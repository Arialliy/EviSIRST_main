#!/usr/bin/env python3
"""Freeze the IRSTD-1K PSBFR-V1 confirmation lockbox.

The only membership parents are the already-frozen
``splits/v2/IRSTD-1K/train.txt`` and ``val.txt`` artifacts.  The former is
deterministically repartitioned into 560 optimization members and 80 locked
confirmation members.  The latter is copied byte-for-byte as
``legacy_dev_val.txt``.

Only masks belonging to the 640 parent-training members are opened, solely to
derive the existing 8-connected target-count/foreground/max-area strata.  No
image, test index, test mask, model, prediction, checkpoint, or metric is read
or executed by this module.

The CLI is check-only by default.  ``--write`` performs a write-once atomic
directory installation; an existing destination must match every expected
byte and may not contain symlinks or extra files.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = Path(__file__).absolute().parents[1]
DATASET = "IRSTD-1K"
DEFAULT_PARENT_SPLIT_DIR = PROJECT_ROOT / "splits" / "v2" / DATASET
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "splits" / "psbfr_v1"
SPLIT_SEED = 2026081701
PARENT_TRAIN_COUNT = 640
PARENT_LEGACY_DEV_COUNT = 160
LOCKBOX_COUNT = 80
NEW_TRAIN_COUNT = 560
SCHEMA = "evisirst_psbfr_confirmation_lockbox/v1"
ALGORITHM_VERSION = "evisirst_psbfr_lockbox_v1"
# This exact namespace reproduces the ranking primitive used by the current
# V2 split builder.  It is copied here so later edits to that module cannot
# silently change this frozen lockbox.
PARENT_RANKING_ALGORITHM_VERSION = "evisirst_group_stratified_sha256_v1"
OUTPUT_ORDER = "parent_v2_train_order"
_EXPECTED_FILENAMES = (
    "train.txt",
    "legacy_dev_val.txt",
    "confirm_lockbox.txt",
    "manifest.json",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MASK_SUFFIXES = (".png", ".bmp")


class PSBFRLockboxError(ValueError):
    """The requested operation violates the frozen lockbox contract."""


@dataclass(frozen=True)
class ParentSnapshot:
    train_ids: tuple[str, ...]
    legacy_dev_ids: tuple[str, ...]
    train_content: bytes
    legacy_dev_content: bytes
    manifest: dict[str, Any]
    manifest_content: bytes
    manifest_sha256: str


@dataclass(frozen=True)
class MaskRecord:
    sample_id: str
    parent_train_position: int
    mask_relpath: str
    mask_sha256: str
    height: int
    width: int
    target_count: int
    foreground_pixels: int
    max_target_area: int
    mask_encoding: str
    nonbinary_pixel_count: int
    stratum: str


@dataclass(frozen=True)
class LockboxBundle:
    train_ids: tuple[str, ...]
    legacy_dev_ids: tuple[str, ...]
    confirm_lockbox_ids: tuple[str, ...]
    train_content: bytes
    legacy_dev_content: bytes
    confirm_lockbox_content: bytes
    manifest: dict[str, Any]
    manifest_content: bytes


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    return _sha256_bytes(
        json.dumps(list(ids), ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _update_length_prefixed(digest: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _stable_rank(namespace: str, value: str, *, split_seed: int) -> int:
    digest = hashlib.sha256()
    for item in (
        PARENT_RANKING_ALGORITHM_VERSION,
        str(split_seed),
        DATASET,
        namespace,
        value,
    ):
        _update_length_prefixed(digest, item)
    return int.from_bytes(digest.digest(), "big")


def _absolute_lexical(path: str | os.PathLike[str]) -> Path:
    """Return an absolute normalized path without resolving symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise PSBFRLockboxError(f"{label} contains a symlink: {current}")


def _require_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    candidate = _absolute_lexical(path)
    _assert_no_symlink_components(candidate, label=label)
    if not candidate.is_dir():
        raise PSBFRLockboxError(f"{label} is not a directory: {candidate}")
    return candidate


def _require_regular_file(path: str | os.PathLike[str], *, label: str) -> Path:
    candidate = _absolute_lexical(path)
    _assert_no_symlink_components(candidate, label=label)
    if not candidate.is_file():
        raise PSBFRLockboxError(f"{label} is not a regular file: {candidate}")
    return candidate


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise PSBFRLockboxError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _parse_ids(content: bytes, *, label: str) -> tuple[str, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PSBFRLockboxError(f"{label} is not UTF-8") from exc
    ids = tuple(text.splitlines())
    if not ids:
        raise PSBFRLockboxError(f"{label} is empty")
    if any(_SAFE_ID.fullmatch(sample_id) is None for sample_id in ids):
        raise PSBFRLockboxError(f"{label} contains an empty or unsafe sample ID")
    if len(ids) != len(set(ids)):
        raise PSBFRLockboxError(f"{label} contains duplicate sample IDs")
    return ids


def load_parent_snapshot(
    parent_split_dir: str | os.PathLike[str] = DEFAULT_PARENT_SPLIT_DIR,
    *,
    expected_train_count: int = PARENT_TRAIN_COUNT,
    expected_legacy_dev_count: int = PARENT_LEGACY_DEV_COUNT,
) -> ParentSnapshot:
    """Load and strictly bind the current V2 train/validation artifacts.

    This function does not inspect a dataset index or any image/mask file.
    """

    directory = _require_directory(parent_split_dir, label="parent split directory")
    train_path = _require_regular_file(directory / "train.txt", label="parent train")
    val_path = _require_regular_file(directory / "val.txt", label="parent val")
    manifest_path = _require_regular_file(
        directory / "manifest.json", label="parent manifest"
    )
    train_content = train_path.read_bytes()
    val_content = val_path.read_bytes()
    manifest_content = manifest_path.read_bytes()
    train_ids = _parse_ids(train_content, label="parent train")
    val_ids = _parse_ids(val_content, label="parent val")
    if len(train_ids) != expected_train_count:
        raise PSBFRLockboxError(
            f"parent train count differs: {len(train_ids)} != {expected_train_count}"
        )
    if len(val_ids) != expected_legacy_dev_count:
        raise PSBFRLockboxError(
            "parent val count differs: "
            f"{len(val_ids)} != {expected_legacy_dev_count}"
        )
    if set(train_ids) & set(val_ids):
        raise PSBFRLockboxError("parent train and val memberships overlap")
    try:
        manifest = json.loads(manifest_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PSBFRLockboxError("parent manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise PSBFRLockboxError("parent manifest must be a JSON object")
    if manifest.get("schema") != "evisirst_v2_train_val_split/v1":
        raise PSBFRLockboxError("unexpected parent split schema")
    if manifest.get("dataset") != DATASET:
        raise PSBFRLockboxError("unexpected parent dataset")
    try:
        parent_algorithm = manifest["algorithm"]
    except (KeyError, TypeError) as exc:
        raise PSBFRLockboxError("parent manifest lacks its algorithm contract") from exc
    if parent_algorithm.get("version") != PARENT_RANKING_ALGORITHM_VERSION:
        raise PSBFRLockboxError("unexpected parent membership algorithm version")
    if parent_algorithm.get("component_connectivity") != 8:
        raise PSBFRLockboxError("parent component connectivity is not 8")
    if parent_algorithm.get("stratification_fields") != [
        "target_count",
        "foreground_pixels",
        "max_target_area",
    ]:
        raise PSBFRLockboxError("unexpected parent stratification fields")

    for role, content, identifiers, expected_count in (
        ("train", train_content, train_ids, expected_train_count),
        ("val", val_content, val_ids, expected_legacy_dev_count),
    ):
        try:
            output = manifest["outputs"][role]
        except (KeyError, TypeError) as exc:
            raise PSBFRLockboxError(f"parent manifest lacks outputs.{role}") from exc
        if output.get("sample_count") != expected_count:
            raise PSBFRLockboxError(f"parent manifest {role} count differs")
        if output.get("relative_path") != f"splits/v2/{DATASET}/{role}.txt":
            raise PSBFRLockboxError(f"parent manifest {role} path differs")
        if output.get("file_sha256") != _sha256_bytes(content):
            raise PSBFRLockboxError(f"parent manifest {role} file hash differs")
        if output.get("ordered_ids_sha256") != _ordered_ids_sha256(identifiers):
            raise PSBFRLockboxError(f"parent manifest {role} ID hash differs")

    try:
        parent_validation = manifest["validation"]
        parent_data_identity = manifest["data_identity"]
    except (KeyError, TypeError) as exc:
        raise PSBFRLockboxError("parent manifest lacks validation/data identity") from exc
    for field in (
        "train_val_disjoint",
        "train_val_union_equals_frozen_train",
        "test_was_not_accessed",
    ):
        if parent_validation.get(field) is not True:
            raise PSBFRLockboxError(f"parent manifest validation.{field} is not true")
    for field in (
        "ordered_image_mask_tree_sha256",
        "ordered_sample_audit_records_sha256",
    ):
        _validate_sha256(parent_data_identity.get(field), label=f"parent {field}")
    try:
        parent_train_distribution = manifest["attribute_distribution"]["train"]
    except (KeyError, TypeError) as exc:
        raise PSBFRLockboxError(
            "parent manifest lacks its train attribute distribution"
        ) from exc
    if parent_train_distribution.get("sample_count") != expected_train_count:
        raise PSBFRLockboxError("parent train attribute count differs")

    return ParentSnapshot(
        train_ids=train_ids,
        legacy_dev_ids=val_ids,
        train_content=train_content,
        legacy_dev_content=val_content,
        manifest=manifest,
        manifest_content=manifest_content,
        manifest_sha256=_sha256_bytes(manifest_content),
    )


def _area_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 4:
        return "1-4"
    if value <= 9:
        return "5-9"
    if value <= 25:
        return "10-25"
    if value <= 64:
        return "26-64"
    return "65+"


def _target_count_bin(value: int) -> str:
    return str(value) if value <= 2 else "3+"


def _stratum(target_count: int, foreground_pixels: int, max_area: int) -> str:
    return (
        f"targets={_target_count_bin(target_count)}|"
        f"foreground={_area_bin(foreground_pixels)}|"
        f"max_target={_area_bin(max_area)}"
    )


def _find_unique_training_mask(mask_directory: Path, sample_id: str) -> Path:
    candidates: list[Path] = []
    for suffix in _MASK_SUFFIXES:
        candidate = mask_directory / f"{sample_id}{suffix}"
        if candidate.is_symlink():
            raise PSBFRLockboxError(f"training mask is a symlink: {candidate}")
        if candidate.is_file():
            _assert_no_symlink_components(candidate, label="training mask")
            candidates.append(candidate)
    if len(candidates) != 1:
        raise PSBFRLockboxError(
            f"expected exactly one training mask for {sample_id!r}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def analyze_training_mask(
    *,
    dataset_root: str | os.PathLike[str],
    sample_id: str,
    parent_train_position: int,
) -> MaskRecord:
    """Derive one stratum from one parent-training mask; no image is opened."""

    if _SAFE_ID.fullmatch(sample_id) is None:
        raise PSBFRLockboxError(f"unsafe sample ID: {sample_id!r}")
    if (
        isinstance(parent_train_position, bool)
        or not isinstance(parent_train_position, int)
        or parent_train_position < 0
    ):
        raise PSBFRLockboxError("parent_train_position must be non-negative")
    root = _require_directory(dataset_root, label="dataset root")
    dataset_directory = _require_directory(root / DATASET, label="dataset directory")
    mask_directory = _require_directory(
        dataset_directory / "masks", label="training mask directory"
    )
    mask_path = _find_unique_training_mask(mask_directory, sample_id)
    content = mask_path.read_bytes()
    try:
        with Image.open(io.BytesIO(content)) as handle:
            mask_array = np.asarray(handle)
    except Exception as exc:
        raise PSBFRLockboxError(f"cannot decode training mask: {sample_id!r}") from exc
    if mask_array.ndim > 2:
        mask_array = mask_array[:, :, 0]
    if mask_array.ndim != 2 or not mask_array.size:
        raise PSBFRLockboxError(f"training mask is not a non-empty 2-D image: {sample_id!r}")
    if not np.isfinite(mask_array).all():
        raise PSBFRLockboxError(f"training mask has non-finite pixels: {sample_id!r}")
    values = np.unique(mask_array)
    value_set = set(values.tolist())
    if value_set.issubset({0, 1}):
        threshold = 0.5
        mask_encoding = "binary_0_1"
        nonbinary_pixel_count = 0
    elif value_set.issubset({0, 255}):
        threshold = 127.0
        mask_encoding = "binary_0_255"
        nonbinary_pixel_count = 0
    elif float(values.min()) >= 0.0 and float(values.max()) <= 255.0:
        threshold = 127.0
        mask_encoding = "grayscale_0_255"
        nonbinary_pixel_count = int(
            np.count_nonzero((mask_array != 0) & (mask_array != 255))
        )
    else:
        raise PSBFRLockboxError(
            f"training mask encoding is outside supported 0..255: {sample_id!r}"
        )
    binary = np.asarray(mask_array > threshold, dtype=np.uint8)
    labels, target_count = ndimage.label(
        binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if target_count:
        areas = np.bincount(labels.reshape(-1), minlength=target_count + 1)[1:]
        foreground_pixels = int(areas.sum())
        max_target_area = int(areas.max())
    else:
        foreground_pixels = 0
        max_target_area = 0
    relative = mask_path.relative_to(root).as_posix()
    return MaskRecord(
        sample_id=sample_id,
        parent_train_position=parent_train_position,
        mask_relpath=relative,
        mask_sha256=_sha256_bytes(content),
        height=int(mask_array.shape[0]),
        width=int(mask_array.shape[1]),
        target_count=int(target_count),
        foreground_pixels=foreground_pixels,
        max_target_area=max_target_area,
        mask_encoding=mask_encoding,
        nonbinary_pixel_count=nonbinary_pixel_count,
        stratum=_stratum(target_count, foreground_pixels, max_target_area),
    )


def load_training_mask_records(
    dataset_root: str | os.PathLike[str], sample_ids: Sequence[str]
) -> tuple[MaskRecord, ...]:
    """Read masks for exactly ``sample_ids``; there is no index discovery."""

    return tuple(
        analyze_training_mask(
            dataset_root=dataset_root,
            sample_id=sample_id,
            parent_train_position=position,
        )
        for position, sample_id in enumerate(sample_ids)
    )


def _largest_remainder_quotas(
    counts: Mapping[str, int], total_target: int, fraction: float, *, split_seed: int
) -> dict[str, int]:
    quotas = {key: int(math.floor(count * fraction)) for key, count in counts.items()}
    remaining = total_target - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            -(counts[key] * fraction - quotas[key]),
            _stable_rank("stratum", key, split_seed=split_seed),
            key,
        ),
    )
    for key in ranked:
        if remaining <= 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise PSBFRLockboxError("cannot allocate the requested lockbox quota")
    return quotas


def _select_lockbox(
    records: Sequence[MaskRecord], *, split_seed: int, lockbox_count: int
) -> tuple[set[str], dict[str, int]]:
    by_stratum: dict[str, list[MaskRecord]] = defaultdict(list)
    for record in records:
        by_stratum[record.stratum].append(record)
    fraction = lockbox_count / len(records)
    quotas = _largest_remainder_quotas(
        {key: len(value) for key, value in by_stratum.items()},
        lockbox_count,
        fraction,
        split_seed=split_seed,
    )
    selected: set[str] = set()
    for stratum, members in by_stratum.items():
        ranked = sorted(
            members,
            key=lambda record: (
                _stable_rank("sample", record.sample_id, split_seed=split_seed),
                record.sample_id,
            ),
        )
        selected.update(record.sample_id for record in ranked[: quotas[stratum]])
    if len(selected) != lockbox_count:
        raise PSBFRLockboxError("selected lockbox count differs from its quota")
    return selected, dict(sorted(quotas.items()))


def _distribution(records: Sequence[MaskRecord]) -> dict[str, Any]:
    count = len(records)
    return {
        "sample_count": count,
        "target_count_total": sum(record.target_count for record in records),
        "target_count_mean": (
            sum(record.target_count for record in records) / count if count else 0.0
        ),
        "foreground_pixel_total": sum(record.foreground_pixels for record in records),
        "foreground_pixel_mean": (
            sum(record.foreground_pixels for record in records) / count
            if count
            else 0.0
        ),
        "target_count_histogram": dict(
            sorted(Counter(str(record.target_count) for record in records).items())
        ),
        "foreground_area_bin_histogram": dict(
            sorted(Counter(_area_bin(record.foreground_pixels) for record in records).items())
        ),
        "max_target_area_bin_histogram": dict(
            sorted(Counter(_area_bin(record.max_target_area) for record in records).items())
        ),
        "mask_shape_histogram": dict(
            sorted(Counter(f"{record.height}x{record.width}" for record in records).items())
        ),
        "mask_encoding_histogram": dict(
            sorted(Counter(record.mask_encoding for record in records).items())
        ),
        "nonbinary_mask_image_count": sum(
            record.nonbinary_pixel_count > 0 for record in records
        ),
        "nonbinary_mask_pixel_count": sum(
            record.nonbinary_pixel_count for record in records
        ),
        "stratum_histogram": dict(
            sorted(Counter(record.stratum for record in records).items())
        ),
    }


def _ordered_mask_tree_sha256(records: Sequence[MaskRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        for value in (record.sample_id, record.mask_relpath, record.mask_sha256):
            _update_length_prefixed(digest, value)
    return digest.hexdigest()


def build_lockbox_bundle(
    *,
    parent: ParentSnapshot,
    records: Sequence[MaskRecord],
    split_seed: int = SPLIT_SEED,
    lockbox_count: int = LOCKBOX_COUNT,
) -> LockboxBundle:
    """Build deterministic bytes without writing them."""

    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise PSBFRLockboxError("split_seed must be an integer")
    if (
        isinstance(lockbox_count, bool)
        or not isinstance(lockbox_count, int)
        or not 0 < lockbox_count < len(parent.train_ids)
    ):
        raise PSBFRLockboxError("lockbox_count must be between 1 and parent_train-1")
    ordered = tuple(sorted(records, key=lambda record: record.parent_train_position))
    record_ids = tuple(record.sample_id for record in ordered)
    if record_ids != parent.train_ids:
        raise PSBFRLockboxError(
            "mask records do not exactly match the ordered parent train membership"
        )
    if tuple(record.parent_train_position for record in ordered) != tuple(
        range(len(ordered))
    ):
        raise PSBFRLockboxError("mask record positions are not contiguous")
    lockbox_set, quotas = _select_lockbox(
        ordered, split_seed=split_seed, lockbox_count=lockbox_count
    )
    train_ids = tuple(sample_id for sample_id in parent.train_ids if sample_id not in lockbox_set)
    lockbox_ids = tuple(sample_id for sample_id in parent.train_ids if sample_id in lockbox_set)
    legacy_ids = parent.legacy_dev_ids
    train_set = set(train_ids)
    legacy_set = set(legacy_ids)
    lockbox_set_checked = set(lockbox_ids)
    if train_set & lockbox_set_checked or train_set & legacy_set or lockbox_set_checked & legacy_set:
        raise PSBFRLockboxError("output memberships are not pairwise disjoint")
    if train_set | lockbox_set_checked != set(parent.train_ids):
        raise PSBFRLockboxError("train/lockbox union differs from parent train")
    if train_set | lockbox_set_checked | legacy_set != set(parent.train_ids) | set(parent.legacy_dev_ids):
        raise PSBFRLockboxError("three-way union differs from parent V2 membership")

    train_content = ("\n".join(train_ids) + "\n").encode("utf-8")
    lockbox_content = ("\n".join(lockbox_ids) + "\n").encode("utf-8")
    legacy_content = parent.legacy_dev_content
    by_id = {record.sample_id: record for record in ordered}
    train_records = tuple(by_id[sample_id] for sample_id in train_ids)
    lockbox_records = tuple(by_id[sample_id] for sample_id in lockbox_ids)
    parent_train_distribution = parent.manifest["attribute_distribution"]["train"]
    observed_parent_train_distribution = _distribution(ordered)
    comparable_distribution = dict(observed_parent_train_distribution)
    comparable_distribution["image_shape_histogram"] = comparable_distribution.pop(
        "mask_shape_histogram"
    )
    if comparable_distribution != parent_train_distribution:
        raise PSBFRLockboxError(
            "mask-derived parent-train attributes differ from the frozen V2 manifest"
        )
    audit_records = [
        {
            "foreground_pixels": record.foreground_pixels,
            "height": record.height,
            "mask_encoding": record.mask_encoding,
            "mask_relpath": record.mask_relpath,
            "mask_sha256": record.mask_sha256,
            "max_target_area": record.max_target_area,
            "nonbinary_pixel_count": record.nonbinary_pixel_count,
            "parent_train_position": record.parent_train_position,
            "sample_id": record.sample_id,
            "split": "confirm_lockbox" if record.sample_id in lockbox_set else "train",
            "stratum": record.stratum,
            "target_count": record.target_count,
            "width": record.width,
        }
        for record in ordered
    ]
    parent_data_identity = parent.manifest["data_identity"]
    parent_outputs = parent.manifest["outputs"]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "status": "frozen_before_any_psbfr_lockbox_model_evaluation",
        "lockbox_not_opened_by_model": True,
        "algorithm": {
            "version": ALGORITHM_VERSION,
            "parent_ranking_algorithm_version": PARENT_RANKING_ALGORITHM_VERSION,
            "split_seed": split_seed,
            "split_seed_used_for_membership": True,
            "selection": "largest_remainder_stratum_quota_then_sha256_rank",
            "membership_hash": "sha256_length_prefixed",
            "component_connectivity": 8,
            "stratification_fields": [
                "target_count",
                "foreground_pixels",
                "max_target_area",
            ],
            "binary_mask_rule": "0/1:>0.5; otherwise 0..255:>127",
            "output_order": OUTPUT_ORDER,
            "parent_train_count": len(parent.train_ids),
            "requested_confirm_lockbox_count": lockbox_count,
            "actual_confirm_lockbox_count": len(lockbox_ids),
            "remaining_train_count": len(train_ids),
            "stratum_lockbox_quotas": quotas,
        },
        "parent_v2": {
            "schema": parent.manifest["schema"],
            "manifest": {
                "relative_path": f"splits/v2/{DATASET}/manifest.json",
                "file_sha256": parent.manifest_sha256,
            },
            "train": {
                "relative_path": f"splits/v2/{DATASET}/train.txt",
                "sample_count": len(parent.train_ids),
                "file_sha256": _sha256_bytes(parent.train_content),
                "ordered_ids_sha256": _ordered_ids_sha256(parent.train_ids),
                "manifest_file_sha256": parent_outputs["train"]["file_sha256"],
                "manifest_ordered_ids_sha256": parent_outputs["train"]["ordered_ids_sha256"],
            },
            "val": {
                "relative_path": f"splits/v2/{DATASET}/val.txt",
                "sample_count": len(parent.legacy_dev_ids),
                "file_sha256": _sha256_bytes(parent.legacy_dev_content),
                "ordered_ids_sha256": _ordered_ids_sha256(parent.legacy_dev_ids),
                "manifest_file_sha256": parent_outputs["val"]["file_sha256"],
                "manifest_ordered_ids_sha256": parent_outputs["val"]["ordered_ids_sha256"],
            },
            "data_identity": {
                "ordered_image_mask_tree_sha256": parent_data_identity[
                    "ordered_image_mask_tree_sha256"
                ],
                "ordered_sample_audit_records_sha256": parent_data_identity[
                    "ordered_sample_audit_records_sha256"
                ],
                "source_index": parent.manifest.get("source_index"),
            },
        },
        "stratification_input": {
            "membership_source": f"splits/v2/{DATASET}/train.txt",
            "member_count": len(ordered),
            "training_masks_opened_for_stratification": True,
            "training_images_opened": False,
            "legacy_dev_masks_opened": False,
            "legacy_dev_images_opened": False,
            "ordered_training_mask_tree_sha256": _ordered_mask_tree_sha256(ordered),
            "ordered_mask_audit_records_sha256": _sha256_bytes(
                _compact_json_bytes(audit_records)
            ),
            "mask_audit_records_embedded": False,
            "attribute_distribution_matches_parent_v2_train": True,
        },
        "outputs": {
            "train": {
                "relative_path": f"splits/psbfr_v1/{DATASET}/train.txt",
                "sample_count": len(train_ids),
                "file_sha256": _sha256_bytes(train_content),
                "ordered_ids_sha256": _ordered_ids_sha256(train_ids),
            },
            "legacy_dev_val": {
                "relative_path": f"splits/psbfr_v1/{DATASET}/legacy_dev_val.txt",
                "sample_count": len(legacy_ids),
                "file_sha256": _sha256_bytes(legacy_content),
                "ordered_ids_sha256": _ordered_ids_sha256(legacy_ids),
                "byte_identical_to_parent_v2_val": True,
            },
            "confirm_lockbox": {
                "relative_path": f"splits/psbfr_v1/{DATASET}/confirm_lockbox.txt",
                "sample_count": len(lockbox_ids),
                "file_sha256": _sha256_bytes(lockbox_content),
                "ordered_ids_sha256": _ordered_ids_sha256(lockbox_ids),
            },
        },
        "attribute_distribution": {
            "parent_v2_train": observed_parent_train_distribution,
            "train": _distribution(train_records),
            "confirm_lockbox": _distribution(lockbox_records),
        },
        "membership_validation": {
            "train_confirm_lockbox_disjoint": True,
            "train_confirm_lockbox_union_equals_parent_v2_train": True,
            "legacy_dev_val_disjoint_from_parent_v2_train": True,
            "all_three_outputs_pairwise_disjoint": True,
            "all_three_outputs_union_equals_parent_v2_train_val": True,
            "legacy_dev_val_byte_identical_to_parent_v2_val": True,
            "parent_train_val_total_count": len(parent.train_ids) + len(parent.legacy_dev_ids),
            "all_three_outputs_total_count": len(train_ids) + len(lockbox_ids) + len(legacy_ids),
        },
        "historical_training_pool_disclosure": {
            "confirm_lockbox_source": "deterministic carve-out from the current V2 training membership",
            "members_were_in_the_historical_training_pool": True,
            "historical_training_pool_non_pristine": True,
            "pristine_external_test": False,
            "required_paper_description": "new locked confirmation set from the original training pool",
            "must_not_be_described_as": "a pristine external test set",
        },
        "generation_audit": {
            "model_imported_or_executed": False,
            "checkpoint_opened": False,
            "prediction_computed": False,
            "metric_computed": False,
            "test_index_opened": False,
            "test_image_opened": False,
            "test_mask_opened": False,
            "test_used_for_membership_or_attributes": False,
            "lockbox_masks_used_only_for_pre_freeze_stratification": True,
            "lockbox_not_opened_by_model": True,
        },
        "immutability": {
            "write_once": True,
            "existing_outputs_require_exact_byte_match": True,
            "symlink_paths_rejected": True,
            "atomic_materialization_unit": f"splits/psbfr_v1/{DATASET}",
        },
    }
    manifest_content = _canonical_json_bytes(manifest)
    return LockboxBundle(
        train_ids=train_ids,
        legacy_dev_ids=legacy_ids,
        confirm_lockbox_ids=lockbox_ids,
        train_content=train_content,
        legacy_dev_content=legacy_content,
        confirm_lockbox_content=lockbox_content,
        manifest=manifest,
        manifest_content=manifest_content,
    )


def generate_lockbox_bundle(
    *,
    dataset_root: str | os.PathLike[str],
    parent_split_dir: str | os.PathLike[str] = DEFAULT_PARENT_SPLIT_DIR,
    expected_train_count: int = PARENT_TRAIN_COUNT,
    expected_legacy_dev_count: int = PARENT_LEGACY_DEV_COUNT,
    split_seed: int = SPLIT_SEED,
    lockbox_count: int = LOCKBOX_COUNT,
) -> LockboxBundle:
    parent = load_parent_snapshot(
        parent_split_dir,
        expected_train_count=expected_train_count,
        expected_legacy_dev_count=expected_legacy_dev_count,
    )
    records = load_training_mask_records(dataset_root, parent.train_ids)
    return build_lockbox_bundle(
        parent=parent,
        records=records,
        split_seed=split_seed,
        lockbox_count=lockbox_count,
    )


def _bundle_files(bundle: LockboxBundle) -> dict[str, bytes]:
    return {
        "train.txt": bundle.train_content,
        "legacy_dev_val.txt": bundle.legacy_dev_content,
        "confirm_lockbox.txt": bundle.confirm_lockbox_content,
        "manifest.json": bundle.manifest_content,
    }


def _destination(output_root: str | os.PathLike[str]) -> tuple[Path, Path]:
    root = _absolute_lexical(output_root)
    _assert_no_symlink_components(root, label="output root")
    destination = root / DATASET
    _assert_no_symlink_components(destination, label="output destination")
    return root, destination


def check_bundle(bundle: LockboxBundle, output_root: str | os.PathLike[str]) -> str:
    """Return prospective/verified status; any partial or differing state fails."""

    _, destination = _destination(output_root)
    if destination.is_symlink():
        raise PSBFRLockboxError(f"output destination is a symlink: {destination}")
    if not destination.exists():
        return "prospective_only"
    if not destination.is_dir():
        raise PSBFRLockboxError(f"output destination is not a directory: {destination}")
    observed_names = {path.name for path in destination.iterdir()}
    expected_names = set(_EXPECTED_FILENAMES)
    if observed_names != expected_names:
        raise PSBFRLockboxError(
            f"output destination is partial or has extras: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    for name, content in _bundle_files(bundle).items():
        path = _require_regular_file(destination / name, label=f"stored {name}")
        if path.read_bytes() != content:
            raise PSBFRLockboxError(f"stored lockbox artifact differs: {path}")
    return "verified_existing"


def _mkdir_without_symlinks(path: Path) -> Path:
    absolute = _absolute_lexical(path)
    _assert_no_symlink_components(absolute, label="output root")
    absolute.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(absolute, label="output root")
    if not absolute.is_dir():
        raise PSBFRLockboxError(f"cannot create output root: {absolute}")
    return absolute


def _write_staged_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically install a directory without replacing an existing path."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(observed_errno, os.strerror(observed_errno), destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    os.rename(source, destination)


def _clean_staging_directory(path: Path) -> None:
    if not path.exists():
        return
    for name in _EXPECTED_FILENAMES:
        candidate = path / name
        if candidate.is_symlink():
            raise PSBFRLockboxError(f"unexpected staging symlink: {candidate}")
        if candidate.exists():
            candidate.unlink()
    path.rmdir()


def materialize_bundle(
    bundle: LockboxBundle, output_root: str | os.PathLike[str]
) -> dict[str, str]:
    """Atomically materialize the complete dataset directory exactly once."""

    initial = check_bundle(bundle, output_root)
    if initial == "verified_existing":
        return {name: "verified_existing" for name in _EXPECTED_FILENAMES}
    root, destination = _destination(output_root)
    root = _mkdir_without_symlinks(root)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{DATASET}.", suffix=".tmp", dir=root)
    )
    try:
        _assert_no_symlink_components(temporary, label="staging directory")
        for name, content in _bundle_files(bundle).items():
            _write_staged_file(temporary / name, content)
        _fsync_directory(temporary)
        try:
            _rename_directory_noreplace(temporary, destination)
        except FileExistsError:
            _clean_staging_directory(temporary)
            if check_bundle(bundle, output_root) != "verified_existing":
                raise PSBFRLockboxError("concurrent output installation differs")
            return {name: "verified_existing" for name in _EXPECTED_FILENAMES}
        _fsync_directory(root)
    finally:
        if temporary.exists():
            _clean_staging_directory(temporary)
    return {name: "written" for name in _EXPECTED_FILENAMES}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only and not args.write:
        args.check_only = True
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = generate_lockbox_bundle(dataset_root=args.dataset_root)
    status: Any
    if args.write:
        status = materialize_bundle(bundle, args.output_root)
    else:
        status = check_bundle(bundle, args.output_root)
    return {
        "schema": SCHEMA + "/cli_result",
        "mode": "write" if args.write else "check_only",
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "parent_train_count": PARENT_TRAIN_COUNT,
        "train_count": len(bundle.train_ids),
        "legacy_dev_val_count": len(bundle.legacy_dev_ids),
        "confirm_lockbox_count": len(bundle.confirm_lockbox_ids),
        "lockbox_not_opened_by_model": True,
        "status": status,
        "destination": f"splits/psbfr_v1/{DATASET}",
        "manifest_sha256": _sha256_bytes(bundle.manifest_content),
    }


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "ALGORITHM_VERSION",
    "DATASET",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PARENT_SPLIT_DIR",
    "LOCKBOX_COUNT",
    "LockboxBundle",
    "MaskRecord",
    "NEW_TRAIN_COUNT",
    "PARENT_LEGACY_DEV_COUNT",
    "PARENT_TRAIN_COUNT",
    "PSBFRLockboxError",
    "ParentSnapshot",
    "SCHEMA",
    "SPLIT_SEED",
    "analyze_training_mask",
    "build_lockbox_bundle",
    "check_bundle",
    "generate_lockbox_bundle",
    "load_parent_snapshot",
    "load_training_mask_records",
    "materialize_bundle",
    "parse_args",
    "run",
]
