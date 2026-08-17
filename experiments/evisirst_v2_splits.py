#!/usr/bin/env python3
"""Build deterministic, train-only V2 train/validation splits for EviSIRST.

The frozen ``img_idx/train_*.txt`` file is the sole membership source.  This
module never opens a test index.  Split membership is controlled only by
``split_seed``; ``run_seed`` is reported by the CLI for downstream provenance,
but is excluded from immutable split artifacts and the membership algorithm.

An explicit sample-to-group JSON mapping is preferred.  Without one, every
sample is treated as its own group and the manifest prominently records the
sample-level fallback.  Outputs are written once under
``splits/v2/<dataset>/{train,val}.txt`` with a deterministic ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

# Support both ``python -m experiments.evisirst_v2_splits`` and direct script
# execution from the repository root without depending on the caller's
# PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import three_dataset_v2_protocol as source_protocol


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "splits" / "v2"
DEFAULT_SPLIT_SEED = 20260811
DEFAULT_RUN_SEED = 42
DEFAULT_VAL_FRACTION = 0.20
SCHEMA = "evisirst_v2_train_val_split/v1"
ALGORITHM_VERSION = "evisirst_group_stratified_sha256_v1"
OUTPUT_ORDER = "frozen_train_index_order"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SUFFIXES = (".png", ".bmp")


class EviSIRSTSplitError(ValueError):
    """The requested split violates the V2 split contract."""


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    index_position: int
    image_relpath: str
    mask_relpath: str
    image_sha256: str
    mask_sha256: str
    height: int
    width: int
    target_count: int
    foreground_pixels: int
    max_target_area: int
    mean_target_area: float
    foreground_fraction: float
    mask_encoding: str
    nonbinary_pixel_count: int
    stratum: str


@dataclass(frozen=True)
class SourceIndexIdentity:
    relative_path: str
    file_sha256: str
    ordered_ids_sha256: str
    count: int


@dataclass(frozen=True)
class SplitBundle:
    dataset: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    train_content: bytes
    val_content: bytes
    manifest: dict[str, Any]
    manifest_content: bytes


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _stable_rank(split_seed: int, dataset: str, namespace: str, value: str) -> int:
    digest = hashlib.sha256()
    for item in (ALGORITHM_VERSION, str(split_seed), dataset, namespace, value):
        _update_length_prefixed(digest, item)
    return int.from_bytes(digest.digest(), "big")


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    return _sha256_bytes(
        json.dumps(list(ids), ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def _relative_regular_file(path: Path, root: Path, *, label: str) -> tuple[Path, str]:
    if path.is_symlink() or not path.is_file():
        raise EviSIRSTSplitError(f"{label} is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise EviSIRSTSplitError(f"{label} escapes dataset_root: {resolved}") from exc
    return resolved, relative.as_posix()


def _unique_file(directory: Path, sample_id: str) -> Path:
    matches = [
        directory / f"{sample_id}{suffix}"
        for suffix in _SUFFIXES
        if (directory / f"{sample_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise EviSIRSTSplitError(
            f"expected one image for {sample_id!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


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


def analyze_sample_file(
    *,
    dataset_root: str | Path,
    sample_id: str,
    index_position: int,
    image_path: str | Path,
    mask_path: str | Path,
) -> SampleRecord:
    """Read one image/mask pair and derive deterministic stratification fields."""

    root = Path(dataset_root).resolve(strict=True)
    if not isinstance(sample_id, str) or _SAFE_ID.fullmatch(sample_id) is None:
        raise EviSIRSTSplitError(f"unsafe sample ID: {sample_id!r}")
    if isinstance(index_position, bool) or index_position < 0:
        raise EviSIRSTSplitError("index_position must be a non-negative integer")
    image, image_relpath = _relative_regular_file(
        Path(image_path), root, label="image"
    )
    mask, mask_relpath = _relative_regular_file(Path(mask_path), root, label="mask")
    with Image.open(image) as handle:
        width, height = handle.size
    with Image.open(mask) as handle:
        mask_array = np.asarray(handle)
    if mask_array.ndim > 2:
        mask_array = mask_array[:, :, 0]
    if mask_array.ndim != 2 or tuple(mask_array.shape) != (height, width):
        raise EviSIRSTSplitError(
            f"image/mask dimensions differ for {sample_id!r}: "
            f"{(height, width)} != {tuple(mask_array.shape)}"
        )
    if not np.isfinite(mask_array).all():
        raise EviSIRSTSplitError(f"mask contains non-finite pixels: {sample_id!r}")

    values = np.unique(mask_array)
    value_set = set(values.tolist())
    if value_set.issubset({0, 1}):
        threshold = 0.5
        mask_encoding = "binary_0_1"
        nonbinary = 0
    elif value_set.issubset({0, 255}):
        threshold = 127.0
        mask_encoding = "binary_0_255"
        nonbinary = 0
    elif float(values.min()) >= 0.0 and float(values.max()) <= 255.0:
        threshold = 127.0
        mask_encoding = "grayscale_0_255"
        nonbinary = int(np.count_nonzero((mask_array != 0) & (mask_array != 255)))
    else:
        raise EviSIRSTSplitError(
            f"mask encoding is outside supported 0..255 range: {sample_id!r}"
        )

    binary = np.asarray(mask_array > threshold, dtype=np.uint8)
    labels, target_count = ndimage.label(
        binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if target_count:
        areas = np.bincount(labels.reshape(-1), minlength=target_count + 1)[1:]
        foreground_pixels = int(areas.sum())
        max_target_area = int(areas.max())
        mean_target_area = float(areas.mean())
    else:
        foreground_pixels = 0
        max_target_area = 0
        mean_target_area = 0.0
    total_pixels = int(height * width)
    return SampleRecord(
        sample_id=sample_id,
        index_position=int(index_position),
        image_relpath=image_relpath,
        mask_relpath=mask_relpath,
        image_sha256=_sha256_file(image),
        mask_sha256=_sha256_file(mask),
        height=int(height),
        width=int(width),
        target_count=int(target_count),
        foreground_pixels=foreground_pixels,
        max_target_area=max_target_area,
        mean_target_area=mean_target_area,
        foreground_fraction=foreground_pixels / total_pixels,
        mask_encoding=mask_encoding,
        nonbinary_pixel_count=nonbinary,
        stratum=_stratum(target_count, foreground_pixels, max_target_area),
    )


def analyze_fixture_index(
    *,
    dataset_root: str | Path,
    dataset: str,
    index_path: str | Path,
    image_dir: str | Path,
    mask_dir: str | Path,
) -> tuple[list[SampleRecord], SourceIndexIdentity]:
    """Analyze a simple train index; intended for tests and local fixtures."""

    root = Path(dataset_root).resolve(strict=True)
    index, index_relative = _relative_regular_file(
        Path(index_path), root, label="train index"
    )
    content = index.read_bytes()
    try:
        sample_ids = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EviSIRSTSplitError("train index is not UTF-8") from exc
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise EviSIRSTSplitError("train index must be non-empty and contain unique IDs")
    records = [
        analyze_sample_file(
            dataset_root=root,
            sample_id=sample_id,
            index_position=position,
            image_path=_unique_file(Path(image_dir), sample_id),
            mask_path=_unique_file(Path(mask_dir), sample_id),
        )
        for position, sample_id in enumerate(sample_ids)
    ]
    return records, SourceIndexIdentity(
        relative_path=index_relative,
        file_sha256=_sha256_bytes(content),
        ordered_ids_sha256=_ordered_ids_sha256(sample_ids),
        count=len(sample_ids),
    )


def load_frozen_training_records(
    dataset_root: str | Path, dataset: str
) -> tuple[list[SampleRecord], SourceIndexIdentity]:
    """Load only one source dataset's frozen train index and train files."""

    dataset = source_protocol.require_dataset(dataset)
    root = Path(dataset_root).resolve(strict=True)
    sample_ids = source_protocol.load_index(root, dataset, "train")
    known_ids = frozenset(sample_ids)
    index_path = source_protocol.index_path(root, dataset, "train")
    _, index_relative = _relative_regular_file(index_path, root, label="train index")
    records: list[SampleRecord] = []
    for position, sample_id in enumerate(sample_ids):
        resolved = source_protocol.resolve_sample(
            root,
            dataset,
            sample_id,
            split="train",
            known_ids=known_ids,
        )
        if resolved.correction_applied:
            raise EviSIRSTSplitError("a test-only correction appeared in train")
        records.append(
            analyze_sample_file(
                dataset_root=root,
                sample_id=sample_id,
                index_position=position,
                image_path=resolved.image_path,
                mask_path=resolved.mask_path,
            )
        )
    content = index_path.read_bytes()
    return records, SourceIndexIdentity(
        relative_path=index_relative,
        file_sha256=_sha256_bytes(content),
        ordered_ids_sha256=_ordered_ids_sha256(sample_ids),
        count=len(sample_ids),
    )


def load_group_mapping(
    path: str | Path, sample_ids: Sequence[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load a complete JSON ``{sample_id: group_id}`` mapping."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise EviSIRSTSplitError(f"group mapping is not a regular file: {candidate}")
    content = candidate.read_bytes()
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EviSIRSTSplitError("group mapping must be a UTF-8 JSON object") from exc
    if isinstance(payload, Mapping) and isinstance(payload.get("groups"), Mapping):
        payload = payload["groups"]
    if not isinstance(payload, Mapping):
        raise EviSIRSTSplitError("group mapping must be an object")
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, str)
            or not value
            or _SAFE_ID.fullmatch(value) is None
        ):
            raise EviSIRSTSplitError("group mapping keys/values must be safe strings")
        mapping[key] = value
    expected = set(sample_ids)
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))[:10]
        extra = sorted(set(mapping) - expected)[:10]
        raise EviSIRSTSplitError(
            f"group mapping must cover the frozen train index exactly; "
            f"missing={missing}, extra={extra}"
        )
    canonical_mapping = _canonical_json_bytes(mapping)
    return mapping, {
        "mode": "explicit_group_mapping",
        "mapping_file_name": candidate.name,
        "mapping_file_sha256": _sha256_bytes(content),
        "canonical_mapping_sha256": _sha256_bytes(canonical_mapping),
    }


def _largest_remainder_quotas(
    counts: Mapping[str, int], total_target: int, fraction: float, *, tie_key: Any
) -> dict[str, int]:
    quotas = {key: int(math.floor(count * fraction)) for key, count in counts.items()}
    remaining = total_target - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            -(counts[key] * fraction - quotas[key]),
            tie_key(key),
            key,
        ),
    )
    for key in ranked:
        if remaining <= 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining != 0:
        raise EviSIRSTSplitError("could not allocate the requested validation quota")
    return quotas


def _sample_level_membership(
    records: Sequence[SampleRecord], dataset: str, split_seed: int, val_count: int
) -> tuple[set[str], dict[str, str], dict[str, Any]]:
    by_stratum: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_stratum[record.stratum].append(record)
    counts = {key: len(value) for key, value in by_stratum.items()}
    fraction = val_count / len(records)
    quotas = _largest_remainder_quotas(
        counts,
        val_count,
        fraction,
        tie_key=lambda value: _stable_rank(split_seed, dataset, "stratum", value),
    )
    validation: set[str] = set()
    for stratum, members in by_stratum.items():
        ordered = sorted(
            members,
            key=lambda record: (
                _stable_rank(split_seed, dataset, "sample", record.sample_id),
                record.sample_id,
            ),
        )
        validation.update(record.sample_id for record in ordered[: quotas[stratum]])
    groups = {record.sample_id: record.sample_id for record in records}
    return validation, groups, {
        "mode": "sample_level_fallback",
        "warning": (
            "no explicit group mapping was supplied; sequence/scene/near-duplicate "
            "leakage cannot be ruled out"
        ),
        "group_count": len(records),
        "stratum_sample_quotas": dict(sorted(quotas.items())),
    }


def _group_level_membership(
    records: Sequence[SampleRecord],
    dataset: str,
    split_seed: int,
    val_fraction: float,
    target_val_count: int,
    group_mapping: Mapping[str, str],
    mapping_metadata: Mapping[str, Any],
) -> tuple[set[str], dict[str, str], dict[str, Any]]:
    by_id = {record.sample_id: record for record in records}
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for sample_id, group_id in group_mapping.items():
        grouped[group_id].append(by_id[sample_id])
    if len(grouped) < 2:
        raise EviSIRSTSplitError("group-aware splitting requires at least two groups")

    group_stratum: dict[str, str] = {}
    for group_id, members in grouped.items():
        counts = Counter(record.stratum for record in members)
        group_stratum[group_id] = min(
            counts,
            key=lambda value: (-counts[value], value),
        )
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id, stratum in group_stratum.items():
        by_stratum[stratum].append(group_id)
    target_group_count = max(
        1, min(len(grouped) - 1, int(math.floor(len(grouped) * val_fraction + 0.5)))
    )
    group_quotas = _largest_remainder_quotas(
        {key: len(value) for key, value in by_stratum.items()},
        target_group_count,
        target_group_count / len(grouped),
        tie_key=lambda value: _stable_rank(split_seed, dataset, "group_stratum", value),
    )
    selected: set[str] = set()
    for stratum, group_ids in by_stratum.items():
        ordered = sorted(
            group_ids,
            key=lambda group_id: (
                _stable_rank(split_seed, dataset, "group", group_id),
                group_id,
            ),
        )
        selected.update(ordered[: group_quotas[stratum]])

    # Preserve the allocated number of groups in every stratum while improving
    # the requested sample count through deterministic within-stratum swaps.
    while True:
        current_count = sum(len(grouped[group_id]) for group_id in selected)
        current_error = abs(current_count - target_val_count)
        best: tuple[int, int, int, str, str] | None = None
        for old_group in sorted(selected):
            stratum = group_stratum[old_group]
            for new_group in by_stratum[stratum]:
                if new_group in selected:
                    continue
                new_count = (
                    current_count
                    - len(grouped[old_group])
                    + len(grouped[new_group])
                )
                error = abs(new_count - target_val_count)
                if error >= current_error:
                    continue
                candidate = (
                    error,
                    _stable_rank(
                        split_seed,
                        dataset,
                        "group_swap",
                        f"{old_group}->{new_group}",
                    ),
                    new_count,
                    old_group,
                    new_group,
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, _, _, old_group, new_group = best
        selected.remove(old_group)
        selected.add(new_group)

    validation = {
        record.sample_id
        for group_id in selected
        for record in grouped[group_id]
    }
    metadata = dict(mapping_metadata)
    metadata.update(
        {
            "group_count": len(grouped),
            "validation_group_count": len(selected),
            "group_stratum_quotas": dict(sorted(group_quotas.items())),
            "target_validation_sample_count": target_val_count,
            "actual_validation_sample_count": len(validation),
            "sample_count_deviation_due_to_group_indivisibility": (
                len(validation) - target_val_count
            ),
        }
    )
    return validation, dict(group_mapping), metadata


def _distribution(records: Sequence[SampleRecord]) -> dict[str, Any]:
    count = len(records)
    target_counts = Counter(str(record.target_count) for record in records)
    foreground_bins = Counter(_area_bin(record.foreground_pixels) for record in records)
    max_area_bins = Counter(_area_bin(record.max_target_area) for record in records)
    shape_counts = Counter(f"{record.height}x{record.width}" for record in records)
    encoding_counts = Counter(record.mask_encoding for record in records)
    stratum_counts = Counter(record.stratum for record in records)
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
        "nonbinary_mask_image_count": sum(
            record.nonbinary_pixel_count > 0 for record in records
        ),
        "nonbinary_mask_pixel_count": sum(
            record.nonbinary_pixel_count for record in records
        ),
        "target_count_histogram": dict(sorted(target_counts.items())),
        "foreground_area_bin_histogram": dict(sorted(foreground_bins.items())),
        "max_target_area_bin_histogram": dict(sorted(max_area_bins.items())),
        "image_shape_histogram": dict(sorted(shape_counts.items())),
        "mask_encoding_histogram": dict(sorted(encoding_counts.items())),
        "stratum_histogram": dict(sorted(stratum_counts.items())),
    }


def _input_tree_sha256(records: Sequence[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        for value in (
            record.sample_id,
            record.image_relpath,
            record.image_sha256,
            record.mask_relpath,
            record.mask_sha256,
        ):
            _update_length_prefixed(digest, value)
    return digest.hexdigest()


def build_split_bundle(
    *,
    dataset: str,
    records: Sequence[SampleRecord],
    source_index: SourceIndexIdentity,
    split_seed: int,
    run_seed: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    group_mapping: Mapping[str, str] | None = None,
    group_mapping_metadata: Mapping[str, Any] | None = None,
) -> SplitBundle:
    """Build deterministic split contents and a self-auditing manifest."""

    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise EviSIRSTSplitError("split_seed must be an integer")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise EviSIRSTSplitError("run_seed must be an integer")
    if not math.isfinite(val_fraction) or not 0.0 < val_fraction < 1.0:
        raise EviSIRSTSplitError("val_fraction must be in (0, 1)")
    ordered = sorted(records, key=lambda record: record.index_position)
    ids = [record.sample_id for record in ordered]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise EviSIRSTSplitError("records must contain at least two unique sample IDs")
    if (
        source_index.count != len(ids)
        or source_index.ordered_ids_sha256 != _ordered_ids_sha256(ids)
    ):
        raise EviSIRSTSplitError("records differ from source index identity")
    target_val_count = max(
        1, min(len(ids) - 1, int(math.floor(len(ids) * val_fraction + 0.5)))
    )

    if group_mapping is None:
        val_set, groups, grouping = _sample_level_membership(
            ordered, dataset, split_seed, target_val_count
        )
    else:
        if set(group_mapping) != set(ids):
            raise EviSIRSTSplitError("group mapping does not exactly cover records")
        val_set, groups, grouping = _group_level_membership(
            ordered,
            dataset,
            split_seed,
            val_fraction,
            target_val_count,
            group_mapping,
            group_mapping_metadata or {"mode": "explicit_group_mapping"},
        )

    train_ids = tuple(sample_id for sample_id in ids if sample_id not in val_set)
    val_ids = tuple(sample_id for sample_id in ids if sample_id in val_set)
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise EviSIRSTSplitError("train/validation membership is invalid")
    if set(train_ids) | set(val_ids) != set(ids):
        raise EviSIRSTSplitError(
            "train/validation union differs from source train index"
        )
    train_groups = {groups[sample_id] for sample_id in train_ids}
    val_groups = {groups[sample_id] for sample_id in val_ids}
    if train_groups & val_groups:
        raise EviSIRSTSplitError("a group crosses train and validation")

    train_content = ("\n".join(train_ids) + "\n").encode("utf-8")
    val_content = ("\n".join(val_ids) + "\n").encode("utf-8")
    by_id = {record.sample_id: record for record in ordered}
    train_records = [by_id[sample_id] for sample_id in train_ids]
    val_records = [by_id[sample_id] for sample_id in val_ids]
    # Keep the committed manifest compact while still cryptographically binding
    # every relative input path, file hash, derived attribute, group, and split.
    # The full records are reproducible from the frozen index and source tree.
    sample_records = [
        {
            "sample_id": record.sample_id,
            "source_index_position": record.index_position,
            "split": "val" if record.sample_id in val_set else "train",
            "group_id": groups[record.sample_id],
            "stratum": record.stratum,
            "image_relpath": record.image_relpath,
            "mask_relpath": record.mask_relpath,
            "image_sha256": record.image_sha256,
            "mask_sha256": record.mask_sha256,
            "height": record.height,
            "width": record.width,
            "target_count": record.target_count,
            "foreground_pixels": record.foreground_pixels,
            "max_target_area": record.max_target_area,
            "mean_target_area": record.mean_target_area,
            "foreground_fraction": record.foreground_fraction,
            "mask_encoding": record.mask_encoding,
            "nonbinary_pixel_count": record.nonbinary_pixel_count,
        }
        for record in ordered
    ]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "algorithm": {
            "version": ALGORITHM_VERSION,
            "membership_hash": "sha256_length_prefixed",
            "stratification_fields": [
                "target_count",
                "foreground_pixels",
                "max_target_area",
            ],
            "component_connectivity": 8,
            "binary_mask_rule": "0/1:>0.5; otherwise 0..255:>127",
            "validation_count_rounding": "floor(N*fraction+0.5), clamped to [1,N-1]",
            "output_order": OUTPUT_ORDER,
        },
        "dataset": dataset,
        "seeds": {
            "split_seed": split_seed,
            "split_seed_used_for_membership": True,
            "run_seed": None,
            "run_seed_used_for_membership": False,
            "run_seed_role": (
                "accepted and reported by the CLI, but intentionally excluded "
                "from immutable split artifacts; record it in each training run"
            ),
        },
        "request": {
            "validation_fraction": val_fraction,
            "target_validation_sample_count": target_val_count,
            "actual_validation_sample_count": len(val_ids),
        },
        "source_index": {
            "split": "train",
            "relative_path": source_index.relative_path,
            "file_sha256": source_index.file_sha256,
            "ordered_ids_sha256": source_index.ordered_ids_sha256,
            "sample_count": source_index.count,
        },
        "data_identity": {
            "paths_are_relative_to": "dataset_root",
            "ordered_image_mask_tree_sha256": _input_tree_sha256(ordered),
            "relative_file_identifier_count": 2 * len(ordered),
            "ordered_sample_audit_records_sha256": _sha256_bytes(
                _canonical_json_bytes(sample_records)
            ),
            "sample_records_embedded": False,
            "sample_records_reconstruction": (
                "read the frozen train index in order and re-run this algorithm "
                "against the verified source tree"
            ),
        },
        "test_access": {
            "test_index_opened": False,
            "test_image_opened": False,
            "test_mask_opened": False,
            "test_used_for_split_or_attributes": False,
        },
        "grouping": grouping,
        "outputs": {
            "train": {
                "relative_path": f"splits/v2/{dataset}/train.txt",
                "sample_count": len(train_ids),
                "file_sha256": _sha256_bytes(train_content),
                "ordered_ids_sha256": _ordered_ids_sha256(train_ids),
            },
            "val": {
                "relative_path": f"splits/v2/{dataset}/val.txt",
                "sample_count": len(val_ids),
                "file_sha256": _sha256_bytes(val_content),
                "ordered_ids_sha256": _ordered_ids_sha256(val_ids),
            },
        },
        "attribute_distribution": {
            "source_train": _distribution(ordered),
            "train": _distribution(train_records),
            "val": _distribution(val_records),
        },
        "validation": {
            "train_val_disjoint": True,
            "train_val_union_equals_frozen_train": True,
            "group_disjoint": True,
            "test_was_not_accessed": True,
        },
    }
    manifest_content = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    return SplitBundle(
        dataset=dataset,
        train_ids=train_ids,
        val_ids=val_ids,
        train_content=train_content,
        val_content=val_content,
        manifest=manifest,
        manifest_content=manifest_content,
    )


def generate_dataset_bundle(
    *,
    dataset_root: str | Path,
    dataset: str,
    split_seed: int,
    run_seed: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    group_mapping_path: str | Path | None = None,
) -> SplitBundle:
    records, source_index = load_frozen_training_records(dataset_root, dataset)
    mapping: dict[str, str] | None = None
    mapping_metadata: dict[str, Any] | None = None
    if group_mapping_path is not None:
        mapping, mapping_metadata = load_group_mapping(
            group_mapping_path, [record.sample_id for record in records]
        )
    return build_split_bundle(
        dataset=dataset,
        records=records,
        source_index=source_index,
        split_seed=split_seed,
        run_seed=run_seed,
        val_fraction=val_fraction,
        group_mapping=mapping,
        group_mapping_metadata=mapping_metadata,
    )


def _write_once(path: Path, content: bytes) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"existing split artifact differs: {path}")
        return "verified_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "written"


def materialize_bundle(bundle: SplitBundle, output_root: str | Path) -> dict[str, str]:
    destination = Path(output_root).resolve() / bundle.dataset
    statuses = {
        "train": _write_once(destination / "train.txt", bundle.train_content),
        "val": _write_once(destination / "val.txt", bundle.val_content),
        "manifest": _write_once(destination / "manifest.json", bundle.manifest_content),
    }
    return statuses


def check_bundle(bundle: SplitBundle, output_root: str | Path) -> str:
    destination = Path(output_root).resolve() / bundle.dataset
    expected = {
        destination / "train.txt": bundle.train_content,
        destination / "val.txt": bundle.val_content,
        destination / "manifest.json": bundle.manifest_content,
    }
    present = [path.exists() for path in expected]
    if not any(present):
        return "prospective_only"
    if not all(present):
        raise FileNotFoundError(f"split output is partial: {destination}")
    for path, content in expected.items():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise EviSIRSTSplitError(f"stored split artifact differs: {path}")
    return "verified_existing"


def _parse_group_mapping_args(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise EviSIRSTSplitError(
                "--group-mapping must use DATASET=/path/to/groups.json"
            )
        dataset, raw_path = value.split("=", 1)
        source_protocol.require_dataset(dataset)
        if dataset in parsed:
            raise EviSIRSTSplitError(f"duplicate group mapping for {dataset}")
        parsed[dataset] = Path(raw_path)
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("all", *source_protocol.DATASETS), default="all"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--run-seed", type=int, default=DEFAULT_RUN_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument(
        "--group-mapping",
        action="append",
        default=[],
        metavar="DATASET=JSON",
        help="complete sample_id -> group_id mapping; repeat per dataset",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only and not args.write:
        args.check_only = True
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    datasets = (
        source_protocol.DATASETS if args.dataset == "all" else (args.dataset,)
    )
    mappings = _parse_group_mapping_args(args.group_mapping)
    unexpected = set(mappings) - set(datasets)
    if unexpected:
        raise EviSIRSTSplitError(
            "group mappings were supplied for unselected datasets: "
            f"{sorted(unexpected)}"
        )
    results: dict[str, Any] = {}
    for dataset in datasets:
        bundle = generate_dataset_bundle(
            dataset_root=args.dataset_root,
            dataset=dataset,
            split_seed=args.split_seed,
            run_seed=args.run_seed,
            val_fraction=args.val_fraction,
            group_mapping_path=mappings.get(dataset),
        )
        if args.write:
            status: Any = materialize_bundle(bundle, args.output_root)
        else:
            status = check_bundle(bundle, args.output_root)
        results[dataset] = {
            "status": status,
            "train_count": len(bundle.train_ids),
            "val_count": len(bundle.val_ids),
            "grouping_mode": bundle.manifest["grouping"]["mode"],
            "test_accessed": False,
            "destination": f"splits/v2/{dataset}",
        }
    return {
        "schema": SCHEMA + "/cli_result",
        "mode": "write" if args.write else "check_only",
        "split_seed": args.split_seed,
        "run_seed": args.run_seed,
        "run_seed_used_for_membership": False,
        "datasets": results,
    }


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RUN_SEED",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_VAL_FRACTION",
    "EviSIRSTSplitError",
    "SCHEMA",
    "SampleRecord",
    "SourceIndexIdentity",
    "SplitBundle",
    "analyze_fixture_index",
    "analyze_sample_file",
    "build_split_bundle",
    "check_bundle",
    "generate_dataset_bundle",
    "load_frozen_training_records",
    "load_group_mapping",
    "materialize_bundle",
    "parse_args",
    "run",
]
