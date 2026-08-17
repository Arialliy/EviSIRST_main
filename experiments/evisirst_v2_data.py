"""Train/validation data layer for the immutable EviSIRST V2 splits.

Only ``splits/v2/<dataset>/{train,val}.txt`` and the frozen source ``train``
index are consumed.  This module has no code path that resolves or opens a
source ``test`` index.  Split membership is controlled by the split artifact;
``run_seed`` controls only sample-local training augmentation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Keep the primary contract as ``import experiments.evisirst_v2_data`` while
# also allowing a direct-file import smoke check from the repository root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import evisirst_v2_splits as split_protocol
from experiments import three_dataset_v2_protocol as source_protocol


DEFAULT_SPLIT_ROOT = Path(__file__).resolve().parents[1] / "splits" / "v2"
TARGET_MODES = ("soft", "binary")
NORMALIZATION_MODES = ("legacy",)
AUGMENTATION_VERSION = "evisirst_v2_run_epoch_sample_occurrence_sha256_v1"
_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_SUFFIXES = (".png", ".bmp")


class EviSIRSTV2DataError(ValueError):
    """A V2 split or sample violates the train/validation data contract."""


@dataclass(frozen=True)
class V2SplitContract:
    dataset: str
    dataset_root: Path
    split_root: Path
    manifest_path: Path
    manifest_sha256: str
    data_tree_sha256: str | None
    data_tree_verified: bool
    manifest: dict[str, Any]
    source_train_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]


@dataclass(frozen=True)
class V2TransformPlan:
    augmentation_seed: int
    occurrence: int
    crop_top: int
    crop_left: int
    crop_size: int
    padded_height: int
    padded_width: int
    crop_attempts: int
    flip_axis0: bool
    flip_axis1: bool
    transpose: bool


@dataclass(frozen=True)
class NormalizationSpec:
    mode: str
    source: str
    mean: float
    std: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big"))
    digest.update(encoded)


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    content = json.dumps(
        list(ids), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(content)


def _stable_uint63(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in (AUGMENTATION_VERSION, *parts):
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def _require_regular_file(path: Path, *, root: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise EviSIRSTV2DataError(f"{label} is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EviSIRSTV2DataError(f"{label} escapes its root: {resolved}") from exc
    return resolved


def _resolve_relative_file(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EviSIRSTV2DataError(f"{label} is not a safe relative path")
    return _require_regular_file(root / candidate, root=root, label=label)


def _decode_index(
    content: bytes, *, label: str, require_canonical_output: bool = True
) -> tuple[str, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EviSIRSTV2DataError(f"{label} is not UTF-8") from exc
    identifiers = tuple(text.splitlines())
    if not identifiers:
        raise EviSIRSTV2DataError(f"{label} is empty")
    if require_canonical_output and content != (
        "\n".join(identifiers) + "\n"
    ).encode("utf-8"):
        raise EviSIRSTV2DataError(f"{label} is not canonical one-ID-per-line text")
    if len(identifiers) != len(set(identifiers)):
        raise EviSIRSTV2DataError(f"{label} contains duplicate IDs")
    if any(_SAMPLE_ID_RE.fullmatch(identifier) is None for identifier in identifiers):
        raise EviSIRSTV2DataError(f"{label} contains an unsafe sample ID")
    return identifiers


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EviSIRSTV2DataError(f"manifest {label} must be an object")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EviSIRSTV2DataError(f"manifest {label} must be an integer")
    return value


def _validate_output_entry(
    *,
    manifest: Mapping[str, Any],
    dataset: str,
    split: str,
    content: bytes,
    ids: Sequence[str],
) -> None:
    outputs = _mapping(manifest.get("outputs"), label="outputs")
    entry = _mapping(outputs.get(split), label=f"outputs.{split}")
    expected_relative = f"splits/v2/{dataset}/{split}.txt"
    if entry.get("relative_path") != expected_relative:
        raise EviSIRSTV2DataError(
            f"manifest outputs.{split}.relative_path differs"
        )
    if entry.get("file_sha256") != _sha256_bytes(content):
        raise EviSIRSTV2DataError(f"{split}.txt SHA-256 differs from manifest")
    if entry.get("ordered_ids_sha256") != _ordered_ids_sha256(ids):
        raise EviSIRSTV2DataError(f"{split}.txt ordered-ID SHA-256 differs")
    if _integer(
        entry.get("sample_count"),
        label=f"outputs.{split}.sample_count",
    ) != len(ids):
        raise EviSIRSTV2DataError(f"{split}.txt count differs from manifest")


def load_v2_split_contract(
    dataset: str,
    *,
    dataset_root: str | Path,
    split_root: str | Path = DEFAULT_SPLIT_ROOT,
    verify_data_tree: bool = True,
) -> V2SplitContract:
    """Read and strictly validate one train-only V2 split artifact.

    Validation covers schema/algorithm identity, output hashes and counts, the
    frozen source train-index hash/order/count, exact union, disjointness, and
    preservation of source-index order.  By default, it also hashes every
    source-train image/mask once and compares the ordered tree digest.  No test
    path is constructed.
    """

    dataset = source_protocol.require_dataset(dataset)
    data_root = Path(dataset_root).resolve(strict=True)
    artifact_root = Path(split_root).resolve(strict=True)
    dataset_directory = artifact_root / dataset
    if dataset_directory.is_symlink() or not dataset_directory.is_dir():
        raise EviSIRSTV2DataError(
            f"split dataset directory is not regular: {dataset_directory}"
        )
    dataset_directory = dataset_directory.resolve(strict=True)
    try:
        dataset_directory.relative_to(artifact_root)
    except ValueError as exc:
        raise EviSIRSTV2DataError("split dataset directory escapes split_root") from exc

    manifest_path = _require_regular_file(
        dataset_directory / "manifest.json",
        root=artifact_root,
        label="manifest",
    )
    train_path = _require_regular_file(
        dataset_directory / "train.txt",
        root=artifact_root,
        label="train split",
    )
    val_path = _require_regular_file(
        dataset_directory / "val.txt",
        root=artifact_root,
        label="validation split",
    )
    manifest_content = manifest_path.read_bytes()
    train_content = train_path.read_bytes()
    val_content = val_path.read_bytes()
    try:
        manifest = json.loads(manifest_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EviSIRSTV2DataError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise EviSIRSTV2DataError("manifest root must be an object")
    if manifest.get("schema") != split_protocol.SCHEMA:
        raise EviSIRSTV2DataError("manifest schema differs")
    if manifest.get("dataset") != dataset:
        raise EviSIRSTV2DataError("manifest dataset differs")
    algorithm = _mapping(manifest.get("algorithm"), label="algorithm")
    if algorithm.get("version") != split_protocol.ALGORITHM_VERSION:
        raise EviSIRSTV2DataError("manifest algorithm version differs")
    if algorithm.get("output_order") != split_protocol.OUTPUT_ORDER:
        raise EviSIRSTV2DataError("manifest output-order contract differs")

    seeds = _mapping(manifest.get("seeds"), label="seeds")
    _integer(seeds.get("split_seed"), label="seeds.split_seed")
    if seeds.get("split_seed_used_for_membership") is not True:
        raise EviSIRSTV2DataError("split seed membership flag differs")
    if seeds.get("run_seed_used_for_membership") is not False:
        raise EviSIRSTV2DataError("run seed must not control split membership")

    train_ids = _decode_index(train_content, label="train.txt")
    val_ids = _decode_index(val_content, label="val.txt")
    _validate_output_entry(
        manifest=manifest,
        dataset=dataset,
        split="train",
        content=train_content,
        ids=train_ids,
    )
    _validate_output_entry(
        manifest=manifest,
        dataset=dataset,
        split="val",
        content=val_content,
        ids=val_ids,
    )

    source = _mapping(manifest.get("source_index"), label="source_index")
    if source.get("split") != "train":
        raise EviSIRSTV2DataError("manifest source index is not train")
    relative_index = source.get("relative_path")
    if not isinstance(relative_index, str):
        raise EviSIRSTV2DataError("manifest source-index path must be relative text")
    expected_train_index = str(
        source_protocol.EXPECTED_SPLITS[dataset]["train"]["index_relpath"]
    )
    if relative_index != expected_train_index:
        # Check the lexical identifier before opening anything.  In particular,
        # a tampered manifest cannot redirect this consumer to a test index.
        raise EviSIRSTV2DataError("manifest source path is not the train index")
    index_parts = Path(relative_index).parts
    if not index_parts or index_parts[0] != dataset:
        raise EviSIRSTV2DataError("source train index is outside the selected dataset")
    source_index_path = _resolve_relative_file(
        data_root, relative_index, label="source train index"
    )
    source_content = source_index_path.read_bytes()
    # Frozen source indices retain their historically hashed byte form (some
    # omit a final newline).  Their exact bytes and parsed order are validated
    # independently below; only generated train/val outputs must be canonical.
    source_ids = _decode_index(
        source_content,
        label="source train index",
        require_canonical_output=False,
    )
    if source.get("file_sha256") != _sha256_bytes(source_content):
        raise EviSIRSTV2DataError("source train-index SHA-256 differs")
    if source.get("ordered_ids_sha256") != _ordered_ids_sha256(source_ids):
        raise EviSIRSTV2DataError("source train-index ordered-ID SHA-256 differs")
    if _integer(source.get("sample_count"), label="source_index.sample_count") != len(
        source_ids
    ):
        raise EviSIRSTV2DataError("source train-index count differs")

    train_set = set(train_ids)
    val_set = set(val_ids)
    source_set = set(source_ids)
    if train_set & val_set:
        raise EviSIRSTV2DataError("train and validation IDs overlap")
    if train_set | val_set != source_set:
        raise EviSIRSTV2DataError("train/validation union differs from source train")
    if tuple(item for item in source_ids if item in train_set) != train_ids:
        raise EviSIRSTV2DataError("train IDs do not preserve source-index order")
    if tuple(item for item in source_ids if item in val_set) != val_ids:
        raise EviSIRSTV2DataError("validation IDs do not preserve source-index order")

    request = _mapping(manifest.get("request"), label="request")
    if _integer(
        request.get("actual_validation_sample_count"),
        label="request.actual_validation_sample_count",
    ) != len(val_ids):
        raise EviSIRSTV2DataError("manifest actual validation count differs")
    distributions = _mapping(
        manifest.get("attribute_distribution"), label="attribute_distribution"
    )
    for name, expected_ids in (
        ("source_train", source_ids),
        ("train", train_ids),
        ("val", val_ids),
    ):
        distribution = _mapping(distributions.get(name), label=f"distribution.{name}")
        if _integer(
            distribution.get("sample_count"),
            label=f"distribution.{name}.sample_count",
        ) != len(expected_ids):
            raise EviSIRSTV2DataError(
                f"manifest {name} attribute-distribution count differs"
            )

    validation = _mapping(manifest.get("validation"), label="validation")
    for key in (
        "train_val_disjoint",
        "train_val_union_equals_frozen_train",
        "group_disjoint",
        "test_was_not_accessed",
    ):
        if validation.get(key) is not True:
            raise EviSIRSTV2DataError(f"manifest validation.{key} is not true")
    test_access = _mapping(manifest.get("test_access"), label="test_access")
    expected_test_access_fields = {
        "test_index_opened",
        "test_image_opened",
        "test_mask_opened",
        "test_used_for_split_or_attributes",
    }
    if set(test_access) != expected_test_access_fields or any(
        value is not False for value in test_access.values()
    ):
        raise EviSIRSTV2DataError("manifest declares test data access")
    grouping = _mapping(manifest.get("grouping"), label="grouping")
    grouping_mode = grouping.get("mode")
    if grouping_mode not in {"sample_level_fallback", "explicit_group_mapping"}:
        raise EviSIRSTV2DataError("manifest grouping mode is unsupported")
    if grouping_mode == "sample_level_fallback":
        warning = grouping.get("warning")
        if not isinstance(warning, str) or not warning.strip():
            raise EviSIRSTV2DataError("sample-level fallback lacks a warning")
    _integer(grouping.get("group_count"), label="grouping.group_count")

    data_identity = _mapping(manifest.get("data_identity"), label="data_identity")
    if data_identity.get("paths_are_relative_to") != "dataset_root":
        raise EviSIRSTV2DataError("manifest data-identity path basis differs")
    if _integer(
        data_identity.get("relative_file_identifier_count"),
        label="data_identity.relative_file_identifier_count",
    ) != 2 * len(source_ids):
        raise EviSIRSTV2DataError("manifest relative-file count differs")
    for key in (
        "ordered_image_mask_tree_sha256",
        "ordered_sample_audit_records_sha256",
    ):
        value = data_identity.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise EviSIRSTV2DataError(f"manifest data_identity.{key} is invalid")

    observed_data_tree: str | None = None
    if not isinstance(verify_data_tree, bool):
        raise EviSIRSTV2DataError("verify_data_tree must be boolean")
    if verify_data_tree:
        observed_data_tree = _ordered_data_tree_sha256(
            dataset_root=data_root,
            dataset=dataset,
            sample_ids=source_ids,
        )
        if observed_data_tree != data_identity["ordered_image_mask_tree_sha256"]:
            raise EviSIRSTV2DataError(
                "current source-train image/mask tree SHA-256 differs from manifest"
            )

    return V2SplitContract(
        dataset=dataset,
        dataset_root=data_root,
        split_root=artifact_root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_bytes(manifest_content),
        data_tree_sha256=observed_data_tree,
        data_tree_verified=verify_data_tree,
        manifest=manifest,
        source_train_ids=source_ids,
        train_ids=train_ids,
        val_ids=val_ids,
    )


def check_v2_splits(
    *,
    dataset_root: str | Path,
    split_root: str | Path = DEFAULT_SPLIT_ROOT,
    datasets: Sequence[str] = source_protocol.DATASETS,
    verify_data_tree: bool = True,
) -> dict[str, Any]:
    """Read-only validation summary for one or more V2 split artifacts."""

    results: dict[str, Any] = {}
    for dataset in datasets:
        contract = load_v2_split_contract(
            dataset,
            dataset_root=dataset_root,
            split_root=split_root,
            verify_data_tree=verify_data_tree,
        )
        results[dataset] = {
            "status": "verified",
            "source_train_count": len(contract.source_train_ids),
            "train_count": len(contract.train_ids),
            "val_count": len(contract.val_ids),
            "manifest_sha256": contract.manifest_sha256,
            "data_tree_sha256": contract.data_tree_sha256,
            "data_tree_verified": contract.data_tree_verified,
            "test_index_opened": False,
        }
    return {
        "schema": "evisirst_v2_data_split_check/v1",
        "datasets": results,
    }


def _normalization_spec(
    dataset: str,
    *,
    normalization_mode: str,
    normalization_values: Mapping[str, float] | None,
) -> NormalizationSpec:
    if normalization_mode not in NORMALIZATION_MODES:
        raise EviSIRSTV2DataError(
            "R1 supports normalization_mode='legacy' only; no alternate "
            "statistics may be selected implicitly"
        )
    expected = source_protocol.get_legacy_normalization(dataset)
    if (
        not math.isfinite(float(expected["mean"]))
        or not math.isfinite(float(expected["std"]))
        or float(expected["std"]) <= 0.0
    ):
        raise EviSIRSTV2DataError("frozen legacy normalization is invalid")
    if normalization_values is not None:
        if set(normalization_values) != {"mean", "std"}:
            raise EviSIRSTV2DataError(
                "normalization_values must contain exactly mean and std"
            )
        try:
            supplied_mean = float(normalization_values["mean"])
            supplied_std = float(normalization_values["std"])
        except (TypeError, ValueError) as exc:
            raise EviSIRSTV2DataError("normalization values must be numeric") from exc
        if not math.isfinite(supplied_mean) or not math.isfinite(supplied_std):
            raise EviSIRSTV2DataError("normalization values must be finite")
        if supplied_mean != expected["mean"] or supplied_std != expected["std"]:
            raise EviSIRSTV2DataError(
                "R1 explicit normalization values differ from frozen legacy values"
            )
    return NormalizationSpec(
        mode="legacy",
        source="three_dataset_v2_protocol.LEGACY_NORMALIZATION",
        mean=float(expected["mean"]),
        std=float(expected["std"]),
    )


def _require_target_mode(target_mode: str) -> str:
    if target_mode not in TARGET_MODES:
        raise EviSIRSTV2DataError(
            f"target_mode must be explicitly one of {TARGET_MODES}"
        )
    return target_mode


def _unique_sample_file(
    directory: Path, sample_id: str, *, dataset_root: Path
) -> Path:
    candidates = [
        directory / f"{sample_id}{suffix}"
        for suffix in _SUPPORTED_SUFFIXES
        if (directory / f"{sample_id}{suffix}").is_file()
    ]
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise EviSIRSTV2DataError(
            f"expected one regular file for {sample_id!r} in {directory}"
        )
    path = candidates[0].resolve(strict=True)
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise EviSIRSTV2DataError("sample file escapes its data directory") from exc
    return path


def _ordered_data_tree_sha256(
    *, dataset_root: Path, dataset: str, sample_ids: Sequence[str]
) -> str:
    """Reproduce the split manifest's ordered relative-path/file digest."""

    digest = hashlib.sha256()
    directory = dataset_root / dataset
    for sample_id in sample_ids:
        image_path = _unique_sample_file(
            directory / "images", sample_id, dataset_root=dataset_root
        )
        mask_path = _unique_sample_file(
            directory / "masks", sample_id, dataset_root=dataset_root
        )
        image_relative = image_path.relative_to(dataset_root).as_posix()
        mask_relative = mask_path.relative_to(dataset_root).as_posix()
        for value in (
            sample_id,
            image_relative,
            _sha256_file(image_path),
            mask_relative,
            _sha256_file(mask_path),
        ):
            _update_length_prefixed(digest, value)
    return digest.hexdigest()


def _load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("I"), dtype=np.float32)
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle, dtype=np.float32)
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
        raise EviSIRSTV2DataError("image/mask pair must be aligned 2-D arrays")
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise EviSIRSTV2DataError("image/mask pair contains non-finite pixels")
    return image, mask


def _target_and_audit(
    raw_mask: np.ndarray, target_mode: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    minimum = float(raw_mask.min())
    maximum = float(raw_mask.max())
    if minimum < 0.0 or maximum > 255.0:
        raise EviSIRSTV2DataError("mask values must stay in [0, 255]")
    unique = np.unique(raw_mask)
    values = set(unique.tolist())
    if values.issubset({0.0, 1.0}):
        encoding = "binary_0_1"
        nonbinary_pixels = 0
    elif values.issubset({0.0, 255.0}):
        encoding = "binary_0_255"
        nonbinary_pixels = 0
    else:
        encoding = "grayscale_0_255"
        nonbinary_pixels = int(np.count_nonzero((raw_mask != 0) & (raw_mask != 255)))

    binary_crop_mask = np.asarray(raw_mask > 127.0, dtype=np.bool_)
    if target_mode == "binary":
        if encoding == "binary_0_1" and maximum > 0.0:
            raise EviSIRSTV2DataError(
                "binary target_mode uses the fixed raw-mask >127 rule; "
                "a positive 0/1 mask must be rescaled explicitly"
            )
        target = binary_crop_mask.astype(np.float32)
        target_rule = "raw_mask>127"
    else:
        divisor = 1.0 if encoding == "binary_0_1" else 255.0
        target = np.asarray(raw_mask / np.float32(divisor), dtype=np.float32)
        target_rule = "raw_mask/1" if divisor == 1.0 else "raw_mask/255"
    audit = {
        "encoding": encoding,
        "minimum": minimum,
        "maximum": maximum,
        "unique_value_count": int(unique.size),
        "nonbinary_pixel_count": nonbinary_pixels,
        "target_mode": target_mode,
        "target_rule": target_rule,
        "positive_crop_rule": "raw_mask>127",
        "positive_crop_pixel_count": int(binary_crop_mask.sum()),
    }
    return target, binary_crop_mask, audit


def _pad(array: np.ndarray, height: int, width: int) -> np.ndarray:
    old_height, old_width = array.shape
    if old_height > height or old_width > width:
        raise EviSIRSTV2DataError("padding target is smaller than input")
    return np.pad(
        array,
        ((0, height - old_height), (0, width - old_width)),
        mode="constant",
    )


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _transform_plan(
    *,
    dataset: str,
    sample_id: str,
    run_seed: int,
    epoch: int,
    occurrence: int,
    height: int,
    width: int,
    binary_crop_mask: np.ndarray,
) -> V2TransformPlan:
    size = source_protocol.PATCH_SIZE
    padded_height = max(height, size)
    padded_width = max(width, size)
    augmentation_seed = _stable_uint63(
        run_seed,
        dataset,
        epoch,
        f"{dataset}::{sample_id}",
        occurrence,
    )
    rng = random.Random(augmentation_seed)
    padded_crop_mask = _pad(binary_crop_mask, padded_height, padded_width)
    attempts = 0
    while True:
        attempts += 1
        top = rng.randint(0, padded_height - size)
        left = rng.randint(0, padded_width - size)
        unbiased_accept = (
            rng.random() > source_protocol.TRAIN_POSITIVE_CROP_PROBABILITY
        )
        if unbiased_accept or bool(
            np.any(padded_crop_mask[top : top + size, left : left + size])
        ):
            break
        if attempts >= 1_000_000:
            raise EviSIRSTV2DataError("positive-biased crop exceeded safety limit")
    return V2TransformPlan(
        augmentation_seed=augmentation_seed,
        occurrence=occurrence,
        crop_top=top,
        crop_left=left,
        crop_size=size,
        padded_height=padded_height,
        padded_width=padded_width,
        crop_attempts=attempts,
        flip_axis0=rng.random() < 0.5,
        flip_axis1=rng.random() < 0.5,
        transpose=rng.random() < 0.5,
    )


class _V2DatasetBase(Dataset):
    def __init__(
        self,
        dataset: str,
        *,
        dataset_root: str | Path,
        split_root: str | Path,
        target_mode: str,
        normalization_mode: str,
        normalization_values: Mapping[str, float] | None,
        return_metadata: bool,
        verify_data_tree: bool,
    ) -> None:
        super().__init__()
        self.contract = load_v2_split_contract(
            dataset,
            dataset_root=dataset_root,
            split_root=split_root,
            verify_data_tree=verify_data_tree,
        )
        self.dataset_name = self.contract.dataset
        self.dataset_root = self.contract.dataset_root
        self.target_mode = _require_target_mode(target_mode)
        self.normalization_spec = _normalization_spec(
            self.dataset_name,
            normalization_mode=normalization_mode,
            normalization_values=normalization_values,
        )
        self.normalization = {
            "mean": self.normalization_spec.mean,
            "std": self.normalization_spec.std,
        }
        self.return_metadata = bool(return_metadata)

    def _paths(self, sample_id: str) -> tuple[Path, Path]:
        directory = self.dataset_root / self.dataset_name
        return (
            _unique_sample_file(
                directory / "images", sample_id, dataset_root=self.dataset_root
            ),
            _unique_sample_file(
                directory / "masks", sample_id, dataset_root=self.dataset_root
            ),
        )

    def _load(
        self, sample_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        image_path, mask_path = self._paths(sample_id)
        image, raw_mask = _load_pair(image_path, mask_path)
        target, binary_crop_mask, audit = _target_and_audit(
            raw_mask, self.target_mode
        )
        image = (
            image - np.float32(self.normalization_spec.mean)
        ) / np.float32(self.normalization_spec.std)
        return image, target, binary_crop_mask, audit

    def _dataset_metadata(self, split: str) -> dict[str, Any]:
        distributions = self.contract.manifest["attribute_distribution"]
        return {
            "schema": "evisirst_v2_dataset_metadata/v1",
            "dataset": self.dataset_name,
            "split": split,
            "split_manifest_sha256": self.contract.manifest_sha256,
            "data_tree_sha256": self.contract.data_tree_sha256,
            "data_tree_verified": self.contract.data_tree_verified,
            "split_seed": self.contract.manifest["seeds"]["split_seed"],
            "target_mode": self.target_mode,
            "binary_crop_rule": "raw_mask>127",
            "normalization": self.normalization_spec.as_dict(),
            "grayscale_mask_audit_from_manifest": {
                "mask_encoding_histogram": dict(
                    distributions[split]["mask_encoding_histogram"]
                ),
                "nonbinary_mask_image_count": distributions[split][
                    "nonbinary_mask_image_count"
                ],
                "nonbinary_mask_pixel_count": distributions[split][
                    "nonbinary_mask_pixel_count"
                ],
            },
            "test_index_opened": False,
        }


class EviSIRSTV2TrainDataset(_V2DatasetBase):
    """V2 train membership with run/epoch/sample/occurrence augmentation."""

    def __init__(
        self,
        dataset: str,
        *,
        dataset_root: str | Path,
        target_mode: str,
        run_seed: int,
        split_root: str | Path = DEFAULT_SPLIT_ROOT,
        normalization_mode: str = "legacy",
        normalization_values: Mapping[str, float] | None = None,
        return_metadata: bool = False,
        verify_data_tree: bool = True,
    ) -> None:
        if isinstance(run_seed, bool) or not isinstance(run_seed, int):
            raise EviSIRSTV2DataError("run_seed must be an integer")
        super().__init__(
            dataset,
            dataset_root=dataset_root,
            split_root=split_root,
            target_mode=target_mode,
            normalization_mode=normalization_mode,
            normalization_values=normalization_values,
            return_metadata=return_metadata,
            verify_data_tree=verify_data_tree,
        )
        self.sample_ids = self.contract.train_ids
        self.run_seed = run_seed
        self.epoch = 0
        self.metadata = self._dataset_metadata("train")
        self.metadata.update(
            {
                "run_seed": run_seed,
                "augmentation_version": AUGMENTATION_VERSION,
                "occurrence_contract": (
                    "integer index implies occurrence=0; a sampler may emit "
                    "(index, occurrence) to diversify repeated samples"
                ),
            }
        )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise EviSIRSTV2DataError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _decode_item_index(self, index: Any) -> tuple[int, int]:
        occurrence = 0
        if isinstance(index, tuple):
            if len(index) != 2:
                raise EviSIRSTV2DataError(
                    "tuple dataset index must be (sample_index, occurrence)"
                )
            index, occurrence = index
        if isinstance(index, bool) or not isinstance(index, int):
            raise EviSIRSTV2DataError("sample index must be an integer")
        if isinstance(occurrence, bool) or not isinstance(occurrence, int):
            raise EviSIRSTV2DataError("occurrence must be an integer")
        if not 0 <= index < len(self.sample_ids) or occurrence < 0:
            raise EviSIRSTV2DataError("sample index/occurrence is out of range")
        return index, occurrence

    def __getitem__(self, index: Any) -> Any:
        position, occurrence = self._decode_item_index(index)
        sample_id = self.sample_ids[position]
        image, target, binary_crop_mask, audit = self._load(sample_id)
        original_height, original_width = image.shape
        plan = _transform_plan(
            dataset=self.dataset_name,
            sample_id=sample_id,
            run_seed=self.run_seed,
            epoch=self.epoch,
            occurrence=occurrence,
            height=original_height,
            width=original_width,
            binary_crop_mask=binary_crop_mask,
        )
        image = _pad(image, plan.padded_height, plan.padded_width)
        target = _pad(target, plan.padded_height, plan.padded_width)
        top, left, size = plan.crop_top, plan.crop_left, plan.crop_size
        image = image[top : top + size, left : left + size]
        target = target[top : top + size, left : left + size]
        if plan.flip_axis0:
            image, target = image[::-1, :], target[::-1, :]
        if plan.flip_axis1:
            image, target = image[:, ::-1], target[:, ::-1]
        if plan.transpose:
            image, target = image.transpose(1, 0), target.transpose(1, 0)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[np.newaxis, :], dtype=np.float32)
        )
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(target[np.newaxis, :], dtype=np.float32)
        )
        if not self.return_metadata:
            return image_tensor, target_tensor
        return {
            "image": image_tensor,
            "mask": target_tensor,
            "dataset_name": self.dataset_name,
            "split": "train",
            "sample_id": sample_id,
            "original_hw": (original_height, original_width),
            "epoch": self.epoch,
            "occurrence": occurrence,
            "run_seed": self.run_seed,
            "augmentation_seed": plan.augmentation_seed,
            "transform_plan": asdict(plan),
            "target_mode": self.target_mode,
            "normalization": self.normalization_spec.as_dict(),
            "mask_audit": audit,
        }


class EviSIRSTV2ValDataset(_V2DatasetBase):
    """V2 validation membership with deterministic original-image padding."""

    def __init__(
        self,
        dataset: str,
        *,
        dataset_root: str | Path,
        target_mode: str,
        split_root: str | Path = DEFAULT_SPLIT_ROOT,
        normalization_mode: str = "legacy",
        normalization_values: Mapping[str, float] | None = None,
        return_metadata: bool = False,
        verify_data_tree: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            dataset_root=dataset_root,
            split_root=split_root,
            target_mode=target_mode,
            normalization_mode=normalization_mode,
            normalization_values=normalization_values,
            return_metadata=return_metadata,
            verify_data_tree=verify_data_tree,
        )
        self.sample_ids = self.contract.val_ids
        self.metadata = self._dataset_metadata("val")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, bool) or not isinstance(index, int):
            raise EviSIRSTV2DataError("validation index must be an integer")
        if not 0 <= index < len(self.sample_ids):
            raise EviSIRSTV2DataError("validation index is out of range")
        sample_id = self.sample_ids[index]
        image, target, _binary_crop_mask, audit = self._load(sample_id)
        height, width = image.shape
        padded_height = _next_multiple(height, source_protocol.PAD_MULTIPLE)
        padded_width = _next_multiple(width, source_protocol.PAD_MULTIPLE)
        image = _pad(image, padded_height, padded_width)
        target = _pad(target, padded_height, padded_width)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[np.newaxis, :], dtype=np.float32)
        )
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(target[np.newaxis, :], dtype=np.float32)
        )
        if not self.return_metadata:
            return image_tensor, target_tensor, (height, width), sample_id
        return {
            "image": image_tensor,
            "mask": target_tensor,
            "dataset_name": self.dataset_name,
            "split": "val",
            "sample_id": sample_id,
            "original_hw": (height, width),
            "padded_hw": (padded_height, padded_width),
            "target_mode": self.target_mode,
            "normalization": self.normalization_spec.as_dict(),
            "mask_audit": audit,
            "random_augmentation_applied": False,
        }


__all__ = [
    "AUGMENTATION_VERSION",
    "DEFAULT_SPLIT_ROOT",
    "EviSIRSTV2DataError",
    "EviSIRSTV2TrainDataset",
    "EviSIRSTV2ValDataset",
    "NORMALIZATION_MODES",
    "NormalizationSpec",
    "TARGET_MODES",
    "V2SplitContract",
    "V2TransformPlan",
    "check_v2_splits",
    "load_v2_split_contract",
]
