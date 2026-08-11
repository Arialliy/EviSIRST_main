"""Public train/test data contract for EviSIRST, including pooled SIRST3.

The three historical source-dataset files remain byte-frozen.  This separate
module adds the SIRST3 joint-training regime without changing that provenance.
SIRST3 is the strict ordered concatenation of the NUAA, NUDT, and IRSTD train
lists; it is not treated as a fourth independent source dataset.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from experiments import three_dataset_v2_protocol as source_protocol


SOURCE_DATASETS = source_protocol.DATASETS
TRAINING_DATASETS = ("SIRST3", *SOURCE_DATASETS)
SIRST3_NORMALIZATION = {
    "mean": 101.06385040283203,
    "std": 34.619606018066406,
}
SIRST3_TRAIN_TREE_SHA256 = (
    "d0dda06a2c8e1f1617c3d2637aa4cde4342f643d55c373bcdccae3b695dd41fe"
)
SIRST3_SPLITS = {
    "train": {
        "count": 1676,
        "file_sha256": (
            "75c32b896b95e29b89edc1f5231f619f275c2b54da0264934e6e0df13d7e7d9a"
        ),
        "ordered_ids_sha256": (
            "66c5a6f43b665e36556a97391c1676e3720b3ae0e72186b4894a9eadb6456355"
        ),
    },
    "test": {
        "count": 1079,
        "file_sha256": (
            "67a0f48b536ea6e2f8c895868c4bcd16c66c7c0a6280fd05ef7cd366d78b8922"
        ),
        "ordered_ids_sha256": (
            "326b4e8470f88b606688ed63606b67fa69f4a9deea2a25597c8cdb4bdf10764e"
        ),
    },
}

_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_SUFFIXES = (".png", ".bmp")


class EviSIRSTDataError(ValueError):
    """A dataset violates the frozen public EviSIRST contract."""


def _require_training_dataset(dataset: str) -> str:
    if dataset not in TRAINING_DATASETS:
        raise EviSIRSTDataError(
            f"training dataset must be one of {TRAINING_DATASETS}, got {dataset!r}"
        )
    return dataset


def _require_source_dataset(dataset: str) -> str:
    if dataset not in SOURCE_DATASETS:
        raise EviSIRSTDataError(
            f"evaluation dataset must be one of {SOURCE_DATASETS}, got {dataset!r}"
        )
    return dataset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_uint63(*parts: Any) -> int:
    """Exact epoch-shuffle seed used by the historical formal trainer."""

    digest = hashlib.sha256()
    for part in parts:
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def _sirst3_index(
    dataset_root: Path,
    split: str,
) -> tuple[list[str], dict[str, str]]:
    if split not in SIRST3_SPLITS:
        raise EviSIRSTDataError(f"unsupported split: {split!r}")
    path = dataset_root / "SIRST3" / "img_idx" / f"{split}_SIRST3.txt"
    if path.is_symlink() or not path.is_file():
        raise EviSIRSTDataError(f"SIRST3 index is not a regular file: {path}")
    content = path.read_bytes()
    expected = SIRST3_SPLITS[split]
    observed_file_sha = hashlib.sha256(content).hexdigest()
    if observed_file_sha != expected["file_sha256"]:
        raise EviSIRSTDataError("SIRST3 index file SHA-256 differs")
    try:
        identifiers = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EviSIRSTDataError(f"SIRST3 index is not UTF-8: {exc}") from exc
    if len(identifiers) != expected["count"] or len(identifiers) != len(
        set(identifiers)
    ):
        raise EviSIRSTDataError("SIRST3 index count or uniqueness differs")
    if any(
        identifier != identifier.strip()
        or _SAMPLE_ID_RE.fullmatch(identifier) is None
        for identifier in identifiers
    ):
        raise EviSIRSTDataError("SIRST3 index contains an unsafe sample ID")
    if source_protocol.ordered_ids_sha256(identifiers) != expected[
        "ordered_ids_sha256"
    ]:
        raise EviSIRSTDataError("SIRST3 ordered-ID SHA-256 differs")

    concatenated: list[str] = []
    source_by_id: dict[str, str] = {}
    for source_dataset in SOURCE_DATASETS:
        source_ids = source_protocol.load_index(
            dataset_root, source_dataset, split
        )
        concatenated.extend(source_ids)
        for sample_id in source_ids:
            if sample_id in source_by_id:
                raise EviSIRSTDataError(
                    f"ambiguous source membership for {sample_id!r}"
                )
            source_by_id[sample_id] = source_dataset
    if concatenated != identifiers:
        raise EviSIRSTDataError(
            f"SIRST3 {split} is not the strict ordered concatenation of "
            f"{SOURCE_DATASETS}"
        )
    return identifiers, source_by_id


def _unique_file(directory: Path, sample_id: str) -> Path:
    candidates = [
        directory / f"{sample_id}{suffix}"
        for suffix in _SUPPORTED_SUFFIXES
        if (directory / f"{sample_id}{suffix}").is_file()
    ]
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise EviSIRSTDataError(
            f"expected one regular data file for {sample_id!r} in {directory}"
        )
    return candidates[0].resolve(strict=True)


def _load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        image_array = np.asarray(image.convert("I"), dtype=np.float32)
    with Image.open(mask_path) as mask:
        mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim > 2:
        mask_array = mask_array[:, :, 0]
    if image_array.ndim != 2 or mask_array.ndim != 2:
        raise EviSIRSTDataError("image and mask must be two-dimensional")
    if image_array.shape != mask_array.shape:
        raise EviSIRSTDataError(
            f"image/mask dimensions differ: {image_path} / {mask_path}"
        )
    if not np.isfinite(image_array).all() or not np.isfinite(mask_array).all():
        raise EviSIRSTDataError("image or mask contains non-finite pixels")
    return image_array, mask_array


def _pad(array: np.ndarray, height: int, width: int) -> np.ndarray:
    old_height, old_width = array.shape
    if old_height > height or old_width > width:
        raise EviSIRSTDataError("padding target is smaller than the image")
    return np.pad(
        array,
        ((0, height - old_height), (0, width - old_width)),
        mode="constant",
    )


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def normalization_for(dataset: str) -> dict[str, float]:
    _require_training_dataset(dataset)
    if dataset == "SIRST3":
        return dict(SIRST3_NORMALIZATION)
    return source_protocol.get_legacy_normalization(dataset)


def _transform_plan(
    *,
    dataset: str,
    source_dataset: str,
    sample_id: str,
    epoch: int,
    height: int,
    width: int,
    mask_any: np.ndarray,
) -> source_protocol.StatelessTransformPlan:
    patch_size = source_protocol.PATCH_SIZE
    padded_height = max(height, patch_size)
    padded_width = max(width, patch_size)
    namespaced_id = f"{source_dataset}::{sample_id}"
    augmentation_seed = source_protocol.stable_sha256_uint64(
        source_protocol.PROTOCOL_SEED,
        dataset,
        epoch,
        namespaced_id,
    )
    rng = random.Random(augmentation_seed)
    attempts = 0
    while True:
        attempts += 1
        top = rng.randint(0, padded_height - patch_size)
        left = rng.randint(0, padded_width - patch_size)
        if (
            rng.random() > source_protocol.TRAIN_POSITIVE_CROP_PROBABILITY
            or bool(np.any(mask_any[top : top + patch_size, left : left + patch_size]))
        ):
            break
        if attempts >= 1_000_000:
            raise EviSIRSTDataError("positive-biased crop exceeded safety limit")
    return source_protocol.StatelessTransformPlan(
        augmentation_seed=augmentation_seed,
        crop_top=top,
        crop_left=left,
        crop_size=patch_size,
        padded_height=padded_height,
        padded_width=padded_width,
        crop_attempts=attempts,
        flip_axis0=rng.random() < 0.5,
        flip_axis1=rng.random() < 0.5,
        transpose=rng.random() < 0.5,
    )


class EviSIRSTTrainDataset(Dataset):
    """Train-only frozen split with source-aware stateless augmentation."""

    def __init__(
        self,
        dataset: str,
        *,
        dataset_root: str | Path,
        patch_size: int = source_protocol.PATCH_SIZE,
        seed: int = source_protocol.PROTOCOL_SEED,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_name = _require_training_dataset(dataset)
        if patch_size != source_protocol.PATCH_SIZE or seed != 42:
            raise EviSIRSTDataError("formal patch size/seed must be 256/42")
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.patch_size = patch_size
        self.seed = seed
        self.return_metadata = bool(return_metadata)
        if self.dataset_name == "SIRST3":
            self.sample_ids, self.source_by_id = _sirst3_index(
                self.dataset_root, "train"
            )
        else:
            self.sample_ids = source_protocol.load_index(
                self.dataset_root, self.dataset_name, "train"
            )
            self.source_by_id = {
                sample_id: self.dataset_name for sample_id in self.sample_ids
            }
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = normalization_for(self.dataset_name)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise EviSIRSTDataError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _paths(self, index: int) -> tuple[Path, Path, str, str]:
        sample_id = self.sample_ids[index]
        source_dataset = self.source_by_id[sample_id]
        if self.dataset_name == "SIRST3":
            directory = self.dataset_root / "SIRST3"
            return (
                _unique_file(directory / "images", sample_id),
                _unique_file(directory / "masks", sample_id),
                sample_id,
                source_dataset,
            )
        sample = source_protocol.resolve_sample(
            self.dataset_root,
            self.dataset_name,
            sample_id,
            split="train",
            known_ids=self._known_ids,
        )
        return sample.image_path, sample.mask_path, sample_id, source_dataset

    def __getitem__(self, index: int) -> Any:
        image_path, mask_path, sample_id, source_dataset = self._paths(index)
        image, raw_mask = _load_pair(image_path, mask_path)
        original_height, original_width = image.shape
        image = (image - np.float32(self.normalization["mean"])) / np.float32(
            self.normalization["std"]
        )
        mask = raw_mask / np.float32(255.0)
        plan = _transform_plan(
            dataset=self.dataset_name,
            source_dataset=source_dataset,
            sample_id=sample_id,
            epoch=self.epoch,
            height=original_height,
            width=original_width,
            mask_any=mask > 0,
        )
        image = _pad(image, plan.padded_height, plan.padded_width)
        mask = _pad(mask, plan.padded_height, plan.padded_width)
        top, left, size = plan.crop_top, plan.crop_left, plan.crop_size
        image = image[top : top + size, left : left + size]
        mask = mask[top : top + size, left : left + size]
        if plan.flip_axis0:
            image, mask = image[::-1, :], mask[::-1, :]
        if plan.flip_axis1:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if plan.transpose:
            image, mask = image.transpose(1, 0), mask.transpose(1, 0)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[np.newaxis, :], dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask[np.newaxis, :], dtype=np.float32)
        )
        if not self.return_metadata:
            return image_tensor, mask_tensor
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset_name": self.dataset_name,
            "source_dataset": source_dataset,
            "sample_id": sample_id,
            "namespaced_sample_id": f"{source_dataset}::{sample_id}",
            "original_hw": (original_height, original_width),
            "epoch": self.epoch,
            "augmentation_seed": plan.augmentation_seed,
            "transform_plan": asdict(plan),
        }


class EviSIRSTTestDataset(Dataset):
    """One held-out source test split using the training-regime normalization."""

    def __init__(
        self,
        training_dataset: str,
        evaluation_dataset: str,
        *,
        dataset_root: str | Path,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.training_dataset = _require_training_dataset(training_dataset)
        self.evaluation_dataset = _require_source_dataset(evaluation_dataset)
        self.dataset_root = Path(dataset_root).resolve(strict=True)
        self.return_metadata = bool(return_metadata)
        self.sample_ids = source_protocol.load_index(
            self.dataset_root, self.evaluation_dataset, "test"
        )
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = normalization_for(self.training_dataset)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Any:
        sample = source_protocol.resolve_sample(
            self.dataset_root,
            self.evaluation_dataset,
            self.sample_ids[index],
            split="test",
            known_ids=self._known_ids,
        )
        image, raw_mask = _load_pair(sample.image_path, sample.mask_path)
        height, width = image.shape
        image = (image - np.float32(self.normalization["mean"])) / np.float32(
            self.normalization["std"]
        )
        mask = raw_mask / np.float32(255.0)
        padded_height = _next_multiple(height, source_protocol.PAD_MULTIPLE)
        padded_width = _next_multiple(width, source_protocol.PAD_MULTIPLE)
        image = _pad(image, padded_height, padded_width)
        mask = _pad(mask, padded_height, padded_width)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[np.newaxis, :], dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask[np.newaxis, :], dtype=np.float32)
        )
        if not self.return_metadata:
            return image_tensor, mask_tensor, (height, width), sample.sample_id
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "training_dataset": self.training_dataset,
            "evaluation_dataset": self.evaluation_dataset,
            "sample_id": sample.sample_id,
            "original_hw": (height, width),
            "correction_applied": sample.correction_applied,
            "correction_id": sample.correction_id,
        }


def sirst3_contract() -> dict[str, Any]:
    return {
        "training_dataset": "SIRST3",
        "source_datasets": list(SOURCE_DATASETS),
        "strict_ordered_concatenation": True,
        "splits": SIRST3_SPLITS,
        "source_ranges": {
            "train": {
                "NUAA-SIRST": [0, 213],
                "NUDT-SIRST": [213, 876],
                "IRSTD-1K": [876, 1676],
            },
            "test": {
                "NUAA-SIRST": [0, 214],
                "NUDT-SIRST": [214, 878],
                "IRSTD-1K": [878, 1079],
            },
        },
        "normalization": dict(SIRST3_NORMALIZATION),
        "normalization_recomputed": False,
        "preflight_train_image_mask_tree_sha256": SIRST3_TRAIN_TREE_SHA256,
        "target_test_used_for_training_or_selection": False,
    }


__all__ = [
    "EviSIRSTDataError",
    "EviSIRSTTestDataset",
    "EviSIRSTTrainDataset",
    "SIRST3_NORMALIZATION",
    "SIRST3_SPLITS",
    "SIRST3_TRAIN_TREE_SHA256",
    "SOURCE_DATASETS",
    "TRAINING_DATASETS",
    "normalization_for",
    "sirst3_contract",
    "stable_uint63",
]
