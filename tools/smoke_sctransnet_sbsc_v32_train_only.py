#!/usr/bin/env python3
"""Run the immutable Stage-C train-only smoke for SCTransNet C3-SBSC V3.2.

This entry point deliberately has no validation/test dataset constructor and
no scheduling knobs.  It consumes only the first 64 IDs of the frozen
IRSTD-1K *training* split, trains a fresh seed-42 V3.2 model for five epochs,
and emits one strict-JSON evidence artifact.  The artifact is diagnostic only:
it is never a checkpoint and it never makes a benchmark-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


_ENTRY_SOURCE = Path(__file__)
if _ENTRY_SOURCE.is_symlink() or not _ENTRY_SOURCE.is_file():
    raise RuntimeError("Stage-C entry source must be a regular non-symlink file")
_ENTRY_SOURCE = _ENTRY_SOURCE.resolve(strict=True)
PROJECT_ROOT = _ENTRY_SOURCE.parents[1]
if PROJECT_ROOT.is_symlink() or not PROJECT_ROOT.is_dir():
    raise RuntimeError("Stage-C repository root must be a regular directory")
# A caller-controlled PYTHONPATH may already contain this repository *after*
# another checkout that also provides ``model``.  Merely checking membership
# would then preserve the hostile checkout's precedence.  Canonicalize every
# entry, remove all aliases of the verified local root, and install exactly one
# authoritative entry at position zero before importing any project module.
_PROJECT_ROOT_TEXT = str(PROJECT_ROOT)
_filtered_sys_path: list[str] = []
for _entry in sys.path:
    if not isinstance(_entry, str):
        _filtered_sys_path.append(_entry)
        continue
    try:
        _canonical_entry = Path(_entry or os.curdir).expanduser().resolve(
            strict=False
        )
    except (OSError, RuntimeError):
        _canonical_entry = None
    if _entry == _PROJECT_ROOT_TEXT or _canonical_entry == PROJECT_ROOT:
        continue
    _filtered_sys_path.append(_entry)
sys.path[:] = [_PROJECT_ROOT_TEXT, *_filtered_sys_path]
_FROZEN_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# This is deliberately set before importing torch, hence before this entry
# point can create a CUDA context.  A conflicting inherited value is rejected
# again before the first CUDA availability/device query.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _FROZEN_CUBLAS_WORKSPACE_CONFIG)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from experiments import evisirst_v2_data as v2_data


SCHEMA = "sctransnet_sbsc_v32_stage_c_train_only_smoke/v1"
SOURCE_SET_SCHEMA = "sctransnet_sbsc_v32_stage_c_train_only_source_set/v1"
COMPACT_DIAGNOSTICS_SCHEMA = (
    "sctransnet_sbsc_v32/compact_projection_diagnostics/v1"
)
CUDA_MEDIAN_ADAPTER_SCHEMA = (
    "sctransnet_sbsc_v32/cuda_strict_median_value_adapter/v1"
)
RUNTIME_INTEGRATION_SCHEMA = "sctransnet_sbsc_v32/runtime_integration/v1"
EXPECTED_RUNTIME_INTEGRATION = {
    "schema": RUNTIME_INTEGRATION_SCHEMA,
    "cuda_median_adapter": CUDA_MEDIAN_ADAPTER_SCHEMA,
}
EXPECTED_V31_SOLVER_SOURCE_SHA256 = (
    "b2d1e3f97607b305551a0602076605968041878eafd822c6f3335840ea725a3a"
)
DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
RUN_SEED = 42
EPOCHS = 5
SAMPLE_COUNT = 64
BATCH_SIZE = 16
WORKERS = 0
INPUT_SIZE = 256
ROUTER_LOSS_WEIGHT = 1.0
BASE_LR = 1e-3
FORMAL_WARMUP_EPOCHS = 10
ROLE_NAMES = ("C", "H", "B")
MAX_MASS_LIMIT = 0.999
GAIN_MIN = 0.0
GAIN_MAX = 0.25
GAIN_SUFFIX = "raw_dual_risk_level_gain"
COMPACT_COUNTER_FIELDS = (
    "projection_rows",
    "solver_attempt_rows",
    "solver_accepted_rows",
    "solver_fallback_rows",
    "emission_checked_rows",
    "emission_accepted_rows",
    "emission_fallback_rows",
)


class StageCTrainOnlySmokeError(RuntimeError):
    """The immutable Stage-C protocol or its evidence contract was violated."""


@dataclass(frozen=True)
class TrainOnlyContract:
    dataset_root: Path
    split_root: Path
    manifest_sha256: str
    train_split_sha256: str
    train_ordered_ids_sha256: str
    source_train_index_sha256: str
    source_train_ordered_ids_sha256: str
    selected_ids_sha256: str
    source_train_count: int
    train_split_count: int
    selected_ids: tuple[str, ...]
    source_index_relative_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT / "splits" / "v2",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    return _sha256_bytes(
        json.dumps(
            list(values), ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    )


def _strict_json_clone(value: Any, *, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StageCTrainOnlySmokeError(
            f"{label} is not finite strict-JSON data"
        ) from exc


def _require_regular_bytes(path: Path, *, root: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise StageCTrainOnlySmokeError(f"{label} is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StageCTrainOnlySmokeError(f"{label} escapes its declared root") from exc
    return resolved.read_bytes()


def _decode_ids(
    content: bytes, *, label: str, canonical_newline: bool
) -> tuple[str, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StageCTrainOnlySmokeError(f"{label} is not UTF-8") from exc
    values = tuple(text.splitlines())
    if not values or len(values) != len(set(values)):
        raise StageCTrainOnlySmokeError(f"{label} is empty or contains duplicates")
    if canonical_newline and content != ("\n".join(values) + "\n").encode("utf-8"):
        raise StageCTrainOnlySmokeError(f"{label} is not canonical one-ID-per-line text")
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(not value or any(character not in safe for character in value) for value in values):
        raise StageCTrainOnlySmokeError(f"{label} contains an unsafe sample ID")
    return values


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageCTrainOnlySmokeError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StageCTrainOnlySmokeError(f"{label} must be an integer >= {minimum}")
    return value


def load_train_only_contract(
    dataset_root: str | Path, split_root: str | Path
) -> TrainOnlyContract:
    """Validate only manifest/train/source-train files; never resolve val/test."""

    raw_dataset_root = Path(dataset_root)
    raw_split_root = Path(split_root)
    if raw_dataset_root.is_symlink() or not raw_dataset_root.is_dir():
        raise StageCTrainOnlySmokeError("dataset-root is not a regular directory")
    if raw_split_root.is_symlink() or not raw_split_root.is_dir():
        raise StageCTrainOnlySmokeError("split-root is not a regular directory")
    data_root = raw_dataset_root.resolve(strict=True)
    artifact_root = raw_split_root.resolve(strict=True)
    split_directory = artifact_root / DATASET
    if split_directory.is_symlink() or not split_directory.is_dir():
        raise StageCTrainOnlySmokeError("IRSTD-1K split directory is not regular")
    split_directory = split_directory.resolve(strict=True)
    try:
        split_directory.relative_to(artifact_root)
    except ValueError as exc:
        raise StageCTrainOnlySmokeError("split directory escapes split-root") from exc

    manifest_content = _require_regular_bytes(
        split_directory / "manifest.json",
        root=artifact_root,
        label="split manifest",
    )
    train_content = _require_regular_bytes(
        split_directory / "train.txt",
        root=artifact_root,
        label="train split",
    )
    try:
        manifest = json.loads(manifest_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageCTrainOnlySmokeError("split manifest is not valid UTF-8 JSON") from exc
    manifest = _mapping(manifest, label="split manifest")
    if manifest.get("schema") != v2_data.split_protocol.SCHEMA:
        raise StageCTrainOnlySmokeError("split manifest schema differs")
    if manifest.get("dataset") != DATASET:
        raise StageCTrainOnlySmokeError("split manifest dataset is not IRSTD-1K")
    test_access = _mapping(manifest.get("test_access"), label="manifest.test_access")
    expected_test_fields = {
        "test_index_opened",
        "test_image_opened",
        "test_mask_opened",
        "test_used_for_split_or_attributes",
    }
    if set(test_access) != expected_test_fields or any(
        value is not False for value in test_access.values()
    ):
        raise StageCTrainOnlySmokeError("split manifest declares test access")
    validation = _mapping(manifest.get("validation"), label="manifest.validation")
    if validation.get("test_was_not_accessed") is not True:
        raise StageCTrainOnlySmokeError("split manifest lacks test non-access evidence")

    outputs = _mapping(manifest.get("outputs"), label="manifest.outputs")
    train_output = _mapping(outputs.get("train"), label="manifest.outputs.train")
    train_ids = _decode_ids(
        train_content, label="train split", canonical_newline=True
    )
    if train_output.get("file_sha256") != _sha256_bytes(train_content):
        raise StageCTrainOnlySmokeError("train split SHA-256 differs from manifest")
    if train_output.get("ordered_ids_sha256") != _ordered_ids_sha256(train_ids):
        raise StageCTrainOnlySmokeError("train ordered-ID SHA-256 differs")
    if _integer(
        train_output.get("sample_count"),
        label="manifest.outputs.train.sample_count",
        minimum=1,
    ) != len(train_ids):
        raise StageCTrainOnlySmokeError("train split count differs from manifest")

    source = _mapping(manifest.get("source_index"), label="manifest.source_index")
    if source.get("split") != "train":
        raise StageCTrainOnlySmokeError("source index is not explicitly train")
    relative = source.get("relative_path")
    if not isinstance(relative, str):
        raise StageCTrainOnlySmokeError("source train-index path is malformed")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise StageCTrainOnlySmokeError("source train-index path is unsafe")
    lowered_parts = tuple(part.lower() for part in relative_path.parts)
    if any("test" in part or part == "val" for part in lowered_parts):
        raise StageCTrainOnlySmokeError("source path is not strictly train-only")
    source_content = _require_regular_bytes(
        data_root / relative_path,
        root=data_root,
        label="source train index",
    )
    source_ids = _decode_ids(
        source_content, label="source train index", canonical_newline=False
    )
    if source.get("file_sha256") != _sha256_bytes(source_content):
        raise StageCTrainOnlySmokeError("source train-index SHA-256 differs")
    if source.get("ordered_ids_sha256") != _ordered_ids_sha256(source_ids):
        raise StageCTrainOnlySmokeError("source train ordered-ID SHA-256 differs")
    if _integer(
        source.get("sample_count"),
        label="manifest.source_index.sample_count",
        minimum=1,
    ) != len(source_ids):
        raise StageCTrainOnlySmokeError("source train-index count differs")
    train_set = set(train_ids)
    if not train_set.issubset(source_ids):
        raise StageCTrainOnlySmokeError("train split is not a source-train subset")
    if tuple(value for value in source_ids if value in train_set) != train_ids:
        raise StageCTrainOnlySmokeError("train split does not preserve source order")
    if len(train_ids) < SAMPLE_COUNT:
        raise StageCTrainOnlySmokeError("train split has fewer than 64 samples")
    selected = train_ids[:SAMPLE_COUNT]
    return TrainOnlyContract(
        dataset_root=data_root,
        split_root=artifact_root,
        manifest_sha256=_sha256_bytes(manifest_content),
        train_split_sha256=_sha256_bytes(train_content),
        train_ordered_ids_sha256=_ordered_ids_sha256(train_ids),
        source_train_index_sha256=_sha256_bytes(source_content),
        source_train_ordered_ids_sha256=_ordered_ids_sha256(source_ids),
        selected_ids_sha256=_ordered_ids_sha256(selected),
        source_train_count=len(source_ids),
        train_split_count=len(train_ids),
        selected_ids=selected,
        source_index_relative_path=relative_path.as_posix(),
    )


class _First64TrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """The V2 train transform restricted to the frozen first-64 membership."""

    def __init__(self, contract: TrainOnlyContract) -> None:
        super().__init__()
        self.dataset_root = contract.dataset_root
        self.sample_ids = contract.selected_ids
        self.epoch = 0
        self.normalization = v2_data._normalization_spec(
            DATASET,
            normalization_mode="legacy",
            normalization_values=None,
        )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= EPOCHS:
            raise StageCTrainOnlySmokeError("dataset epoch is outside Stage C")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise StageCTrainOnlySmokeError("training sample index is invalid")
        sample_id = self.sample_ids[index]
        directory = self.dataset_root / DATASET
        image_path = v2_data._unique_sample_file(
            directory / "images", sample_id, dataset_root=self.dataset_root
        )
        mask_path = v2_data._unique_sample_file(
            directory / "masks", sample_id, dataset_root=self.dataset_root
        )
        image, raw_mask = v2_data._load_pair(image_path, mask_path)
        target, binary_crop_mask, _audit = v2_data._target_and_audit(
            raw_mask, TARGET_MODE
        )
        image = (
            image - np.float32(self.normalization.mean)
        ) / np.float32(self.normalization.std)
        original_height, original_width = image.shape
        plan = v2_data._transform_plan(
            dataset=DATASET,
            sample_id=sample_id,
            run_seed=RUN_SEED,
            epoch=self.epoch,
            occurrence=0,
            height=original_height,
            width=original_width,
            binary_crop_mask=binary_crop_mask,
        )
        image = v2_data._pad(image, plan.padded_height, plan.padded_width)
        target = v2_data._pad(target, plan.padded_height, plan.padded_width)
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
        expected = (1, INPUT_SIZE, INPUT_SIZE)
        if tuple(image_tensor.shape) != expected or tuple(target_tensor.shape) != expected:
            raise StageCTrainOnlySmokeError("training transform did not emit 1x256x256")
        if not bool(torch.isfinite(image_tensor).all()) or not bool(
            torch.isfinite(target_tensor).all()
        ):
            raise StageCTrainOnlySmokeError("training sample is non-finite")
        if bool(((target_tensor < 0.0) | (target_tensor > 1.0)).any()):
            raise StageCTrainOnlySmokeError("training target is outside [0,1]")
        return image_tensor, target_tensor


def build_train_only_dataset(contract: TrainOnlyContract) -> Dataset[Any]:
    return _First64TrainDataset(contract)


def _stable_epoch_seed(epoch: int) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= EPOCHS:
        raise StageCTrainOnlySmokeError("shuffle epoch is outside Stage C")
    digest = hashlib.sha256()
    for value in (SCHEMA, RUN_SEED, DATASET, "shuffle", epoch):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def epoch_order(epoch: int) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_epoch_seed(epoch))
    return tuple(
        int(value)
        for value in torch.randperm(SAMPLE_COUNT, generator=generator).tolist()
    )


def _load_core() -> ModuleType:
    try:
        core = importlib.import_module("experiments.sctransnet_sbsc_v32")
    except (ImportError, ModuleNotFoundError) as exc:
        raise StageCTrainOnlySmokeError("V3.2 core is unavailable") from exc
    required = (
        "build_sctransnet_sbsc_v32_method",
        "validate_sctransnet_sbsc_v32",
        "capture_c3_v32_training_router",
        "tri_router_supervision_loss",
        "build_tri_router_targets_v32",
        "project_sbsc_v32_constraints_",
        "structurally_inactive_parameter_names",
        "_estimate_c3_v31_support_v32",
        "validate_sbsc_v32_runtime_integration",
    )
    if any(not callable(getattr(core, name, None)) for name in required):
        raise StageCTrainOnlySmokeError("V3.2 core API is incomplete")
    source = Path(core.__file__ or "")
    if source.is_symlink() or not source.is_file():
        raise StageCTrainOnlySmokeError("V3.2 core source is not regular")
    expected_source = (
        PROJECT_ROOT / "experiments" / "sctransnet_sbsc_v32.py"
    ).resolve(strict=True)
    try:
        resolved_source = source.resolve(strict=True)
        resolved_source.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise StageCTrainOnlySmokeError("V3.2 core source escapes repository") from exc
    if resolved_source != expected_source:
        raise StageCTrainOnlySmokeError(
            "V3.2 core source is not the canonical repository module"
        )
    if (
        type(getattr(core, "SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA", None))
        is not str
        or core.SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA
        != CUDA_MEDIAN_ADAPTER_SCHEMA
    ):
        raise StageCTrainOnlySmokeError(
            "V3.2 CUDA median adapter schema differs"
        )
    if (
        getattr(core, "EXPECTED_V31_SOLVER_SOURCE_SHA256", None)
        != EXPECTED_V31_SOLVER_SOURCE_SHA256
        or getattr(core, "V31_SOLVER_SOURCE_SHA256", None)
        != EXPECTED_V31_SOLVER_SOURCE_SHA256
    ):
        raise StageCTrainOnlySmokeError(
            "V3.2 frozen V3.1 source contract differs"
        )
    try:
        integration = core.validate_sbsc_v32_runtime_integration()
    except Exception as exc:
        raise StageCTrainOnlySmokeError(
            "V3.2 runtime integration validation failed"
        ) from exc
    if type(integration) is not dict or integration != EXPECTED_RUNTIME_INTEGRATION:
        raise StageCTrainOnlySmokeError(
            "V3.2 runtime integration contract differs"
        )
    return core


def _require_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise StageCTrainOnlySmokeError("device is malformed") from exc
    if device.type not in {"cpu", "cuda"}:
        raise StageCTrainOnlySmokeError("Stage C supports only cpu or cuda")
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _FROZEN_CUBLAS_WORKSPACE_CONFIG:
            raise StageCTrainOnlySmokeError(
                "CUDA requires frozen CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        if not torch.cuda.is_available():
            raise StageCTrainOnlySmokeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise StageCTrainOnlySmokeError("requested CUDA device does not exist")
        return torch.device("cuda", index)
    return torch.device("cpu")


def _configure_runtime_determinism(device: torch.device) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _FROZEN_CUBLAS_WORKSPACE_CONFIG:
        raise StageCTrainOnlySmokeError(
            "deterministic runtime requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RUN_SEED)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise StageCTrainOnlySmokeError("model state is malformed")
        tensor = value.detach().cpu().contiguous()
        descriptor = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        # Flatten before the dtype reinterpretation: PyTorch rejects a direct
        # 0-D Long/Float -> uint8 view even though the underlying bytes are
        # valid.  Reinterpreting a contiguous byte view preserves every bit
        # for bool, integer, floating, and BF16 tensors without numeric casts.
        raw = (
            b""
            if tensor.numel() == 0
            else tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _build_fresh_model(core: ModuleType) -> tuple[nn.Module, dict[str, Any]]:
    result = core.build_sctransnet_sbsc_v32_method(
        method="sbsc_v32",
        dataset=DATASET,
        architecture_seed=ARCHITECTURE_SEED,
        training=True,
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], nn.Module)
        or not isinstance(result[1], Mapping)
    ):
        raise StageCTrainOnlySmokeError("V3.2 builder returned malformed data")
    model, metadata = result
    if (
        metadata.get("cuda_strict_deterministic_median_adapter")
        != CUDA_MEDIAN_ADAPTER_SCHEMA
    ):
        raise StageCTrainOnlySmokeError(
            "V3.2 builder CUDA median adapter schema differs"
        )
    core.validate_sctransnet_sbsc_v32(model, require_zero_gain=True)
    inactive = core.structurally_inactive_parameter_names(model)
    if isinstance(inactive, (str, bytes)):
        raise StageCTrainOnlySmokeError("inactive parameter contract is malformed")
    named = dict(model.named_parameters())
    for name in tuple(inactive):
        if name not in named:
            raise StageCTrainOnlySmokeError("inactive parameter is absent")
        named[name].requires_grad_(False)
    model.train()
    model.mode = "train"
    selected_metadata = {
        key: metadata.get(key)
        for key in (
            "schema",
            "model",
            "method",
            "dataset",
            "architecture_seed",
            "state_key_count",
            "parameter_count",
            "state_sha256",
            "shared_state_sha256",
            "parent_checkpoint",
            "warm_start_used",
            "predecessor_checkpoint_used",
            "test_split_accessed",
            "cuda_strict_deterministic_median_adapter",
        )
        if key in metadata
    }
    return model, _strict_json_clone(selected_metadata, label="builder metadata")


def _extract_compact_diagnostics(record: Mapping[str, Any]) -> dict[str, int]:
    compact = record.get("compact_diagnostics")
    if not isinstance(compact, Mapping):
        raise StageCTrainOnlySmokeError(
            "router capture omits fail-closed compact solver/emission diagnostics"
        )
    allowed = set(COMPACT_COUNTER_FIELDS) | {"schema"}
    if set(compact) - allowed:
        raise StageCTrainOnlySmokeError("compact diagnostics contains unknown fields")
    if "schema" in compact and compact["schema"] != COMPACT_DIAGNOSTICS_SCHEMA:
        raise StageCTrainOnlySmokeError("compact diagnostics schema differs")
    values = {
        field: _integer(compact.get(field), label=f"compact.{field}")
        for field in COMPACT_COUNTER_FIELDS
    }
    if values["projection_rows"] <= 0:
        raise StageCTrainOnlySmokeError("compact diagnostics has no projection rows")
    if (
        values["solver_accepted_rows"] + values["solver_fallback_rows"]
        != values["solver_attempt_rows"]
        or values["solver_attempt_rows"] > values["projection_rows"]
    ):
        raise StageCTrainOnlySmokeError("solver compact counters are inconsistent")
    if (
        values["emission_accepted_rows"] + values["emission_fallback_rows"]
        != values["emission_checked_rows"]
        or values["emission_checked_rows"] != values["projection_rows"]
    ):
        raise StageCTrainOnlySmokeError("emission compact counters are inconsistent")
    return values


def _capture_measurements(
    *,
    core: ModuleType,
    capture: Any,
    masks: torch.Tensor,
    detached_prediction: torch.Tensor,
) -> dict[str, Any]:
    records = getattr(capture, "records", None)
    if not isinstance(records, list) or len(records) != 1:
        raise StageCTrainOnlySmokeError("one batch must produce exactly one capture")
    record = records[0]
    if not isinstance(record, Mapping):
        raise StageCTrainOnlySmokeError("router capture record is malformed")
    logits_values = record.get("logits")
    if not isinstance(logits_values, tuple) or len(logits_values) != 4:
        raise StageCTrainOnlySmokeError("capture must contain four router levels")
    ce: dict[str, list[float]] = {role: [] for role in ROLE_NAMES}
    max_mass: dict[str, list[float]] = {role: [] for role in ROLE_NAMES}
    all_valid_ce: list[float] = []
    for logits in logits_values:
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 4
            or logits.shape[1] != 3
            or logits.shape[0] != masks.shape[0]
            or logits.device != masks.device
            or not bool(torch.isfinite(logits).all())
        ):
            raise StageCTrainOnlySmokeError("captured logits violates BCHW/finite contract")
        height, width = int(logits.shape[-2]), int(logits.shape[-1])
        positions = height * width
        if positions <= 1:
            raise StageCTrainOnlySmokeError("router token count must be >1")
        targets = core.build_tri_router_targets_v32(
            masks, detached_prediction, (height, width)
        )
        probability = getattr(targets, "probability", None)
        valid = getattr(targets, "valid", None)
        if (
            not isinstance(probability, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or tuple(probability.shape) != tuple(logits.shape)
            or tuple(valid.shape) != (logits.shape[0], 3)
            or valid.dtype is not torch.bool
        ):
            raise StageCTrainOnlySmokeError("router target API returned malformed data")
        log_probability = F.log_softmax(logits.float().flatten(2), dim=-1)
        per_role_ce = -(
            probability.float().flatten(2) * log_probability
        ).sum(dim=-1) / math.log(float(positions))
        predicted = F.softmax(logits.float().flatten(2), dim=-1)
        per_role_max = predicted.max(dim=-1).values
        if not bool(torch.isfinite(per_role_ce).all()) or not bool(
            torch.isfinite(per_role_max).all()
        ):
            raise StageCTrainOnlySmokeError("router CE/max-mass is non-finite")
        for role_index, role in enumerate(ROLE_NAMES):
            role_valid = valid[:, role_index]
            role_ce = per_role_ce[:, role_index][role_valid].detach().cpu().tolist()
            role_max = per_role_max[:, role_index][role_valid].detach().cpu().tolist()
            ce[role].extend(float(value) for value in role_ce)
            max_mass[role].extend(float(value) for value in role_max)
            all_valid_ce.extend(float(value) for value in role_ce)
    if not all_valid_ce:
        raise StageCTrainOnlySmokeError("batch has no valid sample-level-role target")
    return {
        "role_ce": ce,
        "role_max_mass": max_mass,
        "all_valid_ce_mean": statistics.fmean(all_valid_ce),
        "compact": _extract_compact_diagnostics(record),
    }


def _gradient_summary(model: nn.Module) -> dict[str, Any]:
    router = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if ".tri_router." in name
    ]
    gain = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("." + GAIN_SUFFIX) or name == GAIN_SUFFIX
    ]
    if len(router) != 2 or len(gain) != 1:
        raise StageCTrainOnlySmokeError("router/gain parameter names differ")

    def group(values: list[tuple[str, nn.Parameter]]) -> dict[str, Any]:
        gradients = [parameter.grad for _name, parameter in values]
        present = all(isinstance(value, torch.Tensor) for value in gradients)
        finite = present and all(bool(torch.isfinite(value).all()) for value in gradients)
        max_abs = (
            max(float(value.detach().abs().max().item()) for value in gradients)
            if finite
            else None
        )
        return {
            "parameter_names": [name for name, _parameter in values],
            "all_present": present,
            "all_finite": finite,
            "max_abs": max_abs,
        }

    summary = {"router": group(router), "gain": group(gain)}
    if not summary["router"]["all_present"] or not summary["gain"]["all_present"]:
        raise StageCTrainOnlySmokeError("router/gain gradient is absent")
    if not summary["router"]["all_finite"] or not summary["gain"]["all_finite"]:
        raise StageCTrainOnlySmokeError("router/gain gradient is non-finite")
    return summary


def _state_summary(model: nn.Module) -> dict[str, Any]:
    router_values: list[torch.Tensor] = []
    gain_values: list[torch.Tensor] = []
    for name, value in model.state_dict().items():
        if ".tri_router." in name:
            router_values.append(value)
        if name.endswith("." + GAIN_SUFFIX) or name == GAIN_SUFFIX:
            gain_values.append(value)
    if len(router_values) != 2 or len(gain_values) != 1:
        raise StageCTrainOnlySmokeError("router/gain state contract differs")
    router_finite = all(bool(torch.isfinite(value).all()) for value in router_values)
    gain_tensor = gain_values[0]
    gain_finite = bool(torch.isfinite(gain_tensor).all())
    gains = [float(value) for value in gain_tensor.detach().cpu().tolist()]
    in_bounds = gain_finite and all(GAIN_MIN <= value <= GAIN_MAX for value in gains)
    if not router_finite or not in_bounds:
        raise StageCTrainOnlySmokeError("router/gain state is non-finite or out of bounds")
    return {
        "router_state_finite": router_finite,
        "gain_state_finite": gain_finite,
        "gain_in_bounds": in_bounds,
        "gain": gains,
    }


def _learning_rate(epoch: int) -> float:
    return BASE_LR * float(epoch) / float(FORMAL_WARMUP_EPOCHS)


def _empty_epoch_accumulator() -> dict[str, Any]:
    return {
        "processed": 0,
        "batches": 0,
        "loss": {"total": 0.0, "segmentation": 0.0, "router": 0.0},
        "ce": {role: [] for role in ROLE_NAMES},
        "max_mass": {role: [] for role in ROLE_NAMES},
        "router_grad_all_finite": True,
        "gain_grad_all_finite": True,
        "router_grad_max_abs": 0.0,
        "gain_grad_max_abs": 0.0,
        "compact": {field: 0 for field in COMPACT_COUNTER_FIELDS},
    }


def _finalize_epoch(
    epoch: int,
    learning_rate: float,
    order: Sequence[int],
    accumulator: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    processed = int(accumulator["processed"])
    if processed != SAMPLE_COUNT or int(accumulator["batches"]) != 4:
        raise StageCTrainOnlySmokeError("epoch did not process 64 samples in four batches")
    roles: dict[str, Any] = {}
    for role in ROLE_NAMES:
        ce = list(accumulator["ce"][role])
        mass = list(accumulator["max_mass"][role])
        if len(ce) != len(mass):
            raise StageCTrainOnlySmokeError("role CE/max-mass counts differ")
        roles[role] = {
            "valid_sample_level_count": len(ce),
            "mean_valid_ce": statistics.fmean(ce) if ce else None,
            "median_max_mass": statistics.median(mass) if mass else None,
        }
    compact = dict(accumulator["compact"])
    solver_denominator = compact["solver_attempt_rows"]
    emission_denominator = compact["emission_checked_rows"]
    compact.update(
        {
            "solver_fallback_rate": (
                compact["solver_fallback_rows"] / solver_denominator
                if solver_denominator
                else None
            ),
            "emission_fallback_rate": (
                compact["emission_fallback_rows"] / emission_denominator
                if emission_denominator
                else None
            ),
        }
    )
    record = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "processed_samples": processed,
        "batch_count": int(accumulator["batches"]),
        "train_order_sha256": _canonical_sha256(list(order)),
        "loss": {
            name: float(accumulator["loss"][name]) / processed
            for name in ("total", "segmentation", "router")
        },
        "roles": roles,
        "gradients": {
            "router_all_present_and_finite": bool(
                accumulator["router_grad_all_finite"]
            ),
            "gain_all_present_and_finite": bool(
                accumulator["gain_grad_all_finite"]
            ),
            "router_max_abs": float(accumulator["router_grad_max_abs"]),
            "gain_max_abs": float(accumulator["gain_grad_max_abs"]),
        },
        "state": dict(state),
        "compact_projection": compact,
    }
    return _strict_json_clone(record, label=f"epoch {epoch} evidence")


def evaluate_stage_c_gate(epochs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(epochs) != EPOCHS or tuple(record.get("epoch") for record in epochs) != tuple(
        range(1, EPOCHS + 1)
    ):
        raise StageCTrainOnlySmokeError("gate requires exactly epochs 1..5")
    first, last = epochs[0], epochs[-1]
    checks: dict[str, bool] = {}
    checks["all_losses_finite"] = all(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in record["loss"].values()
        )
        for record in epochs
    )
    checks["router_gradients_finite"] = all(
        record["gradients"]["router_all_present_and_finite"] is True
        for record in epochs
    )
    checks["gain_gradients_finite"] = all(
        record["gradients"]["gain_all_present_and_finite"] is True
        for record in epochs
    )
    checks["router_and_gain_state_finite"] = all(
        record["state"]["router_state_finite"] is True
        and record["state"]["gain_state_finite"] is True
        and record["state"]["gain_in_bounds"] is True
        for record in epochs
    )
    checks["router_loss_decreased"] = (
        float(last["loss"]["router"]) < float(first["loss"]["router"])
    )
    checks["at_least_one_gain_positive"] = any(
        float(value) > 0.0 for value in last["state"]["gain"]
    )
    for role in ("C", "H"):
        first_role = first["roles"][role]
        last_role = last["roles"][role]
        checks[f"{role}_valid_at_epoch_1_and_5"] = (
            int(first_role["valid_sample_level_count"]) > 0
            and int(last_role["valid_sample_level_count"]) > 0
        )
        first_ce = first_role["mean_valid_ce"]
        last_ce = last_role["mean_valid_ce"]
        checks[f"{role}_mean_valid_ce_decreased"] = (
            isinstance(first_ce, (int, float))
            and not isinstance(first_ce, bool)
            and isinstance(last_ce, (int, float))
            and not isinstance(last_ce, bool)
            and math.isfinite(float(first_ce))
            and math.isfinite(float(last_ce))
            and float(last_ce) < float(first_ce)
        )
    for role in ROLE_NAMES:
        first_role = first["roles"][role]
        last_role = last["roles"][role]
        valid_both = (
            int(first_role["valid_sample_level_count"]) > 0
            and int(last_role["valid_sample_level_count"]) > 0
        )
        median = last_role["median_max_mass"]
        checks[f"{role}_epoch_5_max_mass_not_collapsed_when_applicable"] = (
            not valid_both
            or (
                isinstance(median, (int, float))
                and not isinstance(median, bool)
                and math.isfinite(float(median))
                and float(median) < MAX_MASS_LIMIT
            )
        )
    def compact_rates_valid(record: Mapping[str, Any]) -> bool:
        compact = record.get("compact_projection")
        if not isinstance(compact, Mapping) or not all(
            field in compact for field in COMPACT_COUNTER_FIELDS
        ):
            return False
        if "solver_fallback_rate" not in compact or "emission_fallback_rate" not in compact:
            return False
        counters = tuple(compact[field] for field in COMPACT_COUNTER_FIELDS)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters
        ):
            return False
        if (
            compact["projection_rows"] <= 0
            or compact["solver_accepted_rows"] + compact["solver_fallback_rows"]
            != compact["solver_attempt_rows"]
            or compact["solver_attempt_rows"] > compact["projection_rows"]
            or compact["emission_checked_rows"] != compact["projection_rows"]
            or compact["emission_accepted_rows"] + compact["emission_fallback_rows"]
            != compact["emission_checked_rows"]
        ):
            return False
        solver_attempts = compact["solver_attempt_rows"]
        solver_rate = compact["solver_fallback_rate"]
        if solver_attempts == 0:
            if solver_rate is not None:
                return False
        else:
            if (
                isinstance(solver_rate, bool)
                or not isinstance(solver_rate, (int, float))
                or not math.isfinite(float(solver_rate))
                or not 0.0 <= float(solver_rate) <= 1.0
                or not math.isclose(
                    float(solver_rate),
                    compact["solver_fallback_rows"] / solver_attempts,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                return False
        emission_checked = compact["emission_checked_rows"]
        emission_rate = compact["emission_fallback_rate"]
        return (
            emission_checked > 0
            and not isinstance(emission_rate, bool)
            and isinstance(emission_rate, (int, float))
            and math.isfinite(float(emission_rate))
            and 0.0 <= float(emission_rate) <= 1.0
            and math.isclose(
                float(emission_rate),
                compact["emission_fallback_rows"] / emission_checked,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )

    checks["solver_emission_rates_recorded"] = all(
        compact_rates_valid(record) for record in epochs
    )
    verdict = "GO" if all(checks.values()) else "NO-GO"
    return {
        "verdict": verdict,
        "all_pre_registered_checks_passed": verdict == "GO",
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "scope": "train_only_mechanism_smoke_not_benchmark_performance",
        "official_test_authorized": False,
    }


def _source_provenance(core: ModuleType) -> dict[str, Any]:
    v31 = importlib.import_module("experiments.sctransnet_sbsc_v31")
    sources = {
        "stage_c_runner": _ENTRY_SOURCE,
        "experiments_package": PROJECT_ROOT / "experiments" / "__init__.py",
        "core_builder": Path(core.__file__ or ""),
        "frozen_v31_public_solver": Path(v31.__file__ or ""),
        "paired_initialization_authority": (
            PROJECT_ROOT / "experiments" / "four_dataset_models_seed42_v1.py"
        ),
        "v2_data": Path(v2_data.__file__ or ""),
        "v2_split_validator": (
            PROJECT_ROOT / "experiments" / "evisirst_v2_splits.py"
        ),
        "three_dataset_source_protocol": (
            PROJECT_ROOT / "experiments" / "three_dataset_v2_protocol.py"
        ),
        "model_package": PROJECT_ROOT / "model" / "__init__.py",
        "model_entry": PROJECT_ROOT / "model" / "EviSIRST.py",
        "sctransnet_config": PROJECT_ROOT / "model" / "_internal" / "Config.py",
        "sctransnet": PROJECT_ROOT / "model" / "_internal" / "SCTransNet.py",
    }
    internal_root = PROJECT_ROOT / "model" / "_internal"
    internal_paths = sorted(internal_root.glob("*.py"))
    if not internal_paths:
        raise StageCTrainOnlySmokeError("model internal source tree is missing")
    for path in internal_paths:
        sources.setdefault(f"model_internal/{path.name}", path)

    root = PROJECT_ROOT.resolve(strict=True)
    files: dict[str, dict[str, str]] = {}
    for name, raw_path in sources.items():
        if raw_path.is_symlink() or not raw_path.is_file():
            raise StageCTrainOnlySmokeError(
                f"protocol source is not a regular file: {name}"
            )
        path = raw_path.resolve(strict=True)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise StageCTrainOnlySmokeError(
                f"protocol source escapes repository: {name}"
            ) from exc
        files[name] = {"relative_path": relative, "sha256": _sha256_file(path)}
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


def _data_identity_payload(
    contract: TrainOnlyContract,
    *,
    selected_tree_sha256: str | None,
) -> dict[str, Any]:
    return {
        "manifest_sha256": contract.manifest_sha256,
        "train_split_sha256": contract.train_split_sha256,
        "train_ordered_ids_sha256": contract.train_ordered_ids_sha256,
        "source_train_index_sha256": contract.source_train_index_sha256,
        "source_train_ordered_ids_sha256": (
            contract.source_train_ordered_ids_sha256
        ),
        "selected_first_64_ids_sha256": contract.selected_ids_sha256,
        "selected_first_64_image_mask_tree_sha256": selected_tree_sha256,
        "source_train_count": contract.source_train_count,
        "train_split_count": contract.train_split_count,
        "selected_count": len(contract.selected_ids),
        "source_index_relative_path": contract.source_index_relative_path,
    }


def _immutable_contract_payload() -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "data_role": "train",
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "epochs": EPOCHS,
        "first_train_samples": SAMPLE_COUNT,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "model_initialization": "fresh_v32_no_checkpoint_no_warm_start",
        "optimizer": "Adam_defaults_except_epoch_learning_rate",
        "learning_rate_by_epoch": [_learning_rate(epoch) for epoch in range(1, 6)],
        "segmentation_loss": "sum_of_six_BCELoss_mean_terms",
        "router_loss": "normalized_spatial_router_ce",
        "router_loss_weight": ROUTER_LOSS_WEIGHT,
        "validation_schedule": None,
        "checkpoint_written": False,
    }


def run_stage_c(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_train_only_contract(args.dataset_root, args.split_root)
    dataset = build_train_only_dataset(contract)
    if len(dataset) != SAMPLE_COUNT:
        raise StageCTrainOnlySmokeError("train-only dataset is not exactly first 64")
    selected_tree_sha256 = v2_data._ordered_data_tree_sha256(
        dataset_root=contract.dataset_root,
        dataset=DATASET,
        sample_ids=contract.selected_ids,
    )
    core = _load_core()
    source_provenance = _source_provenance(core)
    model, builder_metadata = _build_fresh_model(core)
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    device = _require_device(args.device)
    model.to(device)
    _configure_runtime_determinism(device)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    epochs: list[dict[str, Any]] = []

    for epoch in range(1, EPOCHS + 1):
        if not callable(getattr(dataset, "set_epoch", None)):
            raise StageCTrainOnlySmokeError("train dataset lacks deterministic set_epoch")
        dataset.set_epoch(epoch)
        order = epoch_order(epoch)
        loader = DataLoader(
            Subset(dataset, order),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        learning_rate = _learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        model.mode = "train"
        accumulator = _empty_epoch_accumulator()
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with core.capture_c3_v32_training_router(model) as capture:
                outputs = model(images)
                if (
                    not isinstance(outputs, (tuple, list))
                    or len(outputs) != 6
                    or any(not isinstance(output, torch.Tensor) for output in outputs)
                ):
                    raise StageCTrainOnlySmokeError(
                        "fresh V3.2 training forward must return six tensors"
                    )
                if any(tuple(output.shape) != tuple(masks.shape) for output in outputs):
                    raise StageCTrainOnlySmokeError("six output geometries differ from masks")
                segmentation_loss = sum(criterion(output, masks) for output in outputs)
                router_loss = core.tri_router_supervision_loss(
                    capture, outputs[-1].detach(), masks
                )
            measurements = _capture_measurements(
                core=core,
                capture=capture,
                masks=masks,
                detached_prediction=outputs[-1].detach(),
            )
            if not math.isclose(
                float(router_loss.detach().item()),
                float(measurements["all_valid_ce_mean"]),
                rel_tol=2e-5,
                abs_tol=2e-6,
            ):
                raise StageCTrainOnlySmokeError(
                    "router loss differs from audited sample-level-role CE"
                )
            total_loss = segmentation_loss + ROUTER_LOSS_WEIGHT * router_loss
            for label, value in (
                ("segmentation", segmentation_loss),
                ("router", router_loss),
                ("total", total_loss),
            ):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 0
                    or not bool(torch.isfinite(value))
                    or float(value.detach().item()) < 0.0
                ):
                    raise StageCTrainOnlySmokeError(f"{label} loss is invalid")
            total_loss.backward()
            gradients = _gradient_summary(model)
            optimizer.step()
            core.project_sbsc_v32_constraints_(model)
            _state_summary(model)

            count = int(images.shape[0])
            accumulator["processed"] += count
            accumulator["batches"] += 1
            accumulator["loss"]["total"] += float(total_loss.detach().item()) * count
            accumulator["loss"]["segmentation"] += (
                float(segmentation_loss.detach().item()) * count
            )
            accumulator["loss"]["router"] += float(router_loss.detach().item()) * count
            for role in ROLE_NAMES:
                accumulator["ce"][role].extend(measurements["role_ce"][role])
                accumulator["max_mass"][role].extend(
                    measurements["role_max_mass"][role]
                )
            accumulator["router_grad_all_finite"] = (
                accumulator["router_grad_all_finite"]
                and gradients["router"]["all_present"]
                and gradients["router"]["all_finite"]
            )
            accumulator["gain_grad_all_finite"] = (
                accumulator["gain_grad_all_finite"]
                and gradients["gain"]["all_present"]
                and gradients["gain"]["all_finite"]
            )
            accumulator["router_grad_max_abs"] = max(
                accumulator["router_grad_max_abs"],
                float(gradients["router"]["max_abs"]),
            )
            accumulator["gain_grad_max_abs"] = max(
                accumulator["gain_grad_max_abs"],
                float(gradients["gain"]["max_abs"]),
            )
            for field in COMPACT_COUNTER_FIELDS:
                accumulator["compact"][field] += measurements["compact"][field]
        epochs.append(
            _finalize_epoch(
                epoch,
                learning_rate,
                order,
                accumulator,
                _state_summary(model),
            )
        )

    gate = evaluate_stage_c_gate(epochs)
    payload = {
        "schema": SCHEMA,
        "execution_status": "completed",
        "immutable_contract": _immutable_contract_payload(),
        "data_identity": _data_identity_payload(
            contract, selected_tree_sha256=selected_tree_sha256
        ),
        "source_provenance": source_provenance,
        "model": {
            "builder_metadata": builder_metadata,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": _state_dict_sha256(model.state_dict()),
        },
        "epochs": epochs,
        "gate": gate,
        "validation_split_accessed": False,
        "validation_dataset_constructed": False,
        "test_split_accessed": False,
        "test_dataset_constructed": False,
        "official_test_accessed": False,
        "checkpoint_written": False,
        "benchmark_performance_claimed": False,
    }
    return _strict_json_clone(payload, label="Stage-C artifact")


def _validate_output_path(
    output: Path, *, dataset_root: Path, split_root: Path
) -> Path:
    if output.exists():
        if output.is_symlink() or output.is_dir():
            raise StageCTrainOnlySmokeError("output must be a new regular JSON file")
        raise StageCTrainOnlySmokeError(
            "output already exists; Stage-C evidence is immutable"
        )
    resolved = output.expanduser().resolve(strict=False)
    for protected in (dataset_root.resolve(strict=True), split_root.resolve(strict=True)):
        try:
            resolved.relative_to(protected)
        except ValueError:
            continue
        raise StageCTrainOnlySmokeError("output cannot be inside a data/split root")
    if resolved.suffix.lower() != ".json":
        raise StageCTrainOnlySmokeError("output must end in .json")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.parent.is_symlink():
        raise StageCTrainOnlySmokeError("output parent cannot be a symlink")
    return resolved


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    clone = _strict_json_clone(dict(payload), label="output artifact")
    if path.exists() or path.is_symlink():
        raise StageCTrainOnlySmokeError(
            "output already exists; Stage-C evidence is immutable"
        )
    encoded = (
        json.dumps(
            clone,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".stage_c_v32_",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # temp and destination share a directory/filesystem.  Hard-linking is
        # an atomic no-replace commit: unlike os.replace, a concurrent writer
        # cannot overwrite an immutable evidence artifact.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise StageCTrainOnlySmokeError(
            "output appeared concurrently; immutable evidence was not overwritten"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _failure_payload(
    exc: Exception, args: argparse.Namespace | None = None
) -> dict[str, Any]:
    data_identity: dict[str, Any] | None = None
    data_identity_error: str | None = None
    source_provenance: dict[str, Any] = {
        "stage_c_source_sha256": _sha256_file(Path(__file__).resolve(strict=True))
    }
    source_provenance_error: str | None = None
    if args is not None:
        try:
            contract = load_train_only_contract(args.dataset_root, args.split_root)
            data_identity = _data_identity_payload(
                contract, selected_tree_sha256=None
            )
        except Exception as evidence_exc:
            data_identity_error = type(evidence_exc).__name__
        try:
            source_provenance = _source_provenance(_load_core())
        except Exception as evidence_exc:
            source_provenance_error = type(evidence_exc).__name__
    payload = {
        "schema": SCHEMA,
        "execution_status": "failed_closed",
        "immutable_contract": _immutable_contract_payload(),
        "data_identity": data_identity,
        "data_identity_error": data_identity_error,
        "source_provenance": source_provenance,
        "source_provenance_error": source_provenance_error,
        "epochs": [],
        "gate": {
            "verdict": "NO-GO",
            "all_pre_registered_checks_passed": False,
            "checks": {"execution_completed": False},
            "failed_checks": ["execution_completed"],
            "scope": "train_only_mechanism_smoke_not_benchmark_performance",
            "official_test_authorized": False,
        },
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "validation_split_accessed": False,
        "validation_dataset_constructed": False,
        "test_split_accessed": False,
        "test_dataset_constructed": False,
        "official_test_accessed": False,
        "checkpoint_written": False,
        "benchmark_performance_claimed": False,
    }
    return _strict_json_clone(payload, label="failed-closed Stage-C artifact")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = _validate_output_path(
            args.output,
            dataset_root=Path(args.dataset_root),
            split_root=Path(args.split_root),
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    try:
        payload = run_stage_c(args)
    except Exception as exc:
        write_strict_json(output, _failure_payload(exc, args))
        print(f"Stage-C NO-GO (failed closed); artifact: {output}")
        return 2
    write_strict_json(output, payload)
    print(f"Stage-C {payload['gate']['verdict']}; artifact: {output}")
    return 0 if payload["gate"]["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
