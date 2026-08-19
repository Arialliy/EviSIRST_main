#!/usr/bin/env python3
"""Formal IRSTD-1K full-800 trainer for the six registered DCS-PG arms.

Epochs 1--500 are training-only.  The official 201-image test split is first
opened and evaluated at epoch 501, then evaluated after every epoch through
1000.  Its 500 records deliberately select independent best-mIoU and best-Pd
checkpoints.  This reproduces an optimistic historical test-selected
convention; it does not support an unbiased-test or stable-over-baseline
claim.  Four control arms are registered but initially blocked; only the two
DCS-PG arms are formally authorized by the frozen rules.

This is an additive runner.  It does not modify or import an old experiment
wrapper and it never overwrites a completed formal artifact.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib
import json
import math
import numbers
import os
import re
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments import evisirst_data as public_data
from experiments import three_dataset_v2_protocol as source_protocol
from experiments.evisirst_data import (
    EviSIRSTTestDataset,
    EviSIRSTTrainDataset,
    stable_uint63,
)
from experiments.evisirst_v2_data import (
    DEFAULT_SPLIT_ROOT,
    load_v2_split_contract,
)
from test import MATCH_RADIUS, PROBABILITY_THRESHOLD, TINY_AREA, evaluate_model
import train as legacy_train
import train_validation_selected as r1_transaction
from model.EviSIRST import initialize_evisirst


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = "IRSTD-1K"
TRAINING_TARGET_RULE = "raw_mask_div_255"
ARCHITECTURE_SEED = 42
RUN_SEED = 42

GUARD_ARCHITECTURE_MODULE = "experiments.irstd_cp_hf_s2_dcspg_v1"
GUARD_ARCHITECTURE_SOURCE = PROJECT_ROOT / "experiments/irstd_cp_hf_s2_dcspg_v1.py"
GUARD_ARCHITECTURE_TEST_SOURCE = (
    PROJECT_ROOT / "tests/test_irstd_cp_hf_s2_dcspg_v1.py"
)
RUNNER_TEST_SOURCE = (
    PROJECT_ROOT / "tests/test_train_irstd_dcspg_ablation_test_selected_v1.py"
)
FORMAL_ADAPTER_SOURCE = PROJECT_ROOT / "run_irstd_dcspg_test_selected_v1.py"
FORMAL_ADAPTER_TEST_SOURCE = (
    PROJECT_ROOT / "tests/test_irstd_dcspg_test_selected_adapter.py"
)
FROZEN_MODEL_BUILDER_SOURCE = (
    PROJECT_ROOT / "experiments/four_dataset_models_seed42_v1.py"
)
GUARD_ARCHITECTURE_BUILDER = "build_irstd_cp_hf_s2_dcspg_v1"
GUARD_ARCHITECTURE_VALIDATOR = "validate_irstd_cp_hf_s2_dcspg_v1"
GUARD_ARCHITECTURE_SCHEMA = "evisirst_irstd_cp_hf_s2_dcspg_architecture_v1"
GUARD_ARCHITECTURE_NAME = "IRSTD-CP-HF-S2-DCSPG-v1"
GUARD_STATE_KEY_COUNT = 577
GUARD_PARAMETER_COUNT = 10_874_616
CP_PARENT_ARCHITECTURE_MODULE = "experiments.irstd_cp_hf_s2_v1"
CP_PARENT_ARCHITECTURE_BUILDER = "build_irstd_cp_hf_s2_v1"
CP_PARENT_ARCHITECTURE_VALIDATOR = "validate_irstd_cp_hf_s2_v1"
CP_PARENT_ARCHITECTURE_SCHEMA = "evisirst_irstd_cp_hf_s2_architecture_v1"
CP_PARENT_ARCHITECTURE_NAME = "IRSTD-CP-HF-S2-v1"
CP_PARENT_STATE_KEY_COUNT = 576
CP_PARENT_PARAMETER_COUNT = 10_874_615
EXPECTED_CP_PARENT_SOURCE_SHA256 = (
    "0dba0021dfe8f46d8c4e2932b478ce03609ab14be73b20c8b77817dce240436a"
)
CLEAN_STATE_KEY_COUNT = 564
CLEAN_PARAMETER_COUNT = 10_870_130

METHOD_IDS = (
    "clean_original",
    "clean_farbg",
    "cp_parent_original",
    "cp_parent_farbg",
    "dcspg_original",
    "dcspg_farbg",
)

# Formal execution remains fail-closed until a post-fix independent audit passes.
ARCHITECTURE_FINAL_AUDIT_GO = False
EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256: str | None = None
EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256: str | None = None

PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments/IRSTD_DCSPG_ABLATION_TEST_SELECTED_V1_PROTOCOL.md"
)
RULES_PATH = (
    PROJECT_ROOT
    / "experiments/irstd_dcspg_ablation_test_selected_v1_rules.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "b4922ac6ddfc05f12689a33c0f8681427c8ee3762da4f849e0891c84528c55ed"
)
EXPECTED_RULES_SHA256 = (
    "e92dfd8774d415b9e1aa51355a69dd9621e20d216dcf271eae81b08755666932"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs/irstd_model_design/dcspg_ablation_v1/test_selected_v1"
)
FORMAL_DATASET_ROOT = Path("/home/ly/SCTransNet_main/datasets")
FORMAL_SPLIT_ROOT = DEFAULT_SPLIT_ROOT

FORMAL_EPOCHS = 1000
FORMAL_BATCH_SIZE = 16
FORMAL_WORKERS = 0
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_TEST_BEGIN = 501
FORMAL_TEST_EVERY = 1
FORMAL_CPU_THREADS = 16
FARBG_LAMBDA = 1.0
FARBG_PROTECT_RADIUS = 3
FARBG_TOPK = 9
FARBG_THRESHOLD = 0.5

EXPECTED_TRAIN_COUNT = 800
EXPECTED_TEST_COUNT = 201
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596"
)
EXPECTED_SOURCE_TRAIN_DATA_TREE_SHA256 = (
    "ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30"
)
EXPECTED_TRAIN_ORDERED_IDS_SHA256 = (
    "681e4d741fb857703471d6555faa0d86e931aa790567c28f4254331ea9ba3d95"
)
EXPECTED_TRAIN_INDEX_FILE_SHA256 = (
    "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744"
)
EXPECTED_TEST_INDEX_FILE_SHA256 = (
    "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579"
)
EXPECTED_TEST_ORDERED_IDS_SHA256 = (
    "48e0661ba187561d1031b2fa22d4b157fd31e00570775b483b51bb21e1def38b"
)
EXPECTED_TEST_IMAGE_MASK_TREE_SHA256 = (
    "20787ff6553c9e24b2370d7f6acbf8739038c877835d4c561040962d75281b3d"
)

RUN_SCHEMA = "evisirst_irstd_dcspg_ablation_test_selected_training/v1"
RECOVERY_SCHEMA = RUN_SCHEMA + "/recovery"
CANDIDATE_SCHEMA = RUN_SCHEMA + "/candidate"
CHECKPOINT_SCHEMA = RUN_SCHEMA + "/checkpoint"
HISTORY_SCHEMA = RUN_SCHEMA + "/history"
SELECTION_SCHEMA = RUN_SCHEMA + "/selection"
SUMMARY_SCHEMA = RUN_SCHEMA + "/summary"
SOURCE_SET_SCHEMA = RUN_SCHEMA + "/source_set"
TEST_ACCESS_STARTED_SCHEMA = RUN_SCHEMA + "/test_access_started"
TEST_ACCESS_VERIFIED_SCHEMA = RUN_SCHEMA + "/test_access_verified"
RULES_SCHEMA = "evisirst_irstd_dcspg_ablation_test_selected_rules/v1"
REQUIRED_ADDITIONAL_SOURCE_FILES = (
    "experiments/four_dataset_models_seed42_v1.py",
    "run_irstd_dcspg_test_selected_v1.py",
    "tests/test_irstd_dcspg_test_selected_adapter.py",
    "tests/test_train_irstd_dcspg_ablation_test_selected_v1.py",
)

ROLE_NAMES = ("best_miou", "best_pd")
PUBLISHED_FILENAMES = {
    "best_miou": "EviSIRST_best_mIoU.pth.tar",
    "best_pd": "EviSIRST_best_Pd.pth.tar",
}
METRIC_FIELDS = (
    "test_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SMOKE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_CANDIDATE_RE = re.compile(r"^epoch_(\d{4})\.pth\.tar$")


class DCSPGAblationError(RuntimeError):
    """An artifact or request violates the frozen V3 contract."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=METHOD_IDS, required=True)
    parser.add_argument("--dataset-root", type=Path, default=FORMAL_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=FORMAL_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--architecture-seed", type=int, default=ARCHITECTURE_SEED)
    parser.add_argument("--run-seed", type=int, default=RUN_SEED)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument("--warmup-epochs", type=int, default=FORMAL_WARMUP_EPOCHS)
    parser.add_argument("--test-begin", type=int, default=FORMAL_TEST_BEGIN)
    parser.add_argument("--test-every", type=int, default=FORMAL_TEST_EVERY)
    parser.add_argument("--cpu-threads", type=int, default=FORMAL_CPU_THREADS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-id", default="fixture")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-images", type=int)
    args = parser.parse_args(argv)

    if args.architecture_seed != ARCHITECTURE_SEED or args.run_seed != RUN_SEED:
        parser.error("this candidate protocol fixes architecture/run seed to 42")
    if not 1 <= args.epochs <= FORMAL_EPOCHS or args.batch_size < 1:
        parser.error("--epochs must be in [1,1000] and --batch-size positive")
    if args.workers < 0 or args.cpu_threads < 1:
        parser.error("--workers must be non-negative and --cpu-threads positive")
    if not math.isfinite(args.base_lr) or not math.isfinite(args.min_lr):
        parser.error("learning rates must be finite")
    if not 0.0 < args.min_lr <= args.base_lr:
        parser.error("learning rates must satisfy 0 < min-lr <= base-lr")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if args.test_every < 1:
        parser.error("test interval must be positive")
    if not 1 <= args.test_begin <= args.epochs:
        parser.error("--test-begin must be within the training run")
    for name in ("max_train_samples", "max_test_images"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if _SMOKE_ID_RE.fullmatch(args.smoke_id) is None:
        parser.error("--smoke-id is not a safe identifier")
    if not args.smoke and (
        args.epochs != FORMAL_EPOCHS
        or args.batch_size != FORMAL_BATCH_SIZE
        or args.workers != FORMAL_WORKERS
        or args.base_lr != FORMAL_BASE_LR
        or args.min_lr != FORMAL_MIN_LR
        or args.warmup_epochs != FORMAL_WARMUP_EPOCHS
        or args.test_begin != FORMAL_TEST_BEGIN
        or args.test_every != FORMAL_TEST_EVERY
        or args.cpu_threads != FORMAL_CPU_THREADS
        or args.max_train_samples is not None
        or args.max_test_images is not None
        or args.smoke_id != "fixture"
    ):
        parser.error("formal mode fixes the complete V3 protocol")
    return args


def test_due(
    epoch: int,
    begin: int = FORMAL_TEST_BEGIN,
    every: int = FORMAL_TEST_EVERY,
) -> bool:
    return epoch >= begin and (epoch - begin) % every == 0


def expected_test_epochs(completed_epoch: int, begin: int, every: int) -> list[int]:
    return [
        epoch
        for epoch in range(1, completed_epoch + 1)
        if test_due(epoch, begin, every)
    ]


def _sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DCSPGAblationError(f"not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DCSPGAblationError("value is not strict canonical JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DCSPGAblationError(f"JSON is not a regular file: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DCSPGAblationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise DCSPGAblationError(f"non-finite JSON token: {token}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DCSPGAblationError(f"malformed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DCSPGAblationError("JSON root must be one object")
    _canonical_bytes(payload)
    return payload


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject a symlink at any existing component before canonicalization."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise DCSPGAblationError(
                f"{label} contains a symlink component: {component}"
            )


def _require_strict_descendant(
    path: Path, *, root: Path, label: str
) -> tuple[Path, Path]:
    """Resolve a hostile path only after rejecting every symlink component."""

    _reject_symlink_components(root, label="output root")
    _reject_symlink_components(path, label=label)
    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise DCSPGAblationError(
            f"{label} resolves outside the output root"
        ) from exc
    if not relative.parts:
        raise DCSPGAblationError(
            f"{label} must resolve strictly inside the output root"
        )
    return resolved_path, resolved_root


def _ensure_regular_directory(path: Path, *, create: bool) -> Path:
    _reject_symlink_components(path, label="directory path")
    if path.is_symlink():
        raise DCSPGAblationError(f"directory is a symlink: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise DCSPGAblationError(f"not a regular directory: {path}")
    return path.resolve(strict=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    parent = _ensure_regular_directory(path.parent, create=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise DCSPGAblationError(f"unsafe JSON destination: {path}")
    content = _canonical_bytes(dict(payload)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    parent = _ensure_regular_directory(path.parent, create=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    content = _canonical_bytes(dict(payload)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    parent = _ensure_regular_directory(path.parent, create=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise DCSPGAblationError(f"unsafe checkpoint destination: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save_no_clobber(
    path: Path, payload: Mapping[str, Any]
) -> None:
    parent = _ensure_regular_directory(path.parent, create=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _update_length_prefixed(digest: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _aggregate_file_manifest(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(files):
        _update_length_prefixed(digest, relative)
        _update_length_prefixed(digest, bytes.fromhex(files[relative]))
    return digest.hexdigest()


def _load_rules() -> dict[str, Any]:
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise DCSPGAblationError("protocol SHA-256 differs")
    if _sha256(RULES_PATH) != EXPECTED_RULES_SHA256:
        raise DCSPGAblationError("rules SHA-256 differs")
    rules = _strict_json(RULES_PATH)
    architecture = rules.get("architecture")
    arms = rules.get("arms")
    farbg = rules.get("far_background_auxiliary")
    execution = rules.get("execution")
    data = rules.get("data")
    selection = rules.get("selection")
    outputs = rules.get("outputs")
    boundary = rules.get("evidence_boundary")
    clean = architecture.get("clean") if isinstance(architecture, Mapping) else None
    cp_parent = (
        architecture.get("cp_parent") if isinstance(architecture, Mapping) else None
    )
    guard = architecture.get("guard") if isinstance(architecture, Mapping) else None
    if (
        rules.get("schema") != RULES_SCHEMA
        or rules.get("protocol_relative_path")
        != "experiments/IRSTD_DCSPG_ABLATION_TEST_SELECTED_V1_PROTOCOL.md"
        or tuple(rules.get("source_manifest_required_files", ()))
        != REQUIRED_ADDITIONAL_SOURCE_FILES
        or not isinstance(architecture, Mapping)
        or not isinstance(clean, Mapping)
        or clean.get("builder") != "model.EviSIRST.initialize_evisirst"
        or clean.get("expected_state_key_count") != CLEAN_STATE_KEY_COUNT
        or clean.get("expected_parameter_count") != CLEAN_PARAMETER_COUNT
        or not isinstance(cp_parent, Mapping)
        or cp_parent.get("module") != CP_PARENT_ARCHITECTURE_MODULE
        or cp_parent.get("builder") != CP_PARENT_ARCHITECTURE_BUILDER
        or cp_parent.get("validator") != CP_PARENT_ARCHITECTURE_VALIDATOR
        or cp_parent.get("architecture_schema") != CP_PARENT_ARCHITECTURE_SCHEMA
        or cp_parent.get("experiment_name") != CP_PARENT_ARCHITECTURE_NAME
        or cp_parent.get("expected_state_key_count")
        != CP_PARENT_STATE_KEY_COUNT
        or cp_parent.get("expected_parameter_count")
        != CP_PARENT_PARAMETER_COUNT
        or not isinstance(guard, Mapping)
        or guard.get("module") != GUARD_ARCHITECTURE_MODULE
        or guard.get("builder") != GUARD_ARCHITECTURE_BUILDER
        or guard.get("validator") != GUARD_ARCHITECTURE_VALIDATOR
        or guard.get("architecture_schema") != GUARD_ARCHITECTURE_SCHEMA
        or guard.get("experiment_name") != GUARD_ARCHITECTURE_NAME
        or guard.get("expected_state_key_count") != GUARD_STATE_KEY_COUNT
        or guard.get("expected_parameter_count") != GUARD_PARAMETER_COUNT
        or tuple(guard.get("source_dependencies", ()))
        != ("experiments/irstd_cp_hf_s2_v1.py",)
        or guard.get("source_dependency_sha256")
        != {
            "experiments/irstd_cp_hf_s2_v1.py": (
                EXPECTED_CP_PARENT_SOURCE_SHA256
            )
        }
        or not isinstance(arms, Mapping)
        or tuple(arms) != METHOD_IDS
        or any(
            not isinstance(arms.get(method_id), Mapping)
            for method_id in METHOD_IDS
        )
        or not isinstance(farbg, Mapping)
        or farbg.get("schema")
        != "evisirst_train_mask_far_background_auxiliary_loss/v1"
        or farbg.get("formal_status") != "FROZEN_NO_SWEEP"
        or farbg.get("lambda") != 1.0
        or farbg.get("protect_dilation_radius_pixels") != 3
        or farbg.get("protect_dilation")
        != "binary_euclidean_disk_offsets_u2_plus_v2_le_9"
        or farbg.get("probability_threshold") != 0.5
        or tuple(farbg.get("selected_training_heads", ()))
        != (
            "raw_d0_probability_head_index_4",
            "final_out_probability_head_index_5",
        )
        or farbg.get("per_sample_topk") != 9
        or farbg.get("head_reduction") != "mean_raw_d0_and_final_out"
        or farbg.get("total_loss")
        != "original_sum_of_six_mean_BCE_plus_lambda_times_far_background_auxiliary"
        or farbg.get("uses_train_mask_only") is not True
        or farbg.get("test_mask_used_for_loss_or_tuning") is not False
        or farbg.get("test_used_to_choose_lambda_or_dilation") is not False
        or farbg.get("hyperparameter_sweep_performed") is not False
        or not isinstance(execution, Mapping)
        or execution.get("architecture_seed") != ARCHITECTURE_SEED
        or execution.get("run_seed") != RUN_SEED
        or execution.get("epochs") != FORMAL_EPOCHS
        or execution.get("batch_size") != FORMAL_BATCH_SIZE
        or execution.get("workers") != FORMAL_WORKERS
        or execution.get("base_lr") != FORMAL_BASE_LR
        or execution.get("min_lr") != FORMAL_MIN_LR
        or execution.get("warmup_epochs") != FORMAL_WARMUP_EPOCHS
        or execution.get("epochs_1_to_500_role")
        != "training_only_no_test_access"
        or execution.get("test_begin_epoch") != FORMAL_TEST_BEGIN
        or execution.get("test_end_epoch") != FORMAL_EPOCHS
        or execution.get("test_every") != FORMAL_TEST_EVERY
        or execution.get("test_history_count") != 500
        or not isinstance(data, Mapping)
        or data.get("expected_train_count") != EXPECTED_TRAIN_COUNT
        or data.get("expected_test_count") != EXPECTED_TEST_COUNT
        or data.get("training_dataset_class")
        != "experiments.evisirst_data.EviSIRSTTrainDataset"
        or data.get("training_target_rule") != TRAINING_TARGET_RULE
        or data.get("expected_train_index_file_sha256")
        != EXPECTED_TRAIN_INDEX_FILE_SHA256
        or data.get("expected_train_ordered_ids_sha256")
        != EXPECTED_TRAIN_ORDERED_IDS_SHA256
        or data.get("expected_split_manifest_sha256")
        != EXPECTED_SPLIT_MANIFEST_SHA256
        or data.get("expected_source_train_data_tree_sha256")
        != EXPECTED_SOURCE_TRAIN_DATA_TREE_SHA256
        or data.get("expected_test_index_file_sha256")
        != EXPECTED_TEST_INDEX_FILE_SHA256
        or data.get("expected_test_ordered_ids_sha256")
        != EXPECTED_TEST_ORDERED_IDS_SHA256
        or data.get("expected_test_image_mask_tree_sha256")
        != EXPECTED_TEST_IMAGE_MASK_TREE_SHA256
        or data.get("optimization_uses_all_source_train_members") is not True
        or data.get("independent_validation_split_in_this_runner") is not False
        or data.get("test_access_is_lazy") is not True
        or data.get("identity_field")
        != "test_index_opened_at_identity_construction"
        or data.get("startup_test_index_opened") is not False
        or data.get("epoch_500_test_split_accessed") is not False
        or data.get("first_test_access_epoch") != FORMAL_TEST_BEGIN
        or data.get("test_access_started_ledger") != "test_access_started.json"
        or data.get("test_access_verified_ledger") != "test_access_verified.json"
        or data.get("started_ledger_precedes_any_live_test_open") is not True
        or not isinstance(selection, Mapping)
        or selection.get("comparison") != "strict_lexicographic_greater_than"
        or selection.get("exact_tie") != "earliest_epoch"
        or selection.get("test_history_is_formal_selection_pool") is not True
        or not isinstance(outputs, Mapping)
        or outputs.get("published_weight_file_count_per_arm") != 2
        or tuple(outputs.get("published_checkpoint_roles", ())) != ROLE_NAMES
        or outputs.get("published_weight_filenames") != PUBLISHED_FILENAMES
        or not isinstance(boundary, Mapping)
        or boundary.get("test_selected") is not True
        or boundary.get("selection_is_optimistic") is not True
        or boundary.get("unbiased_test_claim_supported") is not False
        or boundary.get("stable_over_baseline_claim_supported") is not False
        or tuple(boundary.get("initial_formal_method_ids", ()))
        != ("dcspg_original", "dcspg_farbg")
        or boundary.get("test_used_for_structure_or_hyperparameter_selection")
        is not False
    ):
        raise DCSPGAblationError("rules payload differs from the formal contract")
    return rules


def _method_contract(
    args: argparse.Namespace, rules: Mapping[str, Any]
) -> dict[str, Any]:
    arms = rules["arms"]
    raw = arms.get(args.method_id)
    if not isinstance(raw, Mapping):
        raise DCSPGAblationError("method ID is absent from frozen rules")
    architecture_kind = raw.get("architecture_kind")
    loss_kind = raw.get("loss_kind")
    expected_architecture = (
        "clean"
        if args.method_id.startswith("clean_")
        else "cp_parent"
        if args.method_id.startswith("cp_parent_")
        else "guard"
    )
    expected_loss = (
        "far_background_auxiliary"
        if args.method_id.endswith("_farbg")
        else "original"
    )
    if architecture_kind != expected_architecture or loss_kind != expected_loss:
        raise DCSPGAblationError("method arm decomposition differs")
    if not args.smoke and raw.get("formal_authorized") is not True:
        raise DCSPGAblationError(
            f"formal method arm is registered but not authorized: {args.method_id}"
        )
    farbg = rules["far_background_auxiliary"]
    loss_recipe = {
        "kind": loss_kind,
        "base_loss": "sum_of_six_BCELoss_mean_terms",
        "far_background": dict(farbg) if expected_loss != "original" else None,
    }
    return {
        "method_id": args.method_id,
        "architecture_kind": architecture_kind,
        "loss_kind": loss_kind,
        "formal_authorized": bool(raw.get("formal_authorized")),
        "loss_recipe": loss_recipe,
    }


def _source_paths(rules: Mapping[str, Any]) -> list[Path]:
    architecture = rules["architecture"]
    dependency_paths = [
        PROJECT_ROOT / str(relative)
        for relative in architecture["guard"]["source_dependencies"]
    ]
    paths = [
        Path(__file__).resolve(),
        PROTOCOL_PATH,
        RULES_PATH,
        GUARD_ARCHITECTURE_SOURCE,
        GUARD_ARCHITECTURE_TEST_SOURCE,
        FORMAL_ADAPTER_SOURCE,
        FORMAL_ADAPTER_TEST_SOURCE,
        RUNNER_TEST_SOURCE,
        FROZEN_MODEL_BUILDER_SOURCE,
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "test.py",
        PROJECT_ROOT / "train_validation_selected.py",
        PROJECT_ROOT / "experiments/evisirst_data.py",
        PROJECT_ROOT / "experiments/evisirst_v2_data.py",
        PROJECT_ROOT / "experiments/evisirst_v2_splits.py",
        PROJECT_ROOT / "experiments/three_dataset_v2_protocol.py",
        PROJECT_ROOT / "model/EviSIRST.py",
        PROJECT_ROOT / "model/__init__.py",
        *dependency_paths,
        *sorted((PROJECT_ROOT / "model/_internal").glob("*.py")),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise DCSPGAblationError(f"source dependency unavailable: {path}")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise DCSPGAblationError("source dependency escapes repository") from exc
        if relative not in seen:
            unique.append(resolved)
            seen.add(relative)
    return unique


def _source_manifest(*, formal: bool) -> dict[str, Any]:
    rules = _load_rules()
    paths = _source_paths(rules)
    files = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in paths
    }
    architecture_sha = files[
        GUARD_ARCHITECTURE_SOURCE.relative_to(PROJECT_ROOT).as_posix()
    ]
    architecture_test_sha = files[
        GUARD_ARCHITECTURE_TEST_SOURCE.relative_to(PROJECT_ROOT).as_posix()
    ]
    cp_parent_sha = files["experiments/irstd_cp_hf_s2_v1.py"]
    if cp_parent_sha != EXPECTED_CP_PARENT_SOURCE_SHA256:
        raise DCSPGAblationError("CP-HF-S2 dependency SHA-256 differs")
    if formal and (
        not ARCHITECTURE_FINAL_AUDIT_GO
        or EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 is None
        or EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 is None
        or _SHA_RE.fullmatch(EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256) is None
        or _SHA_RE.fullmatch(EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256) is None
    ):
        raise DCSPGAblationError("formal architecture hashes are not sealed")
    if EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 is not None and (
        architecture_sha != EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256
    ):
        raise DCSPGAblationError("architecture source SHA-256 differs")
    if EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 is not None and (
        architecture_test_sha != EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256
    ):
        raise DCSPGAblationError("architecture test SHA-256 differs")
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "aggregate_sha256": _aggregate_file_manifest(files),
        "architecture_source_sha256": architecture_sha,
        "architecture_test_sha256": architecture_test_sha,
    }


def _file_tree_sha256(
    dataset_root: Path, entries: Sequence[tuple[str, str, Path]]
) -> str:
    digest = hashlib.sha256()
    for role, sample_id, path in entries:
        if path.is_symlink() or not path.is_file():
            raise DCSPGAblationError(f"data entry is not regular: {path}")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(dataset_root)
        except ValueError as exc:
            raise DCSPGAblationError("data entry escapes dataset root") from exc
        _update_length_prefixed(digest, role)
        _update_length_prefixed(digest, sample_id)
        _update_length_prefixed(digest, relative.as_posix())
        _update_length_prefixed(digest, bytes.fromhex(_sha256(resolved)))
    return digest.hexdigest()


def _test_tree_entries(test_dataset: EviSIRSTTestDataset) -> list[tuple[str, str, Path]]:
    entries: list[tuple[str, str, Path]] = []
    for sample_id in test_dataset.sample_ids:
        sample = public_data.source_protocol.resolve_sample(
            test_dataset.dataset_root,
            DATASET,
            sample_id,
            split="test",
            known_ids=test_dataset._known_ids,
        )
        entries.extend(
            (
                (f"{DATASET}:image", sample_id, sample.image_path),
                (f"{DATASET}:mask", sample_id, sample.mask_path),
            )
        )
    return entries


def _data_identity(
    args: argparse.Namespace, rules: Mapping[str, Any]
) -> tuple[EviSIRSTTrainDataset, Any, dict[str, Any]]:
    _reject_symlink_components(args.dataset_root, label="dataset root")
    _reject_symlink_components(args.split_root, label="split root")
    dataset_root = args.dataset_root.resolve(strict=True)
    split_root = args.split_root.resolve(strict=True)
    if not args.smoke:
        if dataset_root != FORMAL_DATASET_ROOT.resolve(strict=True):
            raise DCSPGAblationError(
                f"formal dataset root must be {FORMAL_DATASET_ROOT}"
            )
        if split_root != FORMAL_SPLIT_ROOT.resolve(strict=True):
            raise DCSPGAblationError(
                f"formal split root must be {FORMAL_SPLIT_ROOT}"
            )

    contract = load_v2_split_contract(
        DATASET,
        dataset_root=dataset_root,
        split_root=split_root,
        verify_data_tree=True,
    )
    full_train = EviSIRSTTrainDataset(
        DATASET,
        dataset_root=dataset_root,
        patch_size=256,
        seed=RUN_SEED,
    )
    if tuple(full_train.sample_ids) != contract.source_train_ids:
        raise DCSPGAblationError(
            "legacy full-train order differs from the V2 source-train union"
        )
    train_data: Any = full_train
    if args.max_train_samples is not None:
        train_data = Subset(
            full_train, range(min(args.max_train_samples, len(full_train)))
        )
    if len(train_data) < 1:
        raise DCSPGAblationError("full-train data must be non-empty")

    train_order_sha = source_protocol.ordered_ids_sha256(full_train.sample_ids)
    source_index = contract.manifest.get("source_index")
    if not isinstance(source_index, Mapping):
        raise DCSPGAblationError("V2 source-train index record is malformed")
    train_index_file_sha = source_index.get("file_sha256")
    if not args.smoke and (
        len(train_data) != EXPECTED_TRAIN_COUNT
        or contract.manifest_sha256 != EXPECTED_SPLIT_MANIFEST_SHA256
        or contract.data_tree_sha256
        != EXPECTED_SOURCE_TRAIN_DATA_TREE_SHA256
        or train_index_file_sha != EXPECTED_TRAIN_INDEX_FILE_SHA256
        or train_order_sha != EXPECTED_TRAIN_ORDERED_IDS_SHA256
    ):
        raise DCSPGAblationError("formal full-train identity differs")

    frozen_data = rules["data"]
    identity = {
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "dataset": DATASET,
        "training_dataset_class": (
            "experiments.evisirst_data.EviSIRSTTrainDataset"
        ),
        "training_target_rule": TRAINING_TARGET_RULE,
        "train_count": len(train_data),
        "full_train_count": len(full_train),
        "v2_train_count": len(contract.train_ids),
        "v2_val_count": len(contract.val_ids),
        "v2_train_val_union_equals_source_train": True,
        "independent_validation_split_in_this_runner": False,
        "split_manifest_sha256": contract.manifest_sha256,
        "source_train_data_tree_sha256": contract.data_tree_sha256,
        "source_train_data_tree_verified": contract.data_tree_verified,
        "train_index_relative_path": source_index.get("relative_path"),
        "train_index_file_sha256": train_index_file_sha,
        "train_ordered_ids_sha256": train_order_sha,
        "normalization": dict(full_train.normalization),
        "expected_test_contract": {
            "test_count": frozen_data["expected_test_count"],
            "test_index_file_sha256": frozen_data[
                "expected_test_index_file_sha256"
            ],
            "test_ordered_ids_sha256": frozen_data[
                "expected_test_ordered_ids_sha256"
            ],
            "test_image_mask_tree_sha256": frozen_data[
                "expected_test_image_mask_tree_sha256"
            ],
            "test_image_mask_tree_hash_algorithm": (
            "ordered_length_prefixed(role,sample_id,relative_path,file_sha256)"
            ),
        },
        "test_access_is_lazy": True,
        "test_index_opened_at_identity_construction": False,
        "startup_test_index_opened": False,
        "test_split_accessed_during_preflight": False,
    }
    return full_train, train_data, identity


def _activate_test_data(
    args: argparse.Namespace, data_identity: Mapping[str, Any]
) -> tuple[EviSIRSTTestDataset, Any, dict[str, Any]]:
    """First and only constructor path for official test data."""

    dataset_root = Path(str(data_identity["dataset_root"])).resolve(strict=True)
    full_test = EviSIRSTTestDataset(
        DATASET,
        DATASET,
        dataset_root=dataset_root,
    )
    test_data: Any = full_test
    if args.max_test_images is not None:
        test_data = Subset(
            full_test, range(min(args.max_test_images, len(full_test)))
        )
    if len(test_data) < 1:
        raise DCSPGAblationError("test data must be non-empty")
    test_order_sha = source_protocol.ordered_ids_sha256(full_test.sample_ids)
    test_index_path = source_protocol.index_path(dataset_root, DATASET, "test")
    test_index_file_sha = _sha256(test_index_path)
    test_tree_sha = _file_tree_sha256(
        dataset_root, _test_tree_entries(full_test)
    )
    expected = data_identity["expected_test_contract"]
    if not isinstance(expected, Mapping):
        raise DCSPGAblationError("expected test contract is malformed")
    if not args.smoke and (
        len(test_data) != EXPECTED_TEST_COUNT
        or len(full_test) != EXPECTED_TEST_COUNT
        or test_index_file_sha != expected.get("test_index_file_sha256")
        or test_order_sha != expected.get("test_ordered_ids_sha256")
        or test_tree_sha != expected.get("test_image_mask_tree_sha256")
        or full_test.normalization != data_identity.get("normalization")
    ):
        raise DCSPGAblationError("live official test identity differs")
    observed = {
        "test_count": len(test_data),
        "full_test_count": len(full_test),
        "test_index_relative_path": test_index_path.relative_to(
            dataset_root
        ).as_posix(),
        "test_index_file_sha256": test_index_file_sha,
        "test_ordered_ids_sha256": test_order_sha,
        "test_image_mask_tree_sha256": test_tree_sha,
        "normalization": dict(full_test.normalization),
        "test_index_opened": True,
        "test_split_accessed": True,
        "first_access_epoch": args.test_begin,
    }
    return full_test, test_data, observed


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _metadata_field(
    metadata: Mapping[str, Any], *names: str, default: Any = None
) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return default


def _build_architecture(
    method: Mapping[str, Any]
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    kind = method.get("architecture_kind")
    if kind == "clean":
        model, raw_metadata = initialize_evisirst(
            DATASET, seed=ARCHITECTURE_SEED, training=True
        )
        metadata = dict(raw_metadata)
        pair = metadata.get("pair")
        if (
            metadata.get("method") != "final_scratch"
            or metadata.get("target_survival_registered") is not False
            or metadata.get("warm_start_used") is not False
            or metadata.get("parent_checkpoint") is not None
            or not isinstance(pair, Mapping)
            or pair.get("initialization_mode") != "true_scratch"
            or pair.get("parent_checkpoint_load_count") != 0
            or pair.get("optimizer_state_inherited") is not False
        ):
            raise DCSPGAblationError("clean builder is not true scratch")
        validation: dict[str, Any] = {
            "schema": "evisirst_clean_architecture_validation/v1",
            "name": "EviSIRST-clean",
            "architecture_seed": ARCHITECTURE_SEED,
            "state_key_count": CLEAN_STATE_KEY_COUNT,
            "parameter_count": CLEAN_PARAMETER_COUNT,
            "baseline_checkpoint_loaded": False,
            "warm_start_used": False,
        }
        expected_name = "EviSIRST-clean"
        expected_schema = validation["schema"]
        expected_state_count = CLEAN_STATE_KEY_COUNT
        expected_parameter_count = CLEAN_PARAMETER_COUNT
    else:
        if kind == "cp_parent":
            module_name = CP_PARENT_ARCHITECTURE_MODULE
            builder_name = CP_PARENT_ARCHITECTURE_BUILDER
            validator_name = CP_PARENT_ARCHITECTURE_VALIDATOR
            expected_name = CP_PARENT_ARCHITECTURE_NAME
            expected_schema = CP_PARENT_ARCHITECTURE_SCHEMA
            expected_state_count = CP_PARENT_STATE_KEY_COUNT
            expected_parameter_count = CP_PARENT_PARAMETER_COUNT
        elif kind == "guard":
            module_name = GUARD_ARCHITECTURE_MODULE
            builder_name = GUARD_ARCHITECTURE_BUILDER
            validator_name = GUARD_ARCHITECTURE_VALIDATOR
            expected_name = GUARD_ARCHITECTURE_NAME
            expected_schema = GUARD_ARCHITECTURE_SCHEMA
            expected_state_count = GUARD_STATE_KEY_COUNT
            expected_parameter_count = GUARD_PARAMETER_COUNT
        else:
            raise DCSPGAblationError("unsupported architecture kind")
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise DCSPGAblationError("architecture module is unavailable") from exc
        if (
            getattr(module, "ARCHITECTURE_SCHEMA", None) != expected_schema
            or getattr(module, "EXPERIMENT_NAME", None) != expected_name
        ):
            raise DCSPGAblationError("architecture module identity differs")
        builder = getattr(module, builder_name, None)
        validator = getattr(module, validator_name, None)
        if not callable(builder) or not callable(validator):
            raise DCSPGAblationError("architecture API is incomplete")
        try:
            built = builder(DATASET, seed=ARCHITECTURE_SEED, training=True)
        except Exception as exc:
            raise DCSPGAblationError("architecture builder failed") from exc
        if (
            not isinstance(built, tuple)
            or len(built) != 2
            or not isinstance(built[0], nn.Module)
            or not isinstance(built[1], Mapping)
        ):
            raise DCSPGAblationError("architecture builder return differs")
        model = built[0]
        metadata = dict(built[1])
        try:
            raw_validation = validator(
                model, require_identity_initialization=True
            )
        except Exception as exc:
            raise DCSPGAblationError(
                "architecture identity validation failed"
            ) from exc
        if not isinstance(raw_validation, Mapping):
            raise DCSPGAblationError("architecture validator return differs")
        validation = dict(raw_validation)
        observed_schema = _metadata_field(
            metadata, "architecture_schema", "schema"
        )
        observed_name = _metadata_field(
            metadata,
            "experiment_name",
            "architecture_name",
            "name",
            "experiment",
        )
        baseline_loaded = metadata.get(
            "baseline_checkpoint_loaded",
            metadata.get("parent_checkpoint") is not None,
        )
        if (
            observed_schema != expected_schema
            or observed_name != expected_name
            or _metadata_field(metadata, "architecture_seed", "seed")
            != ARCHITECTURE_SEED
            or _metadata_field(metadata, "state_key_count")
            != expected_state_count
            or _metadata_field(metadata, "parameter_count")
            != expected_parameter_count
            or baseline_loaded is not False
            or metadata.get("warm_start_used") is not False
            or metadata.get("parent_checkpoint") is not None
        ):
            raise DCSPGAblationError("architecture metadata differs")

    state = model.state_dict()
    if (
        len(state) != expected_state_count
        or _parameter_count(model) != expected_parameter_count
        or hasattr(model, "target_survival")
        or any(not parameter.requires_grad for parameter in model.parameters())
    ):
        raise DCSPGAblationError("architecture state/parameter contract differs")
    _validate_state_dict(state, state, expected_count=expected_state_count)
    architecture_identity = {
        "kind": kind,
        "name": expected_name,
        "schema": expected_schema,
        "architecture_seed": ARCHITECTURE_SEED,
        "state_key_count": expected_state_count,
        "parameter_count": expected_parameter_count,
        "full_model_scratch": True,
        "baseline_checkpoint_loaded": False,
        "warm_start_used": False,
    }
    _canonical_bytes(architecture_identity)
    return model, metadata, validation, architecture_identity


def _cpu_state(
    model: nn.Module, *, expected_count: int
) -> dict[str, torch.Tensor]:
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    _validate_state_dict(
        state, model.state_dict(), expected_count=expected_count
    )
    return state


def _validate_state_dict(
    value: Any,
    expected: Mapping[str, torch.Tensor],
    *,
    expected_count: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or list(value) != list(expected):
        raise DCSPGAblationError("checkpoint state key/order differs")
    if len(value) != expected_count or len(expected) != expected_count:
        raise DCSPGAblationError("checkpoint state-key count differs")
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        reference = expected[key]
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise DCSPGAblationError("checkpoint state is malformed")
        if (
            tensor.layout != torch.strided
            or tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
        ):
            raise DCSPGAblationError(
                f"checkpoint tensor contract differs for {key!r}"
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise DCSPGAblationError(
                f"checkpoint tensor is non-finite: {key!r}"
            )
        normalized[key] = tensor.detach().cpu()
    return normalized


def _metric_row(
    metrics: Mapping[str, Any], *, epoch: int, data_role: str
) -> dict[str, Any]:
    if data_role not in {"val", "test"}:
        raise DCSPGAblationError("metric data role is unsupported")
    if set(metrics) != set(METRIC_FIELDS):
        raise DCSPGAblationError("evaluator metric field set differs")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise DCSPGAblationError("metric epoch is invalid")
    row: dict[str, Any] = {"epoch": epoch, "data_role": data_role}
    integer_fields = {
        key for key in METRIC_FIELDS if key.endswith("_count")
    } | {"valid_pixel_count"}
    for key in METRIC_FIELDS:
        value = metrics[key]
        if value is None:
            if key != "tiny_pd":
                raise DCSPGAblationError(f"metric {key!r} is null")
            row[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise DCSPGAblationError(f"metric {key!r} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise DCSPGAblationError(f"metric {key!r} is non-finite")
        if key in integer_fields:
            if not numeric.is_integer() or numeric < 0:
                raise DCSPGAblationError(f"metric {key!r} is not a count")
            row[key] = int(numeric)
        else:
            row[key] = numeric

    for key in (
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
    ):
        if not 0.0 <= float(row[key]) <= 1.0:
            raise DCSPGAblationError(f"metric {key!r} is outside [0,1]")
    if row["tiny_pd"] is not None and not 0.0 <= float(row["tiny_pd"]) <= 1.0:
        raise DCSPGAblationError("tiny_pd is outside [0,1]")
    if (
        float(row["test_loss"]) < 0.0
        or float(row["fa"]) < 0.0
        or float(row["false_objects_per_image"]) < 0.0
        or row["matched_target_count"] > row["target_count"]
        or row["matched_tiny_target_count"] > row["tiny_target_count"]
        or row["unmatched_predicted_object_count"]
        > row["predicted_object_count"]
    ):
        raise DCSPGAblationError("metric vector is internally inconsistent")
    return row


def role_key(row: Mapping[str, Any], role: str) -> tuple[float, ...]:
    tiny = (
        float("-inf")
        if row.get("tiny_pd") is None
        else float(row["tiny_pd"])
    )
    epoch = int(row["epoch"])
    if role == "best_miou":
        return (
            float(row["miou"]),
            float(row["pd"]),
            -float(row["fa"]),
            float(row["niou"]),
            tiny,
            -float(row["test_loss"]),
            -float(epoch),
        )
    if role == "best_pd":
        return (
            float(row["pd"]),
            -float(row["fa"]),
            tiny,
            float(row["miou"]),
            float(row["niou"]),
            -float(row["test_loss"]),
            -float(epoch),
        )
    raise DCSPGAblationError(f"unsupported checkpoint role: {role!r}")


def role_key_record(row: Mapping[str, Any], role: str) -> list[float | None]:
    return [value if math.isfinite(value) else None for value in role_key(row, role)]


def _select_roles(history: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    if not history:
        return {role: None for role in ROLE_NAMES}
    return {
        role: int(max(history, key=lambda row: role_key(row, role))["epoch"])
        for role in ROLE_NAMES
    }


def _validate_metric_history(
    history: Any,
    *,
    completed_epoch: int,
    test_begin: int,
    test_every: int,
) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    if not isinstance(history, list):
        raise DCSPGAblationError("test history is malformed")
    expected_epochs = expected_test_epochs(
        completed_epoch, test_begin, test_every
    )
    observed_epochs = [
        int(row.get("epoch", -1)) if isinstance(row, Mapping) else -1
        for row in history
    ]
    if observed_epochs != expected_epochs:
        raise DCSPGAblationError("test history cadence differs")
    normalized: list[dict[str, Any]] = []
    for raw in history:
        assert isinstance(raw, Mapping)
        if raw.get("data_role") != "test":
            raise DCSPGAblationError("test history role differs")
        metrics = {key: raw.get(key) for key in METRIC_FIELDS}
        if set(raw) != {"epoch", "data_role", *METRIC_FIELDS}:
            raise DCSPGAblationError("test history fields differ")
        normalized.append(
            _metric_row(metrics, epoch=int(raw["epoch"]), data_role="test")
        )
    return normalized, _select_roles(normalized)


def _validate_training_history(
    history: Any,
    *,
    completed_epoch: int,
    processed_samples: int,
    loss_kind: str,
    farbg_lambda: float,
    total_epochs: int | None = None,
    base_lr: float | None = None,
    min_lr: float | None = None,
    warmup_epochs: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise DCSPGAblationError("training history length differs")
    normalized: list[dict[str, Any]] = []
    required = {
        "epoch",
        "base_bce_sum",
        "farbg_loss",
        "total_loss",
        "learning_rate",
        "processed_samples",
    }
    for expected_epoch, raw in enumerate(history, start=1):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise DCSPGAblationError("training history fields differ")
        base_loss = raw.get("base_bce_sum")
        farbg_loss = raw.get("farbg_loss")
        total_loss = raw.get("total_loss")
        learning_rate = raw.get("learning_rate")
        losses = (base_loss, farbg_loss, total_loss)
        if (
            raw.get("epoch") != expected_epoch
            or raw.get("processed_samples") != processed_samples
            or any(
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in losses
            )
            or isinstance(learning_rate, bool)
            or not isinstance(learning_rate, numbers.Real)
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
        ):
            raise DCSPGAblationError("training history record differs")
        expected_far = 0.0 if loss_kind == "original" else float(farbg_loss)
        if float(farbg_loss) != expected_far or not math.isclose(
            float(total_loss),
            float(base_loss) + farbg_lambda * float(farbg_loss),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise DCSPGAblationError("training loss decomposition differs")
        if total_epochs is not None:
            if base_lr is None or min_lr is None or warmup_epochs is None:
                raise DCSPGAblationError("training schedule validation is incomplete")
            expected_lr = legacy_train.learning_rate_for_epoch(
                expected_epoch,
                total_epochs,
                base_lr,
                min_lr,
                warmup_epochs,
            )
            if float(learning_rate) != expected_lr:
                raise DCSPGAblationError("training learning-rate history differs")
        normalized.append(
            {
                "epoch": expected_epoch,
                "base_bce_sum": float(base_loss),
                "farbg_loss": float(farbg_loss),
                "total_loss": float(total_loss),
                "learning_rate": float(learning_rate),
                "processed_samples": processed_samples,
            }
        )
    return normalized


def _disk_kernel_radius3(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(
        -FARBG_PROTECT_RADIUS,
        FARBG_PROTECT_RADIUS + 1,
        device=device,
    )
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    disk = (xx.square() + yy.square()) <= FARBG_PROTECT_RADIUS**2
    if int(torch.count_nonzero(disk).item()) != 29:
        raise DCSPGAblationError("radius-3 Euclidean disk must contain 29 points")
    return disk.to(dtype=dtype).reshape(1, 1, 7, 7)


def _training_heads(
    output: Any, target: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    if (
        not isinstance(target, torch.Tensor)
        or target.ndim != 4
        or target.shape[1] != 1
        or not target.is_floating_point()
        or not bool(torch.isfinite(target).all())
        or bool((target < 0).any())
        or bool((target > 1).any())
    ):
        raise DCSPGAblationError("training target violates probability contract")
    if not isinstance(output, (tuple, list)) or len(output) != 6:
        raise DCSPGAblationError("training model must return six probability heads")
    heads: list[torch.Tensor] = []
    for index, probability in enumerate(output):
        if (
            not isinstance(probability, torch.Tensor)
            or probability.shape != target.shape
            or probability.ndim != 4
            or not probability.is_floating_point()
            or probability.device != target.device
            or probability.dtype != target.dtype
            or not bool(torch.isfinite(probability).all())
            or bool((probability < 0).any())
            or bool((probability > 1).any())
        ):
            raise DCSPGAblationError(
                f"training probability head {index} violates its contract"
            )
        heads.append(probability)
    return tuple(heads)


def far_background_auxiliary_loss(
    output: Any, target: torch.Tensor
) -> torch.Tensor:
    """Frozen train-mask-only r3-disk/topK9 loss on raw d0 and final out."""

    heads = _training_heads(output, target)
    if (
        target.ndim != 4
        or target.shape[1] != 1
        or not target.is_floating_point()
        or not bool(torch.isfinite(target).all())
        or bool((target < 0).any())
        or bool((target > 1).any())
    ):
        raise DCSPGAblationError("training target violates far-background contract")
    ground_truth = (target.detach() > 0.5).to(dtype=heads[-1].dtype)
    disk = _disk_kernel_radius3(device=target.device, dtype=heads[-1].dtype)
    protected = F.conv2d(
        ground_truth,
        disk,
        stride=1,
        padding=FARBG_PROTECT_RADIUS,
    ) > 0
    far_background = ~protected

    per_head: list[torch.Tensor] = []
    for probability in (heads[-2], heads[-1]):
        penalty = (
            F.relu(probability - FARBG_THRESHOLD) / FARBG_THRESHOLD
        ).square()
        per_sample: list[torch.Tensor] = []
        for sample_index in range(int(probability.shape[0])):
            values = penalty[sample_index][far_background[sample_index]]
            count = int(values.numel())
            if count == 0:
                per_sample.append(probability[sample_index].sum() * 0.0)
            else:
                k = min(FARBG_TOPK, count)
                per_sample.append(torch.topk(values, k=k, sorted=False).values.mean())
        per_head.append(torch.stack(per_sample).mean())
    result = 0.5 * (per_head[0] + per_head[1])
    if not bool(torch.isfinite(result)) or result.ndim != 0:
        raise DCSPGAblationError("far-background auxiliary loss is non-finite")
    return result


def training_loss_components(
    output: Any,
    target: torch.Tensor,
    criterion: nn.Module,
    *,
    loss_kind: str,
) -> dict[str, torch.Tensor]:
    """Return auditable base, auxiliary, and total scalar losses."""

    heads = _training_heads(output, target)
    base = sum(criterion(probability.float(), target.float()) for probability in heads)
    if loss_kind == "original":
        auxiliary = base.detach() * 0.0
    elif loss_kind == "far_background_auxiliary":
        auxiliary = far_background_auxiliary_loss(heads, target)
    else:
        raise DCSPGAblationError("unsupported loss kind")
    total = base + FARBG_LAMBDA * auxiliary
    for name, value in (
        ("base_bce_sum", base),
        ("farbg_loss", auxiliary),
        ("total_loss", total),
    ):
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise DCSPGAblationError(f"{name} is not a finite scalar")
    return {
        "base_bce_sum": base,
        "farbg_loss": auxiliary,
        "total_loss": total,
    }


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path]:
    _reject_symlink_components(args.output_root, label="output root")
    output_root = args.output_root.resolve()
    if not args.smoke and output_root != DEFAULT_OUTPUT_ROOT.resolve():
        raise DCSPGAblationError(
            f"formal output root must be {DEFAULT_OUTPUT_ROOT}"
        )
    if args.smoke:
        run_dir = (
            output_root
            / "smoke"
            / args.method_id
            / f"run_seed_{args.run_seed}"
            / f"smoke_{args.smoke_id}"
        )
    else:
        run_dir = (
            output_root
            / "formal"
            / args.method_id
            / DATASET
            / f"run_seed_{args.run_seed}"
        )
    candidates = run_dir / "candidates/test_selected"
    published = run_dir / "published_weights"
    return {
        "output_root": output_root,
        "run_dir": run_dir,
        "candidate_dir": candidates,
        "published_dir": published,
        "latest": run_dir / "last_training_state.pth.tar",
        "training_history": run_dir / "training_history.json",
        "test_history": run_dir / "test_history.json",
        "test_access_started": run_dir / "test_access_started.json",
        "test_access_verified": run_dir / "test_access_verified.json",
        "selection": run_dir / "selection_record.json",
        "summary": run_dir / "summary.json",
        "best_miou": published / PUBLISHED_FILENAMES["best_miou"],
        "best_pd": published / PUBLISHED_FILENAMES["best_pd"],
        "lock": output_root / ".locks" / f"{args.method_id}_seed_{args.run_seed}.lock",
    }


@contextmanager
def _exclusive_run_lock(path: Path) -> Iterator[None]:
    parent = _ensure_regular_directory(path.parent, create=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parent / path.name, flags, 0o664)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DCSPGAblationError("run lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise DCSPGAblationError("method arm is already running") from exc
            raise
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _known_temp(path: Path) -> bool:
    if not path.name.startswith(".") or not path.name.endswith(".tmp"):
        return False
    names = {
        "last_training_state.pth.tar",
        "training_history.json",
        "test_history.json",
        "test_access_started.json",
        "test_access_verified.json",
        "selection_record.json",
        "summary.json",
        *PUBLISHED_FILENAMES.values(),
    }
    names.update(f"epoch_{epoch:04d}.pth.tar" for epoch in range(1, 1001))
    return any(path.name.startswith(f".{name}.") for name in names)


def _prepare_run_directory(
    paths: Mapping[str, Path], *, resume: bool
) -> None:
    run_dir = paths["run_dir"]
    _reject_symlink_components(run_dir, label="run directory")
    resolved_run, _resolved_output = _require_strict_descendant(
        run_dir,
        root=paths["output_root"],
        label="run directory",
    )
    if resolved_run != run_dir.resolve(strict=False):
        raise DCSPGAblationError("run directory canonicalization differs")
    if run_dir.is_symlink() or (run_dir.exists() and not run_dir.is_dir()):
        raise DCSPGAblationError("run directory is unsafe")
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        _ensure_regular_directory(run_dir, create=True)
        return
    if not run_dir.is_dir():
        raise FileNotFoundError(f"resume run directory is absent: {run_dir}")
    allowed_root = {
        "last_training_state.pth.tar",
        "training_history.json",
        "test_history.json",
        "test_access_started.json",
        "test_access_verified.json",
        "selection_record.json",
        "summary.json",
        "candidates",
        "published_weights",
    }
    for item in run_dir.iterdir():
        if item.name in allowed_root:
            if item.is_symlink():
                raise DCSPGAblationError("run artifact is a symlink")
            continue
        if _known_temp(item) and item.is_file() and not item.is_symlink():
            item.unlink()
            continue
        raise DCSPGAblationError(f"unexpected run artifact: {item.name}")
    for directory in (paths["candidate_dir"], paths["published_dir"]):
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise DCSPGAblationError("artifact directory is unsafe")
            for item in directory.iterdir():
                if _known_temp(item) and item.is_file() and not item.is_symlink():
                    item.unlink()


def _phase_flags(*, completed_epoch: int, test_history: Sequence[Any]) -> dict[str, bool]:
    del completed_epoch
    accessed = bool(test_history)
    return {
        "test_access_started": accessed,
        "test_access_verified": accessed,
        "test_index_opened": accessed,
        "test_split_accessed": accessed,
        "test_selected": accessed,
        "selection_is_optimistic": accessed,
        "unbiased_test_claim_supported": False,
        "stable_over_baseline_claim_supported": False,
    }


def _candidate_path(candidate_dir: Path, epoch: int) -> Path:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 1000:
        raise DCSPGAblationError("candidate epoch is invalid")
    return candidate_dir / f"epoch_{epoch:04d}.pth.tar"


def _candidate_epoch(path: Path) -> int:
    match = _CANDIDATE_RE.fullmatch(path.name)
    if match is None:
        raise DCSPGAblationError("candidate filename is invalid")
    return int(match.group(1))


def _candidate_payload(
    *,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    identity: Mapping[str, Any],
    test_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "method_id": identity["method_id"],
        "epoch": epoch,
        "run_identity": dict(identity),
        "test_record": dict(test_record),
        "state_dict": dict(state),
        "candidate_role": "test_selection_frontier",
        "diagnostic_only": False,
        **_phase_flags(completed_epoch=epoch, test_history=[test_record]),
    }


def _validate_candidate(
    path: Path,
    *,
    identity: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    expected_count: int,
    expected_record: Mapping[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise DCSPGAblationError("candidate is not a regular file")
    epoch = _candidate_epoch(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DCSPGAblationError("candidate checkpoint cannot be loaded") from exc
    required = {
        "schema",
        "model",
        "dataset",
        "method_id",
        "epoch",
        "run_identity",
        "test_record",
        "state_dict",
        "candidate_role",
        "diagnostic_only",
        "test_access_started",
        "test_access_verified",
        "test_index_opened",
        "test_split_accessed",
        "test_selected",
        "selection_is_optimistic",
        "unbiased_test_claim_supported",
        "stable_over_baseline_claim_supported",
    }
    record = payload.get("test_record") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != CANDIDATE_SCHEMA
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") != DATASET
        or payload.get("method_id") != identity["method_id"]
        or payload.get("epoch") != epoch
        or payload.get("run_identity") != identity
        or payload.get("candidate_role") != "test_selection_frontier"
        or payload.get("diagnostic_only") is not False
        or not isinstance(record, Mapping)
        or any(payload.get(key) is not value for key, value in _phase_flags(
            completed_epoch=epoch, test_history=[record]
        ).items())
    ):
        raise DCSPGAblationError("candidate identity differs")
    normalized = _metric_row(
        {key: record.get(key) for key in METRIC_FIELDS},
        epoch=epoch,
        data_role="test",
    )
    if dict(record) != normalized:
        raise DCSPGAblationError("candidate test record differs")
    if expected_record is not None and dict(expected_record) != normalized:
        raise DCSPGAblationError("candidate record differs from committed history")
    _validate_state_dict(
        payload.get("state_dict"),
        expected_state,
        expected_count=expected_count,
    )
    return epoch, normalized


def _frontier_artifacts(
    history: Sequence[Mapping[str, Any]], candidate_dir: Path
) -> dict[int, dict[str, Any]]:
    selected = _select_roles(history)
    role_by_epoch: dict[int, list[str]] = {}
    for role, raw_epoch in selected.items():
        if raw_epoch is not None:
            role_by_epoch.setdefault(int(raw_epoch), []).append(role)
    artifacts: dict[int, dict[str, Any]] = {}
    for epoch in sorted(role_by_epoch):
        path = _candidate_path(candidate_dir, epoch)
        if path.is_symlink() or not path.is_file():
            raise DCSPGAblationError("selection-frontier candidate is missing")
        artifacts[epoch] = {
            "relative_path": f"candidates/test_selected/{path.name}",
            "sha256": _sha256(path),
            "roles": sorted(role_by_epoch[epoch]),
        }
    return artifacts


def _save_current_candidate_if_selected(
    *,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    candidate_dir: Path,
) -> None:
    selected_epochs = set(
        epoch_value
        for epoch_value in _select_roles(history).values()
        if epoch_value is not None
    )
    if epoch not in selected_epochs:
        return
    path = _candidate_path(candidate_dir, epoch)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"candidate already exists: {path}")
    record = next(row for row in history if int(row["epoch"]) == epoch)
    _atomic_torch_save_no_clobber(
        path,
        _candidate_payload(
            epoch=epoch,
            state=state,
            identity=identity,
            test_record=record,
        ),
    )


def _reconcile_candidates(
    *,
    candidate_dir: Path,
    identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    artifacts: Mapping[int, Mapping[str, Any]],
    expected_state: Mapping[str, torch.Tensor],
    expected_count: int,
    completed_epoch: int,
    allow_uncommitted_next: bool,
) -> dict[int, dict[str, Any]]:
    expected = _frontier_artifacts(history, candidate_dir) if history else {}
    normalized_artifacts = {int(key): dict(value) for key, value in artifacts.items()}
    if normalized_artifacts != expected:
        raise DCSPGAblationError("candidate artifact map differs from frontier")
    by_epoch = {int(row["epoch"]): row for row in history}
    expected_paths = {_candidate_path(candidate_dir, epoch) for epoch in expected}
    if not candidate_dir.exists():
        if expected_paths:
            raise DCSPGAblationError("candidate directory is absent")
        return expected
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise DCSPGAblationError("candidate directory is unsafe")
    actual_paths: set[Path] = set()
    for path in candidate_dir.iterdir():
        if _known_temp(path) and path.is_file() and not path.is_symlink():
            path.unlink()
            continue
        if path.is_symlink() or not path.is_file():
            raise DCSPGAblationError("candidate directory contains unsafe entry")
        actual_paths.add(path)
    for path in expected_paths:
        epoch = _candidate_epoch(path)
        if path not in actual_paths or _sha256(path) != expected[epoch]["sha256"]:
            raise DCSPGAblationError("committed candidate SHA differs")
        _validate_candidate(
            path,
            identity=identity,
            expected_state=expected_state,
            expected_count=expected_count,
            expected_record=by_epoch[epoch],
        )
    for path in sorted(actual_paths - expected_paths):
        epoch = _candidate_epoch(path)
        expected_record = by_epoch.get(epoch)
        if expected_record is None and not (
            allow_uncommitted_next and epoch == completed_epoch + 1
        ):
            raise DCSPGAblationError("unknown candidate cannot be recovered")
        _validate_candidate(
            path,
            identity=identity,
            expected_state=expected_state,
            expected_count=expected_count,
            expected_record=expected_record,
        )
        path.unlink()
    return expected


def _run_identity(
    *,
    args: argparse.Namespace,
    method: Mapping[str, Any],
    architecture: Mapping[str, Any],
    architecture_validation: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Build the immutable identity without performing any live test access."""

    if int(data_identity.get("train_count", -1)) < 1:
        raise DCSPGAblationError("run identity train count is invalid")
    identity: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "method_id": args.method_id,
        "architecture_seed": args.architecture_seed,
        "run_seed": args.run_seed,
        "architecture": dict(architecture),
        "architecture_validation": dict(architecture_validation),
        "source_manifest": dict(source_manifest),
        "data_identity": dict(data_identity),
        "train_count": int(data_identity["train_count"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_lr": float(args.base_lr),
        "min_lr": float(args.min_lr),
        "warmup_epochs": args.warmup_epochs,
        "optimizer": "torch.optim.Adam_single_group_default_hyperparameters",
        "loss_recipe": dict(method["loss_recipe"]),
        "amp": False,
        "patch_size": 256,
        "training_target_rule": TRAINING_TARGET_RULE,
        "shuffle_seed_algorithm": (
            "sha256_length_prefixed_str_parts_uint63(seed,dataset,shuffle,epoch)"
        ),
        "test_schedule": {
            "begin_epoch": args.test_begin,
            "end_epoch": args.epochs,
            "every": args.test_every,
            "expected_count": len(
                expected_test_epochs(args.epochs, args.test_begin, args.test_every)
            ),
            "lazy_access": True,
        },
        "evaluator": {
            "callable": "test.evaluate_model",
            "common_evaluator_version": "v1",
            "evaluation_head": "out",
            "probability_threshold": float(PROBABILITY_THRESHOLD),
            "match_radius": float(MATCH_RADIUS),
            "tiny_area": int(TINY_AREA),
            "batch_size": 1,
        },
        "selector": {
            "pool": f"official_test_epochs_{args.test_begin}_to_{args.epochs}",
            "comparison": "strict_lexicographic_greater_than",
            "exact_tie": "earliest_epoch_via_negative_epoch",
            "best_miou": ["miou", "pd", "-fa", "niou", "tiny_pd", "-test_loss", "-epoch"],
            "best_pd": ["pd", "-fa", "tiny_pd", "miou", "niou", "-test_loss", "-epoch"],
        },
        "output_directory": str(run_dir.resolve()),
        "device": str(args.device),
        "smoke": bool(args.smoke),
        "smoke_id": args.smoke_id if args.smoke else None,
        "test_index_opened_at_identity_construction": False,
        "startup_test_index_opened": False,
        "test_split_accessed_during_preflight": False,
        "historical_test_selected_baseline_comparison_allowed": True,
        "test_used_for_structure_or_hyperparameter_selection": False,
        "unbiased_test_claim_supported": False,
        "stable_over_baseline_claim_supported": False,
    }
    identity = json.loads(_canonical_bytes(identity).decode("utf-8"))
    identity["identity_sha256"] = _canonical_sha256(identity)
    _validate_identity(identity)
    return identity


def _validate_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise DCSPGAblationError("run identity is malformed")
    normalized = dict(identity)
    digest = normalized.pop("identity_sha256", None)
    if (
        not isinstance(digest, str)
        or _SHA_RE.fullmatch(digest) is None
        or digest != _canonical_sha256(normalized)
        or normalized.get("schema") != RUN_SCHEMA
        or normalized.get("model") != "EviSIRST"
        or normalized.get("dataset") != DATASET
        or normalized.get("test_index_opened_at_identity_construction") is not False
        or normalized.get("startup_test_index_opened") is not False
        or normalized.get("test_split_accessed_during_preflight") is not False
    ):
        raise DCSPGAblationError("run identity digest or startup boundary differs")
    return dict(identity)


def _started_ledger_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    data_identity = identity.get("data_identity")
    schedule = identity.get("test_schedule")
    if not isinstance(data_identity, Mapping) or not isinstance(schedule, Mapping):
        raise DCSPGAblationError("identity lacks lazy-test contract")
    return {
        "schema": TEST_ACCESS_STARTED_SCHEMA,
        "dataset": DATASET,
        "method_id": identity["method_id"],
        "run_identity_sha256": identity["identity_sha256"],
        "expected_test_contract": dict(data_identity["expected_test_contract"]),
        "first_access_epoch": int(schedule["begin_epoch"]),
        "test_access_started": True,
        "live_test_open_may_have_occurred": True,
        "test_index_opened_status": "unknown_until_verified",
        "test_split_accessed_status": "unknown_until_verified",
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
        "stable_over_baseline_claim_supported": False,
    }


def _verified_ledger_payload(
    identity: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": TEST_ACCESS_VERIFIED_SCHEMA,
        "dataset": DATASET,
        "method_id": identity["method_id"],
        "run_identity_sha256": identity["identity_sha256"],
        "first_access_epoch": int(identity["test_schedule"]["begin_epoch"]),
        "observed_test_identity": dict(observed),
        "test_access_started": True,
        "test_access_verified": True,
        "test_index_opened": True,
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
        "stable_over_baseline_claim_supported": False,
    }


def _read_exact_json(path: Path, expected: Mapping[str, Any], *, label: str) -> None:
    observed = _strict_json(path)
    if observed != dict(expected):
        raise DCSPGAblationError(f"{label} differs from its immutable contract")


def _ensure_test_access_started(
    paths: Mapping[str, Path], identity: Mapping[str, Any]
) -> str:
    path = paths["test_access_started"]
    expected = _started_ledger_payload(identity)
    if path.exists() or path.is_symlink():
        _read_exact_json(path, expected, label="test-access-started ledger")
    else:
        _write_json_no_clobber(path, expected)
    return _sha256(path)


def _activate_test_transaction(
    args: argparse.Namespace,
    *,
    paths: Mapping[str, Path],
    identity: Mapping[str, Any],
) -> tuple[EviSIRSTTestDataset, Any, dict[str, Any]]:
    """Commit access intent before calling the sole live-test constructor."""

    _ensure_test_access_started(paths, identity)
    full_test, test_data, observed = _activate_test_data(
        args, identity["data_identity"]
    )
    expected = _verified_ledger_payload(identity, observed)
    verified_path = paths["test_access_verified"]
    if verified_path.exists() or verified_path.is_symlink():
        _read_exact_json(
            verified_path, expected, label="test-access-verified ledger"
        )
    else:
        _write_json_no_clobber(verified_path, expected)
    return full_test, test_data, observed


def _validate_access_ledgers(
    paths: Mapping[str, Path], identity: Mapping[str, Any]
) -> tuple[bool, dict[str, Any] | None]:
    started_path = paths["test_access_started"]
    verified_path = paths["test_access_verified"]
    started = started_path.exists() or started_path.is_symlink()
    verified = verified_path.exists() or verified_path.is_symlink()
    if verified and not started:
        raise DCSPGAblationError("verified test access lacks its started ledger")
    if started:
        _read_exact_json(
            started_path,
            _started_ledger_payload(identity),
            label="test-access-started ledger",
        )
    observed: dict[str, Any] | None = None
    if verified:
        payload = _strict_json(verified_path)
        raw_observed = payload.get("observed_test_identity")
        if not isinstance(raw_observed, Mapping):
            raise DCSPGAblationError("verified test identity is malformed")
        observed = dict(raw_observed)
        if payload != _verified_ledger_payload(identity, observed):
            raise DCSPGAblationError("test-access-verified ledger differs")
    return started, observed


def _safe_torch_load(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DCSPGAblationError(f"{label} is not a regular file")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DCSPGAblationError(f"{label} cannot be loaded safely") from exc
    if not isinstance(payload, Mapping):
        raise DCSPGAblationError(f"{label} root is malformed")
    return payload


def _cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    def clone(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().clone()
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                torch.isfinite(tensor).all()
            ):
                raise DCSPGAblationError("optimizer state contains a non-finite tensor")
            return tensor
        if isinstance(value, Mapping):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        return value

    state = clone(optimizer.state_dict())
    if not isinstance(state, dict):
        raise DCSPGAblationError("optimizer state is malformed")
    return state


def _recovery_payload(
    *,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    rng: Mapping[str, Any],
    identity: Mapping[str, Any],
    training_history: Sequence[Mapping[str, Any]],
    test_history: Sequence[Mapping[str, Any]],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
    test_identity: Mapping[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    best_epochs = _select_roles(test_history)
    return {
        "schema": RECOVERY_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "method_id": identity["method_id"],
        "epoch": epoch,
        "run_identity": dict(identity),
        "state_dict": dict(state),
        "optimizer": dict(optimizer_state),
        "rng": dict(rng),
        "training_history": [dict(row) for row in training_history],
        "test_history": [dict(row) for row in test_history],
        "best_epochs": dict(best_epochs),
        "candidate_artifacts": {
            int(candidate_epoch): dict(record)
            for candidate_epoch, record in candidate_artifacts.items()
        },
        "test_identity": None if test_identity is None else dict(test_identity),
        "elapsed_seconds": float(elapsed_seconds),
        **_phase_flags(completed_epoch=epoch, test_history=test_history),
    }


def _recovery_required_fields() -> set[str]:
    return {
        "schema",
        "model",
        "dataset",
        "method_id",
        "epoch",
        "run_identity",
        "state_dict",
        "optimizer",
        "rng",
        "training_history",
        "test_history",
        "best_epochs",
        "candidate_artifacts",
        "test_identity",
        "elapsed_seconds",
        *_phase_flags(completed_epoch=0, test_history=[]),
    }


def _peek_recovery(
    path: Path, *, identity: Mapping[str, Any]
) -> tuple[Mapping[str, Any], int]:
    payload = _safe_torch_load(path, label="latest recovery")
    epoch = payload.get("epoch")
    if (
        set(payload) != _recovery_required_fields()
        or payload.get("schema") != RECOVERY_SCHEMA
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") != DATASET
        or payload.get("method_id") != identity["method_id"]
        or payload.get("run_identity") != identity
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= int(identity["epochs"])
    ):
        raise DCSPGAblationError("latest recovery identity differs")
    return payload, epoch


def _normalize_candidate_artifacts(value: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise DCSPGAblationError("candidate artifact map is malformed")
    normalized: dict[int, dict[str, Any]] = {}
    for raw_epoch, raw in value.items():
        if (
            isinstance(raw_epoch, bool)
            or not isinstance(raw_epoch, int)
            or not isinstance(raw, Mapping)
        ):
            raise DCSPGAblationError("candidate artifact entry is malformed")
        relative = raw.get("relative_path")
        digest = raw.get("sha256")
        roles = raw.get("roles")
        expected_relative = (
            f"candidates/test_selected/epoch_{raw_epoch:04d}.pth.tar"
        )
        if (
            set(raw) != {"relative_path", "sha256", "roles"}
            or relative != expected_relative
            or not isinstance(digest, str)
            or _SHA_RE.fullmatch(digest) is None
            or not isinstance(roles, list)
            or roles != sorted(set(roles))
            or not roles
            or any(role not in ROLE_NAMES for role in roles)
        ):
            raise DCSPGAblationError("candidate artifact contract differs")
        normalized[raw_epoch] = dict(raw)
    return normalized


def _validate_recovery(
    payload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    candidate_dir: Path,
    expected_count: int,
    observed_test_identity: Mapping[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    epoch = int(payload["epoch"])
    expected_state = model.state_dict()
    state = _validate_state_dict(
        payload.get("state_dict"), expected_state, expected_count=expected_count
    )
    method = identity["loss_recipe"]
    if not isinstance(method, Mapping):
        raise DCSPGAblationError("recovery loss identity is malformed")
    training_history = _validate_training_history(
        payload.get("training_history"),
        completed_epoch=epoch,
        processed_samples=int(identity["train_count"]),
        loss_kind=str(method["kind"]),
        farbg_lambda=FARBG_LAMBDA,
        total_epochs=int(identity["epochs"]),
        base_lr=float(identity["base_lr"]),
        min_lr=float(identity["min_lr"]),
        warmup_epochs=int(identity["warmup_epochs"]),
    )
    schedule = identity["test_schedule"]
    test_history, best_epochs = _validate_metric_history(
        payload.get("test_history"),
        completed_epoch=epoch,
        test_begin=int(schedule["begin_epoch"]),
        test_every=int(schedule["every"]),
    )
    stored_best = payload.get("best_epochs")
    if not isinstance(stored_best, Mapping) or dict(stored_best) != best_epochs:
        raise DCSPGAblationError("recovery dual-role selection differs")
    expected_flags = _phase_flags(completed_epoch=epoch, test_history=test_history)
    if any(payload.get(key) != value for key, value in expected_flags.items()):
        raise DCSPGAblationError("recovery test-access disclosure differs")
    stored_test_identity = payload.get("test_identity")
    if test_history:
        if (
            not isinstance(stored_test_identity, Mapping)
            or observed_test_identity is None
            or dict(stored_test_identity) != dict(observed_test_identity)
        ):
            raise DCSPGAblationError("recovery live test identity differs")
    elif stored_test_identity is not None:
        raise DCSPGAblationError("pre-test recovery contains a test identity")
    elapsed = payload.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, numbers.Real)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise DCSPGAblationError("recovery elapsed time is malformed")
    artifacts = _normalize_candidate_artifacts(payload.get("candidate_artifacts"))
    artifacts = _reconcile_candidates(
        candidate_dir=candidate_dir,
        identity=identity,
        history=test_history,
        artifacts=artifacts,
        expected_state=expected_state,
        expected_count=expected_count,
        completed_epoch=epoch,
        allow_uncommitted_next=True,
    )
    try:
        r1_transaction._validate_and_load_adam_optimizer_state(
            optimizer_state=payload.get("optimizer"),
            model=model,
            optimizer=optimizer,
            identity=identity,
            completed_epoch=epoch,
            total_epochs=int(identity["epochs"]),
        )
    except Exception as exc:
        raise DCSPGAblationError("recovery Adam state differs") from exc
    model.load_state_dict(state, strict=True)
    try:
        legacy_train._restore_rng_state(payload.get("rng"), device)
    except Exception as exc:
        raise DCSPGAblationError("recovery RNG state differs") from exc
    return {
        "epoch": epoch,
        "training_history": training_history,
        "test_history": test_history,
        "best_epochs": best_epochs,
        "candidate_artifacts": artifacts,
        "test_identity": (
            None if stored_test_identity is None else dict(stored_test_identity)
        ),
        "elapsed_seconds": float(elapsed),
    }


def _history_payload(
    *,
    data_role: str,
    identity: Mapping[str, Any],
    completed_epoch: int,
    history: Sequence[Mapping[str, Any]],
    test_history_for_flags: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if data_role == "train":
        history_key = "training_history"
        phase_history: Sequence[Any] = (
            [] if test_history_for_flags is None else test_history_for_flags
        )
    elif data_role == "test":
        history_key = "test_history"
        phase_history = history
    else:
        raise DCSPGAblationError("history sidecar role is unsupported")
    return {
        "schema": HISTORY_SCHEMA,
        "data_role": data_role,
        "method_id": identity["method_id"],
        "run_identity": dict(identity),
        "completed_epoch": completed_epoch,
        history_key: [dict(row) for row in history],
        **_phase_flags(completed_epoch=completed_epoch, test_history=phase_history),
    }


def _sidecar_expectations(
    *,
    data_role: str,
    identity: Mapping[str, Any],
    completed_epoch: int,
    history: Sequence[Mapping[str, Any]],
    test_history_for_flags: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    current = _history_payload(
        data_role=data_role,
        identity=identity,
        completed_epoch=completed_epoch,
        history=history,
        test_history_for_flags=test_history_for_flags,
    )
    expectations = [current]
    if completed_epoch > 1:
        previous_epoch = completed_epoch - 1
        if data_role == "train":
            previous_history = history[:-1]
        else:
            previous_history = [
                row for row in history if int(row["epoch"]) <= previous_epoch
            ]
        expectations.append(
            _history_payload(
                data_role=data_role,
                identity=identity,
                completed_epoch=previous_epoch,
                history=previous_history,
                test_history_for_flags=[
                    row
                    for row in test_history_for_flags
                    if int(row["epoch"]) <= previous_epoch
                ],
            )
        )
    return expectations


def _validate_and_commit_sidecars(
    *,
    paths: Mapping[str, Path],
    identity: Mapping[str, Any],
    completed_epoch: int,
    training_history: Sequence[Mapping[str, Any]],
    test_history: Sequence[Mapping[str, Any]],
) -> None:
    for path_key, role, history in (
        ("training_history", "train", training_history),
        ("test_history", "test", test_history),
    ):
        path = paths[path_key]
        current = _history_payload(
            data_role=role,
            identity=identity,
            completed_epoch=completed_epoch,
            history=history,
            test_history_for_flags=test_history,
        )
        if path.exists() or path.is_symlink():
            observed = _strict_json(path)
            if observed not in _sidecar_expectations(
                data_role=role,
                identity=identity,
                completed_epoch=completed_epoch,
                history=history,
                test_history_for_flags=test_history,
            ):
                raise DCSPGAblationError(f"{role} history sidecar is not committed")
        _write_json_atomic(path, current)


def _audit_artifact_layout(paths: Mapping[str, Path]) -> None:
    run_dir = paths["run_dir"]
    candidate_root = run_dir / "candidates"
    candidate_dir = paths["candidate_dir"]
    if candidate_root.exists():
        if candidate_root.is_symlink() or not candidate_root.is_dir():
            raise DCSPGAblationError("candidate root is unsafe")
        entries = list(candidate_root.iterdir())
        if any(item.name != "test_selected" for item in entries):
            raise DCSPGAblationError("candidate root contains an unknown entry")
        if entries and (
            candidate_dir.is_symlink() or not candidate_dir.is_dir()
        ):
            raise DCSPGAblationError("test-selected candidate directory is unsafe")
    published = paths["published_dir"]
    if published.exists():
        if published.is_symlink() or not published.is_dir():
            raise DCSPGAblationError("published directory is unsafe")
        allowed = set(PUBLISHED_FILENAMES.values())
        for item in published.iterdir():
            if item.name not in allowed or item.is_symlink() or not item.is_file():
                if _known_temp(item) and item.is_file() and not item.is_symlink():
                    item.unlink()
                    continue
                raise DCSPGAblationError("published directory contains unknown data")


def _test_loader(
    test_data: Any, *, args: argparse.Namespace, device: torch.device
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_uint63(args.run_seed, DATASET, "official_test_evaluation")
    )
    return DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )


def _train_one_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    full_train: EviSIRSTTrainDataset,
    train_data: Any,
    args: argparse.Namespace,
    method: Mapping[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    full_train.set_epoch(epoch)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_uint63(args.run_seed, DATASET, "shuffle", epoch)
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )
    learning_rate = legacy_train.learning_rate_for_epoch(
        epoch,
        args.epochs,
        args.base_lr,
        args.min_lr,
        args.warmup_epochs,
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    model.train()
    model.mode = "train"
    sums = {"base_bce_sum": 0.0, "farbg_loss": 0.0, "total_loss": 0.0}
    processed = 0
    for images, masks in loader:
        if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
            raise DCSPGAblationError("training loader returned malformed tensors")
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        components = training_loss_components(
            model(images),
            masks,
            criterion,
            loss_kind=str(method["loss_kind"]),
        )
        components["total_loss"].backward()
        optimizer.step()
        count = int(images.shape[0])
        if count < 1:
            raise DCSPGAblationError("training batch is empty")
        for name in sums:
            sums[name] += float(components[name].detach().item()) * count
        processed += count
    expected_processed = len(train_data)
    if processed != expected_processed or processed < 1:
        raise DCSPGAblationError("processed training sample count differs")
    return {
        "epoch": epoch,
        "base_bce_sum": sums["base_bce_sum"] / processed,
        "farbg_loss": sums["farbg_loss"] / processed,
        "total_loss": sums["total_loss"] / processed,
        "learning_rate": learning_rate,
        "processed_samples": processed,
    }


def _state_tensors_equal(
    observed: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]
) -> bool:
    return list(observed) == list(expected) and all(
        isinstance(observed[key], torch.Tensor)
        and observed[key].dtype == expected[key].dtype
        and observed[key].shape == expected[key].shape
        and torch.equal(observed[key].cpu(), expected[key].cpu())
        for key in expected
    )


def _checkpoint_payload(
    *,
    role: str,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    metrics: Mapping[str, Any],
    identity: Mapping[str, Any],
    test_identity: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    started_sha256: str,
    verified_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "method_id": identity["method_id"],
        "checkpoint_role": role,
        "epoch": epoch,
        "run_identity": dict(identity),
        "state_dict": dict(state),
        "selection_metrics": dict(metrics),
        "selection_role_key": role_key_record(metrics, role),
        "candidate_relative_path": candidate_record["relative_path"],
        "candidate_sha256": candidate_record["sha256"],
        "test_identity": dict(test_identity),
        "test_access_started_ledger_sha256": started_sha256,
        "test_access_verified_ledger_sha256": verified_sha256,
        "this_checkpoint_selected_by_test": True,
        **_phase_flags(completed_epoch=epoch, test_history=[metrics]),
    }


def _validate_checkpoint_payload(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    expected_count: int,
) -> None:
    if set(observed) != set(expected):
        raise DCSPGAblationError("published checkpoint fields differ")
    observed_state = observed.get("state_dict")
    expected_state = expected["state_dict"]
    if not isinstance(observed_state, Mapping) or not isinstance(expected_state, Mapping):
        raise DCSPGAblationError("published checkpoint state is malformed")
    _validate_state_dict(
        observed_state, expected_state, expected_count=expected_count
    )
    if not _state_tensors_equal(observed_state, expected_state):
        raise DCSPGAblationError("published checkpoint tensor values differ")
    for key in expected:
        if key != "state_dict" and observed.get(key) != expected[key]:
            raise DCSPGAblationError(f"published checkpoint field differs: {key}")


def _publish_or_validate(
    path: Path, payload: Mapping[str, Any], *, expected_count: int
) -> str:
    if path.exists() or path.is_symlink():
        observed = _safe_torch_load(path, label="published checkpoint")
        _validate_checkpoint_payload(
            observed, payload, expected_count=expected_count
        )
    else:
        _atomic_torch_save_no_clobber(path, payload)
        reopened = _safe_torch_load(path, label="published checkpoint")
        _validate_checkpoint_payload(
            reopened, payload, expected_count=expected_count
        )
    return _sha256(path)


def _commit_json_no_clobber_or_validate(
    path: Path, payload: Mapping[str, Any], *, label: str
) -> str:
    if path.exists() or path.is_symlink():
        _read_exact_json(path, payload, label=label)
    else:
        _write_json_no_clobber(path, payload)
        _read_exact_json(path, payload, label=label)
    return _sha256(path)


def _finalize(
    *,
    args: argparse.Namespace,
    rules: Mapping[str, Any],
    paths: Mapping[str, Path],
    identity: Mapping[str, Any],
    model: nn.Module,
    expected_count: int,
    training_history: Sequence[Mapping[str, Any]],
    test_history: Sequence[Mapping[str, Any]],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
    test_identity: Mapping[str, Any] | None,
) -> Path:
    if len(training_history) != args.epochs:
        raise DCSPGAblationError("final training history is incomplete")
    normalized_test, best_epochs = _validate_metric_history(
        list(test_history),
        completed_epoch=args.epochs,
        test_begin=args.test_begin,
        test_every=args.test_every,
    )
    expected_count_history = len(
        expected_test_epochs(args.epochs, args.test_begin, args.test_every)
    )
    if len(normalized_test) != expected_count_history:
        raise DCSPGAblationError("final test history count differs")
    if not args.smoke and (
        expected_count_history != 500
        or [int(row["epoch"]) for row in normalized_test]
        != list(range(501, 1001))
    ):
        raise DCSPGAblationError("formal test history must contain exactly epochs 501--1000")
    if test_identity is None or any(best_epochs[role] is None for role in ROLE_NAMES):
        raise DCSPGAblationError("final test selection is incomplete")

    started, ledger_identity = _validate_access_ledgers(paths, identity)
    if not started or ledger_identity is None or ledger_identity != dict(test_identity):
        raise DCSPGAblationError("final test-access ledgers are incomplete")
    started_sha = _sha256(paths["test_access_started"])
    verified_sha = _sha256(paths["test_access_verified"])

    if _source_manifest(formal=not args.smoke) != identity["source_manifest"]:
        raise DCSPGAblationError("source set changed during training")
    _full_train, _train_data, final_data_identity = _data_identity(args, rules)
    if final_data_identity != identity["data_identity"]:
        raise DCSPGAblationError("full-train identity changed during training")

    expected_artifacts = _frontier_artifacts(normalized_test, paths["candidate_dir"])
    normalized_artifacts = _normalize_candidate_artifacts(candidate_artifacts)
    if normalized_artifacts != expected_artifacts:
        raise DCSPGAblationError("final candidate frontier differs")
    by_epoch = {int(row["epoch"]): row for row in normalized_test}
    published_records: dict[str, Any] = {}
    selection_roles: dict[str, Any] = {}
    for role in ROLE_NAMES:
        epoch = int(best_epochs[role])
        artifact = normalized_artifacts.get(epoch)
        if artifact is None or role not in artifact["roles"]:
            raise DCSPGAblationError("selected role lacks its candidate artifact")
        candidate_path = paths["run_dir"] / artifact["relative_path"]
        if _sha256(candidate_path) != artifact["sha256"]:
            raise DCSPGAblationError("selected candidate SHA differs")
        candidate = _safe_torch_load(candidate_path, label="selected candidate")
        _validate_candidate(
            candidate_path,
            identity=identity,
            expected_state=model.state_dict(),
            expected_count=expected_count,
            expected_record=by_epoch[epoch],
        )
        state = _validate_state_dict(
            candidate.get("state_dict"),
            model.state_dict(),
            expected_count=expected_count,
        )
        checkpoint = _checkpoint_payload(
            role=role,
            epoch=epoch,
            state=state,
            metrics=by_epoch[epoch],
            identity=identity,
            test_identity=test_identity,
            candidate_record=artifact,
            started_sha256=started_sha,
            verified_sha256=verified_sha,
        )
        checkpoint_path = paths[role]
        checkpoint_sha = _publish_or_validate(
            checkpoint_path, checkpoint, expected_count=expected_count
        )
        published_records[role] = {
            "relative_path": (
                f"published_weights/{checkpoint_path.name}"
            ),
            "sha256": checkpoint_sha,
            "epoch": epoch,
        }
        selection_roles[role] = {
            "epoch": epoch,
            "metrics": dict(by_epoch[epoch]),
            "role_key": role_key_record(by_epoch[epoch], role),
            "candidate": dict(artifact),
            "published_checkpoint": dict(published_records[role]),
        }

    published_files = sorted(paths["published_dir"].iterdir())
    if (
        len(published_files) != 2
        or {path.name for path in published_files}
        != set(PUBLISHED_FILENAMES.values())
        or any(path.is_symlink() or not path.is_file() for path in published_files)
    ):
        raise DCSPGAblationError("each arm must publish exactly two regular weights")

    selection_payload = {
        "schema": SELECTION_SCHEMA,
        "status": "complete",
        "model": "EviSIRST",
        "dataset": DATASET,
        "method_id": identity["method_id"],
        "run_identity": dict(identity),
        "selection_pool": identity["selector"]["pool"],
        "test_history_count": len(normalized_test),
        "roles": selection_roles,
        "test_identity": dict(test_identity),
        "test_access_started_ledger_sha256": started_sha,
        "test_access_verified_ledger_sha256": verified_sha,
        "historical_test_selected_baseline_comparison_allowed": True,
        "test_used_for_structure_or_hyperparameter_selection": False,
        **_phase_flags(completed_epoch=args.epochs, test_history=normalized_test),
    }
    selection_sha = _commit_json_no_clobber_or_validate(
        paths["selection"], selection_payload, label="selection record"
    )
    for key in ("training_history", "test_history"):
        if paths[key].is_symlink() or not paths[key].is_file():
            raise DCSPGAblationError("final history sidecar is absent")
    summary_payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "model": "EviSIRST",
        "dataset": DATASET,
        "method_id": identity["method_id"],
        "run_identity": dict(identity),
        "completed_epoch": args.epochs,
        "train_count": int(identity["train_count"]),
        "training_history_count": len(training_history),
        "training_history_sha256": _sha256(paths["training_history"]),
        "test_history_count": len(normalized_test),
        "test_history_sha256": _sha256(paths["test_history"]),
        "test_access_started_ledger_sha256": started_sha,
        "test_access_verified_ledger_sha256": verified_sha,
        "selection_record_sha256": selection_sha,
        "published_checkpoints": published_records,
        "published_weight_file_count": 2,
        "complete_six_arm_ablation_claim_supported": False,
        "test_used_for_structure_or_hyperparameter_selection": False,
        **_phase_flags(completed_epoch=args.epochs, test_history=normalized_test),
    }
    _commit_json_no_clobber_or_validate(
        paths["summary"], summary_payload, label="completion summary"
    )
    return paths["published_dir"]


def run(args: argparse.Namespace) -> Path:
    """Execute or strictly resume one independently isolated method arm."""

    torch.set_num_threads(args.cpu_threads)
    rules = _load_rules()
    method = _method_contract(args, rules)
    # The architecture audit gate is checked before device validation, lock
    # creation, or any run-directory mutation.  A NO-GO candidate therefore
    # cannot allocate CUDA state or leave a plausible formal run behind.
    source_manifest = _source_manifest(formal=not args.smoke)
    legacy_train.configure_determinism(args.run_seed)
    device = legacy_train.require_device(args.device)
    paths = resolve_run_paths(args)

    with _exclusive_run_lock(paths["lock"]):
        _prepare_run_directory(paths, resume=args.resume)
        _audit_artifact_layout(paths)
        if not args.resume and any(
            paths[key].exists() or paths[key].is_symlink()
            for key in (
                "latest",
                "selection",
                "summary",
                "test_access_started",
                "test_access_verified",
                "best_miou",
                "best_pd",
            )
        ):
            raise FileExistsError("fresh run would clobber an existing artifact")

        full_train, train_data, data_identity = _data_identity(args, rules)
        model, _metadata, architecture_validation, architecture_identity = (
            _build_architecture(method)
        )
        expected_count = int(architecture_identity["state_key_count"])
        identity = _run_identity(
            args=args,
            method=method,
            architecture=architecture_identity,
            architecture_validation=architecture_validation,
            source_manifest=source_manifest,
            data_identity=data_identity,
            run_dir=paths["run_dir"],
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
        criterion = nn.BCELoss(reduction="mean")

        start_epoch = 1
        training_history: list[dict[str, Any]] = []
        test_history: list[dict[str, Any]] = []
        candidate_artifacts: dict[int, dict[str, Any]] = {}
        test_identity: dict[str, Any] | None = None
        test_data: Any = None
        evaluation_loader: DataLoader | None = None
        elapsed_before = 0.0

        if args.resume:
            payload, completed_epoch = _peek_recovery(
                paths["latest"], identity=identity
            )
            access_started, ledger_test_identity = _validate_access_ledgers(
                paths, identity
            )
            if access_started and completed_epoch < args.test_begin - 1:
                raise DCSPGAblationError(
                    "test access started before the frozen epoch boundary"
                )
            if completed_epoch >= args.test_begin and (
                not access_started or ledger_test_identity is None
            ):
                raise DCSPGAblationError(
                    "post-test recovery lacks committed access ledgers"
                )
            if access_started:
                _full_test, test_data, live_test_identity = (
                    _activate_test_transaction(
                        args, paths=paths, identity=identity
                    )
                )
                if (
                    ledger_test_identity is not None
                    and live_test_identity != ledger_test_identity
                ):
                    raise DCSPGAblationError(
                        "live test identity changed since its verified ledger"
                    )
                test_identity = live_test_identity
                evaluation_loader = _test_loader(
                    test_data, args=args, device=device
                )
            recovered = _validate_recovery(
                payload,
                identity=identity,
                model=model,
                optimizer=optimizer,
                candidate_dir=paths["candidate_dir"],
                expected_count=expected_count,
                observed_test_identity=test_identity,
                device=device,
            )
            training_history = list(recovered["training_history"])
            test_history = list(recovered["test_history"])
            candidate_artifacts = dict(recovered["candidate_artifacts"])
            if recovered["test_identity"] is not None:
                test_identity = dict(recovered["test_identity"])
            elapsed_before = float(recovered["elapsed_seconds"])
            start_epoch = completed_epoch + 1
            _validate_and_commit_sidecars(
                paths=paths,
                identity=identity,
                completed_epoch=completed_epoch,
                training_history=training_history,
                test_history=test_history,
            )
            if completed_epoch < args.epochs and any(
                paths[key].exists() or paths[key].is_symlink()
                for key in ("selection", "summary", "best_miou", "best_pd")
            ):
                raise DCSPGAblationError(
                    "incomplete recovery contains premature final artifacts"
                )
        elif paths["run_dir"].exists() and any(paths["run_dir"].iterdir()):
            # The fresh directory itself is allowed, but no run artifact is.
            unexpected = [
                item
                for item in paths["run_dir"].iterdir()
                if item.name not in {"candidates", "published_weights"}
            ]
            if unexpected:
                raise FileExistsError("fresh run directory is not empty")

        started_at = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            train_record = _train_one_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                full_train=full_train,
                train_data=train_data,
                args=args,
                method=method,
                device=device,
                epoch=epoch,
            )
            training_history.append(train_record)

            metrics: dict[str, Any] | None = None
            if test_due(epoch, args.test_begin, args.test_every):
                if test_data is None:
                    _full_test, test_data, test_identity = (
                        _activate_test_transaction(
                            args, paths=paths, identity=identity
                        )
                    )
                    evaluation_loader = _test_loader(
                        test_data, args=args, device=device
                    )
                if evaluation_loader is None or test_identity is None:
                    raise DCSPGAblationError("official test loader is unavailable")
                training_rng = legacy_train._capture_rng_state(device)
                try:
                    raw_metrics = evaluate_model(
                        model,
                        evaluation_loader,
                        device,
                        threshold=PROBABILITY_THRESHOLD,
                        match_radius=MATCH_RADIUS,
                        tiny_area=TINY_AREA,
                    )
                    metrics = _metric_row(
                        raw_metrics, epoch=epoch, data_role="test"
                    )
                finally:
                    legacy_train._restore_rng_state(training_rng, device)
                test_history.append(metrics)

            state = _cpu_state(model, expected_count=expected_count)
            _save_current_candidate_if_selected(
                epoch=epoch,
                state=state,
                identity=identity,
                history=test_history,
                candidate_dir=paths["candidate_dir"],
            )
            candidate_artifacts = (
                _frontier_artifacts(test_history, paths["candidate_dir"])
                if test_history
                else {}
            )
            elapsed = elapsed_before + time.time() - started_at
            recovery = _recovery_payload(
                epoch=epoch,
                state=state,
                optimizer_state=_cpu_optimizer_state(optimizer),
                rng=legacy_train._capture_rng_state(device),
                identity=identity,
                training_history=training_history,
                test_history=test_history,
                candidate_artifacts=candidate_artifacts,
                test_identity=test_identity if test_history else None,
                elapsed_seconds=elapsed,
            )
            # This replace is the epoch transaction's sole commit point.
            _atomic_torch_save(paths["latest"], recovery)
            candidate_artifacts = _reconcile_candidates(
                candidate_dir=paths["candidate_dir"],
                identity=identity,
                history=test_history,
                artifacts=candidate_artifacts,
                expected_state=model.state_dict(),
                expected_count=expected_count,
                completed_epoch=epoch,
                allow_uncommitted_next=False,
            )
            _validate_and_commit_sidecars(
                paths=paths,
                identity=identity,
                completed_epoch=epoch,
                training_history=training_history,
                test_history=test_history,
            )
            suffix = ""
            if metrics is not None:
                best = _select_roles(test_history)
                suffix = (
                    f" test_mIoU={100.0 * float(metrics['miou']):.6f}%"
                    f" nIoU={100.0 * float(metrics['niou']):.6f}%"
                    f" F1={100.0 * float(metrics['pixel_f1']):.6f}%"
                    f" Pd={100.0 * float(metrics['pd']):.6f}%"
                    f" Fa={1e6 * float(metrics['fa']):.6f}e-6"
                    f" best_mIoU_epoch={best['best_miou']}"
                    f" best_Pd_epoch={best['best_pd']}"
                )
            print(
                f"method={args.method_id} epoch={epoch}/{args.epochs} "
                f"base_bce={train_record['base_bce_sum']:.6f} "
                f"farbg={train_record['farbg_loss']:.6f} "
                f"total={train_record['total_loss']:.6f} "
                f"lr={train_record['learning_rate']:.8f} "
                f"samples={train_record['processed_samples']}{suffix}",
                flush=True,
            )

        return _finalize(
            args=args,
            rules=rules,
            paths=paths,
            identity=identity,
            model=model,
            expected_count=expected_count,
            training_history=training_history,
            test_history=test_history,
            candidate_artifacts=candidate_artifacts,
            test_identity=test_identity,
        )


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(f"completed={result}", flush=True)


if __name__ == "__main__":
    main()
