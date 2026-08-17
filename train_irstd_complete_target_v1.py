#!/usr/bin/env python3
"""Run the preregistered IRSTD-1K complete-target-crop validation pilot.

This is an experimental, validation-only wrapper around the frozen R1
transaction/resume/selector implementation.  It changes only the training
crop policy.  The model graph, architecture initialization, six equally
weighted probability BCE terms, Adam recipe, learning-rate schedule, and
``out``-head validation evaluator remain those of R1.

The wrapper intentionally emits a distinct checkpoint schema.  Its artifacts
are not accepted by the public-test bridge until the paired validation gate is
passed and that bridge is extended in a separate, reviewed change.
"""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import json
import os
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import numpy as np
import scipy
import skimage

from experiments import evisirst_complete_target_crop as complete_crop
from experiments.evisirst_complete_target_crop import (
    EviSIRSTCompleteTargetTrainDataset,
)
from experiments.evisirst_v2_data import (
    DEFAULT_SPLIT_ROOT,
    EviSIRSTV2ValDataset,
    V2SplitContract,
)

import train_validation_selected as r1


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "irstd_performance" / "complete_target_v1"
)
DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
PAIRED_RUN_SEED = 1446202191
FORMAL_EPOCHS = 1000
FORMAL_BATCH_SIZE = 16
FORMAL_WORKERS = 0
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_VAL_INTERVAL = 1
CANONICAL_SPLIT_ROOT = PROJECT_ROOT / "splits" / "v2"
CANONICAL_IRSTD_MANIFEST_SHA256 = (
    "5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596"
)
CANONICAL_IRSTD_DATA_TREE_SHA256 = (
    "ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30"
)
CANONICAL_IRSTD_TRAIN_COUNT = 640
CANONICAL_IRSTD_VAL_COUNT = 160

TRAINING_SCHEMA = "evisirst_irstd_complete_target_training/v1"
CANDIDATE_SCHEMA = "evisirst_irstd_complete_target_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_irstd_complete_target_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_irstd_complete_target_history/v1"
SELECTION_PAYLOAD_SCHEMA = (
    "evisirst_irstd_complete_target_selection_payload/v1"
)
SOURCE_SET_SCHEMA = "evisirst_irstd_complete_target_source_set/v1"
DETERMINISM_SCHEMA = "evisirst_irstd_complete_target_determinism/v1"
EXPERIMENT_SCHEMA = "evisirst_irstd_complete_target_experiment/v1"
PROMOTION_GATE_SCHEMA = "evisirst_irstd_complete_target_promotion_gate/v1"


class CompleteTargetRunnerError(r1.ValidationSelectedTrainingError):
    """The requested run violates the complete-target-v1 contract."""


# Capture the frozen implementation before the bounded runtime patch is ever
# entered.  These references let the wrapper delegate without recursion.
_R1_PROTOCOL_SOURCE_PROVENANCE = r1._protocol_source_provenance
_R1_DETERMINISM_PROTOCOL_IDENTITY = r1._determinism_protocol_identity
_R1_BUILD_FINAL_CHECKPOINT_PAYLOAD = r1.build_final_checkpoint_payload
_R1_LOAD_RESUME_STATE = r1._load_resume_state
_R1_VALIDATE_CANDIDATE_PAYLOAD = r1._validate_candidate_payload
_R1_LOAD_SELECTED_CANDIDATE = r1._load_selected_candidate
_R1_WRITE_JSON = r1._write_json
_R1_ATOMIC_TORCH_SAVE = r1._atomic_torch_save
_R1_TRAINING_SCHEMA = r1.TRAINING_SCHEMA
_R1_VALIDATION_METRICS = r1.ValidationMetrics
_PATCH_LOCK = threading.Lock()
_RUNTIME_STATE = threading.local()


class CompleteTargetValidationMetrics(_R1_VALIDATION_METRICS):
    """R1 metrics plus non-selecting, out-head target-shape diagnostics."""

    def __init__(self, threshold: float, match_radius: float, tiny_area: int) -> None:
        super().__init__(threshold, match_radius, tiny_area)
        self._mechanism = complete_crop.MatchedTargetDiagnostics()

    def update(self, probability: Any, target: Any, loss: float) -> None:
        super().update(probability, target, loss)
        self._mechanism.update(probability, target)

    def compute(self) -> dict[str, Any]:
        metrics = dict(super().compute())
        metrics["complete_target_mechanism_diagnostics"] = (
            self._mechanism.compute()
        )
        return metrics


def promotion_gate() -> dict[str, Any]:
    """Return the immutable, preregistered paired-validation decision gate."""

    return {
        "schema": PROMOTION_GATE_SCHEMA,
        "status": "TBD",
        "paired_baseline": {
            "training_schema": _R1_TRAINING_SCHEMA,
            "dataset": DATASET,
            "target_mode": TARGET_MODE,
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": PAIRED_RUN_SEED,
            "summary_relative_path": (
                "runs/validation_selected/formal/IRSTD-1K/binary/"
                "run_seed_1446202191/summary.json"
            ),
            "selected_validation_metrics": "TBD",
        },
        "variant": {
            "training_schema": TRAINING_SCHEMA,
            "dataset": DATASET,
            "target_mode": TARGET_MODE,
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": PAIRED_RUN_SEED,
            "selected_validation_metrics": "TBD",
        },
        "primary": {
            "metric": "selected_validation_mIoU",
            "comparison": "variant_minus_paired_baseline_raw",
            "operator": ">=",
            "minimum_delta": 0.001,
            "baseline_value": "TBD",
            "variant_value": "TBD",
            "delta": "TBD",
            "passed": "TBD",
        },
        "safety": {
            "failure_rule": (
                "paired Pd decreases by more than 0.003 AND paired Fa increases"
            ),
            "pd_delta_definition": "variant_minus_paired_baseline_raw",
            "pd_failure_operator": "<",
            "pd_failure_threshold": -0.003,
            "fa_delta_definition": "variant_minus_paired_baseline_raw",
            "fa_failure_operator": ">",
            "fa_failure_threshold": 0.0,
            "baseline_pd": "TBD",
            "variant_pd": "TBD",
            "pd_delta": "TBD",
            "baseline_fa": "TBD",
            "variant_fa": "TBD",
            "fa_delta": "TBD",
            "passed": "TBD",
        },
        "mechanism": {
            "pass_rule": "at_least_one_metric_strictly_improves",
            "measurement_protocol": {
                "data_role": "same_canonical_validation_split",
                "evaluation_head": "out",
                "selection_allowed": False,
                "paired_baseline_artifact": {
                    "status": "TBD_pending_paired_baseline_completion",
                    "generation_rule": (
                        "after the paired R1 baseline completes, load its single "
                        "validation-selected checkpoint and run one fixed out-head "
                        "pass over the same canonical IRSTD-1K validation split; "
                        "the diagnostic must not participate in checkpoint selection"
                    ),
                    "input_checkpoint_relative_path": (
                        "runs/validation_selected/formal/IRSTD-1K/binary/"
                        "run_seed_1446202191/EviSIRST.pth.tar"
                    ),
                    "output_artifact_template": (
                        "runs/irstd_performance/complete_target_v1/"
                        "paired_baseline_diagnostics/run_seed_1446202191/"
                        "matched_target_diagnostics.json"
                    ),
                    "result": "TBD",
                },
                "variant_selected_record": (
                    "selected_validation_record.metrics."
                    "complete_target_mechanism_diagnostics"
                ),
            },
            "metrics": [
                {
                    "name": "matched_target_pixel_recall",
                    "selected_record_path": (
                        "selected_validation_record.metrics."
                        "complete_target_mechanism_diagnostics."
                        "matched_target_pixel_recall"
                    ),
                    "source_schema": (
                        "evisirst_matched_target_diagnostics/v1/aggregate"
                    ),
                    "comparison": "variant_minus_paired_baseline_raw",
                    "operator": ">",
                    "baseline_value": "TBD",
                    "variant_value": "TBD",
                    "delta": "TBD",
                },
                {
                    "name": "matched_component_area_ratio_closeness_to_one",
                    "source_metric": "matched_component_area_ratio",
                    "selected_record_path": (
                        "selected_validation_record.metrics."
                        "complete_target_mechanism_diagnostics."
                        "matched_component_area_ratio"
                    ),
                    "source_schema": (
                        "evisirst_matched_target_diagnostics/v1/aggregate"
                    ),
                    "transform": "-abs(raw_value-1.0)",
                    "comparison": "variant_minus_paired_baseline_transformed",
                    "operator": ">",
                    "baseline_value": "TBD",
                    "variant_value": "TBD",
                    "delta": "TBD",
                },
            ],
            "passed": "TBD",
        },
        "decision": {
            "single_seed_required_before_expansion": True,
            "expand_to_three_runtime_seeds_only_if_all_gates_pass": True,
            "three_seed_expansion_status": "blocked_pending_single_seed_gate",
            "public_test_allowed": False,
            "public_test_status": "unsupported_until_separate_gate_extension",
            "result": "TBD",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument("--target-mode", choices=(TARGET_MODE,), default=TARGET_MODE)
    parser.add_argument(
        "--architecture-seed",
        type=int,
        choices=(ARCHITECTURE_SEED,),
        default=ARCHITECTURE_SEED,
    )
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-sample-level-fallback", action="store_true")
    parser.add_argument("--smoke-max-train-samples", type=int)
    parser.add_argument("--smoke-max-val-samples", type=int)
    args = parser.parse_args(argv)

    try:
        r1.require_run_seed(args.run_seed)
    except r1.ValidationSelectedTrainingError as exc:
        parser.error(str(exc))
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for name in ("smoke_max_train_samples", "smoke_max_val_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    smoke = (
        args.smoke_max_train_samples is not None
        or args.smoke_max_val_samples is not None
    )
    if not smoke:
        if args.run_seed != PAIRED_RUN_SEED:
            parser.error(
                "formal complete-target-v1 is preregistered only for paired "
                f"--run-seed {PAIRED_RUN_SEED}"
            )
        if args.epochs != FORMAL_EPOCHS:
            parser.error(
                f"formal complete-target-v1 requires --epochs {FORMAL_EPOCHS}"
            )
        if args.warmup_epochs not in (None, FORMAL_WARMUP_EPOCHS):
            parser.error(
                "formal complete-target-v1 requires --warmup-epochs "
                f"{FORMAL_WARMUP_EPOCHS}"
            )
        args.warmup_epochs = FORMAL_WARMUP_EPOCHS
    else:
        if args.warmup_epochs is None:
            args.warmup_epochs = min(FORMAL_WARMUP_EPOCHS, args.epochs)
        if not 0 <= args.warmup_epochs <= args.epochs:
            parser.error("smoke --warmup-epochs must be in [0, epochs]")

    # These attributes deliberately match the R1 engine namespace.  They are
    # constants, not user-selectable knobs in this variant.
    args.output_root = DEFAULT_OUTPUT_ROOT
    args.batch_size = FORMAL_BATCH_SIZE
    args.workers = FORMAL_WORKERS
    args.base_lr = FORMAL_BASE_LR
    args.min_lr = FORMAL_MIN_LR
    args.val_interval = FORMAL_VAL_INTERVAL
    try:
        _require_variant_args(args)
    except CompleteTargetRunnerError as exc:
        parser.error(str(exc))
    return args


def _require_variant_args(args: argparse.Namespace) -> None:
    exact = {
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "val_interval": FORMAL_VAL_INTERVAL,
    }
    for name, expected in exact.items():
        if getattr(args, name, None) != expected:
            raise CompleteTargetRunnerError(
                f"complete-target-v1 freezes {name}={expected!r}"
            )
    if Path(args.output_root).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise CompleteTargetRunnerError(
            "complete-target-v1 output root is fixed under "
            "runs/irstd_performance/complete_target_v1"
        )
    smoke = r1._is_smoke(args)
    if not smoke and (
        args.run_seed != PAIRED_RUN_SEED
        or args.epochs != FORMAL_EPOCHS
        or args.warmup_epochs != FORMAL_WARMUP_EPOCHS
    ):
        raise CompleteTargetRunnerError(
            "formal run must use the preregistered paired seed and 1000-epoch recipe"
        )
    if not smoke and args.device != "cuda:0":
        raise CompleteTargetRunnerError(
            "formal complete-target-v1 requires logical --device cuda:0"
        )
    if not smoke:
        supplied_split_root = Path(os.path.abspath(args.split_root))
        canonical_split_root = Path(os.path.abspath(CANONICAL_SPLIT_ROOT))
        if supplied_split_root != canonical_split_root:
            raise CompleteTargetRunnerError(
                "formal run requires the repository canonical splits/v2 root"
            )
        if (
            canonical_split_root.is_symlink()
            or not canonical_split_root.is_dir()
        ):
            raise CompleteTargetRunnerError(
                "canonical splits/v2 must be a real repository directory"
            )
        manifest_path = canonical_split_root / DATASET / "manifest.json"
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or _sha256_file(manifest_path) != CANONICAL_IRSTD_MANIFEST_SHA256
        ):
            raise CompleteTargetRunnerError(
                "canonical IRSTD-1K split manifest SHA-256 differs"
            )
    if smoke and not 0 <= args.warmup_epochs <= args.epochs:
        raise CompleteTargetRunnerError("smoke warmup schedule is invalid")


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
    return hashlib.sha256(encoded).hexdigest()


def _runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    cudnn_version = torch.backends.cudnn.version()
    return {
        "requested_device": args.device,
        "physical_device_mapping": (
            "external CUDA_VISIBLE_DEVICES; formal process uses logical cuda:0"
        ),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": (
            None if cudnn_version is None else int(cudnn_version)
        ),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "skimage_version": str(skimage.__version__),
    }


def _variant_source_provenance() -> dict[str, Any]:
    baseline = _R1_PROTOCOL_SOURCE_PROVENANCE()
    baseline_files = baseline.get("files")
    if not isinstance(baseline_files, Mapping):
        raise CompleteTargetRunnerError("R1 source provenance is malformed")
    files = {
        f"r1/{name}": copy.deepcopy(entry)
        for name, entry in sorted(baseline_files.items())
    }
    for name, path in {
        "variant_runner": Path(__file__),
        "complete_target_crop": Path(complete_crop.__file__),
    }.items():
        if path.is_symlink() or not path.is_file():
            raise CompleteTargetRunnerError(f"{name} is not a regular source file")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise CompleteTargetRunnerError(
                f"{name} source is outside the repository"
            ) from exc
        files[name] = {
            "relative_path": relative,
            "sha256": _sha256_file(resolved),
        }
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


def _variant_determinism_protocol_identity() -> dict[str, Any]:
    # The captured R1 function supplies the exact evaluator and selector
    # semantics.  Only its training augmentation identity is replaced.
    identity = copy.deepcopy(_R1_DETERMINISM_PROTOCOL_IDENTITY())
    source = _variant_source_provenance()
    identity["schema"] = DETERMINISM_SCHEMA
    identity["training_data"] = {
        "source_protocol_version": r1.source_protocol.PROTOCOL_VERSION,
        "patch_size": r1.source_protocol.PATCH_SIZE,
        "target_mode": TARGET_MODE,
        "normalization_mode": "legacy",
        "augmentation_version": complete_crop.AUGMENTATION_VERSION,
        "crop_policy": complete_crop.policy_identity(),
        "crop_policy_identity_sha256": complete_crop.policy_identity_sha256(),
        "only_train_crop_policy_differs_from_R1": True,
    }
    identity["source_set_schema"] = source["schema"]
    identity["source_files"] = source["files"]
    identity["source_tree_sha256"] = source["source_tree_sha256"]
    return identity


def build_datasets(
    args: argparse.Namespace,
) -> tuple[
    EviSIRSTCompleteTargetTrainDataset,
    EviSIRSTV2ValDataset,
    dict[str, Any],
]:
    """Construct only complete-target train and unchanged V2 validation data."""

    _require_variant_args(args)
    train_dataset = EviSIRSTCompleteTargetTrainDataset(
        args.dataset,
        dataset_root=args.dataset_root,
        split_root=args.split_root,
        target_mode=args.target_mode,
        run_seed=args.run_seed,
        normalization_mode="legacy",
        verify_data_tree=True,
        formal_num_workers=args.workers,
    )
    grouping = r1.enforce_grouping_policy(
        train_dataset.contract,
        allow_sample_level_fallback=args.allow_sample_level_fallback,
    )
    val_dataset = EviSIRSTV2ValDataset(
        args.dataset,
        dataset_root=args.dataset_root,
        split_root=args.split_root,
        target_mode=args.target_mode,
        normalization_mode="legacy",
        verify_data_tree=True,
    )
    if (
        train_dataset.contract.manifest_sha256
        != val_dataset.contract.manifest_sha256
        or train_dataset.contract.data_tree_sha256
        != val_dataset.contract.data_tree_sha256
        or not train_dataset.contract.data_tree_verified
        or not val_dataset.contract.data_tree_verified
    ):
        raise CompleteTargetRunnerError("train/val data contracts differ")
    if (
        not r1._is_smoke(args)
        and (
            train_dataset.contract.manifest_sha256
            != CANONICAL_IRSTD_MANIFEST_SHA256
            or train_dataset.contract.data_tree_sha256
            != CANONICAL_IRSTD_DATA_TREE_SHA256
            or len(train_dataset) != CANONICAL_IRSTD_TRAIN_COUNT
            or len(val_dataset) != CANONICAL_IRSTD_VAL_COUNT
        )
    ):
        raise CompleteTargetRunnerError(
            "formal IRSTD-1K manifest/data-tree/count identity differs"
        )
    return train_dataset, val_dataset, grouping


def _engine_build_datasets(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    datasets = build_datasets(args)
    _RUNTIME_STATE.train_dataset = datasets[0]
    return datasets


def _active_train_dataset() -> Any:
    dataset = getattr(_RUNTIME_STATE, "train_dataset", None)
    required = (
        "crop_audit_state_dict",
        "load_crop_audit_state_dict",
        "crop_audit_summary",
    )
    if dataset is None or any(not callable(getattr(dataset, name, None)) for name in required):
        raise CompleteTargetRunnerError(
            "complete-target crop audit dataset is unavailable"
        )
    return dataset


def _json_clone(value: Any) -> Any:
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
        raise CompleteTargetRunnerError(
            "crop audit is not strict JSON-native data"
        ) from exc


def _validate_crop_audit_summary(
    summary: Any, *, expected_observations: int
) -> dict[str, Any]:
    if type(summary) is not dict:
        raise CompleteTargetRunnerError("crop audit summary must be an object")
    required_summary_keys = {
        "schema",
        "augmentation_version",
        "formal_num_workers",
        "observation_count",
        "requested_counts",
        "realized_counts",
        "fallback_count",
        "fallback_rate",
        "fallback_reason_counts",
        "cut_component_crop_count",
        "cut_component_crop_rate",
        "total_cut_component_count",
        "max_cut_component_count",
        "test_split_accessed",
    }
    if set(summary) != required_summary_keys:
        raise CompleteTargetRunnerError("crop audit summary keys differ")
    if (
        isinstance(expected_observations, bool)
        or not isinstance(expected_observations, int)
        or expected_observations < 0
        or summary.get("observation_count") != expected_observations
    ):
        raise CompleteTargetRunnerError(
            "crop audit observation count differs from committed training"
        )
    raw_state = {
        "schema": complete_crop.CROP_AUDIT_STATE_SCHEMA,
        "augmentation_version": summary.get("augmentation_version"),
        "formal_num_workers": summary.get("formal_num_workers"),
        "observation_count": summary.get("observation_count"),
        "requested_counts": summary.get("requested_counts"),
        "realized_counts": summary.get("realized_counts"),
        "fallback_count": summary.get("fallback_count"),
        "fallback_reason_counts": summary.get("fallback_reason_counts"),
        "cut_component_crop_count": summary.get("cut_component_crop_count"),
        "total_cut_component_count": summary.get("total_cut_component_count"),
        "max_cut_component_count": summary.get("max_cut_component_count"),
        "test_split_accessed": summary.get("test_split_accessed"),
    }
    verifier = complete_crop.CropAuditAccumulator(formal_num_workers=0)
    try:
        verifier.load_state_dict(_json_clone(raw_state))
    except complete_crop.CompleteTargetCropError as exc:
        raise CompleteTargetRunnerError("crop audit summary is malformed") from exc
    normalized = _json_clone(verifier.compute())
    if _json_clone(summary) != normalized:
        raise CompleteTargetRunnerError(
            "crop audit summary rates differ from its raw counts"
        )
    return normalized


def _identity_train_count(identity: Any) -> int:
    if not isinstance(identity, Mapping):
        raise CompleteTargetRunnerError("crop audit run identity is missing")
    count = identity.get("train_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CompleteTargetRunnerError("crop audit train count is malformed")
    return count


def _current_crop_audit(*, epoch: int, identity: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise CompleteTargetRunnerError("crop audit epoch is malformed")
    dataset = _active_train_dataset()
    state = _json_clone(dataset.crop_audit_state_dict())
    summary = _validate_crop_audit_summary(
        _json_clone(dataset.crop_audit_summary()),
        expected_observations=epoch * _identity_train_count(identity),
    )
    verifier = complete_crop.CropAuditAccumulator(formal_num_workers=0)
    try:
        verifier.load_state_dict(state)
    except complete_crop.CompleteTargetCropError as exc:
        raise CompleteTargetRunnerError("crop audit state is malformed") from exc
    if _json_clone(verifier.compute()) != summary:
        raise CompleteTargetRunnerError("crop audit state and summary differ")
    return state, summary


def _validate_crop_audit_history(
    history: Any,
    *,
    completed_epoch: int,
    identity: Any,
) -> list[dict[str, Any]]:
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise CompleteTargetRunnerError("crop audit history is incomplete")
    train_count = _identity_train_count(identity)
    normalized: list[dict[str, Any]] = []
    previous_summary = complete_crop.CropAuditAccumulator(
        formal_num_workers=0
    ).compute()
    for expected_epoch, record in enumerate(history, start=1):
        if (
            type(record) is not dict
            or set(record) != {"epoch", "summary"}
            or record.get("epoch") != expected_epoch
        ):
            raise CompleteTargetRunnerError("crop audit history order differs")
        current_summary = _validate_crop_audit_summary(
            record.get("summary"),
            expected_observations=expected_epoch * train_count,
        )
        for field in ("requested_counts", "realized_counts"):
            deltas = [
                current_summary[field][name] - previous_summary[field][name]
                for name in current_summary[field]
            ]
            if any(delta < 0 for delta in deltas) or sum(deltas) != train_count:
                raise CompleteTargetRunnerError(
                    f"crop audit {field} per-epoch increment differs"
                )
        fallback_delta = (
            current_summary["fallback_count"]
            - previous_summary["fallback_count"]
        )
        reason_names = set(current_summary["fallback_reason_counts"]) | set(
            previous_summary["fallback_reason_counts"]
        )
        reason_deltas = [
            current_summary["fallback_reason_counts"].get(name, 0)
            - previous_summary["fallback_reason_counts"].get(name, 0)
            for name in reason_names
        ]
        cut_crop_delta = (
            current_summary["cut_component_crop_count"]
            - previous_summary["cut_component_crop_count"]
        )
        total_cut_delta = (
            current_summary["total_cut_component_count"]
            - previous_summary["total_cut_component_count"]
        )
        if (
            fallback_delta < 0
            or any(delta < 0 for delta in reason_deltas)
            or sum(reason_deltas) != fallback_delta
            or cut_crop_delta < 0
            or cut_crop_delta > train_count
            or total_cut_delta < cut_crop_delta
            or current_summary["max_cut_component_count"]
            < previous_summary["max_cut_component_count"]
        ):
            raise CompleteTargetRunnerError(
                "crop audit cumulative history is not monotone"
            )
        normalized.append({"epoch": expected_epoch, "summary": current_summary})
        previous_summary = current_summary
    return normalized


def _prospective_crop_audit_history(
    *, epoch: int, identity: Any, summary: Mapping[str, Any]
) -> list[dict[str, Any]]:
    history = _json_clone(getattr(_RUNTIME_STATE, "crop_audit_history", []))
    train_count = _identity_train_count(identity)
    if history:
        history = _validate_crop_audit_history(
            history,
            completed_epoch=len(history),
            identity=identity,
        )
    normalized_summary = _validate_crop_audit_summary(
        dict(summary), expected_observations=epoch * train_count
    )
    if len(history) == epoch:
        if history[-1]["summary"] != normalized_summary:
            raise CompleteTargetRunnerError(
                "crop audit current summary differs from committed history"
            )
        return history
    if len(history) != epoch - 1:
        raise CompleteTargetRunnerError(
            "crop audit history does not precede the current epoch"
        )
    return [*history, {"epoch": epoch, "summary": normalized_summary}]


def _variant_load_resume_state(**kwargs: Any) -> Any:
    result = _R1_LOAD_RESUME_STATE(**kwargs)
    path = kwargs.get("path")
    identity = kwargs.get("identity")
    if not isinstance(path, Path):
        raise CompleteTargetRunnerError("crop audit resume path is malformed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise CompleteTargetRunnerError("crop audit resume payload is malformed")
    if payload.get("crop_audit_commit_status") != (
        "committed_by_this_atomic_latest"
    ):
        raise CompleteTargetRunnerError(
            "resume crop audit latest commit status differs"
        )
    start_epoch = result[0]
    completed_epoch = start_epoch - 1
    if isinstance(completed_epoch, bool) or completed_epoch < 1:
        raise CompleteTargetRunnerError("crop audit resume epoch is malformed")
    dataset = _active_train_dataset()
    audit_state = payload.get("crop_audit_state")
    try:
        dataset.load_crop_audit_state_dict(_json_clone(audit_state))
    except (TypeError, complete_crop.CompleteTargetCropError) as exc:
        raise CompleteTargetRunnerError(
            "crop audit resume state failed strict validation"
        ) from exc
    _state, current_summary = _current_crop_audit(
        epoch=completed_epoch, identity=identity
    )
    saved_summary = _validate_crop_audit_summary(
        payload.get("crop_audit_summary"),
        expected_observations=(
            completed_epoch * _identity_train_count(identity)
        ),
    )
    if current_summary != saved_summary:
        raise CompleteTargetRunnerError(
            "crop audit restored state differs from saved summary"
        )
    history = _validate_crop_audit_history(
        payload.get("crop_audit_history"),
        completed_epoch=completed_epoch,
        identity=identity,
    )
    if history[-1]["summary"] != current_summary:
        raise CompleteTargetRunnerError(
            "crop audit restored state differs from history"
        )
    _RUNTIME_STATE.crop_audit_history = history
    _RUNTIME_STATE.run_identity = dict(identity)
    return result


def _validate_candidate_crop_audit(
    payload: Any, *, epoch: int, identity: Any
) -> None:
    if not isinstance(payload, Mapping):
        raise CompleteTargetRunnerError("candidate crop audit payload is malformed")
    verifier = complete_crop.CropAuditAccumulator(formal_num_workers=0)
    try:
        verifier.load_state_dict(_json_clone(payload.get("crop_audit_state")))
    except (TypeError, complete_crop.CompleteTargetCropError) as exc:
        raise CompleteTargetRunnerError(
            "candidate crop audit state failed strict validation"
        ) from exc
    summary = _validate_crop_audit_summary(
        payload.get("crop_audit_summary"),
        expected_observations=epoch * _identity_train_count(identity),
    )
    if _json_clone(verifier.compute()) != summary:
        raise CompleteTargetRunnerError(
            "candidate crop audit state and summary differ"
        )
    history = _validate_crop_audit_history(
        payload.get("crop_audit_history"),
        completed_epoch=epoch,
        identity=identity,
    )
    if history[-1]["summary"] != summary:
        raise CompleteTargetRunnerError(
            "candidate crop audit history and summary differ"
        )
    if payload.get("crop_audit_commit_status") != (
        "candidate_pre_latest_uncommitted"
    ):
        raise CompleteTargetRunnerError(
            "candidate crop audit commit status differs"
        )


def _variant_validate_candidate_payload(**kwargs: Any) -> Any:
    result = _R1_VALIDATE_CANDIDATE_PAYLOAD(**kwargs)
    path = kwargs.get("path")
    identity = kwargs.get("identity")
    if not isinstance(path, Path):
        raise CompleteTargetRunnerError("candidate crop audit path is malformed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_candidate_crop_audit(
        payload, epoch=int(result[0]), identity=identity
    )
    return result


def _variant_load_selected_candidate(**kwargs: Any) -> Any:
    state = _R1_LOAD_SELECTED_CANDIDATE(**kwargs)
    selection_payload = kwargs.get("selection_payload")
    candidate_dir = kwargs.get("candidate_dir")
    identity = kwargs.get("identity")
    if not isinstance(selection_payload, Mapping) or not isinstance(
        candidate_dir, Path
    ):
        raise CompleteTargetRunnerError(
            "selected candidate crop audit arguments are malformed"
        )
    epoch = selection_payload.get("selected_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise CompleteTargetRunnerError(
            "selected candidate crop audit epoch is malformed"
        )
    path = r1._candidate_path(candidate_dir, epoch)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_candidate_crop_audit(payload, epoch=epoch, identity=identity)
    selected_artifact = selection_payload.get("selected_candidate")
    candidate_sha256 = _sha256_file(path)
    if (
        not isinstance(selected_artifact, Mapping)
        or selected_artifact.get("file_sha256") != candidate_sha256
    ):
        raise CompleteTargetRunnerError(
            "selected candidate crop audit SHA-256 differs"
        )
    _RUNTIME_STATE.selected_candidate_evidence = {
        "through_epoch": epoch,
        "candidate_sha256": candidate_sha256,
        "state": _json_clone(payload["crop_audit_state"]),
        "summary": _json_clone(payload["crop_audit_summary"]),
        "history": _json_clone(payload["crop_audit_history"]),
    }
    return state


def _selected_candidate_crop_audit(identity: Any) -> dict[str, Any]:
    evidence = getattr(_RUNTIME_STATE, "selected_candidate_evidence", None)
    if type(evidence) is not dict or set(evidence) != {
        "through_epoch",
        "candidate_sha256",
        "state",
        "summary",
        "history",
    }:
        raise CompleteTargetRunnerError(
            "selected checkpoint crop audit evidence is unavailable"
        )
    epoch = evidence["through_epoch"]
    candidate_sha256 = evidence["candidate_sha256"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or not isinstance(candidate_sha256, str)
        or len(candidate_sha256) != 64
        or any(character not in "0123456789abcdef" for character in candidate_sha256)
    ):
        raise CompleteTargetRunnerError(
            "selected checkpoint crop audit identity is malformed"
        )
    verifier = complete_crop.CropAuditAccumulator(formal_num_workers=0)
    try:
        verifier.load_state_dict(_json_clone(evidence["state"]))
    except complete_crop.CompleteTargetCropError as exc:
        raise CompleteTargetRunnerError(
            "selected checkpoint crop audit state is malformed"
        ) from exc
    summary = _validate_crop_audit_summary(
        evidence["summary"],
        expected_observations=epoch * _identity_train_count(identity),
    )
    history = _validate_crop_audit_history(
        evidence["history"],
        completed_epoch=epoch,
        identity=identity,
    )
    if _json_clone(verifier.compute()) != summary or history[-1]["summary"] != summary:
        raise CompleteTargetRunnerError(
            "selected checkpoint crop audit evidence differs"
        )
    return {
        "through_epoch": epoch,
        "candidate_sha256": candidate_sha256,
        "state": _json_clone(evidence["state"]),
        "summary": summary,
        "history": history,
    }


def _variant_run_identity(
    args: argparse.Namespace,
    contract: V2SplitContract,
    grouping: Mapping[str, Any],
    *,
    train_count: int,
    val_count: int,
    smoke: bool,
) -> dict[str, Any]:
    _require_variant_args(args)
    if not smoke and (
        contract.manifest_sha256 != CANONICAL_IRSTD_MANIFEST_SHA256
        or contract.data_tree_sha256 != CANONICAL_IRSTD_DATA_TREE_SHA256
        or train_count != CANONICAL_IRSTD_TRAIN_COUNT
        or val_count != CANONICAL_IRSTD_VAL_COUNT
    ):
        raise CompleteTargetRunnerError(
            "formal run identity differs from the canonical IRSTD-1K split"
        )
    identity = {
        "schema": TRAINING_SCHEMA + "/run_identity",
        "experiment": {
            "schema": EXPERIMENT_SCHEMA,
            "name": "IRSTD-1K complete-target crop v1",
            "status": "experimental_validation_only",
            "only_train_crop_policy_differs_from_R1": True,
            "public_test_supported": False,
            "public_test_gate_status": "unsupported_until_separate_gate_extension",
        },
        "model": "EviSIRST",
        "dataset": args.dataset,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": args.run_seed,
        "target_mode": args.target_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "val_interval": args.val_interval,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "optimizer_hyperparameters": _json_clone(
            r1.R1_ADAM_GROUP_HYPERPARAMETERS
        ),
        "loss": "sum_of_six_BCELoss_mean_terms",
        "deep_supervision_probability_heads": 6,
        "deep_supervision_weights": [1.0] * 6,
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "evaluation_head": "out",
        "evaluation_supplementary_diagnostics": {
            "selection_allowed": False,
            "schema": "evisirst_matched_target_diagnostics/v1/aggregate",
            "record_path": "metrics.complete_target_mechanism_diagnostics",
        },
        "execution_contract": {
            "single_process_only": True,
            "python_threads_running_variant": 1,
            "data_loader_workers": 0,
            "nonblocking_process_lock": ".complete_target_v1.lock",
        },
        "runtime_identity": _runtime_identity(args),
        "selection_rule": r1.selection.INDEPENDENT_RULE_VERSION,
        "determinism_protocol": _variant_determinism_protocol_identity(),
        "crop_policy": complete_crop.policy_identity(),
        "crop_policy_identity_sha256": complete_crop.policy_identity_sha256(),
        "manifest_sha256": contract.manifest_sha256,
        "split_seed": contract.manifest["seeds"]["split_seed"],
        "data_tree_sha256": contract.data_tree_sha256,
        "canonical_split_contract": {
            "split_root_relative_path": "splits/v2",
            "manifest_sha256": CANONICAL_IRSTD_MANIFEST_SHA256,
            "data_tree_sha256": CANONICAL_IRSTD_DATA_TREE_SHA256,
            "train_count": CANONICAL_IRSTD_TRAIN_COUNT,
            "val_count": CANONICAL_IRSTD_VAL_COUNT,
        },
        "grouping_policy": dict(grouping),
        "train_count": train_count,
        "val_count": val_count,
        "smoke": smoke,
        "smoke_max_train_samples": args.smoke_max_train_samples,
        "smoke_max_val_samples": args.smoke_max_val_samples,
        "promotion_gate": promotion_gate(),
        "test_split_accessed": False,
    }
    identity = _json_clone(identity)
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _variant_final_checkpoint_payload(**kwargs: Any) -> dict[str, Any]:
    payload = _R1_BUILD_FINAL_CHECKPOINT_PAYLOAD(**kwargs)
    payload.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_role": "experimental_validation_selected",
            "experiment_schema": EXPERIMENT_SCHEMA,
            "experiment_status": "experimental_validation_only",
            "only_train_crop_policy_differs_from_R1": True,
            "crop_policy": complete_crop.policy_identity(),
            "crop_policy_identity_sha256": complete_crop.policy_identity_sha256(),
            "promotion_gate": promotion_gate(),
            "public_test_supported": False,
            "public_test_gate_status": "unsupported_until_separate_gate_extension",
            "test_split_accessed": False,
        }
    )
    return payload


def _variant_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = dict(payload)
    if output.get("schema") == HISTORY_SCHEMA:
        training_history = output.get("training_history")
        if not isinstance(training_history, list) or not training_history:
            raise CompleteTargetRunnerError(
                "crop audit history requires committed training records"
            )
        epoch = training_history[-1].get("epoch")
        state, summary = _current_crop_audit(
            epoch=epoch, identity=output.get("run_identity")
        )
        output["crop_audit_state"] = state
        output["crop_audit_summary"] = summary
        output["crop_audit_history"] = _prospective_crop_audit_history(
            epoch=epoch,
            identity=output.get("run_identity"),
            summary=summary,
        )
        output["crop_audit_commit_status"] = "pre_latest_transaction_view"
    if output.get("schema") == TRAINING_SCHEMA + "/summary":
        training_history = output.get("training_history")
        validation_history = output.get("validation_history")
        if not isinstance(training_history, list) or not training_history:
            raise CompleteTargetRunnerError(
                "final crop audit requires training history"
            )
        selected_epoch = output.get("selected_epoch")
        selection_payload = output.get("selection")
        if (
            isinstance(selected_epoch, bool)
            or not isinstance(selected_epoch, int)
            or selected_epoch < 1
            or not isinstance(validation_history, list)
            or not isinstance(selection_payload, Mapping)
            or selection_payload.get("selected_epoch") != selected_epoch
        ):
            raise CompleteTargetRunnerError(
                "final selected validation record identity is malformed"
            )
        selected_records = [
            record
            for record in validation_history
            if isinstance(record, Mapping) and record.get("epoch") == selected_epoch
        ]
        if len(selected_records) != 1:
            raise CompleteTargetRunnerError(
                "selected epoch does not identify exactly one validation record"
            )
        output["selected_validation_record"] = _json_clone(selected_records[0])
        output["selected_validation_record_sha256"] = _canonical_sha256(
            output["selected_validation_record"]
        )
        epoch = training_history[-1].get("epoch")
        run_identity = getattr(_RUNTIME_STATE, "run_identity", None)
        if not isinstance(run_identity, Mapping):
            raise CompleteTargetRunnerError("final summary run identity is missing")
        normalized_run_identity = _json_clone(dict(run_identity))
        identity_sha256 = normalized_run_identity.get("identity_sha256")
        unhashed_identity = dict(normalized_run_identity)
        unhashed_identity.pop("identity_sha256", None)
        if (
            not isinstance(identity_sha256, str)
            or identity_sha256 != _canonical_sha256(unhashed_identity)
        ):
            raise CompleteTargetRunnerError(
                "final summary run identity SHA-256 differs"
            )
        output["run_identity"] = normalized_run_identity
        output["training_identity_sha256"] = identity_sha256
        state, summary = _current_crop_audit(
            epoch=epoch,
            identity=normalized_run_identity,
        )
        audit_history = _validate_crop_audit_history(
            getattr(_RUNTIME_STATE, "crop_audit_history", None),
            completed_epoch=epoch,
            identity=normalized_run_identity,
        )
        output["search_run_crop_audit_state"] = state
        output["search_run_crop_audit_summary"] = summary
        output["search_run_crop_audit_history"] = audit_history
        output["search_run_crop_audit_through_epoch"] = epoch
        selected_audit = _selected_candidate_crop_audit(normalized_run_identity)
        if selected_audit["through_epoch"] != selected_epoch:
            raise CompleteTargetRunnerError(
                "selected checkpoint audit epoch differs from selected record"
            )
        output["selected_checkpoint_crop_audit_state"] = selected_audit["state"]
        output["selected_checkpoint_crop_audit_summary"] = selected_audit["summary"]
        output["selected_checkpoint_crop_audit_history"] = selected_audit["history"]
        output["selected_checkpoint_crop_audit_through_epoch"] = selected_epoch
        output["selected_candidate_sha256"] = selected_audit["candidate_sha256"]
        selected_artifact = selection_payload.get("selected_candidate")
        if (
            not isinstance(selected_artifact, Mapping)
            or selected_artifact.get("file_sha256")
            != output["selected_candidate_sha256"]
        ):
            raise CompleteTargetRunnerError(
                "summary selected candidate SHA-256 differs"
            )
        checkpoint_relative = output.get("checkpoint")
        if (
            not isinstance(checkpoint_relative, str)
            or Path(checkpoint_relative).is_absolute()
            or ".." in Path(checkpoint_relative).parts
        ):
            raise CompleteTargetRunnerError(
                "final checkpoint summary path is malformed"
            )
        checkpoint_path = PROJECT_ROOT / checkpoint_relative
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise CompleteTargetRunnerError(
                "final checkpoint is unavailable for summary hashing"
            )
        output["final_checkpoint_sha256"] = _sha256_file(checkpoint_path)
        output["crop_audit_commit_status"] = "complete"
        output.update(
            {
                "checkpoint_role": "experimental_validation_selected",
                "experiment_schema": EXPERIMENT_SCHEMA,
                "experiment_status": "experimental_validation_only",
                "only_train_crop_policy_differs_from_R1": True,
                "crop_policy": complete_crop.policy_identity(),
                "crop_policy_identity_sha256": (
                    complete_crop.policy_identity_sha256()
                ),
                "promotion_gate": promotion_gate(),
                "public_test_supported": False,
                "public_test_gate_status": (
                    "unsupported_until_separate_gate_extension"
                ),
                "test_split_accessed": False,
            }
        )
    _R1_WRITE_JSON(path, output)


def _variant_atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    output = dict(payload)
    if output.get("schema") in {
        TRAINING_SCHEMA,
        CANDIDATE_SCHEMA,
        CHECKPOINT_SCHEMA,
    }:
        output.update(
            {
                "experiment_schema": EXPERIMENT_SCHEMA,
                "experiment_status": "experimental_validation_only",
                "public_test_supported": False,
                "public_test_gate_status": (
                    "unsupported_until_separate_gate_extension"
                ),
                "test_split_accessed": False,
            }
        )
    if output.get("schema") == TRAINING_SCHEMA:
        epoch = output.get("epoch")
        identity = output.get("run_identity")
        state, summary = _current_crop_audit(epoch=epoch, identity=identity)
        prospective = _prospective_crop_audit_history(
            epoch=epoch, identity=identity, summary=summary
        )
        output["crop_audit_state"] = state
        output["crop_audit_summary"] = summary
        output["crop_audit_history"] = prospective
        output["crop_audit_commit_status"] = "committed_by_this_atomic_latest"
        _R1_ATOMIC_TORCH_SAVE(path, output)
        _RUNTIME_STATE.crop_audit_history = prospective
        _RUNTIME_STATE.run_identity = dict(identity)
        return
    if output.get("schema") == CANDIDATE_SCHEMA:
        epoch = output.get("epoch")
        identity = output.get("run_identity")
        state, summary = _current_crop_audit(epoch=epoch, identity=identity)
        output["crop_audit_state"] = state
        output["crop_audit_summary"] = summary
        output["crop_audit_history"] = _prospective_crop_audit_history(
            epoch=epoch, identity=identity, summary=summary
        )
        output["crop_audit_commit_status"] = "candidate_pre_latest_uncommitted"
    if output.get("schema") == CHECKPOINT_SCHEMA:
        identity = output.get("training")
        epoch = identity.get("epochs") if isinstance(identity, Mapping) else None
        state, summary = _current_crop_audit(epoch=epoch, identity=identity)
        output["search_run_crop_audit_state"] = state
        output["search_run_crop_audit_summary"] = summary
        output["search_run_crop_audit_history"] = _validate_crop_audit_history(
            getattr(_RUNTIME_STATE, "crop_audit_history", None),
            completed_epoch=epoch,
            identity=identity,
        )
        output["search_run_crop_audit_through_epoch"] = epoch
        selected_audit = _selected_candidate_crop_audit(identity)
        output["selected_checkpoint_crop_audit_state"] = selected_audit["state"]
        output["selected_checkpoint_crop_audit_summary"] = selected_audit["summary"]
        output["selected_checkpoint_crop_audit_history"] = selected_audit["history"]
        output["selected_checkpoint_crop_audit_through_epoch"] = selected_audit[
            "through_epoch"
        ]
        output["selected_candidate_sha256"] = selected_audit["candidate_sha256"]
        output["crop_audit_commit_status"] = "complete"
    _R1_ATOMIC_TORCH_SAVE(path, output)


@contextmanager
def _variant_runtime() -> Iterator[None]:
    """Temporarily and exception-safely route the R1 engine to this variant."""

    overrides = {
        "TRAINING_SCHEMA": TRAINING_SCHEMA,
        "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "HISTORY_SCHEMA": HISTORY_SCHEMA,
        "SELECTION_PAYLOAD_SCHEMA": SELECTION_PAYLOAD_SCHEMA,
        "ValidationMetrics": CompleteTargetValidationMetrics,
        "build_datasets": _engine_build_datasets,
        "_protocol_source_provenance": _variant_source_provenance,
        "_determinism_protocol_identity": _variant_determinism_protocol_identity,
        "_run_identity": _variant_run_identity,
        "build_final_checkpoint_payload": _variant_final_checkpoint_payload,
        "_load_resume_state": _variant_load_resume_state,
        "_validate_candidate_payload": _variant_validate_candidate_payload,
        "_load_selected_candidate": _variant_load_selected_candidate,
        "_write_json": _variant_write_json,
        "_atomic_torch_save": _variant_atomic_torch_save,
    }
    if not _PATCH_LOCK.acquire(blocking=False):
        raise CompleteTargetRunnerError(
            "complete-target runtime is already active in this process"
        )
    previous = {name: getattr(r1, name) for name in overrides}
    _RUNTIME_STATE.crop_audit_history = []
    _RUNTIME_STATE.train_dataset = None
    _RUNTIME_STATE.run_identity = None
    _RUNTIME_STATE.selected_candidate_evidence = None
    try:
        for name, value in overrides.items():
            setattr(r1, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(r1, name, value)
        for name in (
            "crop_audit_history",
            "train_dataset",
            "run_identity",
            "selected_candidate_evidence",
        ):
            if hasattr(_RUNTIME_STATE, name):
                delattr(_RUNTIME_STATE, name)
        _PATCH_LOCK.release()


@contextmanager
def _run_process_lock(args: argparse.Namespace) -> Iterator[Path]:
    paths = resolve_run_paths(args)
    run_dir = paths["run_dir"]
    if not isinstance(run_dir, Path):
        raise CompleteTargetRunnerError("run directory is malformed")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".complete_target_v1.lock"
    if lock_path.is_symlink():
        raise CompleteTargetRunnerError("run lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise CompleteTargetRunnerError(
                    f"complete-target run is already locked: {run_dir}"
                ) from exc
            raise
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path | bool]:
    _require_variant_args(args)
    return r1.resolve_run_paths(args)


def run(args: argparse.Namespace) -> Path:
    """Execute this variant through R1's transactional validation engine."""

    _require_variant_args(args)
    with _run_process_lock(args), _variant_runtime():
        checkpoint = r1.run(args)
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    checkpoint = run(parse_args(argv))
    print(checkpoint)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CompleteTargetRunnerError",
    "DATASET",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAL_EPOCHS",
    "HISTORY_SCHEMA",
    "PAIRED_RUN_SEED",
    "PROMOTION_GATE_SCHEMA",
    "SELECTION_PAYLOAD_SCHEMA",
    "TARGET_MODE",
    "TRAINING_SCHEMA",
    "build_datasets",
    "parse_args",
    "promotion_gate",
    "resolve_run_paths",
    "run",
]
