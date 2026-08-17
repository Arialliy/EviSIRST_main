#!/usr/bin/env python3
"""Prepare the preregistered IRSTD-1K weighted-deep-supervision fallback.

This validation-only runner reuses the frozen R1 data, model, Adam,
learning-rate, validation, and checkpoint-selection implementations.  Its only
training change is the positional weighting of the six probability-map BCE
terms returned as ``(gt5, gt4, gt3, gt2, d0, out)``.

The formal path is intentionally locked until the completed paired R1 and
complete-target-v1 artifacts prove that the complete-target promotion gate
failed.  Smoke runs are isolated and do not satisfy or bypass that gate.
Public-test loading is neither imported nor supported.
"""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

import run_irstd_complete_target_promotion_gate as canonical_gate
import train_validation_selected as r1
from experiments.evisirst_v2_data import DEFAULT_SPLIT_ROOT, V2SplitContract


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "irstd_performance" / "weighted_ds_v1"
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

DEEP_SUPERVISION_HEAD_ORDER = ("gt5", "gt4", "gt3", "gt2", "d0", "out")
DEEP_SUPERVISION_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
DEEP_SUPERVISION_WEIGHT_SUM = 6.0

TRAINING_SCHEMA = "evisirst_irstd_weighted_ds_training/v1"
CANDIDATE_SCHEMA = "evisirst_irstd_weighted_ds_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_irstd_weighted_ds_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_irstd_weighted_ds_history/v1"
SELECTION_PAYLOAD_SCHEMA = "evisirst_irstd_weighted_ds_selection_payload/v1"
SOURCE_SET_SCHEMA = "evisirst_irstd_weighted_ds_source_set/v1"
DETERMINISM_SCHEMA = "evisirst_irstd_weighted_ds_determinism/v1"
EXPERIMENT_SCHEMA = "evisirst_irstd_weighted_ds_experiment/v1"
PROMOTION_GATE_SCHEMA = "evisirst_irstd_weighted_ds_promotion_gate/v1"
PREDECESSOR_GATE_EVIDENCE_SCHEMA = (
    "evisirst_irstd_canonical_complete_target_gate_evidence/v1"
)

R1_SUMMARY_RELATIVE_PATH = canonical_gate.BASELINE_SUMMARY_RELATIVE_PATH
CANONICAL_GATE_RESULT_RELATIVE_PATH = canonical_gate.OUTPUT_RELATIVE_PATH
CANONICAL_GATE_RESULT_SCHEMA = canonical_gate.RESULT_SCHEMA


class WeightedDSRunnerError(r1.ValidationSelectedTrainingError):
    """The requested run violates the weighted-deep-supervision-v1 contract."""


# Capture all R1 functions before the bounded, process-local runtime patch.
_R1_BUILD_DATASETS = r1.build_datasets
_R1_PROTOCOL_SOURCE_PROVENANCE = r1._protocol_source_provenance
_R1_DETERMINISM_PROTOCOL_IDENTITY = r1._determinism_protocol_identity
_R1_BUILD_FINAL_CHECKPOINT_PAYLOAD = r1.build_final_checkpoint_payload
_R1_WRITE_JSON = r1._write_json
_R1_ATOMIC_TORCH_SAVE = r1._atomic_torch_save
_R1_TRAINING_SCHEMA = r1.TRAINING_SCHEMA
_PATCH_LOCK = threading.Lock()
_RUNTIME_STATE = threading.local()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WeightedDSRunnerError(
            "metadata must be strict JSON without NaN/Infinity"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def loss_policy_identity() -> dict[str, Any]:
    """Return the frozen positional loss contract."""

    if len(DEEP_SUPERVISION_HEAD_ORDER) != 6 or len(DEEP_SUPERVISION_WEIGHTS) != 6:
        raise WeightedDSRunnerError("weighted-DS requires exactly six heads")
    if not math.isclose(
        math.fsum(DEEP_SUPERVISION_WEIGHTS),
        DEEP_SUPERVISION_WEIGHT_SUM,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise WeightedDSRunnerError("weighted-DS weights must sum exactly to 6")
    return {
        "schema": "evisirst_weighted_deep_supervision_loss/v1",
        "output_order": list(DEEP_SUPERVISION_HEAD_ORDER),
        "weights": list(DEEP_SUPERVISION_WEIGHTS),
        "weight_sum": DEEP_SUPERVISION_WEIGHT_SUM,
        "criterion_per_head": "BCELoss(reduction=mean)",
        "formula": "sum_i(weight_i * BCE(probability_head_i, target))",
        "scale_preservation": (
            "weights_sum_to_six_matching_R1_six_unit_weight_BCE_terms"
        ),
    }


def weighted_deep_supervision_loss(
    output: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute the frozen six-head weighted BCE sum without renormalization."""

    policy = loss_policy_identity()
    if not isinstance(output, (tuple, list)) or len(output) != 6:
        raise RuntimeError("training-mode EviSIRST must return six probability maps")
    if not isinstance(target, torch.Tensor):
        raise TypeError("weighted-DS target must be a tensor")
    weighted_terms: list[torch.Tensor] = []
    for index, (probability, weight) in enumerate(
        zip(output, policy["weights"], strict=True)
    ):
        if not isinstance(probability, torch.Tensor) or probability.shape != target.shape:
            raise RuntimeError(
                f"weighted-DS output {index} ({DEEP_SUPERVISION_HEAD_ORDER[index]}) "
                "has the wrong shape"
            )
        if not torch.isfinite(probability).all():
            raise FloatingPointError(
                f"weighted-DS output {index} is non-finite"
            )
        if bool((probability < 0).any()) or bool((probability > 1).any()):
            raise RuntimeError(
                f"weighted-DS output {index} is not a probability map"
            )
        term = criterion(probability.float(), target.float())
        if not isinstance(term, torch.Tensor) or term.numel() != 1:
            raise RuntimeError("weighted-DS criterion must return one scalar per head")
        weighted_terms.append(term * float(weight))
    return sum(weighted_terms[1:], weighted_terms[0])


def promotion_gate() -> dict[str, Any]:
    """Return the immutable paired-validation decision gate for this variant."""

    return {
        "schema": PROMOTION_GATE_SCHEMA,
        "status": "TBD",
        "paired_baseline": {
            "training_schema": _R1_TRAINING_SCHEMA,
            "dataset": DATASET,
            "target_mode": TARGET_MODE,
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": PAIRED_RUN_SEED,
            "summary_relative_path": R1_SUMMARY_RELATIVE_PATH,
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
        "decision": {
            "single_seed_required_before_expansion": True,
            "expand_to_three_runtime_seeds_only_if_primary_and_safety_pass": True,
            "three_seed_expansion_status": "blocked_pending_single_seed_gate",
            "public_test_allowed": False,
            "public_test_status": "unsupported_until_separate_gate_extension",
            "result": "TBD",
        },
    }


def _require_canonical_failure_payload(payload: Any) -> dict[str, Any]:
    """Accept only the canonical evaluator's verified FAIL decision.

    No metric or threshold is recomputed here.  The sole decision authority is
    ``canonical_gate.validate_existing_result()``, which re-evaluates every
    frozen input and compares the complete canonical payload byte-for-byte.
    """

    if not isinstance(payload, Mapping):
        raise WeightedDSRunnerError("canonical promotion-gate payload is malformed")
    fixed = {
        "schema": CANONICAL_GATE_RESULT_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": "val",
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": PAIRED_RUN_SEED,
        "epochs": FORMAL_EPOCHS,
        "test_split_accessed": False,
        "public_test_allowed": False,
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise WeightedDSRunnerError(
                f"canonical promotion-gate payload differs at {key}"
            )
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or any(
        inputs.get(key) is not True
        for key in (
            "all_artifact_hashes_verified",
            "all_source_hashes_currently_verified",
            "canonical_train_val_split_verified",
            "both_selections_recomputed",
        )
    ):
        raise WeightedDSRunnerError(
            "canonical promotion-gate verification evidence is incomplete"
        )
    gate_source = inputs.get("gate_evaluator_source")
    source_path = Path(canonical_gate.__file__).resolve(strict=True)
    if (
        not isinstance(gate_source, Mapping)
        or gate_source.get("relative_path")
        != canonical_gate.GATE_SOURCE_RELATIVE_PATH
        or gate_source.get("sha256") != _sha256_file(source_path)
    ):
        raise WeightedDSRunnerError(
            "canonical promotion-gate source binding differs"
        )
    decision = payload.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("result") != "FAIL"
        or decision.get("overall_passed") is not False
        or decision.get("three_runtime_seed_validation_expansion_allowed")
        is not False
        or decision.get("three_runtime_seed_validation_expansion_status")
        != "blocked"
        or decision.get("public_test_allowed") is not False
    ):
        raise WeightedDSRunnerError(
            "canonical complete-target gate did not authorize weighted-DS fallback"
        )
    return _json_clone(payload)


def validate_canonical_predecessor_failure() -> dict[str, Any]:
    """Validate and bind the immutable canonical predecessor result."""

    if Path(canonical_gate.PROJECT_ROOT).resolve() != PROJECT_ROOT.resolve():
        raise WeightedDSRunnerError(
            "canonical promotion-gate evaluator belongs to another repository"
        )
    try:
        payload = canonical_gate.validate_existing_result()
    except (FileNotFoundError, OSError, canonical_gate.PromotionGateError) as exc:
        raise WeightedDSRunnerError(
            "canonical complete-target promotion result is missing or invalid"
        ) from exc
    canonical_payload = _require_canonical_failure_payload(payload)
    result_path = PROJECT_ROOT / CANONICAL_GATE_RESULT_RELATIVE_PATH
    if result_path.is_symlink() or not result_path.is_file():
        raise WeightedDSRunnerError(
            "canonical complete-target promotion result is not a regular file"
        )
    result_path = result_path.resolve(strict=True)
    try:
        result_path.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise WeightedDSRunnerError(
            "canonical complete-target promotion result escaped the repository"
        ) from exc
    evidence = {
        "schema": PREDECESSOR_GATE_EVIDENCE_SCHEMA,
        "status": "complete",
        "authority": "run_irstd_complete_target_promotion_gate.validate_existing_result",
        "canonical_result_artifact": {
            "relative_path": CANONICAL_GATE_RESULT_RELATIVE_PATH,
            "sha256": _sha256_file(result_path),
        },
        "canonical_payload_schema": CANONICAL_GATE_RESULT_SCHEMA,
        "canonical_payload_sha256": _canonical_sha256(canonical_payload),
        "canonical_payload": canonical_payload,
        "formal_weighted_ds_allowed": True,
        "public_test_accessed": False,
    }
    return _json_clone(evidence)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    smoke = args.smoke_max_train_samples is not None or args.smoke_max_val_samples is not None
    if not smoke:
        if args.run_seed != PAIRED_RUN_SEED:
            parser.error(
                f"formal weighted-DS-v1 requires paired --run-seed {PAIRED_RUN_SEED}"
            )
        if args.epochs != FORMAL_EPOCHS:
            parser.error(f"formal weighted-DS-v1 requires --epochs {FORMAL_EPOCHS}")
        if args.warmup_epochs not in (None, FORMAL_WARMUP_EPOCHS):
            parser.error(
                f"formal weighted-DS-v1 requires --warmup-epochs {FORMAL_WARMUP_EPOCHS}"
            )
        args.warmup_epochs = FORMAL_WARMUP_EPOCHS
    else:
        if args.warmup_epochs is None:
            args.warmup_epochs = min(FORMAL_WARMUP_EPOCHS, args.epochs)
        if not 0 <= args.warmup_epochs <= args.epochs:
            parser.error("smoke --warmup-epochs must be in [0, epochs]")
    args.output_root = DEFAULT_OUTPUT_ROOT
    args.batch_size = FORMAL_BATCH_SIZE
    args.workers = FORMAL_WORKERS
    args.base_lr = FORMAL_BASE_LR
    args.min_lr = FORMAL_MIN_LR
    args.val_interval = FORMAL_VAL_INTERVAL
    try:
        _require_variant_args(args)
    except WeightedDSRunnerError as exc:
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
            raise WeightedDSRunnerError(
                f"weighted-DS-v1 freezes {name}={expected!r}"
            )
    if Path(args.output_root).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise WeightedDSRunnerError(
            "weighted-DS-v1 output root is fixed under "
            "runs/irstd_performance/weighted_ds_v1"
        )
    smoke = r1._is_smoke(args)
    if not smoke and (
        args.run_seed != PAIRED_RUN_SEED
        or args.epochs != FORMAL_EPOCHS
        or args.warmup_epochs != FORMAL_WARMUP_EPOCHS
    ):
        raise WeightedDSRunnerError(
            "formal weighted-DS-v1 requires the paired seed and 1000-epoch recipe"
        )
    if not smoke and args.device != "cuda:0":
        raise WeightedDSRunnerError(
            "formal weighted-DS-v1 requires logical --device cuda:0"
        )
    if not smoke:
        supplied_split_root = Path(os.path.abspath(args.split_root))
        canonical_split_root = Path(os.path.abspath(CANONICAL_SPLIT_ROOT))
        if supplied_split_root != canonical_split_root:
            raise WeightedDSRunnerError(
                "formal run requires the repository canonical splits/v2 root"
            )
        manifest = canonical_split_root / DATASET / "manifest.json"
        if (
            canonical_split_root.is_symlink()
            or not canonical_split_root.is_dir()
            or manifest.is_symlink()
            or not manifest.is_file()
            or _sha256_file(manifest) != CANONICAL_IRSTD_MANIFEST_SHA256
        ):
            raise WeightedDSRunnerError("canonical IRSTD-1K split identity differs")
    if smoke and not 0 <= args.warmup_epochs <= args.epochs:
        raise WeightedDSRunnerError("smoke warmup schedule is invalid")
    loss_policy_identity()


def _runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
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
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _variant_source_provenance() -> dict[str, Any]:
    baseline = _R1_PROTOCOL_SOURCE_PROVENANCE()
    baseline_files = baseline.get("files")
    if not isinstance(baseline_files, Mapping):
        raise WeightedDSRunnerError("R1 source provenance is malformed")
    files = {
        f"r1/{name}": copy.deepcopy(entry)
        for name, entry in sorted(baseline_files.items())
    }
    for name, source in {
        "variant_runner": Path(__file__),
        "canonical_complete_target_gate": Path(canonical_gate.__file__),
    }.items():
        if source.is_symlink() or not source.is_file():
            raise WeightedDSRunnerError(f"{name} is not a regular file")
        resolved = source.resolve(strict=True)
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise WeightedDSRunnerError(f"{name} is outside repository") from exc
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
    identity = copy.deepcopy(_R1_DETERMINISM_PROTOCOL_IDENTITY())
    source = _variant_source_provenance()
    identity["schema"] = DETERMINISM_SCHEMA
    identity["training_loss"] = loss_policy_identity()
    identity["only_training_change_from_R1"] = "deep_supervision_loss_weights"
    identity["source_set_schema"] = source["schema"]
    identity["source_files"] = source["files"]
    identity["source_tree_sha256"] = source["source_tree_sha256"]
    return identity


def build_datasets(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Delegate exactly to the frozen R1 legacy-crop train/val constructor."""

    return _R1_BUILD_DATASETS(args)


def _engine_build_datasets(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    return build_datasets(args)


def _predecessor_evidence_for_identity(smoke: bool) -> dict[str, Any]:
    evidence = getattr(_RUNTIME_STATE, "predecessor_gate_evidence", None)
    if smoke:
        if evidence != {
            "schema": PREDECESSOR_GATE_EVIDENCE_SCHEMA,
            "status": "not_applicable_to_smoke",
            "formal_weighted_ds_allowed": False,
            "public_test_accessed": False,
        }:
            raise WeightedDSRunnerError("smoke predecessor evidence differs")
    else:
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("schema") != PREDECESSOR_GATE_EVIDENCE_SCHEMA
            or evidence.get("status") != "complete"
            or evidence.get("authority")
            != (
                "run_irstd_complete_target_promotion_gate."
                "validate_existing_result"
            )
            or evidence.get("canonical_payload_schema")
            != CANONICAL_GATE_RESULT_SCHEMA
            or evidence.get("formal_weighted_ds_allowed") is not True
            or evidence.get("public_test_accessed") is not False
        ):
            raise WeightedDSRunnerError(
                "formal weighted-DS canonical gate evidence is malformed"
            )
        artifact = evidence.get("canonical_result_artifact")
        payload = _require_canonical_failure_payload(
            evidence.get("canonical_payload")
        )
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("relative_path")
            != CANONICAL_GATE_RESULT_RELATIVE_PATH
            or not isinstance(artifact.get("sha256"), str)
            or len(artifact["sha256"]) != 64
            or evidence.get("canonical_payload_sha256")
            != _canonical_sha256(payload)
        ):
            raise WeightedDSRunnerError(
                "formal weighted-DS canonical gate binding is malformed"
            )
    return _json_clone(evidence)


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
        or not contract.data_tree_verified
        or train_count != CANONICAL_IRSTD_TRAIN_COUNT
        or val_count != CANONICAL_IRSTD_VAL_COUNT
    ):
        raise WeightedDSRunnerError(
            "formal run identity differs from canonical IRSTD-1K"
        )
    identity = {
        "schema": TRAINING_SCHEMA + "/run_identity",
        "experiment": {
            "schema": EXPERIMENT_SCHEMA,
            "name": "IRSTD-1K weighted deep supervision v1",
            "status": "experimental_validation_only",
            "only_training_loss_weights_differ_from_R1": True,
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
        "optimizer_hyperparameters": _json_clone(r1.R1_ADAM_GROUP_HYPERPARAMETERS),
        "loss": "weighted_sum_of_six_BCELoss_mean_terms",
        "loss_policy": loss_policy_identity(),
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "evaluation_head": "out",
        "selection_rule": r1.selection.INDEPENDENT_RULE_VERSION,
        "determinism_protocol": _variant_determinism_protocol_identity(),
        "legacy_R1_train_crop_and_data": True,
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
        "runtime_identity": _runtime_identity(args),
        "predecessor_gate_evidence": _predecessor_evidence_for_identity(smoke),
        "promotion_gate": promotion_gate(),
        "test_split_accessed": False,
    }
    identity = _json_clone(identity)
    identity["identity_sha256"] = _canonical_sha256(identity)
    _RUNTIME_STATE.run_identity = dict(identity)
    return identity


def _variant_final_checkpoint_payload(**kwargs: Any) -> dict[str, Any]:
    payload = _R1_BUILD_FINAL_CHECKPOINT_PAYLOAD(**kwargs)
    identity = kwargs.get("identity")
    if not isinstance(identity, Mapping):
        raise WeightedDSRunnerError("final weighted-DS identity is missing")
    payload.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint_role": "experimental_validation_selected",
            "experiment_schema": EXPERIMENT_SCHEMA,
            "experiment_status": "experimental_validation_only",
            "only_training_loss_weights_differ_from_R1": True,
            "loss_policy": loss_policy_identity(),
            "predecessor_gate_evidence": _json_clone(
                identity["predecessor_gate_evidence"]
            ),
            "promotion_gate": promotion_gate(),
            "public_test_supported": False,
            "public_test_gate_status": "unsupported_until_separate_gate_extension",
            "test_split_accessed": False,
        }
    )
    return payload


def _variant_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = dict(payload)
    if output.get("schema") == TRAINING_SCHEMA + "/summary":
        selected_epoch = output.get("selected_epoch")
        validation_history = output.get("validation_history")
        if isinstance(selected_epoch, bool) or not isinstance(
            selected_epoch, int
        ) or not isinstance(validation_history, list):
            raise WeightedDSRunnerError("weighted-DS selected record is malformed")
        selected = [
            record
            for record in validation_history
            if isinstance(record, Mapping) and record.get("epoch") == selected_epoch
        ]
        if len(selected) != 1:
            raise WeightedDSRunnerError("weighted-DS selected record is ambiguous")
        identity = getattr(_RUNTIME_STATE, "run_identity", None)
        if not isinstance(identity, Mapping):
            raise WeightedDSRunnerError("weighted-DS summary identity is missing")
        output.update(
            {
                "checkpoint_role": "experimental_validation_selected",
                "experiment_schema": EXPERIMENT_SCHEMA,
                "experiment_status": "experimental_validation_only",
                "only_training_loss_weights_differ_from_R1": True,
                "loss_policy": loss_policy_identity(),
                "selected_validation_record": _json_clone(selected[0]),
                "selected_validation_record_sha256": _canonical_sha256(selected[0]),
                "run_identity": _json_clone(identity),
                "training_identity_sha256": identity["identity_sha256"],
                "predecessor_gate_evidence": _json_clone(
                    identity["predecessor_gate_evidence"]
                ),
                "promotion_gate": promotion_gate(),
                "public_test_supported": False,
                "public_test_gate_status": (
                    "unsupported_until_separate_gate_extension"
                ),
                "test_split_accessed": False,
            }
        )
        checkpoint_relative = output.get("checkpoint")
        if not isinstance(checkpoint_relative, str):
            raise WeightedDSRunnerError("weighted-DS checkpoint path is malformed")
        checkpoint_path = PROJECT_ROOT / checkpoint_relative
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise WeightedDSRunnerError("weighted-DS final checkpoint is unavailable")
        output["final_checkpoint_sha256"] = _sha256_file(checkpoint_path)
    _R1_WRITE_JSON(path, output)


def _variant_atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    output = dict(payload)
    if output.get("schema") in {TRAINING_SCHEMA, CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA}:
        output.update(
            {
                "experiment_schema": EXPERIMENT_SCHEMA,
                "experiment_status": "experimental_validation_only",
                "only_training_loss_weights_differ_from_R1": True,
                "loss_policy": loss_policy_identity(),
                "public_test_supported": False,
                "public_test_gate_status": (
                    "unsupported_until_separate_gate_extension"
                ),
                "test_split_accessed": False,
            }
        )
    _R1_ATOMIC_TORCH_SAVE(path, output)


@contextmanager
def _variant_runtime(predecessor_gate_evidence: Mapping[str, Any]) -> Iterator[None]:
    """Temporarily and exception-safely route the R1 engine to this variant."""

    overrides = {
        "TRAINING_SCHEMA": TRAINING_SCHEMA,
        "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "HISTORY_SCHEMA": HISTORY_SCHEMA,
        "SELECTION_PAYLOAD_SCHEMA": SELECTION_PAYLOAD_SCHEMA,
        "build_datasets": _engine_build_datasets,
        "_protocol_source_provenance": _variant_source_provenance,
        "_determinism_protocol_identity": _variant_determinism_protocol_identity,
        "_run_identity": _variant_run_identity,
        "build_final_checkpoint_payload": _variant_final_checkpoint_payload,
        "_write_json": _variant_write_json,
        "_atomic_torch_save": _variant_atomic_torch_save,
    }
    if not _PATCH_LOCK.acquire(blocking=False):
        raise WeightedDSRunnerError("weighted-DS runtime is already active")
    previous = {name: getattr(r1, name) for name in overrides}
    previous_loss = r1.legacy_train.deep_supervision_loss
    _RUNTIME_STATE.predecessor_gate_evidence = _json_clone(
        predecessor_gate_evidence
    )
    _RUNTIME_STATE.run_identity = None
    try:
        for name, value in overrides.items():
            setattr(r1, name, value)
        r1.legacy_train.deep_supervision_loss = weighted_deep_supervision_loss
        yield
    finally:
        r1.legacy_train.deep_supervision_loss = previous_loss
        for name, value in previous.items():
            setattr(r1, name, value)
        for name in ("predecessor_gate_evidence", "run_identity"):
            if hasattr(_RUNTIME_STATE, name):
                delattr(_RUNTIME_STATE, name)
        _PATCH_LOCK.release()


@contextmanager
def _run_process_lock(args: argparse.Namespace) -> Iterator[Path]:
    paths = resolve_run_paths(args)
    run_dir = paths["run_dir"]
    if not isinstance(run_dir, Path):
        raise WeightedDSRunnerError("weighted-DS run directory is malformed")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".weighted_ds_v1.lock"
    if lock_path.is_symlink():
        raise WeightedDSRunnerError("weighted-DS run lock must not be a symlink")
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
                raise WeightedDSRunnerError(
                    f"weighted-DS run is already locked: {run_dir}"
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
    """Run smoke, or formal only after an evidenced predecessor gate failure."""

    _require_variant_args(args)
    smoke = r1._is_smoke(args)
    if smoke:
        predecessor_evidence = {
            "schema": PREDECESSOR_GATE_EVIDENCE_SCHEMA,
            "status": "not_applicable_to_smoke",
            "formal_weighted_ds_allowed": False,
            "public_test_accessed": False,
        }
    else:
        predecessor_evidence = validate_canonical_predecessor_failure()
    with _run_process_lock(args), _variant_runtime(predecessor_evidence):
        checkpoint = r1.run(args)
    return checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    checkpoint = run(parse_args(argv))
    print(checkpoint)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "DATASET",
    "DEEP_SUPERVISION_HEAD_ORDER",
    "DEEP_SUPERVISION_WEIGHTS",
    "DEFAULT_OUTPUT_ROOT",
    "EXPERIMENT_SCHEMA",
    "FORMAL_EPOCHS",
    "HISTORY_SCHEMA",
    "PAIRED_RUN_SEED",
    "PROMOTION_GATE_SCHEMA",
    "SELECTION_PAYLOAD_SCHEMA",
    "TARGET_MODE",
    "TRAINING_SCHEMA",
    "WeightedDSRunnerError",
    "build_datasets",
    "loss_policy_identity",
    "parse_args",
    "promotion_gate",
    "resolve_run_paths",
    "run",
    "validate_canonical_predecessor_failure",
    "weighted_deep_supervision_loss",
]
