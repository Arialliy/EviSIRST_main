#!/usr/bin/env python3
"""Evaluate the fixed single-seed complete-target-v1 validation gate.

The command has no path, split, checkpoint, seed, metric, or threshold
arguments.  It consumes exactly the two completed formal summaries and final
checkpoints plus the fixed paired-baseline matched-target diagnostic.  It
never imports a public-test dataset and never permits public-test evaluation.

Passing this gate authorizes only the preregistered three-runtime-seed
validation expansion.  Public test remains blocked behind a separate review.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from experiments import evisirst_v2_selection as validation_selection


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET = "IRSTD-1K"
DATA_ROLE = "val"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
RUN_SEED = 1446202191
EPOCHS = 1000

CANONICAL_MANIFEST_SHA256 = (
    "5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596"
)
CANONICAL_DATA_TREE_SHA256 = (
    "ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30"
)
CANONICAL_SPLIT_SEED = 20260811
CANONICAL_TRAIN_COUNT = 640
CANONICAL_VAL_COUNT = 160
CANONICAL_SPLIT_FILES = {
    "manifest": {
        "relative_path": "splits/v2/IRSTD-1K/manifest.json",
        "sha256": CANONICAL_MANIFEST_SHA256,
    },
    "train": {
        "relative_path": "splits/v2/IRSTD-1K/train.txt",
        "sha256": "460083baae2ba23f5629e7bd346b5623f78a96856c83693b96bf917f6537ed2d",
        "ordered_ids_sha256": (
            "06c1d6055958b648aeaab9be4dc02721eeaba735d4a39fe3184a8aa4ffc6df33"
        ),
        "sample_count": CANONICAL_TRAIN_COUNT,
    },
    "val": {
        "relative_path": "splits/v2/IRSTD-1K/val.txt",
        "sha256": "05a0d0ecdb1772447c4b5b0b04a5e8cf3a748ba5fdab5cdb3f9323d9089e9576",
        "ordered_ids_sha256": (
            "1800db0d308465c9f8d906731576fdbc35fc1c25bc7790cda5232f20fb9ed408"
        ),
        "sample_count": CANONICAL_VAL_COUNT,
    },
}

BASELINE_SUMMARY_RELATIVE_PATH = (
    "runs/validation_selected/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/summary.json"
)
BASELINE_CHECKPOINT_RELATIVE_PATH = (
    "runs/validation_selected/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/EviSIRST.pth.tar"
)
VARIANT_SUMMARY_RELATIVE_PATH = (
    "runs/irstd_performance/complete_target_v1/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/summary.json"
)
VARIANT_CHECKPOINT_RELATIVE_PATH = (
    "runs/irstd_performance/complete_target_v1/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/EviSIRST.pth.tar"
)
BASELINE_DIAGNOSTIC_RELATIVE_PATH = (
    "runs/irstd_performance/complete_target_v1/paired_baseline_diagnostics/"
    "run_seed_1446202191/matched_target_diagnostics.json"
)
OUTPUT_RELATIVE_PATH = (
    "runs/irstd_performance/complete_target_v1/promotion_gate/"
    "run_seed_1446202191/result.json"
)
GATE_SOURCE_RELATIVE_PATH = "run_irstd_complete_target_promotion_gate.py"

RESULT_SCHEMA = "evisirst_irstd_complete_target_promotion_gate_result/v1"
BASELINE_TRAINING_SCHEMA = "evisirst_validation_selected_training/v1"
BASELINE_CHECKPOINT_SCHEMA = "evisirst_clean_checkpoint/v1"
VARIANT_TRAINING_SCHEMA = "evisirst_irstd_complete_target_training/v1"
VARIANT_CHECKPOINT_SCHEMA = "evisirst_irstd_complete_target_checkpoint/v1"
VARIANT_EXPERIMENT_SCHEMA = "evisirst_irstd_complete_target_experiment/v1"
PREREGISTERED_GATE_SCHEMA = "evisirst_irstd_complete_target_promotion_gate/v1"
BASELINE_DIAGNOSTIC_SCHEMA = "evisirst_irstd_paired_baseline_diagnostic/v1"
MECHANISM_SCHEMA = "evisirst_matched_target_diagnostics/v1/aggregate"

PRIMARY_MINIMUM_DELTA = 0.001
SAFETY_PD_FAILURE_THRESHOLD = -0.003
SAFETY_FA_FAILURE_THRESHOLD = 0.0

_BASELINE_SUMMARY_KEYS = frozenset(
    {
        "schema", "status", "dataset", "checkpoint", "checkpoint_role",
        "selected_epoch", "architecture_seed", "run_seed", "target_mode",
        "split_provenance", "selection", "training_history",
        "validation_history", "candidate_artifacts", "normalization",
        "source_selection", "selection_is_optimistic", "optimistic",
        "test_split_accessed", "smoke", "elapsed_seconds",
    }
)
_VARIANT_SUMMARY_KEYS = frozenset(
    {
        "schema", "status", "dataset", "checkpoint", "checkpoint_role",
        "selected_epoch", "architecture_seed", "run_seed", "target_mode",
        "split_provenance", "selection", "training_history",
        "validation_history", "candidate_artifacts", "normalization",
        "source_selection", "selection_is_optimistic", "optimistic",
        "test_split_accessed", "smoke", "elapsed_seconds",
        "crop_audit_commit_status", "crop_policy",
        "crop_policy_identity_sha256", "experiment_schema",
        "experiment_status", "final_checkpoint_sha256",
        "only_train_crop_policy_differs_from_R1", "promotion_gate",
        "public_test_gate_status", "public_test_supported", "run_identity",
        "search_run_crop_audit_history", "search_run_crop_audit_state",
        "search_run_crop_audit_summary", "search_run_crop_audit_through_epoch",
        "selected_candidate_sha256", "selected_checkpoint_crop_audit_history",
        "selected_checkpoint_crop_audit_state",
        "selected_checkpoint_crop_audit_summary",
        "selected_checkpoint_crop_audit_through_epoch",
        "selected_validation_record", "selected_validation_record_sha256",
        "training_identity_sha256",
    }
)
_BASELINE_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "model", "dataset", "checkpoint_role", "epoch", "seed",
        "architecture_seed", "run_seed", "state_dict", "target_mode",
        "normalization", "normalization_provenance", "training",
        "training_identity_sha256", "split_provenance", "split_seed",
        "split_manifest_sha256", "data_tree_sha256", "data_tree_verified",
        "selection_provenance", "source_selection", "selection_is_optimistic",
        "optimistic", "test_split_accessed", "model_metadata", "smoke",
    }
)
_VARIANT_CHECKPOINT_KEYS = frozenset(
    set(_BASELINE_CHECKPOINT_KEYS)
    | {
        "crop_audit_commit_status", "crop_policy",
        "crop_policy_identity_sha256", "experiment_schema",
        "experiment_status", "only_train_crop_policy_differs_from_R1",
        "promotion_gate", "public_test_gate_status", "public_test_supported",
        "search_run_crop_audit_history", "search_run_crop_audit_state",
        "search_run_crop_audit_summary", "search_run_crop_audit_through_epoch",
        "selected_candidate_sha256", "selected_checkpoint_crop_audit_history",
        "selected_checkpoint_crop_audit_state",
        "selected_checkpoint_crop_audit_summary",
        "selected_checkpoint_crop_audit_through_epoch",
    }
)
_BASELINE_IDENTITY_KEYS = frozenset(
    {
        "schema", "model", "dataset", "architecture_seed", "run_seed",
        "target_mode", "epochs", "batch_size", "workers", "base_lr",
        "min_lr", "warmup_epochs", "val_interval", "normalization_mode",
        "optimizer", "loss", "evaluation", "selection_rule",
        "determinism_protocol", "manifest_sha256", "split_seed",
        "data_tree_sha256", "grouping_policy", "train_count", "val_count",
        "smoke", "smoke_max_train_samples", "smoke_max_val_samples",
        "test_split_accessed", "identity_sha256",
    }
)
_VARIANT_IDENTITY_KEYS = frozenset(
    {
        "schema", "experiment", "model", "dataset", "architecture_seed",
        "run_seed", "target_mode", "epochs", "batch_size", "workers",
        "base_lr", "min_lr", "warmup_epochs", "val_interval",
        "normalization_mode", "optimizer", "optimizer_hyperparameters",
        "loss", "deep_supervision_probability_heads",
        "deep_supervision_weights", "evaluation", "evaluation_head",
        "evaluation_supplementary_diagnostics", "execution_contract",
        "runtime_identity", "selection_rule", "determinism_protocol",
        "crop_policy", "crop_policy_identity_sha256", "manifest_sha256",
        "split_seed", "data_tree_sha256", "canonical_split_contract",
        "grouping_policy", "train_count", "val_count", "smoke",
        "smoke_max_train_samples", "smoke_max_val_samples", "promotion_gate",
        "test_split_accessed", "identity_sha256",
    }
)
_SELECTION_PAYLOAD_KEYS = frozenset(
    {
        "schema", "data_role", "source_selection", "selected_epoch",
        "selected_candidate", "retention_frontier_epochs",
        "selection_provenance", "selection_is_optimistic", "optimistic",
    }
)
_DIAGNOSTIC_KEYS = frozenset(
    {
        "schema", "status", "dataset", "data_role", "diagnostic_only",
        "selection_allowed", "selected_on_same_validation_split", "checkpoint",
        "completed_run_summary", "split", "sources", "evaluation", "metrics",
        "test_split_accessed",
    }
)
_MECHANISM_KEYS = frozenset(
    {
        "schema", "prediction_threshold_rule", "target_threshold_rule",
        "component_connectivity", "component_neighborhood",
        "assignment_algorithm", "match_rule", "image_count",
        "target_component_count", "predicted_component_count",
        "matched_component_count", "matched_overlap_pixel_count",
        "matched_target_pixel_count", "matched_target_pixel_recall",
        "matched_component_area_ratio", "centroid_error",
    }
)

_BASELINE_SOURCE_PATHS = {
    "data": "experiments/evisirst_v2_data.py",
    "frozen_builder": "experiments/four_dataset_models_seed42_v1.py",
    "legacy_data": "experiments/evisirst_data.py",
    "legacy_train": "train.py",
    "model_entry": "model/EviSIRST.py",
    "model_internal/Config.py": "model/_internal/Config.py",
    "model_internal/SCTransNet.py": "model/_internal/SCTransNet.py",
    "model_internal/tpd_clean.py": "model/_internal/tpd_clean.py",
    "model_internal/tpd_clean_v8_mprs_dch.py": "model/_internal/tpd_clean_v8_mprs_dch.py",
    "model_internal/tpd_forward_contract.py": "model/_internal/tpd_forward_contract.py",
    "model_internal/tpd_frequency_gate.py": "model/_internal/tpd_frequency_gate.py",
    "model_internal/tpd_frequency_gate_v2_croa.py": "model/_internal/tpd_frequency_gate_v2_croa.py",
    "model_internal/tpd_ner_v8_mprs_dch.py": "model/_internal/tpd_ner_v8_mprs_dch.py",
    "model_internal/tpd_ner_v8_mprs_dch_v2.py": "model/_internal/tpd_ner_v8_mprs_dch_v2.py",
    "model_internal/tpd_ner_v8_mprs_dch_v3.py": "model/_internal/tpd_ner_v8_mprs_dch_v3.py",
    "model_internal/tpd_ner_v8_mprs_dch_v4_tail_aware.py": "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware.py",
    "model_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py": "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
    "model_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py": "model/_internal/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py",
    "model_internal/tpd_query_frequency_bridge.py": "model/_internal/tpd_query_frequency_bridge.py",
    "model_internal/tpd_relay.py": "model/_internal/tpd_relay.py",
    "model_internal/tpd_sctransnet.py": "model/_internal/tpd_sctransnet.py",
    "model_internal/tpd_survival.py": "model/_internal/tpd_survival.py",
    "model_package": "model/__init__.py",
    "selection": "experiments/evisirst_v2_selection.py",
    "source_protocol": "experiments/three_dataset_v2_protocol.py",
    "split_protocol": "experiments/evisirst_v2_splits.py",
    "trainer": "train_validation_selected.py",
}
_VARIANT_SOURCE_PATHS = {
    **{f"r1/{name}": path for name, path in _BASELINE_SOURCE_PATHS.items()},
    "complete_target_crop": "experiments/evisirst_complete_target_crop.py",
    "variant_runner": "train_irstd_complete_target_v1.py",
}
_DIAGNOSTIC_SOURCE_PATHS = {
    "diagnostic_cli": "run_irstd_paired_baseline_diagnostic.py",
    "r1_metrics": "train_validation_selected.py",
    "matched_target_diagnostics": "experiments/evisirst_complete_target_crop.py",
    "validation_data": "experiments/evisirst_v2_data.py",
    "validation_selection": "experiments/evisirst_v2_selection.py",
    "model_entry": "model/EviSIRST.py",
}


class PromotionGateError(ValueError):
    """A required artifact does not prove the frozen gate inputs."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PromotionGateError("artifact is not strict finite JSON data") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PromotionGateError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_keys(
    value: Any, expected: frozenset[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PromotionGateError(f"{label} must be a string-keyed object")
    missing = sorted(expected.difference(value))
    unexpected = sorted(set(value).difference(expected))
    if missing or unexpected:
        raise PromotionGateError(
            f"{label} keys differ: missing={missing}, unexpected={unexpected}"
        )
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PromotionGateError(f"{label} must be an integer >= {minimum}")
    return value


def _require_metric(
    value: Any, *, label: str, unit_interval: bool, strictly_positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionGateError(f"{label} must be a numeric scalar, not TBD/null")
    metric = float(value)
    if not math.isfinite(metric):
        raise PromotionGateError(f"{label} must be finite")
    if unit_interval and not 0.0 <= metric <= 1.0:
        raise PromotionGateError(f"{label} must be in [0, 1]")
    if not unit_interval and metric < 0.0:
        raise PromotionGateError(f"{label} must be non-negative")
    if strictly_positive and metric <= 0.0:
        raise PromotionGateError(f"{label} must be strictly positive")
    return metric


def _fixed_path(relative_path: str, *, must_exist: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise PromotionGateError("internal artifact path is unsafe")
    root = PROJECT_ROOT.resolve(strict=True)
    path = root.joinpath(*pure.parts)
    current = root
    for component in pure.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise PromotionGateError(f"artifact path contains a symlink: {relative_path}")
    if must_exist:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.resolve(strict=True) != path:
            raise PromotionGateError(f"artifact path was redirected: {relative_path}")
    return path


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PromotionGateError(f"{label} is not a regular file")
        content = handle.read()
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if not content or identity_before != identity_after:
        raise PromotionGateError(f"{label} is empty or changed while being read")
    return content


def _strict_json_object(content: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            parse_float=_finite_json_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PromotionGateError(f"{label} is not strict finite UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise PromotionGateError(f"{label} must contain a JSON object")
    _canonical_json_bytes(value)
    return value


def _load_json(relative_path: str, *, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    path = _fixed_path(relative_path, must_exist=True)
    content = _read_regular_file_once(path, label=label)
    return dict(_strict_json_object(content, label=label)), {
        "relative_path": relative_path,
        "sha256": _sha256_bytes(content),
    }


def _load_checkpoint(
    relative_path: str, *, label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    path = _fixed_path(relative_path, must_exist=True)
    content = _read_regular_file_once(path, label=label)
    try:
        raw = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PromotionGateError(f"{label} is not a weights-only checkpoint") from exc
    if not isinstance(raw, Mapping):
        raise PromotionGateError(f"{label} payload must be a mapping")
    return dict(raw), {"relative_path": relative_path, "sha256": _sha256_bytes(content)}


def _validate_state_dict(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or len(value) != 564:
        raise PromotionGateError(f"{label} must contain exactly 564 state tensors")
    for name, tensor in value.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise PromotionGateError(f"{label} contains a malformed state entry")
        if tensor.is_sparse or not bool(torch.isfinite(tensor).all().item()):
            raise PromotionGateError(f"{label}.{name} is sparse or non-finite")


def _validate_local_file(relative_path: str, sha256: str, *, label: str) -> None:
    _require_sha256(sha256, label=f"{label}.sha256")
    path = _fixed_path(relative_path, must_exist=True)
    if _sha256_file(path) != sha256:
        raise PromotionGateError(f"{label} content SHA-256 differs")


def _validate_source_set(
    protocol: Any,
    *,
    expected_schema: str,
    expected_paths: Mapping[str, str],
    label: str,
) -> str:
    if not isinstance(protocol, Mapping):
        raise PromotionGateError(f"{label} determinism protocol is missing")
    if protocol.get("source_set_schema") != expected_schema:
        raise PromotionGateError(f"{label} source-set schema differs")
    files = protocol.get("source_files")
    if not isinstance(files, Mapping) or set(files) != set(expected_paths):
        raise PromotionGateError(f"{label} source-file registry differs")
    normalized: dict[str, dict[str, str]] = {}
    for name, expected_path in expected_paths.items():
        entry = files.get(name)
        if not isinstance(entry, Mapping) or set(entry) != {"relative_path", "sha256"}:
            raise PromotionGateError(f"{label} source {name!r} entry is malformed")
        if entry.get("relative_path") != expected_path:
            raise PromotionGateError(f"{label} source {name!r} path differs")
        digest = _require_sha256(entry.get("sha256"), label=f"{label}.{name}")
        _validate_local_file(
            expected_path, digest, label=f"{label} source {name!r}"
        )
        normalized[name] = {"relative_path": expected_path, "sha256": digest}
    tree = _require_sha256(
        protocol.get("source_tree_sha256"), label=f"{label}.source_tree_sha256"
    )
    if tree != _canonical_sha256(normalized):
        raise PromotionGateError(f"{label} source-tree SHA-256 differs")
    return tree


def _validate_preregistered_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionGateError("variant preregistered promotion gate is missing")
    fixed_top = {
        "schema": PREREGISTERED_GATE_SCHEMA,
        "status": "TBD",
    }
    for field, expected in fixed_top.items():
        if value.get(field) != expected:
            raise PromotionGateError(
                f"variant preregistered promotion gate {field} differs"
            )
    paired = value.get("paired_baseline")
    paired_fixed = {
        "training_schema": BASELINE_TRAINING_SCHEMA,
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "summary_relative_path": BASELINE_SUMMARY_RELATIVE_PATH,
    }
    if not isinstance(paired, Mapping) or any(
        paired.get(field) != expected for field, expected in paired_fixed.items()
    ):
        raise PromotionGateError("variant preregistered paired baseline differs")
    variant = value.get("variant")
    variant_fixed = {
        "training_schema": VARIANT_TRAINING_SCHEMA,
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
    }
    if not isinstance(variant, Mapping) or any(
        variant.get(field) != expected for field, expected in variant_fixed.items()
    ):
        raise PromotionGateError("variant preregistered variant identity differs")
    primary = value.get("primary")
    primary_fixed = {
        "metric": "selected_validation_mIoU",
        "comparison": "variant_minus_paired_baseline_raw",
        "operator": ">=",
        "minimum_delta": PRIMARY_MINIMUM_DELTA,
    }
    if not isinstance(primary, Mapping) or any(
        primary.get(field) != expected for field, expected in primary_fixed.items()
    ):
        raise PromotionGateError("variant preregistered primary rule differs")
    safety = value.get("safety")
    safety_fixed = {
        "pd_delta_definition": "variant_minus_paired_baseline_raw",
        "pd_failure_operator": "<",
        "pd_failure_threshold": SAFETY_PD_FAILURE_THRESHOLD,
        "fa_delta_definition": "variant_minus_paired_baseline_raw",
        "fa_failure_operator": ">",
        "fa_failure_threshold": SAFETY_FA_FAILURE_THRESHOLD,
    }
    if not isinstance(safety, Mapping) or any(
        safety.get(field) != expected for field, expected in safety_fixed.items()
    ):
        raise PromotionGateError("variant preregistered safety rule differs")
    mechanism = value.get("mechanism")
    if (
        not isinstance(mechanism, Mapping)
        or mechanism.get("pass_rule") != "at_least_one_metric_strictly_improves"
    ):
        raise PromotionGateError("variant preregistered mechanism rule differs")
    protocol = mechanism.get("measurement_protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("data_role") != "same_canonical_validation_split"
        or protocol.get("evaluation_head") != "out"
        or protocol.get("selection_allowed") is not False
        or protocol.get("variant_selected_record")
        != "selected_validation_record.metrics.complete_target_mechanism_diagnostics"
    ):
        raise PromotionGateError(
            "variant preregistered mechanism measurement protocol differs"
        )
    baseline_artifact = protocol.get("paired_baseline_artifact")
    if (
        not isinstance(baseline_artifact, Mapping)
        or baseline_artifact.get("input_checkpoint_relative_path")
        != BASELINE_CHECKPOINT_RELATIVE_PATH
        or baseline_artifact.get("output_artifact_template")
        != BASELINE_DIAGNOSTIC_RELATIVE_PATH
    ):
        raise PromotionGateError(
            "variant preregistered baseline diagnostic identity differs"
        )
    metrics = mechanism.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 2:
        raise PromotionGateError("variant preregistered mechanism metrics differ")
    expected_metrics = (
        {
            "name": "matched_target_pixel_recall",
            "comparison": "variant_minus_paired_baseline_raw",
            "operator": ">",
            "source_schema": MECHANISM_SCHEMA,
        },
        {
            "name": "matched_component_area_ratio_closeness_to_one",
            "source_metric": "matched_component_area_ratio",
            "transform": "-abs(raw_value-1.0)",
            "comparison": "variant_minus_paired_baseline_transformed",
            "operator": ">",
            "source_schema": MECHANISM_SCHEMA,
        },
    )
    for observed, expected in zip(metrics, expected_metrics):
        if not isinstance(observed, Mapping) or any(
            observed.get(field) != expected_value
            for field, expected_value in expected.items()
        ):
            raise PromotionGateError(
                "variant preregistered mechanism metric definition differs"
            )
    decision = value.get("decision")
    decision_fixed = {
        "single_seed_required_before_expansion": True,
        "expand_to_three_runtime_seeds_only_if_all_gates_pass": True,
        "public_test_allowed": False,
        "public_test_status": "unsupported_until_separate_gate_extension",
    }
    if not isinstance(decision, Mapping) or any(
        decision.get(field) != expected for field, expected in decision_fixed.items()
    ):
        raise PromotionGateError("variant preregistered decision rule differs")
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _validate_identity(
    raw: Any, *, variant: bool
) -> tuple[dict[str, Any], str]:
    label = "variant run identity" if variant else "baseline run identity"
    expected_keys = _VARIANT_IDENTITY_KEYS if variant else _BASELINE_IDENTITY_KEYS
    identity = dict(_require_exact_keys(raw, expected_keys, label=label))
    observed_sha = _require_sha256(
        identity.get("identity_sha256"), label=f"{label}.identity_sha256"
    )
    unhashed = dict(identity)
    del unhashed["identity_sha256"]
    if _canonical_sha256(unhashed) != observed_sha:
        raise PromotionGateError(f"{label} SHA-256 differs")

    fixed = {
        "schema": (
            VARIANT_TRAINING_SCHEMA if variant else BASELINE_TRAINING_SCHEMA
        ) + "/run_identity",
        "model": "EviSIRST",
        "dataset": DATASET,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "target_mode": TARGET_MODE,
        "epochs": EPOCHS,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "val_interval": 1,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "selection_rule": validation_selection.INDEPENDENT_RULE_VERSION,
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "split_seed": CANONICAL_SPLIT_SEED,
        "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
        "train_count": CANONICAL_TRAIN_COUNT,
        "val_count": CANONICAL_VAL_COUNT,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    for field, expected in fixed.items():
        if type(identity.get(field)) is not type(expected) or identity.get(field) != expected:
            raise PromotionGateError(f"{label}.{field} differs")

    if variant:
        variant_fixed = {
            "evaluation_head": "out",
            "deep_supervision_probability_heads": 6,
            "deep_supervision_weights": [1.0] * 6,
        }
        for field, expected in variant_fixed.items():
            if identity.get(field) != expected:
                raise PromotionGateError(f"{label}.{field} differs")
        experiment = identity.get("experiment")
        if not isinstance(experiment, Mapping) or any(
            experiment.get(field) != expected
            for field, expected in {
                "schema": VARIANT_EXPERIMENT_SCHEMA,
                "status": "experimental_validation_only",
                "only_train_crop_policy_differs_from_R1": True,
                "public_test_supported": False,
                "public_test_gate_status": "unsupported_until_separate_gate_extension",
            }.items()
        ):
            raise PromotionGateError("variant experiment identity differs")
        canonical = identity.get("canonical_split_contract")
        expected_canonical = {
            "split_root_relative_path": "splits/v2",
            "manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
            "train_count": CANONICAL_TRAIN_COUNT,
            "val_count": CANONICAL_VAL_COUNT,
        }
        if canonical != expected_canonical:
            raise PromotionGateError("variant canonical split contract differs")
        crop_policy = identity.get("crop_policy")
        crop_sha = _require_sha256(
            identity.get("crop_policy_identity_sha256"),
            label="variant crop-policy identity",
        )
        if _canonical_sha256(crop_policy) != crop_sha:
            raise PromotionGateError("variant crop-policy SHA-256 differs")
        _validate_preregistered_gate(identity.get("promotion_gate"))

    source_tree = _validate_source_set(
        identity.get("determinism_protocol"),
        expected_schema=(
            "evisirst_irstd_complete_target_source_set/v1"
            if variant
            else "evisirst_validation_selected_source_set/v2"
        ),
        expected_paths=_VARIANT_SOURCE_PATHS if variant else _BASELINE_SOURCE_PATHS,
        label=label,
    )
    return identity, source_tree


def _validate_split(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionGateError(f"{label} split provenance is missing")
    fixed = {
        "schema": "evisirst_v2_train_val_split/v1",
        "manifest_relative_path": CANONICAL_SPLIT_FILES["manifest"]["relative_path"],
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "split_seed": CANONICAL_SPLIT_SEED,
        "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
        "data_tree_verified": True,
        "train_count": CANONICAL_TRAIN_COUNT,
        "val_count": CANONICAL_VAL_COUNT,
        "test_index_opened": False,
    }
    for field, expected in fixed.items():
        if type(value.get(field)) is not type(expected) or value.get(field) != expected:
            raise PromotionGateError(f"{label} split.{field} differs")
    outputs = value.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"train", "val"}:
        raise PromotionGateError(f"{label} split outputs differ")
    for role in ("train", "val"):
        observed = outputs.get(role)
        expected = CANONICAL_SPLIT_FILES[role]
        wanted = {
            "relative_path": expected["relative_path"],
            "file_sha256": expected["sha256"],
            "ordered_ids_sha256": expected["ordered_ids_sha256"],
            "sample_count": expected["sample_count"],
        }
        if observed != wanted:
            raise PromotionGateError(f"{label} canonical {role} index differs")
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _validate_canonical_split_files() -> None:
    for label, entry in CANONICAL_SPLIT_FILES.items():
        _validate_local_file(entry["relative_path"], entry["sha256"], label=label)


def _validate_histories(
    summary: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = summary.get("training_history")
    validation = summary.get("validation_history")
    expected_epochs = list(range(1, EPOCHS + 1))
    if (
        not isinstance(training, list)
        or len(training) != EPOCHS
        or [record.get("epoch") if isinstance(record, Mapping) else None for record in training]
        != expected_epochs
    ):
        raise PromotionGateError(f"{label} training history is not complete through 1000")
    if (
        not isinstance(validation, list)
        or len(validation) != EPOCHS
        or [record.get("epoch") if isinstance(record, Mapping) else None for record in validation]
        != expected_epochs
    ):
        raise PromotionGateError(f"{label} validation history is not complete through 1000")
    for position, record in enumerate(validation, start=1):
        if (
            not isinstance(record, Mapping)
            or record.get("data_role") != DATA_ROLE
            or record.get("evaluation_head") != "out"
        ):
            raise PromotionGateError(f"{label} validation record {position} differs")
        _require_metric(record.get("mIoU"), label=f"{label}[{position}].mIoU", unit_interval=True)
        _require_metric(record.get("Pd"), label=f"{label}[{position}].Pd", unit_interval=True)
        _require_metric(record.get("Fa"), label=f"{label}[{position}].Fa", unit_interval=False)
    try:
        recomputed = validation_selection.select_independent_checkpoint(validation)
    except (TypeError, ValueError) as exc:
        raise PromotionGateError(f"{label} selection cannot be recomputed") from exc
    selection_payload = dict(
        _require_exact_keys(
            summary.get("selection"), _SELECTION_PAYLOAD_KEYS, label=f"{label} selection"
        )
    )
    selected_epoch = recomputed.get("selected", {}).get("epoch")
    if (
        recomputed.get("evaluated_epochs") != expected_epochs
        or recomputed.get("data_role") != DATA_ROLE
        or recomputed.get("test_selection_supported") is not False
        or selection_payload.get("selection_provenance") != recomputed
        or selection_payload.get("selected_epoch") != selected_epoch
        or summary.get("selected_epoch") != selected_epoch
        or selection_payload.get("data_role") != DATA_ROLE
        or selection_payload.get("source_selection") != "evisirst_v2_validation_split"
        or selection_payload.get("selection_is_optimistic") is not False
        or selection_payload.get("optimistic") is not False
    ):
        raise PromotionGateError(f"{label} stored selection differs from recomputation")
    selected = validation[selected_epoch - 1]
    for metric in ("mIoU", "Fa", "Pd"):
        if selected.get(metric) != recomputed["selected"][metric]:
            raise PromotionGateError(f"{label} selected record {metric} differs")
    return json.loads(_canonical_json_bytes(recomputed).decode("ascii")), dict(selected)


def _validate_summary_common(
    summary: Mapping[str, Any], *, variant: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = "variant summary" if variant else "baseline summary"
    _require_exact_keys(
        summary, _VARIANT_SUMMARY_KEYS if variant else _BASELINE_SUMMARY_KEYS, label=label
    )
    checkpoint_path = VARIANT_CHECKPOINT_RELATIVE_PATH if variant else BASELINE_CHECKPOINT_RELATIVE_PATH
    fixed = {
        "schema": (VARIANT_TRAINING_SCHEMA if variant else BASELINE_TRAINING_SCHEMA) + "/summary",
        "status": "complete",
        "dataset": DATASET,
        "checkpoint": checkpoint_path,
        "checkpoint_role": (
            "experimental_validation_selected" if variant else "validation_selected"
        ),
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "target_mode": TARGET_MODE,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
    }
    for field, expected in fixed.items():
        if type(summary.get(field)) is not type(expected) or summary.get(field) != expected:
            raise PromotionGateError(f"{label}.{field} differs")
    _require_metric(summary.get("elapsed_seconds"), label=f"{label}.elapsed_seconds", unit_interval=False)
    split = _validate_split(summary.get("split_provenance"), label=label)
    recomputed, selected = _validate_histories(summary, label=label)
    if variant:
        variant_fixed = {
            "experiment_schema": VARIANT_EXPERIMENT_SCHEMA,
            "experiment_status": "experimental_validation_only",
            "only_train_crop_policy_differs_from_R1": True,
            "public_test_supported": False,
            "public_test_gate_status": "unsupported_until_separate_gate_extension",
            "crop_audit_commit_status": "complete",
            "search_run_crop_audit_through_epoch": EPOCHS,
            "selected_checkpoint_crop_audit_through_epoch": summary["selected_epoch"],
        }
        for field, expected in variant_fixed.items():
            if summary.get(field) != expected:
                raise PromotionGateError(f"{label}.{field} differs")
        if summary.get("selected_validation_record") != selected:
            raise PromotionGateError("variant selected validation record differs")
        if _canonical_sha256(selected) != summary.get("selected_validation_record_sha256"):
            raise PromotionGateError("variant selected validation record SHA-256 differs")
        _require_sha256(summary.get("final_checkpoint_sha256"), label="variant final checkpoint")
        selected_candidate_sha = _require_sha256(
            summary.get("selected_candidate_sha256"), label="variant selected candidate"
        )
        selected_artifact = summary["selection"].get("selected_candidate")
        if (
            not isinstance(selected_artifact, Mapping)
            or selected_artifact.get("file_sha256") != selected_candidate_sha
        ):
            raise PromotionGateError("variant selected candidate hash differs")
    return recomputed, selected


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    metadata: Mapping[str, str],
    *,
    summary: Mapping[str, Any],
    selection: Mapping[str, Any],
    variant: bool,
) -> tuple[dict[str, Any], str]:
    label = "variant checkpoint" if variant else "baseline checkpoint"
    _require_exact_keys(
        checkpoint,
        _VARIANT_CHECKPOINT_KEYS if variant else _BASELINE_CHECKPOINT_KEYS,
        label=label,
    )
    fixed = {
        "schema": VARIANT_CHECKPOINT_SCHEMA if variant else BASELINE_CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "checkpoint_role": (
            "experimental_validation_selected" if variant else "validation_selected"
        ),
        "epoch": summary["selected_epoch"],
        "seed": ARCHITECTURE_SEED,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "target_mode": TARGET_MODE,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
        "data_tree_verified": True,
        "split_seed": CANONICAL_SPLIT_SEED,
        "split_manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
    }
    for field, expected in fixed.items():
        if type(checkpoint.get(field)) is not type(expected) or checkpoint.get(field) != expected:
            raise PromotionGateError(f"{label}.{field} differs")
    if checkpoint.get("selection_provenance") != selection:
        raise PromotionGateError(f"{label} selection provenance differs")
    if checkpoint.get("split_provenance") != summary.get("split_provenance"):
        raise PromotionGateError(f"{label} split provenance differs")
    if checkpoint.get("normalization") != summary.get("normalization"):
        raise PromotionGateError(f"{label} normalization differs")
    _validate_state_dict(checkpoint.get("state_dict"), label=f"{label}.state_dict")
    identity, source_tree = _validate_identity(checkpoint.get("training"), variant=variant)
    if checkpoint.get("training_identity_sha256") != identity["identity_sha256"]:
        raise PromotionGateError(f"{label} training identity binding differs")
    if variant:
        variant_fixed = {
            "experiment_schema": VARIANT_EXPERIMENT_SCHEMA,
            "experiment_status": "experimental_validation_only",
            "only_train_crop_policy_differs_from_R1": True,
            "public_test_supported": False,
            "public_test_gate_status": "unsupported_until_separate_gate_extension",
            "crop_audit_commit_status": "complete",
            "search_run_crop_audit_through_epoch": EPOCHS,
            "selected_checkpoint_crop_audit_through_epoch": summary["selected_epoch"],
            "selected_candidate_sha256": summary["selected_candidate_sha256"],
        }
        for field, expected in variant_fixed.items():
            if checkpoint.get(field) != expected:
                raise PromotionGateError(f"{label}.{field} differs")
        if checkpoint.get("crop_policy") != summary.get("crop_policy"):
            raise PromotionGateError("variant checkpoint/summary crop policy differs")
        if checkpoint.get("crop_policy_identity_sha256") != summary.get("crop_policy_identity_sha256"):
            raise PromotionGateError("variant checkpoint/summary crop-policy hash differs")
        if metadata["sha256"] != summary.get("final_checkpoint_sha256"):
            raise PromotionGateError("variant final checkpoint content SHA-256 differs")
        if summary.get("run_identity") != identity:
            raise PromotionGateError("variant summary/checkpoint run identity differs")
        if summary.get("training_identity_sha256") != identity["identity_sha256"]:
            raise PromotionGateError("variant summary training identity binding differs")
        if (
            checkpoint.get("promotion_gate") != identity.get("promotion_gate")
            or summary.get("promotion_gate") != identity.get("promotion_gate")
        ):
            raise PromotionGateError(
                "variant checkpoint/summary preregistered gate differs"
            )
        if (
            checkpoint.get("crop_policy") != identity.get("crop_policy")
            or summary.get("crop_policy") != identity.get("crop_policy")
            or checkpoint.get("crop_policy_identity_sha256")
            != identity.get("crop_policy_identity_sha256")
            or summary.get("crop_policy_identity_sha256")
            != identity.get("crop_policy_identity_sha256")
        ):
            raise PromotionGateError(
                "variant checkpoint/summary run-identity crop policy differs"
            )
    return identity, source_tree


def _validate_mechanism(value: Any, *, label: str) -> dict[str, Any]:
    metrics = dict(_require_exact_keys(value, _MECHANISM_KEYS, label=label))
    fixed = {
        "schema": MECHANISM_SCHEMA,
        "prediction_threshold_rule": "probability>0.5",
        "target_threshold_rule": "target>0.5",
        "component_connectivity": 2,
        "component_neighborhood": "8-connected",
        "assignment_algorithm": "Hungarian/scipy.optimize.linear_sum_assignment",
        "match_rule": "centroid_distance<3",
        "image_count": CANONICAL_VAL_COUNT,
    }
    for field, expected in fixed.items():
        if type(metrics.get(field)) is not type(expected) or metrics.get(field) != expected:
            raise PromotionGateError(f"{label}.{field} differs")
    counts = {
        field: _require_int(metrics.get(field), label=f"{label}.{field}")
        for field in (
            "target_component_count", "predicted_component_count",
            "matched_component_count", "matched_overlap_pixel_count",
            "matched_target_pixel_count",
        )
    }
    if (
        counts["matched_component_count"] > counts["target_component_count"]
        or counts["matched_component_count"] > counts["predicted_component_count"]
        or counts["matched_overlap_pixel_count"] > counts["matched_target_pixel_count"]
        or counts["matched_component_count"] < 1
        or counts["matched_target_pixel_count"] < 1
    ):
        raise PromotionGateError(f"{label} count relationships differ")
    recall = _require_metric(
        metrics.get("matched_target_pixel_recall"),
        label=f"{label}.matched_target_pixel_recall",
        unit_interval=True,
    )
    expected_recall = counts["matched_overlap_pixel_count"] / counts["matched_target_pixel_count"]
    if recall != expected_recall:
        raise PromotionGateError(f"{label} matched-target recall differs from counts")
    _require_metric(
        metrics.get("matched_component_area_ratio"),
        label=f"{label}.matched_component_area_ratio",
        unit_interval=False,
        strictly_positive=True,
    )
    _require_metric(metrics.get("centroid_error"), label=f"{label}.centroid_error", unit_interval=False)
    return metrics


def _validate_diagnostic(
    diagnostic: Mapping[str, Any],
    metadata: Mapping[str, str],
    *,
    baseline_summary: Mapping[str, Any],
    baseline_summary_metadata: Mapping[str, str],
    baseline_checkpoint_metadata: Mapping[str, str],
    baseline_training_identity_sha256: str,
    baseline_source_tree: str,
    baseline_selection: Mapping[str, Any],
    baseline_selected_record: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(diagnostic, _DIAGNOSTIC_KEYS, label="baseline diagnostic")
    fixed = {
        "schema": BASELINE_DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": DATA_ROLE,
        "diagnostic_only": True,
        "selection_allowed": False,
        "selected_on_same_validation_split": True,
        "test_split_accessed": False,
    }
    for field, expected in fixed.items():
        if type(diagnostic.get(field)) is not type(expected) or diagnostic.get(field) != expected:
            raise PromotionGateError(f"baseline diagnostic.{field} differs")
    checkpoint = diagnostic.get("checkpoint")
    checkpoint_fixed = {
        "relative_path": BASELINE_CHECKPOINT_RELATIVE_PATH,
        "sha256": baseline_checkpoint_metadata["sha256"],
        "schema": BASELINE_CHECKPOINT_SCHEMA,
        "checkpoint_role": "validation_selected",
        "selected_epoch": baseline_summary["selected_epoch"],
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "target_mode": TARGET_MODE,
        "training_identity_sha256": baseline_training_identity_sha256,
        "strict_564_key_load": True,
    }
    if not isinstance(checkpoint, Mapping):
        raise PromotionGateError("baseline diagnostic checkpoint binding is missing")
    for field, expected in checkpoint_fixed.items():
        if checkpoint.get(field) != expected:
            raise PromotionGateError(f"baseline diagnostic checkpoint.{field} differs")
    if checkpoint.get("selection_provenance_sha256") != _canonical_sha256(
        baseline_selection
    ):
        raise PromotionGateError(
            "baseline diagnostic selection-provenance binding differs"
        )
    completed = diagnostic.get("completed_run_summary")
    expected_selected_sha = _canonical_sha256(baseline_selected_record)
    if (
        not isinstance(completed, Mapping)
        or completed.get("relative_path") != BASELINE_SUMMARY_RELATIVE_PATH
        or completed.get("sha256") != baseline_summary_metadata["sha256"]
        or completed.get("selected_validation_record_sha256") != expected_selected_sha
    ):
        raise PromotionGateError("baseline diagnostic summary binding differs")
    split = diagnostic.get("split")
    split_fixed = {
        "manifest_relative_path": CANONICAL_SPLIT_FILES["manifest"]["relative_path"],
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "val_index_relative_path": CANONICAL_SPLIT_FILES["val"]["relative_path"],
        "val_index_sha256": CANONICAL_SPLIT_FILES["val"]["sha256"],
        "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
        "data_tree_verified": True,
        "train_count": CANONICAL_TRAIN_COUNT,
        "val_count": CANONICAL_VAL_COUNT,
    }
    if split != split_fixed:
        raise PromotionGateError("baseline diagnostic canonical split differs")
    sources = diagnostic.get("sources")
    if (
        not isinstance(sources, Mapping)
        or sources.get("checkpoint_training_source_tree_sha256") != baseline_source_tree
        or sources.get("checkpoint_training_sources_currently_verified") is not True
    ):
        raise PromotionGateError("baseline diagnostic training-source binding differs")
    diagnostic_sources = sources.get("diagnostic")
    if (
        not isinstance(diagnostic_sources, Mapping)
        or diagnostic_sources.get("schema") != "evisirst_paired_baseline_diagnostic_source_set/v1"
    ):
        raise PromotionGateError("baseline diagnostic source set differs")
    diagnostic_tree = _validate_source_set(
        {
            "source_set_schema": diagnostic_sources.get("schema"),
            "source_files": diagnostic_sources.get("files"),
            "source_tree_sha256": diagnostic_sources.get("source_tree_sha256"),
        },
        expected_schema="evisirst_paired_baseline_diagnostic_source_set/v1",
        expected_paths=_DIAGNOSTIC_SOURCE_PATHS,
        label="baseline diagnostic",
    )
    if diagnostic_sources.get("source_tree_sha256") != diagnostic_tree:
        raise PromotionGateError("baseline diagnostic source-tree binding differs")
    evaluation = diagnostic.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("evaluation_head") != "out"
        or evaluation.get("sample_count") != CANONICAL_VAL_COUNT
        or evaluation.get("one_model_forward_per_sample") is not True
        or evaluation.get("prediction_or_target_arrays_written") is not False
    ):
        raise PromotionGateError("baseline diagnostic evaluation contract differs")
    metrics = diagnostic.get("metrics")
    if not isinstance(metrics, Mapping):
        raise PromotionGateError("baseline diagnostic metrics are missing")
    for lower, upper in (("miou", "mIoU"), ("fa", "Fa"), ("pd", "Pd")):
        unit = lower != "fa"
        observed = _require_metric(metrics.get(lower), label=f"diagnostic.{lower}", unit_interval=unit)
        if observed != baseline_selected_record.get(upper):
            raise PromotionGateError(f"baseline diagnostic {lower} differs from selected record")
    mechanism = _validate_mechanism(
        metrics.get("complete_target_mechanism_diagnostics"),
        label="baseline mechanism",
    )
    _require_sha256(metadata["sha256"], label="baseline diagnostic artifact")
    return mechanism


def _assert_inputs_unchanged(observed: Sequence[Mapping[str, str]]) -> None:
    for artifact in observed:
        path = _fixed_path(artifact["relative_path"], must_exist=True)
        if _sha256_file(path) != artifact["sha256"]:
            raise PromotionGateError(
                f"input changed during gate evaluation: {artifact['relative_path']}"
            )


def build_result_payload(
    *,
    baseline_artifacts: Mapping[str, Mapping[str, str]],
    variant_artifacts: Mapping[str, Mapping[str, str]],
    diagnostic_artifact: Mapping[str, str],
    gate_source_artifact: Mapping[str, str],
    baseline_identity: Mapping[str, Any],
    variant_identity: Mapping[str, Any],
    baseline_source_tree: str,
    variant_source_tree: str,
    baseline_selection: Mapping[str, Any],
    variant_selection: Mapping[str, Any],
    baseline_selected: Mapping[str, Any],
    variant_selected: Mapping[str, Any],
    baseline_mechanism: Mapping[str, Any],
    variant_mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_miou = float(baseline_selected["mIoU"])
    variant_miou = float(variant_selected["mIoU"])
    miou_delta_decimal = Decimal(str(variant_miou)) - Decimal(str(baseline_miou))
    miou_delta = float(miou_delta_decimal)
    primary_passed = miou_delta_decimal >= Decimal(str(PRIMARY_MINIMUM_DELTA))

    baseline_pd = float(baseline_selected["Pd"])
    variant_pd = float(variant_selected["Pd"])
    pd_delta_decimal = Decimal(str(variant_pd)) - Decimal(str(baseline_pd))
    pd_delta = float(pd_delta_decimal)
    baseline_fa = float(baseline_selected["Fa"])
    variant_fa = float(variant_selected["Fa"])
    fa_delta_decimal = Decimal(str(variant_fa)) - Decimal(str(baseline_fa))
    fa_delta = float(fa_delta_decimal)
    safety_failed = (
        pd_delta_decimal < Decimal(str(SAFETY_PD_FAILURE_THRESHOLD))
        and fa_delta_decimal > Decimal(str(SAFETY_FA_FAILURE_THRESHOLD))
    )
    safety_passed = not safety_failed

    baseline_recall = float(baseline_mechanism["matched_target_pixel_recall"])
    variant_recall = float(variant_mechanism["matched_target_pixel_recall"])
    recall_delta_decimal = Decimal(str(variant_recall)) - Decimal(str(baseline_recall))
    recall_delta = float(recall_delta_decimal)
    baseline_area = float(baseline_mechanism["matched_component_area_ratio"])
    variant_area = float(variant_mechanism["matched_component_area_ratio"])
    baseline_area_decimal = Decimal(str(baseline_area))
    variant_area_decimal = Decimal(str(variant_area))
    baseline_closeness_decimal = -abs(baseline_area_decimal - Decimal("1.0"))
    variant_closeness_decimal = -abs(variant_area_decimal - Decimal("1.0"))
    closeness_delta_decimal = variant_closeness_decimal - baseline_closeness_decimal
    baseline_closeness = float(baseline_closeness_decimal)
    variant_closeness = float(variant_closeness_decimal)
    closeness_delta = float(closeness_delta_decimal)
    mechanism_passed = (
        recall_delta_decimal > Decimal("0")
        or closeness_delta_decimal > Decimal("0")
    )
    overall_passed = primary_passed and safety_passed and mechanism_passed

    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": DATA_ROLE,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "epochs": EPOCHS,
        "inputs": {
            "paired_baseline": {
                "summary": dict(baseline_artifacts["summary"]),
                "final_checkpoint": dict(baseline_artifacts["checkpoint"]),
                "training_identity_sha256": baseline_identity["identity_sha256"],
                "training_source_tree_sha256": baseline_source_tree,
                "selection_provenance_sha256": _canonical_sha256(baseline_selection),
                "selected_validation_record_sha256": _canonical_sha256(baseline_selected),
                "selected_epoch": baseline_selected["epoch"],
            },
            "complete_target_v1": {
                "summary": dict(variant_artifacts["summary"]),
                "final_checkpoint": dict(variant_artifacts["checkpoint"]),
                "training_identity_sha256": variant_identity["identity_sha256"],
                "training_source_tree_sha256": variant_source_tree,
                "selection_provenance_sha256": _canonical_sha256(variant_selection),
                "selected_validation_record_sha256": _canonical_sha256(variant_selected),
                "selected_epoch": variant_selected["epoch"],
            },
            "paired_baseline_matched_target_diagnostic": dict(diagnostic_artifact),
            "gate_evaluator_source": dict(gate_source_artifact),
            "all_artifact_hashes_verified": True,
            "all_source_hashes_currently_verified": True,
            "canonical_train_val_split_verified": True,
            "both_selections_recomputed": True,
        },
        "primary_gate": {
            "metric": "selected_validation_mIoU",
            "comparison": "variant_minus_paired_baseline_raw",
            "operator": ">=",
            "minimum_delta": PRIMARY_MINIMUM_DELTA,
            "baseline_raw": baseline_miou,
            "variant_raw": variant_miou,
            "delta_raw": miou_delta,
            "passed": primary_passed,
        },
        "safety_gate": {
            "failure_rule": "Pd_delta<-0.003 AND Fa_delta>0",
            "pd": {
                "comparison": "variant_minus_paired_baseline_raw",
                "failure_operator": "<",
                "failure_threshold": SAFETY_PD_FAILURE_THRESHOLD,
                "baseline_raw": baseline_pd,
                "variant_raw": variant_pd,
                "delta_raw": pd_delta,
            },
            "fa": {
                "comparison": "variant_minus_paired_baseline_raw",
                "failure_operator": ">",
                "failure_threshold": SAFETY_FA_FAILURE_THRESHOLD,
                "baseline_raw": baseline_fa,
                "variant_raw": variant_fa,
                "delta_raw": fa_delta,
            },
            "failed": safety_failed,
            "passed": safety_passed,
        },
        "mechanism_gate": {
            "pass_rule": (
                "matched_target_pixel_recall_delta>0 OR "
                "negative_absolute_area_ratio_error_delta>0"
            ),
            "matched_target_pixel_recall": {
                "comparison": "variant_minus_paired_baseline_raw",
                "operator": ">",
                "threshold": 0.0,
                "baseline_raw": baseline_recall,
                "variant_raw": variant_recall,
                "delta_raw": recall_delta,
                "passed": recall_delta_decimal > Decimal("0"),
            },
            "matched_component_area_ratio_closeness_to_one": {
                "source_metric": "matched_component_area_ratio",
                "transform": "-abs(raw_value-1.0)",
                "comparison": "variant_minus_paired_baseline_transformed",
                "operator": ">",
                "threshold": 0.0,
                "baseline_raw": baseline_area,
                "variant_raw": variant_area,
                "baseline_transformed": baseline_closeness,
                "variant_transformed": variant_closeness,
                "delta_transformed": closeness_delta,
                "passed": closeness_delta_decimal > Decimal("0"),
            },
            "passed": mechanism_passed,
        },
        "decision": {
            "overall_passed": overall_passed,
            "result": "PASS" if overall_passed else "FAIL",
            "three_runtime_seed_validation_expansion_allowed": overall_passed,
            "three_runtime_seed_validation_expansion_status": (
                "allowed" if overall_passed else "blocked"
            ),
            "public_test_allowed": False,
            "public_test_status": "blocked_pending_separate_reviewed_gate_extension",
        },
        "test_split_accessed": False,
        "public_test_allowed": False,
    }
    return json.loads(_canonical_json_bytes(payload).decode("ascii"))


def _write_fixed_json_atomic_no_replace(payload: Mapping[str, Any]) -> Path:
    path = _fixed_path(OUTPUT_RELATIVE_PATH, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise PromotionGateError("promotion-gate output directory is a symlink")
    parent = path.parent.resolve(strict=True)
    runs_root = (PROJECT_ROOT.resolve(strict=True) / "runs").resolve(strict=True)
    try:
        parent.relative_to(runs_root)
    except ValueError as exc:
        raise PromotionGateError("promotion-gate output escaped runs/") from exc
    content = json.dumps(
        dict(payload), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(
                    f"promotion-gate result already exists and is immutable: {path}"
                ) from exc
            raise
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def evaluate_payload() -> dict[str, Any]:
    """Read, verify, and recompute the fixed gate without writing a result."""

    _validate_canonical_split_files()
    gate_source_path = _fixed_path(GATE_SOURCE_RELATIVE_PATH, must_exist=True)
    gate_source_meta = {
        "relative_path": GATE_SOURCE_RELATIVE_PATH,
        "sha256": _sha256_file(gate_source_path),
    }
    baseline_summary, baseline_summary_meta = _load_json(
        BASELINE_SUMMARY_RELATIVE_PATH, label="paired baseline summary"
    )
    variant_summary, variant_summary_meta = _load_json(
        VARIANT_SUMMARY_RELATIVE_PATH, label="complete-target-v1 summary"
    )
    diagnostic, diagnostic_meta = _load_json(
        BASELINE_DIAGNOSTIC_RELATIVE_PATH, label="paired baseline diagnostic"
    )
    baseline_checkpoint, baseline_checkpoint_meta = _load_checkpoint(
        BASELINE_CHECKPOINT_RELATIVE_PATH, label="paired baseline final checkpoint"
    )
    variant_checkpoint, variant_checkpoint_meta = _load_checkpoint(
        VARIANT_CHECKPOINT_RELATIVE_PATH, label="complete-target-v1 final checkpoint"
    )

    baseline_selection, baseline_selected = _validate_summary_common(
        baseline_summary, variant=False
    )
    variant_selection, variant_selected = _validate_summary_common(
        variant_summary, variant=True
    )
    baseline_identity, baseline_source_tree = _validate_checkpoint(
        baseline_checkpoint,
        baseline_checkpoint_meta,
        summary=baseline_summary,
        selection=baseline_selection,
        variant=False,
    )
    variant_identity, variant_source_tree = _validate_checkpoint(
        variant_checkpoint,
        variant_checkpoint_meta,
        summary=variant_summary,
        selection=variant_selection,
        variant=True,
    )
    baseline_mechanism = _validate_diagnostic(
        diagnostic,
        diagnostic_meta,
        baseline_summary=baseline_summary,
        baseline_summary_metadata=baseline_summary_meta,
        baseline_checkpoint_metadata=baseline_checkpoint_meta,
        baseline_training_identity_sha256=baseline_identity["identity_sha256"],
        baseline_source_tree=baseline_source_tree,
        baseline_selection=baseline_selection,
        baseline_selected_record=baseline_selected,
    )
    variant_metrics = variant_selected.get("metrics")
    if not isinstance(variant_metrics, Mapping):
        raise PromotionGateError("variant selected mechanism metrics are missing")
    variant_mechanism = _validate_mechanism(
        variant_metrics.get("complete_target_mechanism_diagnostics"),
        label="variant mechanism",
    )

    observed = [
        baseline_summary_meta,
        variant_summary_meta,
        diagnostic_meta,
        baseline_checkpoint_meta,
        variant_checkpoint_meta,
        gate_source_meta,
    ]
    _assert_inputs_unchanged(observed)
    _validate_canonical_split_files()
    if _validate_source_set(
        baseline_identity["determinism_protocol"],
        expected_schema="evisirst_validation_selected_source_set/v2",
        expected_paths=_BASELINE_SOURCE_PATHS,
        label="baseline run identity final check",
    ) != baseline_source_tree:
        raise PromotionGateError("baseline source tree changed during evaluation")
    if _validate_source_set(
        variant_identity["determinism_protocol"],
        expected_schema="evisirst_irstd_complete_target_source_set/v1",
        expected_paths=_VARIANT_SOURCE_PATHS,
        label="variant run identity final check",
    ) != variant_source_tree:
        raise PromotionGateError("variant source tree changed during evaluation")
    diagnostic_sources = diagnostic["sources"]["diagnostic"]
    _validate_source_set(
        {
            "source_set_schema": diagnostic_sources["schema"],
            "source_files": diagnostic_sources["files"],
            "source_tree_sha256": diagnostic_sources["source_tree_sha256"],
        },
        expected_schema="evisirst_paired_baseline_diagnostic_source_set/v1",
        expected_paths=_DIAGNOSTIC_SOURCE_PATHS,
        label="baseline diagnostic final check",
    )
    return build_result_payload(
        baseline_artifacts={
            "summary": baseline_summary_meta,
            "checkpoint": baseline_checkpoint_meta,
        },
        variant_artifacts={
            "summary": variant_summary_meta,
            "checkpoint": variant_checkpoint_meta,
        },
        diagnostic_artifact=diagnostic_meta,
        gate_source_artifact=gate_source_meta,
        baseline_identity=baseline_identity,
        variant_identity=variant_identity,
        baseline_source_tree=baseline_source_tree,
        variant_source_tree=variant_source_tree,
        baseline_selection=baseline_selection,
        variant_selection=variant_selection,
        baseline_selected=baseline_selected,
        variant_selected=variant_selected,
        baseline_mechanism=baseline_mechanism,
        variant_mechanism=variant_mechanism,
    )


def validate_existing_result() -> dict[str, Any]:
    """Strictly validate the fixed immutable result against a fresh evaluation.

    Duplicate keys, non-finite values, unsafe paths, and non-regular files are
    rejected by ``_load_json``.  Canonical-byte equality is deliberately used
    instead of Python mapping equality so JSON number types (for example,
    integer ``1`` versus floating ``1.0``) must also match exactly.
    """

    observed, metadata = _load_json(
        OUTPUT_RELATIVE_PATH, label="existing promotion-gate result"
    )
    expected = evaluate_payload()
    observed_canonical = _canonical_json_bytes(observed)
    expected_canonical = _canonical_json_bytes(expected)
    if observed_canonical != expected_canonical:
        raise PromotionGateError(
            "existing promotion-gate result differs from fresh canonical evaluation"
        )
    path = _fixed_path(OUTPUT_RELATIVE_PATH, must_exist=True)
    if (
        metadata.get("relative_path") != OUTPUT_RELATIVE_PATH
        or _sha256_file(path) != metadata.get("sha256")
    ):
        raise PromotionGateError(
            "existing promotion-gate result changed during validation"
        )
    return json.loads(expected_canonical.decode("ascii"))


def run(_args: argparse.Namespace | None = None) -> Path:
    return _write_fixed_json_atomic_no_replace(evaluate_payload())


def main(argv: Sequence[str] | None = None) -> None:
    output = run(parse_args(argv))
    print(output.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_CHECKPOINT_RELATIVE_PATH",
    "BASELINE_DIAGNOSTIC_RELATIVE_PATH",
    "BASELINE_SUMMARY_RELATIVE_PATH",
    "OUTPUT_RELATIVE_PATH",
    "PromotionGateError",
    "RESULT_SCHEMA",
    "VARIANT_CHECKPOINT_RELATIVE_PATH",
    "VARIANT_SUMMARY_RELATIVE_PATH",
    "build_result_payload",
    "evaluate_payload",
    "main",
    "parse_args",
    "run",
    "validate_existing_result",
]
