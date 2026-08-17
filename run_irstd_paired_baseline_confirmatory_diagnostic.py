#!/usr/bin/env python3
"""Diagnose either preregistered confirmatory IRSTD-1K R1 baseline.

This validation-only command is the two-seed extension of the immutable
single-seed paired-baseline diagnostic.  It accepts no checkpoint, output,
split, dataset, metric, threshold, or selection argument.  Before touching
CUDA it freshly validates the canonical seed-1446202191 promotion result and
requires that result to authorize the three-runtime-seed validation expansion.

For either allowed confirmatory seed it strictly loads the fixed completed R1
summary and final validation-selected checkpoint, recomputes the frozen
selector, verifies the current training/source/split identities and clean
564-key state, and performs one ``out``-head pass over the same canonical V2
validation split.  Public test data is neither imported nor addressable.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import stat
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch.utils.data import DataLoader

import run_irstd_complete_target_promotion_gate as canonical_gate
import run_irstd_paired_baseline_diagnostic as diagnostic_core


PROJECT_ROOT = Path(__file__).resolve().parent
PILOT_RUN_SEED = 1446202191
CONFIRMATORY_RUN_SEEDS = (104728269, 262620274)
DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
WORKERS = 0
LOGICAL_DEVICE = "cuda:0"

DIAGNOSTIC_SCHEMA = (
    "evisirst_irstd_paired_baseline_confirmatory_diagnostic/v1"
)
SOURCE_SET_SCHEMA = (
    "evisirst_irstd_paired_baseline_confirmatory_source_set/v1"
)
IDENTITY_SCHEMA = (
    "evisirst_irstd_paired_baseline_confirmatory_identity/v1"
)
PILOT_AUTHORITY_SCHEMA = (
    "evisirst_irstd_complete_target_confirmatory_authority/v1"
)

_CORE_PATCH_LOCK = threading.Lock()
_GATE_PATCH_LOCK = threading.Lock()
_CORE_PATCH_FIELDS = (
    "RUN_SEED",
    "CHECKPOINT_RELATIVE_PATH",
    "SUMMARY_RELATIVE_PATH",
    "OUTPUT_RELATIVE_PATH",
    "RUN_LOCK_RELATIVE_PATH",
)
_GATE_PATCH_FIELDS = (
    "RUN_SEED",
    "BASELINE_SUMMARY_RELATIVE_PATH",
    "BASELINE_CHECKPOINT_RELATIVE_PATH",
    "BASELINE_DIAGNOSTIC_RELATIVE_PATH",
)


class ConfirmatoryDiagnosticError(
    diagnostic_core.PairedBaselineDiagnosticError
):
    """The fixed confirmatory diagnostic contract was violated."""


def _require_confirmatory_seed(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in CONFIRMATORY_RUN_SEEDS
    ):
        raise ConfirmatoryDiagnosticError(
            "run seed must be one of the two preregistered confirmatory seeds"
        )
    return value


def _relative_paths(run_seed: int) -> dict[str, str]:
    seed = _require_confirmatory_seed(run_seed)
    baseline_root = (
        "runs/validation_selected/formal/IRSTD-1K/binary/"
        f"run_seed_{seed}"
    )
    diagnostic_root = (
        "runs/irstd_performance/complete_target_v1/"
        f"paired_baseline_diagnostics/run_seed_{seed}"
    )
    return {
        "CHECKPOINT_RELATIVE_PATH": f"{baseline_root}/EviSIRST.pth.tar",
        "SUMMARY_RELATIVE_PATH": f"{baseline_root}/summary.json",
        "OUTPUT_RELATIVE_PATH": (
            f"{diagnostic_root}/matched_target_diagnostics.json"
        ),
        "RUN_LOCK_RELATIVE_PATH": f"{diagnostic_root}/run.lock",
    }


def _pilot_core_bindings() -> dict[str, str | int]:
    return {
        "RUN_SEED": PILOT_RUN_SEED,
        "CHECKPOINT_RELATIVE_PATH": (
            "runs/validation_selected/formal/IRSTD-1K/binary/"
            "run_seed_1446202191/EviSIRST.pth.tar"
        ),
        "SUMMARY_RELATIVE_PATH": (
            "runs/validation_selected/formal/IRSTD-1K/binary/"
            "run_seed_1446202191/summary.json"
        ),
        "OUTPUT_RELATIVE_PATH": (
            "runs/irstd_performance/complete_target_v1/"
            "paired_baseline_diagnostics/run_seed_1446202191/"
            "matched_target_diagnostics.json"
        ),
        "RUN_LOCK_RELATIVE_PATH": (
            "runs/irstd_performance/complete_target_v1/"
            "paired_baseline_diagnostics/run_seed_1446202191/run.lock"
        ),
    }


@contextlib.contextmanager
def _bound_diagnostic_core(run_seed: int) -> Iterator[dict[str, str]]:
    """Exception-safely bind the audited core to one fixed confirmatory run."""

    seed = _require_confirmatory_seed(run_seed)
    if not _CORE_PATCH_LOCK.acquire(blocking=False):
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic core is already bound in this process"
        )
    previous = {
        name: getattr(diagnostic_core, name) for name in _CORE_PATCH_FIELDS
    }
    expected = _pilot_core_bindings()
    try:
        if diagnostic_core.PROJECT_ROOT.resolve(
            strict=True
        ) != PROJECT_ROOT.resolve(strict=True):
            raise ConfirmatoryDiagnosticError(
                "single-seed diagnostic core repository root differs"
            )
        for name, value in expected.items():
            if previous[name] != value:
                raise ConfirmatoryDiagnosticError(
                    f"single-seed diagnostic core binding {name} differs"
                )
        paths = _relative_paths(seed)
        diagnostic_core.RUN_SEED = seed
        for name, value in paths.items():
            setattr(diagnostic_core, name, value)
        yield paths
    finally:
        for name, value in previous.items():
            setattr(diagnostic_core, name, value)
        _CORE_PATCH_LOCK.release()


@contextlib.contextmanager
def _bound_baseline_gate_validator(
    run_seed: int, paths: Mapping[str, str]
) -> Iterator[None]:
    """Bind the canonical baseline validator to one confirmatory seed."""

    seed = _require_confirmatory_seed(run_seed)
    expected_paths = _relative_paths(seed)
    if dict(paths) != expected_paths:
        raise ConfirmatoryDiagnosticError("baseline gate artifact paths differ")
    if not _GATE_PATCH_LOCK.acquire(blocking=False):
        raise ConfirmatoryDiagnosticError(
            "canonical baseline validator is already bound in this process"
        )
    previous = {
        name: getattr(canonical_gate, name) for name in _GATE_PATCH_FIELDS
    }
    try:
        if canonical_gate.PROJECT_ROOT.resolve(
            strict=True
        ) != PROJECT_ROOT.resolve(strict=True):
            raise ConfirmatoryDiagnosticError(
                "canonical baseline validator repository root differs"
            )
        canonical_gate.RUN_SEED = seed
        canonical_gate.BASELINE_SUMMARY_RELATIVE_PATH = expected_paths[
            "SUMMARY_RELATIVE_PATH"
        ]
        canonical_gate.BASELINE_CHECKPOINT_RELATIVE_PATH = expected_paths[
            "CHECKPOINT_RELATIVE_PATH"
        ]
        canonical_gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH = expected_paths[
            "OUTPUT_RELATIVE_PATH"
        ]
        yield
    finally:
        for name, value in previous.items():
            setattr(canonical_gate, name, value)
        _GATE_PATCH_LOCK.release()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-seed",
        type=int,
        choices=CONFIRMATORY_RUN_SEEDS,
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--device", choices=(LOGICAL_DEVICE,), default=LOGICAL_DEVICE
    )
    parser.add_argument("--workers", type=int, choices=(WORKERS,), default=WORKERS)
    return parser.parse_args(argv)


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
        raise ConfirmatoryDiagnosticError(
            "confirmatory metadata must be strict finite JSON"
        ) from exc


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_regular_path(relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ConfirmatoryDiagnosticError("fixed artifact path is unsafe")
    repository = PROJECT_ROOT.resolve(strict=True)
    path = repository.joinpath(*pure.parts)
    current = repository
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ConfirmatoryDiagnosticError(
                f"fixed artifact path contains a symlink: {relative_path}"
            )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    if path.resolve(strict=True) != path:
        raise ConfirmatoryDiagnosticError("fixed artifact path was redirected")
    return path


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ConfirmatoryDiagnosticError(f"{label} is not a regular file")
        content = handle.read()
        after = os.fstat(handle.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if not content or identity_before != identity_after:
        raise ConfirmatoryDiagnosticError(f"{label} changed while being read")
    return content


def _require_exact_pass_authority(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ConfirmatoryDiagnosticError(
            "canonical promotion result must be an object"
        )
    decision = payload.get("decision")
    fixed_top = {
        "schema": canonical_gate.RESULT_SCHEMA,
        "status": "complete",
        "run_seed": PILOT_RUN_SEED,
        "test_split_accessed": False,
        "public_test_allowed": False,
    }
    for name, expected in fixed_top.items():
        if type(payload.get(name)) is not type(expected) or payload.get(name) != expected:
            raise ConfirmatoryDiagnosticError(
                f"canonical promotion result {name} differs"
            )
    fixed_decision = {
        "result": "PASS",
        "overall_passed": True,
        "three_runtime_seed_validation_expansion_allowed": True,
        "three_runtime_seed_validation_expansion_status": "allowed",
        "public_test_allowed": False,
    }
    if not isinstance(decision, Mapping):
        raise ConfirmatoryDiagnosticError(
            "canonical promotion decision is missing"
        )
    for name, expected in fixed_decision.items():
        if type(decision.get(name)) is not type(expected) or decision.get(name) != expected:
            raise ConfirmatoryDiagnosticError(
                f"canonical promotion decision {name} does not authorize expansion"
            )
    for gate_name in ("primary_gate", "safety_gate", "mechanism_gate"):
        gate_record = payload.get(gate_name)
        if (
            not isinstance(gate_record, Mapping)
            or gate_record.get("passed") is not True
        ):
            raise ConfirmatoryDiagnosticError(
                f"canonical {gate_name} is not passed"
            )
    return payload


def validate_pilot_authority() -> dict[str, Any]:
    """Freshly re-evaluate and bind the immutable pilot PASS result."""

    if canonical_gate.PROJECT_ROOT.resolve(strict=True) != PROJECT_ROOT.resolve(
        strict=True
    ):
        raise ConfirmatoryDiagnosticError(
            "canonical gate and confirmatory diagnostic repository roots differ"
        )
    try:
        payload = _require_exact_pass_authority(
            canonical_gate.validate_existing_result()
        )
    except (canonical_gate.PromotionGateError, OSError, ValueError) as exc:
        raise ConfirmatoryDiagnosticError(
            "canonical pilot promotion result failed fresh validation"
        ) from exc
    result_path = _fixed_regular_path(canonical_gate.OUTPUT_RELATIVE_PATH)
    content = _read_regular_file_once(
        result_path, label="canonical pilot promotion result"
    )
    try:
        on_disk = json.loads(
            content.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfirmatoryDiagnosticError(
            "canonical pilot promotion result is not strict JSON"
        ) from exc
    if _canonical_json_bytes(on_disk) != _canonical_json_bytes(payload):
        raise ConfirmatoryDiagnosticError(
            "fresh canonical promotion payload differs from its fixed artifact"
        )
    decision = payload["decision"]
    return {
        "schema": PILOT_AUTHORITY_SCHEMA,
        "validator": (
            "run_irstd_complete_target_promotion_gate."
            "validate_existing_result"
        ),
        "result_relative_path": canonical_gate.OUTPUT_RELATIVE_PATH,
        "result_sha256": _sha256_bytes(content),
        "result_schema": payload["schema"],
        "pilot_run_seed": payload["run_seed"],
        "result": decision["result"],
        "overall_passed": decision["overall_passed"],
        "three_runtime_seed_validation_expansion_allowed": decision[
            "three_runtime_seed_validation_expansion_allowed"
        ],
        "public_test_allowed": False,
        "test_split_accessed": False,
    }


def _validate_pilot_authority_record(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "validator",
        "result_relative_path",
        "result_sha256",
        "result_schema",
        "pilot_run_seed",
        "result",
        "overall_passed",
        "three_runtime_seed_validation_expansion_allowed",
        "public_test_allowed",
        "test_split_accessed",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ConfirmatoryDiagnosticError(
            "pilot promotion authority record fields differ"
        )
    fixed = {
        "schema": PILOT_AUTHORITY_SCHEMA,
        "validator": (
            "run_irstd_complete_target_promotion_gate."
            "validate_existing_result"
        ),
        "result_relative_path": canonical_gate.OUTPUT_RELATIVE_PATH,
        "result_schema": canonical_gate.RESULT_SCHEMA,
        "pilot_run_seed": PILOT_RUN_SEED,
        "result": "PASS",
        "overall_passed": True,
        "three_runtime_seed_validation_expansion_allowed": True,
        "public_test_allowed": False,
        "test_split_accessed": False,
    }
    for name, expected in fixed.items():
        if type(value.get(name)) is not type(expected) or value.get(name) != expected:
            raise ConfirmatoryDiagnosticError(
                f"pilot promotion authority {name} differs"
            )
    digest = value.get("result_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ConfirmatoryDiagnosticError(
            "pilot promotion authority result SHA-256 is malformed"
        )
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def confirmatory_source_provenance() -> dict[str, Any]:
    source_paths = {
        "confirmatory_diagnostic_cli": Path(__file__),
        "single_seed_diagnostic_core": Path(diagnostic_core.__file__),
        "canonical_promotion_gate": Path(canonical_gate.__file__),
    }
    repository = PROJECT_ROOT.resolve(strict=True)
    files: dict[str, dict[str, str]] = {}
    for name, raw_path in source_paths.items():
        path = raw_path.resolve(strict=True)
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError as exc:
            raise ConfirmatoryDiagnosticError(
                f"confirmatory source {name} is outside the repository"
            ) from exc
        if raw_path.is_symlink() or not path.is_file():
            raise ConfirmatoryDiagnosticError(
                f"confirmatory source {name} is not a regular file"
            )
        files[name] = {"relative_path": relative, "sha256": _sha256_file(path)}
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _sha256_bytes(_canonical_json_bytes(files)),
    }


def build_result_payload(
    *,
    run_seed: int,
    paths: Mapping[str, str],
    core_payload: Mapping[str, Any],
    core_sources: Mapping[str, Any],
    confirmatory_sources: Mapping[str, Any],
    pilot_authority: Mapping[str, Any],
) -> dict[str, Any]:
    seed = _require_confirmatory_seed(run_seed)
    expected_paths = _relative_paths(seed)
    if dict(paths) != expected_paths:
        raise ConfirmatoryDiagnosticError("confirmatory artifact bindings differ")
    if (
        core_payload.get("schema") != diagnostic_core.DIAGNOSTIC_SCHEMA
        or core_payload.get("diagnostic_only") is not True
        or core_payload.get("selection_allowed") is not False
        or core_payload.get("test_split_accessed") is not False
        or core_payload.get("checkpoint", {}).get("run_seed") != seed
    ):
        raise ConfirmatoryDiagnosticError(
            "single-seed diagnostic core payload differs"
        )
    sources = core_payload.get("sources")
    if (
        not isinstance(sources, Mapping)
        or sources.get("diagnostic") != core_sources
    ):
        raise ConfirmatoryDiagnosticError(
            "single-seed diagnostic source binding differs"
        )
    normalized_authority = _validate_pilot_authority_record(pilot_authority)
    output = dict(core_payload)
    output["schema"] = DIAGNOSTIC_SCHEMA
    output["run_seed"] = seed
    output["promotion_prerequisite"] = normalized_authority
    output_sources = dict(sources)
    output_sources["confirmatory_diagnostic"] = dict(confirmatory_sources)
    output["sources"] = output_sources
    identity = {
        "schema": IDENTITY_SCHEMA,
        "run_seed": seed,
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "data_role": "val",
        "diagnostic_only": True,
        "selection_allowed": False,
        "checkpoint_relative_path": expected_paths[
            "CHECKPOINT_RELATIVE_PATH"
        ],
        "checkpoint_sha256": core_payload["checkpoint"]["sha256"],
        "summary_relative_path": expected_paths["SUMMARY_RELATIVE_PATH"],
        "summary_sha256": core_payload["completed_run_summary"]["sha256"],
        "output_relative_path": expected_paths["OUTPUT_RELATIVE_PATH"],
        "run_lock_relative_path": expected_paths["RUN_LOCK_RELATIVE_PATH"],
        "split_manifest_sha256": core_payload["split"]["manifest_sha256"],
        "data_tree_sha256": core_payload["split"]["data_tree_sha256"],
        "prediction_sha256": core_payload["evaluation"]["prediction_sha256"],
        "target_sha256": core_payload["evaluation"]["target_sha256"],
        "core_source_tree_sha256": core_sources["source_tree_sha256"],
        "confirmatory_source_tree_sha256": confirmatory_sources[
            "source_tree_sha256"
        ],
        "pilot_result_sha256": normalized_authority["result_sha256"],
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = _sha256_bytes(_canonical_json_bytes(identity))
    output["diagnostic_identity"] = identity
    output["diagnostic_only"] = True
    output["selection_allowed"] = False
    output["test_split_accessed"] = False
    return json.loads(_canonical_json_bytes(output).decode("ascii"))


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConfirmatoryDiagnosticError(f"{label} is not a SHA-256 digest")
    return value


def _unwrap_existing_result(
    observed: Mapping[str, Any], *, run_seed: int
) -> dict[str, Any]:
    expected_top = set(canonical_gate._DIAGNOSTIC_KEYS) | {
        "run_seed",
        "promotion_prerequisite",
        "diagnostic_identity",
    }
    if set(observed) != expected_top:
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic top-level fields differ"
        )
    if (
        observed.get("schema") != DIAGNOSTIC_SCHEMA
        or observed.get("run_seed") != run_seed
        or observed.get("diagnostic_only") is not True
        or observed.get("selection_allowed") is not False
        or observed.get("test_split_accessed") is not False
    ):
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic fixed identity differs"
        )
    sources = observed.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "checkpoint_training_source_tree_sha256",
        "checkpoint_training_sources_currently_verified",
        "diagnostic",
        "confirmatory_diagnostic",
    }:
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic source fields differ"
        )
    core = json.loads(_canonical_json_bytes(observed).decode("ascii"))
    del core["run_seed"]
    del core["promotion_prerequisite"]
    del core["diagnostic_identity"]
    core["schema"] = diagnostic_core.DIAGNOSTIC_SCHEMA
    core["sources"] = {
        name: value
        for name, value in core["sources"].items()
        if name != "confirmatory_diagnostic"
    }
    return core


def _validate_core_nested_contract(
    core: Mapping[str, Any],
    *,
    run_seed: int,
    paths: Mapping[str, str],
    selected_record: Mapping[str, Any],
) -> None:
    expected_nested_keys = {
        "checkpoint": {
            "relative_path",
            "sha256",
            "schema",
            "checkpoint_role",
            "selected_epoch",
            "architecture_seed",
            "run_seed",
            "target_mode",
            "training_identity_sha256",
            "selection_provenance_sha256",
            "strict_564_key_load",
        },
        "completed_run_summary": {
            "relative_path",
            "sha256",
            "selected_validation_record_sha256",
        },
        "split": {
            "manifest_relative_path",
            "manifest_sha256",
            "val_index_relative_path",
            "val_index_sha256",
            "data_tree_sha256",
            "data_tree_verified",
            "train_count",
            "val_count",
        },
        "evaluation": {
            "metrics_contract",
            "evaluation_head",
            "probability_threshold",
            "probability_threshold_operator",
            "target_threshold",
            "target_threshold_operator",
            "match_radius",
            "match_distance_operator",
            "connected_component_connectivity",
            "connected_component_neighborhood",
            "assignment_algorithm",
            "tiny_area",
            "tiny_area_operator",
            "sample_count",
            "one_model_forward_per_sample",
            "prediction_digest_schema",
            "prediction_sha256",
            "target_digest_schema",
            "target_sha256",
            "prediction_or_target_arrays_written",
        },
    }
    for name, expected_keys in expected_nested_keys.items():
        value = core.get(name)
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ConfirmatoryDiagnosticError(
                f"confirmatory diagnostic {name} fields differ"
            )
    checkpoint_fixed = {
        "relative_path": paths["CHECKPOINT_RELATIVE_PATH"],
        "schema": diagnostic_core.r1.CHECKPOINT_SCHEMA,
        "checkpoint_role": "validation_selected",
        "architecture_seed": diagnostic_core.ARCHITECTURE_SEED,
        "run_seed": run_seed,
        "target_mode": diagnostic_core.TARGET_MODE,
        "strict_564_key_load": True,
    }
    for name, expected in checkpoint_fixed.items():
        if type(core["checkpoint"].get(name)) is not type(
            expected
        ) or core["checkpoint"].get(name) != expected:
            raise ConfirmatoryDiagnosticError(
                f"confirmatory diagnostic checkpoint.{name} differs"
            )
    if core["completed_run_summary"].get("relative_path") != paths[
        "SUMMARY_RELATIVE_PATH"
    ]:
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic summary path differs"
        )
    for label, value in (
        ("checkpoint", core["checkpoint"].get("sha256")),
        ("training identity", core["checkpoint"].get("training_identity_sha256")),
        ("selection provenance", core["checkpoint"].get("selection_provenance_sha256")),
        ("summary", core["completed_run_summary"].get("sha256")),
        (
            "selected validation record",
            core["completed_run_summary"].get(
                "selected_validation_record_sha256"
            ),
        ),
    ):
        _require_sha256(value, label=f"{label} digest")
    evaluation = core["evaluation"]
    fixed_evaluation = {
        "metrics_contract": diagnostic_core.r1.EVALUATION_PROTOCOL_VERSION,
        "evaluation_head": "out",
        "probability_threshold": diagnostic_core.r1.PROBABILITY_THRESHOLD,
        "probability_threshold_operator": (
            diagnostic_core.r1.PREDICTION_THRESHOLD_OPERATOR
        ),
        "target_threshold": diagnostic_core.r1.TARGET_THRESHOLD,
        "target_threshold_operator": (
            diagnostic_core.r1.TARGET_THRESHOLD_OPERATOR
        ),
        "match_radius": diagnostic_core.r1.MATCH_RADIUS,
        "match_distance_operator": diagnostic_core.r1.MATCH_DISTANCE_OPERATOR,
        "connected_component_connectivity": (
            diagnostic_core.r1.CONNECTED_COMPONENT_CONNECTIVITY
        ),
        "connected_component_neighborhood": (
            diagnostic_core.r1.CONNECTED_COMPONENT_NEIGHBORHOOD
        ),
        "assignment_algorithm": diagnostic_core.r1.ASSIGNMENT_ALGORITHM,
        "tiny_area": diagnostic_core.r1.TINY_AREA,
        "tiny_area_operator": diagnostic_core.r1.TINY_AREA_OPERATOR,
        "sample_count": canonical_gate.CANONICAL_VAL_COUNT,
        "one_model_forward_per_sample": True,
        "prediction_digest_schema": diagnostic_core.PREDICTION_DIGEST_SCHEMA,
        "target_digest_schema": diagnostic_core.TARGET_DIGEST_SCHEMA,
        "prediction_or_target_arrays_written": False,
    }
    for name, expected in fixed_evaluation.items():
        if type(evaluation.get(name)) is not type(expected) or evaluation.get(
            name
        ) != expected:
            raise ConfirmatoryDiagnosticError(
                f"confirmatory diagnostic evaluation.{name} differs"
            )
    _require_sha256(
        evaluation.get("prediction_sha256"), label="prediction digest"
    )
    _require_sha256(evaluation.get("target_sha256"), label="target digest")

    metrics = core.get("metrics")
    selected_metrics = selected_record.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(
        selected_metrics, Mapping
    ):
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic metrics are missing"
        )
    mechanism_key = "complete_target_mechanism_diagnostics"
    if set(metrics) != set(selected_metrics) | {mechanism_key}:
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic metric fields differ"
        )
    descriptive_metrics = {
        name: value for name, value in metrics.items() if name != mechanism_key
    }
    if _canonical_json_bytes(descriptive_metrics) != _canonical_json_bytes(
        selected_metrics
    ):
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic R1 metrics differ from the selected record"
        )
    canonical_gate._validate_mechanism(
        metrics.get(mechanism_key), label="confirmatory baseline mechanism"
    )


def _validate_diagnostic_identity(
    value: Any,
    *,
    run_seed: int,
    paths: Mapping[str, str],
    core: Mapping[str, Any],
    core_sources: Mapping[str, Any],
    confirmatory_sources: Mapping[str, Any],
    pilot_authority: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "run_seed",
        "dataset",
        "target_mode",
        "data_role",
        "diagnostic_only",
        "selection_allowed",
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "summary_relative_path",
        "summary_sha256",
        "output_relative_path",
        "run_lock_relative_path",
        "split_manifest_sha256",
        "data_tree_sha256",
        "prediction_sha256",
        "target_sha256",
        "core_source_tree_sha256",
        "confirmatory_source_tree_sha256",
        "pilot_result_sha256",
        "test_split_accessed",
        "identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic identity fields differ"
        )
    identity = dict(value)
    observed_sha = identity.pop("identity_sha256", None)
    if observed_sha != _sha256_bytes(_canonical_json_bytes(identity)):
        raise ConfirmatoryDiagnosticError(
            "confirmatory diagnostic identity SHA-256 differs"
        )
    fixed = {
        "schema": IDENTITY_SCHEMA,
        "run_seed": run_seed,
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "data_role": "val",
        "diagnostic_only": True,
        "selection_allowed": False,
        "checkpoint_relative_path": paths["CHECKPOINT_RELATIVE_PATH"],
        "checkpoint_sha256": core["checkpoint"]["sha256"],
        "summary_relative_path": paths["SUMMARY_RELATIVE_PATH"],
        "summary_sha256": core["completed_run_summary"]["sha256"],
        "output_relative_path": paths["OUTPUT_RELATIVE_PATH"],
        "run_lock_relative_path": paths["RUN_LOCK_RELATIVE_PATH"],
        "split_manifest_sha256": core["split"]["manifest_sha256"],
        "data_tree_sha256": core["split"]["data_tree_sha256"],
        "prediction_sha256": core["evaluation"]["prediction_sha256"],
        "target_sha256": core["evaluation"]["target_sha256"],
        "core_source_tree_sha256": core_sources["source_tree_sha256"],
        "confirmatory_source_tree_sha256": confirmatory_sources[
            "source_tree_sha256"
        ],
        "pilot_result_sha256": pilot_authority["result_sha256"],
        "test_split_accessed": False,
    }
    for name, expected in fixed.items():
        if type(identity.get(name)) is not type(expected) or identity.get(
            name
        ) != expected:
            raise ConfirmatoryDiagnosticError(
                f"confirmatory diagnostic identity.{name} differs"
            )


def validate_existing_result(run_seed: int) -> dict[str, Any]:
    """Strictly validate one immutable result without repeating GPU inference."""

    seed = _require_confirmatory_seed(run_seed)
    with _bound_diagnostic_core(seed) as paths:
        with _bound_baseline_gate_validator(seed, paths):
            try:
                canonical_gate._validate_canonical_split_files()
                observed, diagnostic_metadata = canonical_gate._load_json(
                    paths["OUTPUT_RELATIVE_PATH"],
                    label=f"seed {seed} confirmatory baseline diagnostic",
                )
                summary, summary_metadata = canonical_gate._load_json(
                    paths["SUMMARY_RELATIVE_PATH"],
                    label=f"seed {seed} paired baseline summary",
                )
                checkpoint, checkpoint_metadata = canonical_gate._load_checkpoint(
                    paths["CHECKPOINT_RELATIVE_PATH"],
                    label=f"seed {seed} paired baseline final checkpoint",
                )
                selection, selected = canonical_gate._validate_summary_common(
                    summary, variant=False
                )
                identity, source_tree = canonical_gate._validate_checkpoint(
                    checkpoint,
                    checkpoint_metadata,
                    summary=summary,
                    selection=selection,
                    variant=False,
                )
                # The core loader additionally binds every key, tensor shape,
                # dtype, and finite value to the live clean 564-key graph.
                _model, strict_metadata = (
                    diagnostic_core.load_fixed_final_model(torch.device("cpu"))
                )
                if (
                    strict_metadata.get("sha256")
                    != checkpoint_metadata["sha256"]
                    or strict_metadata.get("selected_epoch")
                    != summary.get("selected_epoch")
                    or strict_metadata.get("training_identity_sha256")
                    != identity["identity_sha256"]
                ):
                    raise ConfirmatoryDiagnosticError(
                        "strict 564-key checkpoint metadata differs"
                    )

                core = _unwrap_existing_result(observed, run_seed=seed)
                _validate_core_nested_contract(
                    core,
                    run_seed=seed,
                    paths=paths,
                    selected_record=selected,
                )
                explicit_bindings = {
                    "checkpoint SHA-256": (
                        core["checkpoint"].get("sha256"),
                        checkpoint_metadata["sha256"],
                    ),
                    "training identity SHA-256": (
                        core["checkpoint"].get("training_identity_sha256"),
                        identity["identity_sha256"],
                    ),
                    "selection provenance SHA-256": (
                        core["checkpoint"].get("selection_provenance_sha256"),
                        canonical_gate._canonical_sha256(selection),
                    ),
                    "summary SHA-256": (
                        core["completed_run_summary"].get("sha256"),
                        summary_metadata["sha256"],
                    ),
                    "selected record SHA-256": (
                        core["completed_run_summary"].get(
                            "selected_validation_record_sha256"
                        ),
                        canonical_gate._canonical_sha256(selected),
                    ),
                    "training source tree SHA-256": (
                        core["sources"].get(
                            "checkpoint_training_source_tree_sha256"
                        ),
                        source_tree,
                    ),
                }
                for label, (observed_value, expected_value) in (
                    explicit_bindings.items()
                ):
                    if observed_value != expected_value:
                        raise ConfirmatoryDiagnosticError(
                            f"confirmatory diagnostic {label} binding differs"
                        )
                canonical_gate._validate_diagnostic(
                    core,
                    diagnostic_metadata,
                    baseline_summary=summary,
                    baseline_summary_metadata=summary_metadata,
                    baseline_checkpoint_metadata=checkpoint_metadata,
                    baseline_training_identity_sha256=identity[
                        "identity_sha256"
                    ],
                    baseline_source_tree=source_tree,
                    baseline_selection=selection,
                    baseline_selected_record=selected,
                )

                core_sources = diagnostic_core.diagnostic_source_provenance()
                confirmatory_sources = confirmatory_source_provenance()
                pilot_authority = validate_pilot_authority()
                if core.get("sources", {}).get("diagnostic") != core_sources:
                    raise ConfirmatoryDiagnosticError(
                        "confirmatory diagnostic core source hashes differ"
                    )
                observed_sources = observed.get("sources")
                if (
                    not isinstance(observed_sources, Mapping)
                    or observed_sources.get("confirmatory_diagnostic")
                    != confirmatory_sources
                    or observed.get("promotion_prerequisite")
                    != pilot_authority
                ):
                    raise ConfirmatoryDiagnosticError(
                        "confirmatory source or pilot authority binding differs"
                    )
                _validate_diagnostic_identity(
                    observed.get("diagnostic_identity"),
                    run_seed=seed,
                    paths=paths,
                    core=core,
                    core_sources=core_sources,
                    confirmatory_sources=confirmatory_sources,
                    pilot_authority=pilot_authority,
                )
                expected = build_result_payload(
                    run_seed=seed,
                    paths=paths,
                    core_payload=core,
                    core_sources=core_sources,
                    confirmatory_sources=confirmatory_sources,
                    pilot_authority=pilot_authority,
                )
                if _canonical_json_bytes(observed) != _canonical_json_bytes(
                    expected
                ):
                    raise ConfirmatoryDiagnosticError(
                        "existing confirmatory result differs from strict reconstruction"
                    )

                canonical_gate._assert_inputs_unchanged(
                    (
                        summary_metadata,
                        checkpoint_metadata,
                        diagnostic_metadata,
                    )
                )
                canonical_gate._validate_canonical_split_files()
                if (
                    canonical_gate._validate_source_set(
                        identity["determinism_protocol"],
                        expected_schema=(
                            "evisirst_validation_selected_source_set/v2"
                        ),
                        expected_paths=canonical_gate._BASELINE_SOURCE_PATHS,
                        label="confirmatory baseline final source check",
                    )
                    != source_tree
                    or diagnostic_core.diagnostic_source_provenance()
                    != core_sources
                    or confirmatory_source_provenance()
                    != confirmatory_sources
                    or validate_pilot_authority() != pilot_authority
                ):
                    raise ConfirmatoryDiagnosticError(
                        "confirmatory inputs changed during validation"
                    )
                return json.loads(
                    _canonical_json_bytes(observed).decode("ascii")
                )
            except ConfirmatoryDiagnosticError:
                raise
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                raise ConfirmatoryDiagnosticError(
                    f"seed {seed} confirmatory diagnostic failed strict validation"
                ) from exc


def _run_locked(
    args: argparse.Namespace,
    *,
    paths: Mapping[str, str],
    pilot_authority: Mapping[str, Any],
) -> Path:
    output_path = diagnostic_core._run_artifact_path(
        paths["OUTPUT_RELATIVE_PATH"]
    )
    if os.path.lexists(output_path):
        raise FileExistsError(
            "confirmatory paired baseline diagnostic already exists and is "
            f"immutable: {output_path}"
        )
    diagnostic_core.validate_optional_inherited_gpu_lock()
    diagnostic_core.configure_inference_determinism()
    device = diagnostic_core.require_device(args.device)
    core_sources = diagnostic_core.diagnostic_source_provenance()
    confirmatory_sources = confirmatory_source_provenance()
    model, checkpoint_metadata = diagnostic_core.load_fixed_final_model(device)
    dataset = diagnostic_core.build_validation_dataset(
        dataset_root=args.dataset_root,
        checkpoint_metadata=checkpoint_metadata,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
    )
    r1_metrics, mechanism_metrics, digests, sample_count = (
        diagnostic_core.evaluate_out_once(model, loader, device)
    )
    checkpoint_path = diagnostic_core._fixed_path(
        paths["CHECKPOINT_RELATIVE_PATH"], must_exist=True
    )
    summary_path = diagnostic_core._fixed_path(
        paths["SUMMARY_RELATIVE_PATH"], must_exist=True
    )
    if (
        _sha256_file(checkpoint_path) != checkpoint_metadata["sha256"]
        or _sha256_file(summary_path)
        != checkpoint_metadata["summary"]["sha256"]
        or diagnostic_core.r1._protocol_source_provenance()[
            "source_tree_sha256"
        ]
        != checkpoint_metadata["training_source_tree_sha256"]
        or diagnostic_core.diagnostic_source_provenance() != core_sources
        or confirmatory_source_provenance() != confirmatory_sources
        or validate_pilot_authority() != pilot_authority
    ):
        raise ConfirmatoryDiagnosticError(
            "checkpoint, summary, source, or pilot authority changed during "
            "confirmatory evaluation"
        )
    core_payload = diagnostic_core.build_result_payload(
        checkpoint_metadata=checkpoint_metadata,
        dataset=dataset,
        r1_metrics=r1_metrics,
        mechanism_metrics=mechanism_metrics,
        digests=digests,
        sample_count=sample_count,
        diagnostic_sources=core_sources,
    )
    payload = build_result_payload(
        run_seed=args.run_seed,
        paths=paths,
        core_payload=core_payload,
        core_sources=core_sources,
        confirmatory_sources=confirmatory_sources,
        pilot_authority=pilot_authority,
    )
    return diagnostic_core._write_fixed_json_atomic(payload)


def run(args: argparse.Namespace) -> Path:
    seed = _require_confirmatory_seed(args.run_seed)
    if args.device != LOGICAL_DEVICE or args.workers != WORKERS:
        raise ConfirmatoryDiagnosticError(
            "formal confirmatory diagnostic requires logical cuda:0 and workers=0"
        )
    # This is intentionally before run-directory creation, lock acquisition,
    # deterministic CUDA setup, device construction, or checkpoint loading.
    pilot_authority = validate_pilot_authority()
    with _bound_diagnostic_core(seed) as paths:
        with diagnostic_core.exclusive_run_lock():
            return _run_locked(
                args,
                paths=paths,
                pilot_authority=pilot_authority,
            )


def main(argv: Sequence[str] | None = None) -> None:
    output = run(parse_args(argv))
    print(output.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIRMATORY_RUN_SEEDS",
    "ConfirmatoryDiagnosticError",
    "DIAGNOSTIC_SCHEMA",
    "LOGICAL_DEVICE",
    "PILOT_RUN_SEED",
    "SOURCE_SET_SCHEMA",
    "build_result_payload",
    "confirmatory_source_provenance",
    "main",
    "parse_args",
    "run",
    "validate_existing_result",
    "validate_pilot_authority",
]
