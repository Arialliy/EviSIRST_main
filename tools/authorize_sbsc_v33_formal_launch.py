#!/usr/bin/env python3
"""Issue the write-once C3-SBSC V3.3 formal-launch authorization.

This tool performs no training or evaluation.  It independently replays the
frozen method config, unit-suite report, train-only canary V2, paired resource
benchmark, authority files, source manifests, and their common CUDA runtime.
Any missing, stale, incomplete, or merely status-only evidence raises before
the authorization path is written.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ENTRY = Path(__file__)
if ENTRY.is_symlink() or not ENTRY.is_file():
    raise RuntimeError("formal authorizer entry point must be a regular file")
PROJECT_ROOT = ENTRY.resolve(strict=True).parents[1]
if ENTRY.resolve(strict=True) != PROJECT_ROOT / "tools" / ENTRY.name:
    raise RuntimeError("formal authorizer resolves away from tools")
sys.path[:] = [
    str(PROJECT_ROOT),
    *[item for item in sys.path if item not in ("", str(PROJECT_ROOT))],
]

from experiments import sbsc_v33_contracts as contracts
from tools import benchmark_sbsc_v33_resources_v2 as resource
from tools import freeze_sbsc_v33_irstd_formal_method_config as freezer
from tools import run_sbsc_v33_canary_v2 as canary_v2
from tools import run_sbsc_v33_frozen_unit_suite as unit_suite


AUTHORIZATION_SCHEMA = "sctransnet_sbsc_v33/formal_launch_authorization/v1"
ENVIRONMENT_SCHEMA = "sctransnet_sbsc_v33/formal_environment_identity/v1"
AUTHORIZATION_PATH = (
    PROJECT_ROOT / "experiments" / "sbsc_v33_formal_launch_authorization.json"
)
METHOD_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sbsc_v33_methods"
    / "sbsc_v33_third_irstd_formal.json"
)
UNIT_REPORT_PATH = unit_suite.REPORT_PATH
CANARY_REPORT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "sbsc_v33_third"
    / "canary_v2"
    / "IRSTD-1K"
    / "report.json"
)
CANARY_MANIFEST_PATH = CANARY_REPORT_PATH.with_name("manifest.json")
CANARY_RULES_PATH = canary_v2.CANARY_RULES_PATH
RESOURCE_REPORT_PATH = resource.REPORT_PATH
RESOURCE_RULES_PATH = resource.RULES_PATH
EXPECTED_RESOURCE_V2_REPORT_SHA256 = (
    "ec6b04abcc61454a0bbc46bd84167b0bf6770d4ad7ed4397018a4f551e39a6b1"
)

AUTHORIZED_RUN = {
    "method": "sbsc_v33_third",
    "dataset": "IRSTD-1K",
    "run_kind": "formal",
    "output_namespace": "runs/sbsc_v33_third/formal/IRSTD-1K",
}
REPLAY_CHECKS = {
    "method_config_exact_rebuild": True,
    "unit_report_exact_replay": True,
    "canary_v2_exact_replay": True,
    "paired_resource_exact_replay": True,
    "baseline_authority_exact_replay": True,
    "gradient_authorization_exact_replay": True,
    "source_manifests_exact_replay": True,
    "environment_identity_exact_replay": True,
    "evidence_unchanged_before_write": True,
}


class FormalLaunchAuthorizationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise FormalLaunchAuthorizationError(message)


def _lower_sha(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} is not a lowercase SHA-256")
    return value


def _bound_json(path: Path, *, name: str) -> tuple[dict[str, Any], dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} is unavailable")
    relative = contracts.repository_relative_path(path)
    digest = contracts.sha256_file(path)
    return contracts.load_strict_json(path), {"path": relative, "sha256": digest}


def _validate_frozen_resource_v2_reference(
    reference: Mapping[str, Any],
) -> dict[str, str]:
    expected = {
        "path": "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json",
        "sha256": EXPECTED_RESOURCE_V2_REPORT_SHA256,
    }
    if dict(reference) != expected:
        _fail("paired resource V2 physical report SHA/path differs")
    return expected


def _validate_source_manifest(
    value: Any,
    *,
    name: str,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "files", "sha256"}
        or value.get("schema") != "sctransnet_sbsc_v33/source_manifest/v1"
        or not isinstance(value.get("files"), list)
        or not value["files"]
    ):
        _fail(f"{name} source manifest is malformed")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, record in enumerate(value["files"]):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            _fail(f"{name} source record {position} differs")
        relative = record.get("path")
        if type(relative) is not str or relative in seen:
            _fail(f"{name} source path set differs")
        seen.add(relative)
        physical = contracts.require_repository_relative_regular_file(relative)
        digest = _lower_sha(record.get("sha256"), name=f"{name} source SHA")
        size = record.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or contracts.sha256_file(physical) != digest
            or physical.stat().st_size != size
        ):
            _fail(f"{name} source record drifted")
        records.append(dict(record))
    aggregate = _lower_sha(value.get("sha256"), name=f"{name} source aggregate")
    if contracts.canonical_sha256(records) != aggregate:
        _fail(f"{name} source aggregate differs")
    normalized = {"schema": value["schema"], "files": records, "sha256": aggregate}
    if expected is not None and normalized != expected:
        _fail(f"{name} source manifest is not the current manifest")
    return normalized


def validate_method_config(
    config: Mapping[str, Any], *, expected_config: Mapping[str, Any]
) -> dict[str, Any]:
    if dict(config) != dict(expected_config):
        _fail("formal method config is not the exact current frozen rebuild")
    if (
        config.get("schema") != "sctransnet_sbsc_v33/method_config/v1"
        or config.get("status") != "frozen"
        or config.get("write_once") is not True
        or config.get("method") != AUTHORIZED_RUN["method"]
        or config.get("dataset") != AUTHORIZED_RUN["dataset"]
        or config.get("run_kind") != "formal"
        or config.get("output_namespace") != AUTHORIZED_RUN["output_namespace"]
        or config.get("architecture_seed") != 42
        or config.get("run_seed") != 42
        or config.get("router_value_gradient_mode") != "live"
        or config.get("epochs") != 1000
        or config.get("batch_size") != 16
        or config.get("selection_begin_epoch") != 500
        or config.get("selection_end_epoch") != 1000
        or config.get("selection_every") != 1
    ):
        _fail("formal method config identity differs")
    source = _validate_source_manifest(
        config.get("training_source_manifest"),
        name="formal training",
        expected=expected_config.get("training_source_manifest"),
    )
    if config.get("training_source_manifest_sha256") != source["sha256"]:
        _fail("formal training source binding differs")
    return {"source_manifest": source}


def validate_unit_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    try:
        unit_suite.validate_unit_report(report)
    except Exception as exc:
        raise FormalLaunchAuthorizationError(
            f"unit-suite report replay failed: {exc}"
        ) from exc
    if report.get("status") != "PASS":
        _fail("unit-suite report is not PASS")
    source = _validate_source_manifest(
        report.get("source_manifest"), name="unit suite"
    )
    return {
        "source_manifest": source,
        "runtime_identity": dict(report["runtime_identity"]),
        "runtime_identity_sha256": report["runtime_identity_sha256"],
    }


def validate_canary_v2_evidence(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    gradient: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_rules, frozen_method_config_sha = canary_v2.load_frozen_rules()
    frozen_rules_sha = contracts.sha256_file(CANARY_RULES_PATH)
    if (
        dict(rules) != frozen_rules
        or frozen_rules_sha != canary_v2.EXPECTED_RULES_SHA256
        or frozen_method_config_sha != canary_v2.EXPECTED_METHOD_CONFIG_SHA256
    ):
        _fail("canary V2 rules are not the frozen rules")
    if (
        report.get("schema") != canary_v2.REPORT_SCHEMA
        or report.get("status") != "PASS"
        or report.get("hard_gate_pass") is not True
        or report.get("write_once") is not True
        or report.get("dataset") != "IRSTD-1K"
        or report.get("data_role") != "train"
        or report.get("method") != "sbsc_v33_third"
        or report.get("architecture_seed") != 42
        or report.get("run_seed") != 42
        or report.get("router_value_gradient_mode") != "live"
    ):
        _fail("canary V2 PASS identity differs")
    expected_progress = {
        "current_epoch": 20,
        "completed_epochs": 20,
        "started_batches": 1000,
        "completed_batches": 1000,
        "batch_count": 1000,
        "combined_backward_calls": 1000,
        "processed_samples": 16000,
        "optimizer_steps": 1000,
    }
    if report.get("execution_progress") != expected_progress or any(
        report.get(key) != value for key, value in expected_progress.items()
    ):
        _fail("canary V2 20-epoch sample/step ledger differs")
    boundaries = {
        "validation_dataset_constructed": False,
        "validation_loader_constructed": False,
        "test_dataset_constructed": False,
        "test_loader_constructed": False,
        "performance_evaluation": False,
        "model_selection": None,
        "checkpoint_written": False,
        "formal_weights_reusable": False,
    }
    if any(report.get(key) is not value for key, value in boundaries.items()):
        _fail("canary V2 train-only/no-artifact boundary differs")
    epochs = report.get("epoch_records")
    if not isinstance(epochs, list):
        _fail("canary V2 epoch records are missing")
    recomputed_gate = canary_v2.evaluate_canary_gate(
        epochs, rules, execution_complete=True
    )
    if (
        report.get("hard_gate") != recomputed_gate
        or recomputed_gate.get("verdict") != "GO"
        or recomputed_gate.get("reasons") != []
        or set(recomputed_gate.get("checks", {})) != set(canary_v2.HARD_GATE_ORDER)
        or not all(recomputed_gate["checks"].values())
    ):
        _fail("canary V2 hard gate does not independently replay to GO")
    if (
        report.get("rules_path")
        != contracts.repository_relative_path(CANARY_RULES_PATH)
        or report.get("rules_sha256") != frozen_rules_sha
        or report.get("manifest_path")
        != contracts.repository_relative_path(CANARY_MANIFEST_PATH)
        or report.get("manifest_sha256")
        != contracts.sha256_file(CANARY_MANIFEST_PATH)
        or report.get("baseline_authority_sha256") != baseline["manifest_sha256"]
        or report.get("gradient_authorization_sha256")
        != gradient["authorization_sha256"]
    ):
        _fail("canary V2 path/authority binding differs")
    if (
        manifest.get("schema") != canary_v2.MANIFEST_SCHEMA
        or manifest.get("status") != "prepared"
        or manifest.get("write_once") is not True
        or manifest.get("rules_path") != report.get("rules_path")
        or manifest.get("rules_sha256") != report.get("rules_sha256")
        or manifest.get("method_config_sha256")
        != report.get("method_config_sha256")
        or manifest.get("expected_execution")
        != {
            "samples_per_epoch": 800,
            "optimizer_steps_per_epoch": 50,
            "epochs": 20,
            "total_processed_samples": 16000,
            "total_optimizer_steps": 1000,
            "combined_backward_calls_per_step": 1,
        }
    ):
        _fail("canary V2 manifest identity differs")
    artifact_boundaries = manifest.get("artifact_boundaries")
    if not isinstance(artifact_boundaries, Mapping) or any(
        artifact_boundaries.get(key) is not value for key, value in boundaries.items()
    ):
        _fail("canary V2 manifest boundary differs")
    data = manifest.get("data_identity")
    if (
        not isinstance(data, Mapping)
        or data.get("dataset") != "IRSTD-1K"
        or data.get("data_role") != "train"
        or data.get("sample_count") != 800
        or data.get("index_relpath")
        != "IRSTD-1K/img_idx/train_IRSTD-1K.txt"
    ):
        _fail("canary V2 train split identity differs")
    expected_source = canary_v2.build_source_manifest()
    source = _validate_source_manifest(
        manifest.get("training_source_manifest"),
        name="canary V2",
        expected=expected_source,
    )
    if report.get("training_source_manifest_sha256") != source["sha256"]:
        _fail("canary V2 source binding differs")
    authority = manifest.get("authority_bindings")
    if (
        not isinstance(authority, Mapping)
        or not isinstance(authority.get("baseline"), Mapping)
        or authority["baseline"].get("path") != baseline["manifest_path"]
        or authority["baseline"].get("sha256") != baseline["manifest_sha256"]
        or not isinstance(authority.get("gradient"), Mapping)
        or authority["gradient"].get("path") != gradient["authorization_path"]
        or authority["gradient"].get("sha256")
        != gradient["authorization_sha256"]
        or authority["gradient"].get("authorized_router_value_gradient_mode")
        != "live"
    ):
        _fail("canary V2 authority chain differs")
    runtime = manifest.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        _fail("canary V2 runtime identity is missing")
    return {
        "source_manifest": source,
        "runtime_identity": dict(runtime),
        "runtime_identity_sha256": contracts.canonical_sha256(runtime),
    }


def validate_resource_evidence(
    report: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    expected_evidence: Mapping[str, Any],
    expected_source_manifest: Mapping[str, Any],
    batch_plan: Sequence[Sequence[int]],
    input_hashes: Sequence[str],
) -> dict[str, Any]:
    frozen_rules = resource.load_frozen_rules()
    if dict(rules) != frozen_rules:
        _fail("paired resource V2 rules are not the frozen rules")
    source = _validate_source_manifest(
        report.get("training_source_manifest"),
        name="paired resource V2",
        expected=expected_source_manifest,
    )
    try:
        resource.validate_report(
            report,
            rules=rules,
            evidence=expected_evidence,
            source_manifest=expected_source_manifest,
            batch_plan=batch_plan,
            input_hashes=input_hashes,
        )
    except Exception as exc:
        raise FormalLaunchAuthorizationError(
            f"paired resource V2 replay failed: {exc}"
        ) from exc
    hard_gate = report.get("hard_gate")
    if (
        report.get("schema") != resource.REPORT_SCHEMA
        or report.get("status") != "PASS"
        or report.get("hard_gate_pass") is not True
        or report.get("write_once") is not True
        or report.get("rules_path")
        != contracts.repository_relative_path(RESOURCE_RULES_PATH)
        or report.get("rules_sha256") != resource.EXPECTED_RULES_SHA256
        or report.get("training_source_manifest_sha256") != source["sha256"]
        or report.get("superseded_v1_evidence") != resource.V1_EVIDENCE
        or not isinstance(hard_gate, Mapping)
        or hard_gate.get("verdict") != "GO"
        or hard_gate.get("reasons") != []
        or not all(hard_gate.get("checks", {}).values())
        or hard_gate.get("speed_or_ratio_used_as_gate") is not False
    ):
        _fail("paired resource V2 hard gate is not strict replay GO")
    elapsed = report.get("elapsed_seconds")
    runtime = report.get("runtime_identity")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
        or not isinstance(runtime, Mapping)
        or runtime.get("precision") != "FP32"
        or runtime.get("autocast") is not False
        or runtime.get("gradient_scaler") is not False
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("tf32_matmul") is not False
        or runtime.get("tf32_cudnn") is not False
    ):
        _fail("paired resource runtime identity differs")
    return {
        "source_manifest": source,
        "runtime_identity": dict(runtime),
        "runtime_identity_sha256": contracts.canonical_sha256(runtime),
    }


def build_environment_identity(
    *,
    unit: Mapping[str, Any],
    canary: Mapping[str, Any],
    resource_record: Mapping[str, Any],
) -> dict[str, Any]:
    unit_runtime = unit.get("runtime_identity")
    canary_runtime = canary.get("runtime_identity")
    benchmark_runtime = resource_record.get("runtime_identity")
    if not all(
        isinstance(value, Mapping)
        for value in (unit_runtime, canary_runtime, benchmark_runtime)
    ):
        _fail("runtime identities are incomplete")
    common = {
        "python": canary_runtime.get("python"),
        "torch": canary_runtime.get("torch"),
        "cuda_runtime": canary_runtime.get("cuda_runtime"),
        "cublas_workspace_config": canary_runtime.get(
            "cublas_workspace_config"
        ),
        "device_type": "cuda",
    }
    for runtime, label in (
        (unit_runtime, "unit"),
        (benchmark_runtime, "paired resource"),
    ):
        for key in ("python", "torch", "cuda_runtime", "cublas_workspace_config"):
            if runtime.get(key) != common[key]:
                _fail(f"{label}/canary environment field {key!r} differs")
    if (
        unit_runtime.get("device_type") != "cuda"
        or unit_runtime.get("cuda_available") is not True
        or canary_runtime.get("device_type") != "cuda"
        or canary_runtime.get("device_requested") != "cuda:0"
        or benchmark_runtime.get("device_requested") != "cuda:0"
        or benchmark_runtime.get("device") != "cuda:0"
        or benchmark_runtime.get("deterministic_algorithms") is not True
    ):
        _fail("CUDA runtime selection differs across preflight evidence")
    devices = unit_runtime.get("cuda_devices")
    if not isinstance(devices, list) or not devices:
        _fail("unit-suite CUDA device inventory is missing")
    unit_device = next(
        (
            item
            for item in devices
            if isinstance(item, Mapping) and item.get("index") == 0
        ),
        None,
    )
    if (
        not isinstance(unit_device, Mapping)
        or unit_device.get("name") != benchmark_runtime.get("device_name")
        or unit_device.get("total_memory_bytes")
        != benchmark_runtime.get("device_total_memory_bytes")
    ):
        _fail("unit/resource CUDA device identity differs")
    identity = {
        "schema": ENVIRONMENT_SCHEMA,
        "common": common,
        "unit_runtime_identity_sha256": unit["runtime_identity_sha256"],
        "canary_runtime_identity_sha256": canary["runtime_identity_sha256"],
        "paired_resource_runtime_identity_sha256": resource_record[
            "runtime_identity_sha256"
        ],
        "cuda_device": {
            "logical_index": 0,
            "name": unit_device["name"],
            "total_memory_bytes": unit_device["total_memory_bytes"],
            "unit_resource_exact_match": True,
        },
    }
    for record, runtime in (
        (unit, unit_runtime),
        (canary, canary_runtime),
        (resource_record, benchmark_runtime),
    ):
        if record.get("runtime_identity_sha256") != contracts.canonical_sha256(
            runtime
        ):
            _fail("embedded runtime identity digest differs")
    return identity


def validate_environment_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "common",
        "unit_runtime_identity_sha256",
        "canary_runtime_identity_sha256",
        "paired_resource_runtime_identity_sha256",
        "cuda_device",
    }:
        _fail("formal environment identity field set differs")
    common = value.get("common")
    device = value.get("cuda_device")
    if (
        value.get("schema") != ENVIRONMENT_SCHEMA
        or not isinstance(common, Mapping)
        or set(common)
        != {
            "python",
            "torch",
            "cuda_runtime",
            "cublas_workspace_config",
            "device_type",
        }
        or any(
            type(common.get(key)) is not str or not common[key]
            for key in (
                "python",
                "torch",
                "cuda_runtime",
                "cublas_workspace_config",
            )
        )
        or common.get("device_type") != "cuda"
        or common.get("cublas_workspace_config") != ":4096:8"
        or not isinstance(device, Mapping)
        or set(device)
        != {
            "logical_index",
            "name",
            "total_memory_bytes",
            "unit_resource_exact_match",
        }
        or device.get("logical_index") != 0
        or type(device.get("name")) is not str
        or not device["name"]
        or type(device.get("total_memory_bytes")) is not int
        or device["total_memory_bytes"] <= 0
        or device.get("unit_resource_exact_match") is not True
    ):
        _fail("formal environment identity value differs")
    for field in (
        "unit_runtime_identity_sha256",
        "canary_runtime_identity_sha256",
        "paired_resource_runtime_identity_sha256",
    ):
        _lower_sha(value.get(field), name=field)
    return dict(value)


def build_authorization(
    *,
    method_config_reference: Mapping[str, str],
    unit_reference: Mapping[str, str],
    canary_reference: Mapping[str, str],
    resource_reference: Mapping[str, str],
    baseline: Mapping[str, Any],
    gradient: Mapping[str, Any],
    formal_source_sha256: str,
    unit_source_sha256: str,
    canary_source_sha256: str,
    resource_source_sha256: str,
    environment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    for name, reference in (
        ("method config", method_config_reference),
        ("unit", unit_reference),
        ("canary", canary_reference),
        ("resource", resource_reference),
    ):
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            _fail(f"{name} reference field set differs")
        _lower_sha(reference.get("sha256"), name=f"{name} reference SHA")
    expected_paths = {
        "method config": "experiments/sbsc_v33_methods/sbsc_v33_third_irstd_formal.json",
        "unit": "artifacts/sbsc_v33_preflight/unit_test_report.json",
        "canary": "runs/sbsc_v33_third/canary_v2/IRSTD-1K/report.json",
        "resource": "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json",
    }
    for name, reference in (
        ("method config", method_config_reference),
        ("unit", unit_reference),
        ("canary", canary_reference),
        ("resource", resource_reference),
    ):
        if reference["path"] != expected_paths[name]:
            _fail(f"{name} reference path differs")
    source_bindings = {
        "formal_training": _lower_sha(
            formal_source_sha256, name="formal source SHA"
        ),
        "unit_suite": _lower_sha(unit_source_sha256, name="unit source SHA"),
        "canary_v2": _lower_sha(canary_source_sha256, name="canary source SHA"),
        "paired_resource_benchmark": _lower_sha(
            resource_source_sha256, name="resource source SHA"
        ),
    }
    environment = validate_environment_identity(environment_identity)
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "PASS",
        "write_once": True,
        "authorized_run": dict(AUTHORIZED_RUN),
        "method_config_path": method_config_reference["path"],
        "method_config_sha256": method_config_reference["sha256"],
        "gradient_authorization_sha256": gradient["authorization_sha256"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "training_source_manifest_sha256": source_bindings["formal_training"],
        "evidence": {
            "unit_tests": dict(unit_reference),
            "canary": dict(canary_reference),
            "paired_resource_benchmark": dict(resource_reference),
        },
        "environment_identity": environment,
        "environment_identity_sha256": contracts.canonical_sha256(environment),
        "source_bindings": source_bindings,
        "replay_checks": dict(REPLAY_CHECKS),
    }


def validate_authorization(
    value: Mapping[str, Any],
    *,
    expected_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    exact_fields = {
        "schema",
        "status",
        "write_once",
        "authorized_run",
        "method_config_path",
        "method_config_sha256",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_sha256",
        "training_source_manifest_sha256",
        "evidence",
        "environment_identity",
        "environment_identity_sha256",
        "source_bindings",
        "replay_checks",
    }
    if not isinstance(value, Mapping) or set(value) != exact_fields:
        _fail("formal authorization exact 14-field schema differs")
    if (
        value.get("schema") != AUTHORIZATION_SCHEMA
        or value.get("status") != "PASS"
        or value.get("write_once") is not True
        or value.get("authorized_run") != AUTHORIZED_RUN
        or value.get("method_config_path")
        != "experiments/sbsc_v33_methods/sbsc_v33_third_irstd_formal.json"
    ):
        _fail("formal authorization core identity differs")
    for field in (
        "method_config_sha256",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_sha256",
        "training_source_manifest_sha256",
        "environment_identity_sha256",
    ):
        _lower_sha(value.get(field), name=field)
    evidence = value.get("evidence")
    expected_evidence_paths = {
        "unit_tests": "artifacts/sbsc_v33_preflight/unit_test_report.json",
        "canary": "runs/sbsc_v33_third/canary_v2/IRSTD-1K/report.json",
        "paired_resource_benchmark": (
            "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json"
        ),
    }
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        expected_evidence_paths
    ):
        _fail("formal authorization evidence set differs")
    for name, expected_path in expected_evidence_paths.items():
        reference = evidence.get(name)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or reference.get("path") != expected_path
        ):
            _fail(f"formal authorization evidence {name!r} differs")
        _lower_sha(reference.get("sha256"), name=f"{name} evidence SHA")
    _validate_frozen_resource_v2_reference(
        evidence["paired_resource_benchmark"]
    )
    environment = validate_environment_identity(value.get("environment_identity"))
    if value.get("environment_identity_sha256") != contracts.canonical_sha256(
        environment
    ):
        _fail("formal authorization environment digest differs")
    source_bindings = value.get("source_bindings")
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != {
        "formal_training",
        "unit_suite",
        "canary_v2",
        "paired_resource_benchmark",
    }:
        _fail("formal authorization source binding set differs")
    for name, digest in source_bindings.items():
        _lower_sha(digest, name=f"{name} source binding")
    if (
        source_bindings["formal_training"]
        != value.get("training_source_manifest_sha256")
        or value.get("replay_checks") != REPLAY_CHECKS
    ):
        _fail("formal authorization source/replay binding differs")
    if expected_authorization is not None and dict(value) != dict(
        expected_authorization
    ):
        _fail("formal authorization differs from freshly rebuilt evidence")
    return dict(value)


def _replay_formal_preflight() -> dict[str, Any]:
    """Freshly replay all physical evidence and rebuild the only valid auth."""

    # Check the immutable resource evidence anchor before any other dependency;
    # it is loaded and checked again where its full semantics are replayed.
    _initial_resource_report, initial_resource_reference = _bound_json(
        RESOURCE_REPORT_PATH, name="paired resource V2 report"
    )
    _validate_frozen_resource_v2_reference(initial_resource_reference)

    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    if gradient.get("authorized_router_value_gradient_mode") != "live":
        _fail("gradient authorization is not live")

    method_config, method_reference = _bound_json(
        METHOD_CONFIG_PATH, name="formal method config"
    )
    expected_config = freezer.build_config(PROJECT_ROOT / "datasets")
    formal = validate_method_config(method_config, expected_config=expected_config)
    if (
        method_config.get("gradient_authorization_sha256")
        != gradient["authorization_sha256"]
        or method_config.get("baseline_authority_manifest_sha256")
        != baseline["manifest_sha256"]
    ):
        _fail("formal method config authority binding differs")

    unit_report, unit_reference = _bound_json(
        UNIT_REPORT_PATH, name="frozen unit-suite report"
    )
    unit = validate_unit_evidence(unit_report)

    canary_report, canary_reference = _bound_json(
        CANARY_REPORT_PATH, name="canary V2 report"
    )
    canary_manifest, _canary_manifest_reference = _bound_json(
        CANARY_MANIFEST_PATH, name="canary V2 manifest"
    )
    canary_rules, _canary_rules_reference = _bound_json(
        CANARY_RULES_PATH, name="canary V2 rules"
    )
    if (
        canary_reference["sha256"]
        != resource.v1.EXPECTED_CANARY_REPORT_SHA256
        or contracts.sha256_file(CANARY_MANIFEST_PATH)
        != resource.v1.EXPECTED_CANARY_MANIFEST_SHA256
    ):
        _fail("canary V2 physical report/manifest SHA differs")
    canary = validate_canary_v2_evidence(
        canary_report,
        canary_manifest,
        canary_rules,
        baseline=baseline,
        gradient=gradient,
    )

    resource_rules, _resource_rules_reference = _bound_json(
        RESOURCE_RULES_PATH, name="paired resource V2 rules"
    )
    expected_resource_evidence = resource.load_bound_evidence(resource_rules)
    resource_plan, _resource_batches, resource_input_hashes = (
        resource._materialized_snapshot()
    )
    resource_report, resource_reference = _bound_json(
        RESOURCE_REPORT_PATH, name="paired resource V2 report"
    )
    resource_reference = _validate_frozen_resource_v2_reference(
        resource_reference
    )
    resource_record = validate_resource_evidence(
        resource_report,
        resource_rules,
        expected_evidence=expected_resource_evidence,
        expected_source_manifest=resource.build_source_manifest(),
        batch_plan=resource_plan,
        input_hashes=resource_input_hashes,
    )

    environment = build_environment_identity(
        unit=unit, canary=canary, resource_record=resource_record
    )
    authorization = build_authorization(
        method_config_reference=method_reference,
        unit_reference=unit_reference,
        canary_reference=canary_reference,
        resource_reference=resource_reference,
        baseline=baseline,
        gradient=gradient,
        formal_source_sha256=formal["source_manifest"]["sha256"],
        unit_source_sha256=unit["source_manifest"]["sha256"],
        canary_source_sha256=canary["source_manifest"]["sha256"],
        resource_source_sha256=resource_record["source_manifest"]["sha256"],
        environment_identity=environment,
    )
    return validate_authorization(
        authorization, expected_authorization=authorization
    )


def authorize_formal_launch(
    *, output_path: Path = AUTHORIZATION_PATH
) -> dict[str, Any]:
    destination = contracts.validated_output_path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("formal-launch authorization already exists")

    authorization = _replay_formal_preflight()
    validate_authorization(
        authorization, expected_authorization=authorization
    )
    expected_bytes = contracts.canonical_json_bytes(authorization)
    expected_sha256 = contracts.write_once_json(destination, authorization)
    if (
        contracts.sha256_file(destination) != expected_sha256
        or destination.read_bytes() != expected_bytes
    ):
        _fail("formal authorization physical bytes differ after publication")
    reopened = contracts.load_strict_json(destination)
    if reopened != authorization:
        _fail("formal authorization strict reload differs from memory")

    # Rebuild from freshly reloaded physical evidence after publication.  A
    # zero exit is impossible if any dependency changed across the write.
    fresh_expected = _replay_formal_preflight()
    validated = validate_authorization(
        reopened, expected_authorization=fresh_expected
    )
    if validated != authorization or fresh_expected != authorization:
        _fail("formal authorization post-write replay differs")
    return validated


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("formal authorizer accepts no arguments")
    authorization = authorize_formal_launch()
    print(contracts.canonical_json_bytes(authorization).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
