#!/usr/bin/env python3
"""Run the strict-replay V2 paired V3.2/V3.3 resource benchmark.

V1's measurement primitives are retained byte-for-byte.  V2 adds three
fail-closed evidence properties: every active gradient (including gain) must
be present on the final frozen step; all authority/data/source snapshots are
replayed after the CUDA window; and the write-once JSON is strictly reloaded
and independently validated before a zero exit status is possible.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ENTRY = Path(__file__)
if ENTRY.is_symlink() or not ENTRY.is_file():
    raise RuntimeError("resource V2 entry point must be a regular file")
PROJECT_ROOT = ENTRY.resolve(strict=True).parents[1]
if ENTRY.resolve(strict=True) != PROJECT_ROOT / "tools" / ENTRY.name:
    raise RuntimeError("resource V2 entry point resolves away from tools")
sys.path[:] = [
    str(PROJECT_ROOT),
    *[entry for entry in sys.path if entry not in ("", str(PROJECT_ROOT))],
]

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from experiments import sbsc_v33_contracts as contracts
from tools import benchmark_sbsc_v33_resources as v1


RULES_PATH = PROJECT_ROOT / "experiments" / "sbsc_v33_resource_benchmark_rules_v2.json"
REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "sbsc_v33_preflight"
    / "paired_resource_benchmark_v2.json"
)
RULES_SCHEMA = "sctransnet_sbsc_v33/resource_benchmark_rules/v2"
REPORT_SCHEMA = "sctransnet_sbsc_v33/resource_benchmark_report/v2"
MODEL_RECORD_SCHEMA = "sctransnet_sbsc_v33/resource_model_measurement/v2"
EXPECTED_RULES_SHA256 = "6b99407a22bd95029414b43b65bf6cf20dcca530e5c907fa83e2ddbd9540fa8c"

V1_EVIDENCE = {
    "reason": (
        "v1 lacked an exact post-write report replay validator and allowed "
        "conditional final gain-gradient absence"
    ),
    "tool_path": "tools/benchmark_sbsc_v33_resources.py",
    "tool_sha256": "e664c028d9a2a3aef10c3aafcfd2627cbad56918a74901f3ff93d7dc799aae54",
    "rules_path": "experiments/sbsc_v33_resource_benchmark_rules.json",
    "rules_sha256": "110f26c67afc3b873943c62210082a285362be33d29328707c7aeacb40a3622c",
    "tests_path": "tests/test_sbsc_v33_resource_benchmark.py",
    "tests_sha256": "87b8d2f80393493dcdb8e4c94157db33e45dc95dcf9f71acb72cf2c3192ae95e",
    "report_path": "artifacts/sbsc_v33_preflight/paired_resource_benchmark.json",
    "report_sha256": "09ba93e16865bd884c97a1d235856b781c9df6c20ac395c7a61e6ed5bdbe08c0",
    "report_schema": "sctransnet_sbsc_v33/resource_benchmark_report/v1",
    "required_status": "PASS",
    "required_hard_gate_pass": True,
    "required_verdict": "GO",
}
HARD_GATE_ORDER = v1.HARD_GATE_ORDER
MODEL_ORDER = v1.MODEL_ORDER
ARTIFACT_BOUNDARIES = dict(v1.ARTIFACT_BOUNDARIES)


class ResourceBenchmarkV2Error(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise ResourceBenchmarkV2Error(message)


def _sha(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        _fail(f"{name} must be finite and valid")
    return result


def _expected_inactive_names() -> list[str]:
    return sorted(
        {
            *(f"mtc.embeddings_{index}.position_embeddings" for index in range(1, 5)),
            *(
                f"mtc.encoder.layer.{layer}.channel_attn.q{query}_attn{attention}"
                for layer in range(4)
                for query in range(1, 5)
                for attention in range(1, 5)
            ),
        }
    )


def _expected_model_specs() -> list[dict[str, Any]]:
    return [
        {**dict(spec), "active_parameter_tensor_count": 367}
        for spec in v1._model_specs()
    ]


def _verify_v1_evidence(value: Any) -> dict[str, Any]:
    if value != V1_EVIDENCE:
        _fail("superseded V1 evidence record differs")
    for kind in ("tool", "rules", "tests", "report"):
        path = contracts.require_repository_relative_regular_file(value[f"{kind}_path"])
        if contracts.sha256_file(path) != value[f"{kind}_sha256"]:
            _fail(f"superseded V1 {kind} bytes changed")
    report = contracts.load_strict_json(
        contracts.require_repository_relative_regular_file(value["report_path"])
    )
    if (
        report.get("schema") != value["report_schema"]
        or report.get("status") != value["required_status"]
        or report.get("hard_gate_pass") is not value["required_hard_gate_pass"]
        or not isinstance(report.get("hard_gate"), Mapping)
        or report["hard_gate"].get("verdict") != value["required_verdict"]
    ):
        _fail("superseded V1 report identity differs")
    return dict(value)


def load_frozen_rules() -> dict[str, Any]:
    if RULES_PATH.is_symlink() or not RULES_PATH.is_file():
        _fail("resource V2 rules are unavailable")
    if contracts.sha256_file(RULES_PATH) != EXPECTED_RULES_SHA256:
        _fail("resource V2 rules physical SHA-256 drifted")
    rules = contracts.load_strict_json(RULES_PATH)
    expected_fields = {
        "schema",
        "status",
        "scope",
        "write_once",
        "superseded_v1_evidence",
        "identity",
        "models",
        "optimizer",
        "measurement",
        "authority_bindings",
        "source_manifest",
        "hard_gate_order",
        "hard_gates",
        "warning_diagnostics",
        "report_replay",
        "artifact_boundaries",
    }
    legacy_rules = v1.load_frozen_rules()
    if (
        set(rules) != expected_fields
        or rules.get("schema") != RULES_SCHEMA
        or rules.get("status") != "frozen"
        or rules.get("write_once") is not True
        or rules.get("scope")
        != "paired_irstd_1k_train_only_fp32_training_step_resources_strict_replay"
        or rules.get("identity") != legacy_rules["identity"]
        or rules.get("models") != _expected_model_specs()
        or rules.get("optimizer") != legacy_rules["optimizer"]
        or rules.get("authority_bindings") != legacy_rules["authority_bindings"]
        or tuple(rules.get("hard_gate_order", ())) != HARD_GATE_ORDER
        or rules.get("artifact_boundaries")
        != {
            **ARTIFACT_BOUNDARIES,
            "report_path": (
                "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json"
            ),
        }
    ):
        _fail("resource V2 frozen rule identity differs")
    measurement = rules.get("measurement")
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("model_order") != list(MODEL_ORDER)
        or measurement.get("batch_plan_sha256") != v1.EXPECTED_BATCH_PLAN_SHA256
        or measurement.get("warmup_steps_per_model") != v1.WARMUP_STEPS
        or measurement.get("measured_steps_per_model") != v1.MEASURED_STEPS
        or measurement.get("total_steps_per_model") != v1.TOTAL_STEPS
        or measurement.get("same_materialized_cpu_batches_for_both_models") is not True
        or measurement.get("final_step_all_active_gradients_present_and_finite")
        is not True
        or measurement.get("conditional_final_gain_gradient_absence_allowed")
        is not False
    ):
        _fail("resource V2 measurement identity differs")
    replay = rules.get("report_replay")
    if not isinstance(replay, Mapping) or not all(value is True for value in replay.values()):
        _fail("resource V2 replay contract differs")
    warning = rules.get("warning_diagnostics")
    if (
        not isinstance(warning, Mapping)
        or warning.get("affects_verdict") is not False
        or warning.get("ratio_thresholds") is not None
        or warning.get("v33_faster_than_v32_required") is not False
    ):
        _fail("resource V2 warning diagnostics became a hard gate")
    _verify_v1_evidence(rules.get("superseded_v1_evidence"))
    return rules


def _source_paths() -> tuple[str, ...]:
    paths = {
        *v1._source_paths(),
        "tools/benchmark_sbsc_v33_resources_v2.py",
        "tools/__init__.py",
        "experiments/sbsc_v33_resource_benchmark_rules_v2.json",
        "tests/test_sbsc_v33_resource_benchmark.py",
        "artifacts/sbsc_v33_preflight/paired_resource_benchmark.json",
    }
    return tuple(sorted(paths))


def build_source_manifest() -> dict[str, Any]:
    return contracts.build_source_manifest(_source_paths())


def load_bound_evidence(rules: Mapping[str, Any]) -> dict[str, Any]:
    legacy_rules = v1.load_frozen_rules()
    evidence = v1.load_bound_evidence(legacy_rules)
    if rules.get("authority_bindings") != legacy_rules.get("authority_bindings"):
        _fail("resource V2 authority bindings differ from V1 frozen inputs")
    return {
        **evidence,
        "superseded_v1": _verify_v1_evidence(rules.get("superseded_v1_evidence")),
    }


def _input_hashes(batches: Sequence[Mapping[str, Any]]) -> list[str]:
    values = [batch.get("input_pair_sha256") for batch in batches]
    if len(values) != v1.TOTAL_STEPS:
        _fail("resource V2 materialized batch count differs")
    return [_sha(value, name="materialized input pair") for value in values]


def _materialized_snapshot() -> tuple[list[list[int]], list[dict[str, Any]], list[str]]:
    plan = v1.expected_batch_plan()
    dataset = v1._canonical_dataset()
    batches = v1.materialize_training_batches(dataset, plan)
    return plan, batches, _input_hashes(batches)


def _model_record_checks(
    records: Any,
    *,
    top_input_hashes: Sequence[str],
    device_total_memory_bytes: int,
) -> dict[str, bool]:
    if not isinstance(records, Mapping) or set(records) != set(MODEL_ORDER):
        return {name: False for name in HARD_GATE_ORDER}
    exact_keys = {
        "schema",
        "method",
        "model",
        "model_schema",
        "loss_schema",
        "identity_valid",
        "shared_state_sha256",
        "initial_state_sha256",
        "builder_metadata_sha256",
        "router_value_gradient_mode",
        "balance_mode",
        "batch_plan_sha256",
        "input_pair_sha256_by_step",
        "step_records",
        "warmup_step_count",
        "measured_step_count",
        "measured_sample_count",
        "measured_elapsed_seconds",
        "throughput_samples_per_second",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
        "device_total_memory_bytes",
        "finite",
        "training_state_audit",
        "oom_count",
        "complete",
        "state_key_count",
        "parameter_count",
        "inactive_parameter_names",
        "active_parameter_count",
    }
    expected_inactive = _expected_inactive_names()
    identity = True
    same_input = True
    finite = True
    complete = True
    no_oom = True
    peak = True
    shared_hashes: list[Any] = []
    expected_specs = {spec["method"]: spec for spec in _expected_model_specs()}
    for method in MODEL_ORDER:
        record = records.get(method)
        spec = expected_specs[method]
        if not isinstance(record, Mapping) or set(record) != exact_keys:
            return {name: False for name in HARD_GATE_ORDER}
        shared_hashes.append(record.get("shared_state_sha256"))
        identity = identity and (
            record.get("schema") == MODEL_RECORD_SCHEMA
            and record.get("identity_valid") is True
            and record.get("method") == method
            and record.get("model") == spec["model"]
            and record.get("model_schema") == spec["model_schema"]
            and record.get("loss_schema") == spec["loss_schema"]
            and record.get("router_value_gradient_mode")
            == spec["router_value_gradient_mode"]
            and record.get("balance_mode") == spec["balance_mode"]
            and record.get("state_key_count") == spec["state_key_count"]
            and record.get("parameter_count") == spec["parameter_count"]
            and record.get("active_parameter_count")
            == spec["active_parameter_tensor_count"]
            and record.get("inactive_parameter_names") == expected_inactive
            and record.get("batch_plan_sha256") == v1.EXPECTED_BATCH_PLAN_SHA256
        )
        try:
            for key in (
                "shared_state_sha256",
                "initial_state_sha256",
                "builder_metadata_sha256",
            ):
                _sha(record.get(key), name=f"{method}.{key}")
        except ResourceBenchmarkV2Error:
            identity = False
        record_hashes = record.get("input_pair_sha256_by_step")
        steps = record.get("step_records")
        same_input = same_input and (
            isinstance(record_hashes, list)
            and record_hashes == list(top_input_hashes)
            and isinstance(steps, list)
            and len(steps) == v1.TOTAL_STEPS
        )
        measured_elapsed = 0.0
        if isinstance(steps, list):
            for position, step in enumerate(steps):
                phase = "warmup" if position < v1.WARMUP_STEPS else "measured"
                if not isinstance(step, Mapping) or set(step) != {
                    "position",
                    "phase",
                    "batch_size",
                    "input_pair_sha256",
                    "wall_elapsed_seconds",
                    "loss",
                    "finite",
                }:
                    same_input = finite = complete = False
                    continue
                same_input = same_input and (
                    step.get("position") == position
                    and step.get("phase") == phase
                    and step.get("batch_size") == v1.BATCH_SIZE
                    and step.get("input_pair_sha256") == top_input_hashes[position]
                )
                try:
                    elapsed = _finite(
                        step.get("wall_elapsed_seconds"),
                        name=f"{method} step time",
                        positive=True,
                    )
                    losses = step.get("loss")
                    if not isinstance(losses, Mapping) or set(losses) != {
                        "total",
                        "segmentation",
                        "router",
                    }:
                        raise ResourceBenchmarkV2Error("step loss fields differ")
                    values = {
                        key: _finite(value, name=f"{method}.{key}")
                        for key, value in losses.items()
                    }
                    if any(value < 0.0 for value in values.values()) or not math.isclose(
                        values["total"],
                        values["segmentation"] + values["router"],
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-6,
                    ):
                        raise ResourceBenchmarkV2Error("step loss arithmetic differs")
                    if step.get("finite") is not True:
                        raise ResourceBenchmarkV2Error("step self finite flag differs")
                    if phase == "measured":
                        measured_elapsed += elapsed
                except ResourceBenchmarkV2Error:
                    finite = False
        else:
            finite = complete = same_input = False
        audit = record.get("training_state_audit")
        expected_gain_name = (
            v1.core_v32.SBSC_V32_GAIN_STATE_KEY
            if method == "sbsc_v32"
            else v1.core_v33.SBSC_V33_GAIN_STATE_KEY
        )
        audit_ok = (
            isinstance(audit, Mapping)
            and set(audit)
            == {
                "pass",
                "active_parameter_count",
                "missing_unexpected",
                "missing_conditionally_allowed",
                "conditional_missing_gradient_parameter",
                "nonfinite_gradients",
                "nonfinite_parameters",
                "model_state_finite",
                "optimizer_state_finite",
            }
            and audit.get("pass") is True
            and audit.get("active_parameter_count")
            == spec["active_parameter_tensor_count"]
            and audit.get("missing_unexpected") == []
            and audit.get("missing_conditionally_allowed") == []
            and audit.get("conditional_missing_gradient_parameter")
            == expected_gain_name
            and audit.get("nonfinite_gradients") == []
            and audit.get("nonfinite_parameters") == []
            and audit.get("model_state_finite") is True
            and audit.get("optimizer_state_finite") is True
        )
        try:
            observed_measured = _finite(
                record.get("measured_elapsed_seconds"),
                name=f"{method} measured elapsed",
                positive=True,
            )
            throughput = _finite(
                record.get("throughput_samples_per_second"),
                name=f"{method} throughput",
                positive=True,
            )
            timing_ok = math.isclose(
                observed_measured, measured_elapsed, rel_tol=1.0e-12, abs_tol=1.0e-12
            ) and math.isclose(
                throughput,
                v1.MEASURED_SAMPLES / observed_measured,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        except ResourceBenchmarkV2Error:
            timing_ok = False
        finite = finite and audit_ok and timing_ok and record.get("finite") is True
        complete = complete and (
            record.get("complete") is True
            and record.get("warmup_step_count") == v1.WARMUP_STEPS
            and record.get("measured_step_count") == v1.MEASURED_STEPS
            and record.get("measured_sample_count") == v1.MEASURED_SAMPLES
            and isinstance(steps, list)
            and len(steps) == v1.TOTAL_STEPS
        )
        no_oom = no_oom and record.get("oom_count") == 0
        peak = peak and (
            type(record.get("device_total_memory_bytes")) is int
            and record.get("device_total_memory_bytes") == device_total_memory_bytes
            and type(record.get("max_memory_allocated_bytes")) is int
            and type(record.get("max_memory_reserved_bytes")) is int
            and 0
            <= record["max_memory_allocated_bytes"]
            <= device_total_memory_bytes
            and 0
            <= record["max_memory_reserved_bytes"]
            <= device_total_memory_bytes
        )
    identity = identity and len(set(shared_hashes)) == 1 and shared_hashes[0] is not None
    return {
        "identity": bool(identity),
        "same_input": bool(same_input),
        "finite": bool(finite),
        "complete_measurement": bool(complete),
        "no_oom": bool(no_oom),
        "peak_within_device": bool(peak),
    }


def evaluate_hard_gate(
    records: Any,
    *,
    top_input_hashes: Sequence[str],
    device_total_memory_bytes: int,
    global_identity_valid: bool,
) -> dict[str, Any]:
    checks = _model_record_checks(
        records,
        top_input_hashes=top_input_hashes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    checks["identity"] = checks["identity"] and global_identity_valid
    reasons = [name for name in HARD_GATE_ORDER if not checks[name]]
    return {
        "schema": REPORT_SCHEMA + "/hard_gate/v2",
        "hard_gate_order": list(HARD_GATE_ORDER),
        "checks": checks,
        "verdict": "GO" if not reasons else "NO-GO",
        "reasons": reasons,
        "speed_or_ratio_used_as_gate": False,
    }


def _v2_records(records: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in MODEL_ORDER:
        record = dict(records[method])
        record["schema"] = MODEL_RECORD_SCHEMA
        result[method] = record
    return result


def _authority_record(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": {
            "path": evidence["baseline"]["manifest_path"],
            "sha256": evidence["baseline"]["manifest_sha256"],
        },
        "gradient": {
            "path": evidence["gradient"]["authorization_path"],
            "sha256": evidence["gradient"]["authorization_sha256"],
            "authorized_router_value_gradient_mode": evidence["gradient"][
                "authorized_router_value_gradient_mode"
            ],
        },
        "v2_canary": dict(evidence["v2_canary"]),
    }


def build_report(
    *,
    rules: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    batch_plan: Sequence[Sequence[int]],
    batches: Sequence[Mapping[str, Any]],
    model_records: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    records = _v2_records(model_records)
    input_hashes = _input_hashes(batches)
    total_memory = int(runtime_identity["device_total_memory_bytes"])
    gate = evaluate_hard_gate(
        records,
        top_input_hashes=input_hashes,
        device_total_memory_bytes=total_memory,
        global_identity_valid=True,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if gate["verdict"] == "GO" else "FAIL",
        "hard_gate_pass": gate["verdict"] == "GO",
        "write_once": True,
        "scope": rules["scope"],
        "rules_path": contracts.repository_relative_path(RULES_PATH),
        "rules_sha256": EXPECTED_RULES_SHA256,
        "superseded_v1_evidence": dict(evidence["superseded_v1"]),
        "authority_bindings": _authority_record(evidence),
        "training_source_manifest": dict(source_manifest),
        "training_source_manifest_sha256": source_manifest["sha256"],
        "data_identity": {
            "dataset": v1.DATASET,
            "data_role": v1.DATA_ROLE,
            "sample_count": v1.SAMPLE_COUNT,
            "index_relpath": v1.source_protocol.EXPECTED_SPLITS[v1.DATASET]["train"][
                "index_relpath"
            ],
            "index_file_sha256": v1.source_protocol.EXPECTED_SPLITS[v1.DATASET]["train"][
                "file_sha256"
            ],
            "ordered_ids_sha256": v1.source_protocol.EXPECTED_SPLITS[v1.DATASET]["train"][
                "ordered_ids_sha256"
            ],
            "normalization": dict(v1.source_protocol.LEGACY_NORMALIZATION[v1.DATASET]),
            "dataset_epoch": v1.DATASET_EPOCH,
            "batch_size": v1.BATCH_SIZE,
            "patch_size": v1.PATCH_SIZE,
        },
        "batch_plan": {
            "sha256": contracts.canonical_sha256([list(batch) for batch in batch_plan]),
            "indices": [list(batch) for batch in batch_plan],
            "input_pair_sha256_by_step": input_hashes,
            "warmup_steps": v1.WARMUP_STEPS,
            "measured_steps": v1.MEASURED_STEPS,
            "same_materialized_cpu_batches_for_both_models": True,
        },
        "runtime_identity": dict(runtime_identity),
        "model_order": list(MODEL_ORDER),
        "model_records": records,
        "elapsed_seconds": _finite(
            elapsed_seconds, name="resource V2 elapsed", positive=True
        ),
        "pre_write_replay": {
            "rules_unchanged": True,
            "authority_evidence_unchanged": True,
            "training_source_unchanged": True,
            "materialized_input_hashes_unchanged": True,
        },
        "hard_gate": gate,
        "warning_diagnostics": v1.warning_diagnostics(records),
        "artifact_boundaries": dict(ARTIFACT_BOUNDARIES),
    }
    return report


def validate_report(
    report: Mapping[str, Any],
    *,
    rules: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    batch_plan: Sequence[Sequence[int]],
    input_hashes: Sequence[str],
) -> None:
    expected_fields = {
        "schema",
        "status",
        "hard_gate_pass",
        "write_once",
        "scope",
        "rules_path",
        "rules_sha256",
        "superseded_v1_evidence",
        "authority_bindings",
        "training_source_manifest",
        "training_source_manifest_sha256",
        "data_identity",
        "batch_plan",
        "runtime_identity",
        "model_order",
        "model_records",
        "elapsed_seconds",
        "pre_write_replay",
        "hard_gate",
        "warning_diagnostics",
        "artifact_boundaries",
    }
    if set(report) != expected_fields:
        _fail("resource V2 report field set differs")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("write_once") is not True
        or report.get("scope") != rules["scope"]
        or report.get("rules_path") != contracts.repository_relative_path(RULES_PATH)
        or report.get("rules_sha256") != EXPECTED_RULES_SHA256
        or report.get("superseded_v1_evidence") != evidence["superseded_v1"]
        or report.get("authority_bindings") != _authority_record(evidence)
        or report.get("training_source_manifest") != source_manifest
        or report.get("training_source_manifest_sha256") != source_manifest["sha256"]
        or report.get("model_order") != list(MODEL_ORDER)
        or report.get("artifact_boundaries") != ARTIFACT_BOUNDARIES
        or report.get("pre_write_replay")
        != {
            "rules_unchanged": True,
            "authority_evidence_unchanged": True,
            "training_source_unchanged": True,
            "materialized_input_hashes_unchanged": True,
        }
    ):
        _fail("resource V2 report identity/bindings differ")
    expected_data = {
        "dataset": v1.DATASET,
        "data_role": v1.DATA_ROLE,
        "sample_count": v1.SAMPLE_COUNT,
        "index_relpath": v1.source_protocol.EXPECTED_SPLITS[v1.DATASET]["train"][
            "index_relpath"
        ],
        "index_file_sha256": v1.source_protocol.EXPECTED_SPLITS[v1.DATASET]["train"][
            "file_sha256"
        ],
        "ordered_ids_sha256": v1.source_protocol.EXPECTED_SPLITS[v1.DATASET]["train"][
            "ordered_ids_sha256"
        ],
        "normalization": dict(v1.source_protocol.LEGACY_NORMALIZATION[v1.DATASET]),
        "dataset_epoch": v1.DATASET_EPOCH,
        "batch_size": v1.BATCH_SIZE,
        "patch_size": v1.PATCH_SIZE,
    }
    expected_plan = {
        "sha256": v1.EXPECTED_BATCH_PLAN_SHA256,
        "indices": [list(batch) for batch in batch_plan],
        "input_pair_sha256_by_step": list(input_hashes),
        "warmup_steps": v1.WARMUP_STEPS,
        "measured_steps": v1.MEASURED_STEPS,
        "same_materialized_cpu_batches_for_both_models": True,
    }
    if report.get("data_identity") != expected_data or report.get("batch_plan") != expected_plan:
        _fail("resource V2 data/batch identity differs")
    runtime = report.get("runtime_identity")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "python",
            "torch",
            "cuda_runtime",
            "device_requested",
            "device",
            "device_name",
            "device_total_memory_bytes",
            "cublas_workspace_config",
            "deterministic_algorithms",
            "precision",
            "autocast",
            "gradient_scaler",
            "tf32_matmul",
            "tf32_cudnn",
        }
        or runtime.get("device_requested") != "cuda:0"
        or runtime.get("device") != "cuda:0"
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("precision") != "FP32"
        or runtime.get("autocast") is not False
        or runtime.get("gradient_scaler") is not False
        or runtime.get("tf32_matmul") is not False
        or runtime.get("tf32_cudnn") is not False
        or type(runtime.get("device_total_memory_bytes")) is not int
        or runtime["device_total_memory_bytes"] <= 0
    ):
        _fail("resource V2 runtime identity differs")
    _finite(report.get("elapsed_seconds"), name="resource V2 elapsed", positive=True)
    gate = evaluate_hard_gate(
        report.get("model_records"),
        top_input_hashes=input_hashes,
        device_total_memory_bytes=runtime["device_total_memory_bytes"],
        global_identity_valid=True,
    )
    if (
        report.get("hard_gate") != gate
        or report.get("status") != ("PASS" if gate["verdict"] == "GO" else "FAIL")
        or report.get("hard_gate_pass") is not (gate["verdict"] == "GO")
        or report.get("warning_diagnostics")
        != v1.warning_diagnostics(report.get("model_records", {}))
    ):
        _fail("resource V2 independently replayed verdict differs")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if REPORT_PATH.exists() or REPORT_PATH.is_symlink():
        raise FileExistsError("resource V2 write-once report already exists")
    rules_before = load_frozen_rules()
    evidence_before = load_bound_evidence(rules_before)
    source_before = build_source_manifest()
    plan_before, batches_before, hashes_before = _materialized_snapshot()
    device, runtime = v1._require_cuda(args.device)
    total_memory = int(runtime["device_total_memory_bytes"])
    records: dict[str, Any] = {}
    started = time.monotonic()
    specs = {spec["method"]: spec for spec in v1._model_specs()}
    for method in MODEL_ORDER:
        records[method] = v1.benchmark_one_model(
            specs[method],
            batches_before,
            device,
            total_memory,
            rules_before["optimizer"],
        )
    elapsed = max(time.monotonic() - started, sys.float_info.min)

    # Close the entire CUDA-window TOCTOU surface before committing evidence.
    rules_after = load_frozen_rules()
    evidence_after = load_bound_evidence(rules_after)
    source_after = build_source_manifest()
    plan_after, _batches_after, hashes_after = _materialized_snapshot()
    if (
        rules_after != rules_before
        or evidence_after != evidence_before
        or source_after != source_before
        or plan_after != plan_before
        or hashes_after != hashes_before
    ):
        _fail("resource V2 authority/source/data changed during measurement")
    report = build_report(
        rules=rules_before,
        evidence=evidence_before,
        source_manifest=source_before,
        batch_plan=plan_before,
        batches=batches_before,
        model_records=records,
        runtime_identity=runtime,
        elapsed_seconds=elapsed,
    )
    validate_report(
        report,
        rules=rules_after,
        evidence=evidence_after,
        source_manifest=source_after,
        batch_plan=plan_after,
        input_hashes=hashes_after,
    )
    written_sha256 = contracts.write_once_json(REPORT_PATH, report)

    # A zero exit is possible only after strict disk reload and a fresh replay.
    reopened = contracts.load_strict_json(REPORT_PATH)
    if (
        contracts.sha256_file(REPORT_PATH) != written_sha256
        or reopened != report
    ):
        _fail("resource V2 on-disk bytes differ from the committed report")
    final_rules = load_frozen_rules()
    final_evidence = load_bound_evidence(final_rules)
    final_source = build_source_manifest()
    final_plan, _final_batches, final_hashes = _materialized_snapshot()
    validate_report(
        reopened,
        rules=final_rules,
        evidence=final_evidence,
        source_manifest=final_source,
        batch_plan=final_plan,
        input_hashes=final_hashes,
    )
    return reopened


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run_benchmark(parse_args(argv))
    print(contracts.canonical_json_bytes(report).decode("utf-8"), end="")
    return 0 if report["status"] == "PASS" and report["hard_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
