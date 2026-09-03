from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from experiments import sbsc_v33_contracts as contracts
from tools import benchmark_sbsc_v33_resources as benchmark


def _batch_hash(position: int) -> str:
    return hashlib.sha256(f"paired-batch-{position}".encode("utf-8")).hexdigest()


def _valid_model_record(
    method: str,
    *,
    measured_step_seconds: float,
    peak_allocated: int = 500,
    peak_reserved: int = 700,
    device_total: int = 1000,
) -> dict[str, Any]:
    specs = {spec["method"]: spec for spec in benchmark._model_specs()}
    spec = specs[method]
    hashes = [_batch_hash(position) for position in range(benchmark.TOTAL_STEPS)]
    step_records = []
    for position, input_hash in enumerate(hashes):
        elapsed = 0.05 if position < benchmark.WARMUP_STEPS else measured_step_seconds
        step_records.append(
            {
                "position": position,
                "phase": (
                    "warmup"
                    if position < benchmark.WARMUP_STEPS
                    else "measured"
                ),
                "batch_size": benchmark.BATCH_SIZE,
                "input_pair_sha256": input_hash,
                "wall_elapsed_seconds": elapsed,
                "loss": {
                    "total": 1.5,
                    "segmentation": 1.0,
                    "router": 0.5,
                },
                "finite": True,
            }
        )
    measured_elapsed = measured_step_seconds * benchmark.MEASURED_STEPS
    return {
        "schema": benchmark.MODEL_RECORD_SCHEMA,
        "method": method,
        "model": spec["model"],
        "model_schema": spec["model_schema"],
        "loss_schema": spec["loss_schema"],
        "router_value_gradient_mode": spec["router_value_gradient_mode"],
        "balance_mode": spec["balance_mode"],
        "identity_valid": True,
        "shared_state_sha256": "a" * 64,
        "initial_state_sha256": "b" * 64,
        "builder_metadata_sha256": "c" * 64,
        "state_key_count": spec["state_key_count"],
        "parameter_count": spec["parameter_count"],
        "batch_plan_sha256": benchmark.EXPECTED_BATCH_PLAN_SHA256,
        "input_pair_sha256_by_step": hashes,
        "step_records": step_records,
        "warmup_step_count": benchmark.WARMUP_STEPS,
        "measured_step_count": benchmark.MEASURED_STEPS,
        "measured_sample_count": benchmark.MEASURED_SAMPLES,
        "measured_elapsed_seconds": measured_elapsed,
        "throughput_samples_per_second": (
            benchmark.MEASURED_SAMPLES / measured_elapsed
        ),
        "max_memory_allocated_bytes": peak_allocated,
        "max_memory_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": device_total,
        "finite": True,
        "training_state_audit": {
            "pass": True,
            "missing_unexpected": [],
            "missing_conditionally_allowed": [],
        },
        "oom_count": 0,
        "complete": True,
    }


def _valid_records() -> dict[str, dict[str, Any]]:
    # V3.3 is deliberately slower and uses more memory.  This must remain GO.
    return {
        "sbsc_v32": _valid_model_record(
            "sbsc_v32",
            measured_step_seconds=0.10,
            peak_allocated=400,
            peak_reserved=600,
        ),
        "sbsc_v33_third": _valid_model_record(
            "sbsc_v33_third",
            measured_step_seconds=0.20,
            peak_allocated=500,
            peak_reserved=700,
        ),
    }


def _materialized_batch_metadata() -> list[dict[str, Any]]:
    return [
        {
            "position": position,
            "phase": (
                "warmup" if position < benchmark.WARMUP_STEPS else "measured"
            ),
            "indices": list(indices),
            "input_pair_sha256": _batch_hash(position),
        }
        for position, indices in enumerate(benchmark.expected_batch_plan())
    ]


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "device_requested": "cuda:0",
        "device": "cuda:0",
        "device_name": "synthetic-test-device",
        "device_total_memory_bytes": 1000,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "precision": "FP32",
        "autocast": False,
        "gradient_scaler": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }


def _evidence() -> dict[str, Any]:
    return {
        "baseline": {
            "manifest_path": "experiments/sbsc_v33_baseline_authority_manifest.json",
            "manifest_sha256": "1" * 64,
        },
        "gradient": {
            "authorization_path": "experiments/sbsc_v33_gradient_authorization.json",
            "authorization_sha256": "2" * 64,
            "authorized_router_value_gradient_mode": "live",
        },
        "v2_canary": {
            "report_path": "runs/sbsc_v33_third/canary_v2/IRSTD-1K/report.json",
            "report_sha256": benchmark.EXPECTED_CANARY_REPORT_SHA256,
            "report_schema": benchmark.CANARY_REPORT_SCHEMA,
            "manifest_path": "runs/sbsc_v33_third/canary_v2/IRSTD-1K/manifest.json",
            "manifest_sha256": benchmark.EXPECTED_CANARY_MANIFEST_SHA256,
            "rules_path": "experiments/sbsc_v33_canary_rules_v2.json",
            "rules_sha256": benchmark.EXPECTED_CANARY_RULES_SHA256,
            "method_config_sha256": (
                benchmark.EXPECTED_CANARY_METHOD_CONFIG_SHA256
            ),
            "training_source_manifest_sha256": (
                benchmark.EXPECTED_CANARY_SOURCE_MANIFEST_SHA256
            ),
            "status": "PASS",
            "hard_gate_pass": True,
            "verdict": "GO",
        },
    }


def test_frozen_rules_and_preregistered_batch_plan_are_exact() -> None:
    rules = benchmark.load_frozen_rules()
    assert contracts.sha256_file(benchmark.RULES_PATH) == benchmark.EXPECTED_RULES_SHA256
    assert rules["schema"] == benchmark.RULES_SCHEMA
    assert rules["measurement"]["warmup_steps_per_model"] == 4
    assert rules["measurement"]["measured_steps_per_model"] == 12
    assert rules["identity"]["batch_size"] == 16
    assert rules["identity"]["precision"] == "FP32"
    plan = benchmark.expected_batch_plan()
    assert len(plan) == 16
    assert all(len(batch) == 16 for batch in plan)
    assert len({index for batch in plan for index in batch}) == 256
    assert contracts.canonical_sha256(plan) == benchmark.EXPECTED_BATCH_PLAN_SHA256


def test_v2_canary_pass_chain_is_physically_bound_and_replayed() -> None:
    evidence = benchmark.load_bound_evidence(benchmark.load_frozen_rules())
    canary = evidence["v2_canary"]
    assert canary["report_sha256"] == benchmark.EXPECTED_CANARY_REPORT_SHA256
    assert canary["manifest_sha256"] == benchmark.EXPECTED_CANARY_MANIFEST_SHA256
    assert canary["rules_sha256"] == benchmark.EXPECTED_CANARY_RULES_SHA256
    assert canary["method_config_sha256"] == (
        benchmark.EXPECTED_CANARY_METHOD_CONFIG_SHA256
    )
    assert canary["training_source_manifest_sha256"] == (
        benchmark.EXPECTED_CANARY_SOURCE_MANIFEST_SHA256
    )
    assert (canary["status"], canary["hard_gate_pass"], canary["verdict"]) == (
        "PASS",
        True,
        "GO",
    )


def test_missing_bound_canary_report_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = benchmark.PROJECT_ROOT / "runs" / "__missing_resource_canary__.json"
    assert not missing.exists()
    monkeypatch.setattr(benchmark, "CANARY_REPORT_PATH", missing)
    with pytest.raises(benchmark.ResourceBenchmarkError, match="report is unavailable"):
        benchmark.load_bound_evidence(benchmark.load_frozen_rules())


def test_slower_v33_is_warning_only_and_does_not_block_go() -> None:
    records = _valid_records()
    gate = benchmark.evaluate_hard_gate(records, global_identity_valid=True)
    warnings = benchmark.warning_diagnostics(records)
    assert gate["verdict"] == "GO"
    assert all(gate["checks"].values())
    assert gate["speed_or_ratio_used_as_gate"] is False
    assert warnings["affects_verdict"] is False
    assert warnings["ratios"]["v33_to_v32_throughput"] == pytest.approx(0.5)
    assert {warning["kind"] for warning in warnings["warnings"]} == {
        "v33_throughput_below_v32",
        "v33_peak_allocated_above_v32",
        "v33_peak_reserved_above_v32",
    }


@pytest.mark.parametrize(
    ("gate_name", "mutate", "global_identity_valid"),
    [
        ("identity", lambda records: None, False),
        (
            "same_input",
            lambda records: (
                records["sbsc_v33_third"][
                    "input_pair_sha256_by_step"
                ].__setitem__(0, "f" * 64),
                records["sbsc_v33_third"]["step_records"][0].__setitem__(
                    "input_pair_sha256", "f" * 64
                ),
            ),
            True,
        ),
        (
            "finite",
            lambda records: (
                records["sbsc_v33_third"].__setitem__("finite", False),
                records["sbsc_v33_third"]["training_state_audit"].__setitem__(
                    "pass", False
                ),
            ),
            True,
        ),
        (
            "complete_measurement",
            lambda records: records["sbsc_v33_third"].__setitem__(
                "measured_step_count", 11
            ),
            True,
        ),
        (
            "no_oom",
            lambda records: records["sbsc_v33_third"].__setitem__("oom_count", 1),
            True,
        ),
        (
            "peak_within_device",
            lambda records: records["sbsc_v33_third"].__setitem__(
                "max_memory_reserved_bytes", 1001
            ),
            True,
        ),
    ],
)
def test_each_and_only_each_declared_hard_gate_can_fail(
    gate_name: str,
    mutate: Any,
    global_identity_valid: bool,
) -> None:
    records = _valid_records()
    mutate(records)
    gate = benchmark.evaluate_hard_gate(
        records, global_identity_valid=global_identity_valid
    )
    assert gate["verdict"] == "NO-GO"
    assert gate["reasons"] == [gate_name]
    assert gate["checks"][gate_name] is False
    assert all(
        passed is True
        for name, passed in gate["checks"].items()
        if name != gate_name
    )


def test_malformed_model_records_fail_closed_without_exception() -> None:
    gate = benchmark.evaluate_hard_gate(
        {"sbsc_v32": None, "sbsc_v33_third": 7},
        global_identity_valid=True,
    )
    assert gate["verdict"] == "NO-GO"
    assert not any(gate["checks"].values())


def test_conditional_gain_gradient_semantics_match_canary_v2() -> None:
    class ChannelAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_dual_risk_level_gain = nn.Parameter(torch.zeros(4))

    class EncoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.channel_attn = ChannelAttention()

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.ModuleList([nn.Identity(), EncoderLayer()])

    class MTC(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.mtc = MTC()

    model = Tiny()
    gain_name = benchmark.core_v33.SBSC_V33_GAIN_STATE_KEY
    assert gain_name == "mtc.encoder.layer.1.channel_attn.raw_dual_risk_level_gain"
    gain = dict(model.named_parameters())[gain_name]
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    model.weight.grad = torch.ones_like(model.weight)
    gain.grad = None
    audit = benchmark._audit_final_training_state(
        model, optimizer, tuple(dict(model.named_parameters())), "sbsc_v33_third"
    )
    assert audit["pass"] is True
    assert audit["missing_conditionally_allowed"] == [gain_name]

    model.weight.grad = None
    audit = benchmark._audit_final_training_state(
        model, optimizer, tuple(dict(model.named_parameters())), "sbsc_v33_third"
    )
    assert audit["pass"] is False
    assert audit["missing_unexpected"] == ["weight"]

    model.weight.grad = torch.ones_like(model.weight)
    gain.grad = torch.full_like(gain, float("nan"))
    audit = benchmark._audit_final_training_state(
        model, optimizer, tuple(dict(model.named_parameters())), "sbsc_v33_third"
    )
    assert audit["pass"] is False
    assert audit["nonfinite_gradients"] == [gain_name]


def test_report_schema_binds_inputs_authorities_source_and_environment() -> None:
    rules = benchmark.load_frozen_rules()
    source_manifest = {
        "schema": "sctransnet_sbsc_v33/source_manifest/v1",
        "files": [],
        "sha256": "d" * 64,
    }
    report = benchmark.build_report(
        rules=rules,
        evidence=_evidence(),
        source_manifest=source_manifest,
        batch_plan=benchmark.expected_batch_plan(),
        batches=_materialized_batch_metadata(),
        model_records=_valid_records(),
        runtime_identity=_runtime_identity(),
        elapsed_seconds=9.0,
    )
    assert report["schema"] == benchmark.REPORT_SCHEMA
    assert (report["status"], report["hard_gate_pass"]) == ("PASS", True)
    assert report["rules_sha256"] == benchmark.EXPECTED_RULES_SHA256
    assert report["training_source_manifest"] == source_manifest
    assert report["training_source_manifest_sha256"] == "d" * 64
    assert report["authority_bindings"]["v2_canary"]["report_sha256"] == (
        benchmark.EXPECTED_CANARY_REPORT_SHA256
    )
    assert report["runtime_identity"]["precision"] == "FP32"
    assert report["batch_plan"]["sha256"] == benchmark.EXPECTED_BATCH_PLAN_SHA256
    assert report["model_order"] == list(benchmark.MODEL_ORDER)
    assert report["hard_gate"]["verdict"] == "GO"
    assert report["warning_diagnostics"]["affects_verdict"] is False
    assert report["artifact_boundaries"] == benchmark.ARTIFACT_BOUNDARIES
    assert "failure" not in report


def test_report_builder_rejects_unregistered_batch_identity_by_gate() -> None:
    batches = _materialized_batch_metadata()
    batches[0]["indices"] = list(reversed(batches[0]["indices"]))
    report = benchmark.build_report(
        rules=benchmark.load_frozen_rules(),
        evidence=_evidence(),
        source_manifest={
            "schema": "sctransnet_sbsc_v33/source_manifest/v1",
            "files": [],
            "sha256": "d" * 64,
        },
        batch_plan=benchmark.expected_batch_plan(),
        batches=batches,
        model_records=_valid_records(),
        runtime_identity=_runtime_identity(),
        elapsed_seconds=9.0,
    )
    assert report["status"] == "FAIL"
    assert report["hard_gate"]["checks"]["identity"] is False


def test_rules_drift_is_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = tmp_path / "rules.json"
    payload = benchmark.RULES_PATH.read_bytes() + b"\n"
    changed.write_bytes(payload)
    monkeypatch.setattr(benchmark, "RULES_PATH", changed)
    with pytest.raises(benchmark.ResourceBenchmarkError, match="physical SHA-256"):
        benchmark.load_frozen_rules()


def test_existing_report_is_refused_before_any_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "paired_resource_benchmark.json"
    existing.write_text("reserved", encoding="utf-8")
    monkeypatch.setattr(benchmark, "REPORT_PATH", existing)
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark.run_benchmark(argparse.Namespace(device="cuda:0"))


def test_source_manifest_is_complete_for_runtime_dependencies() -> None:
    manifest = benchmark.build_source_manifest()
    paths = {record["path"] for record in manifest["files"]}
    assert {
        "tools/benchmark_sbsc_v33_resources.py",
        "experiments/sbsc_v33_resource_benchmark_rules.json",
        "experiments/sbsc_v33_canary_rules_v2.json",
        "experiments/sbsc_v33_contracts.py",
        "experiments/sctransnet_sbsc_v32.py",
        "experiments/sctransnet_sbsc_v33.py",
        "experiments/evisirst_data.py",
        "experiments/three_dataset_v2_protocol.py",
        "train.py",
    } <= paths
    expected_model_paths = {
        path.relative_to(benchmark.PROJECT_ROOT).as_posix()
        for path in (benchmark.PROJECT_ROOT / "model").rglob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    assert expected_model_paths <= paths
    assert manifest["sha256"] == contracts.canonical_sha256(manifest["files"])


def test_train_only_boundary_has_no_evaluator_test_dataset_or_checkpoint_call() -> None:
    source_path = benchmark.PROJECT_ROOT / "tools" / "benchmark_sbsc_v33_resources.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_data_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "experiments.evisirst_data"
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert imported_data_names == {"EviSIRSTTrainDataset"}
    assert "EviSIRSTTestDataset" not in source
    assert "evaluate_model" not in called_names
    assert "load_state_dict" not in called_attributes
    assert "save" not in called_attributes
    assert "torch.save" not in source
