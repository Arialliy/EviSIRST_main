from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments import sbsc_v33_contracts as contracts
from tools import benchmark_sbsc_v33_resources_v2 as resource


def _v1_report() -> dict:
    return contracts.load_strict_json(
        resource.PROJECT_ROOT
        / "artifacts"
        / "sbsc_v33_preflight"
        / "paired_resource_benchmark.json"
    )


def _records_and_runtime() -> tuple[dict, dict, list[str]]:
    report = _v1_report()
    return (
        resource._v2_records(report["model_records"]),
        dict(report["runtime_identity"]),
        list(report["batch_plan"]["input_pair_sha256_by_step"]),
    )


def _gate(records: dict, runtime: dict, hashes: list[str]) -> dict:
    return resource.evaluate_hard_gate(
        records,
        top_input_hashes=hashes,
        device_total_memory_bytes=runtime["device_total_memory_bytes"],
        global_identity_valid=True,
    )


def _report_fixture() -> tuple[dict, dict, dict, dict, list[list[int]], list[str]]:
    rules = resource.load_frozen_rules()
    evidence = resource.load_bound_evidence(rules)
    source = resource.build_source_manifest()
    legacy = _v1_report()
    plan = resource.v1.expected_batch_plan()
    hashes = list(legacy["batch_plan"]["input_pair_sha256_by_step"])
    batches = [{"input_pair_sha256": value} for value in hashes]
    report = resource.build_report(
        rules=rules,
        evidence=evidence,
        source_manifest=source,
        batch_plan=plan,
        batches=batches,
        model_records=legacy["model_records"],
        runtime_identity=legacy["runtime_identity"],
        elapsed_seconds=1.0,
    )
    return report, rules, evidence, source, plan, hashes


def test_frozen_rules_and_superseded_v1_bytes_are_exact() -> None:
    rules = resource.load_frozen_rules()
    assert contracts.sha256_file(resource.RULES_PATH) == resource.EXPECTED_RULES_SHA256
    assert rules["superseded_v1_evidence"] == resource.V1_EVIDENCE
    for kind in ("tool", "rules", "tests", "report"):
        path = contracts.require_repository_relative_regular_file(
            resource.V1_EVIDENCE[f"{kind}_path"]
        )
        assert contracts.sha256_file(path) == resource.V1_EVIDENCE[f"{kind}_sha256"]


def test_bound_evidence_and_source_manifest_replay() -> None:
    rules = resource.load_frozen_rules()
    evidence = resource.load_bound_evidence(rules)
    source = resource.build_source_manifest()
    assert evidence["superseded_v1"] == resource.V1_EVIDENCE
    assert source["sha256"] == contracts.canonical_sha256(source["files"])
    assert "tools/benchmark_sbsc_v33_resources_v2.py" in {
        item["path"] for item in source["files"]
    }


def test_real_v1_measurements_satisfy_stricter_v2_gate() -> None:
    records, runtime, hashes = _records_and_runtime()
    gate = _gate(records, runtime, hashes)
    assert gate["verdict"] == "GO"
    assert all(gate["checks"].values())
    for record in records.values():
        assert record["training_state_audit"]["missing_conditionally_allowed"] == []


def test_v2_gate_rejects_missing_gain_even_when_self_pass_is_true() -> None:
    records, runtime, hashes = _records_and_runtime()
    record = records["sbsc_v33_third"]
    gain = record["training_state_audit"]["conditional_missing_gradient_parameter"]
    record["training_state_audit"]["missing_conditionally_allowed"] = [gain]
    record["training_state_audit"]["pass"] = True
    gate = _gate(records, runtime, hashes)
    assert gate["checks"]["finite"] is False
    assert gate["verdict"] == "NO-GO"


@pytest.mark.parametrize("location", ["record", "step"])
def test_v2_gate_binds_every_model_hash_to_top_batch_plan(location: str) -> None:
    records, runtime, hashes = _records_and_runtime()
    if location == "record":
        records["sbsc_v32"]["input_pair_sha256_by_step"][3] = "0" * 64
    else:
        records["sbsc_v33_third"]["step_records"][3]["input_pair_sha256"] = "0" * 64
    gate = _gate(records, runtime, hashes)
    assert gate["checks"]["same_input"] is False
    assert gate["verdict"] == "NO-GO"


@pytest.mark.parametrize(
    "mutator,failed_check",
    [
        (lambda record: record.__setitem__("oom_count", 1), "no_oom"),
        (lambda record: record.__setitem__("measured_step_count", 11), "complete_measurement"),
        (
            lambda record: record.__setitem__(
                "max_memory_reserved_bytes", record["device_total_memory_bytes"] + 1
            ),
            "peak_within_device",
        ),
        (
            lambda record: record["training_state_audit"]["nonfinite_gradients"].append(
                "bad"
            ),
            "finite",
        ),
        (
            lambda record: record["training_state_audit"].__setitem__(
                "conditional_missing_gradient_parameter", "wrong.gain"
            ),
            "finite",
        ),
    ],
)
def test_v2_gate_independently_recomputes_each_failure(mutator, failed_check: str) -> None:
    records, runtime, hashes = _records_and_runtime()
    mutator(records["sbsc_v33_third"])
    gate = _gate(records, runtime, hashes)
    assert gate["checks"][failed_check] is False
    assert gate["verdict"] == "NO-GO"


def test_report_roundtrip_is_strictly_replayable() -> None:
    report, rules, evidence, source, plan, hashes = _report_fixture()
    resource.validate_report(
        report,
        rules=rules,
        evidence=evidence,
        source_manifest=source,
        batch_plan=plan,
        input_hashes=hashes,
    )
    assert report["schema"] == resource.REPORT_SCHEMA
    assert report["status"] == "PASS"
    assert report["hard_gate_pass"] is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report.__setitem__("unexpected", True),
        lambda report: report["authority_bindings"]["baseline"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda report: report["batch_plan"]["input_pair_sha256_by_step"].__setitem__(
            0, "0" * 64
        ),
        lambda report: report["runtime_identity"].__setitem__(
            "deterministic_algorithms", False
        ),
        lambda report: report["hard_gate"].__setitem__("verdict", "NO-GO"),
        lambda report: report.__setitem__("status", "FAIL"),
    ],
)
def test_report_replay_rejects_tampering(mutator) -> None:
    report, rules, evidence, source, plan, hashes = _report_fixture()
    mutator(report)
    with pytest.raises(resource.ResourceBenchmarkV2Error):
        resource.validate_report(
            report,
            rules=rules,
            evidence=evidence,
            source_manifest=source,
            batch_plan=plan,
            input_hashes=hashes,
        )


def test_write_once_disk_reload_replays_exact_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, rules, evidence, source, plan, hashes = _report_fixture()
    path = tmp_path / "report.json"
    monkeypatch.setattr(contracts, "validated_output_path", lambda value: value)
    contracts.write_once_json(path, report)
    reopened = contracts.load_strict_json(path)
    resource.validate_report(
        reopened,
        rules=rules,
        evidence=evidence,
        source_manifest=source,
        batch_plan=plan,
        input_hashes=hashes,
    )
    assert reopened == report
    assert contracts.write_once_json(path, report) == contracts.sha256_file(path)
    changed = copy.deepcopy(report)
    changed["status"] = "FAIL"
    with pytest.raises(FileExistsError):
        contracts.write_once_json(path, changed)


def test_run_replays_all_snapshots_before_and_after_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "paired.json"
    legacy = _v1_report()
    plan = resource.v1.expected_batch_plan()
    hashes = list(legacy["batch_plan"]["input_pair_sha256_by_step"])
    batches = [{"input_pair_sha256": value} for value in hashes]
    rules = resource.load_frozen_rules()
    evidence = resource.load_bound_evidence(rules)
    source = resource.build_source_manifest()

    monkeypatch.setattr(resource, "REPORT_PATH", report_path)
    monkeypatch.setattr(contracts, "validated_output_path", lambda value: value)
    monkeypatch.setattr(resource, "load_frozen_rules", lambda: copy.deepcopy(rules))
    monkeypatch.setattr(
        resource, "load_bound_evidence", lambda _rules: copy.deepcopy(evidence)
    )
    monkeypatch.setattr(resource, "build_source_manifest", lambda: copy.deepcopy(source))
    monkeypatch.setattr(
        resource,
        "_materialized_snapshot",
        lambda: (copy.deepcopy(plan), copy.deepcopy(batches), list(hashes)),
    )
    monkeypatch.setattr(
        resource.v1,
        "_require_cuda",
        lambda _device: (object(), copy.deepcopy(legacy["runtime_identity"])),
    )
    monkeypatch.setattr(
        resource.v1,
        "benchmark_one_model",
        lambda spec, *_args, **_kwargs: copy.deepcopy(
            legacy["model_records"][spec["method"]]
        ),
    )
    result = resource.run_benchmark(argparse_namespace(device="cuda:0"))
    assert result["status"] == "PASS"
    assert report_path.is_file()


def test_run_rejects_authority_toctou_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "paired.json"
    legacy = _v1_report()
    plan = resource.v1.expected_batch_plan()
    hashes = list(legacy["batch_plan"]["input_pair_sha256_by_step"])
    batches = [{"input_pair_sha256": value} for value in hashes]
    rules = resource.load_frozen_rules()
    evidence = resource.load_bound_evidence(rules)
    source = resource.build_source_manifest()
    calls = {"count": 0}

    def changing_evidence(_rules):
        calls["count"] += 1
        value = copy.deepcopy(evidence)
        if calls["count"] >= 2:
            value["v2_canary"]["report_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(resource, "REPORT_PATH", report_path)
    monkeypatch.setattr(resource, "load_frozen_rules", lambda: copy.deepcopy(rules))
    monkeypatch.setattr(resource, "load_bound_evidence", changing_evidence)
    monkeypatch.setattr(resource, "build_source_manifest", lambda: copy.deepcopy(source))
    monkeypatch.setattr(
        resource,
        "_materialized_snapshot",
        lambda: (copy.deepcopy(plan), copy.deepcopy(batches), list(hashes)),
    )
    monkeypatch.setattr(
        resource.v1,
        "_require_cuda",
        lambda _device: (object(), copy.deepcopy(legacy["runtime_identity"])),
    )
    monkeypatch.setattr(
        resource.v1,
        "benchmark_one_model",
        lambda spec, *_args, **_kwargs: copy.deepcopy(
            legacy["model_records"][spec["method"]]
        ),
    )
    with pytest.raises(resource.ResourceBenchmarkV2Error, match="changed"):
        resource.run_benchmark(argparse_namespace(device="cuda:0"))
    assert not report_path.exists()


def test_run_rejects_on_disk_bytes_changed_after_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "paired.json"
    legacy = _v1_report()
    plan = resource.v1.expected_batch_plan()
    hashes = list(legacy["batch_plan"]["input_pair_sha256_by_step"])
    batches = [{"input_pair_sha256": value} for value in hashes]
    rules = resource.load_frozen_rules()
    evidence = resource.load_bound_evidence(rules)
    source = resource.build_source_manifest()
    original_write_once = contracts.write_once_json

    monkeypatch.setattr(resource, "REPORT_PATH", report_path)
    monkeypatch.setattr(contracts, "validated_output_path", lambda value: value)
    monkeypatch.setattr(resource, "load_frozen_rules", lambda: copy.deepcopy(rules))
    monkeypatch.setattr(
        resource, "load_bound_evidence", lambda _rules: copy.deepcopy(evidence)
    )
    monkeypatch.setattr(resource, "build_source_manifest", lambda: copy.deepcopy(source))
    monkeypatch.setattr(
        resource,
        "_materialized_snapshot",
        lambda: (copy.deepcopy(plan), copy.deepcopy(batches), list(hashes)),
    )
    monkeypatch.setattr(
        resource.v1,
        "_require_cuda",
        lambda _device: (object(), copy.deepcopy(legacy["runtime_identity"])),
    )
    monkeypatch.setattr(
        resource.v1,
        "benchmark_one_model",
        lambda spec, *_args, **_kwargs: copy.deepcopy(
            legacy["model_records"][spec["method"]]
        ),
    )

    def tampering_write(path, payload):
        digest = original_write_once(path, payload)
        changed = copy.deepcopy(payload)
        changed["elapsed_seconds"] = float(changed["elapsed_seconds"]) + 1.0
        path.write_bytes(contracts.canonical_json_bytes(changed))
        return digest

    monkeypatch.setattr(contracts, "write_once_json", tampering_write)
    with pytest.raises(resource.ResourceBenchmarkV2Error, match="on-disk bytes"):
        resource.run_benchmark(argparse_namespace(device="cuda:0"))


def argparse_namespace(**values):
    from argparse import Namespace

    return Namespace(**values)
