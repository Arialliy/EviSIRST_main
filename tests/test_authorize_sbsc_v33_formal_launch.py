from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import pytest

from experiments import sbsc_v33_contracts as contracts
from tools import authorize_sbsc_v33_formal_launch as authorizer
from tools import benchmark_sbsc_v33_resources_v2 as resource


def _resource_report() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[list[int]],
    list[str],
]:
    rules = resource.load_frozen_rules()
    evidence = resource.load_bound_evidence(rules)
    source = resource.build_source_manifest()
    legacy = contracts.load_strict_json(
        resource.PROJECT_ROOT
        / "artifacts"
        / "sbsc_v33_preflight"
        / "paired_resource_benchmark.json"
    )
    plan = resource.v1.expected_batch_plan()
    hashes = list(legacy["batch_plan"]["input_pair_sha256_by_step"])
    batches = [{"input_pair_sha256": digest} for digest in hashes]
    report = resource.build_report(
        rules=rules,
        evidence=evidence,
        source_manifest=source,
        batch_plan=plan,
        batches=batches,
        model_records=legacy["model_records"],
        runtime_identity=legacy["runtime_identity"],
        elapsed_seconds=10.0,
    )
    return report, rules, evidence, plan, hashes


def _canonical_canary() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        contracts.load_strict_json(authorizer.CANARY_REPORT_PATH),
        contracts.load_strict_json(authorizer.CANARY_MANIFEST_PATH),
        contracts.load_strict_json(authorizer.CANARY_RULES_PATH),
    )


def _authorization_fixture() -> dict[str, Any]:
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    environment = {
        "schema": authorizer.ENVIRONMENT_SCHEMA,
        "common": {
            "python": "3.12.3",
            "torch": "2.9.1+cu130",
            "cuda_runtime": "13.0",
            "cublas_workspace_config": ":4096:8",
            "device_type": "cuda",
        },
        "unit_runtime_identity_sha256": "a" * 64,
        "canary_runtime_identity_sha256": "b" * 64,
        "paired_resource_runtime_identity_sha256": "c" * 64,
        "cuda_device": {
            "logical_index": 0,
            "name": "fixture GPU",
            "total_memory_bytes": 1024,
            "unit_resource_exact_match": True,
        },
    }
    return authorizer.build_authorization(
        method_config_reference={
            "path": "experiments/sbsc_v33_methods/sbsc_v33_third_irstd_formal.json",
            "sha256": "1" * 64,
        },
        unit_reference={
            "path": "artifacts/sbsc_v33_preflight/unit_test_report.json",
            "sha256": "2" * 64,
        },
        canary_reference={
            "path": "runs/sbsc_v33_third/canary_v2/IRSTD-1K/report.json",
            "sha256": "3" * 64,
        },
        resource_reference={
            "path": "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json",
            "sha256": authorizer.EXPECTED_RESOURCE_V2_REPORT_SHA256,
        },
        baseline=baseline,
        gradient=gradient,
        formal_source_sha256="5" * 64,
        unit_source_sha256="6" * 64,
        canary_source_sha256="7" * 64,
        resource_source_sha256="8" * 64,
        environment_identity=environment,
    )


def test_authorization_schema_exactly_keeps_runner_v1_core_and_evidence_set() -> None:
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    environment = {
        "schema": authorizer.ENVIRONMENT_SCHEMA,
        "common": {
            "python": "3.12.3",
            "torch": "2.9.1+cu130",
            "cuda_runtime": "13.0",
            "cublas_workspace_config": ":4096:8",
            "device_type": "cuda",
        },
        "unit_runtime_identity_sha256": "a" * 64,
        "canary_runtime_identity_sha256": "b" * 64,
        "paired_resource_runtime_identity_sha256": "c" * 64,
        "cuda_device": {
            "logical_index": 0,
            "name": "fixture GPU",
            "total_memory_bytes": 1024,
            "unit_resource_exact_match": True,
        },
    }
    authorization = authorizer.build_authorization(
        method_config_reference={
            "path": "experiments/sbsc_v33_methods/sbsc_v33_third_irstd_formal.json",
            "sha256": "1" * 64,
        },
        unit_reference={
            "path": "artifacts/sbsc_v33_preflight/unit_test_report.json",
            "sha256": "2" * 64,
        },
        canary_reference={
            "path": "runs/sbsc_v33_third/canary_v2/IRSTD-1K/report.json",
            "sha256": "3" * 64,
        },
        resource_reference={
            "path": "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json",
            "sha256": authorizer.EXPECTED_RESOURCE_V2_REPORT_SHA256,
        },
        baseline=baseline,
        gradient=gradient,
        formal_source_sha256="5" * 64,
        unit_source_sha256="6" * 64,
        canary_source_sha256="7" * 64,
        resource_source_sha256="8" * 64,
        environment_identity=environment,
    )
    assert authorization["schema"] == authorizer.AUTHORIZATION_SCHEMA
    assert authorization["status"] == "PASS"
    assert authorization["write_once"] is True
    assert authorization["authorized_run"] == authorizer.AUTHORIZED_RUN
    assert set(authorization["evidence"]) == {
        "unit_tests",
        "canary",
        "paired_resource_benchmark",
    }
    assert authorization["training_source_manifest_sha256"] == "5" * 64
    assert authorization["source_bindings"] == {
        "formal_training": "5" * 64,
        "unit_suite": "6" * 64,
        "canary_v2": "7" * 64,
        "paired_resource_benchmark": "8" * 64,
    }
    assert authorization["replay_checks"] == authorizer.REPLAY_CHECKS
    assert authorization["environment_identity_sha256"] == (
        contracts.canonical_sha256(environment)
    )


def test_real_canary_v2_replays_schema_counts_train_only_and_source_chain() -> None:
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    report, manifest, rules = _canonical_canary()
    result = authorizer.validate_canary_v2_evidence(
        report, manifest, rules, baseline=baseline, gradient=gradient
    )
    assert result["source_manifest"]["sha256"] == (
        resource.v1.EXPECTED_CANARY_SOURCE_MANIFEST_SHA256
    )
    assert contracts.sha256_file(authorizer.CANARY_REPORT_PATH) == (
        resource.v1.EXPECTED_CANARY_REPORT_SHA256
    )
    assert contracts.sha256_file(authorizer.CANARY_MANIFEST_PATH) == (
        resource.v1.EXPECTED_CANARY_MANIFEST_SHA256
    )


def test_canary_status_cannot_hide_wrong_1000_step_or_16000_sample_ledger() -> None:
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    report, manifest, rules = _canonical_canary()
    forged = copy.deepcopy(report)
    forged["optimizer_steps"] = 999
    with pytest.raises(
        authorizer.FormalLaunchAuthorizationError, match="sample/step ledger"
    ):
        authorizer.validate_canary_v2_evidence(
            forged, manifest, rules, baseline=baseline, gradient=gradient
        )


def test_paired_resource_pass_is_independently_recomputed() -> None:
    report, rules, evidence, plan, hashes = _resource_report()
    result = authorizer.validate_resource_evidence(
        report,
        rules,
        expected_evidence=evidence,
        expected_source_manifest=resource.build_source_manifest(),
        batch_plan=plan,
        input_hashes=hashes,
    )
    assert result["runtime_identity"]["precision"] == "FP32"
    forged = copy.deepcopy(report)
    forged["model_records"]["sbsc_v33_third"]["oom_count"] = 1
    with pytest.raises(
        authorizer.FormalLaunchAuthorizationError, match="replay failed"
    ):
        authorizer.validate_resource_evidence(
            forged,
            rules,
            expected_evidence=evidence,
            expected_source_manifest=resource.build_source_manifest(),
            batch_plan=plan,
            input_hashes=hashes,
        )


def test_fresh_replay_rejects_nonfrozen_resource_v2_report_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert contracts.sha256_file(authorizer.RESOURCE_REPORT_PATH) == (
        authorizer.EXPECTED_RESOURCE_V2_REPORT_SHA256
    )
    physical_sha256 = contracts.sha256_file

    def forged_sha256(path: Path) -> str:
        if path == authorizer.RESOURCE_REPORT_PATH:
            return "0" * 64
        return physical_sha256(path)

    monkeypatch.setattr(contracts, "sha256_file", forged_sha256)
    with pytest.raises(
        authorizer.FormalLaunchAuthorizationError,
        match="physical report SHA/path differs",
    ):
        authorizer._replay_formal_preflight()


def test_environment_requires_exact_common_runtime_and_cuda_device() -> None:
    report, rules, evidence, plan, hashes = _resource_report()
    resource_record = authorizer.validate_resource_evidence(
        report,
        rules,
        expected_evidence=evidence,
        expected_source_manifest=resource.build_source_manifest(),
        batch_plan=plan,
        input_hashes=hashes,
    )
    canary_runtime = {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "device_requested": "cuda:0",
        "device_type": "cuda",
        "cublas_workspace_config": ":4096:8",
    }
    canary_record = {
        "runtime_identity": canary_runtime,
        "runtime_identity_sha256": contracts.canonical_sha256(canary_runtime),
    }
    benchmark_runtime = resource_record["runtime_identity"]
    unit_runtime = {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "cublas_workspace_config": ":4096:8",
        "device_type": "cuda",
        "cuda_available": True,
        "cuda_devices": [
            {
                "index": 0,
                "name": benchmark_runtime["device_name"],
                "total_memory_bytes": benchmark_runtime[
                    "device_total_memory_bytes"
                ],
            }
        ],
    }
    unit_record = {
        "runtime_identity": unit_runtime,
        "runtime_identity_sha256": contracts.canonical_sha256(unit_runtime),
    }
    identity = authorizer.build_environment_identity(
        unit=unit_record,
        canary=canary_record,
        resource_record=resource_record,
    )
    assert identity["common"]["device_type"] == "cuda"
    drifted = copy.deepcopy(unit_record)
    drifted["runtime_identity"]["torch"] = "different"
    drifted["runtime_identity_sha256"] = contracts.canonical_sha256(
        drifted["runtime_identity"]
    )
    with pytest.raises(
        authorizer.FormalLaunchAuthorizationError, match="environment field"
    ):
        authorizer.build_environment_identity(
            unit=drifted,
            canary=canary_record,
            resource_record=resource_record,
        )


def test_source_manifest_physical_drift_is_rejected() -> None:
    runs = authorizer.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth_source_", dir=runs) as temporary:
        source = Path(temporary) / "source.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        relative = source.relative_to(authorizer.PROJECT_ROOT).as_posix()
        manifest = contracts.build_source_manifest((relative,))
        authorizer._validate_source_manifest(manifest, name="fixture")
        source.write_text("VALUE = 2\n", encoding="utf-8")
        with pytest.raises(
            authorizer.FormalLaunchAuthorizationError, match="drifted"
        ):
            authorizer._validate_source_manifest(manifest, name="fixture")


def test_missing_dependency_fails_before_authorization_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = authorizer.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth_missing_", dir=runs) as temporary:
        directory = Path(temporary)
        missing = directory / "missing_method_config.json"
        output = directory / "authorization.json"
        monkeypatch.setattr(authorizer, "METHOD_CONFIG_PATH", missing)
        with pytest.raises(FileNotFoundError, match="formal method config"):
            authorizer.authorize_formal_launch(output_path=output)
        assert not output.exists()


def test_existing_authorization_is_rejected_before_dependency_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = authorizer.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth_once_", dir=runs) as temporary:
        output = Path(temporary) / "authorization.json"
        output.write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            authorizer, "METHOD_CONFIG_PATH", Path(temporary) / "missing.json"
        )
        with pytest.raises(FileExistsError, match="already exists"):
            authorizer.authorize_formal_launch(output_path=output)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value["evidence"]["paired_resource_benchmark"].__setitem__(
            "path", "artifacts/sbsc_v33_preflight/paired_resource_benchmark.json"
        ),
        lambda value: value["evidence"]["paired_resource_benchmark"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value.__setitem__("environment_identity_sha256", "0" * 64),
        lambda value: value["source_bindings"].__setitem__(
            "formal_training", "0" * 64
        ),
        lambda value: value["replay_checks"].__setitem__(
            "evidence_unchanged_before_write", False
        ),
    ],
)
def test_exact_authorization_validator_rejects_schema_and_binding_attacks(
    mutator,
) -> None:
    expected = _authorization_fixture()
    forged = copy.deepcopy(expected)
    mutator(forged)
    with pytest.raises(authorizer.FormalLaunchAuthorizationError):
        authorizer.validate_authorization(
            forged, expected_authorization=expected
        )


def test_authorizer_returns_only_after_disk_reload_and_second_full_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = authorizer.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth_postwrite_", dir=runs) as temporary:
        output = Path(temporary) / "authorization.json"
        expected = _authorization_fixture()
        calls = 0

        def replay():
            nonlocal calls
            calls += 1
            return copy.deepcopy(expected)

        monkeypatch.setattr(authorizer, "_replay_formal_preflight", replay)
        result = authorizer.authorize_formal_launch(output_path=output)
        assert calls == 2
        assert result == expected
        assert contracts.load_strict_json(output) == expected
        assert output.read_bytes() == contracts.canonical_json_bytes(expected)


def test_authorizer_rejects_strict_reload_tampering_after_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = authorizer.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth_reload_", dir=runs) as temporary:
        output = Path(temporary) / "authorization.json"
        expected = _authorization_fixture()
        monkeypatch.setattr(
            authorizer,
            "_replay_formal_preflight",
            lambda: copy.deepcopy(expected),
        )
        real_load = contracts.load_strict_json

        def forged_load(path: Path):
            value = real_load(path)
            if path == output:
                value["status"] = "FAIL"
            return value

        monkeypatch.setattr(contracts, "load_strict_json", forged_load)
        with pytest.raises(
            authorizer.FormalLaunchAuthorizationError,
            match="strict reload differs",
        ):
            authorizer.authorize_formal_launch(output_path=output)
        assert output.is_file()


def test_authorizer_rejects_evidence_drift_on_post_write_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = authorizer.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth_rebuild_", dir=runs) as temporary:
        output = Path(temporary) / "authorization.json"
        before = _authorization_fixture()
        after = copy.deepcopy(before)
        after["evidence"]["unit_tests"]["sha256"] = "9" * 64
        calls = iter((before, after))
        monkeypatch.setattr(
            authorizer,
            "_replay_formal_preflight",
            lambda: copy.deepcopy(next(calls)),
        )
        with pytest.raises(
            authorizer.FormalLaunchAuthorizationError,
            match="freshly rebuilt evidence",
        ):
            authorizer.authorize_formal_launch(output_path=output)
        assert output.is_file()


def test_authorizer_detects_physical_byte_change_after_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = authorizer.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth_bytes_", dir=runs) as temporary:
        output = Path(temporary) / "authorization.json"
        expected = _authorization_fixture()
        monkeypatch.setattr(
            authorizer,
            "_replay_formal_preflight",
            lambda: copy.deepcopy(expected),
        )
        real_write = contracts.write_once_json

        def corrupting_write(path: Path, payload):
            digest = real_write(path, payload)
            path.write_bytes(path.read_bytes() + b" ")
            return digest

        monkeypatch.setattr(contracts, "write_once_json", corrupting_write)
        with pytest.raises(
            authorizer.FormalLaunchAuthorizationError,
            match="physical bytes differ",
        ):
            authorizer.authorize_formal_launch(output_path=output)
