from __future__ import annotations

import ast
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import sbsc_v33_contracts as contracts
from experiments import sctransnet_sbsc_v33 as core
from experiments import three_dataset_v2_protocol as source_protocol
from tools import run_sbsc_v33_canary_v2 as canary


def _synthetic_capture() -> dict[str, object]:
    batch = 2
    token_hw = (2, 2)
    logits = tuple(torch.zeros(batch, 3, *token_hw) for _ in range(4))
    supports = tuple(torch.softmax(value, dim=1) for value in logits)
    mass = tuple(value.sum(dim=(2, 3)) for value in supports)
    integrated = tuple(value.clone() for value in mass)
    availability = tuple(torch.ones(batch, 3, dtype=torch.bool) for _ in range(4))
    intended = tuple(torch.zeros(batch, 1, 2, 1, dtype=torch.int64) for _ in range(4))
    effective = tuple(value.clone() for value in intended)
    projection_rows = sum(value.numel() for value in intended)
    return {
        "schema": core.SBSC_V33_SCHEMA + "/training_router_capture/v1",
        "module_id": 1,
        "batch_size": batch,
        "token_hw": token_hw,
        "logits": logits,
        "supports": supports,
        "availability": availability,
        "role_mass": mass,
        "integrated_winning_evidence": integrated,
        "intended_mode_codes": intended,
        "effective_mode_codes": effective,
        "value_spatial": torch.ones(batch, 480, *token_hw, requires_grad=True),
        "router_value_gradient_mode": "live",
        "compact_diagnostics": {
            "projection_rows": projection_rows,
            "nonidentity_rows": 0,
            "solver_attempt_rows": 0,
            "solver_accepted_rows": 0,
            "solver_fallback_rows": 0,
            "emission_checked_rows": 0,
            "emission_accepted_rows": 0,
            "emission_fallback_rows": 0,
            "mode_identity_rows": projection_rows,
            "mode_dual_rows": 0,
            "mode_hard_only_rows": 0,
            "mode_background_only_rows": 0,
        },
    }


def _epoch_record(epoch: int) -> dict[str, object]:
    certificate = {
        "projection_rows": 400,
        "nonidentity_rows": 100,
        "solver_attempt_rows": 100,
        "solver_accepted_rows": 95,
        "solver_fallback_rows": 5,
        "emission_checked_rows": 100,
        "emission_accepted_rows": 90,
        "emission_fallback_rows": 10,
        "mode_identity_rows": 310,
        "mode_dual_rows": 30,
        "mode_hard_only_rows": 30,
        "mode_background_only_rows": 30,
    }
    return {
        "epoch": epoch,
        "order_sha256": f"{epoch:064x}",
        "learning_rate": 0.001,
        "processed_samples": 800,
        "optimizer_steps": 50,
        "combined_backward_calls": 50,
        "loss": {"total": 2.0, "segmentation": 1.0, "router": 1.0},
        "router_level_loss": [1.0, 1.0, 1.0, 1.0],
        "role_probability": {
            "minimum": 0.1,
            "maximum": 0.8,
            "simplex_max_abs_error": 1.0e-7,
        },
        "availability": {
            "counts": {"C": 1600, "H": 800, "B": 2400},
            "opportunities": {"C": 3200, "H": 3200, "B": 3200},
            "fraction": {"C": 0.5, "H": 0.25, "B": 0.75},
        },
        "modes": {
            "intended_counts": {
                "identity": 300,
                "dual": 40,
                "hard_only": 30,
                "background_only": 30,
            },
            "effective_counts": {
                "identity": 310,
                "dual": 30,
                "hard_only": 30,
                "background_only": 30,
            },
            "effective_fraction": {
                "identity": 0.775,
                "dual": 0.075,
                "hard_only": 0.075,
                "background_only": 0.075,
            },
        },
        "certificate": certificate,
        "combined_backward": {
            "all_non_gain_active_present": True,
            "all_non_gain_active_finite": True,
            "non_gain_active_parameter_count": 9,
            "nonzero_gradient_elements": 100,
            "router_present_and_finite": True,
            "live_value_spatial_gradient_present_and_finite": True,
            "gain_active_batches": 1,
            "gain_inactive_batches": 49,
            "gain_gradient_nonfinite_batches": 0,
            "gain_gradient_state_counts": {
                "present_finite_required": 1,
                "absent_allowed": 49,
                "finite_zero_allowed": 0,
            },
            "gain_contract_pass": True,
        },
        "gain": [0.01, 0.02, 0.03, 0.04],
        "hard_gate_violations": {name: 0 for name in canary.HARD_GATE_ORDER},
    }


def _records() -> list[dict[str, object]]:
    return [_epoch_record(epoch) for epoch in range(1, 21)]


def _fake_prepared() -> dict[str, object]:
    rules, method_sha = canary.load_frozen_rules()
    split = dict(source_protocol.EXPECTED_SPLITS[canary.DATASET]["train"])
    return {
        "rules": rules,
        "rules_sha256": canary.EXPECTED_RULES_SHA256,
        "method_config_sha256": method_sha,
        "baseline": {
            "manifest_path": "experiments/sbsc_v33_baseline_authority_manifest.json",
            "manifest_sha256": rules["method_config"]["baseline_authority_sha256"],
            "authorities": {
                canary.DATASET: {
                    "evaluation_path": "baseline/evaluation/fake.json",
                    "evaluation_sha256": "1" * 64,
                    "checkpoint": {
                        "path": "baseline/checkpoints/fake.pth.tar",
                        "sha256": "2" * 64,
                    },
                }
            },
        },
        "gradient": {
            "authorization_path": "experiments/sbsc_v33_gradient_authorization.json",
            "authorization_sha256": rules["method_config"][
                "gradient_authorization_sha256"
            ],
            "authorized_router_value_gradient_mode": "live",
            "report_path": "artifacts/fake_gradient.json",
            "report_sha256": "3" * 64,
            "rules_path": "experiments/fake_gradient_rules.json",
            "rules_sha256": "4" * 64,
            "source_manifest_sha256": "5" * 64,
        },
        "source_manifest": {
            "schema": "sctransnet_sbsc_v33/source_manifest/v1",
            "files": [{"path": "fake.py", "sha256": "6" * 64, "size_bytes": 1}],
            "sha256": "7" * 64,
        },
        "builder_metadata_sha256": "8" * 64,
        "initial_state_sha256": "9" * 64,
        "model_validation": {"state_key_count": 513, "parameter_count": 11330188},
        "inactive_names": ("inactive.weight",),
        "active_names": ("active.weight",),
        "device_requested": "cpu",
        "device": torch.device("cpu"),
        "split": split,
    }


def test_rules_and_embedded_method_config_are_physically_frozen() -> None:
    rules, method_sha = canary.load_frozen_rules()
    assert contracts.sha256_file(canary.CANARY_RULES_PATH) == canary.EXPECTED_RULES_SHA256
    assert method_sha == canary.EXPECTED_METHOD_CONFIG_SHA256
    config = rules["method_config"]
    assert config["method"] == "sbsc_v33_third"
    assert config["dataset"] == "IRSTD-1K"
    assert config["architecture_seed"] == config["run_seed"] == 42
    assert config["epochs"] == 20
    assert config["batch_size"] == 16
    assert config["router_value_gradient_mode"] == "live"
    assert config["role_loss_balance_mode"] == "one_third_two_thirds"
    assert config["baseline_authority_sha256"] == contracts.sha256_file(
        contracts.BASELINE_MANIFEST_PATH
    )
    assert config["gradient_authorization_sha256"] == contracts.sha256_file(
        contracts.GRADIENT_AUTHORIZATION_PATH
    )
    assert config["test_loader_constructed"] is False
    assert config["checkpoint_written"] is False
    assert config["formal_weights_reusable"] is False
    assert config["gain_gradient_contract"] == {
        "conditioning_counter": "emission_accepted_rows",
        "when_positive": "present_and_finite",
        "when_zero": "none_or_finite_all_zero",
        "minimum_gain_active_batches_over_complete_run": 1,
        "nonfinite_gain_gradient_batches": 0,
    }
    evidence = rules["superseded_v1_write_once_evidence"]
    for kind in ("runner", "rules", "manifest", "report"):
        path = contracts.require_repository_relative_regular_file(
            evidence[f"{kind}_path"]
        )
        assert contracts.sha256_file(path) == evidence[f"{kind}_sha256"]


def test_cli_exposes_no_training_identity_or_output_overrides() -> None:
    parser = canary.build_parser()
    assert {option for action in parser._actions for option in action.option_strings} == {
        "-h",
        "--help",
        "--device",
    }
    assert canary.parse_args(["--device", "cpu"]).device == "cpu"
    with pytest.raises(SystemExit):
        canary.parse_args(["--epochs", "1"])
    with pytest.raises(SystemExit):
        canary.parse_args(["--method", "sbsc_v33_ord"])


def test_runner_has_no_evaluation_dataset_or_serialization_call() -> None:
    source = Path(canary.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "EviSIRSTTestDataset" not in imported_names
    assert "evaluate_model" not in imported_names
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "save"
        for node in ast.walk(tree)
    )


def test_current_source_manifest_covers_runner_core_data_and_model_tree() -> None:
    manifest = canary.build_source_manifest()
    paths = {record["path"] for record in manifest["files"]}
    assert {
        "tools/run_sbsc_v33_canary_v2.py",
        "experiments/sbsc_v33_canary_rules_v2.json",
        "experiments/sbsc_v33_contracts.py",
        "experiments/sctransnet_sbsc_v33.py",
        "experiments/sctransnet_sbsc_v32.py",
        "experiments/sctransnet_sbsc_v31.py",
        "experiments/evisirst_data.py",
        "experiments/three_dataset_v2_protocol.py",
        "train.py",
        "model/__init__.py",
        "model/_internal/SCTransNet.py",
    } <= paths
    assert "tools/run_sbsc_v33_canary.py" not in paths
    assert "experiments/sbsc_v33_canary_rules.json" not in paths
    ordered_paths = [record["path"] for record in manifest["files"]]
    assert ordered_paths == sorted(set(ordered_paths))
    expected_model_sources = {
        path.relative_to(canary.PROJECT_ROOT).as_posix()
        for path in (canary.PROJECT_ROOT / "model").rglob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    assert {path for path in paths if path.startswith("model/")} == expected_model_sources
    assert manifest["sha256"] == contracts.canonical_sha256(manifest["files"])
    for record in manifest["files"]:
        path = contracts.require_repository_relative_regular_file(record["path"])
        assert contracts.sha256_file(path) == record["sha256"]
        assert path.stat().st_size == record["size_bytes"]


def test_capture_audit_accepts_exact_role_availability_and_certificate() -> None:
    result = canary.audit_capture_record(_synthetic_capture())
    assert result["level_count"] == 4
    assert result["simplex_max_abs_error"] <= 2.0e-6
    assert result["availability_counts"] == {"C": 8, "H": 8, "B": 8}
    assert result["certificate"]["projection_rows"] == 16


def _gradient_fixture(monkeypatch):
    monkeypatch.setattr(core, "SBSC_V33_GAIN_STATE_KEY", "gain")
    monkeypatch.setattr(
        core, "SBSC_V33_ROUTER_STATE_KEYS", ("router_value", "router_head")
    )
    model = torch.nn.Module()
    for name in ("body", "router_value", "router_head", "gain"):
        model.register_parameter(name, torch.nn.Parameter(torch.ones(1)))
    named = dict(model.named_parameters())
    for name in ("body", "router_value", "router_head"):
        named[name].grad = torch.ones_like(named[name])
    value_spatial = torch.ones(1, requires_grad=True)
    value_spatial.grad = torch.ones_like(value_spatial)
    return model, tuple(named), value_spatial, named


@pytest.mark.parametrize("zero_representation", ["none", "finite_zero"])
def test_gain_inactive_batch_accepts_only_none_or_finite_zero(
    monkeypatch, zero_representation
) -> None:
    model, active, value_spatial, named = _gradient_fixture(monkeypatch)
    if zero_representation == "finite_zero":
        named["gain"].grad = torch.zeros_like(named["gain"])
    result = canary._audit_combined_backward(
        model,
        active,
        value_spatial,
        emission_accepted_rows=0,
    )
    assert result["gain_active_batch"] is False
    assert result["gain_gradient_state"] in {
        "absent_allowed",
        "finite_zero_allowed",
    }
    assert result["gain_contract_pass"] is True


@pytest.mark.parametrize("bad_value", [1.0, float("nan")])
def test_gain_inactive_batch_rejects_nonzero_or_nonfinite_gradient(
    monkeypatch, bad_value
) -> None:
    model, active, value_spatial, named = _gradient_fixture(monkeypatch)
    named["gain"].grad = torch.full_like(named["gain"], bad_value)
    with pytest.raises(canary.CanaryError, match="gain-inactive"):
        canary._audit_combined_backward(
            model,
            active,
            value_spatial,
            emission_accepted_rows=0,
        )


def test_gain_active_batch_requires_present_finite_gradient(monkeypatch) -> None:
    model, active, value_spatial, named = _gradient_fixture(monkeypatch)
    with pytest.raises(canary.CanaryError, match="gain gradient is absent"):
        canary._audit_combined_backward(
            model,
            active,
            value_spatial,
            emission_accepted_rows=1,
        )
    named["gain"].grad = torch.zeros_like(named["gain"])
    result = canary._audit_combined_backward(
        model,
        active,
        value_spatial,
        emission_accepted_rows=1,
    )
    assert result["gain_active_batch"] is True
    assert result["gain_gradient_state"] == "present_finite_required"


def test_non_gain_active_gradient_is_unconditionally_required(monkeypatch) -> None:
    model, active, value_spatial, named = _gradient_fixture(monkeypatch)
    named["body"].grad = None
    with pytest.raises(canary.CanaryError, match="missing"):
        canary._audit_combined_backward(
            model,
            active,
            value_spatial,
            emission_accepted_rows=0,
        )


@pytest.mark.parametrize("corruption", ["probability", "availability", "certificate"])
def test_capture_audit_fails_closed_on_each_structural_gate(corruption: str) -> None:
    record = _synthetic_capture()
    if corruption == "probability":
        record["supports"][0][0, 0, 0, 0] += 0.1
        expected = "role_probability"
    elif corruption == "availability":
        record["availability"][0][0, 0] = False
        expected = "availability"
    else:
        record["compact_diagnostics"]["emission_accepted_rows"] = 1
        expected = "certificate"
    with pytest.raises(canary.CanaryError) as captured:
        canary.audit_capture_record(record)
    assert captured.value.category == expected


def test_exact_twenty_epoch_records_pass_all_hard_gates() -> None:
    rules, _method_sha = canary.load_frozen_rules()
    gate = canary.evaluate_canary_gate(_records(), rules, execution_complete=True)
    assert gate["verdict"] == "GO"
    assert all(gate["checks"].values())


def test_complete_run_requires_at_least_one_gain_active_batch() -> None:
    rules, _method_sha = canary.load_frozen_rules()
    records = _records()
    for record in records:
        backward = record["combined_backward"]
        backward["gain_active_batches"] = 0
        backward["gain_inactive_batches"] = 50
        backward["gain_gradient_state_counts"] = {
            "present_finite_required": 0,
            "absent_allowed": 50,
            "finite_zero_allowed": 0,
        }
    gate = canary.evaluate_canary_gate(records, rules, execution_complete=True)
    assert gate["verdict"] == "NO-GO"
    assert gate["checks"]["combined_backward"] is False


@pytest.mark.parametrize(
    ("category", "mutate"),
    [
        ("finite", lambda record: record["loss"].__setitem__("total", float("nan"))),
        ("sample_step_counts", lambda record: record.__setitem__("processed_samples", 799)),
        ("role_probability", lambda record: record["role_probability"].__setitem__("minimum", -0.1)),
        ("availability", lambda record: record["hard_gate_violations"].__setitem__("availability", 1)),
        ("certificate", lambda record: record["certificate"].__setitem__("emission_accepted_rows", 89)),
        ("combined_backward", lambda record: record["combined_backward"].__setitem__("gain_gradient_nonfinite_batches", 1)),
        ("gain_bounds", lambda record: record.__setitem__("gain", [0.0, 0.0, 0.0, 0.251])),
    ],
)
def test_each_hard_gate_is_independently_fail_closed(category, mutate) -> None:
    rules, _method_sha = canary.load_frozen_rules()
    records = _records()
    mutate(records[0])
    gate = canary.evaluate_canary_gate(records, rules, execution_complete=True)
    assert gate["verdict"] == "NO-GO"
    assert gate["checks"][category] is False


def test_distribution_and_trend_anomalies_are_warning_only() -> None:
    rules, _method_sha = canary.load_frozen_rules()
    records = _records()
    for index, record in enumerate(records):
        record["availability"]["counts"] = {"C": 0, "H": 0, "B": 0}
        record["availability"]["fraction"] = {"C": 0.0, "H": 0.0, "B": 0.0}
        record["modes"]["effective_counts"] = {
            "identity": 310,
            "dual": 90,
            "hard_only": 0,
            "background_only": 0,
        }
        record["modes"]["effective_fraction"] = {
            "identity": 0.775,
            "dual": 0.225,
            "hard_only": 0.0,
            "background_only": 0.0,
        }
        record["certificate"]["mode_identity_rows"] = 310
        record["certificate"]["mode_dual_rows"] = 90
        record["certificate"]["mode_hard_only_rows"] = 0
        record["certificate"]["mode_background_only_rows"] = 0
        record["router_level_loss"] = [1.0 if index < 5 else 2.0] * 4
        record["gain"] = [0.0, 0.0, 0.0, 0.0]
    gate = canary.evaluate_canary_gate(records, rules, execution_complete=True)
    warnings = canary.warning_diagnostics(records, rules)
    assert gate["verdict"] == "GO"
    assert warnings["affects_verdict"] is False
    assert warnings["warnings"]


def test_manifest_binds_config_authorities_sources_and_forbidden_boundaries() -> None:
    prepared = _fake_prepared()
    manifest = canary.build_manifest(prepared)
    assert manifest["method_config_sha256"] == canary.EXPECTED_METHOD_CONFIG_SHA256
    assert manifest["authority_bindings"]["baseline"]["sha256"] == prepared[
        "baseline"
    ]["manifest_sha256"]
    assert manifest["authority_bindings"]["gradient"]["sha256"] == prepared[
        "gradient"
    ]["authorization_sha256"]
    assert manifest["training_source_manifest"]["sha256"] == "7" * 64
    assert manifest["data_identity"]["sample_count"] == 800
    assert manifest["artifact_boundaries"]["test_loader_constructed"] is False
    assert manifest["artifact_boundaries"]["checkpoint_written"] is False
    assert manifest["artifact_boundaries"]["formal_weights_reusable"] is False


def test_failure_after_manifest_writes_one_no_go_report_and_no_weights(monkeypatch) -> None:
    runs = canary.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v33_canary_test_", dir=runs) as temporary:
        output = Path(temporary)
        prepared = _fake_prepared()
        monkeypatch.setattr(canary, "DEFAULT_OUTPUT_DIR", output)
        monkeypatch.setattr(canary, "_prepare_canary", lambda _device: prepared)

        def fail_training(_prepared, progress, _records):
            progress.update(
                {
                    "current_epoch": 1,
                    "completed_epochs": 0,
                    "started_batches": 10,
                    "completed_batches": 9,
                    "batch_count": 9,
                    "combined_backward_calls": 10,
                    "processed_samples": 144,
                    "optimizer_steps": 9,
                }
            )
            raise canary.CanaryError("combined_backward", "fixture failure")

        monkeypatch.setattr(canary, "_execute_training", fail_training)
        report = canary.run_canary(SimpleNamespace(device="cpu"))
        assert report["status"] == "FAIL"
        assert report["hard_gate_pass"] is False
        assert report["hard_gate"]["verdict"] == "NO-GO"
        assert report["started_batches"] == 10
        assert report["batch_count"] == report["completed_batches"] == 9
        assert report["combined_backward_calls"] == 10
        assert report["processed_samples"] == 144
        assert report["optimizer_steps"] == 9
        assert report["execution_progress"]["current_epoch"] == 1
        assert report["test_loader_constructed"] is False
        assert report["checkpoint_written"] is False
        assert (output / canary.MANIFEST_NAME).is_file()
        assert (output / canary.REPORT_NAME).is_file()
        assert not list(output.glob("*.pth*"))
        with pytest.raises(FileExistsError):
            canary.run_canary(SimpleNamespace(device="cpu"))


def test_partial_progress_ledger_rejects_inconsistent_sample_or_batch_counts() -> None:
    progress = canary.new_progress_ledger()
    progress.update(
        {
            "current_epoch": 1,
            "started_batches": 2,
            "completed_batches": 1,
            "batch_count": 1,
            "combined_backward_calls": 2,
            "processed_samples": 15,
            "optimizer_steps": 1,
        }
    )
    with pytest.raises(canary.CanaryError, match="progress arithmetic"):
        canary._report_progress(progress)


def test_success_report_is_unambiguous_pass_without_reusable_weights(monkeypatch) -> None:
    runs = canary.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v33_canary_pass_", dir=runs) as temporary:
        output = Path(temporary)
        prepared = _fake_prepared()
        monkeypatch.setattr(canary, "DEFAULT_OUTPUT_DIR", output)
        monkeypatch.setattr(canary, "_prepare_canary", lambda _device: prepared)
        def complete_training(_prepared, progress, records):
            records.extend(_records())
            progress.update(
                {
                    "current_epoch": 20,
                    "completed_epochs": 20,
                    "started_batches": 1000,
                    "completed_batches": 1000,
                    "batch_count": 1000,
                    "combined_backward_calls": 1000,
                    "processed_samples": 16000,
                    "optimizer_steps": 1000,
                }
            )
            return records

        monkeypatch.setattr(canary, "_execute_training", complete_training)
        monkeypatch.setattr(
            canary, "build_source_manifest", lambda: prepared["source_manifest"]
        )
        report = canary.run_canary(SimpleNamespace(device="cpu"))
        assert report["status"] == "PASS"
        assert report["hard_gate_pass"] is True
        assert report["hard_gate"]["verdict"] == "GO"
        assert report["completed_epochs"] == 20
        assert report["optimizer_steps"] == 1000
        assert report["test_loader_constructed"] is False
        assert report["checkpoint_written"] is False
        assert report["formal_weights_reusable"] is False
        assert not list(output.glob("*.pth*"))


def test_underlying_artifact_writer_refuses_different_second_content() -> None:
    runs = canary.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v33_write_once_", dir=runs) as temporary:
        path = Path(temporary) / "manifest.json"
        first = contracts.write_once_json(path, {"schema": "fixture/v1", "value": 1})
        assert first == contracts.sha256_file(path)
        assert contracts.write_once_json(path, {"schema": "fixture/v1", "value": 1}) == first
        with pytest.raises(FileExistsError):
            contracts.write_once_json(path, {"schema": "fixture/v1", "value": 2})
