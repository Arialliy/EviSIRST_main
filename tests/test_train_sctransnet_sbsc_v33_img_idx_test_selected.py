from __future__ import annotations

import copy
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import train_sctransnet_sbsc_v33_img_idx_test_selected as runner
from experiments import sbsc_v33_contracts as contracts
from experiments import sbsc_v33_test_selection as selection
from tools import finalize_sbsc_v33_results as finalizer
from tools import authorize_sbsc_v33_formal_launch as authorizer
from tools import benchmark_sbsc_v33_resources_v2 as resource_v2


def _identity(dataset: str = "IRSTD-1K") -> dict[str, Any]:
    return {
        "method": "sbsc_v33_third",
        "dataset": dataset,
        "architecture_seed": 42,
        "run_seed": 42,
        "split_manifest_sha256": "1" * 64,
        "run_identity_sha256": "2" * 64,
        "evaluation_contract_sha256": "a" * 64,
        "method_config_sha256": "3" * 64,
        "gradient_authorization_sha256": "4" * 64,
        "baseline_authority_manifest_sha256": "5" * 64,
        "formal_launch_authorization_sha256": "6" * 64,
        "training_source_manifest_sha256": "7" * 64,
    }


def _metrics() -> dict[str, float | int | None]:
    return {
        "test_loss": 0.30,
        "miou": 0.70,
        "niou": 0.71,
        "pixel_precision": 0.80,
        "pixel_recall": 0.75,
        "pixel_f1": 0.7741935483870968,
        "pd": 0.90,
        "tiny_pd": 0.80,
        "fa": 2.0e-5,
        "false_objects_per_image": 0.1,
        "target_count": 10,
        "matched_target_count": 9,
        "tiny_target_count": 5,
        "matched_tiny_target_count": 4,
        "predicted_object_count": 11,
        "unmatched_predicted_object_count": 2,
        "valid_pixel_count": 65536,
    }


def test_formal_cli_and_schedule_are_frozen() -> None:
    args = runner.parse_args(
        [
            "--dataset",
            "IRSTD-1K",
            "--dataset-root",
            str(runner.PROJECT_ROOT / "datasets"),
        ]
    )

    assert args.seed == args.architecture_seed == 42
    assert args.epochs == 1000
    assert args.selection_begin == 500
    assert args.selection_every == 1
    assert args.batch_size == 16
    assert args.workers == 0
    assert args.smoke is False
    assert [
        epoch
        for epoch in range(1, args.epochs + 1)
        if runner.selection_due(epoch, args.selection_begin, args.selection_every)
    ] == list(range(500, 1001))


def test_failed_authorization_precedes_seed_device_data_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def reject(_dataset: str) -> dict[str, Any]:
        calls.append("preflight")
        raise RuntimeError("authorization rejected")

    def forbidden(label: str):
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(label)
            raise AssertionError(f"{label} ran before authorization")

        return fail

    monkeypatch.setattr(runner, "_load_formal_preflight", reject)
    monkeypatch.setattr(runner, "configure_determinism", forbidden("seed"))
    monkeypatch.setattr(runner, "require_device", forbidden("device"))
    monkeypatch.setattr(runner, "EviSIRSTTrainDataset", forbidden("train_data"))
    monkeypatch.setattr(runner, "EviSIRSTTestDataset", forbidden("test_data"))
    args = SimpleNamespace(
        dataset="IRSTD-1K",
        dataset_root=tmp_path / "data",
        output_root=tmp_path / "must-not-exist",
        seed=42,
        device="cuda:999",
    )

    with pytest.raises(RuntimeError, match="authorization rejected"):
        runner.run(args)

    assert calls == ["preflight"]
    assert not args.output_root.exists()


def test_non_authorized_dataset_fails_before_loading_any_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("authority")
        raise AssertionError("authority loader must not run")

    monkeypatch.setattr(
        runner.contracts, "load_baseline_authority_manifest", forbidden
    )
    monkeypatch.setattr(runner.contracts, "load_gradient_authorization", forbidden)
    monkeypatch.setattr(runner, "_training_source_manifest", forbidden)

    with pytest.raises(RuntimeError, match="authorizes IRSTD-1K only"):
        runner._load_formal_preflight("NUAA-SIRST")
    assert calls == []


def _launch_closure_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    method_path = runner.PROJECT_ROOT / "test.py"
    monkeypatch.setattr(runner, "METHOD_CONFIG_PATH", method_path)
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    formal_source = {"sha256": "1" * 64}
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
                "name": "fixture GPU",
                "total_memory_bytes": 4096,
            }
        ],
    }
    canary_runtime = {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "cublas_workspace_config": ":4096:8",
        "device_type": "cuda",
        "device_requested": "cuda:0",
    }
    resource_runtime = {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "cublas_workspace_config": ":4096:8",
        "device_requested": "cuda:0",
        "device": "cuda:0",
        "device_name": "fixture GPU",
        "device_total_memory_bytes": 4096,
        "deterministic_algorithms": True,
        "precision": "FP32",
        "autocast": False,
        "gradient_scaler": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }
    unit_record = {
        "source_manifest": {"sha256": "2" * 64},
        "runtime_identity": unit_runtime,
        "runtime_identity_sha256": contracts.canonical_sha256(unit_runtime),
    }
    canary_record = {
        "source_manifest": {"sha256": "3" * 64},
        "runtime_identity": canary_runtime,
        "runtime_identity_sha256": contracts.canonical_sha256(canary_runtime),
    }
    resource_record = {
        "source_manifest": {"sha256": "4" * 64},
        "runtime_identity": resource_runtime,
        "runtime_identity_sha256": contracts.canonical_sha256(resource_runtime),
    }
    environment = authorizer.build_environment_identity(
        unit=unit_record,
        canary=canary_record,
        resource_record=resource_record,
    )
    method_sha = contracts.sha256_file(method_path)
    launch = {
        "schema": authorizer.AUTHORIZATION_SCHEMA,
        "status": "PASS",
        "write_once": True,
        "authorized_run": dict(authorizer.AUTHORIZED_RUN),
        "method_config_path": contracts.repository_relative_path(method_path),
        "method_config_sha256": method_sha,
        "gradient_authorization_sha256": gradient["authorization_sha256"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "training_source_manifest_sha256": formal_source["sha256"],
        "evidence": {
            name: {"path": path, "sha256": f"{index + 5:x}" * 64}
            for index, (name, path) in enumerate(
                runner.FORMAL_EVIDENCE_PATHS.items()
            )
        },
        "environment_identity": environment,
        "environment_identity_sha256": contracts.canonical_sha256(environment),
        "source_bindings": {
            "formal_training": formal_source["sha256"],
            "unit_suite": unit_record["source_manifest"]["sha256"],
            "canary_v2": canary_record["source_manifest"]["sha256"],
            "paired_resource_benchmark": resource_record["source_manifest"][
                "sha256"
            ],
        },
        "replay_checks": dict(authorizer.REPLAY_CHECKS),
    }
    launch["evidence"]["canary"]["sha256"] = (
        runner.EXPECTED_CANARY_V2_REPORT_SHA256
    )
    launch["evidence"]["paired_resource_benchmark"]["sha256"] = (
        runner.EXPECTED_RESOURCE_V2_REPORT_SHA256
    )
    return (
        launch,
        baseline,
        gradient,
        formal_source,
        unit_record,
        canary_record,
        resource_record,
        method_sha,
    )


def test_launch_closure_accepts_only_exact_14_field_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        launch,
        baseline,
        gradient,
        source,
        unit,
        canary,
        resource,
        method_sha,
    ) = _launch_closure_fixture(monkeypatch)
    assert set(launch) == runner.FORMAL_LAUNCH_FIELDS
    runner._validate_formal_launch_closure(
        launch,
        dataset="IRSTD-1K",
        method_config_sha256=method_sha,
        baseline=baseline,
        gradient=gradient,
        current_source=source,
        unit_record=unit,
        canary_record=canary,
        resource_record=resource,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_top_field",
        "wrong_resource_path",
        "wrong_source_binding",
        "false_replay_check",
        "wrong_environment_sha",
    ),
)
def test_launch_closure_rejects_schema_environment_source_or_replay_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    (
        launch,
        baseline,
        gradient,
        source,
        unit,
        canary,
        resource,
        method_sha,
    ) = _launch_closure_fixture(monkeypatch)
    forged = copy.deepcopy(launch)
    if mutation == "extra_top_field":
        forged["unexpected"] = True
    elif mutation == "wrong_resource_path":
        forged["evidence"]["paired_resource_benchmark"]["path"] = (
            "artifacts/sbsc_v33_preflight/paired_resource_benchmark.json"
        )
    elif mutation == "wrong_source_binding":
        forged["source_bindings"]["unit_suite"] = "f" * 64
    elif mutation == "false_replay_check":
        forged["replay_checks"]["unit_report_exact_replay"] = False
    else:
        forged["environment_identity_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        runner._validate_formal_launch_closure(
            forged,
            dataset="IRSTD-1K",
            method_config_sha256=method_sha,
            baseline=baseline,
            gradient=gradient,
            current_source=source,
            unit_record=unit,
            canary_record=canary,
            resource_record=resource,
        )


def test_bound_json_requires_exact_reference_and_physical_sha() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runner_bound_", dir=runs) as temporary:
        path = Path(temporary) / "evidence.json"
        path.write_text('{"status":"PASS"}\n', encoding="utf-8")
        reference = {
            "path": path.relative_to(runner.PROJECT_ROOT).as_posix(),
            "sha256": contracts.sha256_file(path),
        }
        assert runner._load_bound_json(reference, name="fixture") == {
            "status": "PASS"
        }
        forged = dict(reference)
        forged["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="physical SHA-256"):
            runner._load_bound_json(forged, name="fixture")


def test_frozen_resource_digest_rejects_joint_top_model_and_step_hash_tamper() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runner_frozen_", dir=runs) as temporary:
        path = Path(temporary) / "resource.json"
        tampered = contracts.load_strict_json(resource_v2.REPORT_PATH)
        forged_hash = "f" * 64
        tampered["batch_plan"]["input_pair_sha256_by_step"][0] = forged_hash
        for model_record in tampered["model_records"].values():
            model_record["input_pair_sha256_by_step"][0] = forged_hash
            model_record["step_records"][0]["input_pair_sha256"] = forged_hash

        rules = resource_v2.load_frozen_rules()
        resource_v2.validate_report(
            tampered,
            rules=rules,
            evidence=resource_v2.load_bound_evidence(rules),
            source_manifest=resource_v2.build_source_manifest(),
            batch_plan=resource_v2.v1.expected_batch_plan(),
            input_hashes=tampered["batch_plan"]["input_pair_sha256_by_step"],
        )
        path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
        reference = {
            "path": path.relative_to(runner.PROJECT_ROOT).as_posix(),
            "sha256": contracts.sha256_file(path),
        }

        # A merely self-reported physical digest accepts the coherent rewrite.
        assert runner._load_bound_json(reference, name="fixture") == tampered
        with pytest.raises(ValueError, match="frozen path/SHA-256"):
            runner._load_frozen_bound_json(
                reference,
                name="paired resource benchmark",
                expected_path=reference["path"],
                expected_sha256=runner.EXPECTED_RESOURCE_V2_REPORT_SHA256,
            )


def test_full_preflight_calls_strict_replayers_before_seed_gpu_data_or_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = authorizer.freezer.build_config(runner.PROJECT_ROOT / "datasets")
    source = runner._training_source_manifest()
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    method_path = runner.PROJECT_ROOT / "test.py"
    launch_path = runner.PROJECT_ROOT / "train.py"
    monkeypatch.setattr(runner, "METHOD_CONFIG_PATH", method_path)
    monkeypatch.setattr(runner, "FORMAL_LAUNCH_AUTHORIZATION_PATH", launch_path)

    unit_runtime = {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "cublas_workspace_config": ":4096:8",
        "device_type": "cuda",
        "cuda_available": True,
        "cuda_devices": [
            {"index": 0, "name": "fixture GPU", "total_memory_bytes": 4096}
        ],
    }
    canary_runtime = {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "cublas_workspace_config": ":4096:8",
        "device_type": "cuda",
        "device_requested": "cuda:0",
    }
    benchmark_runtime = {
        "python": "3.12.3",
        "torch": "2.9.1+cu130",
        "cuda_runtime": "13.0",
        "cublas_workspace_config": ":4096:8",
        "device_requested": "cuda:0",
        "device": "cuda:0",
        "device_name": "fixture GPU",
        "device_total_memory_bytes": 4096,
        "deterministic_algorithms": True,
        "precision": "FP32",
        "autocast": False,
        "gradient_scaler": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
    }
    records = {
        "unit": {
            "source_manifest": {"sha256": "2" * 64},
            "runtime_identity": unit_runtime,
            "runtime_identity_sha256": contracts.canonical_sha256(unit_runtime),
        },
        "canary": {
            "source_manifest": {"sha256": "3" * 64},
            "runtime_identity": canary_runtime,
            "runtime_identity_sha256": contracts.canonical_sha256(canary_runtime),
        },
        "resource": {
            "source_manifest": {"sha256": "4" * 64},
            "runtime_identity": benchmark_runtime,
            "runtime_identity_sha256": contracts.canonical_sha256(
                benchmark_runtime
            ),
        },
    }
    environment = authorizer.build_environment_identity(
        unit=records["unit"],
        canary=records["canary"],
        resource_record=records["resource"],
    )
    launch = {
        "schema": authorizer.AUTHORIZATION_SCHEMA,
        "status": "PASS",
        "write_once": True,
        "authorized_run": dict(authorizer.AUTHORIZED_RUN),
        "method_config_path": contracts.repository_relative_path(method_path),
        "method_config_sha256": contracts.sha256_file(method_path),
        "gradient_authorization_sha256": gradient["authorization_sha256"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "training_source_manifest_sha256": source["sha256"],
        "evidence": {
            name: {"path": path, "sha256": f"{index + 5:x}" * 64}
            for index, (name, path) in enumerate(
                runner.FORMAL_EVIDENCE_PATHS.items()
            )
        },
        "environment_identity": environment,
        "environment_identity_sha256": contracts.canonical_sha256(environment),
        "source_bindings": {
            "formal_training": source["sha256"],
            "unit_suite": "2" * 64,
            "canary_v2": "3" * 64,
            "paired_resource_benchmark": "4" * 64,
        },
        "replay_checks": dict(authorizer.REPLAY_CHECKS),
    }
    launch["evidence"]["canary"]["sha256"] = (
        runner.EXPECTED_CANARY_V2_REPORT_SHA256
    )
    launch["evidence"]["paired_resource_benchmark"]["sha256"] = (
        runner.EXPECTED_RESOURCE_V2_REPORT_SHA256
    )
    unit_report = {
        "status": "PASS",
        "counts": {
            "passed": authorizer.unit_suite.EXPECTED_PASSED_COUNT,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "deselected": 0,
        },
    }
    canary_report = {
        "schema": authorizer.canary_v2.REPORT_SCHEMA,
        "status": "PASS",
        "hard_gate_pass": True,
        "test_dataset_constructed": False,
        "checkpoint_written": False,
    }
    benchmark_report = {
        "schema": resource_v2.REPORT_SCHEMA,
        "status": "PASS",
        "hard_gate_pass": True,
        "batch_plan": {
            "input_pair_sha256_by_step": [f"{index:064x}" for index in range(16)]
        },
        "hard_gate": {
            "checks": {name: True for name in resource_v2.HARD_GATE_ORDER},
            "verdict": "GO",
            "reasons": [],
        },
    }
    calls: list[str] = []

    def load_json(path: Path) -> dict[str, Any]:
        if path == method_path:
            return copy.deepcopy(config)
        if path == launch_path:
            return copy.deepcopy(launch)
        return {}

    def load_bound(_record: Any, *, name: str) -> dict[str, Any]:
        calls.append(f"load:{name}")
        return {
            "unit tests": unit_report,
            "canary": canary_report,
            "paired resource benchmark": benchmark_report,
        }[name]

    def validate_unit(report: dict[str, Any]) -> dict[str, Any]:
        calls.append("replay:unit")
        assert report["counts"]["passed"] == (
            authorizer.unit_suite.EXPECTED_PASSED_COUNT
        )
        assert all(
            report["counts"][name] == 0
            for name in (
                "failed",
                "skipped",
                "errors",
                "xfailed",
                "xpassed",
                "deselected",
            )
        )
        return records["unit"]

    def validate_canary(report: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("replay:canary")
        assert report["schema"] == authorizer.canary_v2.REPORT_SCHEMA
        assert report["hard_gate_pass"] is True
        assert report["test_dataset_constructed"] is False
        assert report["checkpoint_written"] is False
        return records["canary"]

    def validate_resource(report: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("replay:resource")
        assert report["schema"] == resource_v2.REPORT_SCHEMA
        assert report["hard_gate"]["verdict"] == "GO"
        assert set(report["hard_gate"]["checks"]) == set(
            resource_v2.HARD_GATE_ORDER
        )
        assert all(report["hard_gate"]["checks"].values())
        return records["resource"]

    monkeypatch.setattr(runner.contracts, "load_strict_json", load_json)
    monkeypatch.setattr(
        runner.contracts,
        "load_baseline_authority_manifest",
        lambda: baseline,
    )
    monkeypatch.setattr(
        runner.contracts,
        "load_gradient_authorization",
        lambda: gradient,
    )
    monkeypatch.setattr(
        authorizer.freezer,
        "build_config",
        lambda _dataset_root: copy.deepcopy(config),
    )
    monkeypatch.setattr(runner, "_load_bound_json", load_bound)
    monkeypatch.setattr(authorizer, "validate_unit_evidence", validate_unit)
    monkeypatch.setattr(authorizer, "validate_canary_v2_evidence", validate_canary)
    monkeypatch.setattr(authorizer, "validate_resource_evidence", validate_resource)
    monkeypatch.setattr(resource_v2, "load_frozen_rules", lambda: {})
    monkeypatch.setattr(resource_v2, "load_bound_evidence", lambda _rules: {})
    monkeypatch.setattr(
        resource_v2,
        "build_source_manifest",
        lambda: records["resource"]["source_manifest"],
    )
    monkeypatch.setattr(
        resource_v2,
        "_materialized_snapshot",
        lambda: (_ for _ in ()).throw(
            AssertionError("runner preflight must not construct benchmark data")
        ),
    )
    for name in (
        "configure_determinism",
        "require_device",
        "EviSIRSTTrainDataset",
        "EviSIRSTTestDataset",
        "evaluate_model",
    ):
        monkeypatch.setattr(
            runner,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"{_name} ran during preflight")
            ),
        )

    result = runner._load_formal_preflight("IRSTD-1K")
    assert result["launch"] == launch
    assert calls == [
        "load:unit tests",
        "load:canary",
        "load:paired resource benchmark",
        "replay:unit",
        "replay:canary",
        "replay:resource",
    ]


def test_runner_core_api_signatures_match_the_actual_v33_core() -> None:
    build = inspect.signature(runner.core.build_sctransnet_sbsc_v33_method)
    assert tuple(build.parameters) == (
        "method",
        "dataset",
        "architecture_seed",
        "training",
        "router_value_gradient_mode",
    )
    assert build.parameters["architecture_seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert build.parameters["training"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        build.parameters["router_value_gradient_mode"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )

    losses = inspect.signature(runner.core.training_losses_v33)
    assert tuple(losses.parameters) == (
        "model",
        "images",
        "masks",
        "criterion",
        "balance_mode",
        "router_loss_weight",
    )
    assert losses.parameters["balance_mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        losses.parameters["router_loss_weight"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))


def test_build_model_calls_the_frozen_core_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    model = _TinyModel()
    metadata = {
        "method": runner.METHOD_NAME,
        "model": runner.MODEL_NAME,
        "dataset": "IRSTD-1K",
        "architecture_seed": 42,
        "test_split_accessed": False,
        "pair": {
            "parent_checkpoint": None,
            "warm_start_used": False,
            "predecessor_checkpoint_used": False,
        },
    }

    def build(**kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        calls["build"] = kwargs
        return model, metadata

    def validate(candidate: nn.Module, **kwargs: Any) -> None:
        calls["validate"] = (candidate, kwargs)

    monkeypatch.setattr(runner, "EXPECTED_STATE_KEY_COUNT", 1)
    monkeypatch.setattr(runner, "EXPECTED_PARAMETER_COUNT", 1)
    monkeypatch.setattr(runner, "configure_determinism", lambda seed: calls.setdefault("seed", seed))
    monkeypatch.setattr(runner.core, "build_sctransnet_sbsc_v33_method", build)
    monkeypatch.setattr(runner.core, "validate_sctransnet_sbsc_v33", validate)
    monkeypatch.setattr(runner, "_validate_state", lambda value, expected: value)
    monkeypatch.setattr(runner, "_freeze_structurally_inactive", lambda _model: ())

    built, returned_metadata, inactive = runner._build_model("IRSTD-1K")

    assert built is model
    assert returned_metadata == metadata
    assert inactive == ()
    assert calls["seed"] == 42
    assert calls["build"] == {
        "method": "sbsc_v33_third",
        "dataset": "IRSTD-1K",
        "architecture_seed": 42,
        "training": True,
        "router_value_gradient_mode": "live",
    }
    assert calls["validate"] == (
        model,
        {
            "method": "sbsc_v33_third",
            "router_value_gradient_mode": "live",
            "require_zero_gain": True,
        },
    )


def test_training_loss_wrapper_calls_and_unpacks_the_frozen_core_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    images = torch.zeros(1, 1, 2, 2)
    masks = torch.zeros_like(images)
    criterion = nn.BCELoss(reduction="mean")
    total = model.weight.square() + 3.0
    segmentation = model.weight.square() + 2.0
    router_loss = model.weight.square() + 1.0
    calls: dict[str, Any] = {}

    def losses(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return total, segmentation, router_loss, object()

    monkeypatch.setattr(runner.core, "training_losses_v33", losses)

    observed = runner._training_losses(model, images, masks, criterion)

    assert observed == (total, segmentation, router_loss)
    assert calls["args"] == (model, images, masks, criterion)
    assert calls["kwargs"] == {
        "balance_mode": "one_third_two_thirds",
        "router_loss_weight": 1.0,
    }


@pytest.mark.parametrize(
    "bad_loss",
    (
        torch.tensor([1.0]),
        torch.tensor(float("nan")),
        torch.tensor(-1.0),
    ),
)
def test_training_loss_wrapper_rejects_malformed_scalars(
    monkeypatch: pytest.MonkeyPatch,
    bad_loss: torch.Tensor,
) -> None:
    monkeypatch.setattr(
        runner.core,
        "training_losses_v33",
        lambda *_args, **_kwargs: (
            bad_loss,
            torch.tensor(1.0),
            torch.tensor(1.0),
            object(),
        ),
    )
    with pytest.raises(RuntimeError, match="total loss is malformed"):
        runner._training_losses(
            _TinyModel(),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            nn.BCELoss(),
        )


def test_metric_row_round_trips_through_the_independent_v33_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sha256", lambda _path: "8" * 64)
    identity = _identity()
    record = runner._metric_row(
        _metrics(),
        500,
        selection_identity=identity,
        model_state_sha256="9" * 64,
        sample_count=201,
    )

    payload = selection.select_prefix(
        [record], completed_epoch=500, expected_identity=identity
    )
    selected = payload["roles"]["best_mIoU"]["selected"]
    ranked = payload["ranking_provenance"]["evaluated_records"][0]
    assert record["model"] == selected["model"] == ranked["model"] == runner.MODEL_NAME
    assert record["seed"] == selected["seed"] == ranked["seed"] == 42
    assert record["sample_count"] == selected["sample_count"] == ranked["sample_count"] == 201
    assert (
        record["evaluation_source_sha256"]
        == selected["evaluation_source_sha256"]
        == ranked["evaluation_source_sha256"]
        == "8" * 64
    )
    assert set(selected["metrics"]) == set(_metrics())


def test_selector_rejects_a_runner_record_with_non_full_test_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sha256", lambda _path: "8" * 64)
    identity = _identity()
    record = runner._metric_row(
        _metrics(),
        500,
        selection_identity=identity,
        model_state_sha256="9" * 64,
        sample_count=200,
    )
    with pytest.raises(selection.SBSCV33TestSelectionError, match="sample_count"):
        selection.select_prefix(
            [record], completed_epoch=500, expected_identity=identity
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("miou", 1.1, "outside"),
        ("fa", -1.0, "nonnegative"),
        ("test_loss", float("inf"), "non-finite"),
        ("target_count", 1.5, "nonnegative int"),
    ),
)
def test_metric_row_rejects_invalid_full_evaluator_values(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: Any,
    message: str,
) -> None:
    monkeypatch.setattr(runner, "_sha256", lambda _path: "8" * 64)
    metrics = _metrics()
    metrics[field] = bad_value
    with pytest.raises(ValueError, match=message):
        runner._metric_row(
            metrics,
            500,
            selection_identity=_identity(),
            model_state_sha256="9" * 64,
            sample_count=201,
        )


def test_metric_row_requires_the_complete_evaluator_field_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sha256", lambda _path: "8" * 64)
    metrics = _metrics()
    metrics.pop("pixel_f1")
    with pytest.raises(ValueError, match="unexpected metric field set"):
        runner._metric_row(
            metrics,
            500,
            selection_identity=_identity(),
            model_state_sha256="9" * 64,
            sample_count=201,
        )


def test_training_config_and_501_records_match_the_finalizer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "METHOD_CONFIG_PATH", runner.PROJECT_ROOT / "test.py")
    monkeypatch.setattr(
        runner,
        "FORMAL_LAUNCH_AUTHORIZATION_PATH",
        runner.PROJECT_ROOT / "train.py",
    )
    baseline = contracts.load_baseline_authority_manifest()
    source = runner._training_source_manifest()
    authorization = {
        "authorization_path": "experiments/sbsc_v33_gradient_authorization.json",
        "authorization_sha256": "4" * 64,
        "authorized_router_value_gradient_mode": "live",
    }
    args = SimpleNamespace(
        dataset="IRSTD-1K",
        architecture_seed=42,
        seed=42,
        epochs=1000,
        batch_size=16,
        patch_size=256,
        workers=0,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_epochs=10,
        selection_begin=500,
        selection_every=1,
        smoke=False,
    )
    authorities = {
        "baseline": baseline,
        "gradient": authorization,
        "method_config": {
            "split_contract": {
                "train": {"runner_index_order_sha256": "a" * 64},
                "test": {"runner_index_order_sha256": "b" * 64},
                "normalization": baseline["authorities"]["IRSTD-1K"][
                    "normalization"
                ],
            }
        },
        "method_config_sha256": "3" * 64,
        "launch_authorization_sha256": "6" * 64,
        "training_source_manifest": source,
    }
    training = runner._formal_config(
        args,
        train_count=800,
        test_count=201,
        train_index_sha256="a" * 64,
        test_index_sha256="b" * 64,
        normalization=baseline["authorities"]["IRSTD-1K"]["normalization"],
        source_tree_sha256=source["sha256"],
        authorities=authorities,
    )
    method_config = {
        "method": "sbsc_v33_third",
        "path": training["method_config_path"],
        "sha256": "3" * 64,
        "router_value_gradient_mode": "live",
        "balance_mode": "one_third_two_thirds",
        "training_source_manifest": source,
        "training_source_manifest_sha256": source["sha256"],
        "split_contract": {
            "train": {"runner_index_order_sha256": "a" * 64},
            "test": {"runner_index_order_sha256": "b" * 64},
        },
    }
    monkeypatch.setattr(
        finalizer,
        "_validate_formal_launch_authorization",
        lambda **_kwargs: {
            "path": training["formal_launch_authorization_path"],
            "sha256": training["formal_launch_authorization_sha256"],
        },
    )

    validated_training = finalizer._validate_training_contract(
        training,
        dataset="IRSTD-1K",
        method_config=method_config,
        authorization=authorization,
        baseline=baseline,
    )
    assert validated_training["evaluation_contract"] == training[
        "evaluation_contract"
    ]
    assert validated_training["selection_identity"] == training[
        "selection_identity"
    ]
    completed_training = {
        **training,
        "builder_metadata_sha256": "c" * 64,
        "structurally_inactive_parameter_names": sorted(
            runner.EXPECTED_INACTIVE_PARAMETER_NAMES
        ),
        "state_key_count": runner.EXPECTED_STATE_KEY_COUNT,
        "parameter_count": runner.EXPECTED_PARAMETER_COUNT,
    }
    assert runner._validate_completed_training_config(
        completed_training,
        dataset="IRSTD-1K",
        authorities=authorities,
    ) == completed_training

    monkeypatch.setattr(
        runner,
        "_sha256",
        lambda _path: training["evaluation_source_sha256"],
    )
    history = [
        runner._metric_row(
            _metrics(),
            epoch,
            selection_identity=training["selection_identity"],
            model_state_sha256=f"{epoch:064x}",
            sample_count=201,
        )
        for epoch in range(500, 1001)
    ]
    validated_history = finalizer._validate_history(
        history,
        dataset="IRSTD-1K",
        method_config=method_config,
        training=validated_training,
    )
    assert len(validated_history) == 501
    assert (validated_history[0]["epoch"], validated_history[-1]["epoch"]) == (
        500,
        1000,
    )


def test_publish_once_materializes_two_distinct_write_once_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "best_mIoU.pth.tar"
    second = tmp_path / "best_Pd.pth.tar"
    payload = {"state_dict": {"weight": torch.tensor([1.0])}}

    first_sha = runner._publish_once(first, payload)
    second_sha = runner._publish_once(second, payload)

    assert first_sha == runner._sha256(first)
    assert second_sha == runner._sha256(second)
    assert first.stat().st_ino != second.stat().st_ino
    first_payload = torch.load(first, map_location="cpu", weights_only=True)
    second_payload = torch.load(second, map_location="cpu", weights_only=True)
    assert torch.equal(
        first_payload["state_dict"]["weight"],
        second_payload["state_dict"]["weight"],
    )
    with pytest.raises(FileExistsError, match="already exists"):
        runner._publish_once(first, payload)


def test_resume_repairs_a_winner_missing_between_dual_role_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_sha = "a" * 64
    payload = {
        "epoch": 500,
        "state_dict": {"weight": torch.tensor([1.0])},
        "selection_history": [
            {"epoch": 500, "model_state_sha256": state_sha}
        ],
        "best_epochs": {"best_miou": 500, "best_pd": 500},
    }
    first = tmp_path / "best_miou_state.pth.tar"
    missing = tmp_path / "best_pd_state.pth.tar"
    torch.save(payload, first)
    monkeypatch.setattr(
        runner,
        "_load_recovery",
        lambda path, **_kwargs: dict(
            torch.load(path, map_location="cpu", weights_only=True)
        ),
    )
    monkeypatch.setattr(runner, "_state_dict_sha256", lambda _state: state_sha)

    runner._repair_missing_winner_recoveries(
        candidate_paths=[(500, first)],
        best_epochs={"best_miou": 500, "best_pd": 500},
        best_recovery_paths={"best_miou": first, "best_pd": missing},
        config={},
        expected_state={},
    )

    assert missing.is_file()
    replay = torch.load(missing, map_location="cpu", weights_only=True)
    assert replay["epoch"] == 500
    assert replay["best_epochs"] == {"best_miou": 500, "best_pd": 500}
    assert torch.equal(
        replay["state_dict"]["weight"], payload["state_dict"]["weight"]
    )


def test_resume_refuses_to_guess_an_unrecoverable_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = {
        "epoch": 500,
        "state_dict": {"weight": torch.tensor([1.0])},
        "selection_history": [
            {"epoch": 500, "model_state_sha256": "a" * 64}
        ],
        "best_epochs": {"best_miou": 500, "best_pd": 500},
    }
    source = tmp_path / "best_miou_state.pth.tar"
    missing = tmp_path / "best_pd_state.pth.tar"
    torch.save(payload, source)
    monkeypatch.setattr(
        runner,
        "_load_recovery",
        lambda path, **_kwargs: dict(
            torch.load(path, map_location="cpu", weights_only=True)
        ),
    )
    monkeypatch.setattr(runner, "_state_dict_sha256", lambda _state: "b" * 64)

    with pytest.raises(RuntimeError, match="cannot reconstruct"):
        runner._repair_missing_winner_recoveries(
            candidate_paths=[(500, source)],
            best_epochs={"best_miou": 500, "best_pd": 500},
            best_recovery_paths={"best_miou": source, "best_pd": missing},
            config={},
            expected_state={},
        )
    assert not missing.exists()


@pytest.mark.parametrize(
    "mutation",
    ("metadata", "training", "metrics", "state_hash", "tensor", "extra"),
)
def test_exact_checkpoint_replay_rejects_every_partial_publication_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    state_sha = "a" * 64
    expected = {
        "schema": runner.CHECKPOINT_SCHEMA,
        "model": runner.MODEL_NAME,
        "method": runner.METHOD_NAME,
        "training": {"gradient_authorization_sha256": "1" * 64},
        "metrics": {"miou": 0.7, "pd": 0.9},
        "model_state_sha256": state_sha,
        "state_dict": {"weight": torch.tensor([1.0])},
    }
    observed = copy.deepcopy(expected)
    if mutation == "metadata":
        observed["method"] = "sbsc_v33_half"
    elif mutation == "training":
        observed["training"]["gradient_authorization_sha256"] = "2" * 64
    elif mutation == "metrics":
        observed["metrics"]["miou"] = 0.8
    elif mutation == "state_hash":
        observed["model_state_sha256"] = "b" * 64
    elif mutation == "tensor":
        observed["state_dict"]["weight"] = torch.tensor([2.0])
    elif mutation == "extra":
        observed["unregistered"] = True
    monkeypatch.setattr(runner, "_validate_state", lambda value, _expected: value)
    monkeypatch.setattr(runner, "_state_dict_sha256", lambda _state: state_sha)

    with pytest.raises(ValueError):
        runner._validate_checkpoint_payload_exact(
            observed,
            expected,
            expected_state=expected["state_dict"],
            label="partial best_miou",
        )


def test_exact_checkpoint_replay_accepts_the_complete_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_sha = "a" * 64
    expected = {
        "schema": runner.CHECKPOINT_SCHEMA,
        "model_state_sha256": state_sha,
        "state_dict": {"weight": torch.tensor([1.0])},
        "training": {"method_config_sha256": "1" * 64},
        "metrics": {"miou": 0.7, "pd": 0.9},
    }
    monkeypatch.setattr(runner, "_validate_state", lambda value, _expected: value)
    monkeypatch.setattr(runner, "_state_dict_sha256", lambda _state: state_sha)
    assert runner._validate_checkpoint_payload_exact(
        copy.deepcopy(expected),
        expected,
        expected_state=expected["state_dict"],
        label="partial best_miou",
    )["training"] == expected["training"]


def test_completed_publication_replays_full_checkpoint_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "EXPECTED_STATE_KEY_COUNT", 1)
    roles = {"best_miou": 500, "best_pd": 501}
    state_hashes = {"best_miou": "a" * 64, "best_pd": "b" * 64}
    history = [
        {
            "epoch": epoch,
            "miou": 0.7 if role == "best_miou" else 0.6,
            "pd": 0.9 if role == "best_miou" else 0.95,
            "fa": 1.0e-5,
            "niou": 0.7,
            "tiny_pd": 0.8,
            "test_loss": 0.3,
            "model_state_sha256": state_hashes[role],
        }
        for role, epoch in roles.items()
    ]
    training = {
        "method_config_path": "method.json",
        "method_config_sha256": "1" * 64,
        "gradient_authorization_path": "gradient.json",
        "gradient_authorization_sha256": "2" * 64,
        "baseline_authority_manifest_path": "baseline.json",
        "baseline_authority_manifest_sha256": "3" * 64,
        "formal_launch_authorization_path": "launch.json",
        "formal_launch_authorization_sha256": "4" * 64,
        "training_source_manifest_sha256": "5" * 64,
        "selection_identity": {},
    }
    provenance = {"schema": "replayed"}
    selections = {
        role: {
            "epoch": epoch,
            "metrics": next(row for row in history if row["epoch"] == epoch),
            "role_key": runner.role_key_record(
                next(row for row in history if row["epoch"] == epoch), role
            ),
            "model_state_sha256": state_hashes[role],
            "test_selected": True,
            "selection_is_optimistic": True,
        }
        for role, epoch in roles.items()
    }
    checkpoint_payloads = {
        role: {
            "schema": runner.CHECKPOINT_SCHEMA,
            "model": runner.MODEL_NAME,
            "method": runner.METHOD_NAME,
            "dataset": "IRSTD-1K",
            "checkpoint_role": role,
            "training": copy.deepcopy(training),
            "metrics": copy.deepcopy(selections[role]["metrics"]),
            "model_state_sha256": state_hashes[role],
            "state_dict": {
                "weight": torch.tensor(
                    [1.0 if role == "best_miou" else 2.0]
                )
            },
        }
        for role in roles
    }
    expected_checkpoint_payloads = copy.deepcopy(checkpoint_payloads)
    published_paths = {
        "best_miou": tmp_path / "best_mIoU.pth.tar",
        "best_pd": tmp_path / "best_Pd.pth.tar",
    }
    for role, path in published_paths.items():
        torch.save(checkpoint_payloads[role], path)
    published = {
        role: {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": runner._sha256(path),
            "model_state_sha256": state_hashes[role],
        }
        for role, path in published_paths.items()
    }
    summary = {
        "schema": runner.SUMMARY_SCHEMA,
        "status": "complete",
        "model": runner.MODEL_NAME,
        "method": runner.METHOD_NAME,
        "dataset": "IRSTD-1K",
        "training": training,
        "candidate_count": 501,
        "selections": selections,
        "published_checkpoints": published,
        "two_distinct_physical_checkpoint_files": True,
        "data_role": "test",
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
        "baseline_reference_kind": "historical_existing",
        "method_config_path": training["method_config_path"],
        "method_config_sha256": training["method_config_sha256"],
        "gradient_authorization_path": training["gradient_authorization_path"],
        "gradient_authorization_sha256": training[
            "gradient_authorization_sha256"
        ],
        "baseline_authority_manifest_path": training[
            "baseline_authority_manifest_path"
        ],
        "baseline_authority_manifest_sha256": training[
            "baseline_authority_manifest_sha256"
        ],
        "formal_launch_authorization_path": training[
            "formal_launch_authorization_path"
        ],
        "formal_launch_authorization_sha256": training[
            "formal_launch_authorization_sha256"
        ],
        "training_source_manifest_sha256": training[
            "training_source_manifest_sha256"
        ],
        "selection_history": history,
        "formal_selection_provenance": provenance,
        "elapsed_seconds": 1.0,
    }
    summary_path = tmp_path / "summary.json"

    def write_summary() -> None:
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    write_summary()
    monkeypatch.setattr(
        runner,
        "_validate_completed_training_config",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        runner,
        "_validate_history",
        lambda *_args, **_kwargs: dict(roles),
    )
    monkeypatch.setattr(
        runner.selection,
        "select_final",
        lambda *_args, **_kwargs: copy.deepcopy(provenance),
    )
    monkeypatch.setattr(
        runner,
        "_state_dict_sha256",
        lambda state: (
            state_hashes["best_miou"]
            if float(state["weight"].item()) == 1.0
            else state_hashes["best_pd"]
        ),
    )
    monkeypatch.setattr(
        runner.core, "validate_sbsc_v33_state_dict", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        runner,
        "_slim_checkpoint",
        lambda *, role, **_kwargs: copy.deepcopy(
            expected_checkpoint_payloads[role]
        ),
    )

    runner._validate_completed_publication(
        summary_path,
        published_paths,
        "IRSTD-1K",
        authorities={},
    )

    checkpoint_payloads["best_miou"]["training"] = {"tampered": True}
    torch.save(checkpoint_payloads["best_miou"], published_paths["best_miou"])
    summary["published_checkpoints"]["best_miou"]["sha256"] = runner._sha256(
        published_paths["best_miou"]
    )
    write_summary()
    with pytest.raises(ValueError, match="metadata differs"):
        runner._validate_completed_publication(
            summary_path,
            published_paths,
            "IRSTD-1K",
            authorities={},
        )


def test_completed_cleanup_never_deletes_recovery_before_full_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    output_root = tmp_path / "runs" / runner.METHOD_NAME / "formal"
    run_dir = output_root / "IRSTD-1K"
    dataset_root = tmp_path / "datasets"
    run_dir.mkdir(parents=True)
    dataset_root.mkdir()
    summary = run_dir / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    for name in ("best_mIoU.pth.tar", "best_Pd.pth.tar"):
        (run_dir / name).touch()
    recoveries = [
        run_dir / "last_training_state.pth.tar",
        run_dir / "best_miou_state.pth.tar",
        run_dir / "best_pd_state.pth.tar",
    ]
    for path in recoveries:
        path.write_bytes(b"recovery")
    authorities = {
        "method_config": {
            "output_namespace": "runs/sbsc_v33_third/formal/IRSTD-1K"
        }
    }
    monkeypatch.setattr(runner, "_load_formal_preflight", lambda _dataset: authorities)
    monkeypatch.setattr(runner, "configure_determinism", lambda _seed: None)
    monkeypatch.setattr(runner, "require_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(
        runner,
        "_validate_completed_publication",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("full replay rejected")
        ),
    )
    args = SimpleNamespace(
        dataset="IRSTD-1K",
        dataset_root=dataset_root,
        output_root=output_root,
        seed=42,
        device="cpu",
        resume=True,
    )

    with pytest.raises(ValueError, match="full replay rejected"):
        runner.run(args)
    assert all(path.read_bytes() == b"recovery" for path in recoveries)
