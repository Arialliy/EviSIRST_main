from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from experiments import evisirst_v2_data as v2_data
from tools import smoke_sctransnet_sbsc_v32_train_only as smoke


def _ids(count: int) -> tuple[str, ...]:
    return tuple(f"XDU{index}" for index in range(count))


def _ordered_hash(values: tuple[str, ...]) -> str:
    return smoke._ordered_ids_sha256(values)


def _make_train_only_contract_tree(
    root: Path,
    *,
    source_count: int = 80,
    train_count: int = 64,
) -> tuple[Path, Path]:
    dataset_root = root / "datasets"
    split_root = root / "splits" / "v2"
    source_directory = dataset_root / smoke.DATASET / "img_idx"
    split_directory = split_root / smoke.DATASET
    source_directory.mkdir(parents=True)
    split_directory.mkdir(parents=True)
    source_ids = _ids(source_count)
    train_ids = source_ids[:train_count]
    source_content = "\n".join(source_ids).encode("utf-8")
    train_content = ("\n".join(train_ids) + "\n").encode("utf-8")
    source_relative = f"{smoke.DATASET}/img_idx/train_IRSTD-1K.txt"
    (dataset_root / source_relative).write_bytes(source_content)
    (split_directory / "train.txt").write_bytes(train_content)
    manifest = {
        "schema": v2_data.split_protocol.SCHEMA,
        "dataset": smoke.DATASET,
        "outputs": {
            "train": {
                "file_sha256": smoke._sha256_bytes(train_content),
                "ordered_ids_sha256": _ordered_hash(train_ids),
                "sample_count": len(train_ids),
            },
            # Deliberately present only as inert manifest metadata.  There is
            # no val.txt fixture, so a successful load proves it was not read.
            "val": {"sample_count": source_count - train_count},
        },
        "source_index": {
            "split": "train",
            "relative_path": source_relative,
            "file_sha256": smoke._sha256_bytes(source_content),
            "ordered_ids_sha256": _ordered_hash(source_ids),
            "sample_count": len(source_ids),
        },
        "validation": {"test_was_not_accessed": True},
        "test_access": {
            "test_index_opened": False,
            "test_image_opened": False,
            "test_mask_opened": False,
            "test_used_for_split_or_attributes": False,
        },
    }
    (split_directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return dataset_root, split_root


def _compact() -> dict[str, int | str]:
    return {
        "schema": smoke.COMPACT_DIAGNOSTICS_SCHEMA,
        "projection_rows": 32,
        "solver_attempt_rows": 20,
        "solver_accepted_rows": 18,
        "solver_fallback_rows": 2,
        "emission_checked_rows": 32,
        "emission_accepted_rows": 31,
        "emission_fallback_rows": 1,
    }


def _gate_epochs() -> list[dict]:
    records: list[dict] = []
    for epoch in range(1, 6):
        router_loss = 1.0 - 0.1 * epoch
        records.append(
            {
                "epoch": epoch,
                "loss": {
                    "total": 7.0 - 0.2 * epoch,
                    "segmentation": 6.0 - 0.1 * epoch,
                    "router": router_loss,
                },
                "roles": {
                    role: {
                        "valid_sample_level_count": 12,
                        "mean_valid_ce": router_loss + offset,
                        "median_max_mass": 0.2 + offset,
                    }
                    for role, offset in zip(smoke.ROLE_NAMES, (0.0, 0.01, 0.02))
                },
                "gradients": {
                    "router_all_present_and_finite": True,
                    "gain_all_present_and_finite": True,
                },
                "state": {
                    "router_state_finite": True,
                    "gain_state_finite": True,
                    "gain_in_bounds": True,
                    "gain": [0.01 * epoch, 0.0, 0.0, 0.0],
                },
                "compact_projection": {
                    **{key: value for key, value in _compact().items() if key != "schema"},
                    "solver_fallback_rate": 0.1,
                    "emission_fallback_rate": 1.0 / 32.0,
                },
            }
        )
    return records


def test_cli_exposes_only_data_split_device_and_output() -> None:
    parser = smoke.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert options == {"--dataset-root", "--split-root", "--device", "--output"}
    with pytest.raises(SystemExit):
        smoke.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--output",
                "/tmp/result.json",
                "--epochs",
                "4",
            ]
        )
    with pytest.raises(SystemExit):
        smoke.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--output",
                "/tmp/result.json",
                "--loss-weight",
                "0.5",
            ]
        )


def test_direct_script_help_works_from_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(smoke.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset-root" in completed.stdout
    assert "--split-root" in completed.stdout
    assert "--device" in completed.stdout
    assert "--output" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


def test_direct_script_bootstrap_defeats_hostile_model_path_precedence(
    tmp_path: Path,
) -> None:
    hostile_root = tmp_path / "hostile_checkout"
    hostile_model = hostile_root / "model"
    hostile_model.mkdir(parents=True)
    (hostile_model / "__init__.py").write_text(
        "ORIGIN = 'hostile-model'\n", encoding="utf-8"
    )
    script = Path(smoke.__file__).resolve(strict=True)
    probe = f"""
import json
import runpy
import sys
from pathlib import Path

namespace = runpy.run_path({str(script)!r})
core = namespace["_load_core"]()
import model
import model.SCTransNet as sctransnet
print(json.dumps({{
    "sys_path_0": sys.path[0],
    "core_source": str(Path(core.__file__).resolve(strict=True)),
    "model_source": str(Path(model.__file__).resolve(strict=True)),
    "sctransnet_source": str(Path(sctransnet.__file__).resolve(strict=True)),
}}, sort_keys=True))
"""
    hostile_pythonpath = os.pathsep.join(
        (
            str(hostile_root),
            "/home/ly/BasicIRSTD",
            str(smoke.PROJECT_ROOT),
            str(smoke.PROJECT_ROOT),
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": hostile_pythonpath},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    root = smoke.PROJECT_ROOT.resolve(strict=True)
    assert payload["sys_path_0"] == str(root)
    for key in ("core_source", "model_source", "sctransnet_source"):
        Path(payload[key]).relative_to(root)
    assert payload["core_source"].endswith(
        "/experiments/sctransnet_sbsc_v32.py"
    )
    assert payload["model_source"].endswith("/model/__init__.py")
    assert payload["sctransnet_source"].endswith(
        "/model/_internal/SCTransNet.py"
    )


def test_smoke_loader_rejects_same_path_stale_runtime_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = smoke._load_core()
    stale = ModuleType("experiments.sctransnet_sbsc_v32")
    stale.__file__ = real.__file__
    required = (
        "build_sctransnet_sbsc_v32_method",
        "validate_sctransnet_sbsc_v32",
        "capture_c3_v32_training_router",
        "tri_router_supervision_loss",
        "build_tri_router_targets_v32",
        "project_sbsc_v32_constraints_",
        "structurally_inactive_parameter_names",
        "_estimate_c3_v31_support_v32",
    )
    for name in required:
        setattr(stale, name, getattr(real, name))
    stale.SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA = (
        smoke.CUDA_MEDIAN_ADAPTER_SCHEMA
    )
    stale.EXPECTED_V31_SOLVER_SOURCE_SHA256 = (
        smoke.EXPECTED_V31_SOLVER_SOURCE_SHA256
    )
    stale.V31_SOLVER_SOURCE_SHA256 = smoke.EXPECTED_V31_SOLVER_SOURCE_SHA256
    stale.validate_sbsc_v32_runtime_integration = lambda: {
        "schema": "sctransnet_sbsc_v32/runtime_integration/stale",
        "cuda_median_adapter": smoke.CUDA_MEDIAN_ADAPTER_SCHEMA,
    }
    monkeypatch.setitem(
        sys.modules, "experiments.sctransnet_sbsc_v32", stale
    )
    with pytest.raises(
        smoke.StageCTrainOnlySmokeError,
        match="runtime integration contract differs",
    ):
        smoke._load_core()


def test_immutable_stage_c_constants() -> None:
    assert smoke.DATASET == "IRSTD-1K"
    assert smoke.TARGET_MODE == "binary"
    assert smoke.ARCHITECTURE_SEED == smoke.RUN_SEED == 42
    assert smoke.EPOCHS == 5
    assert smoke.SAMPLE_COUNT == 64
    assert smoke.BATCH_SIZE == 16
    assert smoke.WORKERS == 0
    assert smoke.ROUTER_LOSS_WEIGHT == 1.0
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert [smoke._learning_rate(epoch) for epoch in range(1, 6)] == pytest.approx(
        [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    )


def test_state_hash_preserves_scalar_int64_bool_and_float_raw_bytes() -> None:
    state = {
        "scalar_bool": torch.tensor(True, dtype=torch.bool),
        "scalar_float": torch.tensor(1.25, dtype=torch.float32),
        "scalar_int64": torch.tensor(-7, dtype=torch.int64),
    }
    raw_by_key = {
        "scalar_bool": struct.pack("=?", True),
        "scalar_float": struct.pack("=f", 1.25),
        "scalar_int64": struct.pack("=q", -7),
    }
    expected = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        descriptor = json.dumps(
            [key, str(value.dtype), list(value.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = raw_by_key[key]
        expected.update(len(descriptor).to_bytes(8, "big"))
        expected.update(descriptor)
        expected.update(len(raw).to_bytes(8, "big"))
        expected.update(raw)
    observed = smoke._state_dict_sha256(state)
    assert observed == expected.hexdigest()
    assert observed != smoke._state_dict_sha256(
        {**state, "scalar_bool": torch.tensor(False, dtype=torch.bool)}
    )


def test_state_hash_accepts_real_fresh_v32_model_scalar_buffers() -> None:
    core = smoke._load_core()
    model, metadata = smoke._build_fresh_model(core)
    assert metadata["cuda_strict_deterministic_median_adapter"] == (
        smoke.CUDA_MEDIAN_ADAPTER_SCHEMA
    )
    state = model.state_dict()
    scalar_buffers = {
        name: value for name, value in state.items() if value.ndim == 0
    }
    assert scalar_buffers
    assert any(value.dtype == torch.int64 for value in scalar_buffers.values())
    observed = smoke._state_dict_sha256(state)
    assert len(observed) == 64
    assert observed == metadata["state_sha256"]


def test_source_provenance_binds_stage_c_protocol_and_model_source_closure() -> None:
    source = smoke._source_provenance(smoke._load_core())
    assert source["schema"] == smoke.SOURCE_SET_SCHEMA
    required = {
        "stage_c_runner": "tools/smoke_sctransnet_sbsc_v32_train_only.py",
        "experiments_package": "experiments/__init__.py",
        "core_builder": "experiments/sctransnet_sbsc_v32.py",
        "frozen_v31_public_solver": "experiments/sctransnet_sbsc_v31.py",
        "paired_initialization_authority": (
            "experiments/four_dataset_models_seed42_v1.py"
        ),
        "v2_data": "experiments/evisirst_v2_data.py",
        "v2_split_validator": "experiments/evisirst_v2_splits.py",
        "three_dataset_source_protocol": (
            "experiments/three_dataset_v2_protocol.py"
        ),
        "model_package": "model/__init__.py",
        "model_entry": "model/EviSIRST.py",
        "sctransnet_config": "model/_internal/Config.py",
        "sctransnet": "model/_internal/SCTransNet.py",
    }
    assert required.keys() <= source["files"].keys()
    for name, relative_path in required.items():
        assert source["files"][name]["relative_path"] == relative_path

    internal_keys = {
        f"model_internal/{path.name}"
        for path in (smoke.PROJECT_ROOT / "model" / "_internal").glob("*.py")
    }
    assert internal_keys
    assert internal_keys <= source["files"].keys()

    root = smoke.PROJECT_ROOT.resolve(strict=True)
    for entry in source["files"].values():
        relative = Path(entry["relative_path"])
        assert not relative.is_absolute()
        path = (root / relative).resolve(strict=True)
        path.relative_to(root)
        assert path.is_file()
        assert not (root / relative).is_symlink()
        assert entry["sha256"] == smoke._sha256_file(path)
    assert source["source_tree_sha256"] == smoke._canonical_sha256(source["files"])


def test_source_provenance_rejects_repository_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = smoke._load_core()
    fake_root = tmp_path / "isolated_repo"
    internal_root = fake_root / "model" / "_internal"
    internal_root.mkdir(parents=True)
    (internal_root / "placeholder.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(smoke, "PROJECT_ROOT", fake_root.resolve(strict=True))
    with pytest.raises(
        smoke.StageCTrainOnlySmokeError,
        match="protocol source escapes repository",
    ):
        smoke._source_provenance(core)


def test_train_only_loader_succeeds_without_any_validation_or_test_file(
    tmp_path: Path,
) -> None:
    dataset_root, split_root = _make_train_only_contract_tree(tmp_path)
    assert not (split_root / smoke.DATASET / "val.txt").exists()
    assert not any("test" in path.name.lower() for path in dataset_root.rglob("*"))
    contract = smoke.load_train_only_contract(dataset_root, split_root)
    assert contract.selected_ids == _ids(64)
    assert contract.selected_ids_sha256 == _ordered_hash(_ids(64))
    assert contract.train_split_count == 64
    assert contract.source_train_count == 80
    assert len(contract.manifest_sha256) == 64
    assert len(contract.train_split_sha256) == 64
    assert len(contract.source_train_index_sha256) == 64


def test_train_only_loader_rejects_non_train_source_before_open(
    tmp_path: Path,
) -> None:
    dataset_root, split_root = _make_train_only_contract_tree(tmp_path)
    manifest_path = split_root / smoke.DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_index"]["relative_path"] = (
        f"{smoke.DATASET}/img_idx/test_IRSTD-1K.txt"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="strictly train-only"):
        smoke.load_train_only_contract(dataset_root, split_root)


def test_epoch_order_is_a_fixed_seed42_permutation() -> None:
    first = smoke.epoch_order(1)
    assert first == smoke.epoch_order(1)
    assert first != smoke.epoch_order(2)
    assert sorted(first) == list(range(64))
    with pytest.raises(smoke.StageCTrainOnlySmokeError):
        smoke.epoch_order(0)


def test_compact_diagnostics_fails_closed_when_missing_or_inconsistent() -> None:
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="omits"):
        smoke._extract_compact_diagnostics({})
    valid = smoke._extract_compact_diagnostics(
        {"compact_diagnostics": _compact()}
    )
    assert valid["solver_attempt_rows"] == 20
    bad = _compact()
    bad["solver_accepted_rows"] = 19
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="solver compact"):
        smoke._extract_compact_diagnostics({"compact_diagnostics": bad})
    bad = _compact()
    bad["emission_checked_rows"] = 31
    bad["emission_accepted_rows"] = 30
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="emission compact"):
        smoke._extract_compact_diagnostics({"compact_diagnostics": bad})


class _FakeTargetCore:
    @staticmethod
    def build_tri_router_targets_v32(
        masks: torch.Tensor,
        prediction: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> SimpleNamespace:
        del prediction
        batch = int(masks.shape[0])
        height, width = token_hw
        probability = torch.full(
            (batch, 3, height, width),
            1.0 / float(height * width),
            device=masks.device,
        )
        valid = torch.ones((batch, 3), dtype=torch.bool, device=masks.device)
        return SimpleNamespace(probability=probability, valid=valid)


def test_capture_measurements_are_per_sample_level_role() -> None:
    masks = torch.zeros(2, 1, 8, 8)
    prediction = torch.full_like(masks, 0.2)
    logits = tuple(torch.zeros(2, 3, 2, 2) for _ in range(4))
    capture = SimpleNamespace(
        records=[{"logits": logits, "compact_diagnostics": _compact()}]
    )
    measurements = smoke._capture_measurements(
        core=_FakeTargetCore,
        capture=capture,
        masks=masks,
        detached_prediction=prediction,
    )
    for role in smoke.ROLE_NAMES:
        assert len(measurements["role_ce"][role]) == 8
        assert len(measurements["role_max_mass"][role]) == 8
        assert measurements["role_ce"][role] == pytest.approx([1.0] * 8)
        assert measurements["role_max_mass"][role] == pytest.approx([0.25] * 8)
    assert measurements["all_valid_ce_mean"] == pytest.approx(1.0)


def test_finalize_epoch_records_weighted_losses_roles_and_rates() -> None:
    accumulator = smoke._empty_epoch_accumulator()
    accumulator["processed"] = 64
    accumulator["batches"] = 4
    accumulator["loss"] = {
        "total": 64.0 * 7.0,
        "segmentation": 64.0 * 6.0,
        "router": 64.0,
    }
    for role, values in {
        "C": [0.9, 0.7],
        "H": [0.8, 0.6],
        "B": [0.4, 0.2],
    }.items():
        accumulator["ce"][role] = values
        accumulator["max_mass"][role] = [0.1, 0.3]
    accumulator["router_grad_max_abs"] = 2.0
    accumulator["gain_grad_max_abs"] = 3.0
    accumulator["compact"] = {
        key: int(value)
        for key, value in _compact().items()
        if key != "schema"
    }
    record = smoke._finalize_epoch(
        1,
        1e-4,
        tuple(range(64)),
        accumulator,
        {
            "router_state_finite": True,
            "gain_state_finite": True,
            "gain_in_bounds": True,
            "gain": [0.1, 0.0, 0.0, 0.0],
        },
    )
    assert record["loss"] == pytest.approx(
        {"total": 7.0, "segmentation": 6.0, "router": 1.0}
    )
    assert record["roles"]["C"]["mean_valid_ce"] == pytest.approx(0.8)
    assert record["roles"]["H"]["median_max_mass"] == pytest.approx(0.2)
    assert record["compact_projection"]["solver_fallback_rate"] == pytest.approx(
        0.1
    )
    assert record["compact_projection"]["emission_fallback_rate"] == pytest.approx(
        1.0 / 32.0
    )


def test_pre_registered_gate_go_and_failures() -> None:
    records = _gate_epochs()
    gate = smoke.evaluate_stage_c_gate(records)
    assert gate["verdict"] == "GO"
    assert gate["failed_checks"] == []

    missing_c = _gate_epochs()
    missing_c[-1]["roles"]["C"]["valid_sample_level_count"] = 0
    missing_c[-1]["roles"]["C"]["mean_valid_ce"] = None
    missing_c[-1]["roles"]["C"]["median_max_mass"] = None
    gate = smoke.evaluate_stage_c_gate(missing_c)
    assert gate["verdict"] == "NO-GO"
    assert "C_valid_at_epoch_1_and_5" in gate["failed_checks"]
    assert "C_mean_valid_ce_decreased" in gate["failed_checks"]

    collapsed_h = _gate_epochs()
    collapsed_h[-1]["roles"]["H"]["median_max_mass"] = 0.999
    gate = smoke.evaluate_stage_c_gate(collapsed_h)
    assert gate["verdict"] == "NO-GO"
    assert "H_epoch_5_max_mass_not_collapsed_when_applicable" in gate[
        "failed_checks"
    ]

    zero_gain = _gate_epochs()
    zero_gain[-1]["state"]["gain"] = [0.0, 0.0, 0.0, 0.0]
    assert smoke.evaluate_stage_c_gate(zero_gain)["verdict"] == "NO-GO"


def test_gate_allows_null_solver_rate_only_when_no_solver_attempts() -> None:
    records = _gate_epochs()
    for record in records:
        record["compact_projection"].update(
            {
                "solver_attempt_rows": 0,
                "solver_accepted_rows": 0,
                "solver_fallback_rows": 0,
                "solver_fallback_rate": None,
            }
        )
    assert smoke.evaluate_stage_c_gate(records)["verdict"] == "GO"


def test_gate_rejects_missing_inconsistent_or_nonfinite_fallback_rates() -> None:
    missing = _gate_epochs()
    del missing[0]["compact_projection"]["solver_fallback_rate"]
    gate = smoke.evaluate_stage_c_gate(missing)
    assert gate["verdict"] == "NO-GO"
    assert "solver_emission_rates_recorded" in gate["failed_checks"]

    inconsistent = _gate_epochs()
    inconsistent[0]["compact_projection"]["solver_fallback_rate"] = 0.2
    assert smoke.evaluate_stage_c_gate(inconsistent)["verdict"] == "NO-GO"

    nonfinite = _gate_epochs()
    nonfinite[0]["compact_projection"]["emission_fallback_rate"] = math.nan
    assert smoke.evaluate_stage_c_gate(nonfinite)["verdict"] == "NO-GO"

    zero_attempt_wrong = _gate_epochs()
    zero_attempt_wrong[0]["compact_projection"].update(
        {
            "solver_attempt_rows": 0,
            "solver_accepted_rows": 0,
            "solver_fallback_rows": 0,
            "solver_fallback_rate": 0.0,
        }
    )
    assert smoke.evaluate_stage_c_gate(zero_attempt_wrong)["verdict"] == "NO-GO"


def test_strict_json_rejects_nan_and_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    smoke.write_strict_json(output, {"finite": 1.25, "null": None})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "finite": 1.25,
        "null": None,
    }
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="strict-JSON"):
        smoke.write_strict_json(output, {"bad": math.nan})


def test_existing_output_is_immutable_and_never_overwritten(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    split_root = tmp_path / "splits"
    dataset_root.mkdir()
    split_root.mkdir()
    output = tmp_path / "stage_c.json"
    output.write_text("preserve-me", encoding="utf-8")
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="already exists"):
        smoke._validate_output_path(
            output, dataset_root=dataset_root, split_root=split_root
        )
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="already exists"):
        smoke.write_strict_json(output, {"replacement": True})
    assert output.read_text(encoding="utf-8") == "preserve-me"


def test_cuda_workspace_is_rejected_before_cuda_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    def forbidden_query() -> bool:
        raise AssertionError("CUDA query occurred before workspace verification")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden_query)
    with pytest.raises(smoke.StageCTrainOnlySmokeError, match="CUBLAS"):
        smoke._require_device("cuda:0")


def test_main_failure_is_a_no_go_artifact_with_no_val_or_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "data"
    split_root = tmp_path / "splits"
    dataset_root.mkdir()
    split_root.mkdir()
    output = tmp_path / "out" / "stage_c.json"

    def fail(_args: argparse.Namespace) -> dict:
        raise smoke.StageCTrainOnlySmokeError("compact diagnostics missing")

    monkeypatch.setattr(smoke, "run_stage_c", fail)
    return_code = smoke.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--split-root",
            str(split_root),
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    assert return_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution_status"] == "failed_closed"
    assert payload["gate"]["verdict"] == "NO-GO"
    assert payload["validation_split_accessed"] is False
    assert payload["validation_dataset_constructed"] is False
    assert payload["test_split_accessed"] is False
    assert payload["test_dataset_constructed"] is False
    assert payload["official_test_accessed"] is False
    assert payload["checkpoint_written"] is False


def test_tool_has_no_validation_or_test_dataset_constructor_reference() -> None:
    source = Path(smoke.__file__).read_text(encoding="utf-8")
    assert "EviSIRSTV2ValDataset" not in source
    assert "EviSIRSTV2TestDataset" not in source
    assert "build_validation" not in source
    assert "evaluate_model" not in source
