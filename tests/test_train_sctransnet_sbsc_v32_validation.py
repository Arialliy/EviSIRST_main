from __future__ import annotations

import tempfile
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import train_sctransnet_sbsc_v32_validation as runner


def _args_for_method(method: str, *extra: str):
    return runner.parse_args(
        [
            "--method",
            method,
            "--dataset",
            "IRSTD-1K",
            "--dataset-root",
            "/does/not/get/opened/by_parse",
            "--target-mode",
            "binary",
            *extra,
        ]
    )


def _args(*extra: str):
    return _args_for_method("sctransnet", *extra)


def _record(
    epoch: int,
    *,
    miou: float = 0.7,
    niou: float = 0.69,
    pd: float = 0.9,
    fa: float = 1e-5,
    tiny_pd: float = 0.8,
    loss: float = 0.2,
    model_state_sha256: str = "c" * 64,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "data_role": "val",
        "mIoU": miou,
        "nIoU": niou,
        "Pd": pd,
        "Fa": fa,
        "tinyPd": tiny_pd,
        "loss": loss,
        "evaluation_head": "out",
        "tiny_pd_defined": True,
        "metrics": {
            "miou": miou,
            "niou": niou,
            "pixel_precision": 0.75,
            "pixel_recall": 0.85,
            "pixel_f1": 0.796875,
            "pd": pd,
            "fa": fa,
            "tiny_pd": tiny_pd,
            "false_objects_per_image": 0.125,
            "validation_loss": loss,
        },
        "method": "sctransnet",
        "dataset": "IRSTD-1K",
        "architecture_seed": 42,
        "run_seed": 42,
        "split_manifest_sha256": "b" * 64,
        "run_identity_sha256": "a" * 64,
        "model_state_sha256": model_state_sha256,
        "test_split_accessed": False,
    }


def _identity() -> dict[str, object]:
    return {
        "schema": runner.TRAINING_SCHEMA + "/run_identity",
        "model": "SCTransNet",
        "method": "sctransnet",
        "dataset": "IRSTD-1K",
        "architecture_seed": 42,
        "run_seed": 42,
        "manifest_sha256": "b" * 64,
        "identity_sha256": "a" * 64,
        "test_split_accessed": False,
    }


class _NormalizationSpec:
    def as_dict(self):
        return {"mode": "legacy", "source": "fixture", "mean": 0.0, "std": 1.0}


class _TrainFixture(Dataset):
    def __init__(self, contract, access_log):
        self.contract = contract
        self.access_log = access_log
        self.normalization = {"mean": 0.0, "std": 1.0}
        self.normalization_spec = _NormalizationSpec()
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.access_log.append("train")
        return torch.zeros(1, 8, 8), torch.zeros(1, 8, 8)


class _ValFixture(Dataset):
    def __init__(self, contract, access_log):
        self.contract = contract
        self.access_log = access_log

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.access_log.append("val")
        return torch.zeros(1, 8, 8), torch.zeros(1, 8, 8), (8, 8), "val"


class _SixHeadFixture(nn.Module):
    def __init__(self, method: str = "sctransnet"):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        if method == "sbsc_v32":
            self.mtc = nn.Module()
            self.mtc.encoder = nn.Module()
            self.mtc.encoder.layer = nn.ModuleList([nn.Module(), nn.Module()])
            self.mtc.encoder.layer[1].channel_attn = nn.Module()
            self.mtc.encoder.layer[1].channel_attn.register_parameter(
                "raw_dual_risk_level_gain",
                nn.Parameter(torch.zeros(4, dtype=torch.float32)),
            )
            tri_router = nn.Module()
            tri_router.value_proj = nn.Conv2d(
                480, 8, kernel_size=1, bias=False
            )
            tri_router.head = nn.Conv2d(
                15, 3, kernel_size=3, padding=1, bias=False
            )
            self.mtc.encoder.layer[1].channel_attn.tri_router = tri_router
        self.mode = "train"

    def forward(self, images):
        probability = torch.sigmoid(images * 0.0 + self.bias)
        return tuple(probability for _ in range(6)) if self.mode == "train" else probability


def test_v32_training_loss_is_six_bce_plus_unit_router_auxiliary() -> None:
    model = _SixHeadFixture("sbsc_v32")
    observations: dict[str, object] = {"capture_calls": 0}

    @contextmanager
    def capture(candidate):
        observations["capture_calls"] = int(observations["capture_calls"]) + 1
        yield candidate

    def router_loss(candidate, prediction, target):
        observations["prediction_requires_grad"] = prediction.requires_grad
        observations["target_shape"] = tuple(target.shape)
        return (candidate.bias - 0.25).square()

    core = SimpleNamespace(
        capture_c3_v32_training_router=capture,
        tri_router_supervision_loss=router_loss,
    )
    images = torch.zeros(1, 1, 8, 8)
    masks = torch.zeros_like(images)
    total, segmentation, auxiliary = runner._training_losses(
        model=model,
        images=images,
        masks=masks,
        criterion=nn.BCELoss(reduction="mean"),
        method="sbsc_v32",
        core=core,
    )
    assert observations == {
        "capture_calls": 1,
        "prediction_requires_grad": False,
        "target_shape": (1, 1, 8, 8),
    }
    assert torch.equal(total, segmentation + auxiliary)
    assert float(auxiliary.detach()) == pytest.approx(0.0625)


def test_baseline_training_loss_never_enters_router_capture() -> None:
    model = _SixHeadFixture("sctransnet")

    def forbidden(*args, **kwargs):
        raise AssertionError("baseline reached V3.2 router API")

    core = SimpleNamespace(
        capture_c3_v32_training_router=forbidden,
        tri_router_supervision_loss=forbidden,
    )
    images = torch.zeros(1, 1, 8, 8)
    masks = torch.zeros_like(images)
    total, segmentation, auxiliary = runner._training_losses(
        model=model,
        images=images,
        masks=masks,
        criterion=nn.BCELoss(reduction="mean"),
        method="sctransnet",
        core=core,
    )
    assert torch.equal(total, segmentation)
    assert float(auxiliary) == 0.0


def test_formal_cli_freezes_seed_schedule_methods_and_method_path() -> None:
    args = _args()
    assert args.method == "sctransnet"
    assert args.architecture_seed == args.run_seed == 42
    assert args.epochs == 1000
    assert args.val_interval == 1
    paths = runner.resolve_run_paths(args)
    relative = Path(paths["run_dir"]).relative_to(runner.DEFAULT_OUTPUT_ROOT)
    assert relative == Path("formal/sctransnet/IRSTD-1K/binary/run_seed_42")
    assert Path(paths["best_mIoU_final"]).name == "best_mIoU.pth.tar"
    assert Path(paths["best_Pd_final"]).name == "best_Pd.pth.tar"

    with pytest.raises(SystemExit):
        _args("--run-seed", "43")
    with pytest.raises(SystemExit):
        _args("--architecture-seed", "43")
    with pytest.raises(SystemExit):
        _args("--epochs", "999")
    with pytest.raises(SystemExit) as soft_formal:
        runner.parse_args(
            [
                "--method",
                "sctransnet",
                "--dataset",
                "IRSTD-1K",
                "--dataset-root",
                "/tmp/data",
                "--target-mode",
                "soft",
            ]
        )
    assert soft_formal.value.code == 2
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--method",
                "not_a_method",
                "--dataset",
                "IRSTD-1K",
                "--dataset-root",
                "/tmp/data",
                "--target-mode",
                "binary",
            ]
        )


def test_preflight_is_read_only_and_declares_fresh_or_resume() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        output_root = Path(temporary) / "declared-output"
        args = _args("--output-root", str(output_root), "--preflight")
        manifest = runner.preflight_manifest(args)
        assert manifest["execution_mode"] == "fresh"
        assert manifest["run_kind"] == "formal"
        assert manifest["stage_c_train_only_smoke"] is False
        assert manifest["validation_begin_epoch"] == 500
        assert manifest["validation_record_count"] == 501
        assert manifest["selection_roles"] == ["best_mIoU", "best_Pd"]
        assert manifest["core_imported"] is False
        assert manifest["writes_performed"] is False
        assert manifest["test_split_accessed"] is False
        assert not output_root.exists()

        resumed = _args(
            "--output-root", str(output_root), "--preflight", "--resume"
        )
        assert runner.preflight_manifest(resumed)["execution_mode"] == "resume"


def test_epoch_prefix_is_empty_before_500_and_formal_schedule_has_501_records() -> None:
    assert runner.expected_validation_epochs(
        0, total_epochs=1000, smoke=False
    ) == ()
    assert runner.expected_validation_epochs(
        499, total_epochs=1000, smoke=False
    ) == ()
    assert runner.expected_validation_epochs(
        500, total_epochs=1000, smoke=False
    ) == (500,)
    schedule = runner.expected_validation_epochs(
        1000, total_epochs=1000, smoke=False
    )
    assert schedule[0] == 500
    assert schedule[-1] == 1000
    assert len(schedule) == 501

    assert runner._select_prefix(
        [],
        completed_epoch=499,
        total_epochs=1000,
        smoke=False,
        expected_identity=runner._selection_identity(_identity()),
    )["roles"] == {}
    with pytest.raises(runner.SCTransNetSBSCV32RunnerError):
        runner._select_prefix(
            [_record(499)],
            completed_epoch=499,
            total_epochs=1000,
            smoke=False,
            expected_identity=runner._selection_identity(_identity()),
        )


def test_smoke_can_be_short_but_is_not_the_formal_schedule() -> None:
    args = _args(
        "--epochs",
        "2",
        "--warmup-epochs",
        "0",
        "--smoke-max-train-samples",
        "1",
        "--smoke-max-val-samples",
        "1",
        "--device",
        "cpu",
    )
    assert runner.validation_begin_epoch(args.epochs, smoke=True) == 2
    assert runner.expected_validation_epochs(
        1, total_epochs=2, smoke=True
    ) == ()
    assert runner.expected_validation_epochs(
        2, total_epochs=2, smoke=True
    ) == (2,)
    manifest = runner.preflight_manifest(args)
    assert manifest["run_kind"] == "runner_fixture_smoke"
    assert manifest["stage_c_train_only_smoke"] is False

    # The soft-target escape hatch is confined to explicitly marked fixture
    # smoke runs; it cannot be parsed as a formal run.
    soft_smoke = runner.parse_args(
        [
            "--method",
            "sctransnet",
            "--dataset",
            "IRSTD-1K",
            "--dataset-root",
            "/tmp/data",
            "--target-mode",
            "soft",
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--smoke-max-train-samples",
            "1",
            "--smoke-max-val-samples",
            "1",
        ]
    )
    soft_manifest = runner.preflight_manifest(soft_smoke)
    assert soft_manifest["run_kind"] == "runner_fixture_smoke"
    assert soft_manifest["target_mode"] == "soft"


def test_paired_methods_use_exactly_the_same_shuffle_generator_stream() -> None:
    baseline = _args()
    candidate = SimpleNamespace(**{**vars(baseline), "method": "sbsc_v32"})
    observed_seeds: list[int] = []
    for epoch in (1, 2, 500, 1000):
        left = runner.training_shuffle_generator(baseline, epoch)
        right = runner.training_shuffle_generator(candidate, epoch)
        assert torch.equal(left.get_state(), right.get_state())
        assert torch.equal(
            torch.randperm(37, generator=left),
            torch.randperm(37, generator=right),
        )
        observed_seeds.append(runner.training_shuffle_seed("IRSTD-1K", epoch))
    assert len(set(observed_seeds)) == len(observed_seeds)
    assert runner.SHUFFLE_STREAM == "sctransnet_sbsc_v32_pair"


def test_validation_record_always_discloses_no_test_and_handles_no_tiny_target() -> None:
    record = runner.build_validation_record(
        500,
        {
            "miou": 0.7,
            "niou": 0.69,
            "pd": 0.9,
            "fa": 1e-5,
            "tiny_pd": None,
            "pixel_precision": 0.75,
            "pixel_recall": 0.85,
            "pixel_f1": 0.796875,
            "false_objects_per_image": 0.125,
            "validation_loss": 0.2,
        },
        selection_identity=runner._selection_identity(_identity()),
        model_state_sha256="d" * 64,
    )
    assert record["test_split_accessed"] is False
    assert record["tinyPd"] == 0.0
    assert record["tiny_pd_defined"] is False
    assert {
        "mIoU",
        "nIoU",
        "Pd",
        "Fa",
        "F1",
        "Precision",
        "Recall",
        "tinyPd",
        "false_objects_per_image",
        "loss",
    }.issubset(record)
    assert runner._select_prefix(
        [record],
        completed_epoch=500,
        total_epochs=1000,
        smoke=False,
        expected_identity=runner._selection_identity(_identity()),
    )["retention_frontier_epochs"] == [500]


class _LoadTrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self._expected = {
            key: torch.zeros(4, dtype=torch.float32)
            for key in runner.GAIN_STATE_KEYS
        }
        self._expected.update(
            {
                key: torch.zeros(shape, dtype=dtype)
                for key, (shape, dtype) in runner.ROUTER_STATE_SCHEMA.items()
            }
        )

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {key: value.clone() for key, value in self._expected.items()}

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        self.load_calls += 1
        return torch.nn.modules.module._IncompatibleKeys([], [])


class _NoopCore:
    @staticmethod
    def validate_sbsc_v32_state_dict(state, method):
        return {"method": method}

    @staticmethod
    def validate_sctransnet_sbsc_v32(model, require_zero_gain=False):
        return {}

    @staticmethod
    def project_sbsc_v32_constraints_(model):
        return None


def test_out_of_range_gain_is_rejected_before_live_load_state_dict() -> None:
    model = _LoadTrackingModel()
    state = model.state_dict()
    state[runner.GAIN_STATE_KEYS[0]] = torch.tensor(
        [0.0, 0.0, 0.250001, 0.0], dtype=torch.float32
    )
    gain_contract = {
        "state_keys": list(runner.GAIN_STATE_KEYS),
        "bounds": [0.0, 0.25],
    }
    with pytest.raises(
        runner.SCTransNetSBSCV32RunnerError, match=r"outside \[0, 0.25\]"
    ):
        runner._load_model_state(
            model,
            state,
            method="sbsc_v32",
            gain_contract=gain_contract,
            core=_NoopCore(),  # type: ignore[arg-type]
        )
    assert model.load_calls == 0


def test_frozen_v32_gain_router_contract_and_legacy_state_are_distinct() -> None:
    assert runner.GAIN_MIN == 0.0
    assert runner.GAIN_MAX == 0.25
    assert runner.GAIN_STATE_KEYS == (
        "mtc.encoder.layer.1.channel_attn.raw_dual_risk_level_gain",
    )
    assert runner.EXPECTED_SBSC_V32_STATE_KEY_COUNT == 513
    assert runner.EXPECTED_SBSC_V32_PARAMETER_COUNT == 11_330_188
    assert runner.ROUTER_PARAMETER_COUNT == 4_245
    assert set(runner.ROUTER_STATE_SCHEMA) == {
        "mtc.encoder.layer.1.channel_attn.tri_router.value_proj.weight",
        "mtc.encoder.layer.1.channel_attn.tri_router.head.weight",
    }
    assert runner._cuda_median_adapter_contract(
        "sbsc_v32",
        {
            "cuda_strict_deterministic_median_adapter": (
                runner.CUDA_MEDIAN_ADAPTER_SCHEMA
            )
        },
    ) == runner.CUDA_MEDIAN_ADAPTER_SCHEMA
    assert runner._cuda_median_adapter_contract("sctransnet", {}) is None
    with pytest.raises(
        runner.SCTransNetSBSCV32RunnerError,
        match="adapter schema differs",
    ):
        runner._cuda_median_adapter_contract("sbsc_v32", {})
    with pytest.raises(
        runner.SCTransNetSBSCV32RunnerError,
        match="baseline builder unexpectedly",
    ):
        runner._cuda_median_adapter_contract(
            "sctransnet",
            {
                "cuda_strict_deterministic_median_adapter": (
                    runner.CUDA_MEDIAN_ADAPTER_SCHEMA
                )
            },
        )

    router_state = {
        key: torch.zeros(shape, dtype=dtype)
        for key, (shape, dtype) in runner.ROUTER_STATE_SCHEMA.items()
    }
    router_contract = runner._router_contract(
        "sbsc_v32",
        {
            "router_state_keys": list(runner.ROUTER_STATE_SCHEMA),
            "router_parameter_count": 4_245,
            "loss_schema": runner.TOTAL_LOSS_SCHEMA,
        },
        router_state,
    )
    assert router_contract["auxiliary_loss_weight"] == 1.0
    assert router_contract["parameter_count"] == 4_245

    state_with_forbidden_bias = dict(router_state)
    state_with_forbidden_bias[
        "mtc.encoder.layer.1.channel_attn.tri_router.head.bias"
    ] = torch.zeros(3)
    with pytest.raises(
        runner.SCTransNetSBSCV32RunnerError,
        match="state-key contract differs",
    ):
        runner._router_contract(
            "sbsc_v32",
            {
                "router_state_keys": list(runner.ROUTER_STATE_SCHEMA),
                "router_parameter_count": 4_245,
                "loss_schema": runner.TOTAL_LOSS_SCHEMA,
            },
            state_with_forbidden_bias,
        )

    model = _LoadTrackingModel()
    v2_state = {
        "mtc.encoder.layer.0.channel_attn.raw_transport_gain": torch.tensor(0.1)
    }
    with pytest.raises(runner.SCTransNetSBSCV32RunnerError):
        runner._load_model_state(
            model,
            v2_state,
            method="sbsc_v32",
            gain_contract={
                "state_keys": list(runner.GAIN_STATE_KEYS),
                "bounds": [0.0, 0.25],
            },
            core=_NoopCore(),  # type: ignore[arg-type]
        )
    assert model.load_calls == 0

    v31_state = {
        runner.GAIN_STATE_KEYS[0]: torch.zeros(4, dtype=torch.float32)
    }
    with pytest.raises(
        runner.SCTransNetSBSCV32RunnerError,
        match="checkpoint state keys differ",
    ):
        runner._load_model_state(
            model,
            v31_state,
            method="sbsc_v32",
            gain_contract={
                "state_keys": list(runner.GAIN_STATE_KEYS),
                "bounds": [0.0, 0.25],
            },
            core=_NoopCore(),  # type: ignore[arg-type]
        )
    assert model.load_calls == 0


def test_v2_resume_schema_is_rejected_before_live_state_load() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    model = _LoadTrackingModel()
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        resume_path = run_dir / "last_training_state.pth.tar"
        runner.r1._atomic_torch_save(
            resume_path,
            {
                "schema": "sctransnet_sbsc_v2_validation_training/v1",
                "run_identity": _identity(),
                "method": "sbsc_v2",
                "test_split_accessed": False,
            },
        )
        with pytest.raises(
            runner.SCTransNetSBSCV32RunnerError,
            match="resume run identity differs",
        ):
            runner._load_resume_state(
                path=resume_path,
                run_dir=run_dir,
                candidate_dir=run_dir / "candidates",
                identity=_identity(),
                model=model,
                optimizer=optimizer,
                device=torch.device("cpu"),
                core=_NoopCore(),  # type: ignore[arg-type]
                gain_contract={
                    "state_keys": list(runner.GAIN_STATE_KEYS),
                    "bounds": [0.0, 0.25],
                },
                total_epochs=1000,
                smoke=False,
            )
    assert model.load_calls == 0


def test_candidate_history_latest_transaction_order() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    identity = _identity()
    state = {"weight": torch.tensor([1.0])}
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    events: list[str] = []
    original_save = runner.r1._atomic_torch_save
    original_json = runner.r1._write_json

    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        candidate_dir = run_dir / "candidates"
        history_path = run_dir / "validation_history.json"
        latest_path = run_dir / "last_training_state.pth.tar"

        def traced_save(path, payload):
            path = Path(path)
            events.append("candidate" if path.parent == candidate_dir else "latest")
            return original_save(path, payload)

        def traced_json(path, payload):
            events.append("history")
            return original_json(path, payload)

        with mock.patch.object(
            runner.r1, "_atomic_torch_save", side_effect=traced_save
        ), mock.patch.object(runner.r1, "_write_json", side_effect=traced_json):
            artifacts = runner._commit_epoch(
                epoch=500,
                state=state,
                optimizer=optimizer,
                device=torch.device("cpu"),
                identity=identity,
                training_history=[{"epoch": 500}],
                validation_history=[
                    _record(500, model_state_sha256=runner._state_dict_sha256(state))
                ],
                validation_record=_record(
                    500, model_state_sha256=runner._state_dict_sha256(state)
                ),
                run_dir=run_dir,
                candidate_dir=candidate_dir,
                history_path=history_path,
                latest_path=latest_path,
                expected_state=state,
                gain_contract={"state_keys": (), "bounds": None},
                total_epochs=1000,
                smoke=False,
            )
        assert events[:3] == ["candidate", "history", "latest"]
        assert tuple(artifacts) == (500,)
        assert history_path.is_file()
        assert latest_path.is_file()


def test_nonfinite_live_adam_moment_is_rejected_before_any_commit_write() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    parameter = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["exp_avg"].fill_(float("inf"))
    state = {"weight": torch.tensor([1.0])}
    record = _record(1, model_state_sha256=runner._state_dict_sha256(state))
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        with pytest.raises(
            runner.SCTransNetSBSCV32RunnerError,
            match="optimizer state contains a non-finite tensor",
        ):
            runner._commit_epoch(
                epoch=1,
                state=state,
                optimizer=optimizer,
                device=torch.device("cpu"),
                identity=_identity(),
                training_history=[{"epoch": 1}],
                validation_history=[record],
                validation_record=record,
                run_dir=run_dir,
                candidate_dir=run_dir / "candidates",
                history_path=run_dir / "validation_history.json",
                latest_path=run_dir / "last_training_state.pth.tar",
                expected_state=state,
                gain_contract={"state_keys": (), "bounds": None},
                total_epochs=1,
                smoke=True,
            )
        assert not (run_dir / "candidates").exists()
        assert not (run_dir / "validation_history.json").exists()
        assert not (run_dir / "last_training_state.pth.tar").exists()


def test_crash_recovery_validates_and_removes_only_next_uncommitted_candidate() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    identity = _identity()
    state = {"weight": torch.tensor([1.0])}
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        candidate_dir = run_dir / "candidates"
        path = runner._candidate_path(candidate_dir, 500)
        runner._save_candidate(
            path=path,
            epoch=500,
            state=state,
            identity=identity,
            record=_record(
                500, model_state_sha256=runner._state_dict_sha256(state)
            ),
        )
        assert path.exists()
        restored = runner._validate_resume_candidates(
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            identity=identity,
            history=[],
            artifacts={},
            expected_state=state,
            gain_contract={"state_keys": (), "bounds": None},
            completed_epoch=499,
            total_epochs=1000,
            smoke=False,
        )
        assert restored == {}
        assert not path.exists()

        bad = runner._candidate_path(candidate_dir, 502)
        runner._save_candidate(
            path=bad,
            epoch=502,
            state=state,
            identity=identity,
            record=_record(
                502, model_state_sha256=runner._state_dict_sha256(state)
            ),
        )
        with pytest.raises(
            runner.SCTransNetSBSCV32RunnerError,
            match="not the next crash-recovery epoch",
        ):
            runner._validate_resume_candidates(
                run_dir=run_dir,
                candidate_dir=candidate_dir,
                identity=identity,
                history=[],
                artifacts={},
                expected_state=state,
                gain_contract={"state_keys": (), "bounds": None},
                completed_epoch=499,
                total_epochs=1000,
                smoke=False,
            )


def test_resume_cleans_only_hard_crash_candidate_mkstemp_partial() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    state = {"weight": torch.tensor([1.0])}
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        candidate_dir = run_dir / "candidates"
        candidate_dir.mkdir()
        target = runner._candidate_path(candidate_dir, 500)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=candidate_dir
        )
        try:
            os.write(descriptor, b"partial-torch-save")
        finally:
            os.close(descriptor)
        partial = Path(temporary_name)
        assert partial.is_file()

        restored = runner._validate_resume_candidates(
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            identity=_identity(),
            history=[],
            artifacts={},
            expected_state=state,
            gain_contract={"state_keys": (), "bounds": None},
            completed_epoch=499,
            total_epochs=1000,
            smoke=False,
        )
        assert restored == {}
        assert not partial.exists()


def test_resume_rejects_exact_candidate_temp_symlink_without_unlinking() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    state = {"weight": torch.tensor([1.0])}
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        candidate_dir = run_dir / "candidates"
        candidate_dir.mkdir()
        victim = run_dir / "victim.bin"
        victim.write_bytes(b"do-not-delete")
        hostile = candidate_dir / ".epoch_0500.pth.tar.abcdefgh.tmp"
        hostile.symlink_to(victim)

        with pytest.raises(
            runner.SCTransNetSBSCV32RunnerError,
            match="candidate entry is not a regular file",
        ):
            runner._validate_resume_candidates(
                run_dir=run_dir,
                candidate_dir=candidate_dir,
                identity=_identity(),
                history=[],
                artifacts={},
                expected_state=state,
                gain_contract={"state_keys": (), "bounds": None},
                completed_epoch=499,
                total_epochs=1000,
                smoke=False,
            )
        assert hostile.is_symlink()
        assert victim.read_bytes() == b"do-not-delete"


@pytest.mark.parametrize(
    "name",
    (
        ".epoch_0500.pth.tar.short.tmp",
        ".epoch_500.pth.tar.abcdefgh.tmp",
        "epoch_0500.pth.tar.abcdefgh.tmp",
        ".epoch_0500.pth.tar.ABCDEFGH.tmp",
        ".epoch_0500.pth.tar.abcdefgh.tmp.extra",
        ".epoch_0501.pth.tar.abcdefgh.tmp",
    ),
)
def test_resume_rejects_candidate_temp_near_matches(name: str) -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    state = {"weight": torch.tensor([1.0])}
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        candidate_dir = run_dir / "candidates"
        candidate_dir.mkdir()
        hostile = candidate_dir / name
        hostile.write_bytes(b"hostile-or-ambiguous")

        with pytest.raises(
            runner.SCTransNetSBSCV32RunnerError,
            match="unexpected filename",
        ):
            runner._validate_resume_candidates(
                run_dir=run_dir,
                candidate_dir=candidate_dir,
                identity=_identity(),
                history=[],
                artifacts={},
                expected_state=state,
                gain_contract={"state_keys": (), "bounds": None},
                completed_epoch=499,
                total_epochs=1000,
                smoke=False,
            )
        assert hostile.read_bytes() == b"hostile-or-ambiguous"


def test_same_epoch_winners_still_emit_two_physical_role_weights() -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        run_dir = Path(temporary)
        paths: dict[str, Path | bool] = {
            "best_mIoU_final": run_dir / "best_mIoU.pth.tar",
            "best_Pd_final": run_dir / "best_Pd.pth.tar",
        }
        state = {"weight": torch.tensor([1.0])}
        record = _record(
            777, model_state_sha256=runner._state_dict_sha256(state)
        )
        payloads = {
            role: {
                "schema": runner.CHECKPOINT_SCHEMA,
                "selection_role": role,
                "epoch": 777,
                "state_dict": state,
                "selected_validation_record": record,
                "selected_metrics": dict(record["metrics"]),
                "test_split_accessed": False,
            }
            for role in ("best_mIoU", "best_Pd")
        }
        finals = runner._atomic_write_final_pair(
            run_dir=run_dir, paths=paths, payloads=payloads
        )
        assert set(finals) == {"best_mIoU", "best_Pd"}
        assert finals["best_mIoU"]["epoch"] == 777
        assert finals["best_Pd"]["epoch"] == 777
        assert Path(paths["best_mIoU_final"]).is_file()
        assert Path(paths["best_Pd_final"]).is_file()
        assert len(list(run_dir.glob("best_*.pth.tar"))) == 2
        for role in ("best_mIoU", "best_Pd"):
            payload = torch.load(
                paths[f"{role}_final"], map_location="cpu", weights_only=True
            )
            assert payload["selected_validation_record"] == record
            assert payload["selected_metrics"]["pixel_f1"] == 0.796875


def test_selected_role_hash_must_equal_physical_candidate_state() -> None:
    state = {"weight": torch.tensor([1.0])}
    state_hash = runner._state_dict_sha256(state)
    payload = {
        "roles": {
            role: {"selected": {"epoch": 500, "model_state_sha256": state_hash}}
            for role in ("best_mIoU", "best_Pd")
        }
    }
    assert runner._require_selected_state_hash(
        payload, "best_mIoU", state
    ) == state_hash
    payload["roles"]["best_Pd"]["selected"]["model_state_sha256"] = "f" * 64
    with pytest.raises(
        runner.SCTransNetSBSCV32RunnerError,
        match="differs from physical candidate",
    ):
        runner._require_selected_state_hash(payload, "best_Pd", state)


def test_final_payload_embeds_exact_selected_record_and_all_reported_metrics() -> None:
    state = {"weight": torch.tensor([1.0])}
    state_hash = runner._state_dict_sha256(state)
    record = _record(500, model_state_sha256=state_hash)
    selection_payload = {
        "roles": {
            role: {
                "selected": {
                    "epoch": 500,
                    "model_state_sha256": state_hash,
                }
            }
            for role in ("best_mIoU", "best_Pd")
        }
    }
    payload = runner._final_checkpoint_payload(
        role="best_mIoU",
        state=state,
        args=SimpleNamespace(
            method="sctransnet", dataset="IRSTD-1K", target_mode="binary"
        ),
        identity=_identity(),
        split_provenance={},
        normalization={"mean": 0.0, "std": 1.0},
        normalization_provenance={"mode": "legacy"},
        selection_payload=selection_payload,
        selected_validation_record=record,
        builder_metadata={"method": "sctransnet"},
        smoke=False,
    )
    assert payload["selected_validation_record"] == record
    assert runner.REQUIRED_REPORTED_METRICS.issubset(payload["selected_metrics"])
    assert payload["selected_metrics"]["pixel_precision"] == 0.75


@pytest.mark.parametrize("method", ("sctransnet", "sbsc_v32"))
def test_one_epoch_cpu_fixture_smoke_writes_two_roles_and_never_opens_test(
    method: str,
) -> None:
    runs = runner.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    access_log: list[str] = []
    manifest = {
        "schema": "fixture_split/v1",
        "seeds": {"split_seed": 20260811},
        "source_index": {
            "split": "train",
            "relative_path": "train.txt",
            "file_sha256": "1" * 64,
            "ordered_ids_sha256": "2" * 64,
            "sample_count": 2,
        },
        "outputs": {
            "train": {
                "relative_path": "splits/v2/IRSTD-1K/train.txt",
                "file_sha256": "3" * 64,
                "ordered_ids_sha256": "4" * 64,
                "sample_count": 1,
            },
            "val": {
                "relative_path": "splits/v2/IRSTD-1K/val.txt",
                "file_sha256": "5" * 64,
                "ordered_ids_sha256": "6" * 64,
                "sample_count": 1,
            },
        },
        "grouping": {"mode": "explicit_group_mapping"},
    }
    contract = SimpleNamespace(
        dataset="IRSTD-1K",
        manifest=manifest,
        manifest_sha256="b" * 64,
        data_tree_sha256="c" * 64,
        data_tree_verified=True,
        train_ids=("train",),
        val_ids=("val",),
    )
    train = _TrainFixture(contract, access_log)
    val = _ValFixture(contract, access_log)
    grouping = {
        "mode": "explicit_group_mapping",
        "sample_level_fallback_acknowledged": False,
        "warning": None,
    }

    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        output_root = Path(temporary) / "smoke-output"
        args = _args_for_method(
            method,
            "--output-root",
            str(output_root),
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--batch-size",
            "1",
            "--smoke-max-train-samples",
            "1",
            "--smoke-max-val-samples",
            "1",
            "--device",
            "cpu",
        )

        def fake_build(_args):
            model = _SixHeadFixture(method)
            gain = (
                {"state_keys": list(runner.GAIN_STATE_KEYS), "bounds": [0.0, 0.25]}
                if method == "sbsc_v32"
                else {"state_keys": [], "bounds": None}
            )
            core = SimpleNamespace(__file__=runner.__file__)
            core.project_sbsc_v32_constraints_ = lambda model: (
                model.mtc.encoder.layer[1].channel_attn.raw_dual_risk_level_gain.data.clamp_(
                    0.0, 0.25
                )
            )

            @contextmanager
            def capture(candidate):
                yield candidate

            core.capture_c3_v32_training_router = capture
            core.tri_router_supervision_loss = (
                lambda candidate, prediction, target: (
                    candidate.bias - 0.01
                ).square()
            )
            builder_metadata = {
                "method": method,
                "state_key_count": len(model.state_dict()),
                "cuda_strict_deterministic_median_adapter": (
                    runner.CUDA_MEDIAN_ADAPTER_SCHEMA
                    if method == "sbsc_v32"
                    else None
                ),
            }
            router_contract = {
                "state_keys": [],
                "parameter_count": 0,
                "auxiliary_loss_schema": None,
                "auxiliary_loss_weight": 0.0,
            }
            if method == "sbsc_v32":
                builder_metadata.update(
                    {
                        "router_state_keys": list(runner.ROUTER_STATE_SCHEMA),
                        "router_parameter_count": runner.ROUTER_PARAMETER_COUNT,
                        "loss_schema": runner.TOTAL_LOSS_SCHEMA,
                    }
                )
                router_contract = {
                    "state_keys": list(runner.ROUTER_STATE_SCHEMA),
                    "parameter_count": runner.ROUTER_PARAMETER_COUNT,
                    "auxiliary_loss_schema": (
                        runner.ROUTER_AUXILIARY_LOSS_SCHEMA
                    ),
                    "auxiliary_loss_weight": 1.0,
                }
            return (
                model,
                builder_metadata,
                core,
                {
                    "state_key_count": len(model.state_dict()),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "ordered_state_keys_sha256": "d" * 64,
                    "tensor_schema_sha256": "e" * 64,
                    "gain": gain,
                    "router": router_contract,
                },
                (),
            )

        with mock.patch.object(
            runner.r1, "build_datasets", return_value=(train, val, grouping)
        ), mock.patch.object(runner, "build_model", side_effect=fake_build):
            finals = runner.run(args)

        assert set(finals) == {"best_mIoU", "best_Pd"}
        assert all(path.is_file() for path in finals.values())
        assert access_log == ["train", "val"]
        summary_path = Path(runner.resolve_run_paths(args)["summary"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["final_physical_weight_count"] == 2
        assert summary["test_split_accessed"] is False
        training_record = summary["training_history"][0]
        assert training_record["mean_train_loss"] == pytest.approx(
            training_record["mean_segmentation_loss"]
            + training_record["mean_router_loss"]
        )
        assert training_record["router_auxiliary_weight"] == (
            1.0 if method == "sbsc_v32" else 0.0
        )
        assert len(list(Path(summary_path).parent.glob("best_*.pth.tar"))) == 2
        for path in finals.values():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert payload["builder_metadata"][
                "cuda_strict_deterministic_median_adapter"
            ] == (
                runner.CUDA_MEDIAN_ADAPTER_SCHEMA
                if method == "sbsc_v32"
                else None
            )
            assert payload["training"]["shuffle_stream"] == runner.SHUFFLE_STREAM
            assert payload["training"]["shuffle_stream_shared_across_methods"] is True
            assert payload["training"]["train_patch_size"] == [256, 256]
            assert (
                payload["training"]["validation_geometry_policy"]
                == "original_hw_pad_multiple_32"
            )
            loss_schema = payload["training"]["loss_schema"]
            assert loss_schema["segmentation"] == runner.SEGMENTATION_LOSS_SCHEMA
            assert loss_schema["router_auxiliary_weight"] == (
                1.0 if method == "sbsc_v32" else 0.0
            )
            assert runner.REQUIRED_REPORTED_METRICS.issubset(
                payload["selected_metrics"]
            )


def test_core_import_is_lazy_and_fails_closed() -> None:
    with mock.patch.object(
        runner.importlib,
        "import_module",
        side_effect=ModuleNotFoundError("fixture"),
    ):
        with pytest.raises(
            runner.SCTransNetSBSCV32RunnerError,
            match="core builder is unavailable",
        ):
            runner._load_core_module()


def test_direct_formal_runner_bootstrap_defeats_hostile_pythonpath(
    tmp_path: Path,
) -> None:
    hostile_root = tmp_path / "hostile_checkout"
    hostile_model = hostile_root / "model"
    hostile_experiments = hostile_root / "experiments"
    hostile_model.mkdir(parents=True)
    hostile_experiments.mkdir(parents=True)
    (hostile_model / "__init__.py").write_text(
        "ORIGIN = 'hostile-model'\n", encoding="utf-8"
    )
    (hostile_experiments / "__init__.py").write_text(
        "ORIGIN = 'hostile-experiments'\n", encoding="utf-8"
    )
    root = runner.PROJECT_ROOT.resolve(strict=True)
    root_alias = tmp_path / "repository_alias"
    root_alias.symlink_to(root, target_is_directory=True)
    script = Path(runner.__file__).resolve(strict=True)
    probe = f"""
import json
import os
import runpy
import sys
from pathlib import Path

namespace = runpy.run_path({str(script)!r})
core = namespace["_load_core_module"]()
import experiments
import model
verified_root = Path(namespace["PROJECT_ROOT"]).resolve(strict=True)
canonical_root_count = 0
for entry in sys.path:
    if not isinstance(entry, str):
        continue
    try:
        if Path(entry or os.curdir).resolve(strict=False) == verified_root:
            canonical_root_count += 1
    except (OSError, RuntimeError):
        pass
print(json.dumps({{
    "sys_path_0": sys.path[0],
    "canonical_root_count": canonical_root_count,
    "core_source": str(Path(core.__file__).resolve(strict=True)),
    "experiments_source": str(Path(experiments.__file__).resolve(strict=True)),
    "model_source": str(Path(model.__file__).resolve(strict=True)),
}}, sort_keys=True))
"""
    hostile_pythonpath = os.pathsep.join(
        (
            str(hostile_root),
            "/home/ly/BasicIRSTD",
            str(root_alias),
            str(root),
            str(root),
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
    assert payload["sys_path_0"] == str(root)
    assert payload["canonical_root_count"] == 1
    for key in ("core_source", "experiments_source", "model_source"):
        Path(payload[key]).relative_to(root)


def test_formal_loader_rejects_same_path_stale_runtime_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = runner._load_core_module()
    stale = ModuleType("experiments.sctransnet_sbsc_v32")
    stale.__file__ = real.__file__
    required = (
        "build_sctransnet_sbsc_v32_method",
        "validate_sctransnet_sbsc_v32",
        "validate_sbsc_v32_state_dict",
        "project_sbsc_v32_constraints_",
        "capture_c3_v32_training_router",
        "tri_router_supervision_loss",
        "structurally_inactive_parameter_names",
        "_estimate_c3_v31_support_v32",
    )
    for name in required:
        setattr(stale, name, getattr(real, name))
    stale.SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA = (
        runner.CUDA_MEDIAN_ADAPTER_SCHEMA
    )
    stale.EXPECTED_V31_SOLVER_SOURCE_SHA256 = (
        runner.EXPECTED_V31_SOLVER_SOURCE_SHA256
    )
    stale.V31_SOLVER_SOURCE_SHA256 = runner.EXPECTED_V31_SOLVER_SOURCE_SHA256
    stale.validate_sbsc_v32_runtime_integration = lambda: {
        "schema": "sctransnet_sbsc_v32/runtime_integration/stale",
        "cuda_median_adapter": runner.CUDA_MEDIAN_ADAPTER_SCHEMA,
    }
    monkeypatch.setitem(
        sys.modules, "experiments.sctransnet_sbsc_v32", stale
    )
    with pytest.raises(
        runner.SCTransNetSBSCV32RunnerError,
        match="runtime integration contract differs",
    ):
        runner._load_core_module()


def test_source_provenance_binds_runtime_builder_data_loss_and_model_closure() -> None:
    source = runner._source_provenance(runner._load_core_module())
    relative_paths = {
        entry["relative_path"] for entry in source["files"].values()
    }
    required = {
        "train_sctransnet_sbsc_v32_validation.py",
        "train_validation_selected.py",
        "train.py",
        "experiments/__init__.py",
        "experiments/sctransnet_sbsc_v32.py",
        "experiments/sctransnet_sbsc_v31.py",
        "experiments/sbsc_v32_selection.py",
        "experiments/four_dataset_models_seed42_v1.py",
        "experiments/evisirst_v2_data.py",
        "experiments/evisirst_v2_splits.py",
        "experiments/three_dataset_v2_protocol.py",
        "experiments/evisirst_data.py",
        "experiments/evisirst_v2_selection.py",
        "model/__init__.py",
        "model/EviSIRST.py",
        "model/_internal/Config.py",
        "model/_internal/SCTransNet.py",
    }
    assert required.issubset(relative_paths)
    required_keys = {
        "runner",
        "experiments_package",
        "core_builder",
        "paired_initialization_authority",
        "v2_split_validator",
        "three_dataset_source_protocol",
        "model_package",
        "r1_imported_model_entry",
        "sctransnet_config",
        "sctransnet",
    }
    assert required_keys.issubset(source["files"])
    root = runner.PROJECT_ROOT.resolve(strict=True)
    for entry in source["files"].values():
        relative = Path(entry["relative_path"])
        assert not relative.is_absolute()
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root)
        assert resolved.is_file()
        assert not (root / relative).is_symlink()
        assert entry["sha256"] == runner._sha256_file(resolved)
    assert len(source["source_tree_sha256"]) == 64
