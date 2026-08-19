from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import train_irstd_dcspg_ablation_test_selected_v1 as runner


def _row(epoch: int, *, miou: float = 0.5, pd: float = 0.8, fa: float = 0.01) -> dict[str, object]:
    return {
        "epoch": epoch,
        "data_role": "test",
        "test_loss": 1.0,
        "miou": miou,
        "niou": 0.6,
        "pixel_precision": 0.7,
        "pixel_recall": 0.8,
        "pixel_f1": 0.7466666667,
        "pd": pd,
        "tiny_pd": 0.75,
        "fa": fa,
        "false_objects_per_image": 2.0,
        "target_count": 10,
        "matched_target_count": 8,
        "tiny_target_count": 4,
        "matched_tiny_target_count": 3,
        "predicted_object_count": 12,
        "unmatched_predicted_object_count": 4,
        "valid_pixel_count": 256 * 256,
    }


def test_rules_are_frozen_but_formal_architecture_is_fail_closed() -> None:
    rules = runner._load_rules()
    execution = rules["execution"]
    outputs = rules["outputs"]

    assert execution["epochs"] == 1000
    assert execution["test_begin_epoch"] == 501
    assert execution["test_end_epoch"] == 1000
    assert execution["test_every"] == 1
    assert execution["test_history_count"] == 500
    assert runner.ARCHITECTURE_FINAL_AUDIT_GO is False
    assert runner.EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 is None
    assert runner.EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 is None
    with pytest.raises(runner.DCSPGAblationError, match="not sealed"):
        runner._source_manifest(formal=True)

    source_manifest = runner._source_manifest(formal=False)
    assert len(source_manifest["architecture_source_sha256"]) == 64
    assert len(source_manifest["architecture_test_sha256"]) == 64
    assert outputs["published_weight_file_count_per_arm"] == 2
    assert tuple(outputs["published_checkpoint_roles"]) == runner.ROLE_NAMES
    assert set(outputs["published_weight_filenames"]) == {"best_miou", "best_pd"}


def test_formal_no_go_fails_before_output_or_device_mutation(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    args = runner.parse_args(
        [
            "--method-id",
            "dcspg_original",
            "--device",
            "cuda:999",
            "--output-root",
            str(output),
        ]
    )
    with pytest.raises(runner.DCSPGAblationError, match="not sealed"):
        runner.run(args)
    assert not output.exists()


def test_runner_independently_pins_cp_parent_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = runner._load_rules()
    monkeypatch.setattr(runner, "_load_rules", lambda: copy.deepcopy(rules))
    monkeypatch.setattr(runner, "EXPECTED_CP_PARENT_SOURCE_SHA256", "0" * 64)
    with pytest.raises(runner.DCSPGAblationError, match="dependency SHA-256"):
        runner._source_manifest(formal=False)


def test_formal_cli_forces_1000_epochs_and_every_epoch_test_from_501() -> None:
    args = runner.parse_args(["--method-id", "dcspg_original"])

    assert args.epochs == 1000
    assert args.test_begin == 501
    assert args.test_every == 1
    assert runner.expected_test_epochs(args.epochs, args.test_begin, args.test_every) == list(
        range(501, 1001)
    )

    for forbidden in (
        ["--method-id", "dcspg_original", "--epochs", "999"],
        ["--method-id", "dcspg_original", "--test-begin", "500"],
        ["--method-id", "dcspg_original", "--test-every", "2"],
        ["--method-id", "dcspg_original", "--max-test-images", "1"],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args(forbidden)

    smoke = runner.parse_args(
        [
            "--method-id",
            "dcspg_original",
            "--smoke",
            "--epochs",
            "3",
            "--warmup-epochs",
            "0",
            "--test-begin",
            "2",
            "--max-train-samples",
            "1",
            "--max-test-images",
            "1",
        ]
    )
    assert runner.expected_test_epochs(smoke.epochs, smoke.test_begin, smoke.test_every) == [2, 3]


def test_metric_history_requires_exact_501_to_1000_cadence_and_selects_two_roles() -> None:
    history = [_row(epoch) for epoch in range(501, 1001)]
    history[1] = _row(502, miou=0.9, pd=0.8, fa=0.01)
    history[2] = _row(503, miou=0.7, pd=0.95, fa=0.005)

    normalized, best = runner._validate_metric_history(
        history,
        completed_epoch=1000,
        test_begin=501,
        test_every=1,
    )

    assert len(normalized) == 500
    assert [row["epoch"] for row in normalized] == list(range(501, 1001))
    assert best == {"best_miou": 502, "best_pd": 503}
    assert runner.role_key(normalized[1], "best_miou") > runner.role_key(
        normalized[2], "best_miou"
    )
    assert runner.role_key(normalized[2], "best_pd") > runner.role_key(
        normalized[1], "best_pd"
    )

    missing_epoch = copy.deepcopy(history)
    missing_epoch.pop(10)
    with pytest.raises(runner.DCSPGAblationError, match="cadence"):
        runner._validate_metric_history(
            missing_epoch,
            completed_epoch=1000,
            test_begin=501,
            test_every=1,
        )

    wrong_first = copy.deepcopy(history)
    wrong_first[0]["epoch"] = 500
    with pytest.raises(runner.DCSPGAblationError, match="cadence"):
        runner._validate_metric_history(
            wrong_first,
            completed_epoch=1000,
            test_begin=501,
            test_every=1,
        )


def test_disclosures_flip_only_after_test_history_exists() -> None:
    assert runner._phase_flags(completed_epoch=500, test_history=[]) == {
        "test_access_started": False,
        "test_access_verified": False,
        "test_index_opened": False,
        "test_split_accessed": False,
        "test_selected": False,
        "selection_is_optimistic": False,
        "unbiased_test_claim_supported": False,
        "stable_over_baseline_claim_supported": False,
    }
    post_test = runner._phase_flags(completed_epoch=501, test_history=[_row(501)])
    assert post_test["test_access_started"] is True
    assert post_test["test_access_verified"] is True
    assert post_test["test_selected"] is True
    assert post_test["selection_is_optimistic"] is True
    assert post_test["unbiased_test_claim_supported"] is False
    assert post_test["stable_over_baseline_claim_supported"] is False


def test_strict_json_rejects_duplicate_keys_nan_and_no_clobber(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    with pytest.raises(runner.DCSPGAblationError, match="duplicate"):
        runner._strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(runner.DCSPGAblationError, match="non-finite"):
        runner._strict_json(nonfinite)

    immutable = tmp_path / "immutable.json"
    runner._write_json_no_clobber(immutable, {"x": 1})
    with pytest.raises(FileExistsError):
        runner._write_json_no_clobber(immutable, {"x": 2})
    assert runner._strict_json(immutable) == {"x": 1}


def test_symlink_components_and_symlink_json_fail_closed(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    args = runner.parse_args(
        [
            "--method-id",
            "dcspg_original",
            "--smoke",
            "--epochs",
            "1",
            "--test-begin",
            "1",
            "--warmup-epochs",
            "0",
            "--output-root",
            str(linked / "output"),
        ]
    )
    with pytest.raises(runner.DCSPGAblationError, match="symlink component"):
        runner.resolve_run_paths(args)

    real_json = real / "record.json"
    real_json.write_text('{"ok":true}\n', encoding="utf-8")
    linked_json = tmp_path / "record.json"
    linked_json.symlink_to(real_json)
    with pytest.raises(runner.DCSPGAblationError, match="not a regular file"):
        runner._strict_json(linked_json)


def test_resume_rejects_intermediate_symlink_and_outside_run_dir(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    args = runner.parse_args(
        [
            "--method-id",
            "dcspg_original",
            "--smoke",
            "--epochs",
            "1",
            "--test-begin",
            "1",
            "--warmup-epochs",
            "0",
            "--output-root",
            str(output),
        ]
    )
    paths = runner.resolve_run_paths(args)
    intermediate = output / "smoke" / "dcspg_original"
    intermediate.parent.mkdir(parents=True)
    intermediate.symlink_to(outside, target_is_directory=True)
    with pytest.raises(runner.DCSPGAblationError, match="symlink component"):
        runner._prepare_run_directory(paths, resume=True)

    intermediate.unlink()
    hostile = dict(paths)
    hostile["run_dir"] = outside
    with pytest.raises(runner.DCSPGAblationError, match="outside the output root"):
        runner._prepare_run_directory(hostile, resume=True)


def test_source_manifest_includes_runner_contract_and_aggregate_tracks_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = runner._source_manifest(formal=False)
    files = baseline["files"]
    required = {
        "experiments/four_dataset_models_seed42_v1.py",
        "run_irstd_dcspg_test_selected_v1.py",
        "tests/test_irstd_dcspg_test_selected_adapter.py",
        "tests/test_train_irstd_dcspg_ablation_test_selected_v1.py",
    }
    assert required.issubset(files)

    real_sha256 = runner._sha256
    target = runner.RUNNER_TEST_SOURCE.resolve(strict=True)

    def changed_sha256(path: Path) -> str:
        if path.resolve(strict=True) == target:
            return "f" * 64
        return real_sha256(path)

    monkeypatch.setattr(runner, "_sha256", changed_sha256)
    changed = runner._source_manifest(formal=False)
    assert changed["files"]["tests/test_train_irstd_dcspg_ablation_test_selected_v1.py"] == "f" * 64
    assert changed["aggregate_sha256"] != baseline["aggregate_sha256"]


def test_state_validation_rejects_nan_and_shape_tamper() -> None:
    expected = {"weight": torch.zeros(2, dtype=torch.float32)}
    with pytest.raises(runner.DCSPGAblationError, match="non-finite"):
        runner._validate_state_dict(
            {"weight": torch.tensor([0.0, float("nan")])},
            expected,
            expected_count=1,
        )
    with pytest.raises(runner.DCSPGAblationError, match="contract"):
        runner._validate_state_dict(
            {"weight": torch.zeros(3)}, expected, expected_count=1
        )


def test_far_background_loss_uses_euclidean_disk_and_validates_contract() -> None:
    target = torch.zeros(1, 1, 9, 9)
    target[0, 0, 4, 4] = 1.0
    protected_heads = [torch.zeros_like(target) for _ in range(6)]
    protected_heads[-2][0, 0, 4, 7] = 1.0  # offset (0,3), inside r=3 disk
    protected_heads[-1][0, 0, 4, 7] = 1.0
    assert runner.far_background_auxiliary_loss(protected_heads, target).item() == 0.0

    far_heads = [torch.zeros_like(target) for _ in range(6)]
    far_heads[-2][0, 0, 7, 7] = 1.0  # offset (3,3), outside Euclidean disk
    far_heads[-1][0, 0, 7, 7] = 1.0
    assert runner.far_background_auxiliary_loss(far_heads, target).item() == pytest.approx(
        1.0 / 9.0
    )
    assert int(runner._disk_kernel_radius3(device=target.device, dtype=target.dtype).sum()) == 29

    bad_target = target.clone()
    bad_target[0, 0, 0, 0] = 1.1
    with pytest.raises(runner.DCSPGAblationError, match="target"):
        runner.far_background_auxiliary_loss(far_heads, bad_target)
    bad_heads = list(far_heads)
    bad_heads[0] = bad_heads[0].double()
    with pytest.raises(runner.DCSPGAblationError, match="head 0"):
        runner.far_background_auxiliary_loss(bad_heads, target)


def test_role_tie_break_is_earliest_epoch() -> None:
    rows = [_row(501), _row(502)]
    assert runner._select_roles(rows) == {"best_miou": 501, "best_pd": 501}


def test_candidate_written_before_latest_is_validated_then_removed(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    state = runner._cpu_state(model, expected_count=len(model.state_dict()))
    identity = {"method_id": "dcspg_original"}
    candidate_dir = tmp_path / "candidates" / "test_selected"
    runner._save_current_candidate_if_selected(
        epoch=2,
        state=state,
        identity=identity,
        history=[_row(2)],
        candidate_dir=candidate_dir,
    )
    orphan = candidate_dir / "epoch_0002.pth.tar"
    assert orphan.is_file()

    assert runner._reconcile_candidates(
        candidate_dir=candidate_dir,
        identity=identity,
        history=[],
        artifacts={},
        expected_state=state,
        expected_count=len(state),
        completed_epoch=1,
        allow_uncommitted_next=True,
    ) == {}
    assert not orphan.exists()


def test_adam_and_rng_tamper_fail_closed() -> None:
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    image = torch.zeros(2, 1, 4, 4)
    model(image)[0].sum().backward()
    optimizer.step()
    state = runner._cpu_optimizer_state(optimizer)
    identity = {
        "model": "tiny-fixture",
        "epochs": 1,
        "batch_size": 2,
        "train_count": 2,
        "base_lr": 1e-3,
        "min_lr": 1e-3,
        "warmup_epochs": 0,
    }
    fresh = _TinyModel()
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    runner.r1_transaction._validate_and_load_adam_optimizer_state(
        optimizer_state=copy.deepcopy(state),
        model=fresh,
        optimizer=fresh_optimizer,
        identity=identity,
        completed_epoch=1,
        total_epochs=1,
    )

    tampered = copy.deepcopy(state)
    first_state = next(iter(tampered["state"].values()))
    first_state["exp_avg"].reshape(-1)[0] = float("nan")
    bad_model = _TinyModel()
    bad_optimizer = torch.optim.Adam(bad_model.parameters(), lr=1e-3)
    with pytest.raises(Exception, match="exp_avg tensor contract"):
        runner.r1_transaction._validate_and_load_adam_optimizer_state(
            optimizer_state=tampered,
            model=bad_model,
            optimizer=bad_optimizer,
            identity=identity,
            completed_epoch=1,
            total_epochs=1,
        )

    rng = runner.legacy_train._capture_rng_state(torch.device("cpu"))
    rng["device_type"] = "cuda"
    with pytest.raises(ValueError, match="RNG/device"):
        runner.legacy_train._restore_rng_state(rng, torch.device("cpu"))


def test_formal_lazy_boundary_is_exactly_500_then_501() -> None:
    assert not runner.test_due(500, 501, 1)
    assert runner.test_due(501, 501, 1)
    assert len(runner.expected_test_epochs(500, 501, 1)) == 0
    assert runner.expected_test_epochs(501, 501, 1) == [501]
    assert runner._started_ledger_payload(
        {
            "method_id": "dcspg_original",
            "identity_sha256": "0" * 64,
            "data_identity": {"expected_test_contract": {"count": 201}},
            "test_schedule": {"begin_epoch": 501},
        }
    )["first_access_epoch"] == 501


class _TinyTrain(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.full((1, 4, 4), float(index + 1) / 4.0)
        mask = torch.zeros(1, 4, 4)
        mask[0, index, index] = 1.0
        return image, mask


class _TinyTest(Dataset[Any]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Any:
        del index
        return torch.zeros(1, 4, 4), torch.zeros(1, 4, 4), (4, 4), "tiny"


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.mode = "train"

    def forward(self, image: torch.Tensor) -> Any:
        probability = torch.sigmoid(self.conv(image))
        if self.mode == "test":
            return probability
        return tuple(probability for _ in range(6))


def _install_tiny_run_fixtures(
    monkeypatch: pytest.MonkeyPatch, *, fail_first_test_open: dict[str, bool]
) -> dict[str, int]:
    rules = runner._load_rules()
    source = {
        "schema": runner.SOURCE_SET_SCHEMA,
        "files": {"tiny.py": "0" * 64},
        "aggregate_sha256": "1" * 64,
        "architecture_source_sha256": "2" * 64,
        "architecture_test_sha256": "3" * 64,
    }
    train = _TinyTrain()
    expected_test = {
        "test_count": 1,
        "test_index_file_sha256": "4" * 64,
        "test_ordered_ids_sha256": "5" * 64,
        "test_image_mask_tree_sha256": "6" * 64,
        "test_image_mask_tree_hash_algorithm": "tiny",
    }
    data_identity = {
        "dataset_root": "/tiny",
        "split_root": "/tiny-split",
        "dataset": runner.DATASET,
        "training_dataset_class": "tiny",
        "training_target_rule": runner.TRAINING_TARGET_RULE,
        "train_count": 2,
        "full_train_count": 2,
        "v2_train_count": 1,
        "v2_val_count": 1,
        "v2_train_val_union_equals_source_train": True,
        "independent_validation_split_in_this_runner": False,
        "split_manifest_sha256": "7" * 64,
        "source_train_data_tree_sha256": "8" * 64,
        "source_train_data_tree_verified": True,
        "train_index_relative_path": "tiny/train.txt",
        "train_index_file_sha256": "9" * 64,
        "train_ordered_ids_sha256": "a" * 64,
        "normalization": {"mean": 0.0, "std": 1.0},
        "expected_test_contract": expected_test,
        "test_access_is_lazy": True,
        "test_index_opened_at_identity_construction": False,
        "startup_test_index_opened": False,
        "test_split_accessed_during_preflight": False,
    }
    observed_test = {
        "test_count": 1,
        "full_test_count": 1,
        "test_index_relative_path": "tiny/test.txt",
        "test_index_file_sha256": "4" * 64,
        "test_ordered_ids_sha256": "5" * 64,
        "test_image_mask_tree_sha256": "6" * 64,
        "normalization": {"mean": 0.0, "std": 1.0},
        "test_index_opened": True,
        "test_split_accessed": True,
        "first_access_epoch": 2,
    }
    calls = {"test_open": 0, "evaluate": 0}

    monkeypatch.setattr(runner, "_load_rules", lambda: copy.deepcopy(rules))
    monkeypatch.setattr(
        runner, "_source_manifest", lambda *, formal: copy.deepcopy(source)
    )
    monkeypatch.setattr(
        runner,
        "_data_identity",
        lambda args, frozen: (train, train, copy.deepcopy(data_identity)),
    )

    def build(method: Any) -> Any:
        del method
        model = _TinyModel()
        count = len(model.state_dict())
        params = sum(parameter.numel() for parameter in model.parameters())
        architecture = {
            "kind": "guard",
            "name": "tiny",
            "schema": "tiny/v1",
            "architecture_seed": 42,
            "state_key_count": count,
            "parameter_count": params,
            "full_model_scratch": True,
            "baseline_checkpoint_loaded": False,
            "warm_start_used": False,
        }
        return model, {}, {"schema": "tiny-validation/v1"}, architecture

    monkeypatch.setattr(runner, "_build_architecture", build)

    def activate(args: Any, identity: Any) -> Any:
        del args, identity
        calls["test_open"] += 1
        if fail_first_test_open["value"]:
            fail_first_test_open["value"] = False
            raise RuntimeError("injected crash after access-start ledger")
        test = _TinyTest()
        return test, test, copy.deepcopy(observed_test)

    monkeypatch.setattr(runner, "_activate_test_data", activate)

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        calls["evaluate"] += 1
        return {
            key: value
            for key, value in _row(1).items()
            if key not in {"epoch", "data_role"}
        }

    monkeypatch.setattr(runner, "evaluate_model", evaluate)

    def load_adam(**kwargs: Any) -> None:
        kwargs["optimizer"].load_state_dict(dict(kwargs["optimizer_state"]))

    monkeypatch.setattr(
        runner.r1_transaction,
        "_validate_and_load_adam_optimizer_state",
        load_adam,
    )
    return calls


def _tiny_args(tmp_path: Path, *, resume: bool) -> Any:
    argv = [
        "--method-id",
        "dcspg_original",
        "--smoke",
        "--smoke-id",
        "lazy-crash",
        "--epochs",
        "3",
        "--batch-size",
        "1",
        "--workers",
        "0",
        "--test-begin",
        "2",
        "--warmup-epochs",
        "1",
        "--cpu-threads",
        "1",
        "--device",
        "cpu",
        "--output-root",
        str(tmp_path),
    ]
    if resume:
        argv.append("--resume")
    return runner.parse_args(argv)


def test_lazy_access_crash_ledger_resume_and_exact_two_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail = {"value": True}
    calls = _install_tiny_run_fixtures(
        monkeypatch, fail_first_test_open=fail
    )
    args = _tiny_args(tmp_path, resume=False)
    paths = runner.resolve_run_paths(args)

    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(args)
    latest = torch.load(paths["latest"], map_location="cpu", weights_only=True)
    assert latest["epoch"] == 1
    assert latest["test_history"] == []
    assert latest["test_split_accessed"] is False
    assert calls == {"test_open": 1, "evaluate": 0}
    assert paths["test_access_started"].is_file()
    assert not paths["test_access_verified"].exists()
    started = runner._strict_json(paths["test_access_started"])
    assert started["test_access_started"] is True
    assert started["live_test_open_may_have_occurred"] is True
    assert started["selection_is_optimistic"] is True

    result = runner.run(_tiny_args(tmp_path, resume=True))
    assert result == paths["published_dir"]
    assert calls == {"test_open": 2, "evaluate": 2}
    final = torch.load(paths["latest"], map_location="cpu", weights_only=True)
    assert final["epoch"] == 3
    assert [row["epoch"] for row in final["test_history"]] == [2, 3]
    assert final["test_split_accessed"] is True
    assert final["test_selected"] is True
    assert final["selection_is_optimistic"] is True
    assert final["unbiased_test_claim_supported"] is False
    assert paths["test_access_verified"].is_file()

    published = sorted(paths["published_dir"].iterdir())
    assert [path.name for path in published] == sorted(
        runner.PUBLISHED_FILENAMES.values()
    )
    before = {path.name: runner._sha256(path) for path in published}
    assert runner.run(_tiny_args(tmp_path, resume=True)) == paths["published_dir"]
    after = {path.name: runner._sha256(path) for path in published}
    assert after == before
    summary = runner._strict_json(paths["summary"])
    assert summary["published_weight_file_count"] == 2
    assert summary["test_selected"] is True
    assert summary["selection_is_optimistic"] is True
    assert summary["unbiased_test_claim_supported"] is False
