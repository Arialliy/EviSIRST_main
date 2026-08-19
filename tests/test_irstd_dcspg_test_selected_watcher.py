from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import run_irstd_dcspg_test_selected_when_idle as watcher


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _history(
    method_id: str,
    completed: int,
    *,
    role: str,
    rows: list[dict[str, object]],
    accessed: bool,
) -> dict[str, object]:
    key = "training_history" if role == "train" else "test_history"
    return {
        "schema": "fixture/history",
        "data_role": role,
        "method_id": method_id,
        "completed_epoch": completed,
        key: rows,
        "test_access_started": accessed,
        "test_access_verified": accessed,
    }


def _metric(epoch: int, *, complete: bool = False) -> dict[str, object]:
    if complete:
        return {
            "epoch": epoch,
            "data_role": "test",
            "test_loss": 1.0,
            "miou": 0.5,
            "niou": 0.55,
            "pixel_precision": 0.7,
            "pixel_recall": 0.8,
            "pixel_f1": 0.7466666667,
            "pd": 0.8,
            "tiny_pd": 0.75,
            "fa": 0.01,
            "false_objects_per_image": 2.0,
            "target_count": 10,
            "matched_target_count": 8,
            "tiny_target_count": 4,
            "matched_tiny_target_count": 3,
            "predicted_object_count": 12,
            "unmatched_predicted_object_count": 4,
            "valid_pixel_count": 65536,
        }
    return {
        "epoch": epoch,
        "data_role": "test",
        "test_loss": 1.0,
        "miou": 0.5,
        "niou": 0.55,
        "pd": 0.8,
        "tiny_pd": 0.75,
        "fa": 0.01,
    }


def _make_progress(root: Path, method_id: str, completed: int) -> Path:
    run_dir = (
        root / "formal" / method_id / "IRSTD-1K" / "run_seed_42"
    )
    train = [{"epoch": epoch} for epoch in range(1, completed + 1)]
    test = [_metric(epoch) for epoch in range(501, completed + 1)]
    _write_json(
        run_dir / "training_history.json",
        _history(
            method_id,
            completed,
            role="train",
            rows=train,
            accessed=bool(test),
        ),
    )
    _write_json(
        run_dir / "test_history.json",
        _history(
            method_id,
            completed,
            role="test",
            rows=test,
            accessed=bool(test),
        ),
    )
    if completed >= 501:
        _write_json(run_dir / "test_access_started.json", {"started": True})
        _write_json(run_dir / "test_access_verified.json", {"verified": True})
    return run_dir


DISCLOSURE = {
    "test_access_started": True,
    "test_access_verified": True,
    "test_index_opened": True,
    "test_split_accessed": True,
    "test_selected": True,
    "selection_is_optimistic": True,
    "unbiased_test_claim_supported": False,
    "stable_over_baseline_claim_supported": False,
    "test_used_for_structure_or_hyperparameter_selection": False,
}


def _formal_data_identity(runner: object) -> dict[str, object]:
    rules = runner._load_rules()
    frozen = rules["data"]
    return {
        "dataset_root": "/home/ly/SCTransNet_main/datasets",
        "split_root": "/home/ly/EviSIRST_main/splits/v2",
        "dataset": "IRSTD-1K",
        "training_dataset_class": "experiments.evisirst_data.EviSIRSTTrainDataset",
        "training_target_rule": "raw_mask_div_255",
        "train_count": 800,
        "full_train_count": 800,
        "v2_train_count": 640,
        "v2_val_count": 160,
        "v2_train_val_union_equals_source_train": True,
        "independent_validation_split_in_this_runner": False,
        "split_manifest_sha256": runner.EXPECTED_SPLIT_MANIFEST_SHA256,
        "source_train_data_tree_sha256": runner.EXPECTED_SOURCE_TRAIN_DATA_TREE_SHA256,
        "source_train_data_tree_verified": True,
        "train_index_relative_path": "IRSTD-1K/img_idx/train_IRSTD-1K.txt",
        "train_index_file_sha256": runner.EXPECTED_TRAIN_INDEX_FILE_SHA256,
        "train_ordered_ids_sha256": runner.EXPECTED_TRAIN_ORDERED_IDS_SHA256,
        "normalization": {"mean": 87.4661865234375, "std": 39.71953201293945},
        "expected_test_contract": {
            "test_count": frozen["expected_test_count"],
            "test_index_file_sha256": frozen["expected_test_index_file_sha256"],
            "test_ordered_ids_sha256": frozen["expected_test_ordered_ids_sha256"],
            "test_image_mask_tree_sha256": frozen["expected_test_image_mask_tree_sha256"],
            "test_image_mask_tree_hash_algorithm": "ordered_length_prefixed(role,sample_id,relative_path,file_sha256)",
        },
        "test_access_is_lazy": True,
        "test_index_opened_at_identity_construction": False,
        "startup_test_index_opened": False,
        "test_split_accessed_during_preflight": False,
    }


def _make_complete(root: Path, method_id: str) -> Path:
    import torch
    import train_irstd_dcspg_ablation_test_selected_v1 as runner

    run_dir = root / "formal" / method_id / "IRSTD-1K" / "run_seed_42"
    run_dir.mkdir(parents=True)
    rules = runner._load_rules()
    method = runner._method_contract(
        SimpleNamespace(method_id=method_id, smoke=False), rules
    )
    model, _metadata, architecture_validation, architecture_identity = (
        runner._build_architecture(method)
    )
    state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    args = SimpleNamespace(
        method_id=method_id,
        architecture_seed=42,
        run_seed=42,
        epochs=1000,
        batch_size=16,
        workers=0,
        base_lr=1e-3,
        min_lr=1e-5,
        warmup_epochs=10,
        test_begin=501,
        test_every=1,
        device="cuda:0",
        smoke=False,
        smoke_id="fixture",
    )
    identity = runner._run_identity(
        args=args,
        method=method,
        architecture=architecture_identity,
        architecture_validation=architecture_validation,
        source_manifest=runner._source_manifest(formal=False),
        data_identity=_formal_data_identity(runner),
        run_dir=run_dir,
    )
    training_history = []
    for epoch in range(1, 1001):
        learning_rate = runner.legacy_train.learning_rate_for_epoch(
            epoch, 1000, 1e-3, 1e-5, 10
        )
        far = 0.0 if method["loss_kind"] == "original" else 0.1
        training_history.append(
            {
                "epoch": epoch,
                "base_bce_sum": 1.0,
                "farbg_loss": far,
                "total_loss": 1.0 + far,
                "learning_rate": learning_rate,
                "processed_samples": 800,
            }
        )
    test_history = [_metric(epoch, complete=True) for epoch in range(501, 1001)]
    test_history[700 - 501]["miou"] = 0.95
    test_history[800 - 501]["pd"] = 0.99
    best_epochs = runner._select_roles(test_history)
    by_epoch = {int(row["epoch"]): row for row in test_history}
    test_identity = {
        "test_count": 201,
        "full_test_count": 201,
        "test_index_relative_path": "IRSTD-1K/img_idx/test_IRSTD-1K.txt",
        "test_index_file_sha256": runner.EXPECTED_TEST_INDEX_FILE_SHA256,
        "test_ordered_ids_sha256": runner.EXPECTED_TEST_ORDERED_IDS_SHA256,
        "test_image_mask_tree_sha256": runner.EXPECTED_TEST_IMAGE_MASK_TREE_SHA256,
        "normalization": {"mean": 87.4661865234375, "std": 39.71953201293945},
        "test_index_opened": True,
        "test_split_accessed": True,
        "first_access_epoch": 501,
    }
    started_path = run_dir / "test_access_started.json"
    verified_path = run_dir / "test_access_verified.json"
    _write_json(started_path, runner._started_ledger_payload(identity))
    _write_json(verified_path, runner._verified_ledger_payload(identity, test_identity))
    started_sha = watcher._sha256(started_path)
    verified_sha = watcher._sha256(verified_path)

    candidate_dir = run_dir / "candidates/test_selected"
    candidate_dir.mkdir(parents=True)
    role_by_epoch: dict[int, list[str]] = {}
    for role, epoch in best_epochs.items():
        role_by_epoch.setdefault(int(epoch), []).append(role)
    frontier: dict[int, dict[str, object]] = {}
    for epoch, roles in role_by_epoch.items():
        path = candidate_dir / f"epoch_{epoch:04d}.pth.tar"
        torch.save(
            runner._candidate_payload(
                epoch=epoch,
                state=state,
                identity=identity,
                test_record=by_epoch[epoch],
            ),
            path,
        )
        frontier[epoch] = {
            "relative_path": f"candidates/test_selected/{path.name}",
            "sha256": watcher._sha256(path),
            "roles": sorted(roles),
        }

    latest = runner._recovery_payload(
        epoch=1000,
        state=state,
        optimizer_state={},
        rng={},
        identity=identity,
        training_history=training_history,
        test_history=test_history,
        candidate_artifacts=frontier,
        test_identity=test_identity,
        elapsed_seconds=1.0,
    )
    torch.save(latest, run_dir / "last_training_state.pth.tar")
    _write_json(
        run_dir / "training_history.json",
        runner._history_payload(
            data_role="train",
            identity=identity,
            completed_epoch=1000,
            history=training_history,
            test_history_for_flags=test_history,
        ),
    )
    _write_json(
        run_dir / "test_history.json",
        runner._history_payload(
            data_role="test",
            identity=identity,
            completed_epoch=1000,
            history=test_history,
            test_history_for_flags=test_history,
        ),
    )

    published_dir = run_dir / "published_weights"
    published_dir.mkdir()
    published_records: dict[str, dict[str, object]] = {}
    selection_roles: dict[str, dict[str, object]] = {}
    for role, filename in runner.PUBLISHED_FILENAMES.items():
        epoch = int(best_epochs[role])
        checkpoint = runner._checkpoint_payload(
            role=role,
            epoch=epoch,
            state=state,
            metrics=by_epoch[epoch],
            identity=identity,
            test_identity=test_identity,
            candidate_record=frontier[epoch],
            started_sha256=started_sha,
            verified_sha256=verified_sha,
        )
        checkpoint_path = published_dir / filename
        torch.save(checkpoint, checkpoint_path)
        record = {
            "relative_path": f"published_weights/{filename}",
            "sha256": watcher._sha256(checkpoint_path),
            "epoch": epoch,
        }
        published_records[role] = record
        selection_roles[role] = {
            "epoch": epoch,
            "metrics": dict(by_epoch[epoch]),
            "role_key": runner.role_key_record(by_epoch[epoch], role),
            "candidate": dict(frontier[epoch]),
            "published_checkpoint": dict(record),
        }
    selection = {
        "schema": runner.SELECTION_SCHEMA,
        "status": "complete",
        "model": "EviSIRST",
        "dataset": "IRSTD-1K",
        "method_id": method_id,
        "run_identity": identity,
        "selection_pool": identity["selector"]["pool"],
        "test_history_count": 500,
        "roles": selection_roles,
        "test_identity": test_identity,
        "test_access_started_ledger_sha256": started_sha,
        "test_access_verified_ledger_sha256": verified_sha,
        "historical_test_selected_baseline_comparison_allowed": True,
        "test_used_for_structure_or_hyperparameter_selection": False,
        **runner._phase_flags(completed_epoch=1000, test_history=test_history),
    }
    selection_path = run_dir / "selection_record.json"
    _write_json(selection_path, selection)
    summary = {
        "schema": runner.SUMMARY_SCHEMA,
        "status": "complete",
        "model": "EviSIRST",
        "dataset": "IRSTD-1K",
        "method_id": method_id,
        "run_identity": identity,
        "completed_epoch": 1000,
        "train_count": 800,
        "training_history_count": 1000,
        "training_history_sha256": watcher._sha256(run_dir / "training_history.json"),
        "test_history_count": 500,
        "test_history_sha256": watcher._sha256(run_dir / "test_history.json"),
        "test_access_started_ledger_sha256": started_sha,
        "test_access_verified_ledger_sha256": verified_sha,
        "selection_record_sha256": watcher._sha256(selection_path),
        "published_checkpoints": published_records,
        "published_weight_file_count": 2,
        "complete_six_arm_ablation_claim_supported": False,
        "test_used_for_structure_or_hyperparameter_selection": False,
        **runner._phase_flags(completed_epoch=1000, test_history=test_history),
    }
    _write_json(run_dir / "summary.json", summary)
    return run_dir


def _task_for_root(root: Path, method_id: str = "dcspg_original") -> watcher.Task:
    original = watcher.task_for_method(method_id)
    # Task.run_dir uses the module's FORMAL_OUTPUT_ROOT, so tests patch that
    # global and retain the exact immutable GPU identity here.
    return original


def test_manifest_has_exactly_two_fixed_1000_epoch_tasks() -> None:
    manifest = watcher.task_manifest()
    assert manifest["task_count"] == 2
    assert {entry["method_id"] for entry in manifest["tasks"]} == {
        "dcspg_original",
        "dcspg_farbg",
    }
    assert {entry["epochs"] for entry in manifest["tasks"]} == {1000}
    assert {entry["test_begin_epoch"] for entry in manifest["tasks"]} == {501}
    assert {entry["test_history_count"] for entry in manifest["tasks"]} == {500}
    assert manifest["published_weight_file_count_per_arm"] == 2
    assert manifest["quarantined_output_inspected"] is False
    assert manifest["quarantined_output_reused"] is False


def test_direct_script_bootstraps_project_root_for_offline_runner_import(
    tmp_path: Path,
) -> None:
    program = f"""
import importlib.util
import pathlib
import sys

source = pathlib.Path({os.fspath(watcher.WATCHER_SOURCE)!r})
spec = importlib.util.spec_from_file_location("dcspg_watcher_direct_probe", source)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert str(module.PROJECT_ROOT) in sys.path
runner = __import__("train_irstd_dcspg_ablation_test_selected_v1")
assert pathlib.Path(runner.__file__).resolve() == module.RUNNER_SOURCE
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [os.fspath(watcher.PYTHON_BIN), "-B", "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_initialization_migration_accepts_only_exact_predecessor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initialization = tmp_path / "formal_initialization.json"
    monkeypatch.setattr(watcher, "INITIALIZATION_PATH", initialization)

    _write_json(initialization, watcher._initialization_payload())
    assert watcher._initialization_is_valid()

    _write_json(initialization, watcher._initial_launch_initialization_payload())
    assert not watcher._initialization_is_valid()
    assert watcher._initialization_is_valid(allow_initial_launch=True)

    tampered = watcher._initial_launch_initialization_payload()
    tampered["workers_launched_before_commit"] = True
    _write_json(initialization, tampered)
    assert not watcher._initialization_is_valid(allow_initial_launch=True)


def test_tasks_pin_exact_gpu_index_uuid_and_bus() -> None:
    tasks = {task.method_id: task for task in watcher.TASKS}
    assert (tasks["dcspg_original"].gpu_index, tasks["dcspg_original"].gpu_uuid) == (
        0,
        "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    )
    assert tasks["dcspg_original"].gpu_bus_id == "00000000:16:00.0"
    assert (tasks["dcspg_farbg"].gpu_index, tasks["dcspg_farbg"].gpu_uuid) == (
        1,
        "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    )
    assert tasks["dcspg_farbg"].gpu_bus_id == "00000000:27:00.0"


def test_exact_adapter_argv_for_fresh_and_resume() -> None:
    base = (
        "/home/ly/BasicIRSTD/infrarenet/bin/python",
        "/home/ly/EviSIRST_main/run_irstd_dcspg_test_selected_v1.py",
        "--method-id",
        "dcspg_original",
    )
    assert watcher.adapter_argv("dcspg_original") == base
    assert watcher.adapter_argv("dcspg_original", resume=True) == (*base, "--resume")
    with pytest.raises(watcher.DCSPGWatcherError):
        watcher.adapter_argv("clean_original")


def test_launch_contract_matches_adapter_exact_eight_fields() -> None:
    contract = watcher.launch_contract("dcspg_farbg", resume=True)
    assert set(contract) == {
        "method_id",
        "route",
        "resume",
        "gpu_uuid",
        "gpu_bus_id",
        "singleton_lock_path",
        "task_lock_path",
        "gpu_lock_path",
    }
    assert contract["route"] == "dcspg_farbg"
    assert contract["resume"] is True
    assert Path(contract["singleton_lock_path"]) == watcher.WATCHER_LOCK
    assert Path(contract["task_lock_path"]).name == "dcspg_farbg.lock"
    assert Path(contract["gpu_lock_path"]).name == (
        "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640.lock"
    )


def test_authorization_and_source_placeholders_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "CURRENT_FORMAL_AUTHORIZATION_SHA256", None)
    monkeypatch.setattr(watcher, "FROZEN_WATCHER_TEST_SHA256", None)
    assert watcher.authorization_is_valid() is False
    with pytest.raises(watcher.DCSPGWatcherError, match="not sealed"):
        watcher.frozen_source_allowlist()


def test_active_authorization_preserves_initial_adapter_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        watcher.FORMAL_AUTHORIZATION_SHA256
        == watcher.INITIAL_LAUNCH_AUTHORIZATION_SHA256
    )
    assert (
        watcher._sha256(watcher.ADAPTER_SOURCE)
        == watcher.INITIAL_LAUNCH_ADAPTER_SHA256
    )
    monkeypatch.setattr(watcher, "FORMAL_AUTHORIZATION_SHA256", "a" * 64)
    assert watcher.authorization_is_valid() is False


def test_adapter_hash_normalizes_only_authorization_literal(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text(
        "FORMAL_AUTHORIZATION_SHA256: str | None = None\nVALUE=1\n",
        encoding="utf-8",
    )
    second.write_text(
        'FORMAL_AUTHORIZATION_SHA256: str | None = "' + "a" * 64 + '"\nVALUE=1\n',
        encoding="utf-8",
    )
    assert watcher._normalized_authorization_source_sha256(first) == watcher._normalized_authorization_source_sha256(second)
    second.write_text(
        'FORMAL_AUTHORIZATION_SHA256: str | None = "' + "a" * 64 + '"\nVALUE=2\n',
        encoding="utf-8",
    )
    assert watcher._normalized_authorization_source_sha256(first) != watcher._normalized_authorization_source_sha256(second)


def test_exact_runtime_closure_rejects_four_dataset_and_internal_mutation(
    tmp_path: Path,
) -> None:
    for raw_relative in watcher.FROZEN_RUNTIME_SOURCE_SHA256:
        relative = raw_relative.removesuffix("#normalized_authorization_literal")
        source = watcher.PROJECT_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    watcher.assert_frozen_sources(
        project_root=tmp_path,
        source_allowlist=watcher.FROZEN_RUNTIME_SOURCE_SHA256,
    )
    four_dataset = tmp_path / "experiments/four_dataset_models_seed42_v1.py"
    four_dataset.write_bytes(four_dataset.read_bytes() + b"\n# tamper\n")
    with pytest.raises(watcher.DCSPGWatcherError, match="frozen source differs"):
        watcher.assert_frozen_sources(
            project_root=tmp_path,
            source_allowlist=watcher.FROZEN_RUNTIME_SOURCE_SHA256,
        )
    shutil.copy2(
        watcher.PROJECT_ROOT / "experiments/four_dataset_models_seed42_v1.py",
        four_dataset,
    )
    internal = tmp_path / "model/_internal/tpd_clean.py"
    internal.write_bytes(internal.read_bytes() + b"\n# tamper\n")
    with pytest.raises(watcher.DCSPGWatcherError, match="frozen source differs"):
        watcher.assert_frozen_sources(
            project_root=tmp_path,
            source_allowlist=watcher.FROZEN_RUNTIME_SOURCE_SHA256,
        )
    internal.write_bytes((watcher.PROJECT_ROOT / "model/_internal/tpd_clean.py").read_bytes())
    (tmp_path / "model/_internal/evil.py").write_text("x=1\n", encoding="utf-8")
    with pytest.raises(watcher.DCSPGWatcherError, match="path set"):
        watcher.assert_frozen_sources(
            project_root=tmp_path,
            source_allowlist=watcher.FROZEN_RUNTIME_SOURCE_SHA256,
        )


def test_authorization_closure_survives_same_hash_injected_into_adapter_and_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watcher_source = tmp_path / "watcher.py"
    adapter_source = tmp_path / "adapter.py"
    authorization = tmp_path / "authorization.json"
    initial = "a" * 64
    watcher_source.write_text(
        'FORMAL_AUTHORIZATION_SHA256: str | None = "'
        + initial
        + '"\nCURRENT_FORMAL_AUTHORIZATION_SHA256: str | None = None\nVALUE=1\n',
        encoding="utf-8",
    )
    adapter_source.write_text(
        'FORMAL_AUTHORIZATION_SHA256: str | None = "'
        + initial
        + '"\nVALUE=1\n',
        encoding="utf-8",
    )
    allowlist = {"adapter.py#normalized_authorization_literal": "b" * 64}
    monkeypatch.setattr(watcher, "WATCHER_SOURCE", watcher_source)
    monkeypatch.setattr(watcher, "ADAPTER_SOURCE", adapter_source)
    monkeypatch.setattr(watcher, "AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(watcher, "assert_frozen_sources", lambda: None)
    monkeypatch.setattr(watcher, "frozen_source_allowlist", lambda: allowlist)
    monkeypatch.setattr(watcher, "FORMAL_AUTHORIZATION_SHA256", initial)
    monkeypatch.setattr(watcher, "INITIAL_LAUNCH_AUTHORIZATION_SHA256", initial)
    monkeypatch.setattr(
        watcher,
        "INITIAL_LAUNCH_ADAPTER_SHA256",
        watcher._sha256(adapter_source),
    )
    payload = {
        "schema": watcher.AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED",
        "task_manifest_sha256": watcher._canonical_sha256(watcher.task_manifest()),
        "watcher_normalized_sha256": watcher._normalized_authorization_source_sha256(
            watcher_source
        ),
        "adapter_normalized_sha256": watcher._normalized_authorization_source_sha256(
            adapter_source
        ),
        "source_allowlist_sha256": watcher._canonical_sha256(allowlist),
        "target_output_root_initially_absent": True,
        "quarantined_output_reused": False,
    }
    _write_json(authorization, payload)
    digest = watcher._sha256(authorization)
    monkeypatch.setattr(watcher, "CURRENT_FORMAL_AUTHORIZATION_SHA256", digest)
    assert watcher.authorization_is_valid() is True

    injected = (
        'FORMAL_AUTHORIZATION_SHA256: str | None = "'
        + initial
        + '"\nCURRENT_FORMAL_AUTHORIZATION_SHA256: str | None = "'
        + digest
        + '"\nVALUE=1\n'
    )
    watcher_source.write_text(injected, encoding="utf-8")
    assert watcher.authorization_is_valid() is True


def test_fresh_live_root_is_absent_while_existing_quarantine_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "test_selected_v1"
    quarantine = tmp_path / "test_selected_v1_invalid_pre_audit_20260819T043550"
    quarantine.mkdir()
    monkeypatch.setattr(watcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", live)
    monkeypatch.setattr(watcher, "QUARANTINED_OUTPUT_ROOT", quarantine)
    watcher.assert_initial_target_absent()
    live.mkdir()
    with pytest.raises(watcher.DCSPGWatcherError, match="must be absent"):
        watcher.assert_initial_target_absent()
    with pytest.raises(watcher.DCSPGWatcherError, match="quarantined"):
        watcher.assert_initial_target_absent(quarantine)


@pytest.mark.parametrize("completed", [1, 499, 500, 713, 999])
def test_every_partial_epoch_is_routed_to_strict_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed: int
) -> None:
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    _make_progress(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original", completed)
    result = watcher.probe_progress("dcspg_original")
    assert result["state"] == "resume"
    assert result["resume"] is True
    assert result["test_history_count"] == max(0, completed - 500)


def test_epoch500_has_zero_test_records_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    _make_progress(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original", 500)
    result = watcher.probe_progress("dcspg_original")
    assert result == {
        "method_id": "dcspg_original",
        "state": "resume",
        "completed_epoch": 500,
        "training_history_count": 500,
        "test_history_count": 0,
        "resume": True,
    }


def test_epoch501_activation_crash_after_started_ledger_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    run_dir = _make_progress(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original", 500)
    _write_json(run_dir / "test_access_started.json", {"started": True})
    assert watcher.probe_progress("dcspg_original")["resume"] is True
    _write_json(run_dir / "test_access_verified.json", {"verified": True})
    assert watcher.probe_progress("dcspg_original")["test_history_count"] == 0


def test_test_access_before_epoch500_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    run_dir = _make_progress(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original", 499)
    _write_json(run_dir / "test_access_started.json", {"started": True})
    with pytest.raises(watcher.DCSPGWatcherError, match="before epoch 501"):
        watcher.probe_progress("dcspg_original")


def test_latest_ahead_of_sidecars_routes_to_runner_strict_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    run_dir = _make_progress(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original", 501)
    payload = json.loads((run_dir / "test_history.json").read_text(encoding="utf-8"))
    payload["completed_epoch"] = 500
    payload["test_history"] = []
    _write_json(run_dir / "test_history.json", payload)
    (run_dir / "last_training_state.pth.tar").write_bytes(b"strict-runner-fixture")
    result = watcher.probe_progress("dcspg_original")
    assert result["resume"] is True
    assert result["sidecar_transaction_pending_strict_runner_validation"] is True


def test_epoch1000_partial_finalization_routes_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    run_dir = _make_progress(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original", 1000)
    (run_dir / "last_training_state.pth.tar").write_bytes(b"latest")
    published = run_dir / "published_weights"
    published.mkdir()
    (published / "EviSIRST_best_mIoU.pth.tar").write_bytes(b"one-role-only")
    result = watcher.probe_progress("dcspg_original")
    assert result["state"] == "resume"
    assert result["finalization_transaction_pending_strict_runner_validation"] is True


def test_completion_recomputes_500_epoch_winners_and_two_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "assert_frozen_sources", lambda: None)
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    _make_complete(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original")
    result = watcher.probe_completion("dcspg_original")
    assert result["training_history_count"] == 1000
    assert result["test_history_count"] == 500
    assert result["test_epochs"] == {"first": 501, "last": 1000, "count": 500}
    assert result["published_weight_file_count"] == 2
    assert result["published_checkpoints"]["best_miou"]["epoch"] == 700
    assert result["published_checkpoints"]["best_pd"]["epoch"] == 800
    run_dir = watcher.task_for_method("dcspg_original").run_dir
    assert result["latest_recovery_sha256"] == watcher._sha256(
        run_dir / "last_training_state.pth.tar"
    )


def test_strict_completion_rejects_metadata_ledger_frontier_and_tensor_tampers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(watcher, "assert_frozen_sources", lambda: None)
    monkeypatch.setattr(watcher, "FORMAL_OUTPUT_ROOT", tmp_path / "live")
    run_dir = _make_complete(watcher.FORMAL_OUTPUT_ROOT, "dcspg_original")
    selection_path = run_dir / "selection_record.json"
    summary_path = run_dir / "summary.json"
    started_path = run_dir / "test_access_started.json"
    original_selection = selection_path.read_bytes()
    original_summary = summary_path.read_bytes()
    original_started = started_path.read_bytes()

    selection = json.loads(original_selection)
    selection["roles"]["best_miou"]["epoch"] = 701
    _write_json(selection_path, selection)
    with pytest.raises(watcher.DCSPGWatcherError):
        watcher.probe_completion("dcspg_original")
    selection_path.write_bytes(original_selection)

    selection = json.loads(original_selection)
    identity = selection["run_identity"]
    identity["source_manifest"]["files"]["train.py"] = "0" * 64
    unsigned = dict(identity)
    unsigned.pop("identity_sha256")
    identity["identity_sha256"] = watcher._canonical_sha256(unsigned)
    _write_json(selection_path, selection)
    with pytest.raises(watcher.DCSPGWatcherError, match="source manifest"):
        watcher.probe_completion("dcspg_original")
    selection_path.write_bytes(original_selection)

    started = json.loads(original_started)
    started["first_access_epoch"] = 500
    _write_json(started_path, started)
    with pytest.raises(watcher.DCSPGWatcherError):
        watcher.probe_completion("dcspg_original")
    started_path.write_bytes(original_started)

    candidate_dir = run_dir / "candidates/test_selected"
    (candidate_dir / "epoch_0501.pth.tar").write_bytes(b"old-frontier")
    with pytest.raises(watcher.DCSPGWatcherError, match="frontier file set"):
        watcher.probe_completion("dcspg_original")
    (candidate_dir / "epoch_0501.pth.tar").unlink()

    (run_dir / "published_weights/extra.pth").write_bytes(b"extra")
    with pytest.raises(watcher.DCSPGWatcherError, match="exactly two"):
        watcher.probe_completion("dcspg_original")
    (run_dir / "published_weights/extra.pth").unlink()

    selection = json.loads(original_selection)
    selection["selection_is_optimistic"] = False
    _write_json(selection_path, selection)
    summary = json.loads(original_summary)
    summary["selection_record_sha256"] = watcher._sha256(selection_path)
    _write_json(summary_path, summary)
    with pytest.raises(watcher.DCSPGWatcherError, match="selection record"):
        watcher.probe_completion("dcspg_original")
    selection_path.write_bytes(original_selection)
    summary_path.write_bytes(original_summary)

    weight_path = run_dir / "published_weights/EviSIRST_best_mIoU.pth.tar"
    weight_backup = tmp_path / "best_miou.backup"
    shutil.copy2(weight_path, weight_backup)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    first_key = next(
        key
        for key, tensor in checkpoint["state_dict"].items()
        if tensor.is_floating_point() and tensor.numel() > 0
    )
    checkpoint["state_dict"][first_key] = checkpoint["state_dict"][first_key].clone()
    checkpoint["state_dict"][first_key].reshape(-1)[0] += 1.0
    torch.save(checkpoint, weight_path)
    new_weight_sha = watcher._sha256(weight_path)
    selection = json.loads(original_selection)
    summary = json.loads(original_summary)
    selection["roles"]["best_miou"]["published_checkpoint"]["sha256"] = new_weight_sha
    summary["published_checkpoints"]["best_miou"]["sha256"] = new_weight_sha
    _write_json(selection_path, selection)
    summary["selection_record_sha256"] = watcher._sha256(selection_path)
    _write_json(summary_path, summary)
    with pytest.raises(watcher.DCSPGWatcherError):
        watcher.probe_completion("dcspg_original")
    shutil.copy2(weight_backup, weight_path)
    selection_path.write_bytes(original_selection)
    summary_path.write_bytes(original_summary)

    candidate_path = candidate_dir / "epoch_0700.pth.tar"
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    first_key = next(
        key
        for key, tensor in candidate["state_dict"].items()
        if tensor.is_floating_point() and tensor.numel() > 0
    )
    candidate["state_dict"][first_key] = candidate["state_dict"][first_key].clone()
    candidate["state_dict"][first_key].reshape(-1)[0] += 2.0
    torch.save(candidate, candidate_path)
    candidate_sha = watcher._sha256(candidate_path)
    latest_path = run_dir / "last_training_state.pth.tar"
    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
    latest["candidate_artifacts"][700]["sha256"] = candidate_sha
    torch.save(latest, latest_path)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    checkpoint["candidate_sha256"] = candidate_sha
    torch.save(checkpoint, weight_path)
    weight_sha = watcher._sha256(weight_path)
    selection = json.loads(original_selection)
    summary = json.loads(original_summary)
    selection["roles"]["best_miou"]["candidate"]["sha256"] = candidate_sha
    selection["roles"]["best_miou"]["published_checkpoint"]["sha256"] = weight_sha
    summary["published_checkpoints"]["best_miou"]["sha256"] = weight_sha
    _write_json(selection_path, selection)
    summary["selection_record_sha256"] = watcher._sha256(selection_path)
    _write_json(summary_path, summary)
    with pytest.raises(watcher.DCSPGWatcherError):
        watcher.probe_completion("dcspg_original")


def _proc_entry(root: Path, pid: int, argv: tuple[str, ...], ppid: int) -> None:
    entry = root / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes(
        b"\0".join(value.encode("utf-8") for value in argv) + b"\0"
    )
    (entry / "status").write_text(f"Name:\ttest\nPPid:\t{ppid}\n", encoding="utf-8")


def test_registered_first_worker_is_allowed_but_direct_duplicate_is_rejected(
    tmp_path: Path,
) -> None:
    process = mock.Mock(pid=100)
    process.poll.return_value = None
    task = watcher.task_for_method("dcspg_original")
    job = watcher.RunningJob(
        task=task,
        resume=False,
        process=process,
        gpu=watcher.GPU(task.gpu_index, task.gpu_uuid, task.gpu_bus_id, 0),
        task_lock_fd=10,
        gpu_lock_fd=11,
        singleton_lock_fd=12,
        log_handle=io.BytesIO(),
        log_path=tmp_path / "log",
    )
    _proc_entry(tmp_path, 100, watcher.timed_adapter_argv(task.method_id), 1)
    _proc_entry(tmp_path, 101, watcher.adapter_argv(task.method_id), 100)
    watcher.assert_no_forbidden_processes(authorized_jobs=(job,), proc_root=tmp_path)
    _proc_entry(tmp_path, 102, watcher.adapter_argv("dcspg_farbg"), 1)
    with pytest.raises(watcher.DCSPGWatcherError, match="forbidden"):
        watcher.assert_no_forbidden_processes(authorized_jobs=(job,), proc_root=tmp_path)


def test_worker_environment_and_spawn_pass_all_three_lock_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = watcher.task_for_method("dcspg_farbg")
    fds = [os.open(tmp_path / name, os.O_RDWR | os.O_CREAT, 0o600) for name in ("s", "t", "g")]
    fake_log = io.BytesIO()
    fake_process = mock.Mock(pid=456)
    fake_process.poll.return_value = None
    gpu = watcher.GPU(task.gpu_index, task.gpu_uuid, task.gpu_bus_id, 0)
    monkeypatch.setattr(watcher, "authorization_is_valid", lambda: True)
    monkeypatch.setattr(watcher, "_try_lock", lambda _path: fds[1])
    forbidden = mock.Mock()
    monkeypatch.setattr(watcher, "assert_no_forbidden_processes", forbidden)
    monkeypatch.setattr(watcher, "sample_idle_gpus", lambda: {task.gpu_uuid: gpu})
    monkeypatch.setattr(watcher, "_open_log", lambda _path: fake_log)
    popen = mock.Mock(return_value=fake_process)
    monkeypatch.setattr(watcher.subprocess, "Popen", popen)
    try:
        job = watcher.launch_task(
            task,
            resume=True,
            gpu=gpu,
            gpu_lock_fd=fds[2],
            singleton_lock_fd=fds[0],
            authorized_jobs=(),
        )
        assert job is not None
        kwargs = popen.call_args.kwargs
        assert tuple(kwargs["pass_fds"]) == tuple(fds)
        assert kwargs["start_new_session"] is True
        assert tuple(popen.call_args.args[0]) == watcher.timed_adapter_argv(
            task.method_id, resume=True
        )
        env = kwargs["env"]
        assert env[watcher.METHOD_ENV] == task.method_id
        assert env[watcher.RESUME_ENV] == "1"
        assert env[watcher.GPU_UUID_ENV] == task.gpu_uuid
        assert env[watcher.GPU_BUS_ENV] == task.gpu_bus_id
        assert env[watcher.SINGLETON_FD_ENV] == str(fds[0])
        assert env[watcher.TASK_FD_ENV] == str(fds[1])
        assert env[watcher.GPU_FD_ENV] == str(fds[2])
        assert env["CUDA_VISIBLE_DEVICES"] == task.gpu_uuid
        watcher._release_job(job)
    finally:
        for descriptor in fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_abort_contains_process_group_before_releasing_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = watcher.TASKS[0]
    process = mock.Mock(pid=987)
    process.poll.return_value = None
    process.wait.return_value = 0
    job = watcher.RunningJob(
        task=task,
        resume=False,
        process=process,
        gpu=watcher.GPU(task.gpu_index, task.gpu_uuid, task.gpu_bus_id, 0),
        task_lock_fd=10,
        gpu_lock_fd=11,
        singleton_lock_fd=12,
        log_handle=io.BytesIO(),
        log_path=tmp_path / "log",
    )
    events: list[object] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(watcher, "_release_job", lambda _job: events.append("release"))
    watcher._abort_job(job)
    assert events == [(987, signal.SIGTERM), "release"]
    process.wait.assert_called_once_with(timeout=watcher.TERM_TIMEOUT_SECONDS)


def test_readonly_preflight_performs_no_writes_or_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(watcher, "CURRENT_FORMAL_AUTHORIZATION_SHA256", "a" * 64)
    monkeypatch.setattr(watcher, "_runtime_dependencies_are_valid", lambda: None)
    monkeypatch.setattr(watcher, "assert_frozen_sources", lambda: None)
    monkeypatch.setattr(watcher, "authorization_is_valid", lambda: True)
    monkeypatch.setattr(watcher, "_queue_states", lambda: (True, {task.method_id: {"state": "fresh"} for task in watcher.TASKS}))
    monkeypatch.setattr(watcher, "assert_no_forbidden_processes", lambda: None)
    idle = {
        task.gpu_uuid: watcher.GPU(
            task.gpu_index, task.gpu_uuid, task.gpu_bus_id, 0
        )
        for task in watcher.TASKS
    }
    monkeypatch.setattr(watcher, "sample_idle_gpus", lambda: idle)
    monkeypatch.setattr(watcher, "_lock_status_readonly", lambda _path: "absent")
    monkeypatch.setattr(watcher, "frozen_source_allowlist", lambda: {"x": "b" * 64})
    popen = mock.Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(watcher.subprocess, "Popen", popen)
    result = watcher.readonly_preflight()
    assert result["writes_performed"] is False
    assert result["workers_launched"] is False
    assert all(result["pinned_gpus_idle"].values())
    popen.assert_not_called()


def test_formal_mode_with_none_authorization_fails_before_lock_or_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "CURRENT_FORMAL_AUTHORIZATION_SHA256", None)
    lock = mock.Mock(side_effect=AssertionError("must not create a lock"))
    popen = mock.Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(watcher, "_try_lock", lock)
    monkeypatch.setattr(watcher.subprocess, "Popen", popen)
    with pytest.raises(watcher.DCSPGWatcherError):
        watcher.main(["--formal"])
    lock.assert_not_called()
    popen.assert_not_called()


def test_supervise_mixed_complete_and_resume_ledgers_complete_and_launches_only_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, partial = watcher.TASKS
    complete_payload = {"schema": watcher.COMPLETION_SCHEMA, "method_id": original.method_id}
    states = {
        original.method_id: {"state": "complete", "resume": False},
        partial.method_id: {"state": "resume", "resume": True},
    }
    process = mock.Mock(pid=77)
    process.poll.return_value = 0
    partial_job = watcher.RunningJob(
        task=partial,
        resume=True,
        process=process,
        gpu=watcher.GPU(partial.gpu_index, partial.gpu_uuid, partial.gpu_bus_id, 0),
        task_lock_fd=-1,
        gpu_lock_fd=-1,
        singleton_lock_fd=5,
        log_handle=io.BytesIO(),
        log_path=Path("/tmp/unused"),
    )
    monkeypatch.setattr(watcher, "authorization_is_valid", lambda: True)
    monkeypatch.setattr(watcher, "_queue_states", lambda: (False, states))
    monkeypatch.setattr(watcher, "probe_completion", lambda task: {**complete_payload, "method_id": task.method_id})
    write = mock.Mock()
    monkeypatch.setattr(watcher, "_write_completion", write)
    monkeypatch.setattr(
        watcher,
        "claim_pinned_gpu",
        lambda task, **_kwargs: (
            watcher.GPU(task.gpu_index, task.gpu_uuid, task.gpu_bus_id, 0),
            9,
        ),
    )
    launch = mock.Mock(return_value=partial_job)
    monkeypatch.setattr(watcher, "launch_task", launch)
    monkeypatch.setattr(watcher, "_release_job", lambda _job: None)
    assert watcher.supervise(singleton_lock_fd=5, sleep=lambda _seconds: None) == 0
    launch.assert_called_once()
    assert launch.call_args.kwargs["resume"] is True
    assert launch.call_args.args[0] == partial
    assert [call.args[0].method_id for call in write.call_args_list] == [
        original.method_id,
        partial.method_id,
    ]
