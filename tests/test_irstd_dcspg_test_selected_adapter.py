from __future__ import annotations

import argparse
import fcntl
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

import run_irstd_dcspg_test_selected_v1 as adapter


AUTHORIZATION = "a" * 64
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
GPU_BUS = "00000000:16:00.0"


class _InjectedFailure(RuntimeError):
    pass


@contextmanager
def _locked_files(tmp_path: Path) -> Iterator[tuple[dict[str, Path], dict[str, int]]]:
    paths = {
        "singleton_lock_path": tmp_path / "singleton.lock",
        "task_lock_path": tmp_path / "task.lock",
        "gpu_lock_path": tmp_path / "gpu.lock",
    }
    descriptors: dict[str, int] = {}
    try:
        for name, path in paths.items():
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors[name] = descriptor
        yield paths, descriptors
    finally:
        for descriptor in descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass


def _fake_watcher(
    monkeypatch: pytest.MonkeyPatch,
    *,
    paths: dict[str, Path],
    method_id: str,
    resume: bool,
    valid: bool = True,
) -> types.ModuleType:
    watcher = types.ModuleType(adapter.WATCHER_MODULE)
    watcher.FORMAL_AUTHORIZATION_SHA256 = AUTHORIZATION
    watcher.authorization_is_valid = lambda: valid

    def launch_contract(observed_method: str, *, resume: bool) -> dict[str, Any]:
        return {
            "method_id": method_id,
            "route": f"formal_{method_id}",
            "resume": resume,
            "gpu_uuid": GPU_UUID,
            "gpu_bus_id": GPU_BUS,
            **paths,
        }

    watcher.launch_contract = launch_contract
    monkeypatch.setitem(sys.modules, adapter.WATCHER_MODULE, watcher)
    monkeypatch.setattr(adapter, "FORMAL_AUTHORIZATION_SHA256", AUTHORIZATION)
    return watcher


def _launch_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method_id: str,
    resume: bool,
    descriptors: dict[str, int],
) -> None:
    values = {
        adapter.METHOD_ENV: method_id,
        adapter.ROUTE_ENV: f"formal_{method_id}",
        adapter.RESUME_ENV: "1" if resume else "0",
        adapter.GPU_UUID_ENV: GPU_UUID,
        adapter.GPU_BUS_ENV: GPU_BUS,
        "CUDA_VISIBLE_DEVICES": GPU_UUID,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        adapter.SINGLETON_FD_ENV: str(descriptors["singleton_lock_path"]),
        adapter.TASK_FD_ENV: str(descriptors["task_lock_path"]),
        adapter.GPU_FD_ENV: str(descriptors["gpu_lock_path"]),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)


def _formal_namespace(method_id: str, resume: bool) -> argparse.Namespace:
    return argparse.Namespace(
        method_id=method_id,
        resume=resume,
        device="cuda:0",
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
        cpu_threads=16,
        dataset_root=Path("/home/ly/SCTransNet_main/datasets"),
        split_root=adapter.PROJECT_ROOT / "splits/v2",
        output_root=(
            adapter.PROJECT_ROOT
            / "runs/irstd_model_design/dcspg_ablation_v1/test_selected_v1"
        ),
        smoke=False,
        smoke_id="fixture",
        max_train_samples=None,
        max_test_images=None,
    )


def _fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    method_id: str,
    resume: bool,
    fail: bool,
) -> tuple[types.ModuleType, dict[str, Any]]:
    runner = types.ModuleType(adapter.RUNNER_MODULE)
    runner.ARCHITECTURE_FINAL_AUDIT_GO = False
    runner.EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 = None
    runner.EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 = None
    calls: dict[str, Any] = {"parse": [], "run": 0}

    def parse_args(argv: list[str]) -> argparse.Namespace:
        calls["parse"].append(tuple(argv))
        return _formal_namespace(method_id, resume)

    def run(args: argparse.Namespace) -> Path:
        calls["run"] += 1
        assert args.method_id == method_id
        assert runner.ARCHITECTURE_FINAL_AUDIT_GO is True
        assert runner.EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 == (
            adapter.FINAL_ARCHITECTURE_SOURCE_SHA256
        )
        assert runner.EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 == (
            adapter.FINAL_ARCHITECTURE_TEST_SHA256
        )
        if fail:
            raise _InjectedFailure("injected runner failure")
        return Path("/formal/result")

    runner.parse_args = parse_args
    runner.run = run
    monkeypatch.setitem(sys.modules, adapter.RUNNER_MODULE, runner)
    return runner, calls


def test_static_authorization_blocks_direct_invocation_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "FORMAL_AUTHORIZATION_SHA256", None)
    monkeypatch.setattr(
        adapter,
        "_watcher",
        lambda: (_ for _ in ()).throw(AssertionError("watcher was imported")),
    )
    with pytest.raises(adapter.DCSPGAdapterError, match="not frozen"):
        adapter.run_route("dcspg_original", False)
    with pytest.raises(adapter.DCSPGAdapterError, match="unsupported"):
        adapter.worker_argv("clean_original", False)


def test_worker_argv_is_exact_and_resume_is_explicit() -> None:
    base = (
        os.fspath(adapter.PYTHON_BIN),
        os.fspath(adapter.ADAPTER_PATH),
        "--method-id",
        "dcspg_original",
    )
    assert adapter.worker_argv("dcspg_original", False) == base
    assert adapter.worker_argv("dcspg_original", True) == (*base, "--resume")
    with pytest.raises(adapter.DCSPGAdapterError, match="boolean"):
        adapter.worker_argv("dcspg_original", 1)  # type: ignore[arg-type]


def test_wrong_route_environment_fails_before_runner_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _locked_files(tmp_path) as (paths, descriptors):
        _fake_watcher(
            monkeypatch,
            paths=paths,
            method_id="dcspg_original",
            resume=False,
        )
        _launch_env(
            monkeypatch,
            method_id="dcspg_original",
            resume=False,
            descriptors=descriptors,
        )
        monkeypatch.setenv(adapter.ROUTE_ENV, "wrong_route")
        sentinel = types.ModuleType(adapter.RUNNER_MODULE)
        monkeypatch.setitem(sys.modules, adapter.RUNNER_MODULE, sentinel)
        with pytest.raises(adapter.DCSPGAdapterError, match="environment differs"):
            adapter.run_route("dcspg_original", False)
        assert not hasattr(sentinel, "run")


def test_valid_capability_uses_three_distinct_locked_regular_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _locked_files(tmp_path) as (paths, descriptors):
        _fake_watcher(
            monkeypatch,
            paths=paths,
            method_id="dcspg_farbg",
            resume=True,
        )
        _launch_env(
            monkeypatch,
            method_id="dcspg_farbg",
            resume=True,
            descriptors=descriptors,
        )
        capability = adapter.assert_launch_capability("dcspg_farbg", True)
        assert capability["authorization_sha256"] == AUTHORIZATION
        assert capability["gpu_uuid"] == GPU_UUID
        assert capability["gpu_bus_id"] == GPU_BUS

        monkeypatch.setenv(
            adapter.GPU_FD_ENV, str(descriptors["task_lock_path"])
        )
        with pytest.raises(adapter.DCSPGAdapterError, match="identity differs"):
            adapter.assert_launch_capability("dcspg_farbg", True)


def test_spectator_fd_for_locked_inode_cannot_claim_lock_ownership(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owned.lock"
    owner = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    spectator = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert adapter._assert_inherited_locked_file(str(owner), path) == owner
        with pytest.raises(
            adapter.DCSPGAdapterError, match="does not own the current lock"
        ):
            adapter._assert_inherited_locked_file(str(spectator), path)
    finally:
        os.close(spectator)
        os.close(owner)


def test_gate_is_enabled_only_for_exact_runner_call_and_restored_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method_id = "dcspg_original"
    with _locked_files(tmp_path) as (paths, descriptors):
        _fake_watcher(
            monkeypatch,
            paths=paths,
            method_id=method_id,
            resume=False,
        )
        _launch_env(
            monkeypatch,
            method_id=method_id,
            resume=False,
            descriptors=descriptors,
        )
        runner, calls = _fake_runner(
            monkeypatch,
            method_id=method_id,
            resume=False,
            fail=True,
        )
        with pytest.raises(_InjectedFailure, match="injected"):
            adapter.run_route(method_id, False)
        assert calls == {"parse": [("--method-id", method_id)], "run": 1}
        assert runner.ARCHITECTURE_FINAL_AUDIT_GO is False
        assert runner.EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 is None
        assert runner.EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 is None


def test_valid_resume_route_returns_path_and_restores_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method_id = "dcspg_farbg"
    with _locked_files(tmp_path) as (paths, descriptors):
        _fake_watcher(
            monkeypatch,
            paths=paths,
            method_id=method_id,
            resume=True,
        )
        _launch_env(
            monkeypatch,
            method_id=method_id,
            resume=True,
            descriptors=descriptors,
        )
        runner, calls = _fake_runner(
            monkeypatch,
            method_id=method_id,
            resume=True,
            fail=False,
        )
        assert adapter.run_route(method_id, True) == Path("/formal/result")
        assert calls == {
            "parse": [("--method-id", method_id, "--resume")],
            "run": 1,
        }
        assert runner.ARCHITECTURE_FINAL_AUDIT_GO is False
        assert runner.EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 is None
        assert runner.EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 is None
