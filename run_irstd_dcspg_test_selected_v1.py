#!/usr/bin/env python3
"""Capability-gated adapter for the formal DCS-PG test-selected runner.

Direct execution is deliberately disabled until a separately frozen watcher
authorization hash is installed.  The adapter owns no training or completion
logic: it authenticates one watcher route, proves three inherited locks, then
temporarily supplies the independently reviewed architecture digests to the
underlying runner for exactly one call.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import os
import re
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
ADAPTER_PATH = PROJECT_ROOT / "run_irstd_dcspg_test_selected_v1.py"
WATCHER_MODULE = "tools.run_irstd_dcspg_test_selected_when_idle"
RUNNER_MODULE = "train_irstd_dcspg_ablation_test_selected_v1"

METHOD_IDS = ("dcspg_original", "dcspg_farbg")
FORMAL_AUTHORIZATION_SHA256: str | None = "5cdea5fbba9f9da2e3533fab739f1608493bdd34de98f29c693cfdc1f94694ae"
FINAL_ARCHITECTURE_SOURCE_SHA256 = (
    "70f9354b01bed31b6c64f7fe17d7bd0b9159ccee9887f0b9e636e508b240960d"
)
FINAL_ARCHITECTURE_TEST_SHA256 = (
    "2917f0040fc37344a1279458c3c3a96208dcd2c63193b83ecaba7394edb3f17c"
)

ENV_PREFIX = "EVISIRST_DCSPG_TEST_SELECTED_"
METHOD_ENV = ENV_PREFIX + "METHOD_ID"
ROUTE_ENV = ENV_PREFIX + "ROUTE"
RESUME_ENV = ENV_PREFIX + "RESUME"
GPU_UUID_ENV = ENV_PREFIX + "GPU_UUID"
GPU_BUS_ENV = ENV_PREFIX + "GPU_BUS_ID"
SINGLETON_FD_ENV = ENV_PREFIX + "SINGLETON_LOCK_FD"
TASK_FD_ENV = ENV_PREFIX + "TASK_LOCK_FD"
GPU_FD_ENV = ENV_PREFIX + "GPU_LOCK_FD"

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
_BUS_RE = re.compile(r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")
_ROUTE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class DCSPGAdapterError(RuntimeError):
    """The caller lacks the immutable watcher launch capability."""


def _method(method_id: str) -> str:
    if method_id not in METHOD_IDS:
        raise DCSPGAdapterError("unsupported formal method ID")
    return method_id


def _resume(value: bool) -> bool:
    if type(value) is not bool:
        raise DCSPGAdapterError("resume must be one boolean")
    return value


def _watcher() -> Any:
    try:
        return importlib.import_module(WATCHER_MODULE)
    except Exception as exc:
        raise DCSPGAdapterError("formal watcher is unavailable") from exc


def normalized_authorization_sha256() -> str:
    """Return the matching lower-case watcher authorization or fail closed."""

    expected = FORMAL_AUTHORIZATION_SHA256
    if not isinstance(expected, str) or _SHA_RE.fullmatch(expected) is None:
        raise DCSPGAdapterError("formal adapter authorization is not frozen")
    watcher = _watcher()
    observed = getattr(watcher, "FORMAL_AUTHORIZATION_SHA256", None)
    validator = getattr(watcher, "authorization_is_valid", None)
    try:
        valid = callable(validator) and validator() is True
    except Exception as exc:
        raise DCSPGAdapterError("watcher authorization validation failed") from exc
    if observed != expected or not valid:
        raise DCSPGAdapterError("watcher authorization is invalid")
    return expected


def _launch_contract(method_id: str, resume: bool) -> dict[str, Any]:
    watcher = _watcher()
    provider = getattr(watcher, "launch_contract", None)
    if not callable(provider):
        raise DCSPGAdapterError("watcher launch-contract API is unavailable")
    try:
        raw = provider(method_id, resume=resume)
    except Exception as exc:
        raise DCSPGAdapterError("watcher launch contract failed") from exc
    required = {
        "method_id",
        "route",
        "resume",
        "gpu_uuid",
        "gpu_bus_id",
        "singleton_lock_path",
        "task_lock_path",
        "gpu_lock_path",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise DCSPGAdapterError("watcher launch contract fields differ")
    contract = dict(raw)
    route = contract.get("route")
    uuid = contract.get("gpu_uuid")
    bus = contract.get("gpu_bus_id")
    if (
        contract.get("method_id") != method_id
        or contract.get("resume") is not resume
        or not isinstance(route, str)
        or _ROUTE_RE.fullmatch(route) is None
        or not isinstance(uuid, str)
        or _UUID_RE.fullmatch(uuid) is None
        or not isinstance(bus, str)
        or _BUS_RE.fullmatch(bus) is None
    ):
        raise DCSPGAdapterError("watcher method/route/GPU identity differs")
    for name in ("singleton_lock_path", "task_lock_path", "gpu_lock_path"):
        try:
            path = Path(os.fspath(contract[name]))
        except TypeError as exc:
            raise DCSPGAdapterError("watcher lock path is malformed") from exc
        if not path.is_absolute() or path != path.resolve(strict=False):
            raise DCSPGAdapterError("watcher lock path is not canonical absolute")
        contract[name] = path
    return contract


def _try_existing_lock(path: Path) -> int | None:
    if path.is_symlink() or not path.is_file():
        raise DCSPGAdapterError("required preexisting lock is unsafe")
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DCSPGAdapterError("required preexisting lock cannot be opened") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _assert_inherited_locked_file(raw_fd: str | None, path: Path) -> int:
    if raw_fd is None or not raw_fd.isascii() or not raw_fd.isdecimal():
        raise DCSPGAdapterError("inherited lock FD is missing")
    descriptor = int(raw_fd)
    if descriptor < 3:
        raise DCSPGAdapterError("inherited lock FD is unsafe")
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
    except OSError as exc:
        raise DCSPGAdapterError("inherited lock FD is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(path_stat.st_mode)
        or not stat.S_ISREG(descriptor_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise DCSPGAdapterError("inherited lock FD identity differs")
    contender = _try_existing_lock(path)
    if contender is not None:
        os.close(contender)
        raise DCSPGAdapterError("inherited lock file is not currently locked")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise DCSPGAdapterError(
            "inherited lock FD does not own the current lock"
        ) from exc
    except OSError as exc:
        raise DCSPGAdapterError("inherited lock FD cannot prove ownership") from exc
    return descriptor


def assert_launch_capability(method_id: str, resume: bool) -> Mapping[str, Any]:
    """Validate authorization, route identity, GPU identity, and three locks."""

    method_id = _method(method_id)
    resume = _resume(resume)
    authorization = normalized_authorization_sha256()
    contract = _launch_contract(method_id, resume)
    expected_env = {
        METHOD_ENV: method_id,
        ROUTE_ENV: contract["route"],
        RESUME_ENV: "1" if resume else "0",
        GPU_UUID_ENV: contract["gpu_uuid"],
        GPU_BUS_ENV: contract["gpu_bus_id"],
        "CUDA_VISIBLE_DEVICES": contract["gpu_uuid"],
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    for name, expected in expected_env.items():
        if os.environ.get(name) != expected:
            raise DCSPGAdapterError(f"launch capability environment differs: {name}")
    if "NVIDIA_VISIBLE_DEVICES" in os.environ:
        raise DCSPGAdapterError("NVIDIA_VISIBLE_DEVICES must be absent")

    try:
        descriptors = (
            _assert_inherited_locked_file(
                os.environ.get(SINGLETON_FD_ENV), contract["singleton_lock_path"]
            ),
            _assert_inherited_locked_file(
                os.environ.get(TASK_FD_ENV), contract["task_lock_path"]
            ),
            _assert_inherited_locked_file(
                os.environ.get(GPU_FD_ENV), contract["gpu_lock_path"]
            ),
        )
    except DCSPGAdapterError:
        raise
    except Exception as exc:
        raise DCSPGAdapterError("inherited watcher lock capability differs") from exc
    if len(set(descriptors)) != 3:
        raise DCSPGAdapterError("inherited watcher lock FDs are not distinct")
    identities = set()
    for descriptor in descriptors:
        try:
            descriptor_stat = os.fstat(descriptor)
        except OSError as exc:
            raise DCSPGAdapterError("inherited watcher lock FD is unavailable") from exc
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise DCSPGAdapterError("inherited watcher lock FD is not regular")
        identities.add((descriptor_stat.st_dev, descriptor_stat.st_ino))
    if len(identities) != 3:
        raise DCSPGAdapterError("inherited watcher lock files are not distinct")
    return {**contract, "authorization_sha256": authorization}


@contextmanager
def _temporary_architecture_gate(runner: Any) -> Iterator[None]:
    names = (
        "ARCHITECTURE_FINAL_AUDIT_GO",
        "EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256",
        "EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256",
    )
    if any(not hasattr(runner, name) for name in names):
        raise DCSPGAdapterError("underlying runner gate API is incomplete")
    original = tuple(getattr(runner, name) for name in names)
    if original != (False, None, None):
        raise DCSPGAdapterError("underlying runner static gate is not closed")
    runner.ARCHITECTURE_FINAL_AUDIT_GO = True
    runner.EXPECTED_GUARD_ARCHITECTURE_SOURCE_SHA256 = (
        FINAL_ARCHITECTURE_SOURCE_SHA256
    )
    runner.EXPECTED_GUARD_ARCHITECTURE_TEST_SHA256 = FINAL_ARCHITECTURE_TEST_SHA256
    try:
        yield
    finally:
        for name, value in zip(names, original):
            setattr(runner, name, value)


def _runner_args(runner: Any, method_id: str, resume: bool) -> Any:
    argv = ["--method-id", method_id]
    if resume:
        argv.append("--resume")
    args = runner.parse_args(argv)
    expected = {
        "method_id": method_id,
        "resume": resume,
        "device": "cuda:0",
        "architecture_seed": 42,
        "run_seed": 42,
        "epochs": 1000,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1e-3,
        "min_lr": 1e-5,
        "warmup_epochs": 10,
        "test_begin": 501,
        "test_every": 1,
        "cpu_threads": 16,
        "dataset_root": Path("/home/ly/SCTransNet_main/datasets"),
        "split_root": PROJECT_ROOT / "splits/v2",
        "output_root": (
            PROJECT_ROOT
            / "runs/irstd_model_design/dcspg_ablation_v1/test_selected_v1"
        ),
        "smoke": False,
        "smoke_id": "fixture",
        "max_train_samples": None,
        "max_test_images": None,
    }
    if any(getattr(args, name, object()) != value for name, value in expected.items()):
        raise DCSPGAdapterError("underlying formal runner arguments differ")
    return args


def run_route(method_id: str, resume: bool = False) -> Path:
    method_id = _method(method_id)
    resume = _resume(resume)
    assert_launch_capability(method_id, resume)
    try:
        runner = importlib.import_module(RUNNER_MODULE)
    except Exception as exc:
        raise DCSPGAdapterError("underlying formal runner is unavailable") from exc
    with _temporary_architecture_gate(runner):
        args = _runner_args(runner, method_id, resume)
        result = runner.run(args)
    if not isinstance(result, Path):
        raise DCSPGAdapterError("underlying formal runner returned no path")
    return result


def worker_argv(method_id: str, resume: bool) -> tuple[str, ...]:
    method_id = _method(method_id)
    resume = _resume(resume)
    argv = [os.fspath(PYTHON_BIN), os.fspath(ADAPTER_PATH), "--method-id", method_id]
    if resume:
        argv.append("--resume")
    return tuple(argv)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=METHOD_IDS, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(run_route(args.method_id, args.resume), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "ADAPTER_PATH",
    "DCSPGAdapterError",
    "FINAL_ARCHITECTURE_SOURCE_SHA256",
    "FINAL_ARCHITECTURE_TEST_SHA256",
    "FORMAL_AUTHORIZATION_SHA256",
    "METHOD_IDS",
    "assert_launch_capability",
    "normalized_authorization_sha256",
    "run_route",
    "worker_argv",
]
