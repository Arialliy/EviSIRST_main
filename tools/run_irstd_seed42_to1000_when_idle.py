#!/usr/bin/env python3
"""Fail-closed watcher for the two disclosed Seed-42 continuations to epoch 1000.

This additive watcher authorizes exactly two jobs: the already committed
PSBFR-v1 Seed-42 run and the already committed CP-HF-S2-v1 Seed-42 run.  It
does not authorize a fresh run, another seed, test/lockbox access, or a stable
over-baseline claim.  Each continuation is pinned to the same full GPU UUID
and PCI bus used for its first 500 epochs.

The adapter owns model-aware start/resume/final validation.  This watcher owns
authorization, immutable Stage-1 ledger binding, exact-process exclusion,
the old/new singleton locks, per-task locks, shared full-UUID GPU locks,
idle confirmation, safe logs, launch, containment, and completion ledgers.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

# Direct execution sets ``sys.path[0]`` to the ``tools`` directory rather than
# the repository root.  Insert only the fixed reviewed project root so the
# same frozen module is imported for direct, ``-m``, and test entry points.
_PROJECT_ROOT_BOOTSTRAP = Path("/home/ly/EviSIRST_main")
_PROJECT_ROOT_BOOTSTRAP_TEXT = os.fspath(_PROJECT_ROOT_BOOTSTRAP)
while _PROJECT_ROOT_BOOTSTRAP_TEXT in sys.path:
    sys.path.remove(_PROJECT_ROOT_BOOTSTRAP_TEXT)
sys.path.insert(0, _PROJECT_ROOT_BOOTSTRAP_TEXT)

from tools import run_irstd_ab_model_screen_when_idle as stage1


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
EXPECTED_PYTHON = Path("/usr/bin/python3.12")
ADAPTER_SOURCE = PROJECT_ROOT / "run_irstd_seed42_to1000_v1.py"
WATCHER_SOURCE = PROJECT_ROOT / "tools/run_irstd_seed42_to1000_when_idle.py"
STAGE1_WATCHER_SOURCE = PROJECT_ROOT / "tools/run_irstd_ab_model_screen_when_idle.py"
TIME_BIN = Path("/usr/bin/time")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
PROC_ROOT = Path("/proc")

WATCH_ROOT = PROJECT_ROOT / "runs/irstd_model_design/seed42_to1000_watcher_v1"
WATCHER_LOCK = WATCH_ROOT / "watcher.lock"
TASK_LOCK_ROOT = WATCH_ROOT / "task_locks"
LOG_ROOT = WATCH_ROOT / "logs"
COMPLETION_ROOT = WATCH_ROOT / "epoch1000_commits"
GPU_LOCK_ROOT = PROJECT_ROOT / "runs/.gpu_locks"
OLD_WATCHER_LOCK = (
    PROJECT_ROOT
    / "runs/irstd_model_design/ab_legacy_screen_watcher_v1/watcher.lock"
)
STAGE1_LEDGER_ROOT = (
    PROJECT_ROOT
    / "runs/irstd_model_design/ab_legacy_screen_watcher_v1/epoch500_commits"
)

AUTHORIZATION_PATH = (
    PROJECT_ROOT / "experiments/IRSTD_SEED42_TO1000_V1_AUTHORIZATION.json"
)
# Frozen only after the adapter, watcher, and CPU tests pass independent audit.
FORMAL_AUTHORIZATION_SHA256: str | None = "497f5409e7481eb7df76c8b5baa9c02de0daffead6f9a734d785533d84ae5a4a"
AUTHORIZATION_SCHEMA = "evisirst_irstd_seed42_to1000_authorization/v1"
MANIFEST_SCHEMA = "evisirst_irstd_seed42_to1000_task_manifest/v1"
COMPLETION_SCHEMA = "evisirst_irstd_seed42_to1000_completion/v1"

FINAL_EPOCH = 1000
START_EPOCH = 500
POLL_SECONDS = 10
GPU_CONFIRM_SECONDS = 10
GPU_MEMORY_LIMIT_MIB = 1024
TERM_TIMEOUT_SECONDS = 60
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_ADAPTER_PROBE_PREFIX = "EVISIRST_SEED42_TO1000_WATCHER_PROBE:"

# The old Stage-1 authorization/source set remains a mandatory dependency.
FROZEN_STAGE1_WATCHER_SOURCE_SHA256 = (
    "f840b8c430348bd416915337107cd8ef7880df8f68e61313e901783dc9aa53ee"
)
# The adapter hash normalizes only its authorization literal, exactly like the
# watcher hash.  The two test files have no authorization literal and are
# bound byte-for-byte as executable evidence contracts.
FROZEN_ADAPTER_SOURCE_SHA256: str | None = (
    "edfc366d25ba9986f3ca991cfa882ce2afa350a6e60662563b8e1b7d299f2f25"
)
FROZEN_ADDITIVE_SOURCE_SHA256 = {
    "tests/test_irstd_seed42_to1000_v1.py": (
        "81aaad33883a0dac071f6b682abd5941ae22db772e736f510c41b73736ce3542"
    ),
    "tests/test_irstd_seed42_to1000_watcher.py": (
        "18b685aff1a5b580e48c3d79e7f1547755d2d0a2311f2c889c6859936b233d23"
    ),
}

POST_INTERIM_DISCLOSURE = {
    "epoch500_results_disclosed_before_continuation": True,
    "user_requested_both_routes_to1000": True,
    "post_interim_protocol_amendment": True,
    "exploratory_seed42_full_schedule": True,
    "old_stage1_gate_not_used_for_claim": True,
    "stable_over_baseline_claim_eligible": False,
}


class Seed42ContinuationWatcherError(RuntimeError):
    """The fixed Seed-42 continuation queue cannot be supervised safely."""


class _TerminationRequested(BaseException):
    """Transfer SIGINT/SIGTERM control to the supervision cleanup path."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"termination signal {signum}")
        self.signum = signum


class _TerminationGate:
    """Record signals without asynchronously interrupting launch/cleanup."""

    def __init__(self) -> None:
        self.signum: int | None = None

    def request(self, signum: int, _frame: object) -> None:
        if self.signum is None:
            self.signum = signum

    def raise_if_requested(self) -> None:
        if self.signum is not None:
            raise _TerminationRequested(self.signum)


def _no_termination_requested() -> None:
    return None


@dataclass(frozen=True)
class Task:
    name: str
    route: str
    kind: str
    variant: str
    run_dir: Path
    stage1_ledger: Path
    stage1_ledger_sha256: str
    epoch500_latest_sha256: str
    epoch500_history_sha256: str
    gpu_uuid: str
    gpu_bus_id: str


@dataclass
class RunningJob:
    task: Task
    process: subprocess.Popen[bytes]
    gpu: stage1.GPU
    gpu_lock_fd: int
    task_lock_fd: int
    log_handle: object
    log_path: Path


def _design_run_dir(variant: str) -> Path:
    return (
        PROJECT_ROOT
        / "runs/irstd_model_design"
        / variant
        / "legacy_screen/formal/IRSTD-1K/binary/run_seed_42"
    )


TASKS = (
    Task(
        name="psbfr_s42",
        route="psbfr",
        kind="psbfr",
        variant="psbfr_v1",
        run_dir=_design_run_dir("psbfr_v1"),
        stage1_ledger=STAGE1_LEDGER_ROOT / "psbfr_s42.json",
        stage1_ledger_sha256=(
            "17b9d47591897b8be18b52d0dd6f5ca336f1a50736da3a3b8935098a9f4919a7"
        ),
        epoch500_latest_sha256=(
            "f53792d8fefbcc5fcc41dd4df71047465a6211e4dfaa87e2f063028e106098a4"
        ),
        epoch500_history_sha256=(
            "8bda82ecd091b4c82736475221f9a06c2f4b71cb96a7650b66dde3f319068cab"
        ),
        gpu_uuid="GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
        gpu_bus_id="00000000:16:00.0",
    ),
    Task(
        name="cp_hf_s2_s42",
        route="cp_hf_s2",
        kind="cp_hf_s2",
        variant="cp_hf_s2_v1",
        run_dir=_design_run_dir("cp_hf_s2_v1"),
        stage1_ledger=STAGE1_LEDGER_ROOT / "cp_hf_s2_s42.json",
        stage1_ledger_sha256=(
            "5d0253d906bbab05d095106a7f1e59906386a33d3b8c3c532e065e72cf3e69f3"
        ),
        epoch500_latest_sha256=(
            "6708e425eaec1fa05bb1fa850339d507f5097f509114be10845c38f717e325ed"
        ),
        epoch500_history_sha256=(
            "05b1015bac4abf54cef760bec765c70d71b282810d5c2b6a6e5e4dce43caabcd"
        ),
        gpu_uuid="GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
        gpu_bus_id="00000000:27:00.0",
    ),
)
ROUTES = tuple(task.route for task in TASKS)


def _task(route: str) -> Task:
    matches = [task for task in TASKS if task.route == route]
    if len(matches) != 1:
        raise Seed42ContinuationWatcherError("unsupported continuation route")
    return matches[0]


def _old_task(task: Task) -> stage1.Task:
    matches = [item for item in stage1.TASKS if item.name == task.name]
    if len(matches) != 1:
        raise Seed42ContinuationWatcherError("Stage-1 task mapping differs")
    return matches[0]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise Seed42ContinuationWatcherError(
            "value is not strict canonical JSON"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalized_authorization_source_sha256(path: Path) -> str:
    source = path.read_bytes()
    pattern = re.compile(
        rb"FORMAL_AUTHORIZATION_SHA256: str \| None = "
        rb"(?:None|\"[0-9a-f]{64}\")"
    )
    normalized, count = pattern.subn(
        b'FORMAL_AUTHORIZATION_SHA256: str | None = "<AUTHORIZATION_SHA256>"',
        source,
    )
    if count != 1:
        raise Seed42ContinuationWatcherError(
            f"authorization literal normalization differs: {path.name}"
        )
    return hashlib.sha256(normalized).hexdigest()


def _watcher_source_sha256() -> str:
    return _normalized_authorization_source_sha256(WATCHER_SOURCE)


def _adapter_source_sha256() -> str:
    return _normalized_authorization_source_sha256(ADAPTER_SOURCE)


def frozen_source_allowlist() -> dict[str, str]:
    adapter_sha = FROZEN_ADAPTER_SOURCE_SHA256
    if adapter_sha is None or not _SHA256_RE.fullmatch(adapter_sha):
        raise Seed42ContinuationWatcherError("adapter source is not frozen")
    return {
        "inherited_stage1_source_allowlist_sha256": (
            stage1.frozen_source_allowlist_sha256()
        ),
        "stage1_watcher_source_sha256": FROZEN_STAGE1_WATCHER_SOURCE_SHA256,
        "seed42_continuation_adapter_normalized_sha256": adapter_sha,
        **FROZEN_ADDITIVE_SOURCE_SHA256,
    }


def frozen_source_allowlist_sha256() -> str:
    return _canonical_sha256(frozen_source_allowlist())


def assert_frozen_sources() -> None:
    if not stage1.authorization_is_valid():
        raise Seed42ContinuationWatcherError(
            "frozen Stage-1 source/authorization dependency differs"
        )
    if (
        STAGE1_WATCHER_SOURCE.is_symlink()
        or not STAGE1_WATCHER_SOURCE.is_file()
        or _sha256_file(STAGE1_WATCHER_SOURCE)
        != FROZEN_STAGE1_WATCHER_SOURCE_SHA256
    ):
        raise Seed42ContinuationWatcherError("Stage-1 watcher source differs")
    expected = FROZEN_ADAPTER_SOURCE_SHA256
    if (
        expected is None
        or not _SHA256_RE.fullmatch(expected)
        or ADAPTER_SOURCE.is_symlink()
        or not ADAPTER_SOURCE.is_file()
        or _adapter_source_sha256() != expected
    ):
        raise Seed42ContinuationWatcherError("continuation adapter source differs")
    for relative, expected_additive in FROZEN_ADDITIVE_SOURCE_SHA256.items():
        if (
            not _SHA256_RE.fullmatch(expected_additive)
            or _sha256_file(stage1._regular_project_file(relative))
            != expected_additive
        ):
            raise Seed42ContinuationWatcherError(
                f"additive evidence source differs: {relative}"
            )


def frozen_sources_are_valid() -> bool:
    try:
        assert_frozen_sources()
    except (OSError, Seed42ContinuationWatcherError):
        return False
    return True


def _strict_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise Seed42ContinuationWatcherError(f"not a regular JSON file: {path}")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Seed42ContinuationWatcherError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Seed42ContinuationWatcherError(
                    f"non-finite JSON constant: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Seed42ContinuationWatcherError("JSON artifact is malformed") from exc

    def finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise Seed42ContinuationWatcherError("non-finite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def assert_frozen_epoch500_ledger(route: str) -> dict[str, object]:
    task = _task(route)
    if (
        task.stage1_ledger.is_symlink()
        or not task.stage1_ledger.is_file()
        or _sha256_file(task.stage1_ledger) != task.stage1_ledger_sha256
    ):
        raise Seed42ContinuationWatcherError(
            f"frozen epoch-500 ledger differs: {route}"
        )
    payload = _strict_json(task.stage1_ledger)
    if not isinstance(payload, dict):
        raise Seed42ContinuationWatcherError("epoch-500 ledger is malformed")
    required = {
        "schema": stage1.COMPLETION_SCHEMA,
        "task": task.name,
        "kind": task.kind,
        "variant": task.variant,
        "run_seed": 42,
        "configured_total_epochs": FINAL_EPOCH,
        "stopped_after_epoch": START_EPOCH,
        "latest_sha256": task.epoch500_latest_sha256,
        "history_sha256": task.epoch500_history_sha256,
        "candidate_cleanup_validated": True,
        "lockbox_accessed": False,
        "public_test_allowed": False,
        "test_split_accessed": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise Seed42ContinuationWatcherError(
                f"epoch-500 ledger field differs: {route}.{key}"
            )
    return payload


def assert_epoch500_start(route: str) -> None:
    task = _task(route)
    assert_frozen_sources()
    assert_frozen_epoch500_ledger(route)
    if stage1._probe_task_state(_old_task(task)) != "ready":
        raise Seed42ContinuationWatcherError(
            f"model-aware epoch-500 start validation failed: {route}"
        )
    if (
        _sha256_file(task.run_dir / "last_training_state.pth.tar")
        != task.epoch500_latest_sha256
        or _sha256_file(task.run_dir / "validation_history.json")
        != task.epoch500_history_sha256
    ):
        raise Seed42ContinuationWatcherError(
            f"epoch-500 start snapshot bytes differ: {route}"
        )


_ADAPTER_PROBE_PROGRAM = r'''
import json, sys
import run_irstd_seed42_to1000_v1 as adapter

PREFIX="EVISIRST_SEED42_TO1000_WATCHER_PROBE:"
action=sys.argv[1]
route=sys.argv[2]
if action == "inspect":
    result=adapter.inspect_route_state(route)
elif action == "validate_snapshot":
    result=adapter.validate_start_snapshot(route)
elif action == "create_snapshot":
    path=adapter.create_start_snapshot(route)
    result={"path":str(path),"created_or_identical":True}
elif action == "entry":
    result=adapter.assert_continuation_entry(route)
elif action == "completed":
    result=adapter.assert_completed_state(route)
else:
    raise RuntimeError("unknown fixed adapter probe action")
print(PREFIX+json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False),flush=True)
'''


def _adapter_action(route: str, action: str) -> dict[str, object]:
    _task(route)
    if action not in {
        "inspect",
        "validate_snapshot",
        "create_snapshot",
        "entry",
        "completed",
    }:
        raise Seed42ContinuationWatcherError("unsupported adapter action")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    try:
        completed = subprocess.run(
            [
                os.fspath(PYTHON_BIN),
                "-c",
                _ADAPTER_PROBE_PROGRAM,
                action,
                route,
            ],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Seed42ContinuationWatcherError(
            f"adapter {action} probe failed: {route}"
        ) from exc
    tokens = [
        line[len(_ADAPTER_PROBE_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_ADAPTER_PROBE_PREFIX)
    ]
    if completed.returncode != 0 or len(tokens) != 1:
        detail = completed.stderr.strip()[-1000:]
        raise Seed42ContinuationWatcherError(
            f"adapter {action} validation failed: {route}: {detail}"
        )
    try:
        payload = json.loads(tokens[0])
    except json.JSONDecodeError as exc:
        raise Seed42ContinuationWatcherError(
            "adapter probe returned malformed JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise Seed42ContinuationWatcherError(
            "adapter probe returned a non-object"
        )
    return payload


def _probe_task_state(task: Task) -> str:
    try:
        assert_frozen_sources()
        assert_frozen_epoch500_ledger(task.route)
        payload = _adapter_action(task.route, "inspect")
    except (OSError, Seed42ContinuationWatcherError):
        return "conflict"
    raw = payload.get("state")
    states = {
        "epoch500_ready": "ready",
        "resume_501_999": "resume",
        "finalize_1000": "finalize",
        "complete": "complete_artifacts",
    }
    return states.get(raw, "conflict")


def assert_continuation_entry(route: str) -> dict[str, object]:
    assert_frozen_sources()
    assert_frozen_epoch500_ledger(route)
    result = _adapter_action(route, "entry")
    if result.get("state") not in {
        "epoch500_ready",
        "resume_501_999",
        "finalize_1000",
    }:
        raise Seed42ContinuationWatcherError(
            f"adapter continuation entry state differs: {route}"
        )
    return result


def assert_completed_state(route: str) -> dict[str, object]:
    assert_frozen_sources()
    assert_frozen_epoch500_ledger(route)
    result = _adapter_action(route, "completed")
    if result.get("state") != "complete":
        raise Seed42ContinuationWatcherError(
            f"adapter completed state differs: {route}"
        )
    return result


def create_or_validate_start_snapshot(route: str) -> dict[str, object]:
    state = _probe_task_state(_task(route))
    if state == "ready":
        assert_epoch500_start(route)
        _adapter_action(route, "create_snapshot")
    return _adapter_action(route, "validate_snapshot")


def adapter_argv(route: str) -> tuple[str, ...]:
    _task(route)
    return (
        os.fspath(PYTHON_BIN),
        os.fspath(ADAPTER_SOURCE),
        "--route",
        route,
    )


def timed_argv(argv: Sequence[str]) -> tuple[str, ...]:
    return (os.fspath(TIME_BIN), "-v", *tuple(argv))


def timed_adapter_argv(route: str) -> tuple[str, ...]:
    return timed_argv(adapter_argv(route))


def _read_ppid(pid: int, *, proc_root: Path = PROC_ROOT) -> int | None:
    try:
        lines = (proc_root / str(pid) / "status").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeError):
        return None
    values = [line.split(":", 1)[1].strip() for line in lines if line.startswith("PPid:")]
    if len(values) != 1 or not values[0].isdecimal():
        return None
    return int(values[0])


def _route_process_pair(
    route: str, *, proc_root: Path = PROC_ROOT
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    direct = stage1.exact_argv_processes(
        adapter_argv(route), proc_root=proc_root
    )
    timed = stage1.exact_argv_processes(
        timed_adapter_argv(route), proc_root=proc_root
    )
    return direct, timed


def _authorized_route_process_is_running(
    route: str, *, proc_root: Path = PROC_ROOT
) -> bool:
    direct, timed = _route_process_pair(route, proc_root=proc_root)
    return bool(
        len(direct) == 1
        and len(timed) == 1
        and _read_ppid(direct[0], proc_root=proc_root) == timed[0]
    )


def _assert_inherited_locked_file(raw_fd: str | None, path: Path) -> int:
    if raw_fd is None or not raw_fd.isascii() or not raw_fd.isdecimal():
        raise Seed42ContinuationWatcherError("inherited lock FD is missing")
    descriptor = int(raw_fd)
    if descriptor < 3:
        raise Seed42ContinuationWatcherError("inherited lock FD is unsafe")
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
    except OSError as exc:
        raise Seed42ContinuationWatcherError(
            f"inherited lock FD is unavailable: {path}"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(path_stat.st_mode)
        or not stat.S_ISREG(descriptor_stat.st_mode)
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise Seed42ContinuationWatcherError(
            f"inherited lock FD identity differs: {path}"
        )
    # An independently opened file description must be unable to acquire the
    # lock.  The inherited FD shares the watcher's locked open-file description.
    contender = _try_existing_lock(path)
    if contender is not None:
        os.close(contender)
        raise Seed42ContinuationWatcherError(
            f"inherited file is not currently locked: {path}"
        )
    return descriptor


def assert_launch_capability(route: str) -> None:
    """Prove the current adapter inherited all watcher launch capabilities."""

    task = _task(route)
    required_env = {
        "EVISIRST_SEED42_TO1000_ROUTE": route,
        "EVISIRST_SEED42_TO1000_GPU_UUID": task.gpu_uuid,
        "CUDA_VISIBLE_DEVICES": task.gpu_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    for name, expected in required_env.items():
        if os.environ.get(name) != expected:
            raise Seed42ContinuationWatcherError(
                f"formal launch capability environment differs: {name}"
            )
    if "NVIDIA_VISIBLE_DEVICES" in os.environ:
        raise Seed42ContinuationWatcherError(
            "NVIDIA_VISIBLE_DEVICES must be absent in formal continuation"
        )
    descriptors = {
        _assert_inherited_locked_file(
            os.environ.get("EVISIRST_SEED42_TO1000_TASK_LOCK_FD"),
            TASK_LOCK_ROOT / f"{task.name}.lock",
        ),
        _assert_inherited_locked_file(
            os.environ.get("EVISIRST_SEED42_TO1000_GPU_LOCK_FD"),
            GPU_LOCK_ROOT / f"{task.gpu_uuid}.lock",
        ),
        _assert_inherited_locked_file(
            os.environ.get("EVISIRST_SEED42_TO1000_OLD_WATCHER_LOCK_FD"),
            OLD_WATCHER_LOCK,
        ),
    }
    if len(descriptors) != 3:
        raise Seed42ContinuationWatcherError(
            "formal launch capability FDs are not distinct"
        )


def assert_no_forbidden_processes(route: str) -> None:
    """Reject duplicates and every old direct worker/trainer command.

    When invoked from inside the authorized adapter, the only accepted pair is
    the current adapter with its exact ``/usr/bin/time -v`` parent.
    """

    _task(route)
    for old in stage1.TASKS:
        if stage1.task_processes(old, resume=False) or stage1.task_processes(
            old, resume=True
        ):
            raise Seed42ContinuationWatcherError(
                f"old trainer/worker process is forbidden: {old.name}"
            )
    direct, timed = _route_process_pair(route)
    if not direct and not timed:
        return
    if not _authorized_route_process_is_running(route):
        raise Seed42ContinuationWatcherError(
            f"continuation process multiplicity differs: {route}"
        )
    if os.getpid() != direct[0] or os.getppid() != timed[0]:
        raise Seed42ContinuationWatcherError(
            f"another continuation process already owns route: {route}"
        )
    assert_launch_capability(route)


def _worker_env(
    task: Task,
    gpu: stage1.GPU,
    *,
    gpu_lock_fd: int,
    task_lock_fd: int,
    old_watcher_lock_fd: int,
) -> dict[str, str]:
    env = stage1._scrubbed_gpu_env(gpu)
    env.update(
        {
            "EVISIRST_SEED42_TO1000_ROUTE": task.route,
            "EVISIRST_SEED42_TO1000_GPU_UUID": task.gpu_uuid,
            "EVISIRST_SEED42_TO1000_GPU_LOCK_FD": str(gpu_lock_fd),
            "EVISIRST_SEED42_TO1000_TASK_LOCK_FD": str(task_lock_fd),
            "EVISIRST_SEED42_TO1000_OLD_WATCHER_LOCK_FD": str(
                old_watcher_lock_fd
            ),
        }
    )
    return env


def task_manifest() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for task in TASKS:
        entries.append(
            {
                "name": task.name,
                "route": task.route,
                "kind": task.kind,
                "variant": task.variant,
                "architecture_seed": 42,
                "run_seed": 42,
                "configured_total_epochs": FINAL_EPOCH,
                "required_start_epoch": START_EPOCH,
                "run_dir": task.run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "adapter_argv": list(adapter_argv(task.route)),
                "timed_adapter_argv": list(timed_adapter_argv(task.route)),
                "pinned_gpu_uuid": task.gpu_uuid,
                "pinned_gpu_bus_id": task.gpu_bus_id,
                "stage1_ledger": task.stage1_ledger.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "stage1_ledger_sha256": task.stage1_ledger_sha256,
                "epoch500_latest_sha256": task.epoch500_latest_sha256,
                "epoch500_history_sha256": task.epoch500_history_sha256,
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "SEED42_EXPLORATORY_CONTINUATION_ONLY",
        "tasks": entries,
        "task_count": 2,
        "fresh_training_allowed": False,
        "other_run_seeds_allowed": False,
        "fixed_architecture_seed": 42,
        "fixed_run_seed": 42,
        "candidate_routes_are_separate": True,
        "shared_gpu_lock_root": GPU_LOCK_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "old_stage1_singleton_lock_held_during_supervision": True,
        "lockbox_accessed": False,
        "public_test_allowed": False,
        "test_split_accessed": False,
        **POST_INTERIM_DISCLOSURE,
    }


def authorization_is_valid() -> bool:
    expected = FORMAL_AUTHORIZATION_SHA256
    if expected is None:
        return False
    if not _SHA256_RE.fullmatch(expected):
        raise Seed42ContinuationWatcherError(
            "authorization SHA-256 is malformed"
        )
    try:
        assert_frozen_sources()
        for task in TASKS:
            assert_frozen_epoch500_ledger(task.route)
        if (
            AUTHORIZATION_PATH.is_symlink()
            or not AUTHORIZATION_PATH.is_file()
            or _sha256_file(AUTHORIZATION_PATH) != expected
        ):
            return False
        payload = _strict_json(AUTHORIZATION_PATH)
    except (OSError, Seed42ContinuationWatcherError):
        return False
    required_keys = {
        "schema",
        "status",
        "task_manifest_sha256",
        "watcher_source_sha256",
        "adapter_source_sha256",
        "source_allowlist_sha256",
        "lockbox_accessed",
        "public_test_allowed",
        "test_split_accessed",
        *POST_INTERIM_DISCLOSURE,
    }
    return bool(
        isinstance(payload, dict)
        and set(payload) == required_keys
        and payload.get("schema") == AUTHORIZATION_SCHEMA
        and payload.get("status") == "AUTHORIZED"
        and payload.get("task_manifest_sha256")
        == _canonical_sha256(task_manifest())
        and payload.get("watcher_source_sha256") == _watcher_source_sha256()
        and payload.get("adapter_source_sha256") == _adapter_source_sha256()
        and payload.get("source_allowlist_sha256")
        == frozen_source_allowlist_sha256()
        and payload.get("lockbox_accessed") is False
        and payload.get("public_test_allowed") is False
        and payload.get("test_split_accessed") is False
        and all(payload.get(key) is value for key, value in POST_INTERIM_DISCLOSURE.items())
    )


def _completion_path(task: Task) -> Path:
    return COMPLETION_ROOT / f"{task.name}.json"


def _regular_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Seed42ContinuationWatcherError(
            f"completion artifact is not regular: {path}"
        )
    return _sha256_file(path)


def _completion_payload(task: Task) -> dict[str, object]:
    validation = assert_completed_state(task.route)
    snapshot = _adapter_action(task.route, "validate_snapshot")
    artifacts = {
        name: _regular_hash(task.run_dir / name)
        for name in (
            "last_training_state.pth.tar",
            "validation_history.json",
            "summary.json",
            "EviSIRST_best_mIoU.pth.tar",
            "EviSIRST_best_Pd.pth.tar",
        )
    }
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "task": task.name,
        "route": task.route,
        "kind": task.kind,
        "variant": task.variant,
        "architecture_seed": 42,
        "run_seed": 42,
        "completed_epoch": FINAL_EPOCH,
        "pinned_gpu_uuid": task.gpu_uuid,
        "pinned_gpu_bus_id": task.gpu_bus_id,
        "stage1_ledger_sha256": task.stage1_ledger_sha256,
        "start_snapshot_manifest_sha256": hashlib.sha256(
            _canonical_bytes(snapshot) + b"\n"
        ).hexdigest(),
        "artifact_sha256": artifacts,
        "adapter_validation": validation,
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "source_allowlist_sha256": frozen_source_allowlist_sha256(),
        "watcher_source_sha256": _watcher_source_sha256(),
        "adapter_source_sha256": _adapter_source_sha256(),
        "authorization_sha256": FORMAL_AUTHORIZATION_SHA256,
        "lockbox_accessed": False,
        "public_test_allowed": False,
        "test_split_accessed": False,
        **POST_INTERIM_DISCLOSURE,
    }


def _completion_is_valid(task: Task) -> bool:
    path = _completion_path(task)
    if not path.exists() or path.is_symlink():
        return False
    try:
        payload = _strict_json(path)
        expected = _completion_payload(task)
    except (OSError, Seed42ContinuationWatcherError):
        return False
    return isinstance(payload, dict) and payload == expected


def task_state(task: Task) -> str:
    direct, timed = _route_process_pair(task.route)
    if direct or timed:
        return (
            "running"
            if _authorized_route_process_is_running(task.route)
            else "conflict"
        )
    probed = _probe_task_state(task)
    marker = _completion_path(task)
    if probed == "complete_artifacts":
        if marker.exists() or marker.is_symlink():
            return "complete" if _completion_is_valid(task) else "conflict"
        return "complete_unledgered"
    if marker.exists() or marker.is_symlink():
        return "conflict"
    return probed


def _assert_safe_directory_chain(path: Path) -> None:
    runs = PROJECT_ROOT / "runs"
    if runs.is_symlink() or (runs.exists() and not runs.is_dir()):
        raise Seed42ContinuationWatcherError("repository runs root is unsafe")
    try:
        relative = path.relative_to(runs)
    except ValueError as exc:
        raise Seed42ContinuationWatcherError("watcher path escaped runs") from exc
    current = runs
    for component in relative.parts:
        current /= component
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise Seed42ContinuationWatcherError(f"unsafe watcher path: {current}")


def _try_existing_lock(path: Path) -> int | None:
    if path.is_symlink() or not path.is_file():
        raise Seed42ContinuationWatcherError(
            f"required preexisting lock is unsafe: {path}"
        )
    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def claim_pinned_gpu(
    task: Task,
    *,
    sleep: Callable[[float], None] = time.sleep,
    check_termination: Callable[[], None] = _no_termination_requested,
) -> tuple[stage1.GPU, int] | None:
    def observed(sample: Mapping[str, stage1.GPU]) -> stage1.GPU | None:
        gpu = sample.get(task.gpu_uuid)
        if (
            gpu is None
            or gpu.bus_id.lower() != task.gpu_bus_id.lower()
            or gpu.memory_used_mib > GPU_MEMORY_LIMIT_MIB
        ):
            return None
        return gpu

    check_termination()
    first = observed(stage1.sample_idle_gpus())
    check_termination()
    sleep(GPU_CONFIRM_SECONDS)
    check_termination()
    second = observed(stage1.sample_idle_gpus())
    check_termination()
    if first is None or second is None or (
        first.index,
        first.bus_id,
    ) != (second.index, second.bus_id):
        return None
    descriptor = stage1._try_lock(GPU_LOCK_ROOT / f"{task.gpu_uuid}.lock")
    if descriptor is None:
        return None
    try:
        check_termination()
        third = observed(stage1.sample_idle_gpus())
        check_termination()
        sleep(GPU_CONFIRM_SECONDS)
        check_termination()
        fourth = observed(stage1.sample_idle_gpus())
        check_termination()
        if third is None or fourth is None or (
            third.index,
            third.bus_id,
        ) != (fourth.index, fourth.bus_id):
            os.close(descriptor)
            return None
        return fourth, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def launch_task(
    task: Task,
    *,
    expected_state: str,
    gpu: stage1.GPU,
    gpu_lock_fd: int,
    old_watcher_lock_fd: int,
    check_termination: Callable[[], None] = _no_termination_requested,
) -> RunningJob | None:
    process: subprocess.Popen[bytes] | None = None
    log_handle: object | None = None
    try:
        check_termination()
        if not authorization_is_valid():
            os.close(gpu_lock_fd)
            return None
        check_termination()
        task_fd = stage1._try_lock(TASK_LOCK_ROOT / f"{task.name}.lock")
    except BaseException:
        os.close(gpu_lock_fd)
        raise
    if task_fd is None:
        os.close(gpu_lock_fd)
        return None
    try:
        check_termination()
        if task_state(task) != expected_state:
            os.close(task_fd)
            os.close(gpu_lock_fd)
            return None
        check_termination()
        assert_no_forbidden_processes(task.route)
        check_termination()
        create_or_validate_start_snapshot(task.route)
        check_termination()
        assert_continuation_entry(task.route)
        check_termination()
        final_idle = stage1.sample_idle_gpus()
        check_termination()
        observed = final_idle.get(task.gpu_uuid)
        if observed is None or (
            observed.bus_id.lower() != task.gpu_bus_id.lower()
            or observed.index != gpu.index
        ):
            os.close(task_fd)
            os.close(gpu_lock_fd)
            return None
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"{task.name}.log"
        if log_path.is_symlink() or (
            log_path.exists() and not log_path.is_file()
        ):
            raise Seed42ContinuationWatcherError("task log path is unsafe")
        log_handle = stage1._open_log(log_path)
        try:
            process = subprocess.Popen(
                list(timed_adapter_argv(task.route)),
                cwd=PROJECT_ROOT,
                env=_worker_env(
                    task,
                    gpu,
                    gpu_lock_fd=gpu_lock_fd,
                    task_lock_fd=task_fd,
                    old_watcher_lock_fd=old_watcher_lock_fd,
                ),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                pass_fds=(gpu_lock_fd, task_fd, old_watcher_lock_fd),
                start_new_session=True,
            )
        except BaseException:
            if process is not None:
                provisional = RunningJob(
                    task=task,
                    process=process,
                    gpu=gpu,
                    gpu_lock_fd=gpu_lock_fd,
                    task_lock_fd=task_fd,
                    log_handle=log_handle,
                    log_path=log_path,
                )
                try:
                    _abort_job(provisional)
                finally:
                    task_fd = provisional.task_lock_fd
                    gpu_lock_fd = provisional.gpu_lock_fd
            else:
                log_handle.close()
            raise
    except BaseException:
        try:
            if task_fd >= 0:
                os.close(task_fd)
        finally:
            if gpu_lock_fd >= 0:
                os.close(gpu_lock_fd)
        raise
    if process is None or log_handle is None:
        raise Seed42ContinuationWatcherError("worker launch returned no process")
    return RunningJob(
        task=task,
        process=process,
        gpu=gpu,
        gpu_lock_fd=gpu_lock_fd,
        task_lock_fd=task_fd,
        log_handle=log_handle,
        log_path=log_path,
    )


def _release_job(job: RunningJob) -> None:
    first_error: BaseException | None = None
    try:
        job.log_handle.close()
    except BaseException as exc:
        first_error = exc
    for name in ("task_lock_fd", "gpu_lock_fd"):
        descriptor = getattr(job, name)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                setattr(job, name, -1)
    if first_error is not None:
        raise first_error


def _abort_job(job: RunningJob) -> None:
    _log(
        f"containment requested task={job.task.name} pid={job.process.pid}; "
        "waiting for process-group exit before releasing inherited locks"
    )
    try:
        if job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(job.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
    finally:
        _release_job(job)
        _log(
            f"containment complete task={job.task.name} pid={job.process.pid}; "
            "a later watcher pass may validate and recover fixed transaction "
            "temporaries under the adapter run lock"
        )


def _atomic_write_completion(task: Task) -> Path:
    COMPLETION_ROOT.mkdir(parents=True, exist_ok=True)
    path = _completion_path(task)
    encoded = _canonical_bytes(_completion_payload(task)) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Seed42ContinuationWatcherError(
                "completion ledger no-clobber conflict"
            )
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=COMPLETION_ROOT
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if (
                exc.errno == errno.EEXIST
                and path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == encoded
            ):
                return path
            raise Seed42ContinuationWatcherError(
                "completion ledger no-clobber conflict"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _supervise_unprotected(
    *,
    running: dict[str, RunningJob],
    old_watcher_lock_fd: int,
    sleep: Callable[[float], None],
    check_termination: Callable[[], None],
) -> int:
    check_termination()
    if not authorization_is_valid():
        raise Seed42ContinuationWatcherError(
            "formal continuation authorization is unavailable"
        )
    while True:
        check_termination()
        for name, job in tuple(running.items()):
            check_termination()
            returncode = job.process.poll()
            if returncode is None:
                continue
            if returncode != 0:
                _release_job(job)
                del running[name]
                raise Seed42ContinuationWatcherError(
                    f"continuation worker failed: {name} rc={returncode}"
                )
            check_termination()
            assert_completed_state(job.task.route)
            check_termination()
            _atomic_write_completion(job.task)
            check_termination()
            if not _completion_is_valid(job.task):
                raise Seed42ContinuationWatcherError(
                    f"completion ledger verification failed: {name}"
                )
            _release_job(job)
            del running[name]

        states: dict[str, str] = {}
        for task in TASKS:
            check_termination()
            states[task.name] = (
                "running" if task.name in running else task_state(task)
            )
            check_termination()
        for name, state_value in states.items():
            _log(f"task={name} state={state_value}")
        if any(value == "conflict" for value in states.values()):
            raise Seed42ContinuationWatcherError(
                "Seed-42 continuation queue contains a conflict"
            )
        for task in TASKS:
            if states[task.name] == "complete_unledgered":
                check_termination()
                _atomic_write_completion(task)
                check_termination()
                if not _completion_is_valid(task):
                    raise Seed42ContinuationWatcherError(
                        f"recovered completion ledger differs: {task.name}"
                    )
                states[task.name] = "complete"
        if all(value == "complete" for value in states.values()):
            _log("both separate Seed-42 continuations completed epoch 1000")
            return 0

        for task in TASKS:
            check_termination()
            expected_state = states[task.name]
            if expected_state not in {"ready", "resume", "finalize"}:
                continue
            claimed = claim_pinned_gpu(
                task,
                sleep=sleep,
                check_termination=check_termination,
            )
            check_termination()
            if claimed is None:
                continue
            gpu, gpu_fd = claimed
            job = launch_task(
                task,
                expected_state=expected_state,
                gpu=gpu,
                gpu_lock_fd=gpu_fd,
                old_watcher_lock_fd=old_watcher_lock_fd,
                check_termination=check_termination,
            )
            if job is not None:
                running[task.name] = job
                check_termination()
                _log(
                    f"launched {task.name} pid={job.process.pid} "
                    f"gpu={gpu.uuid} index={gpu.index} bus={gpu.bus_id}"
                )
            else:
                check_termination()
        sleep(POLL_SECONDS)
        check_termination()


def supervise(
    *,
    old_watcher_lock_fd: int,
    sleep: Callable[[float], None] = time.sleep,
    check_termination: Callable[[], None] = _no_termination_requested,
) -> int:
    running: dict[str, RunningJob] = {}
    try:
        return _supervise_unprotected(
            running=running,
            old_watcher_lock_fd=old_watcher_lock_fd,
            sleep=sleep,
            check_termination=check_termination,
        )
    finally:
        for job in tuple(running.values()):
            try:
                _abort_job(job)
            except BaseException as exc:
                _log(f"worker containment failed task={job.task.name}: {exc!r}")
        running.clear()


def _supervise_with_termination_handlers(*, old_watcher_lock_fd: int) -> int:
    previous_handlers: list[tuple[int, object]] = []
    gate = _TerminationGate()

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers.append(
                (signum, signal.signal(signum, gate.request))
            )
        try:
            return supervise(
                old_watcher_lock_fd=old_watcher_lock_fd,
                check_termination=gate.raise_if_requested,
            )
        except _TerminationRequested as exc:
            _log(
                f"watcher termination signal={exc.signum}; "
                "all supervised process groups contained before lock release"
            )
            return 128 + exc.signum
    finally:
        for signum, previous in reversed(previous_handlers):
            signal.signal(signum, previous)


def _lock_available_readonly(path: Path) -> bool:
    descriptor = _try_existing_lock(path)
    if descriptor is None:
        return False
    os.close(descriptor)
    return True


def readonly_preflight() -> dict[str, object]:
    python_target = PYTHON_BIN.resolve(strict=True)
    if (
        python_target != EXPECTED_PYTHON
        or not python_target.is_file()
        or not os.access(python_target, os.X_OK)
    ):
        raise Seed42ContinuationWatcherError("fixed Python target differs")
    for required in (
        ADAPTER_SOURCE,
        WATCHER_SOURCE,
        STAGE1_WATCHER_SOURCE,
        TIME_BIN,
        NVIDIA_SMI,
    ):
        if required.is_symlink() or not required.is_file():
            raise Seed42ContinuationWatcherError(
                f"fixed dependency differs: {required}"
            )
    assert_frozen_sources()
    for task in TASKS:
        assert_frozen_epoch500_ledger(task.route)
    for path in (
        WATCH_ROOT,
        TASK_LOCK_ROOT,
        LOG_ROOT,
        COMPLETION_ROOT,
        GPU_LOCK_ROOT,
    ):
        _assert_safe_directory_chain(path)
    states = {task.name: task_state(task) for task in TASKS}
    idle = stage1.sample_idle_gpus()
    pinned = {
        task.name: bool(
            task.gpu_uuid in idle
            and idle[task.gpu_uuid].bus_id.lower() == task.gpu_bus_id.lower()
        )
        for task in TASKS
    }
    return {
        "schema": "evisirst_irstd_seed42_to1000_readonly_preflight/v1",
        "formal_launch_authorized": authorization_is_valid(),
        "authorization_sha256_frozen": FORMAL_AUTHORIZATION_SHA256 is not None,
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "watcher_source_sha256": _watcher_source_sha256(),
        "adapter_source_sha256": _adapter_source_sha256(),
        "source_allowlist_sha256": frozen_source_allowlist_sha256(),
        "task_states": states,
        "pinned_gpus_idle": pinned,
        "old_stage1_watcher_lock_available": _lock_available_readonly(
            OLD_WATCHER_LOCK
        ),
        "idle_gpu_uuids": sorted(idle),
        "fresh_training_allowed": False,
        "other_run_seeds_allowed": False,
        "lockbox_accessed": False,
        "public_test_supported": False,
        "test_split_accessed": False,
        "writes_performed": False,
        "workers_launched": False,
        **POST_INTERIM_DISCLOSURE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--manifest", action="store_true")
    modes.add_argument("--formal", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.preflight:
        print(json.dumps(readonly_preflight(), sort_keys=True, indent=2))
        return 0
    if args.manifest:
        print(json.dumps(task_manifest(), sort_keys=True, indent=2))
        return 0
    if not authorization_is_valid():
        raise Seed42ContinuationWatcherError(
            "formal mode disabled: frozen authorization is unavailable"
        )
    for path in (
        WATCH_ROOT,
        TASK_LOCK_ROOT,
        LOG_ROOT,
        COMPLETION_ROOT,
        GPU_LOCK_ROOT,
    ):
        _assert_safe_directory_chain(path)
    singleton = stage1._try_lock(WATCHER_LOCK)
    if singleton is None:
        _log("another Seed-42 continuation watcher owns the singleton lock")
        return 0
    old_singleton: int | None = None
    try:
        old_singleton = _try_existing_lock(OLD_WATCHER_LOCK)
        if old_singleton is None:
            raise Seed42ContinuationWatcherError(
                "old Stage-1 watcher singleton is still owned"
            )
        return _supervise_with_termination_handlers(
            old_watcher_lock_fd=old_singleton
        )
    finally:
        if old_singleton is not None:
            os.close(old_singleton)
        os.close(singleton)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FROZEN_ADAPTER_SOURCE_SHA256",
    "FORMAL_AUTHORIZATION_SHA256",
    "POST_INTERIM_DISCLOSURE",
    "ROUTES",
    "Seed42ContinuationWatcherError",
    "TASKS",
    "Task",
    "adapter_argv",
    "assert_completed_state",
    "assert_continuation_entry",
    "assert_epoch500_start",
    "assert_frozen_epoch500_ledger",
    "assert_frozen_sources",
    "assert_launch_capability",
    "assert_no_forbidden_processes",
    "authorization_is_valid",
    "claim_pinned_gpu",
    "create_or_validate_start_snapshot",
    "frozen_source_allowlist",
    "frozen_source_allowlist_sha256",
    "main",
    "readonly_preflight",
    "supervise",
    "task_manifest",
    "task_state",
    "timed_adapter_argv",
]
