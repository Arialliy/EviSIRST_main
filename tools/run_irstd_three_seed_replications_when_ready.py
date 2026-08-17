#!/usr/bin/env python3
"""Run the fixed IRSTD-1K three-runtime-seed validation queue when GPUs are idle.

This is intentionally a no-argument, standard-library-only supervisor.  It
never imports torch in the long-lived watcher: completed artifacts are checked
in the project's fixed virtual environment by the strict product validators.
GPU ownership is based on full NVIDIA UUIDs, four idle observations, and an
advisory lock inherited by ``/usr/bin/time`` and the worker process.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
DATASET_ROOT = Path("/home/ly/SCTransNet_main/datasets")
SPLIT_ROOT = PROJECT_ROOT / "splits/v2"
BASELINE_OUTPUT_ROOT = PROJECT_ROOT / "runs/validation_selected"
WATCH_ROOT = (
    PROJECT_ROOT / "runs/irstd_performance/complete_target_v1/"
    "three_runtime_seed_validation/watcher"
)
WATCHER_LOCK = WATCH_ROOT / "watcher.lock"
GPU_LOCK_ROOT = PROJECT_ROOT / "runs/.gpu_locks"
TASK_LOCK_ROOT = WATCH_ROOT / "task_locks"
LOG_ROOT = WATCH_ROOT / "logs"

BASELINE_TRAINER = PROJECT_ROOT / "train_validation_selected.py"
VARIANT_TRAINER = PROJECT_ROOT / "train_irstd_complete_target_replication_v1.py"
DIAGNOSTIC_RUNNER = (
    PROJECT_ROOT / "run_irstd_paired_baseline_confirmatory_diagnostic.py"
)
THREE_SEED_GATE = PROJECT_ROOT / "run_irstd_complete_target_three_seed_gate.py"
SINGLE_SEED_GATE = PROJECT_ROOT / "run_irstd_complete_target_promotion_gate.py"
TIME_BIN = Path("/usr/bin/time")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
EXPECTED_PYTHON = Path("/usr/bin/python3.12")
PROC_ROOT = Path("/proc")

RUN_SEEDS = (104728269, 262620274)
POLL_SECONDS = 10
GPU_CONFIRM_SECONDS = 10
GPU_MEMORY_LIMIT_MIB = 1024
INITIAL_RETRY_SECONDS = 30
MAX_RETRY_SECONDS = 300
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class ReplicationWatcherError(RuntimeError):
    """The fixed queue cannot be supervised safely."""


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    run_seed: int
    script: Path
    run_dir: Path


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    bus_id: str
    memory_used_mib: int


@dataclass
class RunningJob:
    work_name: str
    process: subprocess.Popen[bytes]
    gpu: GPU
    gpu_lock_fd: int
    task_lock_fd: int
    log_handle: object


def _baseline_dir(seed: int) -> Path:
    return (
        BASELINE_OUTPUT_ROOT / "formal/IRSTD-1K/binary" / f"run_seed_{seed}"
    )


def _variant_dir(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "runs/irstd_performance/complete_target_v1/"
        "three_runtime_seed_validation/formal/IRSTD-1K/binary"
        / f"run_seed_{seed}"
    )


TASKS = (
    Task("b104", "baseline", 104728269, BASELINE_TRAINER, _baseline_dir(104728269)),
    Task("v104", "variant", 104728269, VARIANT_TRAINER, _variant_dir(104728269)),
    Task("b262", "baseline", 262620274, BASELINE_TRAINER, _baseline_dir(262620274)),
    Task("v262", "variant", 262620274, VARIANT_TRAINER, _variant_dir(262620274)),
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def training_argv(task: Task, *, resume: bool) -> tuple[str, ...]:
    if task.kind == "baseline":
        argv = (
            os.fspath(PYTHON_BIN), os.fspath(BASELINE_TRAINER),
            "--dataset", "IRSTD-1K",
            "--dataset-root", os.fspath(DATASET_ROOT),
            "--split-root", os.fspath(SPLIT_ROOT),
            "--output-root", os.fspath(BASELINE_OUTPUT_ROOT),
            "--target-mode", "binary",
            "--architecture-seed", "42",
            "--run-seed", str(task.run_seed),
            "--device", "cuda:0",
            "--epochs", "1000",
            "--batch-size", "16",
            "--workers", "0",
            "--base-lr", "0.001",
            "--min-lr", "0.00001",
            "--warmup-epochs", "10",
            "--val-interval", "1",
            "--allow-sample-level-fallback",
        )
    elif task.kind == "variant":
        argv = (
            os.fspath(PYTHON_BIN), os.fspath(VARIANT_TRAINER),
            "--dataset-root", os.fspath(DATASET_ROOT),
            "--split-root", os.fspath(SPLIT_ROOT),
            "--dataset", "IRSTD-1K",
            "--target-mode", "binary",
            "--architecture-seed", "42",
            "--run-seed", str(task.run_seed),
            "--device", "cuda:0",
            "--epochs", "1000",
            "--warmup-epochs", "10",
            "--allow-sample-level-fallback",
        )
    else:
        raise ReplicationWatcherError(f"unknown task kind: {task.kind}")
    return argv + (("--resume",) if resume else ())


def diagnostic_argv(seed: int) -> tuple[str, ...]:
    if seed not in RUN_SEEDS:
        raise ReplicationWatcherError("diagnostic seed is not preregistered")
    return (
        os.fspath(PYTHON_BIN), os.fspath(DIAGNOSTIC_RUNNER),
        "--run-seed", str(seed),
        "--dataset-root", os.fspath(DATASET_ROOT),
        "--device", "cuda:0", "--workers", "0",
    )


def gate_argv() -> tuple[str, ...]:
    return os.fspath(PYTHON_BIN), os.fspath(THREE_SEED_GATE)


def timed_argv(argv: Sequence[str]) -> tuple[str, ...]:
    return os.fspath(TIME_BIN), "-v", *tuple(argv)


def _probe(code: str, *arguments: object) -> bool:
    completed = subprocess.run(
        [os.fspath(PYTHON_BIN), "-c", code, *(str(value) for value in arguments)],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False, timeout=300,
    )
    return completed.returncode == 0


_AUTHORIZATION_PROBE = (
    "import run_irstd_complete_target_promotion_gate as m;"
    "p=m.validate_existing_result();"
    "assert p['decision']['result']=='PASS';"
    "assert p['decision']['three_runtime_seed_validation_expansion_allowed'] is True"
)
_VARIANT_COMPLETE_PROBE = (
    "import sys; import train_irstd_complete_target_replication_v1 as m;"
    "m.validate_existing_completed_run(int(sys.argv[1]))"
)
_BASELINE_COMPLETE_PROBE = r"""
import sys
import run_irstd_paired_baseline_confirmatory_diagnostic as d
import run_irstd_complete_target_promotion_gate as g
s=int(sys.argv[1])
with d._bound_diagnostic_core(s) as p:
  with d._bound_baseline_gate_validator(s,p):
    g._validate_canonical_split_files()
    summary,sm=g._load_json(p['SUMMARY_RELATIVE_PATH'],label='baseline summary')
    checkpoint,cm=g._load_checkpoint(p['CHECKPOINT_RELATIVE_PATH'],label='baseline checkpoint')
    selection,_=g._validate_summary_common(summary,variant=False)
    g._validate_checkpoint(checkpoint,cm,summary=summary,selection=selection,variant=False)
"""
_DIAGNOSTIC_COMPLETE_PROBE = (
    "import sys; import run_irstd_paired_baseline_confirmatory_diagnostic as m;"
    "m.validate_existing_result(int(sys.argv[1]))"
)
_GATE_COMPLETE_PROBE = (
    "import run_irstd_complete_target_three_seed_gate as m;"
    "p=m.validate_existing_result(); assert p['decision']['result'] in ('PASS','FAIL')"
)


def authorization_is_valid() -> bool:
    return _probe(_AUTHORIZATION_PROBE)


def task_is_complete(task: Task) -> bool:
    return _probe(
        _BASELINE_COMPLETE_PROBE if task.kind == "baseline" else _VARIANT_COMPLETE_PROBE,
        task.run_seed,
    )


def diagnostic_is_complete(seed: int) -> bool:
    return _probe(_DIAGNOSTIC_COMPLETE_PROBE, seed)


def gate_is_complete() -> bool:
    return _probe(_GATE_COMPLETE_PROBE)


def _safe_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not path.is_symlink()


def _read_cmdline(pid_dir: Path) -> tuple[str, ...] | None:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except (OSError, PermissionError):
        return None
    tokens = raw.rstrip(b"\0").split(b"\0") if raw else []
    try:
        return tuple(token.decode("utf-8", errors="strict") for token in tokens)
    except UnicodeDecodeError:
        return None


def exact_argv_processes(
    expected: Sequence[str], *, proc_root: Path = PROC_ROOT, exclude_pid: int | None = None
) -> tuple[int, ...]:
    found: list[int] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise ReplicationWatcherError("cannot enumerate /proc") from exc
    wanted = tuple(expected)
    for entry in entries:
        if not entry.name.isascii() or not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        if pid == exclude_pid:
            continue
        if _read_cmdline(entry) == wanted:
            found.append(pid)
    return tuple(sorted(found))


def work_processes(argv: Sequence[str]) -> tuple[int, ...]:
    """Recognize both the timed wrapper and its exact Python child argv."""

    return tuple(
        sorted(
            set(exact_argv_processes(argv))
            | set(exact_argv_processes(timed_argv(argv)))
        )
    )


def task_state(task: Task) -> str:
    """Return exactly fresh, resume, complete, or conflict."""
    if task_is_complete(task):
        return "complete"
    if work_processes(training_argv(task, resume=False)) or work_processes(
        training_argv(task, resume=True)
    ):
        return "conflict"
    summary = task.run_dir / "summary.json"
    checkpoint = task.run_dir / "EviSIRST.pth.tar"
    latest = task.run_dir / "last_training_state.pth.tar"
    if summary.exists() or checkpoint.exists():
        return "conflict"
    if _safe_regular(latest):
        return "resume"
    if latest.exists() or latest.is_symlink():
        return "conflict"
    if not task.run_dir.exists():
        return "fresh"
    if not task.run_dir.is_dir() or task.run_dir.is_symlink():
        return "conflict"
    permitted = {".complete_target_replication_v1.lock"}
    try:
        occupants = {item.name for item in task.run_dir.iterdir()}
    except OSError:
        return "conflict"
    return "fresh" if occupants <= permitted else "conflict"


def _parse_gpu_rows(text: str) -> dict[str, GPU]:
    result: dict[str, GPU] = {}
    indexes: set[int] = set()
    buses: set[str] = set()
    for raw_line in text.splitlines():
        fields = tuple(field.strip() for field in raw_line.split(","))
        if len(fields) != 4:
            raise ReplicationWatcherError("unexpected nvidia-smi GPU row")
        try:
            index, memory = int(fields[0]), int(fields[3])
        except ValueError as exc:
            raise ReplicationWatcherError("non-integer nvidia-smi field") from exc
        uuid, bus = fields[1], fields[2].lower()
        if not _UUID_RE.fullmatch(uuid) or uuid in result or index in indexes or bus in buses:
            raise ReplicationWatcherError("ambiguous NVIDIA GPU identity")
        if index < 0 or memory < 0 or not bus:
            raise ReplicationWatcherError("invalid NVIDIA GPU row")
        result[uuid] = GPU(index, uuid, bus, memory)
        indexes.add(index)
        buses.add(bus)
    if not result:
        raise ReplicationWatcherError("nvidia-smi returned no GPUs")
    return result


def _parse_compute_uuids(text: str) -> set[str]:
    occupied: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("No running processes"):
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2 or not _UUID_RE.fullmatch(fields[0]):
            raise ReplicationWatcherError("unexpected nvidia-smi compute-app row")
        int(fields[1])
        occupied.add(fields[0])
    return occupied


def sample_idle_gpus() -> dict[str, GPU]:
    gpu_rows = subprocess.run(
        [os.fspath(NVIDIA_SMI), "--query-gpu=index,uuid,pci.bus_id,memory.used", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    apps = subprocess.run(
        [os.fspath(NVIDIA_SMI), "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if gpu_rows.returncode != 0 or apps.returncode != 0:
        raise ReplicationWatcherError("nvidia-smi query failed")
    gpus = _parse_gpu_rows(gpu_rows.stdout)
    occupied = _parse_compute_uuids(apps.stdout)
    return {
        uuid: gpu for uuid, gpu in gpus.items()
        if uuid not in occupied and gpu.memory_used_mib <= GPU_MEMORY_LIMIT_MIB
    }


def _stable_intersection(samples: Sequence[Mapping[str, GPU]]) -> dict[str, GPU]:
    if not samples:
        return {}
    common = set(samples[0])
    for sample in samples[1:]:
        common.intersection_update(sample)
    result: dict[str, GPU] = {}
    for uuid in common:
        identities = {(s[uuid].index, s[uuid].uuid, s[uuid].bus_id) for s in samples}
        if len(identities) == 1:
            result[uuid] = samples[-1][uuid]
    return result


def _open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ReplicationWatcherError(f"lock is a symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _try_lock(path: Path) -> int | None:
    descriptor = _open_lock(path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _require_safe_directory_chain(path: Path) -> None:
    """Reject a lock/log root redirected through any existing symlink."""

    repository_runs = PROJECT_ROOT / "runs"
    if repository_runs.is_symlink() or (
        repository_runs.exists() and not repository_runs.is_dir()
    ):
        raise ReplicationWatcherError("repository runs root is not a safe directory")
    try:
        relative = path.relative_to(repository_runs)
    except ValueError as exc:
        raise ReplicationWatcherError("watcher state root escaped repository runs") from exc
    current = repository_runs
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ReplicationWatcherError(
                f"watcher state path contains a symlink component: {current}"
            )
        if current.exists() and not current.is_dir():
            raise ReplicationWatcherError(
                f"watcher state path contains a non-directory component: {current}"
            )


def claim_idle_gpus(*, sleep: Callable[[float], None] = time.sleep) -> list[tuple[GPU, int]]:
    first = sample_idle_gpus()
    sleep(GPU_CONFIRM_SECONDS)
    second = sample_idle_gpus()
    candidates = _stable_intersection((first, second))
    locked: list[tuple[GPU, int]] = []
    try:
        for gpu in sorted(candidates.values(), key=lambda item: item.index):
            descriptor = _try_lock(GPU_LOCK_ROOT / f"{gpu.uuid}.lock")
            if descriptor is not None:
                locked.append((gpu, descriptor))
        if not locked:
            return []
        third = sample_idle_gpus()
        sleep(GPU_CONFIRM_SECONDS)
        fourth = sample_idle_gpus()
        confirmed = _stable_intersection((third, fourth))
        accepted: list[tuple[GPU, int]] = []
        for gpu, descriptor in locked:
            observed = confirmed.get(gpu.uuid)
            if observed and (observed.index, observed.bus_id) == (gpu.index, gpu.bus_id):
                accepted.append((observed, descriptor))
            else:
                os.close(descriptor)
        return accepted
    except BaseException:
        for _gpu, descriptor in locked:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _scrubbed_gpu_env(gpu: GPU) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    return env


def _release_job(job: RunningJob) -> None:
    job.log_handle.close()
    os.close(job.task_lock_fd)
    os.close(job.gpu_lock_fd)


def launch_gpu_work(work_name: str, argv: Sequence[str], gpu: GPU, gpu_fd: int) -> RunningJob | None:
    task_fd = _try_lock(TASK_LOCK_ROOT / f"{work_name}.lock")
    if task_fd is None:
        os.close(gpu_fd)
        return None
    if work_processes(argv):
        os.close(task_fd)
        os.close(gpu_fd)
        return None
    if not authorization_is_valid():
        os.close(task_fd)
        os.close(gpu_fd)
        _log(f"released {work_name}: canonical PASS authorization is no longer valid")
        return None
    # Strict product probes run between the four-sample claim and this point.
    # Close that TOCTOU window immediately before spawning the timed worker.
    try:
        final_idle = sample_idle_gpus()
    except BaseException:
        os.close(task_fd)
        os.close(gpu_fd)
        raise
    observed = final_idle.get(gpu.uuid)
    if observed is None or (observed.index, observed.bus_id) != (gpu.index, gpu.bus_id):
        os.close(task_fd)
        os.close(gpu_fd)
        _log(f"released {work_name}: GPU identity is no longer idle")
        return None
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{work_name}.log"
    log_handle = log_path.open("ab", buffering=0)
    command = list(timed_argv(argv))
    try:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=_scrubbed_gpu_env(gpu),
            stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
            pass_fds=(gpu_fd, task_fd), start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        os.close(task_fd)
        os.close(gpu_fd)
        raise
    _log(f"launched {work_name} pid={process.pid} gpu={gpu.uuid} index={gpu.index} bus={gpu.bus_id}")
    return RunningJob(work_name, process, gpu, gpu_fd, task_fd, log_handle)


def _diagnostic_output(seed: int) -> Path:
    return (
        PROJECT_ROOT / "runs/irstd_performance/complete_target_v1/"
        "paired_baseline_diagnostics" / f"run_seed_{seed}/matched_target_diagnostics.json"
    )


def _next_work(running: Mapping[str, RunningJob], retry_after: Mapping[str, float]) -> tuple[str, tuple[str, ...]] | None:
    now = time.monotonic()
    for task in TASKS:
        diag_name = f"diag{str(task.run_seed)[:3]}"
        if task.kind == "baseline" and task_state(task) == "complete" and not diagnostic_is_complete(task.run_seed):
            if _diagnostic_output(task.run_seed).exists() or work_processes(diagnostic_argv(task.run_seed)):
                continue
            if diag_name not in running and now >= retry_after.get(diag_name, 0.0):
                return diag_name, diagnostic_argv(task.run_seed)
    for task in TASKS:
        state = task_state(task)
        _log(f"task={task.name} state={state}")
        if state in {"complete", "conflict"} or task.name in running:
            continue
        if now >= retry_after.get(task.name, 0.0):
            return task.name, training_argv(task, resume=(state == "resume"))
    return None


def _all_inputs_complete() -> bool:
    return all(task_state(task) == "complete" for task in TASKS) and all(
        diagnostic_is_complete(seed) for seed in RUN_SEEDS
    )


def _run_gate_once() -> int:
    gate_fd = _try_lock(TASK_LOCK_ROOT / "three_seed_gate.lock")
    if gate_fd is None:
        _log("three-seed gate task lock is already owned")
        return 75
    try:
        if exact_argv_processes(gate_argv()) or not authorization_is_valid():
            _log("three-seed gate launch preflight no longer passes")
            return 75
        env = dict(os.environ)
        for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES"):
            env.pop(name, None)
        completed = subprocess.run(
            list(gate_argv()), cwd=PROJECT_ROOT, env=env,
            stdin=subprocess.DEVNULL, check=False, pass_fds=(gate_fd,),
        )
        return completed.returncode
    finally:
        os.close(gate_fd)


def supervise(*, sleep: Callable[[float], None] = time.sleep) -> int:
    running: dict[str, RunningJob] = {}
    failures: dict[str, int] = {}
    retry_after: dict[str, float] = {}
    while True:
        for name, job in tuple(running.items()):
            returncode = job.process.poll()
            if returncode is None:
                continue
            _release_job(job)
            del running[name]
            if returncode == 0:
                failures.pop(name, None)
                retry_after.pop(name, None)
                _log(f"work={name} exited successfully")
            else:
                failures[name] = failures.get(name, 0) + 1
                delay = min(INITIAL_RETRY_SECONDS * (2 ** (failures[name] - 1)), MAX_RETRY_SECONDS)
                retry_after[name] = time.monotonic() + delay
                _log(f"work={name} failed rc={returncode}; retry in {delay}s")

        if gate_is_complete():
            _log("strict three-seed gate result is complete")
            return 0
        if not authorization_is_valid():
            _log("canonical single-seed PASS authorization is not currently valid")
            sleep(POLL_SECONDS)
            continue
        if not running and _all_inputs_complete():
            status = _run_gate_once()
            if status == 0 and gate_is_complete():
                _log("three-seed CPU gate completed")
                return 0
            _log(f"three-seed CPU gate failed rc={status}; retrying later")
            sleep(INITIAL_RETRY_SECONDS)
            continue

        available = claim_idle_gpus(sleep=sleep)
        for gpu, gpu_fd in available:
            work = _next_work(running, retry_after)
            if work is None:
                os.close(gpu_fd)
                continue
            name, argv = work
            job = launch_gpu_work(name, argv, gpu, gpu_fd)
            if job is not None:
                running[name] = job
        sleep(POLL_SECONDS)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ReplicationWatcherError("fixed watcher accepts no arguments")
    try:
        python_target = PYTHON_BIN.resolve(strict=True)
    except OSError as exc:
        raise ReplicationWatcherError("fixed virtual-environment Python is unavailable") from exc
    if (
        python_target != EXPECTED_PYTHON
        or not python_target.is_file()
        or not os.access(python_target, os.X_OK)
    ):
        raise ReplicationWatcherError("fixed virtual-environment Python target differs")
    for state_root in (WATCH_ROOT, GPU_LOCK_ROOT, TASK_LOCK_ROOT, LOG_ROOT):
        _require_safe_directory_chain(state_root)
    for required in (BASELINE_TRAINER, VARIANT_TRAINER, DIAGNOSTIC_RUNNER, THREE_SEED_GATE, SINGLE_SEED_GATE, TIME_BIN, NVIDIA_SMI):
        if not required.is_file() or required.is_symlink():
            raise ReplicationWatcherError(f"required fixed file is unavailable: {required}")
    singleton = _try_lock(WATCHER_LOCK)
    if singleton is None:
        _log("another fixed replication watcher owns the singleton lock")
        return 0
    try:
        return supervise()
    finally:
        os.close(singleton)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GPU", "TASKS", "Task", "ReplicationWatcherError", "authorization_is_valid",
    "claim_idle_gpus", "diagnostic_argv", "diagnostic_is_complete",
    "exact_argv_processes", "gate_argv", "gate_is_complete", "launch_gpu_work",
    "main", "sample_idle_gpus", "supervise", "task_is_complete", "task_state",
    "timed_argv", "training_argv", "work_processes",
]
