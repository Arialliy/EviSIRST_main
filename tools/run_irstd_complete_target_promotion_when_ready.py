#!/usr/bin/env python3
"""Wait for the fixed IRSTD complete-target inputs, then run the CPU gate.

This watcher has no command-line configuration.  It recognizes the two formal
trainers, the paired-baseline diagnostic, and the promotion gate by exact
``/proc/<pid>/cmdline`` tokens.  The fixed gate is launched only after all five
immutable inputs are regular files and all matching producer processes have
exited.  A separate non-blocking run lock prevents duplicate watchers.

The watcher itself uses only the Python standard library and never queries or
claims a GPU.  A failed gate that produced no result is retried with bounded
exponential backoff; an existing result is never overwritten.
"""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
PROC_ROOT = Path("/proc")

BASELINE_TRAINER = PROJECT_ROOT / "train_validation_selected.py"
VARIANT_TRAINER = PROJECT_ROOT / "train_irstd_complete_target_v1.py"
BASELINE_DIAGNOSTIC = PROJECT_ROOT / "run_irstd_paired_baseline_diagnostic.py"
PROMOTION_GATE = PROJECT_ROOT / "run_irstd_complete_target_promotion_gate.py"

BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "runs/validation_selected/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/summary.json"
)
BASELINE_CHECKPOINT = BASELINE_SUMMARY.with_name("EviSIRST.pth.tar")
VARIANT_SUMMARY = (
    PROJECT_ROOT
    / "runs/irstd_performance/complete_target_v1/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/summary.json"
)
VARIANT_CHECKPOINT = VARIANT_SUMMARY.with_name("EviSIRST.pth.tar")
BASELINE_DIAGNOSTIC_OUTPUT = (
    PROJECT_ROOT
    / "runs/irstd_performance/complete_target_v1/"
    "paired_baseline_diagnostics/run_seed_1446202191/"
    "matched_target_diagnostics.json"
)
RESULT = (
    PROJECT_ROOT
    / "runs/irstd_performance/complete_target_v1/"
    "promotion_gate/run_seed_1446202191/result.json"
)
RUN_LOCK = RESULT.with_name("watcher.run.lock")
FIXED_INPUTS = (
    BASELINE_SUMMARY,
    BASELINE_CHECKPOINT,
    VARIANT_SUMMARY,
    VARIANT_CHECKPOINT,
    BASELINE_DIAGNOSTIC_OUTPUT,
)

POLL_SECONDS = 10
INITIAL_FAILURE_BACKOFF_SECONDS = 30
MAX_FAILURE_BACKOFF_SECONDS = 300


class PromotionWatcherError(RuntimeError):
    """The fixed watcher contract cannot be followed safely."""


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    script: Path
    required_option_values: tuple[tuple[str, str], ...] = ()
    forbidden_tokens: tuple[str, ...] = ()


PROCESS_SPECS = (
    ProcessSpec(
        name="paired_baseline_trainer",
        script=BASELINE_TRAINER,
        required_option_values=(
            ("--dataset", "IRSTD-1K"),
            ("--target-mode", "binary"),
            ("--run-seed", "1446202191"),
        ),
        forbidden_tokens=(
            "--smoke",
            "--smoke-max-train-samples",
            "--smoke-max-val-samples",
        ),
    ),
    ProcessSpec(
        name="complete_target_v1_trainer",
        script=VARIANT_TRAINER,
        required_option_values=(("--run-seed", "1446202191"),),
        forbidden_tokens=(
            "--smoke",
            "--smoke-max-train-samples",
            "--smoke-max-val-samples",
        ),
    ),
    ProcessSpec(
        name="paired_baseline_diagnostic",
        script=BASELINE_DIAGNOSTIC,
        required_option_values=(
            ("--dataset-root", "/home/ly/SCTransNet_main/datasets"),
        ),
    ),
    ProcessSpec(name="promotion_gate", script=PROMOTION_GATE),
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _is_safe_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        return False
    try:
        return path.resolve(strict=True) == path
    except (FileNotFoundError, RuntimeError):
        return False


def _validate_result_path_state() -> bool:
    """Return true for an existing regular result; reject all other occupants."""

    if _is_safe_regular_file(RESULT):
        return True
    if RESULT.exists() or RESULT.is_symlink():
        raise PromotionWatcherError(
            "fixed promotion result exists but is not a safe regular file"
        )
    return False


def fixed_inputs_are_ready() -> bool:
    return all(_is_safe_regular_file(path) for path in FIXED_INPUTS)


def _read_cmdline(pid_directory: Path) -> tuple[str, ...] | None:
    try:
        content = (pid_directory / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    if not content:
        return None
    tokens = content.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    if not tokens:
        return None
    try:
        decoded = tuple(token.decode("utf-8", errors="strict") for token in tokens)
    except UnicodeDecodeError:
        return None
    if any("\x00" in token for token in decoded):
        return None
    return decoded


def _is_python_process(pid_directory: Path) -> bool:
    try:
        executable = (pid_directory / "exe").resolve(strict=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False
    return executable.name.startswith("python")


def _process_cwd(pid_directory: Path) -> Path | None:
    try:
        return (pid_directory / "cwd").resolve(strict=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _argv_has_script(argv: Sequence[str], cwd: Path, expected: Path) -> bool:
    for token in argv:
        if token == os.fspath(expected):
            return True
        if not token.startswith("-") and token.endswith(".py"):
            candidate = Path(token)
            if not candidate.is_absolute():
                candidate = cwd / candidate
            try:
                if candidate.resolve(strict=True) == expected:
                    return True
            except (FileNotFoundError, RuntimeError, OSError):
                continue
    return False


def _argv_has_option_value(
    argv: Sequence[str], option: str, expected: str
) -> bool:
    for index, token in enumerate(argv):
        if token == f"{option}={expected}":
            return True
        if (
            token == option
            and index + 1 < len(argv)
            and argv[index + 1] == expected
        ):
            return True
    return False


def argv_matches_spec(
    argv: Sequence[str], *, cwd: Path, spec: ProcessSpec
) -> bool:
    if not _argv_has_script(argv, cwd, spec.script):
        return False
    if any(
        token == forbidden or token.startswith(f"{forbidden}=")
        for token in argv
        for forbidden in spec.forbidden_tokens
    ):
        return False
    return all(
        _argv_has_option_value(argv, option, value)
        for option, value in spec.required_option_values
    )


def matching_processes(
    *, proc_root: Path = PROC_ROOT, exclude_pid: int | None = None
) -> dict[str, tuple[int, ...]]:
    matches: dict[str, list[int]] = {spec.name: [] for spec in PROCESS_SPECS}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise PromotionWatcherError("cannot enumerate /proc safely") from exc
    for pid_directory in entries:
        if not pid_directory.name.isascii() or not pid_directory.name.isdecimal():
            continue
        pid = int(pid_directory.name)
        if pid == exclude_pid or not _is_python_process(pid_directory):
            continue
        argv = _read_cmdline(pid_directory)
        cwd = _process_cwd(pid_directory)
        if argv is None or cwd is None:
            continue
        for spec in PROCESS_SPECS:
            if argv_matches_spec(argv, cwd=cwd, spec=spec):
                matches[spec.name].append(pid)
    return {name: tuple(sorted(pids)) for name, pids in matches.items()}


@contextmanager
def exclusive_watcher_lock(path: Path = RUN_LOCK) -> Iterator[int | None]:
    """Yield the fixed lock FD, or ``None`` when another watcher owns it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PromotionWatcherError("fixed watcher run lock is a symbolic link")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PromotionWatcherError("cannot open fixed watcher run lock") from exc
    try:
        opened = os.fstat(descriptor)
        on_disk = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(on_disk.st_mode)
            or (opened.st_dev, opened.st_ino) != (on_disk.st_dev, on_disk.st_ino)
        ):
            raise PromotionWatcherError("fixed watcher run lock identity differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        yield descriptor
    finally:
        os.close(descriptor)


def _producer_or_gate_is_running(matches: dict[str, tuple[int, ...]]) -> bool:
    return any(matches.get(spec.name, ()) for spec in PROCESS_SPECS)


def run_gate_once() -> int:
    environment = os.environ.copy()
    # This evaluator is CPU-only.  Do not let an outer training/test shell's
    # accelerator visibility or ordering leak into the fixed child process.
    for name in tuple(environment):
        if name.startswith(("CUDA_", "NVIDIA_VISIBLE_")):
            environment.pop(name)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    completed = subprocess.run(
        [os.fspath(PYTHON_BIN), os.fspath(PROMOTION_GATE)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def watch_forever(
    *,
    sleep: Callable[[float], None] = time.sleep,
    gate_runner: Callable[[], int] = run_gate_once,
) -> int:
    failure_backoff = INITIAL_FAILURE_BACKOFF_SECONDS
    last_status = 0.0
    while True:
        if _validate_result_path_state():
            _log("immutable promotion-gate result already exists; exiting")
            return 0

        processes = matching_processes(exclude_pid=os.getpid())
        if fixed_inputs_are_ready() and not _producer_or_gate_is_running(processes):
            # Close the TOCTOU window as far as possible before spawning.  The
            # gate independently re-reads, hashes, and validates every input.
            if _validate_result_path_state():
                _log("promotion-gate result appeared before launch; exiting")
                return 0
            second_process_check = matching_processes(exclude_pid=os.getpid())
            if (
                not fixed_inputs_are_ready()
                or _producer_or_gate_is_running(second_process_check)
            ):
                sleep(POLL_SECONDS)
                continue

            _log("five fixed inputs are ready and all producers exited; running CPU gate")
            status = gate_runner()
            if _validate_result_path_state():
                if status == 0:
                    _log("promotion gate completed; immutable result exists")
                    return 0
                _log(
                    "gate returned failure after a result appeared; refusing any retry"
                )
                return 1
            _log(
                f"gate failed with status {status} and no result; "
                f"retrying after {failure_backoff}s"
            )
            sleep(failure_backoff)
            failure_backoff = min(
                MAX_FAILURE_BACKOFF_SECONDS, failure_backoff * 2
            )
            continue

        now = time.monotonic()
        if now - last_status >= 300:
            active = [name for name, pids in processes.items() if pids]
            missing = [path.name for path in FIXED_INPUTS if not _is_safe_regular_file(path)]
            _log(
                "waiting for fixed readiness; "
                f"active={active or ['none']} missing={missing or ['none']}"
            )
            last_status = now
        sleep(POLL_SECONDS)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        raise PromotionWatcherError("this fixed watcher accepts no arguments")
    os.umask(0o077)
    with exclusive_watcher_lock() as descriptor:
        if descriptor is None:
            _log("another promotion-gate watcher holds the fixed run lock; exiting")
            return 0
        _log("CPU-only watcher started; waiting for five fixed inputs")
        return watch_forever()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIXED_INPUTS",
    "PROCESS_SPECS",
    "PromotionWatcherError",
    "RUN_LOCK",
    "argv_matches_spec",
    "exclusive_watcher_lock",
    "fixed_inputs_are_ready",
    "main",
    "matching_processes",
    "run_gate_once",
    "watch_forever",
]
