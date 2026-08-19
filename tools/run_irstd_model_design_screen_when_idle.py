#!/usr/bin/env python3
"""Safely schedule the frozen IRSTD model-design legacy screen.

The long-lived watcher is standard-library-only.  Formal work is disabled
until ``FORMAL_AUTHORIZATION_SHA256`` is replaced by the SHA-256 of the fixed
authorization file.  Torch/model/optimizer artifact checks run only in the
fixed project interpreter with CUDA hidden.

Formal workers retain the runner's 1000-epoch schedule.  A worker interposes
only after R1 has atomically committed epoch 500 and its candidate cleanup has
returned, then blocks inside that transaction boundary.  The watcher validates
the complete commit, sends SIGTERM to the worker's independent process group,
and validates again before recording completion.  The training loop therefore
cannot enter epoch 501.
"""

from __future__ import annotations

import argparse
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


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
EXPECTED_PYTHON = Path("/usr/bin/python3.12")
DATASET_ROOT = Path("/home/ly/SCTransNet_main/datasets")
SPLIT_ROOT = PROJECT_ROOT / "splits/v2"
TRAINER = PROJECT_ROOT / "train_irstd_model_design_screen_v1.py"
BASELINE_TRAINER = PROJECT_ROOT / "train_validation_selected.py"
WATCHER_SOURCE = (
    PROJECT_ROOT / "tools/run_irstd_model_design_screen_when_idle.py"
)
TIME_BIN = Path("/usr/bin/time")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
PROC_ROOT = Path("/proc")

WATCH_ROOT = PROJECT_ROOT / "runs/irstd_model_design/legacy_screen_watcher_v1"
WATCHER_LOCK = WATCH_ROOT / "watcher.lock"
TASK_LOCK_ROOT = WATCH_ROOT / "task_locks"
LOG_ROOT = WATCH_ROOT / "logs"
COMPLETION_ROOT = WATCH_ROOT / "epoch500_commits"
GPU_LOCK_ROOT = PROJECT_ROOT / "runs/.gpu_locks"

AUTHORIZATION_PATH = (
    PROJECT_ROOT / "experiments/IRSTD_MODEL_DESIGN_SCREEN_V1_AUTHORIZATION.json"
)
# Deliberately disabled.  Formal launch remains unreachable until a reviewed
# authorization file exists and this value is explicitly frozen to its hash.
FORMAL_AUTHORIZATION_SHA256: str | None = None
AUTHORIZATION_SCHEMA = "evisirst_irstd_model_design_screen_launch_authorization/v1"
COMPLETION_SCHEMA = "evisirst_irstd_model_design_epoch500_stop/v1"

FINAL_CONFIGURED_EPOCH = 1000
STOP_EPOCH = 500
POLL_SECONDS = 10
GPU_CONFIRM_SECONDS = 10
GPU_MEMORY_LIMIT_MIB = 1024
TERM_TIMEOUT_SECONDS = 60
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_PROBE_PREFIX = "EVISIRST_MODEL_DESIGN_SCREEN_PROBE:"


class ModelDesignWatcherError(RuntimeError):
    """The fixed screen queue cannot be supervised safely."""


@dataclass(frozen=True)
class Task:
    name: str
    kind: str
    variant: str
    run_seed: int
    run_dir: Path
    resume_required: bool = False


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    bus_id: str
    memory_used_mib: int


@dataclass
class RunningJob:
    task: Task
    process: subprocess.Popen[bytes]
    gpu: GPU
    gpu_lock_fd: int
    task_lock_fd: int
    log_handle: object


def _run_dir(variant: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "runs/irstd_model_design"
        / variant
        / "legacy_screen/formal/IRSTD-1K/binary"
        / f"run_seed_{seed}"
    )


TASKS = (
    Task(
        "clean_a104_to500",
        "clean_baseline",
        "clean_evisirst",
        104728269,
        PROJECT_ROOT
        / "runs/validation_selected/formal/IRSTD-1K/binary/run_seed_104728269",
        True,
    ),
    Task(
        "d0_s42",
        "model_design",
        "single_residual_v1",
        42,
        _run_dir("single_residual_v1", 42),
    ),
    Task(
        "d0_s1446202191",
        "model_design",
        "single_residual_v1",
        1446202191,
        _run_dir("single_residual_v1", 1446202191),
    ),
    Task("p_s42", "model_design", "psbfr_v1", 42, _run_dir("psbfr_v1", 42)),
    Task(
        "p_s1446202191",
        "model_design",
        "psbfr_v1",
        1446202191,
        _run_dir("psbfr_v1", 1446202191),
    ),
    Task(
        "p_s104728269",
        "model_design",
        "psbfr_v1",
        104728269,
        _run_dir("psbfr_v1", 104728269),
    ),
)


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
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def formal_training_argv(task: Task, *, resume: bool) -> tuple[str, ...]:
    if task.kind == "clean_baseline":
        if not resume:
            raise ModelDesignWatcherError("clean A104 is a resume-only fixed task")
        argv = (
            os.fspath(PYTHON_BIN),
            os.fspath(BASELINE_TRAINER),
            "--dataset",
            "IRSTD-1K",
            "--dataset-root",
            os.fspath(DATASET_ROOT),
            "--split-root",
            os.fspath(SPLIT_ROOT),
            "--output-root",
            os.fspath(PROJECT_ROOT / "runs/validation_selected"),
            "--target-mode",
            "binary",
            "--architecture-seed",
            "42",
            "--run-seed",
            str(task.run_seed),
            "--device",
            "cuda:0",
            "--epochs",
            "1000",
            "--batch-size",
            "16",
            "--workers",
            "0",
            "--base-lr",
            "0.001",
            "--min-lr",
            "0.00001",
            "--warmup-epochs",
            "10",
            "--val-interval",
            "1",
            "--allow-sample-level-fallback",
        )
    elif task.kind == "model_design":
        argv = (
            os.fspath(PYTHON_BIN),
            os.fspath(TRAINER),
            "--dataset-root",
            os.fspath(DATASET_ROOT),
            "--split-root",
            os.fspath(SPLIT_ROOT),
            "--variant",
            task.variant,
            "--dataset",
            "IRSTD-1K",
            "--target-mode",
            "binary",
            "--architecture-seed",
            "42",
            "--run-seed",
            str(task.run_seed),
            "--device",
            "cuda:0",
            "--epochs",
            "1000",
            "--warmup-epochs",
            "10",
            "--allow-sample-level-fallback",
        )
    else:
        raise ModelDesignWatcherError(f"unknown task kind: {task.kind}")
    return argv + (("--resume",) if resume else ())


def smoke_training_argv(task: Task) -> tuple[str, ...]:
    """Return a test-only, promotion-ineligible command in the smoke tree."""

    if task.kind != "model_design":
        raise ModelDesignWatcherError("clean A104 has no smoke-mode command")
    return (
        os.fspath(PYTHON_BIN),
        os.fspath(TRAINER),
        "--dataset-root",
        os.fspath(DATASET_ROOT),
        "--split-root",
        os.fspath(SPLIT_ROOT),
        "--variant",
        task.variant,
        "--run-seed",
        str(task.run_seed),
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--warmup-epochs",
        "0",
        "--smoke-max-train-samples",
        "1",
        "--smoke-max-val-samples",
        "1",
        "--allow-sample-level-fallback",
    )


def timed_argv(argv: Sequence[str]) -> tuple[str, ...]:
    return os.fspath(TIME_BIN), "-v", *tuple(argv)


def task_manifest() -> dict[str, object]:
    entries = []
    for task in TASKS:
        entries.append(
            {
                "name": task.name,
                "kind": task.kind,
                "variant": task.variant,
                "run_seed": task.run_seed,
                "run_dir": task.run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "fresh_argv": (
                    None
                    if task.resume_required
                    else list(formal_training_argv(task, resume=False))
                ),
                "resume_argv": list(formal_training_argv(task, resume=True)),
            }
        )
    return {
        "schema": "evisirst_irstd_model_design_screen_task_manifest/v1",
        "configured_total_epochs": FINAL_CONFIGURED_EPOCH,
        "controlled_stop_epoch": STOP_EPOCH,
        "tasks": entries,
    }


def _strict_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ModelDesignWatcherError(f"not a regular JSON file: {path}")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ModelDesignWatcherError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(token: str) -> object:
        raise ModelDesignWatcherError(f"non-finite JSON constant: {token}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelDesignWatcherError("authorization/marker JSON is malformed") from exc

    def finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ModelDesignWatcherError("non-finite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def authorization_is_valid() -> bool:
    expected = FORMAL_AUTHORIZATION_SHA256
    if expected is None:
        return False
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ModelDesignWatcherError("frozen authorization SHA-256 is malformed")
    if _sha256_file(AUTHORIZATION_PATH) != expected:
        return False
    payload = _strict_json(AUTHORIZATION_PATH)
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == AUTHORIZATION_SCHEMA
        and payload.get("status") == "AUTHORIZED"
        and payload.get("task_manifest_sha256")
        == _canonical_sha256(task_manifest())
        and payload.get("public_test_allowed") is False
        and payload.get("test_split_accessed") is False
    )


def make_cleanup_barrier(
    original: Callable[..., object],
    *,
    pause: Callable[[], object],
    announce: Callable[[], object],
) -> Callable[..., object]:
    """Wrap post-commit cleanup and block before epoch 501 can start.

    ``original`` must return first, proving candidate cleanup completed.  At
    epoch 500 the wrapper then announces readiness and remains blocked until
    the process group receives SIGTERM.  Any direct observation above 500 is
    a fail-closed protocol violation.
    """

    def wrapped(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        epoch = kwargs.get("completed_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ModelDesignWatcherError("cleanup completed_epoch is malformed")
        if epoch > STOP_EPOCH:
            raise ModelDesignWatcherError("epoch 501 is forbidden in legacy screen")
        if epoch == STOP_EPOCH:
            announce()
            while True:
                pause()
        return result

    return wrapped


# Executed only in the fixed venv.  It imports the model runner there, wraps
# R1's post-commit candidate cleanup, and blocks before the loop can advance.
_FORMAL_WORKER_PROGRAM = r'''
import os, signal, sys
from tools.run_irstd_model_design_screen_when_idle import make_cleanup_barrier

kind=sys.argv[1]
variant=sys.argv[2]
seed=int(sys.argv[3])
resume=sys.argv[4] == "resume"
expected_auth=sys.argv[5]
auth_path=sys.argv[6]

if expected_auth == "DISABLED" or not os.path.isfile(auth_path):
    raise RuntimeError("formal authorization is unavailable")
import hashlib
h=hashlib.sha256(open(auth_path,"rb").read()).hexdigest()
if h != expected_auth:
    raise RuntimeError("formal authorization hash differs")

def announce(): print("EVISIRST_MODEL_DESIGN_SCREEN_WORKER:EPOCH500_CLEAN", flush=True)
if kind == "model_design":
  import train_irstd_model_design_screen_v1 as runner
  args=runner.parse_args([
      "--dataset-root","/home/ly/SCTransNet_main/datasets",
      "--split-root","/home/ly/EviSIRST_main/splits/v2",
      "--variant",variant,"--dataset","IRSTD-1K","--target-mode","binary",
      "--architecture-seed","42","--run-seed",str(seed),"--device","cuda:0",
      "--epochs","1000","--warmup-epochs","10",
      "--allow-sample-level-fallback",*( ["--resume"] if resume else [] ),
  ])
  runner.r1._validate_resume_candidates=make_cleanup_barrier(
      runner.r1._validate_resume_candidates,pause=signal.pause,announce=announce)
  runner.run(args)
elif kind == "clean_baseline":
  if not resume: raise RuntimeError("clean A104 is resume-only")
  import train_validation_selected as runner
  args=runner.parse_args([
      "--dataset","IRSTD-1K","--dataset-root","/home/ly/SCTransNet_main/datasets",
      "--split-root","/home/ly/EviSIRST_main/splits/v2",
      "--output-root","/home/ly/EviSIRST_main/runs/validation_selected",
      "--target-mode","binary","--architecture-seed","42","--run-seed",str(seed),
      "--device","cuda:0","--epochs","1000","--batch-size","16","--workers","0",
      "--base-lr","0.001","--min-lr","0.00001","--warmup-epochs","10",
      "--val-interval","1","--allow-sample-level-fallback","--resume"])
  runner._validate_resume_candidates=make_cleanup_barrier(
      runner._validate_resume_candidates,pause=signal.pause,announce=announce)
  runner.run(args)
else: raise RuntimeError("unknown fixed task kind")
raise RuntimeError("formal worker returned without controlled epoch-500 stop")
'''


def worker_argv(task: Task, *, resume: bool) -> tuple[str, ...]:
    expected = FORMAL_AUTHORIZATION_SHA256 or "DISABLED"
    return (
        os.fspath(PYTHON_BIN),
        "-c",
        _FORMAL_WORKER_PROGRAM,
        task.kind,
        task.variant,
        str(task.run_seed),
        "resume" if resume else "fresh",
        expected,
        os.fspath(AUTHORIZATION_PATH),
    )


def timed_worker_argv(task: Task, *, resume: bool) -> tuple[str, ...]:
    return timed_argv(worker_argv(task, resume=resume))


# Read-only fixed-venv probe.  It uses torch only in the short child process.
_STATE_PROBE_PROGRAM = r'''
import hashlib, json, math, os, stat, sys
from pathlib import Path
import torch

PREFIX="EVISIRST_MODEL_DESIGN_SCREEN_PROBE:"
kind=sys.argv[1]; variant=sys.argv[2]; seed=int(sys.argv[3])
def emit(value): print(PREFIX+value,flush=True)
def fail(message): raise RuntimeError(message)
def regular(path):
    st=path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode): fail("not regular")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def finite(value):
    if isinstance(value,float) and not math.isfinite(value): fail("nonfinite")
    if isinstance(value,dict):
        for k,v in value.items():
            if k == "test_split_accessed" and v is not False: fail("test disclosure")
            if k in {"public_test_supported","public_test_allowed"} and v is not False: fail("public disclosure")
            finite(v)
    elif isinstance(value,list):
        for v in value: finite(v)

if kind == "model_design":
 import train_irstd_model_design_screen_v1 as r
 args=r.parse_args([
  "--dataset-root","/home/ly/SCTransNet_main/datasets",
  "--split-root","/home/ly/EviSIRST_main/splits/v2","--variant",variant,
  "--dataset","IRSTD-1K","--target-mode","binary","--architecture-seed","42",
  "--run-seed",str(seed),"--device","cuda:0","--epochs","1000",
  "--warmup-epochs","10","--allow-sample-level-fallback","--resume"])
 context=r._hf_adapter(args)
elif kind == "clean_baseline":
 import contextlib, train_validation_selected as r
 args=r.parse_args([
  "--dataset","IRSTD-1K","--dataset-root","/home/ly/SCTransNet_main/datasets",
  "--split-root","/home/ly/EviSIRST_main/splits/v2",
  "--output-root","/home/ly/EviSIRST_main/runs/validation_selected",
  "--target-mode","binary","--architecture-seed","42","--run-seed",str(seed),
  "--device","cuda:0","--epochs","1000","--batch-size","16","--workers","0",
  "--base-lr","0.001","--min-lr","0.00001","--warmup-epochs","10",
  "--val-interval","1","--allow-sample-level-fallback","--resume"])
 context=contextlib.nullcontext()
else: fail("unknown task kind")
with context:
 engine=r.r1 if kind == "model_design" else r
 paths=r.resolve_run_paths(args); run=paths["run_dir"]
 if not run.exists(): emit("FRESH"); raise SystemExit(0)
 if run.is_symlink() or not run.is_dir(): emit("CONFLICT"); raise SystemExit(0)
 latest=paths["latest"]; history_path=paths["history"]
 forbidden=[paths["summary"],paths["final"]]
 if kind == "model_design": forbidden += [paths["best_mIoU_final"],paths["best_Pd_final"]]
 if any(p.exists() or p.is_symlink() for p in forbidden): emit("CONFLICT"); raise SystemExit(0)
 permitted={".model_design_screen_v1.lock","last_training_state.pth.tar","history.json","candidates"}
 if not latest.exists():
    if {p.name for p in run.iterdir()} <= {".model_design_screen_v1.lock"}: emit("FRESH")
    else: emit("CONFLICT")
    raise SystemExit(0)
 regular(latest); regular(history_path)
 latest_value=torch.load(latest,map_location="cpu",weights_only=True); finite(latest_value)
 if kind == "model_design": identity=r.hf_transaction._current_identity_for_args(args)
 else:
  full_train,full_val,grouping=r.build_datasets(args)
  identity=r._run_identity(args,full_train.contract,grouping,train_count=len(full_train),val_count=len(full_val),smoke=False)
 if (not isinstance(latest_value,dict) or latest_value.get("schema") != r.TRAINING_SCHEMA
     or latest_value.get("run_identity") != identity or latest_value.get("test_split_accessed") is not False): fail("latest identity")
 epoch=latest_value.get("epoch")
 if isinstance(epoch,bool) or not isinstance(epoch,int) or not 1 <= epoch <= 500: fail("epoch outside screen")
 model,_=(r._variant_initialize_evisirst("IRSTD-1K",seed=42,training=True)
           if kind == "model_design" else r.initialize_evisirst("IRSTD-1K",seed=42,training=True))
 r._validate_state_dict(latest_value.get("state_dict"),model.state_dict())
 optimizer=torch.optim.Adam(model.parameters(),lr=args.base_lr)
 engine._validate_and_load_adam_optimizer_state(
   optimizer_state=latest_value.get("optimizer"),model=model,optimizer=optimizer,
   identity=identity,completed_epoch=epoch,total_epochs=1000)
 train,val=engine._validate_resume_history(
   completed_epoch=epoch,interval=1,training_history=latest_value.get("training_history"),
   validation_history=latest_value.get("validation_history"))
 if len(train)!=epoch or len(val)!=epoch or train[-1]["epoch"]!=epoch or val[-1]["epoch"]!=epoch: fail("history endpoint")
 artifacts=latest_value.get("candidate_artifacts")
 frontier=(tuple(r.zero_selection.retention_frontier_epochs(val,margin=None))
           if kind == "model_design" else tuple(r.selection.retention_frontier_epochs(val)))
 if not isinstance(artifacts,dict) or tuple(sorted(artifacts)) != frontier: fail("frontier")
 candidate_dir=paths["candidate_dir"]
 if candidate_dir.is_symlink() or not candidate_dir.is_dir(): fail("candidate dir")
 names=set(); by_epoch={v["epoch"]:v for v in val}
 for ce in frontier:
   meta=artifacts[ce]; name="epoch_%04d.pth.tar"%ce; path=candidate_dir/name; names.add(name); regular(path)
   if meta != {"relative_path":"candidates/"+name,"file_sha256":sha(path)}: fail("candidate sha")
   payload=torch.load(path,map_location="cpu",weights_only=True); finite(payload)
   if (payload.get("schema")!=r.CANDIDATE_SCHEMA or payload.get("run_identity")!=identity
       or payload.get("epoch")!=ce or payload.get("validation_record")!=by_epoch[ce]
       or payload.get("test_split_accessed") is not False): fail("candidate identity")
   r._validate_state_dict(payload.get("state_dict"),model.state_dict())
 if {p.name for p in candidate_dir.iterdir()} != names: fail("candidate directory exactness")
 raw=json.loads(history_path.read_text(encoding="utf-8"),parse_constant=lambda x:fail("json constant")); finite(raw)
 if (raw.get("schema")!=r.HISTORY_SCHEMA or raw.get("run_identity")!=identity
     or raw.get("training_history")!=train or raw.get("validation_history")!=val
     or raw.get("candidate_artifacts")!={str(k):v for k,v in artifacts.items()}
     or raw.get("test_split_accessed") is not False): fail("history json")
 emit("READY" if epoch == 500 else "RESUME")
'''


def _probe_task_state(task: Task) -> str:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    completed = subprocess.run(
        [
            os.fspath(PYTHON_BIN),
            "-c",
            _STATE_PROBE_PROGRAM,
            task.kind,
            task.variant,
            str(task.run_seed),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
    )
    tokens = [
        line[len(_PROBE_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_PROBE_PREFIX)
    ]
    if completed.returncode != 0 or len(tokens) != 1:
        return "conflict"
    value = tokens[0].lower()
    return value if value in {"fresh", "resume", "ready", "conflict"} else "conflict"


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
    expected: Sequence[str],
    *,
    proc_root: Path = PROC_ROOT,
    exclude_pid: int | None = None,
) -> tuple[int, ...]:
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise ModelDesignWatcherError("cannot enumerate /proc") from exc
    wanted = tuple(expected)
    found = []
    for entry in entries:
        if not entry.name.isascii() or not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        if pid != exclude_pid and _read_cmdline(entry) == wanted:
            found.append(pid)
    return tuple(sorted(found))


def task_processes(
    task: Task,
    *,
    resume: bool,
    proc_root: Path = PROC_ROOT,
) -> tuple[int, ...]:
    worker = worker_argv(task, resume=resume)
    return tuple(
        sorted(
            set(exact_argv_processes(worker, proc_root=proc_root))
            | set(exact_argv_processes(timed_argv(worker), proc_root=proc_root))
        )
    )


def _completion_path(task: Task) -> Path:
    return COMPLETION_ROOT / f"{task.name}.json"


def _completion_is_valid(task: Task) -> bool:
    path = _completion_path(task)
    if not path.exists():
        return False
    try:
        payload = _strict_json(path)
    except ModelDesignWatcherError:
        return False
    latest = task.run_dir / "last_training_state.pth.tar"
    history = task.run_dir / "history.json"
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == COMPLETION_SCHEMA
        and payload.get("task") == task.name
        and payload.get("variant") == task.variant
        and payload.get("run_seed") == task.run_seed
        and payload.get("stopped_after_epoch") == STOP_EPOCH
        and payload.get("configured_total_epochs") == FINAL_CONFIGURED_EPOCH
        and payload.get("latest_sha256") == _sha256_file(latest)
        and payload.get("history_sha256") == _sha256_file(history)
        and payload.get("test_split_accessed") is False
    )


def task_state(task: Task) -> str:
    running = task_processes(task, resume=False) or task_processes(task, resume=True)
    probed = _probe_task_state(task)
    if running:
        return "conflict"
    if probed == "ready":
        return "complete" if _completion_is_valid(task) else "conflict"
    if probed in {"fresh", "resume"}:
        if probed == "fresh" and task.resume_required:
            return "conflict"
        return probed
    return "conflict"


def _parse_gpu_rows(text: str) -> dict[str, GPU]:
    result: dict[str, GPU] = {}
    indexes: set[int] = set()
    buses: set[str] = set()
    for raw_line in text.splitlines():
        fields = tuple(field.strip() for field in raw_line.split(","))
        if len(fields) != 4:
            raise ModelDesignWatcherError("unexpected nvidia-smi GPU row")
        try:
            index, memory = int(fields[0]), int(fields[3])
        except ValueError as exc:
            raise ModelDesignWatcherError("non-integer GPU field") from exc
        uuid, bus = fields[1], fields[2].lower()
        if (
            not _UUID_RE.fullmatch(uuid)
            or uuid in result
            or index in indexes
            or bus in buses
            or index < 0
            or memory < 0
            or not bus
        ):
            raise ModelDesignWatcherError("ambiguous NVIDIA GPU identity")
        result[uuid] = GPU(index, uuid, bus, memory)
        indexes.add(index)
        buses.add(bus)
    if not result:
        raise ModelDesignWatcherError("nvidia-smi returned no GPUs")
    return result


def _parse_compute_uuids(text: str) -> set[str]:
    occupied: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("No running processes"):
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2 or not _UUID_RE.fullmatch(fields[0]):
            raise ModelDesignWatcherError("unexpected compute-app row")
        int(fields[1])
        occupied.add(fields[0])
    return occupied


def sample_idle_gpus() -> dict[str, GPU]:
    gpu_rows = subprocess.run(
        [
            os.fspath(NVIDIA_SMI),
            "--query-gpu=index,uuid,pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    apps = subprocess.run(
        [
            os.fspath(NVIDIA_SMI),
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if gpu_rows.returncode != 0 or apps.returncode != 0:
        raise ModelDesignWatcherError("nvidia-smi query failed")
    gpus = _parse_gpu_rows(gpu_rows.stdout)
    occupied = _parse_compute_uuids(apps.stdout)
    return {
        uuid: gpu
        for uuid, gpu in gpus.items()
        if uuid not in occupied and gpu.memory_used_mib <= GPU_MEMORY_LIMIT_MIB
    }


def _stable_intersection(samples: Sequence[Mapping[str, GPU]]) -> dict[str, GPU]:
    if not samples:
        return {}
    common = set(samples[0])
    for sample in samples[1:]:
        common.intersection_update(sample)
    result = {}
    for uuid in common:
        identities = {(sample[uuid].index, sample[uuid].bus_id) for sample in samples}
        if len(identities) == 1:
            result[uuid] = samples[-1][uuid]
    return result


def _open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ModelDesignWatcherError(f"lock is a symlink: {path}")
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


def claim_idle_gpus(
    *, sleep: Callable[[float], None] = time.sleep
) -> list[tuple[GPU, int]]:
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
        accepted = []
        for gpu, descriptor in locked:
            observed = confirmed.get(gpu.uuid)
            if observed and (observed.index, observed.bus_id) == (
                gpu.index,
                gpu.bus_id,
            ):
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


def launch_task(
    task: Task,
    *,
    resume: bool,
    gpu: GPU,
    gpu_lock_fd: int,
) -> RunningJob | None:
    if not authorization_is_valid():
        os.close(gpu_lock_fd)
        return None
    task_fd = _try_lock(TASK_LOCK_ROOT / f"{task.name}.lock")
    if task_fd is None:
        os.close(gpu_lock_fd)
        return None
    argv = worker_argv(task, resume=resume)
    if task_processes(task, resume=False) or task_processes(task, resume=True):
        os.close(task_fd)
        os.close(gpu_lock_fd)
        return None
    final_idle = sample_idle_gpus()
    observed = final_idle.get(gpu.uuid)
    if observed is None or (observed.index, observed.bus_id) != (gpu.index, gpu.bus_id):
        os.close(task_fd)
        os.close(gpu_lock_fd)
        return None
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_handle = (LOG_ROOT / f"{task.name}.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            list(timed_argv(argv)),
            cwd=PROJECT_ROOT,
            env=_scrubbed_gpu_env(gpu),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            pass_fds=(gpu_lock_fd, task_fd),
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        os.close(task_fd)
        os.close(gpu_lock_fd)
        raise
    return RunningJob(task, process, gpu, gpu_lock_fd, task_fd, log_handle)


def _release_job(job: RunningJob) -> None:
    job.log_handle.close()
    os.close(job.task_lock_fd)
    os.close(job.gpu_lock_fd)


def _atomic_write_completion(task: Task) -> Path:
    COMPLETION_ROOT.mkdir(parents=True, exist_ok=True)
    latest = task.run_dir / "last_training_state.pth.tar"
    history = task.run_dir / "history.json"
    payload = {
        "schema": COMPLETION_SCHEMA,
        "task": task.name,
        "variant": task.variant,
        "run_seed": task.run_seed,
        "configured_total_epochs": FINAL_CONFIGURED_EPOCH,
        "stopped_after_epoch": STOP_EPOCH,
        "latest_sha256": _sha256_file(latest),
        "history_sha256": _sha256_file(history),
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "public_test_allowed": False,
        "test_split_accessed": False,
    }
    path = _completion_path(task)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=COMPLETION_ROOT
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def stop_ready_job(job: RunningJob) -> bool:
    """Validate epoch 500, TERM its process group, then validate once more."""

    if job.process.poll() is not None or _probe_task_state(job.task) != "ready":
        return False
    try:
        os.killpg(job.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    try:
        job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(job.process.pid, signal.SIGKILL)
        job.process.wait(timeout=TERM_TIMEOUT_SECONDS)
        raise ModelDesignWatcherError("epoch-500 worker ignored SIGTERM")
    if _probe_task_state(job.task) != "ready":
        raise ModelDesignWatcherError("epoch-500 transaction changed after TERM")
    _atomic_write_completion(job.task)
    if not _completion_is_valid(job.task):
        raise ModelDesignWatcherError("epoch-500 completion marker verification failed")
    return True


def supervise(*, sleep: Callable[[float], None] = time.sleep) -> int:
    if not authorization_is_valid():
        raise ModelDesignWatcherError("formal launch authorization is not frozen")
    running: dict[str, RunningJob] = {}
    while True:
        for name, job in tuple(running.items()):
            if _probe_task_state(job.task) == "ready":
                if not stop_ready_job(job):
                    raise ModelDesignWatcherError("controlled stop precondition failed")
                _release_job(job)
                del running[name]
                continue
            returncode = job.process.poll()
            if returncode is not None:
                _release_job(job)
                del running[name]
                raise ModelDesignWatcherError(
                    f"{name} exited before a clean epoch-500 commit: rc={returncode}"
                )

        states = {task.name: task_state(task) for task in TASKS}
        for name, state_value in states.items():
            _log(f"task={name} state={state_value}")
        if all(value == "complete" for value in states.values()):
            return 0
        if any(value == "conflict" for value in states.values()):
            raise ModelDesignWatcherError("task queue contains a conflict")

        available = claim_idle_gpus(sleep=sleep)
        pending = [
            task
            for task in TASKS
            if states[task.name] in {"fresh", "resume"} and task.name not in running
        ]
        for gpu, gpu_fd in available:
            if not pending:
                os.close(gpu_fd)
                continue
            task = pending.pop(0)
            job = launch_task(
                task,
                resume=states[task.name] == "resume",
                gpu=gpu,
                gpu_lock_fd=gpu_fd,
            )
            if job is None:
                continue
            running[task.name] = job
            _log(
                f"launched {task.name} pid={job.process.pid} "
                f"gpu={gpu.uuid} index={gpu.index} bus={gpu.bus_id}"
            )
        sleep(POLL_SECONDS)


def _require_safe_directory_chain(path: Path) -> None:
    runs = PROJECT_ROOT / "runs"
    if runs.is_symlink() or (runs.exists() and not runs.is_dir()):
        raise ModelDesignWatcherError("repository runs root is unsafe")
    try:
        relative = path.relative_to(runs)
    except ValueError as exc:
        raise ModelDesignWatcherError("watcher state escaped runs") from exc
    current = runs
    for component in relative.parts:
        current /= component
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ModelDesignWatcherError(f"unsafe watcher path: {current}")


def readonly_preflight() -> dict[str, object]:
    """Inspect fixed dependencies, processes, task state, and GPUs without writes."""

    python_target = PYTHON_BIN.resolve(strict=True)
    if (
        python_target != EXPECTED_PYTHON
        or not python_target.is_file()
        or not os.access(python_target, os.X_OK)
    ):
        raise ModelDesignWatcherError("fixed Python target differs")
    for required in (TRAINER, WATCHER_SOURCE, TIME_BIN, NVIDIA_SMI):
        if required.is_symlink() or not required.is_file():
            raise ModelDesignWatcherError(f"fixed dependency differs: {required}")
    for path in (WATCH_ROOT, TASK_LOCK_ROOT, LOG_ROOT, COMPLETION_ROOT, GPU_LOCK_ROOT):
        _require_safe_directory_chain(path)
    states = {task.name: task_state(task) for task in TASKS}
    idle = sample_idle_gpus()
    return {
        "schema": "evisirst_irstd_model_design_screen_readonly_preflight/v1",
        "formal_launch_authorized": authorization_is_valid(),
        "authorization_sha256_frozen": FORMAL_AUTHORIZATION_SHA256 is not None,
        "task_manifest_sha256": _canonical_sha256(task_manifest()),
        "task_states": states,
        "idle_gpu_uuids": sorted(idle),
        "public_test_supported": False,
        "test_split_accessed": False,
        "writes_performed": False,
        "workers_launched": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--formal", action="store_true")
    modes.add_argument("--smoke-commands", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.preflight:
        print(json.dumps(readonly_preflight(), sort_keys=True, indent=2))
        return 0
    if args.smoke_commands:
        payload = {
            task.name: list(smoke_training_argv(task))
            for task in TASKS
            if task.kind == "model_design"
        }
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    if not authorization_is_valid():
        raise ModelDesignWatcherError(
            "formal mode disabled: freeze authorization file and SHA-256 first"
        )
    for path in (WATCH_ROOT, TASK_LOCK_ROOT, LOG_ROOT, COMPLETION_ROOT, GPU_LOCK_ROOT):
        _require_safe_directory_chain(path)
    singleton = _try_lock(WATCHER_LOCK)
    if singleton is None:
        _log("another model-design watcher owns the singleton lock")
        return 0
    try:
        return supervise()
    finally:
        os.close(singleton)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GPU",
    "TASKS",
    "Task",
    "ModelDesignWatcherError",
    "authorization_is_valid",
    "claim_idle_gpus",
    "exact_argv_processes",
    "formal_training_argv",
    "main",
    "make_cleanup_barrier",
    "readonly_preflight",
    "sample_idle_gpus",
    "smoke_training_argv",
    "stop_ready_job",
    "supervise",
    "task_processes",
    "task_state",
    "timed_argv",
    "timed_worker_argv",
    "worker_argv",
]
