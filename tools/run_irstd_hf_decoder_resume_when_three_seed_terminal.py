#!/usr/bin/env python3
"""Resume the fixed HF-decoder pilot after the three-seed gate is terminal.

The long-lived watcher is standard-library-only.  Torch artifact validation is
performed in the fixed project interpreter with every GPU hidden.  The only
GPU command reachable here is the frozen ``--resume`` command below; public
test evaluation is deliberately unreachable.
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
WATCHER_SOURCE = PROJECT_ROOT / "tools/run_irstd_hf_decoder_resume_when_three_seed_terminal.py"
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
EXPECTED_PYTHON = Path("/usr/bin/python3.12")
DATASET_ROOT = Path("/home/ly/SCTransNet_main/datasets")
SPLIT_ROOT = PROJECT_ROOT / "splits/v2"
HF_TRAINER = PROJECT_ROOT / "train_irstd_hf_decoder_v1.py"
THREE_SEED_GATE = PROJECT_ROOT / "run_irstd_complete_target_three_seed_gate.py"
TIME_BIN = Path("/usr/bin/time")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
RUN_DIR = (
    PROJECT_ROOT / "runs/irstd_performance/hf_decoder_v1/formal/"
    "IRSTD-1K/binary/run_seed_1446202191"
)
WATCH_ROOT = PROJECT_ROOT / "runs/irstd_performance/hf_decoder_v1/resume_watcher"
WATCHER_LOCK = WATCH_ROOT / "watcher.lock"
TASK_LOCK = WATCH_ROOT / "task.lock"
LOG_PATH = WATCH_ROOT / "resume.log"
GPU_LOCK_ROOT = PROJECT_ROOT / "runs/.gpu_locks"
PROC_ROOT = Path("/proc")

EXPECTED_RESUME_EPOCH = 780
FINAL_EPOCH = 1000
POLL_SECONDS = 30
GPU_CONFIRM_SECONDS = 10
GPU_MEMORY_LIMIT_MIB = 1024
INITIAL_RETRY_SECONDS = 60
MAX_RETRY_SECONDS = 600
PROBE_PREFIX = "EVISIRST_HF_RESUME_WATCHER:"
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class HFResumeWatcherError(RuntimeError):
    """The fixed HF resume cannot be supervised safely."""


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    bus_id: str
    memory_used_mib: int


def resume_argv() -> tuple[str, ...]:
    return (
        os.fspath(PYTHON_BIN), os.fspath(HF_TRAINER),
        "--dataset-root", os.fspath(DATASET_ROOT),
        "--split-root", os.fspath(SPLIT_ROOT),
        "--dataset", "IRSTD-1K", "--target-mode", "binary",
        "--architecture-seed", "42", "--run-seed", "1446202191",
        "--device", "cuda:0", "--epochs", "1000",
        "--warmup-epochs", "10", "--allow-sample-level-fallback", "--resume",
    )


def timed_argv(argv: Sequence[str]) -> tuple[str, ...]:
    return os.fspath(TIME_BIN), "-v", *tuple(argv)


# This program is intentionally executed only in the fixed project venv with
# CUDA hidden.  It never writes: in particular it does not call the runner's
# mutating _load_resume_state or _finalize_completed helpers.
_PROBE_PROGRAM = r'''
import hashlib, json, math, os, sys
from pathlib import Path

PREFIX = "EVISIRST_HF_RESUME_WATCHER:"
ROOT = Path("/home/ly/EviSIRST_main")
RUN = ROOT / "runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/binary/run_seed_1446202191"

def emit(token):
    print(PREFIX + token, flush=True)

def fail(message):
    raise RuntimeError(message)

def no_link(path):
    try:
        st = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        fail("symlink artifact")
    return True

def regular(path):
    try:
        st = path.lstat()
    except FileNotFoundError:
        fail("missing artifact")
    if not stat.S_ISREG(st.st_mode):
        fail("artifact is not regular")

def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            fail("duplicate JSON key")
        out[key] = value
    return out

def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        fail("non-finite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "test_split_accessed" and item is not False:
                fail("test split access disclosure differs")
            if key in {"public_test_supported", "public_test_allowed"} and item is not False:
                fail("public test disclosure differs")
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)

def strict_json(path):
    regular(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: fail("non-finite JSON constant"),
        object_pairs_hook=unique,
    )
    finite_tree(value)
    return value

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def args_for(r):
    return r.parse_args([
        "--dataset-root", "/home/ly/SCTransNet_main/datasets",
        "--split-root", "/home/ly/EviSIRST_main/splits/v2",
        "--dataset", "IRSTD-1K", "--target-mode", "binary",
        "--architecture-seed", "42", "--run-seed", "1446202191",
        "--device", "cuda:0", "--epochs", "1000",
        "--warmup-epochs", "10", "--allow-sample-level-fallback", "--resume",
    ])

def validate_latest(r, torch, args, paths, *, minimum_epoch, maximum_epoch):
    latest_path = paths["latest"]
    history_path = paths["history"]
    candidate_dir = paths["candidate_dir"]
    regular(latest_path); regular(history_path)
    regular_dir = candidate_dir.lstat()
    if not stat.S_ISDIR(regular_dir.st_mode) or stat.S_ISLNK(regular_dir.st_mode):
        fail("candidate directory differs")
    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
    identity = r._current_identity_for_args(args)
    latest_keys = {
        "schema", "model", "dataset", "epoch", "run_identity", "state_dict",
        "optimizer", "training_history", "validation_history",
        "candidate_artifacts", "rng", "test_split_accessed",
        "architecture_variant", "experiment_schema", "public_test_supported",
        "selection_margin_raw", "selection_window_applied",
    }
    if (not isinstance(latest, dict) or set(latest) != latest_keys
            or latest.get("schema") != r.TRAINING_SCHEMA
            or latest.get("run_identity") != identity):
        fail("resume run identity differs")
    epoch = latest.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not minimum_epoch <= epoch <= maximum_epoch:
        fail("resume epoch differs")
    if latest.get("test_split_accessed") is not False:
        fail("latest test disclosure differs")
    finite_tree(latest)
    model, _ = r._variant_initialize_evisirst("IRSTD-1K", seed=42, training=True)
    r._validate_state_dict(latest.get("state_dict"), model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    r.r1._validate_and_load_adam_optimizer_state(
        optimizer_state=latest.get("optimizer"), model=model, optimizer=optimizer,
        identity=identity, completed_epoch=epoch, total_epochs=1000,
    )
    training, validation = r.r1._validate_resume_history(
        completed_epoch=epoch, interval=1,
        training_history=latest.get("training_history"),
        validation_history=latest.get("validation_history"),
    )
    training_keys = {"epoch", "mean_train_loss", "learning_rate", "processed_samples"}
    validation_keys = {"epoch", "data_role", "mIoU", "Fa", "Pd", "evaluation_head", "metrics"}
    metric_keys = {
        "miou", "niou", "pixel_precision", "pixel_recall", "pixel_f1",
        "pd", "fa", "tiny_pd", "validation_loss", "valid_pixel_count",
        "target_count", "matched_target_count", "tiny_target_count",
        "matched_tiny_target_count", "predicted_object_count",
        "unmatched_predicted_object_count", "false_objects_per_image",
    }
    if any(set(record) != training_keys for record in training):
        fail("training record keys differ")
    if any(set(record) != validation_keys or not isinstance(record.get("metrics"), dict)
           or set(record["metrics"]) != metric_keys for record in validation):
        fail("validation record keys differ")
    provenance = r.zero_selection.select_checkpoints(
        validation, primary_role=r.zero_selection.PRIMARY_ROLE, margin=None
    )
    frontier = tuple(r.zero_selection.retention_frontier_epochs(validation, margin=None))
    artifacts = latest.get("candidate_artifacts")
    if not isinstance(artifacts, dict) or tuple(sorted(artifacts)) != frontier:
        fail("candidate frontier map differs")
    by_epoch = {record["epoch"]: record for record in validation}
    expected_names = set()
    for candidate_epoch in frontier:
        metadata = artifacts[candidate_epoch]
        if not isinstance(metadata, dict): fail("candidate metadata differs")
        name = "epoch_%04d.pth.tar" % candidate_epoch
        path = candidate_dir / name
        expected_names.add(name); regular(path)
        if metadata != {"relative_path": "candidates/" + name, "file_sha256": sha(path)}:
            fail("candidate binding differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        candidate_keys = {
            "schema", "model", "dataset", "epoch", "run_identity",
            "validation_record", "state_dict", "test_split_accessed",
            "architecture_variant", "experiment_schema", "public_test_supported",
            "selection_margin_raw", "selection_window_applied",
        }
        if (not isinstance(payload, dict) or set(payload) != candidate_keys
                or payload.get("schema") != r.CANDIDATE_SCHEMA
                or payload.get("run_identity") != identity
                or payload.get("epoch") != candidate_epoch
                or payload.get("validation_record") != by_epoch[candidate_epoch]
                or payload.get("test_split_accessed") is not False):
            fail("candidate payload differs")
        r._validate_state_dict(payload.get("state_dict"), model.state_dict())
    actual_names = set()
    for entry in candidate_dir.iterdir():
        regular(entry); actual_names.add(entry.name)
    if actual_names != expected_names:
        fail("candidate directory is not exact")
    history = strict_json(history_path)
    history_keys = {
        "schema", "data_role", "test_split_accessed", "run_identity",
        "training_history", "validation_history", "retention_frontier_epochs",
        "candidate_artifacts", "architecture_variant", "experiment_schema",
        "public_test_supported", "selection_margin_raw", "selection_window_applied",
    }
    if (not isinstance(history, dict) or set(history) != history_keys
            or history.get("run_identity") != identity
            or history.get("training_history") != training
            or history.get("validation_history") != validation
            or history.get("retention_frontier_epochs") != list(frontier)
            or history.get("candidate_artifacts") != {str(k): v for k, v in artifacts.items()}
            or history.get("test_split_accessed") is not False):
        fail("JSON history differs from latest commit")
    rng = latest.get("rng")
    if (not isinstance(rng, dict)
            or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda", "device_type"}
            or not isinstance(rng.get("numpy"), dict)
            or set(rng["numpy"]) != {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
            or rng.get("device_type") != "cuda"
            or not isinstance(rng.get("torch_cpu"), torch.Tensor)
            or rng["torch_cpu"].dtype != torch.uint8
            or not isinstance(rng.get("torch_cuda"), torch.Tensor)
            or rng["torch_cuda"].dtype != torch.uint8):
        fail("resume RNG state differs")
    return latest, identity, training, validation, provenance, artifacts, model

def validate_final_payload(r, torch, path, *, identity, model):
    regular(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    generic_keys = {
        "schema", "model", "dataset", "checkpoint_role", "epoch", "seed",
        "architecture_seed", "run_seed", "state_dict", "target_mode",
        "normalization", "normalization_provenance", "training",
        "training_identity_sha256", "split_provenance", "split_seed",
        "split_manifest_sha256", "data_tree_sha256", "data_tree_verified",
        "selection_provenance", "source_selection", "selection_is_optimistic",
        "optimistic", "test_split_accessed", "model_metadata", "smoke",
        "experiment_schema", "experiment_status", "architecture_variant",
        "state_contract", "selection_margin_raw", "selection_window_applied",
        "promotion_gate", "public_test_supported",
    }
    role_keys = generic_keys | {
        "selection_role", "selected_validation_record",
        "selected_validation_record_sha256", "selected_complete_key",
        "selected_complete_key_sha256", "source_candidate",
    }
    expected_keys = role_keys if isinstance(payload, dict) and "selection_role" in payload else generic_keys
    if (not isinstance(payload, dict) or set(payload) != expected_keys
            or payload.get("schema") != r.CHECKPOINT_SCHEMA
            or payload.get("training") != identity
            or payload.get("dataset") != "IRSTD-1K"
            or payload.get("run_seed") != 1446202191
            or payload.get("architecture_seed") != 42
            or payload.get("training_identity_sha256") != identity["identity_sha256"]
            or payload.get("selection_margin_raw") is not None
            or payload.get("selection_window_applied") is not False
            or payload.get("selection_is_optimistic") is not False
            or payload.get("optimistic") is not False
            or payload.get("smoke") is not False
            or payload.get("test_split_accessed") is not False
            or payload.get("public_test_supported") is not False):
        fail("final checkpoint identity differs")
    finite_tree(payload)
    r._validate_state_dict(payload.get("state_dict"), model.state_dict())
    return payload

action = sys.argv[1]
if action == "gate":
    import run_irstd_complete_target_three_seed_gate as gate
    result = gate.validate_existing_result()
    if (result.get("status") != "complete"
            or result.get("decision", {}).get("result") not in {"PASS", "FAIL"}
            or result.get("test_split_accessed") is not False
            or result.get("public_test_allowed") is not False
            or result.get("decision", {}).get("public_test_allowed") is not False):
        fail("aggregate gate is not terminal and validation-only")
    emit("TERMINAL")
elif action == "state":
    import stat, torch
    import train_irstd_hf_decoder_v1 as r
    args = args_for(r); paths = r.resolve_run_paths(args)
    summary = paths["summary"]
    generic = paths["final"]
    primary = paths["best_mIoU_final"]
    secondary = paths["best_Pd_final"]
    if not summary.exists():
        if summary.is_symlink() or generic.exists() or generic.is_symlink() \
                or primary.exists() or primary.is_symlink() \
                or secondary.exists() or secondary.is_symlink():
            fail("partial transaction contains final artifacts")
        validate_latest(r, torch, args, paths, minimum_epoch=780, maximum_epoch=999)
        emit("RESUME")
    else:
        regular(summary)
        strict_summary = strict_json(summary)
        summary_keys = {
            "schema", "status", "dataset", "checkpoint", "checkpoint_role",
            "selected_epoch", "architecture_seed", "run_seed", "target_mode",
            "split_provenance", "selection", "training_history",
            "validation_history", "candidate_artifacts", "normalization",
            "source_selection", "selection_is_optimistic", "optimistic",
            "test_split_accessed", "smoke", "elapsed_seconds",
            "experiment_schema", "architecture_variant", "selection_margin_raw",
            "selection_window_applied", "public_test_supported",
            "selected_validation_record", "selected_validation_record_sha256",
            "selection_roles", "promotion_gate",
        }
        if not isinstance(strict_summary, dict): fail("summary is malformed")
        if "role_final_checkpoints" in strict_summary:
            summary_keys |= {"role_final_checkpoints", "primary_checkpoint_role", "legacy_engine_primary_checkpoint"}
        if set(strict_summary) != summary_keys:
            fail("summary keys differ")
        latest, identity, training, validation, provenance, artifacts, model = validate_latest(
            r, torch, args, paths, minimum_epoch=1000, maximum_epoch=1000
        )
        loaded_summary, loaded_latest, loaded_identity = r._load_completed_transaction(args, paths)
        if (strict_summary != loaded_summary or loaded_identity != identity
                or loaded_latest.get("epoch") != latest.get("epoch")
                or loaded_latest.get("run_identity") != latest.get("run_identity")
                or loaded_latest.get("training_history") != latest.get("training_history")
                or loaded_latest.get("validation_history") != latest.get("validation_history")
                or loaded_latest.get("candidate_artifacts") != latest.get("candidate_artifacts")):
            fail("completed transaction binding differs")
        role_map = loaded_summary.get("role_final_checkpoints")
        if role_map is None:
            template = generic if generic.exists() else primary
            if template.is_symlink() or not template.is_file():
                fail("finalization template is unavailable")
            validate_final_payload(r, torch, template, identity=identity, model=model)
            for path in (primary, secondary):
                if path.exists(): validate_final_payload(r, torch, path, identity=identity, model=model)
                elif path.is_symlink(): fail("broken role-final symlink")
            emit("FINALIZE")
        else:
            template = primary
            primary_payload = validate_final_payload(r, torch, primary, identity=identity, model=model)
            expected_state = r._validate_state_dict(primary_payload.get("state_dict"), model.state_dict())
            normalized = r._validate_committed_role_map(
                run_dir=RUN, role_map=role_map, identity=identity,
                history=validation, expected_state=expected_state,
            )
            for role, path in ((r.zero_selection.PRIMARY_ROLE, primary),
                               (r.zero_selection.SECONDARY_ROLE, secondary)):
                payload = validate_final_payload(r, torch, path, identity=identity, model=model)
                selected = provenance["roles"][role]["selected"]
                source = payload.get("source_candidate")
                expected_artifact = artifacts.get(selected["epoch"])
                expected_source = (
                    {"relative_path": expected_artifact["relative_path"],
                     "sha256": expected_artifact["file_sha256"]}
                    if isinstance(expected_artifact, dict) else None
                )
                if source != expected_source:
                    fail("role source candidate binding differs")
                candidate = RUN / source["relative_path"]
                regular(candidate)
                try:
                    candidate.resolve(strict=True).relative_to(RUN.resolve(strict=True))
                except ValueError:
                    fail("role source candidate escapes run directory")
                if source.get("sha256") != sha(candidate): fail("role source SHA differs")
                cp = torch.load(candidate, map_location="cpu", weights_only=True)
                if (payload.get("selected_validation_record") != cp.get("validation_record")
                        or payload.get("epoch") != selected["epoch"]):
                    fail("role selected record differs")
                role_state = r._validate_state_dict(payload.get("state_dict"), model.state_dict())
                candidate_state = r._validate_state_dict(cp.get("state_dict"), model.state_dict())
                if any(not torch.equal(role_state[k], candidate_state[k]) for k in role_state):
                    fail("role state differs from source candidate")
            if generic.exists() or generic.is_symlink():
                regular(generic); emit("FINALIZE")
            else:
                emit("COMPLETE")
else:
    fail("unknown probe action")
'''


def _probe_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    return env


def _run_probe(action: str) -> str | None:
    completed = subprocess.run(
        [os.fspath(PYTHON_BIN), "-B", "-c", _PROBE_PROGRAM, action],
        cwd=PROJECT_ROOT, env=_probe_environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=600,
    )
    if completed.returncode != 0:
        return None
    tokens = [
        line.removeprefix(PROBE_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(PROBE_PREFIX)
    ]
    return tokens[0] if len(tokens) == 1 else None


def terminal_gate_is_valid() -> bool:
    return _run_probe("gate") == "TERMINAL"


def run_state() -> str:
    token = _run_probe("state")
    return token if token in {"RESUME", "FINALIZE", "COMPLETE"} else "CONFLICT"


def _read_cmdline(pid_dir: Path) -> tuple[str, ...] | None:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except (OSError, PermissionError):
        return None
    try:
        return tuple(part.decode("utf-8", errors="strict") for part in raw.rstrip(b"\0").split(b"\0")) if raw else ()
    except UnicodeDecodeError:
        return None


def exact_argv_processes(expected: Sequence[str], *, proc_root: Path = PROC_ROOT) -> tuple[int, ...]:
    wanted = tuple(expected)
    found: list[int] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise HFResumeWatcherError("cannot enumerate /proc") from exc
    for entry in entries:
        if entry.name.isascii() and entry.name.isdecimal() and _read_cmdline(entry) == wanted:
            found.append(int(entry.name))
    return tuple(sorted(found))


def work_processes() -> tuple[int, ...]:
    return tuple(sorted(
        set(exact_argv_processes(resume_argv(), proc_root=PROC_ROOT))
        | set(exact_argv_processes(timed_argv(resume_argv()), proc_root=PROC_ROOT))
    ))


def _parse_gpu_rows(text: str) -> dict[str, GPU]:
    output: dict[str, GPU] = {}
    indexes: set[int] = set(); buses: set[str] = set()
    for raw in text.splitlines():
        fields = tuple(field.strip() for field in raw.split(","))
        if len(fields) != 4:
            raise HFResumeWatcherError("unexpected NVIDIA GPU row")
        try:
            index, memory = int(fields[0]), int(fields[3])
        except ValueError as exc:
            raise HFResumeWatcherError("non-integer NVIDIA GPU field") from exc
        uuid, bus = fields[1], fields[2].lower()
        if (index < 0 or memory < 0 or not bus or not _UUID_RE.fullmatch(uuid)
                or uuid in output or index in indexes or bus in buses):
            raise HFResumeWatcherError("ambiguous NVIDIA GPU identity")
        output[uuid] = GPU(index, uuid, bus, memory); indexes.add(index); buses.add(bus)
    if not output:
        raise HFResumeWatcherError("no NVIDIA GPUs returned")
    return output


def _parse_compute_uuids(text: str) -> set[str]:
    occupied: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("No running processes"):
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2 or not _UUID_RE.fullmatch(fields[0]):
            raise HFResumeWatcherError("unexpected NVIDIA compute row")
        int(fields[1]); occupied.add(fields[0])
    return occupied


def sample_idle_gpus() -> dict[str, GPU]:
    rows = subprocess.run(
        [os.fspath(NVIDIA_SMI), "--query-gpu=index,uuid,pci.bus_id,memory.used", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    apps = subprocess.run(
        [os.fspath(NVIDIA_SMI), "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if rows.returncode or apps.returncode:
        raise HFResumeWatcherError("nvidia-smi query failed")
    gpus = _parse_gpu_rows(rows.stdout); occupied = _parse_compute_uuids(apps.stdout)
    return {uuid: gpu for uuid, gpu in gpus.items() if uuid not in occupied and gpu.memory_used_mib <= GPU_MEMORY_LIMIT_MIB}


def _open_lock(path: Path, *, inheritable: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise HFResumeWatcherError(f"lock is a symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
    if not inheritable and hasattr(os, "O_CLOEXEC"): flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    os.set_inheritable(fd, inheritable)
    return fd


def _try_lock(path: Path, *, inheritable: bool = False) -> int | None:
    fd = _open_lock(path, inheritable=inheritable)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd); return None
    return fd


def _safe_directory_chain(path: Path) -> None:
    runs = PROJECT_ROOT / "runs"
    if runs.is_symlink() or (runs.exists() and not runs.is_dir()):
        raise HFResumeWatcherError("runs root is unsafe")
    try: relative = path.relative_to(runs)
    except ValueError as exc: raise HFResumeWatcherError("state path escaped runs") from exc
    current = runs
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise HFResumeWatcherError(f"unsafe state path: {current}")


def claim_one_idle_gpu(*, sleep: Callable[[float], None]) -> tuple[GPU, int] | None:
    first = sample_idle_gpus(); sleep(GPU_CONFIRM_SECONDS); second = sample_idle_gpus()
    for uuid in sorted(set(first) & set(second), key=lambda value: second[value].index):
        a, b = first[uuid], second[uuid]
        if (a.index, a.bus_id) != (b.index, b.bus_id): continue
        fd = _try_lock(GPU_LOCK_ROOT / f"{uuid}.lock", inheritable=True)
        if fd is None:
            continue
        # Reobserve while the cooperative UUID lock is held.  launch_once()
        # performs the fourth observation immediately before Popen.
        observed = sample_idle_gpus().get(uuid)
        if observed is not None and (observed.index, observed.bus_id) == (b.index, b.bus_id):
            return observed, fd
        os.close(fd)
    return None


def _worker_environment(gpu: GPU) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"; env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    return env


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _open_log() -> object:
    WATCH_ROOT.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(LOG_PATH, flags, 0o600)
    except OSError as exc:
        raise HFResumeWatcherError("HF resume log is unsafe") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise HFResumeWatcherError("HF resume log is not regular")
        return os.fdopen(descriptor, "ab", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def launch_once(gpu: GPU, gpu_fd: int) -> tuple[subprocess.Popen[bytes], int, object] | None:
    task_fd = _try_lock(TASK_LOCK, inheritable=True)
    if task_fd is None:
        os.close(gpu_fd); return None
    try:
        # Launch-boundary order matters: fresh gate, exact process, then final
        # physical GPU identity/idle observation while both locks are held.
        if not terminal_gate_is_valid() or work_processes() or run_state() not in {"RESUME", "FINALIZE"}:
            os.close(task_fd); os.close(gpu_fd); return None
        observed = sample_idle_gpus().get(gpu.uuid)
        if observed is None or (observed.index, observed.bus_id) != (gpu.index, gpu.bus_id):
            os.close(task_fd); os.close(gpu_fd); return None
        log_handle = _open_log()
        try:
            process = subprocess.Popen(
                list(timed_argv(resume_argv())), cwd=PROJECT_ROOT,
                env=_worker_environment(gpu), stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=subprocess.STDOUT,
                pass_fds=(gpu_fd, task_fd), start_new_session=True,
            )
        except BaseException:
            log_handle.close(); raise
        _log(f"launched fixed HF resume pid={process.pid} gpu={gpu.uuid} index={gpu.index} bus={gpu.bus_id}")
        return process, task_fd, log_handle
    except BaseException:
        try: os.close(task_fd)
        finally: os.close(gpu_fd)
        raise


def supervise(*, sleep: Callable[[float], None] = time.sleep) -> int:
    failures = 0
    while True:
        if not terminal_gate_is_valid():
            sleep(POLL_SECONDS); continue
        if work_processes():
            sleep(POLL_SECONDS); continue
        state = run_state()
        if state == "COMPLETE":
            _log("strict HF completion probe passed"); return 0
        if state == "CONFLICT":
            raise HFResumeWatcherError("HF transaction is not safely resumable")
        claim = claim_one_idle_gpu(sleep=sleep)
        if claim is None:
            sleep(POLL_SECONDS); continue
        gpu, gpu_fd = claim
        launched = launch_once(gpu, gpu_fd)
        if launched is None:
            sleep(POLL_SECONDS); continue
        process, task_fd, log_handle = launched
        while process.poll() is None:
            sleep(POLL_SECONDS)
        returncode = process.returncode
        log_handle.close(); os.close(task_fd); os.close(gpu_fd)
        if returncode == 0:
            failures = 0
        else:
            failures += 1
            delay = min(INITIAL_RETRY_SECONDS * (2 ** (failures - 1)), MAX_RETRY_SECONDS)
            _log(f"HF resume exited rc={returncode}; strict revalidation after {delay}s")
            sleep(delay)


def validate_fixed_dependencies() -> None:
    try: resolved = PYTHON_BIN.resolve(strict=True)
    except OSError as exc: raise HFResumeWatcherError("fixed Python is unavailable") from exc
    if resolved != EXPECTED_PYTHON or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise HFResumeWatcherError("fixed Python target differs")
    for path in (WATCHER_SOURCE, HF_TRAINER, THREE_SEED_GATE, TIME_BIN, NVIDIA_SMI):
        if path.is_symlink() or not path.is_file():
            raise HFResumeWatcherError(f"fixed dependency differs: {path}")
    if Path(__file__).resolve(strict=True) != WATCHER_SOURCE:
        raise HFResumeWatcherError("watcher source path differs")
    for root in (RUN_DIR, WATCH_ROOT, GPU_LOCK_ROOT):
        _safe_directory_chain(root)


def main(argv: Sequence[str] | None = None) -> int:
    if list(sys.argv[1:] if argv is None else argv):
        raise HFResumeWatcherError("fixed HF resume watcher accepts no arguments")
    validate_fixed_dependencies()
    singleton = _try_lock(WATCHER_LOCK)
    if singleton is None:
        _log("another HF resume watcher owns the singleton lock"); return 0
    try: return supervise()
    finally: os.close(singleton)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GPU", "HFResumeWatcherError", "claim_one_idle_gpu", "exact_argv_processes",
    "launch_once", "main", "resume_argv", "run_state", "sample_idle_gpus",
    "supervise", "terminal_gate_is_valid", "timed_argv", "validate_fixed_dependencies",
    "work_processes",
]
