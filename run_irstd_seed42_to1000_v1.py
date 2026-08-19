#!/usr/bin/env python3
"""Authorized Seed-42 continuation for frozen PSBFR and CP-HF-S2 runs.

This additive post-interim amendment preserves each existing run identity and
1000-epoch schedule.  Its results are exploratory and cannot establish a
stable-over-baseline claim without later fresh paired multi-seed evidence.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
ADAPTER_PATH = PROJECT_ROOT / "run_irstd_seed42_to1000_v1.py"
STAGE1_WATCH_ROOT = PROJECT_ROOT / "runs/irstd_model_design/ab_legacy_screen_watcher_v1"
START_SNAPSHOT_ROOT = PROJECT_ROOT / "runs/irstd_model_design/seed42_to1000_v1/start_snapshot"
SNAPSHOT_SCHEMA = "evisirst_seed42_to1000_start_snapshot/v1"
INSPECTION_SCHEMA = "evisirst_seed42_to1000_state_inspection/v1"
FORMAL_AUTHORIZATION_SHA256: str | None = "497f5409e7481eb7df76c8b5baa9c02de0daffead6f9a734d785533d84ae5a4a"
START_EPOCH = 500
FINAL_EPOCH = 1000
_PREFIX = "EVISIRST_SEED42_TO1000_INSPECT:"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

POST_INTERIM_DISCLOSURE = {
    "post_interim_protocol_amendment": True,
    "epoch500_results_disclosed_before_continuation": True,
    "user_requested_both_routes_to1000": True,
    "exploratory_seed42_full_schedule": True,
    "old_stage1_gate_not_used_for_claim": True,
    "stable_over_baseline_claim_eligible": False,
    "lockbox_accessed": False,
    "public_test_allowed": False,
    "test_split_accessed": False,
}


class Seed42ContinuationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RouteSpec:
    route: str
    task: str
    kind: str
    variant: str
    run_dir: Path
    ledger_sha256: str
    latest_sha256: str
    history_sha256: str
    candidates: Mapping[str, str]

    @property
    def ledger(self) -> Path:
        return STAGE1_WATCH_ROOT / "epoch500_commits" / f"{self.task}.json"

    @property
    def snapshot_dir(self) -> Path:
        return START_SNAPSHOT_ROOT / self.route


def _run_dir(variant: str) -> Path:
    return PROJECT_ROOT / "runs/irstd_model_design" / variant / "legacy_screen/formal/IRSTD-1K/binary/run_seed_42"


ROUTE_SPECS = {
    "psbfr": RouteSpec(
        "psbfr", "psbfr_s42", "psbfr", "psbfr_v1", _run_dir("psbfr_v1"),
        "17b9d47591897b8be18b52d0dd6f5ca336f1a50736da3a3b8935098a9f4919a7",
        "f53792d8fefbcc5fcc41dd4df71047465a6211e4dfaa87e2f063028e106098a4",
        "8bda82ecd091b4c82736475221f9a06c2f4b71cb96a7650b66dde3f319068cab",
        {
            "candidates/epoch_0253.pth.tar": "dfc044483db522e876b1e3e004d35b8913e29ccaa726f4cc8ed45b26ed108a06",
            "candidates/epoch_0411.pth.tar": "bdf016bc6a5c73d903d5966af87a995d8e878995e5d21d26085edbc40cbd0123",
        },
    ),
    "cp_hf_s2": RouteSpec(
        "cp_hf_s2", "cp_hf_s2_s42", "cp_hf_s2", "cp_hf_s2_v1", _run_dir("cp_hf_s2_v1"),
        "5d0253d906bbab05d095106a7f1e59906386a33d3b8c3c532e065e72cf3e69f3",
        "6708e425eaec1fa05bb1fa850339d507f5097f509114be10845c38f717e325ed",
        "05b1015bac4abf54cef760bec765c70d71b282810d5c2b6a6e5e4dce43caabcd",
        {
            "candidates/epoch_0395.pth.tar": "743f59d84a25f3da2b61a5cf946e9f50ba1c12243f7ef39a6ec9384ec5ec985f",
            "candidates/epoch_0412.pth.tar": "95f35e5cf0a616477e0365c3c2d2194125d5af6b1bae834bac71db3eef07a5a2",
        },
    ),
}
ROUTES = tuple(ROUTE_SPECS)


def _spec(route: str) -> RouteSpec:
    try:
        return ROUTE_SPECS[route]
    except KeyError as exc:
        raise Seed42ContinuationError("unsupported route") from exc


def _sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Seed42ContinuationError("artifact is not regular")
        with os.fdopen(fd, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise Seed42ContinuationError("non-canonical JSON") from exc


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Seed42ContinuationError("JSON artifact is unsafe")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Seed42ContinuationError("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda token: (_ for _ in ()).throw(Seed42ContinuationError(token)))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Seed42ContinuationError("malformed JSON") from exc
    if not isinstance(value, dict):
        raise Seed42ContinuationError("JSON root differs")
    _canonical(value)
    return value


def _authorization() -> tuple[str, str]:
    from tools import run_irstd_seed42_to1000_when_idle as watcher
    authorization = getattr(watcher, "FORMAL_AUTHORIZATION_SHA256", None)
    if (not isinstance(authorization, str) or not _SHA_RE.fullmatch(authorization)
            or FORMAL_AUTHORIZATION_SHA256 != authorization or not watcher.authorization_is_valid()):
        raise Seed42ContinuationError("continuation authorization is invalid")
    return authorization, _sha256(Path(watcher.__file__).resolve(strict=True))


def _ledger(spec: RouteSpec) -> dict[str, Any]:
    if _sha256(spec.ledger) != spec.ledger_sha256:
        raise Seed42ContinuationError("Stage-1 ledger SHA differs")
    value = _strict_json(spec.ledger)
    required = {
        "schema": "evisirst_irstd_ab_model_screen_epoch500_stop/v1", "task": spec.task,
        "kind": spec.kind, "variant": spec.variant, "run_seed": 42,
        "configured_total_epochs": 1000, "stopped_after_epoch": 500,
        "latest_sha256": spec.latest_sha256, "history_sha256": spec.history_sha256,
        "candidate_cleanup_validated": True, "lockbox_accessed": False,
        "public_test_allowed": False, "test_split_accessed": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise Seed42ContinuationError("Stage-1 ledger payload differs")
    return value


def expected_snapshot_manifest(route: str) -> dict[str, Any]:
    spec = _spec(route)
    ledger = _ledger(spec)
    authorization, watcher_sha = _authorization()
    result: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA, "route": route, "variant": spec.variant,
        "architecture_seed": 42, "run_seed": 42, "committed_epoch": 500,
        "configured_total_epochs": 1000,
        "copy_method": "same_filesystem_hardlink_no_follow_no_clobber",
        "source_run_dir": spec.run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "stage1_completion_ledger": {
            "relative_path": spec.ledger.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": spec.ledger_sha256,
            "canonical_payload_sha256": hashlib.sha256(_canonical(ledger)).hexdigest(),
        },
        "files": {"last_training_state.pth.tar": spec.latest_sha256,
                  "validation_history.json": spec.history_sha256, **dict(spec.candidates)},
        "continuation_adapter_sha256": _sha256(ADAPTER_PATH),
        "continuation_authorization_sha256": authorization,
        "continuation_watcher_source_sha256": watcher_sha,
        **POST_INTERIM_DISCLOSURE,
    }
    return result


def _safe_snapshot_dir(path: Path, create: bool) -> None:
    root = PROJECT_ROOT / "runs"
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise Seed42ContinuationError("snapshot escaped runs") from exc
    current = root
    for part in parts:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise Seed42ContinuationError("unsafe snapshot directory")
        if not current.exists():
            if not create:
                raise Seed42ContinuationError("snapshot directory missing")
            current.mkdir(mode=0o700)


def _link(source: Path, destination: Path, digest: str) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (source.parent.is_symlink() or destination.parent.is_symlink()
            or source.is_symlink() or not source.is_file() or destination.is_symlink()
            or not destination.parent.is_dir()):
        raise Seed42ContinuationError("unsafe snapshot link endpoint")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_dir_fd = os.open(source.parent, directory_flags)
    destination_dir_fd = os.open(destination.parent, directory_flags)
    try:
        source_stat = os.stat(source.name, dir_fd=source_dir_fd, follow_symlinks=False)
        if not stat.S_ISREG(source_stat.st_mode) or _sha256(source) != digest:
            raise Seed42ContinuationError("snapshot source SHA differs")
        if destination.exists():
            linked = os.stat(destination.name, dir_fd=destination_dir_fd, follow_symlinks=False)
            if (not stat.S_ISREG(linked.st_mode) or _sha256(destination) != digest
                    or (linked.st_dev, linked.st_ino) != (source_stat.st_dev, source_stat.st_ino)):
                raise Seed42ContinuationError("snapshot no-clobber conflict")
            return
        try:
            os.link(source.name, destination.name, src_dir_fd=source_dir_fd,
                    dst_dir_fd=destination_dir_fd, follow_symlinks=False)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise Seed42ContinuationError("snapshot no-clobber conflict") from exc
            raise
        linked = os.stat(destination.name, dir_fd=destination_dir_fd, follow_symlinks=False)
        if (not stat.S_ISREG(linked.st_mode)
                or (linked.st_dev, linked.st_ino) != (source_stat.st_dev, source_stat.st_ino)
                or _sha256(destination) != digest):
            raise Seed42ContinuationError("hardlink verification failed")
    finally:
        os.close(destination_dir_fd)
        os.close(source_dir_fd)


def _write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical(value) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise Seed42ContinuationError("manifest no-clobber conflict")
        return
    fd, name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path, follow_symlinks=False)
    finally:
        if temp.exists(): temp.unlink()


_SNAPSHOT_MANIFEST_TEMP_RE = re.compile(r"^\.manifest\.[a-z0-9_]{8}\.tmp$")


def _recover_incomplete_snapshot_manifest_temps(
    snapshot_dir: Path, expected_manifest: Mapping[str, Any]
) -> None:
    manifest = snapshot_dir / "manifest.json"
    matches = [
        path
        for path in snapshot_dir.iterdir()
        if _SNAPSHOT_MANIFEST_TEMP_RE.fullmatch(path.name)
    ]
    expected_bytes = _canonical(expected_manifest) + b"\n"
    if manifest.is_symlink():
        raise Seed42ContinuationError("snapshot manifest is a symlink")
    if manifest.exists():
        if not manifest.is_file() or manifest.read_bytes() != expected_bytes:
            raise Seed42ContinuationError("committed snapshot manifest differs")
        if len(matches) > 1:
            raise Seed42ContinuationError("multiple linked manifest temporaries")
        if matches:
            temporary = matches[0]
            observed = temporary.lstat()
            committed = manifest.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(observed.st_mode)
                or (observed.st_dev, observed.st_ino)
                != (committed.st_dev, committed.st_ino)
                or temporary.read_bytes() != expected_bytes
            ):
                raise Seed42ContinuationError(
                    "linked manifest temporary differs from commit"
                )
            temporary.unlink()
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(snapshot_dir, flags)
    try:
        for path in matches:
            observed = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(observed.st_mode):
                raise Seed42ContinuationError(
                    "snapshot manifest temporary is not regular"
                )
            os.unlink(path.name, dir_fd=descriptor)
        if matches:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_start_snapshot(route: str) -> dict[str, Any]:
    spec = _spec(route)
    _safe_snapshot_dir(spec.snapshot_dir, False)
    observed = _strict_json(spec.snapshot_dir / "manifest.json")
    expected = expected_snapshot_manifest(route)
    if observed != expected:
        raise Seed42ContinuationError("snapshot manifest differs")
    if {p.name for p in spec.snapshot_dir.iterdir()} != {
        "manifest.json", "last_training_state.pth.tar", "validation_history.json", "candidates"
    }:
        raise Seed42ContinuationError("snapshot root entries differ")
    candidate_dir = spec.snapshot_dir / "candidates"
    if candidate_dir.is_symlink() or not candidate_dir.is_dir() or {
        p.name for p in candidate_dir.iterdir()
    } != {Path(name).name for name in spec.candidates}:
        raise Seed42ContinuationError("snapshot candidate entries differ")
    for relative, digest in expected["files"].items():
        path = spec.snapshot_dir / relative
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise Seed42ContinuationError("snapshot artifact differs")
    return observed


def _recoverable_history_epoch(committed_epoch: int, history_epoch: int) -> bool:
    return history_epoch in {
        max(0, committed_epoch - 1),
        committed_epoch,
        min(FINAL_EPOCH, committed_epoch + 1),
    }


def _recoverable_extra_candidate(
    committed_epoch: int, candidate_epoch: int, *, has_committed_record: bool
) -> bool:
    return (
        candidate_epoch <= committed_epoch and has_committed_record
    ) or (
        committed_epoch < FINAL_EPOCH
        and candidate_epoch == committed_epoch + 1
    )


def _assert_sidecar_crash_window(
    *,
    committed_epoch: int,
    history_epoch: int,
    extra_candidate_count: int,
    final_artifact_exists: bool,
) -> None:
    if final_artifact_exists and (
        history_epoch != committed_epoch or extra_candidate_count != 0
    ):
        raise Seed42ContinuationError(
            "finalization requires exact committed history and frontier"
        )


def _assert_summary_evidence(
    summary: Mapping[str, Any],
    *,
    training_history: list[Any],
    validation_history: list[Any],
    candidate_artifacts: Mapping[str, Any],
    selected_record: Mapping[str, Any],
    selected_record_sha256: str,
) -> None:
    if (
        summary.get("training_history") != training_history
        or summary.get("validation_history") != validation_history
        or summary.get("candidate_artifacts") != candidate_artifacts
        or summary.get("selected_validation_record") != selected_record
        or summary.get("selected_validation_record_sha256")
        != selected_record_sha256
    ):
        raise Seed42ContinuationError("summary committed evidence differs")


def _classify_transaction_window(
    *,
    committed_epoch: int,
    summary_exists: bool,
    role_map_present: bool,
    generic_final_exists: bool,
    role_file_count: int,
    primary_role_exists: bool,
    recoverable_temp_count: int = 0,
) -> str:
    if type(committed_epoch) is not int or not START_EPOCH <= committed_epoch <= FINAL_EPOCH:
        raise Seed42ContinuationError("committed epoch is outside continuation")
    if type(role_file_count) is not int or not 0 <= role_file_count <= 2:
        raise Seed42ContinuationError("role-file count differs")
    if committed_epoch < FINAL_EPOCH:
        if summary_exists or role_map_present or generic_final_exists or role_file_count:
            raise Seed42ContinuationError("finalization artifact exists before epoch 1000")
        return "epoch500_ready" if committed_epoch == START_EPOCH else "resume_501_999"
    if not summary_exists and role_file_count:
        raise Seed42ContinuationError("role final exists before engine summary")
    if role_map_present and (not summary_exists or role_file_count != 2):
        raise Seed42ContinuationError("committed role map is incomplete")
    if type(recoverable_temp_count) is not int or recoverable_temp_count < 0:
        raise Seed42ContinuationError("recoverable-temp count differs")
    if (
        summary_exists
        and role_map_present
        and not generic_final_exists
        and recoverable_temp_count == 0
    ):
        return "complete"
    if summary_exists and not role_map_present and not (
        generic_final_exists or primary_role_exists
    ):
        raise Seed42ContinuationError("no trusted primary finalization template")
    return "finalize_1000"


def _semantic_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality, including every tensor byte and metadata key."""
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return bool(
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_semantic_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return bool(
            type(left) is type(right)
            and len(left) == len(right)
            and all(_semantic_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _assert_generic_selection_metadata(
    payload: Mapping[str, Any], *, selected_epoch: int, provenance: Mapping[str, Any]
) -> None:
    # selected_validation_record(+SHA) are added only to the final summary and
    # dual-role files, never to the R1 generic final checkpoint.
    if (
        payload.get("epoch") != selected_epoch
        or payload.get("selection_provenance") != provenance
        or "selected_validation_record" in payload
        or "selected_validation_record_sha256" in payload
    ):
        raise Seed42ContinuationError("generic selection metadata differs")


_INSPECTION_PROGRAM = r'''
import hashlib,json,math,re,stat,sys
from contextlib import ExitStack
from pathlib import Path
import torch
from run_irstd_seed42_to1000_v1 import _assert_generic_selection_metadata,_assert_sidecar_crash_window,_assert_summary_evidence,_classify_transaction_window,_recoverable_extra_candidate,_recoverable_history_epoch
PREFIX="EVISIRST_SEED42_TO1000_INSPECT:"
route=sys.argv[1]
def fail(msg): raise RuntimeError(msg)
def emit(v): print(PREFIX+json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False),flush=True)
def regular(p):
 st=p.lstat()
 if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode): fail("not regular")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def finite(v):
 if isinstance(v,float) and not math.isfinite(v): fail("nonfinite")
 if isinstance(v,torch.Tensor) and v.is_floating_point() and not bool(torch.isfinite(v).all()): fail("nonfinite tensor")
 if isinstance(v,dict):
  for k,x in v.items():
   if k in {"lockbox_accessed","test_split_accessed","public_test_allowed","public_test_supported","test_index_opened"} and x is not False: fail("forbidden disclosure")
   finite(x)
 elif isinstance(v,(list,tuple)):
  for x in v: finite(x)
def unique(pairs):
 out={}
 for k,v in pairs:
  if k in out: fail("duplicate key")
  out[k]=v
 return out
with ExitStack() as stack:
 if route=="psbfr":
  import train_irstd_model_design_screen_v1 as f
  a=f.parse_args(["--dataset-root","/home/ly/SCTransNet_main/datasets","--split-root","/home/ly/EviSIRST_main/splits/v2","--variant","psbfr_v1","--dataset","IRSTD-1K","--target-mode","binary","--architecture-seed","42","--run-seed","42","--device","cuda:0","--epochs","1000","--warmup-epochs","10","--allow-sample-level-fallback","--resume"])
  stack.enter_context(f._hf_adapter(a)); e=f.r1; t=f.hf_transaction; s=f.zero_selection
  p=f.resolve_run_paths(a); identity=t._current_identity_for_args(a); model,_=f._variant_initialize_evisirst("IRSTD-1K",seed=42,training=True)
  validate=f._validate_state_dict; training_schema=f.TRAINING_SCHEMA; candidate_schema=f.CANDIDATE_SCHEMA; history_schema=f.HISTORY_SCHEMA; checkpoint_schema=f.CHECKPOINT_SCHEMA
 elif route=="cp_hf_s2":
  import train_irstd_cp_hf_s2_legacy_screen_v1 as f
  a=f.parse_args(["--dataset-root","/home/ly/SCTransNet_main/datasets","--run-seed","42","--device","cuda:0","--epochs","1000","--warmup-epochs","10","--resume"])
  shared=stack.enter_context(f._screen_transaction_adapter(a)); stack.enter_context(shared._hf_adapter(a)); e=shared.r1; t=shared.hf_transaction; s=shared.zero_selection
  p=shared.resolve_run_paths(a); identity=t._current_identity_for_args(a); model,_=shared._variant_initialize_evisirst("IRSTD-1K",seed=42,training=True)
  validate=shared._validate_state_dict; training_schema=f.TRAINING_SCHEMA; candidate_schema=f.CANDIDATE_SCHEMA; history_schema=f.HISTORY_SCHEMA; checkpoint_schema=f.CHECKPOINT_SCHEMA
 else: fail("route")
 run=p["run_dir"]; latest=p["latest"]; hp=p["history"]; cd=p["candidate_dir"]
 summary=p["summary"]; generic=p["final"]; roles=(p["best_mIoU_final"],p["best_Pd_final"])
 final_artifact_exists=summary.exists() or summary.is_symlink() or generic.exists() or generic.is_symlink() or any(x.exists() or x.is_symlink() for x in roles)
 if run.is_symlink() or not run.is_dir(): fail("run dir")
 allowed={".model_design_screen_v1.lock",latest.name,hp.name,cd.name,p["summary"].name,p["final"].name,p["best_mIoU_final"].name,p["best_Pd_final"].name}
 token=r"[A-Za-z0-9_]{6,32}"
 targets=(latest.name,hp.name,p["summary"].name,p["final"].name,p["best_mIoU_final"].name,p["best_Pd_final"].name)
 run_temp_re=re.compile(r"^(?:\.expected_generic\."+token+r"\.pth\.tar|\.(?:"+"|".join(re.escape(x) for x in targets)+r")\."+token+r"\.tmp|\.\.expected_generic\."+token+r"\.pth\.tar\."+token+r"\.tmp)$")
 recoverable=[]
 for x in run.iterdir():
  if x.name in allowed: continue
  if not run_temp_re.fullmatch(x.name): fail("unexpected run entry")
  regular(x); recoverable.append(x.relative_to(run).as_posix())
 regular(latest); regular(hp); payload=torch.load(latest,map_location="cpu",weights_only=True); finite(payload)
 if not isinstance(payload,dict) or payload.get("schema")!=training_schema or payload.get("run_identity")!=identity or payload.get("test_split_accessed") is not False: fail("latest identity")
 epoch=payload.get("epoch")
 if type(epoch) is not int or not 500<=epoch<=1000: fail("epoch")
 validate(payload.get("state_dict"),model.state_dict())
 opt=torch.optim.Adam(model.parameters(),lr=a.base_lr)
 e._validate_and_load_adam_optimizer_state(optimizer_state=payload.get("optimizer"),model=model,optimizer=opt,identity=identity,completed_epoch=epoch,total_epochs=1000)
 train,val=e._validate_resume_history(completed_epoch=epoch,interval=1,training_history=payload.get("training_history"),validation_history=payload.get("validation_history")); finite(payload.get("rng"))
 if len(train)!=epoch or len(val)!=epoch or train[-1].get("epoch")!=epoch or val[-1].get("epoch")!=epoch: fail("histories")
 artifacts=payload.get("candidate_artifacts"); frontier=tuple(s.retention_frontier_epochs(val,margin=None))
 if not isinstance(artifacts,dict) or tuple(sorted(artifacts))!=frontier: fail("frontier")
 if cd.is_symlink() or not cd.is_dir(): fail("candidate dir")
 by={r["epoch"]:r for r in val}; names=set(); hashes={}
 for ce in frontier:
  name="epoch_%04d.pth.tar"%ce; path=cd/name; names.add(name); regular(path); digest=sha(path); hashes["candidates/"+name]=digest
  if artifacts[ce]!={"relative_path":"candidates/"+name,"file_sha256":digest}: fail("candidate metadata")
  c=torch.load(path,map_location="cpu",weights_only=True); finite(c)
  if c.get("schema")!=candidate_schema or c.get("run_identity")!=identity or c.get("epoch")!=ce or c.get("validation_record")!=by[ce] or c.get("test_split_accessed") is not False: fail("candidate")
  validate(c.get("state_dict"),model.state_dict())
 candidate_temp_re=re.compile(r"^\.epoch_[0-9]{4}\.pth\.tar\."+token+r"\.tmp$")
 actual=set()
 for x in cd.iterdir():
  if candidate_temp_re.fullmatch(x.name): regular(x); recoverable.append(x.relative_to(run).as_posix())
  else: actual.add(x.name)
 for extra_name in sorted(actual-names):
  path=cd/extra_name; regular(path)
  if not extra_name.startswith("epoch_") or not extra_name.endswith(".pth.tar"): fail("extra candidate name")
  try: ce=int(extra_name[6:10])
  except ValueError: fail("extra candidate epoch")
  if extra_name!="epoch_%04d.pth.tar"%ce or ce<1: fail("extra candidate name")
  c=torch.load(path,map_location="cpu",weights_only=True); finite(c)
  expected_record=by.get(ce)
  if not _recoverable_extra_candidate(epoch,ce,has_committed_record=expected_record is not None): fail("extra candidate window")
  if c.get("schema")!=candidate_schema or c.get("run_identity")!=identity or c.get("epoch")!=ce or c.get("test_split_accessed") is not False: fail("extra candidate identity")
  if expected_record is not None and c.get("validation_record")!=expected_record: fail("extra candidate record")
  validate(c.get("state_dict"),model.state_dict())
 history=json.loads(hp.read_text(encoding="utf-8"),object_pairs_hook=unique,parse_constant=lambda x:fail("constant")); finite(history)
 if history.get("schema")!=history_schema or history.get("run_identity")!=identity or history.get("test_split_accessed") is not False: fail("history JSON identity")
 ht=history.get("training_history"); hv=history.get("validation_history")
 if not isinstance(ht,list) or not isinstance(hv,list): fail("history JSON records")
 h_epoch=len(ht)
 if not _recoverable_history_epoch(epoch,h_epoch): fail("history JSON crash window")
 e._validate_resume_history(completed_epoch=h_epoch,interval=1,training_history=ht,validation_history=hv)
 if h_epoch==epoch and (ht!=train or hv!=val or history.get("candidate_artifacts")!={str(k):v for k,v in artifacts.items()}): fail("history JSON committed mismatch")
 _assert_sidecar_crash_window(committed_epoch=epoch,history_epoch=h_epoch,extra_candidate_count=len(actual-names),final_artifact_exists=final_artifact_exists)
 complete=False
 def validate_final(path,role=None):
  regular(path); q=torch.load(path,map_location="cpu",weights_only=True); finite(q)
  if q.get("schema")!=checkpoint_schema or q.get("training")!=identity or q.get("test_split_accessed") is not False or q.get("public_test_supported") is not False or q.get("selection_margin_raw") is not None or q.get("selection_window_applied") is not False: fail("final identity")
  if role is not None and q.get("selection_role")!=role: fail("role final identity")
  validate(q.get("state_dict"),model.state_dict())
  return q
 def validate_generic(path):
  q=validate_final(path); provenance=s.select_checkpoints(val,primary_role=s.PRIMARY_ROLE,margin=None)
  selected=provenance["primary_selected_epoch"]; metadata=artifacts.get(selected)
  _assert_generic_selection_metadata(q,selected_epoch=selected,provenance=provenance)
  if not isinstance(metadata,dict): fail("generic source metadata")
  cp=run/metadata.get("relative_path",""); regular(cp)
  if metadata.get("file_sha256")!=sha(cp): fail("generic source SHA")
  cq=torch.load(cp,map_location="cpu",weights_only=True); qs=validate(q.get("state_dict"),model.state_dict()); cs=validate(cq.get("state_dict"),model.state_dict())
  if set(qs)!=set(cs) or any(not torch.equal(qs[k],cs[k]) for k in qs): fail("generic state differs from source")
 if summary.exists() or summary.is_symlink():
  regular(summary)
  raw=json.loads(summary.read_text(encoding="utf-8"),object_pairs_hook=unique,parse_constant=lambda x:fail("summary constant")); finite(raw)
  sv,lv,ci=t._load_completed_transaction(a,p)
  if ci!=identity or lv.get("epoch")!=1000 or raw!=sv: fail("complete identity")
  provenance=s.select_checkpoints(val,primary_role=s.PRIMARY_ROLE,margin=None); primary_epoch=provenance["primary_selected_epoch"]; primary_record=by[primary_epoch]
  primary_record_sha=hashlib.sha256(json.dumps(primary_record,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
  _assert_summary_evidence(sv,training_history=train,validation_history=val,candidate_artifacts={str(k):v for k,v in artifacts.items()},selected_record=primary_record,selected_record_sha256=primary_record_sha)
  role_map=sv.get("role_final_checkpoints")
  if role_map is not None:
   t._validate_committed_role_map(run_dir=run,role_map=role_map,identity=identity,history=val,expected_state=model.state_dict())
   for rp,role in zip(roles,("best_mIoU","best_Pd")):
    q=torch.load(rp,map_location="cpu",weights_only=True); source=q.get("source_candidate")
    selected=provenance["roles"][role]["selected"]; metadata=artifacts.get(selected["epoch"])
    record=by[selected["epoch"]]; record_sha=hashlib.sha256(json.dumps(record,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
    if not isinstance(metadata,dict) or source!={"relative_path":metadata.get("relative_path"),"sha256":metadata.get("file_sha256")} or q.get("epoch")!=selected["epoch"] or q.get("selected_complete_key")!=selected or q.get("selection_provenance")!=provenance or q.get("selected_validation_record")!=record or q.get("selected_validation_record_sha256")!=record_sha: fail("role source candidate")
    cp=run/metadata.get("relative_path",""); regular(cp)
    try: cp.resolve(strict=True).relative_to(run.resolve(strict=True))
    except ValueError: fail("role source containment")
    if metadata.get("file_sha256")!=sha(cp): fail("role source SHA")
    cq=torch.load(cp,map_location="cpu",weights_only=True); validate(cq.get("state_dict"),model.state_dict())
    qs=validate(q.get("state_dict"),model.state_dict()); cs=validate(cq.get("state_dict"),model.state_dict())
    if set(qs)!=set(cs) or any(not torch.equal(qs[k],cs[k]) for k in qs): fail("role state differs from source")
   complete=not (generic.exists() or generic.is_symlink())
   if generic.exists() or generic.is_symlink(): validate_generic(generic)
  else:
   role_names=("best_mIoU","best_Pd")
   for rp,role in zip(roles,role_names):
    if rp.exists() or rp.is_symlink():
     q=validate_final(rp,role); source=q.get("source_candidate")
     selected=provenance["roles"][role]["selected"]; metadata=artifacts.get(selected["epoch"])
     record=by[selected["epoch"]]; record_sha=hashlib.sha256(json.dumps(record,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
     if not isinstance(metadata,dict) or source!={"relative_path":metadata.get("relative_path"),"sha256":metadata.get("file_sha256")} or q.get("epoch")!=selected["epoch"] or q.get("selected_complete_key")!=selected or q.get("selection_provenance")!=provenance or q.get("selected_validation_record")!=record or q.get("selected_validation_record_sha256")!=record_sha: fail("partial role source")
     cp=run/metadata.get("relative_path",""); regular(cp)
     try: cp.resolve(strict=True).relative_to(run.resolve(strict=True))
     except ValueError: fail("partial role containment")
     if metadata.get("file_sha256")!=sha(cp): fail("partial role source SHA")
     cq=torch.load(cp,map_location="cpu",weights_only=True); qs=validate(q.get("state_dict"),model.state_dict()); cs=validate(cq.get("state_dict"),model.state_dict())
     if set(qs)!=set(cs) or any(not torch.equal(qs[k],cs[k]) for k in qs): fail("partial role state differs")
  if generic.exists() or generic.is_symlink(): validate_generic(generic)
 else:
  if any(x.exists() or x.is_symlink() for x in roles): fail("role final before summary")
  if generic.exists() or generic.is_symlink():
   if epoch!=1000: fail("early final")
   validate_generic(generic)
 state=_classify_transaction_window(committed_epoch=epoch,summary_exists=summary.exists(),role_map_present=(summary.exists() and role_map is not None),generic_final_exists=generic.exists(),role_file_count=sum(x.exists() for x in roles),primary_role_exists=roles[0].exists(),recoverable_temp_count=len(recoverable))
 if complete and state not in {"complete","finalize_1000"}: fail("completion classification")
 provenance=s.select_checkpoints(val,primary_role=s.PRIMARY_ROLE,margin=None); best=provenance["roles"][s.PRIMARY_ROLE]["selected"]
 out={"schema":"evisirst_seed42_to1000_state_inspection/v1","route":route,"state":state,"committed_epoch":epoch,"configured_total_epochs":1000,"latest_sha256":sha(latest),"history_sha256":sha(hp),"candidate_sha256":hashes,"recoverable_temporary_files":sorted(recoverable),"best_mIoU":best["mIoU"],"best_mIoU_epoch":best["epoch"],"selection_rule_version":provenance["rule_version"],"lockbox_accessed":False,"public_test_allowed":False,"test_split_accessed":False}
 if complete: out.update({"summary_sha256":sha(summary),"role_final_sha256":{x.name:sha(x) for x in roles}})
 emit(out)
'''


def inspect_route_state(route: str) -> dict[str, Any]:
    _spec(route)
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""; env["PYTHONDONTWRITEBYTECODE"] = "1"; env.pop("NVIDIA_VISIBLE_DEVICES", None)
    try:
        done = subprocess.run([os.fspath(PYTHON_BIN), "-c", _INSPECTION_PROGRAM, route], cwd=PROJECT_ROOT, env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Seed42ContinuationError("state inspection failed") from exc
    tokens = [line[len(_PREFIX):] for line in done.stdout.splitlines() if line.startswith(_PREFIX)]
    if done.returncode != 0 or len(tokens) != 1:
        raise Seed42ContinuationError("state inspection rejected artifacts: " + done.stderr[-2000:])
    try: report = json.loads(tokens[0])
    except json.JSONDecodeError as exc: raise Seed42ContinuationError("inspection output malformed") from exc
    if (not isinstance(report, dict) or report.get("schema") != INSPECTION_SCHEMA or report.get("route") != route
            or report.get("state") not in {"epoch500_ready","resume_501_999","finalize_1000","complete"}
            or not isinstance(report.get("recoverable_temporary_files"), list)
            or any(not isinstance(item, str) for item in report.get("recoverable_temporary_files", []))
            or any(report.get(k) is not False for k in ("lockbox_accessed","public_test_allowed","test_split_accessed"))):
        raise Seed42ContinuationError("inspection report differs")
    return report


_TEMP_TOKEN = r"[A-Za-z0-9_]{6,32}"
_RUN_TEMP_RE = re.compile(
    r"^(?:\.expected_generic\." + _TEMP_TOKEN + r"\.pth\.tar|"
    r"\.(?:last_training_state\.pth\.tar|validation_history\.json|summary\.json|"
    r"EviSIRST\.pth\.tar|EviSIRST_best_mIoU\.pth\.tar|EviSIRST_best_Pd\.pth\.tar)\."
    + _TEMP_TOKEN
    + r"\.tmp|\.\.expected_generic\."
    + _TEMP_TOKEN
    + r"\.pth\.tar\."
    + _TEMP_TOKEN
    + r"\.tmp)$"
)
_CANDIDATE_TEMP_RE = re.compile(
    r"^\.epoch_[0-9]{4}\.pth\.tar\." + _TEMP_TOKEN + r"\.tmp$"
)


def _recoverable_temp_relative(relative: str) -> bool:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return False
    if len(path.parts) == 1:
        return _RUN_TEMP_RE.fullmatch(path.name) is not None
    return bool(
        len(path.parts) == 2
        and path.parts[0] == "candidates"
        and _CANDIDATE_TEMP_RE.fullmatch(path.name)
    )


def _cleanup_recoverable_temps_under_run_lock(route: str) -> dict[str, Any]:
    """Remove only validated uncommitted writer temps while run lock is held."""
    spec = _spec(route)
    report = inspect_route_state(route)
    raw = report.get("recoverable_temporary_files", [])
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise Seed42ContinuationError("recoverable temp report differs")
    for relative in raw:
        if not _recoverable_temp_relative(relative):
            raise Seed42ContinuationError("unrecognized recoverable temp")
        path = spec.run_dir / relative
        parent = path.parent
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(parent, flags)
        try:
            observed = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(observed.st_mode):
                raise Seed42ContinuationError("recoverable temp is not regular")
            os.unlink(path.name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    cleaned = inspect_route_state(route)
    if cleaned.get("recoverable_temporary_files"):
        raise Seed42ContinuationError("recoverable temp cleanup is incomplete")
    return cleaned


def _accept_locked_clean_state(route: str, report: Mapping[str, Any]) -> dict[str, Any]:
    """Allow cleanup to expose an already-complete transaction to finalizer."""
    validate_start_snapshot(route)
    if report.get("recoverable_temporary_files"):
        raise Seed42ContinuationError("locked cleanup left temporary files")
    if report.get("state") == "complete":
        return dict(report)
    return assert_continuation_entry(route)


def create_start_snapshot(route: str) -> Path:
    spec = _spec(route)
    report = inspect_route_state(route)
    if report.get("state") != "epoch500_ready":
        raise Seed42ContinuationError("snapshot requires exact epoch 500")
    manifest = expected_snapshot_manifest(route)
    if (report.get("latest_sha256") != spec.latest_sha256 or report.get("history_sha256") != spec.history_sha256
            or report.get("candidate_sha256") != dict(spec.candidates)):
        raise Seed42ContinuationError("live epoch-500 hashes differ")
    _safe_snapshot_dir(spec.snapshot_dir, True)
    _recover_incomplete_snapshot_manifest_temps(spec.snapshot_dir, manifest)
    sources = {"last_training_state.pth.tar": spec.run_dir/"last_training_state.pth.tar",
               "validation_history.json": spec.run_dir/"validation_history.json",
               **{name: spec.run_dir/name for name in spec.candidates}}
    for relative, source in sources.items():
        _link(source, spec.snapshot_dir/relative, manifest["files"][relative])
    _write_manifest(spec.snapshot_dir/"manifest.json", manifest)
    validate_start_snapshot(route)
    return spec.snapshot_dir/"manifest.json"


def assert_continuation_entry(route: str) -> dict[str, Any]:
    spec = _spec(route); validate_start_snapshot(route); report = inspect_route_state(route)
    if report["state"] == "complete": raise Seed42ContinuationError("route already complete")
    if report["state"] == "epoch500_ready" and (report["latest_sha256"] != spec.latest_sha256
            or report["history_sha256"] != spec.history_sha256 or report["candidate_sha256"] != dict(spec.candidates)):
        raise Seed42ContinuationError("epoch-500 anchor differs")
    return report


def assert_completed_state(route: str) -> dict[str, Any]:
    validate_start_snapshot(route); report = inspect_route_state(route)
    if report.get("state") != "complete" or report.get("committed_epoch") != 1000:
        raise Seed42ContinuationError("route is not complete")
    return {**report, **POST_INTERIM_DISCLOSURE}


def _psbfr_args() -> argparse.Namespace:
    import train_irstd_model_design_screen_v1 as runner
    return runner.parse_args([
        "--dataset-root","/home/ly/SCTransNet_main/datasets", "--split-root","/home/ly/EviSIRST_main/splits/v2",
        "--variant","psbfr_v1", "--dataset","IRSTD-1K", "--target-mode","binary",
        "--architecture-seed","42", "--run-seed","42", "--device","cuda:0", "--epochs","1000",
        "--warmup-epochs","10", "--allow-sample-level-fallback", "--resume",
    ])


def _cp_args() -> argparse.Namespace:
    import train_irstd_cp_hf_s2_legacy_screen_v1 as runner
    return runner.parse_args([
        "--dataset-root","/home/ly/SCTransNet_main/datasets", "--run-seed","42", "--device","cuda:0",
        "--epochs","1000", "--warmup-epochs","10", "--resume",
    ])


@contextmanager
def _psbfr_entry_guard(runner: Any) -> Iterator[None]:
    original = runner._run_process_lock
    @contextmanager
    def guarded(args: argparse.Namespace) -> Iterator[Path]:
        with original(args) as lock_path:
            cleaned = _cleanup_recoverable_temps_under_run_lock("psbfr")
            _accept_locked_clean_state("psbfr", cleaned)
            yield lock_path
    runner._run_process_lock = guarded
    try:
        yield
    finally:
        runner._run_process_lock = original


@contextmanager
def _cp_entry_guard(cp: Any, screen: Any) -> Iterator[None]:
    """Remove only Stage-1 stops while retaining strict resume/architecture checks."""
    original_context = cp._screen_transaction_adapter
    original_barrier = cp._formal_entry_barrier
    original_pause = cp._install_atomic_epoch500_pause
    trusted_loader = screen.r1._load_resume_state
    trusted_finalize = screen.hf_transaction._finalize_completed
    original_run_lock = screen._run_process_lock
    api = cp.architecture_api()

    @contextmanager
    def guarded_run_lock(args: argparse.Namespace) -> Iterator[Path]:
        with original_run_lock(args) as lock_path:
            cleaned = _cleanup_recoverable_temps_under_run_lock("cp_hf_s2")
            _accept_locked_clean_state("cp_hf_s2", cleaned)
            yield lock_path

    def barrier(args: argparse.Namespace, *, paths: Any = None) -> None:
        del args, paths
        validate_start_snapshot("cp_hf_s2")
        report = inspect_route_state("cp_hf_s2")
        if report.get("state") != "complete":
            assert_continuation_entry("cp_hf_s2")

    def no_pause(dataset: Any, *, smoke: bool) -> Any:
        if smoke or not callable(getattr(dataset, "set_epoch", None)):
            raise Seed42ContinuationError("CP continuation dataset differs")
        return dataset

    @contextmanager
    def continuation_context(args: argparse.Namespace | None) -> Iterator[Any]:
        if args is None or args.run_seed != 42 or args.epochs != 1000 or args.resume is not True:
            raise Seed42ContinuationError("CP continuation args differ")
        with original_context(args) as active:
            blocked_loader = active.r1._load_resume_state
            blocked_finalize = active.hf_transaction._finalize_completed
            blocked_atomic_save = active._variant_atomic_torch_save
            paths = active.resolve_run_paths(args)
            generic_final = paths["final"]
            def continuation_loader(*positional: Any, **keywords: Any) -> Any:
                restored = trusted_loader(*positional, **keywords)
                start = restored[0] if isinstance(restored, tuple) and restored else None
                if type(start) is not int or not 501 <= start <= 1001:
                    raise Seed42ContinuationError("CP restored frontier is outside epoch 500..1000")
                model = keywords.get("model")
                if model is None:
                    raise Seed42ContinuationError("CP resume model missing")
                manifest = api["validate_irstd_cp_hf_s2_v1"](model, require_identity_initialization=False)
                if manifest.get("insertion_point") != "after_up_decoder2_finish_before_gt2_and_up_decoder1":
                    raise Seed42ContinuationError("CP resumed architecture differs")
                return restored
            active.r1._load_resume_state = continuation_loader
            active.hf_transaction._finalize_completed = trusted_finalize
            def continuation_atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
                if path == generic_final and path.exists() and not path.is_symlink():
                    if assert_continuation_entry("cp_hf_s2").get("state") != "finalize_1000":
                        raise Seed42ContinuationError("generic-final recovery state differs")
                    import torch
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=".expected_generic.", suffix=".pth.tar", dir=path.parent
                    )
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    temporary.unlink()
                    try:
                        blocked_atomic_save(temporary, payload)
                        existing = torch.load(path, map_location="cpu", weights_only=True)
                        expected = torch.load(temporary, map_location="cpu", weights_only=True)
                        if not _semantic_equal(existing, expected):
                            raise Seed42ContinuationError(
                                "existing generic final is not semantically identical"
                            )
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                    return
                blocked_atomic_save(path, payload)
            active._variant_atomic_torch_save = continuation_atomic_save
            try:
                yield active
            finally:
                active._variant_atomic_torch_save = blocked_atomic_save
                active.hf_transaction._finalize_completed = blocked_finalize
                active.r1._load_resume_state = blocked_loader

    cp._formal_entry_barrier = barrier
    cp._install_atomic_epoch500_pause = no_pause
    cp._screen_transaction_adapter = continuation_context
    screen._run_process_lock = guarded_run_lock
    try:
        yield
    finally:
        screen._run_process_lock = original_run_lock
        cp._screen_transaction_adapter = original_context
        cp._install_atomic_epoch500_pause = original_pause
        cp._formal_entry_barrier = original_barrier


def run_route(route: str) -> Path:
    _spec(route)
    _authorization()
    from tools import run_irstd_seed42_to1000_when_idle as watcher
    watcher.assert_no_forbidden_processes(route)
    assert_continuation_entry(route)
    if route == "psbfr":
        import train_irstd_model_design_screen_v1 as runner
        with _psbfr_entry_guard(runner):
            result = runner.run(_psbfr_args())
    else:
        import train_irstd_cp_hf_s2_legacy_screen_v1 as cp
        import train_irstd_model_design_screen_v1 as screen
        with _cp_entry_guard(cp, screen):
            result = cp.run(_cp_args())
    if not isinstance(result, Path):
        raise Seed42ContinuationError("runner returned no checkpoint")
    evidence = assert_completed_state(route)
    print(_PREFIX + json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    return result


def route_worker_argv(route: str) -> tuple[str, ...]:
    _spec(route)
    return os.fspath(PYTHON_BIN), os.fspath(ADAPTER_PATH), "--route", route


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=ROUTES, required=True)
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(run_route(args.route))


if __name__ == "__main__":
    main()


__all__ = [
    "ADAPTER_PATH", "FINAL_EPOCH", "FORMAL_AUTHORIZATION_SHA256", "INSPECTION_SCHEMA",
    "POST_INTERIM_DISCLOSURE", "ROUTES", "ROUTE_SPECS", "SNAPSHOT_SCHEMA", "START_EPOCH",
    "START_SNAPSHOT_ROOT", "Seed42ContinuationError", "assert_completed_state",
    "assert_continuation_entry", "create_start_snapshot", "expected_snapshot_manifest",
    "inspect_route_state", "route_worker_argv", "run_route", "validate_start_snapshot",
]
