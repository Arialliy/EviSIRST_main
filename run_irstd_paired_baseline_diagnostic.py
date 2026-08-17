#!/usr/bin/env python3
"""Diagnose the one completed paired IRSTD-1K R1 baseline checkpoint.

This command has no checkpoint-selection surface.  It accepts only the data
root and execution device, then loads the fixed formal seed-1446202191 final
validation-selected checkpoint.  One canonical V2 validation pass computes
the unchanged R1 ``out``-head metrics and the supplementary matched-target
diagnostics used by the complete-target promotion gate.  Public test data is
neither imported nor addressable.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model.EviSIRST as evisirst_model_module
import train_validation_selected as r1
from experiments import evisirst_complete_target_crop as complete_crop
from experiments import evisirst_v2_data as v2_data
from experiments import evisirst_v2_selection as validation_selection
from experiments.evisirst_v2_data import EviSIRSTV2ValDataset
from model.EviSIRST import initialize_evisirst


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = "IRSTD-1K"
DATA_ROLE = "val"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
RUN_SEED = 1446202191
EPOCHS = 1000
WORKERS = 0
CHECKPOINT_RELATIVE_PATH = (
    "runs/validation_selected/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/EviSIRST.pth.tar"
)
SUMMARY_RELATIVE_PATH = (
    "runs/validation_selected/formal/IRSTD-1K/binary/"
    "run_seed_1446202191/summary.json"
)
OUTPUT_RELATIVE_PATH = (
    "runs/irstd_performance/complete_target_v1/"
    "paired_baseline_diagnostics/run_seed_1446202191/"
    "matched_target_diagnostics.json"
)
RUN_LOCK_RELATIVE_PATH = (
    "runs/irstd_performance/complete_target_v1/"
    "paired_baseline_diagnostics/run_seed_1446202191/run.lock"
)
RUN_LOCK_FD_ENV = "EVISIRST_PAIRED_DIAGNOSTIC_RUN_LOCK_FD"
GPU_LOCK_FD_ENV = "EVISIRST_PAIRED_DIAGNOSTIC_GPU_LOCK_FD"
GPU_UUID_ENV = "EVISIRST_PAIRED_DIAGNOSTIC_GPU_UUID"
SPLIT_MANIFEST_RELATIVE_PATH = "splits/v2/IRSTD-1K/manifest.json"
VAL_INDEX_RELATIVE_PATH = "splits/v2/IRSTD-1K/val.txt"

DIAGNOSTIC_SCHEMA = "evisirst_irstd_paired_baseline_diagnostic/v1"
DIAGNOSTIC_SOURCE_SCHEMA = "evisirst_paired_baseline_diagnostic_source_set/v1"
PREDICTION_DIGEST_SCHEMA = "evisirst_out_prediction_digest/v1"
TARGET_DIGEST_SCHEMA = "evisirst_validation_target_digest/v1"

_FINAL_KEYS = frozenset(
    {
        "schema",
        "model",
        "dataset",
        "checkpoint_role",
        "epoch",
        "seed",
        "architecture_seed",
        "run_seed",
        "state_dict",
        "target_mode",
        "normalization",
        "normalization_provenance",
        "training",
        "training_identity_sha256",
        "split_provenance",
        "split_seed",
        "split_manifest_sha256",
        "data_tree_sha256",
        "data_tree_verified",
        "selection_provenance",
        "source_selection",
        "selection_is_optimistic",
        "optimistic",
        "test_split_accessed",
        "model_metadata",
        "smoke",
    }
)
_TRAINING_KEYS = frozenset(
    {
        "schema",
        "model",
        "dataset",
        "architecture_seed",
        "run_seed",
        "target_mode",
        "epochs",
        "batch_size",
        "workers",
        "base_lr",
        "min_lr",
        "warmup_epochs",
        "val_interval",
        "normalization_mode",
        "optimizer",
        "loss",
        "evaluation",
        "selection_rule",
        "determinism_protocol",
        "manifest_sha256",
        "split_seed",
        "data_tree_sha256",
        "grouping_policy",
        "train_count",
        "val_count",
        "smoke",
        "smoke_max_train_samples",
        "smoke_max_val_samples",
        "test_split_accessed",
        "identity_sha256",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "status",
        "dataset",
        "checkpoint",
        "checkpoint_role",
        "selected_epoch",
        "architecture_seed",
        "run_seed",
        "target_mode",
        "split_provenance",
        "selection",
        "training_history",
        "validation_history",
        "candidate_artifacts",
        "normalization",
        "source_selection",
        "selection_is_optimistic",
        "optimistic",
        "test_split_accessed",
        "smoke",
        "elapsed_seconds",
    }
)


class PairedBaselineDiagnosticError(ValueError):
    """The fixed paired-baseline diagnostic contract was violated."""


def _run_artifact_path(relative_path: str) -> Path:
    """Resolve a fixed run artifact and reject redirection outside ``runs/``."""

    path = _fixed_path(relative_path, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    runs_root = (PROJECT_ROOT.resolve(strict=True) / "runs").resolve(strict=True)
    try:
        parent.relative_to(runs_root)
    except ValueError as exc:
        raise PairedBaselineDiagnosticError("diagnostic artifact escaped runs/") from exc
    if parent != path.parent or path.is_symlink():
        raise PairedBaselineDiagnosticError(
            "diagnostic artifact path contains a symbolic-link redirect"
        )
    return path


def _validate_lock_descriptor(
    descriptor: int, path: Path, *, environment_name: str, label: str
) -> None:
    if descriptor <= 2:
        raise PairedBaselineDiagnosticError(
            f"{environment_name} must name a descriptor greater than 2"
        )
    try:
        opened = os.fstat(descriptor)
        expected = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise PairedBaselineDiagnosticError(
            f"inherited {label} descriptor is invalid"
        ) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(expected.st_mode)
        or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise PairedBaselineDiagnosticError(
            f"inherited descriptor is not the fixed {label}"
        )


@contextlib.contextmanager
def exclusive_run_lock():
    """Hold the unique diagnostic lock, accepting a watcher's inherited FD.

    A watcher owns this lock before it starts polling and passes the same open
    file description through ``RUN_LOCK_FD_ENV``.  A direct/manual invocation
    instead opens and non-blockingly locks the identical path itself.  The
    inherited case must never unlock the descriptor because doing so would
    also release the watcher's lock.
    """

    path = _run_artifact_path(RUN_LOCK_RELATIVE_PATH)
    inherited = os.environ.get(RUN_LOCK_FD_ENV)
    descriptor: int
    owns_descriptor = False
    if inherited is not None:
        if not inherited.isascii() or not inherited.isdecimal():
            raise PairedBaselineDiagnosticError(
                f"{RUN_LOCK_FD_ENV} must be a decimal file descriptor"
            )
        descriptor = int(inherited)
        _validate_lock_descriptor(
            descriptor,
            path,
            environment_name=RUN_LOCK_FD_ENV,
            label="run lock",
        )
    else:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise PairedBaselineDiagnosticError(
                "could not open the fixed paired-diagnostic run lock"
            ) from exc
        owns_descriptor = True
        try:
            _validate_lock_descriptor(
                descriptor,
                path,
                environment_name=RUN_LOCK_FD_ENV,
                label="run lock",
            )
        except Exception:
            os.close(descriptor)
            raise

    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PairedBaselineDiagnosticError(
                "another paired-baseline diagnostic or watcher holds the run lock"
            ) from exc
        yield descriptor
    finally:
        if owns_descriptor:
            os.close(descriptor)


def validate_optional_inherited_gpu_lock() -> None:
    """Validate that a watcher-held per-GPU lock reached this process."""

    raw_descriptor = os.environ.get(GPU_LOCK_FD_ENV)
    gpu_uuid = os.environ.get(GPU_UUID_ENV)
    if raw_descriptor is None and gpu_uuid is None:
        return
    if raw_descriptor is None or gpu_uuid is None:
        raise PairedBaselineDiagnosticError(
            "inherited GPU lock descriptor and UUID must be provided together"
        )
    if not raw_descriptor.isascii() or not raw_descriptor.isdecimal():
        raise PairedBaselineDiagnosticError(
            f"{GPU_LOCK_FD_ENV} must be a decimal file descriptor"
        )
    if re.fullmatch(
        r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
        gpu_uuid,
    ) is None:
        raise PairedBaselineDiagnosticError(
            f"{GPU_UUID_ENV} must contain one full NVIDIA GPU UUID"
        )
    path = _run_artifact_path(f"runs/.gpu_locks/{gpu_uuid}.lock")
    descriptor = int(raw_descriptor)
    _validate_lock_descriptor(
        descriptor,
        path,
        environment_name=GPU_LOCK_FD_ENV,
        label="per-GPU lock",
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PairedBaselineDiagnosticError(
            "inherited per-GPU descriptor does not own the GPU lock"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    if args.workers != WORKERS:
        parser.error("--workers is frozen at 0 for the paired diagnostic")
    return args


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PairedBaselineDiagnosticError(
            "metadata must be strict JSON without NaN/Infinity"
        ) from exc


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Any, expected: frozenset[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise PairedBaselineDiagnosticError(f"{label} must be a string-keyed object")
    missing = sorted(expected.difference(value))
    unexpected = sorted(set(value).difference(expected))
    if missing or unexpected:
        raise PairedBaselineDiagnosticError(
            f"{label} keys differ: missing={missing}, unexpected={unexpected}"
        )
    return value


def _fixed_path(relative_path: str, *, must_exist: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise PairedBaselineDiagnosticError("internal artifact path is unsafe")
    repository = PROJECT_ROOT.resolve(strict=True)
    path = repository.joinpath(*pure.parts)
    current = repository
    for component in pure.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise PairedBaselineDiagnosticError(
                f"artifact path contains a symlink: {relative_path}"
            )
    if must_exist:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.resolve(strict=True) != path:
            raise PairedBaselineDiagnosticError("artifact path was redirected")
    return path


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PairedBaselineDiagnosticError(f"{label} is not a regular file")
        content = handle.read()
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if not content or before_identity != after_identity:
        raise PairedBaselineDiagnosticError(f"{label} changed while being read")
    return content


def _strict_json_object(content: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PairedBaselineDiagnosticError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise PairedBaselineDiagnosticError(f"{label} must contain a JSON object")
    _canonical_json_bytes(value)
    return value


def _validate_training_identity(raw: Any) -> dict[str, Any]:
    identity = dict(_require_exact_keys(raw, _TRAINING_KEYS, label="training"))
    observed_identity_sha = identity.get("identity_sha256")
    unhashed = dict(identity)
    del unhashed["identity_sha256"]
    if (
        not isinstance(observed_identity_sha, str)
        or _sha256_bytes(_canonical_json_bytes(unhashed)) != observed_identity_sha
    ):
        raise PairedBaselineDiagnosticError("training identity SHA-256 differs")

    fixed = {
        "schema": r1.TRAINING_SCHEMA + "/run_identity",
        "model": "EviSIRST",
        "dataset": DATASET,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "target_mode": TARGET_MODE,
        "epochs": EPOCHS,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "val_interval": 1,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "selection_rule": validation_selection.INDEPENDENT_RULE_VERSION,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    for field, expected in fixed.items():
        if type(identity.get(field)) is not type(expected) or identity.get(field) != expected:
            raise PairedBaselineDiagnosticError(
                f"training.{field} differs from the paired formal R1 contract"
            )
    for field in ("train_count", "val_count"):
        value = identity[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PairedBaselineDiagnosticError(f"training.{field} is invalid")

    protocol = identity.get("determinism_protocol")
    if not isinstance(protocol, Mapping):
        raise PairedBaselineDiagnosticError("training determinism protocol is absent")
    embedded_files = protocol.get("source_files")
    embedded_tree = protocol.get("source_tree_sha256")
    current_sources = r1._protocol_source_provenance()
    if (
        embedded_files != current_sources["files"]
        or embedded_tree != current_sources["source_tree_sha256"]
    ):
        raise PairedBaselineDiagnosticError(
            "current R1 source tree differs from the checkpoint-bound source tree"
        )
    return identity


def _validate_selection_provenance(value: Any, *, selected_epoch: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedBaselineDiagnosticError("selection provenance is malformed")
    records = value.get("evaluated_records")
    if not isinstance(records, list) or len(records) != EPOCHS:
        raise PairedBaselineDiagnosticError(
            "selection provenance is incomplete; baseline has not completed 1000 epochs"
        )
    try:
        recomputed = validation_selection.select_independent_checkpoint(records)
    except (TypeError, ValueError) as exc:
        raise PairedBaselineDiagnosticError(
            "selection provenance cannot be recomputed"
        ) from exc
    if recomputed != value:
        raise PairedBaselineDiagnosticError("selection provenance recomputation differs")
    if (
        recomputed.get("evaluated_epochs") != list(range(1, EPOCHS + 1))
        or recomputed.get("selected", {}).get("epoch") != selected_epoch
        or recomputed.get("data_role") != DATA_ROLE
        or recomputed.get("test_selection_supported") is not False
    ):
        raise PairedBaselineDiagnosticError("selection provenance identity differs")
    return json.loads(_canonical_json_bytes(recomputed).decode("ascii"))


def _validate_summary(
    *, checkpoint_payload: Mapping[str, Any], selected_epoch: int
) -> dict[str, Any]:
    path = _fixed_path(SUMMARY_RELATIVE_PATH, must_exist=True)
    content = _read_regular_file_once(path, label="completed-run summary")
    summary = dict(
        _require_exact_keys(
            _strict_json_object(content, label="completed-run summary"),
            _SUMMARY_KEYS,
            label="completed-run summary",
        )
    )
    fixed = {
        "schema": r1.TRAINING_SCHEMA + "/summary",
        "status": "complete",
        "dataset": DATASET,
        "checkpoint": CHECKPOINT_RELATIVE_PATH,
        "checkpoint_role": "validation_selected",
        "selected_epoch": selected_epoch,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "target_mode": TARGET_MODE,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
    }
    for field, expected in fixed.items():
        if type(summary.get(field)) is not type(expected) or summary.get(field) != expected:
            raise PairedBaselineDiagnosticError(
                f"completed-run summary {field} differs"
            )
    if (
        summary.get("split_provenance") != checkpoint_payload["split_provenance"]
        or summary.get("normalization") != checkpoint_payload["normalization"]
    ):
        raise PairedBaselineDiagnosticError("summary/checkpoint data binding differs")
    selection = summary.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selected_epoch") != selected_epoch
        or selection.get("selection_provenance")
        != checkpoint_payload["selection_provenance"]
        or selection.get("selection_is_optimistic") is not False
        or selection.get("optimistic") is not False
    ):
        raise PairedBaselineDiagnosticError("summary/checkpoint selection differs")
    training_history = summary.get("training_history")
    validation_history = summary.get("validation_history")
    if (
        not isinstance(training_history, list)
        or not isinstance(validation_history, list)
        or len(training_history) != EPOCHS
        or len(validation_history) != EPOCHS
        or [record.get("epoch") for record in training_history]
        != list(range(1, EPOCHS + 1))
        or [record.get("epoch") for record in validation_history]
        != list(range(1, EPOCHS + 1))
    ):
        raise PairedBaselineDiagnosticError(
            "summary histories do not prove a completed 1000-epoch run"
        )
    selected_record = validation_history[selected_epoch - 1]
    if (
        not isinstance(selected_record, Mapping)
        or selected_record.get("data_role") != DATA_ROLE
        or selected_record.get("evaluation_head") != "out"
        or selected_record.get("mIoU")
        != checkpoint_payload["selection_provenance"]["selected"]["mIoU"]
        or selected_record.get("Fa")
        != checkpoint_payload["selection_provenance"]["selected"]["Fa"]
        or selected_record.get("Pd")
        != checkpoint_payload["selection_provenance"]["selected"]["Pd"]
    ):
        raise PairedBaselineDiagnosticError("selected validation record differs")
    return {
        "relative_path": SUMMARY_RELATIVE_PATH,
        "sha256": _sha256_bytes(content),
        "selected_validation_record_sha256": _sha256_bytes(
            _canonical_json_bytes(selected_record)
        ),
    }


def _normalization_values(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"mean", "std"}:
        raise PairedBaselineDiagnosticError("checkpoint normalization is malformed")
    mean = value.get("mean")
    std = value.get("std")
    if (
        isinstance(mean, bool)
        or isinstance(std, bool)
        or not isinstance(mean, (int, float))
        or not isinstance(std, (int, float))
        or not math.isfinite(float(mean))
        or not math.isfinite(float(std))
        or float(std) <= 0.0
    ):
        raise PairedBaselineDiagnosticError("checkpoint normalization is invalid")
    return {"mean": float(mean), "std": float(std)}


def configure_inference_determinism() -> None:
    torch.manual_seed(ARCHITECTURE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ARCHITECTURE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def require_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise PairedBaselineDiagnosticError("--device is malformed") from exc
    if device.type not in {"cpu", "cuda"}:
        raise PairedBaselineDiagnosticError("--device must be cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise PairedBaselineDiagnosticError("CUDA device index is out of range")
    return device


def load_fixed_final_model(
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly load the sole completed formal paired-baseline endpoint."""

    path = _fixed_path(CHECKPOINT_RELATIVE_PATH, must_exist=True)
    content = _read_regular_file_once(path, label="paired final checkpoint")
    checkpoint_sha256 = _sha256_bytes(content)
    try:
        raw_payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=True
        )
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PairedBaselineDiagnosticError(
            "paired final checkpoint is not a weights-only checkpoint"
        ) from exc
    payload = dict(
        _require_exact_keys(raw_payload, _FINAL_KEYS, label="paired final checkpoint")
    )
    selected_epoch = payload.get("epoch")
    fixed = {
        "schema": r1.CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "checkpoint_role": "validation_selected",
        "seed": ARCHITECTURE_SEED,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "target_mode": TARGET_MODE,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
        "data_tree_verified": True,
    }
    for field, expected in fixed.items():
        if type(payload.get(field)) is not type(expected) or payload.get(field) != expected:
            raise PairedBaselineDiagnosticError(f"checkpoint {field} differs")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not 1 <= selected_epoch <= EPOCHS
    ):
        raise PairedBaselineDiagnosticError("checkpoint selected epoch is invalid")

    training = _validate_training_identity(payload["training"])
    if payload.get("training_identity_sha256") != training["identity_sha256"]:
        raise PairedBaselineDiagnosticError("checkpoint training identity differs")
    selection = _validate_selection_provenance(
        payload["selection_provenance"], selected_epoch=selected_epoch
    )
    split = payload.get("split_provenance")
    if not isinstance(split, Mapping):
        raise PairedBaselineDiagnosticError("checkpoint split provenance is malformed")
    split_bindings = {
        "manifest_relative_path": SPLIT_MANIFEST_RELATIVE_PATH,
        "manifest_sha256": training["manifest_sha256"],
        "split_seed": training["split_seed"],
        "data_tree_sha256": training["data_tree_sha256"],
        "data_tree_verified": True,
        "train_count": training["train_count"],
        "val_count": training["val_count"],
        "test_index_opened": False,
    }
    for field, expected in split_bindings.items():
        if type(split.get(field)) is not type(expected) or split.get(field) != expected:
            raise PairedBaselineDiagnosticError(
                f"checkpoint split_provenance.{field} differs"
            )
    top_split_bindings = {
        "split_manifest_sha256": split["manifest_sha256"],
        "split_seed": split["split_seed"],
        "data_tree_sha256": split["data_tree_sha256"],
        "data_tree_verified": split["data_tree_verified"],
    }
    for field, expected in top_split_bindings.items():
        if payload.get(field) != expected:
            raise PairedBaselineDiagnosticError(f"checkpoint {field} differs")

    normalization = _normalization_values(payload["normalization"])
    summary = _validate_summary(
        checkpoint_payload=payload, selected_epoch=selected_epoch
    )
    model, model_metadata = initialize_evisirst(
        DATASET, seed=ARCHITECTURE_SEED, training=False
    )
    state = r1._validate_state_dict(payload["state_dict"], model.state_dict())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PairedBaselineDiagnosticError("strict model load reported incompatibility")
    model.to(device)
    model.eval()
    model.mode = "test"
    return model, {
        "relative_path": CHECKPOINT_RELATIVE_PATH,
        "sha256": checkpoint_sha256,
        "schema": r1.CHECKPOINT_SCHEMA,
        "selected_epoch": selected_epoch,
        "training_identity_sha256": training["identity_sha256"],
        "training_source_tree_sha256": training["determinism_protocol"][
            "source_tree_sha256"
        ],
        "selection_provenance_sha256": _sha256_bytes(
            _canonical_json_bytes(selection)
        ),
        "split_provenance": dict(split),
        "normalization": normalization,
        "summary": summary,
        "model_metadata": model_metadata,
    }


def build_validation_dataset(
    *, dataset_root: str | os.PathLike[str], checkpoint_metadata: Mapping[str, Any]
) -> EviSIRSTV2ValDataset:
    dataset = EviSIRSTV2ValDataset(
        DATASET,
        dataset_root=dataset_root,
        split_root=PROJECT_ROOT / "splits" / "v2",
        target_mode=TARGET_MODE,
        normalization_mode="legacy",
        normalization_values=checkpoint_metadata["normalization"],
        verify_data_tree=True,
    )
    contract = dataset.contract
    split = checkpoint_metadata["split_provenance"]
    bindings = {
        "manifest_sha256": contract.manifest_sha256,
        "data_tree_sha256": contract.data_tree_sha256,
        "split_seed": contract.manifest["seeds"]["split_seed"],
        "train_count": len(contract.train_ids),
        "val_count": len(contract.val_ids),
    }
    for field, observed in bindings.items():
        if split.get(field) != observed:
            raise PairedBaselineDiagnosticError(
                f"canonical V2 split {field} differs from checkpoint"
            )
    if (
        not contract.data_tree_verified
        or dataset.metadata.get("split") != DATA_ROLE
        or dataset.metadata.get("test_index_opened") is not False
    ):
        raise PairedBaselineDiagnosticError("canonical validation contract is invalid")
    manifest_path = _fixed_path(SPLIT_MANIFEST_RELATIVE_PATH, must_exist=True)
    val_path = _fixed_path(VAL_INDEX_RELATIVE_PATH, must_exist=True)
    if _sha256_file(manifest_path) != contract.manifest_sha256:
        raise PairedBaselineDiagnosticError("canonical split manifest SHA-256 differs")
    expected_val_sha = split.get("outputs", {}).get("val", {}).get("file_sha256")
    if _sha256_file(val_path) != expected_val_sha:
        raise PairedBaselineDiagnosticError("canonical val.txt SHA-256 differs")
    return dataset


def _extract_sample_id(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value or not value.isascii():
        raise PairedBaselineDiagnosticError("validation sample ID is malformed")
    return value


def _digest_update(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _update_array_digest(
    digest: Any, *, sample_id: str, array: np.ndarray
) -> None:
    canonical = np.ascontiguousarray(array, dtype="<f4")
    _digest_update(
        digest,
        _canonical_json_bytes(
            {
                "sample_id": sample_id,
                "shape": list(canonical.shape),
                "dtype": "float32-little-endian",
            }
        ),
    )
    _digest_update(digest, canonical.tobytes(order="C"))


@torch.inference_mode()
def evaluate_out_once(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], int]:
    """Use one frozen ``out`` forward per sample for both metric families."""

    if (
        complete_crop.PREDICTION_THRESHOLD != r1.PROBABILITY_THRESHOLD
        or complete_crop.TARGET_THRESHOLD != r1.TARGET_THRESHOLD
        or complete_crop.MATCH_RADIUS != r1.MATCH_RADIUS
    ):
        raise PairedBaselineDiagnosticError("R1 and mechanism protocols diverged")
    model.eval()
    model.mode = "test"
    criterion = nn.BCELoss(reduction="mean")
    r1_metrics = r1.ValidationMetrics(
        r1.PROBABILITY_THRESHOLD, r1.MATCH_RADIUS, r1.TINY_AREA
    )
    mechanism = complete_crop.MatchedTargetDiagnostics()
    prediction_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    _digest_update(prediction_digest, PREDICTION_DIGEST_SCHEMA.encode("ascii"))
    _digest_update(target_digest, TARGET_DIGEST_SCHEMA.encode("ascii"))
    sample_count = 0
    for images, masks, sizes, raw_sample_ids in loader:
        height, width = r1._extract_hw(sizes)
        sample_id = _extract_sample_id(raw_sample_ids)
        images = images.to(device, non_blocking=True)
        prediction = r1.final_prediction(model(images))[:, :, :height, :width]
        target = masks[:, :, :height, :width].to(device, non_blocking=True)
        if (
            prediction.ndim != 4
            or prediction.shape != target.shape
            or prediction.shape[:2] != (1, 1)
        ):
            raise PairedBaselineDiagnosticError(
                "out probability shape differs from canonical validation target"
            )
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("out probability contains non-finite values")
        if bool(((prediction < 0.0) | (prediction > 1.0)).any()):
            raise PairedBaselineDiagnosticError(
                "out probability is outside [0, 1]"
            )
        loss = criterion(prediction.float(), target.float())
        probability = prediction[0, 0].float().cpu().numpy()
        target_array = target[0, 0].float().cpu().numpy()
        r1_metrics.update(probability, target_array, float(loss.item()))
        mechanism.update(probability, target_array)
        _update_array_digest(
            prediction_digest, sample_id=sample_id, array=probability
        )
        _update_array_digest(target_digest, sample_id=sample_id, array=target_array)
        sample_count += 1
    if sample_count < 1 or sample_count != len(loader.dataset):
        raise PairedBaselineDiagnosticError(
            "validation loader did not process the complete canonical split"
        )
    r1_result = dict(r1_metrics.compute())
    mechanism_result = dict(mechanism.compute())
    _canonical_json_bytes(r1_result)
    _canonical_json_bytes(mechanism_result)
    return (
        r1_result,
        mechanism_result,
        {
            "prediction_sha256": prediction_digest.hexdigest(),
            "target_sha256": target_digest.hexdigest(),
        },
        sample_count,
    )


def diagnostic_source_provenance() -> dict[str, Any]:
    source_paths = {
        "diagnostic_cli": Path(__file__),
        "r1_metrics": Path(r1.__file__),
        "matched_target_diagnostics": Path(complete_crop.__file__),
        "validation_data": Path(v2_data.__file__),
        "validation_selection": Path(validation_selection.__file__),
        "model_entry": Path(evisirst_model_module.__file__),
    }
    files: dict[str, dict[str, str]] = {}
    repository = PROJECT_ROOT.resolve(strict=True)
    for name, raw_path in source_paths.items():
        path = raw_path.resolve(strict=True)
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError as exc:
            raise PairedBaselineDiagnosticError(
                f"diagnostic source {name} is outside the repository"
            ) from exc
        if path.is_symlink() or not path.is_file():
            raise PairedBaselineDiagnosticError(
                f"diagnostic source {name} is not a regular file"
            )
        files[name] = {"relative_path": relative, "sha256": _sha256_file(path)}
    return {
        "schema": DIAGNOSTIC_SOURCE_SCHEMA,
        "files": files,
        "source_tree_sha256": _sha256_bytes(_canonical_json_bytes(files)),
    }


def build_result_payload(
    *,
    checkpoint_metadata: Mapping[str, Any],
    dataset: EviSIRSTV2ValDataset,
    r1_metrics: Mapping[str, Any],
    mechanism_metrics: Mapping[str, Any],
    digests: Mapping[str, str],
    sample_count: int,
    diagnostic_sources: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dataset.contract
    metrics = dict(r1_metrics)
    metrics["complete_target_mechanism_diagnostics"] = dict(mechanism_metrics)
    payload = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": DATA_ROLE,
        "diagnostic_only": True,
        "selection_allowed": False,
        "selected_on_same_validation_split": True,
        "checkpoint": {
            "relative_path": CHECKPOINT_RELATIVE_PATH,
            "sha256": checkpoint_metadata["sha256"],
            "schema": checkpoint_metadata["schema"],
            "checkpoint_role": "validation_selected",
            "selected_epoch": checkpoint_metadata["selected_epoch"],
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": RUN_SEED,
            "target_mode": TARGET_MODE,
            "training_identity_sha256": checkpoint_metadata[
                "training_identity_sha256"
            ],
            "selection_provenance_sha256": checkpoint_metadata[
                "selection_provenance_sha256"
            ],
            "strict_564_key_load": True,
        },
        "completed_run_summary": dict(checkpoint_metadata["summary"]),
        "split": {
            "manifest_relative_path": SPLIT_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": contract.manifest_sha256,
            "val_index_relative_path": VAL_INDEX_RELATIVE_PATH,
            "val_index_sha256": checkpoint_metadata["split_provenance"][
                "outputs"
            ]["val"]["file_sha256"],
            "data_tree_sha256": contract.data_tree_sha256,
            "data_tree_verified": contract.data_tree_verified,
            "train_count": len(contract.train_ids),
            "val_count": len(contract.val_ids),
        },
        "sources": {
            "checkpoint_training_source_tree_sha256": checkpoint_metadata[
                "training_source_tree_sha256"
            ],
            "checkpoint_training_sources_currently_verified": True,
            "diagnostic": dict(diagnostic_sources),
        },
        "evaluation": {
            "metrics_contract": r1.EVALUATION_PROTOCOL_VERSION,
            "evaluation_head": "out",
            "probability_threshold": r1.PROBABILITY_THRESHOLD,
            "probability_threshold_operator": r1.PREDICTION_THRESHOLD_OPERATOR,
            "target_threshold": r1.TARGET_THRESHOLD,
            "target_threshold_operator": r1.TARGET_THRESHOLD_OPERATOR,
            "match_radius": r1.MATCH_RADIUS,
            "match_distance_operator": r1.MATCH_DISTANCE_OPERATOR,
            "connected_component_connectivity": r1.CONNECTED_COMPONENT_CONNECTIVITY,
            "connected_component_neighborhood": r1.CONNECTED_COMPONENT_NEIGHBORHOOD,
            "assignment_algorithm": r1.ASSIGNMENT_ALGORITHM,
            "tiny_area": r1.TINY_AREA,
            "tiny_area_operator": r1.TINY_AREA_OPERATOR,
            "sample_count": sample_count,
            "one_model_forward_per_sample": True,
            "prediction_digest_schema": PREDICTION_DIGEST_SCHEMA,
            "prediction_sha256": digests["prediction_sha256"],
            "target_digest_schema": TARGET_DIGEST_SCHEMA,
            "target_sha256": digests["target_sha256"],
            "prediction_or_target_arrays_written": False,
        },
        "metrics": metrics,
        "test_split_accessed": False,
    }
    return json.loads(_canonical_json_bytes(payload).decode("ascii"))


def _write_fixed_json_atomic(payload: Mapping[str, Any]) -> Path:
    path = _run_artifact_path(OUTPUT_RELATIVE_PATH)
    if os.path.lexists(path):
        raise FileExistsError(
            f"paired baseline diagnostic already exists and is immutable: {path}"
        )
    parent = path.parent.resolve(strict=True)
    content = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-filesystem hard-link is an atomic create-if-absent publish:
            # unlike os.replace(), it can never overwrite a concurrently
            # committed immutable result.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"paired baseline diagnostic already exists and is immutable: {path}"
            ) from exc
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _run_locked(args: argparse.Namespace) -> Path:
    validate_optional_inherited_gpu_lock()
    output_path = _run_artifact_path(OUTPUT_RELATIVE_PATH)
    if os.path.lexists(output_path):
        raise FileExistsError(
            "paired baseline diagnostic already exists and is immutable: "
            f"{output_path}"
        )
    configure_inference_determinism()
    device = require_device(args.device)
    diagnostic_sources = diagnostic_source_provenance()
    model, checkpoint_metadata = load_fixed_final_model(device)
    dataset = build_validation_dataset(
        dataset_root=args.dataset_root, checkpoint_metadata=checkpoint_metadata
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
    )
    r1_metrics, mechanism_metrics, digests, sample_count = evaluate_out_once(
        model, loader, device
    )
    checkpoint_path = _fixed_path(CHECKPOINT_RELATIVE_PATH, must_exist=True)
    summary_path = _fixed_path(SUMMARY_RELATIVE_PATH, must_exist=True)
    if (
        _sha256_file(checkpoint_path) != checkpoint_metadata["sha256"]
        or _sha256_file(summary_path) != checkpoint_metadata["summary"]["sha256"]
        or r1._protocol_source_provenance()["source_tree_sha256"]
        != checkpoint_metadata["training_source_tree_sha256"]
        or diagnostic_source_provenance() != diagnostic_sources
    ):
        raise PairedBaselineDiagnosticError(
            "checkpoint, summary, or diagnostic source changed during evaluation"
        )
    payload = build_result_payload(
        checkpoint_metadata=checkpoint_metadata,
        dataset=dataset,
        r1_metrics=r1_metrics,
        mechanism_metrics=mechanism_metrics,
        digests=digests,
        sample_count=sample_count,
        diagnostic_sources=diagnostic_sources,
    )
    return _write_fixed_json_atomic(payload)


def run(args: argparse.Namespace) -> Path:
    with exclusive_run_lock():
        return _run_locked(args)


def main(argv: Sequence[str] | None = None) -> None:
    output = run(parse_args(argv))
    print(output.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_RELATIVE_PATH",
    "DATASET",
    "DIAGNOSTIC_SCHEMA",
    "GPU_LOCK_FD_ENV",
    "GPU_UUID_ENV",
    "OUTPUT_RELATIVE_PATH",
    "PairedBaselineDiagnosticError",
    "RUN_LOCK_FD_ENV",
    "RUN_LOCK_RELATIVE_PATH",
    "RUN_SEED",
    "SUMMARY_RELATIVE_PATH",
    "build_result_payload",
    "build_validation_dataset",
    "diagnostic_source_provenance",
    "evaluate_out_once",
    "exclusive_run_lock",
    "load_fixed_final_model",
    "parse_args",
    "run",
    "validate_optional_inherited_gpu_lock",
]
