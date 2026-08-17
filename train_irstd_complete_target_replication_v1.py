#!/usr/bin/env python3
"""Run the two authorized IRSTD-1K complete-target-v1 replications.

The completed seed-1446202191 pilot is immutable and is not repeated here.
This entry admits only runtime seeds 104728269 and 262620274, while retaining
architecture seed 42 and every training, validation, crop, optimizer, and
selection choice of ``train_irstd_complete_target_v1.py``.

Formal execution is fail-closed.  Before creating an output directory,
acquiring a run lock, or entering the CUDA training engine, the runner freshly
calls the canonical single-seed gate validator and accepts only its exact PASS
decision authorizing the three-runtime-seed validation expansion.  The full
canonical payload, its canonical SHA-256, the immutable result-file SHA-256,
and the frozen parent source set are embedded in every run identity.  Resume,
candidate, checkpoint, and summary artifacts therefore inherit that binding.
Public-test loading is neither imported nor supported.
"""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

import run_irstd_complete_target_promotion_gate as canonical_gate
import train_irstd_complete_target_v1 as pilot
import train_validation_selected as r1
from experiments import evisirst_complete_target_crop as complete_crop
from experiments.evisirst_v2_data import V2SplitContract


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "irstd_performance"
    / "complete_target_v1"
    / "three_runtime_seed_validation"
)
DATASET = pilot.DATASET
TARGET_MODE = pilot.TARGET_MODE
ARCHITECTURE_SEED = pilot.ARCHITECTURE_SEED
PILOT_RUN_SEED = pilot.PAIRED_RUN_SEED
REPLICATION_RUN_SEEDS = (104728269, 262620274)
THREE_RUNTIME_SEEDS = (PILOT_RUN_SEED, *REPLICATION_RUN_SEEDS)
FORMAL_EPOCHS = pilot.FORMAL_EPOCHS
FORMAL_BATCH_SIZE = pilot.FORMAL_BATCH_SIZE
FORMAL_WORKERS = pilot.FORMAL_WORKERS
FORMAL_BASE_LR = pilot.FORMAL_BASE_LR
FORMAL_MIN_LR = pilot.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = pilot.FORMAL_WARMUP_EPOCHS
FORMAL_VAL_INTERVAL = pilot.FORMAL_VAL_INTERVAL
CANONICAL_SPLIT_ROOT = pilot.CANONICAL_SPLIT_ROOT
CANONICAL_IRSTD_MANIFEST_SHA256 = pilot.CANONICAL_IRSTD_MANIFEST_SHA256
CANONICAL_IRSTD_DATA_TREE_SHA256 = pilot.CANONICAL_IRSTD_DATA_TREE_SHA256
CANONICAL_IRSTD_TRAIN_COUNT = pilot.CANONICAL_IRSTD_TRAIN_COUNT
CANONICAL_IRSTD_VAL_COUNT = pilot.CANONICAL_IRSTD_VAL_COUNT

TRAINING_SCHEMA = "evisirst_irstd_complete_target_replication_training/v1"
CANDIDATE_SCHEMA = "evisirst_irstd_complete_target_replication_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_irstd_complete_target_replication_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_irstd_complete_target_replication_history/v1"
SELECTION_PAYLOAD_SCHEMA = (
    "evisirst_irstd_complete_target_replication_selection_payload/v1"
)
SOURCE_SET_SCHEMA = "evisirst_irstd_complete_target_replication_source_set/v1"
DETERMINISM_SCHEMA = (
    "evisirst_irstd_complete_target_replication_determinism/v1"
)
EXPERIMENT_SCHEMA = "evisirst_irstd_complete_target_replication_experiment/v1"
AUTHORIZATION_SCHEMA = (
    "evisirst_irstd_complete_target_replication_authorization/v1"
)
_CANDIDATE_KEYS = frozenset(
    {
        "schema",
        "model",
        "dataset",
        "epoch",
        "run_identity",
        "validation_record",
        "state_dict",
        "test_split_accessed",
        "experiment_schema",
        "experiment_status",
        "public_test_supported",
        "public_test_gate_status",
        "crop_audit_state",
        "crop_audit_summary",
        "crop_audit_history",
        "crop_audit_commit_status",
    }
)
PARENT_SOURCE_SET_SCHEMA = pilot.SOURCE_SET_SCHEMA
CANONICAL_GATE_RESULT_SCHEMA = canonical_gate.RESULT_SCHEMA
CANONICAL_GATE_RESULT_RELATIVE_PATH = canonical_gate.OUTPUT_RELATIVE_PATH

# These hashes are the immutable implementation used by the completed pilot.
# The full source-tree hash binds every R1 source, not only the three entries
# repeated below for human-auditable failure messages.
FROZEN_PARENT_SOURCE_TREE_SHA256 = (
    "1fedd0dfa1a4094c606dc6e783e62ee8cb89879989e8dd785cb836e1a45bcd96"
)
FROZEN_PILOT_RUNNER_SHA256 = (
    "ed32dc15245b8ca3f3d4503224f4984c37cb83b562908232a13329f78d1c5445"
)
FROZEN_COMPLETE_TARGET_CROP_SHA256 = (
    "c8f3fc0f9169667c21ac0d40424d5b5acd4a3d9447c9362a518882f37209ec08"
)
FROZEN_R1_TRAINER_SHA256 = (
    "6f870677f7724054fc8bc71e07ddde46ed3d316eb837f953be50fe8c99c6e0ed"
)
FROZEN_CANONICAL_GATE_SOURCE_SHA256 = (
    "b1e9ec94b2300759ff8a936ec7fca2b63699998f901de0dccbbe26dae7c77259"
)


class CompleteTargetReplicationError(pilot.CompleteTargetRunnerError):
    """The requested run violates the confirmatory replication contract."""


# Capture the completed pilot implementation before any bounded patch begins.
_PILOT_SOURCE_PROVENANCE = pilot._variant_source_provenance
_PILOT_DETERMINISM_PROTOCOL_IDENTITY = pilot._variant_determinism_protocol_identity
_PILOT_RUN_IDENTITY = pilot._variant_run_identity
_FROZEN_PARENT_SOURCE_SNAPSHOT = _PILOT_SOURCE_PROVENANCE()
_PATCH_LOCK = threading.Lock()
_RUNTIME_STATE = threading.local()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompleteTargetReplicationError(
            "replication metadata must be strict finite JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(data: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise CompleteTargetReplicationError(f"{label} is not strict JSON") from exc
    if type(value) is not dict:
        raise CompleteTargetReplicationError(f"{label} must be a JSON object")
    # Reject values such as 1e999 which the JSON parser converts to infinity.
    _canonical_json_bytes(value)
    return value


def _read_regular_file_stably(path: Path, *, label: str) -> tuple[bytes, str]:
    """Read one non-symlink repository file through a stable descriptor."""

    if path.is_symlink() or not path.is_file():
        raise CompleteTargetReplicationError(f"{label} is not a regular file")
    try:
        path.resolve(strict=True).relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise CompleteTargetReplicationError(
            f"{label} escaped the repository"
        ) from exc
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompleteTargetReplicationError(f"{label} is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise CompleteTargetReplicationError(f"{label} changed while read")
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise CompleteTargetReplicationError(f"{label} size changed while read")
    return data, hashlib.sha256(data).hexdigest()


def _load_torch_regular_stably(
    path: Path, *, label: str
) -> tuple[Mapping[str, Any], str]:
    """Hash and CPU-load one regular torch artifact through one descriptor."""

    if path.is_symlink() or not path.is_file():
        raise CompleteTargetReplicationError(f"{label} is not a regular file")
    try:
        path.resolve(strict=True).relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise CompleteTargetReplicationError(
            f"{label} escaped the repository"
        ) from exc
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompleteTargetReplicationError(f"{label} is not regular")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            try:
                value = torch.load(handle, map_location="cpu", weights_only=True)
            except (EOFError, OSError, RuntimeError, ValueError) as exc:
                raise CompleteTargetReplicationError(
                    f"{label} failed safe CPU loading"
                ) from exc
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise CompleteTargetReplicationError(f"{label} changed while loaded")
    finally:
        os.close(descriptor)
    if not isinstance(value, Mapping):
        raise CompleteTargetReplicationError(f"{label} payload is malformed")
    return value, digest.hexdigest()


def _frozen_parent_source_set() -> dict[str, Any]:
    # The captured function consults pilot module globals.  Snapshot its exact
    # completed-pilot value at import so our own bounded schema patch cannot
    # make the verifier accidentally observe replication schema constants.
    source = _json_clone(_FROZEN_PARENT_SOURCE_SNAPSHOT)
    files = source.get("files")
    expected_entries = {
        "variant_runner": (
            "train_irstd_complete_target_v1.py",
            FROZEN_PILOT_RUNNER_SHA256,
        ),
        "complete_target_crop": (
            "experiments/evisirst_complete_target_crop.py",
            FROZEN_COMPLETE_TARGET_CROP_SHA256,
        ),
        "r1/trainer": (
            "train_validation_selected.py",
            FROZEN_R1_TRAINER_SHA256,
        ),
    }
    if (
        source.get("schema") != PARENT_SOURCE_SET_SCHEMA
        or type(files) is not dict
        or source.get("source_tree_sha256")
        != FROZEN_PARENT_SOURCE_TREE_SHA256
    ):
        raise CompleteTargetReplicationError(
            "completed pilot parent source tree differs from its frozen hash"
        )
    for name, (relative_path, sha256) in expected_entries.items():
        if files.get(name) != {
            "relative_path": relative_path,
            "sha256": sha256,
        }:
            raise CompleteTargetReplicationError(
                f"completed pilot parent source differs at {name}"
            )
    # Verify every recorded file against the repository, including all R1
    # model/data/selection sources represented by the frozen tree hash.
    for name, artifact in files.items():
        if type(name) is not str or type(artifact) is not dict:
            raise CompleteTargetReplicationError("parent source set is malformed")
        relative = artifact.get("relative_path")
        expected_sha = artifact.get("sha256")
        if (
            type(relative) is not str
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or type(expected_sha) is not str
        ):
            raise CompleteTargetReplicationError(
                f"parent source metadata is malformed at {name}"
            )
        source_path = PROJECT_ROOT / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise CompleteTargetReplicationError(
                f"parent source is not a regular file at {name}"
            )
        if _sha256_file(source_path) != expected_sha:
            raise CompleteTargetReplicationError(
                f"parent source SHA-256 differs at {name}"
            )
    return source


def _require_canonical_pass_payload(payload: Any) -> dict[str, Any]:
    if type(payload) is not dict:
        raise CompleteTargetReplicationError(
            "canonical promotion-gate payload is malformed"
        )
    fixed = {
        "schema": CANONICAL_GATE_RESULT_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": "val",
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": PILOT_RUN_SEED,
        "epochs": FORMAL_EPOCHS,
        "test_split_accessed": False,
        "public_test_allowed": False,
    }
    for key, expected in fixed.items():
        if type(payload.get(key)) is not type(expected) or payload.get(key) != expected:
            raise CompleteTargetReplicationError(
                f"canonical promotion-gate payload differs at {key}"
            )
    inputs = payload.get("inputs")
    if type(inputs) is not dict or any(
        inputs.get(key) is not True
        for key in (
            "all_artifact_hashes_verified",
            "all_source_hashes_currently_verified",
            "canonical_train_val_split_verified",
            "both_selections_recomputed",
        )
    ):
        raise CompleteTargetReplicationError(
            "canonical promotion-gate verification evidence is incomplete"
        )
    gate_source = inputs.get("gate_evaluator_source")
    if gate_source != {
        "relative_path": canonical_gate.GATE_SOURCE_RELATIVE_PATH,
        "sha256": FROZEN_CANONICAL_GATE_SOURCE_SHA256,
    }:
        raise CompleteTargetReplicationError(
            "canonical promotion-gate evaluator source binding differs"
        )
    decision = payload.get("decision")
    if (
        type(decision) is not dict
        or decision.get("result") != "PASS"
        or decision.get("overall_passed") is not True
        or decision.get("three_runtime_seed_validation_expansion_allowed")
        is not True
        or decision.get("three_runtime_seed_validation_expansion_status")
        != "allowed"
        or decision.get("public_test_allowed") is not False
    ):
        raise CompleteTargetReplicationError(
            "canonical gate did not authorize runtime-seed replication"
        )
    return _json_clone(payload)


def validate_canonical_expansion_authorization() -> dict[str, Any]:
    """Freshly validate and fully bind the canonical PASS authorization."""

    if Path(canonical_gate.PROJECT_ROOT).resolve() != PROJECT_ROOT.resolve():
        raise CompleteTargetReplicationError(
            "canonical promotion-gate evaluator belongs to another repository"
        )
    gate_source_path = Path(canonical_gate.__file__).resolve(strict=True)
    if _sha256_file(gate_source_path) != FROZEN_CANONICAL_GATE_SOURCE_SHA256:
        raise CompleteTargetReplicationError(
            "canonical promotion-gate evaluator source differs"
        )
    try:
        validated = canonical_gate.validate_existing_result()
    except (FileNotFoundError, OSError, canonical_gate.PromotionGateError) as exc:
        raise CompleteTargetReplicationError(
            "canonical complete-target PASS result is missing or invalid"
        ) from exc
    payload = _require_canonical_pass_payload(validated)

    result_path = PROJECT_ROOT / CANONICAL_GATE_RESULT_RELATIVE_PATH
    result_bytes, result_sha256 = _read_regular_file_stably(
        result_path, label="canonical promotion-gate result"
    )
    observed = _strict_json_object(
        result_bytes, label="canonical promotion-gate result"
    )
    if _canonical_json_bytes(observed) != _canonical_json_bytes(payload):
        raise CompleteTargetReplicationError(
            "canonical promotion-gate result changed after fresh validation"
        )

    parent_sources = _frozen_parent_source_set()
    evidence = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "complete",
        "authority": (
            "run_irstd_complete_target_promotion_gate."
            "validate_existing_result"
        ),
        "canonical_result_artifact": {
            "relative_path": CANONICAL_GATE_RESULT_RELATIVE_PATH,
            "sha256": result_sha256,
        },
        "canonical_payload_schema": CANONICAL_GATE_RESULT_SCHEMA,
        "canonical_payload_sha256": _canonical_sha256(payload),
        "canonical_payload": payload,
        "canonical_gate_evaluator_source": {
            "relative_path": canonical_gate.GATE_SOURCE_RELATIVE_PATH,
            "sha256": FROZEN_CANONICAL_GATE_SOURCE_SHA256,
        },
        "frozen_parent_source_set": parent_sources,
        "frozen_parent_source_tree_sha256": FROZEN_PARENT_SOURCE_TREE_SHA256,
        "completed_pilot_run_seed": PILOT_RUN_SEED,
        "authorized_three_runtime_seeds": list(THREE_RUNTIME_SEEDS),
        "pending_replication_run_seeds": list(REPLICATION_RUN_SEEDS),
        "formal_replication_allowed": True,
        "public_test_supported": False,
        "public_test_accessed": False,
    }
    return _json_clone(evidence)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-root", type=Path, default=CANONICAL_SPLIT_ROOT
    )
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument("--target-mode", choices=(TARGET_MODE,), default=TARGET_MODE)
    parser.add_argument(
        "--architecture-seed",
        type=int,
        choices=(ARCHITECTURE_SEED,),
        default=ARCHITECTURE_SEED,
    )
    parser.add_argument(
        "--run-seed", type=int, choices=REPLICATION_RUN_SEEDS, required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-sample-level-fallback", action="store_true")
    args = parser.parse_args(argv)

    if args.epochs != FORMAL_EPOCHS:
        parser.error(f"formal replication requires --epochs {FORMAL_EPOCHS}")
    if args.warmup_epochs not in (None, FORMAL_WARMUP_EPOCHS):
        parser.error(
            "formal replication requires --warmup-epochs "
            f"{FORMAL_WARMUP_EPOCHS}"
        )
    args.warmup_epochs = FORMAL_WARMUP_EPOCHS
    args.output_root = DEFAULT_OUTPUT_ROOT
    args.batch_size = FORMAL_BATCH_SIZE
    args.workers = FORMAL_WORKERS
    args.base_lr = FORMAL_BASE_LR
    args.min_lr = FORMAL_MIN_LR
    args.val_interval = FORMAL_VAL_INTERVAL
    # The reused engine expects these attributes even though this formal-only
    # entry exposes no smoke bypass.
    args.smoke_max_train_samples = None
    args.smoke_max_val_samples = None
    try:
        _require_replication_args(args)
    except CompleteTargetReplicationError as exc:
        parser.error(str(exc))
    return args


def _require_replication_args(args: argparse.Namespace) -> None:
    exact = {
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "val_interval": FORMAL_VAL_INTERVAL,
        "device": "cuda:0",
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
    }
    for name, expected in exact.items():
        observed = getattr(args, name, None)
        if type(observed) is not type(expected) or observed != expected:
            raise CompleteTargetReplicationError(
                f"complete-target replication freezes {name}={expected!r}"
            )
    run_seed = getattr(args, "run_seed", None)
    if isinstance(run_seed, bool) or run_seed not in REPLICATION_RUN_SEEDS:
        raise CompleteTargetReplicationError(
            "formal replication run_seed must be one of "
            f"{REPLICATION_RUN_SEEDS}; completed pilot seed {PILOT_RUN_SEED} "
            "must not be repeated"
        )
    if Path(args.output_root).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise CompleteTargetReplicationError(
            "replication output root is fixed under complete_target_v1/"
            "three_runtime_seed_validation"
        )
    supplied_split_root = Path(os.path.abspath(args.split_root))
    canonical_split_root = Path(os.path.abspath(CANONICAL_SPLIT_ROOT))
    if supplied_split_root != canonical_split_root:
        raise CompleteTargetReplicationError(
            "formal replication requires the canonical splits/v2 root"
        )
    manifest = canonical_split_root / DATASET / "manifest.json"
    if (
        canonical_split_root.is_symlink()
        or not canonical_split_root.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
        or _sha256_file(manifest) != CANONICAL_IRSTD_MANIFEST_SHA256
    ):
        raise CompleteTargetReplicationError(
            "canonical IRSTD-1K split manifest identity differs"
        )


def _active_authorization() -> dict[str, Any]:
    evidence = getattr(_RUNTIME_STATE, "authorization", None)
    if (
        type(evidence) is not dict
        or evidence.get("schema") != AUTHORIZATION_SCHEMA
        or evidence.get("status") != "complete"
        or evidence.get("formal_replication_allowed") is not True
        or evidence.get("public_test_supported") is not False
        or evidence.get("public_test_accessed") is not False
        or evidence.get("frozen_parent_source_tree_sha256")
        != FROZEN_PARENT_SOURCE_TREE_SHA256
    ):
        raise CompleteTargetReplicationError(
            "active canonical expansion authorization is unavailable"
        )
    return _json_clone(evidence)


def promotion_gate() -> dict[str, Any]:
    """Return the exact active PASS authorization embedded in artifacts."""

    return _active_authorization()


def _replication_source_provenance_from_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = _json_clone(authorization)
    parent_sources = authorization.get("frozen_parent_source_set")
    if (
        type(parent_sources) is not dict
        or parent_sources.get("source_tree_sha256")
        != FROZEN_PARENT_SOURCE_TREE_SHA256
        or parent_sources.get("schema") != PARENT_SOURCE_SET_SCHEMA
        or type(parent_sources.get("files")) is not dict
    ):
        raise CompleteTargetReplicationError(
            "active frozen parent source set is malformed"
        )
    # Recompute before every identity construction so a source change between
    # gate validation and training cannot be silently accepted.
    if _frozen_parent_source_set() != parent_sources:
        raise CompleteTargetReplicationError(
            "parent source set changed after gate authorization"
        )
    files = {
        f"parent/{name}": copy.deepcopy(artifact)
        for name, artifact in sorted(parent_sources["files"].items())
    }
    runner_path = Path(__file__)
    if runner_path.is_symlink() or not runner_path.is_file():
        raise CompleteTargetReplicationError(
            "replication runner is not a regular source file"
        )
    files["replication_runner"] = {
        "relative_path": runner_path.resolve(strict=True)
        .relative_to(PROJECT_ROOT.resolve(strict=True))
        .as_posix(),
        "sha256": _sha256_file(runner_path),
    }
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
        "frozen_parent_source_tree_sha256": FROZEN_PARENT_SOURCE_TREE_SHA256,
    }


def _replication_source_provenance() -> dict[str, Any]:
    return _replication_source_provenance_from_authorization(
        _active_authorization()
    )


def _replication_determinism_protocol_identity() -> dict[str, Any]:
    identity = copy.deepcopy(_PILOT_DETERMINISM_PROTOCOL_IDENTITY())
    source = _replication_source_provenance()
    identity["schema"] = DETERMINISM_SCHEMA
    identity["replication_contract"] = {
        "schema": EXPERIMENT_SCHEMA + "/runtime_seed_contract",
        "architecture_seed": ARCHITECTURE_SEED,
        "completed_pilot_run_seed": PILOT_RUN_SEED,
        "authorized_three_runtime_seeds": list(THREE_RUNTIME_SEEDS),
        "pending_replication_run_seeds": list(REPLICATION_RUN_SEEDS),
        "single_variable_from_pilot": "runtime_seed",
        "public_test_supported": False,
    }
    identity["source_set_schema"] = source["schema"]
    identity["source_files"] = source["files"]
    identity["source_tree_sha256"] = source["source_tree_sha256"]
    identity["frozen_parent_source_tree_sha256"] = (
        FROZEN_PARENT_SOURCE_TREE_SHA256
    )
    return identity


def _replication_run_identity(
    args: argparse.Namespace,
    contract: V2SplitContract,
    grouping: Mapping[str, Any],
    *,
    train_count: int,
    val_count: int,
    smoke: bool,
) -> dict[str, Any]:
    if smoke:
        raise CompleteTargetReplicationError(
            "the confirmatory replication entry has no smoke mode"
        )
    _require_replication_args(args)
    identity = _PILOT_RUN_IDENTITY(
        args,
        contract,
        grouping,
        train_count=train_count,
        val_count=val_count,
        smoke=False,
    )
    identity["schema"] = TRAINING_SCHEMA + "/run_identity"
    identity["experiment"] = {
        "schema": EXPERIMENT_SCHEMA,
        "name": "IRSTD-1K complete-target crop v1 runtime-seed replication",
        "status": "confirmatory_validation_only",
        "single_variable_from_completed_pilot": "runtime_seed",
        "architecture_seed_fixed": ARCHITECTURE_SEED,
        "completed_pilot_run_seed": PILOT_RUN_SEED,
        "authorized_three_runtime_seeds": list(THREE_RUNTIME_SEEDS),
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_pending_separate_review",
    }
    identity["execution_contract"] = {
        "single_process_only": True,
        "python_threads_running_variant": 1,
        "data_loader_workers": FORMAL_WORKERS,
        "nonblocking_process_lock": ".complete_target_replication_v1.lock",
    }
    identity["promotion_gate"] = promotion_gate()
    identity.pop("identity_sha256", None)
    identity = _json_clone(identity)
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


@contextmanager
def _replication_runtime(authorization: Mapping[str, Any]) -> Iterator[None]:
    """Exception-safely route the frozen pilot/R1 engine to replication."""

    if not _PATCH_LOCK.acquire(blocking=False):
        raise CompleteTargetReplicationError(
            "complete-target replication runtime is already active"
        )
    pilot_lock_acquired = False
    previous_pilot: dict[str, Any] = {}
    previous_r1: dict[str, Any] = {}
    try:
        pilot_lock_acquired = pilot._PATCH_LOCK.acquire(blocking=False)
        if not pilot_lock_acquired:
            raise CompleteTargetReplicationError(
                "complete-target pilot runtime is already active"
            )
        pilot_overrides = {
            "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
            "TRAINING_SCHEMA": TRAINING_SCHEMA,
            "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
            "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
            "HISTORY_SCHEMA": HISTORY_SCHEMA,
            "SELECTION_PAYLOAD_SCHEMA": SELECTION_PAYLOAD_SCHEMA,
            "SOURCE_SET_SCHEMA": SOURCE_SET_SCHEMA,
            "DETERMINISM_SCHEMA": DETERMINISM_SCHEMA,
            "EXPERIMENT_SCHEMA": EXPERIMENT_SCHEMA,
            "_require_variant_args": _require_replication_args,
            "promotion_gate": promotion_gate,
            "_variant_source_provenance": _replication_source_provenance,
            "_variant_determinism_protocol_identity": (
                _replication_determinism_protocol_identity
            ),
            "_variant_run_identity": _replication_run_identity,
        }
        previous_pilot = {
            name: getattr(pilot, name) for name in pilot_overrides
        }
        for name, value in pilot_overrides.items():
            setattr(pilot, name, value)

        r1_overrides = {
            "TRAINING_SCHEMA": TRAINING_SCHEMA,
            "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
            "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
            "HISTORY_SCHEMA": HISTORY_SCHEMA,
            "SELECTION_PAYLOAD_SCHEMA": SELECTION_PAYLOAD_SCHEMA,
            "ValidationMetrics": pilot.CompleteTargetValidationMetrics,
            "build_datasets": pilot._engine_build_datasets,
            "_protocol_source_provenance": _replication_source_provenance,
            "_determinism_protocol_identity": (
                _replication_determinism_protocol_identity
            ),
            "_run_identity": _replication_run_identity,
            "build_final_checkpoint_payload": pilot._variant_final_checkpoint_payload,
            "_load_resume_state": pilot._variant_load_resume_state,
            "_validate_candidate_payload": pilot._variant_validate_candidate_payload,
            "_load_selected_candidate": pilot._variant_load_selected_candidate,
            "_write_json": pilot._variant_write_json,
            "_atomic_torch_save": pilot._variant_atomic_torch_save,
        }
        previous_r1 = {name: getattr(r1, name) for name in r1_overrides}
        _RUNTIME_STATE.authorization = _json_clone(authorization)
        pilot._RUNTIME_STATE.crop_audit_history = []
        pilot._RUNTIME_STATE.train_dataset = None
        pilot._RUNTIME_STATE.run_identity = None
        pilot._RUNTIME_STATE.selected_candidate_evidence = None
        for name, value in r1_overrides.items():
            setattr(r1, name, value)
        yield
    finally:
        for name, value in previous_r1.items():
            setattr(r1, name, value)
        for name, value in previous_pilot.items():
            setattr(pilot, name, value)
        for name in (
            "crop_audit_history",
            "train_dataset",
            "run_identity",
            "selected_candidate_evidence",
        ):
            if hasattr(pilot._RUNTIME_STATE, name):
                delattr(pilot._RUNTIME_STATE, name)
        if hasattr(_RUNTIME_STATE, "authorization"):
            delattr(_RUNTIME_STATE, "authorization")
        if pilot_lock_acquired:
            pilot._PATCH_LOCK.release()
        _PATCH_LOCK.release()


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path | bool]:
    _require_replication_args(args)
    paths = r1.resolve_run_paths(args)
    if paths.get("smoke") is not False:
        raise CompleteTargetReplicationError(
            "replication output unexpectedly entered a smoke branch"
        )
    run_dir = paths.get("run_dir")
    expected = (
        DEFAULT_OUTPUT_ROOT
        / "formal"
        / DATASET
        / TARGET_MODE
        / f"run_seed_{args.run_seed}"
    ).resolve()
    if not isinstance(run_dir, Path) or run_dir != expected:
        raise CompleteTargetReplicationError(
            "replication run directory differs from its fixed path"
        )
    return paths


@contextmanager
def _run_process_lock(args: argparse.Namespace) -> Iterator[Path]:
    paths = resolve_run_paths(args)
    run_dir = paths["run_dir"]
    if not isinstance(run_dir, Path):
        raise CompleteTargetReplicationError("replication run directory is malformed")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".complete_target_replication_v1.lock"
    if lock_path.is_symlink():
        raise CompleteTargetReplicationError("replication lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise CompleteTargetReplicationError(
                    f"replication run is already locked: {run_dir}"
                ) from exc
            raise
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def run(args: argparse.Namespace) -> Path:
    """Execute one authorized formal replication through the frozen engine."""

    _require_replication_args(args)
    # This ordering is part of the security/reproducibility contract: the
    # canonical gate and frozen sources are freshly checked before the first
    # output-directory mutation and before R1 can call require_device().
    authorization = validate_canonical_expansion_authorization()
    with _run_process_lock(args), _replication_runtime(authorization):
        checkpoint = r1.run(args)
    return checkpoint


def formal_artifact_contract(run_seed: int) -> dict[str, Any]:
    """Return fixed paths/schemas for an aggregate gate without reading data."""

    if isinstance(run_seed, bool) or run_seed not in REPLICATION_RUN_SEEDS:
        raise CompleteTargetReplicationError(
            f"replication artifact seed must be one of {REPLICATION_RUN_SEEDS}"
        )
    base = (
        "runs/irstd_performance/complete_target_v1/"
        "three_runtime_seed_validation/formal/IRSTD-1K/binary/"
        f"run_seed_{run_seed}"
    )
    return {
        "run_seed": run_seed,
        "summary_relative_path": f"{base}/summary.json",
        "checkpoint_relative_path": f"{base}/EviSIRST.pth.tar",
        "summary_schema": TRAINING_SCHEMA + "/summary",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "run_identity_schema": TRAINING_SCHEMA + "/run_identity",
        "experiment_schema": EXPERIMENT_SCHEMA,
        "source_set_schema": SOURCE_SET_SCHEMA,
        "authorization_schema": AUTHORIZATION_SCHEMA,
        "test_split_accessed": False,
        "public_test_supported": False,
    }


def _require_no_public_test_access(value: Any, *, path: str = "root") -> None:
    """Recursively reject any positive public-test access/permission marker."""

    false_only = {
        "test_split_accessed",
        "test_index_opened",
        "public_test_accessed",
        "public_test_allowed",
        "public_test_supported",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in false_only and child is not False:
                raise CompleteTargetReplicationError(
                    f"public-test marker must be false at {path}.{key}"
                )
            _require_no_public_test_access(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_no_public_test_access(child, path=f"{path}[{index}]")


def _validate_final_state_dict(
    value: Any, *, authorization: Mapping[str, Any]
) -> int:
    if not isinstance(value, Mapping) or len(value) != 564:
        raise CompleteTargetReplicationError(
            "completed replication state_dict is not the clean 564-key graph"
        )
    canonical_artifact = (
        authorization.get("canonical_payload", {})
        .get("inputs", {})
        .get("complete_target_v1", {})
        .get("final_checkpoint")
    )
    if (
        not isinstance(canonical_artifact, Mapping)
        or type(canonical_artifact.get("relative_path")) is not str
        or type(canonical_artifact.get("sha256")) is not str
    ):
        raise CompleteTargetReplicationError(
            "canonical pilot model-structure reference is missing"
        )
    reference_path = PROJECT_ROOT / canonical_artifact["relative_path"]
    reference_payload, reference_sha256 = _load_torch_regular_stably(
        reference_path, label="canonical pilot final checkpoint"
    )
    if reference_sha256 != canonical_artifact["sha256"]:
        raise CompleteTargetReplicationError(
            "canonical pilot final checkpoint SHA-256 differs"
        )
    reference_state = reference_payload.get("state_dict")
    if not isinstance(reference_state, Mapping) or len(reference_state) != 564:
        raise CompleteTargetReplicationError(
            "canonical pilot model-structure reference is malformed"
        )
    if set(value) != set(reference_state):
        raise CompleteTargetReplicationError(
            "completed replication state_dict key names differ"
        )
    _validate_state_dict_against_reference(
        value,
        reference_state,
        label="completed replication state_dict",
        require_tensor_equality=False,
    )
    return len(value)


def _validate_state_dict_against_reference(
    value: Any,
    reference_state: Any,
    *,
    label: str,
    require_tensor_equality: bool,
) -> None:
    """Validate exact keys/shape/dtype/layout/finite, optionally exact values."""

    if (
        not isinstance(value, Mapping)
        or not isinstance(reference_state, Mapping)
        or set(value) != set(reference_state)
    ):
        raise CompleteTargetReplicationError(f"{label} key names differ")
    for key, tensor in value.items():
        reference = reference_state[key]
        if (
            type(key) is not str
            or key.startswith("target_survival")
            or not isinstance(tensor, torch.Tensor)
            or not isinstance(reference, torch.Tensor)
        ):
            raise CompleteTargetReplicationError(
                f"{label} is malformed"
            )
        if (
            tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
            or tensor.layout != reference.layout
            or tensor.is_sparse
        ):
            raise CompleteTargetReplicationError(
                f"{label} tensor contract differs: {key}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise CompleteTargetReplicationError(
                f"{label} tensor is non-finite: {key}"
            )
        if require_tensor_equality and not torch.equal(tensor, reference):
            raise CompleteTargetReplicationError(
                f"{label} tensor values differ: {key}"
            )


def _validate_retained_candidates(
    *,
    run_dir: Path,
    identity: Mapping[str, Any],
    validation_history: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    candidate_artifacts: Any,
    final_state: Mapping[str, torch.Tensor],
    selected_audit: Mapping[str, Any],
    selected_candidate_sha256: Any,
) -> list[dict[str, Any]]:
    """Strictly validate the complete recomputed retention frontier on disk."""

    frontier = r1.selection.retention_frontier_epochs(validation_history)
    if (
        not frontier
        or selection.get("retention_frontier_epochs") != list(frontier)
    ):
        raise CompleteTargetReplicationError(
            "completed replication retention frontier differs"
        )
    selected_epoch = selection.get("selected_epoch")
    if isinstance(selected_epoch, bool) or selected_epoch not in frontier:
        raise CompleteTargetReplicationError(
            "completed replication selected epoch is outside the frontier"
        )
    expected_keys = {str(epoch) for epoch in frontier}
    if type(candidate_artifacts) is not dict or set(candidate_artifacts) != expected_keys:
        raise CompleteTargetReplicationError(
            "completed replication candidate artifact map differs"
        )

    candidate_dir = run_dir / "candidates"
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise CompleteTargetReplicationError(
            "completed replication candidate directory is not a real directory"
        )
    resolved_run = run_dir.resolve(strict=True)
    resolved_candidate_dir = candidate_dir.resolve(strict=True)
    try:
        resolved_candidate_dir.relative_to(resolved_run)
    except ValueError as exc:
        raise CompleteTargetReplicationError(
            "completed replication candidate directory escaped the run"
        ) from exc

    expected_paths: dict[int, Path] = {}
    normalized_artifacts: dict[int, dict[str, str]] = {}
    for epoch in frontier:
        filename = f"epoch_{epoch:04d}.pth.tar"
        relative_path = f"candidates/{filename}"
        entry = candidate_artifacts.get(str(epoch))
        if (
            type(entry) is not dict
            or set(entry) != {"relative_path", "file_sha256"}
            or entry.get("relative_path") != relative_path
            or type(entry.get("file_sha256")) is not str
            or len(entry["file_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entry["file_sha256"]
            )
        ):
            raise CompleteTargetReplicationError(
                f"completed replication candidate metadata differs at epoch {epoch}"
            )
        expected_paths[epoch] = candidate_dir / filename
        normalized_artifacts[epoch] = dict(entry)

    actual_paths: set[Path] = set()
    for path in candidate_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise CompleteTargetReplicationError(
                "completed replication candidate directory contains a non-regular entry"
            )
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_candidate_dir)
        except ValueError as exc:
            raise CompleteTargetReplicationError(
                "completed replication candidate escaped its directory"
            ) from exc
        actual_paths.add(resolved)
    expected_resolved = {
        path.resolve(strict=False) for path in expected_paths.values()
    }
    if actual_paths != expected_resolved:
        raise CompleteTargetReplicationError(
            "completed replication candidate directory has missing or extra files"
        )

    selected_artifact = normalized_artifacts[selected_epoch]
    if (
        selection.get("selected_candidate") != selected_artifact
        or selected_candidate_sha256 != selected_artifact["file_sha256"]
    ):
        raise CompleteTargetReplicationError(
            "completed replication selected candidate metadata differs"
        )
    records_by_epoch = {
        int(record["epoch"]): record for record in validation_history
    }
    evidence: list[dict[str, Any]] = []
    for epoch in frontier:
        path = expected_paths[epoch]
        candidate, observed_sha256 = _load_torch_regular_stably(
            path, label=f"retained candidate epoch {epoch}"
        )
        artifact = normalized_artifacts[epoch]
        if observed_sha256 != artifact["file_sha256"]:
            raise CompleteTargetReplicationError(
                f"retained candidate SHA-256 differs at epoch {epoch}"
            )
        fixed = {
            "schema": CANDIDATE_SCHEMA,
            "model": "EviSIRST",
            "dataset": DATASET,
            "epoch": epoch,
            "run_identity": identity,
            "validation_record": records_by_epoch[epoch],
            "test_split_accessed": False,
            "experiment_schema": EXPERIMENT_SCHEMA,
            "experiment_status": "experimental_validation_only",
            "public_test_supported": False,
            "public_test_gate_status": "unsupported_until_separate_gate_extension",
            "crop_audit_commit_status": "candidate_pre_latest_uncommitted",
        }
        if set(candidate) != set(_CANDIDATE_KEYS):
            raise CompleteTargetReplicationError(
                f"retained candidate keys differ at epoch {epoch}"
            )
        for key, expected in fixed.items():
            observed = candidate.get(key)
            if type(observed) is not type(expected) or observed != expected:
                raise CompleteTargetReplicationError(
                    f"retained candidate differs at epoch {epoch}.{key}"
                )
        _validate_state_dict_against_reference(
            candidate.get("state_dict"),
            final_state,
            label=f"retained candidate epoch {epoch} state_dict",
            require_tensor_equality=epoch == selected_epoch,
        )
        try:
            pilot._validate_candidate_crop_audit(
                candidate, epoch=epoch, identity=identity
            )
        except pilot.CompleteTargetRunnerError as exc:
            raise CompleteTargetReplicationError(
                f"retained candidate crop audit differs at epoch {epoch}"
            ) from exc
        _require_no_public_test_access(
            candidate, path=f"candidate_epoch_{epoch}"
        )
        if epoch == selected_epoch:
            for candidate_key, audit_key in (
                ("crop_audit_state", "state"),
                ("crop_audit_summary", "summary"),
                ("crop_audit_history", "history"),
            ):
                if candidate.get(candidate_key) != selected_audit.get(audit_key):
                    raise CompleteTargetReplicationError(
                        "selected candidate crop audit differs from final checkpoint"
                    )
            if selected_audit.get("through_epoch") != selected_epoch:
                raise CompleteTargetReplicationError(
                    "selected final crop-audit epoch differs"
                )
        evidence.append(
            {
                "epoch": epoch,
                "relative_path": artifact["relative_path"],
                "sha256": observed_sha256,
                "selected": epoch == selected_epoch,
            }
        )
    return evidence


def _validate_final_crop_audit_bundle(
    *,
    state: Any,
    summary: Any,
    history: Any,
    through_epoch: Any,
    expected_epoch: int,
    identity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Close state, computed summary, persisted summary, and history tail."""

    if through_epoch != expected_epoch:
        raise CompleteTargetReplicationError(
            f"{label} crop-audit epoch differs"
        )
    try:
        normalized_state = _json_clone(state)
        verifier = complete_crop.CropAuditAccumulator(formal_num_workers=0)
        verifier.load_state_dict(normalized_state)
        normalized_summary = pilot._validate_crop_audit_summary(
            summary,
            expected_observations=(
                expected_epoch * pilot._identity_train_count(identity)
            ),
        )
        normalized_history = pilot._validate_crop_audit_history(
            history,
            completed_epoch=expected_epoch,
            identity=identity,
        )
    except (
        TypeError,
        complete_crop.CompleteTargetCropError,
        pilot.CompleteTargetRunnerError,
    ) as exc:
        raise CompleteTargetReplicationError(
            f"{label} crop-audit bundle differs"
        ) from exc
    computed_summary = _json_clone(verifier.compute())
    if (
        computed_summary != normalized_summary
        or normalized_history[-1]["summary"] != normalized_summary
    ):
        raise CompleteTargetReplicationError(
            f"{label} crop-audit state/summary/history differ"
        )
    return {
        "state": normalized_state,
        "summary": normalized_summary,
        "history": normalized_history,
        "through_epoch": expected_epoch,
    }


def _validate_ordered_histories(
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training = summary.get("training_history")
    validation = summary.get("validation_history")
    if (
        not isinstance(training, list)
        or not isinstance(validation, list)
        or len(training) != FORMAL_EPOCHS
        or len(validation) != FORMAL_EPOCHS
    ):
        raise CompleteTargetReplicationError(
            "completed replication does not contain 1000 train/val epochs"
        )
    normalized_training: list[dict[str, Any]] = []
    normalized_validation: list[dict[str, Any]] = []
    for epoch, record in enumerate(training, start=1):
        if (
            type(record) is not dict
            or set(record)
            != {
                "epoch",
                "mean_train_loss",
                "learning_rate",
                "processed_samples",
            }
            or record.get("epoch") != epoch
            or record.get("processed_samples") != CANONICAL_IRSTD_TRAIN_COUNT
            or type(record.get("mean_train_loss")) is not float
            or not math.isfinite(record["mean_train_loss"])
            or record["mean_train_loss"] < 0.0
            or type(record.get("learning_rate")) is not float
            or not math.isfinite(record["learning_rate"])
            or record["learning_rate"]
            != r1.legacy_train.learning_rate_for_epoch(
                epoch,
                FORMAL_EPOCHS,
                FORMAL_BASE_LR,
                FORMAL_MIN_LR,
                FORMAL_WARMUP_EPOCHS,
            )
        ):
            raise CompleteTargetReplicationError(
                "completed replication training history differs"
            )
        _canonical_json_bytes(record)
        normalized_training.append(_json_clone(record))
    for epoch, record in enumerate(validation, start=1):
        if (
            type(record) is not dict
            or set(record)
            != {
                "epoch",
                "data_role",
                "mIoU",
                "Fa",
                "Pd",
                "evaluation_head",
                "metrics",
            }
            or record.get("epoch") != epoch
            or record.get("data_role") != "val"
            or record.get("evaluation_head") != "out"
        ):
            raise CompleteTargetReplicationError(
                "completed replication validation history differs"
            )
        # This invokes the frozen validation selector's numeric/type checks.
        r1.selection.select_independent_checkpoint([record])
        normalized_validation.append(_json_clone(record))
    return normalized_training, normalized_validation


def validate_existing_completed_run(run_seed: int) -> dict[str, Any]:
    """CPU-only strict validator for watcher COMPLETE classification.

    The aggregate gate remains the independent decision authority.  This
    helper only proves that one fixed replication directory is internally
    complete, source/gate-bound, validation-selected, and public-test-free.
    It never constructs a dataset or CUDA device.
    """

    contract = formal_artifact_contract(run_seed)
    authorization = validate_canonical_expansion_authorization()
    summary_path = PROJECT_ROOT / contract["summary_relative_path"]
    checkpoint_path = PROJECT_ROOT / contract["checkpoint_relative_path"]
    summary_bytes, summary_sha256 = _read_regular_file_stably(
        summary_path, label="completed replication summary"
    )
    summary = _strict_json_object(
        summary_bytes, label="completed replication summary"
    )
    if set(summary) != set(canonical_gate._VARIANT_SUMMARY_KEYS):
        raise CompleteTargetReplicationError(
            "completed replication summary keys differ"
        )
    fixed_summary = {
        "schema": TRAINING_SCHEMA + "/summary",
        "status": "complete",
        "dataset": DATASET,
        "checkpoint": contract["checkpoint_relative_path"],
        "checkpoint_role": "experimental_validation_selected",
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": run_seed,
        "target_mode": TARGET_MODE,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
        "experiment_schema": EXPERIMENT_SCHEMA,
        "experiment_status": "experimental_validation_only",
        "only_train_crop_policy_differs_from_R1": True,
        "promotion_gate": authorization,
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_until_separate_gate_extension",
        "crop_audit_commit_status": "complete",
    }
    for key, expected in fixed_summary.items():
        if type(summary.get(key)) is not type(expected) or summary.get(key) != expected:
            raise CompleteTargetReplicationError(
                f"completed replication summary differs at {key}"
            )
    training_history, validation_history = _validate_ordered_histories(summary)

    identity = summary.get("run_identity")
    if (
        type(identity) is not dict
        or set(identity) != set(canonical_gate._VARIANT_IDENTITY_KEYS)
        or identity.get("schema") != TRAINING_SCHEMA + "/run_identity"
        or identity.get("run_seed") != run_seed
        or identity.get("epochs") != FORMAL_EPOCHS
        or identity.get("workers") != FORMAL_WORKERS
        or identity.get("promotion_gate") != authorization
        or identity.get("test_split_accessed") is not False
    ):
        raise CompleteTargetReplicationError(
            "completed replication run identity differs"
        )
    identity_sha256 = identity.get("identity_sha256")
    unhashed_identity = dict(identity)
    unhashed_identity.pop("identity_sha256", None)
    if (
        type(identity_sha256) is not str
        or identity_sha256 != _canonical_sha256(unhashed_identity)
        or summary.get("training_identity_sha256") != identity_sha256
    ):
        raise CompleteTargetReplicationError(
            "completed replication identity SHA-256 differs"
        )
    experiment = identity.get("experiment")
    if (
        type(experiment) is not dict
        or experiment.get("schema") != EXPERIMENT_SCHEMA
        or experiment.get("single_variable_from_completed_pilot")
        != "runtime_seed"
        or experiment.get("completed_pilot_run_seed") != PILOT_RUN_SEED
        or experiment.get("authorized_three_runtime_seeds")
        != list(THREE_RUNTIME_SEEDS)
        or experiment.get("public_test_supported") is not False
    ):
        raise CompleteTargetReplicationError(
            "completed replication experiment identity differs"
        )
    expected_sources = _replication_source_provenance_from_authorization(
        authorization
    )
    determinism = identity.get("determinism_protocol")
    if (
        type(determinism) is not dict
        or determinism.get("schema") != DETERMINISM_SCHEMA
        or determinism.get("source_set_schema") != SOURCE_SET_SCHEMA
        or determinism.get("source_files") != expected_sources["files"]
        or determinism.get("source_tree_sha256")
        != expected_sources["source_tree_sha256"]
        or determinism.get("frozen_parent_source_tree_sha256")
        != FROZEN_PARENT_SOURCE_TREE_SHA256
    ):
        raise CompleteTargetReplicationError(
            "completed replication source identity differs"
        )

    selection = summary.get("selection")
    if (
        type(selection) is not dict
        or set(selection) != set(canonical_gate._SELECTION_PAYLOAD_KEYS)
        or selection.get("schema") != SELECTION_PAYLOAD_SCHEMA
        or selection.get("data_role") != "val"
        or selection.get("source_selection") != "evisirst_v2_validation_split"
        or selection.get("selection_is_optimistic") is not False
        or selection.get("optimistic") is not False
    ):
        raise CompleteTargetReplicationError(
            "completed replication selection payload differs"
        )
    recomputed = r1.selection.select_independent_checkpoint(validation_history)
    selected_epoch = recomputed["selected"]["epoch"]
    if (
        selection.get("selection_provenance") != recomputed
        or selection.get("selected_epoch") != selected_epoch
        or summary.get("selected_epoch") != selected_epoch
    ):
        raise CompleteTargetReplicationError(
            "completed replication selector recomputation differs"
        )
    selected_records = [
        record for record in validation_history if record["epoch"] == selected_epoch
    ]
    if (
        len(selected_records) != 1
        or summary.get("selected_validation_record") != selected_records[0]
        or summary.get("selected_validation_record_sha256")
        != _canonical_sha256(selected_records[0])
    ):
        raise CompleteTargetReplicationError(
            "completed replication selected validation record differs"
        )

    checkpoint, checkpoint_sha256 = _load_torch_regular_stably(
        checkpoint_path, label="completed replication final checkpoint"
    )
    if set(checkpoint) != set(canonical_gate._VARIANT_CHECKPOINT_KEYS):
        raise CompleteTargetReplicationError(
            "completed replication checkpoint keys differ"
        )
    fixed_checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "checkpoint_role": "experimental_validation_selected",
        "epoch": selected_epoch,
        "seed": ARCHITECTURE_SEED,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": run_seed,
        "target_mode": TARGET_MODE,
        "training": identity,
        "training_identity_sha256": identity_sha256,
        "selection_provenance": recomputed,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
        "experiment_schema": EXPERIMENT_SCHEMA,
        "experiment_status": "experimental_validation_only",
        "only_train_crop_policy_differs_from_R1": True,
        "promotion_gate": authorization,
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_until_separate_gate_extension",
        "crop_audit_commit_status": "complete",
    }
    for key, expected in fixed_checkpoint.items():
        observed = checkpoint.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise CompleteTargetReplicationError(
                f"completed replication checkpoint differs at {key}"
            )
    state_key_count = _validate_final_state_dict(
        checkpoint.get("state_dict"), authorization=authorization
    )
    if summary.get("final_checkpoint_sha256") != checkpoint_sha256:
        raise CompleteTargetReplicationError(
            "completed replication final checkpoint SHA-256 differs"
        )
    # The summary and final checkpoint must carry byte-for-byte-equivalent
    # audit evidence for both the full search and selected epoch.
    paired_audit_fields = (
        "search_run_crop_audit_state",
        "search_run_crop_audit_summary",
        "search_run_crop_audit_history",
        "search_run_crop_audit_through_epoch",
        "selected_checkpoint_crop_audit_state",
        "selected_checkpoint_crop_audit_summary",
        "selected_checkpoint_crop_audit_history",
        "selected_checkpoint_crop_audit_through_epoch",
        "selected_candidate_sha256",
        "crop_policy",
        "crop_policy_identity_sha256",
    )
    if any(summary.get(key) != checkpoint.get(key) for key in paired_audit_fields):
        raise CompleteTargetReplicationError(
            "completed replication summary/checkpoint audit evidence differs"
        )
    _validate_final_crop_audit_bundle(
        state=summary.get("search_run_crop_audit_state"),
        summary=summary.get("search_run_crop_audit_summary"),
        history=summary.get("search_run_crop_audit_history"),
        through_epoch=summary.get("search_run_crop_audit_through_epoch"),
        expected_epoch=FORMAL_EPOCHS,
        identity=identity,
        label="search-run",
    )
    selected_audit = _validate_final_crop_audit_bundle(
        state=summary.get("selected_checkpoint_crop_audit_state"),
        summary=summary.get("selected_checkpoint_crop_audit_summary"),
        history=summary.get("selected_checkpoint_crop_audit_history"),
        through_epoch=summary.get(
            "selected_checkpoint_crop_audit_through_epoch"
        ),
        expected_epoch=selected_epoch,
        identity=identity,
        label="selected",
    )
    retained_candidates = _validate_retained_candidates(
        run_dir=checkpoint_path.parent,
        identity=identity,
        validation_history=validation_history,
        selection=selection,
        candidate_artifacts=summary.get("candidate_artifacts"),
        final_state=checkpoint["state_dict"],
        selected_audit=selected_audit,
        selected_candidate_sha256=summary.get("selected_candidate_sha256"),
    )
    if summary.get("selected_candidate_sha256") != checkpoint.get(
        "selected_candidate_sha256"
    ):
        raise CompleteTargetReplicationError(
            "completed replication selected candidate SHA differs across final artifacts"
        )
    _require_no_public_test_access(summary, path="summary")
    _require_no_public_test_access(checkpoint, path="checkpoint")
    return {
        "schema": TRAINING_SCHEMA + "/completed_run_evidence",
        "status": "complete",
        "run_seed": run_seed,
        "summary": {
            "relative_path": contract["summary_relative_path"],
            "sha256": summary_sha256,
        },
        "checkpoint": {
            "relative_path": contract["checkpoint_relative_path"],
            "sha256": checkpoint_sha256,
            "state_key_count": state_key_count,
        },
        "training_identity_sha256": identity_sha256,
        "source_tree_sha256": expected_sources["source_tree_sha256"],
        "canonical_authorization_sha256": _canonical_sha256(authorization),
        "selected_epoch": selected_epoch,
        "selected_validation_record_sha256": summary[
            "selected_validation_record_sha256"
        ],
        "retention_frontier_epochs": [
            artifact["epoch"] for artifact in retained_candidates
        ],
        "retained_candidates": retained_candidates,
        "retained_candidate_count": len(retained_candidates),
        "train_epoch_count": len(training_history),
        "validation_epoch_count": len(validation_history),
        "test_split_accessed": False,
        "public_test_supported": False,
    }


def validate_state_dict_contract_for_gate(
    state_dict: Any, authorization: Mapping[str, Any]
) -> int:
    """Expose the exact CPU-only 564-key structure check for tests/gates."""

    normalized = _json_clone(authorization)
    if normalized.get("schema") != AUTHORIZATION_SCHEMA:
        raise CompleteTargetReplicationError(
            "state-dict validation authorization schema differs"
        )
    return _validate_final_state_dict(state_dict, authorization=normalized)


def main(argv: Sequence[str] | None = None) -> None:
    checkpoint = run(parse_args(argv))
    print(checkpoint.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "AUTHORIZATION_SCHEMA",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CompleteTargetReplicationError",
    "DATASET",
    "DEFAULT_OUTPUT_ROOT",
    "DETERMINISM_SCHEMA",
    "EXPERIMENT_SCHEMA",
    "FORMAL_EPOCHS",
    "HISTORY_SCHEMA",
    "PILOT_RUN_SEED",
    "REPLICATION_RUN_SEEDS",
    "SELECTION_PAYLOAD_SCHEMA",
    "SOURCE_SET_SCHEMA",
    "TARGET_MODE",
    "THREE_RUNTIME_SEEDS",
    "TRAINING_SCHEMA",
    "formal_artifact_contract",
    "parse_args",
    "promotion_gate",
    "resolve_run_paths",
    "run",
    "validate_state_dict_contract_for_gate",
    "validate_canonical_expansion_authorization",
    "validate_existing_completed_run",
]
