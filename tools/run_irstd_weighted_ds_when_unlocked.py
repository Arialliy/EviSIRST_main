#!/usr/bin/env python3
"""Run the fixed IRSTD-1K weighted-DS fallback only after a canonical FAIL.

This watcher accepts no arguments.  It is deliberately standard-library-only
at import time: the heavyweight canonical/weighted validators run in the fixed
project virtual environment as short-lived child processes.  A canonical PASS
is terminal.  Missing or not-yet-valid evidence is only observed; it can never
authorize training.

For a canonical FAIL, the watcher requires two consecutive idle observations
ten seconds apart, takes the repository-wide lock for the GPU's complete UUID,
and performs another identity/idle check before launch.  That lock descriptor
is inherited through ``/usr/bin/time`` into the trainer.  The fixed formal run
is resumed exactly when its committed last state exists and no complete summary
does; a validated complete summary is terminal and is never repeated.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Sequence


PROJECT_ROOT = Path("/home/ly/EviSIRST_main")
PYTHON_BIN = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
PYTHON_RESOLVED_TARGET = Path("/usr/bin/python3.12")
TIME_BIN = Path("/usr/bin/time")
NVIDIA_SMI_BIN = Path("/usr/bin/nvidia-smi")
PROC_ROOT = Path("/proc")

TRAINER = PROJECT_ROOT / "train_irstd_weighted_ds_v1.py"
DATASET_ROOT = Path("/home/ly/SCTransNet_main/datasets")
SPLIT_ROOT = PROJECT_ROOT / "splits/v2"
CANONICAL_RESULT = (
    PROJECT_ROOT
    / "runs/irstd_performance/complete_target_v1/promotion_gate/"
    "run_seed_1446202191/result.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "runs/irstd_performance/weighted_ds_v1"
FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "formal/IRSTD-1K/binary/run_seed_1446202191"
)
SUMMARY = FORMAL_RUN_DIR / "summary.json"
LAST_STATE = FORMAL_RUN_DIR / "last_training_state.pth.tar"
WATCHER_LOCK = OUTPUT_ROOT / "watcher.run.lock"
GPU_LOCK_ROOT = PROJECT_ROOT / "runs/.gpu_locks"

DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
RUN_SEED = 1446202191
EPOCHS = 1000
WARMUP_EPOCHS = 10
MEMORY_IDLE_LIMIT_MIB = 1024
POLL_SECONDS = 10
STATUS_SECONDS = 300
INITIAL_FAILURE_BACKOFF_SECONDS = 60
MAX_FAILURE_BACKOFF_SECONDS = 300
VALIDATION_TIMEOUT_SECONDS = 900

GPU_LOCK_FD_ENV = "EVISIRST_WEIGHTED_DS_GPU_LOCK_FD"
GPU_UUID_ENV = "EVISIRST_WEIGHTED_DS_GPU_UUID"
GPU_BUS_ENV = "EVISIRST_WEIGHTED_DS_GPU_BUS_ID"
GPU_INDEX_ENV = "EVISIRST_WEIGHTED_DS_GPU_INDEX"

_GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_GPU_BUS_RE = re.compile(
    r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]"
)
_PROBE_PREFIX = "EVISIRST_WEIGHTED_DS_WATCHER_PROBE:"

_SUMMARY_KEYS = frozenset(
    {
        "schema", "status", "dataset", "checkpoint", "checkpoint_role",
        "selected_epoch", "architecture_seed", "run_seed", "target_mode",
        "split_provenance", "selection", "training_history",
        "validation_history", "candidate_artifacts", "normalization",
        "source_selection", "selection_is_optimistic", "optimistic",
        "test_split_accessed", "smoke", "elapsed_seconds",
        "experiment_schema", "experiment_status",
        "only_training_loss_weights_differ_from_R1", "loss_policy",
        "selected_validation_record", "selected_validation_record_sha256",
        "run_identity", "training_identity_sha256",
        "predecessor_gate_evidence", "promotion_gate",
        "public_test_supported", "public_test_gate_status",
        "final_checkpoint_sha256",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "schema", "experiment", "model", "dataset", "architecture_seed",
        "run_seed", "target_mode", "epochs", "batch_size", "workers",
        "base_lr", "min_lr", "warmup_epochs", "val_interval",
        "normalization_mode", "optimizer", "optimizer_hyperparameters",
        "loss", "loss_policy", "evaluation", "evaluation_head",
        "selection_rule", "determinism_protocol",
        "legacy_R1_train_crop_and_data", "manifest_sha256", "split_seed",
        "data_tree_sha256", "canonical_split_contract", "grouping_policy",
        "train_count", "val_count", "smoke", "smoke_max_train_samples",
        "smoke_max_val_samples", "runtime_identity",
        "predecessor_gate_evidence", "promotion_gate",
        "test_split_accessed", "identity_sha256",
    }
)
_TRAINING_RECORD_KEYS = frozenset(
    {"epoch", "mean_train_loss", "learning_rate", "processed_samples"}
)
_VALIDATION_RECORD_KEYS = frozenset(
    {"epoch", "data_role", "mIoU", "Fa", "Pd", "evaluation_head", "metrics"}
)
_VALIDATION_METRIC_KEYS = frozenset(
    {
        "validation_loss", "miou", "niou", "pixel_precision", "pixel_recall",
        "pixel_f1", "pd", "tiny_pd", "fa", "false_objects_per_image",
        "target_count", "matched_target_count", "tiny_target_count",
        "matched_tiny_target_count", "predicted_object_count",
        "unmatched_predicted_object_count", "valid_pixel_count",
    }
)
_CANDIDATE_METADATA_KEYS = frozenset({"relative_path", "file_sha256"})
_CANDIDATE_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "model", "dataset", "epoch", "run_identity",
        "validation_record", "state_dict", "test_split_accessed",
        "experiment_schema", "experiment_status",
        "only_training_loss_weights_differ_from_R1", "loss_policy",
        "public_test_supported", "public_test_gate_status",
    }
)
_FINAL_CHECKPOINT_KEYS = frozenset(
    {
        "schema", "model", "dataset", "checkpoint_role", "epoch", "seed",
        "architecture_seed", "run_seed", "state_dict", "target_mode",
        "normalization", "normalization_provenance", "training",
        "training_identity_sha256", "split_provenance", "split_seed",
        "split_manifest_sha256", "data_tree_sha256", "data_tree_verified",
        "selection_provenance", "source_selection",
        "selection_is_optimistic", "optimistic", "test_split_accessed",
        "model_metadata", "smoke", "experiment_schema", "experiment_status",
        "only_training_loss_weights_differ_from_R1", "loss_policy",
        "predecessor_gate_evidence", "promotion_gate", "public_test_supported",
        "public_test_gate_status",
    }
)


def _require_finite_json_tree(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite float")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string key")
            _require_finite_json_tree(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_json_tree(child, label=f"{label}[{index}]")


def _strict_json_file(path: Path) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in items:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = value
        return output

    def constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=constant,
    )
    _require_finite_json_tree(value, label="JSON")
    return value


def _require_exact_keys(value: object, expected: object, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} keys differ")


# Executed only in PYTHON_BIN.  Keeping validation out of this process means a
# passive watcher does not import torch, enumerate CUDA, or reserve GPU memory.
_VALIDATOR_PROGRAM = r'''
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from tools.run_irstd_weighted_ds_when_unlocked import (
    _CANDIDATE_CHECKPOINT_KEYS,
    _CANDIDATE_METADATA_KEYS,
    _FINAL_CHECKPOINT_KEYS,
    _IDENTITY_KEYS,
    _SUMMARY_KEYS,
    _TRAINING_RECORD_KEYS,
    _VALIDATION_METRIC_KEYS,
    _VALIDATION_RECORD_KEYS,
    _require_exact_keys as exact_keys,
    _strict_json_file as strict_json,
)

ROOT = Path("/home/ly/EviSIRST_main")
PREFIX = "EVISIRST_WEIGHTED_DS_WATCHER_PROBE:"
sys.path.insert(0, os.fspath(ROOT))


def fail(message):
    raise RuntimeError(message)


def emit(value):
    print(PREFIX + value, flush=True)


def finite_number(value, *, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(label + " must be numeric")
    number = float(value)
    if not math.isfinite(number):
        fail(label + " must be finite")
    if minimum is not None and number < minimum:
        fail(label + " is below its minimum")
    if maximum is not None and number > maximum:
        fail(label + " is above its maximum")
    return number


def safe_regular(path, parent):
    if path.is_symlink() or not path.is_file():
        fail("artifact is not a regular file: " + os.fspath(path))
    resolved = path.resolve(strict=True)
    if resolved != path or resolved.parent != parent.resolve(strict=True):
        fail("artifact escaped its fixed directory: " + os.fspath(path))
    return resolved


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


action = sys.argv[1] if len(sys.argv) == 2 else fail("one fixed action required")

if action == "gate":
    import run_irstd_complete_target_promotion_gate as gate

    payload = gate.validate_existing_result()
    if not isinstance(payload, Mapping):
        fail("canonical payload is malformed")
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        fail("canonical decision is missing")
    result = decision.get("result")
    if result == "PASS":
        if (
            decision.get("overall_passed") is not True
            or decision.get("three_runtime_seed_validation_expansion_allowed")
            is not True
            or payload.get("status") != "complete"
            or payload.get("test_split_accessed") is not False
            or payload.get("public_test_allowed") is not False
        ):
            fail("canonical PASS fields differ")
        emit("PASS")
    elif result == "FAIL":
        import train_irstd_weighted_ds_v1 as runner

        evidence = runner.validate_canonical_predecessor_failure()
        if (
            evidence.get("formal_weighted_ds_allowed") is not True
            or evidence.get("status") != "complete"
            or evidence.get("canonical_payload") != payload
            or evidence.get("public_test_accessed") is not False
        ):
            fail("weighted runner rejected canonical FAIL evidence")
        emit("FAIL")
    else:
        fail("canonical result is neither PASS nor FAIL")

elif action == "summary":
    import train_irstd_weighted_ds_v1 as runner

    evidence = runner.validate_canonical_predecessor_failure()
    args = runner.parse_args(
        [
            "--dataset-root", "/home/ly/SCTransNet_main/datasets",
            "--split-root", "/home/ly/EviSIRST_main/splits/v2",
            "--dataset", "IRSTD-1K",
            "--target-mode", "binary",
            "--architecture-seed", "42",
            "--run-seed", "1446202191",
            "--device", "cuda:0",
            "--epochs", "1000",
            "--warmup-epochs", "10",
            "--allow-sample-level-fallback",
        ]
    )
    paths = runner.resolve_run_paths(args)
    run_dir = paths["run_dir"]
    summary_path = safe_regular(paths["summary"], run_dir)
    final_path = safe_regular(paths["final"], run_dir)
    summary = strict_json(summary_path)
    if not isinstance(summary, Mapping):
        fail("weighted summary is malformed")

    exact_keys(
        summary,
        _SUMMARY_KEYS,
        label="weighted summary",
    )
    finite_number(summary.get("elapsed_seconds"), label="elapsed_seconds", minimum=0)

    expected_checkpoint = final_path.relative_to(ROOT).as_posix()
    fixed_summary = {
        "schema": runner.TRAINING_SCHEMA + "/summary",
        "status": "complete",
        "dataset": runner.DATASET,
        "checkpoint": expected_checkpoint,
        "checkpoint_role": "experimental_validation_selected",
        "architecture_seed": runner.ARCHITECTURE_SEED,
        "run_seed": runner.PAIRED_RUN_SEED,
        "target_mode": runner.TARGET_MODE,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
        "experiment_schema": runner.EXPERIMENT_SCHEMA,
        "experiment_status": "experimental_validation_only",
        "only_training_loss_weights_differ_from_R1": True,
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_until_separate_gate_extension",
    }
    for key, expected in fixed_summary.items():
        if summary.get(key) != expected:
            fail("weighted summary differs at " + key)
    if summary.get("loss_policy") != runner.loss_policy_identity():
        fail("weighted summary loss policy differs")
    if summary.get("predecessor_gate_evidence") != evidence:
        fail("weighted summary predecessor evidence differs")
    if summary.get("promotion_gate") != runner.promotion_gate():
        fail("weighted summary promotion gate differs")
    normalization_spec = runner.r1.v2_data._normalization_spec(
        runner.DATASET,
        normalization_mode="legacy",
        normalization_values=None,
    )
    expected_normalization = {
        "mean": normalization_spec.mean,
        "std": normalization_spec.std,
    }
    if summary.get("normalization") != expected_normalization:
        fail("weighted summary normalization differs")
    manifest_path = safe_regular(
        ROOT / "splits/v2/IRSTD-1K/manifest.json",
        ROOT / "splits/v2/IRSTD-1K",
    )
    if sha256_file(manifest_path) != runner.CANONICAL_IRSTD_MANIFEST_SHA256:
        fail("canonical manifest hash differs")
    manifest = strict_json(manifest_path)
    expected_split = {
        "schema": manifest["schema"],
        "manifest_relative_path": "splits/v2/IRSTD-1K/manifest.json",
        "manifest_sha256": runner.CANONICAL_IRSTD_MANIFEST_SHA256,
        "split_seed": manifest["seeds"]["split_seed"],
        "source_index": manifest["source_index"],
        "outputs": {
            "train": manifest["outputs"]["train"],
            "val": manifest["outputs"]["val"],
        },
        "grouping": manifest["grouping"],
        "data_tree_sha256": runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
        "data_tree_verified": True,
        "train_count": runner.CANONICAL_IRSTD_TRAIN_COUNT,
        "val_count": runner.CANONICAL_IRSTD_VAL_COUNT,
        "test_index_opened": False,
    }
    if summary.get("split_provenance") != expected_split:
        fail("weighted summary split provenance differs")

    identity = summary.get("run_identity")
    if not isinstance(identity, Mapping):
        fail("weighted run identity is missing")
    exact_keys(
        identity,
        _IDENTITY_KEYS,
        label="weighted run identity",
    )
    identity_without_hash = dict(identity)
    identity_sha = identity_without_hash.pop("identity_sha256", None)
    if (
        not isinstance(identity_sha, str)
        or identity_sha != runner._canonical_sha256(identity_without_hash)
        or summary.get("training_identity_sha256") != identity_sha
    ):
        fail("weighted run identity hash differs")
    fixed_identity = {
        "schema": runner.TRAINING_SCHEMA + "/run_identity",
        "model": "EviSIRST",
        "dataset": runner.DATASET,
        "architecture_seed": runner.ARCHITECTURE_SEED,
        "run_seed": runner.PAIRED_RUN_SEED,
        "target_mode": runner.TARGET_MODE,
        "epochs": runner.FORMAL_EPOCHS,
        "batch_size": runner.FORMAL_BATCH_SIZE,
        "workers": runner.FORMAL_WORKERS,
        "base_lr": runner.FORMAL_BASE_LR,
        "min_lr": runner.FORMAL_MIN_LR,
        "warmup_epochs": runner.FORMAL_WARMUP_EPOCHS,
        "val_interval": runner.FORMAL_VAL_INTERVAL,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "optimizer_hyperparameters": runner._json_clone(
            runner.r1.R1_ADAM_GROUP_HYPERPARAMETERS
        ),
        "loss": "weighted_sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "evaluation_head": "out",
        "selection_rule": runner.r1.selection.INDEPENDENT_RULE_VERSION,
        "legacy_R1_train_crop_and_data": True,
        "manifest_sha256": runner.CANONICAL_IRSTD_MANIFEST_SHA256,
        "split_seed": manifest["seeds"]["split_seed"],
        "data_tree_sha256": runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
        "train_count": runner.CANONICAL_IRSTD_TRAIN_COUNT,
        "val_count": runner.CANONICAL_IRSTD_VAL_COUNT,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    for key, expected in fixed_identity.items():
        if identity.get(key) != expected:
            fail("weighted run identity differs at " + key)
    if (
        identity.get("predecessor_gate_evidence") != evidence
        or identity.get("loss_policy") != runner.loss_policy_identity()
        or identity.get("promotion_gate") != runner.promotion_gate()
        or identity.get("determinism_protocol")
        != runner._variant_determinism_protocol_identity()
    ):
        fail("weighted run identity provenance differs")
    canonical_split = identity.get("canonical_split_contract")
    if canonical_split != {
        "split_root_relative_path": "splits/v2",
        "manifest_sha256": runner.CANONICAL_IRSTD_MANIFEST_SHA256,
        "data_tree_sha256": runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
        "train_count": runner.CANONICAL_IRSTD_TRAIN_COUNT,
        "val_count": runner.CANONICAL_IRSTD_VAL_COUNT,
    }:
        fail("weighted canonical split contract differs")
    expected_grouping = {
        "mode": "sample_level_fallback",
        "sample_level_fallback_acknowledged": True,
        "warning": manifest["grouping"]["warning"],
    }
    if identity.get("grouping_policy") != expected_grouping:
        fail("weighted grouping policy differs")
    if identity.get("runtime_identity") != runner._runtime_identity(args):
        fail("weighted runtime device mapping differs")
    if identity.get("experiment") != {
        "schema": runner.EXPERIMENT_SCHEMA,
        "name": "IRSTD-1K weighted deep supervision v1",
        "status": "experimental_validation_only",
        "only_training_loss_weights_differ_from_R1": True,
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_until_separate_gate_extension",
    }:
        fail("weighted experiment identity differs")

    training = summary.get("training_history")
    validation = summary.get("validation_history")
    if (
        not isinstance(training, list)
        or len(training) != runner.FORMAL_EPOCHS
        or [record.get("epoch") for record in training if isinstance(record, Mapping)]
        != list(range(1, runner.FORMAL_EPOCHS + 1))
        or not isinstance(validation, list)
        or len(validation) != runner.FORMAL_EPOCHS
        or [record.get("epoch") for record in validation if isinstance(record, Mapping)]
        != list(range(1, runner.FORMAL_EPOCHS + 1))
    ):
        fail("weighted histories are incomplete")
    for epoch, record in enumerate(training, start=1):
        exact_keys(
            record,
            _TRAINING_RECORD_KEYS,
            label="training record",
        )
        if record["epoch"] != epoch or record["processed_samples"] != runner.CANONICAL_IRSTD_TRAIN_COUNT:
            fail("training record identity differs")
        finite_number(record["mean_train_loss"], label="mean train loss", minimum=0)
        observed_lr = finite_number(record["learning_rate"], label="learning rate", minimum=0)
        expected_lr = runner.r1.legacy_train.learning_rate_for_epoch(
            epoch,
            runner.FORMAL_EPOCHS,
            runner.FORMAL_BASE_LR,
            runner.FORMAL_MIN_LR,
            runner.FORMAL_WARMUP_EPOCHS,
        )
        if observed_lr != expected_lr:
            fail("training learning-rate schedule differs")
    for epoch, record in enumerate(validation, start=1):
        exact_keys(record, _VALIDATION_RECORD_KEYS, label="validation record")
        if (
            record["epoch"] != epoch
            or record["data_role"] != "val"
            or record["evaluation_head"] != "out"
        ):
            fail("validation record identity differs")
        metrics = record["metrics"]
        exact_keys(metrics, _VALIDATION_METRIC_KEYS, label="validation metrics")
        for name in ("validation_loss", "false_objects_per_image"):
            finite_number(metrics[name], label="validation metric " + name, minimum=0)
        for name in (
            "miou", "niou", "pixel_precision", "pixel_recall", "pixel_f1",
            "pd", "fa",
        ):
            finite_number(
                metrics[name],
                label="validation metric " + name,
                minimum=0,
                maximum=1,
            )
        if metrics["tiny_pd"] is not None:
            finite_number(metrics["tiny_pd"], label="tiny_pd", minimum=0, maximum=1)
        for name in (
            "target_count", "matched_target_count", "tiny_target_count",
            "matched_tiny_target_count", "predicted_object_count",
            "unmatched_predicted_object_count", "valid_pixel_count",
        ):
            if isinstance(metrics[name], bool) or not isinstance(metrics[name], int) or metrics[name] < 0:
                fail("validation count " + name + " differs")
        if (
            metrics["target_count"] != 239
            or metrics["tiny_target_count"] != 21
            or metrics["valid_pixel_count"] != 41943040
            or metrics["matched_target_count"] > metrics["target_count"]
            or metrics["matched_target_count"] > metrics["predicted_object_count"]
            or metrics["matched_tiny_target_count"] > metrics["tiny_target_count"]
            or metrics["unmatched_predicted_object_count"] > metrics["predicted_object_count"]
            or metrics["pd"]
            != metrics["matched_target_count"] / metrics["target_count"]
            or metrics["tiny_pd"]
            != metrics["matched_tiny_target_count"] / metrics["tiny_target_count"]
            or metrics["false_objects_per_image"]
            != metrics["unmatched_predicted_object_count"] / runner.CANONICAL_IRSTD_VAL_COUNT
            or record["mIoU"] != metrics["miou"]
            or record["Fa"] != metrics["fa"]
            or record["Pd"] != metrics["pd"]
        ):
            fail("validation metric consistency differs")
    provenance = runner.r1.selection.select_independent_checkpoint(validation)
    frontier = runner.r1.selection.retention_frontier_epochs(validation)
    raw_artifacts = summary.get("candidate_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        fail("weighted candidate artifacts are malformed")
    model, expected_model_metadata = runner.r1.initialize_evisirst(
        runner.DATASET, seed=runner.ARCHITECTURE_SEED, training=True
    )
    model_state_contract = model.state_dict()
    artifacts = {}
    for key, artifact in raw_artifacts.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal():
            fail("weighted candidate epoch key is malformed")
        epoch = int(key)
        if epoch in artifacts or not isinstance(artifact, Mapping):
            fail("weighted candidate artifact is malformed")
        exact_keys(
            artifact,
            _CANDIDATE_METADATA_KEYS,
            label="weighted candidate metadata",
        )
        relative = artifact.get("relative_path")
        expected_relative = "candidates/epoch_%04d.pth.tar" % epoch
        if relative != expected_relative:
            fail("weighted candidate path differs")
        candidate = safe_regular(run_dir / relative, run_dir / "candidates")
        if artifact.get("file_sha256") != sha256_file(candidate):
            fail("weighted candidate hash differs")
        candidate_payload = runner.torch.load(
            candidate, map_location="cpu", weights_only=True
        )
        exact_keys(
            candidate_payload,
            _CANDIDATE_CHECKPOINT_KEYS,
            label="weighted candidate checkpoint",
        )
        if (
            candidate_payload.get("schema") != runner.CANDIDATE_SCHEMA
            or candidate_payload.get("model") != "EviSIRST"
            or candidate_payload.get("dataset") != runner.DATASET
            or candidate_payload.get("epoch") != epoch
            or candidate_payload.get("run_identity") != identity
            or candidate_payload.get("validation_record") != validation[epoch - 1]
            or candidate_payload.get("test_split_accessed") is not False
            or candidate_payload.get("experiment_schema") != runner.EXPERIMENT_SCHEMA
            or candidate_payload.get("experiment_status")
            != "experimental_validation_only"
            or candidate_payload.get("only_training_loss_weights_differ_from_R1")
            is not True
            or candidate_payload.get("loss_policy") != runner.loss_policy_identity()
            or candidate_payload.get("public_test_supported") is not False
            or candidate_payload.get("public_test_gate_status")
            != "unsupported_until_separate_gate_extension"
        ):
            fail("weighted candidate checkpoint identity differs")
        runner.r1._validate_state_dict(
            candidate_payload.get("state_dict"), model_state_contract
        )
        artifacts[epoch] = dict(artifact)
    if tuple(sorted(artifacts)) != frontier:
        fail("weighted retention frontier differs")
    selected_epoch = int(provenance["selected"]["epoch"])
    expected_selection = {
        "schema": runner.SELECTION_PAYLOAD_SCHEMA,
        "data_role": "val",
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "selected_epoch": selected_epoch,
        "selected_candidate": artifacts[selected_epoch],
        "retention_frontier_epochs": list(frontier),
        "selection_provenance": provenance,
    }
    if summary.get("selection") != expected_selection:
        fail("weighted selection was not reproducible")
    selected_records = [
        record for record in validation
        if isinstance(record, Mapping) and record.get("epoch") == selected_epoch
    ]
    if len(selected_records) != 1:
        fail("weighted selected record is ambiguous")
    selected_record = selected_records[0]
    if (
        summary.get("selected_epoch") != selected_epoch
        or summary.get("selected_validation_record") != selected_record
        or summary.get("selected_validation_record_sha256")
        != runner._canonical_sha256(selected_record)
    ):
        fail("weighted selected record binding differs")

    final_sha = sha256_file(final_path)
    if summary.get("final_checkpoint_sha256") != final_sha:
        fail("weighted final checkpoint hash differs")
    checkpoint = runner.torch.load(final_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        fail("weighted final checkpoint is malformed")
    exact_keys(
        checkpoint,
        _FINAL_CHECKPOINT_KEYS,
        label="weighted final checkpoint",
    )
    fixed_checkpoint = {
        "schema": runner.CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": runner.DATASET,
        "checkpoint_role": "experimental_validation_selected",
        "epoch": selected_epoch,
        "seed": runner.ARCHITECTURE_SEED,
        "architecture_seed": runner.ARCHITECTURE_SEED,
        "run_seed": runner.PAIRED_RUN_SEED,
        "target_mode": runner.TARGET_MODE,
        "training_identity_sha256": identity_sha,
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
        "experiment_schema": runner.EXPERIMENT_SCHEMA,
        "experiment_status": "experimental_validation_only",
        "only_training_loss_weights_differ_from_R1": True,
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_until_separate_gate_extension",
    }
    for key, expected in fixed_checkpoint.items():
        if checkpoint.get(key) != expected:
            fail("weighted final checkpoint differs at " + key)
    if (
        checkpoint.get("normalization") != expected_normalization
        or checkpoint.get("normalization_provenance")
        != normalization_spec.as_dict()
        or checkpoint.get("split_provenance") != expected_split
        or checkpoint.get("split_seed") != expected_split["split_seed"]
        or checkpoint.get("split_manifest_sha256")
        != runner.CANONICAL_IRSTD_MANIFEST_SHA256
        or checkpoint.get("data_tree_sha256")
        != runner.CANONICAL_IRSTD_DATA_TREE_SHA256
        or checkpoint.get("data_tree_verified") is not True
        or checkpoint.get("promotion_gate") != runner.promotion_gate()
    ):
        fail("weighted final checkpoint data provenance differs")
    state = checkpoint.get("state_dict")
    validated_final_state = runner.r1._validate_state_dict(
        state, model_state_contract
    )
    selected_candidate = runner.torch.load(
        run_dir / artifacts[selected_epoch]["relative_path"],
        map_location="cpu",
        weights_only=True,
    )
    validated_selected_state = runner.r1._validate_state_dict(
        selected_candidate.get("state_dict"), model_state_contract
    )
    if any(
        not runner.torch.equal(validated_final_state[key], validated_selected_state[key])
        for key in validated_final_state
    ):
        fail("weighted final and selected-candidate states differ")
    if (
        checkpoint.get("training") != identity
        or checkpoint.get("predecessor_gate_evidence") != evidence
        or checkpoint.get("loss_policy") != runner.loss_policy_identity()
        or checkpoint.get("selection_provenance") != provenance
        or checkpoint.get("source_selection") != "evisirst_v2_validation_split"
        or checkpoint.get("model_metadata") != expected_model_metadata
    ):
        fail("weighted final checkpoint provenance differs")
    emit("SUMMARY_OK")
else:
    fail("unknown validation action")
'''


class WeightedDSWatcherError(RuntimeError):
    """The fixed fallback watcher cannot proceed safely."""


class GateDecision(Enum):
    WAIT = "WAIT"
    INVALID = "INVALID"
    PASS = "PASS"
    FAIL = "FAIL"


class RunMode(Enum):
    FRESH = "FRESH"
    RESUME = "RESUME"
    WAIT = "WAIT"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class GPUIdentity:
    index: int
    uuid: str
    bus_id: str
    memory_used_mib: int


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _safe_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        return False
    try:
        return path.resolve(strict=True) == path
    except (FileNotFoundError, RuntimeError, OSError):
        return False


def _require_missing_or_regular(path: Path, *, label: str) -> bool:
    """Return whether a fixed path is regular; reject every unsafe occupant."""

    if _safe_regular_file(path):
        return True
    if path.exists() or path.is_symlink():
        raise WeightedDSWatcherError(f"{label} is not a safe regular file")
    return False


def _require_fixed_dependency(
    path: Path,
    *,
    label: str,
    allow_symlink: bool,
    expected_resolved: Path | None = None,
    require_executable: bool,
) -> Path:
    try:
        lexical = path.lstat()
        resolved = path.resolve(strict=True)
        target = resolved.stat(follow_symlinks=False)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise WeightedDSWatcherError(f"{label} is unavailable or unsafe") from exc
    if (
        not stat.S_ISREG(target.st_mode)
        or (stat.S_ISLNK(lexical.st_mode) and not allow_symlink)
        or (not stat.S_ISLNK(lexical.st_mode) and resolved != path)
        or (expected_resolved is not None and resolved != expected_resolved)
        or (require_executable and not os.access(resolved, os.X_OK))
    ):
        raise WeightedDSWatcherError(f"{label} is unavailable or unsafe")
    return resolved


def validate_fixed_dependencies() -> None:
    """Validate fixed launchers without creating locks or run directories."""

    _require_fixed_dependency(
        PYTHON_BIN,
        label="fixed Python",
        allow_symlink=True,
        expected_resolved=PYTHON_RESOLVED_TARGET,
        require_executable=True,
    )
    for executable, label, require_executable in (
        (TIME_BIN, "fixed time executable", True),
        (NVIDIA_SMI_BIN, "fixed nvidia-smi", True),
        (TRAINER, "fixed weighted trainer", False),
    ):
        _require_fixed_dependency(
            executable,
            label=label,
            allow_symlink=False,
            require_executable=require_executable,
        )


def _require_safe_lock_parent(parent: Path) -> None:
    runs_root = PROJECT_ROOT / "runs"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise WeightedDSWatcherError("repository runs directory is unsafe")
    try:
        parent.relative_to(runs_root)
    except ValueError as exc:
        raise WeightedDSWatcherError("lock path escapes repository runs") from exc
    parent.mkdir(parents=True, exist_ok=True)
    current = runs_root
    for component in parent.relative_to(runs_root).parts:
        current = current / component
        if current.is_symlink() or not current.is_dir():
            raise WeightedDSWatcherError("lock parent contains an unsafe component")
    if parent.resolve(strict=True) != parent:
        raise WeightedDSWatcherError("lock parent identity differs")


def _open_lock(path: Path, *, inheritable: bool) -> int:
    _require_safe_lock_parent(path.parent)
    if path.is_symlink():
        raise WeightedDSWatcherError(f"lock is a symbolic link: {path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise WeightedDSWatcherError(f"cannot open fixed lock: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        on_disk = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(on_disk.st_mode)
            or (opened.st_dev, opened.st_ino) != (on_disk.st_dev, on_disk.st_ino)
        ):
            raise WeightedDSWatcherError(f"lock identity differs: {path}")
        os.set_inheritable(descriptor, inheritable)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def exclusive_watcher_lock(path: Path = WATCHER_LOCK) -> Iterator[int | None]:
    descriptor = _open_lock(path, inheritable=False)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        yield descriptor
    finally:
        os.close(descriptor)


def _validation_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.fspath(PROJECT_ROOT),
        }
    )
    return environment


def _run_validation_probe(
    action: str,
    *,
    process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str | None:
    try:
        completed = process_runner(
            [os.fspath(PYTHON_BIN), "-B", "-c", _VALIDATOR_PROGRAM, action],
            cwd=PROJECT_ROOT,
            env=_validation_environment(),
            capture_output=True,
            text=True,
            timeout=VALIDATION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    tokens = [
        line[len(_PROBE_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_PROBE_PREFIX)
    ]
    if len(tokens) != 1:
        return None
    return tokens[0]


def probe_canonical_decision() -> GateDecision:
    if not _require_missing_or_regular(
        CANONICAL_RESULT, label="canonical promotion result"
    ):
        return GateDecision.WAIT
    token = _run_validation_probe("gate")
    if token == GateDecision.PASS.value:
        return GateDecision.PASS
    if token == GateDecision.FAIL.value:
        return GateDecision.FAIL
    return GateDecision.INVALID


def validate_complete_summary() -> bool:
    if not _require_missing_or_regular(SUMMARY, label="weighted complete summary"):
        return False
    return _run_validation_probe("summary") == "SUMMARY_OK"


def inspect_run_mode() -> RunMode:
    if _require_missing_or_regular(SUMMARY, label="weighted complete summary"):
        return RunMode.COMPLETE
    if FORMAL_RUN_DIR.is_symlink():
        raise WeightedDSWatcherError("weighted formal run directory is a symlink")
    if not FORMAL_RUN_DIR.exists():
        return RunMode.FRESH
    if not FORMAL_RUN_DIR.is_dir() or FORMAL_RUN_DIR.resolve(strict=True) != FORMAL_RUN_DIR:
        raise WeightedDSWatcherError("weighted formal run directory is unsafe")
    if _require_missing_or_regular(LAST_STATE, label="weighted last state"):
        return RunMode.RESUME

    # A runner may create its own lock before doing any work.  Nothing else is
    # safe to reinterpret as a fresh run when no committed last state exists.
    allowed = {".weighted_ds_v1.lock"}
    entries = tuple(FORMAL_RUN_DIR.iterdir())
    if all(entry.name in allowed and _safe_regular_file(entry) for entry in entries):
        return RunMode.FRESH
    return RunMode.WAIT


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
        return tuple(token.decode("utf-8", errors="strict") for token in tokens)
    except UnicodeDecodeError:
        return None


def _process_cwd(pid_directory: Path) -> Path | None:
    try:
        return (pid_directory / "cwd").resolve(strict=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _is_python_process(pid_directory: Path) -> bool:
    try:
        executable = (pid_directory / "exe").resolve(strict=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False
    return executable.name.startswith("python")


def _argv_has_script(argv: Sequence[str], *, cwd: Path, expected: Path) -> bool:
    for token in argv:
        if token == os.fspath(expected):
            return True
        if token.startswith("-") or not token.endswith(".py"):
            continue
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
        if token == option and index + 1 < len(argv) and argv[index + 1] == expected:
            return True
    return False


def argv_is_formal_weighted_trainer(argv: Sequence[str], *, cwd: Path) -> bool:
    if not _argv_has_script(argv, cwd=cwd, expected=TRAINER):
        return False
    if not _argv_has_option_value(argv, "--run-seed", str(RUN_SEED)):
        return False
    smoke_options = (
        "--smoke",
        "--smoke-max-train-samples",
        "--smoke-max-val-samples",
    )
    return not any(
        token == option or token.startswith(option + "=")
        for token in argv
        for option in smoke_options
    )


def matching_formal_trainers(
    *, proc_root: Path = PROC_ROOT, exclude_pid: int | None = None
) -> tuple[int, ...]:
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise WeightedDSWatcherError("cannot enumerate /proc safely") from exc
    matches: list[int] = []
    for pid_directory in entries:
        if not pid_directory.name.isascii() or not pid_directory.name.isdecimal():
            continue
        pid = int(pid_directory.name)
        if pid == exclude_pid or not _is_python_process(pid_directory):
            continue
        argv = _read_cmdline(pid_directory)
        cwd = _process_cwd(pid_directory)
        if argv is not None and cwd is not None and argv_is_formal_weighted_trainer(
            argv, cwd=cwd
        ):
            matches.append(pid)
    return tuple(sorted(matches))


def _run_capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(
            args=list(command), returncode=127, stdout="", stderr=str(exc)
        )


def _parse_gpu_row(row: str) -> GPUIdentity | None:
    fields = tuple(field.strip() for field in row.split(","))
    if len(fields) != 4:
        return None
    raw_index, uuid, bus_id, raw_memory = fields
    if (
        not raw_index.isascii()
        or not raw_index.isdecimal()
        or _GPU_UUID_RE.fullmatch(uuid) is None
        or _GPU_BUS_RE.fullmatch(bus_id) is None
        or not raw_memory.isascii()
        or not raw_memory.isdecimal()
    ):
        return None
    return GPUIdentity(
        index=int(raw_index),
        uuid=uuid,
        bus_id=bus_id.upper(),
        memory_used_mib=int(raw_memory),
    )


def list_gpu_identities() -> tuple[GPUIdentity, ...]:
    completed = _run_capture(
        [
            os.fspath(NVIDIA_SMI_BIN),
            "--query-gpu=index,uuid,pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.returncode != 0:
        return ()
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    identities = tuple(_parse_gpu_row(row) for row in rows)
    if not identities or any(identity is None for identity in identities):
        return ()
    concrete = tuple(identity for identity in identities if identity is not None)
    if (
        len({identity.index for identity in concrete}) != len(concrete)
        or len({identity.uuid for identity in concrete}) != len(concrete)
        or len({identity.bus_id for identity in concrete}) != len(concrete)
    ):
        return ()
    return tuple(sorted(concrete, key=lambda identity: identity.index))


def gpu_identity_is_idle(expected: GPUIdentity) -> bool:
    identity_query = _run_capture(
        [
            os.fspath(NVIDIA_SMI_BIN),
            "-i",
            expected.uuid,
            "--query-gpu=index,uuid,pci.bus_id,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if identity_query.returncode != 0:
        return False
    rows = [row for row in identity_query.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        return False
    observed = _parse_gpu_row(rows[0])
    if (
        observed is None
        or observed.index != expected.index
        or observed.uuid != expected.uuid
        or observed.bus_id != expected.bus_id
        or observed.memory_used_mib > MEMORY_IDLE_LIMIT_MIB
    ):
        return False
    applications = _run_capture(
        [
            os.fspath(NVIDIA_SMI_BIN),
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if applications.returncode != 0:
        return False
    app_uuids = tuple(
        line.strip() for line in applications.stdout.splitlines() if line.strip()
    )
    if any(_GPU_UUID_RE.fullmatch(uuid) is None for uuid in app_uuids):
        return False
    return sum(uuid == expected.uuid for uuid in app_uuids) == 0


@contextmanager
def claimed_gpu_lock(identity: GPUIdentity) -> Iterator[int | None]:
    lock_path = GPU_LOCK_ROOT / f"{identity.uuid}.lock"
    descriptor = _open_lock(lock_path, inheritable=True)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        # This is the required third check after two ten-second observations.
        if not gpu_identity_is_idle(identity):
            yield None
            return
        yield descriptor
    finally:
        os.close(descriptor)


def fixed_training_command(*, resume: bool) -> list[str]:
    command = [
        os.fspath(TIME_BIN),
        "-v",
        os.fspath(PYTHON_BIN),
        os.fspath(TRAINER),
        "--dataset-root",
        os.fspath(DATASET_ROOT),
        "--split-root",
        os.fspath(SPLIT_ROOT),
        "--dataset",
        DATASET,
        "--target-mode",
        TARGET_MODE,
        "--architecture-seed",
        str(ARCHITECTURE_SEED),
        "--run-seed",
        str(RUN_SEED),
        "--device",
        "cuda:0",
        "--epochs",
        str(EPOCHS),
        "--warmup-epochs",
        str(WARMUP_EPOCHS),
        "--allow-sample-level-fallback",
    ]
    if resume:
        command.append("--resume")
    return command


def run_trainer_once(*, resume: bool, gpu: GPUIdentity, gpu_lock_fd: int) -> int:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu.uuid,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            GPU_LOCK_FD_ENV: str(gpu_lock_fd),
            GPU_UUID_ENV: gpu.uuid,
            GPU_BUS_ENV: gpu.bus_id,
            GPU_INDEX_ENV: str(gpu.index),
        }
    )
    try:
        completed = subprocess.run(
            fixed_training_command(resume=resume),
            cwd=PROJECT_ROOT,
            env=environment,
            pass_fds=(gpu_lock_fd,),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 127
    return int(completed.returncode)


def _update_idle_streaks(
    previous: dict[str, tuple[GPUIdentity, int]],
    identities: Sequence[GPUIdentity],
) -> dict[str, tuple[GPUIdentity, int]]:
    current: dict[str, tuple[GPUIdentity, int]] = {}
    for identity in identities:
        if not gpu_identity_is_idle(identity):
            continue
        old = previous.get(identity.uuid)
        count = old[1] + 1 if old is not None and old[0] == identity else 1
        current[identity.uuid] = (identity, count)
    return current


def watch_forever(
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    gate_probe: Callable[[], GateDecision] = probe_canonical_decision,
    trainer_runner: Callable[..., int] = run_trainer_once,
) -> int:
    idle_streaks: dict[str, tuple[GPUIdentity, int]] = {}
    failure_backoff = INITIAL_FAILURE_BACKOFF_SECONDS
    last_status = 0.0

    while True:
        decision = gate_probe()
        if decision is GateDecision.PASS:
            _log("canonical complete-target gate is validated PASS; fallback is permanently blocked")
            return 0
        if decision is not GateDecision.FAIL:
            idle_streaks.clear()
            now = monotonic()
            if now - last_status >= STATUS_SECONDS:
                _log(
                    "waiting for a complete, freshly validated canonical decision; "
                    f"state={decision.value}"
                )
                last_status = now
            sleep(POLL_SECONDS)
            continue

        mode = inspect_run_mode()
        if mode is RunMode.COMPLETE:
            if validate_complete_summary():
                _log("weighted-DS formal summary is complete and validated; exiting")
                return 0
            idle_streaks.clear()
            _log("weighted summary exists but validation is not yet successful; waiting")
            sleep(POLL_SECONDS)
            continue
        if mode is RunMode.WAIT:
            idle_streaks.clear()
            _log("weighted run has no resumable commit and is not safely fresh; waiting")
            sleep(POLL_SECONDS)
            continue

        active = matching_formal_trainers(exclude_pid=os.getpid())
        if active:
            idle_streaks.clear()
            now = monotonic()
            if now - last_status >= STATUS_SECONDS:
                _log(f"matching formal weighted trainer already running; pids={list(active)}")
                last_status = now
            sleep(POLL_SECONDS)
            continue

        idle_streaks = _update_idle_streaks(idle_streaks, list_gpu_identities())
        ready = sorted(
            (
                identity
                for identity, observations in idle_streaks.values()
                if observations >= 2
            ),
            key=lambda identity: identity.index,
        )
        launched = False
        for gpu in ready:
            with claimed_gpu_lock(gpu) as gpu_lock_fd:
                if gpu_lock_fd is None:
                    idle_streaks.pop(gpu.uuid, None)
                    continue

                # Fail closed on every mutable prerequisite immediately before
                # launch.  The runner itself repeats the canonical FAIL check.
                fresh_decision = gate_probe()
                if fresh_decision is GateDecision.PASS:
                    _log("canonical PASS observed at launch boundary; fallback blocked")
                    return 0
                fresh_mode = inspect_run_mode()
                if (
                    fresh_decision is not GateDecision.FAIL
                    or fresh_mode not in {RunMode.FRESH, RunMode.RESUME}
                    or matching_formal_trainers(exclude_pid=os.getpid())
                    or not gpu_identity_is_idle(gpu)
                ):
                    idle_streaks.pop(gpu.uuid, None)
                    continue

                resume = fresh_mode is RunMode.RESUME
                _log(
                    "launching fixed weighted-DS formal run; "
                    f"resume={resume} gpu_index={gpu.index} "
                    f"gpu_bus={gpu.bus_id} gpu_uuid={gpu.uuid}"
                )
                status = trainer_runner(
                    resume=resume, gpu=gpu, gpu_lock_fd=gpu_lock_fd
                )
                launched = True

            idle_streaks.clear()
            if inspect_run_mode() is RunMode.COMPLETE and validate_complete_summary():
                _log("weighted-DS formal run completed and validated")
                return 0
            _log(
                f"weighted trainer returned status {status} without a validated "
                f"complete summary; retrying after {failure_backoff}s"
            )
            sleep(failure_backoff)
            failure_backoff = min(
                MAX_FAILURE_BACKOFF_SECONDS, failure_backoff * 2
            )
            break

        if launched:
            continue
        now = monotonic()
        if now - last_status >= STATUS_SECONDS:
            streak_report = {
                uuid: observations
                for uuid, (_identity, observations) in sorted(idle_streaks.items())
            }
            _log(
                "canonical FAIL validated; waiting for two consecutive idle GPU "
                f"observations; streaks={streak_report or {'none': 0}}"
            )
            last_status = now
        sleep(POLL_SECONDS)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        raise WeightedDSWatcherError("this fixed watcher accepts no arguments")
    os.umask(0o077)
    validate_fixed_dependencies()
    with exclusive_watcher_lock(WATCHER_LOCK) as descriptor:
        if descriptor is None:
            _log("another weighted-DS watcher holds the fixed run lock; exiting")
            return 0
        _log("fixed weighted-DS fallback watcher started")
        return watch_forever()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_RESULT",
    "FORMAL_RUN_DIR",
    "GPUIdentity",
    "GPU_LOCK_ROOT",
    "GateDecision",
    "RunMode",
    "SUMMARY",
    "WATCHER_LOCK",
    "WeightedDSWatcherError",
    "argv_is_formal_weighted_trainer",
    "claimed_gpu_lock",
    "exclusive_watcher_lock",
    "fixed_training_command",
    "gpu_identity_is_idle",
    "inspect_run_mode",
    "list_gpu_identities",
    "main",
    "matching_formal_trainers",
    "probe_canonical_decision",
    "run_trainer_once",
    "validate_fixed_dependencies",
    "validate_complete_summary",
    "watch_forever",
]
