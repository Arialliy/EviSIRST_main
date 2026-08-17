#!/usr/bin/env python3
"""Evaluate EviSIRST or its SCTransNet baseline on one test split.

The default evaluator uses one common paper protocol for both models:
strict probability > 0.5, global foreground IoU, image-normalized IoU,
pixel F1, and one-to-one component matching for Pd/Fa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from skimage import measure
from torch.utils.data import DataLoader, Subset

from experiments.evisirst_data import (
    EviSIRSTTestDataset,
    TRAINING_DATASETS,
)
from experiments import evisirst_v2_selection as v2_selection
from load_models import DATASETS as EVALUATION_DATASETS
from load_models import load_baseline, load_evisirst
from model.EviSIRST import initialize_evisirst


PROJECT_ROOT = Path(__file__).resolve().parent
PROBABILITY_THRESHOLD = 0.5
MATCH_RADIUS = 3.0
TINY_AREA = 9

_CHECKPOINT_AUDIT_FIELDS = (
    "source_selection",
    "selection_is_optimistic",
    "selection_provenance",
    "run_seed",
    "split_manifest_sha256",
    "target_mode",
    "smoke",
    "training_identity_sha256",
)
_VALIDATION_SELECTED_REVALIDATION_FIELDS = (
    "dataset",
    "seed",
    "architecture_seed",
    "training",
    "test_split_accessed",
    "split_provenance",
    "split_seed",
    "data_tree_sha256",
    "data_tree_verified",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_TARGET_MODES = frozenset(("soft", "binary"))
_VALIDATION_DATASETS = frozenset(
    ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
)
_VALIDATION_SELECTED_SOURCE = "evisirst_v2_validation_split"
_VALIDATION_SELECTION_SCHEMA = "evisirst_v2_validation_selection/v1"
_VALIDATION_SELECTION_RULE = "evisirst_v2_independent_val_lexicographic/v1"
_VALIDATION_SELECTION_KIND = "independent_checkpoint"
_VALIDATION_SELECTION_TOLERANCE = 0.001
_VALIDATION_TRAINING_SCHEMA = (
    "evisirst_validation_selected_training/v1/run_identity"
)
_VALIDATION_DETERMINISM_SCHEMA = (
    "evisirst_validation_selected_determinism/v1"
)
_VALIDATION_SPLIT_SCHEMA = "evisirst_v2_train_val_split/v1"
_MAX_VALIDATION_RUN_SEED = (1 << 32) - 1
_MAX_AUDIT_JSON_DEPTH = 16
_MAX_AUDIT_JSON_NODES = 100000
_MAX_AUDIT_STRING_LENGTH = 16384


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("evisirst", "baseline"), default="evisirst")
    parser.add_argument("--dataset", choices=EVALUATION_DATASETS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional train.py EviSIRST checkpoint; omitted loads the published weight",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.model != "evisirst":
        parser.error("--checkpoint currently supports only --model evisirst")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    return args


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {index}")
    return device


def configure_inference_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_selection_provenance(value: Any) -> dict[str, Any]:
    """Copy one bounded strict-JSON provenance object without stringifying it."""

    node_count = 0

    def visit(item: Any, *, depth: int, context: str) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_AUDIT_JSON_NODES:
            raise ValueError("selection_provenance is too large")
        if depth > _MAX_AUDIT_JSON_DEPTH:
            raise ValueError("selection_provenance is too deeply nested")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int):
            if not -(2**63) <= item <= 2**63 - 1:
                raise ValueError(f"{context} integer is outside signed 64-bit range")
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{context} contains a non-finite float")
            return item
        if isinstance(item, str):
            if len(item) > _MAX_AUDIT_STRING_LENGTH:
                raise ValueError(f"{context} string is too long")
            return item
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key, nested in item.items():
                if not isinstance(key, str) or not key or len(key) > 256:
                    raise ValueError(f"{context} keys must be short non-empty strings")
                normalized[key] = visit(
                    nested,
                    depth=depth + 1,
                    context=f"{context}.{key}",
                )
            return normalized
        if isinstance(item, (tuple, list)):
            return [
                visit(
                    nested,
                    depth=depth + 1,
                    context=f"{context}[{index}]",
                )
                for index, nested in enumerate(item)
            ]
        raise ValueError(f"{context} contains a non-JSON value: {type(item).__name__}")

    normalized = visit(value, depth=0, context="selection_provenance")
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("selection_provenance must be a non-empty JSON object")
    return normalized


def _validate_validation_selected_identity(
    payload: Mapping[str, Any], audit: Mapping[str, Any]
) -> None:
    """Bind a validation-selected checkpoint to its trainer run identity."""

    required = (
        "dataset",
        "epoch",
        "seed",
        "architecture_seed",
        "training",
        "test_split_accessed",
        "split_provenance",
        "split_seed",
        "data_tree_sha256",
        "data_tree_verified",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(
            "validation_selected checkpoint is missing identity fields: "
            + ", ".join(missing)
        )

    dataset = payload["dataset"]
    if not isinstance(dataset, str) or dataset not in _VALIDATION_DATASETS:
        raise ValueError(
            "validation_selected checkpoint dataset is not a V2 source dataset"
        )
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("validation_selected checkpoint epoch must be positive")
    architecture_seed = payload["architecture_seed"]
    seed = payload["seed"]
    if (
        isinstance(architecture_seed, bool)
        or not isinstance(architecture_seed, int)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or architecture_seed != seed
        or seed != 42
    ):
        raise ValueError(
            "validation_selected architecture_seed and checkpoint seed must "
            "both equal 42"
        )
    test_split_accessed = payload["test_split_accessed"]
    if test_split_accessed is not False:
        raise ValueError(
            "validation_selected checkpoint must set test_split_accessed=false"
        )

    raw_training = payload["training"]
    if not isinstance(raw_training, Mapping):
        raise ValueError("validation_selected checkpoint training must be a mapping")
    try:
        training = _normalize_selection_provenance(raw_training)
    except ValueError as exc:
        raise ValueError(
            "validation_selected checkpoint training must be bounded strict JSON"
        ) from exc
    raw_identity_sha256 = training.get("identity_sha256")
    if (
        not isinstance(raw_identity_sha256, str)
        or _SHA256_PATTERN.fullmatch(raw_identity_sha256) is None
    ):
        raise ValueError("checkpoint training.identity_sha256 is malformed")
    training_identity_sha256 = raw_identity_sha256.lower()
    if training_identity_sha256 != audit["training_identity_sha256"]:
        raise ValueError(
            "checkpoint training.identity_sha256 differs from "
            "training_identity_sha256"
        )

    canonical_training = dict(training)
    del canonical_training["identity_sha256"]
    try:
        canonical_bytes = json.dumps(
            canonical_training,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint training identity is not canonical JSON") from exc
    recomputed_identity_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    if training_identity_sha256 != recomputed_identity_sha256:
        raise ValueError("checkpoint training identity SHA-256 does not recompute")
    if training.get("schema") != _VALIDATION_TRAINING_SCHEMA:
        raise ValueError(
            "checkpoint training schema differs from the public R1 contract"
        )
    if training.get("model") != "EviSIRST":
        raise ValueError(
            "checkpoint training model differs from the public R1 contract"
        )

    required_training_contract = (
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
        "split_seed",
        "data_tree_sha256",
        "train_count",
        "val_count",
    )
    missing_training_contract = [
        field for field in required_training_contract if field not in training
    ]
    if missing_training_contract:
        raise ValueError(
            "checkpoint training identity is missing R1 contract fields: "
            + ", ".join(missing_training_contract)
        )
    epochs = training["epochs"]
    batch_size = training["batch_size"]
    workers = training["workers"]
    warmup_epochs = training["warmup_epochs"]
    val_interval = training["val_interval"]
    train_count = training["train_count"]
    val_count = training["val_count"]
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or epoch > epochs
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 0
        or isinstance(warmup_epochs, bool)
        or not isinstance(warmup_epochs, int)
        or not 0 <= warmup_epochs <= epochs
        or isinstance(val_interval, bool)
        or not isinstance(val_interval, int)
        or not 1 <= val_interval <= epochs
        or epoch % val_interval != 0
        or isinstance(train_count, bool)
        or not isinstance(train_count, int)
        or train_count < 1
        or isinstance(val_count, bool)
        or not isinstance(val_count, int)
        or val_count < 1
    ):
        raise ValueError(
            "checkpoint training epoch/data-loader schedule is malformed"
        )
    base_lr = training["base_lr"]
    min_lr = training["min_lr"]
    if (
        isinstance(base_lr, bool)
        or not isinstance(base_lr, (int, float))
        or isinstance(min_lr, bool)
        or not isinstance(min_lr, (int, float))
        or not math.isfinite(float(base_lr))
        or not math.isfinite(float(min_lr))
        or not 0.0 < float(min_lr) <= float(base_lr)
    ):
        raise ValueError("checkpoint training learning-rate contract is malformed")
    fixed_training_contract = {
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "selection_rule": _VALIDATION_SELECTION_RULE,
    }
    for field, expected in fixed_training_contract.items():
        if training[field] != expected:
            raise ValueError(
                f"checkpoint training {field} differs from the public R1 contract"
            )
    determinism = training["determinism_protocol"]
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("schema") != _VALIDATION_DETERMINISM_SCHEMA
        or not isinstance(determinism.get("selection"), Mapping)
        or determinism["selection"].get("rule_version")
        != _VALIDATION_SELECTION_RULE
        or determinism["selection"].get("miou_candidate_tolerance")
        != _VALIDATION_SELECTION_TOLERANCE
    ):
        raise ValueError(
            "checkpoint training determinism selection contract differs"
        )
    validation_evaluation = determinism.get("validation_evaluation")
    expected_validation_evaluation = {
        "version": "evisirst_validation_metrics/v1",
        "evaluation_head": "out",
        "prediction_threshold": 0.5,
        "prediction_threshold_operator": ">",
        "target_threshold": 0.5,
        "target_threshold_operator": ">",
        "match_radius": 3.0,
        "match_distance_operator": "<",
        "connected_component_connectivity": 2,
        "connected_component_neighborhood": "8-connected",
        "assignment_algorithm": "Hungarian/scipy.optimize.linear_sum_assignment",
        "tiny_area": 9,
        "tiny_area_operator": "<=",
    }
    if not isinstance(validation_evaluation, Mapping) or set(
        validation_evaluation
    ) != set(expected_validation_evaluation):
        raise ValueError(
            "checkpoint training validation evaluator fields differ"
        )
    for field, expected in expected_validation_evaluation.items():
        observed = validation_evaluation[field]
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                "checkpoint training validation evaluator "
                f"{field} differs from the public contract"
            )

    expected_training_fields = {
        "dataset": dataset,
        "run_seed": audit["run_seed"],
        "target_mode": audit["target_mode"],
        "manifest_sha256": audit["split_manifest_sha256"],
        "smoke": audit["smoke"],
        "architecture_seed": architecture_seed,
        "test_split_accessed": test_split_accessed,
    }
    missing_training = [
        field for field in expected_training_fields if field not in training
    ]
    if missing_training:
        raise ValueError(
            "checkpoint training identity is missing fields: "
            + ", ".join(missing_training)
        )
    for field, expected in expected_training_fields.items():
        observed = training[field]
        if field == "manifest_sha256":
            if (
                not isinstance(observed, str)
                or _SHA256_PATTERN.fullmatch(observed) is None
                or observed.lower() != expected
            ):
                raise ValueError(
                    "checkpoint training manifest_sha256 differs from "
                    "top-level identity"
                )
        elif type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"checkpoint training {field} differs from top-level identity"
            )
    provenance = audit["selection_provenance"]
    selected = provenance.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("checkpoint selection_provenance.selected must be a mapping")
    selected_epoch = selected.get("epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch != epoch
    ):
        raise ValueError(
            "checkpoint selected validation epoch differs from checkpoint epoch"
        )

    raw_split_provenance = payload["split_provenance"]
    if not isinstance(raw_split_provenance, Mapping):
        raise ValueError("checkpoint split_provenance must be a mapping")
    try:
        split_provenance = _normalize_selection_provenance(
            raw_split_provenance
        )
    except ValueError as exc:
        raise ValueError(
            "checkpoint split_provenance must be bounded strict JSON"
        ) from exc
    manifest_sha256 = split_provenance.get("manifest_sha256")
    if (
        split_provenance.get("schema") != _VALIDATION_SPLIT_SCHEMA
        or not isinstance(manifest_sha256, str)
        or _SHA256_PATTERN.fullmatch(manifest_sha256) is None
        or manifest_sha256.lower() != audit["split_manifest_sha256"]
    ):
        raise ValueError(
            "checkpoint split_provenance schema/manifest differs from "
            "top-level identity"
        )
    expected_manifest_path = f"splits/v2/{dataset}/manifest.json"
    source_index = split_provenance.get("source_index")
    outputs = split_provenance.get("outputs")
    if (
        split_provenance.get("manifest_relative_path")
        != expected_manifest_path
        or not isinstance(source_index, Mapping)
        or source_index.get("split") != "train"
        or source_index.get("relative_path")
        != f"{dataset}/img_idx/train_{dataset}.txt"
        or source_index.get("sample_count") != train_count + val_count
        or not isinstance(outputs, Mapping)
        or set(outputs) != {"train", "val"}
    ):
        raise ValueError(
            "checkpoint split_provenance source/output roles differ from "
            "the frozen V2 train/val contract"
        )
    for field in ("file_sha256", "ordered_ids_sha256"):
        value = source_index.get(field)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"checkpoint split source_index {field} is malformed"
            )
    for role, count in (("train", train_count), ("val", val_count)):
        output = outputs[role]
        if (
            not isinstance(output, Mapping)
            or output.get("relative_path") != f"splits/v2/{dataset}/{role}.txt"
            or output.get("sample_count") != count
        ):
            raise ValueError(
                f"checkpoint split {role} output differs from the V2 contract"
            )
        for field in ("file_sha256", "ordered_ids_sha256"):
            value = output.get(field)
            if (
                not isinstance(value, str)
                or _SHA256_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(
                    f"checkpoint split {role} {field} is malformed"
                )
    data_tree_sha256 = payload["data_tree_sha256"]
    split_seed = payload["split_seed"]
    if (
        not isinstance(data_tree_sha256, str)
        or _SHA256_PATTERN.fullmatch(data_tree_sha256) is None
        or split_provenance.get("data_tree_sha256") != data_tree_sha256.lower()
        or training["data_tree_sha256"] != data_tree_sha256.lower()
        or isinstance(split_seed, bool)
        or not isinstance(split_seed, int)
        or split_seed < 0
        or split_provenance.get("split_seed") != split_seed
        or training["split_seed"] != split_seed
        or split_provenance.get("train_count") != train_count
        or split_provenance.get("val_count") != val_count
    ):
        raise ValueError(
            "checkpoint split/data-tree identity differs across provenance"
        )
    if (
        payload["data_tree_verified"] is not True
        or split_provenance.get("data_tree_verified") is not True
        or split_provenance.get("test_index_opened") is not False
    ):
        raise ValueError(
            "checkpoint split_provenance must verify data and set "
            "test_index_opened=false"
        )


def _checkpoint_audit_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach optional clean-checkpoint audit fields."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint audit metadata source must be a mapping")
    validation_selected = payload.get("checkpoint_role") == "validation_selected"
    if validation_selected:
        missing = [field for field in _CHECKPOINT_AUDIT_FIELDS if field not in payload]
        if missing:
            raise ValueError(
                "validation_selected checkpoint is missing audit fields: "
                + ", ".join(missing)
            )
    audit: dict[str, Any] = {}
    if "source_selection" in payload:
        value = payload["source_selection"]
        if (
            not isinstance(value, str)
            or not value
            or value.strip() != value
            or len(value) > 256
            or not value.isprintable()
        ):
            raise ValueError("checkpoint source_selection is malformed")
        audit["source_selection"] = value
    if "selection_is_optimistic" in payload:
        value = payload["selection_is_optimistic"]
        if not isinstance(value, bool):
            raise ValueError("checkpoint selection_is_optimistic must be a boolean")
        audit["selection_is_optimistic"] = value
    if "selection_provenance" in payload:
        audit["selection_provenance"] = _normalize_selection_provenance(
            payload["selection_provenance"]
        )
    if "run_seed" in payload:
        value = payload["run_seed"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 2**63
        ):
            raise ValueError("checkpoint run_seed must be a non-negative integer")
        audit["run_seed"] = value
    if "split_manifest_sha256" in payload:
        value = payload["split_manifest_sha256"]
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("checkpoint split_manifest_sha256 is malformed")
        audit["split_manifest_sha256"] = value.lower()
    if "target_mode" in payload:
        value = payload["target_mode"]
        if not isinstance(value, str) or value not in _TARGET_MODES:
            raise ValueError(
                f"checkpoint target_mode must be one of {sorted(_TARGET_MODES)}"
            )
        audit["target_mode"] = value
    if "smoke" in payload:
        value = payload["smoke"]
        if not isinstance(value, bool):
            raise ValueError("checkpoint smoke must be an explicit boolean")
        if value:
            raise ValueError("smoke checkpoint cannot enter public test evaluation")
        audit["smoke"] = value
    if "training_identity_sha256" in payload:
        value = payload["training_identity_sha256"]
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("checkpoint training_identity_sha256 is malformed")
        audit["training_identity_sha256"] = value.lower()
    if validation_selected:
        if audit["run_seed"] > _MAX_VALIDATION_RUN_SEED:
            raise ValueError(
                "validation_selected run_seed exceeds the R1 uint32 contract"
            )
        if audit["source_selection"] != _VALIDATION_SELECTED_SOURCE:
            raise ValueError(
                "validation_selected checkpoint source_selection must identify "
                "the frozen V2 validation split"
            )
        if audit["selection_is_optimistic"] is not False:
            raise ValueError(
                "validation_selected checkpoint cannot be selection-optimistic"
            )
        provenance = audit["selection_provenance"]
        expected_provenance_fields = {
            "schema": _VALIDATION_SELECTION_SCHEMA,
            "rule_version": _VALIDATION_SELECTION_RULE,
            "selection_kind": _VALIDATION_SELECTION_KIND,
        }
        for field, expected in expected_provenance_fields.items():
            if provenance.get(field) != expected:
                raise ValueError(
                    "validation_selected selection_provenance "
                    f"{field} differs from the public contract"
                )
        tolerance = provenance.get("candidate_tolerance_raw")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) != _VALIDATION_SELECTION_TOLERANCE
        ):
            raise ValueError(
                "validation_selected candidate_tolerance_raw differs from "
                "the public contract"
            )
        if provenance.get("data_role") != "val":
            raise ValueError(
                "validation_selected selection_provenance data_role must be 'val'"
            )
        if provenance.get("test_selection_supported") is not False:
            raise ValueError(
                "validation_selected provenance must set "
                "test_selection_supported=false"
            )
        evaluated_records = provenance.get("evaluated_records")
        if not isinstance(evaluated_records, list) or not evaluated_records:
            raise ValueError(
                "validation_selected provenance must contain complete "
                "evaluated_records"
            )
        try:
            recomputed_provenance = v2_selection.select_independent_checkpoint(
                evaluated_records
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "validation_selected evaluated_records violate the public "
                "selection rule"
            ) from exc
        provenance_bytes = json.dumps(
            provenance,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        recomputed_bytes = json.dumps(
            recomputed_provenance,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if provenance_bytes != recomputed_bytes:
            raise ValueError(
                "validation_selected selection_provenance does not exactly "
                "recompute from evaluated_records"
            )
        if audit["smoke"] is not False:
            raise ValueError("validation_selected checkpoint must set smoke=false")
        _validate_validation_selected_identity(payload, audit)
    return audit


def _state_from_checkpoint(
    path: Path,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_sha256 = _sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "evisirst_clean_checkpoint/v1"
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") not in TRAINING_DATASETS
        or payload.get("seed") != 42
    ):
        raise ValueError("custom checkpoint identity differs from this test run")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or len(state) != 564 or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("custom EviSIRST checkpoint must contain 564 tensors")
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("custom EviSIRST checkpoint contains non-finite tensors")
    normalized_payload = dict(payload)
    normalized_payload.update(_checkpoint_audit_metadata(payload))
    return state, normalized_payload, checkpoint_sha256


def load_model(
    model_name: str,
    dataset: str,
    checkpoint: Path | None,
) -> tuple[nn.Module, dict[str, Any]]:
    if checkpoint is None:
        return (
            load_evisirst(dataset)
            if model_name == "evisirst"
            else load_baseline(dataset)
        )
    state, payload, checkpoint_sha256 = _state_from_checkpoint(checkpoint)
    training_dataset = str(payload["dataset"])
    model, metadata = initialize_evisirst(
        training_dataset, seed=42, training=False
    )
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("custom EviSIRST checkpoint state keys differ")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"custom checkpoint tensor contract differs for {key!r}")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("custom checkpoint strict load failed")
    model.eval()
    model.mode = "test"
    ready = dict(metadata)
    ready.update(
        {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_role": str(payload.get("checkpoint_role", "custom")),
            "epoch": payload.get("epoch"),
            "checkpoint_sha256": checkpoint_sha256,
            "training_dataset": training_dataset,
            "evaluation_dataset": dataset,
            "strict_load": True,
        }
    )
    ready.update(
        {
            field: payload[field]
            for field in _CHECKPOINT_AUDIT_FIELDS
            if field in payload
        }
    )
    if payload.get("checkpoint_role") == "validation_selected":
        ready.update(
            {
                field: payload[field]
                for field in _VALIDATION_SELECTED_REVALIDATION_FIELDS
                if field in payload
            }
        )
    return model, ready


class ValidationMetrics:
    """Additive fixed-threshold metrics used by the final paper protocol."""

    def __init__(self, threshold: float, match_radius: float, tiny_area: int) -> None:
        self.threshold = threshold
        self.match_radius = match_radius
        self.tiny_area = tiny_area
        self.intersection = 0
        self.union = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.image_ious: list[float] = []
        self.losses: list[float] = []
        self.target_count = 0
        self.matched_target_count = 0
        self.tiny_target_count = 0
        self.matched_tiny_target_count = 0
        self.predicted_object_count = 0
        self.unmatched_predicted_object_count = 0
        self.unmatched_predicted_pixels = 0
        self.valid_pixels = 0

    def update(self, probability: np.ndarray, target: np.ndarray, loss: float) -> None:
        prediction = probability > self.threshold
        target_binary = target > 0.5
        intersection = int(np.logical_and(prediction, target_binary).sum())
        union = int(np.logical_or(prediction, target_binary).sum())
        self.intersection += intersection
        self.union += union
        self.tp += intersection
        self.fp += int(np.logical_and(prediction, ~target_binary).sum())
        self.fn += int(np.logical_and(~prediction, target_binary).sum())
        self.image_ious.append(1.0 if union == 0 else intersection / union)
        self.losses.append(float(loss))
        self.valid_pixels += int(target_binary.size)

        predicted = measure.regionprops(measure.label(prediction, connectivity=2))
        targets = measure.regionprops(measure.label(target_binary, connectivity=2))
        self.predicted_object_count += len(predicted)
        self.target_count += len(targets)
        self.tiny_target_count += sum(region.area <= self.tiny_area for region in targets)

        matched_targets: set[int] = set()
        matched_predictions: set[int] = set()
        if targets and predicted:
            distances = np.empty((len(targets), len(predicted)), dtype=np.float64)
            for target_index, target_region in enumerate(targets):
                target_centroid = np.asarray(target_region.centroid)
                for pred_index, pred_region in enumerate(predicted):
                    distances[target_index, pred_index] = np.linalg.norm(
                        np.asarray(pred_region.centroid) - target_centroid
                    )
            reward = (min(len(targets), len(predicted)) + 1) * max(
                1.0, self.match_radius
            )
            real_cost = np.where(
                distances < self.match_radius,
                distances - reward,
                reward,
            )
            cost = np.concatenate(
                (real_cost, np.zeros((len(targets), len(targets)))), axis=1
            )
            assigned_targets, assigned_columns = linear_sum_assignment(cost)
            for target_index, column in zip(assigned_targets, assigned_columns):
                if column < len(predicted) and distances[target_index, column] < self.match_radius:
                    matched_targets.add(int(target_index))
                    matched_predictions.add(int(column))

        self.matched_target_count += len(matched_targets)
        self.matched_tiny_target_count += sum(
            targets[index].area <= self.tiny_area for index in matched_targets
        )
        unmatched = [
            region
            for index, region in enumerate(predicted)
            if index not in matched_predictions
        ]
        self.unmatched_predicted_object_count += len(unmatched)
        self.unmatched_predicted_pixels += sum(int(region.area) for region in unmatched)

    def compute(self) -> dict[str, float | int | None]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        denominator = precision + recall
        return {
            "test_loss": float(np.mean(self.losses)),
            "miou": self.intersection / max(1, self.union),
            "niou": float(np.mean(self.image_ious)),
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_f1": 0.0 if denominator == 0 else 2.0 * precision * recall / denominator,
            "pd": self.matched_target_count / max(1, self.target_count),
            "tiny_pd": (
                self.matched_tiny_target_count / self.tiny_target_count
                if self.tiny_target_count
                else None
            ),
            "fa": self.unmatched_predicted_pixels / max(1, self.valid_pixels),
            "false_objects_per_image": self.unmatched_predicted_object_count
            / max(1, len(self.image_ious)),
            "target_count": self.target_count,
            "matched_target_count": self.matched_target_count,
            "tiny_target_count": self.tiny_target_count,
            "matched_tiny_target_count": self.matched_tiny_target_count,
            "predicted_object_count": self.predicted_object_count,
            "unmatched_predicted_object_count": self.unmatched_predicted_object_count,
            "valid_pixel_count": self.valid_pixels,
        }


def _extract_hw(value: Any) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        flattened = value.reshape(-1)
        return int(flattened[0]), int(flattened[1])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(torch.as_tensor(value[0]).reshape(-1)[0]), int(
            torch.as_tensor(value[1]).reshape(-1)[0]
        )
    raise TypeError(f"unsupported image-size value: {type(value)!r}")


def final_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[-1]
    raise TypeError("model returned an unsupported prediction object")


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = PROBABILITY_THRESHOLD,
    match_radius: float = MATCH_RADIUS,
    tiny_area: int = TINY_AREA,
) -> dict[str, float | int | None]:
    model.eval()
    model.mode = "test"
    criterion = nn.BCELoss(reduction="mean")
    metrics = ValidationMetrics(threshold, match_radius, tiny_area)
    for images, masks, sizes, _sample_ids in loader:
        height, width = _extract_hw(sizes)
        images = images.to(device, non_blocking=True)
        raw_prediction = final_prediction(model(images))
        if not isinstance(raw_prediction, torch.Tensor) or raw_prediction.ndim != 4:
            raise RuntimeError("model must return one four-dimensional probability map")
        prediction = raw_prediction[:, :, :height, :width]
        target = masks[:, :, :height, :width].to(device, non_blocking=True)
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise RuntimeError("model output shape differs from the target")
        if prediction.shape[0] != 1 or prediction.shape[1] != 1:
            raise RuntimeError("test.py requires a B=1, C=1 probability map")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("model output contains non-finite values")
        if bool((prediction < 0).any()) or bool((prediction > 1).any()):
            raise RuntimeError("model output is not in the probability range [0, 1]")
        loss = criterion(prediction.float(), target.float())
        metrics.update(
            prediction[0, 0].float().cpu().numpy(),
            target[0, 0].float().cpu().numpy(),
            float(loss.item()),
        )
    return metrics.compute()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_result_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build the result JSON checkpoint block with validated audit provenance."""

    raw_path = metadata.get("checkpoint_path")
    if isinstance(raw_path, str) and raw_path:
        checkpoint_path = Path(raw_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        resolved_checkpoint = checkpoint_path.resolve()
        try:
            display_path = resolved_checkpoint.relative_to(PROJECT_ROOT).as_posix()
            path_scope = "repository-relative"
        except ValueError:
            display_path = f"<external>/{resolved_checkpoint.name}"
            path_scope = "external-path-redacted"
    else:
        display_path = None
        path_scope = "unavailable"
    result = {
        "path": display_path,
        "path_scope": path_scope,
        "epoch": metadata.get("epoch"),
        "role": metadata.get("checkpoint_role"),
        "sha256": metadata.get("checkpoint_sha256"),
    }
    result.update(_checkpoint_audit_metadata(metadata))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_inference_determinism()
    device = _device(args.device)
    model, metadata = load_model(args.model, args.dataset, args.checkpoint)
    training_dataset = str(metadata.get("training_dataset", args.dataset))
    if training_dataset not in TRAINING_DATASETS:
        raise ValueError("model training-dataset metadata is unsupported")
    dataset = EviSIRSTTestDataset(
        training_dataset,
        args.dataset,
        dataset_root=args.dataset_root,
    )
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    metrics = evaluate_model(
        model,
        loader,
        device,
        threshold=PROBABILITY_THRESHOLD,
        match_radius=MATCH_RADIUS,
        tiny_area=TINY_AREA,
    )
    result = {
        "schema": "evisirst_public_evaluation/v1",
        "model": args.model,
        "training_dataset": training_dataset,
        "evaluation_dataset": args.dataset,
        "dataset": args.dataset,
        "dataset_source": {
            "kind": "external_cli_dataset_root",
            "root_recorded": False,
            "dataset": args.dataset,
        },
        "sample_count": len(dataset),
        "threshold": PROBABILITY_THRESHOLD,
        "threshold_operator": ">",
        "match_radius": MATCH_RADIUS,
        "match_radius_operator": "<",
        "tiny_area": TINY_AREA,
        "protocol": "evisirst_public_common_evaluator_v1",
        "normalization_dataset": training_dataset,
        "normalization": dict(dataset.dataset.normalization)
        if isinstance(dataset, Subset)
        else dict(dataset.normalization),
        "checkpoint": _checkpoint_result_metadata(metadata),
        "metrics": metrics,
    }
    result["metrics_display"] = {
        "miou_percent": 100 * float(metrics["miou"]),
        "niou_percent": 100 * float(metrics["niou"]),
        "f1_percent": 100 * float(metrics["pixel_f1"]),
        "pd_percent": 100 * float(metrics["pd"]),
        "fa_times_1e6": 1e6 * float(metrics["fa"]),
    }
    if args.output_json is not None:
        _write_json_atomic(args.output_json, result)
    return result


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    metrics = result["metrics"]
    print(
        f"{result['model']} {result['dataset']} "
        f"mIoU={100 * float(metrics['miou']):.4f} "
        f"nIoU={100 * float(metrics['niou']):.4f} "
        f"F1={100 * float(metrics['pixel_f1']):.4f} "
        f"Pd={100 * float(metrics['pd']):.4f} "
        f"Fa={1e6 * float(metrics['fa']):.4f}e-6"
    )


if __name__ == "__main__":
    main()
