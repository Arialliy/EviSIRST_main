#!/usr/bin/env python3
"""Train one EviSIRST source model with V2 validation-only selection.

This R1 entry point preserves the clean model's historical six-output BCE,
Adam, and learning-rate recipe.  It consumes only the immutable V2 ``train``
and ``val`` artifacts; public test data is never constructed or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from skimage import measure
from torch.utils.data import DataLoader, Subset

import train as legacy_train
import model.EviSIRST as evisirst_model_module
from experiments import four_dataset_models_seed42_v1 as frozen_builder
from experiments import evisirst_v2_data as v2_data
from experiments import evisirst_v2_selection as selection
from experiments import three_dataset_v2_protocol as source_protocol
from experiments.evisirst_v2_data import (
    DEFAULT_SPLIT_ROOT,
    EviSIRSTV2TrainDataset,
    EviSIRSTV2ValDataset,
    TARGET_MODES,
    V2SplitContract,
)
from model.EviSIRST import initialize_evisirst


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "validation_selected"
ARCHITECTURE_SEED = 42
# ``legacy_train.configure_determinism`` forwards this value to NumPy's
# legacy MT19937 seeder, whose accepted range is exactly uint32.
MAX_RUN_SEED = (1 << 32) - 1
TRAINING_SCHEMA = "evisirst_validation_selected_training/v1"
CANDIDATE_SCHEMA = "evisirst_validation_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_clean_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_validation_history/v1"
SELECTION_PAYLOAD_SCHEMA = "evisirst_validation_selection_payload/v1"
R1_RECIPE = {
    "optimizer": "Adam",
    "loss": "sum_of_six_BCELoss_mean_terms",
    "base_lr": 1e-3,
    "min_lr": 1e-5,
    "warmup_epochs": 10,
    "amp": False,
    "normalization_mode": "legacy",
    "architecture_seed": ARCHITECTURE_SEED,
    "evaluation_head": "out (common evaluate_model final_prediction)",
}
R1_ADAM_GROUP_HYPERPARAMETERS = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "decoupled_weight_decay": False,
}
# These tensors are registered by the frozen SCTransNet graph but are not
# connected to its training forward.  Adam therefore never creates moments
# for them.  The set is an explicit architecture contract rather than a list
# learned from the untrusted resume payload.
R1_STRUCTURALLY_INACTIVE_PARAMETER_NAMES = frozenset(
    {
        "mtc.embeddings_3.position_embeddings",
        "mtc.embeddings_4.position_embeddings",
    }
    | {
        f"mtc.encoder.layer.{layer}.channel_attn.q{query}_attn{attention}"
        for layer in range(4)
        for query in range(1, 5)
        for attention in range(1, 5)
    }
)
PROBABILITY_THRESHOLD = 0.5
MATCH_RADIUS = 3.0
TINY_AREA = 9
EVALUATION_PROTOCOL_VERSION = "evisirst_validation_metrics/v1"
PREDICTION_THRESHOLD_OPERATOR = ">"
TARGET_THRESHOLD = 0.5
TARGET_THRESHOLD_OPERATOR = ">"
MATCH_DISTANCE_OPERATOR = "<"
CONNECTED_COMPONENT_CONNECTIVITY = 2
CONNECTED_COMPONENT_NEIGHBORHOOD = "8-connected"
ASSIGNMENT_ALGORITHM = "Hungarian/scipy.optimize.linear_sum_assignment"
TINY_AREA_OPERATOR = "<="
_CANDIDATE_NAME_RE = re.compile(r"^epoch_([0-9]+)\.pth\.tar$")


class ValidationSelectedTrainingError(ValueError):
    """The requested run violates the validation-selected R1 contract."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=source_protocol.DATASETS, required=True
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-mode", choices=TARGET_MODES, required=True)
    parser.add_argument("--architecture-seed", type=int, default=ARCHITECTURE_SEED)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-sample-level-fallback", action="store_true")
    parser.add_argument(
        "--smoke-max-train-samples",
        type=int,
        help="explicit smoke-only train subset; routes output under smoke/",
    )
    parser.add_argument(
        "--smoke-max-val-samples",
        type=int,
        help="explicit smoke-only validation subset; routes output under smoke/",
    )
    args = parser.parse_args(argv)
    if args.architecture_seed != ARCHITECTURE_SEED:
        parser.error("--architecture-seed is frozen at 42")
    if not 0 <= args.run_seed <= MAX_RUN_SEED:
        parser.error(f"--run-seed must be in [0, {MAX_RUN_SEED}]")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.val_interval < 1 or args.val_interval > args.epochs:
        parser.error("--val-interval must be in [1, epochs]")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if (
        not math.isfinite(args.base_lr)
        or not math.isfinite(args.min_lr)
        or not 0.0 < args.min_lr <= args.base_lr
    ):
        parser.error("learning rates must satisfy 0 < min-lr <= base-lr")
    for name in ("smoke_max_train_samples", "smoke_max_val_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_source_provenance() -> dict[str, Any]:
    """Hash the explicitly enumerated protocol and frozen-model source set."""

    source_paths = {
        "trainer": Path(__file__),
        "legacy_train": Path(legacy_train.__file__),
        "legacy_data": PROJECT_ROOT / "experiments" / "evisirst_data.py",
        "data": Path(v2_data.__file__),
        "split_protocol": Path(v2_data.split_protocol.__file__),
        "selection": Path(selection.__file__),
        "source_protocol": Path(source_protocol.__file__),
        "frozen_builder": Path(frozen_builder.__file__),
        "model_entry": Path(evisirst_model_module.__file__),
        "model_package": PROJECT_ROOT / "model" / "__init__.py",
    }
    internal_root = PROJECT_ROOT / "model" / "_internal"
    internal_paths = sorted(internal_root.glob("*.py"))
    if not internal_paths:
        raise ValidationSelectedTrainingError(
            "frozen model source tree is missing"
        )
    for path in internal_paths:
        source_paths[f"model_internal/{path.name}"] = path
    sources: dict[str, dict[str, str]] = {}
    for name, path in source_paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValidationSelectedTrainingError(
                f"{name} protocol source is not a regular file"
            )
        path = path.resolve(strict=True)
        try:
            relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise ValidationSelectedTrainingError(
                f"{name} protocol source is outside the repository"
            ) from exc
        sources[name] = {
            "relative_path": relative_path,
            "sha256": _sha256_file(path),
        }
    return {
        "schema": "evisirst_validation_selected_source_set/v2",
        "files": sources,
        "source_tree_sha256": _sha256_bytes(_canonical_json_bytes(sources)),
    }


def _determinism_protocol_identity() -> dict[str, Any]:
    """Return explicit protocol semantics that must match on strict resume."""

    source_provenance = _protocol_source_provenance()
    return {
        "schema": "evisirst_validation_selected_determinism/v1",
        "training_data": {
            "source_protocol_version": source_protocol.PROTOCOL_VERSION,
            "patch_size": source_protocol.PATCH_SIZE,
            "train_positive_crop_probability": (
                source_protocol.TRAIN_POSITIVE_CROP_PROBABILITY
            ),
            "augmentation_version": v2_data.AUGMENTATION_VERSION,
        },
        "validation_evaluation": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "evaluation_head": "out",
            "prediction_threshold": PROBABILITY_THRESHOLD,
            "prediction_threshold_operator": PREDICTION_THRESHOLD_OPERATOR,
            "target_threshold": TARGET_THRESHOLD,
            "target_threshold_operator": TARGET_THRESHOLD_OPERATOR,
            "match_radius": MATCH_RADIUS,
            "match_distance_operator": MATCH_DISTANCE_OPERATOR,
            "connected_component_connectivity": CONNECTED_COMPONENT_CONNECTIVITY,
            "connected_component_neighborhood": CONNECTED_COMPONENT_NEIGHBORHOOD,
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "tiny_area": TINY_AREA,
            "tiny_area_operator": TINY_AREA_OPERATOR,
        },
        "selection": {
            "rule_version": selection.INDEPENDENT_RULE_VERSION,
            "miou_candidate_tolerance": selection.MIOU_CANDIDATE_TOLERANCE,
        },
        "source_set_schema": source_provenance["schema"],
        "source_files": source_provenance["files"],
        "source_tree_sha256": source_provenance["source_tree_sha256"],
    }


def stable_uint63(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in ("evisirst_validation_selected_runtime_seed/v1", *parts):
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


class ValidationMetrics:
    """Additive common metrics used for validation-only model selection."""

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

    def update(
        self, probability: np.ndarray, target: np.ndarray, loss: float
    ) -> None:
        prediction = probability > self.threshold
        target_binary = target > TARGET_THRESHOLD
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

        predicted = measure.regionprops(
            measure.label(
                prediction, connectivity=CONNECTED_COMPONENT_CONNECTIVITY
            )
        )
        targets = measure.regionprops(
            measure.label(
                target_binary, connectivity=CONNECTED_COMPONENT_CONNECTIVITY
            )
        )
        self.predicted_object_count += len(predicted)
        self.target_count += len(targets)
        self.tiny_target_count += sum(
            int(region.area <= self.tiny_area) for region in targets
        )
        matched_targets: set[int] = set()
        matched_predictions: set[int] = set()
        if targets and predicted:
            distances = np.empty((len(targets), len(predicted)), dtype=np.float64)
            for target_index, target_region in enumerate(targets):
                target_centroid = np.asarray(target_region.centroid)
                for prediction_index, prediction_region in enumerate(predicted):
                    distances[target_index, prediction_index] = np.linalg.norm(
                        np.asarray(prediction_region.centroid) - target_centroid
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
                if (
                    column < len(predicted)
                    and distances[target_index, column] < self.match_radius
                ):
                    matched_targets.add(int(target_index))
                    matched_predictions.add(int(column))
        self.matched_target_count += len(matched_targets)
        self.matched_tiny_target_count += sum(
            int(targets[index].area <= self.tiny_area)
            for index in matched_targets
        )
        unmatched = [
            region
            for index, region in enumerate(predicted)
            if index not in matched_predictions
        ]
        self.unmatched_predicted_object_count += len(unmatched)
        self.unmatched_predicted_pixels += sum(int(region.area) for region in unmatched)

    def compute(self) -> dict[str, float | int | None]:
        precision = float(self.tp / max(1, self.tp + self.fp))
        recall = float(self.tp / max(1, self.tp + self.fn))
        denominator = precision + recall
        return {
            "validation_loss": float(np.mean(self.losses)),
            "miou": float(self.intersection / max(1, self.union)),
            "niou": float(np.mean(self.image_ious)),
            "pixel_precision": precision,
            "pixel_recall": recall,
            "pixel_f1": float(
                0.0 if denominator == 0 else 2.0 * precision * recall / denominator
            ),
            "pd": float(
                self.matched_target_count / max(1, self.target_count)
            ),
            "tiny_pd": (
                float(self.matched_tiny_target_count / self.tiny_target_count)
                if self.tiny_target_count
                else None
            ),
            "fa": float(
                self.unmatched_predicted_pixels / max(1, self.valid_pixels)
            ),
            "false_objects_per_image": float(
                self.unmatched_predicted_object_count
                / max(1, len(self.image_ious))
            ),
            "target_count": int(self.target_count),
            "matched_target_count": int(self.matched_target_count),
            "tiny_target_count": int(self.tiny_target_count),
            "matched_tiny_target_count": int(self.matched_tiny_target_count),
            "predicted_object_count": int(self.predicted_object_count),
            "unmatched_predicted_object_count": int(
                self.unmatched_predicted_object_count
            ),
            "valid_pixel_count": int(self.valid_pixels),
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
    """Evaluate only the final/out probability head on a validation loader."""

    model.eval()
    model.mode = "test"
    criterion = nn.BCELoss(reduction="mean")
    metrics = ValidationMetrics(threshold, match_radius, tiny_area)
    for images, masks, sizes, _sample_ids in loader:
        height, width = _extract_hw(sizes)
        images = images.to(device, non_blocking=True)
        prediction = final_prediction(model(images))[:, :, :height, :width]
        target = masks[:, :, :height, :width].to(device, non_blocking=True)
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise RuntimeError("model output shape differs from validation target")
        if prediction.shape[0] != 1 or prediction.shape[1] != 1:
            raise RuntimeError("validation requires a B=1, C=1 probability map")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("model output contains non-finite values")
        if bool((prediction < 0).any()) or bool((prediction > 1).any()):
            raise RuntimeError("model output is outside probability range [0, 1]")
        loss = criterion(prediction.float(), target.float())
        metrics.update(
            prediction[0, 0].float().cpu().numpy(),
            target[0, 0].float().cpu().numpy(),
            float(loss.item()),
        )
    return metrics.compute()


def _is_smoke(args: argparse.Namespace) -> bool:
    return (
        args.smoke_max_train_samples is not None
        or args.smoke_max_val_samples is not None
    )


def require_run_seed(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_RUN_SEED
    ):
        raise ValidationSelectedTrainingError(
            f"run_seed must be an integer in [0, {MAX_RUN_SEED}]"
        )
    return value


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path | bool]:
    """Resolve a smoke-isolated output strictly below ``PROJECT_ROOT/runs``."""

    raw_root = args.output_root
    candidate = raw_root if raw_root.is_absolute() else PROJECT_ROOT / raw_root
    repository_runs = PROJECT_ROOT / "runs"
    if repository_runs.is_symlink():
        raise ValidationSelectedTrainingError(
            "the repository runs directory must not be a symlink"
        )
    allowed_root = repository_runs.resolve()
    try:
        allowed_root.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValidationSelectedTrainingError(
            "the repository runs directory resolves outside the repository"
        ) from exc

    # Check the lexical path before resolving it so that a symlink which stays
    # inside ``runs`` cannot silently redirect a declared output tree.  A
    # resolved-containment check alone cannot distinguish that redirection.
    lexical_allowed_root = Path(os.path.abspath(repository_runs))
    lexical_candidate = Path(os.path.abspath(candidate))
    try:
        lexical_relative_root = lexical_candidate.relative_to(
            lexical_allowed_root
        )
    except ValueError as exc:
        raise ValidationSelectedTrainingError(
            "output root must be a strict child of the repository runs directory"
        ) from exc
    if not lexical_relative_root.parts:
        raise ValidationSelectedTrainingError(
            "the repository runs directory itself is too broad for output"
        )
    current = lexical_allowed_root
    for component in lexical_relative_root.parts:
        current = current / component
        if current.is_symlink():
            raise ValidationSelectedTrainingError(
                "output root must not contain symlink components"
            )

    output_root = candidate.resolve()
    try:
        resolved_relative_root = output_root.relative_to(allowed_root)
    except ValueError as exc:
        raise ValidationSelectedTrainingError(
            "resolved output root escapes the repository runs directory"
        ) from exc
    if not resolved_relative_root.parts:
        raise ValidationSelectedTrainingError(
            "the repository runs directory itself is too broad for output"
        )

    smoke = _is_smoke(args)
    branch = "smoke" if smoke else "formal"
    run_dir = (
        output_root
        / branch
        / args.dataset
        / args.target_mode
        / f"run_seed_{args.run_seed}"
    )
    if smoke:
        smoke_identity = {
            "epochs": args.epochs,
            "val_interval": args.val_interval,
            "max_train": args.smoke_max_train_samples,
            "max_val": args.smoke_max_val_samples,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "warmup_epochs": args.warmup_epochs,
        }
        suffix = _sha256_bytes(_canonical_json_bytes(smoke_identity))[:12]
        run_dir = run_dir / f"smoke_{suffix}"

    relative_run = run_dir.relative_to(output_root)
    current = output_root
    for component in relative_run.parts:
        current = current / component
        if current.is_symlink():
            raise ValidationSelectedTrainingError(
                "run path must not contain symlink components"
            )
    resolved_run_dir = run_dir.resolve()
    try:
        resolved_relative_run = resolved_run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValidationSelectedTrainingError(
            "resolved run path escapes the declared output root"
        ) from exc
    if not resolved_relative_run.parts:
        raise ValidationSelectedTrainingError(
            "resolved run path must be a strict child of the output root"
        )
    try:
        resolved_run_dir.relative_to(allowed_root)
    except ValueError as exc:
        raise ValidationSelectedTrainingError(
            "resolved run path escapes the repository runs directory"
        ) from exc
    run_dir = resolved_run_dir
    return {
        "output_root": output_root,
        "run_dir": run_dir,
        "candidate_dir": run_dir / "candidates",
        "latest": run_dir / "last_training_state.pth.tar",
        "history": run_dir / "validation_history.json",
        "final": run_dir / "EviSIRST.pth.tar",
        "summary": run_dir / "summary.json",
        "smoke": smoke,
    }


def enforce_grouping_policy(
    contract: V2SplitContract, *, allow_sample_level_fallback: bool
) -> dict[str, Any]:
    grouping = contract.manifest.get("grouping")
    if not isinstance(grouping, Mapping):
        raise ValidationSelectedTrainingError("split grouping metadata is missing")
    mode = grouping.get("mode")
    if mode == "sample_level_fallback":
        if not allow_sample_level_fallback:
            raise ValidationSelectedTrainingError(
                "split uses sample-level fallback; rerun with the explicit "
                "--allow-sample-level-fallback acknowledgement"
            )
        warning = grouping.get("warning")
        if not isinstance(warning, str) or not warning.strip():
            raise ValidationSelectedTrainingError("fallback warning is missing")
        return {
            "mode": mode,
            "sample_level_fallback_acknowledged": True,
            "warning": warning,
        }
    if mode != "explicit_group_mapping":
        raise ValidationSelectedTrainingError("unsupported split grouping mode")
    return {
        "mode": mode,
        "sample_level_fallback_acknowledged": False,
        "warning": None,
    }


def build_datasets(
    args: argparse.Namespace,
) -> tuple[EviSIRSTV2TrainDataset, EviSIRSTV2ValDataset, dict[str, Any]]:
    """Construct only V2 train/val datasets and enforce the fallback gate."""

    train_dataset = EviSIRSTV2TrainDataset(
        args.dataset,
        dataset_root=args.dataset_root,
        split_root=args.split_root,
        target_mode=args.target_mode,
        run_seed=args.run_seed,
        normalization_mode="legacy",
        verify_data_tree=True,
    )
    grouping = enforce_grouping_policy(
        train_dataset.contract,
        allow_sample_level_fallback=args.allow_sample_level_fallback,
    )
    val_dataset = EviSIRSTV2ValDataset(
        args.dataset,
        dataset_root=args.dataset_root,
        split_root=args.split_root,
        target_mode=args.target_mode,
        normalization_mode="legacy",
        verify_data_tree=True,
    )
    if (
        train_dataset.contract.manifest_sha256
        != val_dataset.contract.manifest_sha256
        or train_dataset.contract.data_tree_sha256
        != val_dataset.contract.data_tree_sha256
        or not train_dataset.contract.data_tree_verified
        or not val_dataset.contract.data_tree_verified
    ):
        raise ValidationSelectedTrainingError("train/val data contracts differ")
    return train_dataset, val_dataset, grouping


def _candidate_path(candidate_dir: Path, epoch: int) -> Path:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValidationSelectedTrainingError("candidate epoch must be positive")
    return candidate_dir / f"epoch_{epoch:04d}.pth.tar"


def frontier_file_plan(
    history: Sequence[Mapping[str, Any]], candidate_dir: str | Path
) -> dict[str, Any]:
    """Return the exact candidate files retained/deleted by the frontier."""

    directory = Path(candidate_dir)
    frontier = selection.retention_frontier_epochs(history)
    evaluated = sorted(int(record["epoch"]) for record in history)
    frontier_set = set(frontier)
    return {
        "frontier_epochs": frontier,
        "keep": tuple(_candidate_path(directory, epoch) for epoch in frontier),
        "delete": tuple(
            _candidate_path(directory, epoch)
            for epoch in evaluated
            if epoch not in frontier_set
        ),
    }


def build_validation_record(
    epoch: int, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValidationSelectedTrainingError("validation epoch must be positive")
    try:
        cloned_metrics = json.loads(
            json.dumps(dict(metrics), allow_nan=False, sort_keys=True)
        )
        record = {
            "epoch": int(epoch),
            "data_role": "val",
            "mIoU": float(metrics["miou"]),
            "Fa": float(metrics["fa"]),
            "Pd": float(metrics["pd"]),
            "evaluation_head": "out",
            "metrics": cloned_metrics,
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationSelectedTrainingError(
            "public validation metrics are incomplete or non-finite"
        ) from exc
    # Reuse the authoritative normalizer and validation-role checks.
    selection.select_independent_checkpoint([record])
    return record


def build_selection_payload(
    history: Sequence[Mapping[str, Any]],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    provenance = selection.select_independent_checkpoint(history)
    selected_epoch = int(provenance["selected"]["epoch"])
    if selected_epoch not in candidate_artifacts:
        raise ValidationSelectedTrainingError(
            "selected epoch is absent from retained frontier artifacts"
        )
    frontier = selection.retention_frontier_epochs(history)
    if tuple(sorted(candidate_artifacts)) != frontier:
        raise ValidationSelectedTrainingError(
            "candidate artifacts differ from the retention frontier"
        )
    return {
        "schema": SELECTION_PAYLOAD_SCHEMA,
        "data_role": "val",
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "selected_epoch": selected_epoch,
        "selected_candidate": dict(candidate_artifacts[selected_epoch]),
        "retention_frontier_epochs": list(frontier),
        "selection_provenance": provenance,
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
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


def _split_provenance(contract: V2SplitContract) -> dict[str, Any]:
    manifest = contract.manifest
    return {
        "schema": manifest["schema"],
        "manifest_relative_path": (
            f"splits/v2/{contract.dataset}/manifest.json"
        ),
        "manifest_sha256": contract.manifest_sha256,
        "split_seed": manifest["seeds"]["split_seed"],
        "source_index": dict(manifest["source_index"]),
        "outputs": {
            "train": dict(manifest["outputs"]["train"]),
            "val": dict(manifest["outputs"]["val"]),
        },
        "grouping": dict(manifest["grouping"]),
        "data_tree_sha256": contract.data_tree_sha256,
        "data_tree_verified": contract.data_tree_verified,
        "train_count": len(contract.train_ids),
        "val_count": len(contract.val_ids),
        "test_index_opened": False,
    }


def _run_identity(
    args: argparse.Namespace,
    contract: V2SplitContract,
    grouping: Mapping[str, Any],
    *,
    train_count: int,
    val_count: int,
    smoke: bool,
) -> dict[str, Any]:
    identity = {
        "schema": TRAINING_SCHEMA + "/run_identity",
        "model": "EviSIRST",
        "dataset": args.dataset,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": args.run_seed,
        "target_mode": args.target_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "val_interval": args.val_interval,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "selection_rule": selection.INDEPENDENT_RULE_VERSION,
        "determinism_protocol": _determinism_protocol_identity(),
        "manifest_sha256": contract.manifest_sha256,
        "split_seed": contract.manifest["seeds"]["split_seed"],
        "data_tree_sha256": contract.data_tree_sha256,
        "grouping_policy": dict(grouping),
        "train_count": train_count,
        "val_count": val_count,
        "smoke": smoke,
        "smoke_max_train_samples": args.smoke_max_train_samples,
        "smoke_max_val_samples": args.smoke_max_val_samples,
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = _sha256_bytes(_canonical_json_bytes(identity))
    return identity


def _validate_state_dict(
    value: Any, expected: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValidationSelectedTrainingError("checkpoint state keys differ")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValidationSelectedTrainingError("checkpoint state is malformed")
        reference = expected[key]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValidationSelectedTrainingError(
                f"checkpoint tensor contract differs for {key!r}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValidationSelectedTrainingError(
                f"checkpoint tensor is non-finite: {key!r}"
            )
        state[key] = tensor.detach().cpu()
    if len(state) != 564 or any(key.startswith("target_survival") for key in state):
        raise ValidationSelectedTrainingError(
            "checkpoint is not the clean 564-key graph"
        )
    return state


def _same_typed_value(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(observed) == len(expected) and all(
            _same_typed_value(left, right)
            for left, right in zip(observed, expected)
        )
    return bool(observed == expected)


def _require_optimizer_identity(
    identity: Mapping[str, Any], *, completed_epoch: int, total_epochs: int
) -> tuple[float, float, int]:
    required_integer_fields = ("epochs", "batch_size", "train_count")
    for name in required_integer_fields:
        value = identity.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationSelectedTrainingError(
                f"resume optimizer identity {name} is malformed"
            )
    if identity["epochs"] != total_epochs:
        raise ValidationSelectedTrainingError(
            "resume optimizer total epoch identity differs"
        )
    warmup_epochs = identity.get("warmup_epochs")
    if (
        isinstance(warmup_epochs, bool)
        or not isinstance(warmup_epochs, int)
        or not 0 <= warmup_epochs <= total_epochs
    ):
        raise ValidationSelectedTrainingError(
            "resume optimizer warmup identity is malformed"
        )
    learning_rates: list[float] = []
    for name in ("base_lr", "min_lr"):
        value = identity.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationSelectedTrainingError(
                f"resume optimizer identity {name} is malformed"
            )
        learning_rates.append(float(value))
    base_lr, min_lr = learning_rates
    if (
        not math.isfinite(base_lr)
        or not math.isfinite(min_lr)
        or not 0.0 < min_lr <= base_lr
    ):
        raise ValidationSelectedTrainingError(
            "resume optimizer learning-rate identity is malformed"
        )
    expected_lr = legacy_train.learning_rate_for_epoch(
        completed_epoch,
        total_epochs,
        base_lr,
        min_lr,
        warmup_epochs,
    )
    expected_step = completed_epoch * math.ceil(
        identity["train_count"] / identity["batch_size"]
    )
    return base_lr, expected_lr, expected_step


def _validate_adam_group(
    group: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    expected_lr: float,
    label: str,
) -> list[int]:
    expected_keys = {"lr", "params", *R1_ADAM_GROUP_HYPERPARAMETERS}
    if set(group) != expected_keys or set(reference) != expected_keys:
        raise ValidationSelectedTrainingError(
            f"{label} Adam param-group fields differ from the R1 recipe"
        )
    lr = group.get("lr")
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(float(lr))
        or float(lr) != expected_lr
    ):
        raise ValidationSelectedTrainingError(
            f"{label} Adam learning rate differs from the saved-epoch schedule"
        )
    for name, recipe_value in R1_ADAM_GROUP_HYPERPARAMETERS.items():
        current_value = reference.get(name)
        if not _same_typed_value(current_value, recipe_value):
            raise ValidationSelectedTrainingError(
                f"current Adam {name} differs from the R1 recipe"
            )
        if not _same_typed_value(group.get(name), current_value):
            raise ValidationSelectedTrainingError(
                f"{label} Adam {name} differs from the R1 recipe"
            )
    raw_ids = group.get("params")
    reference_ids = reference.get("params")
    if not isinstance(raw_ids, list) or not isinstance(reference_ids, list):
        raise ValidationSelectedTrainingError(
            f"{label} Adam parameter IDs are malformed"
        )
    if raw_ids != reference_ids:
        raise ValidationSelectedTrainingError(
            f"{label} Adam parameter ID structure differs"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw_ids
    ) or len(set(raw_ids)) != len(raw_ids):
        raise ValidationSelectedTrainingError(
            f"{label} Adam parameter IDs are malformed"
        )
    return list(raw_ids)


def _validate_adam_parameter_state(
    state: Any,
    *,
    parameter: torch.Tensor,
    expected_device: torch.device,
    expected_step: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {"step", "exp_avg", "exp_avg_sq"}
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValidationSelectedTrainingError(
            f"{label} Adam step/moment fields are incomplete or extra"
        )
    step = state["step"]
    if (
        not isinstance(step, torch.Tensor)
        or step.shape != torch.Size([])
        or step.dtype != torch.float32
        or step.device.type != "cpu"
        or not bool(torch.isfinite(step))
    ):
        raise ValidationSelectedTrainingError(
            f"{label} Adam step tensor is malformed"
        )
    step_value = float(step.item())
    if step_value != expected_step or not step_value.is_integer():
        raise ValidationSelectedTrainingError(
            f"{label} Adam step is outside the completed training stage"
        )
    for name in ("exp_avg", "exp_avg_sq"):
        moment = state[name]
        if (
            not isinstance(moment, torch.Tensor)
            or moment.layout != torch.strided
            or moment.shape != parameter.shape
            or moment.dtype != parameter.dtype
            or moment.device != expected_device
            or not bool(torch.isfinite(moment).all())
        ):
            raise ValidationSelectedTrainingError(
                f"{label} Adam {name} tensor contract differs"
            )
    if bool((state["exp_avg_sq"] < 0).any()):
        raise ValidationSelectedTrainingError(
            f"{label} Adam exp_avg_sq contains a negative value"
        )
    return state["step"], state["exp_avg"], state["exp_avg_sq"]


def _require_disjoint_adam_storage(
    tensors: Sequence[torch.Tensor], *, label: str
) -> None:
    """Reject shared optimizer storage that would couple future Adam updates."""

    storage_owners: dict[tuple[str, int], int] = {}
    for index, tensor in enumerate(tensors):
        if not tensor.is_contiguous() or tensor.numel() < 1:
            raise ValidationSelectedTrainingError(
                f"{label} Adam state tensors must be non-empty contiguous tensors"
            )
        storage_identity = (
            str(tensor.device),
            tensor.untyped_storage().data_ptr(),
        )
        if storage_identity in storage_owners:
            raise ValidationSelectedTrainingError(
                f"{label} Adam state tensors must not share storage"
            )
        storage_owners[storage_identity] = index


def _expected_adam_state_ids(
    *,
    model: nn.Module,
    live_parameters: Sequence[torch.nn.Parameter],
    parameter_ids: Sequence[int],
    identity: Mapping[str, Any],
) -> set[int]:
    """Bind lazy Adam state coverage to the trusted model graph, not payload."""

    model_parameters = list(model.parameters())
    if len(model_parameters) != len(live_parameters) or any(
        observed is not expected
        for observed, expected in zip(live_parameters, model_parameters)
    ):
        raise ValidationSelectedTrainingError(
            "current Adam parameters differ from model parameter order"
        )
    if len(parameter_ids) != len(live_parameters):
        raise ValidationSelectedTrainingError(
            "current Adam parameter IDs differ from live parameters"
        )

    name_by_object_id = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    if len(name_by_object_id) != len(model_parameters) or any(
        id(parameter) not in name_by_object_id for parameter in model_parameters
    ):
        raise ValidationSelectedTrainingError(
            "model parameter names are incomplete or aliased"
        )

    production_r1 = identity.get("model") == "EviSIRST"
    if production_r1:
        observed_names = set(name_by_object_id.values())
        missing_inactive = sorted(
            R1_STRUCTURALLY_INACTIVE_PARAMETER_NAMES.difference(observed_names)
        )
        if missing_inactive:
            raise ValidationSelectedTrainingError(
                "frozen R1 inactive-parameter contract differs from the model"
            )
        named_parameters = dict(model.named_parameters())
        if any(
            not named_parameters[name].requires_grad
            for name in R1_STRUCTURALLY_INACTIVE_PARAMETER_NAMES
        ):
            raise ValidationSelectedTrainingError(
                "frozen R1 inactive parameters must remain registered trainable "
                "tensors without optimizer moments"
            )
    else:
        # Small fixture models have no frozen-graph dead parameters.
        observed_names = set()

    expected: set[int] = set()
    for parameter_id, parameter in zip(parameter_ids, live_parameters):
        name = name_by_object_id[id(parameter)]
        structurally_inactive = (
            production_r1
            and name in R1_STRUCTURALLY_INACTIVE_PARAMETER_NAMES
        )
        if parameter.requires_grad and not structurally_inactive:
            expected.add(parameter_id)
    return expected


def _validate_and_load_adam_optimizer_state(
    *,
    optimizer_state: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: Mapping[str, Any],
    completed_epoch: int,
    total_epochs: int,
) -> None:
    """Strictly validate the R1 Adam state before and after loading it."""

    if type(optimizer) is not torch.optim.Adam or optimizer.state:
        raise ValidationSelectedTrainingError(
            "resume requires one fresh torch.optim.Adam optimizer"
        )
    base_lr, expected_lr, expected_step = _require_optimizer_identity(
        identity,
        completed_epoch=completed_epoch,
        total_epochs=total_epochs,
    )
    current = optimizer.state_dict()
    if set(current) != {"state", "param_groups"}:
        raise ValidationSelectedTrainingError("current Adam state is malformed")
    current_groups = current["param_groups"]
    if not isinstance(current_groups, list) or len(current_groups) != 1:
        raise ValidationSelectedTrainingError(
            "R1 requires exactly one Adam parameter group"
        )
    current_group = current_groups[0]
    current_ids = _validate_adam_group(
        current_group,
        reference=current_group,
        expected_lr=base_lr,
        label="current",
    )
    live_parameters = optimizer.param_groups[0].get("params")
    if (
        not isinstance(live_parameters, list)
        or len(live_parameters) != len(current_ids)
        or any(
            not isinstance(parameter, torch.nn.Parameter)
            for parameter in live_parameters
        )
    ):
        raise ValidationSelectedTrainingError(
            "current Adam live parameter structure is malformed"
        )
    parameters_by_id = dict(zip(current_ids, live_parameters))
    expected_state_ids = _expected_adam_state_ids(
        model=model,
        live_parameters=live_parameters,
        parameter_ids=current_ids,
        identity=identity,
    )

    if not isinstance(optimizer_state, Mapping) or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise ValidationSelectedTrainingError(
            "resume optimizer top-level structure is malformed"
        )
    saved_groups = optimizer_state["param_groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != 1:
        raise ValidationSelectedTrainingError(
            "resume Adam parameter-group structure differs"
        )
    saved_ids = _validate_adam_group(
        saved_groups[0],
        reference=current_group,
        expected_lr=expected_lr,
        label="resume",
    )
    saved_state = optimizer_state["state"]
    if not isinstance(saved_state, Mapping):
        raise ValidationSelectedTrainingError("resume Adam state map is malformed")
    if set(saved_state) != expected_state_ids:
        raise ValidationSelectedTrainingError(
            "resume Adam state must cover every trainable parameter exactly"
        )
    saved_state_tensors: list[torch.Tensor] = []
    for parameter_id, state in saved_state.items():
        if (
            isinstance(parameter_id, bool)
            or not isinstance(parameter_id, int)
            or parameter_id not in parameters_by_id
        ):
            raise ValidationSelectedTrainingError(
                "resume Adam state contains an unknown parameter ID"
            )
        saved_state_tensors.extend(
            _validate_adam_parameter_state(
                state,
                parameter=parameters_by_id[parameter_id],
                expected_device=torch.device("cpu"),
                expected_step=expected_step,
                label="resume",
            )
        )
    _require_disjoint_adam_storage(saved_state_tensors, label="resume")
    if not set(saved_state).issubset(saved_ids):
        raise ValidationSelectedTrainingError(
            "resume Adam state and parameter groups disagree"
        )

    try:
        optimizer.load_state_dict(dict(optimizer_state))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValidationSelectedTrainingError(
            "resume Adam load failed after validation"
        ) from exc

    loaded_groups = optimizer.param_groups
    loaded_parameters = loaded_groups[0].get("params") if loaded_groups else None
    if (
        len(loaded_groups) != 1
        or not isinstance(loaded_parameters, list)
        or len(loaded_parameters) != len(live_parameters)
        or any(
            observed is not expected
            for observed, expected in zip(loaded_parameters, live_parameters)
        )
    ):
        raise ValidationSelectedTrainingError(
            "loaded Adam live parameter structure differs"
        )
    loaded_group = dict(loaded_groups[0])
    loaded_group["params"] = saved_ids
    _validate_adam_group(
        loaded_group,
        reference=current_group,
        expected_lr=expected_lr,
        label="loaded",
    )
    expected_live_states = {parameters_by_id[key] for key in saved_state}
    if set(optimizer.state) != expected_live_states:
        raise ValidationSelectedTrainingError("loaded Adam state mapping differs")
    loaded_state_tensors: list[torch.Tensor] = []
    for parameter in expected_live_states:
        loaded_state_tensors.extend(
            _validate_adam_parameter_state(
                optimizer.state[parameter],
                parameter=parameter,
                expected_device=parameter.device,
                expected_step=expected_step,
                label="loaded",
            )
        )
    _require_disjoint_adam_storage(loaded_state_tensors, label="loaded")


def validate_resume_identity(
    payload: Any, expected_identity: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Reject any resume payload whose complete immutable identity differs."""

    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TRAINING_SCHEMA
        or payload.get("run_identity") != expected_identity
    ):
        raise ValidationSelectedTrainingError("resume run identity differs")
    return payload


def _regular_file(path: Path, *, parent: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValidationSelectedTrainingError(f"{label} is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(parent.resolve(strict=True))
    except ValueError as exc:
        raise ValidationSelectedTrainingError(
            f"{label} escapes the run directory"
        ) from exc
    return resolved


def _candidate_artifacts(
    history: Sequence[Mapping[str, Any]], candidate_dir: Path
) -> dict[int, dict[str, Any]]:
    if not history:
        if candidate_dir.exists() and any(candidate_dir.iterdir()):
            raise ValidationSelectedTrainingError(
                "candidate directory is non-empty before validation"
            )
        return {}
    plan = frontier_file_plan(history, candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for path in plan["keep"]:
        if path.is_symlink() or not path.is_file():
            raise ValidationSelectedTrainingError(
                "retention-frontier candidate is missing or non-regular"
            )
    # Do not delete dominated files here.  The new latest state is the commit
    # point; pruning before it would make the previous latest unrecoverable.
    artifacts: dict[int, dict[str, Any]] = {}
    for epoch, path in zip(plan["frontier_epochs"], plan["keep"]):
        artifacts[int(epoch)] = {
            "relative_path": f"candidates/{path.name}",
            "file_sha256": _sha256_file(path),
        }
    return artifacts


def _history_payload(
    *,
    identity: Mapping[str, Any],
    training_history: Sequence[Mapping[str, Any]],
    validation_history: Sequence[Mapping[str, Any]],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": HISTORY_SCHEMA,
        "data_role": "val",
        "test_split_accessed": False,
        "run_identity": dict(identity),
        "training_history": [dict(item) for item in training_history],
        "validation_history": [dict(item) for item in validation_history],
        "retention_frontier_epochs": sorted(candidate_artifacts),
        "candidate_artifacts": {
            str(epoch): dict(candidate_artifacts[epoch])
            for epoch in sorted(candidate_artifacts)
        },
    }


def _expected_validation_epochs(completed_epoch: int, interval: int) -> list[int]:
    return list(range(interval, completed_epoch + 1, interval))


def _validate_resume_history(
    *,
    completed_epoch: int,
    interval: int,
    training_history: Any,
    validation_history: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        not isinstance(training_history, list)
        or len(training_history) != completed_epoch
    ):
        raise ValidationSelectedTrainingError("resume training history is incomplete")
    normalized_training: list[dict[str, Any]] = []
    for expected_epoch, record in enumerate(training_history, start=1):
        if not isinstance(record, Mapping) or record.get("epoch") != expected_epoch:
            raise ValidationSelectedTrainingError(
                "resume training history order differs"
            )
        normalized_training.append(dict(record))
    if not isinstance(validation_history, list):
        raise ValidationSelectedTrainingError("resume validation history is malformed")
    expected_epochs = _expected_validation_epochs(completed_epoch, interval)
    observed_epochs = [
        int(record.get("epoch", -1))
        for record in validation_history
        if isinstance(record, Mapping)
    ]
    if (
        len(observed_epochs) != len(validation_history)
        or observed_epochs != expected_epochs
    ):
        raise ValidationSelectedTrainingError("resume validation schedule differs")
    normalized_validation = [dict(record) for record in validation_history]
    if normalized_validation:
        selection.select_independent_checkpoint(normalized_validation)
    return normalized_training, normalized_validation


def _candidate_epoch_from_name(path: Path) -> int:
    match = _CANDIDATE_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValidationSelectedTrainingError(
            "candidate directory contains an unexpected filename"
        )
    epoch = int(match.group(1))
    if epoch < 1:
        raise ValidationSelectedTrainingError("candidate filename epoch is invalid")
    return epoch


def _validate_candidate_payload(
    *,
    path: Path,
    run_dir: Path,
    identity: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    expected_record: Mapping[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    path = _regular_file(path, parent=run_dir, label="candidate")
    filename_epoch = _candidate_epoch_from_name(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CANDIDATE_SCHEMA
        or payload.get("run_identity") != identity
        or payload.get("epoch") != filename_epoch
        or not isinstance(payload.get("validation_record"), Mapping)
    ):
        raise ValidationSelectedTrainingError("candidate identity differs")
    record = dict(payload["validation_record"])
    if record.get("epoch") != filename_epoch:
        raise ValidationSelectedTrainingError("candidate validation epoch differs")
    selection.select_independent_checkpoint([record])
    if expected_record is not None and record != expected_record:
        raise ValidationSelectedTrainingError("candidate validation record differs")
    _validate_state_dict(payload.get("state_dict"), expected_state)
    return filename_epoch, record


def _validate_resume_candidates(
    *,
    run_dir: Path,
    candidate_dir: Path,
    identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    artifacts: Any,
    expected_state: Mapping[str, torch.Tensor],
    completed_epoch: int,
) -> dict[int, dict[str, Any]]:
    frontier = selection.retention_frontier_epochs(history) if history else ()
    if not isinstance(artifacts, Mapping):
        raise ValidationSelectedTrainingError("resume candidate map is malformed")
    normalized: dict[int, dict[str, Any]] = {}
    for raw_epoch, raw_artifact in artifacts.items():
        if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int):
            raise ValidationSelectedTrainingError("resume candidate epoch is malformed")
        if not isinstance(raw_artifact, Mapping):
            raise ValidationSelectedTrainingError(
                "resume candidate metadata is malformed"
            )
        normalized[raw_epoch] = dict(raw_artifact)
    if tuple(sorted(normalized)) != frontier:
        raise ValidationSelectedTrainingError("resume candidates differ from frontier")
    expected_files: set[Path] = set()
    by_epoch = {int(record["epoch"]): record for record in history}
    for epoch in frontier:
        expected_path = _candidate_path(candidate_dir, epoch)
        relative = normalized[epoch].get("relative_path")
        if relative != f"candidates/{expected_path.name}":
            raise ValidationSelectedTrainingError("resume candidate path differs")
        path = _regular_file(expected_path, parent=run_dir, label="candidate")
        expected_files.add(path)
        if normalized[epoch].get("file_sha256") != _sha256_file(path):
            raise ValidationSelectedTrainingError("resume candidate SHA-256 differs")
        _validate_candidate_payload(
            path=path,
            run_dir=run_dir,
            identity=identity,
            expected_state=expected_state,
            expected_record=by_epoch[epoch],
        )
    actual_files: set[Path] = set()
    if candidate_dir.exists():
        for path in candidate_dir.iterdir():
            regular = _regular_file(
                path, parent=run_dir, label="candidate directory entry"
            )
            actual_files.add(regular)
    missing = expected_files - actual_files
    if missing:
        raise ValidationSelectedTrainingError("resume frontier candidate is missing")
    # Crash recovery is deliberately after complete validation.  An extra is
    # safe only if it is a committed dominated record, or a candidate written
    # after the old latest but before the next atomic latest commit.
    for path in sorted(actual_files - expected_files):
        filename_epoch = _candidate_epoch_from_name(path)
        expected_record = by_epoch.get(filename_epoch)
        if filename_epoch <= completed_epoch and expected_record is None:
            raise ValidationSelectedTrainingError(
                "extra candidate is not in committed validation history"
            )
        observed_epoch, _ = _validate_candidate_payload(
            path=path,
            run_dir=run_dir,
            identity=identity,
            expected_state=expected_state,
            expected_record=expected_record,
        )
        if observed_epoch in frontier:
            raise ValidationSelectedTrainingError(
                "frontier candidate was misclassified as an extra"
            )
        path.unlink()
    return normalized


def _load_resume_state(
    *,
    path: Path,
    run_dir: Path,
    candidate_dir: Path,
    identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    total_epochs: int,
    val_interval: int,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    path = _regular_file(path, parent=run_dir, label="resume state")
    payload = validate_resume_identity(
        torch.load(path, map_location="cpu", weights_only=True), identity
    )
    completed = payload.get("epoch")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ValidationSelectedTrainingError("resume epoch is malformed")
    if completed < 1 or completed > total_epochs:
        raise ValidationSelectedTrainingError("resume epoch is outside this run")
    state = _validate_state_dict(payload.get("state_dict"), model.state_dict())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValidationSelectedTrainingError("resume model strict load failed")
    _validate_and_load_adam_optimizer_state(
        optimizer_state=payload.get("optimizer"),
        model=model,
        optimizer=optimizer,
        identity=identity,
        completed_epoch=completed,
        total_epochs=total_epochs,
    )
    train_history, val_history = _validate_resume_history(
        completed_epoch=completed,
        interval=val_interval,
        training_history=payload.get("training_history"),
        validation_history=payload.get("validation_history"),
    )
    artifacts = _validate_resume_candidates(
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        identity=identity,
        history=val_history,
        artifacts=payload.get("candidate_artifacts"),
        expected_state=model.state_dict(),
        completed_epoch=completed,
    )
    legacy_train._restore_rng_state(payload.get("rng"), device)
    return completed + 1, train_history, val_history, artifacts


def _save_candidate(
    *,
    path: Path,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    identity: Mapping[str, Any],
    validation_record: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"candidate already exists: {path}")
    _atomic_torch_save(
        path,
        {
            "schema": CANDIDATE_SCHEMA,
            "model": "EviSIRST",
            "dataset": identity["dataset"],
            "epoch": epoch,
            "run_identity": dict(identity),
            "validation_record": dict(validation_record),
            "state_dict": dict(state),
            "test_split_accessed": False,
        },
    )


def _load_selected_candidate(
    *,
    run_dir: Path,
    candidate_dir: Path,
    selection_payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    epoch = int(selection_payload["selected_epoch"])
    path = _candidate_path(candidate_dir, epoch)
    path = _regular_file(path, parent=run_dir, label="selected candidate")
    artifact = selection_payload["selected_candidate"]
    if artifact.get("file_sha256") != _sha256_file(path):
        raise ValidationSelectedTrainingError("selected candidate SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    record = next(item for item in history if int(item["epoch"]) == epoch)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CANDIDATE_SCHEMA
        or payload.get("run_identity") != identity
        or payload.get("epoch") != epoch
        or payload.get("validation_record") != record
    ):
        raise ValidationSelectedTrainingError("selected candidate identity differs")
    return _validate_state_dict(payload.get("state_dict"), expected_state)


def build_final_checkpoint_payload(
    *,
    args: argparse.Namespace,
    state_dict: Mapping[str, torch.Tensor],
    normalization: Mapping[str, float],
    normalization_provenance: Mapping[str, Any],
    identity: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if (
        selection_payload.get("data_role") != "val"
        or selection_payload.get("selection_is_optimistic") is not False
        or selection_payload.get("optimistic") is not False
    ):
        raise ValidationSelectedTrainingError(
            "final checkpoint requires non-optimistic validation selection"
        )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": args.dataset,
        "checkpoint_role": "validation_selected",
        "epoch": selection_payload["selected_epoch"],
        "seed": ARCHITECTURE_SEED,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": args.run_seed,
        "state_dict": dict(state_dict),
        "target_mode": args.target_mode,
        "normalization": dict(normalization),
        "normalization_provenance": dict(normalization_provenance),
        "training": dict(identity),
        "training_identity_sha256": identity["identity_sha256"],
        "split_provenance": dict(split_provenance),
        "split_seed": split_provenance["split_seed"],
        "split_manifest_sha256": split_provenance["manifest_sha256"],
        "data_tree_sha256": split_provenance["data_tree_sha256"],
        "data_tree_verified": split_provenance["data_tree_verified"],
        "selection_provenance": dict(
            selection_payload["selection_provenance"]
        ),
        "source_selection": selection_payload["source_selection"],
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "model_metadata": dict(model_metadata),
        "smoke": bool(smoke),
    }


def run(args: argparse.Namespace) -> Path:
    require_run_seed(args.run_seed)
    paths = resolve_run_paths(args)
    run_dir = paths["run_dir"]
    candidate_dir = paths["candidate_dir"]
    latest_path = paths["latest"]
    history_path = paths["history"]
    final_path = paths["final"]
    summary_path = paths["summary"]
    assert isinstance(run_dir, Path)
    assert isinstance(candidate_dir, Path)
    assert isinstance(latest_path, Path)
    assert isinstance(history_path, Path)
    assert isinstance(final_path, Path)
    assert isinstance(summary_path, Path)
    smoke = bool(paths["smoke"])

    # V2 datasets are the only data constructors reachable from this entry.
    full_train, full_val, grouping = build_datasets(args)
    train_data: Any = full_train
    val_data: Any = full_val
    if args.smoke_max_train_samples is not None:
        train_data = Subset(
            full_train,
            range(min(args.smoke_max_train_samples, len(full_train))),
        )
    if args.smoke_max_val_samples is not None:
        val_data = Subset(
            full_val,
            range(min(args.smoke_max_val_samples, len(full_val))),
        )
    if len(train_data) < 1 or len(val_data) < 1:
        raise ValidationSelectedTrainingError("train/validation data is empty")

    identity = _run_identity(
        args,
        full_train.contract,
        grouping,
        train_count=len(train_data),
        val_count=len(val_data),
        smoke=smoke,
    )
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists: {summary_path}")
    if not args.resume and any(
        path.exists() for path in (latest_path, final_path, history_path, candidate_dir)
    ):
        raise FileExistsError(f"run directory already contains artifacts: {run_dir}")
    legacy_train.configure_determinism(ARCHITECTURE_SEED)
    device = legacy_train.require_device(args.device)
    model, model_metadata = initialize_evisirst(
        args.dataset,
        seed=ARCHITECTURE_SEED,
        training=True,
    )
    model.to(device)
    if len(model.state_dict()) != 564 or hasattr(model, "target_survival"):
        raise ValidationSelectedTrainingError("R1 requires the clean 564-key model")
    # Initial weights are fixed by architecture_seed; subsequent runtime RNG is
    # controlled independently by run_seed.
    legacy_train.configure_determinism(args.run_seed)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    start_epoch = 1
    training_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    candidate_artifacts: dict[int, dict[str, Any]] = {}
    if args.resume:
        (
            start_epoch,
            training_history,
            validation_history,
            candidate_artifacts,
        ) = _load_resume_state(
            path=latest_path,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            identity=identity,
            model=model,
            optimizer=optimizer,
            device=device,
            total_epochs=args.epochs,
            val_interval=args.val_interval,
        )
        # The JSON history is not the commit point and may be ahead/behind if a
        # crash occurred around the atomic latest write.  Rebuild it solely
        # from the strictly validated committed state before continuing.
        _write_json(
            history_path,
            _history_payload(
                identity=identity,
                training_history=training_history,
                validation_history=validation_history,
                candidate_artifacts=candidate_artifacts,
            ),
        )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        full_train.set_epoch(epoch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            stable_uint63(args.run_seed, args.dataset, "shuffle", epoch)
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=generator,
            drop_last=False,
        )
        learning_rate = legacy_train.learning_rate_for_epoch(
            epoch,
            args.epochs,
            args.base_lr,
            args.min_lr,
            args.warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        model.mode = "train"
        loss_sum = 0.0
        processed = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = legacy_train.deep_supervision_loss(
                model(images), masks, criterion
            )
            loss.backward()
            optimizer.step()
            count = int(images.shape[0])
            processed += count
            loss_sum += float(loss.detach().item()) * count
        if processed != len(train_data):
            raise RuntimeError("processed training sample count differs")
        mean_loss = loss_sum / processed
        training_record = {
            "epoch": epoch,
            "mean_train_loss": mean_loss,
            "learning_rate": learning_rate,
            "processed_samples": processed,
        }
        training_history.append(training_record)

        state = legacy_train._cpu_state(model)
        if epoch % args.val_interval == 0:
            metrics = evaluate_model(model, val_loader, device)
            validation_record = build_validation_record(epoch, metrics)
            validation_history.append(validation_record)
            candidate_path = _candidate_path(candidate_dir, epoch)
            _save_candidate(
                path=candidate_path,
                epoch=epoch,
                state=state,
                identity=identity,
                validation_record=validation_record,
            )
            candidate_artifacts = _candidate_artifacts(
                validation_history, candidate_dir
            )
        _write_json(
            history_path,
            _history_payload(
                identity=identity,
                training_history=training_history,
                validation_history=validation_history,
                candidate_artifacts=candidate_artifacts,
            ),
        )
        _atomic_torch_save(
            latest_path,
            {
                "schema": TRAINING_SCHEMA,
                "model": "EviSIRST",
                "dataset": args.dataset,
                "epoch": epoch,
                "run_identity": identity,
                "state_dict": state,
                "optimizer": optimizer.state_dict(),
                "training_history": training_history,
                "validation_history": validation_history,
                "candidate_artifacts": candidate_artifacts,
                "rng": legacy_train._capture_rng_state(device),
                "test_split_accessed": False,
            },
        )
        if validation_history and validation_history[-1]["epoch"] == epoch:
            # The atomic latest write above is the commit point.  Only now may
            # fully validated dominated/orphan candidates be removed.
            candidate_artifacts = _validate_resume_candidates(
                run_dir=run_dir,
                candidate_dir=candidate_dir,
                identity=identity,
                history=validation_history,
                artifacts=candidate_artifacts,
                expected_state=model.state_dict(),
                completed_epoch=epoch,
            )
        validation_text = ""
        if validation_history and validation_history[-1]["epoch"] == epoch:
            record = validation_history[-1]
            validation_text = (
                f" val_mIoU={record['mIoU']:.6f} val_Fa={record['Fa']:.8f}"
                f" val_Pd={record['Pd']:.6f}"
            )
        print(
            f"epoch={epoch}/{args.epochs} loss={mean_loss:.6f} "
            f"lr={learning_rate:.8f}{validation_text}",
            flush=True,
        )

    if not validation_history:
        raise ValidationSelectedTrainingError("no validation epoch was evaluated")
    selection_payload = build_selection_payload(
        validation_history, candidate_artifacts
    )
    selected_state = _load_selected_candidate(
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        selection_payload=selection_payload,
        identity=identity,
        history=validation_history,
        expected_state=model.state_dict(),
    )
    split_provenance = _split_provenance(full_train.contract)
    final_payload = build_final_checkpoint_payload(
        args=args,
        state_dict=selected_state,
        normalization=full_train.normalization,
        normalization_provenance=full_train.normalization_spec.as_dict(),
        identity=identity,
        split_provenance=split_provenance,
        selection_payload=selection_payload,
        model_metadata=model_metadata,
        smoke=smoke,
    )
    _atomic_torch_save(final_path, final_payload)
    _write_json(
        summary_path,
        {
            "schema": TRAINING_SCHEMA + "/summary",
            "status": "complete",
            "dataset": args.dataset,
            "checkpoint": str(final_path.relative_to(PROJECT_ROOT)),
            "checkpoint_role": "validation_selected",
            "selected_epoch": selection_payload["selected_epoch"],
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": args.run_seed,
            "target_mode": args.target_mode,
            "split_provenance": split_provenance,
            "selection": selection_payload,
            "training_history": training_history,
            "validation_history": validation_history,
            "candidate_artifacts": {
                str(epoch): artifact
                for epoch, artifact in sorted(candidate_artifacts.items())
            },
            "normalization": dict(full_train.normalization),
            "source_selection": selection_payload["source_selection"],
            "selection_is_optimistic": False,
            "optimistic": False,
            "test_split_accessed": False,
            "smoke": smoke,
            "elapsed_seconds": time.time() - started,
        },
    )
    return final_path


def main(argv: list[str] | None = None) -> None:
    checkpoint = run(parse_args(argv))
    print(checkpoint)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "DEFAULT_OUTPUT_ROOT",
    "HISTORY_SCHEMA",
    "MAX_RUN_SEED",
    "PROJECT_ROOT",
    "R1_RECIPE",
    "SELECTION_PAYLOAD_SCHEMA",
    "TRAINING_SCHEMA",
    "ValidationSelectedTrainingError",
    "build_datasets",
    "build_final_checkpoint_payload",
    "build_selection_payload",
    "build_validation_record",
    "enforce_grouping_policy",
    "evaluate_model",
    "final_prediction",
    "frontier_file_plan",
    "parse_args",
    "resolve_run_paths",
    "require_run_seed",
    "run",
    "stable_uint63",
    "validate_resume_identity",
]
