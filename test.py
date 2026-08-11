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
from load_models import DATASETS as EVALUATION_DATASETS
from load_models import load_baseline, load_evisirst
from model.EviSIRST import initialize_evisirst


PROBABILITY_THRESHOLD = 0.5
MATCH_RADIUS = 3.0
TINY_AREA = 9


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
    return state, dict(payload), checkpoint_sha256


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
        "dataset_root": str(args.dataset_root.resolve()),
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
        "checkpoint": {
            "path": metadata.get("checkpoint_path"),
            "epoch": metadata.get("epoch"),
            "role": metadata.get("checkpoint_role"),
            "sha256": metadata.get("checkpoint_sha256"),
        },
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
