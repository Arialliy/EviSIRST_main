#!/usr/bin/env python3
"""Read-only semantic and numerical diagnostics for C3-SBSC V3.1.

The formal diagnostic adapters copy only the 510 shared SCTransNet tensors
from one of two pre-registered validation-selected source checkpoints.  They
are deliberately marked ``diagnostic_only`` and cannot pass the formal V3.1
runner validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage import measure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.four_dataset_models_seed42_v1 import state_dict_sha256
from experiments import evisirst_v2_data as v2_data
from experiments import sctransnet_sbsc_v2 as v2_core
from experiments import sctransnet_sbsc_v21 as v21_core
from experiments import sctransnet_sbsc_v31 as v31_core


DIAGNOSTIC_SCHEMA = "sctransnet_sbsc_v31_diagnostic/v1"
SINGLE_SOURCE_SCHEMA = DIAGNOSTIC_SCHEMA + "/single_source/v1"
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
TOKEN_DOWNSAMPLE = 16
MATCH_RADIUS = 3.0
PROBABILITY_THRESHOLD = 0.5
NONTRIVIAL_L1 = 1e-4
HIGH_CONFIDENCE = 0.5
EXPECTED_VALIDATION_IMAGES = 160
ZERO_GAIN = (0.0, 0.0, 0.0, 0.0)
STRESS_GAIN = (0.25, 0.25, 0.25, 0.25)
REGION_NAMES = ("target", "ring", "false_object", "far")
SUPPORT_NAMES = ("consistent", "contradictory", "common")
SUPPORT_CODES = {
    "consistent": "C",
    "contradictory": "H",
    "common": "B",
}
DISTANCE_BINS = ("contact", "near", "mid", "far")
AREA_BINS = ("1-4", "5-16", "17-64", "65-inf")
DIAGNOSTIC_INTERVENTIONS = (
    "full",
    "consistent_common_only",
    "consistent_contradictory_only",
    "uniform_consistent",
    "uniform_contradictory",
    "uniform_common",
    "swap_consistent_contradictory",
    "cross_image_support",
    "spatial_shuffle_support",
    "disable_hard_constraint",
    "disable_background_constraint",
)


class C3V31DiagnosticError(RuntimeError):
    """The frozen read-only diagnostic contract cannot be evaluated."""


@dataclass(frozen=True)
class SourceSpec:
    method: str
    epoch: int
    checkpoint_relative_path: str
    checkpoint_file_sha256: str
    checkpoint_schema: str


SOURCE_SPECS = {
    "sctransnet": SourceSpec(
        method="sctransnet",
        epoch=670,
        checkpoint_relative_path=(
            "runs/sctransnet_sbsc_v2_validation/formal/sctransnet/"
            "IRSTD-1K/binary/run_seed_42/best_mIoU.pth.tar"
        ),
        checkpoint_file_sha256=(
            "840d596bbd6f309fffa967be05903315e663d88d8fae6f393a067a17241ca8cb"
        ),
        checkpoint_schema="sctransnet_sbsc_v2_validation_checkpoint/v1",
    ),
    "sbsc_v21": SourceSpec(
        method="sbsc_v21",
        epoch=543,
        checkpoint_relative_path=(
            "runs/sctransnet_sbsc_v21_validation/formal/sbsc_v21/"
            "IRSTD-1K/binary/run_seed_42/best_mIoU.pth.tar"
        ),
        checkpoint_file_sha256=(
            "25fda97b94c13f3ddf5a6fe358712daf3929bb2ff2b0ff260d4afb776def55d1"
        ),
        checkpoint_schema="sctransnet_sbsc_v21_validation_checkpoint/v1",
    ),
}


@dataclass(frozen=True)
class C3V31DiagnosticAdapter:
    source_model: torch.nn.Module
    anchor_model: torch.nn.Module
    diagnostic_model: torch.nn.Module
    evidence: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state or not all(
        type(key) is str and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError("checkpoint state_dict must map string keys to tensors")
    if all(key.startswith("module.") for key in state):
        return {key[7:]: value for key, value in state.items()}
    if any(key.startswith("module.") for key in state):
        raise ValueError("mixed DataParallel prefixes are forbidden")
    return dict(state)


def _load_source_checkpoint(
    spec: SourceSpec,
    checkpoint_path: Path | None,
) -> tuple[Path, dict[str, Any], dict[str, torch.Tensor]]:
    path = (
        PROJECT_ROOT / spec.checkpoint_relative_path
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"source checkpoint is not a regular file: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != spec.checkpoint_file_sha256:
        raise RuntimeError("source checkpoint SHA256 differs from the frozen spec")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("source checkpoint payload must be a dict")
    expected_identity = {
        "schema": spec.checkpoint_schema,
        "method": spec.method,
        "dataset": "IRSTD-1K",
        "epoch": spec.epoch,
        "architecture_seed": 42,
        "run_seed": 42,
        "test_split_accessed": False,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"source checkpoint identity differs at {key}")
    state = _canonical_state(payload.get("state_dict", {}))
    if spec.method == "sctransnet":
        v2_core.validate_sbsc_v2_state_dict(state, "sctransnet")
    else:
        v21_core.validate_sbsc_v21_state_dict(state, "sbsc_v21")
    try:
        v31_core.validate_sbsc_v31_state_dict(state, "sbsc_v31")
    except (TypeError, ValueError):
        pass
    else:
        raise RuntimeError(
            "V3.1 candidate validator unexpectedly accepted a source state"
        )
    return path.resolve(strict=True), payload, state


def build_c3_v31_diagnostic_adapter(
    source_method: str,
    *,
    checkpoint_path: Path | None = None,
) -> C3V31DiagnosticAdapter:
    """Build source and V3.1 diagnostic models without checkpoint migration."""

    if type(source_method) is not str or source_method not in SOURCE_SPECS:
        raise ValueError(f"source_method must be one of {tuple(SOURCE_SPECS)}")
    spec = SOURCE_SPECS[source_method]
    path, _payload, source_state = _load_source_checkpoint(
        spec, checkpoint_path
    )

    if source_method == "sctransnet":
        source_model, _source_metadata = v2_core.build_sctransnet_sbsc_v2_method(
            "sctransnet", "IRSTD-1K", training=False
        )
    else:
        source_model, _source_metadata = v21_core.build_sctransnet_sbsc_v21_method(
            "sbsc_v21", "IRSTD-1K", training=False
        )
    source_model.load_state_dict(source_state, strict=True)
    source_model.eval()
    source_model.mode = "test"
    if source_method == "sbsc_v21":
        v21_core.validate_sctransnet_sbsc_v21(
            source_model, require_zero_gain=False
        )

    shared_state = dict(source_state)
    removed_gain = None
    if source_method == "sbsc_v21":
        removed_gain = shared_state.pop(v21_core.SBSC_V21_GAIN_STATE_KEY)
        if tuple(removed_gain.shape) != ():
            raise RuntimeError("V2.1 diagnostic source gain schema differs")
    if len(shared_state) != 510:
        raise RuntimeError("diagnostic adapter requires exactly 510 shared tensors")

    anchor_model, diagnostic_model, _diagnostic_metadata = (
        v31_core.build_paired_sctransnet_sbsc_v31("IRSTD-1K")
    )
    anchor_model.load_state_dict(shared_state, strict=True)
    anchor_model.eval()
    anchor_model.mode = "train"
    diagnostic_state = diagnostic_model.state_dict()
    shared_keys = set(diagnostic_state) - {v31_core.SBSC_V31_GAIN_STATE_KEY}
    if set(shared_state) != shared_keys:
        raise RuntimeError("source shared state differs from V3.1 authority")
    for key in sorted(shared_keys):
        diagnostic_state[key] = shared_state[key].detach().clone()
    diagnostic_state[v31_core.SBSC_V31_GAIN_STATE_KEY] = torch.zeros(
        4, dtype=torch.float32
    )
    diagnostic_model.load_state_dict(diagnostic_state, strict=True)
    v31_core.validate_sctransnet_sbsc_v31(
        diagnostic_model, require_zero_gain=True
    )
    diagnostic_model.eval()
    diagnostic_model.mode = "train"
    diagnostic_model.diagnostic_only = True
    diagnostic_model.diagnostic_source_method = source_method
    diagnostic_model.diagnostic_source_epoch = spec.epoch
    diagnostic_model.diagnostic_source_file_sha256 = spec.checkpoint_file_sha256

    source_shared_sha = state_dict_sha256(shared_state)
    anchor_shared_sha = state_dict_sha256(anchor_model.state_dict())
    diagnostic_shared_sha = state_dict_sha256(
        diagnostic_model.state_dict(), sorted(shared_keys)
    )
    if source_shared_sha != anchor_shared_sha or source_shared_sha != diagnostic_shared_sha:
        raise RuntimeError("diagnostic shared-state SHA256 differs")
    evidence = {
        "schema": DIAGNOSTIC_SCHEMA + "/adapter/v1",
        "diagnostic_only": True,
        "source_method": source_method,
        "source_epoch": spec.epoch,
        "checkpoint_relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_file_sha256": spec.checkpoint_file_sha256,
        "source_state_sha256": state_dict_sha256(source_state),
        "shared_state_sha256": source_shared_sha,
        "shared_state_key_count": len(shared_state),
        "anchor_state_sha256": anchor_shared_sha,
        "diagnostic_state_sha256": state_dict_sha256(
            diagnostic_model.state_dict()
        ),
        "removed_source_gain": source_method == "sbsc_v21",
        "v31_candidate_validator_rejected_source_state": True,
        "v31_gain_zero": True,
        "test_split_accessed": False,
    }
    return C3V31DiagnosticAdapter(
        source_model=source_model,
        anchor_model=anchor_model,
        diagnostic_model=diagnostic_model,
        evidence=evidence,
    )


def _matched_prediction_indices(
    target_regions: Sequence[Any],
    prediction_regions: Sequence[Any],
) -> set[int]:
    if not target_regions or not prediction_regions:
        return set()
    distances = np.empty(
        (len(target_regions), len(prediction_regions)), dtype=np.float64
    )
    for target_index, target_region in enumerate(target_regions):
        target_centroid = np.asarray(target_region.centroid)
        for prediction_index, prediction_region in enumerate(prediction_regions):
            distances[target_index, prediction_index] = np.linalg.norm(
                np.asarray(prediction_region.centroid) - target_centroid
            )
    reward = (min(len(target_regions), len(prediction_regions)) + 1) * max(
        1.0, MATCH_RADIUS
    )
    real_cost = np.where(distances < MATCH_RADIUS, distances - reward, reward)
    cost = np.concatenate(
        (real_cost, np.zeros((len(target_regions), len(target_regions)))), axis=1
    )
    rows, columns = linear_sum_assignment(cost)
    return {
        int(column)
        for row, column in zip(rows, columns)
        if column < len(prediction_regions)
        and distances[row, column] < MATCH_RADIUS
    }


def _distance_bin(distance: float) -> str:
    if distance < 0.0 or math.isnan(distance):
        raise ValueError("component distance must be non-negative")
    if distance < 3.0:
        return "contact"
    if distance < 8.0:
        return "near"
    if distance < 16.0:
        return "mid"
    return "far"


def _area_bin(area: int) -> str:
    if isinstance(area, bool) or not isinstance(area, int) or area < 1:
        raise ValueError("component area must be a positive integer")
    if area <= 4:
        return "1-4"
    if area <= 16:
        return "5-16"
    if area <= 64:
        return "17-64"
    return "65-inf"


def prediction_component_taxonomy(
    target: np.ndarray,
    source_probability: np.ndarray,
    *,
    valid_hw: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Return strict-JSON source-specific prediction-component evidence."""

    target_array = np.asarray(target)
    probability = np.asarray(source_probability)
    if target_array.ndim != 2 or probability.shape != target_array.shape:
        raise ValueError("target and source_probability must share one 2D shape")
    if valid_hw is None:
        valid_hw = (int(target_array.shape[0]), int(target_array.shape[1]))
    if (
        len(valid_hw) != 2
        or not 0 < int(valid_hw[0]) <= int(target_array.shape[0])
        or not 0 < int(valid_hw[1]) <= int(target_array.shape[1])
    ):
        raise ValueError("valid_hw is outside the padded image")
    valid = np.zeros_like(target_array, dtype=bool)
    valid[: int(valid_hw[0]), : int(valid_hw[1])] = True
    target_binary = (target_array > 0.5) & valid
    prediction = (probability > PROBABILITY_THRESHOLD) & valid
    targets = measure.regionprops(measure.label(target_binary, connectivity=2))
    predictions = measure.regionprops(measure.label(prediction, connectivity=2))
    matched = _matched_prediction_indices(targets, predictions)
    records: list[dict[str, Any]] = []
    for index, region in enumerate(predictions):
        centroid = np.asarray(region.centroid, dtype=np.float64)
        if targets:
            nearest = min(
                float(
                    np.linalg.norm(
                        centroid - np.asarray(target_region.centroid)
                    )
                )
                for target_region in targets
            )
            encoded_distance: float | str = nearest
        else:
            nearest = math.inf
            encoded_distance = "+inf"
        area = int(region.area)
        records.append(
            {
                "prediction_component_index": index,
                "centroid_row": float(centroid[0]),
                "centroid_column": float(centroid[1]),
                "area_pixels": area,
                "matched_by_frozen_hungarian": index in matched,
                "nearest_gt_centroid_distance_px": encoded_distance,
                "distance_bin": _distance_bin(nearest),
                "area_bin": _area_bin(area),
            }
        )
    return records


def build_pixel_regions(
    target: np.ndarray,
    source_probability: np.ndarray,
    *,
    valid_hw: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Build the four frozen disjoint regions on the padded pixel canvas."""

    target_array = np.asarray(target)
    probability = np.asarray(source_probability)
    if target_array.ndim != 2 or probability.shape != target_array.shape:
        raise ValueError("target and source_probability must share one 2D shape")
    if valid_hw is None:
        valid_hw = (int(target_array.shape[0]), int(target_array.shape[1]))
    if (
        len(valid_hw) != 2
        or not 0 < int(valid_hw[0]) <= int(target_array.shape[0])
        or not 0 < int(valid_hw[1]) <= int(target_array.shape[1])
    ):
        raise ValueError("valid_hw is outside the padded image")
    valid = np.zeros_like(target_array, dtype=bool)
    valid[: int(valid_hw[0]), : int(valid_hw[1])] = True
    target_binary = (target_array > 0.5) & valid
    prediction = (probability > 0.5) & valid
    target_regions = measure.regionprops(measure.label(target_binary, connectivity=2))
    prediction_labels = measure.label(prediction, connectivity=2)
    prediction_regions = measure.regionprops(prediction_labels)
    matched_predictions = _matched_prediction_indices(
        target_regions, prediction_regions
    )
    unmatched = np.zeros_like(prediction, dtype=bool)
    for index, region in enumerate(prediction_regions):
        if index not in matched_predictions:
            unmatched[prediction_labels == region.label] = True
    dilated = ndimage.binary_dilation(
        target_binary, structure=np.ones((7, 7), dtype=bool)
    )
    ring = dilated & (~target_binary) & valid
    unmatched &= ~target_binary & ~ring
    far = (~dilated) & (~prediction) & valid

    return {
        "target": target_binary,
        "ring": ring,
        "false_object": unmatched,
        "far": far,
    }


def build_token_regions(
    target: np.ndarray,
    source_probability: np.ndarray,
    *,
    token_size: tuple[int, int] | None = None,
    valid_hw: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Map target/ring/unmatched-FP/far masks to disjoint token regions."""

    target_array = np.asarray(target)
    probability = np.asarray(source_probability)
    if target_array.ndim != 2 or probability.shape != target_array.shape:
        raise ValueError("target and source_probability must share one 2D shape")
    if token_size is None:
        if (
            target_array.shape[0] % TOKEN_DOWNSAMPLE != 0
            or target_array.shape[1] % TOKEN_DOWNSAMPLE != 0
        ):
            raise ValueError("padded image geometry must be divisible by 16")
        token_size = (
            target_array.shape[0] // TOKEN_DOWNSAMPLE,
            target_array.shape[1] // TOKEN_DOWNSAMPLE,
        )
    pixel_regions = build_pixel_regions(
        target_array, probability, valid_hw=valid_hw
    )
    pooled: dict[str, torch.Tensor] = {}
    for name, mask in pixel_regions.items():
        tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        pooled[name] = F.adaptive_max_pool2d(tensor, token_size)[0, 0].bool()
    pooled["ring"] &= ~pooled["target"]
    pooled["false_object"] &= ~(pooled["target"] | pooled["ring"])
    pooled["far"] &= ~(
        pooled["target"] | pooled["ring"] | pooled["false_object"]
    )
    return pooled


def support_region_stat(
    support: torch.Tensor,
    region: torch.Tensor,
) -> dict[str, float] | None:
    values = support.detach().float().reshape(-1)
    mask = region.detach().bool().reshape(-1).to(values.device)
    if values.numel() != mask.numel():
        raise ValueError("support and token region sizes differ")
    count = int(mask.sum().item())
    if count == 0:
        return None
    mass = float(values[mask].sum().item())
    enrichment = mass / (count / float(values.numel()) + 1e-6)
    return {"mass": mass, "enrichment": enrichment, "token_count": count}


def paired_bootstrap_lower_bound(
    left: Sequence[float | None],
    right: Sequence[float | None],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int]:
    if type(seed) is not int or seed != BOOTSTRAP_SEED:
        raise ValueError("paired bootstrap freezes seed=42")
    if type(resamples) is not int or resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("paired bootstrap freezes 10,000 resamples")
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have equal length")
    differences = np.asarray(
        [
            float(a) - float(b)
            for a, b in zip(left, right)
            if a is not None and b is not None
        ],
        dtype=np.float64,
    )
    if differences.size == 0 or not np.isfinite(differences).all():
        raise ValueError("paired bootstrap requires finite defined pairs")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, differences.size, size=(resamples, differences.size)
    )
    means = differences[indices].mean(axis=1)
    return {
        "pair_count": int(differences.size),
        "mean_difference": float(differences.mean()),
        "lower_bound_95": float(np.percentile(means, 2.5)),
        "seed": seed,
        "resamples": resamples,
    }


def strict_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    """Serialize evidence with sorted keys and no NaN/Infinity extensions."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise C3V31DiagnosticError(
            "diagnostic evidence is not finite strict JSON"
        ) from error


def _strict_json_clone(value: Any) -> Any:
    return json.loads(strict_json_dumps(value, indent=None))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        strict_json_dumps(value, indent=None).encode("utf-8")
    ).hexdigest()


def _with_evidence_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    report = _strict_json_clone(dict(value))
    if "evidence_sha256" in report:
        raise C3V31DiagnosticError("evidence_sha256 must be computed, not supplied")
    report["evidence_sha256"] = _canonical_sha256(report)
    return _strict_json_clone(report)


def _verify_evidence_sha256(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise C3V31DiagnosticError("hashed evidence must be a mapping")
    observed = value.get("evidence_sha256")
    if type(observed) is not str or len(observed) != 64:
        raise C3V31DiagnosticError("evidence SHA256 is missing")
    payload = dict(value)
    del payload["evidence_sha256"]
    if _canonical_sha256(payload) != observed:
        raise C3V31DiagnosticError("evidence SHA256 verification failed")


def _reject_nonstandard_json_constant(value: str) -> None:
    raise C3V31DiagnosticError(f"non-standard JSON constant is forbidden: {value}")


def load_strict_json_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if report_path.is_symlink() or not report_path.is_file():
        raise C3V31DiagnosticError("report input must be a regular non-symlink file")
    try:
        value = json.loads(
            report_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise C3V31DiagnosticError("report input is not readable strict JSON") from error
    if not isinstance(value, dict):
        raise C3V31DiagnosticError("report input must contain one JSON object")
    return _strict_json_clone(value)


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically replace one explicit JSON file in its existing directory."""

    requested = Path(path)
    if not requested.name or requested.name in {".", ".."}:
        raise C3V31DiagnosticError("output path must name one JSON file")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise C3V31DiagnosticError("output parent directory does not exist") from error
    target = parent / requested.name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise C3V31DiagnosticError("output must be a regular non-symlink file")
    payload = strict_json_dumps(value) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _require_source_methods(source_methods: Sequence[str]) -> tuple[str, ...]:
    if isinstance(source_methods, (str, bytes)) or not isinstance(
        source_methods, Sequence
    ):
        raise TypeError("source_methods must be a sequence")
    methods = tuple(source_methods)
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("source_methods must be non-empty and unique")
    if any(type(method) is not str or method not in SOURCE_SPECS for method in methods):
        raise ValueError(f"source_methods must be drawn from {tuple(SOURCE_SPECS)}")
    return methods


def build_validation_dataset(
    dataset_root: str | Path,
    *,
    split_root: str | Path = v2_data.DEFAULT_SPLIT_ROOT,
) -> v2_data.EviSIRSTV2ValDataset:
    """Open exactly the frozen IRSTD-1K V2 validation membership."""

    dataset = v2_data.EviSIRSTV2ValDataset(
        "IRSTD-1K",
        dataset_root=dataset_root,
        split_root=split_root,
        target_mode="binary",
        normalization_mode="legacy",
        return_metadata=True,
        # The diagnostic is validation-only: never hash/open training images.
        # Every validation file is hashed independently below.
        verify_data_tree=False,
    )
    if len(dataset) != EXPECTED_VALIDATION_IMAGES:
        raise C3V31DiagnosticError(
            "frozen IRSTD-1K validation must contain exactly 160 images"
        )
    if (
        dataset.metadata.get("split") != "val"
        or dataset.metadata.get("test_index_opened") is not False
        or tuple(dataset.sample_ids) != tuple(dataset.contract.val_ids)
    ):
        raise C3V31DiagnosticError("validation-only dataset identity differs")
    if dataset.contract.data_tree_verified or dataset.contract.data_tree_sha256 is not None:
        raise C3V31DiagnosticError(
            "validation-only construction unexpectedly opened the full data tree"
        )
    return dataset


def validation_data_tree_sha256(
    dataset: v2_data.EviSIRSTV2ValDataset,
) -> str:
    """Hash only the ordered validation image/mask files, never train/test."""

    digest = hashlib.sha256()
    for sample_id in dataset.sample_ids:
        image_path, mask_path = dataset._paths(sample_id)
        for value in (
            sample_id,
            image_path.relative_to(dataset.dataset_root).as_posix(),
            sha256_file(image_path),
            mask_path.relative_to(dataset.dataset_root).as_posix(),
            sha256_file(mask_path),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _output_tuple(output: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)) and output and all(
        isinstance(value, torch.Tensor) for value in output
    ):
        return tuple(output)
    raise C3V31DiagnosticError("model returned an unsupported output object")


def _final_probability(output: Any) -> torch.Tensor:
    return _output_tuple(output)[-1]


@torch.inference_mode()
def _capture_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    gain_override: Sequence[float],
    amp_bfloat16: bool,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    if model.training or model.mode != "train":
        raise C3V31DiagnosticError(
            "diagnostic model must be eval() with six-head mode='train'"
        )
    device_type = image.device.type
    autocast = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if amp_bfloat16
        else nullcontext()
    )
    with autocast, v31_core.capture_c3_v31_diagnostics(
        model,
        gain_override=gain_override,
        intervention="full",
    ) as collector:
        outputs = _output_tuple(model(image))
    if len(outputs) != 6 or len(collector.records) != 1:
        raise C3V31DiagnosticError(
            "one diagnostic forward must emit six heads and one capture record"
        )
    return outputs, collector.records[0]


def validate_token_geometry(
    capture: Mapping[str, Any],
    padded_hw: tuple[int, int],
) -> tuple[int, int]:
    """Bidirectionally bind captured support N to the padded input geometry."""

    if (
        len(padded_hw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in padded_hw)
        or any(value <= 0 or value % TOKEN_DOWNSAMPLE != 0 for value in padded_hw)
    ):
        raise C3V31DiagnosticError(
            "padded validation geometry must be positive and divisible by 16"
        )
    token_hw = (
        padded_hw[0] // TOKEN_DOWNSAMPLE,
        padded_hw[1] // TOKEN_DOWNSAMPLE,
    )
    expected_positions = token_hw[0] * token_hw[1]
    levels = capture.get("levels")
    if not isinstance(levels, tuple) or len(levels) != 4:
        raise C3V31DiagnosticError("capture must contain exactly four levels")
    observed: set[int] = set()
    observed_batches: set[int] = set()
    for level_index, level in enumerate(levels):
        if not isinstance(level, Mapping) or level.get("level_index") != level_index:
            raise C3V31DiagnosticError("capture level order differs")
        support = level.get("support")
        if not isinstance(support, v31_core.C3V31LevelSupport):
            raise C3V31DiagnosticError("capture support record differs")
        for name in (
            "consistent_support",
            "contradictory_support",
            "common_support",
        ):
            tensor = getattr(support, name)
            if tensor.ndim != 4 or int(tensor.shape[0]) <= 0:
                raise C3V31DiagnosticError("captured support geometry differs")
            observed_batches.add(int(tensor.shape[0]))
            observed.add(int(tensor.shape[-1]))
    if observed != {expected_positions} or len(observed_batches) != 1:
        raise C3V31DiagnosticError(
            "captured support N disagrees with padded input token geometry"
        )
    return token_hw


def _reject_base_contract_failure(
    capture: Mapping[str, Any], *, execution: str
) -> None:
    levels = capture.get("levels")
    if not isinstance(levels, tuple) or len(levels) != 4:
        raise C3V31DiagnosticError(f"{execution} capture level schema differs")
    failed = [
        index if not isinstance(level, Mapping) else int(level.get("level_index", index))
        for index, level in enumerate(levels)
        if not isinstance(level, Mapping)
        or level.get("base_contract_failure") is not False
    ]
    if failed:
        raise C3V31DiagnosticError(
            f"{execution} base attention contract failed at levels {failed}"
        )


def _masked_maximum(
    values: torch.Tensor, mask: torch.Tensor
) -> float | None:
    expanded = torch.broadcast_to(mask, values.shape)
    selected = values.detach().float()[expanded]
    return None if selected.numel() == 0 else float(selected.max().item())


def _masked_minimum(
    values: torch.Tensor, mask: torch.Tensor
) -> float | None:
    expanded = torch.broadcast_to(mask, values.shape)
    selected = values.detach().float()[expanded]
    return None if selected.numel() == 0 else float(selected.min().item())


def _support_cell_records(
    capture: Mapping[str, Any],
    regions: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    if set(regions) != set(REGION_NAMES):
        raise C3V31DiagnosticError("token-region schema differs")
    records: list[dict[str, Any]] = []
    for level in capture["levels"]:
        support = level["support"]
        projection = level["projection"]
        if not isinstance(projection, v31_core.C3V31Projection):
            raise C3V31DiagnosticError("level projection record is missing")
        row_mass = level["A0_raw"].detach().float().sum(
            dim=-1, keepdim=True
        )
        qout = level["Aout_fp32"].detach().float() / row_mass
        nontrivial = (
            qout - projection.q0.detach().float()
        ).abs().sum(dim=-1).ge(NONTRIVIAL_L1)
        heads = int(support.consistent_support.shape[1])
        for head in range(heads):
            valid = {
                SUPPORT_CODES[name]: bool(
                    getattr(support, f"{name}_valid")[0, head, 0, 0].item()
                )
                for name in SUPPORT_NAMES
            }
            region_records: dict[str, dict[str, Any]] = {}
            for region_name in REGION_NAMES:
                per_support: dict[str, Any] = {}
                for support_name in SUPPORT_NAMES:
                    code = SUPPORT_CODES[support_name]
                    per_support[code] = (
                        support_region_stat(
                            getattr(support, f"{support_name}_support")[
                                0, head, 0
                            ],
                            regions[region_name],
                        )
                        if valid[code]
                        else None
                    )
                region_records[region_name] = per_support

            reverse_reasons: list[str] = []

            def mass(region_name: str, code: str) -> float | None:
                value = region_records[region_name][code]
                return None if value is None else float(value["mass"])

            target_c, target_h, target_b = (
                mass("target", code) for code in ("C", "H", "B")
            )
            false_h, false_c = mass("false_object", "H"), mass(
                "false_object", "C"
            )
            far_b, far_c = mass("far", "B"), mass("far", "C")
            if target_c is not None and target_h is not None and target_c <= target_h:
                reverse_reasons.append("target_C_not_above_H")
            if target_c is not None and target_b is not None and target_c <= target_b:
                reverse_reasons.append("target_C_not_above_B")
            if false_h is not None and false_c is not None and false_h <= false_c:
                reverse_reasons.append("false_object_H_not_above_C")
            if far_b is not None and far_c is not None and far_b <= far_c:
                reverse_reasons.append("far_B_not_above_C")

            reliability = float(support.reliability[0, head, 0, 0].item())
            nontrivial_rows = int(nontrivial[0, head].sum().item())
            high_confidence_wrong = bool(
                reliability >= HIGH_CONFIDENCE
                and nontrivial_rows > 0
                and reverse_reasons
            )
            records.append(
                {
                    "level_index": int(level["level_index"]),
                    "head_index": head,
                    "valid": valid,
                    "reliability": reliability,
                    "regions": region_records,
                    "nontrivial_row_count": nontrivial_rows,
                    "reverse_enrichment_reasons": reverse_reasons,
                    "high_confidence_wrong_correction": high_confidence_wrong,
                }
            )
    return records


def _level_numerical_record(
    level: Mapping[str, Any], *, has_target: bool, execution: str
) -> dict[str, Any]:
    if execution not in {"fp32", "bf16"}:
        raise ValueError("execution must be 'fp32' or 'bf16'")
    projection = level.get("projection")
    if not isinstance(projection, v31_core.C3V31Projection):
        raise C3V31DiagnosticError("level projection is unavailable")
    support = level.get("support")
    if not isinstance(support, v31_core.C3V31LevelSupport):
        raise C3V31DiagnosticError("level support is unavailable")
    q0 = projection.q0.detach().float()
    qhat = projection.qhat.detach().float()
    row_mass = level["A0_raw"].detach().float().sum(dim=-1, keepdim=True)
    qout = level["Aout_fp32"].detach().float() / row_mass
    q_emitted = level["Aout_emitted"].detach().float() / row_mass
    accepted = projection.accepted.detach().bool()
    fallback = projection.solver_fallback.detach().bool()
    eligible = accepted | fallback
    emission_fallback = level["emission_fallback"].detach().bool()
    terminal_count = (
        projection.support_identity.to(torch.int64)
        + projection.risk_identity.to(torch.int64)
        + projection.solver_fallback.to(torch.int64)
        + projection.accepted.to(torch.int64)
    )
    nontrivial = (qout - q0).abs().sum(dim=-1, keepdim=True).ge(
        NONTRIVIAL_L1
    )
    nontrivial_emitted = (q_emitted - q0).abs().sum(
        dim=-1, keepdim=True
    ).ge(NONTRIVIAL_L1)
    solver_input_finite = torch.ones_like(accepted)
    for value in (
        q0,
        level["consistent_benefit"],
        level["hard_risk"],
        level["background_risk"],
        support.reliability,
    ):
        per_row = torch.isfinite(value.detach().float()).all(
            dim=-1, keepdim=True
        )
        solver_input_finite &= torch.broadcast_to(per_row, accepted.shape)
    projection_tensors = tuple(
        value
        for name in projection.__dataclass_fields__
        for value in (getattr(projection, name),)
        if isinstance(value, torch.Tensor)
        and (value.is_floating_point() or value.is_complex())
    )
    support_tensors = tuple(
        value
        for name in support.__dataclass_fields__
        for value in (getattr(support, name),)
        if isinstance(value, torch.Tensor)
        and (value.is_floating_point() or value.is_complex())
    )
    level_tensors = tuple(
        value
        for value in level.values()
        if isinstance(value, torch.Tensor)
        and (value.is_floating_point() or value.is_complex())
    )
    finite = all(
        bool(torch.isfinite(value.detach().float()).all())
        for value in (*projection_tensors, *support_tensors, *level_tensors)
    )
    status_counts = {
        name: int(getattr(projection, name).sum().item())
        for name in (
            "support_identity",
            "risk_identity",
            "solver_fallback",
            "accepted",
            "hard_inactive",
            "hard_defined",
            "background_defined",
            "hard_active",
            "background_active",
            "projected_newton_used",
            "projected_gradient_used",
        )
    }
    active_set_counts = {
        str(code): int(projection.active_set_code.eq(code).sum().item())
        for code in (-1, 0, 1, 2, 3)
    }
    failure_counts = {
        name: int(getattr(projection, name).sum().item())
        for name in (
            "bracket_failed_h",
            "bracket_failed_b",
            "correlation_blocked",
            "line_search_failed",
            "live_recert_failed",
        )
    }
    candidate_fields = (
        "candidate_valid",
        "candidate_allowed",
        "candidate_finite",
        "candidate_simplex",
        "candidate_support_preserved",
        "candidate_lambda_ok",
        "candidate_kkt_ok",
        "candidate_stationarity_ok",
        "candidate_risk_ok",
        "candidate_objective_ok",
    )
    candidate_counts = {
        name: [
            int(value)
            for value in getattr(projection, name)
            .reshape(-1, 4)
            .sum(dim=0)
            .cpu()
            .tolist()
        ]
        for name in candidate_fields
    }
    candidate_numeric_fields = (
        "candidate_objective",
        "candidate_lambda_h",
        "candidate_lambda_b",
        "candidate_kkt_max",
        "candidate_stationarity",
        "candidate_risk_max",
        "candidate_raw_risk_max",
    )
    candidate_numeric_extrema = {}
    for name in candidate_numeric_fields:
        rows_by_slot = getattr(projection, name).detach().float().reshape(-1, 4)
        candidate_numeric_extrema[name] = {
            "min": [float(value) for value in rows_by_slot.amin(dim=0).cpu()],
            "max": [float(value) for value in rows_by_slot.amax(dim=0).cpu()],
        }
    emission_success = accepted & (~emission_fallback)
    qout32_certificate_ok = level["qout32_certificate_ok"].detach().bool()
    emitted_certificate_ok = level["emitted_certificate_ok"].detach().bool()
    if (
        tuple(qout32_certificate_ok.shape) != tuple(accepted.shape)
        or tuple(emitted_certificate_ok.shape) != tuple(accepted.shape)
        or tuple(emission_fallback.shape) != tuple(accepted.shape)
    ):
        raise C3V31DiagnosticError("emission certificate geometry differs")
    certificate_accounting_consistent = torch.equal(
        emission_fallback,
        (~qout32_certificate_ok) | (~emitted_certificate_ok),
    )
    emission_tolerance = float(level["emission_tolerance"])
    if not math.isfinite(emission_tolerance) or emission_tolerance <= 0.0:
        raise C3V31DiagnosticError("emission tolerance is invalid")
    return {
        "level_index": int(level["level_index"]),
        "execution": execution,
        "row_count": int(q0.numel() // q0.shape[-1]),
        "finite": finite,
        "nonfinite_input_count": int((~solver_input_finite).sum().item()),
        "terminal_one_hot": bool(terminal_count.eq(1).all()),
        "qhat_row_mass_error_max": float(
            qhat.sum(dim=-1).sub(1.0).abs().max().item()
        ),
        "qout_row_mass_error_max": float(
            qout.sum(dim=-1).sub(1.0).abs().max().item()
        ),
        "qhat_min": float(qhat.min().item()),
        "qout_min": float(qout.min().item()),
        "q_emitted_row_mass_error_max": float(
            q_emitted.sum(dim=-1).sub(1.0).abs().max().item()
        ),
        "q_emitted_min": float(q_emitted.min().item()),
        "accepted_hard_delta_max": _masked_maximum(
            projection.hard_risk_delta,
            accepted & projection.hard_defined,
        ),
        "accepted_background_delta_max": _masked_maximum(
            projection.background_risk_delta,
            accepted & projection.background_defined,
        ),
        "qout_hard_delta_max": _masked_maximum(
            level["hard_risk_delta_qout"],
            accepted & projection.hard_defined,
        ),
        "qout_background_delta_max": _masked_maximum(
            level["background_risk_delta_qout"],
            accepted & projection.background_defined,
        ),
        "accepted_hard_delta_raw_max": _masked_maximum(
            projection.hard_risk_delta_raw, accepted
        ),
        "accepted_background_delta_raw_max": _masked_maximum(
            projection.background_risk_delta_raw, accepted
        ),
        "qout_hard_delta_raw_max": _masked_maximum(
            level["hard_risk_delta_qout_raw"], accepted
        ),
        "qout_background_delta_raw_max": _masked_maximum(
            level["background_risk_delta_qout_raw"], accepted
        ),
        "emitted_hard_delta_max": _masked_maximum(
            level["hard_risk_delta_emitted"],
            emission_success & projection.hard_defined,
        ),
        "emitted_background_delta_max": _masked_maximum(
            level["background_risk_delta_emitted"],
            emission_success & projection.background_defined,
        ),
        "emitted_hard_delta_raw_max": _masked_maximum(
            level["hard_risk_delta_emitted_raw"], emission_success
        ),
        "emitted_background_delta_raw_max": _masked_maximum(
            level["background_risk_delta_emitted_raw"], emission_success
        ),
        "accepted_objective_min": _masked_minimum(
            projection.objective_certificate, accepted
        ),
        "qout_objective_min": _masked_minimum(
            level["objective_qout"], accepted
        ),
        "emitted_objective_min": _masked_minimum(
            level["objective_emitted"], emission_success
        ),
        "effective_kkt_max": _masked_maximum(
            projection.kkt_max, accepted
        ),
        "solver_eligible_rows": int(eligible.sum().item()),
        "target_solver_eligible_rows": int(eligible.sum().item()) if has_target else 0,
        "target_nontrivial_accepted_rows": (
            int((eligible & accepted & nontrivial).sum().item())
            if has_target
            else 0
        ),
        "emission_fallback_rows": int(
            (eligible & emission_fallback).sum().item()
        ),
        "all_emission_fallback_rows": int(emission_fallback.sum().item()),
        "target_nontrivial_emitted_rows": (
            int((eligible & (~emission_fallback) & nontrivial_emitted).sum().item())
            if has_target
            else 0
        ),
        "emission_tolerance": emission_tolerance,
        "qout32_certificate_failure_rows": int(
            (eligible & (~qout32_certificate_ok)).sum().item()
        ),
        "emitted_certificate_failure_rows": int(
            (eligible & (~emitted_certificate_ok)).sum().item()
        ),
        "emission_certificate_accounting_consistent": (
            certificate_accounting_consistent
        ),
        "emission_reason_counts": {
            str(code): int(level["emission_reason_code"].eq(code).sum().item())
            for code in (0, 1, 2)
        },
        "solver_reason_counts": {
            str(code): int(projection.reason_code.eq(code).sum().item())
            for code in range(8)
        },
        "status_counts": status_counts,
        "active_set_counts": active_set_counts,
        "failure_counts": failure_counts,
        "candidate_counts_by_slot_empty_h_b_hb": candidate_counts,
        "candidate_numeric_extrema_by_slot_empty_h_b_hb": (
            candidate_numeric_extrema
        ),
        "iteration_max": {
            "singleton_h": int(projection.singleton_iterations_h.max().item()),
            "singleton_b": int(projection.singleton_iterations_b.max().item()),
            "dual": int(projection.dual_iterations.max().item()),
        },
    }


def _amp_level_comparison(
    fp32_level: Mapping[str, Any],
    bf16_level: Mapping[str, Any],
) -> dict[str, Any]:
    fp_projection = fp32_level["projection"]
    bf_projection = bf16_level["projection"]
    if not isinstance(fp_projection, v31_core.C3V31Projection) or not isinstance(
        bf_projection, v31_core.C3V31Projection
    ):
        raise C3V31DiagnosticError("AMP projection record differs")

    def emitted_probability(level: Mapping[str, Any]) -> torch.Tensor:
        mass = level["A0_raw"].detach().float().sum(dim=-1, keepdim=True)
        return level["Aout_emitted"].detach().float() / mass

    fp_direction = emitted_probability(fp32_level) - fp_projection.q0.detach().float()
    bf_direction = emitted_probability(bf16_level) - bf_projection.q0.detach().float()
    if tuple(fp_direction.shape) != tuple(bf_direction.shape):
        raise C3V31DiagnosticError("FP32/BF16 projection geometry differs")
    fp_rows = fp_direction.reshape(-1, fp_direction.shape[-1])
    bf_rows = bf_direction.reshape(-1, bf_direction.shape[-1])
    fp_emission_fallback = fp32_level["emission_fallback"].detach().bool()
    bf_emission_fallback = bf16_level["emission_fallback"].detach().bool()
    fp_eligible = fp_projection.accepted | fp_projection.solver_fallback
    bf_eligible = bf_projection.accepted | bf_projection.solver_fallback
    comparison_eligible = fp_eligible | bf_eligible
    both_nontrivial = (
        fp_direction.abs().sum(dim=-1, keepdim=True).ge(NONTRIVIAL_L1)
        & bf_direction.abs().sum(dim=-1, keepdim=True).ge(NONTRIVIAL_L1)
        & (~fp_emission_fallback)
        & (~bf_emission_fallback)
        & comparison_eligible
    ).reshape(-1)
    if bool(both_nontrivial.any()):
        cosine = F.cosine_similarity(
            fp_rows[both_nontrivial], bf_rows[both_nontrivial], dim=-1
        )
        cosine_values = [float(value) for value in cosine.cpu().tolist()]
    else:
        cosine_values = []
    mismatch = (
        fp_projection.accepted.ne(bf_projection.accepted)
        | fp_projection.solver_fallback.ne(bf_projection.solver_fallback)
    ) & comparison_eligible
    emission_mismatch = fp_emission_fallback.ne(
        bf_emission_fallback
    ) & comparison_eligible
    bf_support = bf16_level["support"]
    bf_internal_tensors = [
        value
        for name, value in bf16_level.items()
        if name not in {"A0_raw", "Aout_emitted"}
        and isinstance(value, torch.Tensor)
        and value.is_floating_point()
    ]
    bf_internal_tensors.extend(
        value
        for name in bf_projection.__dataclass_fields__
        for value in (getattr(bf_projection, name),)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    bf_internal_tensors.extend(
        value
        for name in bf_support.__dataclass_fields__
        for value in (getattr(bf_support, name),)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    return {
        "level_index": int(fp32_level["level_index"]),
        "eligible_rows": int(comparison_eligible.sum().item()),
        "status_mismatch_rows": int(mismatch.sum().item()),
        "emission_status_mismatch_rows": int(emission_mismatch.sum().item()),
        "nontrivial_both_rows": len(cosine_values),
        "correction_cosines": cosine_values,
        "internal_solver_and_relations_fp32": bool(bf_internal_tensors)
        and all(value.dtype is torch.float32 for value in bf_internal_tensors),
        "emitted_attention_restores_ambient_dtype": (
            bf16_level["Aout_emitted"].dtype
            == bf16_level["A0_raw"].dtype
        ),
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _median_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def _maximum_or_none(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return None if not defined else max(defined)


def _minimum_or_none(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return None if not defined else min(defined)


def _paired_direction(
    images: Sequence[Mapping[str, Any]],
    *,
    left_support: str,
    right_support: str,
    region: str,
) -> dict[str, Any]:
    """Bootstrap comparison-specific common-valid image-level differences."""

    image_units: list[dict[str, Any]] = []
    image_differences: list[float] = []
    for image in images:
        paired_unit_differences: list[float] = []
        for cell in image["support_cells"]:
            left_valid = bool(cell["valid"][left_support])
            right_valid = bool(cell["valid"][right_support])
            left = cell["regions"][region][left_support]
            right = cell["regions"][region][right_support]
            if left_valid and right_valid and left is not None and right is not None:
                paired_unit_differences.append(
                    float(left["mass"]) - float(right["mass"])
                )
        image_difference = _mean_or_none(paired_unit_differences)
        if image_difference is not None:
            image_differences.append(image_difference)
        image_units.append(
            {
                "sample_id": image["sample_id"],
                "paired_unit_count": len(paired_unit_differences),
                "image_mean_difference": image_difference,
                "defined": image_difference is not None,
            }
        )
    try:
        result = paired_bootstrap_lower_bound(
            image_differences, [0.0] * len(image_differences)
        )
    except ValueError:
        result = {
            "pair_count": 0,
            "mean_difference": None,
            "lower_bound_95": None,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
        }
    result.update(
        {
            "left_support": left_support,
            "right_support": right_support,
            "region": region,
            "eligible_image_count": len(image_differences),
            "undefined_image_count": len(images) - len(image_differences),
            "eligible_sample_ids_in_order": [
                unit["sample_id"] for unit in image_units if unit["defined"]
            ],
            "image_units": image_units,
            "passed": bool(
                result["lower_bound_95"] is not None
                and float(result["lower_bound_95"]) > 0.0
            ),
        }
    )
    return result


def summarize_semantic_evidence(
    images: Sequence[Mapping[str, Any]],
    *,
    source_method: str,
) -> dict[str, Any]:
    if source_method not in SOURCE_SPECS:
        raise ValueError("unknown semantic source")
    directions = {
        "target_C_minus_H": _paired_direction(
            images,
            left_support="C",
            right_support="H",
            region="target",
        ),
        "target_C_minus_B": _paired_direction(
            images,
            left_support="C",
            right_support="B",
            region="target",
        ),
        "false_object_H_minus_C": _paired_direction(
            images,
            left_support="H",
            right_support="C",
            region="false_object",
        ),
        "far_B_minus_C": _paired_direction(
            images,
            left_support="B",
            right_support="C",
            region="far",
        ),
    }
    unit_enrichment: dict[tuple[str, str], list[float]] = {
        (support, region): []
        for support in ("C", "H", "B")
        for region in REGION_NAMES
    }
    valid_numerators = Counter({"H": 0, "B": 0, "C_and_B_target": 0})
    valid_denominators = Counter({"H": 0, "B": 0, "C_and_B_target": 0})
    reverse_cells = 0
    high_confidence_wrong = 0
    empty_regions = Counter({region: 0 for region in REGION_NAMES})
    component_distance = Counter({name: 0 for name in DISTANCE_BINS})
    component_area = Counter({name: 0 for name in AREA_BINS})

    for image in images:
        for region in REGION_NAMES:
            empty_regions[region] += int(bool(image["empty_regions"][region]))
        for component in image["prediction_components"]:
            component_distance[component["distance_bin"]] += 1
            component_area[component["area_bin"]] += 1
        for cell in image["support_cells"]:
            valid_numerators["H"] += int(cell["valid"]["H"])
            valid_numerators["B"] += int(cell["valid"]["B"])
            valid_denominators["H"] += 1
            valid_denominators["B"] += 1
            if image["has_target"]:
                valid_numerators["C_and_B_target"] += int(
                    cell["valid"]["C"] and cell["valid"]["B"]
                )
                valid_denominators["C_and_B_target"] += 1
            reverse_cells += int(bool(cell["reverse_enrichment_reasons"]))
            high_confidence_wrong += int(
                cell["high_confidence_wrong_correction"]
            )
            for region in REGION_NAMES:
                for support in ("C", "H", "B"):
                    stat = cell["regions"][region][support]
                    if stat is not None:
                        unit_enrichment[(support, region)].append(
                            float(stat["enrichment"])
                        )

    def rate(name: str) -> float | None:
        denominator = valid_denominators[name]
        return (
            None
            if denominator == 0
            else float(valid_numerators[name] / denominator)
        )

    valid_rates = {
        "consistent_and_common_target_images": rate("C_and_B_target"),
        "background": rate("B"),
        "hard": rate("H"),
        "counts": {
            name: {
                "valid": int(valid_numerators[name]),
                "total": int(valid_denominators[name]),
            }
            for name in ("C_and_B_target", "B", "H")
        },
    }
    enrichment_medians = {
        f"{support}_{region}": _median_or_none(values)
        for (support, region), values in unit_enrichment.items()
    }
    common_gates = {
        "four_paired_directions": all(
            direction["passed"] for direction in directions.values()
        ),
        "background_valid_rate_ge_0_99": bool(
            valid_rates["background"] is not None
            and valid_rates["background"] >= 0.99
        ),
        "hard_valid_rate_ge_0_10": bool(
            valid_rates["hard"] is not None and valid_rates["hard"] >= 0.10
        ),
        "no_reverse_enrichment_cells": reverse_cells == 0,
        "no_high_confidence_wrong_correction": high_confidence_wrong == 0,
    }
    source_specific_gates: dict[str, bool] = {}
    if source_method == "sbsc_v21":
        source_specific_gates = {
            "consistent_target_enrichment_median_ge_2": bool(
                enrichment_medians["C_target"] is not None
                and enrichment_medians["C_target"] >= 2.0
            ),
            "contradictory_false_object_enrichment_median_ge_1_5": bool(
                enrichment_medians["H_false_object"] is not None
                and enrichment_medians["H_false_object"] >= 1.5
            ),
            "consistent_common_target_valid_rate_ge_0_70": bool(
                valid_rates["consistent_and_common_target_images"] is not None
                and valid_rates["consistent_and_common_target_images"] >= 0.70
            ),
        }
    gates = {**common_gates, **source_specific_gates}
    return {
        "schema": DIAGNOSTIC_SCHEMA + "/semantic_summary/v1",
        "source_method": source_method,
        "image_count": len(images),
        "directions": directions,
        "enrichment_medians": enrichment_medians,
        "valid_rates": valid_rates,
        "empty_region_image_counts": dict(empty_regions),
        "reverse_enrichment_cell_count": reverse_cells,
        "high_confidence_wrong_correction_count": high_confidence_wrong,
        "prediction_component_taxonomy": {
            "distance_bins": dict(component_distance),
            "area_bins": dict(component_area),
        },
        "gates": gates,
        "status": "GO" if all(gates.values()) else "NO-GO",
    }


def summarize_numerical_evidence(
    images: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    levels = [level for image in images for level in image["numerical_levels"]]
    bf16_levels = [
        level for image in images for level in image["bf16_numerical_levels"]
    ]
    amp_levels = [level for image in images for level in image["amp_levels"]]
    if (
        not levels
        or len(bf16_levels) != len(levels)
        or len(amp_levels) != len(levels)
    ):
        raise C3V31DiagnosticError("numerical evidence is empty")

    status_names = (
        "support_identity",
        "risk_identity",
        "solver_fallback",
        "accepted",
        "hard_inactive",
        "hard_defined",
        "background_defined",
        "hard_active",
        "background_active",
        "projected_newton_used",
        "projected_gradient_used",
    )
    failure_names = (
        "bracket_failed_h",
        "bracket_failed_b",
        "correlation_blocked",
        "line_search_failed",
        "live_recert_failed",
    )

    def execution_summary(
        execution_levels: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        eligible_rows = sum(
            int(level["solver_eligible_rows"]) for level in execution_levels
        )
        total_rows = sum(int(level["row_count"]) for level in execution_levels)
        fallback_rows = sum(
            int(level["status_counts"]["solver_fallback"])
            for level in execution_levels
        )
        support_identity_rows = sum(
            int(level["status_counts"]["support_identity"])
            for level in execution_levels
        )
        target_eligible_rows = sum(
            int(level["target_solver_eligible_rows"])
            for level in execution_levels
        )
        target_nontrivial_rows = sum(
            int(level["target_nontrivial_accepted_rows"])
            for level in execution_levels
        )
        emission_fallback_rows = sum(
            int(level["emission_fallback_rows"])
            for level in execution_levels
        )
        target_nontrivial_emitted_rows = sum(
            int(level["target_nontrivial_emitted_rows"])
            for level in execution_levels
        )
        candidate_names = tuple(
            execution_levels[0][
                "candidate_counts_by_slot_empty_h_b_hb"
            ]
        )
        candidate_numeric_names = tuple(
            execution_levels[0][
                "candidate_numeric_extrema_by_slot_empty_h_b_hb"
            ]
        )
        return {
            "row_count": total_rows,
            "solver_eligible_rows": eligible_rows,
            "solver_fallback_rows": fallback_rows,
            "solver_fallback_rate": (
                None
                if eligible_rows == 0
                else float(fallback_rows / eligible_rows)
            ),
            "support_identity_rows": support_identity_rows,
            "support_identity_rate": (
                None
                if total_rows == 0
                else float(support_identity_rows / total_rows)
            ),
            "target_solver_eligible_rows": target_eligible_rows,
            "target_nontrivial_accepted_rows": target_nontrivial_rows,
            "target_nontrivial_accepted_rate": (
                None
                if target_eligible_rows == 0
                else float(target_nontrivial_rows / target_eligible_rows)
            ),
            "emission_fallback_rows": emission_fallback_rows,
            "emission_fallback_rate": (
                None
                if eligible_rows == 0
                else float(emission_fallback_rows / eligible_rows)
            ),
            "target_nontrivial_emitted_rows": (
                target_nontrivial_emitted_rows
            ),
            "target_nontrivial_emitted_rate": (
                None
                if target_eligible_rows == 0
                else float(
                    target_nontrivial_emitted_rows / target_eligible_rows
                )
            ),
            "emission_tolerance_min": min(
                float(level["emission_tolerance"])
                for level in execution_levels
            ),
            "emission_tolerance_max": max(
                float(level["emission_tolerance"])
                for level in execution_levels
            ),
            "qout32_certificate_failure_rows": sum(
                int(level["qout32_certificate_failure_rows"])
                for level in execution_levels
            ),
            "emitted_certificate_failure_rows": sum(
                int(level["emitted_certificate_failure_rows"])
                for level in execution_levels
            ),
            "emission_certificate_accounting_consistent": all(
                bool(level["emission_certificate_accounting_consistent"])
                for level in execution_levels
            ),
            "nonfinite_input_count": sum(
                int(level["nonfinite_input_count"])
                for level in execution_levels
            ),
            "status_counts": {
                name: sum(
                    int(level["status_counts"][name])
                    for level in execution_levels
                )
                for name in status_names
            },
            "active_set_counts": {
                str(code): sum(
                    int(level["active_set_counts"][str(code)])
                    for level in execution_levels
                )
                for code in (-1, 0, 1, 2, 3)
            },
            "solver_reason_counts": {
                str(code): sum(
                    int(level["solver_reason_counts"][str(code)])
                    for level in execution_levels
                )
                for code in range(8)
            },
            "emission_reason_counts": {
                str(code): sum(
                    int(level["emission_reason_counts"][str(code)])
                    for level in execution_levels
                )
                for code in (0, 1, 2)
            },
            "failure_counts": {
                name: sum(
                    int(level["failure_counts"][name])
                    for level in execution_levels
                )
                for name in failure_names
            },
            "candidate_counts_by_slot_empty_h_b_hb": {
                name: [
                    sum(
                        int(
                            level[
                                "candidate_counts_by_slot_empty_h_b_hb"
                            ][name][slot]
                        )
                        for level in execution_levels
                    )
                    for slot in range(4)
                ]
                for name in candidate_names
            },
            "candidate_numeric_extrema_by_slot_empty_h_b_hb": {
                name: {
                    "min": [
                        min(
                            float(
                                level[
                                    "candidate_numeric_extrema_by_slot_empty_h_b_hb"
                                ][name]["min"][slot]
                            )
                            for level in execution_levels
                        )
                        for slot in range(4)
                    ],
                    "max": [
                        max(
                            float(
                                level[
                                    "candidate_numeric_extrema_by_slot_empty_h_b_hb"
                                ][name]["max"][slot]
                            )
                            for level in execution_levels
                        )
                        for slot in range(4)
                    ],
                }
                for name in candidate_numeric_names
            },
            "iteration_max": {
                name: max(
                    int(level["iteration_max"][name])
                    for level in execution_levels
                )
                for name in ("singleton_h", "singleton_b", "dual")
            },
        }

    fp32_execution = execution_summary(levels)
    bf16_execution = execution_summary(bf16_levels)
    all_levels = [*levels, *bf16_levels]
    amp_eligible = sum(int(level["eligible_rows"]) for level in amp_levels)
    amp_mismatch = sum(int(level["status_mismatch_rows"]) for level in amp_levels)
    amp_emission_mismatch = sum(
        int(level["emission_status_mismatch_rows"]) for level in amp_levels
    )
    amp_internal_fp32 = all(
        bool(level["internal_solver_and_relations_fp32"])
        for level in amp_levels
    )
    amp_ambient_dtype = all(
        bool(level["emitted_attention_restores_ambient_dtype"])
        for level in amp_levels
    )
    cosines = [
        float(value)
        for level in amp_levels
        for value in level["correction_cosines"]
    ]
    amp_mismatch_rate = (
        None if amp_eligible == 0 else float(amp_mismatch / amp_eligible)
    )
    amp_emission_mismatch_rate = (
        None
        if amp_eligible == 0
        else float(amp_emission_mismatch / amp_eligible)
    )
    qhat_mass = max(
        float(level["qhat_row_mass_error_max"]) for level in all_levels
    )
    qout_mass = max(
        float(level["qout_row_mass_error_max"]) for level in all_levels
    )
    qhat_min = min(float(level["qhat_min"]) for level in all_levels)
    qout_min = min(float(level["qout_min"]) for level in all_levels)
    hard_delta = _maximum_or_none(
        [level["accepted_hard_delta_max"] for level in all_levels]
    )
    background_delta = _maximum_or_none(
        [level["accepted_background_delta_max"] for level in all_levels]
    )
    qout_hard_delta = _maximum_or_none(
        [level["qout_hard_delta_max"] for level in all_levels]
    )
    qout_background_delta = _maximum_or_none(
        [level["qout_background_delta_max"] for level in all_levels]
    )
    accepted_hard_delta_raw = _maximum_or_none(
        [level["accepted_hard_delta_raw_max"] for level in all_levels]
    )
    accepted_background_delta_raw = _maximum_or_none(
        [level["accepted_background_delta_raw_max"] for level in all_levels]
    )
    qout_hard_delta_raw = _maximum_or_none(
        [level["qout_hard_delta_raw_max"] for level in all_levels]
    )
    qout_background_delta_raw = _maximum_or_none(
        [level["qout_background_delta_raw_max"] for level in all_levels]
    )
    emitted_hard_delta = _maximum_or_none(
        [level["emitted_hard_delta_max"] for level in all_levels]
    )
    emitted_background_delta = _maximum_or_none(
        [level["emitted_background_delta_max"] for level in all_levels]
    )
    emitted_hard_delta_raw = _maximum_or_none(
        [level["emitted_hard_delta_raw_max"] for level in all_levels]
    )
    emitted_background_delta_raw = _maximum_or_none(
        [level["emitted_background_delta_raw_max"] for level in all_levels]
    )
    objective = _minimum_or_none(
        [level["accepted_objective_min"] for level in all_levels]
    )
    qout_objective = _minimum_or_none(
        [level["qout_objective_min"] for level in all_levels]
    )
    emitted_objective = _minimum_or_none(
        [level["emitted_objective_min"] for level in all_levels]
    )
    kkt = _maximum_or_none(
        [level["effective_kkt_max"] for level in all_levels]
    )
    zero_exact = all(bool(image["gain_zero_six_head_bitwise_equal"]) for image in images)
    all_finite = all(bool(level["finite"]) for level in all_levels) and all(
        bool(image["all_model_outputs_finite"]) for image in images
    )
    terminal_one_hot = all(
        bool(level["terminal_one_hot"]) for level in all_levels
    )
    nonfinite_input_count = (
        int(fp32_execution["nonfinite_input_count"])
        + int(bf16_execution["nonfinite_input_count"])
    )

    def at_most(value: float | None, limit: float) -> bool:
        return value is None or value <= limit

    def at_least(value: float | None, limit: float) -> bool:
        return value is None or value >= limit

    gates = {
        "gain_zero_six_head_bitwise_equal": zero_exact,
        "all_outputs_and_solver_tensors_finite": all_finite,
        "nonfinite_input_count_zero": nonfinite_input_count == 0,
        "four_terminal_states_one_hot": terminal_one_hot,
        "emission_certificate_accounting_consistent": bool(
            fp32_execution["emission_certificate_accounting_consistent"]
            and bf16_execution["emission_certificate_accounting_consistent"]
        ),
        "qhat_row_mass_error_le_1e_6": qhat_mass <= 1e-6,
        "qout_row_mass_error_le_1e_6": qout_mass <= 1e-6,
        "qhat_min_ge_minus_1e_7": qhat_min >= -1e-7,
        "qout_min_ge_minus_1e_7": qout_min >= -1e-7,
        "accepted_defined_risk_delta_le_1e_6": (
            at_most(hard_delta, 1e-6)
            and at_most(background_delta, 1e-6)
        ),
        "qout_defined_risk_delta_le_1e_6": (
            at_most(qout_hard_delta, 1e-6)
            and at_most(qout_background_delta, 1e-6)
        ),
        "emitted_defined_risk_delta_le_1e_6": (
            at_most(emitted_hard_delta, 1e-6)
            and at_most(emitted_background_delta, 1e-6)
        ),
        "consistent_objective_ge_minus_1e_6": (
            at_least(objective, -1e-6)
            and at_least(qout_objective, -1e-6)
            and at_least(emitted_objective, -1e-6)
        ),
        "effective_projected_kkt_le_1e_6": at_most(kkt, 1e-6),
        "solver_fallback_rate_le_0_05": bool(
            fp32_execution["solver_fallback_rate"] is not None
            and fp32_execution["solver_fallback_rate"] <= 0.05
            and bf16_execution["solver_fallback_rate"] is not None
            and bf16_execution["solver_fallback_rate"] <= 0.05
        ),
        "target_nontrivial_accepted_rate_ge_0_20": bool(
            fp32_execution["target_nontrivial_accepted_rate"] is not None
            and fp32_execution["target_nontrivial_accepted_rate"] >= 0.20
        ),
        "fp32_emission_fallback_rate_le_0_01": bool(
            fp32_execution["emission_fallback_rate"] is not None
            and fp32_execution["emission_fallback_rate"] <= 0.01
        ),
        "bf16_emission_fallback_rate_le_0_05": bool(
            bf16_execution["emission_fallback_rate"] is not None
            and bf16_execution["emission_fallback_rate"] <= 0.05
        ),
        "fp32_nontrivial_emitted_rate_ge_0_20": bool(
            fp32_execution["target_nontrivial_emitted_rate"] is not None
            and fp32_execution["target_nontrivial_emitted_rate"] >= 0.20
        ),
        "bf16_nontrivial_emitted_rate_ge_0_20": bool(
            bf16_execution["target_nontrivial_emitted_rate"] is not None
            and bf16_execution["target_nontrivial_emitted_rate"] >= 0.20
        ),
        "amp_correction_cosine_ge_0_99": bool(
            cosines and min(cosines) >= 0.99
        ),
        "amp_status_mismatch_rate_le_0_01": bool(
            amp_mismatch_rate is not None and amp_mismatch_rate <= 0.01
        ),
        "amp_emission_status_mismatch_rate_le_0_01": bool(
            amp_emission_mismatch_rate is not None
            and amp_emission_mismatch_rate <= 0.01
        ),
        "amp_internal_solver_and_relations_fp32": amp_internal_fp32,
        "amp_emitted_attention_restores_ambient_dtype": amp_ambient_dtype,
    }
    output_changes = [
        float(image["full_vs_zero_output_mean_abs_change"]) for image in images
    ]
    regional_output_changes = {
        region: [
            float(value)
            for image in images
            for value in (
                image["full_vs_zero_output_mean_abs_change_by_pixel_region"][
                    region
                ],
            )
            if value is not None
        ]
        for region in REGION_NAMES
    }
    return {
        "schema": DIAGNOSTIC_SCHEMA + "/numerical_summary/v1",
        "image_count": len(images),
        "row_count": fp32_execution["row_count"],
        "execution": {
            "fp32": fp32_execution,
            "bf16": bf16_execution,
        },
        "status_counts": fp32_execution["status_counts"],
        "failure_counts": fp32_execution["failure_counts"],
        "active_set_counts": fp32_execution["active_set_counts"],
        "solver_reason_counts": fp32_execution["solver_reason_counts"],
        "emission_reason_counts": fp32_execution["emission_reason_counts"],
        "candidate_counts_by_slot_empty_h_b_hb": fp32_execution[
            "candidate_counts_by_slot_empty_h_b_hb"
        ],
        "candidate_numeric_extrema_by_slot_empty_h_b_hb": fp32_execution[
            "candidate_numeric_extrema_by_slot_empty_h_b_hb"
        ],
        "iteration_max": fp32_execution["iteration_max"],
        "nonfinite_input_count": nonfinite_input_count,
        "qhat_row_mass_error_max": qhat_mass,
        "qout_row_mass_error_max": qout_mass,
        "qhat_min": qhat_min,
        "qout_min": qout_min,
        "accepted_hard_delta_max": hard_delta,
        "accepted_background_delta_max": background_delta,
        "qout_hard_delta_max": qout_hard_delta,
        "qout_background_delta_max": qout_background_delta,
        "accepted_hard_delta_raw_max": accepted_hard_delta_raw,
        "accepted_background_delta_raw_max": accepted_background_delta_raw,
        "qout_hard_delta_raw_max": qout_hard_delta_raw,
        "qout_background_delta_raw_max": qout_background_delta_raw,
        "emitted_hard_delta_max": emitted_hard_delta,
        "emitted_background_delta_max": emitted_background_delta,
        "emitted_hard_delta_raw_max": emitted_hard_delta_raw,
        "emitted_background_delta_raw_max": emitted_background_delta_raw,
        "accepted_objective_min": objective,
        "qout_objective_min": qout_objective,
        "emitted_objective_min": emitted_objective,
        "effective_kkt_max": kkt,
        "solver_eligible_rows": fp32_execution["solver_eligible_rows"],
        "solver_fallback_rate": fp32_execution["solver_fallback_rate"],
        "support_identity_rate": fp32_execution["support_identity_rate"],
        "target_solver_eligible_rows": fp32_execution[
            "target_solver_eligible_rows"
        ],
        "target_nontrivial_accepted_rows": fp32_execution[
            "target_nontrivial_accepted_rows"
        ],
        "target_nontrivial_accepted_rate": fp32_execution[
            "target_nontrivial_accepted_rate"
        ],
        "fp32_emission_fallback_rate": fp32_execution[
            "emission_fallback_rate"
        ],
        "bf16_emission_fallback_rate": bf16_execution[
            "emission_fallback_rate"
        ],
        "fp32_nontrivial_emitted_rate": fp32_execution[
            "target_nontrivial_emitted_rate"
        ],
        "bf16_nontrivial_emitted_rate": bf16_execution[
            "target_nontrivial_emitted_rate"
        ],
        "amp_eligible_rows": amp_eligible,
        "amp_status_mismatch_rows": amp_mismatch,
        "amp_status_mismatch_rate": amp_mismatch_rate,
        "amp_emission_status_mismatch_rows": amp_emission_mismatch,
        "amp_emission_status_mismatch_rate": amp_emission_mismatch_rate,
        "amp_nontrivial_both_rows": len(cosines),
        "amp_correction_cosine_min": None if not cosines else min(cosines),
        "amp_internal_solver_and_relations_fp32": amp_internal_fp32,
        "amp_emitted_attention_restores_ambient_dtype": amp_ambient_dtype,
        "full_vs_zero_output_mean_abs_change_mean": float(
            np.mean(np.asarray(output_changes, dtype=np.float64))
        ),
        "full_vs_zero_output_mean_abs_change_by_pixel_region": {
            region: {
                "defined_image_count": len(values),
                "mean": _mean_or_none(values),
            }
            for region, values in regional_output_changes.items()
        },
        "gates": gates,
        "status": "GO" if all(gates.values()) else "NO-GO",
    }


def run_synthetic_contracts() -> dict[str, Any]:
    """Run the three frozen attention-level semantic boundary cases."""

    batch, heads, positions = 1, 4, 256

    def emitted_fixture(
        base: torch.Tensor, projection: v31_core.C3V31Projection
    ) -> torch.Tensor:
        candidate = base + float(STRESS_GAIN[0]) * (projection.qhat - base)
        return torch.where(projection.accepted, candidate, base)

    constant_key = F.normalize(
        torch.ones(batch, heads, 8, positions), dim=-1
    )
    zero_validations = tuple(
        torch.zeros(batch, heads, 1, positions) for _ in range(4)
    )
    background_support = v31_core.estimate_c3_v31_support_from_validations(
        constant_key, zero_validations
    )
    q0 = torch.full((batch, heads, 1, 8), 1.0 / 8.0)
    background_projection = v31_core.solve_dual_risk_projection_v31(
        q0,
        torch.zeros_like(q0),
        torch.zeros_like(q0),
        torch.zeros_like(q0),
        background_support.levels[0].reliability,
        hard_active=background_support.levels[0].contradictory_valid,
        background_active=background_support.levels[0].common_valid,
    )
    background_only = {
        "shape": [batch, heads, positions],
        "key_has_variation_false": not bool(
            background_support.key_has_variation.any()
        ),
        "reliability_zero": all(
            torch.count_nonzero(level.reliability).item() == 0
            for level in background_support.levels
        ),
        "projection_exact_identity": torch.equal(
            background_projection.qhat, q0
        ),
        "emitted_output_exact_identity": torch.equal(
            emitted_fixture(q0, background_projection), q0
        ),
    }
    background_only["passed"] = all(
        value for key, value in background_only.items() if key not in {"shape"}
    )

    varied_key = torch.zeros(batch, heads, 8, positions)
    varied_key[..., 42] = 1.0
    varied_key[..., 43] = -0.25
    varied_key = F.normalize(varied_key, dim=-1)
    hot_validations = [value.clone() for value in zero_validations]
    for peer_index in (1, 2, 3):
        hot_validations[peer_index][..., 42] = 1.0
    hot_support = v31_core.estimate_c3_v31_support_from_validations(
        varied_key, tuple(hot_validations)
    )
    hot_level = hot_support.levels[0]
    equal_relation_projection = v31_core.solve_dual_risk_projection_v31(
        q0,
        torch.zeros_like(q0),
        torch.zeros_like(q0),
        torch.zeros_like(q0),
        hot_level.reliability,
        hard_active=hot_level.contradictory_valid,
        background_active=hot_level.common_valid,
    )
    consistent_hot = {
        "shape": [batch, heads, positions],
        "token_index": 42,
        "peer_indices": list(hot_level.peer_indices),
        "consistent_valid": bool(hot_level.consistent_valid.all()),
        "hot_token_has_consistent_mass": bool(
            hot_level.consistent_support[..., 42].gt(0.0).all()
        ),
        "zero_relative_relations_exact_identity": torch.equal(
            equal_relation_projection.qhat, q0
        ),
        "emitted_output_exact_identity": torch.equal(
            emitted_fixture(q0, equal_relation_projection), q0
        ),
    }
    consistent_hot["passed"] = all(
        value
        for key, value in consistent_hot.items()
        if key not in {"shape", "token_index", "peer_indices"}
    )

    baseline = v31_core.estimate_c3_v31_support_from_validations(
        varied_key, zero_validations
    )
    shallow_validations = [value.clone() for value in zero_validations]
    shallow_validations[0][..., 42] = 1.0
    shallow = v31_core.estimate_c3_v31_support_from_validations(
        varied_key, tuple(shallow_validations)
    )
    before = baseline.levels[0]
    after = shallow.levels[0]
    shallow_projection = v31_core.solve_dual_risk_projection_v31(
        q0,
        torch.zeros_like(q0),
        torch.zeros_like(q0),
        torch.zeros_like(q0),
        after.reliability,
        hard_active=after.contradictory_valid,
        background_active=after.common_valid,
    )
    independent_fields = (
        "peer_consensus",
        "peer_dispersion",
        "agreement",
        "consistent_raw",
        "contradictory_raw",
        "common_raw",
        "consistent_mass",
        "contradictory_mass",
        "common_mass",
        "consistent_support",
        "contradictory_support",
        "common_support",
        "consistent_valid",
        "contradictory_valid",
        "common_valid",
        "consistent_positive_strength",
        "consistent_common_separation",
        "reliability",
    )
    shallow_only = {
        "shape": [batch, heads, positions],
        "token_index": 42,
        "peer_indices": list(after.peer_indices),
        "self_validation_changed": not torch.equal(
            before.self_validation, after.self_validation
        ),
        "all_peer_derived_fields_bitwise_equal": all(
            torch.equal(getattr(before, name), getattr(after, name))
            for name in independent_fields
        ),
        "required_support_invalid": not bool(
            (after.consistent_valid & after.common_valid).any()
        ),
        "reliability_zero": torch.count_nonzero(after.reliability).item() == 0,
        "projection_exact_identity": torch.equal(shallow_projection.qhat, q0),
        "emitted_output_exact_identity": torch.equal(
            emitted_fixture(q0, shallow_projection), q0
        ),
    }
    shallow_only["passed"] = all(
        value
        for key, value in shallow_only.items()
        if key not in {"shape", "token_index", "peer_indices"}
    )
    cases = {
        "background_only": background_only,
        "consistent_hot": consistent_hot,
        "shallow_only": shallow_only,
    }
    return {
        "schema": DIAGNOSTIC_SCHEMA + "/synthetic/v1",
        "fixture_geometry": {"batch": batch, "heads": heads, "positions": positions},
        "cases": cases,
        "status": "GO" if all(case["passed"] for case in cases.values()) else "NO-GO",
    }


def _require_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise C3V31DiagnosticError("diagnostic device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise C3V31DiagnosticError("requested CUDA diagnostic device is unavailable")
    return device


def _sample_record(
    *,
    adapter: C3V31DiagnosticAdapter,
    item: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if item.get("split") != "val":
        raise C3V31DiagnosticError("diagnostic received a non-validation sample")
    image = item.get("image")
    mask = item.get("mask")
    sample_id = item.get("sample_id")
    original_hw = item.get("original_hw")
    padded_hw = item.get("padded_hw")
    if (
        not isinstance(image, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or image.ndim != 3
        or mask.ndim != 3
        or tuple(image.shape) != tuple(mask.shape)
        or type(sample_id) is not str
        or not isinstance(original_hw, tuple)
        or not isinstance(padded_hw, tuple)
    ):
        raise C3V31DiagnosticError("validation sample schema differs")
    original_hw = tuple(int(value) for value in original_hw)
    padded_hw = tuple(int(value) for value in padded_hw)
    if tuple(image.shape[-2:]) != padded_hw or any(
        original_hw[index] > padded_hw[index] for index in (0, 1)
    ):
        raise C3V31DiagnosticError("validation sample geometry differs")
    batched = image.unsqueeze(0).to(device)
    target = mask[0].detach().float().cpu().numpy()

    with torch.inference_mode():
        source_output = adapter.source_model(batched)
        source_probability = _final_probability(source_output)
        anchor_outputs = _output_tuple(adapter.anchor_model(batched))
        zero_outputs, zero_capture = _capture_forward(
            adapter.diagnostic_model,
            batched,
            gain_override=ZERO_GAIN,
            amp_bfloat16=False,
        )
        full_outputs, full_capture = _capture_forward(
            adapter.diagnostic_model,
            batched,
            gain_override=STRESS_GAIN,
            amp_bfloat16=False,
        )
        bf16_outputs, bf16_capture = _capture_forward(
            adapter.diagnostic_model,
            batched,
            gain_override=STRESS_GAIN,
            amp_bfloat16=True,
        )

    for execution, capture in (
        ("gain_zero_fp32", zero_capture),
        ("stress_fp32", full_capture),
        ("stress_bf16", bf16_capture),
    ):
        _reject_base_contract_failure(capture, execution=execution)

    if len(anchor_outputs) != 6:
        raise C3V31DiagnosticError("anchor model did not expose all six heads")
    gain_zero_equal = all(
        torch.equal(anchor, candidate)
        for anchor, candidate in zip(anchor_outputs, zero_outputs)
    )
    all_outputs = (
        *_output_tuple(source_output),
        *anchor_outputs,
        *zero_outputs,
        *full_outputs,
        *bf16_outputs,
    )
    all_outputs_finite = all(
        bool(torch.isfinite(output.detach().float()).all())
        for output in all_outputs
    )
    if tuple(source_probability.shape) != (1, 1, *padded_hw):
        raise C3V31DiagnosticError("source probability geometry differs")
    source_probability_np = source_probability[0, 0].detach().float().cpu().numpy()
    token_hw = validate_token_geometry(full_capture, padded_hw)
    if validate_token_geometry(zero_capture, padded_hw) != token_hw or (
        validate_token_geometry(bf16_capture, padded_hw) != token_hw
    ):
        raise C3V31DiagnosticError("zero/full/AMP token geometry differs")
    regions = build_token_regions(
        target,
        source_probability_np,
        token_size=token_hw,
        valid_hw=original_hw,
    )
    pixel_regions = build_pixel_regions(
        target,
        source_probability_np,
        valid_hw=original_hw,
    )
    support_cells = _support_cell_records(full_capture, regions)
    has_target = bool((target[: original_hw[0], : original_hw[1]] > 0.5).any())
    numerical_levels = [
        _level_numerical_record(
            level, has_target=has_target, execution="fp32"
        )
        for level in full_capture["levels"]
    ]
    bf16_numerical_levels = [
        _level_numerical_record(
            level, has_target=has_target, execution="bf16"
        )
        for level in bf16_capture["levels"]
    ]
    amp_levels = [
        _amp_level_comparison(fp32_level, bf16_level)
        for fp32_level, bf16_level in zip(
            full_capture["levels"], bf16_capture["levels"]
        )
    ]
    valid_slice = (
        slice(None),
        slice(None),
        slice(0, original_hw[0]),
        slice(0, original_hw[1]),
    )
    pixel_change = (
        full_outputs[-1][0, 0].detach().float()
        - zero_outputs[-1][0, 0].detach().float()
    ).abs().cpu()
    output_change = float(pixel_change[valid_slice[-2:]].mean().item())
    output_change_by_region: dict[str, float | None] = {}
    for name in REGION_NAMES:
        mask_tensor = torch.from_numpy(pixel_regions[name]).bool()
        output_change_by_region[name] = (
            None
            if not bool(mask_tensor.any())
            else float(pixel_change[mask_tensor].mean().item())
        )
    return {
        "schema": DIAGNOSTIC_SCHEMA + "/image/v1",
        "sample_id": sample_id,
        "data_role": "val",
        "original_hw": list(original_hw),
        "padded_hw": list(padded_hw),
        "token_hw": list(token_hw),
        "token_count": int(token_hw[0] * token_hw[1]),
        "has_target": has_target,
        "empty_regions": {
            name: not bool(regions[name].any()) for name in REGION_NAMES
        },
        "prediction_components": prediction_component_taxonomy(
            target,
            source_probability_np,
            valid_hw=original_hw,
        ),
        "support_cells": support_cells,
        "gain_zero_six_head_bitwise_equal": gain_zero_equal,
        "all_model_outputs_finite": all_outputs_finite,
        "full_vs_zero_output_mean_abs_change": output_change,
        "full_vs_zero_output_mean_abs_change_by_pixel_region": (
            output_change_by_region
        ),
        "numerical_levels": numerical_levels,
        "bf16_numerical_levels": bf16_numerical_levels,
        "amp_levels": amp_levels,
        "test_split_accessed": False,
    }


def run_source_diagnostic(
    source_method: str,
    dataset: v2_data.EviSIRSTV2ValDataset,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Run one source independently; no statistics cross source boundaries."""

    if source_method not in SOURCE_SPECS:
        raise ValueError("unknown diagnostic source")
    if len(dataset) != EXPECTED_VALIDATION_IMAGES:
        raise C3V31DiagnosticError("diagnostic requires all 160 validation images")
    if (
        not isinstance(getattr(dataset, "metadata", None), Mapping)
        or dataset.metadata.get("split") != "val"
        or dataset.metadata.get("test_index_opened") is not False
        or tuple(getattr(dataset, "sample_ids", ()))
        != tuple(getattr(dataset.contract, "val_ids", ()))
        or bool(getattr(dataset.contract, "data_tree_verified", True))
        or getattr(dataset.contract, "data_tree_sha256", None) is not None
    ):
        raise C3V31DiagnosticError("source diagnostic requires V2 validation only")
    resolved_device = _require_device(device)
    validation_tree_sha = validation_data_tree_sha256(dataset)
    adapter = build_c3_v31_diagnostic_adapter(source_method)
    for model in (
        adapter.source_model,
        adapter.anchor_model,
        adapter.diagnostic_model,
    ):
        model.to(resolved_device)
        model.eval()
    adapter.source_model.mode = "test"
    adapter.anchor_model.mode = "train"
    adapter.diagnostic_model.mode = "train"
    named_models = {
        "source": adapter.source_model,
        "anchor": adapter.anchor_model,
        "diagnostic": adapter.diagnostic_model,
    }
    state_before = {
        name: state_dict_sha256(model.state_dict())
        for name, model in named_models.items()
    }
    images: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            if not isinstance(item, Mapping):
                raise C3V31DiagnosticError(
                    "validation dataset must return metadata mappings"
                )
            images.append(
                _sample_record(
                    adapter=adapter,
                    item=item,
                    device=resolved_device,
                )
            )
    state_after = {
        name: state_dict_sha256(model.state_dict())
        for name, model in named_models.items()
    }
    state_unchanged = {
        name: state_before[name] == state_after[name] for name in named_models
    }
    if not all(state_unchanged.values()):
        raise C3V31DiagnosticError("diagnostic mutated model state")
    semantic = summarize_semantic_evidence(
        images, source_method=source_method
    )
    numerical = summarize_numerical_evidence(images)
    gates = {
        "semantic": semantic["status"] == "GO",
        "numerical": numerical["status"] == "GO",
        "all_160_validation_images": len(images) == EXPECTED_VALIDATION_IMAGES,
        "state_unchanged": all(state_unchanged.values()),
        "all_model_states_unchanged": all(state_unchanged.values()),
    }
    report = {
        "schema": DIAGNOSTIC_SCHEMA + "/source/v1",
        "source_method": source_method,
        "source_epoch": SOURCE_SPECS[source_method].epoch,
        "adapter": dict(adapter.evidence),
        "dataset": {
            "name": "IRSTD-1K",
            "split": "val",
            "sample_count": len(images),
            "split_manifest_sha256": dataset.contract.manifest_sha256,
            "validation_only_data_tree_sha256": validation_tree_sha,
            "full_source_data_tree_opened": False,
            "target_mode": dataset.target_mode,
            "test_index_opened": False,
        },
        "gain_overrides": {
            "zero_anchor": list(ZERO_GAIN),
            "operator_stress": list(STRESS_GAIN),
            "parameter_was_not_modified": True,
        },
        "semantic": semantic,
        "numerical": numerical,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_unchanged,
        "images": images,
        "gates": gates,
        "status": "GO" if all(gates.values()) else "NO-GO",
        "training_started": False,
        "test_split_accessed": False,
    }
    return _with_evidence_sha256(report)


def _frozen_protocol_record() -> dict[str, Any]:
    return {
        "dataset": "IRSTD-1K",
        "data_role": "val",
        "validation_image_count": EXPECTED_VALIDATION_IMAGES,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "match_radius_px": MATCH_RADIUS,
        "token_downsample": TOKEN_DOWNSAMPLE,
        "zero_gain": list(ZERO_GAIN),
        "stress_gain": list(STRESS_GAIN),
        "nontrivial_l1": NONTRIVIAL_L1,
        "high_confidence": HIGH_CONFIDENCE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "diagnostic_interventions": list(DIAGNOSTIC_INTERVENTIONS),
        "source_statistics_are_separate": True,
    }


def _validate_source_report_contract(
    source_method: str, source_report: Mapping[str, Any]
) -> None:
    if source_method not in SOURCE_SPECS or not isinstance(source_report, Mapping):
        raise C3V31DiagnosticError("single-source report identity is invalid")
    _verify_evidence_sha256(source_report)
    spec = SOURCE_SPECS[source_method]
    adapter = source_report.get("adapter")
    dataset = source_report.get("dataset")
    gates = source_report.get("gates")
    gain_overrides = source_report.get("gain_overrides")
    semantic = source_report.get("semantic")
    numerical = source_report.get("numerical")
    images = source_report.get("images")
    state_before = source_report.get("model_state_sha256_before")
    state_after = source_report.get("model_state_sha256_after")
    state_unchanged = source_report.get("model_state_unchanged")
    if (
        source_report.get("schema") != DIAGNOSTIC_SCHEMA + "/source/v1"
        or source_report.get("source_method") != source_method
        or source_report.get("source_epoch") != spec.epoch
        or not isinstance(adapter, Mapping)
        or adapter.get("source_method") != source_method
        or adapter.get("source_epoch") != spec.epoch
        or adapter.get("checkpoint_file_sha256")
        != spec.checkpoint_file_sha256
        or adapter.get("checkpoint_relative_path")
        != spec.checkpoint_relative_path
        or adapter.get("schema") != DIAGNOSTIC_SCHEMA + "/adapter/v1"
        or adapter.get("diagnostic_only") is not True
        or adapter.get("test_split_accessed") is not False
        or adapter.get("shared_state_key_count") != 510
        or adapter.get("v31_candidate_validator_rejected_source_state")
        is not True
        or adapter.get("v31_gain_zero") is not True
        or any(
            type(adapter.get(name)) is not str or len(adapter[name]) != 64
            for name in (
                "source_state_sha256",
                "shared_state_sha256",
                "anchor_state_sha256",
                "diagnostic_state_sha256",
            )
        )
        or not isinstance(dataset, Mapping)
        or dataset.get("name") != "IRSTD-1K"
        or dataset.get("split") != "val"
        or dataset.get("sample_count") != EXPECTED_VALIDATION_IMAGES
        or dataset.get("full_source_data_tree_opened") is not False
        or dataset.get("test_index_opened") is not False
        or dataset.get("target_mode") != "binary"
        or type(dataset.get("split_manifest_sha256")) is not str
        or len(dataset["split_manifest_sha256"]) != 64
        or type(dataset.get("validation_only_data_tree_sha256")) is not str
        or len(dataset["validation_only_data_tree_sha256"]) != 64
        or source_report.get("training_started") is not False
        or source_report.get("test_split_accessed") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or not isinstance(gain_overrides, Mapping)
        or gain_overrides.get("zero_anchor") != list(ZERO_GAIN)
        or gain_overrides.get("operator_stress") != list(STRESS_GAIN)
        or gain_overrides.get("parameter_was_not_modified") is not True
        or not isinstance(semantic, Mapping)
        or semantic.get("schema") != DIAGNOSTIC_SCHEMA + "/semantic_summary/v1"
        or semantic.get("source_method") != source_method
        or semantic.get("image_count") != EXPECTED_VALIDATION_IMAGES
        or not isinstance(numerical, Mapping)
        or numerical.get("schema") != DIAGNOSTIC_SCHEMA + "/numerical_summary/v1"
        or numerical.get("image_count") != EXPECTED_VALIDATION_IMAGES
        or not isinstance(images, list)
        or len(images) != EXPECTED_VALIDATION_IMAGES
        or not isinstance(state_before, Mapping)
        or not isinstance(state_after, Mapping)
        or not isinstance(state_unchanged, Mapping)
        or set(state_before) != {"source", "anchor", "diagnostic"}
        or state_before != state_after
        or set(state_unchanged) != {"source", "anchor", "diagnostic"}
        or not all(value is True for value in state_unchanged.values())
    ):
        raise C3V31DiagnosticError("single-source report contract differs")
    sample_ids: list[str] = []
    for image in images:
        if (
            not isinstance(image, Mapping)
            or type(image.get("sample_id")) is not str
            or image.get("data_role") != "val"
            or image.get("test_split_accessed") is not False
        ):
            raise C3V31DiagnosticError("single-source image evidence differs")
        sample_ids.append(image["sample_id"])
    if len(set(sample_ids)) != EXPECTED_VALIDATION_IMAGES:
        raise C3V31DiagnosticError("single-source image IDs are not unique")
    expected_status = (
        "GO"
        if all(bool(value) for value in gates.values())
        and source_report["semantic"].get("status") == "GO"
        and source_report["numerical"].get("status") == "GO"
        else "NO-GO"
    )
    if source_report.get("status") != expected_status:
        raise C3V31DiagnosticError("single-source status is not reproducible")


def _build_single_source_report(
    source_method: str,
    source_report: Mapping[str, Any],
    synthetic: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_source_report_contract(source_method, source_report)
    synthetic_copy = _strict_json_clone(synthetic)
    if (
        not isinstance(synthetic_copy, dict)
        or synthetic_copy.get("schema") != DIAGNOSTIC_SCHEMA + "/synthetic/v1"
        or synthetic_copy.get("status") not in {"GO", "NO-GO"}
    ):
        raise C3V31DiagnosticError("synthetic report contract differs")
    dataset = source_report["dataset"]
    gates = {
        "synthetic_contracts": synthetic_copy["status"] == "GO",
        "source_semantic_and_numerical_gates": source_report["status"] == "GO",
        "all_160_validation_images": (
            dataset["sample_count"] == EXPECTED_VALIDATION_IMAGES
        ),
        "validation_only": (
            source_report["test_split_accessed"] is False
            and source_report["training_started"] is False
        ),
    }
    report = {
        "schema": SINGLE_SOURCE_SCHEMA,
        "source_method": source_method,
        "source_epoch": SOURCE_SPECS[source_method].epoch,
        "status": "GO" if all(gates.values()) else "NO-GO",
        "decision": (
            "READY_FOR_TWO_SOURCE_MERGE"
            if all(gates.values())
            else "SOURCE_NO_GO"
        ),
        "protocol": _frozen_protocol_record(),
        "dataset_binding": {
            "name": dataset["name"],
            "split": dataset["split"],
            "sample_count": dataset["sample_count"],
            "split_manifest_sha256": dataset["split_manifest_sha256"],
            "validation_only_data_tree_sha256": dataset[
                "validation_only_data_tree_sha256"
            ],
            "target_mode": dataset["target_mode"],
        },
        "synthetic": synthetic_copy,
        "synthetic_evidence_sha256": _canonical_sha256(synthetic_copy),
        "source": _strict_json_clone(source_report),
        "source_report_evidence_sha256": source_report["evidence_sha256"],
        "gates": gates,
        "training_started": False,
        "test_split_accessed": False,
        "official_test_accessed": False,
    }
    return _with_evidence_sha256(report)


def run_single_source_validation_diagnostic(
    source_method: str,
    *,
    dataset_root: str | Path,
    split_root: str | Path = v2_data.DEFAULT_SPLIT_ROOT,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    """Run one independent source, suitable for one-GPU-per-source execution."""

    if source_method not in SOURCE_SPECS:
        raise ValueError("single-source diagnostic requires one frozen source")
    dataset = build_validation_dataset(dataset_root, split_root=split_root)
    synthetic = run_synthetic_contracts()
    source_report = run_source_diagnostic(source_method, dataset, device=device)
    return _build_single_source_report(source_method, source_report, synthetic)


def _validate_single_source_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise C3V31DiagnosticError("single-source evidence must be a mapping")
    _verify_evidence_sha256(report)
    source_method = report.get("source_method")
    if type(source_method) is not str or source_method not in SOURCE_SPECS:
        raise C3V31DiagnosticError("single-source method is invalid")
    source = report.get("source")
    synthetic = report.get("synthetic")
    if not isinstance(source, Mapping) or not isinstance(synthetic, Mapping):
        raise C3V31DiagnosticError("single-source nested evidence is missing")
    rebuilt = _build_single_source_report(source_method, source, synthetic)
    if rebuilt != _strict_json_clone(report):
        raise C3V31DiagnosticError("single-source report is not reproducible")
    return rebuilt


def merge_single_source_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strictly merge exactly two independently produced source reports."""

    if isinstance(reports, (str, bytes)) or len(reports) != 2:
        raise C3V31DiagnosticError("merge requires exactly two source reports")
    validated = [_validate_single_source_report(report) for report in reports]
    by_method = {report["source_method"]: report for report in validated}
    if set(by_method) != set(SOURCE_SPECS) or len(by_method) != 2:
        raise C3V31DiagnosticError("merge requires one report per frozen source")
    ordered = [by_method[method] for method in SOURCE_SPECS]
    dataset_binding = ordered[0]["dataset_binding"]
    if any(report["dataset_binding"] != dataset_binding for report in ordered[1:]):
        raise C3V31DiagnosticError("source reports bind different validation data")
    synthetic = ordered[0]["synthetic"]
    if any(
        report["synthetic_evidence_sha256"]
        != ordered[0]["synthetic_evidence_sha256"]
        or report["synthetic"] != synthetic
        for report in ordered[1:]
    ):
        raise C3V31DiagnosticError("source reports bind different synthetic evidence")
    sources = {
        report["source_method"]: report["source"] for report in ordered
    }
    gates = {
        "both_frozen_sources_present": True,
        "synthetic_contracts": synthetic["status"] == "GO",
        "all_source_semantic_and_numerical_gates": all(
            report["status"] == "GO" for report in ordered
        ),
        "validation_only": all(
            report["test_split_accessed"] is False
            and report["training_started"] is False
            for report in ordered
        ),
        "dataset_binding_identical": True,
        "single_source_evidence_verified": True,
    }
    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "GO" if all(gates.values()) else "NO-GO",
        "decision": (
            "ALLOW_1_EPOCH_SMOKE"
            if all(gates.values())
            else "RETURN_TO_C3_SUPPORT_RISK_MECHANISM"
        ),
        "execution_mode": "merged_single_source_reports",
        "protocol": _frozen_protocol_record(),
        "requested_sources": list(SOURCE_SPECS),
        "missing_sources": [],
        "dataset_binding": dataset_binding,
        "synthetic": synthetic,
        "sources": sources,
        "merge_provenance": {
            report["source_method"]: report["evidence_sha256"]
            for report in ordered
        },
        "gates": gates,
        "training_started": False,
        "test_split_accessed": False,
        "official_test_accessed": False,
    }
    return _with_evidence_sha256(report)


def merge_single_source_report_files(
    paths: Sequence[str | Path],
) -> dict[str, Any]:
    if isinstance(paths, (str, bytes)) or len(paths) != 2:
        raise C3V31DiagnosticError("merge requires exactly two report files")
    return merge_single_source_reports(
        [load_strict_json_report(path) for path in paths]
    )


def run_validation_diagnostic(
    source_methods: Sequence[str],
    *,
    dataset_root: str | Path,
    split_root: str | Path = v2_data.DEFAULT_SPLIT_ROOT,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    """Run the frozen two-source validation-only GO/NO-GO diagnostic."""

    methods = _require_source_methods(source_methods)
    dataset = build_validation_dataset(dataset_root, split_root=split_root)
    synthetic = run_synthetic_contracts()
    sources = {
        method: run_source_diagnostic(method, dataset, device=device)
        for method in methods
    }
    missing_sources = sorted(set(SOURCE_SPECS) - set(methods))
    gates = {
        "both_frozen_sources_present": not missing_sources,
        "synthetic_contracts": synthetic["status"] == "GO",
        "all_source_semantic_and_numerical_gates": bool(sources)
        and all(report["status"] == "GO" for report in sources.values()),
        "validation_only": all(
            report["test_split_accessed"] is False
            and report["training_started"] is False
            for report in sources.values()
        ),
    }
    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "GO" if all(gates.values()) else "NO-GO",
        "decision": (
            "ALLOW_1_EPOCH_SMOKE"
            if all(gates.values())
            else "RETURN_TO_C3_SUPPORT_RISK_MECHANISM"
        ),
        "protocol": _frozen_protocol_record(),
        "requested_sources": list(methods),
        "missing_sources": missing_sources,
        "synthetic": synthetic,
        "sources": sources,
        "gates": gates,
        "training_started": False,
        "test_split_accessed": False,
        "official_test_accessed": False,
    }
    return _with_evidence_sha256(report)


def preflight(source_methods: Sequence[str]) -> dict[str, Any]:
    methods = _require_source_methods(source_methods)
    adapters = [build_c3_v31_diagnostic_adapter(method) for method in methods]
    report = {
        "schema": DIAGNOSTIC_SCHEMA + "/preflight/v1",
        "status": "ready",
        "sources": [dict(adapter.evidence) for adapter in adapters],
        "zero_gain": list(ZERO_GAIN),
        "stress_gain": list(STRESS_GAIN),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "diagnostic_interventions": list(DIAGNOSTIC_INTERVENTIONS),
        "dataset_opened": False,
        "gpu_opened": False,
        "official_test_accessed": False,
        "training_started": False,
    }
    return _with_evidence_sha256(report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("all", *SOURCE_SPECS),
        default="all",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--single-source", action="store_true")
    mode.add_argument(
        "--merge",
        type=Path,
        nargs=2,
        metavar=("SC_SOURCE_JSON", "V21_SOURCE_JSON"),
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--split-root", type=Path, default=v2_data.DEFAULT_SPLIT_ROOT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    methods = tuple(SOURCE_SPECS) if args.source == "all" else (args.source,)
    if args.merge is not None:
        if args.dataset_root is not None:
            raise SystemExit("--merge does not open --dataset-root")
        report = merge_single_source_report_files(args.merge)
    elif args.preflight:
        report = preflight(methods)
    elif args.single_source:
        if args.source == "all":
            raise SystemExit("--single-source requires one explicit --source")
        if args.dataset_root is None:
            raise SystemExit("--dataset-root is required for the validation diagnostic")
        report = run_single_source_validation_diagnostic(
            args.source,
            dataset_root=args.dataset_root,
            split_root=args.split_root,
            device=args.device,
        )
    else:
        if args.dataset_root is None:
            raise SystemExit("--dataset-root is required for the validation diagnostic")
        report = run_validation_diagnostic(
            methods,
            dataset_root=args.dataset_root,
            split_root=args.split_root,
            device=args.device,
        )
    if args.output is None:
        print(strict_json_dumps(report))
    else:
        output = write_json_atomic(args.output, report)
        print(
            strict_json_dumps(
                {
                    "schema": report.get("schema"),
                    "status": report.get("status"),
                    "decision": report.get("decision"),
                    "source_method": report.get("source_method"),
                    "evidence_sha256": report.get("evidence_sha256"),
                    "output": str(output),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
