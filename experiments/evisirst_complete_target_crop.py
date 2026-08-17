"""Auditable complete-target crops and component diagnostics for EviSIRST V2.

This is an experimental, non-frozen augmentation module.  It deliberately
does not resolve dataset paths or split names: the dataset adapter below
inherits the existing V2 *train* dataset and only replaces its spatial
augmentation.  There is no code path that opens a test index.

Half of the stateless augmentation requests choose a crop containing one
complete 8-connected target component with an eight-pixel margin.  A request
is realized only when every other component intersecting the crop is also
complete.  The other half is an unconstrained uniform crop.  An infeasible
complete-target request falls back explicitly to a uniform crop and records
the reason and any cut component labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage import measure

from experiments import evisirst_v2_data as v2_data
from experiments import three_dataset_v2_protocol as source_protocol


POLICY_SCHEMA = "evisirst_complete_target_crop_policy/v1"
AUGMENTATION_VERSION = (
    "evisirst_complete_target_crop_run_dataset_epoch_sample_occurrence_sha256_v1"
)
CROP_AUDIT_SCHEMA = "evisirst_complete_target_crop_audit/v1"
CROP_AUDIT_STATE_SCHEMA = "evisirst_complete_target_crop_audit_state/v1"
DIAGNOSTIC_SCHEMA = "evisirst_matched_target_diagnostics/v1"
if source_protocol.PATCH_SIZE != 256:
    raise RuntimeError("complete-target v1 requires the frozen 256-pixel patch")
CROP_SIZE = source_protocol.PATCH_SIZE
COMPLETE_TARGET_PROBABILITY = 0.5
COMPLETE_TARGET_MARGIN = 8
COMPONENT_CONNECTIVITY = 2
COMPONENT_NEIGHBORHOOD = "8-connected"
PREDICTION_THRESHOLD = 0.5
TARGET_THRESHOLD = 0.5
MATCH_RADIUS = 3.0
FORMAL_NUM_WORKERS = 0
TEST_SPLIT_ACCESSED = False
_REQUESTS = ("complete_target", "uniform")
_REALIZATIONS = ("complete_target", "uniform")


class CompleteTargetCropError(ValueError):
    """The complete-target augmentation contract was violated."""


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
        raise CompleteTargetCropError(
            f"value cannot be encoded as canonical JSON: {exc}"
        ) from exc


def policy_identity() -> dict[str, Any]:
    """Return the exact JSON-native policy identity for checkpoint binding."""

    return {
        "schema": POLICY_SCHEMA,
        "augmentation_version": AUGMENTATION_VERSION,
        "crop_size": CROP_SIZE,
        "padding": "bottom/right constant-zero to at least crop_size",
        "request_probability": {
            "complete_target": COMPLETE_TARGET_PROBABILITY,
            "uniform": 1.0 - COMPLETE_TARGET_PROBABILITY,
        },
        "complete_target_margin_pixels_each_side": COMPLETE_TARGET_MARGIN,
        "component_connectivity": COMPONENT_CONNECTIVITY,
        "component_neighborhood": COMPONENT_NEIGHBORHOOD,
        "complete_request_constraint": (
            "selected component plus margin is inside crop; every other "
            "intersecting component is wholly inside crop"
        ),
        "infeasible_request_behavior": "explicit_uniform_fallback",
        "uniform_crop_distribution": "uniform_integer_top_and_left",
        "flip_axis0_probability": 0.5,
        "flip_axis1_probability": 0.5,
        "transpose_probability": 0.5,
        "non_crop_augmentation": {
            "source": "experiments.evisirst_v2_data._transform_plan",
            "augmentation_version": v2_data.AUGMENTATION_VERSION,
            "reuse": "exact flip_axis0/flip_axis1/transpose tuple",
            "legacy_crop_coordinates_used": False,
        },
        "stateless_seed_fields": [
            "augmentation_version",
            "run_seed",
            "dataset",
            "epoch",
            "sample_id",
            "occurrence",
        ],
        "formal_num_workers": FORMAL_NUM_WORKERS,
        "split_scope": "V2 train membership only",
        "test_split_accessed": TEST_SPLIT_ACCESSED,
        "diagnostics": {
            "prediction_rule": "probability>0.5",
            "target_rule": "target>0.5",
            "component_neighborhood": COMPONENT_NEIGHBORHOOD,
            "assignment": "Hungarian/scipy.optimize.linear_sum_assignment",
            "match_rule": "centroid_distance<3",
            "matched_target_pixel_recall": (
                "matched-pair intersection pixels / matched target pixels"
            ),
            "matched_component_area_ratio": (
                "mean(predicted component area / target component area)"
            ),
            "centroid_error": "mean matched centroid Euclidean distance",
        },
    }


def policy_identity_sha256() -> str:
    return hashlib.sha256(_canonical_json_bytes(policy_identity())).hexdigest()


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompleteTargetCropError(f"{label} must be a non-negative integer")
    return value


def _require_seed_inputs(
    *, run_seed: Any, dataset: Any, epoch: Any, sample_id: Any, occurrence: Any
) -> None:
    _require_nonnegative_integer(run_seed, "run_seed")
    _require_nonnegative_integer(epoch, "epoch")
    _require_nonnegative_integer(occurrence, "occurrence")
    if not isinstance(dataset, str) or not dataset:
        raise CompleteTargetCropError("dataset must be a non-empty string")
    if not isinstance(sample_id, str) or not sample_id:
        raise CompleteTargetCropError("sample_id must be a non-empty string")


def stateless_augmentation_seed(
    *,
    run_seed: int,
    dataset: str,
    epoch: int,
    sample_id: str,
    occurrence: int,
    augmentation_version: str = AUGMENTATION_VERSION,
) -> int:
    """Derive a type-stable unsigned 64-bit seed from every required field."""

    _require_seed_inputs(
        run_seed=run_seed,
        dataset=dataset,
        epoch=epoch,
        sample_id=sample_id,
        occurrence=occurrence,
    )
    if not isinstance(augmentation_version, str) or not augmentation_version:
        raise CompleteTargetCropError(
            "augmentation_version must be a non-empty string"
        )
    payload = [
        augmentation_version,
        run_seed,
        dataset,
        epoch,
        sample_id,
        occurrence,
    ]
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _subseed(augmentation_seed: int, stream: str) -> int:
    return stateless_augmentation_seed(
        run_seed=augmentation_seed,
        dataset="augmentation_substream",
        epoch=0,
        sample_id=stream,
        occurrence=0,
        augmentation_version=f"{AUGMENTATION_VERSION}/substream/v1",
    )


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or not (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ):
        raise CompleteTargetCropError("binary_mask must be a numeric 2-D array")
    if not bool(np.isfinite(array).all()):
        raise CompleteTargetCropError("binary_mask contains non-finite values")
    if not bool(np.logical_or(array == 0, array == 1).all()):
        raise CompleteTargetCropError(
            "binary_mask must contain only boolean/0/1 values"
        )
    return np.asarray(array, dtype=np.bool_)


@dataclass(frozen=True)
class _Component:
    label: int
    area: int
    min_row: int
    min_col: int
    max_row_exclusive: int
    max_col_exclusive: int
    coordinates: np.ndarray


def _components(binary_mask: np.ndarray) -> tuple[_Component, ...]:
    labels = measure.label(binary_mask, connectivity=COMPONENT_CONNECTIVITY)
    result = []
    for region in measure.regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        result.append(
            _Component(
                label=int(region.label),
                area=int(region.area),
                min_row=int(min_row),
                min_col=int(min_col),
                max_row_exclusive=int(max_row),
                max_col_exclusive=int(max_col),
                coordinates=np.asarray(region.coords, dtype=np.int64),
            )
        )
    return tuple(result)


def _cut_component_labels(
    components: tuple[_Component, ...],
    *,
    top: int,
    left: int,
    size: int,
) -> tuple[int, ...]:
    bottom = top + size
    right = left + size
    cut = []
    for component in components:
        if (
            component.max_row_exclusive <= top
            or component.min_row >= bottom
            or component.max_col_exclusive <= left
            or component.min_col >= right
        ):
            continue
        if (
            component.min_row >= top
            and component.max_row_exclusive <= bottom
            and component.min_col >= left
            and component.max_col_exclusive <= right
        ):
            continue
        coordinates = component.coordinates
        inside = (
            (coordinates[:, 0] >= top)
            & (coordinates[:, 0] < bottom)
            & (coordinates[:, 1] >= left)
            & (coordinates[:, 1] < right)
        )
        if bool(inside.any()):
            cut.append(component.label)
    return tuple(cut)


def _margin_origin_ranges(
    component: _Component,
    *,
    padded_height: int,
    padded_width: int,
) -> tuple[int, int, int, int] | None:
    top_low = max(
        0,
        component.max_row_exclusive + COMPLETE_TARGET_MARGIN - CROP_SIZE,
    )
    top_high = min(
        padded_height - CROP_SIZE,
        component.min_row - COMPLETE_TARGET_MARGIN,
    )
    left_low = max(
        0,
        component.max_col_exclusive + COMPLETE_TARGET_MARGIN - CROP_SIZE,
    )
    left_high = min(
        padded_width - CROP_SIZE,
        component.min_col - COMPLETE_TARGET_MARGIN,
    )
    if top_low > top_high or left_low > left_high:
        return None
    return top_low, top_high, left_low, left_high


def _coprime_stride(total: int, rng: random.Random) -> int:
    if total == 1:
        return 1
    while True:
        stride = rng.randrange(1, total)
        if math.gcd(stride, total) == 1:
            return stride


def _complete_origin(
    components: tuple[_Component, ...],
    *,
    padded_height: int,
    padded_width: int,
    augmentation_seed: int,
) -> tuple[int, int, _Component | None, int, str | None]:
    if not components:
        return 0, 0, None, 0, "no_target_components"
    eligible = [
        (component, ranges)
        for component in components
        if (
            ranges := _margin_origin_ranges(
                component,
                padded_height=padded_height,
                padded_width=padded_width,
            )
        )
        is not None
    ]
    if not eligible:
        return 0, 0, None, 0, "no_margin_feasible_component"
    rng = random.Random(_subseed(augmentation_seed, "complete_target_crop"))
    rng.shuffle(eligible)
    checks = 0
    for component, ranges in eligible:
        top_low, top_high, left_low, left_high = ranges
        column_count = left_high - left_low + 1
        total = (top_high - top_low + 1) * column_count
        start = rng.randrange(total)
        stride = _coprime_stride(total, rng)
        for offset in range(total):
            flat_index = (start + offset * stride) % total
            top = top_low + flat_index // column_count
            left = left_low + flat_index % column_count
            checks += 1
            if not _cut_component_labels(
                components, top=top, left=left, size=CROP_SIZE
            ):
                return top, left, component, checks, None
    return (
        0,
        0,
        None,
        checks,
        "no_complete_target_crop_without_cut_components",
    )


def _uniform_origin(
    *, padded_height: int, padded_width: int, augmentation_seed: int
) -> tuple[int, int]:
    rng = random.Random(_subseed(augmentation_seed, "uniform_crop"))
    return (
        rng.randint(0, padded_height - CROP_SIZE),
        rng.randint(0, padded_width - CROP_SIZE),
    )


@dataclass(frozen=True)
class CompleteTargetTransformPlan:
    augmentation_version: str
    augmentation_seed: int
    legacy_augmentation_version: str
    legacy_augmentation_seed: int
    legacy_crop_attempts_before_flip: int
    occurrence: int
    crop_top: int
    crop_left: int
    crop_size: int
    padded_height: int
    padded_width: int
    requested_strategy: str
    realized_strategy: str
    fallback_reason: str | None
    selected_component_label: int | None
    selected_component_area: int | None
    component_count: int
    complete_candidate_checks: int
    cut_component_labels: tuple[int, ...]
    flip_axis0: bool
    flip_axis1: bool
    transpose: bool

    @property
    def fallback(self) -> bool:
        return self.fallback_reason is not None

    @property
    def cut_component_count(self) -> int:
        return len(self.cut_component_labels)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "schema": CROP_AUDIT_SCHEMA,
            "augmentation_version": self.augmentation_version,
            "augmentation_seed": self.augmentation_seed,
            "legacy_augmentation_version": self.legacy_augmentation_version,
            "legacy_augmentation_seed": self.legacy_augmentation_seed,
            "legacy_crop_attempts_before_flip": (
                self.legacy_crop_attempts_before_flip
            ),
            "occurrence": self.occurrence,
            "requested_strategy": self.requested_strategy,
            "realized_strategy": self.realized_strategy,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "selected_component_label": self.selected_component_label,
            "selected_component_area": self.selected_component_area,
            "component_count": self.component_count,
            "complete_candidate_checks": self.complete_candidate_checks,
            "cut_component_count": self.cut_component_count,
            "cut_component_labels": list(self.cut_component_labels),
            "test_split_accessed": TEST_SPLIT_ACCESSED,
        }


def derive_complete_target_transform_plan(
    binary_mask: np.ndarray,
    *,
    run_seed: int,
    dataset: str,
    epoch: int,
    sample_id: str,
    occurrence: int,
) -> CompleteTargetTransformPlan:
    """Build the versioned stateless crop/flip/transpose plan for one sample."""

    mask = _binary_mask(binary_mask)
    _require_seed_inputs(
        run_seed=run_seed,
        dataset=dataset,
        epoch=epoch,
        sample_id=sample_id,
        occurrence=occurrence,
    )
    height, width = mask.shape
    if height < 1 or width < 1:
        raise CompleteTargetCropError("binary_mask dimensions must be positive")
    padded_height = max(height, CROP_SIZE)
    padded_width = max(width, CROP_SIZE)
    augmentation_seed = stateless_augmentation_seed(
        run_seed=run_seed,
        dataset=dataset,
        epoch=epoch,
        sample_id=sample_id,
        occurrence=occurrence,
    )
    components = _components(mask)
    request_rng = random.Random(_subseed(augmentation_seed, "request"))
    requested_complete = request_rng.random() < COMPLETE_TARGET_PROBABILITY
    requested_strategy = "complete_target" if requested_complete else "uniform"
    fallback_reason = None
    selected = None
    checks = 0
    if requested_complete:
        top, left, selected, checks, fallback_reason = _complete_origin(
            components,
            padded_height=padded_height,
            padded_width=padded_width,
            augmentation_seed=augmentation_seed,
        )
    if not requested_complete or fallback_reason is not None:
        top, left = _uniform_origin(
            padded_height=padded_height,
            padded_width=padded_width,
            augmentation_seed=augmentation_seed,
        )
    realized_strategy = (
        "complete_target"
        if requested_complete and fallback_reason is None
        else "uniform"
    )
    cut_labels = _cut_component_labels(
        components, top=top, left=left, size=CROP_SIZE
    )
    if realized_strategy == "complete_target" and cut_labels:
        raise CompleteTargetCropError(
            "internal error: a complete-target crop cuts a component"
        )
    # The experiment is a crop-only intervention.  Reproduce the existing V2
    # planner all the way through its rejection loop, discard only its crop
    # coordinates, and reuse its exact per-sample flip/transpose tuple.
    legacy_plan = v2_data._transform_plan(
        dataset=dataset,
        sample_id=sample_id,
        run_seed=run_seed,
        epoch=epoch,
        occurrence=occurrence,
        height=height,
        width=width,
        binary_crop_mask=mask,
    )
    return CompleteTargetTransformPlan(
        augmentation_version=AUGMENTATION_VERSION,
        augmentation_seed=augmentation_seed,
        legacy_augmentation_version=v2_data.AUGMENTATION_VERSION,
        legacy_augmentation_seed=legacy_plan.augmentation_seed,
        legacy_crop_attempts_before_flip=legacy_plan.crop_attempts,
        occurrence=occurrence,
        crop_top=top,
        crop_left=left,
        crop_size=CROP_SIZE,
        padded_height=padded_height,
        padded_width=padded_width,
        requested_strategy=requested_strategy,
        realized_strategy=realized_strategy,
        fallback_reason=fallback_reason,
        selected_component_label=None if selected is None else selected.label,
        selected_component_area=None if selected is None else selected.area,
        component_count=len(components),
        complete_candidate_checks=checks,
        cut_component_labels=cut_labels,
        flip_axis0=legacy_plan.flip_axis0,
        flip_axis1=legacy_plan.flip_axis1,
        transpose=legacy_plan.transpose,
    )


def _aligned_float_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise CompleteTargetCropError(f"{label} must be a 2-D array")
    if not (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ) or not bool(np.isfinite(array).all()):
        raise CompleteTargetCropError(f"{label} must contain finite numbers")
    return np.asarray(array, dtype=np.float32)


def apply_crop_and_augment(
    image: np.ndarray,
    target: np.ndarray,
    plan: CompleteTargetTransformPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the original zero-pad/crop/two-flip/transpose operations."""

    if not isinstance(plan, CompleteTargetTransformPlan):
        raise CompleteTargetCropError("plan has the wrong type")
    image_array = _aligned_float_array(image, "image")
    target_array = _aligned_float_array(target, "target")
    if image_array.shape != target_array.shape:
        raise CompleteTargetCropError("image and target shapes must match")
    height, width = image_array.shape
    if (
        plan.padded_height != max(height, CROP_SIZE)
        or plan.padded_width != max(width, CROP_SIZE)
    ):
        raise CompleteTargetCropError("plan padding does not match input shape")
    padding = (
        (0, plan.padded_height - height),
        (0, plan.padded_width - width),
    )
    image_array = np.pad(image_array, padding, mode="constant")
    target_array = np.pad(target_array, padding, mode="constant")
    top, left, size = plan.crop_top, plan.crop_left, plan.crop_size
    image_array = image_array[top : top + size, left : left + size]
    target_array = target_array[top : top + size, left : left + size]
    if plan.flip_axis0:
        image_array, target_array = image_array[::-1, :], target_array[::-1, :]
    if plan.flip_axis1:
        image_array, target_array = image_array[:, ::-1], target_array[:, ::-1]
    if plan.transpose:
        image_array = image_array.transpose(1, 0)
        target_array = target_array.transpose(1, 0)
    return (
        np.ascontiguousarray(image_array, dtype=np.float32),
        np.ascontiguousarray(target_array, dtype=np.float32),
    )


class CropAuditAccumulator:
    """In-process crop accounting; the formal protocol requires workers=0."""

    def __init__(self, *, formal_num_workers: int = FORMAL_NUM_WORKERS) -> None:
        if (
            isinstance(formal_num_workers, bool)
            or not isinstance(formal_num_workers, int)
            or formal_num_workers != FORMAL_NUM_WORKERS
        ):
            raise CompleteTargetCropError("formal_num_workers must be exactly 0")
        self._reset()

    def _reset(self) -> None:
        self.requested: Counter[str] = Counter()
        self.realized: Counter[str] = Counter()
        self.fallback_reasons: Counter[str] = Counter()
        self.observation_count = 0
        self.fallback_count = 0
        self.cut_component_crop_count = 0
        self.total_cut_component_count = 0
        self.max_cut_component_count = 0

    def update(self, plan: CompleteTargetTransformPlan) -> None:
        if not isinstance(plan, CompleteTargetTransformPlan):
            raise CompleteTargetCropError("crop audit accepts transform plans only")
        self.observation_count += 1
        self.requested[plan.requested_strategy] += 1
        self.realized[plan.realized_strategy] += 1
        if plan.fallback:
            self.fallback_count += 1
            self.fallback_reasons[str(plan.fallback_reason)] += 1
        cut_count = plan.cut_component_count
        if cut_count:
            self.cut_component_crop_count += 1
        self.total_cut_component_count += cut_count
        self.max_cut_component_count = max(self.max_cut_component_count, cut_count)

    def state_dict(self) -> dict[str, Any]:
        """Return only canonical raw counts; rates are always recomputed."""

        return {
            "schema": CROP_AUDIT_STATE_SCHEMA,
            "augmentation_version": AUGMENTATION_VERSION,
            "formal_num_workers": FORMAL_NUM_WORKERS,
            "observation_count": self.observation_count,
            "requested_counts": {
                name: int(self.requested[name]) for name in _REQUESTS
            },
            "realized_counts": {
                name: int(self.realized[name]) for name in _REALIZATIONS
            },
            "fallback_count": self.fallback_count,
            "fallback_reason_counts": dict(sorted(self.fallback_reasons.items())),
            "cut_component_crop_count": self.cut_component_crop_count,
            "total_cut_component_count": self.total_cut_component_count,
            "max_cut_component_count": self.max_cut_component_count,
            "test_split_accessed": TEST_SPLIT_ACCESSED,
        }

    @staticmethod
    def _state_count(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CompleteTargetCropError(
                f"crop audit state {label} must be a non-negative integer"
            )
        return value

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Strictly restore an audit state after validating count invariants."""

        if type(payload) is not dict:
            raise CompleteTargetCropError(
                "crop audit state must be a canonical JSON object"
            )
        expected_keys = {
            "schema",
            "augmentation_version",
            "formal_num_workers",
            "observation_count",
            "requested_counts",
            "realized_counts",
            "fallback_count",
            "fallback_reason_counts",
            "cut_component_crop_count",
            "total_cut_component_count",
            "max_cut_component_count",
            "test_split_accessed",
        }
        if set(payload) != expected_keys:
            raise CompleteTargetCropError("crop audit state keys differ")
        if payload["schema"] != CROP_AUDIT_STATE_SCHEMA:
            raise CompleteTargetCropError("crop audit state schema differs")
        if payload["augmentation_version"] != AUGMENTATION_VERSION:
            raise CompleteTargetCropError(
                "crop audit state augmentation version differs"
            )
        if (
            isinstance(payload["formal_num_workers"], bool)
            or not isinstance(payload["formal_num_workers"], int)
            or payload["formal_num_workers"] != FORMAL_NUM_WORKERS
        ):
            raise CompleteTargetCropError("crop audit state workers differ")
        if payload["test_split_accessed"] is not TEST_SPLIT_ACCESSED:
            raise CompleteTargetCropError(
                "crop audit state test-split declaration differs"
            )
        requested_payload = payload["requested_counts"]
        realized_payload = payload["realized_counts"]
        fallback_reason_payload = payload["fallback_reason_counts"]
        if type(requested_payload) is not dict or set(
            requested_payload
        ) != set(_REQUESTS):
            raise CompleteTargetCropError(
                "crop audit requested-count keys differ"
            )
        if type(realized_payload) is not dict or set(
            realized_payload
        ) != set(_REALIZATIONS):
            raise CompleteTargetCropError("crop audit realized-count keys differ")
        if type(fallback_reason_payload) is not dict:
            raise CompleteTargetCropError(
                "crop audit fallback reasons must be a mapping"
            )
        allowed_reasons = {
            "no_target_components",
            "no_margin_feasible_component",
            "no_complete_target_crop_without_cut_components",
        }
        if any(
            not isinstance(reason, str)
            or not reason
            or reason not in allowed_reasons
            for reason in fallback_reason_payload
        ):
            raise CompleteTargetCropError(
                "crop audit fallback reason keys differ"
            )
        observation_count = self._state_count(
            payload["observation_count"], "observation_count"
        )
        requested = {
            name: self._state_count(
                requested_payload[name], f"requested_counts.{name}"
            )
            for name in _REQUESTS
        }
        realized = {
            name: self._state_count(
                realized_payload[name], f"realized_counts.{name}"
            )
            for name in _REALIZATIONS
        }
        fallback_count = self._state_count(
            payload["fallback_count"], "fallback_count"
        )
        fallback_reasons = {
            reason: self._state_count(
                count, f"fallback_reason_counts.{reason}"
            )
            for reason, count in fallback_reason_payload.items()
        }
        if any(count == 0 for count in fallback_reasons.values()):
            raise CompleteTargetCropError(
                "crop audit fallback reason counts must be positive"
            )
        cut_crop_count = self._state_count(
            payload["cut_component_crop_count"], "cut_component_crop_count"
        )
        total_cut_count = self._state_count(
            payload["total_cut_component_count"], "total_cut_component_count"
        )
        max_cut_count = self._state_count(
            payload["max_cut_component_count"], "max_cut_component_count"
        )
        if sum(requested.values()) != observation_count:
            raise CompleteTargetCropError(
                "crop audit requested counts do not sum to observations"
            )
        if sum(realized.values()) != observation_count:
            raise CompleteTargetCropError(
                "crop audit realized counts do not sum to observations"
            )
        if sum(fallback_reasons.values()) != fallback_count:
            raise CompleteTargetCropError(
                "crop audit fallback reasons do not sum to fallback_count"
            )
        if fallback_count != requested["complete_target"] - realized[
            "complete_target"
        ]:
            raise CompleteTargetCropError(
                "crop audit complete-target fallback conservation failed"
            )
        if realized["uniform"] != requested["uniform"] + fallback_count:
            raise CompleteTargetCropError(
                "crop audit uniform fallback conservation failed"
            )
        if cut_crop_count > observation_count or total_cut_count < cut_crop_count:
            raise CompleteTargetCropError("crop audit cut-component counts differ")
        if cut_crop_count == 0 and (total_cut_count != 0 or max_cut_count != 0):
            raise CompleteTargetCropError("empty crop audit cut counts differ")
        if cut_crop_count > 0 and (
            max_cut_count < 1
            or max_cut_count > total_cut_count
            or total_cut_count > cut_crop_count * max_cut_count
        ):
            raise CompleteTargetCropError("crop audit maximum cut count differs")
        self._reset()
        self.requested.update(requested)
        self.realized.update(realized)
        self.fallback_reasons.update(fallback_reasons)
        self.observation_count = observation_count
        self.fallback_count = fallback_count
        self.cut_component_crop_count = cut_crop_count
        self.total_cut_component_count = total_cut_count
        self.max_cut_component_count = max_cut_count

    def compute(self) -> dict[str, Any]:
        denominator = max(1, self.observation_count)
        state = self.state_dict()
        return {
            "schema": f"{CROP_AUDIT_SCHEMA}/aggregate",
            "augmentation_version": state["augmentation_version"],
            "formal_num_workers": state["formal_num_workers"],
            "observation_count": state["observation_count"],
            "requested_counts": state["requested_counts"],
            "realized_counts": state["realized_counts"],
            "fallback_count": state["fallback_count"],
            "fallback_rate": self.fallback_count / denominator,
            "fallback_reason_counts": state["fallback_reason_counts"],
            "cut_component_crop_count": state["cut_component_crop_count"],
            "cut_component_crop_rate": self.cut_component_crop_count / denominator,
            "total_cut_component_count": state["total_cut_component_count"],
            "max_cut_component_count": state["max_cut_component_count"],
            "test_split_accessed": state["test_split_accessed"],
        }


class EviSIRSTCompleteTargetTrainDataset(v2_data.EviSIRSTV2TrainDataset):
    """V2 train-only adapter whose sole recipe change is the crop policy."""

    def __init__(
        self,
        dataset: str,
        *,
        dataset_root: str | Path,
        target_mode: str,
        run_seed: int,
        split_root: str | Path = v2_data.DEFAULT_SPLIT_ROOT,
        normalization_mode: str = "legacy",
        normalization_values: Mapping[str, float] | None = None,
        return_metadata: bool = False,
        verify_data_tree: bool = True,
        formal_num_workers: int = FORMAL_NUM_WORKERS,
    ) -> None:
        self.crop_audit = CropAuditAccumulator(
            formal_num_workers=formal_num_workers
        )
        super().__init__(
            dataset,
            dataset_root=dataset_root,
            target_mode=target_mode,
            run_seed=run_seed,
            split_root=split_root,
            normalization_mode=normalization_mode,
            normalization_values=normalization_values,
            return_metadata=return_metadata,
            verify_data_tree=verify_data_tree,
        )
        self.metadata.update(
            {
                "augmentation_version": AUGMENTATION_VERSION,
                "crop_policy": policy_identity(),
                "crop_policy_sha256": policy_identity_sha256(),
                "formal_num_workers": FORMAL_NUM_WORKERS,
                "test_index_opened": TEST_SPLIT_ACCESSED,
            }
        )

    def __getitem__(self, index: Any) -> Any:
        position, occurrence = self._decode_item_index(index)
        sample_id = self.sample_ids[position]
        image, target, binary_crop_mask, mask_audit = self._load(sample_id)
        original_height, original_width = image.shape
        plan = derive_complete_target_transform_plan(
            binary_crop_mask,
            run_seed=self.run_seed,
            dataset=self.dataset_name,
            epoch=self.epoch,
            sample_id=sample_id,
            occurrence=occurrence,
        )
        image, target = apply_crop_and_augment(image, target, plan)
        self.crop_audit.update(plan)
        image_tensor = torch.from_numpy(image[np.newaxis, :])
        target_tensor = torch.from_numpy(target[np.newaxis, :])
        if not self.return_metadata:
            return image_tensor, target_tensor
        return {
            "image": image_tensor,
            "mask": target_tensor,
            "dataset_name": self.dataset_name,
            "split": "train",
            "sample_id": sample_id,
            "original_hw": (original_height, original_width),
            "epoch": self.epoch,
            "occurrence": occurrence,
            "run_seed": self.run_seed,
            "augmentation_seed": plan.augmentation_seed,
            "transform_plan": asdict(plan),
            "crop_audit": plan.audit_dict(),
            "target_mode": self.target_mode,
            "normalization": self.normalization_spec.as_dict(),
            "mask_audit": mask_audit,
        }

    def crop_audit_summary(self) -> dict[str, Any]:
        return self.crop_audit.compute()

    def crop_audit_state_dict(self) -> dict[str, Any]:
        return self.crop_audit.state_dict()

    def load_crop_audit_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.crop_audit.load_state_dict(payload)


def _diagnostic_binary(value: np.ndarray, threshold: float, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ):
        raise CompleteTargetCropError(f"{label} must be a numeric 2-D array")
    if not bool(np.isfinite(array).all()):
        raise CompleteTargetCropError(f"{label} contains non-finite values")
    return np.asarray(array > threshold, dtype=np.bool_)


def matched_target_diagnostics(
    probability: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    """Diagnose matched target shape fidelity under the frozen Pd semantics."""

    prediction_binary = _diagnostic_binary(
        probability, PREDICTION_THRESHOLD, "probability"
    )
    target_binary = _diagnostic_binary(target, TARGET_THRESHOLD, "target")
    if prediction_binary.shape != target_binary.shape:
        raise CompleteTargetCropError("probability and target shapes must match")
    prediction_labels = measure.label(
        prediction_binary, connectivity=COMPONENT_CONNECTIVITY
    )
    target_labels = measure.label(target_binary, connectivity=COMPONENT_CONNECTIVITY)
    predicted = measure.regionprops(prediction_labels)
    targets = measure.regionprops(target_labels)
    matched: list[tuple[int, int, float]] = []
    if targets and predicted:
        distances = np.empty((len(targets), len(predicted)), dtype=np.float64)
        for target_index, target_region in enumerate(targets):
            target_centroid = np.asarray(target_region.centroid)
            for prediction_index, prediction_region in enumerate(predicted):
                distances[target_index, prediction_index] = np.linalg.norm(
                    np.asarray(prediction_region.centroid) - target_centroid
                )
        reward = (min(len(targets), len(predicted)) + 1) * max(1.0, MATCH_RADIUS)
        real_cost = np.where(
            distances < MATCH_RADIUS,
            distances - reward,
            reward,
        )
        cost = np.concatenate(
            (real_cost, np.zeros((len(targets), len(targets)))), axis=1
        )
        assigned_targets, assigned_columns = linear_sum_assignment(cost)
        for target_index, column in zip(assigned_targets, assigned_columns):
            if column < len(predicted) and distances[target_index, column] < MATCH_RADIUS:
                matched.append(
                    (int(target_index), int(column), float(distances[target_index, column]))
                )

    matches = []
    overlap_sum = 0
    matched_target_area_sum = 0
    area_ratio_sum = 0.0
    centroid_error_sum = 0.0
    for target_index, prediction_index, centroid_error in matched:
        target_region = targets[target_index]
        prediction_region = predicted[prediction_index]
        target_pixels = target_labels == target_region.label
        predicted_pixels = prediction_labels == prediction_region.label
        overlap = int(np.logical_and(target_pixels, predicted_pixels).sum())
        target_area = int(target_region.area)
        predicted_area = int(prediction_region.area)
        pixel_recall = overlap / target_area
        area_ratio = predicted_area / target_area
        overlap_sum += overlap
        matched_target_area_sum += target_area
        area_ratio_sum += area_ratio
        centroid_error_sum += centroid_error
        matches.append(
            {
                "target_component_label": int(target_region.label),
                "predicted_component_label": int(prediction_region.label),
                "target_area": target_area,
                "predicted_area": predicted_area,
                "overlap_pixels": overlap,
                "matched_target_pixel_recall": pixel_recall,
                "matched_component_area_ratio": area_ratio,
                "centroid_error": centroid_error,
            }
        )
    match_count = len(matches)
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "prediction_threshold_rule": "probability>0.5",
        "target_threshold_rule": "target>0.5",
        "component_connectivity": COMPONENT_CONNECTIVITY,
        "component_neighborhood": COMPONENT_NEIGHBORHOOD,
        "assignment_algorithm": "Hungarian/scipy.optimize.linear_sum_assignment",
        "match_rule": "centroid_distance<3",
        "target_component_count": len(targets),
        "predicted_component_count": len(predicted),
        "matched_component_count": match_count,
        "matched_overlap_pixel_count": overlap_sum,
        "matched_target_pixel_count": matched_target_area_sum,
        "matched_target_pixel_recall": (
            None
            if matched_target_area_sum == 0
            else overlap_sum / matched_target_area_sum
        ),
        "matched_component_area_ratio": (
            None if match_count == 0 else area_ratio_sum / match_count
        ),
        "centroid_error": (
            None if match_count == 0 else centroid_error_sum / match_count
        ),
        "matches": matches,
    }


class MatchedTargetDiagnostics:
    """Additive dataset-level form of :func:`matched_target_diagnostics`."""

    def __init__(self) -> None:
        self.image_count = 0
        self.target_component_count = 0
        self.predicted_component_count = 0
        self.matched_component_count = 0
        self.matched_overlap_pixel_count = 0
        self.matched_target_pixel_count = 0
        self.area_ratio_sum = 0.0
        self.centroid_error_sum = 0.0

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        result = matched_target_diagnostics(probability, target)
        self.image_count += 1
        self.target_component_count += result["target_component_count"]
        self.predicted_component_count += result["predicted_component_count"]
        self.matched_component_count += result["matched_component_count"]
        self.matched_overlap_pixel_count += result["matched_overlap_pixel_count"]
        self.matched_target_pixel_count += result["matched_target_pixel_count"]
        self.area_ratio_sum += sum(
            match["matched_component_area_ratio"] for match in result["matches"]
        )
        self.centroid_error_sum += sum(
            match["centroid_error"] for match in result["matches"]
        )

    def compute(self) -> dict[str, Any]:
        return {
            "schema": f"{DIAGNOSTIC_SCHEMA}/aggregate",
            "prediction_threshold_rule": "probability>0.5",
            "target_threshold_rule": "target>0.5",
            "component_connectivity": COMPONENT_CONNECTIVITY,
            "component_neighborhood": COMPONENT_NEIGHBORHOOD,
            "assignment_algorithm": "Hungarian/scipy.optimize.linear_sum_assignment",
            "match_rule": "centroid_distance<3",
            "image_count": self.image_count,
            "target_component_count": self.target_component_count,
            "predicted_component_count": self.predicted_component_count,
            "matched_component_count": self.matched_component_count,
            "matched_overlap_pixel_count": self.matched_overlap_pixel_count,
            "matched_target_pixel_count": self.matched_target_pixel_count,
            "matched_target_pixel_recall": (
                None
                if self.matched_target_pixel_count == 0
                else self.matched_overlap_pixel_count
                / self.matched_target_pixel_count
            ),
            "matched_component_area_ratio": (
                None
                if self.matched_component_count == 0
                else self.area_ratio_sum / self.matched_component_count
            ),
            "centroid_error": (
                None
                if self.matched_component_count == 0
                else self.centroid_error_sum / self.matched_component_count
            ),
        }


__all__ = [
    "AUGMENTATION_VERSION",
    "COMPLETE_TARGET_MARGIN",
    "COMPLETE_TARGET_PROBABILITY",
    "COMPONENT_CONNECTIVITY",
    "CROP_SIZE",
    "CROP_AUDIT_STATE_SCHEMA",
    "FORMAL_NUM_WORKERS",
    "CompleteTargetCropError",
    "CompleteTargetTransformPlan",
    "CropAuditAccumulator",
    "EviSIRSTCompleteTargetTrainDataset",
    "MatchedTargetDiagnostics",
    "apply_crop_and_augment",
    "derive_complete_target_transform_plan",
    "matched_target_diagnostics",
    "policy_identity",
    "policy_identity_sha256",
    "stateless_augmentation_seed",
]
