"""Validation-only checkpoint and fixed-spec selection contracts for V2.

This module is deliberately independent of models, checkpoints, datasets, and
PyTorch.  It consumes already-computed per-epoch validation metrics and emits
JSON-serializable provenance.  Every input record must explicitly declare
``data_role="val"``; there is no test-selection entry point or role override.

Metric conventions
------------------
``mIoU`` and ``Pd`` are raw proportions in ``[0, 1]``.  Consequently, the
candidate window ``0.001`` means 0.10 percentage points.  ``Fa`` must be a
finite non-negative value and must use one consistent unit within a call.

For an independent model, checkpoint choice is:

1. find the best validation mIoU;
2. retain epochs no more than 0.001 below it;
3. minimize Fa, maximize Pd, then choose the earliest epoch.

For a joint model, per-domain validation metrics are aggregated with equal
domain weight.  Choice is: maximize worst-domain mIoU, apply the same 0.001
window, maximize macro mIoU, minimize macro Fa, maximize macro Pd, then use
the earliest epoch only as a deterministic final tie-break.

The fixed-spec registry stores head/calibration values only after the caller
marks them ``fixed_after_validation`` and supplies validation provenance.  It
never imports, accepts, or invokes a model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any


MIOU_CANDIDATE_TOLERANCE = 0.001
INDEPENDENT_RULE_VERSION = "evisirst_v2_independent_val_lexicographic/v1"
JOINT_RULE_VERSION = "evisirst_v2_joint_val_lexicographic/v1"
PROVENANCE_SCHEMA = "evisirst_v2_validation_selection/v1"
REGISTRY_SCHEMA = "evisirst_v2_fixed_validation_spec_registry/v1"
REGISTRY_RULE_VERSION = "evisirst_v2_fixed_after_validation_only/v1"
FIXED_SELECTION_STATUS = "fixed_after_validation"

_INDEPENDENT_REQUIRED_KEYS = frozenset(("epoch", "data_role", "mIoU", "Fa", "Pd"))
_DOMAIN_REQUIRED_KEYS = frozenset(("mIoU", "Fa", "Pd"))
_JOINT_REQUIRED_KEYS = frozenset(("epoch", "data_role", "domains"))
_SPEC_KINDS = frozenset(("head", "calibration"))


class EviSIRSTSelectionError(ValueError):
    """Input data violates the validation-only V2 selection contract."""


def _require_keys(
    value: Mapping[str, Any], required: frozenset[str], *, context: str
) -> None:
    missing = sorted(required.difference(value))
    if missing:
        raise EviSIRSTSelectionError(
            f"{context} missing required keys: {', '.join(missing)}"
        )


def _require_validation_role(value: Any, *, context: str) -> None:
    if value != "val":
        raise EviSIRSTSelectionError(
            f"{context} data_role must be exactly 'val'; test selection is forbidden"
        )


def _normalize_epoch(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EviSIRSTSelectionError(f"{context} epoch must be an integer")
    epoch = int(value)
    if epoch < 0:
        raise EviSIRSTSelectionError(f"{context} epoch must be non-negative")
    return epoch


def _normalize_metric(
    value: Any,
    *,
    name: str,
    context: str,
    unit_interval: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EviSIRSTSelectionError(f"{context} {name} must be a real scalar")
    metric = float(value)
    if not math.isfinite(metric):
        raise EviSIRSTSelectionError(f"{context} {name} must be finite")
    if unit_interval and not 0.0 <= metric <= 1.0:
        raise EviSIRSTSelectionError(
            f"{context} {name} must be a raw proportion in [0, 1]"
        )
    if not unit_interval and metric < 0.0:
        raise EviSIRSTSelectionError(f"{context} {name} must be non-negative")
    return metric


def _materialize_records(records: Iterable[Mapping[str, Any]]) -> list[Any]:
    if isinstance(records, (str, bytes, Mapping)):
        raise EviSIRSTSelectionError("records must be an iterable of epoch mappings")
    try:
        materialized = list(records)
    except TypeError as exc:
        raise EviSIRSTSelectionError(
            "records must be an iterable of epoch mappings"
        ) from exc
    if not materialized:
        raise EviSIRSTSelectionError("at least one validation epoch is required")
    return materialized


def _reject_duplicate_epoch(epoch: int, seen: set[int]) -> None:
    if epoch in seen:
        raise EviSIRSTSelectionError(f"duplicate epoch is forbidden: {epoch}")
    seen.add(epoch)


def _normalize_independent_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for position, record in enumerate(_materialize_records(records)):
        context = f"record[{position}]"
        if not isinstance(record, Mapping):
            raise EviSIRSTSelectionError(f"{context} must be a mapping")
        _require_keys(record, _INDEPENDENT_REQUIRED_KEYS, context=context)
        _require_validation_role(record["data_role"], context=context)
        epoch = _normalize_epoch(record["epoch"], context=context)
        _reject_duplicate_epoch(epoch, seen_epochs)
        normalized.append(
            {
                "epoch": epoch,
                "mIoU": _normalize_metric(
                    record["mIoU"],
                    name="mIoU",
                    context=context,
                    unit_interval=True,
                ),
                "Fa": _normalize_metric(
                    record["Fa"],
                    name="Fa",
                    context=context,
                    unit_interval=False,
                ),
                "Pd": _normalize_metric(
                    record["Pd"],
                    name="Pd",
                    context=context,
                    unit_interval=True,
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["epoch"])


def _normalize_joint_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    normalized: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    expected_domains: tuple[str, ...] | None = None

    for position, record in enumerate(_materialize_records(records)):
        context = f"record[{position}]"
        if not isinstance(record, Mapping):
            raise EviSIRSTSelectionError(f"{context} must be a mapping")
        _require_keys(record, _JOINT_REQUIRED_KEYS, context=context)
        _require_validation_role(record["data_role"], context=context)
        epoch = _normalize_epoch(record["epoch"], context=context)
        _reject_duplicate_epoch(epoch, seen_epochs)

        raw_domains = record["domains"]
        if not isinstance(raw_domains, Mapping) or not raw_domains:
            raise EviSIRSTSelectionError(
                f"{context} domains must be a non-empty mapping"
            )
        if any(not isinstance(name, str) or not name for name in raw_domains):
            raise EviSIRSTSelectionError(
                f"{context} domain names must be non-empty strings"
            )
        domain_names = tuple(sorted(raw_domains))
        if expected_domains is None:
            expected_domains = domain_names
        elif domain_names != expected_domains:
            raise EviSIRSTSelectionError(
                f"{context} domain set differs from earlier epochs: "
                f"expected {expected_domains}, got {domain_names}"
            )

        domain_metrics: dict[str, dict[str, float]] = {}
        for domain_name in domain_names:
            domain_context = f"{context}.domains[{domain_name!r}]"
            metrics = raw_domains[domain_name]
            if not isinstance(metrics, Mapping):
                raise EviSIRSTSelectionError(
                    f"{domain_context} must be a metric mapping"
                )
            _require_keys(metrics, _DOMAIN_REQUIRED_KEYS, context=domain_context)
            domain_metrics[domain_name] = {
                "mIoU": _normalize_metric(
                    metrics["mIoU"],
                    name="mIoU",
                    context=domain_context,
                    unit_interval=True,
                ),
                "Fa": _normalize_metric(
                    metrics["Fa"],
                    name="Fa",
                    context=domain_context,
                    unit_interval=False,
                ),
                "Pd": _normalize_metric(
                    metrics["Pd"],
                    name="Pd",
                    context=domain_context,
                    unit_interval=True,
                ),
            }

        domain_count = len(domain_names)
        miou_values = [domain_metrics[name]["mIoU"] for name in domain_names]
        fa_values = [domain_metrics[name]["Fa"] for name in domain_names]
        pd_values = [domain_metrics[name]["Pd"] for name in domain_names]
        worst_miou = min(miou_values)
        normalized.append(
            {
                "epoch": epoch,
                "domain_metrics": domain_metrics,
                "worst_domain_mIoU": worst_miou,
                "worst_domains": [
                    name
                    for name in domain_names
                    if domain_metrics[name]["mIoU"] == worst_miou
                ],
                "macro_mIoU": math.fsum(miou_values) / domain_count,
                "macro_Fa": math.fsum(fa_values) / domain_count,
                "macro_Pd": math.fsum(pd_values) / domain_count,
            }
        )

    assert expected_domains is not None
    return sorted(normalized, key=lambda item: item["epoch"]), expected_domains


def _trace_stage(
    *,
    stage: str,
    criterion: str,
    value: float | int,
    survivors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "criterion": criterion,
        "value": value,
        "surviving_epochs": [item["epoch"] for item in survivors],
    }


def _independent_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": item["epoch"],
        "mIoU": item["mIoU"],
        "Fa": item["Fa"],
        "Pd": item["Pd"],
        "post_window_rank_key": [item["Fa"], -item["Pd"], item["epoch"]],
    }


def select_independent_checkpoint(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one independent-model checkpoint from validation records only."""

    normalized = _normalize_independent_records(records)
    best_miou = max(item["mIoU"] for item in normalized)
    best_only = [item for item in normalized if item["mIoU"] == best_miou]
    candidate_floor = best_miou - MIOU_CANDIDATE_TOLERANCE
    candidates = [item for item in normalized if item["mIoU"] >= candidate_floor]

    min_fa = min(item["Fa"] for item in candidates)
    after_fa = [item for item in candidates if item["Fa"] == min_fa]
    max_pd = max(item["Pd"] for item in after_fa)
    after_pd = [item for item in after_fa if item["Pd"] == max_pd]
    selected_epoch = min(item["epoch"] for item in after_pd)
    after_epoch = [item for item in after_pd if item["epoch"] == selected_epoch]
    selected = after_epoch[0]

    decision_trace = [
        _trace_stage(
            stage="best_validation_mIoU",
            criterion="maximize mIoU",
            value=best_miou,
            survivors=best_only,
        ),
        _trace_stage(
            stage="mIoU_candidate_window",
            criterion="mIoU >= best_mIoU - 0.001 (raw proportion, inclusive)",
            value=candidate_floor,
            survivors=candidates,
        ),
        _trace_stage(
            stage="validation_Fa",
            criterion="minimize Fa within the mIoU window",
            value=min_fa,
            survivors=after_fa,
        ),
        _trace_stage(
            stage="validation_Pd",
            criterion="maximize Pd among Fa ties",
            value=max_pd,
            survivors=after_pd,
        ),
        _trace_stage(
            stage="epoch",
            criterion="choose earliest epoch among remaining exact ties",
            value=selected_epoch,
            survivors=after_epoch,
        ),
    ]

    return {
        "schema": PROVENANCE_SCHEMA,
        "rule_version": INDEPENDENT_RULE_VERSION,
        "selection_kind": "independent_checkpoint",
        "data_role": "val",
        "test_selection_supported": False,
        "metric_scale": {
            "mIoU": "raw_proportion",
            "Pd": "raw_proportion",
            "Fa": "caller_consistent_nonnegative_unit",
        },
        "candidate_tolerance_raw": MIOU_CANDIDATE_TOLERANCE,
        "evaluated_epochs": [item["epoch"] for item in normalized],
        # Keep the compact, complete selector input so downstream public
        # loaders can recompute the winner rather than trusting a self-declared
        # ``selected`` block.  These are selection scalars only, not labels or
        # prediction arrays.
        "evaluated_records": [
            {
                "epoch": item["epoch"],
                "data_role": "val",
                "mIoU": item["mIoU"],
                "Fa": item["Fa"],
                "Pd": item["Pd"],
            }
            for item in normalized
        ],
        "candidates": [_independent_candidate(item) for item in candidates],
        "selected": _independent_candidate(selected),
        "selection_reason": {
            "summary": (
                f"epoch {selected_epoch} survived the mIoU window and won "
                "the ordered Fa, Pd, epoch tie-breaks"
            ),
            "lexicographic_order": [
                "best_mIoU_then_raw_0.001_window",
                "Fa:min",
                "Pd:max",
                "epoch:min",
            ],
            "decision_trace": decision_trace,
        },
    }


def retention_frontier_epochs(
    history: Iterable[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Return epochs whose checkpoint can still win at some mIoU threshold.

    This function uses the same validation-only record contract as
    :func:`select_independent_checkpoint`.  For two distinct records ``A`` and
    ``B``, ``A`` dominates ``B`` when both conditions hold, with at least one
    strict inequality::

        A.mIoU >= B.mIoU
        (A.Fa, -A.Pd, A.epoch) <= (B.Fa, -B.Pd, B.epoch)

    Whenever ``B`` passes a threshold, ``A`` therefore also passes it and is
    no worse under every post-threshold selection key.  ``B`` can never be
    selected and its checkpoint is safe to delete.  Returned epochs are the
    non-dominated frontier in ascending order.

    The proof assumes immutable validation metrics and the declared
    ``Fa:min, Pd:max, epoch:min`` key.  Recompute the frontier if metrics or the
    selection rule change.  No checkpoint or filesystem operation occurs.
    """

    normalized = _normalize_independent_records(history)
    frontier: list[int] = []
    for candidate in normalized:
        candidate_key = (
            candidate["Fa"],
            -candidate["Pd"],
            candidate["epoch"],
        )
        dominated = False
        for challenger in normalized:
            if challenger["epoch"] == candidate["epoch"]:
                continue
            challenger_key = (
                challenger["Fa"],
                -challenger["Pd"],
                challenger["epoch"],
            )
            weakly_better_miou = challenger["mIoU"] >= candidate["mIoU"]
            weakly_better_lex_key = challenger_key <= candidate_key
            at_least_one_strict = (
                challenger["mIoU"] > candidate["mIoU"]
                or challenger_key < candidate_key
            )
            if (
                weakly_better_miou
                and weakly_better_lex_key
                and at_least_one_strict
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["epoch"])
    return tuple(frontier)


def _joint_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": item["epoch"],
        "worst_domain_mIoU": item["worst_domain_mIoU"],
        "worst_domains": list(item["worst_domains"]),
        "macro_mIoU": item["macro_mIoU"],
        "macro_Fa": item["macro_Fa"],
        "macro_Pd": item["macro_Pd"],
        "domain_metrics": {
            name: dict(metrics)
            for name, metrics in item["domain_metrics"].items()
        },
        "post_window_rank_key": [
            -item["macro_mIoU"],
            item["macro_Fa"],
            -item["macro_Pd"],
            item["epoch"],
        ],
    }


def select_joint_checkpoint(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one joint-model checkpoint from per-domain validation metrics."""

    normalized, domain_names = _normalize_joint_records(records)
    best_worst_miou = max(item["worst_domain_mIoU"] for item in normalized)
    best_only = [
        item for item in normalized if item["worst_domain_mIoU"] == best_worst_miou
    ]
    candidate_floor = best_worst_miou - MIOU_CANDIDATE_TOLERANCE
    candidates = [
        item for item in normalized if item["worst_domain_mIoU"] >= candidate_floor
    ]

    max_macro_miou = max(item["macro_mIoU"] for item in candidates)
    after_macro_miou = [
        item for item in candidates if item["macro_mIoU"] == max_macro_miou
    ]
    min_macro_fa = min(item["macro_Fa"] for item in after_macro_miou)
    after_macro_fa = [
        item for item in after_macro_miou if item["macro_Fa"] == min_macro_fa
    ]
    max_macro_pd = max(item["macro_Pd"] for item in after_macro_fa)
    after_macro_pd = [
        item for item in after_macro_fa if item["macro_Pd"] == max_macro_pd
    ]
    selected_epoch = min(item["epoch"] for item in after_macro_pd)
    after_epoch = [
        item for item in after_macro_pd if item["epoch"] == selected_epoch
    ]
    selected = after_epoch[0]

    decision_trace = [
        _trace_stage(
            stage="best_worst_domain_validation_mIoU",
            criterion="maximize minimum domain mIoU",
            value=best_worst_miou,
            survivors=best_only,
        ),
        _trace_stage(
            stage="worst_domain_mIoU_candidate_window",
            criterion=(
                "worst-domain mIoU >= best worst-domain mIoU - 0.001 "
                "(raw proportion, inclusive)"
            ),
            value=candidate_floor,
            survivors=candidates,
        ),
        _trace_stage(
            stage="macro_validation_mIoU",
            criterion="maximize equal-domain macro mIoU within the window",
            value=max_macro_miou,
            survivors=after_macro_miou,
        ),
        _trace_stage(
            stage="macro_validation_Fa",
            criterion="minimize equal-domain macro Fa among macro-mIoU ties",
            value=min_macro_fa,
            survivors=after_macro_fa,
        ),
        _trace_stage(
            stage="macro_validation_Pd",
            criterion="maximize equal-domain macro Pd among macro-Fa ties",
            value=max_macro_pd,
            survivors=after_macro_pd,
        ),
        _trace_stage(
            stage="epoch",
            criterion="deterministic earliest epoch among remaining exact ties",
            value=selected_epoch,
            survivors=after_epoch,
        ),
    ]

    return {
        "schema": PROVENANCE_SCHEMA,
        "rule_version": JOINT_RULE_VERSION,
        "selection_kind": "joint_checkpoint",
        "data_role": "val",
        "test_selection_supported": False,
        "domains": list(domain_names),
        "domain_weighting": "equal_domain_macro",
        "metric_scale": {
            "mIoU": "raw_proportion",
            "Pd": "raw_proportion",
            "Fa": "caller_consistent_nonnegative_unit",
        },
        "candidate_tolerance_raw": MIOU_CANDIDATE_TOLERANCE,
        "evaluated_epochs": [item["epoch"] for item in normalized],
        "candidates": [_joint_candidate(item) for item in candidates],
        "selected": _joint_candidate(selected),
        "selection_reason": {
            "summary": (
                f"epoch {selected_epoch} survived the worst-domain mIoU "
                "window and won the macro mIoU, macro Fa, macro Pd, epoch "
                "tie-breaks"
            ),
            "lexicographic_order": [
                "best_worst_domain_mIoU_then_raw_0.001_window",
                "macro_mIoU:max",
                "macro_Fa:min",
                "macro_Pd:max",
                "epoch:min_deterministic_final_tie_break",
            ],
            "decision_trace": decision_trace,
        },
    }


def _json_clone(value: Any, *, context: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise EviSIRSTSelectionError(
            f"{context} must be JSON-serializable and contain no NaN/Infinity"
        ) from exc


class ValidationSpecRegistry:
    """Write-once registry for head/calibration specs fixed on validation.

    Registration deep-copies JSON data, and duplicate registry keys are
    rejected.  There is intentionally no replacement/update operation: a new
    validation decision must receive a new registry key and provenance.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def register(
        self,
        *,
        registry_key: str,
        spec_kind: str,
        fixed_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Register one JSON spec already selected and frozen on validation."""

        if not isinstance(registry_key, str) or not registry_key.strip():
            raise EviSIRSTSelectionError(
                "registry_key must be a non-empty string"
            )
        if registry_key in self._entries:
            raise EviSIRSTSelectionError(
                f"duplicate registry_key is forbidden: {registry_key!r}"
            )
        if spec_kind not in _SPEC_KINDS:
            raise EviSIRSTSelectionError(
                f"spec_kind must be one of {sorted(_SPEC_KINDS)}"
            )
        if not isinstance(fixed_spec, Mapping):
            raise EviSIRSTSelectionError("fixed_spec must be a mapping")

        cloned = _json_clone(fixed_spec, context="fixed_spec")
        required = frozenset(
            (
                "data_role",
                "selection_status",
                "value",
                "selection_provenance",
            )
        )
        _require_keys(cloned, required, context="fixed_spec")
        _require_validation_role(cloned["data_role"], context="fixed_spec")
        if cloned["selection_status"] != FIXED_SELECTION_STATUS:
            raise EviSIRSTSelectionError(
                "fixed_spec selection_status must be "
                f"{FIXED_SELECTION_STATUS!r}"
            )
        if not isinstance(cloned["value"], dict) or not cloned["value"]:
            raise EviSIRSTSelectionError(
                "fixed_spec value must be a non-empty JSON object"
            )

        provenance = cloned["selection_provenance"]
        if not isinstance(provenance, dict):
            raise EviSIRSTSelectionError(
                "fixed_spec selection_provenance must be a JSON object"
            )
        _require_keys(
            provenance,
            frozenset(("data_role", "rule_version", "selection_reason")),
            context="fixed_spec selection_provenance",
        )
        _require_validation_role(
            provenance["data_role"],
            context="fixed_spec selection_provenance",
        )
        if not isinstance(provenance["rule_version"], str) or not provenance[
            "rule_version"
        ]:
            raise EviSIRSTSelectionError(
                "fixed_spec selection_provenance rule_version must be non-empty"
            )

        entry = {
            "registry_key": registry_key,
            "spec_kind": spec_kind,
            "data_role": "val",
            "selection_status": FIXED_SELECTION_STATUS,
            "value": cloned["value"],
            "selection_provenance": provenance,
        }
        self._entries[registry_key] = entry
        return _json_clone(entry, context="registry entry")

    def snapshot(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable registry provenance object."""

        snapshot = {
            "schema": REGISTRY_SCHEMA,
            "rule_version": REGISTRY_RULE_VERSION,
            "data_role": "val",
            "test_selection_supported": False,
            "entries": [
                self._entries[key] for key in sorted(self._entries)
            ],
        }
        return _json_clone(snapshot, context="registry snapshot")

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "MIOU_CANDIDATE_TOLERANCE",
    "INDEPENDENT_RULE_VERSION",
    "JOINT_RULE_VERSION",
    "PROVENANCE_SCHEMA",
    "REGISTRY_SCHEMA",
    "REGISTRY_RULE_VERSION",
    "FIXED_SELECTION_STATUS",
    "EviSIRSTSelectionError",
    "select_independent_checkpoint",
    "retention_frontier_epochs",
    "select_joint_checkpoint",
    "ValidationSpecRegistry",
]
