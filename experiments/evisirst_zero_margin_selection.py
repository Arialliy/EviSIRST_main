"""Strict validation-only checkpoint selection without an mIoU window.

The selector is intentionally independent of PyTorch, datasets, and checkpoint
I/O.  It consumes immutable validation records and emits complete,
JSON-serializable provenance from which both role winners can be recomputed.

HF Stage A has one publication role, ``best_mIoU``.  ``best_Pd`` is retained as
an explicitly secondary operating-point checkpoint; it can never silently
replace the publication checkpoint.

The two descending lexicographic keys are exactly::

    best_mIoU = (mIoU, Pd, -Fa, nIoU, tinyPd, -loss, -epoch)
    best_Pd   = (Pd, -Fa, tinyPd, mIoU, nIoU, -loss, -epoch)

There is no tolerance/candidate window.  The public ``margin`` argument exists
only to make accidental reuse of the historical 0.001 policy fail closed: only
``None`` and numeric zero are accepted, and provenance canonicalizes it to
``null`` with ``window_applied=false``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any


PROVENANCE_SCHEMA = "evisirst_zero_margin_validation_selection/v1"
RULE_VERSION = "evisirst_zero_margin_dual_role_lexicographic/v1"
PRIMARY_ROLE = "best_mIoU"
SECONDARY_ROLE = "best_Pd"
VALID_ROLES = (PRIMARY_ROLE, SECONDARY_ROLE)
SELECTION_MARGIN_RAW = None

_TOP_LEVEL_METRICS = {
    "mIoU": "mIoU",
    "nIoU": "nIoU",
    "Pd": "Pd",
    "Fa": "Fa",
    "tinyPd": "tinyPd",
    "loss": "loss",
}
_NESTED_METRICS = {
    "mIoU": "miou",
    "nIoU": "niou",
    "Pd": "pd",
    "Fa": "fa",
    "tinyPd": "tiny_pd",
    "loss": "validation_loss",
}
_UNIT_INTERVAL_METRICS = frozenset(("mIoU", "nIoU", "Pd", "tinyPd"))
_NONNEGATIVE_METRICS = frozenset(("Fa", "loss"))
_TEST_DISCLOSURE_FIELDS = frozenset(
    (
        "test_split_accessed",
        "test_index_opened",
        "test_selected",
        "test_selection_supported",
    )
)
_ROLE_RANK_ORDERS = {
    PRIMARY_ROLE: (
        "mIoU:max",
        "Pd:max",
        "Fa:min",
        "nIoU:max",
        "tinyPd:max",
        "loss:min",
        "epoch:min",
    ),
    SECONDARY_ROLE: (
        "Pd:max",
        "Fa:min",
        "tinyPd:max",
        "mIoU:max",
        "nIoU:max",
        "loss:min",
        "epoch:min",
    ),
}


class EviSIRSTZeroMarginSelectionError(ValueError):
    """An input violates the strict validation-selection contract."""


def _normalize_margin(margin: Any) -> None:
    if margin is None:
        return None
    if isinstance(margin, bool) or not isinstance(margin, Real):
        raise EviSIRSTZeroMarginSelectionError(
            "margin must be null/None or numeric zero; selection windows are forbidden"
        )
    numeric = float(margin)
    if not math.isfinite(numeric) or numeric != 0.0:
        raise EviSIRSTZeroMarginSelectionError(
            "margin must be null/None or numeric zero; selection windows are forbidden"
        )
    return None


def _materialize_records(records: Iterable[Mapping[str, Any]]) -> list[Any]:
    if isinstance(records, (str, bytes, Mapping)):
        raise EviSIRSTZeroMarginSelectionError(
            "records must be an iterable of validation-record mappings"
        )
    try:
        materialized = list(records)
    except TypeError as exc:
        raise EviSIRSTZeroMarginSelectionError(
            "records must be an iterable of validation-record mappings"
        ) from exc
    if not materialized:
        raise EviSIRSTZeroMarginSelectionError(
            "at least one validation record is required"
        )
    return materialized


def _normalize_epoch(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} epoch must be a positive integer"
        )
    epoch = int(value)
    if epoch < 1:
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} epoch must be a positive integer"
        )
    return epoch


def _normalize_metric(value: Any, *, name: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} {name} must be a real scalar"
        )
    metric = float(value)
    if not math.isfinite(metric):
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} {name} must be finite"
        )
    if name in _UNIT_INTERVAL_METRICS and not 0.0 <= metric <= 1.0:
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} {name} must be a raw proportion in [0, 1]"
        )
    if name in _NONNEGATIVE_METRICS and metric < 0.0:
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} {name} must be non-negative"
        )
    return metric


def _validate_test_disclosures(record: Mapping[str, Any], *, context: str) -> None:
    for field in sorted(_TEST_DISCLOSURE_FIELDS.intersection(record)):
        if record[field] is not False:
            raise EviSIRSTZeroMarginSelectionError(
                f"{context} {field} must be explicit false; test-split access "
                "and test selection are forbidden"
            )


def _extract_metric(
    record: Mapping[str, Any],
    nested: Mapping[str, Any] | None,
    *,
    name: str,
    context: str,
) -> tuple[float, str]:
    top_key = _TOP_LEVEL_METRICS[name]
    nested_key = _NESTED_METRICS[name]
    has_top = top_key in record
    has_nested = nested is not None and nested_key in nested
    if not has_top and not has_nested:
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} missing required metric {top_key!r}; accepted nested "
            f"alias is metrics[{nested_key!r}]"
        )

    top_value = (
        _normalize_metric(record[top_key], name=name, context=context)
        if has_top
        else None
    )
    nested_value = (
        _normalize_metric(
            nested[nested_key],
            name=name,
            context=f"{context}.metrics",
        )
        if has_nested and nested is not None
        else None
    )
    if has_top and has_nested and top_value != nested_value:
        raise EviSIRSTZeroMarginSelectionError(
            f"{context} conflicting top-level {top_key} and "
            f"metrics[{nested_key!r}] values"
        )
    if has_top and has_nested:
        assert top_value is not None
        return top_value, "top_level_and_metrics_verified_equal"
    if has_top:
        assert top_value is not None
        return top_value, "top_level"
    assert nested_value is not None
    return nested_value, "metrics"


def _normalize_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for position, raw_record in enumerate(_materialize_records(records)):
        context = f"record[{position}]"
        if not isinstance(raw_record, Mapping):
            raise EviSIRSTZeroMarginSelectionError(
                f"{context} must be a mapping"
            )
        if raw_record.get("data_role") != "val":
            raise EviSIRSTZeroMarginSelectionError(
                f"{context} data_role must be exactly 'val'; test selection is forbidden"
            )
        if "epoch" not in raw_record:
            raise EviSIRSTZeroMarginSelectionError(
                f"{context} missing required key: epoch"
            )
        _validate_test_disclosures(raw_record, context=context)
        epoch = _normalize_epoch(raw_record["epoch"], context=context)
        if epoch in seen_epochs:
            raise EviSIRSTZeroMarginSelectionError(
                f"duplicate epoch is forbidden: {epoch}"
            )
        seen_epochs.add(epoch)

        raw_nested = raw_record.get("metrics")
        if raw_nested is not None and not isinstance(raw_nested, Mapping):
            raise EviSIRSTZeroMarginSelectionError(
                f"{context} metrics must be a mapping when supplied"
            )
        nested = raw_nested if isinstance(raw_nested, Mapping) else None
        metrics: dict[str, float] = {}
        metric_sources: dict[str, str] = {}
        for name in _TOP_LEVEL_METRICS:
            value, source = _extract_metric(
                raw_record, nested, name=name, context=context
            )
            metrics[name] = value
            metric_sources[name] = source
        normalized.append(
            {
                "epoch": epoch,
                "data_role": "val",
                **metrics,
                "metric_sources": metric_sources,
            }
        )
    return sorted(normalized, key=lambda item: item["epoch"])


def _rank_key(record: Mapping[str, Any], role: str) -> tuple[float, ...]:
    epoch = float(record["epoch"])
    if role == PRIMARY_ROLE:
        return (
            float(record["mIoU"]),
            float(record["Pd"]),
            -float(record["Fa"]),
            float(record["nIoU"]),
            float(record["tinyPd"]),
            -float(record["loss"]),
            -epoch,
        )
    if role == SECONDARY_ROLE:
        return (
            float(record["Pd"]),
            -float(record["Fa"]),
            float(record["tinyPd"]),
            float(record["mIoU"]),
            float(record["nIoU"]),
            -float(record["loss"]),
            -epoch,
        )
    raise EviSIRSTZeroMarginSelectionError(f"unsupported role: {role!r}")


def _candidate(record: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    return {
        "epoch": int(record["epoch"]),
        "mIoU": float(record["mIoU"]),
        "nIoU": float(record["nIoU"]),
        "Pd": float(record["Pd"]),
        "Fa": float(record["Fa"]),
        "tinyPd": float(record["tinyPd"]),
        "loss": float(record["loss"]),
        "descending_rank_key": list(_rank_key(record, role)),
    }


def _normalize_primary_role(primary_role: Any) -> str:
    if primary_role != PRIMARY_ROLE:
        raise EviSIRSTZeroMarginSelectionError(
            "HF Stage A primary_role must be exactly 'best_mIoU'; best_Pd is "
            "a secondary operating-point checkpoint only"
        )
    return PRIMARY_ROLE


def select_checkpoints(
    records: Iterable[Mapping[str, Any]],
    *,
    primary_role: str = PRIMARY_ROLE,
    margin: float | None = None,
) -> dict[str, Any]:
    """Select both strict validation roles and return auditable provenance."""

    _normalize_margin(margin)
    _normalize_primary_role(primary_role)
    normalized = _normalize_records(records)
    winners = {
        role: max(normalized, key=lambda item, role=role: _rank_key(item, role))
        for role in VALID_ROLES
    }
    roles: dict[str, dict[str, Any]] = {}
    for role in VALID_ROLES:
        winner = winners[role]
        roles[role] = {
            "role": role,
            "is_primary_publication_role": role == PRIMARY_ROLE,
            "rank_order": list(_ROLE_RANK_ORDERS[role]),
            "selected": _candidate(winner, role=role),
            "decision": (
                "strict descending lexicographic maximum over every evaluated "
                "validation record; no candidate window"
            ),
        }

    primary_selected = dict(roles[PRIMARY_ROLE]["selected"])
    evaluated_records = [
        {
            "epoch": int(record["epoch"]),
            "data_role": "val",
            "mIoU": float(record["mIoU"]),
            "nIoU": float(record["nIoU"]),
            "Pd": float(record["Pd"]),
            "Fa": float(record["Fa"]),
            "tinyPd": float(record["tinyPd"]),
            "loss": float(record["loss"]),
            "metric_sources": dict(record["metric_sources"]),
        }
        for record in normalized
    ]
    return {
        "schema": PROVENANCE_SCHEMA,
        "rule_version": RULE_VERSION,
        "selection_kind": "dual_role_validation_checkpoint",
        "data_role": "val",
        "test_selection_supported": False,
        "test_split_accessed": False,
        "window_applied": False,
        "selection_margin_raw": SELECTION_MARGIN_RAW,
        "candidate_tolerance_raw": 0.0,
        "metric_scale": {
            "mIoU": "raw_proportion",
            "nIoU": "raw_proportion",
            "Pd": "raw_proportion",
            "tinyPd": "raw_proportion",
            "Fa": "caller_consistent_nonnegative_unit",
            "loss": "finite_nonnegative_scalar",
        },
        "primary_role": PRIMARY_ROLE,
        "primary_selected_epoch": int(primary_selected["epoch"]),
        "primary_selected": primary_selected,
        # Compatibility with consumers expecting an existing-style top-level
        # selected block.  It is deliberately bound to best_mIoU only.
        "selected": dict(primary_selected),
        "roles": roles,
        "evaluated_epochs": [int(record["epoch"]) for record in normalized],
        "evaluated_records": evaluated_records,
        "selection_reason": {
            "summary": (
                "best_mIoU is the HF Stage A publication checkpoint; best_Pd "
                "is retained only as a secondary operating point"
            ),
            "no_threshold_or_window": True,
        },
    }


def select_best_miou_checkpoint(
    records: Iterable[Mapping[str, Any]], *, margin: float | None = None
) -> dict[str, Any]:
    """Return provenance with top-level ``selected`` bound to best_mIoU."""

    return select_checkpoints(records, primary_role=PRIMARY_ROLE, margin=margin)


def select_best_pd_checkpoint(
    records: Iterable[Mapping[str, Any]], *, margin: float | None = None
) -> dict[str, Any]:
    """Return the explicitly secondary best-Pd role provenance."""

    provenance = select_checkpoints(
        records, primary_role=PRIMARY_ROLE, margin=margin
    )
    role = provenance["roles"][SECONDARY_ROLE]
    return {
        "schema": PROVENANCE_SCHEMA,
        "rule_version": RULE_VERSION,
        "selection_kind": "secondary_validation_checkpoint",
        "data_role": "val",
        "test_selection_supported": False,
        "test_split_accessed": False,
        "window_applied": False,
        "selection_margin_raw": SELECTION_MARGIN_RAW,
        "candidate_tolerance_raw": 0.0,
        "role": SECONDARY_ROLE,
        "is_primary_publication_role": False,
        "rank_order": list(role["rank_order"]),
        "selected": dict(role["selected"]),
        "evaluated_epochs": list(provenance["evaluated_epochs"]),
        "evaluated_records": list(provenance["evaluated_records"]),
    }


def _normalize_retain_roles(retain_roles: Iterable[str]) -> tuple[str, ...]:
    if isinstance(retain_roles, (str, bytes, Mapping)):
        raise EviSIRSTZeroMarginSelectionError(
            "retain_roles must be an iterable of distinct role names"
        )
    try:
        roles = tuple(retain_roles)
    except TypeError as exc:
        raise EviSIRSTZeroMarginSelectionError(
            "retain_roles must be an iterable of distinct role names"
        ) from exc
    if not roles:
        raise EviSIRSTZeroMarginSelectionError(
            "retain_roles must include the primary best_mIoU role"
        )
    if len(set(roles)) != len(roles):
        raise EviSIRSTZeroMarginSelectionError(
            "retain_roles must not contain duplicates"
        )
    unsupported = sorted(set(roles).difference(VALID_ROLES))
    if unsupported:
        raise EviSIRSTZeroMarginSelectionError(
            f"unsupported retention roles: {', '.join(unsupported)}"
        )
    if PRIMARY_ROLE not in roles:
        raise EviSIRSTZeroMarginSelectionError(
            "retain_roles must include the primary best_mIoU role"
        )
    return roles


def retention_frontier_epochs(
    records: Iterable[Mapping[str, Any]],
    *,
    retain_roles: Iterable[str] = VALID_ROLES,
    margin: float | None = None,
) -> tuple[int, ...]:
    """Return the minimal past checkpoint set needed by retained role winners.

    With immutable metrics and exact lexicographic maxima, every losing past
    checkpoint is permanently dominated by the current winner for that role.
    A future record can either lose or replace the winner, so retaining the
    union of current role winners is sufficient.  The primary best-mIoU role is
    mandatory; best-Pd may be omitted only by an explicit caller policy.
    """

    roles = _normalize_retain_roles(retain_roles)
    provenance = select_checkpoints(
        records, primary_role=PRIMARY_ROLE, margin=margin
    )
    return tuple(
        sorted(
            {
                int(provenance["roles"][role]["selected"]["epoch"])
                for role in roles
            }
        )
    )


__all__ = [
    "EviSIRSTZeroMarginSelectionError",
    "PRIMARY_ROLE",
    "PROVENANCE_SCHEMA",
    "RULE_VERSION",
    "SECONDARY_ROLE",
    "SELECTION_MARGIN_RAW",
    "VALID_ROLES",
    "retention_frontier_epochs",
    "select_best_miou_checkpoint",
    "select_best_pd_checkpoint",
    "select_checkpoints",
]
