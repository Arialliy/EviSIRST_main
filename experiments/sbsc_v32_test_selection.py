"""Frozen test-selected dual-role checkpoint selection for SBSC V3.2.

The formal ``img_idx/test`` protocol evaluates every epoch from 500 through
1000 inclusive.  Because those test measurements select checkpoints, every
record and every returned payload explicitly discloses that the result is
optimistic and cannot support an unbiased-test claim.

There is no tolerance window.  The descending lexicographic keys are exactly::

    best_mIoU = (mIoU, Pd, -Fa, nIoU, tinyPd, -loss, -epoch)
    best_Pd   = (Pd, -Fa, tinyPd, mIoU, nIoU, -loss, -epoch)

This module performs no checkpoint I/O.  If both roles select the same epoch,
the caller still owns materialization of two distinct physical weight files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any


RECORD_SCHEMA = "sctransnet_sbsc_v32_test_evaluation_record/v1"
SELECTION_SCHEMA = "sctransnet_sbsc_v32_test_selected_dual_role/v1"
SCHEDULE_SCHEMA = "sctransnet_sbsc_v32_test_schedule/v1"
RANKING_SCHEMA = "sctransnet_sbsc_v32_test_selected_ranking/v1"

_FROZEN_TEST_BEGIN_EPOCH = 500
_FROZEN_TOTAL_EPOCHS = 1000
_FROZEN_VALID_ROLES = ("best_mIoU", "best_Pd")
_FROZEN_PRIMARY_ROLE = "best_mIoU"
_FROZEN_SECONDARY_ROLE = "best_Pd"
_FROZEN_RULE_VERSION = "evisirst_zero_margin_dual_role_lexicographic/v1"

# Public descriptive aliases.  Formal checks below use private literals so an
# imported alias cannot be monkeypatched to weaken the frozen protocol.
TEST_BEGIN_EPOCH = _FROZEN_TEST_BEGIN_EPOCH
TOTAL_EPOCHS = _FROZEN_TOTAL_EPOCHS
VALID_ROLES = _FROZEN_VALID_ROLES
PRIMARY_ROLE = _FROZEN_PRIMARY_ROLE
SECONDARY_ROLE = _FROZEN_SECONDARY_ROLE
RULE_VERSION = _FROZEN_RULE_VERSION

_IDENTITY_FIELDS = (
    "method",
    "dataset",
    "architecture_seed",
    "run_seed",
    "split_manifest_sha256",
    "run_identity_sha256",
)
_METHODS = frozenset(("sbsc_v32",))
_DATASETS = frozenset(("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"))
_DISCLOSURES = {
    "data_role": "test",
    "test_split_accessed": True,
    "test_selected": True,
    "selection_is_optimistic": True,
    "unbiased_test_claim_supported": False,
}
_METRICS = ("mIoU", "nIoU", "Pd", "Fa", "tinyPd", "loss")
_NESTED_METRICS = {
    "mIoU": "miou",
    "nIoU": "niou",
    "Pd": "pd",
    "Fa": "fa",
    "tinyPd": "tiny_pd",
    "loss": "test_loss",
}
_UNIT_INTERVAL_METRICS = frozenset(("mIoU", "nIoU", "Pd", "tinyPd"))
_NONNEGATIVE_METRICS = frozenset(("Fa", "loss"))
_ROLE_RANK_ORDERS = {
    _FROZEN_PRIMARY_ROLE: (
        "mIoU:max",
        "Pd:max",
        "Fa:min",
        "nIoU:max",
        "tinyPd:max",
        "loss:min",
        "epoch:min",
    ),
    _FROZEN_SECONDARY_ROLE: (
        "Pd:max",
        "Fa:min",
        "tinyPd:max",
        "mIoU:max",
        "nIoU:max",
        "loss:min",
        "epoch:min",
    ),
}


class SBSCV32TestSelectionError(ValueError):
    """A history violates the frozen test-selected contract."""


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SBSCV32TestSelectionError(
            "selection metadata must be finite strict JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SBSCV32TestSelectionError(f"{label} must be lowercase SHA-256")
    return value


def _require_completed_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SBSCV32TestSelectionError("completed_epoch must be an integer")
    completed = int(value)
    if not 0 <= completed <= _FROZEN_TOTAL_EPOCHS:
        raise SBSCV32TestSelectionError(
            f"completed_epoch must be in [0, {_FROZEN_TOTAL_EPOCHS}]"
        )
    return completed


def expected_test_epochs(completed_epoch: int) -> tuple[int, ...]:
    """Return the exact committed test cadence for one training prefix."""

    completed = _require_completed_epoch(completed_epoch)
    if completed < _FROZEN_TEST_BEGIN_EPOCH:
        return ()
    return tuple(range(_FROZEN_TEST_BEGIN_EPOCH, completed + 1))


def _validated_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_IDENTITY_FIELDS):
        raise SBSCV32TestSelectionError(
            "expected_identity must contain the exact frozen identity fields"
        )
    identity = {field: value[field] for field in _IDENTITY_FIELDS}
    if identity["method"] not in _METHODS:
        raise SBSCV32TestSelectionError(
            "expected_identity method must be exactly 'sbsc_v32'"
        )
    if identity["dataset"] not in _DATASETS:
        raise SBSCV32TestSelectionError("expected_identity dataset differs")
    for field in ("architecture_seed", "run_seed"):
        if type(identity[field]) is not int or identity[field] != 42:
            raise SBSCV32TestSelectionError(f"expected_identity {field} must be 42")
    for field in ("split_manifest_sha256", "run_identity_sha256"):
        identity[field] = _sha256_text(identity[field], label=field)
    return identity


def _materialize(records: Any) -> list[Mapping[str, Any]]:
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(
        records, Sequence
    ):
        raise SBSCV32TestSelectionError(
            "test_history must be a sequence of mappings"
        )
    materialized = list(records)
    if any(not isinstance(record, Mapping) for record in materialized):
        raise SBSCV32TestSelectionError(
            "every test history record must be a mapping"
        )
    return materialized


def _metric(value: Any, *, name: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SBSCV32TestSelectionError(f"{context} {name} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SBSCV32TestSelectionError(f"{context} {name} must be finite")
    if name in _UNIT_INTERVAL_METRICS and not 0.0 <= normalized <= 1.0:
        raise SBSCV32TestSelectionError(
            f"{context} {name} must be a raw proportion in [0, 1]"
        )
    if name in _NONNEGATIVE_METRICS and normalized < 0.0:
        raise SBSCV32TestSelectionError(
            f"{context} {name} must be non-negative"
        )
    return normalized


def _optional_tiny_metric(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    return _metric(value, name="tinyPd", context=context)


def _validated_metrics(
    record: Mapping[str, Any], *, context: str
) -> dict[str, float | None]:
    nested = record.get("metrics")
    if nested is not None and not isinstance(nested, Mapping):
        raise SBSCV32TestSelectionError(
            f"{context} metrics must be a mapping when supplied"
        )
    normalized: dict[str, float | None] = {}
    for name in _METRICS:
        if name not in record:
            raise SBSCV32TestSelectionError(
                f"{context} missing required top-level metric {name!r}"
            )
        value = (
            _optional_tiny_metric(record[name], context=context)
            if name == "tinyPd"
            else _metric(record[name], name=name, context=context)
        )
        alias = _NESTED_METRICS[name]
        if isinstance(nested, Mapping) and alias in nested:
            nested_value = (
                _optional_tiny_metric(
                    nested[alias], context=f"{context}.metrics"
                )
                if name == "tinyPd"
                else _metric(
                    nested[alias], name=name, context=f"{context}.metrics"
                )
            )
            if nested_value != value:
                raise SBSCV32TestSelectionError(
                    f"{context} conflicting {name} and metrics[{alias!r}] values"
                )
        normalized[name] = value
    return normalized


def _rank_key(record: Mapping[str, Any], role: str) -> tuple[float, ...]:
    epoch = float(record["epoch"])
    tiny_pd = (
        float("-inf")
        if record["tinyPd"] is None
        else float(record["tinyPd"])
    )
    if role == _FROZEN_PRIMARY_ROLE:
        return (
            float(record["mIoU"]),
            float(record["Pd"]),
            -float(record["Fa"]),
            float(record["nIoU"]),
            tiny_pd,
            -float(record["loss"]),
            -epoch,
        )
    if role == _FROZEN_SECONDARY_ROLE:
        return (
            float(record["Pd"]),
            -float(record["Fa"]),
            tiny_pd,
            float(record["mIoU"]),
            float(record["nIoU"]),
            -float(record["loss"]),
            -epoch,
        )
    raise SBSCV32TestSelectionError(f"unsupported role: {role!r}")


def _validated_records(
    records: Any,
    *,
    completed_epoch: int,
    expected_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    materialized = _materialize(records)
    identity = _validated_identity(expected_identity)
    expected = expected_test_epochs(completed_epoch)
    validated: list[dict[str, Any]] = []
    observed: list[int] = []
    for position, record in enumerate(materialized):
        context = f"record[{position}]"
        if record.get("schema") != RECORD_SCHEMA:
            raise SBSCV32TestSelectionError(
                f"{context} schema must be exactly {RECORD_SCHEMA!r}"
            )
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, Integral):
            raise SBSCV32TestSelectionError(f"{context} epoch must be an integer")
        epoch = int(epoch)
        if not _FROZEN_TEST_BEGIN_EPOCH <= epoch <= _FROZEN_TOTAL_EPOCHS:
            raise SBSCV32TestSelectionError(
                f"{context} epoch must be in "
                f"[{_FROZEN_TEST_BEGIN_EPOCH}, {_FROZEN_TOTAL_EPOCHS}]"
            )
        observed.append(epoch)
        for field, expected_value in _DISCLOSURES.items():
            observed_value = record.get(field)
            differs = (
                observed_value != expected_value
                if field == "data_role"
                else observed_value is not expected_value
            )
            if differs:
                raise SBSCV32TestSelectionError(
                    f"{context} {field} must be explicit {expected_value!r}"
                )
        for field, expected_value in identity.items():
            if record.get(field) != expected_value:
                raise SBSCV32TestSelectionError(
                    f"{context} {field} differs from expected identity"
                )
        state_hash = _sha256_text(
            record.get("model_state_sha256"),
            label=f"{context} model_state_sha256",
        )
        metrics = _validated_metrics(record, context=context)
        validated.append(
            {
                "schema": RECORD_SCHEMA,
                "epoch": epoch,
                **_DISCLOSURES,
                **metrics,
                "model_state_sha256": state_hash,
            }
        )
    if tuple(observed) != expected:
        raise SBSCV32TestSelectionError(
            "test history cadence differs: "
            f"expected {expected[:2]}...{expected[-2:] if expected else ()} "
            f"({len(expected)} records), got {tuple(observed[:2])}..."
            f"{tuple(observed[-2:]) if observed else ()} "
            f"({len(observed)} records)"
        )
    return validated


def _selected(record: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    rank_key = _rank_key(record, role)
    return {
        "epoch": int(record["epoch"]),
        "mIoU": float(record["mIoU"]),
        "nIoU": float(record["nIoU"]),
        "Pd": float(record["Pd"]),
        "Fa": float(record["Fa"]),
        "tinyPd": (
            None if record["tinyPd"] is None else float(record["tinyPd"])
        ),
        "loss": float(record["loss"]),
        "descending_rank_key": [
            value if math.isfinite(value) else None for value in rank_key
        ],
        "model_state_sha256": str(record["model_state_sha256"]),
    }


def _empty_payload(
    completed_epoch: int, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    identity = _validated_identity(expected_identity)
    return {
        "schema": SELECTION_SCHEMA,
        "schedule_schema": SCHEDULE_SCHEMA,
        "rule_version": _FROZEN_RULE_VERSION,
        **_DISCLOSURES,
        "completed_epoch": completed_epoch,
        "test_begin_epoch": _FROZEN_TEST_BEGIN_EPOCH,
        "total_epochs": _FROZEN_TOTAL_EPOCHS,
        "test_record_count": 0,
        "selection_identity": identity,
        "selection_identity_sha256": _canonical_sha256(identity),
        "roles": {},
        "retention_frontier_epochs": [],
        "distinct_physical_weights_required": True,
        "physical_weight_materialization_owner": "runner",
    }


def select_prefix(
    test_history: Any,
    *,
    completed_epoch: int,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and rank the exact test-selected prefix through one epoch."""

    completed = _require_completed_epoch(completed_epoch)
    records = _validated_records(
        test_history,
        completed_epoch=completed,
        expected_identity=expected_identity,
    )
    if not records:
        return _empty_payload(completed, expected_identity=expected_identity)

    identity = _validated_identity(expected_identity)
    winners = {
        role: max(records, key=lambda record, role=role: _rank_key(record, role))
        for role in _FROZEN_VALID_ROLES
    }
    roles = {
        role: {
            "role": role,
            "is_primary_reporting_role": role == _FROZEN_PRIMARY_ROLE,
            "rank_order": list(_ROLE_RANK_ORDERS[role]),
            "selected": _selected(winners[role], role=role),
            "decision": (
                "strict descending lexicographic maximum over every evaluated "
                "test record; no tolerance or candidate window"
            ),
        }
        for role in _FROZEN_VALID_ROLES
    }
    frontier = sorted(
        {
            int(roles[role]["selected"]["epoch"])
            for role in _FROZEN_VALID_ROLES
        }
    )
    evaluated_records = []
    for record in records:
        metrics = {
            name: (
                None
                if name == "tinyPd" and record[name] is None
                else float(record[name])
            )
            for name in _METRICS
        }
        evaluated_records.append(
            {
                "epoch": int(record["epoch"]),
                "data_role": "test",
                **metrics,
                "model_state_sha256": str(record["model_state_sha256"]),
            }
        )
    ranking = {
        "schema": RANKING_SCHEMA,
        "rule_version": _FROZEN_RULE_VERSION,
        "selection_kind": "dual_role_test_selected_checkpoint",
        **_DISCLOSURES,
        "window_applied": False,
        "selection_margin_raw": None,
        "candidate_tolerance_raw": 0.0,
        "primary_role": _FROZEN_PRIMARY_ROLE,
        "primary_selected_epoch": int(
            roles[_FROZEN_PRIMARY_ROLE]["selected"]["epoch"]
        ),
        "roles": roles,
        "evaluated_epochs": [int(record["epoch"]) for record in records],
        "evaluated_records": evaluated_records,
    }
    return {
        "schema": SELECTION_SCHEMA,
        "schedule_schema": SCHEDULE_SCHEMA,
        "rule_version": _FROZEN_RULE_VERSION,
        **_DISCLOSURES,
        "completed_epoch": completed,
        "test_begin_epoch": _FROZEN_TEST_BEGIN_EPOCH,
        "total_epochs": _FROZEN_TOTAL_EPOCHS,
        "test_record_count": len(records),
        "selection_identity": identity,
        "selection_identity_sha256": _canonical_sha256(identity),
        "test_history_sha256": _canonical_sha256(list(test_history)),
        "model_state_sha256_by_epoch": {
            str(record["epoch"]): str(record["model_state_sha256"])
            for record in records
        },
        "roles": roles,
        "retention_frontier_epochs": frontier,
        "ranking_provenance": ranking,
        "distinct_physical_weights_required": True,
        "physical_weight_materialization_owner": "runner",
    }


def select_final(
    test_history: Any, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Require all 501 test records and return both final role winners."""

    payload = select_prefix(
        test_history,
        completed_epoch=_FROZEN_TOTAL_EPOCHS,
        expected_identity=expected_identity,
    )
    expected_count = _FROZEN_TOTAL_EPOCHS - _FROZEN_TEST_BEGIN_EPOCH + 1
    if payload["test_record_count"] != expected_count:
        raise SBSCV32TestSelectionError("final test history is incomplete")
    if set(payload["roles"]) != set(_FROZEN_VALID_ROLES):
        raise SBSCV32TestSelectionError("final dual-role selection is incomplete")
    return payload


def retention_frontier_epochs(
    test_history: Any,
    *,
    completed_epoch: int,
    expected_identity: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the union of the current winners' checkpoint epochs."""

    payload = select_prefix(
        test_history,
        completed_epoch=completed_epoch,
        expected_identity=expected_identity,
    )
    return tuple(payload["retention_frontier_epochs"])


__all__ = [
    "PRIMARY_ROLE",
    "RANKING_SCHEMA",
    "RECORD_SCHEMA",
    "RULE_VERSION",
    "SBSCV32TestSelectionError",
    "SECONDARY_ROLE",
    "SELECTION_SCHEMA",
    "SCHEDULE_SCHEMA",
    "TEST_BEGIN_EPOCH",
    "TOTAL_EPOCHS",
    "VALID_ROLES",
    "expected_test_epochs",
    "retention_frontier_epochs",
    "select_final",
    "select_prefix",
]
