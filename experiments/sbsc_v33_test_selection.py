"""Fail-closed test-selected dual-role checkpoint selection for SBSC V3.3.

Formal V3.3 runs evaluate the original ``img_idx/test`` split at every epoch
from 500 through 1000 inclusive.  The resulting 501 records select two
separately reported weights with exact, zero-margin lexicographic rules::

    best_mIoU = (mIoU, Pd, -Fa, nIoU, tinyPd, -loss, -epoch)
    best_Pd   = (Pd, -Fa, tinyPd, mIoU, nIoU, -loss, -epoch)

The final ``-epoch`` component makes an exact tie resolve to the earliest
epoch.  This module performs no checkpoint I/O; the runner remains responsible
for publishing two distinct physical files even when both roles select the
same epoch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


RECORD_SCHEMA = "sctransnet_sbsc_v33_test_evaluation_record/v1"
SELECTION_SCHEMA = "sctransnet_sbsc_v33_test_selected_dual_role/v1"
SCHEDULE_SCHEMA = "sctransnet_sbsc_v33_test_schedule/v1"
RANKING_SCHEMA = "sctransnet_sbsc_v33_test_selected_ranking/v1"

_FROZEN_TEST_BEGIN_EPOCH = 500
_FROZEN_TOTAL_EPOCHS = 1000
_FROZEN_VALID_ROLES = ("best_mIoU", "best_Pd")
_FROZEN_PRIMARY_ROLE = "best_mIoU"
_FROZEN_SECONDARY_ROLE = "best_Pd"
_FROZEN_RULE_VERSION = "evisirst_v33_zero_margin_dual_role_lexicographic/v1"

# Public descriptive aliases cannot weaken the private literals used below.
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
    "evaluation_contract_sha256",
    "method_config_sha256",
    "gradient_authorization_sha256",
    "baseline_authority_manifest_sha256",
    "formal_launch_authorization_sha256",
    "training_source_manifest_sha256",
)
_IDENTITY_SHA_FIELDS = (
    "split_manifest_sha256",
    "run_identity_sha256",
    "evaluation_contract_sha256",
    "method_config_sha256",
    "gradient_authorization_sha256",
    "baseline_authority_manifest_sha256",
    "formal_launch_authorization_sha256",
    "training_source_manifest_sha256",
)
_METHODS = frozenset(("sbsc_v33_third", "sbsc_v33_ord", "sbsc_v33_half"))
_DATASETS = frozenset(("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"))
_MODEL_NAME = "SCTransNet-C3-SBSC-V3.3"
_EXPECTED_TEST_COUNTS = {
    "NUAA-SIRST": 214,
    "NUDT-SIRST": 664,
    "IRSTD-1K": 201,
}
_DISCLOSURES = {
    "data_role": "test",
    "test_split_accessed": True,
    "test_selected": True,
    "selection_is_optimistic": True,
    "unbiased_test_claim_supported": False,
}

# Exact output of the common evaluator.  V3.3 selection records must retain
# the complete vector because reporting is not limited to the ranking metric.
_EVALUATOR_METRICS = (
    "test_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
_PROBABILITY_METRICS = frozenset(
    (
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
    )
)
_NONNEGATIVE_METRICS = frozenset(
    ("test_loss", "fa", "false_objects_per_image")
)
_COUNT_METRICS = frozenset(
    (
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    )
)
_RANKING_ALIASES = {
    "mIoU": "miou",
    "nIoU": "niou",
    "Pd": "pd",
    "Fa": "fa",
    "tinyPd": "tiny_pd",
    "loss": "test_loss",
}
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


class SBSCV33TestSelectionError(ValueError):
    """A history violates the frozen V3.3 test-selection contract."""


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
        raise SBSCV33TestSelectionError(
            "selection metadata must be finite strict JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SBSCV33TestSelectionError(f"{label} must be lowercase SHA-256")
    return value


def _require_completed_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SBSCV33TestSelectionError("completed_epoch must be an integer")
    completed = int(value)
    if not 0 <= completed <= _FROZEN_TOTAL_EPOCHS:
        raise SBSCV33TestSelectionError(
            f"completed_epoch must be in [0, {_FROZEN_TOTAL_EPOCHS}]"
        )
    return completed


def expected_test_epochs(completed_epoch: int) -> tuple[int, ...]:
    """Return the exact committed test cadence for a training prefix."""

    completed = _require_completed_epoch(completed_epoch)
    if completed < _FROZEN_TEST_BEGIN_EPOCH:
        return ()
    return tuple(range(_FROZEN_TEST_BEGIN_EPOCH, completed + 1))


def _validated_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_IDENTITY_FIELDS):
        raise SBSCV33TestSelectionError(
            "expected_identity must contain the exact frozen V3.3 fields"
        )
    identity = {field: value[field] for field in _IDENTITY_FIELDS}
    if identity["method"] not in _METHODS:
        raise SBSCV33TestSelectionError(
            "expected_identity method must be exactly one of "
            "sbsc_v33_third/sbsc_v33_ord/sbsc_v33_half"
        )
    if identity["dataset"] not in _DATASETS:
        raise SBSCV33TestSelectionError("expected_identity dataset differs")
    for field in ("architecture_seed", "run_seed"):
        if type(identity[field]) is not int or identity[field] != 42:
            raise SBSCV33TestSelectionError(f"expected_identity {field} must be 42")
    for field in _IDENTITY_SHA_FIELDS:
        identity[field] = _sha256_text(identity[field], label=field)
    return identity


def _materialize(records: Any) -> list[Mapping[str, Any]]:
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(
        records, Sequence
    ):
        raise SBSCV33TestSelectionError(
            "test_history must be a sequence of mappings"
        )
    materialized = list(records)
    if any(not isinstance(record, Mapping) for record in materialized):
        raise SBSCV33TestSelectionError(
            "every test history record must be a mapping"
        )
    return materialized


def _real(value: Any, *, name: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SBSCV33TestSelectionError(f"{context} {name} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SBSCV33TestSelectionError(f"{context} {name} must be finite")
    return normalized


def _probability(value: Any, *, name: str, context: str) -> float:
    normalized = _real(value, name=name, context=context)
    if not 0.0 <= normalized <= 1.0:
        raise SBSCV33TestSelectionError(
            f"{context} {name} must be a raw proportion in [0, 1]"
        )
    return normalized


def _nonnegative(value: Any, *, name: str, context: str) -> float:
    normalized = _real(value, name=name, context=context)
    if normalized < 0.0:
        raise SBSCV33TestSelectionError(f"{context} {name} must be non-negative")
    return normalized


def _count(value: Any, *, name: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SBSCV33TestSelectionError(
            f"{context} {name} must be a non-negative integer"
        )
    normalized = int(value)
    if normalized < 0:
        raise SBSCV33TestSelectionError(
            f"{context} {name} must be a non-negative integer"
        )
    return normalized


def _one_evaluator_metric(value: Any, *, name: str, context: str) -> float | int | None:
    if name == "tiny_pd" and value is None:
        return None
    if name in _PROBABILITY_METRICS:
        return _probability(value, name=name, context=context)
    if name in _NONNEGATIVE_METRICS:
        return _nonnegative(value, name=name, context=context)
    if name in _COUNT_METRICS:
        return _count(value, name=name, context=context)
    raise AssertionError(f"unclassified evaluator metric: {name}")


def _validated_evaluator_metrics(
    record: Mapping[str, Any], *, context: str
) -> dict[str, float | int | None]:
    nested = record.get("metrics")
    if nested is not None and (
        not isinstance(nested, Mapping) or set(nested) != set(_EVALUATOR_METRICS)
    ):
        raise SBSCV33TestSelectionError(
            f"{context} metrics must contain the exact evaluator field set"
        )

    normalized: dict[str, float | int | None] = {}
    for name in _EVALUATOR_METRICS:
        if name not in record:
            raise SBSCV33TestSelectionError(
                f"{context} missing required evaluator metric {name!r}"
            )
        value = _one_evaluator_metric(record[name], name=name, context=context)
        if isinstance(nested, Mapping):
            nested_value = _one_evaluator_metric(
                nested[name], name=name, context=f"{context}.metrics"
            )
            if nested_value != value:
                raise SBSCV33TestSelectionError(
                    f"{context} conflicting top-level and nested {name} values"
                )
        normalized[name] = value

    if (
        normalized["matched_target_count"] > normalized["target_count"]
        or normalized["matched_tiny_target_count"]
        > normalized["tiny_target_count"]
        or normalized["unmatched_predicted_object_count"]
        > normalized["predicted_object_count"]
    ):
        raise SBSCV33TestSelectionError(f"{context} evaluator counts are inconsistent")
    if normalized["target_count"] <= 0:
        raise SBSCV33TestSelectionError(
            f"{context} target_count must be positive"
        )
    if (normalized["tiny_target_count"] == 0) is not (
        normalized["tiny_pd"] is None
    ):
        raise SBSCV33TestSelectionError(
            f"{context} tiny-Pd denominator/null contract differs"
        )
    expected_pd = (
        normalized["matched_target_count"] / normalized["target_count"]
    )
    if not math.isclose(
        normalized["pd"], expected_pd, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise SBSCV33TestSelectionError(
            f"{context} pd differs from evaluator counts"
        )
    if normalized["tiny_pd"] is not None:
        expected_tiny_pd = (
            normalized["matched_tiny_target_count"]
            / normalized["tiny_target_count"]
        )
        if not math.isclose(
            normalized["tiny_pd"],
            expected_tiny_pd,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SBSCV33TestSelectionError(
                f"{context} tiny_pd differs from evaluator counts"
            )
    precision = normalized["pixel_precision"]
    recall = normalized["pixel_recall"]
    expected_f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    if not math.isclose(
        normalized["pixel_f1"], expected_f1, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise SBSCV33TestSelectionError(
            f"{context} pixel_f1 is inconsistent with precision/recall"
        )

    tiny_undefined = record.get("tiny_pd_was_undefined")
    if type(tiny_undefined) is not bool or tiny_undefined is not (
        normalized["tiny_pd"] is None
    ):
        raise SBSCV33TestSelectionError(
            f"{context} tiny_pd_was_undefined disclosure differs"
        )

    for alias, metric_name in _RANKING_ALIASES.items():
        if alias not in record:
            raise SBSCV33TestSelectionError(
                f"{context} missing required ranking alias {alias!r}"
            )
        alias_value = _one_evaluator_metric(
            record[alias], name=metric_name, context=context
        )
        if alias_value != normalized[metric_name]:
            raise SBSCV33TestSelectionError(
                f"{context} conflicting {alias} and {metric_name} values"
            )
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
    raise SBSCV33TestSelectionError(f"unsupported role: {role!r}")


def _validated_records(
    records: Any,
    *,
    completed_epoch: int,
    expected_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    materialized = _materialize(records)
    identity = _validated_identity(expected_identity)
    expected = expected_test_epochs(completed_epoch)
    validated: list[dict[str, Any]] = []
    observed: list[int] = []
    required_fields = {
        "schema",
        "epoch",
        "model",
        "seed",
        "sample_count",
        "evaluation_source_sha256",
        "model_state_sha256",
        "tiny_pd_was_undefined",
        *_DISCLOSURES,
        *_IDENTITY_FIELDS,
        *_EVALUATOR_METRICS,
        *_RANKING_ALIASES,
    }
    allowed_fields = required_fields | {"metrics"}

    for position, record in enumerate(materialized):
        context = f"record[{position}]"
        missing_fields = required_fields - set(record)
        unexpected_fields = set(record) - allowed_fields
        if missing_fields:
            raise SBSCV33TestSelectionError(
                f"{context} missing required fields: {sorted(missing_fields)}"
            )
        if unexpected_fields:
            raise SBSCV33TestSelectionError(
                f"{context} has unexpected fields: {sorted(unexpected_fields)}"
            )
        if record.get("schema") != RECORD_SCHEMA:
            raise SBSCV33TestSelectionError(
                f"{context} schema must be exactly {RECORD_SCHEMA!r}"
            )
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, Integral):
            raise SBSCV33TestSelectionError(f"{context} epoch must be an integer")
        epoch = int(epoch)
        if not _FROZEN_TEST_BEGIN_EPOCH <= epoch <= _FROZEN_TOTAL_EPOCHS:
            raise SBSCV33TestSelectionError(
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
                raise SBSCV33TestSelectionError(
                    f"{context} {field} must be explicit {expected_value!r}"
                )
        for field, expected_value in identity.items():
            if record.get(field) != expected_value:
                raise SBSCV33TestSelectionError(
                    f"{context} {field} differs from expected identity"
                )

        if record.get("model") != _MODEL_NAME:
            raise SBSCV33TestSelectionError(f"{context} model identity differs")
        if type(record.get("seed")) is not int or record["seed"] != 42:
            raise SBSCV33TestSelectionError(f"{context} seed must be 42")
        if (
            type(record.get("sample_count")) is not int
            or record["sample_count"] != _EXPECTED_TEST_COUNTS[identity["dataset"]]
        ):
            raise SBSCV33TestSelectionError(
                f"{context} sample_count differs from the full test split"
            )
        evaluation_source_sha256 = _sha256_text(
            record.get("evaluation_source_sha256"),
            label=f"{context} evaluation_source_sha256",
        )

        metrics = _validated_evaluator_metrics(record, context=context)
        state_hash = _sha256_text(
            record.get("model_state_sha256"),
            label=f"{context} model_state_sha256",
        )
        validated.append(
            {
                "schema": RECORD_SCHEMA,
                "epoch": epoch,
                **_DISCLOSURES,
                **identity,
                "model": _MODEL_NAME,
                "seed": 42,
                "sample_count": record["sample_count"],
                "evaluation_source_sha256": evaluation_source_sha256,
                **metrics,
                **{
                    alias: metrics[name]
                    for alias, name in _RANKING_ALIASES.items()
                },
                "tiny_pd_was_undefined": metrics["tiny_pd"] is None,
                "model_state_sha256": state_hash,
            }
        )

    if tuple(observed) != expected:
        raise SBSCV33TestSelectionError(
            "test history cadence differs: "
            f"expected {expected[:2]}...{expected[-2:] if expected else ()} "
            f"({len(expected)} records), got {tuple(observed[:2])}..."
            f"{tuple(observed[-2:]) if observed else ()} "
            f"({len(observed)} records)"
        )
    return validated, _canonical_sha256(materialized)


def _selected(record: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    rank_key = _rank_key(record, role)
    metrics = {name: record[name] for name in _EVALUATOR_METRICS}
    return {
        "epoch": int(record["epoch"]),
        "model": str(record["model"]),
        "seed": int(record["seed"]),
        "sample_count": int(record["sample_count"]),
        "evaluation_source_sha256": str(record["evaluation_source_sha256"]),
        "mIoU": float(record["mIoU"]),
        "nIoU": float(record["nIoU"]),
        "Pd": float(record["Pd"]),
        "Fa": float(record["Fa"]),
        "tinyPd": (
            None if record["tinyPd"] is None else float(record["tinyPd"])
        ),
        "loss": float(record["loss"]),
        "metrics": metrics,
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
    records, history_sha256 = _validated_records(
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
            "separate_weight_reporting_required": True,
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
        {int(roles[role]["selected"]["epoch"]) for role in _FROZEN_VALID_ROLES}
    )
    evaluated_records = [
        {
            "epoch": int(record["epoch"]),
            "data_role": "test",
            "model": str(record["model"]),
            "seed": int(record["seed"]),
            "sample_count": int(record["sample_count"]),
            "evaluation_source_sha256": str(
                record["evaluation_source_sha256"]
            ),
            "metrics": {
                name: record[name] for name in _EVALUATOR_METRICS
            },
            **{
                alias: record[alias] for alias in _RANKING_ALIASES
            },
            "model_state_sha256": str(record["model_state_sha256"]),
        }
        for record in records
    ]
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
        "test_history_sha256": history_sha256,
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
        raise SBSCV33TestSelectionError("final test history is incomplete")
    if set(payload["roles"]) != set(_FROZEN_VALID_ROLES):
        raise SBSCV33TestSelectionError("final dual-role selection is incomplete")
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
    "SBSCV33TestSelectionError",
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
