"""Frozen validation-only dual-role selection for SCTransNet/SBSC V3.2.

The formal schedule evaluates the immutable validation split at every epoch
from 500 through 1000 inclusive.  This module deliberately has no model,
dataset, checkpoint, or test-set dependency.  It validates both an in-flight
prefix and the complete 501-record history before delegating the actual
zero-margin lexicographic ranking to the audited shared selector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from numbers import Integral
from typing import Any

from experiments import evisirst_zero_margin_selection as _ranking


SELECTION_SCHEMA = "sctransnet_sbsc_v32_validation_dual_role/v1"
SCHEDULE_SCHEMA = "sctransnet_sbsc_v32_validation_schedule/v1"
_FROZEN_VALIDATION_BEGIN_EPOCH = 500
_FROZEN_TOTAL_EPOCHS = 1000
_FROZEN_VALID_ROLES = ("best_mIoU", "best_Pd")
_FROZEN_PRIMARY_ROLE = "best_mIoU"
_FROZEN_SECONDARY_ROLE = "best_Pd"
_FROZEN_RULE_VERSION = "evisirst_zero_margin_dual_role_lexicographic/v1"

# Descriptive public aliases.  Executable checks below use private literals so
# mutating an imported public name cannot weaken the formal gate.
VALIDATION_BEGIN_EPOCH = _FROZEN_VALIDATION_BEGIN_EPOCH
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
_METHODS = frozenset(("sctransnet", "sbsc_v32"))
_DATASETS = frozenset(("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"))


class SBSCV32SelectionError(ValueError):
    """A history violates the frozen validation-selection contract."""


def _require_completed_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SBSCV32SelectionError("completed_epoch must be an integer")
    completed = int(value)
    if not 0 <= completed <= _FROZEN_TOTAL_EPOCHS:
        raise SBSCV32SelectionError(
            f"completed_epoch must be in [0, {_FROZEN_TOTAL_EPOCHS}]"
        )
    return completed


def expected_validation_epochs(completed_epoch: int) -> tuple[int, ...]:
    """Return the exact committed validation cadence for one prefix."""

    completed = _require_completed_epoch(completed_epoch)
    if completed < _FROZEN_VALIDATION_BEGIN_EPOCH:
        return ()
    return tuple(range(_FROZEN_VALIDATION_BEGIN_EPOCH, completed + 1))


def _materialize(records: Any) -> list[Mapping[str, Any]]:
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(
        records, Sequence
    ):
        raise SBSCV32SelectionError(
            "validation_history must be a sequence of mappings"
        )
    materialized = list(records)
    if any(not isinstance(record, Mapping) for record in materialized):
        raise SBSCV32SelectionError(
            "every validation history record must be a mapping"
        )
    return materialized


def _sha256_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SBSCV32SelectionError(f"{label} must be lowercase SHA-256")
    return value


def _validated_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_IDENTITY_FIELDS):
        raise SBSCV32SelectionError(
            "expected_identity must contain the exact frozen identity fields"
        )
    identity = {field: value[field] for field in _IDENTITY_FIELDS}
    if identity["method"] not in _METHODS:
        raise SBSCV32SelectionError("expected_identity method differs")
    if identity["dataset"] not in _DATASETS:
        raise SBSCV32SelectionError("expected_identity dataset differs")
    for field in ("architecture_seed", "run_seed"):
        if type(identity[field]) is not int or identity[field] != 42:
            raise SBSCV32SelectionError(f"expected_identity {field} must be 42")
    for field in ("split_manifest_sha256", "run_identity_sha256"):
        identity[field] = _sha256_text(identity[field], label=field)
    return identity


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_records(
    records: Any,
    *,
    completed_epoch: int,
    expected_identity: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    materialized = _materialize(records)
    identity = _validated_identity(expected_identity)
    expected = expected_validation_epochs(completed_epoch)
    observed: list[int] = []
    for position, record in enumerate(materialized):
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, Integral):
            raise SBSCV32SelectionError(
                f"record[{position}] epoch must be an integer"
            )
        observed.append(int(epoch))
        if record.get("data_role") != "val":
            raise SBSCV32SelectionError(
                f"record[{position}] data_role must be exactly 'val'"
            )
        # Absence is not evidence of non-access.  The formal runner must make
        # this disclosure explicit on every selector input record so that the
        # selector cannot silently turn an unknown provenance state into a
        # negative test-access claim in its output payload.
        if record.get("test_split_accessed") is not False:
            raise SBSCV32SelectionError(
                f"record[{position}] test_split_accessed must be explicit false"
            )
        for field, expected_value in identity.items():
            if record.get(field) != expected_value:
                raise SBSCV32SelectionError(
                    f"record[{position}] {field} differs from expected identity"
                )
        _sha256_text(
            record.get("model_state_sha256"),
            label=f"record[{position}] model_state_sha256",
        )
    if tuple(observed) != expected:
        raise SBSCV32SelectionError(
            "validation history cadence differs: "
            f"expected {expected[:2]}...{expected[-2:] if expected else ()} "
            f"({len(expected)} records), got {tuple(observed[:2])}..."
            f"{tuple(observed[-2:]) if observed else ()} "
            f"({len(observed)} records)"
        )
    if materialized:
        try:
            _ranking.select_checkpoints(materialized)
        except _ranking.EviSIRSTZeroMarginSelectionError as exc:
            raise SBSCV32SelectionError(str(exc)) from exc
    return materialized


def _empty_payload(
    completed_epoch: int, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    identity = _validated_identity(expected_identity)
    return {
        "schema": SELECTION_SCHEMA,
        "schedule_schema": SCHEDULE_SCHEMA,
        "rule_version": _FROZEN_RULE_VERSION,
        "data_role": "val",
        "completed_epoch": completed_epoch,
        "validation_begin_epoch": _FROZEN_VALIDATION_BEGIN_EPOCH,
        "total_epochs": _FROZEN_TOTAL_EPOCHS,
        "validation_record_count": 0,
        "selection_identity": identity,
        "selection_identity_sha256": _canonical_sha256(identity),
        "roles": {},
        "retention_frontier_epochs": [],
        "test_split_accessed": False,
    }


def select_prefix(
    validation_history: Any,
    *,
    completed_epoch: int,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and rank exactly the committed prefix through one epoch."""

    completed = _require_completed_epoch(completed_epoch)
    records = _validated_records(
        validation_history,
        completed_epoch=completed,
        expected_identity=expected_identity,
    )
    if not records:
        return _empty_payload(
            completed, expected_identity=expected_identity
        )
    identity = _validated_identity(expected_identity)
    provenance = _ranking.select_checkpoints(records)
    if (
        provenance.get("rule_version") != _FROZEN_RULE_VERSION
        or set(provenance.get("roles", {})) != set(_FROZEN_VALID_ROLES)
    ):
        raise SBSCV32SelectionError(
            "shared ranking returned a non-frozen role contract"
        )
    state_hashes = {
        int(record["epoch"]): str(record["model_state_sha256"])
        for record in records
    }
    roles: dict[str, dict[str, Any]] = {}
    for role in _FROZEN_VALID_ROLES:
        role_payload = dict(provenance["roles"][role])
        selected = dict(role_payload["selected"])
        selected_epoch = int(selected["epoch"])
        selected["model_state_sha256"] = state_hashes[selected_epoch]
        role_payload["selected"] = selected
        roles[role] = role_payload
    frontier = sorted(
        {
            int(roles[role]["selected"]["epoch"])
            for role in _FROZEN_VALID_ROLES
        }
    )
    return {
        "schema": SELECTION_SCHEMA,
        "schedule_schema": SCHEDULE_SCHEMA,
        "rule_version": _FROZEN_RULE_VERSION,
        "data_role": "val",
        "completed_epoch": completed,
        "validation_begin_epoch": _FROZEN_VALIDATION_BEGIN_EPOCH,
        "total_epochs": _FROZEN_TOTAL_EPOCHS,
        "validation_record_count": len(records),
        "selection_identity": identity,
        "selection_identity_sha256": _canonical_sha256(identity),
        "validation_history_sha256": _canonical_sha256(records),
        "model_state_sha256_by_epoch": {
            str(epoch): state_hashes[epoch] for epoch in sorted(state_hashes)
        },
        "roles": roles,
        "retention_frontier_epochs": frontier,
        "ranking_provenance": provenance,
        "test_split_accessed": False,
    }


def select_final(
    validation_history: Any, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Require all 501 records and return the two final role winners."""

    payload = select_prefix(
        validation_history,
        completed_epoch=_FROZEN_TOTAL_EPOCHS,
        expected_identity=expected_identity,
    )
    expected_count = (
        _FROZEN_TOTAL_EPOCHS - _FROZEN_VALIDATION_BEGIN_EPOCH + 1
    )
    if payload["validation_record_count"] != expected_count:
        raise SBSCV32SelectionError("final validation history is incomplete")
    if set(payload["roles"]) != set(_FROZEN_VALID_ROLES):
        raise SBSCV32SelectionError("final dual-role selection is incomplete")
    return payload


def retention_frontier_epochs(
    validation_history: Any,
    *,
    completed_epoch: int,
    expected_identity: Mapping[str, Any],
) -> tuple[int, ...]:
    payload = select_prefix(
        validation_history,
        completed_epoch=completed_epoch,
        expected_identity=expected_identity,
    )
    return tuple(payload["retention_frontier_epochs"])


__all__ = [
    "PRIMARY_ROLE",
    "RULE_VERSION",
    "SBSCV32SelectionError",
    "SECONDARY_ROLE",
    "SELECTION_SCHEMA",
    "SCHEDULE_SCHEMA",
    "TOTAL_EPOCHS",
    "VALIDATION_BEGIN_EPOCH",
    "VALID_ROLES",
    "expected_validation_epochs",
    "retention_frontier_epochs",
    "select_final",
    "select_prefix",
]
