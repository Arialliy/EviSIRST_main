"""Test-safe ledger contract for fixed EviSIRST head diagnostics.

This protocol layer records exactly one already-fixed head evaluation.  It
does not load a checkpoint, run a model, read labels, compare candidates, or
offer a ``best``/search API.  Model execution remains in
``model.evisirst_head_diagnostics``.

The ``head_spec`` argument accepts that module's frozen ``FixedHeadSpec``
structurally, or its exact JSON projection with ``head``, ``alpha``, and
``probability_epsilon`` fields.  Avoiding a runtime model import lets ledger
validation run in lightweight audit environments without PyTorch.

Checkpoint identity uses a normalized repository-relative artifact path plus
its SHA-256.  Absolute host paths and parent traversal are rejected so ledger
files do not leak workstation layout or become non-portable.

Two and only two roles exist:

``historical_test_diagnostic``
    Allows direct ``out``, direct ``d0``, or exactly ``logit_blend(alpha=0.5)``.
    The ledger always sets ``diagnostic_only=true`` and
    ``selection_allowed=false``, regardless of the checkpoint's reported
    ``selection_is_optimistic`` value.

``val``
    Allows one caller-supplied fixed alpha and requires an immutable split
    manifest SHA-256 binding.  The caller may explicitly mark the resulting
    entry as eligible for a later validation-only selection step only when the
    checkpoint is not itself optimistic.  This module still performs no
    selection itself.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import PurePosixPath
from typing import Any, Protocol


LEDGER_SCHEMA = "evisirst_fixed_head_diagnostic_ledger/v1"
PROTOCOL_VERSION = "evisirst_head_diagnostic_role_contract/v1"
HISTORICAL_TEST_DIAGNOSTIC_ROLE = "historical_test_diagnostic"
VALIDATION_ROLE = "val"

_ROLES = frozenset((HISTORICAL_TEST_DIAGNOSTIC_ROLE, VALIDATION_ROLE))
_HEADS = frozenset(("out", "d0", "logit_blend"))
_CHECKPOINT_KEYS = frozenset(
    (
        "checkpoint_path",
        "checkpoint_sha256",
        "source_selection",
        "selection_is_optimistic",
    )
)
_HEAD_SPEC_KEYS = frozenset(("head", "alpha", "probability_epsilon"))
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RESULT_KEYS = frozenset(("dataset", "metrics", "prediction_artifact_sha256"))
_RESULT_METRIC_KEYS = frozenset(
    (
        "test_loss",
        "validation_loss",
        "mIoU",
        "miou",
        "nIoU",
        "niou",
        "F1",
        "f1",
        "pixel_f1",
        "pixel_precision",
        "pixel_recall",
        "Pd",
        "pd",
        "tiny_pd",
        "Fa",
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
)


class HeadDiagnosticProtocolError(ValueError):
    """A ledger request violates the fixed, role-separated protocol."""


class FixedHeadSpecLike(Protocol):
    """Structural subset of ``model.evisirst_head_diagnostics.FixedHeadSpec``."""

    head: str
    alpha: float | None
    probability_epsilon: float


class FrozenJSONDict(dict[str, Any]):
    """A JSON-serializable dict with mutation methods disabled."""

    __slots__ = ()

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("diagnostic ledger entries are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _require_exact_keys(
    value: Mapping[str, Any], required: frozenset[str], *, context: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise HeadDiagnosticProtocolError(f"{context} keys must be strings")
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing:
        raise HeadDiagnosticProtocolError(
            f"{context} missing required provenance keys: {', '.join(missing)}"
        )
    if unexpected:
        raise HeadDiagnosticProtocolError(
            f"{context} has unexpected keys: {', '.join(unexpected)}"
        )


def _strict_json_clone(value: Any, *, context: str) -> Any:
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
        raise HeadDiagnosticProtocolError(
            f"{context} must be strict JSON and contain no NaN/Infinity"
        ) from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenJSONDict(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _normalize_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HeadDiagnosticProtocolError(
            f"{field} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _normalize_checkpoint_path(value: Any) -> str:
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise HeadDiagnosticProtocolError(
            "checkpoint_path must be a path string"
        ) from exc
    if (
        not isinstance(path, str)
        or not path
        or "\x00" in path
        or "\n" in path
        or "\\" in path
    ):
        raise HeadDiagnosticProtocolError(
            "checkpoint_path must be a portable single-line POSIX path"
        )
    pure_path = PurePosixPath(path)
    normalized = pure_path.as_posix()
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or not pure_path.parts
        or ":" in pure_path.parts[0]
        or pure_path.parts[0] == "~"
        or normalized in ("", ".")
        or normalized != path
    ):
        raise HeadDiagnosticProtocolError(
            "checkpoint_path must be a normalized repository-relative artifact path"
        )
    return normalized


def _normalize_checkpoint_provenance(
    checkpoint_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(checkpoint_provenance, Mapping):
        raise HeadDiagnosticProtocolError(
            "checkpoint_provenance must be a mapping"
        )
    _require_exact_keys(
        checkpoint_provenance,
        _CHECKPOINT_KEYS,
        context="checkpoint_provenance",
    )
    source_selection = checkpoint_provenance["source_selection"]
    if (
        not isinstance(source_selection, str)
        or not source_selection
        or source_selection.strip() != source_selection
    ):
        raise HeadDiagnosticProtocolError(
            "source_selection must be a non-empty trimmed string"
        )
    selection_is_optimistic = checkpoint_provenance[
        "selection_is_optimistic"
    ]
    if not isinstance(selection_is_optimistic, bool):
        raise HeadDiagnosticProtocolError(
            "selection_is_optimistic must be an explicit JSON boolean"
        )
    return {
        "checkpoint_path": _normalize_checkpoint_path(
            checkpoint_provenance["checkpoint_path"]
        ),
        "checkpoint_sha256": _normalize_sha256(
            checkpoint_provenance["checkpoint_sha256"],
            field="checkpoint_sha256",
        ),
        "source_selection": source_selection,
        # Preserve the supplied truth value; never rewrite historical
        # provenance merely to make it agree with the role policy.
        "selection_is_optimistic": selection_is_optimistic,
    }


def _head_spec_fields(head_spec: Any) -> dict[str, Any]:
    if isinstance(head_spec, Mapping):
        _require_exact_keys(head_spec, _HEAD_SPEC_KEYS, context="head_spec")
        values = dict(head_spec)
    else:
        missing = [
            field for field in _HEAD_SPEC_KEYS if not hasattr(head_spec, field)
        ]
        if missing:
            raise HeadDiagnosticProtocolError(
                "head_spec must be a FixedHeadSpec-compatible object; missing "
                + ", ".join(sorted(missing))
            )
        values = {
            field: getattr(head_spec, field) for field in _HEAD_SPEC_KEYS
        }

    head = values["head"]
    if head not in _HEADS:
        raise HeadDiagnosticProtocolError(
            f"head must be one of {sorted(_HEADS)}, got {head!r}"
        )
    epsilon = values["probability_epsilon"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise HeadDiagnosticProtocolError(
            "probability_epsilon must be a real scalar"
        )
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise HeadDiagnosticProtocolError(
            "probability_epsilon must be finite and strictly between 0 and 0.5"
        )

    alpha = values["alpha"]
    if head == "logit_blend":
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise HeadDiagnosticProtocolError(
                "logit_blend requires one fixed real alpha"
            )
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise HeadDiagnosticProtocolError(
                "logit_blend alpha must be finite and in [0, 1]"
            )
    elif alpha is not None:
        raise HeadDiagnosticProtocolError(
            f"alpha is not applicable to the direct {head!r} head"
        )

    return {
        "head": head,
        "alpha": alpha,
        "probability_epsilon": epsilon,
        "utility_contract": "model.evisirst_head_diagnostics.FixedHeadSpec",
    }


def _normalize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise HeadDiagnosticProtocolError(
            "result must be a mapping for one fixed head"
        )
    _require_exact_keys(result, _RESULT_KEYS, context="result")
    dataset = result["dataset"]
    if (
        not isinstance(dataset, str)
        or not dataset
        or dataset.strip() != dataset
        or len(dataset) > 256
        or not dataset.isprintable()
    ):
        raise HeadDiagnosticProtocolError(
            "result dataset must be a non-empty portable string"
        )
    metrics = result["metrics"]
    if not isinstance(metrics, Mapping) or not metrics:
        raise HeadDiagnosticProtocolError(
            "result metrics must be a non-empty summary mapping"
        )
    if any(not isinstance(key, str) for key in metrics):
        raise HeadDiagnosticProtocolError("result metric keys must be strings")
    unexpected_metrics = sorted(set(metrics).difference(_RESULT_METRIC_KEYS))
    if unexpected_metrics:
        raise HeadDiagnosticProtocolError(
            "result metrics contain undeclared keys: "
            + ", ".join(str(key) for key in unexpected_metrics)
        )
    normalized_metrics: dict[str, int | float | None] = {}
    for key, value in metrics.items():
        if value is None:
            normalized_metrics[key] = None
        elif isinstance(value, bool) or not isinstance(value, Real):
            raise HeadDiagnosticProtocolError(
                f"result metric {key!r} must be a finite scalar or None"
            )
        elif isinstance(value, int):
            normalized_metrics[key] = value
        else:
            normalized = float(value)
            if not math.isfinite(normalized):
                raise HeadDiagnosticProtocolError(
                    f"result metric {key!r} must be finite"
                )
            normalized_metrics[key] = normalized
    return {
        "dataset": dataset,
        "metrics": normalized_metrics,
        "prediction_artifact_sha256": _normalize_sha256(
            result["prediction_artifact_sha256"],
            field="prediction_artifact_sha256",
        ),
    }


def build_diagnostic_ledger_entry(
    *,
    role: str,
    head_spec: FixedHeadSpecLike | Mapping[str, Any],
    checkpoint_provenance: Mapping[str, Any],
    result: Mapping[str, Any],
    split_manifest_sha256: str | None = None,
    validation_selection_allowed: bool = False,
    validation_checkpoint_selected_on_same_split: bool = False,
) -> FrozenJSONDict:
    """Build one immutable strict-JSON ledger entry for a fixed head.

    ``validation_selection_allowed`` and
    ``validation_checkpoint_selected_on_same_split`` are meaningful only for
    ``role="val"``.  The latter is a one-way safety gate: a checkpoint whose
    epoch was already chosen or retained using this same validation split may
    be described with additional fixed heads, but those results are forced to
    ``diagnostic_only=true`` and can never enter another selection step.  The
    function accepts no labels, metric objective, candidate collection,
    threshold grid, or search configuration.
    """

    if role not in _ROLES:
        raise HeadDiagnosticProtocolError(
            f"role must be one of {sorted(_ROLES)}; generic test is unsupported"
        )
    if not isinstance(validation_selection_allowed, bool):
        raise HeadDiagnosticProtocolError(
            "validation_selection_allowed must be an explicit boolean"
        )
    if not isinstance(validation_checkpoint_selected_on_same_split, bool):
        raise HeadDiagnosticProtocolError(
            "validation_checkpoint_selected_on_same_split must be an "
            "explicit boolean"
        )

    checkpoint = _normalize_checkpoint_provenance(checkpoint_provenance)
    fixed_head = _head_spec_fields(head_spec)
    result_payload = _normalize_result(result)

    if role == HISTORICAL_TEST_DIAGNOSTIC_ROLE:
        if validation_checkpoint_selected_on_same_split:
            raise HeadDiagnosticProtocolError(
                "validation_checkpoint_selected_on_same_split is valid only "
                "for role='val'"
            )
        if validation_selection_allowed:
            raise HeadDiagnosticProtocolError(
                "historical test diagnostics can never allow selection"
            )
        if split_manifest_sha256 is not None:
            raise HeadDiagnosticProtocolError(
                "historical test diagnostics do not accept a validation split hash"
            )
        if (
            fixed_head["head"] == "logit_blend"
            and fixed_head["alpha"] != 0.5
        ):
            raise HeadDiagnosticProtocolError(
                "historical test logit_blend requires alpha exactly 0.5"
            )
        diagnostic_only = True
        selection_allowed = False
        manifest_hash = None
        optimistic_disclosure = (
            "reported_optimistic"
            if checkpoint["selection_is_optimistic"]
            else "reported_not_optimistic_value_preserved;"
            "historical_test_role_still_forbids_selection"
        )
    else:
        if (
            validation_checkpoint_selected_on_same_split
            and validation_selection_allowed
        ):
            raise HeadDiagnosticProtocolError(
                "a checkpoint already selected on this validation split "
                "cannot allow secondary head selection"
            )
        if (
            validation_selection_allowed
            and checkpoint["selection_is_optimistic"]
        ):
            raise HeadDiagnosticProtocolError(
                "validation selection cannot reuse an optimistic checkpoint"
            )
        manifest_hash = _normalize_sha256(
            split_manifest_sha256,
            field="split_manifest_sha256",
        )
        if checkpoint["selection_is_optimistic"]:
            diagnostic_only = True
            selection_allowed = False
            optimistic_disclosure = (
                "reported_optimistic_validation_contaminated;"
                "diagnostic_only_no_unbiased_evidence"
            )
        elif validation_checkpoint_selected_on_same_split:
            diagnostic_only = True
            selection_allowed = False
            optimistic_disclosure = (
                "reported_not_optimistic;"
                "same_validation_split_reused_after_epoch_selection;"
                "diagnostic_only_no_secondary_head_selection"
            )
        else:
            diagnostic_only = False
            selection_allowed = validation_selection_allowed
            optimistic_disclosure = "reported_not_optimistic"

    payload = {
        "schema": LEDGER_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "role": role,
        "diagnostic_only": diagnostic_only,
        "selection_allowed": selection_allowed,
        "checkpoint_provenance": checkpoint,
        "checkpoint_optimistic_disclosure": optimistic_disclosure,
        "split_manifest_sha256": manifest_hash,
        "fixed_head_spec": fixed_head,
        "result": result_payload,
        "protocol_assertions": {
            "single_fixed_head_only": True,
            "alpha_search_performed_by_protocol": False,
            "candidate_comparison_performed_by_protocol": False,
            "labels_read_by_protocol": False,
            "best_api_available": False,
            "validation_checkpoint_selected_on_same_split": (
                validation_checkpoint_selected_on_same_split
            ),
        },
    }
    strict_payload = _strict_json_clone(payload, context="ledger entry")
    frozen = _freeze_json(strict_payload)
    assert isinstance(frozen, FrozenJSONDict)
    return frozen


__all__ = [
    "LEDGER_SCHEMA",
    "PROTOCOL_VERSION",
    "HISTORICAL_TEST_DIAGNOSTIC_ROLE",
    "VALIDATION_ROLE",
    "HeadDiagnosticProtocolError",
    "FixedHeadSpecLike",
    "FrozenJSONDict",
    "build_diagnostic_ledger_entry",
]
