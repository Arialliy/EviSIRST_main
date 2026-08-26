#!/usr/bin/env python3
"""SCTransNet C3-SBSC V3.1 core and exact paired-construction contract.

C3-SBSC V3.1 changes exactly one operator: zero-based SCTB layer 1 (the second
SCTB) channel
attention.  The remaining three operators, encoder, decoder, and deep
supervision graph stay as the original SCTransNet implementation.  Its only
learned extension is one V3.1-specific four-level non-negative gain vector. At the
exact zero initialization the complete six-output network is bitwise equal to
the paired SCTransNet constructed from the same architecture seed.

This module intentionally contains no dataset, evaluator, checkpoint
selection, runner, or test-split dependency.
"""

from __future__ import annotations

import copy
import math
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from experiments.four_dataset_models_seed42_v1 import (
    _construct_original,
    state_dict_sha256,
)
from model.SCTransNet import Attention_org, SCTransNet


SBSC_V31_SCHEMA = (
    "sctransnet_sbsc_v31/loo_tri_support_dual_risk_projection/v1"
)

# Executable validators use private literals rather than mutable public
# aliases.  In particular, V3.1 has a distinct method and state key from V2,
# so a V2 checkpoint cannot be silently interpreted with V3.1 semantics.
_FROZEN_ARCHITECTURE_SEED = 42
_FROZEN_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
_FROZEN_METHODS = ("sctransnet", "sbsc_v31")
_FROZEN_BLOCK_COUNT = 4
_FROZEN_REPLACED_BLOCK_INDEX = 1
_FROZEN_BASELINE_STATE_KEY_COUNT = 510
_FROZEN_BASELINE_PARAMETER_COUNT = 11_325_939
_FROZEN_CANDIDATE_STATE_KEY_COUNT = 511
_FROZEN_CANDIDATE_PARAMETER_COUNT = 11_325_943
_FROZEN_GAIN_MIN = 0.0
_FROZEN_GAIN_MAX = 0.25
_FROZEN_EPS = 1e-6
_FROZEN_RISK_VARIANCE_MIN = 1e-8
_FROZEN_RISK_SCALE_MIN = 1e-4
_FROZEN_LAMBDA_GRID = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_FROZEN_LAMBDA_MAX = 32.0
_FROZEN_SINGLETON_ITERATIONS = 8
_FROZEN_DUAL_ITERATIONS = 6
_FROZEN_DUAL_ALPHAS = (1.0, 0.5, 0.25, 0.125)
_FROZEN_PROJECTED_GRADIENT_STEP = 0.5
_FROZEN_EIGENVALUE_FLOOR = 1e-6
_FROZEN_CORRELATION_GUARD = 0.999
_FROZEN_RISK_ROOT_GUARD = 5e-7
_FROZEN_RISK_TOLERANCE = 1e-6
_FROZEN_KKT_TOLERANCE = 1e-6
_FROZEN_STATIONARITY_TOLERANCE = 5e-6
_FROZEN_LAMBDA_ON = 1e-5
_FROZEN_OBJECTIVE_TOLERANCE = 1e-6
_FROZEN_GAIN_SUFFIX = "raw_dual_risk_level_gain"
_FROZEN_GAIN_STATE_KEY = (
    "mtc.encoder.layer.1.channel_attn." + _FROZEN_GAIN_SUFFIX
)

ARCHITECTURE_SEED = _FROZEN_ARCHITECTURE_SEED
SUPPORTED_DATASETS = _FROZEN_DATASETS
SUPPORTED_METHODS = _FROZEN_METHODS
EXPECTED_SBSC_V31_BLOCKS = 1
EXPECTED_SBSC_V31_STATE_KEY_COUNT = _FROZEN_CANDIDATE_STATE_KEY_COUNT
EXPECTED_SBSC_V31_PARAMETER_COUNT = _FROZEN_CANDIDATE_PARAMETER_COUNT
SBSC_V31_GAIN_MIN = _FROZEN_GAIN_MIN
SBSC_V31_GAIN_MAX = _FROZEN_GAIN_MAX
SBSC_V31_EPS = _FROZEN_EPS
SBSC_V31_GAIN_SUFFIX = _FROZEN_GAIN_SUFFIX
SBSC_V31_GAIN_STATE_KEY = _FROZEN_GAIN_STATE_KEY

_FROZEN_SOLVER_CONSTANTS = (
    ("simplex_tolerance", _FROZEN_EPS),
    ("risk_direction_variance_min", _FROZEN_RISK_VARIANCE_MIN),
    ("risk_scale_min", _FROZEN_RISK_SCALE_MIN),
    ("lambda_grid", _FROZEN_LAMBDA_GRID),
    ("lambda_max", _FROZEN_LAMBDA_MAX),
    ("singleton_iterations", float(_FROZEN_SINGLETON_ITERATIONS)),
    ("dual_iterations", float(_FROZEN_DUAL_ITERATIONS)),
    ("dual_alphas", _FROZEN_DUAL_ALPHAS),
    ("projected_gradient_step", _FROZEN_PROJECTED_GRADIENT_STEP),
    ("eigenvalue_floor", _FROZEN_EIGENVALUE_FLOOR),
    ("correlation_guard", _FROZEN_CORRELATION_GUARD),
    ("risk_root_guard", _FROZEN_RISK_ROOT_GUARD),
    ("risk_tolerance", _FROZEN_RISK_TOLERANCE),
    ("projected_kkt_tolerance", _FROZEN_KKT_TOLERANCE),
    ("stationarity_tolerance", _FROZEN_STATIONARITY_TOLERANCE),
    ("lambda_on", _FROZEN_LAMBDA_ON),
    ("objective_tolerance", _FROZEN_OBJECTIVE_TOLERANCE),
)

_AUTHORITY_STATE_SCHEMA: tuple[
    tuple[str, tuple[int, ...], torch.dtype], ...
] | None = None
_AUTHORITY_STATE_SCHEMA_LOCK = threading.Lock()
_AUTHORITY_NONREPLACEMENT_STRUCTURE: tuple[tuple[Any, ...], ...] | None = None
_AUTHORITY_NONREPLACEMENT_STRUCTURE_LOCK = threading.Lock()

_FROZEN_DIAGNOSTIC_INTERVENTIONS = (
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


@dataclass(frozen=True)
class C3V31LevelSupport:
    """One level's detached leave-one-level-out tri-support evidence."""

    level_index: int
    peer_indices: tuple[int, int, int]
    self_validation: torch.Tensor
    peer_consensus: torch.Tensor
    peer_dispersion: torch.Tensor
    agreement: torch.Tensor
    consistent_raw: torch.Tensor
    contradictory_raw: torch.Tensor
    common_raw: torch.Tensor
    consistent_mass: torch.Tensor
    contradictory_mass: torch.Tensor
    common_mass: torch.Tensor
    consistent_support: torch.Tensor
    contradictory_support: torch.Tensor
    common_support: torch.Tensor
    consistent_valid: torch.Tensor
    contradictory_valid: torch.Tensor
    common_valid: torch.Tensor
    consistent_positive_strength: torch.Tensor
    consistent_common_separation: torch.Tensor
    reliability: torch.Tensor


@dataclass(frozen=True)
class C3V31Support:
    """Detached K-rarity evidence and four LOO tri-support records."""

    rarity: torch.Tensor
    log_rarity_score: torch.Tensor
    bounded_score: torch.Tensor
    centered_score: torch.Tensor
    normalized_positive_rarity: torch.Tensor
    normalized_background_rarity: torch.Tensor
    relative_confidence: torch.Tensor
    absolute_confidence: torch.Tensor
    key_confidence: torch.Tensor
    key_has_variation: torch.Tensor
    key_confidence_valid: torch.Tensor
    levels: tuple[C3V31LevelSupport, ...]


@dataclass(frozen=True)
class C3V31Projection:
    """Auditable per-row result of the frozen EASN-2 projection."""

    q0: torch.Tensor
    qhat: torch.Tensor
    phi_h: torch.Tensor
    phi_b: torch.Tensor
    risk_scale_h: torch.Tensor
    risk_scale_b: torch.Tensor
    lambda_h: torch.Tensor
    lambda_b: torch.Tensor
    hard_defined: torch.Tensor
    background_defined: torch.Tensor
    hard_active: torch.Tensor
    background_active: torch.Tensor
    active_set_code: torch.Tensor
    hard_risk_delta: torch.Tensor
    background_risk_delta: torch.Tensor
    hard_risk_delta_raw: torch.Tensor
    background_risk_delta_raw: torch.Tensor
    kkt_h: torch.Tensor
    kkt_b: torch.Tensor
    kkt_max: torch.Tensor
    stationarity_residual: torch.Tensor
    objective_certificate: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_objective: torch.Tensor
    candidate_lambda_h: torch.Tensor
    candidate_lambda_b: torch.Tensor
    candidate_kkt_max: torch.Tensor
    candidate_stationarity: torch.Tensor
    candidate_risk_max: torch.Tensor
    candidate_raw_risk_max: torch.Tensor
    candidate_allowed: torch.Tensor
    candidate_finite: torch.Tensor
    candidate_simplex: torch.Tensor
    candidate_support_preserved: torch.Tensor
    candidate_lambda_ok: torch.Tensor
    candidate_kkt_ok: torch.Tensor
    candidate_stationarity_ok: torch.Tensor
    candidate_risk_ok: torch.Tensor
    candidate_objective_ok: torch.Tensor
    support_identity: torch.Tensor
    risk_identity: torch.Tensor
    solver_fallback: torch.Tensor
    accepted: torch.Tensor
    hard_inactive: torch.Tensor
    reason_code: torch.Tensor
    singleton_iterations_h: torch.Tensor
    singleton_iterations_b: torch.Tensor
    dual_iterations: torch.Tensor
    bracket_failed_h: torch.Tensor
    bracket_failed_b: torch.Tensor
    correlation_blocked: torch.Tensor
    line_search_failed: torch.Tensor
    projected_newton_used: torch.Tensor
    projected_gradient_used: torch.Tensor
    live_recert_failed: torch.Tensor
    solver_constants: tuple[tuple[str, Any], ...]


@dataclass
class C3V31DiagnosticCollector:
    """One context-local, read-only diagnostic capture ledger."""

    intervention: str
    gain_override: torch.Tensor | None
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class _C3V31DiagnosticRequest:
    module_id: int
    collector: C3V31DiagnosticCollector


_C3_V31_DIAGNOSTIC_REQUEST: ContextVar[
    _C3V31DiagnosticRequest | None
] = ContextVar("c3_v31_diagnostic_request", default=None)


def _require_architecture_seed(seed: int) -> int:
    if type(seed) is not int or seed != _FROZEN_ARCHITECTURE_SEED:
        raise ValueError("SCTransNet-SBSC V3.1 fixes architecture_seed=42")
    return seed


def _require_dataset(dataset: str) -> str:
    if type(dataset) is not str or dataset not in _FROZEN_DATASETS:
        raise ValueError(
            f"dataset must be one of {_FROZEN_DATASETS}, got {dataset!r}"
        )
    return dataset


def _require_method(method: str) -> str:
    if type(method) is not str or method not in _FROZEN_METHODS:
        raise ValueError(
            f"method must be one of {_FROZEN_METHODS}, got {method!r}"
        )
    return method


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _state_schema_records(
    state: Mapping[str, torch.Tensor],
) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    return tuple(
        (key, tuple(value.shape), value.dtype)
        for key, value in sorted(state.items())
    )


def _cache_authority_state_schema(model: SCTransNet) -> None:
    """Cache only an independently constructed exact SCTransNet schema."""

    global _AUTHORITY_STATE_SCHEMA
    if type(model) is not SCTransNet:
        raise TypeError("authority schema requires exact SCTransNet")
    state = model.state_dict()
    if len(state) != _FROZEN_BASELINE_STATE_KEY_COUNT:
        raise RuntimeError("authority SCTransNet state-key count differs")
    if _parameter_count(model) != _FROZEN_BASELINE_PARAMETER_COUNT:
        raise RuntimeError("authority SCTransNet parameter count differs")
    records = _state_schema_records(state)
    with _AUTHORITY_STATE_SCHEMA_LOCK:
        if _AUTHORITY_STATE_SCHEMA is None:
            _AUTHORITY_STATE_SCHEMA = records
        elif _AUTHORITY_STATE_SCHEMA != records:
            raise RuntimeError("authority SCTransNet state schema changed")


def _authority_state_schema(
) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    global _AUTHORITY_STATE_SCHEMA
    if _AUTHORITY_STATE_SCHEMA is None:
        authority = _construct_original(_FROZEN_ARCHITECTURE_SEED)
        _cache_authority_state_schema(authority)
    assert _AUTHORITY_STATE_SCHEMA is not None
    return _AUTHORITY_STATE_SCHEMA


def _structure_literal(value: Any) -> Any | None:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, torch.Size):
        return ("torch.Size", tuple(int(item) for item in value))
    if type(value) in (tuple, list) and all(
        item is None or type(item) in (bool, int, float, str)
        for item in value
    ):
        return (type(value).__name__, tuple(value))
    return None


def _nonreplacement_structure_records(
    model: nn.Module,
) -> tuple[tuple[Any, ...], ...]:
    excluded = "mtc.encoder.layer.1.channel_attn"
    records: list[tuple[Any, ...]] = []
    for name, module in model.named_modules():
        if name == excluded or name.startswith(excluded + "."):
            continue
        attributes = []
        for attribute, value in sorted(module.__dict__.items()):
            if attribute.startswith("_") or attribute in {"training", "mode"}:
                continue
            literal = _structure_literal(value)
            if literal is not None:
                attributes.append((attribute, literal))
        records.append(
            (
                name,
                type(module),
                module.extra_repr(),
                tuple(attributes),
            )
        )
    return tuple(records)


def _authority_nonreplacement_structure() -> tuple[tuple[Any, ...], ...]:
    global _AUTHORITY_NONREPLACEMENT_STRUCTURE
    with _AUTHORITY_NONREPLACEMENT_STRUCTURE_LOCK:
        if _AUTHORITY_NONREPLACEMENT_STRUCTURE is None:
            with torch.random.fork_rng(devices=[]):
                authority = _construct_original(_FROZEN_ARCHITECTURE_SEED)
            _AUTHORITY_NONREPLACEMENT_STRUCTURE = (
                _nonreplacement_structure_records(authority)
            )
    return _AUTHORITY_NONREPLACEMENT_STRUCTURE


def _ste_clip(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Forward clamp with an identity backward derivative."""

    clipped = raw.clamp(float(lower), float(upper))
    return raw + (clipped - raw).detach()


def _validated_query_tuple(
    normalized_queries: Sequence[torch.Tensor],
    normalized_key: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if isinstance(normalized_queries, (str, bytes, torch.Tensor)) or not isinstance(
        normalized_queries, Sequence
    ):
        raise TypeError("normalized_queries must be a sequence of four tensors")
    queries = tuple(normalized_queries)
    if len(queries) != 4:
        raise ValueError("normalized_queries must contain exactly four levels")
    batch, heads, _channels, positions = normalized_key.shape
    for index, query in enumerate(queries):
        if not isinstance(query, torch.Tensor) or query.ndim != 4:
            raise TypeError(f"normalized_queries[{index}] must be a 4D tensor")
        if (
            int(query.shape[0]) != int(batch)
            or int(query.shape[1]) != int(heads)
            or int(query.shape[-1]) != int(positions)
            or query.device != normalized_key.device
        ):
            raise ValueError("query/key batch, head, position, or device differs")
        if not query.is_floating_point():
            raise TypeError("normalized queries must be floating point")
    return queries


def _validated_validation_tuple(
    signed_validations: Sequence[torch.Tensor],
    normalized_key: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if isinstance(signed_validations, (str, bytes, torch.Tensor)) or not isinstance(
        signed_validations, Sequence
    ):
        raise TypeError("signed_validations must be a sequence of four tensors")
    validations = tuple(signed_validations)
    if len(validations) != 4:
        raise ValueError("signed_validations must contain exactly four levels")
    expected = (
        int(normalized_key.shape[0]),
        int(normalized_key.shape[1]),
        1,
        int(normalized_key.shape[-1]),
    )
    for index, validation in enumerate(validations):
        if not isinstance(validation, torch.Tensor) or validation.ndim != 4:
            raise TypeError(f"signed_validations[{index}] must be a 4D tensor")
        if tuple(validation.shape) != expected:
            raise ValueError("signed validation geometry differs from key")
        if (
            validation.device != normalized_key.device
            or not validation.is_floating_point()
        ):
            raise TypeError(
                "signed validations must be floating point on the key device"
            )
    return validations


def estimate_c3_v31_support_from_validations(
    normalized_key: torch.Tensor,
    signed_validations: Sequence[torch.Tensor],
    *,
    eps: float = SBSC_V31_EPS,
    detach_support: bool = True,
) -> C3V31Support:
    """Build frozen LOO C/H/B support from explicit signed validations."""

    if not isinstance(normalized_key, torch.Tensor) or normalized_key.ndim != 4:
        raise TypeError("normalized_key must be a 4D tensor")
    if not normalized_key.is_floating_point():
        raise TypeError("normalized_key must be floating point")
    if not math.isfinite(float(eps)) or float(eps) != _FROZEN_EPS:
        raise ValueError("C3-SBSC V3.1 freezes eps=1e-6")
    if type(detach_support) is not bool or not detach_support:
        raise ValueError("C3-SBSC V3.1 freezes detach_support=True")
    validations = _validated_validation_tuple(
        signed_validations, normalized_key
    )

    with torch.no_grad(), torch.autocast(
        device_type=normalized_key.device.type, enabled=False
    ):
        key = normalized_key.detach().float()
        values = tuple(value.detach().float() for value in validations)
        if not bool(torch.isfinite(key).all()) or any(
            not bool(torch.isfinite(value).all()) for value in values
        ):
            raise ValueError("normalized key and signed validations must be finite")
        positions = int(key.shape[-1])
        if positions < 2:
            raise ValueError("C3-SBSC V3.1 requires at least two positions")

        centered_key = key - key.mean(dim=-1, keepdim=True)
        rarity = torch.sqrt(centered_key.square().mean(dim=-2))
        raw_log_rarity = torch.log(rarity.clamp_min(_FROZEN_EPS))
        log_rarity_score = raw_log_rarity - raw_log_rarity.mean(
            dim=-1, keepdim=True
        )
        bounded_score = torch.tanh(log_rarity_score)
        centered_score = bounded_score - bounded_score.mean(
            dim=-1, keepdim=True
        )
        constant = raw_log_rarity.amax(dim=-1, keepdim=True).eq(
            raw_log_rarity.amin(dim=-1, keepdim=True)
        )
        log_rarity_score = torch.where(
            constant, torch.zeros_like(log_rarity_score), log_rarity_score
        )
        bounded_score = torch.where(
            constant, torch.zeros_like(bounded_score), bounded_score
        )
        centered_score = torch.where(
            constant, torch.zeros_like(centered_score), centered_score
        )

        positive_rarity = F.relu(centered_score)
        background_rarity = F.relu(-centered_score)
        positive_denom = positive_rarity.amax(dim=-1, keepdim=True)
        background_denom = background_rarity.amax(dim=-1, keepdim=True)
        background_mass = background_rarity.sum(dim=-1, keepdim=True)
        key_has_variation_3d = (
            (~constant)
            & positive_denom.gt(_FROZEN_EPS)
            & background_mass.gt(_FROZEN_EPS)
        )
        positive_envelope = torch.where(
            key_has_variation_3d,
            positive_rarity / (positive_denom + _FROZEN_EPS),
            torch.zeros_like(positive_rarity),
        ).unsqueeze(-2)
        background_envelope = torch.where(
            key_has_variation_3d,
            background_rarity / (background_denom + _FROZEN_EPS),
            torch.zeros_like(background_rarity),
        ).unsqueeze(-2)
        relative_confidence = torch.where(
            key_has_variation_3d,
            bounded_score.abs().amax(dim=-1, keepdim=True),
            torch.zeros_like(positive_denom),
        ).unsqueeze(-2)
        absolute_confidence = torch.tanh(
            math.sqrt(float(positions))
            * rarity.amax(dim=-1, keepdim=True)
        ).unsqueeze(-2)
        key_confidence = relative_confidence * absolute_confidence
        key_has_variation = key_has_variation_3d.unsqueeze(-2)
        key_confidence_valid = (
            key_has_variation & key_confidence.gt(_FROZEN_EPS)
        )
        uniform = torch.full_like(
            positive_envelope, 1.0 / float(positions)
        )

        levels: list[C3V31LevelSupport] = []
        for level_index in range(4):
            peer_indices = tuple(
                index for index in range(4) if index != level_index
            )
            if len(peer_indices) != 3:
                raise AssertionError("LOO peer tuple must contain three levels")
            peer_values = torch.cat(
                [values[index] for index in peer_indices], dim=-2
            )
            peer_consensus = peer_values.median(
                dim=-2, keepdim=True
            ).values
            peer_dispersion = (
                peer_values - peer_consensus
            ).abs().median(dim=-2, keepdim=True).values
            agreement = (1.0 - peer_dispersion / 2.0).clamp(0.0, 1.0)

            consistent_raw = (
                positive_envelope
                * F.relu(peer_consensus)
                * agreement
            )
            contradictory_raw = positive_envelope * (
                F.relu(-peer_consensus) * agreement
                + peer_dispersion / 2.0
            )
            common_raw = background_envelope
            consistent_mass = consistent_raw.sum(dim=-1, keepdim=True)
            contradictory_mass = contradictory_raw.sum(
                dim=-1, keepdim=True
            )
            common_mass = common_raw.sum(dim=-1, keepdim=True)
            consistent_valid = (
                key_has_variation
                & consistent_mass.gt(_FROZEN_EPS)
            )
            contradictory_valid = (
                key_has_variation
                & contradictory_mass.gt(_FROZEN_EPS)
            )
            common_valid = (
                key_has_variation
                & common_mass.gt(_FROZEN_EPS)
            )
            consistent_support = torch.where(
                consistent_valid,
                consistent_raw / consistent_mass.clamp_min(_FROZEN_EPS),
                uniform,
            )
            contradictory_support = torch.where(
                contradictory_valid,
                contradictory_raw
                / contradictory_mass.clamp_min(_FROZEN_EPS),
                uniform,
            )
            common_support = torch.where(
                common_valid,
                common_raw / common_mass.clamp_min(_FROZEN_EPS),
                uniform,
            )
            positive_strength = (
                F.relu(peer_consensus) * agreement
            ).amax(dim=-1, keepdim=True)
            separation = 0.5 * (
                consistent_support - common_support
            ).abs().sum(dim=-1, keepdim=True)
            reliability = torch.where(
                consistent_valid & common_valid & key_confidence_valid,
                (
                    key_confidence
                    * positive_strength
                    * separation
                ).clamp_min(0.0).pow(1.0 / 3.0),
                torch.zeros_like(consistent_mass),
            ).clamp(0.0, 1.0)

            levels.append(
                C3V31LevelSupport(
                    level_index=level_index,
                    peer_indices=peer_indices,
                    self_validation=values[level_index],
                    peer_consensus=peer_consensus,
                    peer_dispersion=peer_dispersion,
                    agreement=agreement,
                    consistent_raw=consistent_raw,
                    contradictory_raw=contradictory_raw,
                    common_raw=common_raw,
                    consistent_mass=consistent_mass,
                    contradictory_mass=contradictory_mass,
                    common_mass=common_mass,
                    consistent_support=consistent_support,
                    contradictory_support=contradictory_support,
                    common_support=common_support,
                    consistent_valid=consistent_valid,
                    contradictory_valid=contradictory_valid,
                    common_valid=common_valid,
                    consistent_positive_strength=positive_strength,
                    consistent_common_separation=separation,
                    reliability=reliability,
                )
            )

    return C3V31Support(
        rarity=rarity,
        log_rarity_score=log_rarity_score,
        bounded_score=bounded_score,
        centered_score=centered_score,
        normalized_positive_rarity=positive_envelope,
        normalized_background_rarity=background_envelope,
        relative_confidence=relative_confidence,
        absolute_confidence=absolute_confidence,
        key_confidence=key_confidence,
        key_has_variation=key_has_variation,
        key_confidence_valid=key_confidence_valid,
        levels=tuple(levels),
    )


def estimate_c3_v31_support(
    normalized_key: torch.Tensor,
    normalized_queries: Sequence[torch.Tensor],
    base_attentions: Sequence[torch.Tensor],
    *,
    eps: float = SBSC_V31_EPS,
    detach_support: bool = True,
) -> C3V31Support:
    """Compute signed validations, then delegate to the pure LOO estimator."""

    if not isinstance(normalized_key, torch.Tensor) or normalized_key.ndim != 4:
        raise TypeError("normalized_key must be a 4D tensor")
    if not normalized_key.is_floating_point():
        raise TypeError("normalized_key must be floating point")
    if not math.isfinite(float(eps)) or float(eps) != _FROZEN_EPS:
        raise ValueError("C3-SBSC V3.1 freezes eps=1e-6")
    if type(detach_support) is not bool or not detach_support:
        raise ValueError("C3-SBSC V3.1 freezes detach_support=True")
    queries = _validated_query_tuple(normalized_queries, normalized_key)
    if isinstance(base_attentions, (str, bytes, torch.Tensor)) or not isinstance(
        base_attentions, Sequence
    ):
        raise TypeError("base_attentions must be a sequence of four tensors")
    attentions = tuple(base_attentions)
    if len(attentions) != 4:
        raise ValueError("base_attentions must contain exactly four levels")
    for index, (attention, query) in enumerate(zip(attentions, queries)):
        expected_shape = (
            int(query.shape[0]),
            int(query.shape[1]),
            int(query.shape[2]),
            int(normalized_key.shape[2]),
        )
        if not isinstance(attention, torch.Tensor) or attention.ndim != 4:
            raise TypeError(f"base_attentions[{index}] must be a 4D tensor")
        if tuple(attention.shape) != expected_shape:
            raise ValueError("base attention geometry differs from Q/K")
        if (
            attention.device != normalized_key.device
            or not attention.is_floating_point()
        ):
            raise TypeError(
                "base attentions must be floating point on the Q/K device"
            )

    with torch.no_grad(), torch.autocast(
        device_type=normalized_key.device.type, enabled=False
    ):
        key = normalized_key.detach().float()
        query_values = tuple(query.detach().float() for query in queries)
        attention_values = tuple(
            attention.detach().float() for attention in attentions
        )
        if not bool(torch.isfinite(key).all()) or any(
            not bool(torch.isfinite(query).all()) for query in query_values
        ) or any(
            not bool(torch.isfinite(attention).all())
            for attention in attention_values
        ):
            raise ValueError("normalized Q/K and base attentions must be finite")
        validations: list[torch.Tensor] = []
        for query, attention in zip(query_values, attention_values):
            query_conditioned_key = attention @ key
            query_moment = (query * query_conditioned_key).mean(
                dim=-2, keepdim=True
            )
            centered = query_moment - query_moment.mean(
                dim=-1, keepdim=True
            )
            standardized = centered / (
                torch.sqrt(centered.square().mean(dim=-1, keepdim=True))
                + _FROZEN_EPS
            )
            validations.append(torch.tanh(standardized))

    return estimate_c3_v31_support_from_validations(
        normalized_key,
        tuple(validations),
        eps=eps,
        detach_support=detach_support,
    )


def _row_law(
    q0: torch.Tensor,
    centered_benefit: torch.Tensor,
    phi_h: torch.Tensor,
    phi_b: torch.Tensor,
    lambda_h: torch.Tensor,
    lambda_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable multiplicative law on the exact positive support of q0."""

    positive_support = q0.gt(0.0)
    safe_q0 = torch.where(positive_support, q0, torch.ones_like(q0))
    log_q0 = torch.log(safe_q0)
    logits = (
        log_q0
        + centered_benefit
        - lambda_h * phi_h
        - lambda_b * phi_b
    )
    logits = torch.where(
        positive_support,
        logits,
        torch.full_like(logits, -torch.inf),
    )
    log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_q = logits - log_normalizer
    q = torch.where(positive_support, torch.exp(log_q), torch.zeros_like(q0))
    return (
        q,
        torch.where(positive_support, log_q, torch.zeros_like(log_q)),
        torch.where(positive_support, log_q0, torch.zeros_like(log_q0)),
        log_normalizer,
    )


def _exact_row_kl(q: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
    """Exact J+ KL value with xlogy and no smoothing epsilon."""

    positive_support = q0.gt(0.0)
    safe_q0 = torch.where(positive_support, q0, torch.ones_like(q0))
    terms = torch.xlogy(q, q) - torch.xlogy(q, safe_q0)
    return torch.where(
        positive_support, terms, torch.zeros_like(terms)
    ).sum(dim=-1, keepdim=True)


def _broadcast_solver_row(
    value: torch.Tensor,
    row_shape: tuple[int, ...],
    *,
    name: str,
    boolean: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if boolean:
        if value.dtype is not torch.bool:
            raise TypeError(f"{name} must have bool dtype")
    elif not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    try:
        return torch.broadcast_to(value, row_shape + (1,))
    except RuntimeError as error:
        raise ValueError(
            f"{name} is not broadcastable to row geometry"
        ) from error


def _solve_singleton_root(
    q0: torch.Tensor,
    centered_benefit: torch.Tensor,
    phi: torch.Tensor,
    attempted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen grid-bracketed eight-update singleton Newton root."""

    zeros = torch.zeros_like(attempted, dtype=q0.dtype)
    q_empty, _log_q, _log_q0, _log_z = _row_law(
        q0, centered_benefit, phi, torch.zeros_like(phi), zeros, zeros
    )
    e0 = (q_empty * phi).sum(dim=-1, keepdim=True)
    already_safe = attempted & e0.le(0.0)
    needs_root = attempted & e0.gt(0.0)
    lower = torch.zeros_like(e0)
    upper = torch.zeros_like(e0)
    found = torch.zeros_like(attempted)
    previous = _FROZEN_LAMBDA_GRID[0]
    for grid_value in _FROZEN_LAMBDA_GRID[1:]:
        grid = torch.full_like(e0, float(grid_value))
        q_grid, _lq, _lq0, _lz = _row_law(
            q0,
            centered_benefit,
            phi,
            torch.zeros_like(phi),
            grid,
            zeros,
        )
        e_grid = (q_grid * phi).sum(dim=-1, keepdim=True)
        take = needs_root & (~found) & e_grid.le(-_FROZEN_RISK_ROOT_GUARD)
        lower = torch.where(take, torch.full_like(lower, previous), lower)
        upper = torch.where(take, grid, upper)
        found = found | take
        previous = grid_value

    current = upper
    converged = torch.zeros_like(found)
    iterations = torch.zeros_like(found, dtype=torch.int64)
    for _iteration in range(_FROZEN_SINGLETON_ITERATIONS):
        q_current, _lq, _lq0, _lz = _row_law(
            q0,
            centered_benefit,
            phi,
            torch.zeros_like(phi),
            current,
            zeros,
        )
        expectation = (q_current * phi).sum(dim=-1, keepdim=True)
        current_converged = (
            found
            & (~converged)
            & expectation.le(0.0)
            & expectation.abs().le(_FROZEN_KKT_TOLERANCE)
        )
        converged = converged | current_converged
        active = found & (~converged)
        variance = (
            q_current * (phi - expectation).square()
        ).sum(dim=-1, keepdim=True)
        newton = current + (
            expectation + _FROZEN_RISK_ROOT_GUARD
        ) / variance.clamp_min(_FROZEN_EIGENVALUE_FLOOR)
        use_newton = (
            active
            & torch.isfinite(newton)
            & newton.gt(lower)
            & newton.lt(upper)
        )
        trial = torch.where(use_newton, newton, 0.5 * (lower + upper))
        q_trial, _lq, _lq0, _lz = _row_law(
            q0,
            centered_benefit,
            phi,
            torch.zeros_like(phi),
            trial,
            zeros,
        )
        e_trial = (q_trial * phi).sum(dim=-1, keepdim=True)
        move_lower = active & e_trial.gt(-_FROZEN_RISK_ROOT_GUARD)
        lower = torch.where(move_lower, trial, lower)
        upper = torch.where(active & (~move_lower), trial, upper)
        current = torch.where(active, trial, current)
        iterations = iterations + active.to(torch.int64)
        trial_converged = (
            active
            & e_trial.le(0.0)
            & e_trial.abs().le(_FROZEN_KKT_TOLERANCE)
        )
        converged = converged | trial_converged
        lower = torch.where(trial_converged, trial, lower)
        upper = torch.where(trial_converged, trial, upper)

    terminal = torch.where(converged, current, upper)
    root = torch.where(found, terminal, zeros)
    root_available = already_safe | found
    candidate_allowed = needs_root & found
    return root, root_available, candidate_allowed, iterations


def _candidate_certificate(
    *,
    q: torch.Tensor,
    log_q: torch.Tensor,
    log_q0: torch.Tensor,
    q0: torch.Tensor,
    benefit: torch.Tensor,
    centered_benefit: torch.Tensor,
    hard_risk: torch.Tensor,
    background_risk: torch.Tensor,
    phi_h: torch.Tensor,
    phi_b: torch.Tensor,
    lambda_h: torch.Tensor,
    lambda_b: torch.Tensor,
    reliability: torch.Tensor,
    hard_defined: torch.Tensor,
    background_defined: torch.Tensor,
    hard_effective: torch.Tensor,
    background_effective: torch.Tensor,
    allowed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    finite = (
        torch.isfinite(q).all(dim=-1, keepdim=True)
        & torch.isfinite(log_q).all(dim=-1, keepdim=True)
        & torch.isfinite(lambda_h)
        & torch.isfinite(lambda_b)
    )
    simplex = (
        q.amin(dim=-1, keepdim=True).ge(-1e-7)
        & q.sum(dim=-1, keepdim=True).sub(1.0).abs().le(_FROZEN_EPS)
    )
    positive_support = q0.gt(0.0)
    support_preserved = ((~positive_support) | q.gt(0.0)).all(
        dim=-1, keepdim=True
    )
    lambda_ok = (
        lambda_h.ge(0.0)
        & lambda_b.ge(0.0)
        & lambda_h.le(_FROZEN_LAMBDA_MAX)
        & lambda_b.le(_FROZEN_LAMBDA_MAX)
    )

    hard_delta_raw = ((q - q0) * hard_risk).sum(dim=-1, keepdim=True)
    background_delta_raw = (
        (q - q0) * background_risk
    ).sum(dim=-1, keepdim=True)
    hard_delta = torch.where(
        hard_defined, hard_delta_raw, torch.zeros_like(hard_delta_raw)
    )
    background_delta = torch.where(
        background_defined,
        background_delta_raw,
        torch.zeros_like(background_delta_raw),
    )
    risk_ok = (
        ((~hard_defined) | hard_delta.le(_FROZEN_RISK_TOLERANCE))
        & (
            (~background_defined)
            | background_delta.le(_FROZEN_RISK_TOLERANCE)
        )
    )

    e_h = (q * phi_h).sum(dim=-1, keepdim=True)
    e_b = (q * phi_b).sum(dim=-1, keepdim=True)
    kkt_h = torch.where(
        hard_effective,
        torch.where(
            lambda_h.gt(_FROZEN_LAMBDA_ON), e_h.abs(), F.relu(e_h)
        ),
        torch.zeros_like(e_h),
    )
    kkt_b = torch.where(
        background_effective,
        torch.where(
            lambda_b.gt(_FROZEN_LAMBDA_ON), e_b.abs(), F.relu(e_b)
        ),
        torch.zeros_like(e_b),
    )
    kkt_max = torch.maximum(kkt_h, kkt_b)
    kkt_ok = kkt_max.le(_FROZEN_KKT_TOLERANCE)

    safe_materialized_q = torch.where(
        positive_support & q.gt(0.0), q, torch.ones_like(q)
    )
    safe_materialized_q0 = torch.where(
        positive_support, q0, torch.ones_like(q0)
    )
    materialized_log_q = torch.log(safe_materialized_q)
    materialized_log_q0 = torch.log(safe_materialized_q0)
    stationarity = (
        materialized_log_q
        - materialized_log_q0
        - centered_benefit
        + lambda_h * phi_h
        + lambda_b * phi_b
    )
    count = positive_support.sum(dim=-1, keepdim=True).clamp_min(1)
    stationarity_mean = torch.where(
        positive_support, stationarity, torch.zeros_like(stationarity)
    ).sum(dim=-1, keepdim=True) / count
    stationarity_residual = torch.where(
        positive_support,
        (stationarity - stationarity_mean).abs(),
        torch.zeros_like(stationarity),
    ).amax(dim=-1, keepdim=True)
    stationarity_ok = stationarity_residual.le(
        _FROZEN_STATIONARITY_TOLERANCE
    )

    kl = _exact_row_kl(q, q0)
    base_benefit = (q0 * benefit).sum(dim=-1, keepdim=True)
    objective = reliability * (
        (q * benefit).sum(dim=-1, keepdim=True) - base_benefit
    ) - kl
    objective_ok = objective.ge(-_FROZEN_OBJECTIVE_TOLERANCE)
    pre_kkt = finite & simplex & support_preserved & lambda_ok
    through_kkt = pre_kkt & kkt_ok & stationarity_ok
    through_risk = through_kkt & risk_ok
    valid = allowed & through_risk & objective_ok
    return {
        "valid": valid,
        "allowed": allowed,
        "finite": finite,
        "simplex": simplex,
        "support_preserved": support_preserved,
        "lambda_ok": lambda_ok,
        "through_kkt": allowed & through_kkt,
        "through_risk": allowed & through_risk,
        "risk_ok": risk_ok,
        "objective_ok": objective_ok,
        "hard_delta": hard_delta,
        "background_delta": background_delta,
        "hard_delta_raw": hard_delta_raw,
        "background_delta_raw": background_delta_raw,
        "kkt_h": kkt_h,
        "kkt_b": kkt_b,
        "kkt_max": kkt_max,
        "kkt_ok": kkt_ok,
        "stationarity": stationarity_residual,
        "stationarity_ok": stationarity_ok,
        "objective": objective,
    }


def solve_dual_risk_projection_v31(
    base_probability: torch.Tensor,
    consistent_benefit: torch.Tensor,
    hard_risk: torch.Tensor,
    background_risk: torch.Tensor,
    reliability: torch.Tensor,
    *,
    hard_active: torch.Tensor,
    background_active: torch.Tensor,
) -> C3V31Projection:
    """Solve the frozen two-risk KL projection with EASN-2."""

    inputs = (base_probability, consistent_benefit, hard_risk, background_risk)
    if any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("probability, benefit, and risks must be Tensors")
    if base_probability.ndim < 2:
        raise ValueError("base_probability must have a row dimension")
    if any(tuple(value.shape) != tuple(base_probability.shape) for value in inputs[1:]):
        raise ValueError("probability, benefit, and risk tensors must share shape")
    if any(not value.is_floating_point() for value in inputs):
        raise TypeError("probability, benefit, and risk tensors must be floating point")
    if any(value.dtype is not torch.float32 for value in inputs):
        raise TypeError("probability, benefit, and risk tensors must be FP32")
    if any(value.device != base_probability.device for value in inputs[1:]):
        raise ValueError("all solver inputs must share one device")

    row_shape = tuple(base_probability.shape[:-1])
    positions = int(base_probability.shape[-1])
    if positions < 2:
        raise ValueError("solver requires at least two columns per row")
    reliability_row = _broadcast_solver_row(
        reliability,
        row_shape,
        name="reliability",
        boolean=False,
    )
    hard_requested = _broadcast_solver_row(
        hard_active,
        row_shape,
        name="hard_active",
        boolean=True,
    )
    background_requested = _broadcast_solver_row(
        background_active,
        row_shape,
        name="background_active",
        boolean=True,
    )
    if reliability_row.dtype is not torch.float32:
        raise TypeError("reliability must be FP32")
    if any(
        value.device != base_probability.device
        for value in (reliability_row, hard_requested, background_requested)
    ):
        raise ValueError("reliability and requested-risk masks must share the solver device")

    with torch.autocast(device_type=base_probability.device.type, enabled=False):
        q0_live = base_probability.float()
        benefit_live = consistent_benefit.float()
        hard_live = hard_risk.float()
        background_live = background_risk.float()
        reliability_live = reliability_row.float()
        q0 = q0_live.detach().reshape(-1, positions)
        benefit = benefit_live.detach().reshape(-1, positions)
        risk_h = hard_live.detach().reshape(-1, positions)
        risk_b = background_live.detach().reshape(-1, positions)
        kappa = reliability_live.detach().reshape(-1, 1)
        requested_h = hard_requested.detach().reshape(-1, 1)
        requested_b = background_requested.detach().reshape(-1, 1)
        rows = int(q0.shape[0])

        if (
            not bool(torch.isfinite(q0).all())
            or bool((q0 < 0.0).any())
            or bool(
                q0.sum(dim=-1, keepdim=True)
                .sub(1.0)
                .abs()
                .gt(_FROZEN_EPS)
                .any()
            )
        ):
            raise ValueError(
                "base_probability must be finite, non-negative, and sum to one"
            )

        input_ok = (
            torch.isfinite(benefit).all(dim=-1, keepdim=True)
            & torch.isfinite(risk_h).all(dim=-1, keepdim=True)
            & torch.isfinite(risk_b).all(dim=-1, keepdim=True)
            & torch.isfinite(kappa)
            & kappa.ge(0.0)
            & kappa.le(1.0)
        )
        safe_benefit = torch.where(input_ok, benefit, torch.zeros_like(benefit))
        safe_h = torch.where(input_ok, risk_h, torch.zeros_like(risk_h))
        safe_b = torch.where(input_ok, risk_b, torch.zeros_like(risk_b))
        safe_kappa = torch.where(input_ok, kappa, torch.zeros_like(kappa))
        support_identity = input_ok & safe_kappa.le(0.0)
        preeligible = input_ok & (~support_identity)

        base_benefit = (q0 * safe_benefit).sum(dim=-1, keepdim=True)
        centered_benefit = safe_kappa * (safe_benefit - base_benefit)
        beta_h = (q0 * safe_h).sum(dim=-1, keepdim=True)
        beta_b = (q0 * safe_b).sum(dim=-1, keepdim=True)
        direction_h = safe_h - beta_h
        direction_b = safe_b - beta_b
        variance_h = (q0 * direction_h.square()).sum(dim=-1, keepdim=True)
        variance_b = (q0 * direction_b.square()).sum(dim=-1, keepdim=True)
        scale_h = direction_h.abs().amax(dim=-1, keepdim=True)
        scale_b = direction_b.abs().amax(dim=-1, keepdim=True)

        zero = torch.zeros((rows, 1), device=q0.device, dtype=torch.float32)
        q_empty, log_empty, log_q0, _empty_log_z = _row_law(
            q0,
            centered_benefit,
            torch.zeros_like(q0),
            torch.zeros_like(q0),
            zero,
            zero,
        )
        hard_variance_ok = variance_h.ge(_FROZEN_RISK_VARIANCE_MIN)
        background_variance_ok = variance_b.ge(_FROZEN_RISK_VARIANCE_MIN)
        hard_defined = requested_h & hard_variance_ok
        background_defined = requested_b & background_variance_ok
        risk_identity = preeligible & requested_b & (~background_variance_ok)

        hard_effective = (
            preeligible
            & hard_defined
            & (~risk_identity)
        )
        background_effective = (
            preeligible
            & background_defined
            & (~risk_identity)
        )
        eligible = preeligible & (~risk_identity)
        hard_inactive = preeligible & (~hard_defined)

        phi_h = torch.where(
            hard_effective,
            direction_h / scale_h.clamp_min(_FROZEN_RISK_SCALE_MIN),
            torch.zeros_like(direction_h),
        )
        phi_b = torch.where(
            background_effective,
            direction_b / scale_b.clamp_min(_FROZEN_RISK_SCALE_MIN),
            torch.zeros_like(direction_b),
        )

        root_h, root_h_available, allow_h, iterations_h = _solve_singleton_root(
            q0, centered_benefit, phi_h, eligible & hard_effective
        )
        root_b, root_b_available, allow_b, iterations_b = _solve_singleton_root(
            q0, centered_benefit, phi_b, eligible & background_effective
        )
        bracket_failed_h = eligible & hard_effective & (~root_h_available)
        bracket_failed_b = eligible & background_effective & (~root_b_available)
        zeros_like_q = torch.zeros_like(q0)
        q_h, log_h, _logq0_h, _logzh = _row_law(
            q0, centered_benefit, phi_h, zeros_like_q, root_h, zero
        )
        q_b, log_b, _logq0_b, _logzb = _row_law(
            q0, centered_benefit, zeros_like_q, phi_b, zero, root_b
        )

        certificate_common = {
            "q0": q0,
            "benefit": safe_benefit,
            "centered_benefit": centered_benefit,
            "hard_risk": safe_h,
            "background_risk": safe_b,
            "phi_h": phi_h,
            "phi_b": phi_b,
            "reliability": safe_kappa,
            "hard_defined": hard_defined,
            "background_defined": background_defined,
            "hard_effective": hard_effective,
            "background_effective": background_effective,
        }
        cert_empty = _candidate_certificate(
            q=q_empty,
            log_q=log_empty,
            log_q0=log_q0,
            lambda_h=zero,
            lambda_b=zero,
            allowed=eligible,
            **certificate_common,
        )
        cert_h = _candidate_certificate(
            q=q_h,
            log_q=log_h,
            log_q0=log_q0,
            lambda_h=root_h,
            lambda_b=zero,
            allowed=allow_h,
            **certificate_common,
        )
        cert_b = _candidate_certificate(
            q=q_b,
            log_q=log_b,
            log_q0=log_q0,
            lambda_h=zero,
            lambda_b=root_b,
            allowed=allow_b,
            **certificate_common,
        )

        first_three_valid = cert_empty["valid"] | cert_h["valid"] | cert_b["valid"]
        dual_requested = (
            eligible
            & hard_effective
            & background_effective
            & (~first_three_valid)
        )
        covariance_hb = (q0 * direction_h * direction_b).sum(
            dim=-1, keepdim=True
        )
        # The eigenvalue floor belongs only to the safeguarded Newton inverse.
        # Applying it here would hide perfectly correlated low-variance risks.
        correlation = covariance_hb / torch.sqrt(
            (variance_h * variance_b).clamp_min(torch.finfo(torch.float32).tiny)
        )
        correlation_blocked = dual_requested & correlation.abs().gt(
            _FROZEN_CORRELATION_GUARD
        )
        root_missing = dual_requested & (~(root_h_available & root_b_available))
        dual_alive = (
            dual_requested
            & root_h_available
            & root_b_available
            & (~correlation_blocked)
        )
        dual_lambda_h = root_h.clone()
        dual_lambda_b = root_b.clone()
        dual_iterations = torch.zeros_like(iterations_h)
        line_search_failed = torch.zeros_like(dual_alive)
        projected_newton_used = torch.zeros_like(dual_alive)
        projected_gradient_used = torch.zeros_like(dual_alive)
        dual_converged = torch.zeros_like(dual_alive)

        for _iteration in range(_FROZEN_DUAL_ITERATIONS):
            q_dual_step, _lq, _lq0, log_z_current = _row_law(
                q0,
                centered_benefit,
                phi_h,
                phi_b,
                dual_lambda_h,
                dual_lambda_b,
            )
            current_certificate = _candidate_certificate(
                q=q_dual_step,
                log_q=_lq,
                log_q0=_lq0,
                lambda_h=dual_lambda_h,
                lambda_b=dual_lambda_b,
                allowed=dual_alive & (~dual_converged),
                **certificate_common,
            )
            dual_converged = dual_converged | current_certificate["valid"]
            active_before = dual_alive & (~dual_converged)
            e_h = (q_dual_step * phi_h).sum(dim=-1, keepdim=True)
            e_b = (q_dual_step * phi_b).sum(dim=-1, keepdim=True)
            var_h_step = (
                q_dual_step * (phi_h - e_h).square()
            ).sum(dim=-1, keepdim=True)
            var_b_step = (
                q_dual_step * (phi_b - e_b).square()
            ).sum(dim=-1, keepdim=True)
            cov_step = (
                q_dual_step * (phi_h - e_h) * (phi_b - e_b)
            ).sum(dim=-1, keepdim=True)
            covariance = torch.stack(
                (
                    torch.cat((var_h_step, cov_step), dim=-1),
                    torch.cat((cov_step, var_b_step), dim=-1),
                ),
                dim=-2,
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            inverse = eigenvectors @ torch.diag_embed(
                eigenvalues.clamp_min(_FROZEN_EIGENVALUE_FLOOR).reciprocal()
            ) @ eigenvectors.transpose(-2, -1)
            residual = torch.cat(
                (
                    e_h + _FROZEN_RISK_ROOT_GUARD,
                    e_b + _FROZEN_RISK_ROOT_GUARD,
                ),
                dim=-1,
            )
            newton_delta = (inverse @ residual.unsqueeze(-1)).squeeze(-1)
            current_lambda = torch.cat(
                (dual_lambda_h, dual_lambda_b), dim=-1
            )
            current_dual = log_z_current - _FROZEN_RISK_ROOT_GUARD * (
                dual_lambda_h + dual_lambda_b
            )
            trials: list[torch.Tensor] = []
            trial_values: list[torch.Tensor] = []
            trial_validities: list[torch.Tensor] = []
            for alpha in _FROZEN_DUAL_ALPHAS:
                raw_trial = current_lambda + float(alpha) * newton_delta
                trial = raw_trial.clamp(0.0, _FROZEN_LAMBDA_MAX)
                numerically_valid = torch.isfinite(raw_trial).all(
                    dim=-1, keepdim=True
                ) & trial.sub(current_lambda).abs().amax(
                    dim=-1, keepdim=True
                ).gt(0.0)
                _q_trial, _lqt, _lq0t, trial_log_z = _row_law(
                    q0,
                    centered_benefit,
                    phi_h,
                    phi_b,
                    trial[:, :1],
                    trial[:, 1:],
                )
                dual_value = trial_log_z - _FROZEN_RISK_ROOT_GUARD * (
                    trial[:, :1] + trial[:, 1:]
                )
                valid_trial = (
                    active_before
                    & numerically_valid
                    & torch.isfinite(dual_value)
                    & dual_value.le(current_dual)
                )
                trials.append(trial)
                trial_values.append(
                    torch.where(
                        valid_trial,
                        dual_value,
                        torch.full_like(dual_value, torch.inf),
                    )
                )
                trial_validities.append(valid_trial)
            trial_value_matrix = torch.cat(trial_values, dim=-1)
            trial_valid_matrix = torch.cat(trial_validities, dim=-1)
            winner = trial_value_matrix.argmin(dim=-1)
            trial_tensor = torch.stack(trials, dim=1)
            chosen = torch.gather(
                trial_tensor,
                1,
                winner[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            any_trial = trial_valid_matrix.any(dim=-1, keepdim=True)
            projected_gradient_raw = current_lambda + (
                _FROZEN_PROJECTED_GRADIENT_STEP * residual
            )
            projected_gradient_trial = projected_gradient_raw.clamp(
                0.0, _FROZEN_LAMBDA_MAX
            )
            _q_pg, _lqpg, _lq0pg, projected_gradient_log_z = _row_law(
                q0,
                centered_benefit,
                phi_h,
                phi_b,
                projected_gradient_trial[:, :1],
                projected_gradient_trial[:, 1:],
            )
            projected_gradient_dual = (
                projected_gradient_log_z
                - _FROZEN_RISK_ROOT_GUARD
                * (
                    projected_gradient_trial[:, :1]
                    + projected_gradient_trial[:, 1:]
                )
            )
            projected_gradient_valid = (
                active_before
                & (~any_trial)
                & torch.isfinite(projected_gradient_raw).all(
                    dim=-1, keepdim=True
                )
                & projected_gradient_trial.sub(current_lambda).abs().amax(
                    dim=-1, keepdim=True
                ).gt(0.0)
                & torch.isfinite(projected_gradient_dual)
                & projected_gradient_dual.le(current_dual)
            )
            update_newton = active_before & any_trial
            update_gradient = projected_gradient_valid
            update = update_newton | update_gradient
            chosen_update = torch.where(
                update_newton.expand_as(chosen),
                chosen,
                projected_gradient_trial,
            )
            dual_lambda_h = torch.where(
                update, chosen_update[:, :1], dual_lambda_h
            )
            dual_lambda_b = torch.where(
                update, chosen_update[:, 1:], dual_lambda_b
            )
            dual_iterations = dual_iterations + update.to(torch.int64)
            projected_newton_used = projected_newton_used | update_newton
            projected_gradient_used = (
                projected_gradient_used | update_gradient
            )
            failed_update = active_before & (~update)
            line_search_failed = line_search_failed | failed_update
            dual_alive = dual_alive & (~failed_update)

        q_dual, log_dual, _logq0_dual, _logzd = _row_law(
            q0,
            centered_benefit,
            phi_h,
            phi_b,
            dual_lambda_h,
            dual_lambda_b,
        )
        cert_dual = _candidate_certificate(
            q=q_dual,
            log_q=log_dual,
            log_q0=log_q0,
            lambda_h=dual_lambda_h,
            lambda_b=dual_lambda_b,
            allowed=dual_alive,
            **certificate_common,
        )

        certificates = (cert_empty, cert_h, cert_b, cert_dual)
        candidate_valid = torch.cat(
            [certificate["valid"] for certificate in certificates], dim=-1
        )
        sentinel = -torch.finfo(torch.float32).max
        candidate_objective = torch.cat(
            [
                torch.where(
                    certificate["valid"],
                    certificate["objective"],
                    torch.full_like(certificate["objective"], sentinel),
                )
                for certificate in certificates
            ],
            dim=-1,
        )
        winner = candidate_objective.argmax(dim=-1)
        accepted_before_live = candidate_valid.any(dim=-1, keepdim=True)
        lambda_h_candidates = torch.cat(
            (zero, root_h, zero, dual_lambda_h), dim=-1
        )
        lambda_b_candidates = torch.cat(
            (zero, zero, root_b, dual_lambda_b), dim=-1
        )
        candidate_kkt_max = torch.cat(
            [certificate["kkt_max"] for certificate in certificates], dim=-1
        )
        candidate_stationarity = torch.cat(
            [certificate["stationarity"] for certificate in certificates],
            dim=-1,
        )
        candidate_risk_max = torch.cat(
            [
                torch.maximum(
                    certificate["hard_delta"],
                    certificate["background_delta"],
                )
                for certificate in certificates
            ],
            dim=-1,
        )
        candidate_raw_risk_max = torch.cat(
            [
                torch.maximum(
                    certificate["hard_delta_raw"],
                    certificate["background_delta_raw"],
                )
                for certificate in certificates
            ],
            dim=-1,
        )
        candidate_allowed = torch.cat(
            [certificate["allowed"] for certificate in certificates], dim=-1
        )
        candidate_finite = torch.cat(
            [certificate["finite"] for certificate in certificates], dim=-1
        )
        candidate_simplex = torch.cat(
            [certificate["simplex"] for certificate in certificates], dim=-1
        )
        candidate_support_preserved = torch.cat(
            [
                certificate["support_preserved"]
                for certificate in certificates
            ],
            dim=-1,
        )
        candidate_lambda_ok = torch.cat(
            [certificate["lambda_ok"] for certificate in certificates], dim=-1
        )
        candidate_kkt_ok = torch.cat(
            [certificate["kkt_ok"] for certificate in certificates], dim=-1
        )
        candidate_stationarity_ok = torch.cat(
            [
                certificate["stationarity_ok"]
                for certificate in certificates
            ],
            dim=-1,
        )
        candidate_risk_ok = torch.cat(
            [certificate["risk_ok"] for certificate in certificates], dim=-1
        )
        candidate_objective_ok = torch.cat(
            [
                certificate["objective_ok"]
                for certificate in certificates
            ],
            dim=-1,
        )
        selected_lambda_h = torch.gather(
            lambda_h_candidates, 1, winner[:, None]
        )
        selected_lambda_b = torch.gather(
            lambda_b_candidates, 1, winner[:, None]
        )

        # Recompute the selected law on live FP32 tensors using detached duals
        # and scales, then re-certify before exposing a differentiable qhat.
        accepted_mask_rows = accepted_before_live.reshape(row_shape + (1,))
        live_benefit_rows = torch.where(
            accepted_mask_rows, benefit_live, torch.zeros_like(benefit_live)
        ).reshape(-1, positions)
        live_h_rows = torch.where(
            accepted_mask_rows, hard_live, torch.zeros_like(hard_live)
        ).reshape(-1, positions)
        live_b_rows = torch.where(
            accepted_mask_rows, background_live, torch.zeros_like(background_live)
        ).reshape(-1, positions)
        live_kappa_rows = torch.where(
            accepted_mask_rows,
            reliability_live,
            torch.zeros_like(reliability_live),
        ).reshape(-1, 1)
        live_q0_rows = q0_live.reshape(-1, positions)
        live_centered_benefit = live_kappa_rows * (
            live_benefit_rows
            - (live_q0_rows * live_benefit_rows).sum(dim=-1, keepdim=True)
        )
        live_direction_h = live_h_rows - (
            live_q0_rows * live_h_rows
        ).sum(dim=-1, keepdim=True)
        live_direction_b = live_b_rows - (
            live_q0_rows * live_b_rows
        ).sum(dim=-1, keepdim=True)
        live_phi_h = torch.where(
            hard_effective,
            live_direction_h / scale_h.detach().clamp_min(_FROZEN_RISK_SCALE_MIN),
            torch.zeros_like(live_direction_h),
        )
        live_phi_b = torch.where(
            background_effective,
            live_direction_b / scale_b.detach().clamp_min(_FROZEN_RISK_SCALE_MIN),
            torch.zeros_like(live_direction_b),
        )
        q_live, log_q_live, log_q0_live, _live_log_z = _row_law(
            live_q0_rows,
            live_centered_benefit,
            live_phi_h,
            live_phi_b,
            selected_lambda_h.detach(),
            selected_lambda_b.detach(),
        )
        live_certificate = _candidate_certificate(
            q=q_live.detach(),
            log_q=log_q_live.detach(),
            log_q0=log_q0_live.detach(),
            q0=live_q0_rows.detach(),
            benefit=live_benefit_rows.detach(),
            centered_benefit=live_centered_benefit.detach(),
            hard_risk=live_h_rows.detach(),
            background_risk=live_b_rows.detach(),
            phi_h=live_phi_h.detach(),
            phi_b=live_phi_b.detach(),
            lambda_h=selected_lambda_h.detach(),
            lambda_b=selected_lambda_b.detach(),
            reliability=live_kappa_rows.detach(),
            hard_defined=hard_defined,
            background_defined=background_defined,
            hard_effective=hard_effective,
            background_effective=background_effective,
            allowed=accepted_before_live,
        )
        accepted = accepted_before_live & live_certificate["valid"]
        live_recert_failed = accepted_before_live & (~accepted)
        qhat_rows = q0_live.reshape(-1, positions) + accepted.to(
            q0_live.dtype
        ) * (q_live - q0_live.reshape(-1, positions))
        selected_lambda_h = torch.where(
            accepted, selected_lambda_h.detach(), zero
        )
        selected_lambda_b = torch.where(
            accepted, selected_lambda_b.detach(), zero
        )
        active_set_code = torch.where(
            accepted,
            winner[:, None].to(torch.int64),
            torch.full((rows, 1), -1, device=q0.device, dtype=torch.int64),
        )

        solver_fallback = (~support_identity) & (~risk_identity) & (~accepted)
        ill_conditioned = (
            bracket_failed_h
            | bracket_failed_b
            | root_missing
            | correlation_blocked
            | line_search_failed
        )
        any_through_kkt = torch.cat(
            [certificate["through_kkt"] for certificate in certificates],
            dim=-1,
        ).any(dim=-1, keepdim=True)
        any_through_risk = torch.cat(
            [certificate["through_risk"] for certificate in certificates],
            dim=-1,
        ).any(dim=-1, keepdim=True)
        reason_code = torch.zeros((rows, 1), device=q0.device, dtype=torch.int64)
        reason_code = torch.where(support_identity, torch.ones_like(reason_code), reason_code)
        reason_code = torch.where(
            risk_identity, torch.full_like(reason_code, 2), reason_code
        )
        unresolved = solver_fallback
        reason_code = torch.where(
            unresolved & (~input_ok), torch.full_like(reason_code, 3), reason_code
        )
        unresolved = unresolved & input_ok
        reason_code = torch.where(
            unresolved & ill_conditioned,
            torch.full_like(reason_code, 4),
            reason_code,
        )
        unresolved = unresolved & (~ill_conditioned)
        reason_code = torch.where(
            unresolved & (~any_through_kkt),
            torch.full_like(reason_code, 5),
            reason_code,
        )
        unresolved = unresolved & any_through_kkt
        reason_code = torch.where(
            unresolved & (~any_through_risk),
            torch.full_like(reason_code, 6),
            reason_code,
        )
        unresolved = unresolved & any_through_risk
        reason_code = torch.where(
            unresolved, torch.full_like(reason_code, 7), reason_code
        )

        final_hard_delta = torch.where(
            accepted,
            live_certificate["hard_delta"],
            torch.zeros_like(zero),
        )
        final_background_delta = torch.where(
            accepted,
            live_certificate["background_delta"],
            torch.zeros_like(zero),
        )
        final_hard_delta_raw = torch.where(
            accepted,
            live_certificate["hard_delta_raw"],
            torch.zeros_like(zero),
        )
        final_background_delta_raw = torch.where(
            accepted,
            live_certificate["background_delta_raw"],
            torch.zeros_like(zero),
        )
        final_kkt_h = torch.where(
            accepted, live_certificate["kkt_h"], torch.zeros_like(zero)
        )
        final_kkt_b = torch.where(
            accepted, live_certificate["kkt_b"], torch.zeros_like(zero)
        )
        final_stationarity = torch.where(
            accepted, live_certificate["stationarity"], torch.zeros_like(zero)
        )
        final_objective = torch.where(
            accepted, live_certificate["objective"], torch.zeros_like(zero)
        )

    scalar_shape = row_shape + (1,)
    return C3V31Projection(
        q0=q0_live,
        qhat=qhat_rows.reshape(base_probability.shape),
        phi_h=phi_h.reshape(base_probability.shape),
        phi_b=phi_b.reshape(base_probability.shape),
        risk_scale_h=scale_h.reshape(scalar_shape),
        risk_scale_b=scale_b.reshape(scalar_shape),
        lambda_h=selected_lambda_h.reshape(scalar_shape),
        lambda_b=selected_lambda_b.reshape(scalar_shape),
        hard_defined=hard_defined.reshape(scalar_shape),
        background_defined=background_defined.reshape(scalar_shape),
        hard_active=hard_effective.reshape(scalar_shape),
        background_active=background_effective.reshape(scalar_shape),
        active_set_code=active_set_code.reshape(scalar_shape),
        hard_risk_delta=final_hard_delta.reshape(scalar_shape),
        background_risk_delta=final_background_delta.reshape(scalar_shape),
        hard_risk_delta_raw=final_hard_delta_raw.reshape(scalar_shape),
        background_risk_delta_raw=final_background_delta_raw.reshape(
            scalar_shape
        ),
        kkt_h=final_kkt_h.reshape(scalar_shape),
        kkt_b=final_kkt_b.reshape(scalar_shape),
        kkt_max=torch.maximum(final_kkt_h, final_kkt_b).reshape(scalar_shape),
        stationarity_residual=final_stationarity.reshape(scalar_shape),
        objective_certificate=final_objective.reshape(scalar_shape),
        candidate_valid=candidate_valid.reshape(row_shape + (4,)),
        candidate_objective=candidate_objective.reshape(row_shape + (4,)),
        candidate_lambda_h=lambda_h_candidates.reshape(row_shape + (4,)),
        candidate_lambda_b=lambda_b_candidates.reshape(row_shape + (4,)),
        candidate_kkt_max=candidate_kkt_max.reshape(row_shape + (4,)),
        candidate_stationarity=candidate_stationarity.reshape(row_shape + (4,)),
        candidate_risk_max=candidate_risk_max.reshape(row_shape + (4,)),
        candidate_raw_risk_max=candidate_raw_risk_max.reshape(
            row_shape + (4,)
        ),
        candidate_allowed=candidate_allowed.reshape(row_shape + (4,)),
        candidate_finite=candidate_finite.reshape(row_shape + (4,)),
        candidate_simplex=candidate_simplex.reshape(row_shape + (4,)),
        candidate_support_preserved=candidate_support_preserved.reshape(
            row_shape + (4,)
        ),
        candidate_lambda_ok=candidate_lambda_ok.reshape(row_shape + (4,)),
        candidate_kkt_ok=candidate_kkt_ok.reshape(row_shape + (4,)),
        candidate_stationarity_ok=candidate_stationarity_ok.reshape(
            row_shape + (4,)
        ),
        candidate_risk_ok=candidate_risk_ok.reshape(row_shape + (4,)),
        candidate_objective_ok=candidate_objective_ok.reshape(
            row_shape + (4,)
        ),
        support_identity=support_identity.reshape(scalar_shape),
        risk_identity=risk_identity.reshape(scalar_shape),
        solver_fallback=solver_fallback.reshape(scalar_shape),
        accepted=accepted.reshape(scalar_shape),
        hard_inactive=hard_inactive.reshape(scalar_shape),
        reason_code=reason_code.reshape(scalar_shape),
        singleton_iterations_h=iterations_h.reshape(scalar_shape),
        singleton_iterations_b=iterations_b.reshape(scalar_shape),
        dual_iterations=dual_iterations.reshape(scalar_shape),
        bracket_failed_h=bracket_failed_h.reshape(scalar_shape),
        bracket_failed_b=bracket_failed_b.reshape(scalar_shape),
        correlation_blocked=correlation_blocked.reshape(scalar_shape),
        line_search_failed=line_search_failed.reshape(scalar_shape),
        projected_newton_used=projected_newton_used.reshape(scalar_shape),
        projected_gradient_used=projected_gradient_used.reshape(scalar_shape),
        live_recert_failed=live_recert_failed.reshape(scalar_shape),
        solver_constants=_FROZEN_SOLVER_CONSTANTS,
    )
class C3DualRiskProjectionV31(Attention_org):
    """Layer-1 SSCA replacement with one closed-loop C3 projection operator."""

    def __init__(
        self,
        config: Any,
        vis: bool,
        channel_num: list[int] | tuple[int, ...],
        *,
        layer_index: int,
        gain_limit: float = SBSC_V31_GAIN_MAX,
        eps: float = SBSC_V31_EPS,
        detach_support: bool = True,
    ) -> None:
        super().__init__(config, vis, channel_num)
        if type(layer_index) is not int or layer_index != _FROZEN_REPLACED_BLOCK_INDEX:
            raise ValueError("C3-SBSC V3.1 replaces only zero-based SCTB layer 1")
        if float(gain_limit) != _FROZEN_GAIN_MAX:
            raise ValueError("C3-SBSC V3.1 freezes gain_limit=0.25")
        if float(eps) != _FROZEN_EPS:
            raise ValueError("C3-SBSC V3.1 freezes eps=1e-6")
        if type(detach_support) is not bool or not detach_support:
            raise ValueError("C3-SBSC V3.1 freezes detach_support=True")
        self.layer_index = layer_index
        self.gain_limit = float(gain_limit)
        self.eps = float(eps)
        self.detach_support = detach_support
        self.raw_dual_risk_level_gain = nn.Parameter(
            torch.zeros(4, dtype=torch.float32)
        )

    @classmethod
    def from_ssca(
        cls,
        source: Attention_org,
        *,
        layer_index: int,
    ) -> "C3DualRiskProjectionV31":
        if type(source) is not Attention_org:
            raise TypeError("source must be the exact frozen Attention_org class")
        config = SimpleNamespace(KV_size=int(source.KV_size))
        reference = next(source.parameters())
        with torch.random.fork_rng(devices=[]):
            replacement = cls(
                config,
                bool(source.vis),
                tuple(int(value) for value in source.channel_num),
                layer_index=layer_index,
            )
        replacement.to(device=reference.device, dtype=reference.dtype)
        incompatible = replacement.load_state_dict(
            source.state_dict(), strict=False
        )
        if incompatible.missing_keys != [_FROZEN_GAIN_SUFFIX]:
            raise RuntimeError(
                "unexpected missing V3.1 replacement state: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected legacy replacement state: "
                f"{incompatible.unexpected_keys}"
            )
        with torch.no_grad():
            replacement.raw_dual_risk_level_gain.data = (
                replacement.raw_dual_risk_level_gain.detach().float()
            )
            replacement.raw_dual_risk_level_gain.zero_()
        replacement.train(source.training)
        return replacement

    def effective_level_gain(
        self,
        override: torch.Tensor | Sequence[float] | None = None,
    ) -> torch.Tensor:
        if override is None:
            return _ste_clip(
                self.raw_dual_risk_level_gain,
                _FROZEN_GAIN_MIN,
                self.gain_limit,
            )
        value = torch.as_tensor(
            override,
            device=self.raw_dual_risk_level_gain.device,
            dtype=torch.float32,
        )
        if tuple(value.shape) != (4,) or not bool(torch.isfinite(value).all()):
            raise ValueError("level gain override must be one finite FP32 vector [4]")
        if bool(
            ((value < _FROZEN_GAIN_MIN) | (value > self.gain_limit)).any()
        ):
            raise ValueError("level gain override is outside [0, 0.25]")
        return value

    @torch.no_grad()
    def project_level_gain_(self) -> None:
        self.raw_dual_risk_level_gain.clamp_(
            _FROZEN_GAIN_MIN, self.gain_limit
        )

    def _conditional_attention(
        self,
        query_fp32: torch.Tensor,
        key_fp32: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        positions = int(query_fp32.shape[-1])
        relation = (
            float(positions)
            * (
                (query_fp32 * support.float())
                @ key_fp32.transpose(-2, -1)
            )
            / math.sqrt(self.KV_size)
        )
        return self.softmax(self.psi(relation))

    def _intervene_level_support(
        self,
        level: C3V31LevelSupport,
        bundle: C3V31Support,
        intervention: str,
    ) -> C3V31LevelSupport:
        if intervention in (
            "full",
            "consistent_common_only",
            "consistent_contradictory_only",
            "disable_hard_constraint",
            "disable_background_constraint",
        ):
            return level
        updates: dict[str, Any] = {}
        uniform = torch.full_like(
            level.consistent_support,
            1.0 / float(level.consistent_support.shape[-1]),
        )
        if intervention == "uniform_consistent":
            updates["consistent_support"] = torch.where(
                level.consistent_valid, uniform, level.consistent_support
            )
        elif intervention == "uniform_contradictory":
            updates["contradictory_support"] = torch.where(
                level.contradictory_valid,
                uniform,
                level.contradictory_support,
            )
        elif intervention == "uniform_common":
            updates["common_support"] = torch.where(
                level.common_valid, uniform, level.common_support
            )
        elif intervention == "swap_consistent_contradictory":
            updates.update(
                {
                    "consistent_raw": level.contradictory_raw,
                    "contradictory_raw": level.consistent_raw,
                    "consistent_mass": level.contradictory_mass,
                    "contradictory_mass": level.consistent_mass,
                    "consistent_support": level.contradictory_support,
                    "contradictory_support": level.consistent_support,
                    "consistent_valid": level.contradictory_valid,
                    "contradictory_valid": level.consistent_valid,
                }
            )
        elif intervention == "cross_image_support":
            if int(level.consistent_support.shape[0]) < 2:
                raise ValueError("cross_image_support requires batch >= 2")
            for name in (
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
            ):
                updates[name] = torch.roll(getattr(level, name), 1, dims=0)
        elif intervention == "spatial_shuffle_support":
            generator = torch.Generator(device="cpu").manual_seed(42)
            permutation = torch.randperm(
                int(level.consistent_support.shape[-1]), generator=generator
            ).to(level.consistent_support.device)
            for name in (
                "consistent_raw",
                "contradictory_raw",
                "common_raw",
                "consistent_support",
                "contradictory_support",
                "common_support",
            ):
                updates[name] = getattr(level, name).index_select(-1, permutation)
        else:
            raise ValueError(f"unsupported C3 diagnostic intervention: {intervention}")

        transformed = replace(level, **updates)
        if intervention != "cross_image_support":
            separation = 0.5 * (
                transformed.consistent_support - transformed.common_support
            ).abs().sum(dim=-1, keepdim=True)
            reliability = torch.where(
                transformed.consistent_valid
                & transformed.common_valid
                & bundle.key_confidence_valid,
                (
                    bundle.key_confidence
                    * transformed.consistent_positive_strength
                    * separation
                ).clamp_min(0.0).pow(1.0 / 3.0),
                torch.zeros_like(transformed.consistent_mass),
            ).clamp(0.0, 1.0)
            transformed = replace(
                transformed,
                consistent_common_separation=separation,
                reliability=reliability,
            )
        return transformed

    def _project_level(
        self,
        *,
        level_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        base_attention: torch.Tensor,
        support: C3V31LevelSupport,
        key_confidence: torch.Tensor,
        key_confidence_valid: torch.Tensor,
        gain: torch.Tensor,
        gain_override: torch.Tensor | None,
        intervention: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if base_attention.dtype in (torch.float16, torch.bfloat16):
            ambient_tolerance = max(
                _FROZEN_EPS,
                4.0 * torch.finfo(base_attention.dtype).eps,
            )
        else:
            ambient_tolerance = _FROZEN_EPS
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_fp32 = query.float()
            key_fp32 = key.float()
            base_fp32 = base_attention.float()
            row_mass = base_fp32.sum(dim=-1, keepdim=True)
            if (
                not bool(torch.isfinite(base_fp32).all())
                or bool((base_fp32 < 0.0).any())
                or bool((row_mass <= 0.0).any())
            ):
                return base_attention, {
                    "level_index": level_index,
                    "intervention": intervention,
                    "base_contract_failure": True,
                    "A0_raw": base_attention,
                    "projection": None,
                    "emission_fallback": torch.ones(
                        base_attention.shape[:-1] + (1,),
                        device=base_attention.device,
                        dtype=torch.bool,
                    ),
                    "emission_reason_code": torch.full(
                        base_attention.shape[:-1] + (1,),
                        1,
                        device=base_attention.device,
                        dtype=torch.int64,
                    ),
                    "emission_tolerance": float(ambient_tolerance),
                    "qout32_certificate_ok": torch.zeros(
                        base_attention.shape[:-1] + (1,),
                        device=base_attention.device,
                        dtype=torch.bool,
                    ),
                    "emitted_certificate_ok": torch.zeros(
                        base_attention.shape[:-1] + (1,),
                        device=base_attention.device,
                        dtype=torch.bool,
                    ),
                    "raw_gain": self.raw_dual_risk_level_gain,
                    "effective_gain": gain,
                    "gain_override": gain_override,
                }

            q0 = base_fp32 / row_mass
            q_consistent = self._conditional_attention(
                query_fp32, key_fp32, support.consistent_support
            )
            q_contradictory = (
                q0
                if intervention == "consistent_common_only"
                else self._conditional_attention(
                    query_fp32, key_fp32, support.contradictory_support
                )
            )
            q_common = (
                q0
                if intervention == "consistent_contradictory_only"
                else self._conditional_attention(
                    query_fp32, key_fp32, support.common_support
                )
            )
            q_consistent = q_consistent / q_consistent.sum(
                dim=-1, keepdim=True
            )
            q_contradictory = q_contradictory / q_contradictory.sum(
                dim=-1, keepdim=True
            )
            q_common = q_common / q_common.sum(dim=-1, keepdim=True)
            consistent_benefit = torch.tanh(
                0.5
                * (
                    torch.log(q_consistent + _FROZEN_EPS)
                    - torch.log(q0 + _FROZEN_EPS)
                )
            )
            hard_risk = torch.tanh(
                0.5
                * (
                    torch.log(q_contradictory + _FROZEN_EPS)
                    - torch.log(q0 + _FROZEN_EPS)
                )
            )
            background_risk = torch.tanh(
                0.5
                * (
                    torch.log(q_common + _FROZEN_EPS)
                    - torch.log(q0 + _FROZEN_EPS)
                )
            )
            if intervention == "consistent_contradictory_only":
                # The C+H deletion branch must not retain either Common
                # validity or the full-path C--B separation through kappa.
                consistent_contradictory_separation = 0.5 * (
                    support.consistent_support
                    - support.contradictory_support
                ).abs().sum(dim=-1, keepdim=True)
                required_support = (
                    support.consistent_valid
                    & support.contradictory_valid
                    & key_confidence_valid
                )
                reliability = torch.where(
                    required_support,
                    (
                        key_confidence
                        * support.consistent_positive_strength
                        * consistent_contradictory_separation
                    ).clamp_min(0.0).pow(1.0 / 3.0),
                    torch.zeros_like(support.reliability),
                ).clamp(0.0, 1.0)
            else:
                required_support = (
                    support.consistent_valid & support.common_valid
                )
                reliability = torch.where(
                    required_support,
                    support.reliability,
                    torch.zeros_like(support.reliability),
                )
            projection = solve_dual_risk_projection_v31(
                q0,
                consistent_benefit,
                hard_risk,
                background_risk,
                reliability,
                hard_active=(
                    support.contradictory_valid
                    & (intervention != "consistent_common_only")
                    & (intervention != "disable_hard_constraint")
                ),
                background_active=(
                    support.common_valid
                    & (intervention != "consistent_contradictory_only")
                    & (intervention != "disable_background_constraint")
                ),
            )
            ahat = row_mass * projection.qhat
            effective_gain = gain[level_index].float()
            projected_fp32_candidate = base_fp32 + effective_gain * (
                ahat - base_fp32
            )
            # Identity/fallback rows must not rely on the numerically lossy
            # divide-then-multiply reconstruction q0 -> row_mass*q0.
            aout_fp32_candidate = torch.where(
                projection.accepted,
                projected_fp32_candidate,
                base_fp32,
            )
            qout_candidate = aout_fp32_candidate / row_mass

            with torch.no_grad():
                hard_defined = projection.hard_defined
                background_defined = projection.background_defined
                hard_delta_out_raw = (
                    (qout_candidate.detach() - q0.detach())
                    * hard_risk.detach()
                ).sum(dim=-1, keepdim=True)
                background_delta_out_raw = (
                    (qout_candidate.detach() - q0.detach())
                    * background_risk.detach()
                ).sum(dim=-1, keepdim=True)
                hard_delta_out = torch.where(
                    hard_defined,
                    hard_delta_out_raw,
                    torch.zeros_like(qout_candidate[..., :1]),
                )
                background_delta_out = torch.where(
                    background_defined,
                    background_delta_out_raw,
                    torch.zeros_like(qout_candidate[..., :1]),
                )
                kl_out = _exact_row_kl(
                    qout_candidate.detach(), q0.detach()
                )
                objective_out = reliability.detach() * (
                    (
                        qout_candidate.detach()
                        * consistent_benefit.detach()
                    ).sum(dim=-1, keepdim=True)
                    - (
                        q0.detach() * consistent_benefit.detach()
                    ).sum(dim=-1, keepdim=True)
                ) - kl_out
                qout32_ok = (
                    torch.isfinite(qout_candidate.detach()).all(
                        dim=-1, keepdim=True
                    )
                    & qout_candidate.detach().amin(
                        dim=-1, keepdim=True
                    ).ge(-1e-7)
                    & qout_candidate.detach().sum(
                        dim=-1, keepdim=True
                    ).sub(1.0).abs().le(_FROZEN_EPS)
                    & (
                        (~q0.detach().gt(0.0))
                        | qout_candidate.detach().gt(0.0)
                    ).all(dim=-1, keepdim=True)
                    & ((~hard_defined) | hard_delta_out.le(_FROZEN_RISK_TOLERANCE))
                    & (
                        (~background_defined)
                        | background_delta_out.le(_FROZEN_RISK_TOLERANCE)
                    )
                    & objective_out.ge(-_FROZEN_OBJECTIVE_TOLERANCE)
                )
            aout_fp32 = torch.where(
                qout32_ok, aout_fp32_candidate, base_fp32
            )
            correction = (aout_fp32 - base_fp32).to(
                dtype=base_attention.dtype
            )
            emitted_candidate = base_attention + correction

            with torch.no_grad():
                emitted_fp32 = emitted_candidate.detach().float()
                q_emitted = emitted_fp32 / row_mass.detach()
                emitted_tolerance = ambient_tolerance
                emitted_hard_delta_raw = (
                    (q_emitted - q0.detach()) * hard_risk.detach()
                ).sum(dim=-1, keepdim=True)
                emitted_background_delta_raw = (
                    (q_emitted - q0.detach()) * background_risk.detach()
                ).sum(dim=-1, keepdim=True)
                emitted_hard_delta = torch.where(
                    hard_defined,
                    emitted_hard_delta_raw,
                    torch.zeros_like(q_emitted[..., :1]),
                )
                emitted_background_delta = torch.where(
                    background_defined,
                    emitted_background_delta_raw,
                    torch.zeros_like(q_emitted[..., :1]),
                )
                emitted_kl = _exact_row_kl(q_emitted, q0.detach())
                emitted_objective = reliability.detach() * (
                    (q_emitted * consistent_benefit.detach()).sum(
                        dim=-1, keepdim=True
                    )
                    - (
                        q0.detach() * consistent_benefit.detach()
                    ).sum(dim=-1, keepdim=True)
                ) - emitted_kl
                emitted_ok = (
                    torch.isfinite(q_emitted).all(dim=-1, keepdim=True)
                    & q_emitted.amin(dim=-1, keepdim=True).ge(
                        -float(emitted_tolerance)
                    )
                    & emitted_fp32.sum(dim=-1, keepdim=True)
                    .sub(row_mass.detach())
                    .abs()
                    .le(float(emitted_tolerance))
                    & (
                        (~q0.detach().gt(0.0)) | q_emitted.gt(0.0)
                    ).all(dim=-1, keepdim=True)
                    & (
                        (~hard_defined)
                        | emitted_hard_delta.le(float(emitted_tolerance))
                    )
                    & (
                        (~background_defined)
                        | emitted_background_delta.le(float(emitted_tolerance))
                    )
                    & emitted_objective.ge(-_FROZEN_OBJECTIVE_TOLERANCE)
                )
                emission_reason = torch.where(
                    ~qout32_ok,
                    torch.ones_like(qout32_ok, dtype=torch.int64),
                    torch.where(
                        ~emitted_ok,
                        torch.full_like(qout32_ok, 2, dtype=torch.int64),
                        torch.zeros_like(qout32_ok, dtype=torch.int64),
                    ),
                )
                emission_fallback = ~qout32_ok | ~emitted_ok
                zero_emitted = torch.zeros_like(emitted_objective)
                hard_delta_emitted_raw = torch.where(
                    emission_fallback, zero_emitted, emitted_hard_delta_raw
                )
                background_delta_emitted_raw = torch.where(
                    emission_fallback,
                    zero_emitted,
                    emitted_background_delta_raw,
                )
                hard_delta_emitted = torch.where(
                    emission_fallback, zero_emitted, emitted_hard_delta
                )
                background_delta_emitted = torch.where(
                    emission_fallback, zero_emitted, emitted_background_delta
                )
                objective_emitted = torch.where(
                    emission_fallback, zero_emitted, emitted_objective
                )
            attention = torch.where(
                emission_fallback, base_attention, emitted_candidate
            )

        return attention, {
            "level_index": level_index,
            "intervention": intervention,
            "base_contract_failure": False,
            "A0_raw": base_attention,
            "q0": q0,
            "q_consistent": q_consistent,
            "q_contradictory": q_contradictory,
            "q_common": q_common,
            "consistent_benefit": consistent_benefit,
            "hard_risk": hard_risk,
            "background_risk": background_risk,
            "effective_reliability": reliability,
            "projection": projection,
            "Ahat": ahat,
            "Aout_fp32": aout_fp32,
            "Aout_emitted": attention,
            "hard_risk_delta_qout": hard_delta_out,
            "background_risk_delta_qout": background_delta_out,
            "hard_risk_delta_qout_raw": hard_delta_out_raw,
            "background_risk_delta_qout_raw": background_delta_out_raw,
            "objective_qout": objective_out,
            "hard_risk_delta_emitted": hard_delta_emitted,
            "background_risk_delta_emitted": background_delta_emitted,
            "hard_risk_delta_emitted_raw": hard_delta_emitted_raw,
            "background_risk_delta_emitted_raw": (
                background_delta_emitted_raw
            ),
            "objective_emitted": objective_emitted,
            "emission_fallback": emission_fallback,
            "emission_reason_code": emission_reason,
            "emission_tolerance": float(emitted_tolerance),
            "qout32_certificate_ok": qout32_ok,
            "emitted_certificate_ok": emitted_ok,
            "raw_gain": self.raw_dual_risk_level_gain,
            "effective_gain": gain,
            "gain_override": gain_override,
        }

    def _forward_impl(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        emb3: torch.Tensor,
        emb4: torch.Tensor,
        emb_all: torch.Tensor,
        *,
        collect_diagnostics: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, Any], ...]]:
        if any(value is None for value in (emb1, emb2, emb3, emb4, emb_all)):
            raise ValueError("frozen SCTransNet requires all four query levels")
        request = _C3_V31_DIAGNOSTIC_REQUEST.get()
        context_capture = request is not None and request.module_id == id(self)
        intervention = (
            request.collector.intervention if context_capture else "full"
        )
        collect_diagnostics = collect_diagnostics or context_capture
        _batch, _channels, height, width = emb1.shape
        queries = (
            self.q1(self.mhead1(emb1)),
            self.q2(self.mhead2(emb2)),
            self.q3(self.mhead3(emb3)),
            self.q4(self.mhead4(emb4)),
        )
        raw_key = self.k(self.mheadk(emb_all))
        value = self.v(self.mheadv(emb_all))
        raw_queries = tuple(
            rearrange(
                query,
                "b (head c) h w -> b head c (h w)",
                head=self.num_attention_heads,
            )
            for query in queries
        )
        raw_key = rearrange(
            raw_key,
            "b (head c) h w -> b head c (h w)",
            head=self.num_attention_heads,
        )
        value = rearrange(
            value,
            "b (head c) h w -> b head c (h w)",
            head=self.num_attention_heads,
        )
        queries = tuple(F.normalize(query, dim=-1) for query in raw_queries)
        key = F.normalize(raw_key, dim=-1)

        key_transpose = key.transpose(-2, -1)
        scale = math.sqrt(self.KV_size)
        base_relations = tuple(
            (query @ key_transpose) / scale for query in queries
        )
        base_attentions = tuple(
            self.softmax(self.psi(relation)) for relation in base_relations
        )
        # Preserve the inherited ambient-dtype A0 expression above, while all
        # new support and conditional cross-relations use independent FP32
        # normalizations and autocast-disabled arithmetic.
        with torch.autocast(device_type=raw_key.device.type, enabled=False):
            c3_queries = tuple(
                F.normalize(query.float(), dim=-1) for query in raw_queries
            )
            c3_key = F.normalize(raw_key.float(), dim=-1)
        support = estimate_c3_v31_support(
            c3_key,
            c3_queries,
            base_attentions,
            eps=self.eps,
            detach_support=self.detach_support,
        )
        gain_override = (
            request.collector.gain_override
            if context_capture and request is not None
            else None
        )
        gain = self.effective_level_gain(gain_override)

        attentions: list[torch.Tensor] = []
        diagnostics: list[dict[str, Any]] = []
        for level_index, (query, base_attention, level_support) in enumerate(
            zip(c3_queries, base_attentions, support.levels)
        ):
            level_support = self._intervene_level_support(
                level_support, support, intervention
            )
            attention, level_diagnostics = self._project_level(
                level_index=level_index,
                query=query,
                key=c3_key,
                base_attention=base_attention,
                support=level_support,
                key_confidence=support.key_confidence,
                key_confidence_valid=support.key_confidence_valid,
                gain=gain,
                gain_override=gain_override,
                intervention=intervention,
            )
            attentions.append(attention)
            if collect_diagnostics:
                level_diagnostics["support"] = level_support
                diagnostics.append(level_diagnostics)

        outputs = []
        projections = (
            self.project_out1,
            self.project_out2,
            self.project_out3,
            self.project_out4,
        )
        for attention, projection in zip(attentions, projections):
            out = (attention @ value).mean(dim=1)
            out = rearrange(
                out,
                "b c (h w) -> b c h w",
                h=height,
                w=width,
            )
            outputs.append(projection(out))
        if context_capture and request is not None:
            request.collector.records.append(
                {
                    "schema": SBSC_V31_SCHEMA + "/diagnostics/v1",
                    "module_id": id(self),
                    "intervention": intervention,
                    "gain_override": gain_override,
                    "levels": tuple(diagnostics),
                    "solver_constants": _FROZEN_SOLVER_CONSTANTS,
                }
            )
        return (
            (outputs[0], outputs[1], outputs[2], outputs[3], None),
            tuple(diagnostics),
        )

    def forward_with_c3_diagnostics(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        emb3: torch.Tensor,
        emb4: torch.Tensor,
        emb_all: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, Any], ...]]:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError(
                "forward_with_c3_diagnostics requires eval() and no_grad()"
            )
        return self._forward_impl(
            emb1,
            emb2,
            emb3,
            emb4,
            emb_all,
            collect_diagnostics=True,
        )

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        outputs, _diagnostics = self._forward_impl(
            emb1,
            emb2,
            emb3,
            emb4,
            emb_all,
            collect_diagnostics=False,
        )
        return outputs
def _sbsc_v31_modules(
    model: nn.Module,
) -> tuple[C3DualRiskProjectionV31, ...]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, C3DualRiskProjectionV31)
    )


@contextmanager
def capture_c3_v31_diagnostics(
    model: nn.Module,
    *,
    gain_override: torch.Tensor | Sequence[float] | None = None,
    intervention: str = "full",
):
    """Capture one or more ordinary forwards without hooks or state mutation."""

    if type(intervention) is not str or intervention not in _FROZEN_DIAGNOSTIC_INTERVENTIONS:
        raise ValueError(
            f"intervention must be one of {_FROZEN_DIAGNOSTIC_INTERVENTIONS}"
        )
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("diagnostic capture requires model.eval() and no_grad()")
    modules = _sbsc_v31_modules(model)
    if len(modules) != 1 or type(modules[0]) is not C3DualRiskProjectionV31:
        raise RuntimeError("diagnostic capture requires exactly one exact V3.1 block")
    if _C3_V31_DIAGNOSTIC_REQUEST.get() is not None:
        raise RuntimeError("nested C3 diagnostic capture is forbidden")
    normalized_override = None
    if gain_override is not None:
        normalized_override = modules[0].effective_level_gain(
            gain_override
        ).detach().clone()
    collector = C3V31DiagnosticCollector(
        intervention=intervention,
        gain_override=normalized_override,
        records=[],
    )
    token = _C3_V31_DIAGNOSTIC_REQUEST.set(
        _C3V31DiagnosticRequest(
            module_id=id(modules[0]), collector=collector
        )
    )
    try:
        yield collector
    finally:
        _C3_V31_DIAGNOSTIC_REQUEST.reset(token)


def replace_sctransnet_l1_ssca_with_sbsc_v31(
    model: SCTransNet,
) -> tuple[str, ...]:
    """Transactionally replace only zero-based SCTB layer 1."""

    if type(model) is not SCTransNet:
        raise TypeError("SBSC V3.1 replacement requires exact SCTransNet")
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("SCTransNet must contain exactly four SCTBs")
    sources = tuple(block.channel_attn for block in layers)
    if any(type(source) is not Attention_org for source in sources):
        raise TypeError("all four source operators must be exact Attention_org")

    source = sources[_FROZEN_REPLACED_BLOCK_INDEX]
    replacement = C3DualRiskProjectionV31.from_ssca(
        source,
        layer_index=_FROZEN_REPLACED_BLOCK_INDEX,
    )
    source_state = source.state_dict()
    replacement_state = replacement.state_dict()
    if set(replacement_state) - set(source_state) != {_FROZEN_GAIN_SUFFIX}:
        raise RuntimeError("SBSC V3.1 replacement added unexpected state")
    if any(
        not torch.equal(source_state[key], replacement_state[key])
        for key in source_state
    ):
        raise RuntimeError("SBSC V3.1 replacement changed shared state")

    layers[_FROZEN_REPLACED_BLOCK_INDEX].channel_attn = replacement
    if any(
        type(layers[index].channel_attn) is not Attention_org
        for index in (0, 2, 3)
    ):
        raise RuntimeError("SBSC V3.1 changed a non-L1 SSCA operator")
    return ("mtc.encoder.layer.1.channel_attn",)


@torch.no_grad()
def project_sbsc_v31_gain_(model: nn.Module) -> None:
    """Project the single live V3.1 gain after every optimizer step."""

    modules = _sbsc_v31_modules(model)
    if len(modules) != 1:
        raise RuntimeError("gain projection requires exactly one V3.1 block")
    if type(modules[0]) is not C3DualRiskProjectionV31:
        raise TypeError("gain projection rejects V3.1 subclasses")
    modules[0].project_level_gain_()


def project_sbsc_v31_constraints_(model: nn.Module) -> None:
    """Runner-facing alias for the complete V3.1 post-step constraint."""

    project_sbsc_v31_gain_(model)


def _reject_instance_forward_mutation_or_hooks(model: nn.Module) -> None:
    shadowed_names = ("forward", "state_dict", "named_parameters", "parameters")
    hook_names = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    for name, module in model.named_modules():
        location = name or "<root>"
        if any(attribute in module.__dict__ for attribute in shadowed_names):
            raise RuntimeError(f"instance method shadow detected at {location!r}")
        if any(bool(getattr(module, attribute, None)) for attribute in hook_names):
            raise RuntimeError(f"module hooks are forbidden at {location!r}")


def validate_sctransnet_sbsc_v31(
    model: nn.Module,
    require_zero_gain: bool = False,
) -> dict[str, Any]:
    """Validate the exact one-replacement V3.1 graph and bounded state."""

    if type(require_zero_gain) is not bool:
        raise TypeError("require_zero_gain must be bool")
    if type(model) is not SCTransNet:
        raise TypeError("validator requires exact unwrapped SCTransNet")
    if bool(getattr(model, "diagnostic_only", False)):
        raise RuntimeError("diagnostic-only adapters are forbidden in formal runs")
    _reject_instance_forward_mutation_or_hooks(model)
    if (
        _nonreplacement_structure_records(model)
        != _authority_nonreplacement_structure()
    ):
        raise RuntimeError(
            "SCTransNet structure outside the L1 SSCA replacement differs"
        )
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("model must retain exactly four SCTBs")
    modules = tuple(layer.channel_attn for layer in layers)
    if type(modules[1]) is not C3DualRiskProjectionV31:
        raise TypeError("zero-based SCTB layer 1 must contain exact V3.1 class")
    if any(type(modules[index]) is not Attention_org for index in (0, 2, 3)):
        raise TypeError("SCTB layers 0, 2, and 3 must retain exact Attention_org")
    discovered = _sbsc_v31_modules(model)
    if len(discovered) != 1 or discovered[0] is not modules[1]:
        raise RuntimeError("V3.1 module must exist only at SCTB layer 1")

    module = modules[1]
    assert type(module) is C3DualRiskProjectionV31
    if module.layer_index != 1:
        raise RuntimeError("V3.1 layer index differs")
    if module.detach_support is not True or module.eps != _FROZEN_EPS:
        raise RuntimeError("V3.1 support contract differs")
    if module.gain_limit != _FROZEN_GAIN_MAX:
        raise RuntimeError("V3.1 gain bound differs")
    if module.num_attention_heads != 1 or module.KV_size != 480:
        raise RuntimeError("V3.1 attention geometry differs")
    if tuple(module.channel_num) != (32, 64, 128, 256):
        raise RuntimeError("V3.1 channel contract differs")
    if module.vis is not False:
        raise RuntimeError("V3.1 freezes vis=False")
    if type(module.psi) is not nn.InstanceNorm2d:
        raise TypeError("V3.1 must retain exact InstanceNorm2d")
    if (
        module.psi.num_features != 1
        or module.psi.eps != 1e-5
        or module.psi.momentum != 0.1
        or module.psi.affine is not False
        or module.psi.track_running_stats is not False
    ):
        raise RuntimeError("V3.1 InstanceNorm2d contract differs")
    if type(module.softmax) is not nn.Softmax or module.softmax.dim != 3:
        raise TypeError("V3.1 must retain exact Softmax(dim=3)")
    gain = module._parameters.get(_FROZEN_GAIN_SUFFIX)
    if (
        gain is not module.raw_dual_risk_level_gain
        or tuple(gain.shape) != (4,)
        or gain.dtype is not torch.float32
    ):
        raise RuntimeError("V3.1 gain registration differs")
    forbidden_solver_overrides = (
        "risk_scale_min",
        "lambda_grid",
        "lambda_max",
        "singleton_iterations",
        "dual_iterations",
        "dual_alphas",
        "eigenvalue_floor",
        "correlation_guard",
        "risk_root_guard",
        "risk_tolerance",
        "kkt_tolerance",
        "stationarity_tolerance",
        "lambda_on",
    )
    if any(name in module.__dict__ for name in forbidden_solver_overrides):
        raise RuntimeError("V3.1 solver constants cannot be instance overrides")

    state_validation = validate_sbsc_v31_state_dict(
        model.state_dict(), "sbsc_v31"
    )
    if require_zero_gain and torch.count_nonzero(
        model.state_dict()[_FROZEN_GAIN_STATE_KEY]
    ).item() != 0:
        raise RuntimeError("V3.1 identity gain is not exactly zero")
    if _parameter_count(model) != _FROZEN_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("V3.1 parameter-count contract differs")
    return {
        "schema": SBSC_V31_SCHEMA,
        "state_key_count": len(model.state_dict()),
        "parameter_count": _parameter_count(model),
        "replaced_block_indices": [1],
        "gain_state_keys": [_FROZEN_GAIN_STATE_KEY],
        "gain_bounds": [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX],
        "loo_tri_support": True,
        "independent_tri_support_cross_relation": True,
        "dual_risk_projection": True,
        "per_row_certificate": True,
        "safe_fallback": True,
        "support_stop_gradient": True,
        "gain_applied_after_projection": True,
        "encoder_changed_outside_l1_ssca": False,
        "decoder_changed": False,
        "deep_supervision_changed": False,
        "state_validation": state_validation,
    }


def _state_mapping(source: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if not isinstance(source, Mapping):
        raise TypeError("state must be a mapping")
    nested = source.get("state_dict")
    candidate = nested if isinstance(nested, Mapping) else source
    if not all(
        type(key) is str and isinstance(value, torch.Tensor)
        for key, value in candidate.items()
    ):
        raise TypeError("state must map string keys to Tensors")
    return candidate


def validate_sbsc_v31_state_dict(
    state: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Validate exact V3.1/baseline checkpoint state before live loading."""

    method = _require_method(method)
    tensors = _state_mapping(state)
    prefix = ""
    if tensors and all(key.startswith("module.") for key in tensors):
        prefix = "module."
    canonical: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        canonical_key = key[len(prefix):] if prefix else key
        if canonical_key in canonical:
            raise ValueError("state contains duplicate canonical keys")
        canonical[canonical_key] = value

    expected_schema = {
        key: (shape, dtype)
        for key, shape, dtype in _authority_state_schema()
    }
    if method == "sbsc_v31":
        expected_schema[_FROZEN_GAIN_STATE_KEY] = ((4,), torch.float32)
    if set(canonical) != set(expected_schema):
        missing = sorted(set(expected_schema) - set(canonical))
        unexpected = sorted(set(canonical) - set(expected_schema))
        raise ValueError(
            "state key set differs from the exact V3.1 architecture schema; "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    for key, (expected_shape, expected_dtype) in expected_schema.items():
        value = canonical[key]
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"state shape differs for {key!r}")
        if value.dtype != expected_dtype:
            raise TypeError(f"state dtype differs for {key!r}")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"state tensor {key!r} must be finite")

    gain_values: list[float] = []
    if method == "sbsc_v31":
        value = canonical[_FROZEN_GAIN_STATE_KEY]
        if not torch.isfinite(value).all():
            raise ValueError("V3.1 gain state must be finite")
        if value.dtype is not torch.float32 or tuple(value.shape) != (4,):
            raise TypeError("V3.1 gain state must be one FP32 vector [4]")
        if bool(
            ((value < _FROZEN_GAIN_MIN) | (value > _FROZEN_GAIN_MAX)).any()
        ):
            raise ValueError("V3.1 gain state is outside [0, 0.25]")
        gain_values.extend(value.detach().cpu().tolist())
    return {
        "method": method,
        "state_key_count": len(tensors),
        "gain_state_keys": (
            [f"{prefix}{_FROZEN_GAIN_STATE_KEY}"]
            if method == "sbsc_v31"
            else []
        ),
        "gain_values": gain_values,
        "gain_bounds": (
            [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX]
            if method == "sbsc_v31"
            else None
        ),
        "data_parallel_prefix": bool(prefix),
    }


def structurally_inactive_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Return unchanged SCTransNet parameters absent from the V3.1 graph."""

    names = tuple(name for name, _parameter in model.named_parameters())
    inactive = tuple(
        name
        for name in names
        if (
            name.endswith("position_embeddings")
            and ".mtc.embeddings_" in f".{name}"
        )
        or any(
            name.endswith(f"q{query}_attn{key}")
            for query in range(1, 5)
            for key in range(1, 5)
        )
    )
    if len(inactive) != 68:
        raise RuntimeError(
            f"expected 68 structurally inactive parameters, got {len(inactive)}"
        )
    return inactive


def _method_metadata(
    model: SCTransNet,
    *,
    method: str,
    dataset: str,
    architecture_seed: int,
    shared_state_sha256: str,
) -> dict[str, Any]:
    state = model.state_dict()
    gain_keys = [key for key in (_FROZEN_GAIN_STATE_KEY,) if key in state]
    return {
        "schema": SBSC_V31_SCHEMA,
        "method": method,
        "dataset": dataset,
        "architecture_seed": architecture_seed,
        "state_key_count": len(state),
        "parameter_count": _parameter_count(model),
        "state_sha256": state_dict_sha256(state),
        "shared_state_sha256": shared_state_sha256,
        "paired_initialization": True,
        "test_split_accessed": False,
        "gain_state_keys": gain_keys,
        "gain_bounds": (
            [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX] if gain_keys else None
        ),
        "replaced_block_indices": [1] if method == "sbsc_v31" else [],
        "loo_tri_support": method == "sbsc_v31",
        "independent_tri_support_cross_relation": method == "sbsc_v31",
        "dual_risk_projection": method == "sbsc_v31",
    }


def build_paired_sctransnet_sbsc_v31(
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
) -> tuple[SCTransNet, SCTransNet, dict[str, Any]]:
    """Build bitwise-paired seed-42 SCTransNet and SBSC V3.1 from scratch."""

    dataset = _require_dataset(dataset)
    architecture_seed = _require_architecture_seed(architecture_seed)
    baseline = _construct_original(architecture_seed)
    _cache_authority_state_schema(baseline)
    candidate = copy.deepcopy(baseline)
    replace_sctransnet_l1_ssca_with_sbsc_v31(candidate)

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    if set(candidate_state) - set(baseline_state) != {_FROZEN_GAIN_STATE_KEY}:
        raise RuntimeError("candidate adds state outside the single V3.1 gain")
    if set(baseline_state) - set(candidate_state):
        raise RuntimeError("candidate removed baseline state")
    if any(
        not torch.equal(baseline_state[key], candidate_state[key])
        for key in baseline_state
    ):
        raise RuntimeError("paired shared state is not bitwise equal")
    shared_hash = state_dict_sha256(baseline_state)
    if shared_hash != state_dict_sha256(candidate_state, baseline_state.keys()):
        raise RuntimeError("paired shared-state hashes differ")
    if _parameter_count(baseline) != _FROZEN_BASELINE_PARAMETER_COUNT:
        raise RuntimeError("baseline parameter-count contract differs")
    validate_sctransnet_sbsc_v31(candidate, require_zero_gain=True)
    validate_sbsc_v31_state_dict(baseline_state, "sctransnet")

    baseline_metadata = _method_metadata(
        baseline,
        method="sctransnet",
        dataset=dataset,
        architecture_seed=architecture_seed,
        shared_state_sha256=shared_hash,
    )
    candidate_metadata = _method_metadata(
        candidate,
        method="sbsc_v31",
        dataset=dataset,
        architecture_seed=architecture_seed,
        shared_state_sha256=shared_hash,
    )
    metadata = dict(candidate_metadata)
    metadata.update(
        {
            "pair_schema": SBSC_V31_SCHEMA,
            "baseline": baseline_metadata,
            "candidate": candidate_metadata,
            "shared_state_key_count": len(baseline_state),
            "shared_state_bitwise_equal": True,
            "parent_checkpoint": None,
            "warm_start_used": False,
            "predecessor_checkpoint_used": False,
            "model_construction_preserves_caller_rng_stream": True,
        }
    )
    return baseline, candidate, metadata


def build_sctransnet_sbsc_v31_method(
    method: str,
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
) -> tuple[SCTransNet, dict[str, Any]]:
    """Build the complete pair first, then return one requested role."""

    method = _require_method(method)
    if type(training) is not bool:
        raise TypeError("training must be bool")
    baseline, candidate, pair_metadata = build_paired_sctransnet_sbsc_v31(
        dataset,
        architecture_seed,
    )
    model = baseline if method == "sctransnet" else candidate
    model.train(training)
    model.mode = "train" if training else "test"
    metadata = dict(
        pair_metadata["baseline" if method == "sctransnet" else "candidate"]
    )
    metadata.update(
        {
            "training": training,
            "pair": pair_metadata,
            "test_split_accessed": False,
        }
    )
    return model, metadata


C3V31 = C3DualRiskProjectionV31


__all__ = [
    "ARCHITECTURE_SEED",
    "EXPECTED_SBSC_V31_BLOCKS",
    "EXPECTED_SBSC_V31_PARAMETER_COUNT",
    "EXPECTED_SBSC_V31_STATE_KEY_COUNT",
    "C3DualRiskProjectionV31",
    "C3V31",
    "C3V31DiagnosticCollector",
    "C3V31LevelSupport",
    "C3V31Projection",
    "C3V31Support",
    "SBSC_V31_EPS",
    "SBSC_V31_GAIN_MAX",
    "SBSC_V31_GAIN_MIN",
    "SBSC_V31_GAIN_STATE_KEY",
    "SBSC_V31_GAIN_SUFFIX",
    "SBSC_V31_SCHEMA",
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
    "build_paired_sctransnet_sbsc_v31",
    "build_sctransnet_sbsc_v31_method",
    "capture_c3_v31_diagnostics",
    "estimate_c3_v31_support",
    "estimate_c3_v31_support_from_validations",
    "project_sbsc_v31_constraints_",
    "project_sbsc_v31_gain_",
    "replace_sctransnet_l1_ssca_with_sbsc_v31",
    "solve_dual_risk_projection_v31",
    "structurally_inactive_parameter_names",
    "validate_sbsc_v31_state_dict",
    "validate_sctransnet_sbsc_v31",
]
