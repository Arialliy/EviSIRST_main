#!/usr/bin/env python3
"""SCTransNet-SBSC V2.1 core and exact paired-construction contract.

SBSC V2.1 changes exactly one operator: zero-based SCTB layer 1 (the second
SCTB) channel
attention.  The remaining three operators, encoder, decoder, and deep
supervision graph stay as the original SCTransNet implementation.  Its only
learned extension is one V2.1-specific non-negative scalar gain.  At the
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
from dataclasses import dataclass
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


SBSC_V21_SCHEMA = (
    "sctransnet_sbsc_v21/query_validated_rare_simplex_transport/v1"
)

# Executable validators use private literals rather than mutable public
# aliases.  In particular, V2.1 has a distinct method and state key from V2,
# so a V2 checkpoint cannot be silently interpreted with V2.1 semantics.
_FROZEN_ARCHITECTURE_SEED = 42
_FROZEN_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
_FROZEN_METHODS = ("sctransnet", "sbsc_v21")
_FROZEN_BLOCK_COUNT = 4
_FROZEN_REPLACED_BLOCK_INDEX = 1
_FROZEN_BASELINE_STATE_KEY_COUNT = 510
_FROZEN_BASELINE_PARAMETER_COUNT = 11_325_939
_FROZEN_CANDIDATE_STATE_KEY_COUNT = 511
_FROZEN_CANDIDATE_PARAMETER_COUNT = 11_325_940
_FROZEN_GAIN_MIN = 0.0
_FROZEN_GAIN_MAX = 0.5
_FROZEN_EPS = 1e-6
_FROZEN_GAIN_SUFFIX = "raw_simplex_gain"
_FROZEN_GAIN_STATE_KEY = (
    "mtc.encoder.layer.1.channel_attn." + _FROZEN_GAIN_SUFFIX
)

ARCHITECTURE_SEED = _FROZEN_ARCHITECTURE_SEED
SUPPORTED_DATASETS = _FROZEN_DATASETS
SUPPORTED_METHODS = _FROZEN_METHODS
EXPECTED_SBSC_V21_BLOCKS = 1
EXPECTED_SBSC_V21_STATE_KEY_COUNT = _FROZEN_CANDIDATE_STATE_KEY_COUNT
EXPECTED_SBSC_V21_PARAMETER_COUNT = _FROZEN_CANDIDATE_PARAMETER_COUNT
SBSC_V21_GAIN_MIN = _FROZEN_GAIN_MIN
SBSC_V21_GAIN_MAX = _FROZEN_GAIN_MAX
SBSC_V21_EPS = _FROZEN_EPS
SBSC_V21_GAIN_SUFFIX = _FROZEN_GAIN_SUFFIX
SBSC_V21_GAIN_STATE_KEY = _FROZEN_GAIN_STATE_KEY

_AUTHORITY_STATE_SCHEMA: tuple[
    tuple[str, tuple[int, ...], torch.dtype], ...
] | None = None
_AUTHORITY_STATE_SCHEMA_LOCK = threading.Lock()


@dataclass(frozen=True)
class SBSCV21LevelSupport:
    """One detached query level's signed validation and support pair."""

    query_moment: torch.Tensor
    centered_query_moment: torch.Tensor
    standardized_query_moment: torch.Tensor
    signed_query_validation: torch.Tensor
    positive_mass: torch.Tensor
    negative_mass: torch.Tensor
    positive_support: torch.Tensor
    negative_support: torch.Tensor
    balance: torch.Tensor
    query_confidence: torch.Tensor
    confidence: torch.Tensor
    has_two_sided_mass: torch.Tensor


@dataclass(frozen=True)
class SBSCV21Support:
    """Detached K-rarity evidence and four query-validated support pairs."""

    rarity: torch.Tensor
    log_rarity_score: torch.Tensor
    bounded_score: torch.Tensor
    centered_score: torch.Tensor
    normalized_positive_rarity: torch.Tensor
    relative_confidence: torch.Tensor
    absolute_confidence: torch.Tensor
    key_confidence: torch.Tensor
    has_spatial_variation: torch.Tensor
    levels: tuple[SBSCV21LevelSupport, ...]


def _require_architecture_seed(seed: int) -> int:
    if type(seed) is not int or seed != _FROZEN_ARCHITECTURE_SEED:
        raise ValueError("SCTransNet-SBSC V2.1 fixes architecture_seed=42")
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


def estimate_sbsc_v21_support(
    normalized_key: torch.Tensor,
    normalized_queries: Sequence[torch.Tensor],
    base_attentions: Sequence[torch.Tensor],
    *,
    eps: float = SBSC_V21_EPS,
    detach_support: bool = True,
) -> SBSCV21Support:
    """Estimate the frozen two-sided query-validated rare support.

    ``base_attentions[i]`` is the untouched SSCA law ``A0_i``.  On the
    detached support path it produces ``M_i=A0_i@K`` and a signed spatial
    validation from ``mean_c(Q_i*M_i)``.  The returned weights/confidences are
    detached; the live conditional relations later reuse the original Q/K.
    """

    if not isinstance(normalized_key, torch.Tensor) or normalized_key.ndim != 4:
        raise TypeError("normalized_key must be a 4D tensor")
    if not normalized_key.is_floating_point():
        raise TypeError("normalized_key must be floating point")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    if float(eps) != _FROZEN_EPS:
        raise ValueError("SBSC V2.1 freezes eps=1e-6")
    if type(detach_support) is not bool or not detach_support:
        raise ValueError("SBSC V2.1 freezes detach_support=True")
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
        if attention.device != normalized_key.device or not attention.is_floating_point():
            raise TypeError("base attentions must be floating point on the Q/K device")

    with torch.no_grad(), torch.autocast(
        device_type=normalized_key.device.type, enabled=False
    ):
        key = normalized_key.detach().float()
        query_values = tuple(query.detach().float() for query in queries)
        attention_values = tuple(
            attention.detach().float() for attention in attentions
        )
        if not torch.isfinite(key).all() or any(
            not torch.isfinite(query).all() for query in query_values
        ) or any(not torch.isfinite(attention).all() for attention in attention_values):
            raise ValueError("normalized Q/K and base attentions must be finite")

        positions = int(key.shape[-1])
        if positions < 2:
            raise ValueError("SBSC V2.1 requires at least two positions")

        # Frozen V2 K-rarity confidence, followed by the V2.1 normalized
        # positive-rarity envelope rho.
        centered_key = key - key.mean(dim=-1, keepdim=True)
        rarity = torch.sqrt(centered_key.square().mean(dim=-2))
        log_rarity = torch.log(rarity.clamp_min(_FROZEN_EPS))
        centered_log_rarity = log_rarity - log_rarity.mean(
            dim=-1, keepdim=True
        )
        bounded = torch.tanh(centered_log_rarity)
        centered_score = bounded - bounded.mean(dim=-1, keepdim=True)
        constant = log_rarity.amax(dim=-1, keepdim=True).eq(
            log_rarity.amin(dim=-1, keepdim=True)
        )
        centered_log_rarity = torch.where(
            constant, torch.zeros_like(centered_log_rarity), centered_log_rarity
        )
        bounded = torch.where(constant, torch.zeros_like(bounded), bounded)
        centered_score = torch.where(
            constant, torch.zeros_like(centered_score), centered_score
        )
        positive_rarity = F.relu(centered_score)
        rho_denom = positive_rarity.amax(dim=-1, keepdim=True)
        negative_rarity_mass = F.relu(-centered_score).sum(
            dim=-1, keepdim=True
        )
        has_variation = (
            (~constant)
            & rho_denom.gt(_FROZEN_EPS)
            & negative_rarity_mass.gt(_FROZEN_EPS)
        )
        rho = torch.where(
            has_variation,
            positive_rarity / (rho_denom + _FROZEN_EPS),
            torch.zeros_like(positive_rarity),
        ).unsqueeze(-2)
        relative_confidence = torch.where(
            has_variation,
            bounded.abs().amax(dim=-1, keepdim=True),
            torch.zeros_like(rho_denom),
        )
        absolute_confidence = torch.tanh(
            math.sqrt(float(positions))
            * rarity.amax(dim=-1, keepdim=True)
        )
        key_confidence = (
            relative_confidence * absolute_confidence
        ).unsqueeze(-2)
        has_variation_4d = has_variation.unsqueeze(-2)

        uniform = torch.full_like(rho, 1.0 / float(positions))
        levels: list[SBSCV21LevelSupport] = []
        for query, attention in zip(query_values, attention_values):
            query_conditioned_key = attention @ key
            query_moment = (query * query_conditioned_key).mean(
                dim=-2, keepdim=True
            )
            centered_query_moment = query_moment - query_moment.mean(
                dim=-1, keepdim=True
            )
            query_mean_square = centered_query_moment.square().mean(
                dim=-1, keepdim=True
            )
            standardized_query_moment = centered_query_moment / (
                torch.sqrt(query_mean_square) + _FROZEN_EPS
            )
            signed_query_validation = torch.tanh(
                standardized_query_moment
            )
            positive = rho * F.relu(signed_query_validation)
            negative = rho * F.relu(-signed_query_validation)
            positive_mass = positive.sum(dim=-1, keepdim=True)
            negative_mass = negative.sum(dim=-1, keepdim=True)
            has_two_sided_mass = (
                has_variation_4d
                & positive_mass.gt(_FROZEN_EPS)
                & negative_mass.gt(_FROZEN_EPS)
            )
            positive_support = torch.where(
                has_two_sided_mass,
                positive / positive_mass.clamp_min(_FROZEN_EPS),
                uniform,
            )
            negative_support = torch.where(
                has_two_sided_mass,
                negative / negative_mass.clamp_min(_FROZEN_EPS),
                uniform,
            )
            balance = torch.where(
                has_two_sided_mass,
                2.0
                * torch.minimum(positive_mass, negative_mass)
                / (positive_mass + negative_mass + _FROZEN_EPS),
                torch.zeros_like(positive_mass),
            )
            query_confidence = torch.tanh(
                float(positions) * torch.sqrt(query_mean_square)
            )
            confidence = torch.where(
                has_two_sided_mass,
                balance
                * torch.sqrt(
                    (key_confidence * query_confidence).clamp_min(0.0)
                ),
                torch.zeros_like(balance),
            ).clamp(0.0, 1.0)
            levels.append(
                SBSCV21LevelSupport(
                    query_moment=query_moment,
                    centered_query_moment=centered_query_moment,
                    standardized_query_moment=standardized_query_moment,
                    signed_query_validation=signed_query_validation,
                    positive_mass=positive_mass,
                    negative_mass=negative_mass,
                    positive_support=positive_support,
                    negative_support=negative_support,
                    balance=balance,
                    query_confidence=query_confidence,
                    confidence=confidence,
                    has_two_sided_mass=has_two_sided_mass,
                )
            )

    return SBSCV21Support(
        rarity=rarity,
        log_rarity_score=centered_log_rarity,
        bounded_score=bounded,
        centered_score=centered_score,
        normalized_positive_rarity=rho,
        relative_confidence=relative_confidence.unsqueeze(-2),
        absolute_confidence=absolute_confidence.unsqueeze(-2),
        key_confidence=key_confidence,
        has_spatial_variation=has_variation,
        levels=tuple(levels),
    )


class QueryValidatedRareSimplexTransportV21(Attention_org):
    """Second-SCTB SSCA replacement with query-validated simplex transport."""

    def __init__(
        self,
        config: Any,
        vis: bool,
        channel_num: list[int] | tuple[int, ...],
        *,
        layer_index: int,
        gain_limit: float = SBSC_V21_GAIN_MAX,
        eps: float = SBSC_V21_EPS,
        detach_support: bool = True,
    ) -> None:
        super().__init__(config, vis, channel_num)
        if type(layer_index) is not int or layer_index != _FROZEN_REPLACED_BLOCK_INDEX:
            raise ValueError("SBSC V2.1 replaces only zero-based SCTB layer 1")
        if float(gain_limit) != _FROZEN_GAIN_MAX:
            raise ValueError("SBSC V2.1 freezes gain_limit=0.5")
        if float(eps) != _FROZEN_EPS:
            raise ValueError("SBSC V2.1 freezes eps=1e-6")
        if type(detach_support) is not bool or not detach_support:
            raise ValueError("SBSC V2.1 freezes detach_support=True")
        self.layer_index = layer_index
        self.gain_limit = float(gain_limit)
        self.eps = float(eps)
        self.detach_support = detach_support
        self.raw_simplex_gain = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_ssca(
        cls,
        source: Attention_org,
        *,
        layer_index: int,
    ) -> "QueryValidatedRareSimplexTransportV21":
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
                "unexpected missing V2.1 replacement state: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected legacy replacement state: "
                f"{incompatible.unexpected_keys}"
            )
        with torch.no_grad():
            replacement.raw_simplex_gain.zero_()
        replacement.train(source.training)
        return replacement

    def effective_transport_gain(
        self,
        override: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if override is None:
            return _ste_clip(
                self.raw_simplex_gain,
                _FROZEN_GAIN_MIN,
                self.gain_limit,
            )
        value = torch.as_tensor(
            override,
            device=self.raw_simplex_gain.device,
            dtype=self.raw_simplex_gain.dtype,
        )
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError("transport gain override must be one finite scalar")
        value = value.reshape(())
        scalar = float(value.detach().cpu().item())
        if not _FROZEN_GAIN_MIN <= scalar <= self.gain_limit:
            raise ValueError("transport gain override is outside [0, 0.5]")
        return value

    @torch.no_grad()
    def project_transport_gain_(self) -> None:
        self.raw_simplex_gain.clamp_(
            _FROZEN_GAIN_MIN, self.gain_limit
        )

    def _simplex_tangent_transport(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        base_attention: torch.Tensor,
        support: SBSCV21LevelSupport,
        gain: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply the frozen multiplicative simplex-tangent transport."""

        with torch.autocast(device_type=query.device.type, enabled=False):
            query_fp32 = query.float()
            key_fp32 = key.float()
            positions = int(query.shape[-1])
            positive_relation = (
                float(positions)
                * (
                    (query_fp32 * support.positive_support.float())
                    @ key_fp32.transpose(-2, -1)
                )
                / math.sqrt(self.KV_size)
            )
            negative_relation = (
                float(positions)
                * (
                    (query_fp32 * support.negative_support.float())
                    @ key_fp32.transpose(-2, -1)
                )
                / math.sqrt(self.KV_size)
            )
            positive_attention = self.softmax(self.psi(positive_relation))
            negative_attention = self.softmax(self.psi(negative_relation))
            base_fp32 = base_attention.float()
            log_ratio_direction = torch.tanh(
                0.5
                * (
                    torch.log(positive_attention + _FROZEN_EPS)
                    - torch.log(negative_attention + _FROZEN_EPS)
                )
            )
            base_mean_direction = (
                base_fp32 * log_ratio_direction
            ).sum(dim=-1, keepdim=True)
            total_variation = 0.5 * (
                positive_attention - negative_attention
            ).abs().sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            step = gain.float() * support.confidence.float() * total_variation
            centered_direction = log_ratio_direction - base_mean_direction
            factor = 1.0 + step * centered_direction
            if not bool(torch.isfinite(factor).all()):
                raise FloatingPointError("V2.1 simplex factor is non-finite")
            if bool((factor < -4.0 * _FROZEN_EPS).any()) or bool(
                (factor > 2.0 + 4.0 * _FROZEN_EPS).any()
            ):
                raise FloatingPointError("V2.1 simplex factor left [0, 2]")
            factor = factor.clamp(0.0, 2.0)
            attention_fp32 = base_fp32 * factor
            correction_fp32 = attention_fp32 - base_fp32
        correction = correction_fp32.to(dtype=base_attention.dtype)
        attention = base_attention + correction
        return attention, {
            "base_attention": base_attention,
            "positive_attention": positive_attention,
            "negative_attention": negative_attention,
            "log_ratio_direction": log_ratio_direction,
            "base_mean_direction": base_mean_direction,
            "centered_direction": centered_direction,
            "total_variation": total_variation,
            "transport_step": step,
            "simplex_factor": factor,
            "transport_correction_fp32": correction_fp32,
            "transport_correction": correction,
            "attention": attention,
        }

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        if any(value is None for value in (emb1, emb2, emb3, emb4, emb_all)):
            raise ValueError("frozen SCTransNet requires all four query levels")
        _batch, _channels, height, width = emb1.shape
        queries = (
            self.q1(self.mhead1(emb1)),
            self.q2(self.mhead2(emb2)),
            self.q3(self.mhead3(emb3)),
            self.q4(self.mhead4(emb4)),
        )
        raw_key = self.k(self.mheadk(emb_all))
        value = self.v(self.mheadv(emb_all))
        queries = tuple(
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
        queries = tuple(F.normalize(query, dim=-1) for query in queries)
        key = F.normalize(raw_key, dim=-1)

        # Preserve the original SSCA expression order and ambient dtype.  The
        # V2.1 correction is added only after the exact anchor is complete.
        key_transpose = key.transpose(-2, -1)
        scale = math.sqrt(self.KV_size)
        base_relations = tuple(
            (query @ key_transpose) / scale for query in queries
        )
        base_attentions = tuple(
            self.softmax(self.psi(relation)) for relation in base_relations
        )
        support = estimate_sbsc_v21_support(
            key,
            queries,
            base_attentions,
            eps=self.eps,
            detach_support=self.detach_support,
        )
        gain = self.effective_transport_gain()

        attentions: list[torch.Tensor] = []
        for query, base_attention, level_support in zip(
            queries, base_attentions, support.levels
        ):
            attention, _diagnostics = self._simplex_tangent_transport(
                query,
                key,
                base_attention,
                level_support,
                gain,
            )
            attentions.append(attention)

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
        return outputs[0], outputs[1], outputs[2], outputs[3], None


def _sbsc_v21_modules(
    model: nn.Module,
) -> tuple[QueryValidatedRareSimplexTransportV21, ...]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, QueryValidatedRareSimplexTransportV21)
    )


def replace_sctransnet_l1_ssca_with_sbsc_v21(
    model: SCTransNet,
) -> tuple[str, ...]:
    """Transactionally replace only zero-based SCTB layer 1."""

    if type(model) is not SCTransNet:
        raise TypeError("SBSC V2.1 replacement requires exact SCTransNet")
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("SCTransNet must contain exactly four SCTBs")
    sources = tuple(block.channel_attn for block in layers)
    if any(type(source) is not Attention_org for source in sources):
        raise TypeError("all four source operators must be exact Attention_org")

    source = sources[_FROZEN_REPLACED_BLOCK_INDEX]
    replacement = QueryValidatedRareSimplexTransportV21.from_ssca(
        source,
        layer_index=_FROZEN_REPLACED_BLOCK_INDEX,
    )
    source_state = source.state_dict()
    replacement_state = replacement.state_dict()
    if set(replacement_state) - set(source_state) != {_FROZEN_GAIN_SUFFIX}:
        raise RuntimeError("SBSC V2.1 replacement added unexpected state")
    if any(
        not torch.equal(source_state[key], replacement_state[key])
        for key in source_state
    ):
        raise RuntimeError("SBSC V2.1 replacement changed shared state")

    layers[_FROZEN_REPLACED_BLOCK_INDEX].channel_attn = replacement
    if any(
        type(layers[index].channel_attn) is not Attention_org
        for index in (0, 2, 3)
    ):
        raise RuntimeError("SBSC V2.1 changed a non-L1 SSCA operator")
    return ("mtc.encoder.layer.1.channel_attn",)


@torch.no_grad()
def project_sbsc_v21_gain_(model: nn.Module) -> None:
    """Project the single live V2.1 gain after every optimizer step."""

    modules = _sbsc_v21_modules(model)
    if len(modules) != 1:
        raise RuntimeError("gain projection requires exactly one V2.1 block")
    if type(modules[0]) is not QueryValidatedRareSimplexTransportV21:
        raise TypeError("gain projection rejects V2.1 subclasses")
    modules[0].project_transport_gain_()


def project_sbsc_v21_constraints_(model: nn.Module) -> None:
    """Runner-facing alias for the complete V2.1 post-step constraint."""

    project_sbsc_v21_gain_(model)


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


def validate_sctransnet_sbsc_v21(
    model: nn.Module,
    require_zero_gain: bool = False,
) -> dict[str, Any]:
    """Validate the exact one-replacement V2.1 graph and bounded state."""

    if type(require_zero_gain) is not bool:
        raise TypeError("require_zero_gain must be bool")
    if type(model) is not SCTransNet:
        raise TypeError("validator requires exact unwrapped SCTransNet")
    _reject_instance_forward_mutation_or_hooks(model)
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("model must retain exactly four SCTBs")
    modules = tuple(layer.channel_attn for layer in layers)
    if type(modules[1]) is not QueryValidatedRareSimplexTransportV21:
        raise TypeError("zero-based SCTB layer 1 must contain exact V2.1 class")
    if any(type(modules[index]) is not Attention_org for index in (0, 2, 3)):
        raise TypeError("SCTB layers 0, 2, and 3 must retain exact Attention_org")
    discovered = _sbsc_v21_modules(model)
    if len(discovered) != 1 or discovered[0] is not modules[1]:
        raise RuntimeError("V2.1 module must exist only at SCTB layer 1")

    module = modules[1]
    assert type(module) is QueryValidatedRareSimplexTransportV21
    if module.layer_index != 1:
        raise RuntimeError("V2.1 layer index differs")
    if module.detach_support is not True or module.eps != _FROZEN_EPS:
        raise RuntimeError("V2.1 support contract differs")
    if module.gain_limit != _FROZEN_GAIN_MAX:
        raise RuntimeError("V2.1 gain bound differs")
    if module.num_attention_heads != 1 or module.KV_size != 480:
        raise RuntimeError("V2.1 attention geometry differs")
    if tuple(module.channel_num) != (32, 64, 128, 256):
        raise RuntimeError("V2.1 channel contract differs")
    if module.vis is not False:
        raise RuntimeError("V2.1 freezes vis=False")
    if type(module.psi) is not nn.InstanceNorm2d:
        raise TypeError("V2.1 must retain exact InstanceNorm2d")
    if (
        module.psi.num_features != 1
        or module.psi.eps != 1e-5
        or module.psi.momentum != 0.1
        or module.psi.affine is not False
        or module.psi.track_running_stats is not False
    ):
        raise RuntimeError("V2.1 InstanceNorm2d contract differs")
    if type(module.softmax) is not nn.Softmax or module.softmax.dim != 3:
        raise TypeError("V2.1 must retain exact Softmax(dim=3)")
    gain = module._parameters.get(_FROZEN_GAIN_SUFFIX)
    if gain is not module.raw_simplex_gain or tuple(gain.shape) != ():
        raise RuntimeError("V2.1 gain registration differs")

    state_validation = validate_sbsc_v21_state_dict(
        model.state_dict(), "sbsc_v21"
    )
    if require_zero_gain and torch.count_nonzero(
        model.state_dict()[_FROZEN_GAIN_STATE_KEY]
    ).item() != 0:
        raise RuntimeError("V2.1 identity gain is not exactly zero")
    if _parameter_count(model) != _FROZEN_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("V2.1 parameter-count contract differs")
    return {
        "schema": SBSC_V21_SCHEMA,
        "state_key_count": len(model.state_dict()),
        "parameter_count": _parameter_count(model),
        "replaced_block_indices": [1],
        "gain_state_keys": [_FROZEN_GAIN_STATE_KEY],
        "gain_bounds": [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX],
        "query_validated_rare_support": True,
        "simplex_tangent_transport": True,
        "support_stop_gradient": True,
        "transport_preserves_simplex": True,
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


def validate_sbsc_v21_state_dict(
    state: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Validate exact V2.1/baseline checkpoint state before live loading."""

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
    if method == "sbsc_v21":
        expected_schema[_FROZEN_GAIN_STATE_KEY] = ((), torch.float32)
    if set(canonical) != set(expected_schema):
        missing = sorted(set(expected_schema) - set(canonical))
        unexpected = sorted(set(canonical) - set(expected_schema))
        raise ValueError(
            "state key set differs from the exact V2.1 architecture schema; "
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
    if method == "sbsc_v21":
        value = canonical[_FROZEN_GAIN_STATE_KEY]
        if not torch.isfinite(value).all():
            raise ValueError("V2.1 gain state must be finite")
        scalar = float(value.detach().cpu().reshape(()).item())
        if not _FROZEN_GAIN_MIN <= scalar <= _FROZEN_GAIN_MAX:
            raise ValueError("V2.1 gain state is outside [0, 0.5]")
        gain_values.append(scalar)
    return {
        "method": method,
        "state_key_count": len(tensors),
        "gain_state_keys": (
            [f"{prefix}{_FROZEN_GAIN_STATE_KEY}"]
            if method == "sbsc_v21"
            else []
        ),
        "gain_values": gain_values,
        "gain_bounds": (
            [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX]
            if method == "sbsc_v21"
            else None
        ),
        "data_parallel_prefix": bool(prefix),
    }


def structurally_inactive_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Return unchanged SCTransNet parameters absent from the V2.1 graph."""

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
        "schema": SBSC_V21_SCHEMA,
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
        "replaced_block_indices": [1] if method == "sbsc_v21" else [],
    }


def build_paired_sctransnet_sbsc_v21(
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
) -> tuple[SCTransNet, SCTransNet, dict[str, Any]]:
    """Build bitwise-paired seed-42 SCTransNet and SBSC V2.1 from scratch."""

    dataset = _require_dataset(dataset)
    architecture_seed = _require_architecture_seed(architecture_seed)
    baseline = _construct_original(architecture_seed)
    _cache_authority_state_schema(baseline)
    candidate = copy.deepcopy(baseline)
    replace_sctransnet_l1_ssca_with_sbsc_v21(candidate)

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    if set(candidate_state) - set(baseline_state) != {_FROZEN_GAIN_STATE_KEY}:
        raise RuntimeError("candidate adds state outside the single V2.1 gain")
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
    validate_sctransnet_sbsc_v21(candidate, require_zero_gain=True)
    validate_sbsc_v21_state_dict(baseline_state, "sctransnet")

    baseline_metadata = _method_metadata(
        baseline,
        method="sctransnet",
        dataset=dataset,
        architecture_seed=architecture_seed,
        shared_state_sha256=shared_hash,
    )
    candidate_metadata = _method_metadata(
        candidate,
        method="sbsc_v21",
        dataset=dataset,
        architecture_seed=architecture_seed,
        shared_state_sha256=shared_hash,
    )
    metadata = dict(candidate_metadata)
    metadata.update(
        {
            "pair_schema": SBSC_V21_SCHEMA,
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


def build_sctransnet_sbsc_v21_method(
    method: str,
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
) -> tuple[SCTransNet, dict[str, Any]]:
    """Build the complete pair first, then return one requested role."""

    method = _require_method(method)
    if type(training) is not bool:
        raise TypeError("training must be bool")
    baseline, candidate, pair_metadata = build_paired_sctransnet_sbsc_v21(
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


SBSCV21 = QueryValidatedRareSimplexTransportV21


__all__ = [
    "ARCHITECTURE_SEED",
    "EXPECTED_SBSC_V21_BLOCKS",
    "EXPECTED_SBSC_V21_PARAMETER_COUNT",
    "EXPECTED_SBSC_V21_STATE_KEY_COUNT",
    "QueryValidatedRareSimplexTransportV21",
    "SBSCV21",
    "SBSCV21LevelSupport",
    "SBSCV21Support",
    "SBSC_V21_EPS",
    "SBSC_V21_GAIN_MAX",
    "SBSC_V21_GAIN_MIN",
    "SBSC_V21_GAIN_STATE_KEY",
    "SBSC_V21_GAIN_SUFFIX",
    "SBSC_V21_SCHEMA",
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
    "build_paired_sctransnet_sbsc_v21",
    "build_sctransnet_sbsc_v21_method",
    "estimate_sbsc_v21_support",
    "project_sbsc_v21_constraints_",
    "project_sbsc_v21_gain_",
    "replace_sctransnet_l1_ssca_with_sbsc_v21",
    "structurally_inactive_parameter_names",
    "validate_sbsc_v21_state_dict",
    "validate_sctransnet_sbsc_v21",
]
