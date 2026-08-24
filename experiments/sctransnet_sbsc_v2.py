#!/usr/bin/env python3
"""SCTransNet-SBSC V2 core, paired builder, and causal diagnostics.

SBSC V2 replaces only the four SSCA relation operators in SCTransNet.  Its
support is parameter free; the complete learned extension is one bounded
scalar transport gain per transformer block.  A zero gain is an exact SSCA
identity, while the straight-through clamp leaves a usable gradient at zero.

This module deliberately contains no dataset reader, evaluator, or checkpoint
selection logic.  In particular, constructing a model cannot access a test
split.
"""

from __future__ import annotations

import copy
import math
import threading
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from experiments.four_dataset_models_seed42_v1 import (
    _construct_original,
    state_dict_sha256,
)
from model.SCTransNet import Attention_org, SCTransNet


SBSC_V2_SCHEMA = "sctransnet_sbsc_v2/conditional_cross_moment_transport/v2"

# Validators use these private literals, never their public aliases.  Thus a
# caller cannot make malformed state pass by monkeypatching exported counts or
# bounds to agree with the malformed object.
_FROZEN_ARCHITECTURE_SEED = 42
_FROZEN_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
_FROZEN_METHODS = ("sctransnet", "sbsc_v2")
_FROZEN_BLOCK_COUNT = 4
_FROZEN_BASELINE_STATE_KEY_COUNT = 510
_FROZEN_BASELINE_PARAMETER_COUNT = 11_325_939
_FROZEN_CANDIDATE_STATE_KEY_COUNT = 514
_FROZEN_CANDIDATE_PARAMETER_COUNT = 11_325_943
_FROZEN_GAIN_MIN = 0.0
_FROZEN_GAIN_MAX = 0.25
_FROZEN_EPS = 1e-6
_FROZEN_ABSOLUTE_CONFIDENCE_TEMPERATURE = 1.0
_FROZEN_GAIN_SUFFIX = "raw_transport_gain"

ARCHITECTURE_SEED = _FROZEN_ARCHITECTURE_SEED
SUPPORTED_DATASETS = _FROZEN_DATASETS
SUPPORTED_METHODS = _FROZEN_METHODS
EXPECTED_SBSC_V2_BLOCKS = _FROZEN_BLOCK_COUNT
EXPECTED_SBSC_V2_STATE_KEY_COUNT = _FROZEN_CANDIDATE_STATE_KEY_COUNT
EXPECTED_SBSC_V2_PARAMETER_COUNT = _FROZEN_CANDIDATE_PARAMETER_COUNT
SBSC_V2_GAIN_MIN = _FROZEN_GAIN_MIN
SBSC_V2_GAIN_MAX = _FROZEN_GAIN_MAX
SBSC_V2_EPS = _FROZEN_EPS
SBSC_V2_ABSOLUTE_CONFIDENCE_TEMPERATURE = (
    _FROZEN_ABSOLUTE_CONFIDENCE_TEMPERATURE
)
SBSC_V2_GAIN_SUFFIX = _FROZEN_GAIN_SUFFIX

_GAIN_STATE_KEYS = tuple(
    f"mtc.encoder.layer.{index}.channel_attn.{_FROZEN_GAIN_SUFFIX}"
    for index in range(_FROZEN_BLOCK_COUNT)
)

_AUTHORITY_STATE_SCHEMA: tuple[
    tuple[str, tuple[int, ...], torch.dtype], ...
] | None = None
_AUTHORITY_STATE_SCHEMA_LOCK = threading.Lock()

InterventionMode = Literal[
    "normal",
    "zero",
    "reverse",
    "shuffle",
    "cross_image",
]
_INTERVENTION_MODES = frozenset(
    {"normal", "zero", "reverse", "shuffle", "cross_image"}
)


@dataclass(frozen=True)
class SBSCV2Support:
    """Parameter-free support derived from post-L2-normalized shared K."""

    rarity: torch.Tensor
    log_rarity_score: torch.Tensor
    bounded_score: torch.Tensor
    centered_score: torch.Tensor
    candidate: torch.Tensor
    counter_support: torch.Tensor
    relative_confidence: torch.Tensor
    absolute_confidence: torch.Tensor
    confidence: torch.Tensor
    has_spatial_variation: torch.Tensor


@dataclass(frozen=True)
class SBSCV2Intervention:
    """One whole-graph, forward-local causal intervention specification.

    ``spatial_permutations`` maps the four zero-based SCTB indices to fixed
    permutations of the current token axis.  ``batch_permutation`` must be a
    fixed-point-free permutation for the ``cross_image`` intervention.
    """

    mode: InterventionMode = "normal"
    spatial_permutations: Mapping[int, torch.Tensor] | None = None
    batch_permutation: torch.Tensor | None = None


@dataclass
class _SBSCV2ForwardContext:
    module_ids: frozenset[int]
    intervention: SBSCV2Intervention
    diagnostics: dict[int, dict[str, Any]]


_FORWARD_CONTEXT: ContextVar[_SBSCV2ForwardContext | None] = ContextVar(
    "sctransnet_sbsc_v2_forward_context",
    default=None,
)


def _require_architecture_seed(seed: int) -> int:
    if type(seed) is not int or seed != _FROZEN_ARCHITECTURE_SEED:
        raise ValueError("SCTransNet-SBSC V2 fixes architecture_seed=42")
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
    """Forward clamp with identity backward derivative everywhere."""

    clipped = raw.clamp(float(lower), float(upper))
    return raw + (clipped - raw).detach()


def estimate_sbsc_v2_support(
    normalized_key: torch.Tensor,
    *,
    eps: float = SBSC_V2_EPS,
    detach_support: bool = True,
) -> SBSCV2Support:
    """Create Jordan candidate/counter-support densities from normalized K.

    The input is the actual K after SCTransNet's spatial L2 normalization and
    has ``B x heads x key_channels x positions`` layout.  The support path is
    detached by default; K remains non-detached in all cross-moment operands.
    """

    if not isinstance(normalized_key, torch.Tensor) or normalized_key.ndim != 4:
        shape = getattr(normalized_key, "shape", None)
        raise ValueError(f"normalized_key must be BHKN, got {shape}")
    if not normalized_key.is_floating_point():
        raise TypeError("normalized_key must be floating point")
    if int(normalized_key.shape[-1]) < 2:
        raise ValueError("SBSC V2 requires at least two spatial positions")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    source = normalized_key.detach() if detach_support else normalized_key
    with torch.autocast(device_type=normalized_key.device.type, enabled=False):
        work = source.float()
        if not torch.isfinite(work).all():
            raise ValueError("normalized_key must contain only finite values")

        # Per-channel spatial centering is used only to measure rarity.  The
        # cross-moments below use the unchanged normalized Q and K tensors.
        centered_key = work - work.mean(dim=-1, keepdim=True)
        # Keep the unclamped RMS for the absolute continuity gate.  Only the
        # logarithmic branch receives the numerical floor.
        rarity = torch.sqrt(centered_key.square().mean(dim=-2))
        log_rarity = torch.log(rarity.clamp_min(float(eps)))
        z = log_rarity - log_rarity.mean(dim=-1, keepdim=True)
        bounded = torch.tanh(z)
        h = bounded - bounded.mean(dim=-1, keepdim=True)

        # An exactly spatially constant rarity field is an exact no-evidence
        # case.  Explicit overrides prevent residual arbitrary support.
        constant = log_rarity.amax(dim=-1, keepdim=True).eq(
            log_rarity.amin(dim=-1, keepdim=True)
        )
        z = torch.where(constant, torch.zeros_like(z), z)
        bounded = torch.where(constant, torch.zeros_like(bounded), bounded)
        h = torch.where(constant, torch.zeros_like(h), h)

        positive = F.relu(h)
        negative = F.relu(-h)
        positive_mass = positive.sum(dim=-1, keepdim=True)
        negative_mass = negative.sum(dim=-1, keepdim=True)
        has_variation = (
            (~constant)
            & positive_mass.gt(float(eps))
            & negative_mass.gt(float(eps))
        )
        positions = int(h.shape[-1])
        uniform = torch.full_like(h, 1.0 / float(positions))
        candidate = torch.where(
            has_variation,
            positive / positive_mass.clamp_min(float(eps)),
            uniform,
        )
        counter_support = torch.where(
            has_variation,
            negative / negative_mass.clamp_min(float(eps)),
            uniform,
        )
        relative_confidence = torch.where(
            has_variation,
            bounded.abs().amax(dim=-1, keepdim=True),
            torch.zeros_like(positive_mass),
        )
        absolute_confidence = torch.tanh(
            math.sqrt(float(positions))
            * rarity.amax(dim=-1, keepdim=True)
            / _FROZEN_ABSOLUTE_CONFIDENCE_TEMPERATURE
        )
        confidence = (
            relative_confidence * absolute_confidence
        ).unsqueeze(-2)

    return SBSCV2Support(
        rarity=rarity,
        log_rarity_score=z,
        bounded_score=bounded,
        centered_score=h,
        candidate=candidate,
        counter_support=counter_support,
        relative_confidence=relative_confidence.unsqueeze(-2),
        absolute_confidence=absolute_confidence.unsqueeze(-2),
        confidence=confidence,
        has_spatial_variation=has_variation,
    )


def _validate_permutation(
    permutation: torch.Tensor,
    size: int,
    *,
    label: str,
    require_derangement: bool,
) -> torch.Tensor:
    if not isinstance(permutation, torch.Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if permutation.ndim != 1 or int(permutation.numel()) != int(size):
        raise ValueError(f"{label} must have shape ({size},)")
    if permutation.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"{label} must have an integer dtype")
    host = permutation.detach().cpu().to(dtype=torch.long)
    if not torch.equal(torch.sort(host).values, torch.arange(size)):
        raise ValueError(f"{label} must be a permutation of range({size})")
    if require_derangement and torch.any(host.eq(torch.arange(size))):
        raise ValueError(f"{label} must have no fixed points")
    return host


class SupportBalancedConditionalCrossMomentV2(Attention_org):
    """Drop-in SSCA replacement with bounded signed relation transport."""

    def __init__(
        self,
        config: Any,
        vis: bool,
        channel_num: list[int] | tuple[int, ...],
        *,
        layer_index: int,
        gain_limit: float = SBSC_V2_GAIN_MAX,
        eps: float = SBSC_V2_EPS,
        absolute_confidence_temperature: float = (
            SBSC_V2_ABSOLUTE_CONFIDENCE_TEMPERATURE
        ),
        detach_support: bool = True,
    ) -> None:
        super().__init__(config, vis, channel_num)
        if type(layer_index) is not int or not 0 <= layer_index < 4:
            raise ValueError("layer_index must be an integer in [0, 3]")
        if float(gain_limit) != _FROZEN_GAIN_MAX:
            raise ValueError("SBSC V2 freezes gain_limit=0.25")
        if float(eps) != _FROZEN_EPS:
            raise ValueError("SBSC V2 freezes eps=1e-6")
        if (
            float(absolute_confidence_temperature)
            != _FROZEN_ABSOLUTE_CONFIDENCE_TEMPERATURE
        ):
            raise ValueError(
                "SBSC V2 freezes absolute_confidence_temperature=1.0"
            )
        if type(detach_support) is not bool or not detach_support:
            raise ValueError("SBSC V2 freezes detach_support=True")
        self.layer_index = layer_index
        self.gain_limit = float(gain_limit)
        self.eps = float(eps)
        self.absolute_confidence_temperature = float(
            absolute_confidence_temperature
        )
        self.detach_support = detach_support
        self.raw_transport_gain = nn.Parameter(torch.zeros(()))

    @classmethod
    def from_ssca(
        cls,
        source: Attention_org,
        *,
        layer_index: int,
    ) -> "SupportBalancedConditionalCrossMomentV2":
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
                "unexpected missing replacement state: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected legacy replacement state: "
                f"{incompatible.unexpected_keys}"
            )
        with torch.no_grad():
            replacement.raw_transport_gain.zero_()
        replacement.train(source.training)
        return replacement

    def effective_transport_gain(
        self,
        override: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if override is None:
            return _ste_clip(
                self.raw_transport_gain,
                _FROZEN_GAIN_MIN,
                self.gain_limit,
            )
        value = torch.as_tensor(
            override,
            device=self.raw_transport_gain.device,
            dtype=self.raw_transport_gain.dtype,
        )
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError("transport gain override must be one finite scalar")
        value = value.reshape(())
        scalar = float(value.detach().cpu().item())
        if not _FROZEN_GAIN_MIN <= scalar <= self.gain_limit:
            raise ValueError("transport gain override is outside [0, 0.25]")
        return value

    @torch.no_grad()
    def project_transport_gain_(self) -> None:
        self.raw_transport_gain.clamp_(
            _FROZEN_GAIN_MIN, self.gain_limit
        )

    def _apply_context(
        self,
        support: SBSCV2Support,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None, str]:
        candidate = support.candidate
        counter = support.counter_support
        confidence = support.confidence
        override: float | None = None
        mode = "normal"
        context = _FORWARD_CONTEXT.get()
        if context is None or id(self) not in context.module_ids:
            return candidate, counter, confidence, override, mode

        intervention = context.intervention
        mode = intervention.mode
        if mode == "zero":
            override = 0.0
        elif mode == "reverse":
            candidate, counter = counter, candidate
        elif mode == "shuffle":
            assert intervention.spatial_permutations is not None
            permutation = intervention.spatial_permutations[self.layer_index]
            host = _validate_permutation(
                permutation,
                int(candidate.shape[-1]),
                label=f"spatial_permutations[{self.layer_index}]",
                require_derangement=True,
            )
            device_permutation = host.to(candidate.device)
            candidate = candidate.index_select(-1, device_permutation)
            counter = counter.index_select(-1, device_permutation)
        elif mode == "cross_image":
            batch_size = int(candidate.shape[0])
            if batch_size < 2:
                raise ValueError("cross_image diagnostics require batch size >= 2")
            assert intervention.batch_permutation is not None
            host = _validate_permutation(
                intervention.batch_permutation,
                batch_size,
                label="batch_permutation",
                require_derangement=True,
            )
            device_permutation = host.to(candidate.device)
            candidate = candidate.index_select(0, device_permutation)
            counter = counter.index_select(0, device_permutation)
            confidence = confidence.index_select(0, device_permutation)
        elif mode != "normal":  # defended again at the module boundary
            raise RuntimeError(f"unsupported intervention mode: {mode!r}")
        return candidate, counter, confidence, override, mode

    def _relation_transport(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        base_attention: torch.Tensor,
        candidate: torch.Tensor,
        counter: torch.Tensor,
        confidence: torch.Tensor,
        gain: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # The conditional branch is deliberately FP32 even under AMP.  A0 is
        # supplied from the untouched ambient-dtype SSCA path below; only the
        # completed correction is cast back before the identity-preserving add.
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_fp32 = query.float()
            key_fp32 = key.float()
            key_transpose_fp32 = key_fp32.transpose(-2, -1)
            candidate_fp32 = candidate.float()
            counter_fp32 = counter.float()
            positions = int(query.shape[-1])
            scale = math.sqrt(self.KV_size)
            positive_relation = (
                float(positions)
                * (
                    (query_fp32 * candidate_fp32.unsqueeze(-2))
                    @ key_transpose_fp32
                )
                / scale
            )
            negative_relation = (
                float(positions)
                * (
                    (query_fp32 * counter_fp32.unsqueeze(-2))
                    @ key_transpose_fp32
                )
                / scale
            )
            positive_attention = self.softmax(self.psi(positive_relation))
            negative_attention = self.softmax(self.psi(negative_relation))
            delta = positive_attention - negative_attention
            delta_zero = delta - delta.mean(dim=-1, keepdim=True)
            row_l1 = delta_zero.abs().sum(dim=-1, keepdim=True)
            transport_direction = delta_zero / torch.maximum(
                torch.ones_like(row_l1), row_l1
            )
            correction_fp32 = (
                gain.float() * confidence.float() * transport_direction
            )
        correction = correction_fp32.to(dtype=base_attention.dtype)
        attention = base_attention + correction
        return attention, {
            "base_attention": base_attention,
            "positive_attention": positive_attention,
            "negative_attention": negative_attention,
            "delta": delta,
            "delta_zero": delta_zero,
            "transport_direction": transport_direction,
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

        support = estimate_sbsc_v2_support(
            key,
            eps=self.eps,
            detach_support=self.detach_support,
        )
        candidate, counter, confidence, gain_override, mode = (
            self._apply_context(support)
        )
        gain = self.effective_transport_gain(gain_override)

        # Keep the original SSCA relation/IN/softmax expression order and
        # ambient dtype exactly.  This is the zero-gain identity anchor.
        key_transpose = key.transpose(-2, -1)
        scale = math.sqrt(self.KV_size)
        base_relations = tuple(
            (query @ key_transpose) / scale for query in queries
        )
        base_attentions = tuple(
            self.softmax(self.psi(relation)) for relation in base_relations
        )

        attentions: list[torch.Tensor] = []
        branch_diagnostics: list[dict[str, torch.Tensor]] = []
        for query, base_attention in zip(queries, base_attentions):
            attention, diagnostics = self._relation_transport(
                query,
                key,
                base_attention,
                candidate,
                counter,
                confidence,
                gain,
            )
            attentions.append(attention)
            branch_diagnostics.append(diagnostics)

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

        context = _FORWARD_CONTEXT.get()
        if context is not None and id(self) in context.module_ids:
            if self.layer_index in context.diagnostics:
                raise RuntimeError("SBSC V2 layer executed twice in one diagnostic")
            context.diagnostics[self.layer_index] = {
                "layer_index": self.layer_index,
                "mode": mode,
                "rarity": support.rarity.detach(),
                "log_rarity_score": support.log_rarity_score.detach(),
                "bounded_score": support.bounded_score.detach(),
                "centered_score": support.centered_score.detach(),
                "candidate": candidate.detach(),
                "counter_support": counter.detach(),
                "relative_confidence": support.relative_confidence.detach(),
                "absolute_confidence": support.absolute_confidence.detach(),
                "confidence": confidence.detach(),
                "has_spatial_variation": support.has_spatial_variation.detach(),
                "effective_gain": gain.detach(),
                "branches": tuple(
                    {
                        key_name: tensor.detach()
                        for key_name, tensor in branch.items()
                    }
                    for branch in branch_diagnostics
                ),
            }
        return outputs[0], outputs[1], outputs[2], outputs[3], None


def _sbsc_v2_modules(
    model: nn.Module,
) -> tuple[SupportBalancedConditionalCrossMomentV2, ...]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, SupportBalancedConditionalCrossMomentV2)
    )


def replace_sctransnet_ssca_with_sbsc_v2(model: SCTransNet) -> tuple[str, ...]:
    """Transactionally replace all four SSCA modules in an exact SCTransNet."""

    if type(model) is not SCTransNet:
        raise TypeError("SBSC V2 replacement requires the exact SCTransNet class")
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("SCTransNet must contain exactly four SCTBs")
    sources = tuple(block.channel_attn for block in layers)
    if any(type(source) is not Attention_org for source in sources):
        raise TypeError("all four source operators must be exact Attention_org")

    # Build and validate all modules before assigning any of them.
    replacements = tuple(
        SupportBalancedConditionalCrossMomentV2.from_ssca(
            source,
            layer_index=index,
        )
        for index, source in enumerate(sources)
    )
    for source, replacement in zip(sources, replacements):
        source_state = source.state_dict()
        replacement_state = replacement.state_dict()
        if set(replacement_state) - set(source_state) != {_FROZEN_GAIN_SUFFIX}:
            raise RuntimeError("SBSC V2 replacement added unexpected state")
        if any(
            not torch.equal(source_state[key], replacement_state[key])
            for key in source_state
        ):
            raise RuntimeError("SBSC V2 replacement changed shared state")
    for block, replacement in zip(layers, replacements):
        block.channel_attn = replacement
    return tuple(
        f"mtc.encoder.layer.{index}.channel_attn"
        for index in range(_FROZEN_BLOCK_COUNT)
    )


@torch.no_grad()
def project_sbsc_v2_gains_(model: nn.Module) -> None:
    """Required post-optimizer-step projection for all four gains."""

    modules = _sbsc_v2_modules(model)
    if len(modules) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("gain projection requires exactly four SBSC V2 blocks")
    if any(type(module) is not SupportBalancedConditionalCrossMomentV2 for module in modules):
        raise TypeError("gain projection rejects SBSC V2 subclasses")
    for module in modules:
        module.project_transport_gain_()


def _reject_instance_forward_mutation_or_hooks(model: nn.Module) -> None:
    shadowed_names = (
        "forward",
        "state_dict",
        "named_parameters",
        "parameters",
    )
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
            raise RuntimeError(
                f"instance method shadow detected at {location!r}"
            )
        if any(bool(getattr(module, attribute, None)) for attribute in hook_names):
            raise RuntimeError(f"module hooks are forbidden at {location!r}")


def validate_sctransnet_sbsc_v2(
    model: nn.Module,
    require_zero_gain: bool = False,
) -> dict[str, Any]:
    """Validate the frozen four-block graph and its bounded gain state."""

    if type(require_zero_gain) is not bool:
        raise TypeError("require_zero_gain must be bool")
    if type(model) is not SCTransNet:
        raise TypeError("validator requires the exact unwrapped SCTransNet class")
    _reject_instance_forward_mutation_or_hooks(model)
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("model must retain exactly four SCTBs")
    modules = tuple(layer.channel_attn for layer in layers)
    if any(
        type(module) is not SupportBalancedConditionalCrossMomentV2
        for module in modules
    ):
        raise TypeError("each SCTB site must contain the exact SBSC V2 class")
    discovered = _sbsc_v2_modules(model)
    if len(discovered) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("model must contain exactly four SBSC V2 blocks")
    if tuple(id(module) for module in discovered) != tuple(
        id(module) for module in modules
    ):
        raise RuntimeError("SBSC V2 modules are outside the four SCTB sites")
    expected_channels = (32, 64, 128, 256)
    for index, module in enumerate(modules):
        if module.layer_index != index:
            raise RuntimeError("SBSC V2 layer indices differ from encoder order")
        if module.detach_support is not True:
            raise RuntimeError("SBSC V2 support path must be detached")
        if module.eps != _FROZEN_EPS:
            raise RuntimeError("SBSC V2 eps contract differs")
        if (
            module.absolute_confidence_temperature
            != _FROZEN_ABSOLUTE_CONFIDENCE_TEMPERATURE
        ):
            raise RuntimeError(
                "SBSC V2 absolute confidence temperature differs"
            )
        if module.gain_limit != _FROZEN_GAIN_MAX:
            raise RuntimeError("SBSC V2 gain bound differs")
        if module.num_attention_heads != 1:
            raise RuntimeError("SBSC V2 must retain one attention head")
        if module.KV_size != 480:
            raise RuntimeError("SBSC V2 KV_size must remain 480")
        if tuple(module.channel_num) != expected_channels:
            raise RuntimeError("SBSC V2 channel contract differs")
        if module.vis is not False:
            raise RuntimeError("SBSC V2 freezes vis=False")
        if type(module.psi) is not nn.InstanceNorm2d:
            raise TypeError("SBSC V2 must retain exact InstanceNorm2d")
        if (
            module.psi.num_features != 1
            or module.psi.eps != 1e-5
            or module.psi.momentum != 0.1
            or module.psi.affine is not False
            or module.psi.track_running_stats is not False
        ):
            raise RuntimeError("SBSC V2 InstanceNorm2d contract differs")
        if type(module.softmax) is not nn.Softmax or module.softmax.dim != 3:
            raise TypeError("SBSC V2 must retain exact Softmax(dim=3)")
        gain = module._parameters.get(_FROZEN_GAIN_SUFFIX)
        if gain is not module.raw_transport_gain or tuple(gain.shape) != ():
            raise RuntimeError("SBSC V2 gain registration differs")
    state_validation = validate_sbsc_v2_state_dict(
        model.state_dict(), "sbsc_v2"
    )
    if require_zero_gain and any(
        torch.count_nonzero(model.state_dict()[key]).item() != 0
        for key in _GAIN_STATE_KEYS
    ):
        raise RuntimeError("SBSC V2 identity gains are not exactly zero")
    if _parameter_count(model) != _FROZEN_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("SBSC V2 parameter-count contract differs")
    return {
        "schema": SBSC_V2_SCHEMA,
        "state_key_count": len(model.state_dict()),
        "parameter_count": _parameter_count(model),
        "gain_state_keys": list(_GAIN_STATE_KEYS),
        "gain_bounds": [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX],
        "support_source": "post_spatial_l2_normalized_shared_K",
        "support_stop_gradient": True,
        "absolute_confidence_temperature": (
            _FROZEN_ABSOLUTE_CONFIDENCE_TEMPERATURE
        ),
        "conditional_cross_moment": True,
        "transport_row_sum_zero": True,
        "transport_row_l1_bound": 1.0,
        "encoder_changed_outside_ssca": False,
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


def validate_sbsc_v2_state_dict(
    state: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Validate checkpoint architecture/gain state before loading a model."""

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

    authority_records = _authority_state_schema()
    expected_schema = {
        key: (shape, dtype) for key, shape, dtype in authority_records
    }
    if method == "sbsc_v2":
        expected_schema.update(
            {
                key: ((), torch.float32)
                for key in _GAIN_STATE_KEYS
            }
        )
    if set(canonical) != set(expected_schema):
        missing = sorted(set(expected_schema) - set(canonical))
        unexpected = sorted(set(canonical) - set(expected_schema))
        raise ValueError(
            "state key set differs from the exact architecture schema; "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    for key, (expected_shape, expected_dtype) in expected_schema.items():
        value = canonical[key]
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"state shape differs for {key!r}")
        if value.dtype != expected_dtype:
            raise TypeError(f"state dtype differs for {key!r}")

    actual = {
        f"{prefix}{key}" for key in _GAIN_STATE_KEYS if key in canonical
    }
    gain_values: list[float] = []
    for canonical_key in _GAIN_STATE_KEYS:
        if canonical_key not in canonical:
            continue
        value = canonical[canonical_key]
        if not torch.isfinite(value).all():
            raise ValueError(
                f"gain state {prefix + canonical_key!r} must be finite"
            )
        scalar = float(value.detach().cpu().reshape(()).item())
        if not _FROZEN_GAIN_MIN <= scalar <= _FROZEN_GAIN_MAX:
            raise ValueError(
                f"gain state {prefix + canonical_key!r} is outside [0, 0.25]"
            )
        gain_values.append(scalar)
    return {
        "method": method,
        "state_key_count": len(tensors),
        "gain_state_keys": sorted(actual),
        "gain_values": gain_values,
        "gain_bounds": [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX],
        "data_parallel_prefix": bool(prefix),
    }


def structurally_inactive_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Return SCTransNet parameters registered but absent from its forward."""

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
    # 4 unused positional parameters + 16 unused scalars in each of 4 SSCA
    # blocks.  Fail closed if upstream SCTransNet changes this contract.
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
    gain_keys = [key for key in _GAIN_STATE_KEYS if key in state]
    return {
        "schema": SBSC_V2_SCHEMA,
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
        "gain_bounds": [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX],
    }


def build_paired_sctransnet_sbsc_v2(
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
) -> tuple[SCTransNet, SCTransNet, dict[str, Any]]:
    """Build bitwise-paired seed-42 SCTransNet and SBSC V2 from scratch."""

    dataset = _require_dataset(dataset)
    architecture_seed = _require_architecture_seed(architecture_seed)
    baseline = _construct_original(architecture_seed)
    _cache_authority_state_schema(baseline)
    candidate = copy.deepcopy(baseline)
    replace_sctransnet_ssca_with_sbsc_v2(candidate)

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    if set(candidate_state) - set(baseline_state) != set(_GAIN_STATE_KEYS):
        raise RuntimeError("candidate adds state outside the four V2 gains")
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
    validate_sctransnet_sbsc_v2(candidate, require_zero_gain=True)
    validate_sbsc_v2_state_dict(baseline_state, "sctransnet")

    baseline_metadata = _method_metadata(
        baseline,
        method="sctransnet",
        dataset=dataset,
        architecture_seed=architecture_seed,
        shared_state_sha256=shared_hash,
    )
    candidate_metadata = _method_metadata(
        candidate,
        method="sbsc_v2",
        dataset=dataset,
        architecture_seed=architecture_seed,
        shared_state_sha256=shared_hash,
    )
    metadata = dict(candidate_metadata)
    metadata.update(
        {
            "pair_schema": SBSC_V2_SCHEMA,
            "baseline": baseline_metadata,
            "candidate": candidate_metadata,
            "shared_state_key_count": len(baseline_state),
            "shared_state_bitwise_equal": True,
            "parent_checkpoint": None,
            "warm_start_used": False,
            "model_construction_preserves_caller_rng_stream": True,
        }
    )
    return baseline, candidate, metadata


def build_sctransnet_sbsc_v2_method(
    method: str,
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
) -> tuple[SCTransNet, dict[str, Any]]:
    """Build the complete pair first, then return exactly one requested role."""

    method = _require_method(method)
    if type(training) is not bool:
        raise TypeError("training must be bool")
    baseline, candidate, pair_metadata = build_paired_sctransnet_sbsc_v2(
        dataset,
        architecture_seed,
    )
    model = baseline if method == "sctransnet" else candidate
    model.train(training)
    model.mode = "train" if training else "test"
    metadata = dict(pair_metadata["baseline" if method == "sctransnet" else "candidate"])
    metadata.update(
        {
            "training": training,
            "pair": pair_metadata,
            "test_split_accessed": False,
        }
    )
    return model, metadata


def _validate_intervention_structure(
    intervention: SBSCV2Intervention,
) -> None:
    if not isinstance(intervention, SBSCV2Intervention):
        raise TypeError("intervention must be SBSCV2Intervention")
    if intervention.mode not in _INTERVENTION_MODES:
        raise ValueError(f"unsupported intervention mode: {intervention.mode!r}")
    if intervention.mode == "shuffle":
        mapping = intervention.spatial_permutations
        if not isinstance(mapping, Mapping) or set(mapping) != set(range(4)):
            raise ValueError("shuffle requires permutations for layers 0,1,2,3")
    elif intervention.spatial_permutations is not None:
        raise ValueError("spatial_permutations are valid only for shuffle")
    if intervention.mode == "cross_image":
        if not isinstance(intervention.batch_permutation, torch.Tensor):
            raise ValueError("cross_image requires batch_permutation")
    elif intervention.batch_permutation is not None:
        raise ValueError("batch_permutation is valid only for cross_image")


def forward_sctransnet_sbsc_v2_intervention(
    model: SCTransNet,
    inputs: torch.Tensor,
    intervention: SBSCV2Intervention,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """Run one full-graph diagnostic without mutating modules or parameters."""

    validate_sctransnet_sbsc_v2(model, require_zero_gain=False)
    if model.training:
        raise RuntimeError(
            "SBSC V2 diagnostics require model.eval() so BatchNorm state "
            "cannot change"
        )
    _validate_intervention_structure(intervention)
    modules = _sbsc_v2_modules(model)
    context = _SBSCV2ForwardContext(
        module_ids=frozenset(id(module) for module in modules),
        intervention=intervention,
        diagnostics={},
    )
    token = _FORWARD_CONTEXT.set(context)
    try:
        outputs = model(inputs)
        if set(context.diagnostics) != set(range(4)):
            raise RuntimeError("diagnostic forward did not execute all four blocks")
        diagnostics = tuple(context.diagnostics[index] for index in range(4))
    finally:
        _FORWARD_CONTEXT.reset(token)
    return outputs, diagnostics


# Short aliases useful in notebooks while preserving explicit runner APIs.
SBSCV2 = SupportBalancedConditionalCrossMomentV2
forward_sbsc_v2_intervention = forward_sctransnet_sbsc_v2_intervention


__all__ = [
    "ARCHITECTURE_SEED",
    "EXPECTED_SBSC_V2_BLOCKS",
    "EXPECTED_SBSC_V2_PARAMETER_COUNT",
    "EXPECTED_SBSC_V2_STATE_KEY_COUNT",
    "SBSCV2",
    "SBSCV2Intervention",
    "SBSCV2Support",
    "SBSC_V2_ABSOLUTE_CONFIDENCE_TEMPERATURE",
    "SBSC_V2_EPS",
    "SBSC_V2_GAIN_MAX",
    "SBSC_V2_GAIN_MIN",
    "SBSC_V2_GAIN_SUFFIX",
    "SBSC_V2_SCHEMA",
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
    "SupportBalancedConditionalCrossMomentV2",
    "build_paired_sctransnet_sbsc_v2",
    "build_sctransnet_sbsc_v2_method",
    "estimate_sbsc_v2_support",
    "forward_sbsc_v2_intervention",
    "forward_sctransnet_sbsc_v2_intervention",
    "project_sbsc_v2_gains_",
    "replace_sctransnet_ssca_with_sbsc_v2",
    "structurally_inactive_parameter_names",
    "validate_sbsc_v2_state_dict",
    "validate_sctransnet_sbsc_v2",
]
