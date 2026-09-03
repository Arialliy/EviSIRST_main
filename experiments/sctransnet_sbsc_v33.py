#!/usr/bin/env python3
"""SCTransNet C3-SBSC V3.3 role-exclusive projection core.

V3.3 keeps the frozen V3.1 numerical solver and the V3.2 router topology, but
changes the learned evidence semantics in three coupled steps: token-wise
exclusive C/H/B roles, mass-aware role availability, and explicit routing of
dual, hard-only, background-only, and identity rows.  Construction is paired
at architecture seed 42 and never consumes the caller's RNG stream.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from experiments import sctransnet_sbsc_v31 as v31
from experiments import sctransnet_sbsc_v32 as v32
from experiments.four_dataset_models_seed42_v1 import (
    _construct_original,
    state_dict_sha256,
)
from model.SCTransNet import Attention_org, SCTransNet


SBSC_V33_SCHEMA = (
    "sctransnet_sbsc_v33/role_exclusive_mass_aware_mode_routed/v1"
)
SBSC_V33_MODEL_NAME = "SCTransNet-C3-SBSC-V3.3"

ARCHITECTURE_SEED = 42
SUPPORTED_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
SUPPORTED_METHODS = (
    "sctransnet",
    "sbsc_v33_third",
    "sbsc_v33_ord",
    "sbsc_v33_half",
)
SUPPORTED_GRADIENT_MODES = ("live", "detached")
SUPPORTED_BALANCE_MODES = (
    "ordinary",
    "half_half",
    "one_third_two_thirds",
)

SBSC_V33_GAIN_MIN = v32.SBSC_V32_GAIN_MIN
SBSC_V33_GAIN_MAX = v32.SBSC_V32_GAIN_MAX
SBSC_V33_EPS = v32.SBSC_V32_EPS
SBSC_V33_GAIN_SUFFIX = v32.SBSC_V32_GAIN_SUFFIX
SBSC_V33_GAIN_STATE_KEY = v32.SBSC_V32_GAIN_STATE_KEY
SBSC_V33_ROUTER_PARAMETER_COUNT = v32.SBSC_V32_ROUTER_PARAMETER_COUNT
SBSC_V33_LOSS_SCHEMA = (
    "sum_of_six_BCELoss_mean_terms_plus_unit_weight_"
    "mean_of_four_role_axis_router_ce"
)

_FROZEN_BLOCK_COUNT = 4
_FROZEN_REPLACED_BLOCK_INDEX = 1
_FROZEN_BASELINE_STATE_KEY_COUNT = 510
_FROZEN_BASELINE_PARAMETER_COUNT = 11_325_939
_FROZEN_CANDIDATE_STATE_KEY_COUNT = 513
_FROZEN_CANDIDATE_PARAMETER_COUNT = 11_330_188
_FROZEN_ROUTER_PREFIX = (
    "mtc.encoder.layer.1.channel_attn.exclusive_tri_router."
)
_FROZEN_ROUTER_STATE_SCHEMA = (
    (_FROZEN_ROUTER_PREFIX + "value_proj.weight", (8, 480, 1, 1)),
    (_FROZEN_ROUTER_PREFIX + "head.weight", (3, 15, 3, 3)),
)
SBSC_V33_ROUTER_STATE_KEYS = tuple(
    key for key, _shape in _FROZEN_ROUTER_STATE_SCHEMA
)
EXPECTED_SBSC_V33_ROUTER_INIT_SHA256 = (
    v32.EXPECTED_SBSC_V32_ROUTER_INIT_SHA256
)
EXPECTED_SBSC_V33_STATE_KEY_COUNT = _FROZEN_CANDIDATE_STATE_KEY_COUNT
EXPECTED_SBSC_V33_PARAMETER_COUNT = _FROZEN_CANDIDATE_PARAMETER_COUNT

MODE_IDENTITY = 0
MODE_DUAL = 1
MODE_HARD_ONLY = 2
MODE_BACKGROUND_ONLY = 3
MODE_NAMES = (
    "identity",
    "dual",
    "hard_only",
    "background_only",
)


@dataclass(frozen=True)
class RoleTargetsV33:
    """Detached token-wise C/H/B target simplex."""

    probability: torch.Tensor
    target_region: torch.Tensor
    background_region: torch.Tensor
    token_hw: tuple[int, int]


@dataclass(frozen=True)
class RoutedLevelV33:
    """One level's V3.1-compatible support plus V3.3 audit tensors."""

    support: v31.C3V31LevelSupport
    logits: torch.Tensor
    role_probability: torch.Tensor
    role_mass: torch.Tensor
    integrated_winning_evidence: torch.Tensor
    support_weighted_existence: torch.Tensor
    availability: torch.Tensor


@dataclass
class C3V33TrainingRouterCollector:
    """Context-local ledger for one gradient-enabled training forward."""

    records: list[dict[str, Any]]


@dataclass(frozen=True)
class RouterLossBreakdownV33:
    """Exact four-level mean plus auditable per-image/per-role terms."""

    total: torch.Tensor
    per_level: tuple[torch.Tensor, ...]
    per_level_per_image: tuple[torch.Tensor, ...]
    per_level_per_role: tuple[torch.Tensor, ...]
    level_reduction: str
    balance_mode: str


@dataclass(frozen=True)
class _C3V33TrainingRouterRequest:
    module_id: int
    collector: C3V33TrainingRouterCollector


_C3_V33_TRAINING_ROUTER_REQUEST: ContextVar[
    _C3V33TrainingRouterRequest | None
] = ContextVar("c3_v33_training_router_request", default=None)


def v31_ambient_emission_tolerance(dtype: torch.dtype) -> float:
    """Return the exact ambient-dtype tolerance frozen by V3.1."""

    if dtype in (torch.float16, torch.bfloat16):
        return max(SBSC_V33_EPS, 4.0 * torch.finfo(dtype).eps)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"unsupported attention dtype: {dtype}")
    return SBSC_V33_EPS


def router_value_input_v33(
    value_spatial: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Apply the globally authorized V-branch gradient boundary."""

    if not isinstance(value_spatial, torch.Tensor) or value_spatial.ndim != 4:
        raise TypeError("value_spatial must be a 4D tensor")
    if mode == "live":
        return value_spatial.float()
    if mode == "detached":
        return value_spatial.detach().float()
    raise ValueError(f"unsupported router value-gradient mode: {mode!r}")


class _RoleExclusiveTriEvidenceRouterV33(nn.Module):
    """The unchanged 4,245-parameter V3.2 topology under a new state prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.value_proj = nn.Conv2d(480, 8, kernel_size=1, bias=False)
        self.activation = nn.SiLU()
        self.head = nn.Conv2d(15, 3, kernel_size=3, padding=1, bias=False)

    def encode_value(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.value_proj(value))

    def route(
        self,
        encoded_value: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> torch.Tensor:
        if encoded_value.ndim != 4 or descriptor.ndim != 4:
            raise ValueError("router inputs must be 4D spatial tensors")
        if encoded_value.shape[:1] + encoded_value.shape[-2:] != (
            descriptor.shape[:1] + descriptor.shape[-2:]
        ):
            raise ValueError("router value/descriptor geometry differs")
        if encoded_value.shape[1] != 8 or descriptor.shape[1] != 7:
            raise ValueError("router requires exactly 8 live and 7 static channels")
        return self.head(torch.cat((encoded_value, descriptor), dim=1))


def _mass_aware_role_statistics(
    logits: torch.Tensor,
    *,
    finite_row: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert role-axis logits into normalized supports and availability."""

    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("V3.3 router logits must have shape [B,3,H,W]")
    if finite_row.shape != (logits.shape[0],):
        raise ValueError("finite_row must contain one flag per image")
    safe_logits = torch.where(
        finite_row[:, None, None, None],
        logits,
        torch.zeros_like(logits),
    )
    probability = F.softmax(safe_logits, dim=1)
    flat = probability.flatten(2)
    other_max = torch.stack(
        (
            flat[:, 1:].amax(dim=1),
            flat[:, (0, 2)].amax(dim=1),
            flat[:, :2].amax(dim=1),
        ),
        dim=1,
    )
    margin = (flat - other_max).clamp_min(0.0)
    mass = flat.sum(dim=-1)
    integrated = (flat * margin).sum(dim=-1)
    existence = integrated / mass.clamp_min(SBSC_V33_EPS)
    positions = int(flat.shape[-1])
    availability = (
        finite_row[:, None]
        & mass.ge(1.0)
        & integrated.ge(1.0 / float(positions))
    )
    normalized = flat / mass[:, :, None].clamp_min(SBSC_V33_EPS)
    uniform = torch.full_like(normalized, 1.0 / float(positions))
    support = torch.where(availability[:, :, None], normalized, uniform)
    if not bool(torch.isfinite(probability).all()):
        raise RuntimeError("role-axis probability is non-finite")
    if not bool(
        probability.sum(dim=1).sub(1.0).abs().le(2.0e-6).all()
    ):
        raise RuntimeError("role-axis probabilities do not form a simplex")
    if not bool(
        support.sum(dim=-1).sub(1.0).abs().le(2.0e-6).all()
    ):
        raise RuntimeError("spatial role supports do not sum to one")
    return probability, support, mass, integrated, existence, availability


class RoleExclusiveTriEvidenceProjectionV33(v31.C3DualRiskProjectionV31):
    """Layer-1 SSCA replacement implementing the V3.3 evidence chain."""

    def __init__(
        self,
        config: Any,
        vis: bool,
        channel_num: list[int] | tuple[int, ...],
        *,
        layer_index: int,
        router_value_gradient_mode: str,
        gain_limit: float = SBSC_V33_GAIN_MAX,
        eps: float = SBSC_V33_EPS,
        detach_support: bool = True,
    ) -> None:
        if router_value_gradient_mode not in SUPPORTED_GRADIENT_MODES:
            raise ValueError("router_value_gradient_mode must be live or detached")
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(ARCHITECTURE_SEED)
            super().__init__(
                config,
                vis,
                channel_num,
                layer_index=layer_index,
                gain_limit=gain_limit,
                eps=eps,
                detach_support=detach_support,
            )
            self.exclusive_tri_router = _RoleExclusiveTriEvidenceRouterV33()
        self.router_value_gradient_mode = router_value_gradient_mode

    @classmethod
    def from_ssca(
        cls,
        source: Attention_org,
        *,
        layer_index: int,
        router_value_gradient_mode: str,
    ) -> "RoleExclusiveTriEvidenceProjectionV33":
        if type(source) is not Attention_org:
            raise TypeError("source must be the exact frozen Attention_org class")
        if type(layer_index) is not int or layer_index != 1:
            raise ValueError("V3.3 replaces only zero-based SCTB layer 1")
        if router_value_gradient_mode not in SUPPORTED_GRADIENT_MODES:
            raise ValueError("router_value_gradient_mode must be live or detached")
        config = SimpleNamespace(KV_size=int(source.KV_size))
        reference = next(source.parameters())
        replacement = cls(
            config,
            bool(source.vis),
            tuple(int(value) for value in source.channel_num),
            layer_index=layer_index,
            router_value_gradient_mode=router_value_gradient_mode,
        )
        router_state = {
            key: value.detach().clone()
            for key, value in replacement.exclusive_tri_router.state_dict().items()
        }
        replacement.to(device=reference.device, dtype=reference.dtype)
        incompatible = replacement.load_state_dict(source.state_dict(), strict=False)
        expected_missing = {
            SBSC_V33_GAIN_SUFFIX,
            "exclusive_tri_router.value_proj.weight",
            "exclusive_tri_router.head.weight",
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise RuntimeError(
                "unexpected missing V3.3 replacement state: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected legacy V3.3 replacement state: "
                f"{incompatible.unexpected_keys}"
            )
        replacement.exclusive_tri_router.to(
            device=reference.device,
            dtype=torch.float32,
        )
        replacement.exclusive_tri_router.load_state_dict(router_state, strict=True)
        with torch.no_grad():
            replacement.raw_dual_risk_level_gain.data = (
                replacement.raw_dual_risk_level_gain.detach().float()
            )
            replacement.raw_dual_risk_level_gain.zero_()
        if (
            torch.count_nonzero(
                replacement.exclusive_tri_router.value_proj.weight
            ).item()
            == 0
            or torch.count_nonzero(
                replacement.exclusive_tri_router.head.weight
            ).item()
            == 0
        ):
            raise RuntimeError("V3.3 router must have non-zero initialization")
        replacement.train(source.training)
        return replacement

    @staticmethod
    def _spatial_energy(raw: torch.Tensor) -> torch.Tensor:
        return v32.LearnedTriEvidenceProjectionV32._spatial_energy(raw)

    def _route_supports(
        self,
        *,
        static_support: v31.C3V31Support,
        raw_queries: tuple[torch.Tensor, ...],
        raw_key: torch.Tensor,
        value_spatial: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> tuple[RoutedLevelV33, ...]:
        height, width = token_hw
        positions = height * width
        if raw_key.shape[-1] != positions or value_spatial.shape[-2:] != token_hw:
            raise ValueError("router token geometry differs")
        value_input = router_value_input_v33(
            value_spatial,
            mode=self.router_value_gradient_mode,
        )
        with torch.autocast(device_type=value_spatial.device.type, enabled=False):
            encoded_value = self.exclusive_tri_router.encode_value(value_input)
            key_energy = self._spatial_energy(raw_key)
            routed: list[RoutedLevelV33] = []
            for level, raw_query in zip(
                static_support.levels,
                raw_queries,
                strict=True,
            ):
                query_energy = self._spatial_energy(raw_query)

                def spatial(value: torch.Tensor) -> torch.Tensor:
                    if tuple(value.shape[-2:]) != (1, positions):
                        raise ValueError("LOO descriptor geometry differs")
                    return value.reshape(value.shape[0], 1, height, width)

                descriptor = torch.cat(
                    (
                        spatial(level.peer_consensus.detach().float()),
                        spatial(level.peer_dispersion.detach().float()),
                        spatial(level.agreement.detach().float()),
                        spatial(
                            static_support.normalized_positive_rarity.detach().float()
                        ),
                        spatial(
                            static_support.normalized_background_rarity.detach().float()
                        ),
                        spatial(query_energy),
                        spatial(key_energy),
                    ),
                    dim=1,
                ).detach()
                logits = self.exclusive_tri_router.route(
                    encoded_value,
                    descriptor,
                ).float()
                finite_row = torch.isfinite(logits).flatten(1).all(dim=1)
                (
                    probability,
                    support,
                    mass,
                    integrated,
                    existence,
                    availability,
                ) = _mass_aware_role_statistics(logits, finite_row=finite_row)

                support4 = support.unsqueeze(2)
                mass4 = mass[:, :, None, None]
                availability4 = availability[:, :, None, None]
                existence4 = existence[:, :, None, None]
                d_cb = 0.5 * (
                    support4[:, 0:1] - support4[:, 2:3]
                ).abs().sum(dim=-1, keepdim=True)
                reliability = torch.where(
                    availability4[:, 0:1]
                    & availability4[:, 2:3]
                    & static_support.key_confidence_valid,
                    (
                        static_support.key_confidence
                        * existence4[:, 0:1]
                        * d_cb
                    ).clamp_min(0.0).pow(1.0 / 3.0),
                    torch.zeros_like(mass4[:, 0:1]),
                ).clamp(0.0, 1.0)
                routed_support = replace(
                    level,
                    consistent_raw=probability[:, 0:1].flatten(2).unsqueeze(1),
                    contradictory_raw=probability[:, 1:2].flatten(2).unsqueeze(1),
                    common_raw=probability[:, 2:3].flatten(2).unsqueeze(1),
                    consistent_mass=mass4[:, 0:1],
                    contradictory_mass=mass4[:, 1:2],
                    common_mass=mass4[:, 2:3],
                    consistent_support=support4[:, 0:1],
                    contradictory_support=support4[:, 1:2],
                    common_support=support4[:, 2:3],
                    consistent_valid=availability4[:, 0:1],
                    contradictory_valid=availability4[:, 1:2],
                    common_valid=availability4[:, 2:3],
                    consistent_positive_strength=existence4[:, 0:1],
                    consistent_common_separation=d_cb,
                    reliability=reliability,
                )
                routed.append(
                    RoutedLevelV33(
                        support=routed_support,
                        logits=logits,
                        role_probability=probability,
                        role_mass=mass,
                        integrated_winning_evidence=integrated,
                        support_weighted_existence=existence,
                        availability=availability,
                    )
                )
        return tuple(routed)

    @staticmethod
    def _slice_level_support(
        support: v31.C3V31LevelSupport,
        indices: torch.Tensor,
    ) -> v31.C3V31LevelSupport:
        updates: dict[str, Any] = {}
        for field in fields(v31.C3V31LevelSupport):
            value = getattr(support, field.name)
            if isinstance(value, torch.Tensor):
                updates[field.name] = value.index_select(0, indices)
        return replace(support, **updates)

    def _project_available_subset(
        self,
        *,
        eligible_images: torch.Tensor,
        level_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        base_attention: torch.Tensor,
        support: v31.C3V31LevelSupport,
        key_confidence: torch.Tensor,
        key_confidence_valid: torch.Tensor,
        gain: torch.Tensor,
        intervention: str,
    ) -> tuple[
        torch.Tensor,
        dict[str, Any],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run one frozen solver branch only on numerically eligible images."""

        if (
            eligible_images.ndim != 1
            or eligible_images.shape[0] != base_attention.shape[0]
            or eligible_images.dtype is not torch.bool
        ):
            raise ValueError("eligible_images must be one bool per batch item")
        row_shape = base_attention.shape[:-1] + (1,)
        emitted_ok = torch.zeros(
            row_shape,
            device=base_attention.device,
            dtype=torch.bool,
        )
        emission_fallback = torch.ones_like(emitted_ok)
        hard_defined = torch.zeros_like(emitted_ok)
        background_defined = torch.zeros_like(emitted_ok)
        accepted = torch.zeros_like(emitted_ok)
        solver_fallback = torch.zeros_like(emitted_ok)
        indices = torch.nonzero(eligible_images, as_tuple=False).flatten()
        if indices.numel() == 0:
            return (
                base_attention,
                {
                    "schema": SBSC_V33_SCHEMA + "/empty_branch/v1",
                    "intervention": intervention,
                    "subset_indices": indices,
                    "emitted_certificate_ok": emitted_ok,
                    "emission_fallback": emission_fallback,
                    "projection_accepted": accepted,
                    "projection": None,
                },
                hard_defined,
                background_defined,
                accepted,
                solver_fallback,
            )
        subset_support = self._slice_level_support(support, indices)
        subset_attention, subset_diagnostics = self._project_level(
            level_index=level_index,
            query=query.index_select(0, indices),
            key=key.index_select(0, indices),
            base_attention=base_attention.index_select(0, indices),
            support=subset_support,
            key_confidence=key_confidence.index_select(0, indices),
            key_confidence_valid=key_confidence_valid.index_select(0, indices),
            gain=gain,
            gain_override=None,
            intervention=intervention,
        )
        attention = base_attention.index_copy(0, indices, subset_attention)
        subset_emitted = subset_diagnostics.get("emitted_certificate_ok")
        subset_fallback = subset_diagnostics.get("emission_fallback")
        if (
            not isinstance(subset_emitted, torch.Tensor)
            or not isinstance(subset_fallback, torch.Tensor)
        ):
            raise RuntimeError("subset projection omitted emission certificates")
        emitted_ok = emitted_ok.index_copy(0, indices, subset_emitted.detach())
        emission_fallback = emission_fallback.index_copy(
            0,
            indices,
            subset_fallback.detach(),
        )
        projection = subset_diagnostics.get("projection")
        if projection is not None:
            hard_defined = hard_defined.index_copy(
                0,
                indices,
                projection.hard_defined.detach(),
            )
            background_defined = background_defined.index_copy(
                0,
                indices,
                projection.background_defined.detach(),
            )
            accepted = accepted.index_copy(
                0,
                indices,
                projection.accepted.detach(),
            )
            solver_fallback = solver_fallback.index_copy(
                0,
                indices,
                projection.solver_fallback.detach(),
            )
        branch_diagnostics = {
            "schema": SBSC_V33_SCHEMA + "/subset_branch/v1",
            "intervention": intervention,
            "subset_indices": indices,
            "subset_diagnostics": subset_diagnostics,
            "emitted_certificate_ok": emitted_ok,
            "emission_fallback": emission_fallback,
            "projection_accepted": accepted,
            "projection": projection,
        }
        return (
            attention,
            branch_diagnostics,
            hard_defined,
            background_defined,
            accepted,
            solver_fallback,
        )

    def _project_selected_rows(
        self,
        *,
        selected_rows: torch.Tensor,
        level_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        base_attention: torch.Tensor,
        support: v31.C3V31LevelSupport,
        key_confidence: torch.Tensor,
        key_confidence_valid: torch.Tensor,
        gain: torch.Tensor,
        intervention: str,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
        """Run a differentiable branch on exactly the emitted attention rows."""

        row_shape = base_attention.shape[:-1] + (1,)
        if (
            selected_rows.shape != row_shape
            or selected_rows.dtype is not torch.bool
            or selected_rows.device != base_attention.device
        ):
            raise ValueError("selected_rows must match the attention row geometry")
        batch, heads, row_count, attention_positions = base_attention.shape
        if heads != 1:
            raise RuntimeError("V3.3 freezes one attention head")
        token_positions = int(query.shape[-1])
        if (
            query.shape[:3] != (batch, 1, row_count)
            or key.shape[0] != batch
            or key.shape[1] != 1
            or key.shape[-1] != token_positions
        ):
            raise ValueError("selected-row Q/K geometry differs")
        flat_selected = selected_rows.reshape(batch * row_count)
        flat_indices = torch.nonzero(flat_selected, as_tuple=False).flatten()
        emitted_ok = torch.zeros(
            row_shape,
            device=base_attention.device,
            dtype=torch.bool,
        )
        emission_fallback = torch.ones_like(emitted_ok)
        accepted = torch.zeros_like(emitted_ok)
        solver_fallback = torch.zeros_like(emitted_ok)
        if flat_indices.numel() == 0:
            return (
                base_attention,
                {
                    "schema": SBSC_V33_SCHEMA + "/empty_selected_rows/v1",
                    "intervention": intervention,
                    "flat_row_indices": flat_indices,
                    "emitted_certificate_ok": emitted_ok,
                    "emission_fallback": emission_fallback,
                    "projection_accepted": accepted,
                    "projection": None,
                },
                accepted,
                solver_fallback,
            )
        sample_indices = torch.div(
            flat_indices,
            row_count,
            rounding_mode="floor",
        )
        flat_query = query.permute(0, 2, 1, 3).reshape(
            batch * row_count,
            1,
            1,
            token_positions,
        )
        flat_base = base_attention.permute(0, 2, 1, 3).reshape(
            batch * row_count,
            1,
            1,
            attention_positions,
        )
        subset_support = self._slice_level_support(support, sample_indices)
        subset_attention, subset_diagnostics = self._project_level(
            level_index=level_index,
            query=flat_query.index_select(0, flat_indices),
            key=key.index_select(0, sample_indices),
            base_attention=flat_base.index_select(0, flat_indices),
            support=subset_support,
            key_confidence=key_confidence.index_select(0, sample_indices),
            key_confidence_valid=key_confidence_valid.index_select(
                0,
                sample_indices,
            ),
            gain=gain,
            gain_override=None,
            intervention=intervention,
        )
        # The frozen solver contains discrete certificates/root-search paths
        # whose formal forward is correct but whose unused zero-gradient
        # branches can yield 0*NaN during autograd.  Preserve its emitted
        # tensor bit-for-bit, detach the numerical solver backward, and attach
        # a stable conditional-consistent surrogate.  Gain receives the exact
        # detached Ahat-base direction; Q/K/router support receive the smooth
        # conditional-attention direction only on actually selected rows.
        projection = subset_diagnostics.get("projection")
        actual_ahat = subset_diagnostics.get("Ahat")
        if projection is not None and isinstance(actual_ahat, torch.Tensor):
            subset_query = flat_query.index_select(0, flat_indices)
            subset_key = key.index_select(0, sample_indices)
            subset_base = flat_base.index_select(0, flat_indices)
            with torch.autocast(
                device_type=subset_base.device.type,
                enabled=False,
            ):
                base_fp32 = subset_base.float()
                row_mass = base_fp32.sum(dim=-1, keepdim=True)
                proxy_q = self._conditional_attention(
                    subset_query.float(),
                    subset_key.float(),
                    subset_support.consistent_support,
                )
                proxy_q = proxy_q / proxy_q.sum(dim=-1, keepdim=True)
                proxy_ahat = row_mass * proxy_q
                accepted_mask = projection.accepted.detach()
                exact_ahat = torch.where(
                    accepted_mask,
                    actual_ahat.detach().float(),
                    base_fp32.detach(),
                )
                effective_gain = gain[level_index].float()
                surrogate_fp32 = base_fp32 + effective_gain * (
                    exact_ahat
                    - base_fp32
                    + proxy_ahat
                    - proxy_ahat.detach()
                )
                surrogate = subset_base + (
                    surrogate_fp32 - base_fp32
                ).to(dtype=subset_base.dtype)
            subset_attention = (
                subset_attention.detach()
                + surrogate
                - surrogate.detach()
            )
        flat_attention = flat_base.index_copy(
            0,
            flat_indices,
            subset_attention,
        )
        attention = flat_attention.reshape(
            batch,
            row_count,
            1,
            attention_positions,
        ).permute(0, 2, 1, 3)
        subset_emitted = subset_diagnostics.get("emitted_certificate_ok")
        subset_fallback = subset_diagnostics.get("emission_fallback")
        if (
            not isinstance(subset_emitted, torch.Tensor)
            or not isinstance(subset_fallback, torch.Tensor)
        ):
            raise RuntimeError("selected-row branch omitted emission certificates")

        def scatter_flag(value: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
            flat = base.permute(0, 2, 1, 3).reshape(batch * row_count, 1, 1, 1)
            scattered = flat.index_copy(0, flat_indices, value.detach())
            return scattered.reshape(batch, row_count, 1, 1).permute(0, 2, 1, 3)

        emitted_ok = scatter_flag(subset_emitted, emitted_ok)
        emission_fallback = scatter_flag(subset_fallback, emission_fallback)
        if projection is not None:
            accepted = scatter_flag(projection.accepted, accepted)
            solver_fallback = scatter_flag(
                projection.solver_fallback,
                solver_fallback,
            )
        branch_diagnostics = {
            "schema": SBSC_V33_SCHEMA + "/selected_rows_branch/v1",
            "intervention": intervention,
            "flat_row_indices": flat_indices,
            "sample_indices": sample_indices,
            "subset_diagnostics": subset_diagnostics,
            "emitted_certificate_ok": emitted_ok,
            "emission_fallback": emission_fallback,
            "projection_accepted": accepted,
            "projection": projection,
            "backward_surrogate": "consistent_conditional_straight_through_v1",
        }
        return attention, branch_diagnostics, accepted, solver_fallback

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
        request = _C3_V33_TRAINING_ROUTER_REQUEST.get()
        context_capture = request is not None and request.module_id == id(self)
        if context_capture and (not self.training or not torch.is_grad_enabled()):
            raise RuntimeError(
                "V3.3 router capture requires a gradient-enabled training forward"
            )
        batch, _channels, height, width = emb1.shape
        query_spatial = (
            self.q1(self.mhead1(emb1)),
            self.q2(self.mhead2(emb2)),
            self.q3(self.mhead3(emb3)),
            self.q4(self.mhead4(emb4)),
        )
        key_spatial = self.k(self.mheadk(emb_all))
        value_spatial = self.v(self.mheadv(emb_all))
        token_hw = (int(key_spatial.shape[-2]), int(key_spatial.shape[-1]))
        if token_hw != (height, width):
            raise ValueError("V3.3 Q/K/V token grids must be identical")
        raw_queries = tuple(
            rearrange(
                query,
                "b (head c) h w -> b head c (h w)",
                head=self.num_attention_heads,
            )
            for query in query_spatial
        )
        raw_key = rearrange(
            key_spatial,
            "b (head c) h w -> b head c (h w)",
            head=self.num_attention_heads,
        )
        value = rearrange(
            value_spatial,
            "b (head c) h w -> b head c (h w)",
            head=self.num_attention_heads,
        )
        if not bool(
            torch.stack(
                tuple(
                    torch.isfinite(tensor).all()
                    for tensor in (*raw_queries, raw_key, value)
                )
            ).all()
        ):
            raise RuntimeError("V3.3 base Q/K/V contains non-finite values")
        queries = tuple(F.normalize(query, dim=-1) for query in raw_queries)
        key = F.normalize(raw_key, dim=-1)
        key_transpose = key.transpose(-2, -1)
        scale = math.sqrt(self.KV_size)
        base_attentions = tuple(
            self.softmax(self.psi((query @ key_transpose) / scale))
            for query in queries
        )
        if not bool(
            torch.stack(
                tuple(torch.isfinite(value).all() for value in base_attentions)
            ).all()
        ):
            raise RuntimeError("V3.3 base attention contains non-finite values")
        with torch.autocast(device_type=raw_key.device.type, enabled=False):
            c3_queries = tuple(
                F.normalize(query.float(), dim=-1) for query in raw_queries
            )
            c3_key = F.normalize(raw_key.float(), dim=-1)
        static_support = v32._estimate_c3_v31_support_v32(
            c3_key,
            c3_queries,
            base_attentions,
            eps=self.eps,
            detach_support=True,
        )
        routed_levels = self._route_supports(
            static_support=static_support,
            raw_queries=raw_queries,
            raw_key=raw_key,
            value_spatial=value_spatial,
            token_hw=token_hw,
        )
        gain = self.effective_level_gain(None)
        attentions: list[torch.Tensor] = []
        diagnostics: list[dict[str, Any]] = []
        intended_modes: list[torch.Tensor] = []
        effective_modes: list[torch.Tensor] = []
        compact = {
            "projection_rows": 0,
            "nonidentity_rows": 0,
            "solver_attempt_rows": 0,
            "solver_accepted_rows": 0,
            "solver_fallback_rows": 0,
            "emission_checked_rows": 0,
            "emission_accepted_rows": 0,
            "emission_fallback_rows": 0,
            "mode_identity_rows": 0,
            "mode_dual_rows": 0,
            "mode_hard_only_rows": 0,
            "mode_background_only_rows": 0,
        }
        for level_index, (
            query,
            base_attention,
            routed,
        ) in enumerate(zip(c3_queries, base_attentions, routed_levels, strict=True)):
            availability = routed.availability
            full_eligible = availability[:, 0] & availability[:, 2]
            hard_eligible = availability[:, 0] & availability[:, 1]
            # Mode planning is deliberately detached.  It determines discrete
            # row identities, after which each differentiable V3.1 branch is
            # re-run only on rows that will actually be emitted.  This avoids
            # 0*NaN gradients from solver branches that a later where() masks.
            with torch.no_grad():
                (
                    _planned_full_attention,
                    planned_full_diagnostics,
                    _planned_full_hard_defined,
                    background_defined,
                    _planned_full_accepted,
                    _planned_full_solver_fallback,
                ) = self._project_available_subset(
                    eligible_images=full_eligible,
                    level_index=level_index,
                    query=query,
                    key=c3_key,
                    base_attention=base_attention,
                    support=routed.support,
                    key_confidence=static_support.key_confidence,
                    key_confidence_valid=static_support.key_confidence_valid,
                    gain=gain,
                    intervention="full",
                )
                (
                    _planned_hard_attention,
                    planned_hard_diagnostics,
                    hard_defined,
                    _planned_hard_background_defined,
                    _planned_hard_accepted,
                    _planned_hard_solver_fallback,
                ) = self._project_available_subset(
                    eligible_images=hard_eligible,
                    level_index=level_index,
                    query=query,
                    key=c3_key,
                    base_attention=base_attention,
                    support=routed.support,
                    key_confidence=static_support.key_confidence,
                    key_confidence_valid=static_support.key_confidence_valid,
                    gain=gain,
                    intervention="consistent_contradictory_only",
                )
            candidate_available = routed.support.consistent_valid.expand_as(
                hard_defined
            )
            intended_mode = torch.full_like(
                hard_defined,
                MODE_IDENTITY,
                dtype=torch.int64,
            )
            intended_mode = torch.where(
                candidate_available & hard_defined & background_defined,
                torch.full_like(intended_mode, MODE_DUAL),
                intended_mode,
            )
            intended_mode = torch.where(
                candidate_available & hard_defined & ~background_defined,
                torch.full_like(intended_mode, MODE_HARD_ONLY),
                intended_mode,
            )
            intended_mode = torch.where(
                candidate_available & ~hard_defined & background_defined,
                torch.full_like(intended_mode, MODE_BACKGROUND_ONLY),
                intended_mode,
            )
            full_selected_rows = intended_mode.eq(MODE_DUAL) | intended_mode.eq(
                MODE_BACKGROUND_ONLY
            )
            hard_selected_rows = intended_mode.eq(MODE_HARD_ONLY)
            (
                full_attention,
                full_diagnostics,
                full_accepted,
                full_solver_fallback,
            ) = self._project_selected_rows(
                selected_rows=full_selected_rows,
                level_index=level_index,
                query=query,
                key=c3_key,
                base_attention=base_attention,
                support=routed.support,
                key_confidence=static_support.key_confidence,
                key_confidence_valid=static_support.key_confidence_valid,
                gain=gain,
                intervention="full",
            )
            (
                hard_attention,
                hard_diagnostics,
                hard_accepted,
                hard_solver_fallback,
            ) = self._project_selected_rows(
                selected_rows=hard_selected_rows,
                level_index=level_index,
                query=query,
                key=c3_key,
                base_attention=base_attention,
                support=routed.support,
                key_confidence=static_support.key_confidence,
                key_confidence_valid=static_support.key_confidence_valid,
                gain=gain,
                intervention="consistent_contradictory_only",
            )
            attention, effective_mode, merge_fallback = (
                merge_mode_routed_attention_v33(
                    base_attention=base_attention,
                    full_attention=full_attention,
                    hard_attention=hard_attention,
                    intended_mode=intended_mode,
                    full_diagnostics=full_diagnostics,
                    hard_diagnostics=hard_diagnostics,
                )
            )
            attentions.append(attention)
            intended_modes.append(intended_mode)
            effective_modes.append(effective_mode)

            selected_projection = torch.where(
                intended_mode.eq(MODE_HARD_ONLY),
                hard_accepted,
                full_accepted,
            )
            selected_solver_fallback = torch.where(
                intended_mode.eq(MODE_HARD_ONLY),
                hard_solver_fallback,
                full_solver_fallback,
            )
            attempted = intended_mode.ne(MODE_IDENTITY)
            compact["projection_rows"] += int(intended_mode.numel())
            compact["nonidentity_rows"] += int(attempted.sum().item())
            compact["solver_attempt_rows"] += int(
                (attempted & (selected_projection | selected_solver_fallback))
                .sum()
                .item()
            )
            compact["solver_accepted_rows"] += int(
                (attempted & selected_projection).sum().item()
            )
            compact["solver_fallback_rows"] += int(
                (attempted & selected_solver_fallback).sum().item()
            )
            compact["emission_checked_rows"] += int(attempted.sum().item())
            compact["emission_fallback_rows"] += int(merge_fallback.sum().item())
            compact["emission_accepted_rows"] += int(
                (attempted & ~merge_fallback).sum().item()
            )
            for code, name in enumerate(MODE_NAMES):
                compact[f"mode_{name}_rows"] += int(
                    effective_mode.eq(code).sum().item()
                )

            level_diagnostics = {
                "schema": SBSC_V33_SCHEMA + "/projection_level/v1",
                "level_index": level_index,
                "support": routed.support,
                "router_logits": routed.logits,
                "router_supports": routed.role_probability,
                "role_mass": routed.role_mass,
                "integrated_winning_evidence": (
                    routed.integrated_winning_evidence
                ),
                "support_weighted_existence": (
                    routed.support_weighted_existence
                ),
                "availability": routed.availability,
                "intended_mode_code": intended_mode,
                "effective_mode_code": effective_mode,
                "merge_fallback": merge_fallback,
                "full_branch": full_diagnostics,
                "hard_only_branch": hard_diagnostics,
                "planned_full_branch": planned_full_diagnostics,
                "planned_hard_only_branch": planned_hard_diagnostics,
                "emission_fallback": merge_fallback,
                "emitted_certificate_ok": ~merge_fallback,
            }
            if collect_diagnostics:
                diagnostics.append(level_diagnostics)

        projections = (
            self.project_out1,
            self.project_out2,
            self.project_out3,
            self.project_out4,
        )
        outputs: list[torch.Tensor] = []
        for attention, projection_layer in zip(
            attentions,
            projections,
            strict=True,
        ):
            out = (attention @ value).mean(dim=1)
            out = rearrange(
                out,
                "b c (h w) -> b c h w",
                h=height,
                w=width,
            )
            outputs.append(projection_layer(out))
        if context_capture and request is not None:
            request.collector.records.append(
                {
                    "schema": SBSC_V33_SCHEMA + "/training_router_capture/v1",
                    "module_id": id(self),
                    "batch_size": batch,
                    "token_hw": token_hw,
                    "logits": tuple(level.logits for level in routed_levels),
                    "supports": tuple(
                        level.role_probability for level in routed_levels
                    ),
                    "availability": tuple(
                        level.availability for level in routed_levels
                    ),
                    "role_mass": tuple(level.role_mass for level in routed_levels),
                    "integrated_winning_evidence": tuple(
                        level.integrated_winning_evidence
                        for level in routed_levels
                    ),
                    "intended_mode_codes": tuple(intended_modes),
                    "effective_mode_codes": tuple(effective_modes),
                    "value_spatial": value_spatial,
                    "router_value_gradient_mode": self.router_value_gradient_mode,
                    "compact_diagnostics": compact,
                }
            )
        return (
            (outputs[0], outputs[1], outputs[2], outputs[3], None),
            tuple(diagnostics),
        )


def _branch_emission_ok(
    diagnostics: Mapping[str, Any],
    *,
    expected_shape: torch.Size,
) -> torch.Tensor:
    emitted = diagnostics.get("emitted_certificate_ok")
    fallback = diagnostics.get("emission_fallback")
    projection_accepted = diagnostics.get("projection_accepted")
    if (
        not isinstance(emitted, torch.Tensor)
        or not isinstance(fallback, torch.Tensor)
        or not isinstance(projection_accepted, torch.Tensor)
        or emitted.dtype is not torch.bool
        or fallback.dtype is not torch.bool
        or projection_accepted.dtype is not torch.bool
        or emitted.shape != expected_shape
        or fallback.shape != expected_shape
        or projection_accepted.shape != expected_shape
    ):
        raise RuntimeError("projection branch certificate tensors are malformed")
    return (
        emitted.detach()
        & ~fallback.detach()
        & projection_accepted.detach()
    )


def merge_mode_routed_attention_v33(
    *,
    base_attention: torch.Tensor,
    full_attention: torch.Tensor,
    hard_attention: torch.Tensor,
    intended_mode: torch.Tensor,
    full_diagnostics: Mapping[str, Any],
    hard_diagnostics: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one certified V3.1 branch and fail closed per attention row."""

    if (
        not isinstance(base_attention, torch.Tensor)
        or base_attention.ndim != 4
        or full_attention.shape != base_attention.shape
        or hard_attention.shape != base_attention.shape
        or full_attention.dtype != base_attention.dtype
        or hard_attention.dtype != base_attention.dtype
        or full_attention.device != base_attention.device
        or hard_attention.device != base_attention.device
    ):
        raise ValueError("mode-routed attention tensors differ")
    expected_mode_shape = base_attention.shape[:-1] + (1,)
    if (
        not isinstance(intended_mode, torch.Tensor)
        or intended_mode.shape != expected_mode_shape
        or intended_mode.dtype is not torch.int64
        or intended_mode.device != base_attention.device
        or bool(
            (
                (intended_mode < MODE_IDENTITY)
                | (intended_mode > MODE_BACKGROUND_ONLY)
            ).any()
        )
    ):
        raise ValueError("intended mode tensor is malformed")
    full_ok = _branch_emission_ok(
        full_diagnostics,
        expected_shape=expected_mode_shape,
    )
    hard_ok = _branch_emission_ok(
        hard_diagnostics,
        expected_shape=expected_mode_shape,
    )
    full_selected = intended_mode.eq(MODE_DUAL) | intended_mode.eq(
        MODE_BACKGROUND_ONLY
    )
    hard_selected = intended_mode.eq(MODE_HARD_ONLY)
    candidate = torch.where(
        hard_selected,
        hard_attention,
        torch.where(full_selected, full_attention, base_attention),
    )
    branch_ok = (
        intended_mode.eq(MODE_IDENTITY)
        | (full_selected & full_ok)
        | (hard_selected & hard_ok)
    )
    tolerance = v31_ambient_emission_tolerance(base_attention.dtype)
    with torch.no_grad():
        candidate_fp32 = candidate.detach().float()
        base_fp32 = base_attention.detach().float()
        contract_ok = (
            torch.isfinite(candidate_fp32).all(dim=-1, keepdim=True)
            & candidate_fp32.amin(dim=-1, keepdim=True).ge(-tolerance)
            & candidate_fp32.sum(dim=-1, keepdim=True)
            .sub(base_fp32.sum(dim=-1, keepdim=True))
            .abs()
            .le(tolerance)
        )
        accepted = branch_ok & contract_ok
        effective_mode = torch.where(
            accepted,
            intended_mode,
            torch.full_like(intended_mode, MODE_IDENTITY),
        )
        merge_fallback = intended_mode.ne(MODE_IDENTITY) & ~accepted
    emitted = torch.where(
        effective_mode.ne(MODE_IDENTITY),
        candidate,
        base_attention,
    )
    return emitted, effective_mode, merge_fallback


def _v33_modules(
    model: nn.Module,
) -> tuple[RoleExclusiveTriEvidenceProjectionV33, ...]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, RoleExclusiveTriEvidenceProjectionV33)
    )


@contextmanager
def capture_c3_v33_training_router(model: nn.Module):
    """Capture exactly one V3.3 block without hooks or persistent mutation."""

    if type(model) is not SCTransNet:
        raise TypeError("V3.3 training capture requires exact SCTransNet")
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError("V3.3 training capture requires training with gradients")
    modules = _v33_modules(model)
    if (
        len(modules) != 1
        or type(modules[0]) is not RoleExclusiveTriEvidenceProjectionV33
    ):
        raise RuntimeError("training capture requires one exact V3.3 block")
    if _C3_V33_TRAINING_ROUTER_REQUEST.get() is not None:
        raise RuntimeError("nested V3.3 training router capture is forbidden")
    collector = C3V33TrainingRouterCollector(records=[])
    token = _C3_V33_TRAINING_ROUTER_REQUEST.set(
        _C3V33TrainingRouterRequest(
            module_id=id(modules[0]),
            collector=collector,
        )
    )
    try:
        yield collector
    finally:
        _C3_V33_TRAINING_ROUTER_REQUEST.reset(token)


def _require_token_hw(
    token_hw: tuple[int, int] | Sequence[int],
) -> tuple[int, int]:
    values = tuple(token_hw)
    if (
        len(values) != 2
        or any(type(value) is not int or value <= 0 for value in values)
        or values[0] * values[1] <= 1
    ):
        raise ValueError("token_hw must contain two positive ints with N>1")
    return int(values[0]), int(values[1])


def build_role_targets_v33(
    target: torch.Tensor,
    detached_prediction: torch.Tensor,
    token_hw: tuple[int, int] | Sequence[int],
) -> RoleTargetsV33:
    """Build a strict token-wise C/H/B target simplex."""

    if not isinstance(target, torch.Tensor) or not isinstance(
        detached_prediction,
        torch.Tensor,
    ):
        raise TypeError("target and prediction must be tensors")
    if (
        target.ndim != 4
        or detached_prediction.ndim != 4
        or tuple(target.shape) != tuple(detached_prediction.shape)
        or target.shape[0] <= 0
        or target.shape[1] != 1
    ):
        raise ValueError("target/prediction must have equal nonempty [B,1,H,W]")
    if target.device != detached_prediction.device:
        raise TypeError("target and prediction must share a device")
    if detached_prediction.requires_grad:
        raise ValueError("router teacher prediction must be explicitly detached")
    values = _require_token_hw(token_hw)
    with torch.no_grad(), torch.autocast(
        device_type=target.device.type,
        enabled=False,
    ):
        y = target.detach().float()
        p = detached_prediction.detach().float()
        if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(p).all()):
            raise ValueError("router supervision inputs must be finite")
        if bool(((y < 0.0) | (y > 1.0)).any()) or bool(
            ((p < 0.0) | (p > 1.0)).any()
        ):
            raise ValueError("router supervision inputs must be in [0,1]")
        pooled_target = F.adaptive_max_pool2d(y, values)
        pooled_prediction = F.adaptive_max_pool2d(p, values)
        background = 1.0 - pooled_target
        probability = torch.cat(
            (
                pooled_target,
                background * pooled_prediction,
                background * (1.0 - pooled_prediction),
            ),
            dim=1,
        )
        if not bool(torch.isfinite(probability).all()):
            raise RuntimeError("V3.3 role target is non-finite")
        if bool(((probability < 0.0) | (probability > 1.0)).any()):
            raise RuntimeError("V3.3 role target is outside [0,1]")
        if not bool(
            probability.sum(dim=1).sub(1.0).abs().le(1.0e-6).all()
        ):
            raise RuntimeError("V3.3 role target is not a role simplex")
    return RoleTargetsV33(
        probability=probability,
        target_region=pooled_target,
        background_region=background,
        token_hw=values,
    )


def _balanced_role_ce_per_image_v33(
    logits: torch.Tensor,
    targets: RoleTargetsV33,
    *,
    balance_mode: str,
) -> torch.Tensor:
    """Return auditable CE contributions with exact shape [B,3]."""

    if balance_mode not in SUPPORTED_BALANCE_MODES:
        raise ValueError(f"unsupported V3.3 balance mode: {balance_mode!r}")
    if type(targets) is not RoleTargetsV33:
        raise TypeError("targets must be exact RoleTargetsV33")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 4
        or logits.shape[1] != 3
        or tuple(logits.shape) != tuple(targets.probability.shape)
        or logits.device != targets.probability.device
    ):
        raise ValueError("logits and role target must share exact [B,3,H,W]")
    if (
        targets.probability.dtype is not torch.float32
        or targets.target_region.dtype is not torch.float32
        or targets.background_region.dtype is not torch.float32
        or targets.target_region.shape != logits.shape[:1] + (1,) + logits.shape[2:]
        or targets.background_region.shape != targets.target_region.shape
        or targets.token_hw != (int(logits.shape[-2]), int(logits.shape[-1]))
    ):
        raise ValueError("role target tensor contract differs")
    if not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError("router logits must be finite")
    if not bool(torch.isfinite(targets.probability).all()):
        raise ValueError("role targets must be finite")
    if bool(
        (
            (targets.probability < 0.0)
            | (targets.probability > 1.0)
        ).any()
    ) or not bool(
        targets.probability.sum(dim=1).sub(1.0).abs().le(1.0e-6).all()
    ):
        raise ValueError("role targets must be a [0,1] simplex")
    if not bool(
        targets.target_region.add(targets.background_region)
        .sub(1.0)
        .abs()
        .le(1.0e-6)
        .all()
    ):
        raise ValueError("target/background regions must be complementary")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        log_probability = F.log_softmax(logits.float(), dim=1)
        contribution = -targets.probability * log_probability
        values = contribution.flatten(2)
        if balance_mode == "ordinary":
            per_role = values.mean(dim=-1)
        else:
            target_weight = targets.target_region.flatten(2)
            background_weight = targets.background_region.flatten(2)
            target_mass = target_weight.sum(dim=-1)
            background_mass = background_weight.sum(dim=-1)
            target_valid = target_mass.gt(SBSC_V33_EPS)
            background_valid = background_mass.gt(SBSC_V33_EPS)
            if not bool((target_valid | background_valid).all()):
                raise RuntimeError("every image must contain a valid spatial region")
            target_mean = (
                (values * target_weight).sum(dim=-1)
                / target_mass.clamp_min(SBSC_V33_EPS)
            )
            background_mean = (
                (values * background_weight).sum(dim=-1)
                / background_mass.clamp_min(SBSC_V33_EPS)
            )
            if balance_mode == "half_half":
                nominal_target = 0.5
                nominal_background = 0.5
            else:
                nominal_target = 1.0 / 3.0
                nominal_background = 2.0 / 3.0
            active_target = target_valid.to(values.dtype) * nominal_target
            active_background = (
                background_valid.to(values.dtype) * nominal_background
            )
            active_total = active_target + active_background
            per_role = (
                active_target * target_mean
                + active_background * background_mean
            ) / active_total.clamp_min(SBSC_V33_EPS)
    if (
        per_role.shape != (logits.shape[0], 3)
        or not bool(torch.isfinite(per_role).all())
        or bool((per_role < 0.0).any())
    ):
        raise RuntimeError("per-role CE tensor is malformed")
    return per_role


def token_role_ce_per_image_v33(
    logits: torch.Tensor,
    targets: RoleTargetsV33,
    *,
    balance_mode: str,
) -> torch.Tensor:
    """Return exact per-image role-axis CE with shape [B]."""

    per_role = _balanced_role_ce_per_image_v33(
        logits,
        targets,
        balance_mode=balance_mode,
    )
    per_image = per_role.sum(dim=1)
    if (
        per_image.shape != (logits.shape[0],)
        or not bool(torch.isfinite(per_image).all())
        or bool((per_image < 0.0).any())
    ):
        raise RuntimeError("per-image role CE is malformed")
    return per_image


def router_loss_v33(
    capture: C3V33TrainingRouterCollector,
    detached_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    balance_mode: str,
) -> RouterLossBreakdownV33:
    """Validate one capture and return the exact mean of four level losses."""

    if type(capture) is not C3V33TrainingRouterCollector:
        raise TypeError("capture must be exact V3.3 collector")
    if len(capture.records) != 1:
        raise RuntimeError("router loss requires exactly one captured forward")
    record = capture.records[0]
    if record.get("schema") != SBSC_V33_SCHEMA + "/training_router_capture/v1":
        raise RuntimeError("router capture schema differs")
    if type(record.get("module_id")) is not int or record["module_id"] <= 0:
        raise RuntimeError("router capture module identity is malformed")
    batch_size = record.get("batch_size")
    token_hw = record.get("token_hw")
    if type(batch_size) is not int or batch_size <= 0:
        raise RuntimeError("router capture batch size is malformed")
    expected_hw = _require_token_hw(token_hw)
    fields = {
        "logits": record.get("logits"),
        "supports": record.get("supports"),
        "availability": record.get("availability"),
        "intended_mode_codes": record.get("intended_mode_codes"),
        "effective_mode_codes": record.get("effective_mode_codes"),
    }
    if any(
        not isinstance(value, tuple) or len(value) != 4
        for value in fields.values()
    ):
        raise RuntimeError("V3.3 capture must contain four records per field")
    target_cache: dict[tuple[int, int], RoleTargetsV33] = {}
    per_level: list[torch.Tensor] = []
    per_level_per_image: list[torch.Tensor] = []
    per_level_per_role: list[torch.Tensor] = []
    for level_index, logits in enumerate(fields["logits"]):
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 4
            or logits.shape[0] != batch_size
            or logits.shape[1] != 3
            or logits.device != target.device
            or (int(logits.shape[-2]), int(logits.shape[-1])) != expected_hw
            or not bool(torch.isfinite(logits.detach()).all())
        ):
            raise RuntimeError(f"level {level_index} router logits are malformed")
        support = fields["supports"][level_index]
        availability = fields["availability"][level_index]
        intended = fields["intended_mode_codes"][level_index]
        effective = fields["effective_mode_codes"][level_index]
        if (
            not isinstance(support, torch.Tensor)
            or support.shape != logits.shape
            or support.device != logits.device
            or not bool(torch.isfinite(support).all())
            or not torch.allclose(
                support,
                F.softmax(logits.float(), dim=1),
                rtol=0.0,
                atol=2.0e-6,
            )
            or not isinstance(availability, torch.Tensor)
            or availability.shape != (batch_size, 3)
            or availability.dtype is not torch.bool
            or not isinstance(intended, torch.Tensor)
            or not isinstance(effective, torch.Tensor)
            or intended.ndim != 4
            or intended.shape[0] != batch_size
            or intended.shape[1] != 1
            or intended.shape[2] <= 0
            or intended.shape[3] != 1
            or effective.shape != intended.shape
            or intended.dtype is not torch.int64
            or effective.dtype is not torch.int64
            or bool(((intended < 0) | (intended > 3)).any())
            or bool(((effective < 0) | (effective > 3)).any())
        ):
            raise RuntimeError(f"level {level_index} capture identity is malformed")
        targets = target_cache.get(expected_hw)
        if targets is None:
            targets = build_role_targets_v33(
                target,
                detached_prediction,
                expected_hw,
            )
            target_cache[expected_hw] = targets
        per_role = _balanced_role_ce_per_image_v33(
            logits,
            targets,
            balance_mode=balance_mode,
        )
        per_image = per_role.sum(dim=1)
        level_loss = per_image.mean()
        per_level_per_role.append(per_role)
        per_level_per_image.append(per_image)
        per_level.append(level_loss)
    total = torch.stack(per_level, dim=0).mean()
    if (
        total.ndim != 0
        or not bool(torch.isfinite(total))
        or float(total.detach()) < 0.0
    ):
        raise RuntimeError("aggregated V3.3 router loss is malformed")
    return RouterLossBreakdownV33(
        total=total,
        per_level=tuple(per_level),
        per_level_per_image=tuple(per_level_per_image),
        per_level_per_role=tuple(per_level_per_role),
        level_reduction="mean",
        balance_mode=balance_mode,
    )


def training_losses_v33(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    *,
    balance_mode: str,
    router_loss_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    RouterLossBreakdownV33,
]:
    """Compute the six segmentation terms and one four-level router term."""

    if type(router_loss_weight) not in (int, float) or float(
        router_loss_weight
    ) != 1.0:
        raise ValueError("V3.3 freezes router_loss_weight=1.0")
    with capture_c3_v33_training_router(model) as capture:
        outputs = model(images)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("V3.3 training forward must return six outputs")
    breakdown = router_loss_v33(
        capture,
        outputs[-1].detach(),
        masks,
        balance_mode=balance_mode,
    )
    segmentation_loss = sum(criterion(output, masks) for output in outputs)
    router_loss = breakdown.total
    total_loss = segmentation_loss + router_loss
    for name, value in (
        ("total", total_loss),
        ("segmentation", segmentation_loss),
        ("router", router_loss),
        *tuple(
            (f"router_level_{index}", level_loss)
            for index, level_loss in enumerate(breakdown.per_level)
        ),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 0
            or not bool(torch.isfinite(value))
            or float(value.detach()) < 0.0
        ):
            raise RuntimeError(f"{name} loss is malformed")
    return total_loss, segmentation_loss, router_loss, breakdown


def _require_dataset(dataset: str) -> str:
    if type(dataset) is not str or dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}")
    return dataset


def _require_method(method: str) -> str:
    if type(method) is not str or method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    return method


def _require_architecture_seed(seed: int) -> int:
    if type(seed) is not int or seed != ARCHITECTURE_SEED:
        raise ValueError("V3.3 freezes architecture_seed=42")
    return seed


def _state_mapping(source: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if not isinstance(source, Mapping):
        raise TypeError("state must be a mapping")
    nested = source.get("state_dict")
    candidate = nested if isinstance(nested, Mapping) else source
    if not all(
        type(key) is str and isinstance(value, torch.Tensor)
        for key, value in candidate.items()
    ):
        raise TypeError("state must map string keys to tensors")
    return candidate


def validate_sbsc_v33_state_dict(
    state: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Validate the exact baseline or V3.3 state before loading it."""

    v32._require_frozen_v31_solver_source()
    method = _require_method(method)
    tensors = _state_mapping(state)
    prefix = ""
    if tensors and all(key.startswith("module.") for key in tensors):
        prefix = "module."
    canonical: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        canonical_key = key[len(prefix) :] if prefix else key
        if canonical_key in canonical:
            raise ValueError("state contains duplicate canonical keys")
        canonical[canonical_key] = value
    expected = {
        key: (shape, dtype)
        for key, shape, dtype in v31._authority_state_schema()
    }
    is_candidate = method != "sctransnet"
    if is_candidate:
        expected[SBSC_V33_GAIN_STATE_KEY] = ((4,), torch.float32)
        for key, shape in _FROZEN_ROUTER_STATE_SCHEMA:
            expected[key] = (shape, torch.float32)
    if set(canonical) != set(expected):
        missing = sorted(set(expected) - set(canonical))
        unexpected = sorted(set(canonical) - set(expected))
        raise ValueError(
            "state key set differs from exact V3.3 schema; "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    for key, (shape, dtype) in expected.items():
        value = canonical[key]
        if tuple(value.shape) != shape:
            raise ValueError(f"state shape differs for {key!r}")
        if value.dtype != dtype:
            raise TypeError(f"state dtype differs for {key!r}")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"state tensor {key!r} must be finite")
    gains: list[float] = []
    if is_candidate:
        gain = canonical[SBSC_V33_GAIN_STATE_KEY]
        if bool(((gain < SBSC_V33_GAIN_MIN) | (gain > SBSC_V33_GAIN_MAX)).any()):
            raise ValueError("V3.3 gain state is outside [0,0.25]")
        gains = [float(value) for value in gain.detach().cpu().tolist()]
    return {
        "schema": SBSC_V33_SCHEMA + "/state_validation/v1",
        "method": method,
        "state_key_count": len(tensors),
        "gain_values": gains,
        "gain_state_keys": (
            [f"{prefix}{SBSC_V33_GAIN_STATE_KEY}"] if is_candidate else []
        ),
        "router_state_keys": (
            [f"{prefix}{key}" for key in SBSC_V33_ROUTER_STATE_KEYS]
            if is_candidate
            else []
        ),
        "data_parallel_prefix": bool(prefix),
    }


def replace_sctransnet_l1_ssca_with_sbsc_v33(
    model: SCTransNet,
    *,
    router_value_gradient_mode: str,
) -> tuple[str, ...]:
    """Transactionally replace exactly zero-based SCTB layer 1."""

    if type(model) is not SCTransNet:
        raise TypeError("SBSC V3.3 replacement requires exact SCTransNet")
    if router_value_gradient_mode not in SUPPORTED_GRADIENT_MODES:
        raise ValueError("router_value_gradient_mode must be live or detached")
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("SCTransNet must contain exactly four SCTBs")
    sources = tuple(block.channel_attn for block in layers)
    if any(type(source) is not Attention_org for source in sources):
        raise TypeError("all source SSCA operators must be exact Attention_org")
    source = sources[_FROZEN_REPLACED_BLOCK_INDEX]
    replacement = RoleExclusiveTriEvidenceProjectionV33.from_ssca(
        source,
        layer_index=_FROZEN_REPLACED_BLOCK_INDEX,
        router_value_gradient_mode=router_value_gradient_mode,
    )
    source_state = source.state_dict()
    replacement_state = replacement.state_dict()
    expected_added = {
        SBSC_V33_GAIN_SUFFIX,
        "exclusive_tri_router.value_proj.weight",
        "exclusive_tri_router.head.weight",
    }
    if set(replacement_state) - set(source_state) != expected_added:
        raise RuntimeError("SBSC V3.3 replacement added unexpected state")
    if set(source_state) - set(replacement_state):
        raise RuntimeError("SBSC V3.3 replacement removed source state")
    if any(
        not torch.equal(source_state[key], replacement_state[key])
        for key in source_state
    ):
        raise RuntimeError("SBSC V3.3 replacement changed shared source state")
    layers[_FROZEN_REPLACED_BLOCK_INDEX].channel_attn = replacement
    return ("mtc.encoder.layer.1.channel_attn",)


@torch.no_grad()
def project_sbsc_v33_constraints_(model: nn.Module) -> None:
    """Clamp the one V3.3 gain vector after an optimizer step."""

    modules = _v33_modules(model)
    if (
        len(modules) != 1
        or type(modules[0]) is not RoleExclusiveTriEvidenceProjectionV33
    ):
        raise RuntimeError("gain projection requires one exact V3.3 block")
    modules[0].project_level_gain_()


def validate_sctransnet_sbsc_v33(
    model: nn.Module,
    *,
    method: str,
    router_value_gradient_mode: str,
    require_zero_gain: bool = False,
) -> dict[str, Any]:
    """Validate the exact V3.3 graph, state prefix, dtype, and mode identity."""

    method = _require_method(method)
    if method == "sctransnet":
        raise ValueError("V3.3 graph validator requires a candidate method")
    if router_value_gradient_mode not in SUPPORTED_GRADIENT_MODES:
        raise ValueError("router_value_gradient_mode must be live or detached")
    if type(require_zero_gain) is not bool:
        raise TypeError("require_zero_gain must be bool")
    v32._require_frozen_v31_solver_source()
    if type(model) is not SCTransNet:
        raise TypeError("validator requires exact unwrapped SCTransNet")
    if bool(getattr(model, "diagnostic_only", False)):
        raise RuntimeError("diagnostic-only adapters are forbidden")
    v31._reject_instance_forward_mutation_or_hooks(model)
    if (
        v31._nonreplacement_structure_records(model)
        != v31._authority_nonreplacement_structure()
    ):
        raise RuntimeError("SCTransNet structure outside L1 SSCA differs")
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("model must retain exactly four SCTBs")
    modules = tuple(layer.channel_attn for layer in layers)
    if type(modules[1]) is not RoleExclusiveTriEvidenceProjectionV33:
        raise TypeError("zero-based SCTB layer 1 must contain exact V3.3 class")
    if any(type(modules[index]) is not Attention_org for index in (0, 2, 3)):
        raise TypeError("SCTB layers 0,2,3 must retain exact Attention_org")
    discovered = _v33_modules(model)
    if len(discovered) != 1 or discovered[0] is not modules[1]:
        raise RuntimeError("V3.3 module must exist only at SCTB layer 1")
    module = discovered[0]
    if (
        module.layer_index != 1
        or module.gain_limit != SBSC_V33_GAIN_MAX
        or module.eps != SBSC_V33_EPS
        or module.detach_support is not True
        or module.router_value_gradient_mode != router_value_gradient_mode
        or module.num_attention_heads != 1
        or module.KV_size != 480
        or tuple(module.channel_num) != (32, 64, 128, 256)
        or module.vis is not False
    ):
        raise RuntimeError("V3.3 scalar or attention geometry differs")
    expected_children = (
        "psi",
        "softmax",
        "mhead1",
        "mhead2",
        "mhead3",
        "mhead4",
        "mheadk",
        "mheadv",
        "q1",
        "q2",
        "q3",
        "q4",
        "k",
        "v",
        "project_out1",
        "project_out2",
        "project_out3",
        "project_out4",
        "exclusive_tri_router",
    )
    if tuple(module._modules) != expected_children:
        raise RuntimeError("V3.3 L1 SSCA module tree differs")
    router = module.exclusive_tri_router
    if type(router) is not _RoleExclusiveTriEvidenceRouterV33:
        raise TypeError("V3.3 requires the exact exclusive router")
    if tuple(router._modules) != ("value_proj", "activation", "head"):
        raise RuntimeError("V3.3 router module tree differs")
    if (
        type(router.value_proj) is not nn.Conv2d
        or router.value_proj.in_channels != 480
        or router.value_proj.out_channels != 8
        or router.value_proj.kernel_size != (1, 1)
        or router.value_proj.bias is not None
        or type(router.activation) is not nn.SiLU
        or router.activation.inplace is not False
        or type(router.head) is not nn.Conv2d
        or router.head.in_channels != 15
        or router.head.out_channels != 3
        or router.head.kernel_size != (3, 3)
        or router.head.padding != (1, 1)
        or router.head.bias is not None
    ):
        raise RuntimeError("V3.3 router layer contract differs")
    gain = module._parameters.get(SBSC_V33_GAIN_SUFFIX)
    if (
        gain is not module.raw_dual_risk_level_gain
        or tuple(gain.shape) != (4,)
        or gain.dtype is not torch.float32
        or gain.requires_grad is not True
        or any(parameter.dtype is not torch.float32 for parameter in router.parameters())
        or any(parameter.requires_grad is not True for parameter in router.parameters())
    ):
        raise RuntimeError("V3.3 gain/router parameter contract differs")
    validation = validate_sbsc_v33_state_dict(model.state_dict(), method)
    if require_zero_gain and torch.count_nonzero(gain).item() != 0:
        raise RuntimeError("V3.3 identity gain is not exactly zero")
    if v31._parameter_count(model) != _FROZEN_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("V3.3 parameter-count contract differs")
    if len(model.state_dict()) != _FROZEN_CANDIDATE_STATE_KEY_COUNT:
        raise RuntimeError("V3.3 state-key count contract differs")
    return {
        "schema": SBSC_V33_SCHEMA,
        "model": SBSC_V33_MODEL_NAME,
        "method": method,
        "state_key_count": len(model.state_dict()),
        "parameter_count": v31._parameter_count(model),
        "router_parameter_count": SBSC_V33_ROUTER_PARAMETER_COUNT,
        "router_state_keys": list(SBSC_V33_ROUTER_STATE_KEYS),
        "gain_state_keys": [SBSC_V33_GAIN_STATE_KEY],
        "gain_bounds": [SBSC_V33_GAIN_MIN, SBSC_V33_GAIN_MAX],
        "replaced_block_indices": [1],
        "role_axis_softmax": True,
        "mass_aware_availability": True,
        "mode_routed_projection": True,
        "router_value_gradient_mode": router_value_gradient_mode,
        "loss_schema": SBSC_V33_LOSS_SCHEMA,
        "state_validation": validation,
    }


def _method_metadata(
    model: SCTransNet,
    *,
    method: str,
    dataset: str,
    shared_state_sha256: str,
    router_value_gradient_mode: str | None,
) -> dict[str, Any]:
    is_candidate = method != "sctransnet"
    return {
        "schema": SBSC_V33_SCHEMA,
        "model": SBSC_V33_MODEL_NAME if is_candidate else "SCTransNet",
        "method": method,
        "dataset": dataset,
        "architecture_seed": ARCHITECTURE_SEED,
        "state_key_count": len(model.state_dict()),
        "parameter_count": v31._parameter_count(model),
        "state_sha256": state_dict_sha256(model.state_dict()),
        "shared_state_sha256": shared_state_sha256,
        "paired_initialization": True,
        "test_split_accessed": False,
        "router_value_gradient_mode": (
            router_value_gradient_mode if is_candidate else None
        ),
        "gain_state_keys": [SBSC_V33_GAIN_STATE_KEY] if is_candidate else [],
        "router_state_keys": list(SBSC_V33_ROUTER_STATE_KEYS) if is_candidate else [],
        "router_parameter_count": (
            SBSC_V33_ROUTER_PARAMETER_COUNT if is_candidate else 0
        ),
        "router_initial_state_sha256": (
            state_dict_sha256(
                model.mtc.encoder.layer[1]
                .channel_attn.exclusive_tri_router.state_dict()
            )
            if is_candidate
            else None
        ),
        "loss_schema": (
            SBSC_V33_LOSS_SCHEMA
            if is_candidate
            else "sum_of_six_BCELoss_mean_terms"
        ),
        "replaced_block_indices": [1] if is_candidate else [],
        "v31_solver_source_sha256": (
            v32.V31_SOLVER_SOURCE_SHA256 if is_candidate else None
        ),
    }


def build_paired_sctransnet_sbsc_v33(
    dataset: str,
    *,
    method: str = "sbsc_v33_third",
    architecture_seed: int = ARCHITECTURE_SEED,
    router_value_gradient_mode: str,
) -> tuple[SCTransNet, SCTransNet, dict[str, Any]]:
    """Build a fresh bitwise-paired SCTransNet/V3.3 pair."""

    v32._require_frozen_v31_solver_source()
    dataset = _require_dataset(dataset)
    method = _require_method(method)
    if method == "sctransnet":
        raise ValueError("paired V3.3 builder requires a candidate method")
    _require_architecture_seed(architecture_seed)
    if router_value_gradient_mode not in SUPPORTED_GRADIENT_MODES:
        raise ValueError("router_value_gradient_mode must be live or detached")
    baseline = _construct_original(ARCHITECTURE_SEED)
    candidate = copy.deepcopy(baseline)
    replace_sctransnet_l1_ssca_with_sbsc_v33(
        candidate,
        router_value_gradient_mode=router_value_gradient_mode,
    )
    router_initial_hash = state_dict_sha256(
        candidate.mtc.encoder.layer[1]
        .channel_attn.exclusive_tri_router.state_dict()
    )
    if router_initial_hash != EXPECTED_SBSC_V33_ROUTER_INIT_SHA256:
        raise RuntimeError("V3.3 router seed-42 initialization differs")
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    expected_added = {SBSC_V33_GAIN_STATE_KEY, *SBSC_V33_ROUTER_STATE_KEYS}
    if set(candidate_state) - set(baseline_state) != expected_added:
        raise RuntimeError("paired V3.3 state additions differ")
    if set(baseline_state) - set(candidate_state):
        raise RuntimeError("paired V3.3 candidate removed baseline state")
    if any(
        not torch.equal(baseline_state[key], candidate_state[key])
        for key in baseline_state
    ):
        raise RuntimeError("paired shared state is not bitwise equal")
    shared_hash = state_dict_sha256(baseline_state)
    if shared_hash != state_dict_sha256(candidate_state, baseline_state.keys()):
        raise RuntimeError("paired shared-state hashes differ")
    if v31._parameter_count(baseline) != _FROZEN_BASELINE_PARAMETER_COUNT:
        raise RuntimeError("baseline parameter-count contract differs")
    validate_sbsc_v33_state_dict(baseline_state, "sctransnet")
    validate_sctransnet_sbsc_v33(
        candidate,
        method=method,
        router_value_gradient_mode=router_value_gradient_mode,
        require_zero_gain=True,
    )
    baseline_metadata = _method_metadata(
        baseline,
        method="sctransnet",
        dataset=dataset,
        shared_state_sha256=shared_hash,
        router_value_gradient_mode=None,
    )
    candidate_metadata = _method_metadata(
        candidate,
        method=method,
        dataset=dataset,
        shared_state_sha256=shared_hash,
        router_value_gradient_mode=router_value_gradient_mode,
    )
    metadata = dict(candidate_metadata)
    metadata.update(
        {
            "pair_schema": SBSC_V33_SCHEMA,
            "baseline": baseline_metadata,
            "candidate": candidate_metadata,
            "shared_state_key_count": len(baseline_state),
            "shared_state_bitwise_equal": True,
            "parent_checkpoint": None,
            "warm_start_used": False,
            "predecessor_checkpoint_used": False,
            "single_seed_only": True,
            "model_construction_preserves_caller_rng_stream": True,
        }
    )
    return baseline, candidate, metadata


def build_sctransnet_sbsc_v33_method(
    method: str,
    dataset: str,
    *,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
    router_value_gradient_mode: str,
) -> tuple[SCTransNet, dict[str, Any]]:
    """Build a complete fresh pair and return exactly one method role."""

    method = _require_method(method)
    if type(training) is not bool:
        raise TypeError("training must be bool")
    if method == "sctransnet":
        baseline, _candidate, pair = build_paired_sctransnet_sbsc_v33(
            dataset,
            method="sbsc_v33_third",
            architecture_seed=architecture_seed,
            router_value_gradient_mode=router_value_gradient_mode,
        )
        model = baseline
        metadata = dict(pair["baseline"])
    else:
        _baseline, candidate, pair = build_paired_sctransnet_sbsc_v33(
            dataset,
            method=method,
            architecture_seed=architecture_seed,
            router_value_gradient_mode=router_value_gradient_mode,
        )
        model = candidate
        metadata = dict(pair["candidate"])
    model.train(training)
    model.mode = "train" if training else "test"
    metadata.update(
        {
            "training": training,
            "pair": pair,
            "test_split_accessed": False,
        }
    )
    return model, metadata


def build_sctransnet_sbsc_v33(
    *,
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
    router_value_gradient_mode: str,
) -> tuple[SCTransNet, dict[str, Any]]:
    """Convenience builder for the pre-registered primary third method."""

    return build_sctransnet_sbsc_v33_method(
        "sbsc_v33_third",
        dataset,
        architecture_seed=architecture_seed,
        training=training,
        router_value_gradient_mode=router_value_gradient_mode,
    )


def structurally_inactive_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return v31.structurally_inactive_parameter_names(model)


C3V33 = RoleExclusiveTriEvidenceProjectionV33
solve_dual_risk_projection_v31 = v32.solve_dual_risk_projection_v31


__all__ = [
    "ARCHITECTURE_SEED",
    "C3V33",
    "C3V33TrainingRouterCollector",
    "EXPECTED_SBSC_V33_PARAMETER_COUNT",
    "EXPECTED_SBSC_V33_ROUTER_INIT_SHA256",
    "EXPECTED_SBSC_V33_STATE_KEY_COUNT",
    "MODE_BACKGROUND_ONLY",
    "MODE_DUAL",
    "MODE_HARD_ONLY",
    "MODE_IDENTITY",
    "MODE_NAMES",
    "RoleExclusiveTriEvidenceProjectionV33",
    "RoleTargetsV33",
    "RouterLossBreakdownV33",
    "SBSC_V33_EPS",
    "SBSC_V33_GAIN_MAX",
    "SBSC_V33_GAIN_MIN",
    "SBSC_V33_GAIN_STATE_KEY",
    "SBSC_V33_LOSS_SCHEMA",
    "SBSC_V33_MODEL_NAME",
    "SBSC_V33_ROUTER_PARAMETER_COUNT",
    "SBSC_V33_ROUTER_STATE_KEYS",
    "SBSC_V33_SCHEMA",
    "SUPPORTED_BALANCE_MODES",
    "SUPPORTED_DATASETS",
    "SUPPORTED_GRADIENT_MODES",
    "SUPPORTED_METHODS",
    "build_paired_sctransnet_sbsc_v33",
    "build_role_targets_v33",
    "build_sctransnet_sbsc_v33",
    "build_sctransnet_sbsc_v33_method",
    "capture_c3_v33_training_router",
    "merge_mode_routed_attention_v33",
    "project_sbsc_v33_constraints_",
    "replace_sctransnet_l1_ssca_with_sbsc_v33",
    "router_loss_v33",
    "router_value_input_v33",
    "solve_dual_risk_projection_v31",
    "structurally_inactive_parameter_names",
    "token_role_ce_per_image_v33",
    "training_losses_v33",
    "v31_ambient_emission_tolerance",
    "validate_sbsc_v33_state_dict",
    "validate_sctransnet_sbsc_v33",
]
