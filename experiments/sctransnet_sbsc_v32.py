#!/usr/bin/env python3
"""SCTransNet C3-SBSC V3.2 learned tri-evidence projection core.

V3.2 replaces exactly zero-based SCTB layer 1.  A very small train-supervised
router combines detached Q/K leave-one-level-out descriptors with live V
content, while the risk projection and certificates are delegated to the
frozen V3.1 implementation.  The ordinary SCTransNet forward interface is not
changed and a zero gain is an exact paired-baseline identity.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import sys
import threading
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from experiments import sctransnet_sbsc_v31 as v31
from experiments.four_dataset_models_seed42_v1 import (
    _construct_original,
    state_dict_sha256,
)
from model.SCTransNet import Attention_org, SCTransNet


SBSC_V32_SCHEMA = (
    "sctransnet_sbsc_v32/learned_tri_evidence_dual_risk_projection/v1"
)
SBSC_V32_MODEL_NAME = "SCTransNet-C3-SBSC-V3.2"

_FROZEN_ARCHITECTURE_SEED = 42
_FROZEN_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
_FROZEN_METHODS = ("sctransnet", "sbsc_v32")
_FROZEN_BLOCK_COUNT = 4
_FROZEN_REPLACED_BLOCK_INDEX = 1
_FROZEN_BASELINE_STATE_KEY_COUNT = 510
_FROZEN_BASELINE_PARAMETER_COUNT = 11_325_939
_FROZEN_CANDIDATE_STATE_KEY_COUNT = 513
_FROZEN_CANDIDATE_PARAMETER_COUNT = 11_330_188
_FROZEN_ROUTER_PARAMETER_COUNT = 4_245
_FROZEN_GAIN_MIN = 0.0
_FROZEN_GAIN_MAX = 0.25
_FROZEN_EPS = 1e-6
_FROZEN_GAIN_SUFFIX = "raw_dual_risk_level_gain"
_FROZEN_GAIN_STATE_KEY = (
    "mtc.encoder.layer.1.channel_attn." + _FROZEN_GAIN_SUFFIX
)
_FROZEN_ROUTER_PREFIX = "mtc.encoder.layer.1.channel_attn.tri_router."
_FROZEN_ROUTER_STATE_SCHEMA = (
    (_FROZEN_ROUTER_PREFIX + "value_proj.weight", (8, 480, 1, 1)),
    (_FROZEN_ROUTER_PREFIX + "head.weight", (3, 15, 3, 3)),
)
_FROZEN_LOSS_SCHEMA = (
    "sum_of_six_BCELoss_mean_terms_plus_"
    "unit_weight_normalized_spatial_router_ce"
)
_FROZEN_CUDA_MEDIAN_ADAPTER_SCHEMA = (
    "sctransnet_sbsc_v32/cuda_strict_median_value_adapter/v1"
)
_FROZEN_RUNTIME_INTEGRATION_SCHEMA = (
    "sctransnet_sbsc_v32/runtime_integration/v1"
)
_FROZEN_CUDA_MEDIAN_WARNING_PATTERN = (
    r"^median CUDA with indices output does not have a deterministic "
    r"implementation, but you set "
    r"'torch\.use_deterministic_algorithms\(True, warn_only=True\)'\. "
    r"You can file an issue at https://github\.com/pytorch/pytorch/issues "
    r"to help us prioritize adding deterministic support for this operation\. "
    r"\(Triggered internally at [^)]*Context\.cpp:\d+\.\)$"
)
_FROZEN_CUDA_MEDIAN_WARNING_COUNT = 8
_FROZEN_CUDA_MEDIAN_WARNING_LINES = (547, 552)
_CUDA_DETERMINISM_LOCK = threading.RLock()

ARCHITECTURE_SEED = _FROZEN_ARCHITECTURE_SEED
SUPPORTED_DATASETS = _FROZEN_DATASETS
SUPPORTED_METHODS = _FROZEN_METHODS
EXPECTED_SBSC_V32_BLOCKS = 1
EXPECTED_SBSC_V32_STATE_KEY_COUNT = _FROZEN_CANDIDATE_STATE_KEY_COUNT
EXPECTED_SBSC_V32_PARAMETER_COUNT = _FROZEN_CANDIDATE_PARAMETER_COUNT
SBSC_V32_ROUTER_PARAMETER_COUNT = _FROZEN_ROUTER_PARAMETER_COUNT
SBSC_V32_GAIN_MIN = _FROZEN_GAIN_MIN
SBSC_V32_GAIN_MAX = _FROZEN_GAIN_MAX
SBSC_V32_EPS = _FROZEN_EPS
SBSC_V32_GAIN_SUFFIX = _FROZEN_GAIN_SUFFIX
SBSC_V32_GAIN_STATE_KEY = _FROZEN_GAIN_STATE_KEY
SBSC_V32_ROUTER_STATE_KEYS = tuple(
    key for key, _shape in _FROZEN_ROUTER_STATE_SCHEMA
)
SBSC_V32_LOSS_SCHEMA = _FROZEN_LOSS_SCHEMA
SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA = _FROZEN_CUDA_MEDIAN_ADAPTER_SCHEMA
SBSC_V32_RUNTIME_INTEGRATION_SCHEMA = _FROZEN_RUNTIME_INTEGRATION_SCHEMA
EXPECTED_V31_SOLVER_SOURCE_SHA256 = (
    "b2d1e3f97607b305551a0602076605968041878eafd822c6f3335840ea725a3a"
)
EXPECTED_SBSC_V32_ROUTER_INIT_SHA256 = (
    "a8f549893b4819442dfe3dcb2fb3822062501fac817c3dd6b0ea949dd3a4376d"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


V31_SOLVER_SOURCE_SHA256 = _sha256_file(Path(v31.__file__).resolve())
_FROZEN_V31_SUPPORT_FUNCTION = v31.estimate_c3_v31_support
_FROZEN_V31_SOLVER_FUNCTION = v31.solve_dual_risk_projection_v31
_FROZEN_V31_PROJECT_LEVEL_FUNCTION = v31.C3DualRiskProjectionV31._project_level
_FROZEN_V31_CONDITIONAL_FUNCTION = (
    v31.C3DualRiskProjectionV31._conditional_attention
)


def _require_frozen_v31_solver_source() -> None:
    actual = _sha256_file(Path(v31.__file__).resolve())
    if (
        actual != EXPECTED_V31_SOLVER_SOURCE_SHA256
        or V31_SOLVER_SOURCE_SHA256 != EXPECTED_V31_SOLVER_SOURCE_SHA256
        or v31.estimate_c3_v31_support is not _FROZEN_V31_SUPPORT_FUNCTION
        or v31.solve_dual_risk_projection_v31 is not _FROZEN_V31_SOLVER_FUNCTION
        or v31.C3DualRiskProjectionV31._project_level
        is not _FROZEN_V31_PROJECT_LEVEL_FUNCTION
        or v31.C3DualRiskProjectionV31._conditional_attention
        is not _FROZEN_V31_CONDITIONAL_FUNCTION
    ):
        raise RuntimeError(
            "V3.2 frozen V3.1 source SHA256 differs from authority or its "
            "implementation was replaced"
        )


def _estimate_c3_v31_support_v32(
    normalized_key: torch.Tensor,
    normalized_queries: Sequence[torch.Tensor],
    base_attentions: Sequence[torch.Tensor],
    *,
    eps: float,
    detach_support: bool,
) -> v31.C3V31Support:
    """Call frozen V3.1 support with a CUDA-strict median value adapter.

    V3.1 asks ``median(dim).values`` for exactly three peers.  CUDA reports
    the unused index selection as nondeterministic, but tied median values are
    equal.  Only under strict CUDA determinism, temporarily downgrade that
    one exact warning while preserving every caller determinism setting.
    """

    is_cuda = (
        isinstance(normalized_key, torch.Tensor)
        and normalized_key.device.type == "cuda"
    )
    if not is_cuda:
        return v31.estimate_c3_v31_support(
            normalized_key,
            normalized_queries,
            base_attentions,
            eps=eps,
            detach_support=detach_support,
        )

    with _CUDA_DETERMINISM_LOCK:
        strict_cuda = (
            torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
        )
        if not strict_cuda:
            return v31.estimate_c3_v31_support(
                normalized_key,
                normalized_queries,
                base_attentions,
                eps=eps,
                detach_support=detach_support,
            )
        previous_enabled = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("error")
                warnings.filterwarnings(
                    "always",
                    message=_FROZEN_CUDA_MEDIAN_WARNING_PATTERN,
                    category=UserWarning,
                )
                result = v31.estimate_c3_v31_support(
                    normalized_key,
                    normalized_queries,
                    base_attentions,
                    eps=eps,
                    detach_support=detach_support,
                )
        finally:
            torch.use_deterministic_algorithms(
                previous_enabled, warn_only=previous_warn_only
            )
        _validate_cuda_median_warning_evidence(captured)
        return result


def _validate_cuda_median_warning_evidence(
    captured: Sequence[warnings.WarningMessage],
) -> None:
    """Accept exactly the eight value-invariant V3.1 median warnings."""

    if len(captured) != _FROZEN_CUDA_MEDIAN_WARNING_COUNT:
        raise RuntimeError(
            "V3.2 CUDA median adapter requires exactly eight frozen warnings"
        )
    expected_source = str(Path(v31.__file__).resolve())
    line_counts = {line: 0 for line in _FROZEN_CUDA_MEDIAN_WARNING_LINES}
    for warning in captured:
        if (
            warning.category is not UserWarning
            or re.fullmatch(
                _FROZEN_CUDA_MEDIAN_WARNING_PATTERN,
                str(warning.message),
            )
            is None
            or str(Path(warning.filename).resolve()) != expected_source
            or warning.lineno not in line_counts
        ):
            raise RuntimeError(
                "V3.2 CUDA median adapter observed an unexpected warning"
            )
        line_counts[warning.lineno] += 1
    if any(count != 4 for count in line_counts.values()):
        raise RuntimeError(
            "V3.2 CUDA median adapter warning line counts differ"
        )


_FROZEN_V32_SUPPORT_ADAPTER_FUNCTION = _estimate_c3_v31_support_v32


@dataclass(frozen=True)
class TriRouterTargetsV32:
    """Detached FP32 C/H/B soft targets for one token geometry."""

    raw: torch.Tensor
    probability: torch.Tensor
    mass: torch.Tensor
    valid: torch.Tensor
    token_hw: tuple[int, int]


@dataclass
class C3V32TrainingRouterCollector:
    """Context-local ledger containing live router logits."""

    records: list[dict[str, Any]]


@dataclass(frozen=True)
class _C3V32TrainingRouterRequest:
    module_id: int
    collector: C3V32TrainingRouterCollector


_C3_V32_TRAINING_ROUTER_REQUEST: ContextVar[
    _C3V32TrainingRouterRequest | None
] = ContextVar("c3_v32_training_router_request", default=None)


class _LearnedTriEvidenceRouterV32(nn.Module):
    """The only new learned submodule: 4,245 parameters and two state keys."""

    def __init__(self) -> None:
        super().__init__()
        self.value_proj = nn.Conv2d(480, 8, kernel_size=1, bias=False)
        self.activation = nn.SiLU()
        self.head = nn.Conv2d(15, 3, kernel_size=3, padding=1, bias=False)

    def encode_value(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.value_proj(value))

    def route(
        self, encoded_value: torch.Tensor, descriptor: torch.Tensor
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


class LearnedTriEvidenceProjectionV32(v31.C3DualRiskProjectionV31):
    """Layer-1 SSCA replacement with learned C/H/B spatial evidence."""

    def __init__(
        self,
        config: Any,
        vis: bool,
        channel_num: list[int] | tuple[int, ...],
        *,
        layer_index: int,
        gain_limit: float = SBSC_V32_GAIN_MAX,
        eps: float = SBSC_V32_EPS,
        detach_support: bool = True,
    ) -> None:
        # One fork covers the complete replacement construction.  Thus the
        # router receives the frozen seed-42 stream *after* the inherited
        # SSCA draws, exactly as specified, without consuming caller RNG.
        with torch.random.fork_rng(devices=[]):
            # Seed only the forked CPU generator.  torch.manual_seed would
            # also reset every initialized CUDA generator despite devices=[].
            torch.default_generator.manual_seed(_FROZEN_ARCHITECTURE_SEED)
            super().__init__(
                config,
                vis,
                channel_num,
                layer_index=layer_index,
                gain_limit=gain_limit,
                eps=eps,
                detach_support=detach_support,
            )
            self.tri_router = _LearnedTriEvidenceRouterV32()

    @classmethod
    def from_ssca(
        cls,
        source: Attention_org,
        *,
        layer_index: int,
    ) -> "LearnedTriEvidenceProjectionV32":
        if type(source) is not Attention_org:
            raise TypeError("source must be the exact frozen Attention_org class")
        if type(layer_index) is not int or layer_index != 1:
            raise ValueError("V3.2 replaces only zero-based SCTB layer 1")
        config = SimpleNamespace(KV_size=int(source.KV_size))
        reference = next(source.parameters())
        replacement = cls(
            config,
            bool(source.vis),
            tuple(int(value) for value in source.channel_num),
            layer_index=layer_index,
        )
        router_state = {
            key: value.detach().clone()
            for key, value in replacement.tri_router.state_dict().items()
        }
        replacement.to(device=reference.device, dtype=reference.dtype)
        incompatible = replacement.load_state_dict(source.state_dict(), strict=False)
        expected_missing = {
            _FROZEN_GAIN_SUFFIX,
            "tri_router.value_proj.weight",
            "tri_router.head.weight",
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise RuntimeError(
                "unexpected missing V3.2 replacement state: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected legacy V3.2 replacement state: "
                f"{incompatible.unexpected_keys}"
            )
        replacement.tri_router.to(
            device=reference.device, dtype=torch.float32
        )
        replacement.tri_router.load_state_dict(router_state, strict=True)
        with torch.no_grad():
            replacement.raw_dual_risk_level_gain.data = (
                replacement.raw_dual_risk_level_gain.detach().float()
            )
            replacement.raw_dual_risk_level_gain.zero_()
        if (
            torch.count_nonzero(replacement.tri_router.value_proj.weight).item()
            == 0
            or torch.count_nonzero(replacement.tri_router.head.weight).item() == 0
        ):
            raise RuntimeError("V3.2 router must have non-zero default initialization")
        replacement.train(source.training)
        return replacement

    @staticmethod
    def _spatial_energy(raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError("raw Q/K energy input must have shape [B,1,C,N]")
        with torch.no_grad(), torch.autocast(
            device_type=raw.device.type, enabled=False
        ):
            value = raw.detach().float()
            rms = torch.sqrt(value.square().mean(dim=-2, keepdim=True))
            log_rms = torch.log(rms + _FROZEN_EPS)
            centered = log_rms - log_rms.mean(dim=-1, keepdim=True)
            variance = centered.square().mean(dim=-1, keepdim=True)
            return torch.tanh(
                centered / (torch.sqrt(variance) + _FROZEN_EPS)
            )

    def _route_supports(
        self,
        *,
        static_support: v31.C3V31Support,
        raw_queries: tuple[torch.Tensor, ...],
        raw_key: torch.Tensor,
        value_spatial: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> tuple[
        tuple[v31.C3V31LevelSupport, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        height, width = token_hw
        positions = height * width
        if raw_key.shape[-1] != positions or value_spatial.shape[-2:] != token_hw:
            raise ValueError("router token geometry differs")
        with torch.autocast(device_type=value_spatial.device.type, enabled=False):
            encoded_value = self.tri_router.encode_value(value_spatial.float())
            key_energy = self._spatial_energy(raw_key)
            routed_levels: list[v31.C3V31LevelSupport] = []
            logits_records: list[torch.Tensor] = []
            support_records: list[torch.Tensor] = []
            for level, raw_query in zip(static_support.levels, raw_queries):
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
                logits = self.tri_router.route(encoded_value, descriptor).float()
                finite_row = torch.isfinite(logits).flatten(1).all(dim=1)
                safe_logits = torch.where(
                    finite_row[:, None, None, None],
                    logits,
                    torch.zeros_like(logits),
                )
                probability = F.softmax(safe_logits.flatten(2), dim=-1)
                consistent = probability[:, 0:1].unsqueeze(1)
                contradictory = probability[:, 1:2].unsqueeze(1)
                common = probability[:, 2:3].unsqueeze(1)
                valid = finite_row[:, None, None, None]
                mass = valid.to(probability.dtype)
                separation = 0.5 * (consistent - common).abs().sum(
                    dim=-1, keepdim=True
                )
                reliability = torch.where(
                    valid & static_support.key_confidence_valid,
                    torch.sqrt(
                        (
                            static_support.key_confidence * separation
                        ).clamp(0.0, 1.0)
                    ),
                    torch.zeros_like(separation),
                )
                routed_levels.append(
                    replace(
                        level,
                        consistent_raw=consistent,
                        contradictory_raw=contradictory,
                        common_raw=common,
                        consistent_mass=mass,
                        contradictory_mass=mass,
                        common_mass=mass,
                        consistent_support=consistent,
                        contradictory_support=contradictory,
                        common_support=common,
                        consistent_valid=valid,
                        contradictory_valid=valid,
                        common_valid=valid,
                        consistent_positive_strength=separation,
                        consistent_common_separation=separation,
                        reliability=reliability,
                    )
                )
                logits_records.append(logits)
                support_records.append(probability.reshape(
                    probability.shape[0], 3, height, width
                ))
        return (
            tuple(routed_levels),
            tuple(logits_records),
            tuple(support_records),
        )

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
        request = _C3_V32_TRAINING_ROUTER_REQUEST.get()
        context_capture = request is not None and request.module_id == id(self)
        if context_capture and (not self.training or not torch.is_grad_enabled()):
            raise RuntimeError(
                "V3.2 router capture is valid only for a gradient-enabled "
                "training forward"
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
            raise ValueError("V3.2 Q/K/V token grids must be identical")
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
        base_qkv_finite = torch.stack(
            tuple(
                torch.isfinite(tensor).all()
                for tensor in (*raw_queries, raw_key, value)
            )
        ).all()
        if not bool(base_qkv_finite):
            raise RuntimeError(
                "V3.2 base Q/K/V contract contains nonfinite values"
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
        base_attention_finite = torch.stack(
            tuple(
                torch.isfinite(attention).all()
                for attention in base_attentions
            )
        ).all()
        if not bool(base_attention_finite):
            raise RuntimeError(
                "V3.2 base attention contract contains nonfinite values"
            )
        with torch.autocast(device_type=raw_key.device.type, enabled=False):
            c3_queries = tuple(
                F.normalize(query.float(), dim=-1) for query in raw_queries
            )
            c3_key = F.normalize(raw_key.float(), dim=-1)
        static_support = _estimate_c3_v31_support_v32(
            c3_key,
            c3_queries,
            base_attentions,
            eps=self.eps,
            detach_support=True,
        )
        routed_levels, router_logits, router_supports = self._route_supports(
            static_support=static_support,
            raw_queries=raw_queries,
            raw_key=raw_key,
            value_spatial=value_spatial,
            token_hw=token_hw,
        )
        gain = self.effective_level_gain(None)
        attentions: list[torch.Tensor] = []
        diagnostics: list[dict[str, Any]] = []
        compact_diagnostics = {
            "projection_rows": 0,
            "solver_attempt_rows": 0,
            "solver_accepted_rows": 0,
            "solver_fallback_rows": 0,
            "emission_checked_rows": 0,
            "emission_accepted_rows": 0,
            "emission_fallback_rows": 0,
        }
        for level_index, (query, base_attention, level_support) in enumerate(
            zip(c3_queries, base_attentions, routed_levels)
        ):
            attention, level_diagnostics = self._project_level(
                level_index=level_index,
                query=query,
                key=c3_key,
                base_attention=base_attention,
                support=level_support,
                key_confidence=static_support.key_confidence,
                key_confidence_valid=static_support.key_confidence_valid,
                gain=gain,
                gain_override=None,
                intervention="full",
            )
            attentions.append(attention)
            if context_capture:
                projection = level_diagnostics.get("projection")
                base_contract_failure = bool(
                    level_diagnostics.get("base_contract_failure", False)
                )
                if base_contract_failure or projection is None:
                    row_flag = level_diagnostics["emission_fallback"].detach()
                    rows = int(row_flag.numel())
                    compact_diagnostics["projection_rows"] += rows
                    compact_diagnostics["emission_checked_rows"] += rows
                    compact_diagnostics["emission_fallback_rows"] += rows
                else:
                    accepted = projection.accepted.detach()
                    solver_fallback = projection.solver_fallback.detach()
                    emission_fallback = level_diagnostics[
                        "emission_fallback"
                    ].detach()
                    solver_attempt = accepted | solver_fallback
                    rows = int(accepted.numel())
                    compact_diagnostics["projection_rows"] += rows
                    compact_diagnostics["solver_attempt_rows"] += int(
                        solver_attempt.sum().item()
                    )
                    compact_diagnostics["solver_accepted_rows"] += int(
                        accepted.sum().item()
                    )
                    compact_diagnostics["solver_fallback_rows"] += int(
                        solver_fallback.sum().item()
                    )
                    compact_diagnostics["emission_checked_rows"] += rows
                    compact_diagnostics["emission_fallback_rows"] += int(
                        emission_fallback.sum().item()
                    )
                    compact_diagnostics["emission_accepted_rows"] += int(
                        (~emission_fallback).sum().item()
                    )
            if collect_diagnostics:
                level_diagnostics["support"] = level_support
                level_diagnostics["router_logits"] = router_logits[level_index]
                level_diagnostics["router_supports"] = router_supports[level_index]
                diagnostics.append(level_diagnostics)

        projections = (
            self.project_out1,
            self.project_out2,
            self.project_out3,
            self.project_out4,
        )
        outputs: list[torch.Tensor] = []
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
                    "schema": SBSC_V32_SCHEMA + "/training_router_capture/v1",
                    "module_id": id(self),
                    "batch_size": batch,
                    "token_hw": token_hw,
                    "logits": router_logits,
                    "supports": router_supports,
                    "compact_diagnostics": compact_diagnostics,
                }
            )
        return (
            (outputs[0], outputs[1], outputs[2], outputs[3], None),
            tuple(diagnostics),
        )


def _v32_modules(model: nn.Module) -> tuple[LearnedTriEvidenceProjectionV32, ...]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, LearnedTriEvidenceProjectionV32)
    )


@contextmanager
def capture_c3_v32_training_router(model: nn.Module):
    """Capture live router logits during one ordinary training forward."""

    if type(model) is not SCTransNet:
        raise TypeError("V3.2 training capture requires exact SCTransNet")
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError("V3.2 training capture requires training with gradients")
    modules = _v32_modules(model)
    if len(modules) != 1 or type(modules[0]) is not LearnedTriEvidenceProjectionV32:
        raise RuntimeError("training capture requires exactly one exact V3.2 block")
    if _C3_V32_TRAINING_ROUTER_REQUEST.get() is not None:
        raise RuntimeError("nested V3.2 training router capture is forbidden")
    collector = C3V32TrainingRouterCollector(records=[])
    token = _C3_V32_TRAINING_ROUTER_REQUEST.set(
        _C3V32TrainingRouterRequest(
            module_id=id(modules[0]), collector=collector
        )
    )
    try:
        yield collector
    finally:
        _C3_V32_TRAINING_ROUTER_REQUEST.reset(token)


def build_tri_router_targets_v32(
    target: torch.Tensor,
    detached_prediction: torch.Tensor,
    token_hw: tuple[int, int] | Sequence[int],
) -> TriRouterTargetsV32:
    """Build threshold-free C/H/B targets from one training batch."""

    if not isinstance(target, torch.Tensor) or not isinstance(
        detached_prediction, torch.Tensor
    ):
        raise TypeError("target and prediction must be tensors")
    if target.ndim != 4 or detached_prediction.ndim != 4:
        raise ValueError("target and prediction must be BCHW")
    if tuple(target.shape) != tuple(detached_prediction.shape) or target.shape[1] != 1:
        raise ValueError("target/prediction geometry differs")
    if target.device != detached_prediction.device:
        raise TypeError("target and prediction must share a device")
    if detached_prediction.requires_grad:
        raise ValueError("router teacher prediction must be explicitly detached")
    values = tuple(token_hw)
    if (
        len(values) != 2
        or any(type(value) is not int or value <= 0 for value in values)
        or values[0] * values[1] <= 1
    ):
        raise ValueError("token_hw must contain two positive ints with N>1")
    with torch.no_grad(), torch.autocast(
        device_type=target.device.type, enabled=False
    ):
        y = target.detach().float()
        p = detached_prediction.detach().float()
        if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(p).all()):
            raise ValueError("router supervision inputs must be finite")
        if bool(((y < 0.0) | (y > 1.0)).any()) or bool(
            ((p < 0.0) | (p > 1.0)).any()
        ):
            raise ValueError("router supervision inputs must be probabilities")
        pooled_target = F.adaptive_max_pool2d(y, values)
        pooled_prediction = F.adaptive_max_pool2d(p, values)
        outside = 1.0 - pooled_target
        raw = torch.cat(
            (
                pooled_target,
                outside
                * (-torch.log((1.0 - pooled_prediction).clamp_min(_FROZEN_EPS))),
                outside * (1.0 - pooled_prediction),
            ),
            dim=1,
        )
        mass = raw.flatten(2).sum(dim=-1)
        valid = mass > _FROZEN_EPS
        probability = torch.where(
            valid[:, :, None, None],
            raw / mass[:, :, None, None].clamp_min(_FROZEN_EPS),
            torch.zeros_like(raw),
        )
    return TriRouterTargetsV32(
        raw=raw,
        probability=probability,
        mass=mass,
        valid=valid,
        token_hw=(values[0], values[1]),
    )


def tri_router_supervision_loss(
    capture: C3V32TrainingRouterCollector,
    detached_prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the unit-weight normalized spatial router CE scalar."""

    if type(capture) is not C3V32TrainingRouterCollector:
        raise TypeError("capture must be a V3.2 training router collector")
    if not isinstance(detached_prediction, torch.Tensor) or not isinstance(
        target, torch.Tensor
    ):
        raise TypeError("router prediction and target must be tensors")
    if len(capture.records) != 1:
        raise RuntimeError("router loss requires exactly one captured forward")
    record = capture.records[0]
    logits_values = record.get("logits")
    if not isinstance(logits_values, tuple) or len(logits_values) != 4:
        raise RuntimeError("router capture must contain four live logits")
    for logits in logits_values:
        if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
            raise RuntimeError("captured router logits must be BCHW")
    if not bool(
        torch.stack(
            tuple(torch.isfinite(logits.detach()).all() for logits in logits_values)
        ).all()
    ):
        raise RuntimeError("captured router logits must be finite")
    losses: list[torch.Tensor] = []
    target_cache: dict[tuple[int, int], TriRouterTargetsV32] = {}
    for logits in logits_values:
        if logits.shape[1] != 3 or logits.device != target.device:
            raise RuntimeError("captured router logits geometry/device differs")
        token_hw = (int(logits.shape[-2]), int(logits.shape[-1]))
        targets = target_cache.get(token_hw)
        if targets is None:
            targets = build_tri_router_targets_v32(
                target, detached_prediction, token_hw
            )
            target_cache[token_hw] = targets
        if int(logits.shape[0]) != int(targets.probability.shape[0]):
            raise RuntimeError("captured router logits batch size differs")
        positions = int(logits.shape[-2] * logits.shape[-1])
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probability = F.log_softmax(
                logits.float().flatten(2), dim=-1
            )
            per_role = -(
                targets.probability.flatten(2) * log_probability
            ).sum(dim=-1) / math.log(float(positions))
        losses.extend(per_role[targets.valid].unbind())
    if not losses:
        raise RuntimeError("all V3.2 router supervision roles are empty")
    return torch.stack(losses).mean()


def replace_sctransnet_l1_ssca_with_sbsc_v32(
    model: SCTransNet,
) -> tuple[str, ...]:
    """Transactionally replace exactly one L1 Attention_org."""

    if type(model) is not SCTransNet:
        raise TypeError("SBSC V3.2 replacement requires exact SCTransNet")
    layers = tuple(model.mtc.encoder.layer)
    if len(layers) != _FROZEN_BLOCK_COUNT:
        raise RuntimeError("SCTransNet must contain exactly four SCTBs")
    sources = tuple(block.channel_attn for block in layers)
    if any(type(source) is not Attention_org for source in sources):
        raise TypeError("all four source operators must be exact Attention_org")
    source = sources[_FROZEN_REPLACED_BLOCK_INDEX]
    replacement = LearnedTriEvidenceProjectionV32.from_ssca(
        source, layer_index=_FROZEN_REPLACED_BLOCK_INDEX
    )
    source_state = source.state_dict()
    replacement_state = replacement.state_dict()
    expected_added = {
        _FROZEN_GAIN_SUFFIX,
        "tri_router.value_proj.weight",
        "tri_router.head.weight",
    }
    if set(replacement_state) - set(source_state) != expected_added:
        raise RuntimeError("SBSC V3.2 replacement added unexpected state")
    if any(
        not torch.equal(source_state[key], replacement_state[key])
        for key in source_state
    ):
        raise RuntimeError("SBSC V3.2 replacement changed shared state")
    layers[_FROZEN_REPLACED_BLOCK_INDEX].channel_attn = replacement
    return ("mtc.encoder.layer.1.channel_attn",)


@torch.no_grad()
def project_sbsc_v32_constraints_(model: nn.Module) -> None:
    """Clamp the one V3.2 gain vector after an optimizer step."""

    modules = _v32_modules(model)
    if len(modules) != 1 or type(modules[0]) is not LearnedTriEvidenceProjectionV32:
        raise RuntimeError("gain projection requires one exact V3.2 block")
    modules[0].project_level_gain_()


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


def _require_method(method: str) -> str:
    if type(method) is not str or method not in _FROZEN_METHODS:
        raise ValueError(f"method must be one of {_FROZEN_METHODS}")
    return method


def _require_dataset(dataset: str) -> str:
    if type(dataset) is not str or dataset not in _FROZEN_DATASETS:
        raise ValueError(f"dataset must be one of {_FROZEN_DATASETS}")
    return dataset


def _require_architecture_seed(seed: int) -> int:
    if type(seed) is not int or seed != _FROZEN_ARCHITECTURE_SEED:
        raise ValueError("V3.2 freezes architecture_seed=42")
    return seed


def _validate_frozen_l1_ssca_operators(
    module: LearnedTriEvidenceProjectionV32,
) -> None:
    """Freeze every inherited L1 SSCA operator attribute used by forward."""

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
        "tri_router",
    )
    if tuple(module._modules) != expected_children:
        raise RuntimeError("V3.2 L1 SSCA module tree differs")
    conv_schema = {
        "mhead1": (32, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "mhead2": (64, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "mhead3": (128, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "mhead4": (256, 256, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "mheadk": (480, 480, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "mheadv": (480, 480, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "q1": (32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 16),
        "q2": (64, 64, (3, 3), (1, 1), (1, 1), (1, 1), 32),
        "q3": (128, 128, (3, 3), (1, 1), (1, 1), (1, 1), 64),
        "q4": (256, 256, (3, 3), (1, 1), (1, 1), (1, 1), 128),
        "k": (480, 480, (3, 3), (1, 1), (1, 1), (1, 1), 480),
        "v": (480, 480, (3, 3), (1, 1), (1, 1), (1, 1), 480),
        "project_out1": (32, 32, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "project_out2": (64, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "project_out3": (128, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1),
        "project_out4": (256, 256, (1, 1), (1, 1), (0, 0), (1, 1), 1),
    }
    for name, expected in conv_schema.items():
        conv = getattr(module, name)
        actual = (
            conv.in_channels if type(conv) is nn.Conv2d else None,
            conv.out_channels if type(conv) is nn.Conv2d else None,
            conv.kernel_size if type(conv) is nn.Conv2d else None,
            conv.stride if type(conv) is nn.Conv2d else None,
            conv.padding if type(conv) is nn.Conv2d else None,
            conv.dilation if type(conv) is nn.Conv2d else None,
            conv.groups if type(conv) is nn.Conv2d else None,
        )
        if (
            type(conv) is not nn.Conv2d
            or actual != expected
            or conv.padding_mode != "zeros"
            or conv.bias is not None
            or conv.transposed is not False
            or conv.output_padding != (0, 0)
            or conv.weight.requires_grad is not True
            or "_conv_forward" in conv.__dict__
        ):
            raise RuntimeError(f"V3.2 inherited L1 SSCA conv {name!r} differs")


def validate_sbsc_v32_state_dict(
    state: Mapping[str, Any], method: str
) -> dict[str, Any]:
    """Validate exact baseline/V3.2 state before any live load."""

    _require_frozen_v31_solver_source()
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
    expected = {
        key: (shape, dtype)
        for key, shape, dtype in v31._authority_state_schema()
    }
    if method == "sbsc_v32":
        expected[_FROZEN_GAIN_STATE_KEY] = ((4,), torch.float32)
        for key, shape in _FROZEN_ROUTER_STATE_SCHEMA:
            expected[key] = (shape, torch.float32)
    if set(canonical) != set(expected):
        missing = sorted(set(expected) - set(canonical))
        unexpected = sorted(set(canonical) - set(expected))
        raise ValueError(
            "state key set differs from the exact V3.2 architecture schema; "
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
    if method == "sbsc_v32":
        gain = canonical[_FROZEN_GAIN_STATE_KEY]
        if bool(((gain < _FROZEN_GAIN_MIN) | (gain > _FROZEN_GAIN_MAX)).any()):
            raise ValueError("V3.2 gain state is outside [0,0.25]")
        gains = [float(value) for value in gain.detach().cpu().tolist()]
    return {
        "schema": SBSC_V32_SCHEMA + "/state_validation/v1",
        "method": method,
        "state_key_count": len(tensors),
        "gain_state_keys": (
            [f"{prefix}{_FROZEN_GAIN_STATE_KEY}"] if method == "sbsc_v32" else []
        ),
        "router_state_keys": (
            [f"{prefix}{key}" for key in SBSC_V32_ROUTER_STATE_KEYS]
            if method == "sbsc_v32"
            else []
        ),
        "gain_values": gains,
        "gain_bounds": (
            [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX]
            if method == "sbsc_v32"
            else None
        ),
        "data_parallel_prefix": bool(prefix),
    }


def _require_exact_runtime_module_source(
    module_name: str, relative_path: str
) -> None:
    """Bind an imported dependency to the source file hashed by provenance."""

    module = sys.modules.get(module_name)
    source_text = getattr(module, "__file__", None)
    project_root = Path(__file__).resolve(strict=True).parents[1]
    expected = project_root / relative_path
    if (
        module is None
        or not isinstance(source_text, str)
        or expected.is_symlink()
        or not expected.is_file()
    ):
        raise RuntimeError("V3.2 runtime dependency source is unavailable")
    source = Path(source_text)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("V3.2 runtime dependency source is not regular")
    if source.resolve(strict=True) != expected.resolve(strict=True):
        raise RuntimeError("V3.2 runtime dependency source differs from provenance")


def validate_sbsc_v32_runtime_integration() -> dict[str, str]:
    """Validate the adapter call edge and actual imported source closure."""

    _require_frozen_v31_solver_source()
    forward = LearnedTriEvidenceProjectionV32._forward_impl
    adapter_name = "_estimate_c3_v31_support_v32"
    if (
        adapter_name not in forward.__code__.co_names
        or forward.__globals__.get(adapter_name)
        is not _FROZEN_V32_SUPPORT_ADAPTER_FUNCTION
        or globals().get(adapter_name)
        is not _FROZEN_V32_SUPPORT_ADAPTER_FUNCTION
    ):
        raise RuntimeError("V3.2 forward is not bound to the frozen CUDA adapter")
    _require_exact_runtime_module_source(
        v31.__name__, "experiments/sctransnet_sbsc_v31.py"
    )
    _require_exact_runtime_module_source(
        SCTransNet.__module__, "model/_internal/SCTransNet.py"
    )
    _require_exact_runtime_module_source(
        state_dict_sha256.__module__,
        "experiments/four_dataset_models_seed42_v1.py",
    )
    return {
        "schema": _FROZEN_RUNTIME_INTEGRATION_SCHEMA,
        "cuda_median_adapter": _FROZEN_CUDA_MEDIAN_ADAPTER_SCHEMA,
    }


def validate_sctransnet_sbsc_v32(
    model: nn.Module, require_zero_gain: bool = False
) -> dict[str, Any]:
    """Validate the exact learned-router V3.2 graph and state."""

    if type(require_zero_gain) is not bool:
        raise TypeError("require_zero_gain must be bool")
    validate_sbsc_v32_runtime_integration()
    if type(model) is not SCTransNet:
        raise TypeError("validator requires exact unwrapped SCTransNet")
    if bool(getattr(model, "diagnostic_only", False)):
        raise RuntimeError("diagnostic-only adapters are forbidden in formal runs")
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
    if type(modules[1]) is not LearnedTriEvidenceProjectionV32:
        raise TypeError("zero-based SCTB layer 1 must contain exact V3.2 class")
    if any(type(modules[index]) is not Attention_org for index in (0, 2, 3)):
        raise TypeError("SCTB layers 0,2,3 must retain exact Attention_org")
    discovered = _v32_modules(model)
    if len(discovered) != 1 or discovered[0] is not modules[1]:
        raise RuntimeError("V3.2 module must exist only at SCTB layer 1")
    module = modules[1]
    assert type(module) is LearnedTriEvidenceProjectionV32
    replacement_helper_names = (
        "_forward_impl",
        "_route_supports",
        "_spatial_energy",
        "effective_level_gain",
        "_project_level",
        "_conditional_attention",
        "project_level_gain_",
    )
    if any(name in module.__dict__ for name in replacement_helper_names):
        raise RuntimeError("V3.2 replacement helper instance shadow detected")
    if (
        module.layer_index != 1
        or module.gain_limit != _FROZEN_GAIN_MAX
        or module.eps != _FROZEN_EPS
        or module.detach_support is not True
    ):
        raise RuntimeError("V3.2 frozen scalar contract differs")
    if module.num_attention_heads != 1 or module.KV_size != 480:
        raise RuntimeError("V3.2 attention geometry differs")
    if tuple(module.channel_num) != (32, 64, 128, 256):
        raise RuntimeError("V3.2 channel geometry differs")
    _validate_frozen_l1_ssca_operators(module)
    if module.vis is not False:
        raise RuntimeError("V3.2 freezes vis=False")
    if type(module.psi) is not nn.InstanceNorm2d:
        raise TypeError("V3.2 must retain exact InstanceNorm2d")
    if (
        module.psi.num_features != 1
        or module.psi.eps != 1e-5
        or module.psi.momentum != 0.1
        or module.psi.affine is not False
        or module.psi.track_running_stats is not False
        or "_apply_instance_norm" in module.psi.__dict__
        or "_check_input_dim" in module.psi.__dict__
    ):
        raise RuntimeError("V3.2 InstanceNorm2d contract differs")
    if type(module.softmax) is not nn.Softmax or module.softmax.dim != 3:
        raise TypeError("V3.2 must retain exact Softmax(dim=3)")
    if type(module.tri_router) is not _LearnedTriEvidenceRouterV32:
        raise TypeError("V3.2 requires the exact learned router")
    if tuple(module.tri_router._modules) != (
        "value_proj",
        "activation",
        "head",
    ) or module.tri_router._parameters or module.tri_router._buffers:
        raise RuntimeError("V3.2 router module tree differs")
    if any(
        name in module.tri_router.__dict__ for name in ("encode_value", "route")
    ):
        raise RuntimeError("V3.2 router helper instance shadow detected")
    if (
        type(module.tri_router.value_proj) is not nn.Conv2d
        or module.tri_router.value_proj.in_channels != 480
        or module.tri_router.value_proj.out_channels != 8
        or module.tri_router.value_proj.kernel_size != (1, 1)
        or module.tri_router.value_proj.stride != (1, 1)
        or module.tri_router.value_proj.padding != (0, 0)
        or module.tri_router.value_proj.dilation != (1, 1)
        or module.tri_router.value_proj.groups != 1
        or module.tri_router.value_proj.padding_mode != "zeros"
        or module.tri_router.value_proj.bias is not None
        or "_conv_forward" in module.tri_router.value_proj.__dict__
        or type(module.tri_router.activation) is not nn.SiLU
        or module.tri_router.activation.inplace is not False
        or type(module.tri_router.head) is not nn.Conv2d
        or module.tri_router.head.in_channels != 15
        or module.tri_router.head.out_channels != 3
        or module.tri_router.head.kernel_size != (3, 3)
        or module.tri_router.head.stride != (1, 1)
        or module.tri_router.head.padding != (1, 1)
        or module.tri_router.head.dilation != (1, 1)
        or module.tri_router.head.groups != 1
        or module.tri_router.head.padding_mode != "zeros"
        or module.tri_router.head.bias is not None
        or "_conv_forward" in module.tri_router.head.__dict__
    ):
        raise RuntimeError("V3.2 router layer contract differs")
    gain = module._parameters.get(_FROZEN_GAIN_SUFFIX)
    if (
        gain is not module.raw_dual_risk_level_gain
        or tuple(gain.shape) != (4,)
        or gain.dtype is not torch.float32
        or gain.requires_grad is not True
    ):
        raise RuntimeError("V3.2 gain registration differs")
    if any(
        parameter.requires_grad is not True
        for parameter in module.tri_router.parameters()
    ):
        raise RuntimeError("V3.2 router parameters must require gradients")
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
    validation = validate_sbsc_v32_state_dict(model.state_dict(), "sbsc_v32")
    if require_zero_gain and torch.count_nonzero(gain).item() != 0:
        raise RuntimeError("V3.2 identity gain is not exactly zero")
    if v31._parameter_count(model) != _FROZEN_CANDIDATE_PARAMETER_COUNT:
        raise RuntimeError("V3.2 parameter-count contract differs")
    if len(model.state_dict()) != _FROZEN_CANDIDATE_STATE_KEY_COUNT:
        raise RuntimeError("V3.2 state-key count contract differs")
    return {
        "schema": SBSC_V32_SCHEMA,
        "model": SBSC_V32_MODEL_NAME,
        "state_key_count": len(model.state_dict()),
        "parameter_count": v31._parameter_count(model),
        "router_parameter_count": _FROZEN_ROUTER_PARAMETER_COUNT,
        "router_state_keys": list(SBSC_V32_ROUTER_STATE_KEYS),
        "gain_state_keys": [_FROZEN_GAIN_STATE_KEY],
        "gain_bounds": [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX],
        "replaced_block_indices": [1],
        "independent_spatial_role_softmax": True,
        "live_value_evidence": True,
        "train_only_router_supervision": True,
        "loss_schema": _FROZEN_LOSS_SCHEMA,
        "v31_solver_source_sha256": V31_SOLVER_SOURCE_SHA256,
        "cuda_strict_deterministic_median_adapter": (
            _FROZEN_CUDA_MEDIAN_ADAPTER_SCHEMA
        ),
        "state_validation": validation,
    }


def structurally_inactive_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return v31.structurally_inactive_parameter_names(model)


def _method_metadata(
    model: SCTransNet,
    *,
    method: str,
    dataset: str,
    shared_state_sha256: str,
) -> dict[str, Any]:
    is_candidate = method == "sbsc_v32"
    return {
        "schema": SBSC_V32_SCHEMA,
        "model": SBSC_V32_MODEL_NAME if is_candidate else "SCTransNet",
        "method": method,
        "dataset": dataset,
        "architecture_seed": _FROZEN_ARCHITECTURE_SEED,
        "state_key_count": len(model.state_dict()),
        "parameter_count": v31._parameter_count(model),
        "state_sha256": state_dict_sha256(model.state_dict()),
        "shared_state_sha256": shared_state_sha256,
        "paired_initialization": True,
        "test_split_accessed": False,
        "gain_state_keys": [_FROZEN_GAIN_STATE_KEY] if is_candidate else [],
        "gain_bounds": [_FROZEN_GAIN_MIN, _FROZEN_GAIN_MAX] if is_candidate else None,
        "router_state_keys": list(SBSC_V32_ROUTER_STATE_KEYS) if is_candidate else [],
        "router_parameter_count": _FROZEN_ROUTER_PARAMETER_COUNT if is_candidate else 0,
        "router_initial_state_sha256": (
            state_dict_sha256(
                model.mtc.encoder.layer[1].channel_attn.tri_router.state_dict()
            )
            if is_candidate
            else None
        ),
        "loss_schema": _FROZEN_LOSS_SCHEMA if is_candidate else "sum_of_six_BCELoss_mean_terms",
        "replaced_block_indices": [1] if is_candidate else [],
        "v31_solver_source_sha256": V31_SOLVER_SOURCE_SHA256 if is_candidate else None,
        "cuda_strict_deterministic_median_adapter": (
            _FROZEN_CUDA_MEDIAN_ADAPTER_SCHEMA if is_candidate else None
        ),
    }


def build_paired_sctransnet_sbsc_v32(
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
) -> tuple[SCTransNet, SCTransNet, dict[str, Any]]:
    """Build a fresh bitwise-paired SCTransNet/V3.2 pair."""

    _require_frozen_v31_solver_source()
    dataset = _require_dataset(dataset)
    _require_architecture_seed(architecture_seed)
    baseline = _construct_original(_FROZEN_ARCHITECTURE_SEED)
    candidate = copy.deepcopy(baseline)
    replace_sctransnet_l1_ssca_with_sbsc_v32(candidate)
    router_initial_hash = state_dict_sha256(
        candidate.mtc.encoder.layer[1].channel_attn.tri_router.state_dict()
    )
    if router_initial_hash != EXPECTED_SBSC_V32_ROUTER_INIT_SHA256:
        raise RuntimeError("V3.2 router seed-42 initialization differs")
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    expected_added = {_FROZEN_GAIN_STATE_KEY, *SBSC_V32_ROUTER_STATE_KEYS}
    if set(candidate_state) - set(baseline_state) != expected_added:
        raise RuntimeError("paired candidate state additions differ")
    if set(baseline_state) - set(candidate_state):
        raise RuntimeError("paired candidate removed baseline state")
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
    validate_sctransnet_sbsc_v32(candidate, require_zero_gain=True)
    validate_sbsc_v32_state_dict(baseline_state, "sctransnet")
    baseline_metadata = _method_metadata(
        baseline,
        method="sctransnet",
        dataset=dataset,
        shared_state_sha256=shared_hash,
    )
    candidate_metadata = _method_metadata(
        candidate,
        method="sbsc_v32",
        dataset=dataset,
        shared_state_sha256=shared_hash,
    )
    metadata = dict(candidate_metadata)
    metadata.update(
        {
            "pair_schema": SBSC_V32_SCHEMA,
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


def build_sctransnet_sbsc_v32_method(
    method: str,
    dataset: str,
    architecture_seed: int = ARCHITECTURE_SEED,
    training: bool = True,
) -> tuple[SCTransNet, dict[str, Any]]:
    """Build the complete fresh pair and return exactly one method role."""

    method = _require_method(method)
    if type(training) is not bool:
        raise TypeError("training must be bool")
    baseline, candidate, pair = build_paired_sctransnet_sbsc_v32(
        dataset, architecture_seed
    )
    model = baseline if method == "sctransnet" else candidate
    model.train(training)
    model.mode = "train" if training else "test"
    metadata = dict(pair["baseline" if method == "sctransnet" else "candidate"])
    metadata.update(
        {
            "training": training,
            "pair": pair,
            "test_split_accessed": False,
        }
    )
    return model, metadata


solve_dual_risk_projection_v31 = _FROZEN_V31_SOLVER_FUNCTION
C3V32 = LearnedTriEvidenceProjectionV32


__all__ = [
    "ARCHITECTURE_SEED",
    "C3V32",
    "C3V32TrainingRouterCollector",
    "EXPECTED_SBSC_V32_BLOCKS",
    "EXPECTED_SBSC_V32_PARAMETER_COUNT",
    "EXPECTED_SBSC_V32_STATE_KEY_COUNT",
    "EXPECTED_SBSC_V32_ROUTER_INIT_SHA256",
    "EXPECTED_V31_SOLVER_SOURCE_SHA256",
    "LearnedTriEvidenceProjectionV32",
    "SBSC_V32_EPS",
    "SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA",
    "SBSC_V32_GAIN_MAX",
    "SBSC_V32_GAIN_MIN",
    "SBSC_V32_GAIN_STATE_KEY",
    "SBSC_V32_GAIN_SUFFIX",
    "SBSC_V32_LOSS_SCHEMA",
    "SBSC_V32_MODEL_NAME",
    "SBSC_V32_ROUTER_PARAMETER_COUNT",
    "SBSC_V32_ROUTER_STATE_KEYS",
    "SBSC_V32_RUNTIME_INTEGRATION_SCHEMA",
    "SBSC_V32_SCHEMA",
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
    "TriRouterTargetsV32",
    "V31_SOLVER_SOURCE_SHA256",
    "build_paired_sctransnet_sbsc_v32",
    "build_sctransnet_sbsc_v32_method",
    "build_tri_router_targets_v32",
    "capture_c3_v32_training_router",
    "project_sbsc_v32_constraints_",
    "replace_sctransnet_l1_ssca_with_sbsc_v32",
    "solve_dual_risk_projection_v31",
    "structurally_inactive_parameter_names",
    "tri_router_supervision_loss",
    "validate_sbsc_v32_state_dict",
    "validate_sbsc_v32_runtime_integration",
    "validate_sctransnet_sbsc_v32",
]
