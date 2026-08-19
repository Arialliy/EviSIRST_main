"""IRSTD CP-HF-S2 with a dilated coarse-support precision guard.

The previous CP-HF-S2 candidate improved target detection but increased
false objects.  Its existing deep-supervision fusion logit ``d0`` contains
useful multiscale negative evidence, yet the frozen inference path discards
``d0`` and returns only the final decoder logit ``out``.  This architecture
keeps the audited CP-HF-S2 feature correction and adds one operation:

``support = max_pool2d(sigmoid(d0), kernel_size=9)``

``guarded_out = out - strength * sigmoid(out) * (1 - support)``

Both logits are detached when forming the coarse-support evidence.  The
non-negative strength is bounded by three and is exactly zero at construction,
so the complete graph is an exact clean-model identity at initialization.
The boundary parameter nevertheless receives a first-step gradient.  A
negative parameter update safely falls back to zero correction.  The fifth
training head remains the raw ``d0`` probability; the sixth training head and
the evaluation result are the guarded ``out`` probability.

No checkpoint is loaded and no frozen source is modified by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.module as torch_module_hooks

from experiments import irstd_cp_hf_s2_v1 as cp_hf
from model.EviSIRST import EviSIRST, initialize_evisirst
from model import (
    tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival as formal_base,
)
from model.tpd_frequency_gate import FixedHaarAnalysis
from model.tpd_frequency_gate_v2_croa import (
    QueryFrequencyLevelGateV2CROA,
    QueryOnlyFrequencyGateV2CROA,
    validate_formal_qfg_v2_croa,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    TailAwarePersistentDCOffsetEvidenceRelay,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward
from model.tpd_sctransnet import ExplicitRelayUpBlock


EXPERIMENT_SCHEMA = "evisirst_irstd_cp_hf_s2_dcspg_v1"
ARCHITECTURE_SCHEMA = "evisirst_irstd_cp_hf_s2_dcspg_architecture_v1"
EXPERIMENT_NAME = "IRSTD-CP-HF-S2-DCSPG-v1"
ARCHITECTURE_NAME = EXPERIMENT_NAME
SUPPORTED_DATASET = "IRSTD-1K"

_CANONICAL_SOURCE_DEPENDENCY_ITEMS: tuple[tuple[str, str], ...] = (
    (
        "experiments/irstd_cp_hf_s2_v1.py",
        "0dba0021dfe8f46d8c4e2932b478ce03609ab14be73b20c8b77817dce240436a",
    ),
)
_CANONICAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Public discovery fields are descriptive copies only.  The validator below
# captures the canonical path/root/digest in defaults at function definition,
# so synchronously rewriting these public globals cannot redirect trust.
SOURCE_DEPENDENCIES: tuple[str, ...] = tuple(
    path for path, _digest in _CANONICAL_SOURCE_DEPENDENCY_ITEMS
)
SOURCE_DEPENDENCY_SHA256: dict[str, str] = dict(
    _CANONICAL_SOURCE_DEPENDENCY_ITEMS
)
PROJECT_ROOT = _CANONICAL_PROJECT_ROOT
AUTHORITATIVE_BASE_MANIFEST_SHA256 = (
    "01836fa26f47f63c4a4362e22a3f66955a61a3fc2adba9fd76d63c6758b15b00"
)
CANONICAL_BASE_STATE_KEYS_SHA256 = (
    "fa5b57ea63c1ea2a7d9711fa935b87bcdc3501d6439539b26da476a4fe7def0d"
)

BASE_STATE_KEY_COUNT = cp_hf.BASE_STATE_KEY_COUNT
BASE_PARAMETER_COUNT = cp_hf.BASE_PARAMETER_COUNT
CP_MODULE_NAME = cp_hf.MODULE_NAME
CP_STATE_PREFIX = cp_hf.STATE_PREFIX
CP_STATE_KEY_COUNT = cp_hf.EXTENSION_STATE_KEY_COUNT
CP_PARAMETER_COUNT = cp_hf.EXTENSION_PARAMETER_COUNT

GUARD_MODULE_NAME = "dilated_coarse_support_precision_guard"
GUARD_STATE_PREFIX = f"{GUARD_MODULE_NAME}."
GUARD_STATE_KEYS = (f"{GUARD_STATE_PREFIX}raw_strength",)
GUARD_STATE_KEY_COUNT = 1
GUARD_PARAMETER_COUNT = 1
SUPPORT_KERNEL_SIZE = 9
MAX_LOGIT_DELTA = 3.0

EXTENSION_STATE_KEY_COUNT = CP_STATE_KEY_COUNT + GUARD_STATE_KEY_COUNT
EXTENSION_PARAMETER_COUNT = CP_PARAMETER_COUNT + GUARD_PARAMETER_COUNT
FORMAL_STATE_KEY_COUNT = BASE_STATE_KEY_COUNT + EXTENSION_STATE_KEY_COUNT
FORMAL_PARAMETER_COUNT = BASE_PARAMETER_COUNT + EXTENSION_PARAMETER_COUNT

_BASE_STATE_KEYS_ATTRIBUTE = "_irstd_cp_hf_s2_dcspg_base_state_keys"
_BASE_MANIFEST_ATTRIBUTE = "_irstd_cp_hf_s2_dcspg_base_manifest"
_ARCHITECTURE_SEED_ATTRIBUTE = "_irstd_cp_hf_s2_dcspg_architecture_seed"
_INITIALIZATION_SEED_ATTRIBUTE = "_irstd_cp_hf_s2_dcspg_initialization_seed"

_FROZEN_MODEL_FORWARD = EviSIRST.forward
_FROZEN_MODEL_MANIFEST = EviSIRST.architecture_manifest
_FROZEN_MODEL_EXPLICIT_EMBEDDINGS = EviSIRST.explicit_embeddings
_FROZEN_QFG_CLASS = QueryOnlyFrequencyGateV2CROA
_FROZEN_QFG_LEVEL_CLASS = QueryFrequencyLevelGateV2CROA
_FROZEN_HAAR_CLASS = FixedHaarAnalysis
_FROZEN_NER_CLASS = TailAwarePersistentDCOffsetEvidenceRelay
_FROZEN_DECODER_CLASS = ExplicitRelayUpBlock
_FROZEN_CONV2D_CLASS = nn.Conv2d
_FROZEN_SEQUENTIAL_CLASS = nn.Sequential
_FROZEN_MODULE_LIST_CLASS = nn.ModuleList
_FROZEN_GELU_CLASS = nn.GELU
_FROZEN_CP_FORWARD = cp_hf.ContextPurifiedHighFrequencyS2.forward
_FROZEN_CP_COMPONENTS = (
    cp_hf.ContextPurifiedHighFrequencyS2.refinement_components
)
_FROZEN_CP_MANIFEST = cp_hf.ContextPurifiedHighFrequencyS2.architecture_manifest
_FROZEN_CONV2D_FORWARD = nn.Conv2d.forward
_FROZEN_SEQUENTIAL_FORWARD = nn.Sequential.forward
_FROZEN_GELU_FORWARD = nn.GELU.forward
_FROZEN_HAAR_FORWARD = FixedHaarAnalysis.forward
_FROZEN_QFG_PREPARE = QueryOnlyFrequencyGateV2CROA.prepare
_FROZEN_QFG_APPLY_PREPARED = QueryOnlyFrequencyGateV2CROA.apply_prepared
_FROZEN_QFG_MANIFEST = QueryOnlyFrequencyGateV2CROA.architecture_manifest
_FROZEN_QFG_NORMALIZE_QUERY_SIZES = (
    QueryOnlyFrequencyGateV2CROA._normalize_query_sizes
)
_FROZEN_QFG_LEVEL_PREPARE = QueryFrequencyLevelGateV2CROA.prepare
_FROZEN_QFG_LEVEL_APPLY_PREPARED = (
    QueryFrequencyLevelGateV2CROA.apply_prepared
)
_FROZEN_QFG_LEVEL_ALIGN_PRIOR = QueryFrequencyLevelGateV2CROA._align_prior
_FROZEN_NER_FORWARD_STAGE = (
    TailAwarePersistentDCOffsetEvidenceRelay.forward_stage
)
_FROZEN_NER_DC_SUPPORT = TailAwarePersistentDCOffsetEvidenceRelay.dc_support
_FROZEN_NER_PERSISTENT_TAIL_SUPPORT = (
    TailAwarePersistentDCOffsetEvidenceRelay._persistent_tail_support
)
_FROZEN_NER_TAIL_SUPPORT = (
    TailAwarePersistentDCOffsetEvidenceRelay._tail_support
)
_FROZEN_DECODER_PREPARE = ExplicitRelayUpBlock.prepare
_FROZEN_DECODER_FINISH = ExplicitRelayUpBlock.finish
_FROZEN_FREQUENCY_ENCODER_FORWARD = frequency_encoder_forward
_FROZEN_QFG_VALIDATOR = validate_formal_qfg_v2_croa


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _validate_source_dependencies(
    *,
    _project_root: Path = _CANONICAL_PROJECT_ROOT,
    _dependency_items: tuple[tuple[str, str], ...] = (
        (
            "experiments/irstd_cp_hf_s2_v1.py",
            "0dba0021dfe8f46d8c4e2932b478ce03609ab14be73b20c8b77817dce240436a",
        ),
    ),
) -> dict[str, str]:
    """Hash canonical dependencies without trusting writable public fields."""

    observed: dict[str, str] = {}
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for relative_path, expected_sha256 in _dependency_items:
        pure = PurePosixPath(relative_path)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative_path
            or ".." in pure.parts
            or relative_path in ("", ".")
        ):
            raise RuntimeError("DCS-PG source-dependency path is not normalized")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise RuntimeError("DCS-PG source-dependency SHA-256 is malformed")
        path = _project_root.joinpath(*pure.parts)
        current = _project_root
        for component in pure.parts:
            current = current / component
            if current.is_symlink():
                raise RuntimeError("DCS-PG source dependency contains a symlink")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise RuntimeError("DCS-PG source dependency cannot be opened safely") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError("DCS-PG source dependency is not a regular file")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError("DCS-PG source dependency changed while hashing")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError("DCS-PG source-dependency SHA-256 differs")
        observed[relative_path] = actual_sha256
    return observed


_FORWARD_HOOK_ATTRIBUTES = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)

_GLOBAL_RUNTIME_HOOK_ATTRIBUTES = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
)


def _validate_runtime_graph_seal(model: nn.Module) -> None:
    for attribute in _GLOBAL_RUNTIME_HOOK_ATTRIBUTES:
        hooks = getattr(torch_module_hooks, attribute, None)
        if hooks:
            raise RuntimeError(f"global runtime hook registry is nonempty: {attribute}")
    for name, module in model.named_modules():
        label = name or "<root>"
        if "forward" in vars(module):
            raise RuntimeError(f"instance forward shadow exists at {label}")
        for attribute in _FORWARD_HOOK_ATTRIBUTES:
            hooks = getattr(module, attribute, None)
            if hooks:
                raise RuntimeError(f"runtime hook exists at {label}.{attribute}")


def _validate_bound_method(
    instance: Any, *, name: str, expected: Any, label: str
) -> None:
    if name in vars(instance):
        raise RuntimeError(f"instance method shadow exists at {label}.{name}")
    class_value = getattr(type(instance), name, None)
    if class_value is not expected:
        raise RuntimeError(f"class method identity differs at {label}.{name}")
    bound = getattr(instance, name, None)
    if (
        getattr(bound, "__self__", None) is not instance
        or getattr(bound, "__func__", None) is not expected
    ):
        raise RuntimeError(f"bound method identity differs at {label}.{name}")


def _validate_static_method(
    instance: Any, *, name: str, expected: Any, label: str
) -> None:
    if name in vars(instance):
        raise RuntimeError(f"instance method shadow exists at {label}.{name}")
    class_value = getattr(type(instance), name, None)
    bound = getattr(instance, name, None)
    if class_value is not expected or bound is not expected:
        raise RuntimeError(f"class method identity differs at {label}.{name}")


def _validate_conv2d_contract(
    module: nn.Module,
    *,
    label: str,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
    bias: bool = False,
) -> None:
    if type(module) is not _FROZEN_CONV2D_CLASS:
        raise RuntimeError(f"QFG internal module type differs at {label}")
    expected = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": (1, 1),
        "padding": padding,
        "dilation": (1, 1),
        "groups": groups,
        "padding_mode": "zeros",
    }
    for name, value in expected.items():
        if getattr(module, name, None) != value:
            raise RuntimeError(f"QFG internal module {name} differs at {label}")
    if (module.bias is not None) is bias:
        return
    raise RuntimeError(f"QFG internal module bias contract differs at {label}")


def _validate_qfg_level_contract(
    module: nn.Module,
    *,
    index: int,
    feature_channels: int,
    expected_alignment: tuple[int, int],
) -> None:
    label = f"tpd_qfg.levels[{index}]"
    if type(module) is not _FROZEN_QFG_LEVEL_CLASS:
        raise RuntimeError(f"QFG level type differs at {label}")
    _validate_bound_method(
        module,
        name="prepare",
        expected=_FROZEN_QFG_LEVEL_PREPARE,
        label=label,
    )
    _validate_bound_method(
        module,
        name="apply_prepared",
        expected=_FROZEN_QFG_LEVEL_APPLY_PREPARED,
        label=label,
    )
    _validate_static_method(
        module,
        name="_align_prior",
        expected=_FROZEN_QFG_LEVEL_ALIGN_PRIOR,
        label=label,
    )
    expected_attributes = {
        "feature_channels": feature_channels,
        "mode": "high_low",
        "hidden_channels": 8,
        "expected_alignment": expected_alignment,
        "detach_frequency_source": True,
        "alpha_effective_init": 0.1,
        "eps": 1.0e-6,
        "validate_finite": True,
    }
    for name, expected in expected_attributes.items():
        observed = getattr(module, name, None)
        if observed != expected or (
            isinstance(expected, bool) and observed is not expected
        ):
            raise RuntimeError(f"QFG level {name} differs at {label}")
    if type(getattr(module, "_prepared_owner_token", None)) is not object:
        raise RuntimeError(f"QFG level owner token differs at {label}")
    if tuple(module._modules) != (
        "haar",
        "prior_projection",
        "spatial_projection",
        "gate_out",
    ):
        raise RuntimeError(f"QFG internal module set differs at {label}")

    haar = module.haar
    if type(haar) is not _FROZEN_HAAR_CLASS or haar.validate_finite is not False:
        raise RuntimeError(f"QFG Haar contract differs at {label}")
    if tuple(haar._buffers) != ("kernels",) or tuple(haar._parameters):
        raise RuntimeError(f"QFG Haar state contract differs at {label}")
    expected_haar = torch.tensor(
        (
            ((1.0, 1.0), (1.0, 1.0)),
            ((-1.0, -1.0), (1.0, 1.0)),
            ((-1.0, 1.0), (-1.0, 1.0)),
            ((1.0, -1.0), (-1.0, 1.0)),
        ),
        device=haar.kernels.device,
        dtype=haar.kernels.dtype,
    ).unsqueeze(1) / 2.0
    if not torch.equal(haar.kernels, expected_haar):
        raise RuntimeError(f"QFG Haar kernels differ at {label}")

    _validate_conv2d_contract(
        module.prior_projection,
        label=f"{label}.prior_projection",
        in_channels=4 * feature_channels,
        out_channels=8,
        kernel_size=(1, 1),
    )
    spatial = module.spatial_projection
    if (
        type(spatial) is not _FROZEN_SEQUENTIAL_CLASS
        or len(spatial) != 2
        or type(spatial[1]) is not _FROZEN_GELU_CLASS
        or spatial[1].approximate != "none"
    ):
        raise RuntimeError(f"QFG spatial projection structure differs at {label}")
    _validate_conv2d_contract(
        spatial[0],
        label=f"{label}.spatial_projection.0",
        in_channels=8,
        out_channels=8,
        kernel_size=(3, 3),
        padding=(1, 1),
        groups=8,
    )
    _validate_conv2d_contract(
        module.gate_out,
        label=f"{label}.gate_out",
        in_channels=8,
        out_channels=1,
        kernel_size=(1, 1),
    )


def _validate_qfg_runtime_contract(module: nn.Module) -> None:
    if type(module) is not _FROZEN_QFG_CLASS:
        raise RuntimeError("formal QFG module type differs")
    _validate_bound_method(
        module,
        name="prepare",
        expected=_FROZEN_QFG_PREPARE,
        label="tpd_qfg",
    )
    _validate_bound_method(
        module,
        name="apply_prepared",
        expected=_FROZEN_QFG_APPLY_PREPARED,
        label="tpd_qfg",
    )
    _validate_bound_method(
        module,
        name="architecture_manifest",
        expected=_FROZEN_QFG_MANIFEST,
        label="tpd_qfg",
    )
    _validate_static_method(
        module,
        name="_normalize_query_sizes",
        expected=_FROZEN_QFG_NORMALIZE_QUERY_SIZES,
        label="tpd_qfg",
    )
    if type(getattr(module, "_prepared_owner_token", None)) is not object:
        raise RuntimeError("formal QFG owner token differs")
    levels = getattr(module, "levels", None)
    if type(levels) is not _FROZEN_MODULE_LIST_CLASS or len(levels) != 4:
        raise RuntimeError("formal QFG levels container differs")
    if tuple(module._modules) != ("levels",):
        raise RuntimeError("formal QFG internal module set differs")
    expected_channels = (32, 64, 128, 256)
    expected_alignments = ((8, 8), (4, 4), (2, 2), (1, 1))
    owner_tokens = [module._prepared_owner_token]
    for index, (level, channels, alignment) in enumerate(
        zip(levels, expected_channels, expected_alignments)
    ):
        _validate_qfg_level_contract(
            level,
            index=index,
            feature_channels=channels,
            expected_alignment=alignment,
        )
        owner_tokens.append(level._prepared_owner_token)
    if len({id(token) for token in owner_tokens}) != len(owner_tokens):
        raise RuntimeError("formal QFG owner tokens are not distinct")


def _validate_forward_helper_bindings(model: EviSIRST) -> None:
    if frequency_encoder_forward is not _FROZEN_FREQUENCY_ENCODER_FORWARD:
        raise RuntimeError("frequency encoder helper identity differs")
    _validate_bound_method(
        model,
        name="explicit_embeddings",
        expected=_FROZEN_MODEL_EXPLICIT_EMBEDDINGS,
        label="model",
    )
    _validate_qfg_runtime_contract(model.tpd_qfg)

    relay = getattr(model, "tpd_ner", None)
    if type(relay) is not _FROZEN_NER_CLASS:
        raise RuntimeError("formal NER relay type differs")
    for name, expected in (
        ("forward_stage", _FROZEN_NER_FORWARD_STAGE),
        ("dc_support", _FROZEN_NER_DC_SUPPORT),
        ("_persistent_tail_support", _FROZEN_NER_PERSISTENT_TAIL_SUPPORT),
        ("_tail_support", _FROZEN_NER_TAIL_SUPPORT),
    ):
        _validate_bound_method(relay, name=name, expected=expected, label="tpd_ner")

    for name, stage in (
        ("up_decoder4", 4),
        ("up_decoder3", 3),
        ("up_decoder2", 2),
    ):
        decoder = getattr(model, name, None)
        if type(decoder) is not _FROZEN_DECODER_CLASS or decoder.stage != stage:
            raise RuntimeError(f"formal decoder contract differs at {name}")
        _validate_bound_method(
            decoder,
            name="prepare",
            expected=_FROZEN_DECODER_PREPARE,
            label=name,
        )
        _validate_bound_method(
            decoder,
            name="finish",
            expected=_FROZEN_DECODER_FINISH,
            label=name,
        )


def _validate_canonical_base_state_keys(
    keys: tuple[str, ...],
    *,
    label: str,
    _expected_sha256: str = (
        "fa5b57ea63c1ea2a7d9711fa935b87bcdc3501d6439539b26da476a4fe7def0d"
    ),
) -> str:
    observed_sha256 = _canonical_sha256(keys)
    if observed_sha256 != _expected_sha256:
        raise RuntimeError(f"{label} canonical state-key sequence differs")
    return observed_sha256


def _validate_mode_training_state(model: nn.Module) -> None:
    mode = getattr(model, "mode", None)
    if mode not in ("train", "test"):
        raise RuntimeError("formal model.mode must be exactly 'train' or 'test'")
    expected_training = mode == "train"
    if model.training is not expected_training:
        raise RuntimeError("formal model.mode and model.training differ")
    for name, module in model.named_modules():
        if module.training is not expected_training:
            raise RuntimeError(
                f"formal child training state differs at {name or '<root>'}"
            )


def _resolve_architecture_seed(
    *, seed: int | None, architecture_seed: int | None
) -> int:
    for name, value in (("seed", seed), ("architecture_seed", architecture_seed)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(f"{name} must be an integer or None")
    if seed is None and architecture_seed is None:
        return 42
    if seed is None:
        assert architecture_seed is not None
        return architecture_seed
    if architecture_seed is None:
        return seed
    if seed != architecture_seed:
        raise ValueError("seed and architecture_seed must agree when both are set")
    return seed


class DilatedCoarseSupportPrecisionGuard(nn.Module):
    """One-sided bounded suppression outside detached coarse ``d0`` support."""

    def __init__(
        self,
        *,
        support_kernel_size: int = SUPPORT_KERNEL_SIZE,
        max_logit_delta: float = MAX_LOGIT_DELTA,
    ) -> None:
        super().__init__()
        if type(support_kernel_size) is not int:
            raise TypeError("support_kernel_size must be an integer")
        if support_kernel_size != SUPPORT_KERNEL_SIZE:
            raise ValueError(
                f"support_kernel_size must be exactly {SUPPORT_KERNEL_SIZE}"
            )
        if isinstance(max_logit_delta, bool) or not isinstance(
            max_logit_delta, (int, float)
        ):
            raise TypeError("max_logit_delta must be a real number")
        normalized = float(max_logit_delta)
        if not math.isfinite(normalized) or normalized != MAX_LOGIT_DELTA:
            raise ValueError(f"max_logit_delta must be exactly {MAX_LOGIT_DELTA}")
        self.support_kernel_size = support_kernel_size
        self.max_logit_delta = normalized
        self.raw_strength = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def strength(self) -> torch.Tensor:
        # At raw_strength==0 this is exactly zero in the parameter dtype, but
        # torch.clamp retains the softplus derivative at its lower boundary.
        zero_centered = F.softplus(self.raw_strength) - math.log(2.0)
        unit_strength = torch.clamp(zero_centered, min=0.0, max=1.0)
        return self.max_logit_delta * unit_strength

    @staticmethod
    def _validate_logits(out_logit: torch.Tensor, d0_logit: torch.Tensor) -> None:
        if not isinstance(out_logit, torch.Tensor) or not isinstance(
            d0_logit, torch.Tensor
        ):
            raise TypeError("out_logit and d0_logit must be tensors")
        if out_logit.ndim != 4 or d0_logit.ndim != 4:
            raise ValueError("out_logit and d0_logit must be BCHW tensors")
        if out_logit.shape != d0_logit.shape:
            raise ValueError("out_logit and d0_logit shapes differ")
        if out_logit.shape[1] != 1:
            raise ValueError("out_logit and d0_logit must have one channel")
        if not out_logit.is_floating_point() or not d0_logit.is_floating_point():
            raise TypeError("out_logit and d0_logit must be floating point")
        if out_logit.device != d0_logit.device:
            raise ValueError("out_logit and d0_logit devices differ")
        if out_logit.dtype != d0_logit.dtype:
            raise ValueError("out_logit and d0_logit dtypes differ")
        if not bool(torch.isfinite(out_logit).all()) or not bool(
            torch.isfinite(d0_logit).all()
        ):
            raise ValueError("out_logit and d0_logit must be finite")

    def refinement_components(
        self, out_logit: torch.Tensor, d0_logit: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return auditable stopped probabilities and bounded correction."""

        self._validate_logits(out_logit, d0_logit)
        out_probability = torch.sigmoid(out_logit.detach())
        d0_probability = torch.sigmoid(d0_logit.detach())
        support_padding = self.support_kernel_size // 2
        dilated_d0_support = F.max_pool2d(
            d0_probability,
            kernel_size=self.support_kernel_size,
            stride=1,
            padding=support_padding,
        )
        coarse_unsupported_evidence = 1.0 - dilated_d0_support
        unsupported_out = out_probability * coarse_unsupported_evidence
        correction = -self.strength.to(
            device=out_logit.device, dtype=out_logit.dtype
        ) * unsupported_out
        return {
            "stopped_out_probability": out_probability,
            "stopped_d0_probability": d0_probability,
            "dilated_d0_support": dilated_d0_support,
            "coarse_unsupported_evidence": coarse_unsupported_evidence,
            "unsupported_out": unsupported_out,
            "correction": correction,
        }

    def forward(
        self, out_logit: torch.Tensor, d0_logit: torch.Tensor
    ) -> torch.Tensor:
        correction = self.refinement_components(out_logit, d0_logit)[
            "correction"
        ]
        return out_logit + correction

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "schema": "evisirst_dcspg_module/v1",
            "module": type(self).__name__,
            "inputs": ["raw_out_logit", "raw_d0_logit"],
            "out_evidence_stop_gradient": True,
            "d0_evidence_stop_gradient": True,
            "support": "max_pool2d(sigmoid(d0),kernel9,stride1,padding4)",
            "support_kernel_size": self.support_kernel_size,
            "coarse_unsupported_evidence": "one_minus_dilated_d0_support",
            "unsupported_out": (
                "sigmoid(out)*coarse_unsupported_evidence"
            ),
            "ground_truth_far_background_claimed": False,
            "strength": (
                "max_logit_delta*clamp(softplus(raw_strength)-log(2),0,1)"
            ),
            "max_logit_delta": self.max_logit_delta,
            "identity_initialization": "raw_strength_exact_zero",
            "negative_raw_strength_policy": "exact_zero_safe_fallback",
            "correction_sign": "non_positive_only",
            "guarantee": "elementwise_logit_correction_in_closed_interval[-3,0]",
            "training_head_5": "raw_d0_probability",
            "training_head_6": "guarded_out_probability",
            "evaluation_head": "guarded_out_probability",
            "inference_requires_raw_d0": True,
            "inference_additional_compute": (
                "gt_conv2_to_gt_conv5_interpolation_outconv_and_maxpool9"
            ),
        }


_FROZEN_GUARD_FORWARD = DilatedCoarseSupportPrecisionGuard.forward
_FROZEN_GUARD_COMPONENTS = (
    DilatedCoarseSupportPrecisionGuard.refinement_components
)
_FROZEN_GUARD_MANIFEST = (
    DilatedCoarseSupportPrecisionGuard.architecture_manifest
)
_FROZEN_GUARD_STRENGTH_GETTER = (
    DilatedCoarseSupportPrecisionGuard.strength.fget
)


def cp_hf_s2_dcspg_forward_with_relay(self: EviSIRST, x: torch.Tensor):
    """Frozen relay flow, CP-HF-S2 at d2, then DCS-PG at final logit."""

    x1 = self.inc(x)
    x2 = self.down_encoder1(self.pool(x1))
    x3 = self.down_encoder2(self.pool(x2))
    x4 = self.down_encoder3(self.pool(x3))
    d5 = self.down_encoder4(self.pool(x4))
    f1, f2, f3, f4 = x1, x2, x3, x4

    emb1, emb2, emb3, emb4, evidence1, evidence2 = self.explicit_embeddings(
        x1, x2, x3, x4
    )
    h11, h12, h13 = evidence1
    h21, h22 = evidence2
    prepared_qfg = self.tpd_qfg.prepare(
        (x1, x2, x3, x4),
        tuple(
            tuple(embedding.shape[-2:])
            for embedding in (emb1, emb2, emb3, emb4)
        ),
    )
    encoded1, encoded2, encoded3, encoded4, _ = frequency_encoder_forward(
        self.mtc.encoder,
        emb1,
        emb2,
        emb3,
        emb4,
        self.tpd_qfg,
        prepared_qfg,
    )
    x1 = self.mtc.reconstruct_1(encoded1) + f1
    x2 = self.mtc.reconstruct_2(encoded2) + f2
    x3 = self.mtc.reconstruct_3(encoded3) + f3
    x4 = self.mtc.reconstruct_4(encoded4) + f4
    x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4

    up4, skip4 = self.up_decoder4.prepare(d5, x4)
    q4, mask4 = self.tpd_ner.forward_stage(
        4, (h13, h22, up4), tuple(up4.shape[-2:])
    )
    d4 = self.up_decoder4.finish(up4, skip4, mask4)
    up3, skip3 = self.up_decoder3.prepare(d4, x3)
    q3, mask3 = self.tpd_ner.forward_stage(
        3, (h12, h21, q4, up3), tuple(up3.shape[-2:])
    )
    d3 = self.up_decoder3.finish(up3, skip3, mask3)
    up2, skip2 = self.up_decoder2.prepare(d3, x2)
    _, mask2 = self.tpd_ner.forward_stage(
        2, (h11, q3, up2), tuple(up2.shape[-2:])
    )
    d2 = self.up_decoder2.finish(up2, skip2, mask2)
    refinement = getattr(self, CP_MODULE_NAME, None)
    if not isinstance(refinement, cp_hf.ContextPurifiedHighFrequencyS2):
        raise RuntimeError("formal CP-HF-S2 module is absent or replaced")
    d2 = refinement(d2, x2)

    out = self.outc(self.up_decoder1(d2, x1))
    gt_5 = self.gt_conv5(d5)
    gt_4 = self.gt_conv4(d4)
    gt_3 = self.gt_conv3(d3)
    gt_2 = self.gt_conv2(d2)
    gt5 = F.interpolate(gt_5, scale_factor=16, mode="bilinear", align_corners=True)
    gt4 = F.interpolate(gt_4, scale_factor=8, mode="bilinear", align_corners=True)
    gt3 = F.interpolate(gt_3, scale_factor=4, mode="bilinear", align_corners=True)
    gt2 = F.interpolate(gt_2, scale_factor=2, mode="bilinear", align_corners=True)
    d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))

    guard = getattr(self, GUARD_MODULE_NAME, None)
    if not isinstance(guard, DilatedCoarseSupportPrecisionGuard):
        raise RuntimeError("formal DCS-PG module is absent or replaced")
    guarded_out = guard(out, d0)
    if self.mode != "train":
        return torch.sigmoid(guarded_out)
    return (
        torch.sigmoid(gt5),
        torch.sigmoid(gt4),
        torch.sigmoid(gt3),
        torch.sigmoid(gt2),
        torch.sigmoid(d0),
        torch.sigmoid(guarded_out),
    )


def _require_clean_base(base: nn.Module) -> tuple[tuple[str, ...], dict[str, Any]]:
    if type(base) is not EviSIRST:
        raise TypeError("CP-HF-S2-DCSPG requires the exact clean EviSIRST class")
    if getattr(base, "mode", None) != "train" or not base.training:
        raise RuntimeError("install CP-HF-S2-DCSPG on a training-mode base")
    if getattr(base, "deepsuper", None) is not True:
        raise RuntimeError("CP-HF-S2-DCSPG requires deep supervision")
    if hasattr(base, CP_MODULE_NAME) or hasattr(base, GUARD_MODULE_NAME):
        raise RuntimeError("CP-HF-S2-DCSPG requires a clean unextended base")
    for marker in (
        _BASE_STATE_KEYS_ATTRIBUTE,
        _BASE_MANIFEST_ATTRIBUTE,
        _ARCHITECTURE_SEED_ATTRIBUTE,
        _INITIALIZATION_SEED_ATTRIBUTE,
    ):
        if hasattr(base, marker):
            raise RuntimeError("CP-HF-S2-DCSPG installation marker already exists")
    state_keys = tuple(base.state_dict())
    if len(state_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError("clean base state-key count differs")
    if _parameter_count(base) != BASE_PARAMETER_COUNT:
        raise RuntimeError("clean base parameter count differs")
    if any(not parameter.requires_grad for parameter in base.parameters()):
        raise RuntimeError("all clean base parameters must be trainable")
    return state_keys, copy.deepcopy(base.architecture_manifest())


def install_irstd_cp_hf_s2_dcspg_v1(
    base: nn.Module, *, architecture_seed: int = 42
) -> tuple[nn.Module, dict[str, Any]]:
    """Install CP-HF-S2 and DCS-PG without loading any checkpoint."""

    if isinstance(architecture_seed, bool) or not isinstance(
        architecture_seed, int
    ):
        raise TypeError("architecture_seed must be an integer")
    base_state_keys, base_manifest = _require_clean_base(base)
    model, cp_metadata = cp_hf.install_irstd_cp_hf_s2_v1(
        base, architecture_seed=architecture_seed
    )
    guard = DilatedCoarseSupportPrecisionGuard()
    guard.to(device=model.outc.weight.device, dtype=model.outc.weight.dtype)
    guard.train(model.training)
    model.add_module(GUARD_MODULE_NAME, guard)
    model._forward_with_relay = MethodType(
        cp_hf_s2_dcspg_forward_with_relay, model
    )
    initialization_seed = cp_hf.derive_cp_hf_s2_initialization_seed(
        architecture_seed
    )
    setattr(model, _BASE_STATE_KEYS_ATTRIBUTE, base_state_keys)
    setattr(model, _BASE_MANIFEST_ATTRIBUTE, base_manifest)
    setattr(model, _ARCHITECTURE_SEED_ATTRIBUTE, architecture_seed)
    setattr(model, _INITIALIZATION_SEED_ATTRIBUTE, initialization_seed)
    manifest = validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )
    metadata = {
        "schema": EXPERIMENT_SCHEMA,
        "architecture_schema": ARCHITECTURE_SCHEMA,
        "name": EXPERIMENT_NAME,
        "architecture_name": ARCHITECTURE_NAME,
        "experiment": EXPERIMENT_NAME,
        "dataset_scope": SUPPORTED_DATASET,
        "architecture_seed": architecture_seed,
        "seed": architecture_seed,
        "initialization_seed": initialization_seed,
        "initialization_mode": "full_model_scratch_isolated_cp_extension",
        "baseline_checkpoint_loaded": False,
        "parent_checkpoint": None,
        "warm_start_used": False,
        "full_model_scratch_training": True,
        "optimizer_parameter_scope": "all_model_parameters",
        "cp_hf_s2_install_metadata": cp_metadata,
        "architecture_manifest": manifest,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
        "source_dependency_paths": list(manifest["source_dependency_paths"]),
        "source_dependencies": dict(manifest["source_dependencies"]),
    }
    return model, metadata


def build_irstd_cp_hf_s2_dcspg_v1(
    dataset: str = SUPPORTED_DATASET,
    *,
    seed: int | None = None,
    architecture_seed: int | None = None,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a checkpoint-free full-model-scratch CP-HF-S2-DCSPG graph."""

    if dataset != SUPPORTED_DATASET:
        raise ValueError(
            f"CP-HF-S2-DCSPG supports only {SUPPORTED_DATASET!r}"
        )
    if type(training) is not bool:
        raise TypeError("training must be bool")
    resolved_seed = _resolve_architecture_seed(
        seed=seed, architecture_seed=architecture_seed
    )
    base, base_metadata = initialize_evisirst(
        dataset, seed=resolved_seed, training=True
    )
    model, metadata = install_irstd_cp_hf_s2_dcspg_v1(
        base, architecture_seed=resolved_seed
    )
    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()
    metadata = dict(metadata)
    metadata.update(
        {
            "training_mode": training,
            "base_model_metadata": dict(base_metadata),
            "all_parameters_trainable": all(
                parameter.requires_grad for parameter in model.parameters()
            ),
        }
    )
    return model, metadata


def _validate_cp_extension(
    extension: cp_hf.ContextPurifiedHighFrequencyS2,
    *,
    architecture_seed: int,
    initialization_seed: int,
) -> None:
    cp_hf._validate_module_structure(extension)
    if (
        extension.architecture_seed != architecture_seed
        or extension.initialization_seed != initialization_seed
    ):
        raise RuntimeError("CP-HF-S2 module seed attributes differ")
    expected_manifest = {
        "schema": "evisirst_cp_hf_s2_module/v1",
        "module": "ContextPurifiedHighFrequencyS2",
        "architecture_seed": architecture_seed,
        "initialization_seed": initialization_seed,
        "initialization_seed_derivation": (
            "uint63_be_sha256_prefix8_"
            "EviSIRST/CP-HF-S2-v1/extension/{architecture_seed}"
        ),
        "feature_channels": cp_hf.FEATURE_CHANNELS,
        "skip_channels": cp_hf.SKIP_CHANNELS,
        "hidden_channels": cp_hf.HIDDEN_CHANNELS,
        "feature_evidence_stop_gradient": True,
        "skip_evidence_stop_gradient": True,
        "skip_projection": "conv1x1_with_bias",
        "skip_spatial_alignment": "strict_equal_no_interpolation",
        "low_pass": "fixed_binomial5_reflect_padding",
        "low_pass_kernel_size": cp_hf.LOW_PASS_KERNEL_SIZE,
        "channel_gate": "gap_conv1x1_silu_conv1x1_sigmoid",
        "spatial_gate": "three_mean_abs_maps_conv3x3_sigmoid",
        "residual": "depthwise3x3_silu_pointwise1x1_tanh",
        "scale": "max_feature_delta_times_tanh_raw_scale",
        "max_feature_delta": cp_hf.MAX_FEATURE_DELTA,
        "identity_initialization": "raw_scale_exact_zero",
        "additional_zero_terminal": False,
        "guarantee": "elementwise_absolute_feature_correction_le_0.25",
        "final_logit_bound_claimed": False,
        "purification_guaranteed": False,
    }
    if extension.architecture_manifest() != expected_manifest:
        raise RuntimeError("CP-HF-S2 component manifest differs")


def _validate_live_base_formal_contract(
    model: EviSIRST,
    *,
    base_keys: tuple[str, ...],
    require_identity_initialization: bool,
    _authoritative_base_manifest_sha256: str = (
        "01836fa26f47f63c4a4362e22a3f66955a61a3fc2adba9fd76d63c6758b15b00"
    ),
) -> dict[str, Any]:
    """Apply the live equivalent of the frozen 564-key base validator."""

    if hasattr(model, "target_survival"):
        raise RuntimeError("formal base unexpectedly retains Survival heads")
    if (
        model.deepsuper is not True
        or model.relay_enabled is not True
        or getattr(model, "_nested_relay_installed", None) is not True
    ):
        raise RuntimeError("formal base relay/deep-supervision contract differs")
    if model.tokenizer_variant != formal_base.FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError("formal base tokenizer variant differs")
    if model.relay_width != formal_base.DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal base relay width differs")
    if (
        model.relay_initialization_seed
        != formal_base.DEFAULT_RELAY_INITIALIZATION_SEED
    ):
        raise RuntimeError("formal base relay initialization seed differs")
    if model.tpd_ner.dc_support_mode != formal_base.DEFAULT_DC_SUPPORT_MODE:
        raise RuntimeError("formal base requires complement-tail support")
    if dict(model.tpd_ner.tail_z_thresholds) != dict(
        formal_base.DEFAULT_TAIL_Z_THRESHOLDS
    ):
        raise RuntimeError("formal base tail thresholds differ")
    context_gate = formal_base._formal_context_gate(model)
    qfg_core_manifest = _FROZEN_QFG_VALIDATOR(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialization,
    )

    state = model.state_dict()
    observed_base_keys = tuple(
        key
        for key in state
        if not key.startswith((CP_STATE_PREFIX, GUARD_STATE_PREFIX))
    )
    observed_base_state_keys_sha256 = _validate_canonical_base_state_keys(
        observed_base_keys,
        label="live formal base",
    )
    if observed_base_keys != base_keys or len(observed_base_keys) != 564:
        raise RuntimeError("live formal base state-key contract differs")
    qfg_keys = tuple(
        key for key in observed_base_keys if key.startswith(formal_base.QFG_STATE_PREFIX)
    )
    if set(qfg_keys) != set(formal_base.QFG_STATE_KEYS):
        raise RuntimeError("live formal base QFG state keys differ")
    if any(
        key.startswith(formal_base.SURVIVAL_STATE_PREFIX)
        for key in observed_base_keys
    ):
        raise RuntimeError("live formal base retains Survival state")
    base_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith((CP_STATE_PREFIX, GUARD_STATE_PREFIX))
    )
    if (
        base_parameter_count
        != formal_base.PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
    ):
        raise RuntimeError("live formal base parameter count differs")
    if _parameter_count(model.tpd_qfg) != formal_base.PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise RuntimeError("live formal base QFG parameter count differs")

    live_manifest = model.architecture_manifest()
    live_manifest_sha256 = _canonical_sha256(live_manifest)
    if live_manifest_sha256 != _authoritative_base_manifest_sha256:
        raise RuntimeError("live base manifest differs from authoritative build")
    return {
        "schema": "evisirst_dcspg_live_base_formal_validation/v1",
        "equivalent_validator": (
            "validate_formal_qfg_v2_croa_inference_model_"
            "with_extension_state_excluded"
        ),
        "state_key_count": len(observed_base_keys),
        "parameter_count": base_parameter_count,
        "qfg_state_key_count": len(qfg_keys),
        "qfg_parameter_count": _parameter_count(model.tpd_qfg),
        "canonical_base_state_keys_sha256": (
            observed_base_state_keys_sha256
        ),
        "context_gate": context_gate,
        "qfg_core_manifest": qfg_core_manifest,
        "authoritative_manifest_sha256": _authoritative_base_manifest_sha256,
        "live_manifest_sha256": live_manifest_sha256,
    }


def validate_irstd_cp_hf_s2_dcspg_v1(
    model: nn.Module, *, require_identity_initialization: bool = False
) -> dict[str, Any]:
    """Strictly validate state, binding, structure, seeds, and trainability."""

    if type(require_identity_initialization) is not bool:
        raise TypeError("require_identity_initialization must be bool")
    if type(model) is not EviSIRST:
        raise TypeError("formal CP-HF-S2-DCSPG must retain exact EviSIRST class")
    source_dependencies = _validate_source_dependencies()
    _validate_mode_training_state(model)
    _validate_runtime_graph_seal(model)
    _validate_bound_method(
        model,
        name="forward",
        expected=_FROZEN_MODEL_FORWARD,
        label="model",
    )
    _validate_bound_method(
        model,
        name="architecture_manifest",
        expected=_FROZEN_MODEL_MANIFEST,
        label="model",
    )
    if _FROZEN_CONV2D_CLASS.forward is not _FROZEN_CONV2D_FORWARD:
        raise RuntimeError("Conv2d class forward identity differs")
    if _FROZEN_SEQUENTIAL_CLASS.forward is not _FROZEN_SEQUENTIAL_FORWARD:
        raise RuntimeError("Sequential class forward identity differs")
    if _FROZEN_GELU_CLASS.forward is not _FROZEN_GELU_FORWARD:
        raise RuntimeError("GELU class forward identity differs")
    if _FROZEN_HAAR_CLASS.forward is not _FROZEN_HAAR_FORWARD:
        raise RuntimeError("FixedHaarAnalysis class forward identity differs")
    _validate_forward_helper_bindings(model)
    if getattr(model, "deepsuper", None) is not True:
        raise RuntimeError("formal DCS-PG requires deep supervision heads")
    expected_auxiliary_channels = {
        "gt_conv2": 32,
        "gt_conv3": 64,
        "gt_conv4": 128,
        "gt_conv5": 256,
    }
    for name, input_channels in expected_auxiliary_channels.items():
        head = getattr(model, name, None)
        if (
            type(head) is not nn.Sequential
            or len(head) != 1
            or type(head[0]) is not nn.Conv2d
            or head[0].in_channels != input_channels
            or head[0].out_channels != 1
            or head[0].kernel_size != (1, 1)
        ):
            raise RuntimeError(f"raw d0 auxiliary head contract differs: {name}")
    outconv = getattr(model, "outconv", None)
    if (
        type(outconv) is not nn.Conv2d
        or outconv.in_channels != 5
        or outconv.out_channels != 1
        or outconv.kernel_size != (1, 1)
    ):
        raise RuntimeError("raw d0 fusion head contract differs")
    bound = getattr(model, "_forward_with_relay", None)
    if (
        getattr(bound, "__self__", None) is not model
        or getattr(bound, "__func__", None)
        is not cp_hf_s2_dcspg_forward_with_relay
    ):
        raise RuntimeError("CP-HF-S2-DCSPG relay binding differs")

    architecture_seed = getattr(model, _ARCHITECTURE_SEED_ATTRIBUTE, None)
    initialization_seed = getattr(model, _INITIALIZATION_SEED_ATTRIBUTE, None)
    if isinstance(architecture_seed, bool) or not isinstance(
        architecture_seed, int
    ):
        raise RuntimeError("architecture seed marker differs")
    expected_initialization_seed = cp_hf.derive_cp_hf_s2_initialization_seed(
        architecture_seed
    )
    if initialization_seed != expected_initialization_seed:
        raise RuntimeError("initialization seed marker differs")
    # The reused frozen installer owns parallel markers; reject divergence.
    if (
        getattr(model, "_irstd_cp_hf_s2_architecture_seed", None)
        != architecture_seed
        or getattr(model, "_irstd_cp_hf_s2_initialization_seed", None)
        != initialization_seed
    ):
        raise RuntimeError("reused CP-HF-S2 seed binding differs")

    extension = getattr(model, CP_MODULE_NAME, None)
    if type(extension) is not cp_hf.ContextPurifiedHighFrequencyS2:
        raise RuntimeError("formal CP-HF-S2 component is absent or replaced")
    _validate_bound_method(
        extension,
        name="forward",
        expected=_FROZEN_CP_FORWARD,
        label=CP_MODULE_NAME,
    )
    _validate_bound_method(
        extension,
        name="refinement_components",
        expected=_FROZEN_CP_COMPONENTS,
        label=CP_MODULE_NAME,
    )
    _validate_bound_method(
        extension,
        name="architecture_manifest",
        expected=_FROZEN_CP_MANIFEST,
        label=CP_MODULE_NAME,
    )
    _validate_cp_extension(
        extension,
        architecture_seed=architecture_seed,
        initialization_seed=initialization_seed,
    )
    guard = getattr(model, GUARD_MODULE_NAME, None)
    if type(guard) is not DilatedCoarseSupportPrecisionGuard:
        raise RuntimeError("formal DCS-PG component is absent or replaced")
    _validate_bound_method(
        guard,
        name="forward",
        expected=_FROZEN_GUARD_FORWARD,
        label=GUARD_MODULE_NAME,
    )
    _validate_bound_method(
        guard,
        name="refinement_components",
        expected=_FROZEN_GUARD_COMPONENTS,
        label=GUARD_MODULE_NAME,
    )
    _validate_bound_method(
        guard,
        name="architecture_manifest",
        expected=_FROZEN_GUARD_MANIFEST,
        label=GUARD_MODULE_NAME,
    )
    strength_property = getattr(type(guard), "strength", None)
    if (
        not isinstance(strength_property, property)
        or strength_property.fget is not _FROZEN_GUARD_STRENGTH_GETTER
    ):
        raise RuntimeError("DCS-PG strength property identity differs")
    if (
        guard.support_kernel_size != SUPPORT_KERNEL_SIZE
        or guard.max_logit_delta != MAX_LOGIT_DELTA
    ):
        raise RuntimeError("DCS-PG fixed configuration differs")
    if guard.architecture_manifest() != DilatedCoarseSupportPrecisionGuard(
        max_logit_delta=MAX_LOGIT_DELTA
    ).architecture_manifest():
        raise RuntimeError("DCS-PG component manifest differs")

    state = model.state_dict()
    base_keys = getattr(model, _BASE_STATE_KEYS_ATTRIBUTE, None)
    if not isinstance(base_keys, tuple) or len(base_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError("base state-key snapshot is absent or malformed")
    base_state_keys_sha256 = _validate_canonical_base_state_keys(
        base_keys,
        label="stored base snapshot",
    )
    observed_base_keys = tuple(
        key
        for key in state
        if not key.startswith((CP_STATE_PREFIX, GUARD_STATE_PREFIX))
    )
    observed_base_state_keys_sha256 = _validate_canonical_base_state_keys(
        observed_base_keys,
        label="live base",
    )
    if observed_base_keys != base_keys:
        raise RuntimeError("live base state keys differ from stored snapshot")
    expected_cp_shapes = cp_hf._expected_extension_shapes()
    expected_state_keys = (
        base_keys + tuple(expected_cp_shapes) + GUARD_STATE_KEYS
    )
    if tuple(state) != expected_state_keys:
        raise RuntimeError("formal CP-HF-S2-DCSPG state names or order differ")
    for key, shape in expected_cp_shapes.items():
        if tuple(state[key].shape) != shape:
            raise RuntimeError(f"CP-HF-S2 state shape differs for {key}")
    if tuple(state[GUARD_STATE_KEYS[0]].shape) != ():
        raise RuntimeError("DCS-PG raw_strength state shape differs")
    for key, value in state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"formal state is non-finite for {key}")
    expected_kernel = cp_hf._binomial_kernel_5().to(
        device=extension.low_pass_kernel.device,
        dtype=extension.low_pass_kernel.dtype,
    )
    if not torch.equal(extension.low_pass_kernel, expected_kernel):
        raise RuntimeError("CP-HF-S2 fixed binomial kernel differs")
    if len(state) != FORMAL_STATE_KEY_COUNT:
        raise RuntimeError("formal state-key count differs")
    if _parameter_count(extension) != CP_PARAMETER_COUNT:
        raise RuntimeError("CP-HF-S2 component parameter count differs")
    if _parameter_count(guard) != GUARD_PARAMETER_COUNT:
        raise RuntimeError("DCS-PG parameter count differs")
    if _parameter_count(model) != FORMAL_PARAMETER_COUNT:
        raise RuntimeError("formal total parameter count differs")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("formal graph contains frozen parameters")
    base_formal_validation = _validate_live_base_formal_contract(
        model,
        base_keys=base_keys,
        require_identity_initialization=require_identity_initialization,
    )
    if require_identity_initialization:
        if torch.count_nonzero(extension.raw_scale.detach()).item() != 0:
            raise RuntimeError("CP-HF-S2 raw_scale is not identity initialized")
        if torch.count_nonzero(guard.raw_strength.detach()).item() != 0:
            raise RuntimeError("DCS-PG raw_strength is not identity initialized")
        if torch.count_nonzero(guard.strength.detach()).item() != 0:
            raise RuntimeError("DCS-PG strength is not exact-zero initialized")

    base_manifest = getattr(model, _BASE_MANIFEST_ATTRIBUTE, None)
    if not isinstance(base_manifest, Mapping):
        raise RuntimeError("base architecture manifest snapshot is absent")
    if model.architecture_manifest() != dict(base_manifest):
        raise RuntimeError("base architecture manifest changed after installation")
    reused_base_keys = getattr(model, "_irstd_cp_hf_s2_base_state_keys", None)
    reused_base_manifest = getattr(model, "_irstd_cp_hf_s2_base_manifest", None)
    if reused_base_keys != base_keys or reused_base_manifest != base_manifest:
        raise RuntimeError("reused CP-HF-S2 base snapshot differs")

    return {
        "schema": ARCHITECTURE_SCHEMA,
        "name": EXPERIMENT_NAME,
        "architecture_name": ARCHITECTURE_NAME,
        "experiment": EXPERIMENT_NAME,
        "base_model": "EviSIRST",
        "dataset_scope": SUPPORTED_DATASET,
        "architecture_seed": architecture_seed,
        "seed": architecture_seed,
        "initialization_seed": initialization_seed,
        "binding_target": "_forward_with_relay",
        "bound_function": "cp_hf_s2_dcspg_forward_with_relay",
        "feature_insertion": "CP-HF-S2_after_up_decoder2_finish",
        "precision_guard_insertion": "after_raw_d0_and_raw_out_before_sigmoid",
        "training_head_5": "raw_d0_probability",
        "training_head_6": "guarded_out_probability",
        "evaluation_head": "guarded_out_probability",
        "raw_d0_required_modules": [
            "gt_conv2",
            "gt_conv3",
            "gt_conv4",
            "gt_conv5",
            "outconv",
        ],
        "inference_requires_raw_d0": True,
        "inference_early_out_bypass_allowed": False,
        "inference_additional_compute": (
            "gt_conv2_to_gt_conv5_interpolation_outconv_and_maxpool9"
        ),
        "base_state_key_count": BASE_STATE_KEY_COUNT,
        "canonical_base_state_keys_sha256": base_state_keys_sha256,
        "live_base_state_keys_sha256": observed_base_state_keys_sha256,
        "cp_state_key_count": CP_STATE_KEY_COUNT,
        "guard_state_key_count": GUARD_STATE_KEY_COUNT,
        "extension_state_key_count": EXTENSION_STATE_KEY_COUNT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "base_parameter_count": BASE_PARAMETER_COUNT,
        "cp_parameter_count": CP_PARAMETER_COUNT,
        "guard_parameter_count": GUARD_PARAMETER_COUNT,
        "extension_parameter_count": EXTENSION_PARAMETER_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
        "cp_component_manifest": extension.architecture_manifest(),
        "guard_component_manifest": guard.architecture_manifest(),
        "base_formal_validation": base_formal_validation,
        "compatible_diagnostic_components": [
            "stopped_out_probability",
            "stopped_d0_probability",
            "dilated_d0_support",
            "coarse_unsupported_evidence",
            "unsupported_out",
            "correction",
        ],
        "identity_relative_to_clean_at_initialization": True,
        "baseline_checkpoint_loaded": False,
        "warm_start_used": False,
        "loss_change": False,
        "data_change": False,
        "checkpoint_selector_change": False,
        "evaluation_data_role_policy": "external_runner_owned",
        "source_dependency_paths": list(source_dependencies),
        "source_dependencies": source_dependencies,
    }


__all__ = [
    "ARCHITECTURE_NAME",
    "ARCHITECTURE_SCHEMA",
    "AUTHORITATIVE_BASE_MANIFEST_SHA256",
    "BASE_PARAMETER_COUNT",
    "BASE_STATE_KEY_COUNT",
    "CANONICAL_BASE_STATE_KEYS_SHA256",
    "CP_MODULE_NAME",
    "CP_PARAMETER_COUNT",
    "CP_STATE_KEY_COUNT",
    "EXPERIMENT_NAME",
    "EXPERIMENT_SCHEMA",
    "EXTENSION_PARAMETER_COUNT",
    "EXTENSION_STATE_KEY_COUNT",
    "FORMAL_PARAMETER_COUNT",
    "FORMAL_STATE_KEY_COUNT",
    "GUARD_MODULE_NAME",
    "GUARD_PARAMETER_COUNT",
    "GUARD_STATE_KEY_COUNT",
    "GUARD_STATE_KEYS",
    "GUARD_STATE_PREFIX",
    "MAX_LOGIT_DELTA",
    "SUPPORT_KERNEL_SIZE",
    "DilatedCoarseSupportPrecisionGuard",
    "SOURCE_DEPENDENCIES",
    "SOURCE_DEPENDENCY_SHA256",
    "SUPPORTED_DATASET",
    "build_irstd_cp_hf_s2_dcspg_v1",
    "cp_hf_s2_dcspg_forward_with_relay",
    "install_irstd_cp_hf_s2_dcspg_v1",
    "validate_irstd_cp_hf_s2_dcspg_v1",
]
