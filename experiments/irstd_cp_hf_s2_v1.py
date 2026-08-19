"""IRSTD CP-HF-S2 V1: a bounded, read-only decoder refinement.

This experimental module leaves every frozen EviSIRST source untouched.  It
binds one instance-local copy of the audited relay forward and makes exactly
one graph change: immediately after ``up_decoder2.finish`` it replaces ``d2``
with

``d2 + a * C * Q * R``.

``C`` and ``Q`` are channel/spatial gates, ``R`` is a bounded high-frequency
residual, and ``a = 0.25 * tanh(raw_scale)``.  Both ``d2`` and the ``x2`` skip
are stop-gradient evidence for the extension.  Therefore the extension has
no correction Jacobian into the base graph, while the ordinary identity path
continues to train the complete base model.  ``raw_scale`` is the only
zero-initialized gate; it receives a gradient on the first optimization step.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.EviSIRST import EviSIRST, initialize_evisirst
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
    validate_formal_qfg_v2_croa_inference_model,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward


EXPERIMENT_SCHEMA = "evisirst_irstd_cp_hf_s2_v1"
ARCHITECTURE_SCHEMA = "evisirst_irstd_cp_hf_s2_architecture_v1"
EXPERIMENT_NAME = "IRSTD-CP-HF-S2-v1"
SUPPORTED_DATASET = "IRSTD-1K"

BASE_STATE_KEY_COUNT = 564
BASE_PARAMETER_COUNT = 10_870_130
MODULE_NAME = "decoder_cp_hf_s2"
STATE_PREFIX = f"{MODULE_NAME}."
FEATURE_CHANNELS = 32
SKIP_CHANNELS = 64
HIDDEN_CHANNELS = 8
LOW_PASS_KERNEL_SIZE = 5
MAX_FEATURE_DELTA = 0.25
EXTENSION_PARAMETER_COUNT = 4_485
EXTENSION_STATE_KEY_COUNT = 12
FORMAL_PARAMETER_COUNT = BASE_PARAMETER_COUNT + EXTENSION_PARAMETER_COUNT
FORMAL_STATE_KEY_COUNT = BASE_STATE_KEY_COUNT + EXTENSION_STATE_KEY_COUNT

_BASE_STATE_KEYS_ATTRIBUTE = "_irstd_cp_hf_s2_base_state_keys"
_BASE_MANIFEST_ATTRIBUTE = "_irstd_cp_hf_s2_base_manifest"
_ARCHITECTURE_SEED_ATTRIBUTE = "_irstd_cp_hf_s2_architecture_seed"
_INITIALIZATION_SEED_ATTRIBUTE = "_irstd_cp_hf_s2_initialization_seed"
_FROZEN_FORWARD_WITH_RELAY = (
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet._forward_with_relay
)


def derive_cp_hf_s2_initialization_seed(architecture_seed: int) -> int:
    """Derive the extension's isolated uint63 initialization substream."""

    if isinstance(architecture_seed, bool) or not isinstance(
        architecture_seed, int
    ):
        raise TypeError("architecture_seed must be an integer")
    payload = (
        "EviSIRST/CP-HF-S2-v1/extension/" + str(architecture_seed)
    ).encode("ascii")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big", signed=False
    ) % (1 << 63)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _binomial_kernel_5() -> torch.Tensor:
    vector = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    return (vector[:, None] * vector[None, :] / 256.0).reshape(1, 1, 5, 5)


class ContextPurifiedHighFrequencyS2(nn.Module):
    """Bounded S2 feature correction computed from read-only evidence."""

    def __init__(
        self,
        *,
        feature_channels: int = FEATURE_CHANNELS,
        skip_channels: int = SKIP_CHANNELS,
        hidden_channels: int = HIDDEN_CHANNELS,
        low_pass_kernel_size: int = LOW_PASS_KERNEL_SIZE,
        max_feature_delta: float = MAX_FEATURE_DELTA,
        architecture_seed: int = 42,
        initialization_seed: int | None = None,
    ) -> None:
        super().__init__()
        if feature_channels != FEATURE_CHANNELS:
            raise ValueError(f"feature_channels must be {FEATURE_CHANNELS}")
        if skip_channels != SKIP_CHANNELS:
            raise ValueError(f"skip_channels must be {SKIP_CHANNELS}")
        if hidden_channels != HIDDEN_CHANNELS:
            raise ValueError(f"hidden_channels must be {HIDDEN_CHANNELS}")
        if low_pass_kernel_size != LOW_PASS_KERNEL_SIZE:
            raise ValueError(
                f"low_pass_kernel_size must be {LOW_PASS_KERNEL_SIZE}"
            )
        if float(max_feature_delta) != MAX_FEATURE_DELTA:
            raise ValueError(f"max_feature_delta must be {MAX_FEATURE_DELTA}")
        derived = derive_cp_hf_s2_initialization_seed(architecture_seed)
        if initialization_seed is None:
            initialization_seed = derived
        if initialization_seed != derived:
            raise ValueError(
                "initialization_seed must be derived from architecture_seed"
            )

        self.feature_channels = feature_channels
        self.skip_channels = skip_channels
        self.hidden_channels = hidden_channels
        self.low_pass_kernel_size = low_pass_kernel_size
        self.max_feature_delta = float(max_feature_delta)
        self.architecture_seed = architecture_seed
        self.initialization_seed = initialization_seed

        # This is deliberately the only zero-initialized learnable gate.
        self.raw_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.register_buffer("low_pass_kernel", _binomial_kernel_5())
        self.skip_projection = nn.Conv2d(
            skip_channels, feature_channels, kernel_size=1, bias=True
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                3 * feature_channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                feature_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_branch = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                groups=feature_channels,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.max_feature_delta * torch.tanh(self.raw_scale)

    def _validate_inputs(
        self, feature: torch.Tensor, skip: torch.Tensor
    ) -> None:
        if not isinstance(feature, torch.Tensor) or not isinstance(
            skip, torch.Tensor
        ):
            raise TypeError("feature and skip must be tensors")
        if feature.ndim != 4 or skip.ndim != 4:
            raise ValueError("feature and skip must be BCHW tensors")
        if feature.shape[0] != skip.shape[0]:
            raise ValueError("feature and skip batch sizes differ")
        if feature.shape[1] != self.feature_channels:
            raise ValueError("feature channel contract differs")
        if skip.shape[1] != self.skip_channels:
            raise ValueError("skip channel contract differs")
        if not feature.is_floating_point() or not skip.is_floating_point():
            raise TypeError("feature and skip must be floating point")
        if feature.device != skip.device or feature.dtype != skip.dtype:
            raise ValueError("feature and skip device/dtype differ")
        if feature.shape[-2:] != skip.shape[-2:]:
            raise ValueError("feature and skip spatial shapes differ")
        if min(feature.shape[-2:]) <= self.low_pass_kernel_size // 2:
            raise ValueError("feature is too small for reflect low-pass padding")

    def refinement_components(
        self, feature: torch.Tensor, skip: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute named components without caching or mutating the base."""

        self._validate_inputs(feature, skip)
        stopped_feature = feature.detach()
        stopped_skip = skip.detach()
        projected = self.skip_projection(stopped_skip)
        padding = self.low_pass_kernel_size // 2
        padded = F.pad(
            projected,
            (padding, padding, padding, padding),
            mode="reflect",
        )
        kernel = self.low_pass_kernel.to(
            device=projected.device, dtype=projected.dtype
        ).expand(self.feature_channels, 1, -1, -1)
        low = F.conv2d(padded, kernel, groups=self.feature_channels)
        high = projected - low
        high_abs = high.abs()
        channel = self.channel_gate(
            torch.cat((stopped_feature, low, high_abs), dim=1)
        )
        spatial = self.spatial_gate(
            torch.cat(
                (
                    stopped_feature.abs().mean(dim=1, keepdim=True),
                    low.abs().mean(dim=1, keepdim=True),
                    high_abs.mean(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )
        residual = self.residual_branch(high)
        correction = (
            self.residual_scale.to(dtype=feature.dtype)
            * channel
            * spatial
            * residual
        )
        return {
            "projected_skip": projected,
            "low": low,
            "high": high,
            "channel_gate": channel,
            "spatial_gate": spatial,
            "bounded_residual": residual,
            "correction": correction,
        }

    def forward(
        self, feature: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        correction = self.refinement_components(feature, skip)["correction"]
        return feature + correction

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "schema": "evisirst_cp_hf_s2_module/v1",
            "module": type(self).__name__,
            "architecture_seed": self.architecture_seed,
            "initialization_seed": self.initialization_seed,
            "initialization_seed_derivation": (
                "uint63_be_sha256_prefix8_"
                "EviSIRST/CP-HF-S2-v1/extension/{architecture_seed}"
            ),
            "feature_channels": self.feature_channels,
            "skip_channels": self.skip_channels,
            "hidden_channels": self.hidden_channels,
            "feature_evidence_stop_gradient": True,
            "skip_evidence_stop_gradient": True,
            "skip_projection": "conv1x1_with_bias",
            "skip_spatial_alignment": "strict_equal_no_interpolation",
            "low_pass": "fixed_binomial5_reflect_padding",
            "low_pass_kernel_size": self.low_pass_kernel_size,
            "channel_gate": "gap_conv1x1_silu_conv1x1_sigmoid",
            "spatial_gate": "three_mean_abs_maps_conv3x3_sigmoid",
            "residual": "depthwise3x3_silu_pointwise1x1_tanh",
            "scale": "max_feature_delta_times_tanh_raw_scale",
            "max_feature_delta": self.max_feature_delta,
            "identity_initialization": "raw_scale_exact_zero",
            "additional_zero_terminal": False,
            "guarantee": "elementwise_absolute_feature_correction_le_0.25",
            "final_logit_bound_claimed": False,
            "purification_guaranteed": False,
        }


def _copied_forward_with_optional_cp_hf(
    self: EviSIRST,
    x: torch.Tensor,
    *,
    apply_cp_hf: bool,
):
    """Audited frozen relay flow plus one optional post-d2 operation."""

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
    if apply_cp_hf:
        refinement = getattr(self, MODULE_NAME, None)
        if not isinstance(refinement, ContextPurifiedHighFrequencyS2):
            raise RuntimeError("formal CP-HF-S2 module is absent or replaced")
        d2 = refinement(d2, x2)
    out = self.outc(self.up_decoder1(d2, x1))

    if not self.deepsuper:
        return torch.sigmoid(out)
    gt_5 = self.gt_conv5(d5)
    gt_4 = self.gt_conv4(d4)
    gt_3 = self.gt_conv3(d3)
    gt_2 = self.gt_conv2(d2)
    gt5 = F.interpolate(gt_5, scale_factor=16, mode="bilinear", align_corners=True)
    gt4 = F.interpolate(gt_4, scale_factor=8, mode="bilinear", align_corners=True)
    gt3 = F.interpolate(gt_3, scale_factor=4, mode="bilinear", align_corners=True)
    gt2 = F.interpolate(gt_2, scale_factor=2, mode="bilinear", align_corners=True)
    d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))
    if self.mode != "train":
        return torch.sigmoid(out)
    return (
        torch.sigmoid(gt5),
        torch.sigmoid(gt4),
        torch.sigmoid(gt3),
        torch.sigmoid(gt2),
        torch.sigmoid(d0),
        torch.sigmoid(out),
    )


def baseline_reference_forward_with_relay(self: EviSIRST, x: torch.Tensor):
    """Copied baseline mode used to prove parity with the frozen forward."""

    return _copied_forward_with_optional_cp_hf(
        self, x, apply_cp_hf=False
    )


def cp_hf_s2_forward_with_relay(self: EviSIRST, x: torch.Tensor):
    """Formal CP-HF-S2 relay path."""

    return _copied_forward_with_optional_cp_hf(self, x, apply_cp_hf=True)


def _require_clean_base(base: nn.Module) -> tuple[str, ...]:
    if type(base) is not EviSIRST:
        raise TypeError("CP-HF-S2 requires the exact clean EviSIRST class")
    if getattr(base, "mode", None) != "train" or not base.training:
        raise RuntimeError("install CP-HF-S2 on a training-mode base")
    if getattr(base, "deepsuper", None) is not True:
        raise RuntimeError("CP-HF-S2 requires deep supervision")
    if hasattr(base, "target_survival"):
        raise RuntimeError("CP-HF-S2 requires the TSS-free clean graph")
    if hasattr(base, MODULE_NAME):
        raise RuntimeError("CP-HF-S2 is already installed")
    if any(
        hasattr(base, name)
        for name in (
            _BASE_STATE_KEYS_ATTRIBUTE,
            _BASE_MANIFEST_ATTRIBUTE,
            _ARCHITECTURE_SEED_ATTRIBUTE,
            _INITIALIZATION_SEED_ATTRIBUTE,
        )
    ):
        raise RuntimeError("CP-HF-S2 installation markers already exist")
    bound = getattr(base, "_forward_with_relay", None)
    if getattr(bound, "__func__", None) is not _FROZEN_FORWARD_WITH_RELAY:
        raise RuntimeError("base relay forward is not the frozen implementation")
    if getattr(base.outc, "_forward_pre_hooks", None) or getattr(
        base.outc, "_forward_hooks", None
    ):
        raise RuntimeError("clean outc unexpectedly contains local hooks")
    state_keys = tuple(base.state_dict())
    if len(state_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError("clean base state-key count differs from 564")
    if _parameter_count(base) != BASE_PARAMETER_COUNT:
        raise RuntimeError("clean base parameter count differs")
    if any(not value.requires_grad for value in base.parameters()):
        raise RuntimeError("formal CP-HF-S2 requires all base parameters trainable")
    return state_keys


def _new_extension(
    reference: torch.Tensor, *, architecture_seed: int
) -> ContextPurifiedHighFrequencyS2:
    initialization_seed = derive_cp_hf_s2_initialization_seed(
        architecture_seed
    )
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(initialization_seed)
        module = ContextPurifiedHighFrequencyS2(
            architecture_seed=architecture_seed,
            initialization_seed=initialization_seed,
        )
    module.to(device=reference.device, dtype=reference.dtype)
    return module


def install_irstd_cp_hf_s2_v1(
    base: nn.Module, *, architecture_seed: int = 42
) -> tuple[nn.Module, dict[str, Any]]:
    """Install the formal extension without modifying public model sources."""

    base_state_keys = _require_clean_base(base)
    base_validation = validate_formal_qfg_v2_croa_inference_model(
        base, require_identity_initialized_qfg=True
    )
    base_manifest = copy.deepcopy(base.architecture_manifest())
    initialization_seed = derive_cp_hf_s2_initialization_seed(
        architecture_seed
    )
    extension = _new_extension(
        base.outc.weight, architecture_seed=architecture_seed
    )
    extension.train(base.training)
    base.add_module(MODULE_NAME, extension)
    base._forward_with_relay = MethodType(cp_hf_s2_forward_with_relay, base)
    setattr(base, _BASE_STATE_KEYS_ATTRIBUTE, base_state_keys)
    setattr(base, _BASE_MANIFEST_ATTRIBUTE, base_manifest)
    setattr(base, _ARCHITECTURE_SEED_ATTRIBUTE, architecture_seed)
    setattr(base, _INITIALIZATION_SEED_ATTRIBUTE, initialization_seed)
    manifest = validate_irstd_cp_hf_s2_v1(
        base, require_identity_initialization=True
    )
    return base, {
        "schema": EXPERIMENT_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "dataset_scope": SUPPORTED_DATASET,
        "initialization_mode": "full_model_scratch_isolated_extension",
        "parent_checkpoint": None,
        "warm_start_used": False,
        "optimizer_parameter_scope": "all_model_parameters",
        "architecture_seed": architecture_seed,
        "initialization_seed": initialization_seed,
        "base_formal_validation": base_validation,
        "architecture_manifest": manifest,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
    }


def build_irstd_cp_hf_s2_v1(
    dataset: str = SUPPORTED_DATASET,
    *,
    seed: int = 42,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a checkpoint-free, full-model-scratch CP-HF-S2 graph."""

    if dataset != SUPPORTED_DATASET:
        raise ValueError(f"CP-HF-S2 supports only {SUPPORTED_DATASET!r}")
    if type(training) is not bool:
        raise TypeError("training must be bool")
    base, base_metadata = initialize_evisirst(dataset, seed=seed, training=True)
    model, extension_metadata = install_irstd_cp_hf_s2_v1(
        base, architecture_seed=seed
    )
    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()
    metadata = dict(extension_metadata)
    metadata.update(
        {
            "training_mode": training,
            "base_model_metadata": dict(base_metadata),
            "full_model_scratch_training": True,
            "all_parameters_trainable": all(
                parameter.requires_grad for parameter in model.parameters()
            ),
        }
    )
    return model, metadata


def _expected_extension_shapes() -> dict[str, tuple[int, ...]]:
    return {
        f"{STATE_PREFIX}raw_scale": (),
        f"{STATE_PREFIX}low_pass_kernel": (1, 1, 5, 5),
        f"{STATE_PREFIX}skip_projection.weight": (32, 64, 1, 1),
        f"{STATE_PREFIX}skip_projection.bias": (32,),
        f"{STATE_PREFIX}channel_gate.1.weight": (8, 96, 1, 1),
        f"{STATE_PREFIX}channel_gate.1.bias": (8,),
        f"{STATE_PREFIX}channel_gate.3.weight": (32, 8, 1, 1),
        f"{STATE_PREFIX}channel_gate.3.bias": (32,),
        f"{STATE_PREFIX}spatial_gate.0.weight": (1, 3, 3, 3),
        f"{STATE_PREFIX}spatial_gate.0.bias": (1,),
        f"{STATE_PREFIX}residual_branch.0.weight": (32, 1, 3, 3),
        f"{STATE_PREFIX}residual_branch.2.weight": (32, 32, 1, 1),
    }


def _validate_module_structure(
    extension: ContextPurifiedHighFrequencyS2,
) -> None:
    if (
        extension.feature_channels != FEATURE_CHANNELS
        or extension.skip_channels != SKIP_CHANNELS
        or extension.hidden_channels != HIDDEN_CHANNELS
        or extension.low_pass_kernel_size != LOW_PASS_KERNEL_SIZE
        or extension.max_feature_delta != MAX_FEATURE_DELTA
    ):
        raise RuntimeError("CP-HF-S2 frozen scalar/channel contract differs")
    projection = extension.skip_projection
    if (
        type(projection) is not nn.Conv2d
        or projection.in_channels != SKIP_CHANNELS
        or projection.out_channels != FEATURE_CHANNELS
        or projection.kernel_size != (1, 1)
        or projection.stride != (1, 1)
        or projection.bias is None
    ):
        raise RuntimeError("CP-HF-S2 skip projection structure differs")
    channel = extension.channel_gate
    if (
        type(channel) is not nn.Sequential
        or len(channel) != 5
        or type(channel[0]) is not nn.AdaptiveAvgPool2d
        or channel[0].output_size != 1
        or type(channel[1]) is not nn.Conv2d
        or channel[1].in_channels != 3 * FEATURE_CHANNELS
        or channel[1].out_channels != HIDDEN_CHANNELS
        or channel[1].kernel_size != (1, 1)
        or channel[1].bias is None
        or type(channel[2]) is not nn.SiLU
        or type(channel[3]) is not nn.Conv2d
        or channel[3].in_channels != HIDDEN_CHANNELS
        or channel[3].out_channels != FEATURE_CHANNELS
        or channel[3].kernel_size != (1, 1)
        or channel[3].bias is None
        or type(channel[4]) is not nn.Sigmoid
    ):
        raise RuntimeError("CP-HF-S2 channel gate structure differs")
    spatial = extension.spatial_gate
    if (
        type(spatial) is not nn.Sequential
        or len(spatial) != 2
        or type(spatial[0]) is not nn.Conv2d
        or spatial[0].in_channels != 3
        or spatial[0].out_channels != 1
        or spatial[0].kernel_size != (3, 3)
        or spatial[0].padding != (1, 1)
        or spatial[0].bias is None
        or type(spatial[1]) is not nn.Sigmoid
    ):
        raise RuntimeError("CP-HF-S2 spatial gate structure differs")
    residual = extension.residual_branch
    if (
        type(residual) is not nn.Sequential
        or len(residual) != 4
        or type(residual[0]) is not nn.Conv2d
        or residual[0].in_channels != FEATURE_CHANNELS
        or residual[0].out_channels != FEATURE_CHANNELS
        or residual[0].kernel_size != (3, 3)
        or residual[0].padding != (1, 1)
        or residual[0].groups != FEATURE_CHANNELS
        or residual[0].bias is not None
        or type(residual[1]) is not nn.SiLU
        or type(residual[2]) is not nn.Conv2d
        or residual[2].in_channels != FEATURE_CHANNELS
        or residual[2].out_channels != FEATURE_CHANNELS
        or residual[2].kernel_size != (1, 1)
        or residual[2].bias is not None
        or type(residual[3]) is not nn.Tanh
    ):
        raise RuntimeError("CP-HF-S2 residual branch structure differs")


def validate_irstd_cp_hf_s2_v1(
    model: nn.Module, *, require_identity_initialization: bool = False
) -> dict[str, Any]:
    """Validate binding, state contract, finite values, and architecture."""

    if type(model) is not EviSIRST:
        raise TypeError("CP-HF-S2 model must retain the exact EviSIRST class")
    extension = getattr(model, MODULE_NAME, None)
    if not isinstance(extension, ContextPurifiedHighFrequencyS2):
        raise RuntimeError("formal CP-HF-S2 extension is absent or replaced")
    _validate_module_structure(extension)
    bound = getattr(model, "_forward_with_relay", None)
    if (
        getattr(bound, "__self__", None) is not model
        or getattr(bound, "__func__", None) is not cp_hf_s2_forward_with_relay
    ):
        raise RuntimeError("CP-HF-S2 relay binding differs")
    base_keys = getattr(model, _BASE_STATE_KEYS_ATTRIBUTE, None)
    if not isinstance(base_keys, tuple) or len(base_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError("base state-key snapshot is absent or malformed")
    state = model.state_dict()
    non_extension = tuple(
        key for key in state if not key.startswith(STATE_PREFIX)
    )
    if non_extension != base_keys:
        raise RuntimeError("CP-HF-S2 changed base state names or order")
    expected = _expected_extension_shapes()
    observed_extension = {
        key: value for key, value in state.items() if key.startswith(STATE_PREFIX)
    }
    if set(observed_extension) != set(expected):
        raise RuntimeError("CP-HF-S2 extension state-key set differs")
    for key, shape in expected.items():
        value = observed_extension[key]
        if tuple(value.shape) != shape:
            raise RuntimeError(f"CP-HF-S2 state shape differs for {key}")
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"CP-HF-S2 state is non-finite for {key}")
    expected_kernel = _binomial_kernel_5().to(
        device=extension.low_pass_kernel.device,
        dtype=extension.low_pass_kernel.dtype,
    )
    if not torch.equal(extension.low_pass_kernel, expected_kernel):
        raise RuntimeError("CP-HF-S2 fixed binomial kernel differs")
    if len(state) != FORMAL_STATE_KEY_COUNT:
        raise RuntimeError("formal CP-HF-S2 state-key count differs")
    if _parameter_count(extension) != EXTENSION_PARAMETER_COUNT:
        raise RuntimeError("CP-HF-S2 extension parameter count differs")
    if _parameter_count(model) != FORMAL_PARAMETER_COUNT:
        raise RuntimeError("formal CP-HF-S2 parameter count differs")
    if any(not value.requires_grad for value in model.parameters()):
        raise RuntimeError("formal CP-HF-S2 contains frozen parameters")
    if require_identity_initialization and torch.count_nonzero(
        extension.raw_scale.detach()
    ).item() != 0:
        raise RuntimeError("CP-HF-S2 raw_scale is not identity initialized")
    architecture_seed = getattr(model, _ARCHITECTURE_SEED_ATTRIBUTE, None)
    initialization_seed = getattr(model, _INITIALIZATION_SEED_ATTRIBUTE, None)
    if isinstance(architecture_seed, bool) or not isinstance(
        architecture_seed, int
    ):
        raise RuntimeError("CP-HF-S2 architecture seed marker differs")
    if isinstance(initialization_seed, bool) or not isinstance(
        initialization_seed, int
    ):
        raise RuntimeError("CP-HF-S2 initialization seed marker differs")
    if initialization_seed != derive_cp_hf_s2_initialization_seed(
        architecture_seed
    ):
        raise RuntimeError("CP-HF-S2 initialization seed binding differs")
    if (
        extension.architecture_seed != architecture_seed
        or extension.initialization_seed != initialization_seed
    ):
        raise RuntimeError("CP-HF-S2 module seed attributes differ")
    base_manifest = getattr(model, _BASE_MANIFEST_ATTRIBUTE, None)
    if not isinstance(base_manifest, Mapping):
        raise RuntimeError("CP-HF-S2 base manifest snapshot is absent")
    if model.architecture_manifest() != base_manifest:
        raise RuntimeError("CP-HF-S2 current base architecture manifest differs")
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "model": EXPERIMENT_NAME,
        "base_model": "EviSIRST",
        "dataset_scope": SUPPORTED_DATASET,
        "binding_target": "_forward_with_relay",
        "bound_function": "cp_hf_s2_forward_with_relay",
        "insertion_point": "after_up_decoder2_finish_before_gt2_and_up_decoder1",
        "affected_outputs": ["gt2", "d0", "out"],
        "unchanged_outputs": ["gt5", "gt4", "gt3"],
        "base_state_key_count": BASE_STATE_KEY_COUNT,
        "extension_state_key_count": EXTENSION_STATE_KEY_COUNT,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "base_parameter_count": BASE_PARAMETER_COUNT,
        "extension_parameter_count": EXTENSION_PARAMETER_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
        "architecture_seed": architecture_seed,
        "initialization_seed": initialization_seed,
        "module_manifest": extension.architecture_manifest(),
        "base_architecture_schema": base_manifest.get("schema"),
        "loss_change": False,
        "data_change": False,
        "checkpoint_selector_change": False,
        "public_test_supported": False,
    }


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "BASE_PARAMETER_COUNT",
    "BASE_STATE_KEY_COUNT",
    "ContextPurifiedHighFrequencyS2",
    "EXPERIMENT_NAME",
    "EXPERIMENT_SCHEMA",
    "EXTENSION_PARAMETER_COUNT",
    "EXTENSION_STATE_KEY_COUNT",
    "FORMAL_PARAMETER_COUNT",
    "FORMAL_STATE_KEY_COUNT",
    "MAX_FEATURE_DELTA",
    "MODULE_NAME",
    "STATE_PREFIX",
    "SUPPORTED_DATASET",
    "baseline_reference_forward_with_relay",
    "build_irstd_cp_hf_s2_v1",
    "cp_hf_s2_forward_with_relay",
    "derive_cp_hf_s2_initialization_seed",
    "install_irstd_cp_hf_s2_v1",
    "validate_irstd_cp_hf_s2_v1",
]
