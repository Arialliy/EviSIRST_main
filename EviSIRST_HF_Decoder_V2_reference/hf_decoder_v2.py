"""Baseline-preserving, context-purified HF residual adapters for EviSIRST.

The adapter is an exact identity at initialization because ``raw_scale`` is
initialized to zero. It should be placed *after* an existing clean decoder
stage, not used to replace the decoder itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FixedBinomialLowPass(nn.Module):
    """Channel-wise 5x5 binomial low-pass filter with no trainable state."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size != 5:
            raise ValueError("V2 protocol fixes the binomial kernel to 5x5")
        vector = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
        kernel = torch.outer(vector, vector)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None], persistent=True)
        self.padding = kernel_size // 2

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected BCHW tensor, got shape={tuple(x.shape)}")
        if min(x.shape[-2:]) <= self.padding:
            raise ValueError("feature map is too small for reflect-padded 5x5 low-pass")
        weight = self.kernel.to(dtype=x.dtype).expand(x.shape[1], 1, -1, -1)
        padded = F.pad(x, (self.padding,) * 4, mode="reflect")
        return F.conv2d(padded, weight, groups=x.shape[1])


class ContextPurifiedHFAdapter(nn.Module):
    """A bounded, low-frequency-guided HF residual adapter.

    Args:
        feature_channels: channels of the clean decoder feature ``feature``.
        skip_channels: channels of the encoder skip feature ``skip``.
        reduction: bottleneck ratio for the channel gate.
        max_residual_scale: absolute upper bound of the learned residual scale.

    The output is

        feature + alpha * channel_gate * spatial_gate * residual(high)

    where ``alpha = max_residual_scale * tanh(raw_scale)`` and raw_scale starts
    at exactly zero. Consequently, a newly constructed adapter is bit-identical
    to the clean path in FP32 for finite inputs.
    """

    def __init__(
        self,
        feature_channels: int,
        skip_channels: int | None = None,
        *,
        reduction: int = 8,
        max_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be positive")
        if reduction < 1:
            raise ValueError("reduction must be positive")
        if not 0.0 < max_residual_scale <= 1.0:
            raise ValueError("max_residual_scale must be in (0, 1]")

        skip_channels = feature_channels if skip_channels is None else skip_channels
        if skip_channels < 1:
            raise ValueError("skip_channels must be positive")

        self.feature_channels = int(feature_channels)
        self.skip_channels = int(skip_channels)
        self.max_residual_scale = float(max_residual_scale)
        self.low_pass = FixedBinomialLowPass(kernel_size=5)
        self.skip_projection: nn.Module
        if self.skip_channels == self.feature_channels:
            self.skip_projection = nn.Identity()
        else:
            self.skip_projection = nn.Conv2d(
                self.skip_channels,
                self.feature_channels,
                kernel_size=1,
                bias=True,
            )

        hidden = max(8, self.feature_channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3 * self.feature_channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.feature_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_branch = nn.Sequential(
            nn.Conv2d(
                self.feature_channels,
                self.feature_channels,
                kernel_size=3,
                padding=1,
                groups=self.feature_channels,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.feature_channels,
                self.feature_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Tanh(),
        )
        self.raw_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def residual_scale(self) -> Tensor:
        return self.max_residual_scale * torch.tanh(self.raw_scale)

    def forward(self, feature: Tensor, skip: Tensor) -> Tensor:
        if feature.ndim != 4 or skip.ndim != 4:
            raise ValueError("feature and skip must both be BCHW tensors")
        if feature.shape[0] != skip.shape[0]:
            raise ValueError("feature and skip batch sizes differ")
        if feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"feature channels={feature.shape[1]}, expected={self.feature_channels}"
            )
        if skip.shape[1] != self.skip_channels:
            raise ValueError(f"skip channels={skip.shape[1]}, expected={self.skip_channels}")
        if skip.shape[-2:] != feature.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        skip = self.skip_projection(skip)
        low = self.low_pass(skip)
        high = skip - low
        high_abs = high.abs()

        channel_context = torch.cat((feature, low, high_abs), dim=1)
        channel_gate = self.channel_gate(channel_context)
        spatial_context = torch.cat(
            (
                feature.abs().mean(dim=1, keepdim=True),
                low.abs().mean(dim=1, keepdim=True),
                high_abs.mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = self.spatial_gate(spatial_context)
        residual = self.residual_branch(high)
        return feature + self.residual_scale.to(feature.dtype) * channel_gate * spatial_gate * residual


ADAPTER_PREFIXES = ("hf_v2_d2.", "hf_v2_d1.")
HF_V2_VARIANTS = ("hf_v2_s2", "hf_v2_s21")


def attach_hf_decoder_v2(
    model: nn.Module,
    *,
    variant: str,
    base_channel: int,
) -> nn.Module:
    """Register V2 adapters on an already-built clean-R1 graph.

    For the public SCTransNet channel contract, decoder stage ``d2`` has
    ``base_channel`` channels while skip ``x2`` has ``2 * base_channel``;
    decoder stage ``d1`` and skip ``x1`` both have ``base_channel`` channels.
    The clean model must not call this function.
    """

    if variant not in HF_V2_VARIANTS:
        raise ValueError(f"unsupported HF V2 variant {variant!r}: {HF_V2_VARIANTS}")
    if base_channel < 1:
        raise ValueError("base_channel must be positive")
    if hasattr(model, "hf_v2_d2") or hasattr(model, "hf_v2_d1"):
        raise RuntimeError("HF V2 adapters are already registered")
    model.add_module(
        "hf_v2_d2",
        ContextPurifiedHFAdapter(
            feature_channels=base_channel,
            skip_channels=2 * base_channel,
        ),
    )
    if variant == "hf_v2_s21":
        model.add_module(
            "hf_v2_d1",
            ContextPurifiedHFAdapter(
                feature_channels=base_channel,
                skip_channels=base_channel,
            ),
        )
    model.hf_v2_variant = variant
    return model


def adapter_state_keys(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        key for key in model.state_dict() if key.startswith(ADAPTER_PREFIXES)
    )


def _extract_state_dict(payload: Any) -> Mapping[str, Tensor]:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("checkpoint does not contain a non-empty state_dict")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in payload.items()):
        raise TypeError("state_dict must map string keys to tensors")
    return payload


def load_clean_r1_warm_start(
    model: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Load a clean-R1 checkpoint while permitting only V2 adapter keys to be missing.

    Full V2 resume must continue to use ``strict=True``; this function is only
    for the one-way baseline -> V2 mechanism-screen initialization.
    """

    payload: Any
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    else:
        payload = checkpoint
    state = _extract_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    invalid_missing = tuple(
        key for key in missing if not key.startswith(ADAPTER_PREFIXES)
    )
    if unexpected or invalid_missing:
        raise RuntimeError(
            "baseline warm-start contract failed: "
            f"unexpected={unexpected}, invalid_missing={invalid_missing}"
        )
    expected_adapter_keys = set(adapter_state_keys(model))
    if set(missing) != expected_adapter_keys:
        raise RuntimeError(
            "missing-key set differs from the exact V2 adapter state: "
            f"missing={sorted(missing)}, expected={sorted(expected_adapter_keys)}"
        )
    for name, module in model.named_modules():
        if isinstance(module, ContextPurifiedHFAdapter):
            if module.raw_scale.detach().item() != 0.0:
                raise RuntimeError(f"{name}.raw_scale is not exact zero after warm-start")
    return {
        "warm_start": "clean_r1_to_hf_decoder_v2",
        "missing_adapter_keys": sorted(missing),
        "unexpected_keys": [],
        "strict_resume_required_after_first_save": True,
    }


@torch.no_grad()
def assert_zero_init_identity(
    adapter: ContextPurifiedHFAdapter,
    feature: Tensor,
    skip: Tensor,
) -> None:
    """Require exact tensor equality before V2 training is authorized."""

    adapter.eval()
    output = adapter(feature, skip)
    if not torch.equal(output, feature):
        max_abs = float((output - feature).abs().max().item())
        raise AssertionError(f"zero-init adapter is not exact identity; max_abs={max_abs}")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "ADAPTER_PREFIXES",
    "ContextPurifiedHFAdapter",
    "HF_V2_VARIANTS",
    "FixedBinomialLowPass",
    "adapter_state_keys",
    "attach_hf_decoder_v2",
    "assert_zero_init_identity",
    "load_clean_r1_warm_start",
    "trainable_parameter_count",
]
