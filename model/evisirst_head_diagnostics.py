"""Read-only diagnostics for the frozen EviSIRST V3 output heads.

The frozen V3 forward API exposes its six auxiliary outputs only when the
custom ``model.mode`` attribute is ``"train"``.  Those outputs are already
sigmoid probabilities, not the original pre-sigmoid logits.  This module
temporarily selects that API branch while keeping PyTorch modules in eval
mode, then restores both kinds of state exactly.

For ``logit_blend``, the logits are therefore *recovered* with a clamped
inverse sigmoid.  This is stable for probabilities equal to zero or one, but
cannot reconstruct information lost through sigmoid saturation or finite
precision.

There is intentionally no alpha search or optimization API here.  A blend
alpha must be chosen on a calibration/validation split by the caller and
passed in as an immutable :class:`FixedHeadSpec` before final evaluation.
Do not choose alpha from final-test labels or metrics.

The temporary state switch mutates the supplied model object and is not safe
for concurrent forwards on that same instance.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Iterator, Literal, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


LEGACY_HEAD_ORDER: Tuple[str, ...] = (
    "gt5",
    "gt4",
    "gt3",
    "gt2",
    "d0",
    "out",
)
"""Positional order returned by the frozen V3 custom ``mode='train'`` API."""

HeadName = Literal["out", "d0", "logit_blend"]
_FIXED_HEAD_NAMES = frozenset(("out", "d0", "logit_blend"))
_DEFAULT_PROBABILITY_EPSILON = 1e-6


def _validated_unit_interval_scalar(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result!r}")
    return result


def _validated_probability_epsilon(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            "probability_epsilon must be a real scalar, "
            f"got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"probability_epsilon must be finite, got {result!r}"
        )
    if not 0.0 < result < 0.5:
        raise ValueError(
            "probability_epsilon must be strictly between 0 and 0.5, "
            f"got {result!r}"
        )
    return result


def _validate_probability_tensor(probability: Tensor, *, name: str) -> None:
    if not isinstance(probability, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not probability.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if probability.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool(((probability < 0.0) | (probability > 1.0)).any()):
        raise ValueError(f"{name} contains values outside [0, 1]")


@dataclass(frozen=True)
class EviSIRSTProbabilityHeads:
    """Named view of the frozen V3 six-probability return contract."""

    gt5: Tensor
    gt4: Tensor
    gt3: Tensor
    gt2: Tensor
    d0: Tensor
    out: Tensor

    @classmethod
    def from_legacy_output(
        cls, output: Sequence[Tensor]
    ) -> "EviSIRSTProbabilityHeads":
        """Validate and name the positional frozen-V3 output tuple."""

        if not isinstance(output, (tuple, list)):
            raise TypeError(
                "expected the custom mode='train' forward to return a tuple "
                "or list of six probability tensors"
            )
        if len(output) != len(LEGACY_HEAD_ORDER):
            raise ValueError(
                "expected exactly six probability tensors in order "
                f"{LEGACY_HEAD_ORDER}, got {len(output)}"
            )

        tensors = tuple(output)
        for name, tensor in zip(LEGACY_HEAD_ORDER, tensors):
            _validate_probability_tensor(tensor, name=name)

        reference = tensors[0]
        for name, tensor in zip(LEGACY_HEAD_ORDER[1:], tensors[1:]):
            if tensor.shape != reference.shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match "
                    f"gt5 shape {tuple(reference.shape)}"
                )
            if tensor.device != reference.device:
                raise ValueError(
                    f"{name} device {tensor.device} does not match "
                    f"gt5 device {reference.device}"
                )
            if tensor.dtype != reference.dtype:
                raise ValueError(
                    f"{name} dtype {tensor.dtype} does not match "
                    f"gt5 dtype {reference.dtype}"
                )

        return cls(*tensors)

    def as_dict(self) -> Dict[str, Tensor]:
        """Return a new insertion-ordered mapping in the legacy head order."""

        return {name: getattr(self, name) for name in LEGACY_HEAD_ORDER}


@dataclass(frozen=True)
class FixedHeadSpec:
    """A pre-committed head rule for evaluation.

    ``alpha`` is mandatory only for ``logit_blend`` and follows
    ``(1 - alpha) * logit(out) + alpha * logit(d0)``.  Passing alpha for a
    direct head is rejected so that a stale calibration value cannot be
    silently ignored.
    """

    head: HeadName
    alpha: Optional[float] = None
    probability_epsilon: float = _DEFAULT_PROBABILITY_EPSILON

    def __post_init__(self) -> None:
        if self.head not in _FIXED_HEAD_NAMES:
            raise ValueError(
                f"head must be one of {sorted(_FIXED_HEAD_NAMES)}, "
                f"got {self.head!r}"
            )

        epsilon = _validated_probability_epsilon(self.probability_epsilon)
        object.__setattr__(self, "probability_epsilon", epsilon)

        if self.head == "logit_blend":
            if self.alpha is None:
                raise ValueError(
                    "alpha must be supplied for logit_blend; calibrate it "
                    "before final evaluation"
                )
            alpha = _validated_unit_interval_scalar(self.alpha, name="alpha")
            object.__setattr__(self, "alpha", alpha)
        elif self.alpha is not None:
            raise ValueError(
                f"alpha is not applicable to the direct {self.head!r} head"
            )


@dataclass(frozen=True)
class HeadPrediction:
    """Selected probability map together with auditable selection metadata."""

    probability: Tensor
    head: HeadName
    alpha: Optional[float]
    probability_epsilon: Optional[float]
    source_contract: str = "frozen_v3_legacy_six_probability_maps"
    logit_recovery: Optional[str] = None

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Serializable metadata to retain beside downstream metrics."""

        return {
            "head": self.head,
            "alpha": self.alpha,
            "probability_epsilon": self.probability_epsilon,
            "source_contract": self.source_contract,
            "source_head_order": LEGACY_HEAD_ORDER,
            "logit_recovery": self.logit_recovery,
            "blend_formula": (
                "(1-alpha)*logit(out)+alpha*logit(d0)"
                if self.head == "logit_blend"
                else None
            ),
            "epsilon_policy": (
                "max(requested_epsilon,dtype_machine_epsilon)"
                if self.head == "logit_blend"
                else None
            ),
        }


@contextmanager
def expose_legacy_probability_outputs(model: nn.Module) -> Iterator[nn.Module]:
    """Temporarily expose all six probabilities without enabling training.

    ``model.eval()`` is deliberately separate from the frozen model's custom
    string ``mode``.  Every module's original ``training`` flag is saved and
    restored individually, preserving even a pre-existing mixed train/eval
    configuration.  Restoration happens on normal return and on exceptions.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not hasattr(model, "mode"):
        raise AttributeError(
            "model must expose the frozen EviSIRST custom 'mode' attribute"
        )

    original_mode = model.mode
    training_states = tuple(
        (module, bool(module.training)) for module in model.modules()
    )

    try:
        model.eval()
        model.mode = "train"
        yield model
    finally:
        # Direct assignment is intentional: calling model.train(was_training)
        # would recursively erase a pre-existing mixed child configuration.
        for module, was_training in training_states:
            module.training = was_training
        model.mode = original_mode


def extract_probability_heads(
    model: nn.Module, *forward_args: Any, **forward_kwargs: Any
) -> EviSIRSTProbabilityHeads:
    """Run one inference-only forward and return all six named probabilities."""

    with expose_legacy_probability_outputs(model):
        with torch.inference_mode():
            output = model(*forward_args, **forward_kwargs)
    return EviSIRSTProbabilityHeads.from_legacy_output(output)


def probability_to_logit(
    probability: Tensor,
    *,
    epsilon: float = _DEFAULT_PROBABILITY_EPSILON,
) -> Tensor:
    """Recover a finite approximate logit from a probability tensor.

    The frozen API does not expose raw logits.  Values are clamped before the
    inverse sigmoid.  Float16 and bfloat16 inputs are promoted to float32.  If
    the requested epsilon is below the working dtype's machine epsilon, the
    latter is used as the effective clamp so that the upper boundary remains
    strictly below one.
    """

    _validate_probability_tensor(probability, name="probability")
    requested_epsilon = _validated_probability_epsilon(epsilon)

    if probability.dtype in (torch.float16, torch.bfloat16):
        working_probability = probability.float()
    else:
        working_probability = probability

    dtype_epsilon = float(torch.finfo(working_probability.dtype).eps)
    effective_epsilon = max(requested_epsilon, dtype_epsilon)
    clamped = working_probability.clamp(
        min=effective_epsilon,
        max=1.0 - effective_epsilon,
    )
    return torch.log(clamped) - torch.log1p(-clamped)


def fixed_logit_blend(
    out_probability: Tensor,
    d0_probability: Tensor,
    *,
    alpha: float,
    epsilon: float = _DEFAULT_PROBABILITY_EPSILON,
) -> Tensor:
    """Blend two heads with one caller-supplied, pre-calibrated alpha."""

    _validate_probability_tensor(out_probability, name="out_probability")
    _validate_probability_tensor(d0_probability, name="d0_probability")
    if out_probability.shape != d0_probability.shape:
        raise ValueError("out_probability and d0_probability shapes must match")
    if out_probability.device != d0_probability.device:
        raise ValueError("out_probability and d0_probability devices must match")

    blend_alpha = _validated_unit_interval_scalar(alpha, name="alpha")
    out_logit = probability_to_logit(out_probability, epsilon=epsilon)
    d0_logit = probability_to_logit(d0_probability, epsilon=epsilon)
    if out_logit.dtype != d0_logit.dtype:
        common_dtype = torch.promote_types(out_logit.dtype, d0_logit.dtype)
        out_logit = out_logit.to(common_dtype)
        d0_logit = d0_logit.to(common_dtype)

    blended_logit = (1.0 - blend_alpha) * out_logit + blend_alpha * d0_logit
    return torch.sigmoid(blended_logit)


def select_fixed_head(
    heads: EviSIRSTProbabilityHeads, spec: FixedHeadSpec
) -> HeadPrediction:
    """Apply one immutable, pre-selected head rule; no selection is performed."""

    if not isinstance(heads, EviSIRSTProbabilityHeads):
        raise TypeError("heads must be an EviSIRSTProbabilityHeads instance")
    if not isinstance(spec, FixedHeadSpec):
        raise TypeError("spec must be a FixedHeadSpec instance")

    if spec.head == "out":
        return HeadPrediction(
            probability=heads.out,
            head="out",
            alpha=None,
            probability_epsilon=None,
        )
    if spec.head == "d0":
        return HeadPrediction(
            probability=heads.d0,
            head="d0",
            alpha=None,
            probability_epsilon=None,
        )

    # FixedHeadSpec guarantees that alpha is present for this branch.
    assert spec.alpha is not None
    probability = fixed_logit_blend(
        heads.out,
        heads.d0,
        alpha=spec.alpha,
        epsilon=spec.probability_epsilon,
    )
    return HeadPrediction(
        probability=probability,
        head="logit_blend",
        alpha=spec.alpha,
        probability_epsilon=spec.probability_epsilon,
        logit_recovery="clamped_inverse_sigmoid_from_probability",
    )


def predict_fixed_head(
    model: nn.Module,
    spec: FixedHeadSpec,
    *forward_args: Any,
    **forward_kwargs: Any,
) -> HeadPrediction:
    """Extract six probabilities and apply a caller-supplied fixed rule."""

    heads = extract_probability_heads(model, *forward_args, **forward_kwargs)
    return select_fixed_head(heads, spec)


__all__ = [
    "LEGACY_HEAD_ORDER",
    "EviSIRSTProbabilityHeads",
    "FixedHeadSpec",
    "HeadPrediction",
    "expose_legacy_probability_outputs",
    "extract_probability_heads",
    "probability_to_logit",
    "fixed_logit_blend",
    "select_fixed_head",
    "predict_fixed_head",
]
