"""Stage-A IRSTD context-guided high-frequency decoder residual.

This experiment deliberately leaves the frozen public EviSIRST sources
untouched.  :func:`install_irstd_hf_decoder_v1` registers one residual module
on a clean 564-state-key EviSIRST instance and installs a single pre-hook on
``outc``.  The hook therefore receives exactly the output of
``up_decoder1(d2, x1)`` and transforms it immediately before the final 1x1
segmentation convolution.

The formal Stage-A path is full-model scratch training: every base parameter
and every extension parameter remains trainable.  ``gamma == 0`` makes the
new branch an exact identity at construction; this is an initialization
contract, not a warm-start or freezing policy.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.EviSIRST import EviSIRST, initialize_evisirst
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    validate_formal_qfg_v2_croa_inference_model,
)


EXPERIMENT_SCHEMA = "evisirst_irstd_hf_decoder_stage_a_v1"
ARCHITECTURE_SCHEMA = "evisirst_irstd_hf_decoder_architecture_v1"
EXPERIMENT_NAME = "IRSTD-HF-Decoder-v1"
EXPERIMENT_STAGE = "A"
SUPPORTED_DATASET = "IRSTD-1K"

BASE_STATE_KEY_COUNT = 564
BASE_PARAMETER_COUNT = 10_870_130
HF_DECODER_MODULE_NAME = "decoder_hf_residual"
HF_DECODER_STATE_PREFIX = f"{HF_DECODER_MODULE_NAME}."
HF_DECODER_CHANNELS = 32
HF_DECODER_HIDDEN_CHANNELS = 16
HF_DECODER_KERNEL_SIZE = 5
HF_DECODER_INITIALIZATION_SEED = 716_725_840_394_809_692
HF_DECODER_PARAMETER_COUNT = 2_913
HF_DECODER_STATE_KEY_COUNT = 8
FORMAL_PARAMETER_COUNT = BASE_PARAMETER_COUNT + HF_DECODER_PARAMETER_COUNT
FORMAL_STATE_KEY_COUNT = BASE_STATE_KEY_COUNT + HF_DECODER_STATE_KEY_COUNT

HF_DECODER_STATE_KEYS = (
    f"{HF_DECODER_STATE_PREFIX}gamma",
    f"{HF_DECODER_STATE_PREFIX}gate.0.weight",
    f"{HF_DECODER_STATE_PREFIX}gate.1.bias",
    f"{HF_DECODER_STATE_PREFIX}gate.1.weight",
    f"{HF_DECODER_STATE_PREFIX}gate.3.bias",
    f"{HF_DECODER_STATE_PREFIX}gate.3.weight",
    f"{HF_DECODER_STATE_PREFIX}high_projection.0.weight",
    f"{HF_DECODER_STATE_PREFIX}high_projection.1.weight",
)

_BASE_STATE_KEYS_ATTRIBUTE = "_irstd_hf_decoder_v1_base_state_keys"
_BASE_MANIFEST_ATTRIBUTE = "_irstd_hf_decoder_v1_base_manifest"
_HOOK_ID_ATTRIBUTE = "_irstd_hf_decoder_v1_hook_id"


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _all_zero(tensor: torch.Tensor) -> bool:
    return torch.count_nonzero(tensor.detach()).item() == 0


class ContextGuidedHighFrequencyResidual(nn.Module):
    """Identity-initialized, context-gated high-frequency residual.

    A reflect-padded local mean supplies low-frequency context ``L`` and
    ``H = feature - L`` supplies signed high-frequency evidence.  A gate built
    from ``concat(L, abs(H))`` conditions a depthwise/pointwise high-frequency
    projection.  The learned scalar terminal is zero at construction, so the
    returned tensor is bitwise equal to the input for finite floating-point
    inputs while gradients can first activate ``gamma``.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = HF_DECODER_HIDDEN_CHANNELS,
        kernel_size: int = HF_DECODER_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.hidden_channels = _positive_int(
            hidden_channels, "hidden_channels"
        )
        self.kernel_size = _positive_int(kernel_size, "kernel_size")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        self.gate = nn.Sequential(
            nn.Conv2d(
                2 * self.channels,
                self.hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(1, self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_channels,
                self.channels,
                kernel_size=1,
                bias=True,
            ),
        )
        self.high_projection = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                groups=self.channels,
                bias=False,
            ),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=1,
                bias=False,
            ),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.reset_identity()

    def reset_identity(self) -> None:
        """Restore the two formal zero terminals without reinitializing capacity."""

        with torch.no_grad():
            self.gamma.zero_()
            terminal = self.gate[-1]
            if not isinstance(terminal, nn.Conv2d):
                raise RuntimeError("gate terminal is not Conv2d")
            terminal.weight.zero_()
            if terminal.bias is None:
                raise RuntimeError("gate terminal must have a bias")
            terminal.bias.zero_()

    def _validate_feature(self, feature: torch.Tensor) -> None:
        if not isinstance(feature, torch.Tensor):
            raise TypeError("feature must be a Tensor")
        if feature.ndim != 4:
            raise ValueError(
                f"feature must be BCHW, got shape={tuple(feature.shape)}"
            )
        if feature.shape[1] != self.channels:
            raise ValueError(
                f"feature has C={feature.shape[1]}, expected C={self.channels}"
            )
        if not feature.is_floating_point():
            raise TypeError("feature must use a floating-point dtype")
        padding = self.kernel_size // 2
        if feature.shape[-2] <= padding or feature.shape[-1] <= padding:
            raise ValueError(
                "feature spatial dimensions must exceed reflect padding "
                f"({padding})"
            )

    def residual_components(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(low, high, gate, projected_high)`` without tensor caching."""

        self._validate_feature(feature)
        padding = self.kernel_size // 2
        padded = F.pad(
            feature,
            (padding, padding, padding, padding),
            mode="reflect",
        )
        low = F.avg_pool2d(
            padded,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )
        high = feature - low
        gate = torch.sigmoid(self.gate(torch.cat((low, high.abs()), dim=1)))
        projected_high = self.high_projection(high * gate)
        return low, high, gate, projected_high

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, _, projected_high = self.residual_components(feature)
        return feature + torch.tanh(self.gamma) * projected_high

    def outc_forward_pre_hook(
        self,
        outc: nn.Module,
        inputs: tuple[Any, ...],
    ) -> tuple[torch.Tensor]:
        """Transform the sole positional feature passed to the final head."""

        if not isinstance(outc, nn.Conv2d):
            raise TypeError("HF decoder hook target must be Conv2d")
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("outc must receive exactly one Tensor input")
        return (self(inputs[0]),)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "module": "ContextGuidedHighFrequencyResidual",
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "low_pass": "reflect_pad_avg_pool2d",
            "low_pass_kernel_size": self.kernel_size,
            "high_frequency": "feature_minus_local_mean",
            "gate_input": ["low_frequency", "absolute_high_frequency"],
            "gate_activation": "sigmoid",
            "high_projection": "depthwise_3x3_then_pointwise_1x1",
            "residual_terminal": "tanh_scalar_gamma",
            "identity_terminal_initialization": "exact_zero",
            "gate_terminal_initialization": "exact_zero",
        }


def _require_clean_base(base: nn.Module) -> tuple[str, ...]:
    if type(base) is not EviSIRST:
        raise TypeError("Stage-A HF decoder requires the exact EviSIRST class")
    if getattr(base, "mode", None) != "train" or not base.training:
        raise RuntimeError("install the Stage-A branch on a training-mode base")
    if getattr(base, "deepsuper", None) is not True:
        raise RuntimeError("Stage-A HF decoder requires deep supervision")
    if hasattr(base, "target_survival"):
        raise RuntimeError("clean 564-key base must not register TSS")
    if hasattr(base, HF_DECODER_MODULE_NAME):
        raise RuntimeError("HF decoder residual is already registered")
    if any(
        hasattr(base, attribute)
        for attribute in (
            _BASE_STATE_KEYS_ATTRIBUTE,
            _BASE_MANIFEST_ATTRIBUTE,
            _HOOK_ID_ATTRIBUTE,
        )
    ):
        raise RuntimeError("HF decoder installation markers already exist")
    if not isinstance(getattr(base, "up_decoder1", None), nn.Module):
        raise TypeError("base.up_decoder1 must be a Module")
    outc = getattr(base, "outc", None)
    if not isinstance(outc, nn.Conv2d):
        raise TypeError("base.outc must be Conv2d")
    if (
        outc.in_channels != HF_DECODER_CHANNELS
        or outc.out_channels != 1
        or outc.kernel_size != (1, 1)
        or outc.stride != (1, 1)
    ):
        raise RuntimeError("base.outc differs from the formal 32->1 1x1 head")
    if outc._forward_pre_hooks:
        raise RuntimeError("base.outc already has a forward pre-hook")

    state_keys = tuple(base.state_dict())
    if len(state_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError(
            f"base must have {BASE_STATE_KEY_COUNT} state keys, "
            f"got {len(state_keys)}"
        )
    if any(key.startswith(HF_DECODER_STATE_PREFIX) for key in state_keys):
        raise RuntimeError("clean base unexpectedly contains HF decoder state")
    if _parameter_count(base) != BASE_PARAMETER_COUNT:
        raise RuntimeError("base parameter count differs from formal EviSIRST")
    frozen = [name for name, parameter in base.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(
            "Stage-A requires every base parameter trainable; frozen: "
            + ", ".join(frozen[:5])
        )
    return state_keys


def _new_formal_residual(reference: torch.Tensor) -> ContextGuidedHighFrequencyResidual:
    # New capacity uses one architecture-owned CPU substream and does not
    # rewrite or consume the caller's runtime RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(HF_DECODER_INITIALIZATION_SEED)
        residual = ContextGuidedHighFrequencyResidual(
            channels=HF_DECODER_CHANNELS,
            hidden_channels=HF_DECODER_HIDDEN_CHANNELS,
            kernel_size=HF_DECODER_KERNEL_SIZE,
        )
    residual.to(device=reference.device, dtype=reference.dtype)
    return residual


def install_irstd_hf_decoder_v1(
    base: nn.Module,
    *,
    require_all_base_trainable: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Install the formal branch in-place while preserving all base key names."""

    if type(require_all_base_trainable) is not bool:
        raise TypeError("require_all_base_trainable must be bool")
    base_state_keys = _require_clean_base(base)
    if not require_all_base_trainable:
        raise ValueError(
            "formal Stage-A construction does not permit a frozen-base contract"
        )
    base_validation = validate_formal_qfg_v2_croa_inference_model(
        base,
        require_identity_initialized_qfg=True,
    )
    base_manifest = copy.deepcopy(base.architecture_manifest())

    residual = _new_formal_residual(base.outc.weight)
    residual.train(base.training)
    base.add_module(HF_DECODER_MODULE_NAME, residual)
    handle = base.outc.register_forward_pre_hook(
        residual.outc_forward_pre_hook,
        with_kwargs=False,
    )
    setattr(base, _BASE_STATE_KEYS_ATTRIBUTE, base_state_keys)
    setattr(base, _BASE_MANIFEST_ATTRIBUTE, base_manifest)
    setattr(base, _HOOK_ID_ATTRIBUTE, int(handle.id))

    manifest = validate_irstd_hf_decoder_v1(
        base,
        require_identity_initialization=True,
        require_all_trainable=True,
    )
    metadata = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "stage": EXPERIMENT_STAGE,
        "dataset_scope": SUPPORTED_DATASET,
        "initialization_mode": "true_scratch_extension_install",
        "parent_checkpoint": None,
        "warm_start_used": False,
        "base_parameters_frozen": False,
        "optimizer_parameter_scope": "all_model_parameters",
        "base_formal_validation": base_validation,
        "architecture_manifest": manifest,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
    }
    return base, metadata


def build_irstd_hf_decoder_v1(
    dataset: str = SUPPORTED_DATASET,
    *,
    seed: int = 42,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a checkpoint-free, full-model-scratch Stage-A graph."""

    if dataset != SUPPORTED_DATASET:
        raise ValueError(
            f"Stage-A HF decoder supports only {SUPPORTED_DATASET!r}"
        )
    if type(training) is not bool:
        raise TypeError("training must be bool")
    base, base_metadata = initialize_evisirst(dataset, seed=seed, training=True)
    model, extension_metadata = install_irstd_hf_decoder_v1(base)
    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()
    metadata = dict(extension_metadata)
    metadata.update(
        {
            "architecture_seed": seed,
            "training_mode": training,
            "base_model_metadata": dict(base_metadata),
            "full_model_scratch_training": True,
            "all_base_parameters_trainable": all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith(HF_DECODER_STATE_PREFIX)
            ),
            "all_extension_parameters_trainable": all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith(HF_DECODER_STATE_PREFIX)
            ),
        }
    )
    return model, metadata


def _validate_extension_shapes(
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = {
        f"{HF_DECODER_STATE_PREFIX}gamma": (1,),
        f"{HF_DECODER_STATE_PREFIX}gate.0.weight": (16, 64, 1, 1),
        f"{HF_DECODER_STATE_PREFIX}gate.1.weight": (16,),
        f"{HF_DECODER_STATE_PREFIX}gate.1.bias": (16,),
        f"{HF_DECODER_STATE_PREFIX}gate.3.weight": (32, 16, 1, 1),
        f"{HF_DECODER_STATE_PREFIX}gate.3.bias": (32,),
        f"{HF_DECODER_STATE_PREFIX}high_projection.0.weight": (32, 1, 3, 3),
        f"{HF_DECODER_STATE_PREFIX}high_projection.1.weight": (32, 32, 1, 1),
    }
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise RuntimeError(f"formal HF decoder state shape differs: {key}")


def _validate_hook(model: nn.Module, residual: nn.Module) -> int:
    hook_id = getattr(model, _HOOK_ID_ATTRIBUTE, None)
    if type(hook_id) is not int:
        raise RuntimeError("HF decoder hook id marker is absent")
    hooks = model.outc._forward_pre_hooks
    if set(hooks) != {hook_id}:
        raise RuntimeError("outc must contain exactly the formal HF decoder hook")
    hook = hooks[hook_id]
    if (
        getattr(hook, "__self__", None) is not residual
        or getattr(hook, "__func__", None)
        is not ContextGuidedHighFrequencyResidual.outc_forward_pre_hook
    ):
        raise RuntimeError("outc hook is not bound to the formal residual")
    return hook_id


def _validated_contract_facts(
    model: nn.Module,
    *,
    require_identity_initialization: bool,
    require_all_trainable: bool,
) -> dict[str, Any]:
    if type(model) is not EviSIRST:
        raise TypeError("formal HF decoder model must retain the exact base class")
    residual = getattr(model, HF_DECODER_MODULE_NAME, None)
    if type(residual) is not ContextGuidedHighFrequencyResidual:
        raise TypeError("formal HF decoder residual type differs")
    if (
        residual.channels != HF_DECODER_CHANNELS
        or residual.hidden_channels != HF_DECODER_HIDDEN_CHANNELS
        or residual.kernel_size != HF_DECODER_KERNEL_SIZE
    ):
        raise RuntimeError("formal HF decoder configuration differs")
    if not isinstance(getattr(model, "outc", None), nn.Conv2d):
        raise TypeError("formal model outc must be Conv2d")
    hook_id = _validate_hook(model, residual)

    state = model.state_dict()
    extension_keys = tuple(
        sorted(key for key in state if key.startswith(HF_DECODER_STATE_PREFIX))
    )
    if extension_keys != HF_DECODER_STATE_KEYS:
        raise RuntimeError("formal HF decoder state keys differ")
    if len(state) != FORMAL_STATE_KEY_COUNT:
        raise RuntimeError("formal HF decoder total state-key count differs")
    base_keys = tuple(key for key in state if not key.startswith(HF_DECODER_STATE_PREFIX))
    installed_base_keys = getattr(model, _BASE_STATE_KEYS_ATTRIBUTE, None)
    if type(installed_base_keys) is not tuple or base_keys != installed_base_keys:
        raise RuntimeError("base state names changed after HF decoder installation")
    if len(base_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError("formal HF decoder base state-key count differs")
    _validate_extension_shapes(state)

    if _parameter_count(residual) != HF_DECODER_PARAMETER_COUNT:
        raise RuntimeError("formal HF decoder extension parameter count differs")
    if _parameter_count(model) != FORMAL_PARAMETER_COUNT:
        raise RuntimeError("formal HF decoder total parameter count differs")
    if residual.gamma.device != model.outc.weight.device:
        raise RuntimeError("HF decoder and outc devices differ")
    if residual.gamma.dtype != model.outc.weight.dtype:
        raise RuntimeError("HF decoder and outc dtypes differ")

    base_manifest = getattr(model, _BASE_MANIFEST_ATTRIBUTE, None)
    if not isinstance(base_manifest, Mapping):
        raise RuntimeError("base architecture manifest snapshot is absent")
    if model.architecture_manifest() != dict(base_manifest):
        raise RuntimeError("base architecture manifest changed after installation")

    base_frozen = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith(HF_DECODER_STATE_PREFIX)
        and not parameter.requires_grad
    ]
    extension_frozen = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith(HF_DECODER_STATE_PREFIX)
        and not parameter.requires_grad
    ]
    if require_all_trainable and (base_frozen or extension_frozen):
        raise RuntimeError(
            "formal Stage-A model contains frozen parameters: "
            + ", ".join((base_frozen + extension_frozen)[:5])
        )
    if require_identity_initialization:
        terminal = residual.gate[-1]
        if (
            not _all_zero(residual.gamma)
            or not _all_zero(terminal.weight)
            or terminal.bias is None
            or not _all_zero(terminal.bias)
        ):
            raise RuntimeError("formal HF decoder identity initialization differs")
    return {
        "residual": residual,
        "base_manifest": copy.deepcopy(dict(base_manifest)),
        "hook_id": hook_id,
        "base_frozen": base_frozen,
        "extension_frozen": extension_frozen,
    }


def _architecture_manifest_from_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    residual = facts["residual"]
    if not isinstance(residual, ContextGuidedHighFrequencyResidual):
        raise TypeError("validated residual fact differs")
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "stage": EXPERIMENT_STAGE,
        "dataset_scope": SUPPORTED_DATASET,
        "base_model": "EviSIRST",
        "base_architecture_manifest": copy.deepcopy(facts["base_manifest"]),
        "integration": "registered_outc_forward_pre_hook",
        "insertion_source": "base.up_decoder1(d2,x1)_output",
        "insertion_target": "base.outc_input",
        "insertion_order": "up_decoder1_then_hf_residual_then_outc",
        "forward_rewrite": False,
        "source_base_files_modified": False,
        "hf_decoder_module_name": HF_DECODER_MODULE_NAME,
        "hf_decoder_state_prefix": HF_DECODER_STATE_PREFIX,
        "base_state_key_count": BASE_STATE_KEY_COUNT,
        "hf_decoder_state_key_count": HF_DECODER_STATE_KEY_COUNT,
        "hf_decoder_state_keys": list(HF_DECODER_STATE_KEYS),
        "total_state_key_count": FORMAL_STATE_KEY_COUNT,
        "base_parameter_count": BASE_PARAMETER_COUNT,
        "hf_decoder_parameter_count": HF_DECODER_PARAMETER_COUNT,
        "total_parameter_count": FORMAL_PARAMETER_COUNT,
        "initialization_seed": HF_DECODER_INITIALIZATION_SEED,
        "initialization_seed_role": "architecture_only_not_run_seed",
        "identity_initialization": "gamma_exact_zero",
        "full_model_scratch_training": True,
        "warm_start_required": False,
        "base_parameters_frozen": False,
        "optimizer_parameter_scope": "all_model_parameters",
        "component_manifest": residual.architecture_manifest(),
    }


def irstd_hf_decoder_v1_architecture_manifest(
    model: nn.Module,
) -> dict[str, Any]:
    """Return the strict Stage-A architecture manifest for ``model``."""

    facts = _validated_contract_facts(
        model,
        require_identity_initialization=False,
        require_all_trainable=True,
    )
    return _architecture_manifest_from_facts(facts)


def validate_irstd_hf_decoder_v1(
    model: nn.Module,
    *,
    require_identity_initialization: bool = False,
    require_all_trainable: bool = True,
) -> dict[str, Any]:
    """Strictly validate state, hook, trainability, and architecture fields."""

    if type(require_identity_initialization) is not bool:
        raise TypeError("require_identity_initialization must be bool")
    if type(require_all_trainable) is not bool:
        raise TypeError("require_all_trainable must be bool")
    facts = _validated_contract_facts(
        model,
        require_identity_initialization=require_identity_initialization,
        require_all_trainable=require_all_trainable,
    )
    manifest = _architecture_manifest_from_facts(facts)
    required = {
        "schema": ARCHITECTURE_SCHEMA,
        "integration": "registered_outc_forward_pre_hook",
        "insertion_source": "base.up_decoder1(d2,x1)_output",
        "insertion_target": "base.outc_input",
        "total_state_key_count": FORMAL_STATE_KEY_COUNT,
        "hf_decoder_parameter_count": HF_DECODER_PARAMETER_COUNT,
        "base_parameters_frozen": False,
        "optimizer_parameter_scope": "all_model_parameters",
    }
    for name, expected in required.items():
        if manifest.get(name) != expected:
            raise RuntimeError(f"architecture manifest field differs: {name}")
    return manifest


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "BASE_PARAMETER_COUNT",
    "BASE_STATE_KEY_COUNT",
    "ContextGuidedHighFrequencyResidual",
    "EXPERIMENT_NAME",
    "EXPERIMENT_SCHEMA",
    "FORMAL_PARAMETER_COUNT",
    "FORMAL_STATE_KEY_COUNT",
    "HF_DECODER_INITIALIZATION_SEED",
    "HF_DECODER_MODULE_NAME",
    "HF_DECODER_PARAMETER_COUNT",
    "HF_DECODER_STATE_KEY_COUNT",
    "HF_DECODER_STATE_KEYS",
    "HF_DECODER_STATE_PREFIX",
    "SUPPORTED_DATASET",
    "build_irstd_hf_decoder_v1",
    "install_irstd_hf_decoder_v1",
    "irstd_hf_decoder_v1_architecture_manifest",
    "validate_irstd_hf_decoder_v1",
]
