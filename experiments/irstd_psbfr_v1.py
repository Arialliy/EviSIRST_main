"""IRSTD prediction-supported bounded frequency refinement (PSBFR) V1.

This experimental architecture leaves the frozen public EviSIRST sources
untouched.  It registers one refinement module on a clean 564-state-key model
and installs a single forward hook on ``outc``.  The hook observes both the
final decoder feature ``F`` and its clean logit ``z0`` and returns

``z0 + 2 * S * tanh(r)``,

where ``r`` is predicted from stop-gradient low/high-frequency decoder
evidence and ``S`` is a stop-gradient, spatially dilated clean prediction.
The correction terminal is exactly zero at construction.  Consequently the
installed graph starts as an exact clean-logit identity while the terminal
can receive a gradient on the first optimization step.  Stop-gradient inputs
prevent the correction Jacobian from feeding back into the base decoder.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.EviSIRST import EviSIRST, initialize_evisirst
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    validate_formal_qfg_v2_croa_inference_model,
)


EXPERIMENT_SCHEMA = "evisirst_irstd_psbfr_stage_a_v1"
ARCHITECTURE_SCHEMA = "evisirst_irstd_psbfr_architecture_v1"
EXPERIMENT_NAME = "IRSTD-PSBFR-v1"
EXPERIMENT_STAGE = "A"
SUPPORTED_DATASET = "IRSTD-1K"


def derive_psbfr_initialization_seed(architecture_seed: int) -> int:
    """Derive PSBFR's uint63 initialization substream from an init seed."""

    if isinstance(architecture_seed, bool) or not isinstance(
        architecture_seed, int
    ):
        raise TypeError("architecture_seed must be an integer")
    payload = (
        "EviSIRST/PSBFR-v1/extension/" + str(architecture_seed)
    ).encode("ascii")
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, byteorder="big", signed=False) % (1 << 63)


BASE_STATE_KEY_COUNT = 564
BASE_PARAMETER_COUNT = 10_870_130
PSBFR_MODULE_NAME = "decoder_bounded_logit_refinement"
PSBFR_STATE_PREFIX = f"{PSBFR_MODULE_NAME}."
PSBFR_CHANNELS = 32
PSBFR_HIDDEN_CHANNELS = 16
PSBFR_LOW_PASS_KERNEL_SIZE = 5
PSBFR_SUPPORT_KERNEL_SIZE = 7
PSBFR_MAX_LOGIT_DELTA = 2.0
# Compatibility name: this is specifically the seed-42 derived substream.
PSBFR_INITIALIZATION_SEED = derive_psbfr_initialization_seed(42)
PSBFR_PARAMETER_COUNT = 1_073
PSBFR_STATE_KEY_COUNT = 5
FORMAL_PARAMETER_COUNT = BASE_PARAMETER_COUNT + PSBFR_PARAMETER_COUNT
FORMAL_STATE_KEY_COUNT = BASE_STATE_KEY_COUNT + PSBFR_STATE_KEY_COUNT

PSBFR_STATE_KEYS = (
    f"{PSBFR_STATE_PREFIX}corrector.0.weight",
    f"{PSBFR_STATE_PREFIX}corrector.1.bias",
    f"{PSBFR_STATE_PREFIX}corrector.1.weight",
    f"{PSBFR_STATE_PREFIX}corrector.3.bias",
    f"{PSBFR_STATE_PREFIX}corrector.3.weight",
)

_BASE_STATE_KEYS_ATTRIBUTE = "_irstd_psbfr_v1_base_state_keys"
_BASE_MANIFEST_ATTRIBUTE = "_irstd_psbfr_v1_base_manifest"
_HOOK_ID_ATTRIBUTE = "_irstd_psbfr_v1_hook_id"
_ARCHITECTURE_SEED_ATTRIBUTE = "_irstd_psbfr_v1_architecture_seed"
_INITIALIZATION_SEED_ATTRIBUTE = "_irstd_psbfr_v1_initialization_seed"


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not normalized > 0.0 or not torch.isfinite(torch.tensor(normalized)):
        raise ValueError(f"{name} must be finite and positive")
    return normalized


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _all_zero(tensor: torch.Tensor) -> bool:
    return torch.count_nonzero(tensor.detach()).item() == 0


class PredictionSupportedBoundedFrequencyRefinement(nn.Module):
    """Read-only decoder evidence producing a bounded clean-logit correction."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = PSBFR_HIDDEN_CHANNELS,
        low_pass_kernel_size: int = PSBFR_LOW_PASS_KERNEL_SIZE,
        support_kernel_size: int = PSBFR_SUPPORT_KERNEL_SIZE,
        max_logit_delta: float = PSBFR_MAX_LOGIT_DELTA,
        architecture_seed: int = 42,
        initialization_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.hidden_channels = _positive_int(
            hidden_channels, "hidden_channels"
        )
        self.low_pass_kernel_size = _positive_int(
            low_pass_kernel_size, "low_pass_kernel_size"
        )
        self.support_kernel_size = _positive_int(
            support_kernel_size, "support_kernel_size"
        )
        if self.low_pass_kernel_size % 2 != 1:
            raise ValueError("low_pass_kernel_size must be odd")
        if self.support_kernel_size % 2 != 1:
            raise ValueError("support_kernel_size must be odd")
        self.max_logit_delta = _positive_float(
            max_logit_delta, "max_logit_delta"
        )
        derived_seed = derive_psbfr_initialization_seed(architecture_seed)
        if initialization_seed is None:
            initialization_seed = derived_seed
        if isinstance(initialization_seed, bool) or not isinstance(
            initialization_seed, int
        ):
            raise TypeError("initialization_seed must be an integer")
        if initialization_seed != derived_seed:
            raise ValueError(
                "initialization_seed must be derived from architecture_seed"
            )
        self.architecture_seed = architecture_seed
        self.initialization_seed = initialization_seed

        self.corrector = nn.Sequential(
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
                1,
                kernel_size=1,
                bias=True,
            ),
        )
        self.reset_identity()

    @property
    def terminal(self) -> nn.Conv2d:
        terminal = self.corrector[-1]
        if not isinstance(terminal, nn.Conv2d):
            raise RuntimeError("PSBFR correction terminal is not Conv2d")
        return terminal

    def reset_identity(self) -> None:
        """Zero only the correction terminal; no second zero gate is used."""

        with torch.no_grad():
            self.terminal.weight.zero_()
            if self.terminal.bias is None:
                raise RuntimeError("PSBFR correction terminal must have a bias")
            self.terminal.bias.zero_()

    def _validate_inputs(
        self,
        feature: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> None:
        if not isinstance(feature, torch.Tensor):
            raise TypeError("feature must be a Tensor")
        if not isinstance(base_logit, torch.Tensor):
            raise TypeError("base_logit must be a Tensor")
        if feature.ndim != 4:
            raise ValueError(
                f"feature must be BCHW, got shape={tuple(feature.shape)}"
            )
        if base_logit.ndim != 4:
            raise ValueError(
                "base_logit must be BCHW, got "
                f"shape={tuple(base_logit.shape)}"
            )
        if feature.shape[1] != self.channels:
            raise ValueError(
                f"feature has C={feature.shape[1]}, expected C={self.channels}"
            )
        if base_logit.shape[1] != 1:
            raise ValueError("base_logit must have exactly one channel")
        if (
            feature.shape[0] != base_logit.shape[0]
            or feature.shape[-2:] != base_logit.shape[-2:]
        ):
            raise ValueError("feature and base_logit batch/spatial shapes differ")
        if not feature.is_floating_point() or not base_logit.is_floating_point():
            raise TypeError("feature and base_logit must use floating-point dtypes")
        if feature.device != base_logit.device:
            raise ValueError("feature and base_logit devices differ")
        if feature.dtype != base_logit.dtype:
            raise ValueError("feature and base_logit dtypes differ")
        padding = self.low_pass_kernel_size // 2
        if feature.shape[-2] <= padding or feature.shape[-1] <= padding:
            raise ValueError(
                "feature spatial dimensions must exceed reflect padding "
                f"({padding})"
            )

    def refinement_components(
        self,
        feature: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return ``(low, high, raw, support, correction)`` without caching.

        Both evidence inputs are deliberately detached.  Gradients from the
        correction therefore update only ``corrector``; the identity ``z0``
        path continues to train the complete base model normally.
        """

        self._validate_inputs(feature, base_logit)
        stopped_feature = feature.detach()
        padding = self.low_pass_kernel_size // 2
        padded = F.pad(
            stopped_feature,
            (padding, padding, padding, padding),
            mode="reflect",
        )
        low = F.avg_pool2d(
            padded,
            kernel_size=self.low_pass_kernel_size,
            stride=1,
            padding=0,
        )
        high = stopped_feature - low
        raw = self.corrector(torch.cat((low, high.abs()), dim=1))
        stopped_probability = torch.sigmoid(base_logit.detach())
        support_padding = self.support_kernel_size // 2
        support = F.max_pool2d(
            stopped_probability,
            kernel_size=self.support_kernel_size,
            stride=1,
            padding=support_padding,
        )
        correction = self.max_logit_delta * support * torch.tanh(raw)
        return low, high, raw, support, correction

    def forward(
        self,
        feature: torch.Tensor,
        base_logit: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, _, correction = self.refinement_components(
            feature, base_logit
        )
        return base_logit + correction

    def outc_forward_hook(
        self,
        outc: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> torch.Tensor:
        """Replace the clean ``outc`` logit with its bounded refinement."""

        if not isinstance(outc, nn.Conv2d):
            raise TypeError("PSBFR hook target must be Conv2d")
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("outc must receive exactly one Tensor input")
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("outc must return exactly one Tensor")
        return self(inputs[0], output)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "module": "PredictionSupportedBoundedFrequencyRefinement",
            "architecture_seed": self.architecture_seed,
            "initialization_seed": self.initialization_seed,
            "initialization_seed_derivation": (
                "uint63_be_sha256_prefix8_"
                "EviSIRST/PSBFR-v1/extension/{architecture_seed}"
            ),
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "feature_evidence_stop_gradient": True,
            "low_pass": "reflect_pad_avg_pool2d",
            "low_pass_kernel_size": self.low_pass_kernel_size,
            "high_frequency": "stopped_feature_minus_local_mean",
            "corrector_input": ["low_frequency", "absolute_high_frequency"],
            "support_is_corrector_input": False,
            "corrector": "conv1x1_groupnorm1_gelu_conv1x1",
            "corrector_groupnorm_groups": 1,
            "support_source": "sigmoid_stop_gradient_clean_outc_logit",
            "support_dilation": "max_pool2d",
            "support_kernel_size": self.support_kernel_size,
            "correction": "max_logit_delta_times_support_times_tanh_raw",
            "max_logit_delta": self.max_logit_delta,
            "identity_initialization": "correction_terminal_exact_zero",
            "additional_zero_scalar_gate": False,
        }


def _require_clean_base(base: nn.Module) -> tuple[str, ...]:
    if type(base) is not EviSIRST:
        raise TypeError("PSBFR requires the exact EviSIRST class")
    if getattr(base, "mode", None) != "train" or not base.training:
        raise RuntimeError("install PSBFR on a training-mode base")
    if getattr(base, "deepsuper", None) is not True:
        raise RuntimeError("PSBFR requires deep supervision")
    if hasattr(base, "target_survival"):
        raise RuntimeError("clean 564-key base must not register TSS")
    if hasattr(base, PSBFR_MODULE_NAME):
        raise RuntimeError("PSBFR is already registered")
    if any(
        hasattr(base, attribute)
        for attribute in (
            _BASE_STATE_KEYS_ATTRIBUTE,
            _BASE_MANIFEST_ATTRIBUTE,
            _HOOK_ID_ATTRIBUTE,
            _ARCHITECTURE_SEED_ATTRIBUTE,
            _INITIALIZATION_SEED_ATTRIBUTE,
        )
    ):
        raise RuntimeError("PSBFR installation markers already exist")
    if not isinstance(getattr(base, "up_decoder1", None), nn.Module):
        raise TypeError("base.up_decoder1 must be a Module")
    outc = getattr(base, "outc", None)
    if not isinstance(outc, nn.Conv2d):
        raise TypeError("base.outc must be Conv2d")
    if (
        outc.in_channels != PSBFR_CHANNELS
        or outc.out_channels != 1
        or outc.kernel_size != (1, 1)
        or outc.stride != (1, 1)
    ):
        raise RuntimeError("base.outc differs from the formal 32->1 1x1 head")
    if outc._forward_pre_hooks or outc._forward_hooks:
        raise RuntimeError("base.outc already has a local forward hook")

    state_keys = tuple(base.state_dict())
    if len(state_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError(
            f"base must have {BASE_STATE_KEY_COUNT} state keys, "
            f"got {len(state_keys)}"
        )
    if any(key.startswith(PSBFR_STATE_PREFIX) for key in state_keys):
        raise RuntimeError("clean base unexpectedly contains PSBFR state")
    if _parameter_count(base) != BASE_PARAMETER_COUNT:
        raise RuntimeError("base parameter count differs from formal EviSIRST")
    frozen = [
        name for name, parameter in base.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen:
        raise RuntimeError(
            "PSBFR requires every base parameter trainable; frozen: "
            + ", ".join(frozen[:5])
        )
    return state_keys


def _new_formal_refinement(
    reference: torch.Tensor,
    *,
    architecture_seed: int,
) -> PredictionSupportedBoundedFrequencyRefinement:
    # New capacity owns a CPU RNG substream and cannot consume the caller's
    # architecture or runtime RNG stream.
    initialization_seed = derive_psbfr_initialization_seed(architecture_seed)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(initialization_seed)
        refinement = PredictionSupportedBoundedFrequencyRefinement(
            channels=PSBFR_CHANNELS,
            hidden_channels=PSBFR_HIDDEN_CHANNELS,
            low_pass_kernel_size=PSBFR_LOW_PASS_KERNEL_SIZE,
            support_kernel_size=PSBFR_SUPPORT_KERNEL_SIZE,
            max_logit_delta=PSBFR_MAX_LOGIT_DELTA,
            architecture_seed=architecture_seed,
            initialization_seed=initialization_seed,
        )
    refinement.to(device=reference.device, dtype=reference.dtype)
    return refinement


def install_irstd_psbfr_v1(
    base: nn.Module,
    *,
    architecture_seed: int = 42,
    require_all_base_trainable: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Install formal PSBFR in-place while preserving every base state name."""

    if type(require_all_base_trainable) is not bool:
        raise TypeError("require_all_base_trainable must be bool")
    initialization_seed = derive_psbfr_initialization_seed(architecture_seed)
    base_state_keys = _require_clean_base(base)
    if not require_all_base_trainable:
        raise ValueError("formal PSBFR does not permit a frozen-base contract")
    base_validation = validate_formal_qfg_v2_croa_inference_model(
        base,
        require_identity_initialized_qfg=True,
    )
    base_manifest = copy.deepcopy(base.architecture_manifest())

    refinement = _new_formal_refinement(
        base.outc.weight,
        architecture_seed=architecture_seed,
    )
    refinement.train(base.training)
    base.add_module(PSBFR_MODULE_NAME, refinement)
    handle = base.outc.register_forward_hook(
        refinement.outc_forward_hook,
        with_kwargs=False,
    )
    setattr(base, _BASE_STATE_KEYS_ATTRIBUTE, base_state_keys)
    setattr(base, _BASE_MANIFEST_ATTRIBUTE, base_manifest)
    setattr(base, _HOOK_ID_ATTRIBUTE, int(handle.id))
    setattr(base, _ARCHITECTURE_SEED_ATTRIBUTE, architecture_seed)
    setattr(base, _INITIALIZATION_SEED_ATTRIBUTE, initialization_seed)

    manifest = validate_irstd_psbfr_v1(
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
        "architecture_seed": architecture_seed,
        "initialization_seed": initialization_seed,
        "base_formal_validation": base_validation,
        "architecture_manifest": manifest,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
    }
    return base, metadata


def build_irstd_psbfr_v1(
    dataset: str = SUPPORTED_DATASET,
    *,
    seed: int = 42,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a checkpoint-free, full-model-scratch formal PSBFR graph."""

    if dataset != SUPPORTED_DATASET:
        raise ValueError(f"PSBFR supports only {SUPPORTED_DATASET!r}")
    if type(training) is not bool:
        raise TypeError("training must be bool")
    base, base_metadata = initialize_evisirst(dataset, seed=seed, training=True)
    model, extension_metadata = install_irstd_psbfr_v1(
        base,
        architecture_seed=seed,
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
            "architecture_seed": seed,
            "training_mode": training,
            "base_model_metadata": dict(base_metadata),
            "full_model_scratch_training": True,
            "all_base_parameters_trainable": all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith(PSBFR_STATE_PREFIX)
            ),
            "all_extension_parameters_trainable": all(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith(PSBFR_STATE_PREFIX)
            ),
        }
    )
    return model, metadata


def _validate_extension_shapes(state: Mapping[str, torch.Tensor]) -> None:
    expected = {
        f"{PSBFR_STATE_PREFIX}corrector.0.weight": (16, 64, 1, 1),
        f"{PSBFR_STATE_PREFIX}corrector.1.weight": (16,),
        f"{PSBFR_STATE_PREFIX}corrector.1.bias": (16,),
        f"{PSBFR_STATE_PREFIX}corrector.3.weight": (1, 16, 1, 1),
        f"{PSBFR_STATE_PREFIX}corrector.3.bias": (1,),
    }
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise RuntimeError(f"formal PSBFR state shape differs: {key}")


def _validate_hook(model: nn.Module, refinement: nn.Module) -> int:
    hook_id = getattr(model, _HOOK_ID_ATTRIBUTE, None)
    if type(hook_id) is not int:
        raise RuntimeError("PSBFR hook id marker is absent")
    if model.outc._forward_pre_hooks:
        raise RuntimeError("formal PSBFR outc must not contain a pre-hook")
    hooks = model.outc._forward_hooks
    if set(hooks) != {hook_id}:
        raise RuntimeError("outc must contain exactly the formal PSBFR hook")
    hook = hooks[hook_id]
    if (
        getattr(hook, "__self__", None) is not refinement
        or getattr(hook, "__func__", None)
        is not PredictionSupportedBoundedFrequencyRefinement.outc_forward_hook
    ):
        raise RuntimeError("outc hook is not bound to the formal PSBFR module")
    return hook_id


def _validated_contract_facts(
    model: nn.Module,
    *,
    require_identity_initialization: bool,
    require_all_trainable: bool,
) -> dict[str, Any]:
    if type(model) is not EviSIRST:
        raise TypeError("formal PSBFR model must retain the exact base class")
    refinement = getattr(model, PSBFR_MODULE_NAME, None)
    if type(refinement) is not PredictionSupportedBoundedFrequencyRefinement:
        raise TypeError("formal PSBFR module type differs")
    if (
        refinement.channels != PSBFR_CHANNELS
        or refinement.hidden_channels != PSBFR_HIDDEN_CHANNELS
        or refinement.low_pass_kernel_size != PSBFR_LOW_PASS_KERNEL_SIZE
        or refinement.support_kernel_size != PSBFR_SUPPORT_KERNEL_SIZE
        or refinement.max_logit_delta != PSBFR_MAX_LOGIT_DELTA
    ):
        raise RuntimeError("formal PSBFR configuration differs")
    architecture_seed = getattr(model, _ARCHITECTURE_SEED_ATTRIBUTE, None)
    initialization_seed = getattr(model, _INITIALIZATION_SEED_ATTRIBUTE, None)
    try:
        derived_seed = derive_psbfr_initialization_seed(architecture_seed)
    except TypeError as exc:
        raise RuntimeError("formal PSBFR architecture seed is absent") from exc
    if (
        initialization_seed != derived_seed
        or refinement.architecture_seed != architecture_seed
        or refinement.initialization_seed != derived_seed
    ):
        raise RuntimeError("formal PSBFR initialization seed binding differs")
    if not isinstance(getattr(model, "outc", None), nn.Conv2d):
        raise TypeError("formal model outc must be Conv2d")
    hook_id = _validate_hook(model, refinement)

    state = model.state_dict()
    extension_keys = tuple(
        sorted(key for key in state if key.startswith(PSBFR_STATE_PREFIX))
    )
    if extension_keys != PSBFR_STATE_KEYS:
        raise RuntimeError("formal PSBFR state keys differ")
    if len(state) != FORMAL_STATE_KEY_COUNT:
        raise RuntimeError("formal PSBFR total state-key count differs")
    base_keys = tuple(
        key for key in state if not key.startswith(PSBFR_STATE_PREFIX)
    )
    installed_base_keys = getattr(model, _BASE_STATE_KEYS_ATTRIBUTE, None)
    if type(installed_base_keys) is not tuple or base_keys != installed_base_keys:
        raise RuntimeError("base state names changed after PSBFR installation")
    if len(base_keys) != BASE_STATE_KEY_COUNT:
        raise RuntimeError("formal PSBFR base state-key count differs")
    _validate_extension_shapes(state)

    if _parameter_count(refinement) != PSBFR_PARAMETER_COUNT:
        raise RuntimeError("formal PSBFR extension parameter count differs")
    if _parameter_count(model) != FORMAL_PARAMETER_COUNT:
        raise RuntimeError("formal PSBFR total parameter count differs")
    reference = refinement.terminal.weight
    if reference.device != model.outc.weight.device:
        raise RuntimeError("PSBFR and outc devices differ")
    if reference.dtype != model.outc.weight.dtype:
        raise RuntimeError("PSBFR and outc dtypes differ")

    base_manifest = getattr(model, _BASE_MANIFEST_ATTRIBUTE, None)
    if not isinstance(base_manifest, Mapping):
        raise RuntimeError("base architecture manifest snapshot is absent")
    if model.architecture_manifest() != dict(base_manifest):
        raise RuntimeError("base architecture manifest changed after installation")

    base_frozen = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith(PSBFR_STATE_PREFIX)
        and not parameter.requires_grad
    ]
    extension_frozen = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith(PSBFR_STATE_PREFIX)
        and not parameter.requires_grad
    ]
    if require_all_trainable and (base_frozen or extension_frozen):
        raise RuntimeError(
            "formal PSBFR model contains frozen parameters: "
            + ", ".join((base_frozen + extension_frozen)[:5])
        )
    if require_identity_initialization:
        terminal = refinement.terminal
        if (
            not _all_zero(terminal.weight)
            or terminal.bias is None
            or not _all_zero(terminal.bias)
        ):
            raise RuntimeError("formal PSBFR identity initialization differs")
    return {
        "refinement": refinement,
        "base_manifest": copy.deepcopy(dict(base_manifest)),
        "hook_id": hook_id,
        "base_frozen": base_frozen,
        "extension_frozen": extension_frozen,
        "architecture_seed": architecture_seed,
        "initialization_seed": derived_seed,
    }


def _architecture_manifest_from_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    refinement = facts["refinement"]
    if not isinstance(
        refinement, PredictionSupportedBoundedFrequencyRefinement
    ):
        raise TypeError("validated PSBFR fact differs")
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "stage": EXPERIMENT_STAGE,
        "dataset_scope": SUPPORTED_DATASET,
        "base_model": "EviSIRST",
        "base_architecture_manifest": copy.deepcopy(facts["base_manifest"]),
        "integration": "registered_outc_forward_hook",
        "insertion_source": "base.up_decoder1(d2,x1)_output_and_clean_outc_logit",
        "insertion_target": "base.outc_output",
        "insertion_order": "up_decoder1_then_outc_then_bounded_logit_refinement",
        "forward_rewrite": False,
        "source_base_files_modified": False,
        "psbfr_module_name": PSBFR_MODULE_NAME,
        "psbfr_state_prefix": PSBFR_STATE_PREFIX,
        "base_state_key_count": BASE_STATE_KEY_COUNT,
        "psbfr_state_key_count": PSBFR_STATE_KEY_COUNT,
        "psbfr_state_keys": list(PSBFR_STATE_KEYS),
        "total_state_key_count": FORMAL_STATE_KEY_COUNT,
        "base_parameter_count": BASE_PARAMETER_COUNT,
        "psbfr_parameter_count": PSBFR_PARAMETER_COUNT,
        "total_parameter_count": FORMAL_PARAMETER_COUNT,
        "architecture_seed": facts["architecture_seed"],
        "initialization_seed": facts["initialization_seed"],
        "initialization_seed_derivation": (
            "uint63_be_sha256_prefix8_"
            "EviSIRST/PSBFR-v1/extension/{architecture_seed}"
        ),
        "initialization_seed_role": (
            "deterministically_derived_from_architecture_seed_not_run_seed"
        ),
        "identity_initialization": "correction_terminal_exact_zero",
        "full_model_scratch_training": True,
        "warm_start_required": False,
        "base_parameters_frozen": False,
        "optimizer_parameter_scope": "all_model_parameters",
        "component_manifest": refinement.architecture_manifest(),
    }


def irstd_psbfr_v1_architecture_manifest(model: nn.Module) -> dict[str, Any]:
    """Return the strict formal PSBFR architecture manifest for ``model``."""

    facts = _validated_contract_facts(
        model,
        require_identity_initialization=False,
        require_all_trainable=True,
    )
    return _architecture_manifest_from_facts(facts)


def validate_irstd_psbfr_v1(
    model: nn.Module,
    *,
    require_identity_initialization: bool = False,
    require_all_trainable: bool = True,
) -> dict[str, Any]:
    """Strictly validate PSBFR state, hook, initialization, and trainability."""

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
        "integration": "registered_outc_forward_hook",
        "insertion_target": "base.outc_output",
        "total_state_key_count": FORMAL_STATE_KEY_COUNT,
        "psbfr_parameter_count": PSBFR_PARAMETER_COUNT,
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
    "EXPERIMENT_NAME",
    "EXPERIMENT_SCHEMA",
    "FORMAL_PARAMETER_COUNT",
    "FORMAL_STATE_KEY_COUNT",
    "PSBFR_INITIALIZATION_SEED",
    "PSBFR_MAX_LOGIT_DELTA",
    "PSBFR_MODULE_NAME",
    "PSBFR_PARAMETER_COUNT",
    "PSBFR_STATE_KEY_COUNT",
    "PSBFR_STATE_KEYS",
    "PSBFR_STATE_PREFIX",
    "PredictionSupportedBoundedFrequencyRefinement",
    "SUPPORTED_DATASET",
    "build_irstd_psbfr_v1",
    "derive_psbfr_initialization_seed",
    "install_irstd_psbfr_v1",
    "irstd_psbfr_v1_architecture_manifest",
    "validate_irstd_psbfr_v1",
]
