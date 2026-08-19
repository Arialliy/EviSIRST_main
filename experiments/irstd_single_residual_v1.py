"""IRSTD-1K D0 diagnostic: one encoder skip residual per level.

The frozen EviSIRST relay forward currently forms each reconstructed encoder
feature as ``(reconstruct(encoded) + f) + f``.  This diagnostic keeps the
entire 564-state graph unchanged and binds one controlled instance-level
``_forward_with_relay`` whose only computational change is to omit the second
``+ f`` at levels 1--4.

No runner, loss, data transform, checkpoint selector, or public model source
is changed here.  The baseline-reference entry point intentionally exercises
the copied relay flow with residual multiplicity two so tests can prove that
the copy is bitwise equal to the frozen implementation before D0 is used.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.EviSIRST import EviSIRST, initialize_evisirst
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward


EXPERIMENT_SCHEMA = "evisirst_irstd_single_residual_diagnostic_v1"
ARCHITECTURE_SCHEMA = "evisirst_irstd_single_residual_architecture_v1"
EXPERIMENT_NAME = "IRSTD-Single-Residual-v1"
SUPPORTED_DATASET = "IRSTD-1K"

FORMAL_STATE_KEY_COUNT = 564
FORMAL_PARAMETER_COUNT = 10_870_130
ENCODER_LEVELS = (1, 2, 3, 4)
BASELINE_RESIDUAL_MULTIPLICITY = 2
SINGLE_RESIDUAL_MULTIPLICITY = 1

_INSTALLED_ATTRIBUTE = "_irstd_single_residual_v1_installed"
_BASE_STATE_KEYS_ATTRIBUTE = "_irstd_single_residual_v1_base_state_keys"
_BASE_MANIFEST_ATTRIBUTE = "_irstd_single_residual_v1_base_manifest"

_FROZEN_FORWARD_WITH_RELAY = (
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet._forward_with_relay
)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _residualized_encoder_features(
    reconstructed: Sequence[torch.Tensor],
    skips: Sequence[torch.Tensor],
    *,
    residual_multiplicity: int,
) -> tuple[torch.Tensor, ...]:
    """Apply the frozen addition order with one or two copies of each skip."""

    if len(reconstructed) != len(ENCODER_LEVELS) or len(skips) != len(
        ENCODER_LEVELS
    ):
        raise ValueError("exactly four reconstructed features and skips are required")
    if residual_multiplicity not in (
        SINGLE_RESIDUAL_MULTIPLICITY,
        BASELINE_RESIDUAL_MULTIPLICITY,
    ):
        raise ValueError("residual_multiplicity must be exactly 1 or 2")

    values = tuple(value + skip for value, skip in zip(reconstructed, skips))
    if residual_multiplicity == BASELINE_RESIDUAL_MULTIPLICITY:
        values = tuple(value + skip for value, skip in zip(values, skips))
    return values


def _copied_forward_with_relay(
    self: EviSIRST,
    x: torch.Tensor,
    *,
    residual_multiplicity: int,
):
    """Copy of the frozen QFG/CROA relay flow with one controlled switch."""

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
    x1, x2, x3, x4 = _residualized_encoder_features(
        (
            self.mtc.reconstruct_1(encoded1),
            self.mtc.reconstruct_2(encoded2),
            self.mtc.reconstruct_3(encoded3),
            self.mtc.reconstruct_4(encoded4),
        ),
        (f1, f2, f3, f4),
        residual_multiplicity=residual_multiplicity,
    )

    up4, skip4 = self.up_decoder4.prepare(d5, x4)
    q4, mask4 = self.tpd_ner.forward_stage(
        4,
        (h13, h22, up4),
        tuple(up4.shape[-2:]),
    )
    d4 = self.up_decoder4.finish(up4, skip4, mask4)

    up3, skip3 = self.up_decoder3.prepare(d4, x3)
    q3, mask3 = self.tpd_ner.forward_stage(
        3,
        (h12, h21, q4, up3),
        tuple(up3.shape[-2:]),
    )
    d3 = self.up_decoder3.finish(up3, skip3, mask3)

    up2, skip2 = self.up_decoder2.prepare(d3, x2)
    _, mask2 = self.tpd_ner.forward_stage(
        2,
        (h11, q3, up2),
        tuple(up2.shape[-2:]),
    )
    d2 = self.up_decoder2.finish(up2, skip2, mask2)
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
    """Copied baseline path used only to audit parity with the frozen source."""

    return _copied_forward_with_relay(
        self,
        x,
        residual_multiplicity=BASELINE_RESIDUAL_MULTIPLICITY,
    )


def single_residual_forward_with_relay(self: EviSIRST, x: torch.Tensor):
    """D0 relay path: one reconstructed-feature skip residual at each level."""

    return _copied_forward_with_relay(
        self,
        x,
        residual_multiplicity=SINGLE_RESIDUAL_MULTIPLICITY,
    )


def _require_clean_base(base: nn.Module) -> tuple[str, ...]:
    if type(base) is not EviSIRST:
        raise TypeError("D0 requires the exact clean EviSIRST class")
    if getattr(base, _INSTALLED_ATTRIBUTE, False):
        raise RuntimeError("IRSTD Single-Residual-v1 is already installed")
    if any(
        hasattr(base, name)
        for name in (_BASE_STATE_KEYS_ATTRIBUTE, _BASE_MANIFEST_ATTRIBUTE)
    ):
        raise RuntimeError("D0 installation markers already exist")
    bound = getattr(base, "_forward_with_relay", None)
    if getattr(bound, "__func__", None) is not _FROZEN_FORWARD_WITH_RELAY:
        raise RuntimeError("base _forward_with_relay is not the frozen implementation")
    if hasattr(base, "target_survival"):
        raise RuntimeError("D0 requires the TSS-free clean inference graph")
    if getattr(base.outc, "_forward_pre_hooks", None):
        raise RuntimeError("D0 requires a clean outc without forward pre-hooks")

    state_keys = tuple(base.state_dict())
    if len(state_keys) != FORMAL_STATE_KEY_COUNT:
        raise RuntimeError("D0 base must contain exactly 564 state keys")
    if _parameter_count(base) != FORMAL_PARAMETER_COUNT:
        raise RuntimeError("D0 base parameter count differs from 10,870,130")
    frozen = [name for name, value in base.named_parameters() if not value.requires_grad]
    if frozen:
        raise RuntimeError("D0 requires every base parameter trainable")
    return state_keys


def install_irstd_single_residual_v1(
    base: nn.Module,
) -> tuple[nn.Module, dict[str, Any]]:
    """Bind D0 in place without adding parameters, buffers, or state keys."""

    state_keys = _require_clean_base(base)
    base_manifest = copy.deepcopy(base.architecture_manifest())
    base._forward_with_relay = MethodType(single_residual_forward_with_relay, base)
    setattr(base, _INSTALLED_ATTRIBUTE, True)
    setattr(base, _BASE_STATE_KEYS_ATTRIBUTE, state_keys)
    setattr(base, _BASE_MANIFEST_ATTRIBUTE, base_manifest)

    manifest = validate_irstd_single_residual_v1(base)
    metadata = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "dataset_scope": SUPPORTED_DATASET,
        "initialization_mode": "full_model_scratch_no_new_capacity",
        "parent_checkpoint": None,
        "warm_start_used": False,
        "optimizer_parameter_scope": "all_model_parameters",
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
        "architecture_manifest": manifest,
    }
    return base, metadata


def build_irstd_single_residual_v1(
    dataset: str = SUPPORTED_DATASET,
    *,
    seed: int = 42,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the checkpoint-free D0 diagnostic graph."""

    if dataset != SUPPORTED_DATASET:
        raise ValueError(f"D0 supports only {SUPPORTED_DATASET!r}")
    if type(training) is not bool:
        raise TypeError("training must be bool")
    base, base_metadata = initialize_evisirst(dataset, seed=seed, training=True)
    model, metadata = install_irstd_single_residual_v1(base)
    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()
    ready = dict(metadata)
    ready.update(
        {
            "architecture_seed": seed,
            "training_mode": training,
            "base_model_metadata": dict(base_metadata),
            "full_model_scratch_training": True,
        }
    )
    return model, ready


def validate_irstd_single_residual_v1(model: nn.Module) -> dict[str, Any]:
    """Validate D0 binding, unchanged state schema, and standalone manifest."""

    if type(model) is not EviSIRST:
        raise TypeError("D0 model must retain the exact EviSIRST class")
    if getattr(model, _INSTALLED_ATTRIBUTE, None) is not True:
        raise RuntimeError("D0 installed marker is absent")
    bound = getattr(model, "_forward_with_relay", None)
    if (
        getattr(bound, "__self__", None) is not model
        or getattr(bound, "__func__", None) is not single_residual_forward_with_relay
    ):
        raise RuntimeError("D0 _forward_with_relay binding differs")

    state = model.state_dict()
    state_keys = tuple(state)
    installed_keys = getattr(model, _BASE_STATE_KEYS_ATTRIBUTE, None)
    if type(installed_keys) is not tuple or state_keys != installed_keys:
        raise RuntimeError("D0 changed base state names or order")
    if len(state_keys) != FORMAL_STATE_KEY_COUNT:
        raise RuntimeError("D0 state-key count differs from 564")
    if _parameter_count(model) != FORMAL_PARAMETER_COUNT:
        raise RuntimeError("D0 parameter count differs from 10,870,130")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise RuntimeError("D0 state contains non-finite values")
    frozen = [name for name, value in model.named_parameters() if not value.requires_grad]
    if frozen:
        raise RuntimeError("D0 contains frozen parameters")
    if hasattr(model, "target_survival"):
        raise RuntimeError("D0 unexpectedly registers TSS")

    base_manifest = getattr(model, _BASE_MANIFEST_ATTRIBUTE, None)
    if not isinstance(base_manifest, Mapping):
        raise RuntimeError("D0 base manifest snapshot is absent")
    return {
        "schema": ARCHITECTURE_SCHEMA,
        "model": EXPERIMENT_NAME,
        "base_model": "EviSIRST",
        "dataset_scope": SUPPORTED_DATASET,
        "binding_target": "_forward_with_relay",
        "bound_function": "single_residual_forward_with_relay",
        "encoder_levels_changed": ENCODER_LEVELS,
        "baseline_encoder_formula": "reconstruct_l(encoded_l) + 2 * f_l",
        "diagnostic_encoder_formula": "reconstruct_l(encoded_l) + f_l",
        "only_graph_change": "remove_second_skip_addition_at_levels_1_to_4",
        "new_modules": 0,
        "new_parameters": 0,
        "new_state_keys": 0,
        "state_key_count": FORMAL_STATE_KEY_COUNT,
        "parameter_count": FORMAL_PARAMETER_COUNT,
        "loss_change": False,
        "data_change": False,
        "checkpoint_selector_change": False,
        "base_architecture_schema": base_manifest.get("schema"),
    }


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "BASELINE_RESIDUAL_MULTIPLICITY",
    "ENCODER_LEVELS",
    "EXPERIMENT_NAME",
    "EXPERIMENT_SCHEMA",
    "FORMAL_PARAMETER_COUNT",
    "FORMAL_STATE_KEY_COUNT",
    "SINGLE_RESIDUAL_MULTIPLICITY",
    "SUPPORTED_DATASET",
    "baseline_reference_forward_with_relay",
    "build_irstd_single_residual_v1",
    "install_irstd_single_residual_v1",
    "single_residual_forward_with_relay",
    "validate_irstd_single_residual_v1",
]
