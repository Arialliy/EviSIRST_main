from __future__ import annotations

import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from experiments.irstd_hf_decoder_v1 import (
    ARCHITECTURE_SCHEMA,
    BASE_STATE_KEY_COUNT,
    ContextGuidedHighFrequencyResidual,
    FORMAL_PARAMETER_COUNT,
    FORMAL_STATE_KEY_COUNT,
    HF_DECODER_MODULE_NAME,
    HF_DECODER_PARAMETER_COUNT,
    HF_DECODER_STATE_KEY_COUNT,
    HF_DECODER_STATE_KEYS,
    HF_DECODER_STATE_PREFIX,
    build_irstd_hf_decoder_v1,
    install_irstd_hf_decoder_v1,
    validate_irstd_hf_decoder_v1,
)
from model.EviSIRST import initialize_evisirst


class ContextGuidedHighFrequencyResidualTest(unittest.TestCase):
    def test_zero_terminal_is_bitwise_identity_and_gamma_can_start(self) -> None:
        torch.manual_seed(7)
        residual = ContextGuidedHighFrequencyResidual(
            channels=4,
            hidden_channels=3,
            kernel_size=3,
        )
        feature = torch.randn(2, 4, 9, 11, requires_grad=True)
        output = residual(feature)
        self.assertTrue(torch.equal(output, feature))

        output.sum().backward()
        self.assertIsNotNone(residual.gamma.grad)
        self.assertGreater(float(residual.gamma.grad.abs().item()), 0.0)
        self.assertTrue(torch.equal(feature.grad, torch.ones_like(feature)))
        self.assertEqual(
            torch.count_nonzero(residual.gate[-1].weight.grad).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(residual.high_projection[0].weight.grad).item(),
            0,
        )

    def test_hook_is_exactly_before_outc_and_nonzero_gamma_changes_logits(self) -> None:
        torch.manual_seed(11)
        outc = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        residual = ContextGuidedHighFrequencyResidual(
            channels=4,
            hidden_channels=2,
            kernel_size=3,
        )
        feature = torch.randn(1, 4, 8, 8)
        baseline = outc(feature)
        handle = outc.register_forward_pre_hook(residual.outc_forward_pre_hook)
        try:
            identity_logits = outc(feature)
            self.assertTrue(torch.equal(identity_logits, baseline))
            with torch.no_grad():
                residual.gamma.fill_(0.5)
                residual.high_projection[0].weight.zero_()
                residual.high_projection[0].weight[:, 0, 1, 1] = 1.0
                residual.high_projection[1].weight.zero_()
                for channel in range(4):
                    residual.high_projection[1].weight[channel, channel, 0, 0] = 1.0
            changed_logits = outc(feature)
            self.assertFalse(torch.equal(changed_logits, baseline))
        finally:
            handle.remove()

    def test_component_validates_channels_dtype_and_reflect_extent(self) -> None:
        residual = ContextGuidedHighFrequencyResidual(4, 2, 5)
        with self.assertRaisesRegex(ValueError, "C=3"):
            residual(torch.zeros(1, 3, 8, 8))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            residual(torch.zeros(1, 4, 8, 8, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "reflect padding"):
            residual(torch.zeros(1, 4, 2, 8))


class IRSTDHFDecoderFormalContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = build_irstd_hf_decoder_v1(
            "IRSTD-1K",
            seed=42,
            training=True,
        )

    def test_formal_builder_has_564_base_plus_eight_named_states(self) -> None:
        state = self.model.state_dict()
        extension = tuple(
            sorted(key for key in state if key.startswith(HF_DECODER_STATE_PREFIX))
        )
        base = tuple(
            key for key in state if not key.startswith(HF_DECODER_STATE_PREFIX)
        )
        self.assertEqual(len(base), BASE_STATE_KEY_COUNT)
        self.assertEqual(extension, HF_DECODER_STATE_KEYS)
        self.assertEqual(len(extension), HF_DECODER_STATE_KEY_COUNT)
        self.assertEqual(len(state), FORMAL_STATE_KEY_COUNT)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            FORMAL_PARAMETER_COUNT,
        )
        residual = getattr(self.model, HF_DECODER_MODULE_NAME)
        self.assertEqual(
            sum(parameter.numel() for parameter in residual.parameters()),
            HF_DECODER_PARAMETER_COUNT,
        )

    def test_validator_and_manifest_enforce_stage_a_contract(self) -> None:
        manifest = validate_irstd_hf_decoder_v1(
            self.model,
            require_identity_initialization=True,
        )
        self.assertEqual(manifest["schema"], ARCHITECTURE_SCHEMA)
        self.assertEqual(
            manifest["insertion_order"],
            "up_decoder1_then_hf_residual_then_outc",
        )
        self.assertEqual(manifest["base_state_key_count"], 564)
        self.assertEqual(manifest["hf_decoder_state_key_count"], 8)
        self.assertEqual(manifest["total_state_key_count"], 572)
        self.assertFalse(manifest["base_parameters_frozen"])
        self.assertEqual(
            manifest["optimizer_parameter_scope"], "all_model_parameters"
        )
        self.assertTrue(self.metadata["full_model_scratch_training"])
        self.assertFalse(self.metadata["warm_start_used"])
        self.assertTrue(self.metadata["all_base_parameters_trainable"])
        self.assertTrue(self.metadata["all_extension_parameters_trainable"])

    def test_every_base_and_extension_parameter_is_trainable(self) -> None:
        frozen = [
            name
            for name, parameter in self.model.named_parameters()
            if not parameter.requires_grad
        ]
        self.assertEqual(frozen, [])

    def test_formal_state_strict_load_and_missing_extension_key_rejected(self) -> None:
        state = OrderedDict(
            (key, value.detach().clone())
            for key, value in self.model.state_dict().items()
        )
        self.model.load_state_dict(state, strict=True)
        missing = OrderedDict(state)
        missing.pop(HF_DECODER_STATE_KEYS[0])
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.model.load_state_dict(missing, strict=True)

    def test_validator_rejects_removed_hook_freezing_and_nonidentity_terminal(self) -> None:
        hook_id = getattr(self.model, "_irstd_hf_decoder_v1_hook_id")
        hook = self.model.outc._forward_pre_hooks.pop(hook_id)
        try:
            with self.assertRaisesRegex(RuntimeError, "exactly the formal"):
                validate_irstd_hf_decoder_v1(self.model)
        finally:
            self.model.outc._forward_pre_hooks[hook_id] = hook

        base_name, base_parameter = next(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if not name.startswith(HF_DECODER_STATE_PREFIX)
        )
        base_parameter.requires_grad_(False)
        try:
            with self.assertRaisesRegex(RuntimeError, "frozen parameters"):
                validate_irstd_hf_decoder_v1(self.model)
        finally:
            base_parameter.requires_grad_(True)
        self.assertTrue(base_name)

        residual = getattr(self.model, HF_DECODER_MODULE_NAME)
        with torch.no_grad():
            residual.gamma.fill_(0.1)
        try:
            with self.assertRaisesRegex(RuntimeError, "identity initialization"):
                validate_irstd_hf_decoder_v1(
                    self.model,
                    require_identity_initialization=True,
                )
            validate_irstd_hf_decoder_v1(
                self.model,
                require_identity_initialization=False,
            )
        finally:
            residual.reset_identity()

    def test_duplicate_install_and_nonformal_dataset_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            install_irstd_hf_decoder_v1(self.model)
        with self.assertRaisesRegex(ValueError, "supports only"):
            build_irstd_hf_decoder_v1("NUDT-SIRST")

    def test_installer_refuses_a_frozen_base_contract(self) -> None:
        base, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
        first_parameter = next(base.parameters())
        first_parameter.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "every base parameter"):
            install_irstd_hf_decoder_v1(base)


if __name__ == "__main__":
    unittest.main()
