from __future__ import annotations

import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from experiments.irstd_psbfr_v1 import (
    ARCHITECTURE_SCHEMA,
    BASE_PARAMETER_COUNT,
    BASE_STATE_KEY_COUNT,
    FORMAL_PARAMETER_COUNT,
    FORMAL_STATE_KEY_COUNT,
    PSBFR_MAX_LOGIT_DELTA,
    PSBFR_INITIALIZATION_SEED,
    PSBFR_MODULE_NAME,
    PSBFR_PARAMETER_COUNT,
    PSBFR_STATE_KEY_COUNT,
    PSBFR_STATE_KEYS,
    PSBFR_STATE_PREFIX,
    PredictionSupportedBoundedFrequencyRefinement,
    build_irstd_psbfr_v1,
    derive_psbfr_initialization_seed,
    install_irstd_psbfr_v1,
    validate_irstd_psbfr_v1,
)
from model.EviSIRST import initialize_evisirst


class PredictionSupportedBoundedFrequencyRefinementTest(unittest.TestCase):
    def test_public_initialization_seed_derivation_is_exact(self) -> None:
        self.assertEqual(
            derive_psbfr_initialization_seed(42),
            4_455_125_435_119_193_184,
        )
        self.assertEqual(
            PSBFR_INITIALIZATION_SEED,
            derive_psbfr_initialization_seed(42),
        )
        self.assertEqual(
            derive_psbfr_initialization_seed(43),
            4_210_749_627_286_790_960,
        )
        with self.assertRaisesRegex(TypeError, "integer"):
            derive_psbfr_initialization_seed(True)

    def test_identity_bootstrap_and_second_step_trainability(self) -> None:
        torch.manual_seed(7)
        refinement = PredictionSupportedBoundedFrequencyRefinement(
            channels=4,
            hidden_channels=3,
            low_pass_kernel_size=3,
            support_kernel_size=5,
            max_logit_delta=2.0,
        )
        feature = torch.randn(2, 4, 9, 11, requires_grad=True)
        base_logit = torch.randn(2, 1, 9, 11, requires_grad=True)
        output = refinement(feature, base_logit)
        self.assertTrue(torch.equal(output, base_logit))

        loss = output.square().mean()
        loss.backward()
        terminal = refinement.terminal
        self.assertIsNotNone(terminal.weight.grad)
        self.assertIsNotNone(terminal.bias.grad)
        self.assertGreater(float(terminal.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(terminal.bias.grad.abs().sum()), 0.0)
        self.assertEqual(
            torch.count_nonzero(refinement.corrector[0].weight.grad).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(refinement.corrector[1].weight.grad).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(refinement.corrector[1].bias.grad).item(), 0
        )
        self.assertIsNone(feature.grad)
        self.assertTrue(
            torch.allclose(
                base_logit.grad,
                2.0 * base_logit.detach() / base_logit.numel(),
                rtol=1e-6,
                atol=0.0,
            )
        )

        optimizer = torch.optim.SGD(refinement.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        refinement(feature.detach(), base_logit.detach()).square().mean().backward()
        self.assertGreater(
            float(refinement.corrector[0].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(refinement.corrector[1].weight.grad.abs().sum()), 0.0
        )

    def test_feature_and_support_are_detached_and_correction_is_bounded(self) -> None:
        torch.manual_seed(11)
        refinement = PredictionSupportedBoundedFrequencyRefinement(4, 3, 3, 5, 2.0)
        with torch.no_grad():
            refinement.terminal.weight.fill_(0.25)
            refinement.terminal.bias.fill_(0.1)
        feature = torch.randn(1, 4, 9, 10, requires_grad=True)
        base_logit = torch.randn(1, 1, 9, 10, requires_grad=True)
        low, high, raw, support, correction = refinement.refinement_components(
            feature, base_logit
        )
        self.assertFalse(low.requires_grad)
        self.assertFalse(high.requires_grad)
        self.assertFalse(support.requires_grad)
        self.assertTrue(raw.requires_grad)
        feature_grad, logit_grad = torch.autograd.grad(
            correction.sum(),
            (feature, base_logit),
            allow_unused=True,
        )
        self.assertIsNone(feature_grad)
        self.assertIsNone(logit_grad)
        self.assertTrue(
            bool(
                (
                    correction.abs()
                    <= refinement.max_logit_delta * support + 1e-7
                ).all()
            )
        )
        self.assertEqual(refinement.corrector[0].in_channels, 8)
        self.assertEqual(refinement.corrector[0].in_channels, 2 * refinement.channels)

    def test_forward_hook_is_identity_then_changes_only_the_logit(self) -> None:
        torch.manual_seed(13)
        outc = nn.Conv2d(4, 1, kernel_size=1)
        refinement = PredictionSupportedBoundedFrequencyRefinement(4, 3, 3, 5, 2.0)
        feature = torch.randn(1, 4, 9, 9)
        baseline = outc(feature)
        handle = outc.register_forward_hook(refinement.outc_forward_hook)
        try:
            identity = outc(feature)
            self.assertTrue(torch.equal(identity, baseline))
            with torch.no_grad():
                refinement.terminal.weight.fill_(0.2)
                refinement.terminal.bias.fill_(-0.1)
            changed = outc(feature)
            self.assertFalse(torch.equal(changed, baseline))
            delta = changed - baseline
            self.assertLessEqual(
                float(delta.detach().abs().max()), PSBFR_MAX_LOGIT_DELTA + 1e-6
            )
        finally:
            handle.remove()

    def test_configuration_and_input_contracts_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "odd"):
            PredictionSupportedBoundedFrequencyRefinement(4, 2, 4, 5, 2.0)
        with self.assertRaisesRegex(ValueError, "odd"):
            PredictionSupportedBoundedFrequencyRefinement(4, 2, 3, 4, 2.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            PredictionSupportedBoundedFrequencyRefinement(4, 2, 3, 5, 0.0)
        refinement = PredictionSupportedBoundedFrequencyRefinement(4, 2, 5, 7, 2.0)
        with self.assertRaisesRegex(ValueError, "C=3"):
            refinement(torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "exactly one channel"):
            refinement(torch.zeros(1, 4, 8, 8), torch.zeros(1, 2, 8, 8))
        with self.assertRaisesRegex(ValueError, "reflect padding"):
            refinement(torch.zeros(1, 4, 2, 8), torch.zeros(1, 1, 2, 8))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            refinement(
                torch.zeros(1, 4, 8, 8, dtype=torch.int64),
                torch.zeros(1, 1, 8, 8, dtype=torch.int64),
            )


class IRSTDPSBFRFormalContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = build_irstd_psbfr_v1(
            "IRSTD-1K",
            seed=42,
            training=True,
        )

    def test_formal_state_parameter_and_manifest_contract(self) -> None:
        state = self.model.state_dict()
        extension = tuple(
            sorted(key for key in state if key.startswith(PSBFR_STATE_PREFIX))
        )
        base = tuple(
            key for key in state if not key.startswith(PSBFR_STATE_PREFIX)
        )
        self.assertEqual(len(base), BASE_STATE_KEY_COUNT)
        self.assertEqual(extension, PSBFR_STATE_KEYS)
        self.assertEqual(len(extension), PSBFR_STATE_KEY_COUNT)
        self.assertEqual(len(state), FORMAL_STATE_KEY_COUNT)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            FORMAL_PARAMETER_COUNT,
        )
        refinement = getattr(self.model, PSBFR_MODULE_NAME)
        self.assertEqual(
            sum(parameter.numel() for parameter in refinement.parameters()),
            PSBFR_PARAMETER_COUNT,
        )
        manifest = validate_irstd_psbfr_v1(
            self.model,
            require_identity_initialization=True,
        )
        self.assertEqual(manifest["schema"], ARCHITECTURE_SCHEMA)
        self.assertEqual(manifest["integration"], "registered_outc_forward_hook")
        self.assertEqual(manifest["base_state_key_count"], 564)
        self.assertEqual(manifest["psbfr_state_key_count"], 5)
        self.assertEqual(manifest["total_state_key_count"], 569)
        self.assertEqual(manifest["base_parameter_count"], BASE_PARAMETER_COUNT)
        self.assertEqual(manifest["psbfr_parameter_count"], 1_073)
        self.assertEqual(manifest["architecture_seed"], 42)
        self.assertEqual(
            manifest["initialization_seed"], PSBFR_INITIALIZATION_SEED
        )
        component = manifest["component_manifest"]
        self.assertEqual(component["architecture_seed"], 42)
        self.assertEqual(
            component["initialization_seed"], PSBFR_INITIALIZATION_SEED
        )
        self.assertTrue(component["feature_evidence_stop_gradient"])
        self.assertFalse(component["support_is_corrector_input"])
        self.assertEqual(component["corrector_groupnorm_groups"], 1)
        self.assertEqual(component["max_logit_delta"], 2.0)
        self.assertTrue(self.metadata["full_model_scratch_training"])
        self.assertFalse(self.metadata["warm_start_used"])
        self.assertTrue(self.metadata["all_base_parameters_trainable"])
        self.assertTrue(self.metadata["all_extension_parameters_trainable"])

    def test_strict_state_load_and_missing_extension_state_are_rejected(self) -> None:
        state = OrderedDict(
            (key, value.detach().clone())
            for key, value in self.model.state_dict().items()
        )
        self.model.load_state_dict(state, strict=True)
        missing = OrderedDict(state)
        missing.pop(PSBFR_STATE_KEYS[0])
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.model.load_state_dict(missing, strict=True)

    def test_validator_rejects_hook_tampering_freezing_and_nonidentity(self) -> None:
        hook_id = getattr(self.model, "_irstd_psbfr_v1_hook_id")
        hook = self.model.outc._forward_hooks.pop(hook_id)
        try:
            with self.assertRaisesRegex(RuntimeError, "exactly the formal"):
                validate_irstd_psbfr_v1(self.model)
        finally:
            self.model.outc._forward_hooks[hook_id] = hook

        base_name, base_parameter = next(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if not name.startswith(PSBFR_STATE_PREFIX)
        )
        base_parameter.requires_grad_(False)
        try:
            with self.assertRaisesRegex(RuntimeError, "frozen parameters"):
                validate_irstd_psbfr_v1(self.model)
        finally:
            base_parameter.requires_grad_(True)
        self.assertTrue(base_name)

        refinement = getattr(self.model, PSBFR_MODULE_NAME)
        with torch.no_grad():
            refinement.terminal.bias.fill_(0.1)
        try:
            with self.assertRaisesRegex(RuntimeError, "identity initialization"):
                validate_irstd_psbfr_v1(
                    self.model,
                    require_identity_initialization=True,
                )
            validate_irstd_psbfr_v1(
                self.model,
                require_identity_initialization=False,
            )
        finally:
            refinement.reset_identity()

        architecture_seed = getattr(
            self.model, "_irstd_psbfr_v1_architecture_seed"
        )
        setattr(self.model, "_irstd_psbfr_v1_architecture_seed", 43)
        try:
            with self.assertRaisesRegex(RuntimeError, "seed binding"):
                validate_irstd_psbfr_v1(self.model)
        finally:
            setattr(
                self.model,
                "_irstd_psbfr_v1_architecture_seed",
                architecture_seed,
            )

    def test_duplicate_nonformal_dataset_and_frozen_base_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            install_irstd_psbfr_v1(self.model)
        with self.assertRaisesRegex(ValueError, "supports only"):
            build_irstd_psbfr_v1("NUDT-SIRST")
        base, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
        first_parameter = next(base.parameters())
        first_parameter.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "every base parameter"):
            install_irstd_psbfr_v1(base)

    def test_install_preserves_rng_base_state_full_output_and_first_base_gradient(self) -> None:
        clean, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
        variant, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
        clean_state = clean.state_dict()
        variant_state = variant.state_dict()
        self.assertEqual(tuple(clean_state), tuple(variant_state))
        for key in clean_state:
            self.assertTrue(torch.equal(clean_state[key], variant_state[key]), key)

        torch.manual_seed(91_337)
        caller_rng = torch.get_rng_state().clone()
        before_install = {
            key: value.detach().clone() for key, value in variant_state.items()
        }
        install_irstd_psbfr_v1(variant)
        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng))
        installed = variant.state_dict()
        for key, value in before_install.items():
            self.assertTrue(torch.equal(installed[key], value), key)

        clean.train()
        clean.mode = "train"
        variant.train()
        variant.mode = "train"
        torch.manual_seed(1_271)
        image = torch.randn(1, 1, 256, 256)
        forward_rng = torch.get_rng_state().clone()

        clean_output = clean(image)
        self.assertIsInstance(clean_output, tuple)
        clean_loss = sum(probability.float().mean() for probability in clean_output)
        clean_loss.backward()
        detached_clean_output = tuple(value.detach().clone() for value in clean_output)
        clean_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in clean.named_parameters()
        }
        del clean_output, clean_loss

        torch.set_rng_state(forward_rng)
        variant_output = variant(image)
        self.assertIsInstance(variant_output, tuple)
        self.assertEqual(len(variant_output), 6)
        for clean_probability, variant_probability in zip(
            detached_clean_output, variant_output
        ):
            self.assertTrue(torch.equal(clean_probability, variant_probability))
        variant_loss = sum(
            probability.float().mean() for probability in variant_output
        )
        variant_loss.backward()

        for name, parameter in variant.named_parameters():
            if name.startswith(PSBFR_STATE_PREFIX):
                continue
            expected = clean_gradients[name]
            if expected is None:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.equal(parameter.grad, expected), name)
        refinement = getattr(variant, PSBFR_MODULE_NAME)
        self.assertGreater(float(refinement.terminal.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(refinement.terminal.bias.grad.abs().sum()), 0.0)

    def test_extension_initialization_tracks_architecture_seed_reproducibly(self) -> None:
        same_seed_model, same_metadata = build_irstd_psbfr_v1(
            "IRSTD-1K",
            seed=42,
            training=True,
        )
        # The currently frozen public base constructor itself admits only 42.
        # Vary the extension-owned architecture seed through the installer so
        # this test isolates the PSBFR substream contract.
        different_base, _ = initialize_evisirst(
            "IRSTD-1K", seed=42, training=True
        )
        torch.manual_seed(820_431)
        caller_rng = torch.get_rng_state().clone()
        different_seed_model, different_metadata = install_irstd_psbfr_v1(
            different_base,
            architecture_seed=43,
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), caller_rng))

        reference_extension = {
            key: value
            for key, value in self.model.state_dict().items()
            if key.startswith(PSBFR_STATE_PREFIX)
        }
        repeated_extension = {
            key: value
            for key, value in same_seed_model.state_dict().items()
            if key.startswith(PSBFR_STATE_PREFIX)
        }
        different_extension = {
            key: value
            for key, value in different_seed_model.state_dict().items()
            if key.startswith(PSBFR_STATE_PREFIX)
        }
        self.assertEqual(tuple(reference_extension), tuple(repeated_extension))
        for key in reference_extension:
            self.assertTrue(
                torch.equal(reference_extension[key], repeated_extension[key]),
                key,
            )
        self.assertFalse(
            torch.equal(
                reference_extension[
                    f"{PSBFR_STATE_PREFIX}corrector.0.weight"
                ],
                different_extension[
                    f"{PSBFR_STATE_PREFIX}corrector.0.weight"
                ],
            )
        )
        for suffix in ("corrector.3.weight", "corrector.3.bias"):
            key = f"{PSBFR_STATE_PREFIX}{suffix}"
            self.assertEqual(torch.count_nonzero(reference_extension[key]).item(), 0)
            self.assertEqual(torch.count_nonzero(different_extension[key]).item(), 0)

        self.assertEqual(same_metadata["architecture_seed"], 42)
        self.assertEqual(
            same_metadata["initialization_seed"],
            derive_psbfr_initialization_seed(42),
        )
        self.assertEqual(different_metadata["architecture_seed"], 43)
        self.assertEqual(
            different_metadata["initialization_seed"],
            derive_psbfr_initialization_seed(43),
        )
        different_manifest = validate_irstd_psbfr_v1(
            different_seed_model,
            require_identity_initialization=True,
        )
        self.assertEqual(different_manifest["architecture_seed"], 43)
        self.assertEqual(
            different_manifest["initialization_seed"],
            derive_psbfr_initialization_seed(43),
        )


if __name__ == "__main__":
    unittest.main()
