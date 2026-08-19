from __future__ import annotations

import unittest
from collections import OrderedDict

import torch

from experiments.irstd_single_residual_v1 import (
    ARCHITECTURE_SCHEMA,
    FORMAL_PARAMETER_COUNT,
    FORMAL_STATE_KEY_COUNT,
    _residualized_encoder_features,
    baseline_reference_forward_with_relay,
    build_irstd_single_residual_v1,
    install_irstd_single_residual_v1,
    validate_irstd_single_residual_v1,
)
from model.EviSIRST import initialize_evisirst


class IRSTDSingleResidualV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_num_threads)

    def test_baseline_reference_copy_is_fixed_input_bitwise_equal(self) -> None:
        model, _ = initialize_evisirst("IRSTD-1K", seed=42, training=False)
        generator = torch.Generator().manual_seed(20260817)
        sample = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            frozen = model._forward_with_relay(sample)
            copied = baseline_reference_forward_with_relay(model, sample)
        self.assertTrue(torch.equal(frozen, copied))

    def test_d0_removes_exactly_one_skip_at_all_four_levels(self) -> None:
        reconstructed = tuple(
            torch.full((1, level, 2, 2), float(10 * level))
            for level in range(1, 5)
        )
        skips = tuple(
            torch.full_like(value, float(level))
            for level, value in enumerate(reconstructed, start=1)
        )
        baseline = _residualized_encoder_features(
            reconstructed,
            skips,
            residual_multiplicity=2,
        )
        d0 = _residualized_encoder_features(
            reconstructed,
            skips,
            residual_multiplicity=1,
        )
        self.assertEqual(len(d0), 4)
        for level in range(4):
            self.assertTrue(torch.equal(baseline[level] - d0[level], skips[level]))

    def test_install_preserves_564_keys_and_parameter_count_and_manifest(self) -> None:
        base, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
        original_keys = tuple(base.state_dict())
        model, metadata = install_irstd_single_residual_v1(base)
        self.assertEqual(tuple(model.state_dict()), original_keys)
        self.assertEqual(len(model.state_dict()), FORMAL_STATE_KEY_COUNT)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            FORMAL_PARAMETER_COUNT,
        )
        manifest = validate_irstd_single_residual_v1(model)
        self.assertEqual(manifest["schema"], ARCHITECTURE_SCHEMA)
        self.assertEqual(manifest["encoder_levels_changed"], (1, 2, 3, 4))
        self.assertEqual(manifest["new_parameters"], 0)
        self.assertEqual(manifest["new_state_keys"], 0)
        self.assertEqual(metadata["architecture_manifest"], manifest)

    def test_train_eval_api_finite_outputs_and_gradients(self) -> None:
        model, metadata = build_irstd_single_residual_v1(training=True)
        sample = torch.randn(2, 1, 32, 32)
        outputs = model(sample)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(torch.isfinite(value).all().item() for value in outputs))
        sum(value.mean() for value in outputs).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all().item() for value in gradients))
        self.assertTrue(any(torch.count_nonzero(value).item() for value in gradients))
        self.assertTrue(metadata["full_model_scratch_training"])

        model.mode = "test"
        model.eval()
        with torch.no_grad():
            evaluated = model(sample[:1])
        self.assertIsInstance(evaluated, torch.Tensor)
        self.assertEqual(tuple(evaluated.shape), (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(evaluated).all().item())

    def test_clean_state_strict_loads_and_missing_key_is_rejected(self) -> None:
        clean, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
        state = OrderedDict(
            (key, value.detach().clone()) for key, value in clean.state_dict().items()
        )
        d0, _ = build_irstd_single_residual_v1(training=True)
        incompatible = d0.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        missing = OrderedDict(state)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            d0.load_state_dict(missing, strict=True)

    def test_duplicate_install_nonformal_dataset_and_binding_tamper_rejected(self) -> None:
        model, _ = build_irstd_single_residual_v1(training=True)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            install_irstd_single_residual_v1(model)
        with self.assertRaisesRegex(ValueError, "supports only"):
            build_irstd_single_residual_v1("NUDT-SIRST")

        original = model._forward_with_relay
        model._forward_with_relay = baseline_reference_forward_with_relay.__get__(
            model, type(model)
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "binding differs"):
                validate_irstd_single_residual_v1(model)
        finally:
            model._forward_with_relay = original


if __name__ == "__main__":
    unittest.main()
