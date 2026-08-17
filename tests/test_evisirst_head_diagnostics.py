from __future__ import annotations

import unittest

import torch
from torch import nn

from model.evisirst_head_diagnostics import (
    LEGACY_HEAD_ORDER,
    EviSIRSTProbabilityHeads,
    FixedHeadSpec,
    extract_probability_heads,
    fixed_logit_blend,
    probability_to_logit,
    select_fixed_head,
)


class FakeSixProbabilityModel(nn.Module):
    """Parameter-free stand-in for the frozen model's forward contract."""

    probabilities = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8)

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.child = nn.Identity()
        self.mode = "test"
        self.fail = fail
        self.observed_state = None

    def forward(self, input_tensor: torch.Tensor):
        self.observed_state = (
            self.training,
            self.child.training,
            self.mode,
            torch.is_inference_mode_enabled(),
        )
        if self.fail:
            raise RuntimeError("synthetic forward failure")
        return tuple(
            torch.full_like(input_tensor, probability)
            for probability in self.probabilities
        )


def make_probability_heads(
    *, out_probability: float = 0.2, d0_probability: float = 0.8
) -> EviSIRSTProbabilityHeads:
    tensors = tuple(torch.tensor([value]) for value in (0.1, 0.2, 0.3, 0.4))
    return EviSIRSTProbabilityHeads(
        *tensors,
        d0=torch.tensor([d0_probability]),
        out=torch.tensor([out_probability]),
    )


class EviSIRSTHeadDiagnosticsTest(unittest.TestCase):
    def test_extracts_legacy_order_and_restores_exact_mixed_state(self) -> None:
        model = FakeSixProbabilityModel()
        model.train()
        model.child.eval()
        model.mode = "original-mode"
        before_training = tuple(module.training for module in model.modules())

        heads = extract_probability_heads(model, torch.zeros(1, 1, 2, 2))

        self.assertEqual(
            model.observed_state,
            (False, False, "train", True),
        )
        self.assertEqual(
            tuple(module.training for module in model.modules()),
            before_training,
        )
        self.assertEqual(model.mode, "original-mode")
        self.assertEqual(tuple(heads.as_dict()), LEGACY_HEAD_ORDER)
        for name, expected in zip(LEGACY_HEAD_ORDER, model.probabilities):
            torch.testing.assert_close(
                getattr(heads, name),
                torch.full((1, 1, 2, 2), expected),
            )

    def test_direct_heads_are_fixed_and_reject_irrelevant_alpha(self) -> None:
        heads = make_probability_heads(out_probability=0.25, d0_probability=0.75)

        out_prediction = select_fixed_head(heads, FixedHeadSpec("out"))
        d0_prediction = select_fixed_head(heads, FixedHeadSpec("d0"))

        self.assertIs(out_prediction.probability, heads.out)
        self.assertIs(d0_prediction.probability, heads.d0)
        self.assertEqual(out_prediction.provenance["head"], "out")
        self.assertIsNone(out_prediction.provenance["alpha"])
        with self.assertRaisesRegex(ValueError, "not applicable"):
            FixedHeadSpec("out", alpha=0.5)

    def test_logit_blend_matches_explicit_math_and_records_provenance(self) -> None:
        out_probability = torch.tensor([0.2, 0.35])
        d0_probability = torch.tensor([0.8, 0.65])
        alpha = 0.25
        heads = EviSIRSTProbabilityHeads(
            out_probability,
            out_probability,
            out_probability,
            out_probability,
            d0_probability,
            out_probability,
        )

        prediction = select_fixed_head(
            heads,
            FixedHeadSpec("logit_blend", alpha=alpha),
        )
        expected = torch.sigmoid(
            (1.0 - alpha) * torch.logit(out_probability)
            + alpha * torch.logit(d0_probability)
        )

        torch.testing.assert_close(prediction.probability, expected)
        self.assertEqual(prediction.provenance["head"], "logit_blend")
        self.assertEqual(prediction.provenance["alpha"], alpha)
        self.assertEqual(
            prediction.provenance["logit_recovery"],
            "clamped_inverse_sigmoid_from_probability",
        )

    def test_probability_to_logit_clamps_saturated_probabilities(self) -> None:
        saturated = torch.tensor([0.0, 1.0], dtype=torch.float16)

        recovered = probability_to_logit(saturated)
        blended = fixed_logit_blend(saturated, saturated, alpha=0.5)

        self.assertEqual(recovered.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(recovered).all()))
        self.assertTrue(bool(torch.isfinite(blended).all()))
        self.assertTrue(bool(((blended > 0.0) & (blended < 1.0)).all()))

    def test_blend_requires_one_valid_preselected_alpha(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be supplied"):
            FixedHeadSpec("logit_blend")
        for invalid_alpha in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(alpha=invalid_alpha):
                with self.assertRaises(ValueError):
                    FixedHeadSpec("logit_blend", alpha=invalid_alpha)
        with self.assertRaises(TypeError):
            FixedHeadSpec("logit_blend", alpha=True)

    def test_forward_exception_restores_mode_and_all_training_flags(self) -> None:
        model = FakeSixProbabilityModel(fail=True)
        model.train()
        model.child.eval()
        model.mode = "before-failure"
        before_training = tuple(module.training for module in model.modules())

        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
            extract_probability_heads(model, torch.zeros(1, 1, 2, 2))

        self.assertEqual(
            tuple(module.training for module in model.modules()),
            before_training,
        )
        self.assertEqual(model.mode, "before-failure")

    def test_malformed_legacy_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly six"):
            EviSIRSTProbabilityHeads.from_legacy_output(
                tuple(torch.tensor([0.5]) for _ in range(5))
            )


if __name__ == "__main__":
    unittest.main()
