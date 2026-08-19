from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from experiments.irstd_psbfr_v1 import (
    PredictionSupportedBoundedFrequencyRefinement,
)
import run_irstd_psbfr_correction_diagnostic_v1 as diagnostic


class _ToyPSBFR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.outc = nn.Conv2d(4, 1, 1)
        self.decoder_bounded_logit_refinement = (
            PredictionSupportedBoundedFrequencyRefinement(4, 3, 3, 5, 2.0)
        )
        self.outc.register_forward_hook(
            self.decoder_bounded_logit_refinement.outc_forward_hook
        )
        self.mode = "train"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.outc(image))


class PSBFRCorrectionDiagnosticTest(unittest.TestCase):
    def _batch(self):
        image = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 3:5, 3:5] = 1.0
        return [(image, mask, (8, 8), ["fixture"])]

    def test_one_pass_reports_bound_nonzero_saturation_and_signed_regions(self) -> None:
        torch.manual_seed(7)
        model = _ToyPSBFR()
        refinement = model.decoder_bounded_logit_refinement
        with torch.no_grad():
            refinement.terminal.weight.fill_(0.2)
            refinement.terminal.bias.fill_(-0.05)
        summary = diagnostic.diagnose_validation_pass(
            model, self._batch(), torch.device("cpu")
        )
        self.assertEqual(summary["validation_sample_count"], 1)
        self.assertEqual(summary["evaluated_pixel_count"], 64)
        self.assertEqual(summary["bound_violation_count"], 0)
        self.assertGreater(summary["nonzero_correction_count"], 0)
        self.assertGreaterEqual(summary["tanh_saturation_fraction"], 0.0)
        self.assertLessEqual(summary["tanh_saturation_fraction"], 1.0)
        regions = summary["signed_region_diagnostics"]
        self.assertEqual(set(regions), {"target", "boundary", "background"})
        self.assertEqual(sum(region["count"] for region in regions.values()), 64)
        for region in regions.values():
            if region["count"]:
                self.assertAlmostEqual(
                    region["positive_fraction"]
                    + region["negative_fraction"]
                    + region["zero_fraction"],
                    1.0,
                )

    def test_identity_corrector_is_detected_as_zero_but_bound_still_holds(self) -> None:
        model = _ToyPSBFR()
        summary = diagnostic.diagnose_validation_pass(
            model, self._batch(), torch.device("cpu")
        )
        self.assertEqual(summary["bound_violation_count"], 0)
        self.assertEqual(summary["nonzero_correction_count"], 0)
        self.assertEqual(summary["correction_absolute_max"], 0.0)

    def test_source_has_no_public_test_or_lockbox_dataset_dependency(self) -> None:
        source = Path(diagnostic.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        self.assertFalse(any(name == "test" for name in imported_names))
        self.assertNotIn("EviSIRSTTestDataset", source)
        self.assertNotIn("confirm_lockbox", source)
        self.assertIn('"selection_allowed": False', source)
        self.assertIn('"public_test_allowed": False', source)
        self.assertIn('"lockbox_accessed": False', source)


if __name__ == "__main__":
    unittest.main()
