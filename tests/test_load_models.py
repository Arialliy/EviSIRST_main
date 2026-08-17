from __future__ import annotations

import gc
import unittest
from pathlib import Path

import pytest
import torch

from load_models import (
    BASELINE_CHECKPOINTS,
    DATASETS,
    EVISIRST_PACKAGES,
    load_baseline,
    load_evisirst,
)
from model.EviSIRST import EviSIRST, build_evisirst, load_pretrained


pytestmark = [pytest.mark.integration, pytest.mark.artifact]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _baseline_path(dataset: str) -> str:
    return (
        f"baseline/checkpoints/{dataset}/"
        f"{BASELINE_CHECKPOINTS[dataset]['file']}"
    )


def _require_artifacts(*relative_paths: str) -> None:
    missing = [
        relative_path
        for relative_path in relative_paths
        if not (REPOSITORY_ROOT / relative_path).exists()
    ]
    if missing:
        pytest.skip(
            "checkpoint artifact(s) are not installed: "
            + ", ".join(missing)
            + "; run tools/verify_checkpoint_artifacts.py --require after fetching them"
        )


class LoadModelsTest(unittest.TestCase):
    def test_all_designated_baselines_strict_load(self) -> None:
        _require_artifacts(*(_baseline_path(dataset) for dataset in DATASETS))
        for dataset in DATASETS:
            model, metadata = load_baseline(dataset)
            self.assertTrue(metadata["strict_load"])
            self.assertEqual(metadata["dataset"], dataset)
            self.assertEqual(metadata["checkpoint_role"], "baseline")
            self.assertIn("selection_is_optimistic", metadata)
            self.assertIn("selection_provenance", metadata)
            self.assertEqual(metadata["state_key_count"], 510)
            self.assertTrue(metadata["checkpoint_path"].endswith("SCTransNet.pth.tar"))
            self.assertFalse(model.training)
            self.assertEqual(model.mode, "test")
            del model
            gc.collect()

    def test_nudt_designated_baseline_is_epoch1000(self) -> None:
        _require_artifacts(_baseline_path("NUDT-SIRST"))
        model, metadata = load_baseline("NUDT-SIRST")
        self.assertEqual(metadata["epoch"], 1000)
        self.assertEqual(
            metadata["checkpoint_sha256"],
            "5baa4e0859060228f079e97f8ce4309a71c30f829c701f3c0b63ba0f67862172",
        )
        self.assertEqual(
            metadata["source_selection"],
            "epoch1000_user_designated_baseline",
        )
        self.assertFalse(metadata["selection_is_optimistic"])
        del model
        gc.collect()

    def test_evisirst_public_loader(self) -> None:
        _require_artifacts(EVISIRST_PACKAGES["NUAA-SIRST"]["relative_path"])
        model, metadata = load_pretrained("NUAA-SIRST")
        self.assertTrue(metadata["strict_load"])
        self.assertTrue(metadata["v3_canonical_package_validation"])
        self.assertEqual(metadata["checkpoint_role"], "final")
        self.assertEqual(metadata["source_selection"], "historical_best_miou")
        self.assertTrue(metadata["selection_is_optimistic"])
        self.assertEqual(
            metadata["selection_provenance"]["data_role"], "test"
        )
        self.assertTrue(metadata["checkpoint_path"].endswith("EviSIRST.pth.tar"))
        self.assertIsInstance(model, EviSIRST)
        self.assertFalse(model.training)
        self.assertEqual(model.mode, "test")

    def test_all_published_evisirst_paths(self) -> None:
        _require_artifacts(
            *(EVISIRST_PACKAGES[dataset]["relative_path"] for dataset in DATASETS)
        )
        expected_epochs = {
            "NUAA-SIRST": 850,
            "NUDT-SIRST": 420,
            "IRSTD-1K": 830,
        }
        for dataset in DATASETS:
            model, metadata = load_evisirst(dataset)
            self.assertEqual(metadata["epoch"], expected_epochs[dataset])
            self.assertIn(f"/results/{dataset}/EviSIRST.pth.tar", metadata["checkpoint_path"])
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
            del model
            gc.collect()

    def test_evisirst_public_builder(self) -> None:
        _require_artifacts(EVISIRST_PACKAGES["NUDT-SIRST"]["relative_path"])
        model = build_evisirst("NUDT-SIRST")
        self.assertIsInstance(model, EviSIRST)
        self.assertFalse(model.training)
        self.assertEqual(model.mode, "test")

    def test_baseline_public_forward(self) -> None:
        _require_artifacts(_baseline_path("NUAA-SIRST"))
        model, _ = load_baseline("NUAA-SIRST")
        probe = torch.linspace(-1.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
        with torch.no_grad():
            output = model(probe)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
