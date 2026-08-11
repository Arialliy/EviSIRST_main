from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

import train_sirst3_test_selected as selected


class _FractionalMaskDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        image = torch.zeros(1, 1, 2, dtype=torch.float32)
        target = torch.tensor([[[1.0, 128.0 / 255.0]]], dtype=torch.float32)
        return image, target, (1, 2), "fractional"


class _TwoPositiveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mode = "test"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.full_like(image, 0.9)


class SIRST3TestSelectedTests(unittest.TestCase):
    def test_official_histogram_semantics_are_not_common_binary_iou(self) -> None:
        metrics = selected.evaluate_official_selection_miou(
            _TwoPositiveModel(),
            DataLoader(_FractionalMaskDataset(), batch_size=1, shuffle=False),
            torch.device("cpu"),
        )
        self.assertEqual(metrics["intersection"], 1)
        self.assertEqual(metrics["union"], 3)
        self.assertEqual(metrics["sample_count"], 1)
        self.assertEqual(metrics["miou"], 1.0 / 3.0)

    def test_selection_schedule_matches_official_epochs(self) -> None:
        selected_epochs = [
            epoch
            for epoch in range(1, 1001)
            if selected._selection_due(epoch, 500, 1)
        ]
        self.assertEqual(selected_epochs[0], 500)
        self.assertEqual(selected_epochs[-1], 1000)
        self.assertEqual(len(selected_epochs), 501)

    def test_history_uses_strict_earliest_tie(self) -> None:
        history = [
            {"epoch": 2, "miou": 0.5, "intersection": 1, "union": 2, "sample_count": 1},
            {"epoch": 3, "miou": 0.5, "intersection": 2, "union": 4, "sample_count": 1},
            {"epoch": 4, "miou": 0.6, "intersection": 3, "union": 5, "sample_count": 1},
        ]
        score, epoch = selected._validate_history(
            history, completed_epoch=4, begin=2, every=1
        )
        self.assertEqual(score, 0.6)
        self.assertEqual(epoch, 4)

    def test_nonformal_override_requires_smoke(self) -> None:
        with self.assertRaises(SystemExit):
            selected.parse_args(
                [
                    "--dataset-root",
                    "/tmp",
                    "--output-root",
                    "/tmp/out",
                    "--epochs",
                    "2",
                    "--warmup-epochs",
                    "0",
                    "--selection-begin-epoch",
                    "1",
                ]
            )
        args = selected.parse_args(
            [
                "--dataset-root",
                "/tmp",
                "--output-root",
                "/tmp/out",
                "--epochs",
                "2",
                "--warmup-epochs",
                "0",
                "--selection-begin-epoch",
                "1",
                "--smoke",
                "--max-train-samples",
                "1",
                "--max-selection-images",
                "1",
            ]
        )
        self.assertTrue(args.smoke)

    def test_checkpoint_metadata_separates_run_and_checkpoint_selection(self) -> None:
        state = {"weight": torch.ones(1)}
        endpoint = selected._slim_checkpoint(
            state=state,
            epoch=1000,
            role="final",
            config={},
            train_index_order_sha256="0" * 64,
            selection_score=None,
        )
        self.assertTrue(endpoint["run_used_test_for_selection"])
        self.assertTrue(endpoint["run_selection_is_optimistic"])
        self.assertFalse(endpoint["this_checkpoint_selected_by_test"])
        self.assertFalse(endpoint["test_selected"])


if __name__ == "__main__":
    unittest.main()
