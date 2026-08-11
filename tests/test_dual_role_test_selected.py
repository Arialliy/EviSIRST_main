from __future__ import annotations

import unittest

from train_dual_role_test_selected import (
    _validate_history,
    role_key,
    selection_due,
)


def row(epoch: int, *, miou: float, pd: float, fa: float) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "test_loss": 0.1,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.5,
        "pixel_recall": 0.5,
        "pixel_f1": 0.5,
        "pd": pd,
        "tiny_pd": pd,
        "fa": fa,
        "false_objects_per_image": 0.0,
        "target_count": 2,
        "matched_target_count": 1,
        "tiny_target_count": 2,
        "matched_tiny_target_count": 1,
        "predicted_object_count": 1,
        "unmatched_predicted_object_count": 0,
        "valid_pixel_count": 16,
    }


class DualRoleSelectionTest(unittest.TestCase):
    def test_formal_cadence_has_501_epochs(self) -> None:
        epochs = [e for e in range(1, 1001) if selection_due(e, 500, 1)]
        self.assertEqual(epochs, list(range(500, 1001)))

    def test_roles_can_select_different_epochs(self) -> None:
        history = [
            row(500, miou=0.8, pd=0.9, fa=0.01),
            row(501, miou=0.7, pd=1.0, fa=0.02),
        ]
        self.assertEqual(
            _validate_history(history, completed_epoch=501, begin=500, every=1),
            {"best_miou": 500, "best_pd": 501},
        )

    def test_exact_role_key_tie_keeps_earlier_epoch(self) -> None:
        first = row(500, miou=0.8, pd=0.9, fa=0.01)
        second = row(501, miou=0.8, pd=0.9, fa=0.01)
        self.assertGreater(role_key(first, "best_miou"), role_key(second, "best_miou"))
        self.assertGreater(role_key(first, "best_pd"), role_key(second, "best_pd"))

    def test_best_miou_uses_historical_secondary_fields(self) -> None:
        first = row(500, miou=0.8, pd=0.9, fa=0.02)
        second = row(501, miou=0.8, pd=0.9, fa=0.01)
        self.assertGreater(role_key(second, "best_miou"), role_key(first, "best_miou"))


if __name__ == "__main__":
    unittest.main()
