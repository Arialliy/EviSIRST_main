from __future__ import annotations

import unittest

from train_independent_test_selected import (
    FORMAL_EPOCHS,
    FORMAL_SELECTION_BEGIN,
    FORMAL_SELECTION_EVERY,
    _validate_history,
    selection_due,
)


def _row(epoch: int, miou: float) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "test_loss": 0.1,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.5,
        "pixel_recall": 0.5,
        "pixel_f1": 0.5,
        "pd": 0.5,
        "tiny_pd": 0.5,
        "fa": 0.0,
        "false_objects_per_image": 0.0,
        "target_count": 2,
        "matched_target_count": 1,
        "tiny_target_count": 2,
        "matched_tiny_target_count": 1,
        "predicted_object_count": 1,
        "unmatched_predicted_object_count": 0,
        "valid_pixel_count": 16,
    }


class IndependentSelectionContractTest(unittest.TestCase):
    def test_formal_cadence_is_every_epoch_500_through_1000(self) -> None:
        epochs = [
            epoch
            for epoch in range(1, FORMAL_EPOCHS + 1)
            if selection_due(epoch, FORMAL_SELECTION_BEGIN, FORMAL_SELECTION_EVERY)
        ]
        self.assertEqual(epochs, list(range(500, 1001)))
        self.assertEqual(len(epochs), 501)

    def test_strict_miou_tie_keeps_earlier_epoch(self) -> None:
        score, epoch = _validate_history(
            [_row(500, 0.3), _row(501, 0.4), _row(502, 0.4)],
            completed_epoch=502,
            begin=500,
            every=1,
        )
        self.assertEqual(score, 0.4)
        self.assertEqual(epoch, 501)

    def test_history_cannot_skip_an_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "cadence"):
            _validate_history(
                [_row(500, 0.3), _row(502, 0.4)],
                completed_epoch=502,
                begin=500,
                every=1,
            )


if __name__ == "__main__":
    unittest.main()
