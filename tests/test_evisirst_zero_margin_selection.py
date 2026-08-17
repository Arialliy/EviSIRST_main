from __future__ import annotations

import json
import unittest

from experiments import evisirst_zero_margin_selection as selection


def record(
    epoch: int,
    *,
    miou: float,
    niou: float,
    pd: float,
    fa: float,
    tiny_pd: float,
    loss: float,
    data_role: str = "val",
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "data_role": data_role,
        "mIoU": miou,
        "nIoU": niou,
        "Pd": pd,
        "Fa": fa,
        "tinyPd": tiny_pd,
        "loss": loss,
    }


def runner_record(
    epoch: int,
    *,
    miou: float,
    niou: float,
    pd: float,
    fa: float,
    tiny_pd: float,
    loss: float,
) -> dict[str, object]:
    metrics = {
        "miou": miou,
        "niou": niou,
        "pd": pd,
        "fa": fa,
        "tiny_pd": tiny_pd,
        "validation_loss": loss,
    }
    return {
        "epoch": epoch,
        "data_role": "val",
        "mIoU": miou,
        "Pd": pd,
        "Fa": fa,
        "evaluation_head": "out",
        "metrics": metrics,
    }


class ZeroMarginRoleSelectionTest(unittest.TestCase):
    def test_no_miou_window_even_for_a_tiny_difference(self) -> None:
        history = [
            record(
                1,
                miou=0.8000,
                niou=0.60,
                pd=0.70,
                fa=0.10,
                tiny_pd=0.50,
                loss=0.20,
            ),
            record(
                2,
                miou=0.7999,
                niou=0.99,
                pd=0.99,
                fa=0.00,
                tiny_pd=0.99,
                loss=0.01,
            ),
        ]

        provenance = selection.select_checkpoints(history)

        self.assertEqual(provenance["primary_selected_epoch"], 1)
        self.assertEqual(provenance["selected"]["epoch"], 1)
        self.assertEqual(
            provenance["roles"]["best_mIoU"]["selected"]["epoch"], 1
        )
        self.assertEqual(
            provenance["roles"]["best_Pd"]["selected"]["epoch"], 2
        )
        self.assertIsNone(provenance["selection_margin_raw"])
        self.assertEqual(provenance["candidate_tolerance_raw"], 0.0)
        self.assertFalse(provenance["window_applied"])

    def test_best_miou_uses_the_complete_declared_key(self) -> None:
        common = {
            "miou": 0.8,
            "niou": 0.7,
            "pd": 0.9,
            "fa": 0.02,
            "tiny_pd": 0.8,
            "loss": 0.1,
        }
        history = [record(9, **common)]
        variants = (
            {"epoch": 1, "miou": 0.81},
            {"epoch": 2, "pd": 0.91},
            {"epoch": 3, "fa": 0.01},
            {"epoch": 4, "niou": 0.71},
            {"epoch": 5, "tiny_pd": 0.81},
            {"epoch": 6, "loss": 0.09},
            {"epoch": 7},
        )
        # Test each tie-break independently against epoch 9.  A better field
        # only matters after all preceding fields are exactly tied.
        field_order = ("miou", "pd", "fa", "niou", "tiny_pd", "loss")
        for index, variant in enumerate(variants):
            candidate = dict(common)
            candidate.update(variant)
            for earlier_field in field_order[:index]:
                candidate[earlier_field] = common[earlier_field]
            pair = [record(9, **common), record(**candidate)]
            selected = selection.select_checkpoints(pair)["selected"]
            with self.subTest(variant=variant):
                self.assertEqual(selected["epoch"], variant["epoch"])

        self.assertEqual(
            selection.select_checkpoints(
                [record(9, **common), record(7, **common)]
            )["selected"]["epoch"],
            7,
        )
        self.assertEqual(
            selection.select_checkpoints([record(9, **common)])["roles"]
            ["best_mIoU"]["rank_order"],
            [
                "mIoU:max",
                "Pd:max",
                "Fa:min",
                "nIoU:max",
                "tinyPd:max",
                "loss:min",
                "epoch:min",
            ],
        )

    def test_best_pd_uses_the_complete_declared_key(self) -> None:
        history = [
            record(
                8,
                miou=0.90,
                niou=0.80,
                pd=0.94,
                fa=0.02,
                tiny_pd=0.80,
                loss=0.10,
            ),
            record(
                7,
                miou=0.70,
                niou=0.70,
                pd=0.95,
                fa=0.20,
                tiny_pd=0.50,
                loss=0.20,
            ),
            record(
                6,
                miou=0.75,
                niou=0.72,
                pd=0.95,
                fa=0.01,
                tiny_pd=0.90,
                loss=0.30,
            ),
            record(
                5,
                miou=0.76,
                niou=0.73,
                pd=0.95,
                fa=0.01,
                tiny_pd=0.91,
                loss=0.30,
            ),
        ]

        provenance = selection.select_checkpoints(history)
        role = provenance["roles"]["best_Pd"]
        self.assertEqual(role["selected"]["epoch"], 5)
        self.assertFalse(role["is_primary_publication_role"])
        self.assertEqual(
            role["rank_order"],
            [
                "Pd:max",
                "Fa:min",
                "tinyPd:max",
                "mIoU:max",
                "nIoU:max",
                "loss:min",
                "epoch:min",
            ],
        )
        secondary = selection.select_best_pd_checkpoint(history)
        self.assertEqual(secondary["selected"]["epoch"], 5)
        self.assertFalse(secondary["is_primary_publication_role"])

    def test_runner_record_layout_is_supported_and_conflicts_fail_closed(self) -> None:
        compatible = runner_record(
            12,
            miou=0.71,
            niou=0.72,
            pd=0.93,
            fa=1.2e-5,
            tiny_pd=0.91,
            loss=0.002,
        )
        provenance = selection.select_checkpoints([compatible])
        evaluated = provenance["evaluated_records"][0]
        self.assertEqual(evaluated["nIoU"], 0.72)
        self.assertEqual(evaluated["tinyPd"], 0.91)
        self.assertEqual(evaluated["loss"], 0.002)
        self.assertEqual(
            evaluated["metric_sources"]["mIoU"],
            "top_level_and_metrics_verified_equal",
        )
        self.assertEqual(evaluated["metric_sources"]["nIoU"], "metrics")

        conflicting = dict(compatible)
        conflicting["metrics"] = dict(compatible["metrics"])
        conflicting["metrics"]["miou"] = 0.70
        with self.assertRaisesRegex(
            selection.EviSIRSTZeroMarginSelectionError, "conflicting"
        ):
            selection.select_checkpoints([conflicting])

    def test_provenance_is_complete_strict_json_and_validation_only(self) -> None:
        provenance = selection.select_best_miou_checkpoint(
            [
                record(
                    4,
                    miou=0.7,
                    niou=0.69,
                    pd=0.9,
                    fa=1e-5,
                    tiny_pd=0.85,
                    loss=0.01,
                )
            ],
            margin=0,
        )

        encoded = json.dumps(provenance, allow_nan=False, sort_keys=True)
        self.assertIn(selection.PROVENANCE_SCHEMA, encoded)
        self.assertIn(selection.RULE_VERSION, encoded)
        self.assertEqual(provenance["data_role"], "val")
        self.assertFalse(provenance["test_selection_supported"])
        self.assertFalse(provenance["test_split_accessed"])
        self.assertEqual(provenance["primary_role"], "best_mIoU")
        self.assertEqual(provenance["evaluated_epochs"], [4])


class ZeroMarginRetentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.history = [
            record(
                1,
                miou=0.80,
                niou=0.70,
                pd=0.90,
                fa=0.02,
                tiny_pd=0.80,
                loss=0.10,
            ),
            record(
                2,
                miou=0.79,
                niou=0.71,
                pd=0.95,
                fa=0.01,
                tiny_pd=0.90,
                loss=0.09,
            ),
            record(
                3,
                miou=0.70,
                niou=0.60,
                pd=0.80,
                fa=0.04,
                tiny_pd=0.70,
                loss=0.20,
            ),
        ]

    def test_default_retains_union_of_both_role_winners(self) -> None:
        self.assertEqual(
            selection.retention_frontier_epochs(reversed(self.history)), (1, 2)
        )

    def test_primary_only_retention_is_explicit_and_never_omits_best_miou(self) -> None:
        self.assertEqual(
            selection.retention_frontier_epochs(
                self.history, retain_roles=("best_mIoU",)
            ),
            (1,),
        )
        with self.assertRaisesRegex(
            selection.EviSIRSTZeroMarginSelectionError, "must include"
        ):
            selection.retention_frontier_epochs(
                self.history, retain_roles=("best_Pd",)
            )


class ZeroMarginInputValidationTest(unittest.TestCase):
    def _valid(self) -> dict[str, object]:
        return record(
            1,
            miou=0.8,
            niou=0.7,
            pd=0.9,
            fa=0.01,
            tiny_pd=0.8,
            loss=0.1,
        )

    def test_rejects_any_nonzero_or_malformed_margin(self) -> None:
        for margin in (0.001, -0.001, float("nan"), True, "0"):
            with self.subTest(margin=margin):
                with self.assertRaisesRegex(
                    selection.EviSIRSTZeroMarginSelectionError, "margin"
                ):
                    selection.select_checkpoints([self._valid()], margin=margin)

    def test_rejects_test_role_and_positive_test_disclosures(self) -> None:
        test_role = self._valid()
        test_role["data_role"] = "test"
        with self.assertRaisesRegex(
            selection.EviSIRSTZeroMarginSelectionError, "data_role"
        ):
            selection.select_checkpoints([test_role])

        for field in (
            "test_split_accessed",
            "test_index_opened",
            "test_selected",
            "test_selection_supported",
        ):
            disclosed = self._valid()
            disclosed[field] = True
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    selection.EviSIRSTZeroMarginSelectionError, field
                ):
                    selection.select_checkpoints([disclosed])

    def test_rejects_missing_duplicate_nonfinite_and_out_of_range_values(self) -> None:
        missing = self._valid()
        del missing["tinyPd"]
        cases = (
            [missing],
            [self._valid(), self._valid()],
            [{**self._valid(), "nIoU": float("nan")}],
            [{**self._valid(), "Pd": 95.0}],
            [{**self._valid(), "Fa": -1.0}],
            [{**self._valid(), "loss": None}],
            [{**self._valid(), "epoch": True}],
        )
        for invalid in cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(selection.EviSIRSTZeroMarginSelectionError):
                    selection.select_checkpoints(invalid)

    def test_publication_role_cannot_be_changed_to_best_pd(self) -> None:
        with self.assertRaisesRegex(
            selection.EviSIRSTZeroMarginSelectionError, "primary_role"
        ):
            selection.select_checkpoints(
                [self._valid()], primary_role="best_Pd"
            )


if __name__ == "__main__":
    unittest.main()
