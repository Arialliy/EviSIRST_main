from __future__ import annotations

import copy
import json
import unittest

from experiments import evisirst_v2_selection as selection


def independent_record(
    epoch: int,
    miou: float,
    fa: float,
    pd: float,
    *,
    data_role: str = "val",
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "data_role": data_role,
        "mIoU": miou,
        "Fa": fa,
        "Pd": pd,
    }


def joint_record(
    epoch: int,
    domain_mious: tuple[float, float, float],
    *,
    macro_fa: float,
    macro_pd: float,
    data_role: str = "val",
) -> dict[str, object]:
    domains = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
    return {
        "epoch": epoch,
        "data_role": data_role,
        "domains": {
            name: {"mIoU": miou, "Fa": macro_fa, "Pd": macro_pd}
            for name, miou in zip(domains, domain_mious)
        },
    }


class IndependentCheckpointSelectionTest(unittest.TestCase):
    def test_full_lexicographic_rule_and_inclusive_candidate_window(self) -> None:
        records = [
            independent_record(1, 0.8000, 0.005, 0.90),
            independent_record(2, 0.7992, 0.004, 0.88),
            independent_record(3, 0.7995, 0.004, 0.91),
            independent_record(4, 0.7989, 0.001, 0.99),
            independent_record(5, 0.7995, 0.004, 0.91),
            independent_record(6, 0.7990, 0.010, 0.50),
        ]

        provenance = selection.select_independent_checkpoint(records)

        self.assertEqual(provenance["selected"]["epoch"], 3)
        self.assertEqual(
            [candidate["epoch"] for candidate in provenance["candidates"]],
            [1, 2, 3, 5, 6],
        )
        self.assertEqual(
            provenance["selection_reason"]["lexicographic_order"],
            [
                "best_mIoU_then_raw_0.001_window",
                "Fa:min",
                "Pd:max",
                "epoch:min",
            ],
        )
        trace = provenance["selection_reason"]["decision_trace"]
        self.assertEqual(trace[2]["surviving_epochs"], [2, 3, 5])
        self.assertEqual(trace[3]["surviving_epochs"], [3, 5])
        self.assertEqual(trace[4]["surviving_epochs"], [3])

    def test_provenance_is_strict_json_and_explicitly_validation_only(self) -> None:
        provenance = selection.select_independent_checkpoint(
            [independent_record(7, 0.75, 3.0e-6, 0.92)]
        )

        encoded = json.dumps(provenance, allow_nan=False, sort_keys=True)
        self.assertIn(selection.INDEPENDENT_RULE_VERSION, encoded)
        self.assertEqual(provenance["data_role"], "val")
        self.assertFalse(provenance["test_selection_supported"])
        self.assertIn("candidates", provenance)
        self.assertIn("selection_reason", provenance)

    def test_rejects_test_role_duplicate_epoch_nan_and_missing_metrics(self) -> None:
        invalid_cases = {
            "test role": [independent_record(1, 0.8, 0.01, 0.9, data_role="test")],
            "duplicate epoch": [
                independent_record(1, 0.8, 0.01, 0.9),
                independent_record(1, 0.7, 0.02, 0.8),
            ],
            "NaN": [independent_record(1, float("nan"), 0.01, 0.9)],
            "missing": [
                {"epoch": 1, "data_role": "val", "mIoU": 0.8, "Pd": 0.9}
            ],
        }
        expected_messages = {
            "test role": "data_role",
            "duplicate epoch": "duplicate epoch",
            "NaN": "finite",
            "missing": "missing required keys",
        }
        for case, records in invalid_cases.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    selection.EviSIRSTSelectionError,
                    expected_messages[case],
                ):
                    selection.select_independent_checkpoint(records)

    def test_rejects_percent_scale_miou(self) -> None:
        with self.assertRaisesRegex(
            selection.EviSIRSTSelectionError,
            "raw proportion",
        ):
            selection.select_independent_checkpoint(
                [independent_record(1, 80.0, 0.01, 0.9)]
            )

    def test_retention_frontier_removes_only_dominated_epochs(self) -> None:
        history = [
            independent_record(1, 0.700, 0.050, 0.80),
            independent_record(2, 0.690, 0.060, 0.75),
            independent_record(3, 0.720, 0.070, 0.90),
            independent_record(4, 0.710, 0.040, 0.70),
            independent_record(5, 0.720, 0.070, 0.90),
            independent_record(6, 0.680, 0.030, 0.50),
        ]

        frontier = selection.retention_frontier_epochs(reversed(history))

        # epoch 4 dominates 1 and 2; epoch 3 wins the exact metric tie with 5.
        # Epochs 3, 4, and 6 trade mIoU against the lexicographic selection key.
        self.assertEqual(frontier, (3, 4, 6))

    def test_every_threshold_winner_is_on_retention_frontier(self) -> None:
        history = [
            independent_record(1, 0.61, 0.08, 0.80),
            independent_record(2, 0.63, 0.09, 0.95),
            independent_record(3, 0.65, 0.07, 0.70),
            independent_record(4, 0.67, 0.11, 0.99),
            independent_record(5, 0.66, 0.06, 0.75),
            independent_record(6, 0.64, 0.05, 0.60),
            independent_record(7, 0.67, 0.11, 0.99),
            independent_record(8, 0.60, 0.04, 0.50),
        ]
        frontier = set(selection.retention_frontier_epochs(history))
        miou_levels = sorted({float(record["mIoU"]) for record in history})

        # Eligibility changes only at an observed mIoU.  Exact boundaries and
        # one midpoint between each pair therefore enumerate every distinct
        # non-empty candidate set induced by a real-valued threshold.
        thresholds = [miou_levels[0]]
        thresholds.extend(miou_levels)
        thresholds.extend(
            (lower + upper) / 2.0
            for lower, upper in zip(miou_levels, miou_levels[1:])
        )
        for threshold in sorted(set(thresholds)):
            eligible = [
                record for record in history if float(record["mIoU"]) >= threshold
            ]
            winner = min(
                eligible,
                key=lambda record: (
                    float(record["Fa"]),
                    -float(record["Pd"]),
                    int(record["epoch"]),
                ),
            )
            with self.subTest(threshold=threshold, winner=winner["epoch"]):
                self.assertIn(winner["epoch"], frontier)

    def test_retention_frontier_reuses_strict_validation_history_contract(self) -> None:
        invalid_histories = [
            [independent_record(1, 0.8, 0.01, 0.9, data_role="test")],
            [
                independent_record(1, 0.8, 0.01, 0.9),
                independent_record(1, 0.7, 0.02, 0.8),
            ],
            [independent_record(1, 0.8, float("nan"), 0.9)],
            [{"epoch": 1, "data_role": "val", "mIoU": 0.8, "Fa": 0.01}],
        ]
        for history in invalid_histories:
            with self.subTest(history=history):
                with self.assertRaises(selection.EviSIRSTSelectionError):
                    selection.retention_frontier_epochs(history)


class JointCheckpointSelectionTest(unittest.TestCase):
    def test_joint_rule_derives_equal_domain_worst_and_macro_metrics(self) -> None:
        records = [
            joint_record(10, (0.7000, 0.8500, 0.8500), macro_fa=0.05, macro_pd=0.90),
            joint_record(11, (0.6995, 0.8800, 0.8800), macro_fa=0.04, macro_pd=0.90),
            joint_record(12, (0.6995, 0.8800, 0.8800), macro_fa=0.03, macro_pd=0.90),
            joint_record(13, (0.6995, 0.8800, 0.8800), macro_fa=0.03, macro_pd=0.95),
            joint_record(14, (0.6995, 0.8800, 0.8800), macro_fa=0.03, macro_pd=0.95),
            joint_record(15, (0.6988, 0.9900, 0.9900), macro_fa=0.00, macro_pd=1.00),
        ]

        provenance = selection.select_joint_checkpoint(records)

        self.assertEqual(provenance["selected"]["epoch"], 13)
        self.assertEqual(
            [candidate["epoch"] for candidate in provenance["candidates"]],
            [10, 11, 12, 13, 14],
        )
        self.assertAlmostEqual(
            provenance["selected"]["worst_domain_mIoU"], 0.6995
        )
        self.assertAlmostEqual(
            provenance["selected"]["macro_mIoU"],
            (0.6995 + 0.88 + 0.88) / 3.0,
        )
        self.assertEqual(provenance["domain_weighting"], "equal_domain_macro")
        self.assertEqual(
            provenance["selection_reason"]["decision_trace"][-1][
                "surviving_epochs"
            ],
            [13],
        )

    def test_joint_rejects_test_role_missing_domain_metric_and_domain_drift(self) -> None:
        test_role = joint_record(
            1,
            (0.7, 0.8, 0.9),
            macro_fa=0.01,
            macro_pd=0.9,
            data_role="test",
        )
        with self.assertRaisesRegex(selection.EviSIRSTSelectionError, "data_role"):
            selection.select_joint_checkpoint([test_role])

        missing = joint_record(
            1, (0.7, 0.8, 0.9), macro_fa=0.01, macro_pd=0.9
        )
        del missing["domains"]["IRSTD-1K"]["Fa"]
        with self.assertRaisesRegex(
            selection.EviSIRSTSelectionError,
            "missing required keys",
        ):
            selection.select_joint_checkpoint([missing])

        first = joint_record(
            1, (0.7, 0.8, 0.9), macro_fa=0.01, macro_pd=0.9
        )
        second = joint_record(
            2, (0.7, 0.8, 0.9), macro_fa=0.01, macro_pd=0.9
        )
        del second["domains"]["IRSTD-1K"]
        with self.assertRaisesRegex(selection.EviSIRSTSelectionError, "domain set"):
            selection.select_joint_checkpoint([first, second])


class ValidationSpecRegistryTest(unittest.TestCase):
    def _provenance(self) -> dict[str, object]:
        return selection.select_independent_checkpoint(
            [independent_record(9, 0.8, 0.01, 0.9)]
        )

    def _fixed_spec(
        self,
        value: dict[str, object],
        *,
        data_role: str = "val",
        selection_status: str = selection.FIXED_SELECTION_STATUS,
    ) -> dict[str, object]:
        return {
            "data_role": data_role,
            "selection_status": selection_status,
            "value": value,
            "selection_provenance": self._provenance(),
        }

    def test_registry_accepts_only_detached_fixed_validation_specs(self) -> None:
        registry = selection.ValidationSpecRegistry()
        head_spec = self._fixed_spec({"head": "logit_blend", "alpha": 0.25})

        registered = registry.register(
            registry_key="primary_head",
            spec_kind="head",
            fixed_spec=head_spec,
        )
        head_spec["value"]["alpha"] = 0.75
        registered["value"]["alpha"] = 1.0
        registry.register(
            registry_key="probability_calibration",
            spec_kind="calibration",
            fixed_spec=self._fixed_spec({"temperature": 1.1, "bias": -0.05}),
        )
        snapshot = registry.snapshot()

        self.assertEqual(len(registry), 2)
        self.assertEqual(snapshot["entries"][0]["value"]["alpha"], 0.25)
        self.assertEqual(snapshot["data_role"], "val")
        self.assertFalse(snapshot["test_selection_supported"])
        json.dumps(snapshot, allow_nan=False, sort_keys=True)

    def test_registry_rejects_nonvalidation_unfixed_duplicate_and_nan_specs(self) -> None:
        invalid_specs = {
            "test": self._fixed_spec({"head": "out"}, data_role="test"),
            "unfixed": self._fixed_spec(
                {"head": "out"}, selection_status="candidate"
            ),
            "NaN": self._fixed_spec({"temperature": float("nan")}),
        }
        for case, fixed_spec in invalid_specs.items():
            with self.subTest(case=case):
                registry = selection.ValidationSpecRegistry()
                with self.assertRaises(selection.EviSIRSTSelectionError):
                    registry.register(
                        registry_key="fixed",
                        spec_kind="head" if case != "NaN" else "calibration",
                        fixed_spec=fixed_spec,
                    )

        registry = selection.ValidationSpecRegistry()
        fixed = self._fixed_spec({"head": "out"})
        registry.register(
            registry_key="primary_head",
            spec_kind="head",
            fixed_spec=fixed,
        )
        with self.assertRaisesRegex(
            selection.EviSIRSTSelectionError,
            "duplicate registry_key",
        ):
            registry.register(
                registry_key="primary_head",
                spec_kind="head",
                fixed_spec=copy.deepcopy(fixed),
            )

    def test_registry_rejects_test_selected_provenance(self) -> None:
        fixed = self._fixed_spec({"head": "d0"})
        fixed["selection_provenance"]["data_role"] = "test"
        with self.assertRaisesRegex(selection.EviSIRSTSelectionError, "data_role"):
            selection.ValidationSpecRegistry().register(
                registry_key="primary_head",
                spec_kind="head",
                fixed_spec=fixed,
            )


if __name__ == "__main__":
    unittest.main()
