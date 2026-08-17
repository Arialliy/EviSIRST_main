from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from experiments import evisirst_head_diagnostic_protocol as protocol


@dataclass(frozen=True)
class FixtureFixedHeadSpec:
    """Dependency-free structural stand-in for the model utility spec."""

    head: str
    alpha: float | None = None
    probability_epsilon: float = 1e-6


def checkpoint_provenance(*, optimistic: bool = True) -> dict[str, object]:
    return {
        "checkpoint_path": "artifacts/checkpoints/EviSIRST.pth.tar",
        "checkpoint_sha256": "a" * 64,
        "source_selection": "historical_test_best_miou",
        "selection_is_optimistic": optimistic,
    }


def result_payload() -> dict[str, object]:
    return {
        "dataset": "Fixture-SIRST",
        "metrics": {"mIoU": 0.8, "Fa": 1.0e-6, "Pd": 0.9},
        "prediction_artifact_sha256": "b" * 64,
    }


class HistoricalTestDiagnosticProtocolTest(unittest.TestCase):
    def test_historical_role_allows_only_three_preregistered_fixed_heads(self) -> None:
        specs = (
            FixtureFixedHeadSpec("out"),
            FixtureFixedHeadSpec("d0"),
            FixtureFixedHeadSpec("logit_blend", alpha=0.5),
        )
        for spec in specs:
            with self.subTest(head=spec.head):
                entry = protocol.build_diagnostic_ledger_entry(
                    role=protocol.HISTORICAL_TEST_DIAGNOSTIC_ROLE,
                    head_spec=spec,
                    checkpoint_provenance=checkpoint_provenance(),
                    result=result_payload(),
                )
                self.assertTrue(entry["diagnostic_only"])
                self.assertFalse(entry["selection_allowed"])
                self.assertIsNone(entry["split_manifest_sha256"])
                self.assertEqual(entry["fixed_head_spec"]["head"], spec.head)

    def test_historical_blend_alpha_must_be_exactly_point_five(self) -> None:
        for alpha in (0.0, 0.25, 0.5000000001, 0.75, 1.0):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    protocol.HeadDiagnosticProtocolError,
                    "exactly 0.5",
                ):
                    protocol.build_diagnostic_ledger_entry(
                        role=protocol.HISTORICAL_TEST_DIAGNOSTIC_ROLE,
                        head_spec=FixtureFixedHeadSpec(
                            "logit_blend", alpha=alpha
                        ),
                        checkpoint_provenance=checkpoint_provenance(),
                        result=result_payload(),
                    )

    def test_false_optimistic_value_is_preserved_but_never_enables_selection(
        self,
    ) -> None:
        entry = protocol.build_diagnostic_ledger_entry(
            role=protocol.HISTORICAL_TEST_DIAGNOSTIC_ROLE,
            head_spec=FixtureFixedHeadSpec("out"),
            checkpoint_provenance=checkpoint_provenance(optimistic=False),
            result=result_payload(),
        )

        self.assertFalse(
            entry["checkpoint_provenance"]["selection_is_optimistic"]
        )
        self.assertTrue(entry["diagnostic_only"])
        self.assertFalse(entry["selection_allowed"])
        self.assertIn(
            "value_preserved",
            entry["checkpoint_optimistic_disclosure"],
        )

    def test_historical_role_rejects_any_selection_permission(self) -> None:
        with self.assertRaisesRegex(
            protocol.HeadDiagnosticProtocolError,
            "never allow selection",
        ):
            protocol.build_diagnostic_ledger_entry(
                role=protocol.HISTORICAL_TEST_DIAGNOSTIC_ROLE,
                head_spec=FixtureFixedHeadSpec("out"),
                checkpoint_provenance=checkpoint_provenance(),
                result=result_payload(),
                validation_selection_allowed=True,
            )


class ValidationDiagnosticProtocolTest(unittest.TestCase):
    def test_same_split_epoch_selected_checkpoint_forces_diagnostic_only(
        self,
    ) -> None:
        entry = protocol.build_diagnostic_ledger_entry(
            role=protocol.VALIDATION_ROLE,
            head_spec=FixtureFixedHeadSpec("logit_blend", alpha=0.5),
            checkpoint_provenance=checkpoint_provenance(optimistic=False),
            result=result_payload(),
            split_manifest_sha256="9" * 64,
            validation_selection_allowed=False,
            validation_checkpoint_selected_on_same_split=True,
        )

        self.assertTrue(entry["diagnostic_only"])
        self.assertFalse(entry["selection_allowed"])
        self.assertFalse(
            entry["checkpoint_provenance"]["selection_is_optimistic"]
        )
        self.assertTrue(
            entry["protocol_assertions"][
                "validation_checkpoint_selected_on_same_split"
            ]
        )
        self.assertIn(
            "no_secondary_head_selection",
            entry["checkpoint_optimistic_disclosure"],
        )

    def test_same_split_epoch_selected_checkpoint_rejects_selection(self) -> None:
        with self.assertRaisesRegex(
            protocol.HeadDiagnosticProtocolError,
            "cannot allow secondary head selection",
        ):
            protocol.build_diagnostic_ledger_entry(
                role=protocol.VALIDATION_ROLE,
                head_spec=FixtureFixedHeadSpec("out"),
                checkpoint_provenance=checkpoint_provenance(optimistic=False),
                result=result_payload(),
                split_manifest_sha256="8" * 64,
                validation_selection_allowed=True,
                validation_checkpoint_selected_on_same_split=True,
            )

    def test_validation_accepts_one_fixed_alpha_and_manifest_binding(self) -> None:
        result = result_payload()
        result["metrics"]["validation_loss"] = 0.1
        entry = protocol.build_diagnostic_ledger_entry(
            role=protocol.VALIDATION_ROLE,
            head_spec=FixtureFixedHeadSpec("logit_blend", alpha=0.25),
            checkpoint_provenance=checkpoint_provenance(optimistic=False),
            result=result,
            split_manifest_sha256="c" * 64,
            validation_selection_allowed=True,
        )

        self.assertFalse(entry["diagnostic_only"])
        self.assertTrue(entry["selection_allowed"])
        self.assertEqual(entry["split_manifest_sha256"], "c" * 64)
        self.assertEqual(entry["fixed_head_spec"]["alpha"], 0.25)
        self.assertEqual(entry["result"]["metrics"]["validation_loss"], 0.1)

    def test_validation_requires_manifest_hash_and_generic_test_role_is_absent(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            protocol.HeadDiagnosticProtocolError,
            "split_manifest_sha256",
        ):
            protocol.build_diagnostic_ledger_entry(
                role=protocol.VALIDATION_ROLE,
                head_spec=FixtureFixedHeadSpec("out"),
                checkpoint_provenance=checkpoint_provenance(optimistic=False),
                result=result_payload(),
            )

    def test_validation_selection_rejects_optimistic_checkpoint(self) -> None:
        with self.assertRaisesRegex(
            protocol.HeadDiagnosticProtocolError,
            "cannot reuse an optimistic checkpoint",
        ):
            protocol.build_diagnostic_ledger_entry(
                role=protocol.VALIDATION_ROLE,
                head_spec=FixtureFixedHeadSpec("out"),
                checkpoint_provenance=checkpoint_provenance(optimistic=True),
                result=result_payload(),
                split_manifest_sha256="e" * 64,
                validation_selection_allowed=True,
            )

    def test_optimistic_validation_without_selection_is_contaminated_diagnostic(
        self,
    ) -> None:
        entry = protocol.build_diagnostic_ledger_entry(
            role=protocol.VALIDATION_ROLE,
            head_spec=FixtureFixedHeadSpec("out"),
            checkpoint_provenance=checkpoint_provenance(optimistic=True),
            result=result_payload(),
            split_manifest_sha256="f" * 64,
            validation_selection_allowed=False,
        )

        self.assertTrue(entry["diagnostic_only"])
        self.assertFalse(entry["selection_allowed"])
        self.assertIn("contaminated", entry["checkpoint_optimistic_disclosure"])
        self.assertIn(
            "no_unbiased_evidence", entry["checkpoint_optimistic_disclosure"]
        )
        with self.assertRaisesRegex(
            protocol.HeadDiagnosticProtocolError,
            "generic test is unsupported",
        ):
            protocol.build_diagnostic_ledger_entry(
                role="test",
                head_spec=FixtureFixedHeadSpec("out"),
                checkpoint_provenance=checkpoint_provenance(),
                result=result_payload(),
            )


class LedgerIntegrityTest(unittest.TestCase):
    def test_checkpoint_provenance_is_complete_and_strictly_validated(self) -> None:
        for missing_key in (
            "checkpoint_path",
            "checkpoint_sha256",
            "source_selection",
            "selection_is_optimistic",
        ):
            with self.subTest(missing_key=missing_key):
                checkpoint = checkpoint_provenance()
                del checkpoint[missing_key]
                with self.assertRaisesRegex(
                    protocol.HeadDiagnosticProtocolError,
                    "missing required provenance keys",
                ):
                    protocol.build_diagnostic_ledger_entry(
                        role=protocol.HISTORICAL_TEST_DIAGNOSTIC_ROLE,
                        head_spec=FixtureFixedHeadSpec("d0"),
                        checkpoint_provenance=checkpoint,
                        result=result_payload(),
                    )

        invalid_checkpoints = (
            {
                **checkpoint_provenance(),
                "checkpoint_path": "/home/private/checkpoints/model.pth.tar",
            },
            {**checkpoint_provenance(), "checkpoint_path": "../model.pth.tar"},
            {
                **checkpoint_provenance(),
                "checkpoint_path": "artifacts\\model.pth.tar",
            },
            {**checkpoint_provenance(), "checkpoint_path": "C:/private/model.pth"},
            {**checkpoint_provenance(), "checkpoint_path": "~/.private/model.pth"},
            {**checkpoint_provenance(), "checkpoint_sha256": "not-a-hash"},
            {**checkpoint_provenance(), "source_selection": ""},
            {**checkpoint_provenance(), "selection_is_optimistic": 1},
        )
        for checkpoint in invalid_checkpoints:
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaises(protocol.HeadDiagnosticProtocolError):
                    protocol.build_diagnostic_ledger_entry(
                        role=protocol.HISTORICAL_TEST_DIAGNOSTIC_ROLE,
                        head_spec=FixtureFixedHeadSpec("out"),
                        checkpoint_provenance=checkpoint,
                        result=result_payload(),
                    )

    def test_ledger_is_strict_json_detached_and_recursively_immutable(self) -> None:
        checkpoint = checkpoint_provenance(optimistic=False)
        head_spec = {
            "head": "logit_blend",
            "alpha": 0.25,
            "probability_epsilon": 1e-6,
        }
        result = result_payload()
        entry = protocol.build_diagnostic_ledger_entry(
            role=protocol.VALIDATION_ROLE,
            head_spec=head_spec,
            checkpoint_provenance=checkpoint,
            result=result,
            split_manifest_sha256="d" * 64,
            validation_selection_allowed=True,
        )

        checkpoint["source_selection"] = "mutated"
        head_spec["alpha"] = 0.75
        result["metrics"]["mIoU"] = 0.0
        self.assertNotEqual(
            entry["checkpoint_provenance"]["source_selection"], "mutated"
        )
        self.assertEqual(entry["fixed_head_spec"]["alpha"], 0.25)
        self.assertEqual(entry["result"]["metrics"]["mIoU"], 0.8)

        with self.assertRaises(TypeError):
            entry["selection_allowed"] = False
        with self.assertRaises(TypeError):
            entry["result"]["metrics"]["mIoU"] = 0.0

        encoded = json.dumps(entry, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema"], protocol.LEDGER_SCHEMA)
        self.assertFalse(decoded["protocol_assertions"]["best_api_available"])
        self.assertFalse(decoded["checkpoint_provenance"]["checkpoint_path"].startswith("/"))
        self.assertNotIn("/home/", encoded)

    def test_non_json_nan_candidate_and_label_payloads_are_rejected(self) -> None:
        invalid_results = (
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"mIoU": float("nan")},
                "prediction_artifact_sha256": "b" * 64,
            },
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"mIoU": object()},
                "prediction_artifact_sha256": "b" * 64,
            },
            {"candidates": [{"alpha": 0.25}, {"alpha": 0.5}]},
            {"labels": [0, 1]},
            {"best_head": "d0"},
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"target": 1},
                "prediction_artifact_sha256": "b" * 64,
            },
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"groundTruth": 1},
                "prediction_artifact_sha256": "b" * 64,
            },
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"y_true": 1},
                "prediction_artifact_sha256": "b" * 64,
            },
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"values": [0.1, 0.2]},
                "prediction_artifact_sha256": "b" * 64,
            },
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"mIoU": [0.8]},
                "prediction_artifact_sha256": "b" * 64,
            },
            {
                "dataset": "Fixture-SIRST",
                "metrics": {"mIoU": {"value": 0.8}},
                "prediction_artifact_sha256": "b" * 64,
            },
        )
        for result in invalid_results:
            with self.subTest(result=result):
                with self.assertRaises(protocol.HeadDiagnosticProtocolError):
                    protocol.build_diagnostic_ledger_entry(
                        role=protocol.HISTORICAL_TEST_DIAGNOSTIC_ROLE,
                        head_spec=FixtureFixedHeadSpec("out"),
                        checkpoint_provenance=checkpoint_provenance(),
                        result=result,
                    )

    def test_public_api_contains_no_search_or_best_function(self) -> None:
        self.assertFalse(
            any(
                "search" in name.lower() or "best" in name.lower()
                for name in protocol.__all__
            )
        )


if __name__ == "__main__":
    unittest.main()
