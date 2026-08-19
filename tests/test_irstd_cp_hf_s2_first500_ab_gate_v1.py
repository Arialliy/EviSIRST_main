from __future__ import annotations

import unittest
from types import SimpleNamespace

import run_irstd_cp_hf_s2_first500_ab_gate_v1 as gate


def _history(offset: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for epoch in range(1, 501):
        miou = 0.60 + offset + epoch * 1.0e-6
        pd = 0.90
        fa = 1.0e-5
        loss = 1.0 / epoch
        records.append(
            {
                "epoch": epoch,
                "data_role": "val",
                "evaluation_head": "out",
                "mIoU": miou,
                "nIoU": miou - 0.01,
                "Pd": pd,
                "Fa": fa,
                "tinyPd": 0.80,
                "loss": loss,
                "metrics": {
                    "miou": miou,
                    "niou": miou - 0.01,
                    "pd": pd,
                    "fa": fa,
                    "tiny_pd": 0.80,
                    "validation_loss": loss,
                },
                "test_split_accessed": False,
            }
        )
    return records


class CPHFS2First500GateTest(unittest.TestCase):
    def _seed_map(self, offset: float):
        return {seed: _history(offset) for seed in gate.RUN_SEEDS}

    def test_two_candidates_are_decided_independently_without_winner(self) -> None:
        payload = gate.evaluate_payload(
            clean_histories=self._seed_map(0.0),
            candidate_histories={
                "psbfr_v1": self._seed_map(0.003),
                "cp_hf_s2_v1": self._seed_map(-0.001),
            },
            mechanism_checks={
                "psbfr_v1": {
                    "bound_holds": True,
                    "corrector_nonzero": True,
                    "saturation_below_limit": True,
                },
                "cp_hf_s2_v1": {
                    "adapter_executed": True,
                    "adapter_departed_identity": True,
                    "finite": True,
                },
            },
        )
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(
            list(payload["decisions"]), ["cp_hf_s2_v1", "psbfr_v1"]
        )
        self.assertEqual(payload["decisions"]["psbfr_v1"]["result"], "GO")
        self.assertEqual(payload["decisions"]["cp_hf_s2_v1"]["result"], "STOP")
        self.assertFalse(payload["candidate_ranking_performed"])
        self.assertIsNone(payload["winner_selected"])
        self.assertFalse(payload["stability_claim_allowed"])
        self.assertFalse(payload["lockbox_accessed"])
        self.assertFalse(payload["test_selected"])
        self.assertFalse(payload["test_selection_supported"])

    def test_additional_or_missing_candidate_is_invalid_not_selected(self) -> None:
        payload = gate.evaluate_payload(
            clean_histories=self._seed_map(0.0),
            candidate_histories={"cp_hf_s2_v1": self._seed_map(0.01)},
            mechanism_checks={},
        )
        self.assertEqual(payload["status"], "INCOMPLETE")
        self.assertFalse(payload["candidate_ranking_performed"])
        self.assertFalse(payload["immutable_result_written"])

    def test_first500_is_contiguous_zero_margin_and_ignores_later_records(self) -> None:
        records = _history(0.0)
        records.append(
            {
                **records[-1],
                "epoch": 501,
                "mIoU": 0.999,
                "metrics": {**records[-1]["metrics"], "miou": 0.999},
            }
        )
        selected = gate.select_first500(records)
        self.assertEqual(selected["records_used"], 500)
        self.assertFalse(selected["records_after_epoch_500_used"])
        self.assertEqual(
            selected["selection_provenance"]["rule_version"],
            "evisirst_zero_margin_dual_role_lexicographic/v1",
        )
        broken = _history(0.0)
        broken[10]["epoch"] = 99
        with self.assertRaisesRegex(Exception, "continuous"):
            gate.select_first500(broken)

    def test_all_three_seeds_are_required(self) -> None:
        incomplete = self._seed_map(0.0)
        incomplete.pop(42)
        with self.assertRaises(gate.CPHFS2First500GateNotReady):
            gate.candidate_decision(
                candidate="cp_hf_s2_v1",
                clean_histories=incomplete,
                candidate_histories=self._seed_map(0.003),
                mechanism_checks={
                    "adapter_executed": True,
                    "adapter_departed_identity": True,
                    "finite": True,
                },
            )

    def test_gate_has_no_path_seed_or_threshold_cli(self) -> None:
        gate.parse_args([])
        for option in ("--seed", "42", "--threshold", "0", "--candidate", "x"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                gate.parse_args([option])

    def test_read_only_cp_contract_accepts_no_training_args(self) -> None:
        fixture = SimpleNamespace(
            VARIANT_ROOTS={},
            PROTOCOL_RELATIVE_PATH="original-protocol",
            RULES_RELATIVE_PATH="original-rules",
        )
        with gate._cp_reader_contract(fixture):
            self.assertIn("cp_hf_s2_v1", fixture.VARIANT_ROOTS)
        self.assertEqual(fixture.VARIANT_ROOTS, {})
        self.assertEqual(fixture.PROTOCOL_RELATIVE_PATH, "original-protocol")
        self.assertEqual(fixture.RULES_RELATIVE_PATH, "original-rules")

    def test_combined_s2_run_and_writer_are_sealed(self) -> None:
        self.assertFalse(gate.S2_ROUTE_LEDGER_EXECUTION_SEALED)
        with self.assertRaises(gate.CPHFS2First500GateNotReady):
            gate.run()
        with self.assertRaises(gate.CPHFS2First500GateNotReady):
            gate.write_complete_payload(
                {"schema": gate.RESULT_SCHEMA, "status": "complete"}
            )

    def test_generic_clean_source_hash_tamper_is_rejected(self) -> None:
        identity = {
            "determinism_protocol": {
                "source_files": {
                    "fixture": {
                        "relative_path": "experiments/evisirst_zero_margin_selection.py",
                        "sha256": "0" * 64,
                    }
                },
                "source_tree_sha256": "0" * 64,
            }
        }
        with self.assertRaisesRegex(Exception, "source hash differs"):
            gate._validate_identity_source_tree(identity, role="clean")


if __name__ == "__main__":
    unittest.main()
