from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import run_irstd_cp_hf_s2_stage1_route_gate_v1 as gate


class Stage1RouteGateTest(unittest.TestCase):
    def test_routes_are_independent_and_never_choose_a_winner(self) -> None:
        clean = {
            "selected_mIoU": 0.70,
            "selected_Pd": 0.90,
            "selected_Fa": 1.0e-5,
        }
        cp_go = gate.route_decision(
            candidate="cp_hf_s2_v1",
            clean=clean,
            variant={
                "selected_mIoU": 0.701,
                "selected_Pd": 0.90,
                "selected_Fa": 1.0e-5,
            },
            mechanism_checks={
                "adapter_executed": True,
                "adapter_departed_identity": True,
                "finite": True,
            },
        )
        self.assertEqual(cp_go["result"], "GO")
        self.assertEqual(
            cp_go["authorized_remaining_run_seeds"], [1446202191, 104728269]
        )
        psbfr_stop = gate.route_decision(
            candidate="psbfr_v1",
            clean=clean,
            variant={
                "selected_mIoU": 0.699,
                "selected_Pd": 0.90,
                "selected_Fa": 1.0e-5,
            },
            mechanism_checks={
                "bound_holds": True,
                "corrector_nonzero": True,
                "saturation_below_limit": True,
            },
        )
        self.assertEqual(psbfr_stop["result"], "STOP")
        self.assertEqual(psbfr_stop["authorized_remaining_run_seeds"], [])

    def test_safety_failure_stops_a_positive_miou_route(self) -> None:
        decision = gate.route_decision(
            candidate="cp_hf_s2_v1",
            clean={
                "selected_mIoU": 0.70,
                "selected_Pd": 0.90,
                "selected_Fa": 1.0e-5,
            },
            variant={
                "selected_mIoU": 0.71,
                "selected_Pd": 0.896,
                "selected_Fa": 2.0e-5,
            },
            mechanism_checks={
                "adapter_executed": True,
                "adapter_departed_identity": True,
                "finite": True,
            },
        )
        self.assertTrue(decision["safety_failure"])
        self.assertEqual(decision["result"], "STOP")

    def test_waiting_route_does_not_block_ready_route(self) -> None:
        ready = {
            "schema": gate.RESULT_SCHEMA,
            "status": "complete",
            "candidate": "cp_hf_s2_v1",
        }

        def evaluate(candidate: str):
            if candidate == "psbfr_v1":
                raise gate.Stage1RouteGateNotReady("missing PSBFR")
            return ready

        with mock.patch.object(gate, "evaluate_route", side_effect=evaluate), mock.patch.object(
            gate, "write_route_payload", return_value=Path("/tmp/cp.json")
        ):
            outcomes = gate.run()
        self.assertTrue(outcomes["psbfr_v1"].startswith("WAIT:"))
        self.assertEqual(outcomes["cp_hf_s2_v1"], "/tmp/cp.json")

    def test_writer_rejects_minimal_complete_payload(self) -> None:
        fresh = {
            "schema": gate.RESULT_SCHEMA,
            "status": "complete",
            "candidate": "cp_hf_s2_v1",
            "lockbox_accessed": False,
            "test_selected": False,
            "test_selection_supported": False,
            "public_test_supported": False,
            "test_split_accessed": False,
            "public_test_allowed": False,
        }
        with mock.patch.object(gate, "evaluate_route", return_value=fresh):
            with self.assertRaisesRegex(Exception, "fresh fixed evidence"):
                gate.write_route_payload(
                    "cp_hf_s2_v1",
                    {"schema": gate.RESULT_SCHEMA, "status": "complete"},
                )

    def test_cli_has_no_candidate_or_threshold_choice(self) -> None:
        gate.parse_args([])
        for args in (["--candidate", "cp_hf_s2_v1"], ["--threshold", "0"]):
            with self.assertRaises(SystemExit):
                gate.parse_args(args)

    def test_reserved_combined_s2_path_must_remain_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reserved.json"
            with mock.patch.object(gate.shared, "OUTPUT_PATH", path):
                gate._require_reserved_s2_absent()
                path.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(Exception, "must remain absent"):
                    gate._require_reserved_s2_absent()


if __name__ == "__main__":
    unittest.main()
