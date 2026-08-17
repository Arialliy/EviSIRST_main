from __future__ import annotations

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import run_irstd_hf_decoder_seed42_decision_gate as gate


def _artifact(path: str, token: str) -> dict[str, object]:
    return {"relative_path": path, "sha256": token * 64, "size_bytes": 1}


def _checkpoint_evidence(arm: str, epoch: int, token: str) -> dict[str, object]:
    return {
        "final_checkpoint": _artifact(f"runs/{arm}/final.pth.tar", token),
        "selected_candidate": _artifact(f"runs/{arm}/candidate.pth.tar", token),
        "selected_epoch": epoch,
        "state_key_count": 572 if arm == "H42" else 564,
        "state_dict_sha256": token * 64,
        "final_candidate_tensor_equality": True,
        "training_identity_sha256": token * 64,
    }


def _metrics(epoch: int, miou: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "mIoU": miou,
        "nIoU": 0.65,
        "F1": 0.81,
        "pixel_precision": 0.82,
        "pixel_recall": 0.80,
        "false_objects_per_image": 0.2,
        "Pd": 0.94,
        "Fa": 0.00001,
        "tinyPd": 0.85,
        "loss": 0.0003,
    }


def _payload(*, a_miou: float = 0.6991869918699187, h_miou: float = 0.6899375975039002) -> dict[str, object]:
    return gate.build_result_payload(
        a42_summary_artifact=_artifact("runs/A42/summary.json", "a"),
        h42_summary_artifact=_artifact("runs/H42/summary.json", "b"),
        a42_checkpoint_evidence=_checkpoint_evidence("A42", 411, "c"),
        h42_checkpoint_evidence=_checkpoint_evidence("H42", 402, "d"),
        a42_fresh_selection={"selected": {"epoch": 411}},
        h42_fresh_selection={"selected": {"epoch": 402}},
        a42_metrics=_metrics(411, a_miou),
        h42_metrics=_metrics(402, h_miou),
        a42_stored_selection={
            "rule_version": "historical-window",
            "candidate_tolerance_raw": 0.001,
            "window_applied": True,
            "selected_epoch": 411,
        },
        h42_stored_selection={
            "rule_version": gate.zero_selection.RULE_VERSION,
            "candidate_tolerance_raw": 0.0,
            "window_applied": False,
            "selected_epoch": 402,
        },
        selector_artifact={
            **_artifact(gate.SELECTOR_RELATIVE_PATH, "e"),
            "rule_version": gate.zero_selection.RULE_VERSION,
        },
        split_artifacts={"manifest": _artifact("splits/manifest.json", "f")},
        split_provenance={"schema": "fixture-split/v1"},
        gate_source_artifact=_artifact(gate.GATE_SOURCE_RELATIVE_PATH, "1"),
    )


class FixedDecisionTest(unittest.TestCase):
    def test_cli_has_no_override_surface(self) -> None:
        self.assertEqual(vars(gate.parse_args([])), {})
        with self.assertRaises(SystemExit):
            gate.parse_args(["--run-seed", "7"])
        self.assertEqual(gate.RUN_SEED, 42)
        self.assertEqual(gate.ARCHITECTURE_SEED, 42)

    def test_decimal_stop_selects_a42_and_never_opens_later_gates(self) -> None:
        payload = _payload()
        decision = payload["decision"]
        self.assertEqual(payload["schema"], gate.RESULT_SCHEMA)
        self.assertEqual(decision["result"], "STOP")
        self.assertEqual(decision["mIoU_delta_decimal"], "-0.0092493943660185")
        self.assertEqual(decision["formal_model"]["role"], "A42_clean_R1")
        self.assertEqual(
            decision["formal_model"]["relative_path"],
            gate.A42_FINAL_RELATIVE_PATH,
        )
        self.assertFalse(decision["public_test_allowed"])
        self.assertFalse(decision["multi_seed_expansion_allowed"])
        self.assertFalse(payload["test_split_accessed"])

    def test_zero_delta_is_stop_and_strict_positive_delta_is_go(self) -> None:
        boundary = _payload(a_miou=0.7, h_miou=0.7)
        positive = _payload(a_miou=0.7, h_miou=0.7000000000000001)
        self.assertEqual(boundary["decision"]["result"], "STOP")
        self.assertEqual(positive["decision"]["result"], "GO")
        self.assertEqual(
            positive["decision"]["formal_model"]["relative_path"],
            gate.H42_FINAL_RELATIVE_PATH,
        )
        self.assertFalse(positive["decision"]["multi_seed_expansion_allowed"])

    def test_source_does_not_import_test_dataset_or_evaluator(self) -> None:
        source = Path(gate.__file__).read_text(encoding="utf-8")
        imported: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertNotIn("test", imported)
        self.assertNotIn("experiments.evisirst_data", imported)
        self.assertNotIn("EviSIRSTTestDataset", source)


class IntegrityPrimitiveTest(unittest.TestCase):
    def test_self_rehashed_identity_tamper_still_fails_frozen_binding(self) -> None:
        fixtures = (
            ("A42", gate.A42_FINAL_RELATIVE_PATH),
            ("H42", gate.H42_FINAL_RELATIVE_PATH),
        )
        for arm, relative_path in fixtures:
            if not (gate.PROJECT_ROOT / relative_path).is_file():
                self.skipTest("completed formal Seed42 checkpoints are unavailable")
            checkpoint, _ = gate._load_checkpoint(
                relative_path, label=f"{arm} fixture final"
            )
            identity = copy.deepcopy(checkpoint["training"])
            gate._validate_identity(identity, arm=arm, label=f"{arm} fixture")
            identity["grouping_policy"] = {"mode": "tampered_but_rehashed"}
            identity.pop("identity_sha256")
            identity["identity_sha256"] = gate._canonical_sha256(identity)
            with self.subTest(arm=arm), self.assertRaisesRegex(
                gate.Seed42DecisionGateError, f"frozen {arm} identity"
            ):
                gate._validate_identity(identity, arm=arm, label=f"{arm} fixture")

    def test_state_comparison_is_tensor_exact_and_hash_bound(self) -> None:
        final = {
            "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "count": torch.tensor(3, dtype=torch.int64),
        }
        candidate = {key: value.clone() for key, value in final.items()}
        observed = gate._compare_state_dicts(final, candidate, label="fixture")
        self.assertEqual(observed, gate._state_dict_sha256(final))
        candidate["weight"][0, 1] = 2.5
        with self.assertRaisesRegex(gate.Seed42DecisionGateError, "tensor differs"):
            gate._compare_state_dicts(final, candidate, label="fixture")

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    gate.Seed42DecisionGateError, "strict finite"
                ):
                    gate._strict_json_object(raw, label="fixture")

    def test_test_disclosure_must_be_explicit_false(self) -> None:
        gate._require_false_disclosures(
            {"nested": {"test_split_accessed": False}}, label="fixture"
        )
        with self.assertRaisesRegex(gate.Seed42DecisionGateError, "explicit false"):
            gate._require_false_disclosures(
                {"nested": {"test_split_accessed": True}}, label="fixture"
            )


class ImmutableLedgerTest(unittest.TestCase):
    def test_atomic_hardlink_no_replace_and_fresh_validation(self) -> None:
        payload = _payload()
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            repository.mkdir(exist_ok=True)
            with (
                mock.patch.object(gate, "PROJECT_ROOT", repository),
                mock.patch.object(
                    gate, "OUTPUT_RELATIVE_PATH", "runs/gate/result.json"
                ),
                mock.patch.object(gate, "evaluate_payload", return_value=payload),
            ):
                output = gate._write_fixed_json_atomic_no_replace(payload)
                observed_inode = output.stat().st_ino
                self.assertEqual(gate.validate_existing_result(), payload)
                self.assertEqual(output.stat().st_ino, observed_inode)
                with self.assertRaises(FileExistsError):
                    gate._write_fixed_json_atomic_no_replace(payload)

                changed = json.loads(output.read_text(encoding="utf-8"))
                changed["decision"]["result"] = "GO"
                output.write_text(json.dumps(changed) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    gate.Seed42DecisionGateError, "fresh canonical"
                ):
                    gate.validate_existing_result()


class CompletedArtifactIntegrationTest(unittest.TestCase):
    def test_completed_artifacts_freshly_evaluate_to_stop_without_writing(self) -> None:
        required = [
            gate.PROJECT_ROOT / gate.A42_SUMMARY_RELATIVE_PATH,
            gate.PROJECT_ROOT / gate.A42_FINAL_RELATIVE_PATH,
            gate.PROJECT_ROOT / gate.H42_SUMMARY_RELATIVE_PATH,
            gate.PROJECT_ROOT / gate.H42_FINAL_RELATIVE_PATH,
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("completed formal Seed42 artifacts are unavailable")
        output = gate.PROJECT_ROOT / gate.OUTPUT_RELATIVE_PATH
        before = output.read_bytes() if output.exists() else None
        payload = gate.evaluate_payload()
        after = output.read_bytes() if output.exists() else None
        self.assertEqual(after, before)
        self.assertEqual(payload["decision"]["result"], "STOP")
        self.assertEqual(payload["decision"]["formal_model"]["role"], "A42_clean_R1")
        self.assertEqual(
            payload["inputs"]["A42_clean_R1"]["selected_epoch"], 411
        )
        self.assertEqual(
            payload["inputs"]["H42_clean_R1_plus_HF_Decoder_V1"]["selected_epoch"],
            402,
        )
        self.assertEqual(
            payload["inputs"]["A42_clean_R1"]["stored_selection"][
                "candidate_tolerance_raw"
            ],
            0.001,
        )
        self.assertTrue(
            payload["inputs"]["A42_clean_R1"]["stored_selection"][
                "window_applied"
            ]
        )
        self.assertEqual(
            payload["inputs"]["A42_clean_R1"][
                "selected_validation_record_sha256"
            ],
            "f7a56520729535efd82c865fa7cd8ffe41c3d27bf76942bbca67a56dcf06cd56",
        )
        self.assertEqual(
            payload["inputs"]["H42_clean_R1_plus_HF_Decoder_V1"][
                "selected_validation_record_sha256"
            ],
            "018635a974daad7ebf3f682ee9fe0c96d4e0a57f983a13baa10ec8ff4be7bb6e",
        )
        self.assertFalse(payload["public_test_allowed"])
        self.assertFalse(payload["multi_seed_expansion_allowed"])


if __name__ == "__main__":
    unittest.main()
