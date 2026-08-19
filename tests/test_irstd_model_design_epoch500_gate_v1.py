from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments import evisirst_zero_margin_selection as zero_selection
import run_irstd_model_design_epoch500_gate_v1 as gate
import train_irstd_model_design_screen_v1 as runner


def _record(epoch: int, miou: float | None = None) -> dict[str, object]:
    value = (epoch / 1000.0) if miou is None else miou
    pd = 0.8 + epoch / 10000.0
    fa = 1e-4 - epoch / 10_000_000.0
    return {
        "epoch": epoch,
        "data_role": "val",
        "evaluation_head": "out",
        "mIoU": value,
        "Pd": pd,
        "Fa": fa,
        "metrics": {
            "miou": value,
            "niou": max(value - 0.0001, 0.0),
            "pd": pd,
            "fa": fa,
            "tiny_pd": 0.7,
            "validation_loss": 1.0 - value,
        },
    }


def _identity(variant: str, seed: int) -> dict[str, object]:
    args = runner.parse_args(
        [
            "--dataset-root",
            "/tmp/data",
            "--variant",
            variant,
            "--run-seed",
            str(seed),
        ]
    )
    contract = SimpleNamespace(
        manifest={"seeds": {"split_seed": 20260811}},
        manifest_sha256=runner.CANONICAL_IRSTD_MANIFEST_SHA256,
        data_tree_sha256=runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
    )
    with runner._hf_adapter(args):
        return runner._variant_run_identity(
            args,
            contract,
            {"mode": "scene_group"},
            train_count=640,
            val_count=160,
            smoke=False,
        )


class Epoch500GateTest(unittest.TestCase):
    def test_first500_selector_is_direct_frozen_selector_and_ignores_later(self) -> None:
        records = [_record(epoch) for epoch in range(1, 501)]
        records.extend([_record(501, 0.999), _record(502, 1.0)])
        fresh, metrics = gate._first500_selection(records)
        expected = zero_selection.select_checkpoints(
            records[:500],
            primary_role=zero_selection.PRIMARY_ROLE,
            margin=None,
        )
        self.assertEqual(fresh, expected)
        self.assertEqual(metrics["epoch"], 500)
        self.assertEqual(metrics["mIoU"], 0.5)

    def test_identity_binds_1000_schedule_sources_architecture_and_split(self) -> None:
        identity = _identity("psbfr_v1", 42)
        validated = gate._validate_identity(identity, seed=42, variant="psbfr_v1")
        self.assertEqual(validated, identity)
        for mutation, message in (
            (("epochs", 500), "1000-epoch"),
            (("manifest_sha256", "0" * 64), "1000-epoch"),
        ):
            tampered = copy.deepcopy(identity)
            tampered[mutation[0]] = mutation[1]
            unhashed = dict(tampered)
            unhashed.pop("identity_sha256")
            tampered["identity_sha256"] = gate._canonical_sha256(unhashed)
            with self.assertRaisesRegex(gate.ModelDesignEpoch500GateError, message):
                gate._validate_identity(tampered, seed=42, variant="psbfr_v1")

        tampered = copy.deepcopy(identity)
        tampered["state_contract"]["architecture_source_sha256"] = "0" * 64
        unhashed = dict(tampered)
        unhashed.pop("identity_sha256")
        tampered["identity_sha256"] = gate._canonical_sha256(unhashed)
        with self.assertRaisesRegex(gate.ModelDesignEpoch500GateError, "architecture"):
            gate._validate_identity(tampered, seed=42, variant="psbfr_v1")

    def test_variant_arm_requires_exact_atomic_500_and_candidate_binding(self) -> None:
        identity = _identity("psbfr_v1", 42)
        records = [_record(epoch) for epoch in range(1, 501)]
        fresh, metrics = gate._first500_selection(records)
        selected_epoch = int(metrics["epoch"])
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT / "runs") as temporary:
            root = Path(temporary)
            run_dir = root / "run_seed_42"
            candidates = run_dir / "candidates"
            candidates.mkdir(parents=True)
            state = {
                f"state.{index}": torch.tensor([float(index)])
                for index in range(569)
            }
            candidate_path = candidates / f"epoch_{selected_epoch:04d}.pth.tar"
            torch.save(
                {
                    "epoch": selected_epoch,
                    "run_identity": identity,
                    "validation_record": records[selected_epoch - 1],
                    "state_dict": state,
                    "test_split_accessed": False,
                },
                candidate_path,
            )
            artifact = {
                "relative_path": f"candidates/epoch_{selected_epoch:04d}.pth.tar",
                "file_sha256": gate._sha256_file(candidate_path),
            }
            latest = {
                "epoch": 500,
                "run_identity": identity,
                "training_history": [{"epoch": epoch} for epoch in range(1, 501)],
                "validation_history": records,
                "candidate_artifacts": {selected_epoch: artifact},
                "test_split_accessed": False,
            }
            torch.save(latest, run_dir / "last_training_state.pth.tar")
            (run_dir / "validation_history.json").write_text(
                json.dumps(
                    {
                        "run_identity": identity,
                        "validation_history": records,
                        "test_split_accessed": False,
                    }
                ),
                encoding="utf-8",
            )
            roots = dict(gate.VARIANT_ROOTS)
            roots["psbfr_v1"] = root
            with mock.patch.object(gate, "VARIANT_ROOTS", roots):
                evidence = gate._read_arm(seed=42, variant="psbfr_v1")
                self.assertEqual(evidence["selected_metrics"], metrics)
                self.assertEqual(
                    evidence["fresh_zero_margin_selection"], fresh
                )
                self.assertTrue(evidence["atomic_pause_at_exactly_500"])
                self.assertEqual(
                    evidence["selected_candidate"]["state_key_count"], 569
                )
                latest["epoch"] = 501
                latest["training_history"].append({"epoch": 501})
                latest["validation_history"].append(_record(501))
                torch.save(latest, run_dir / "last_training_state.pth.tar")
                with self.assertRaisesRegex(
                    gate.ModelDesignEpoch500GateError, "exactly epoch 500"
                ):
                    gate._read_arm(seed=42, variant="psbfr_v1")

    def test_decision_recomputes_mean_safety_and_psbfr_diagnostics(self) -> None:
        def arm(seed: int, miou: float, pd: float, fa: float):
            return {
                "run_seed": seed,
                "selected_metrics": {
                    "epoch": 10,
                    "mIoU": miou,
                    "Pd": pd,
                    "Fa": fa,
                },
            }

        pairs = [
            (arm(1, 0.70, 0.90, 1e-5), arm(1, 0.703, 0.90, 1e-5)),
            (arm(2, 0.69, 0.91, 2e-5), arm(2, 0.693, 0.91, 2e-5)),
            (arm(3, 0.68, 0.92, 2e-5), arm(3, 0.683, 0.92, 2e-5)),
        ]
        rules = {
            "mean_delta_mIoU_minimum": 0.002,
            "tanh_absolute_saturation_fraction_maximum": 0.1,
        }
        diagnostic = {
            "summary": {
                "bound_violation_count": 0,
                "nonzero_correction_count": 1,
                "tanh_saturation_fraction": 0.01,
            }
        }
        decision = gate._decision("psbfr_v1", pairs, rules, diagnostic)
        self.assertEqual(decision["result"], "GO")
        failed = copy.deepcopy(diagnostic)
        failed["summary"]["bound_violation_count"] = 1
        self.assertEqual(
            gate._decision("psbfr_v1", pairs, rules, failed)["result"], "STOP"
        )

    def test_wait_never_writes_and_terminal_output_is_no_replace(self) -> None:
        waiting = {"schema": gate.RESULT_SCHEMA, "status": "waiting_for_epoch_500"}
        with self.assertRaises(gate.ModelDesignEpoch500GateNotReady):
            gate._write_no_replace(waiting)
        complete = {"schema": gate.RESULT_SCHEMA, "status": "complete"}
        with tempfile.TemporaryDirectory(dir=gate.PROJECT_ROOT / "runs") as temporary:
            with mock.patch.object(gate, "PROJECT_ROOT", Path(temporary)), mock.patch.object(
                gate, "OUTPUT_RELATIVE_PATH", "result.json"
            ):
                path = gate._write_no_replace(complete)
                self.assertTrue(path.is_file())
                with self.assertRaises(FileExistsError):
                    gate._write_no_replace(complete)


if __name__ == "__main__":
    unittest.main()
