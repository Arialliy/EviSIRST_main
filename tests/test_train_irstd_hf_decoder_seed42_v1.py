from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import train_irstd_hf_decoder_seed42_v1 as runner


def _record(
    epoch: int,
    *,
    miou: float,
    niou: float,
    pd: float,
    fa: float,
    tiny_pd: float,
    loss: float,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "data_role": "val",
        "mIoU": miou,
        "Pd": pd,
        "Fa": fa,
        "evaluation_head": "out",
        "metrics": {
            "miou": miou,
            "niou": niou,
            "pd": pd,
            "fa": fa,
            "tiny_pd": tiny_pd,
            "validation_loss": loss,
        },
    }


class HFDecoderSeed42WrapperTest(unittest.TestCase):
    def _formal(self, *extra: str):
        return runner.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--allow-sample-level-fallback",
                *extra,
            ]
        )

    def _smoke(self, *extra: str):
        return self._formal(
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--smoke-max-train-samples",
            "1",
            "--smoke-max-val-samples",
            "1",
            *extra,
        )

    @staticmethod
    def _state(offset: float = 0.0) -> dict[str, torch.Tensor]:
        state = {
            f"base.{index}": torch.tensor([float(index) + offset])
            for index in range(runner.hf_decoder.BASE_STATE_KEY_COUNT)
        }
        state.update(
            {
                key: torch.tensor([float(index) + offset])
                for index, key in enumerate(
                    runner.hf_decoder.HF_DECODER_STATE_KEYS
                )
            }
        )
        return state

    def test_formal_cli_is_seed42_and_output_schema_is_isolated(self) -> None:
        args = self._formal()
        self.assertEqual(args.architecture_seed, 42)
        self.assertEqual(args.run_seed, 42)
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.warmup_epochs, 10)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.output_root, runner.DEFAULT_OUTPUT_ROOT)
        paths = runner.resolve_run_paths(args)
        self.assertEqual(
            Path(paths["run_dir"]).relative_to(runner.DEFAULT_OUTPUT_ROOT),
            Path("formal/IRSTD-1K/binary/run_seed_42"),
        )
        self.assertEqual(
            Path(paths["best_mIoU_final"]).name,
            "EviSIRST_best_mIoU.pth.tar",
        )
        self.assertEqual(
            Path(paths["best_Pd_final"]).name,
            "EviSIRST_best_Pd.pth.tar",
        )
        self.assertNotEqual(runner.TRAINING_SCHEMA, runner.frozen.TRAINING_SCHEMA)
        self.assertNotEqual(
            runner.DEFAULT_OUTPUT_ROOT, runner.frozen.DEFAULT_OUTPUT_ROOT
        )

    def test_non42_runtime_seed_is_rejected_even_for_smoke(self) -> None:
        with self.assertRaisesRegex(
            runner.Seed42HFDecoderRunnerError, "runtime seed must be 42"
        ):
            self._smoke("--run-seed", "43")
        with self.assertRaises(SystemExit):
            self._formal("--epochs", "999")
        with self.assertRaises(SystemExit):
            self._formal("--device", "cpu")

    def test_context_restores_every_frozen_runner_global(self) -> None:
        names = tuple(runner._PATCHES)
        before = {name: getattr(runner.frozen, name) for name in names}
        with runner._seed42_contract():
            self.assertEqual(runner.frozen.PAIRED_RUN_SEED, 42)
            self.assertEqual(runner.frozen.TRAINING_SCHEMA, runner.TRAINING_SCHEMA)
            self.assertEqual(
                runner.frozen.DEFAULT_OUTPUT_ROOT, runner.DEFAULT_OUTPUT_ROOT
            )
            self.assertIs(
                runner.frozen._variant_source_provenance,
                runner._seed42_source_provenance,
            )
        for name, value in before.items():
            self.assertIs(getattr(runner.frozen, name), value)

    def test_help_describes_seed42_revalidation_and_restores_frozen_doc(self) -> None:
        frozen_doc = runner.frozen.__doc__
        output = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(output):
            runner.parse_args(["--help"])
        rendered = output.getvalue()
        self.assertIn("fixed-Seed-42 revalidation", rendered)
        self.assertNotIn("Stage-A validation pilot", rendered)
        self.assertIs(runner.frozen.__doc__, frozen_doc)

    def test_source_and_identity_bind_wrapper_protocol_and_frozen_dependencies(self) -> None:
        source = runner.source_provenance()
        self.assertEqual(source["schema"], runner.SOURCE_SET_SCHEMA)
        expected = {
            "seed42_wrapper": "train_irstd_hf_decoder_seed42_v1.py",
            "seed42_protocol": (
                "experiments/IRSTD_HF_DECODER_SEED42_V1_PROTOCOL.md"
            ),
            "frozen_hf_runner": "train_irstd_hf_decoder_v1.py",
            "frozen_hf_architecture": "experiments/irstd_hf_decoder_v1.py",
            "frozen_zero_margin_selector": (
                "experiments/evisirst_zero_margin_selection.py"
            ),
            "frozen_hf_protocol": "experiments/IRSTD_HF_DECODER_V1_PROTOCOL.md",
        }
        for name, relative_path in expected.items():
            self.assertEqual(
                source["files"][name]["relative_path"], relative_path
            )
            self.assertEqual(len(source["files"][name]["sha256"]), 64)
            if name in runner.FROZEN_DEPENDENCY_SHA256:
                self.assertEqual(
                    source["files"][name]["sha256"],
                    runner.FROZEN_DEPENDENCY_SHA256[name],
                )
        self.assertTrue(
            any(name.startswith("r1/model_internal/") for name in source["files"])
        )
        self.assertEqual(len(source["source_tree_sha256"]), 64)

        identity = runner.determinism_protocol_identity()
        self.assertEqual(identity["schema"], runner.DETERMINISM_SCHEMA)
        self.assertEqual(
            identity["formal_seed_contract"],
            {
                "architecture_seed": 42,
                "runtime_seed": 42,
                "single_seed_only": True,
                "split_seed": 20260811,
            },
        )
        self.assertEqual(identity["source_files"], source["files"])
        self.assertFalse(identity["test_access"]["supported"])

    def test_run_identity_freezes_seed_split_recipe_and_572_states(self) -> None:
        args = self._formal()
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture",
        }
        contract = SimpleNamespace(
            manifest={"seeds": {"split_seed": 20260811}},
            manifest_sha256=runner.frozen.CANONICAL_IRSTD_MANIFEST_SHA256,
            data_tree_sha256=runner.frozen.CANONICAL_IRSTD_DATA_TREE_SHA256,
        )
        with runner._seed42_contract():
            identity = runner.frozen._variant_run_identity(
                args,
                contract,
                grouping,
                train_count=640,
                val_count=160,
                smoke=False,
            )
        self.assertEqual(identity["schema"], runner.TRAINING_SCHEMA + "/run_identity")
        self.assertEqual(identity["architecture_seed"], 42)
        self.assertEqual(identity["run_seed"], 42)
        self.assertEqual(identity["split_seed"], 20260811)
        self.assertEqual(identity["epochs"], 1000)
        self.assertEqual(identity["training_crop"], "clean_R1_unchanged")
        self.assertEqual(identity["state_contract"]["total_state_key_count"], 572)
        self.assertEqual(identity["selection_roles"], ["best_mIoU", "best_Pd"])
        self.assertIsNone(identity["selection_margin_raw"])
        self.assertFalse(identity["test_split_accessed"])
        self.assertEqual(
            identity["experiment"]["protocol_relative_path"],
            "experiments/IRSTD_HF_DECODER_SEED42_V1_PROTOCOL.md",
        )

    def test_resume_delegates_under_seed42_contract(self) -> None:
        args = self._smoke("--resume")
        expected = runner.DEFAULT_OUTPUT_ROOT / "fixture" / "EviSIRST_best_mIoU.pth.tar"

        def delegated(observed_args):
            self.assertIs(observed_args, args)
            self.assertTrue(observed_args.resume)
            self.assertEqual(runner.frozen.PAIRED_RUN_SEED, 42)
            self.assertEqual(runner.frozen.TRAINING_SCHEMA, runner.TRAINING_SCHEMA)
            self.assertEqual(
                runner.frozen.DEFAULT_OUTPUT_ROOT, runner.DEFAULT_OUTPUT_ROOT
            )
            return expected

        with mock.patch.object(runner, "_FROZEN_RUN", side_effect=delegated) as call:
            observed = runner.run(args)
        self.assertEqual(observed, expected)
        call.assert_called_once_with(args)

    def test_dual_role_finalizer_uses_seed42_schema_and_is_idempotent(self) -> None:
        history = [
            _record(1, miou=.71, niou=.70, pd=.80, fa=.02, tiny_pd=.7, loss=.2),
            _record(2, miou=.70, niou=.69, pd=.95, fa=.01, tiny_pd=.9, loss=.3),
        ]
        provenance = runner.zero_selection.select_checkpoints(history)
        identity = {
            "schema": runner.TRAINING_SCHEMA + "/run_identity",
            "dataset": runner.DATASET,
            "run_seed": 42,
            "identity_sha256": "a" * 64,
        }
        state1 = self._state()
        state2 = self._state(1.0)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            candidates = run_dir / "candidates"
            candidates.mkdir()
            artifact_map: dict[int, dict[str, object]] = {}
            for epoch, state in ((1, state1), (2, state2)):
                path = candidates / f"epoch_{epoch:04d}.pth.tar"
                torch.save(
                    {
                        "schema": runner.CANDIDATE_SCHEMA,
                        "run_identity": identity,
                        "epoch": epoch,
                        "validation_record": history[epoch - 1],
                        "state_dict": state,
                        "test_split_accessed": False,
                    },
                    path,
                )
                artifact_map[epoch] = {
                    "relative_path": f"candidates/{path.name}",
                    "file_sha256": runner._sha256_file(path),
                }
            primary = {
                "schema": runner.CHECKPOINT_SCHEMA,
                "training": identity,
                "state_dict": state1,
                "selection_provenance": provenance,
                "test_split_accessed": False,
            }
            kwargs = {
                "run_dir": run_dir,
                "args": SimpleNamespace(dataset=runner.DATASET),
                "identity": identity,
                "history": history,
                "candidate_artifacts": artifact_map,
                "expected_state": state1,
                "primary_payload": primary,
            }
            finals = runner._atomic_write_dual_role_finals(**kwargs)
            repeated = runner._atomic_write_dual_role_finals(**kwargs)
            self.assertEqual(repeated, finals)
            self.assertEqual(finals["best_mIoU"]["epoch"], 1)
            self.assertEqual(finals["best_Pd"]["epoch"], 2)
            for role, evidence in finals.items():
                payload = torch.load(
                    run_dir / evidence["relative_path"],
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(payload["schema"], runner.CHECKPOINT_SCHEMA)
                self.assertEqual(payload["selection_role"], role)
                self.assertEqual(len(payload["state_dict"]), 572)
                self.assertFalse(payload["selection_window_applied"])
                self.assertFalse(payload["test_split_accessed"])

    def test_protocol_forbids_public_test_and_seed_cherry_picking(self) -> None:
        protocol = runner.PROTOCOL_PATH.read_text(encoding="utf-8")
        self.assertIn("Runtime/training seed: `42`", protocol)
        self.assertIn("split seed `20260811`", protocol)
        self.assertIn("1000 epochs", protocol)
        self.assertIn("best_mIoU", protocol)
        self.assertIn("best_Pd", protocol)
        self.assertIn("not selected after observing", protocol)
        self.assertIn("Test: unavailable", protocol)
        gate = runner.interpretation_gate()
        self.assertFalse(
            gate["formal_seed_policy"]["post_hoc_choose_better_seed"]
        )
        self.assertFalse(gate["completion"]["public_test_allowed"])
        self.assertFalse(gate["completion"]["test_split_accessed"])


if __name__ == "__main__":
    unittest.main()
