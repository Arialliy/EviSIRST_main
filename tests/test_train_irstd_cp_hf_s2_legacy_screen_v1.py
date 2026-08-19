from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import train_irstd_cp_hf_s2_legacy_screen_v1 as runner


class CPHFS2LegacyScreenContractTest(unittest.TestCase):
    def _args(self, seed: int = 42, *extra: str):
        return runner.parse_args(
            ["--dataset-root", "/tmp/irstd", "--run-seed", str(seed), *extra]
        )

    def _identity(self) -> dict[str, object]:
        return {
            "schema": runner.TRAINING_SCHEMA + "/run_identity",
            "architecture_seed": 42,
            "run_seed": 42,
            "epochs": 1000,
            "source_tree_sha256": "a" * 64,
            "manifest_sha256": (
                "5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596"
            ),
            "data_tree_sha256": (
                "ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30"
            ),
            "test_split_accessed": False,
        }

    def test_rules_fix_three_seeds_schedule_pause_and_two_candidates(self) -> None:
        rules = runner.load_rules()
        run = rules["legacy_screen_run"]
        comparison = rules["comparison"]
        self.assertEqual(tuple(run["run_seeds"]), runner.FORMAL_RUN_SEEDS)
        self.assertEqual(run["configured_total_epochs"], 1000)
        self.assertEqual(run["operational_pause_after_atomic_epoch"], 500)
        self.assertFalse(run["screen_records_after_epoch_500_allowed"])
        self.assertEqual(run["runner_enforced_pause_marker"], "EPOCH500_CLEAN")
        self.assertTrue(run["epoch_501_entry_forbidden"])
        self.assertTrue(run["resume_at_epoch_500_returns_clean_pause"])
        self.assertIsNone(run["selection_margin_raw"])
        self.assertEqual(
            comparison["candidate_registry"], ["psbfr_v1", "cp_hf_s2_v1"]
        )
        self.assertTrue(comparison["candidate_decisions_are_independent"])
        self.assertFalse(comparison["candidate_ranking_or_winner_selection"])
        architecture = rules["architecture"]
        self.assertEqual(architecture["state_key_count"], 576)
        self.assertEqual(architecture["parameter_count"], 10_874_615)
        self.assertEqual(architecture["extension_state_prefix"], "decoder_cp_hf_s2.")
        self.assertEqual(len(architecture["source_sha256"]), 64)

    def test_cli_and_paths_are_fixed(self) -> None:
        for seed in runner.FORMAL_RUN_SEEDS:
            args = self._args(seed)
            self.assertEqual(args.epochs, 1000)
            self.assertEqual(args.warmup_epochs, 10)
            self.assertEqual(args.device, "cuda:0")
            self.assertEqual(args.variant, "cp_hf_s2_v1")
            paths = runner.resolve_run_paths(args)
            self.assertEqual(
                paths["run_dir"].relative_to(runner.DEFAULT_OUTPUT_ROOT),
                Path("formal") / "IRSTD-1K" / "binary" / f"run_seed_{seed}",
            )
        with self.assertRaises(SystemExit):
            self._args(7)
        with self.assertRaises(SystemExit):
            self._args(42, "--epochs", "500")
        with self.assertRaises(SystemExit):
            self._args(42, "--device", "cpu")

        smoke = self._args(
            42,
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--smoke-max-train-samples",
            "1",
            "--smoke-max-val-samples",
            "1",
        )
        smoke_paths = runner.resolve_run_paths(smoke)
        relative = smoke_paths["run_dir"].relative_to(runner.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(relative.parts[:4], ("smoke", "IRSTD-1K", "binary", "run_seed_42"))
        self.assertTrue(relative.parts[-1].startswith("smoke_"))
        with self.assertRaises(SystemExit):
            self._args(42, "--smoke-max-train-samples", "1")

    def test_split_and_source_seals_without_architecture_import(self) -> None:
        split = runner.split_provenance()
        self.assertEqual(split["train_count"], 640)
        self.assertEqual(split["val_count"], 160)
        self.assertFalse(split["test_split_accessed"])
        source = runner.source_provenance(require_architecture=False)
        self.assertIn("screen_runner", source["files"])
        self.assertIn("screen_amendment", source["files"])
        self.assertEqual(len(source["source_tree_sha256"]), 64)

    def test_rules_source_hash_drift_and_float_seed_fail_closed(self) -> None:
        rules = runner.load_rules()
        rules["cp_hf_s2_mechanism_diagnostic"]["source_sha256"] = "0" * 64
        runs_root = runner.PROJECT_ROOT / "runs"
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            path = Path(temporary) / "rules.json"
            path.write_text(json.dumps(rules), encoding="utf-8")
            with mock.patch.object(runner, "RULES_PATH", path):
                with self.assertRaisesRegex(Exception, "rules identity differs"):
                    runner.load_rules()
        args = self._args(42)
        args.run_seed = 42.0
        with self.assertRaisesRegex(Exception, "run seed"):
            runner.require_args(args)

    def test_resume_candidate_and_final_envelopes_fail_closed(self) -> None:
        identity = self._identity()
        resume = {
            "schema": runner.TRAINING_SCHEMA,
            "run_identity": identity,
            "epoch": 2,
            "state_dict": {"base.weight": 1},
            "optimizer": {},
            "rng": {},
            "training_history": [{"epoch": 1}, {"epoch": 2}],
            "validation_history": [{"epoch": 1}, {"epoch": 2}],
            "candidate_artifacts": {},
        }
        resume.update(runner._strict_false_disclosures())
        self.assertIs(
            runner.validate_resume_envelope(
                resume,
                expected_identity=identity,
                expected_state_keys={"base.weight"},
            ),
            resume,
        )
        tampered = copy.deepcopy(resume)
        tampered["run_identity"]["run_seed"] = 7
        with self.assertRaisesRegex(Exception, "identity"):
            runner.validate_resume_envelope(tampered, expected_identity=identity)
        leaked = copy.deepcopy(resume)
        leaked["rng"]["test_split_accessed"] = True
        with self.assertRaisesRegex(Exception, "must be false"):
            runner.validate_resume_envelope(leaked, expected_identity=identity)

        candidate = {
            "schema": runner.CANDIDATE_SCHEMA,
            "run_identity": identity,
            "epoch": 2,
            "state_dict": {"base.weight": 1},
            "validation_record": {"epoch": 2, "test_split_accessed": False},
        }
        candidate.update(runner._strict_false_disclosures())
        runner.validate_candidate_envelope(
            candidate, expected_identity=identity, epoch=2
        )
        final = {
            "schema": runner.CHECKPOINT_SCHEMA,
            "training": identity,
            "selection_role": "best_mIoU",
            "source_candidate": {
                "relative_path": "candidates/epoch_0002.pth.tar",
                "sha256": "b" * 64,
            },
            "selected_complete_key": {"epoch": 2, "mIoU": 0.5},
            "selection_provenance": {"rule_version": "fixture"},
            "state_dict": {"base.weight": 1},
        }
        final["selected_complete_key_sha256"] = runner._canonical_sha256(
            final["selected_complete_key"]
        )
        final.update(runner._strict_false_disclosures())
        runner.validate_final_envelope(
            final, expected_identity=identity, role="best_mIoU"
        )

    def test_json_writer_is_idempotent_but_never_clobbers(self) -> None:
        payload = {"schema": "fixture/v1", "test_split_accessed": False}
        runs_root = runner.PROJECT_ROOT / "runs"
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            path = Path(temporary) / "decision.json"
            runner.write_json_no_clobber(path, payload)
            runner.write_json_no_clobber(path, payload)
            with self.assertRaises(FileExistsError):
                runner.write_json_no_clobber(
                    path,
                    {"schema": "different/v1", "test_split_accessed": False},
                )
            target = Path(temporary) / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            link = Path(temporary) / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(Exception, "unsafe"):
                runner.write_json_no_clobber(link, payload)
            real_parent = Path(temporary) / "real_parent"
            real_parent.mkdir()
            linked_parent = Path(temporary) / "linked_parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(Exception, "real directory"):
                runner.write_json_no_clobber(linked_parent / "result.json", payload)
            candidate = Path(temporary) / "candidate.pth.tar"
            candidate.write_bytes(b"sentinel")
            with self.assertRaises(FileExistsError):
                runner._write_bytes_no_clobber(
                    candidate, b"replacement", allow_identical=False
                )
            self.assertEqual(candidate.read_bytes(), b"sentinel")
            fifo = Path(temporary) / "fifo.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(Exception, "not regular"):
                runner.write_json_no_clobber(fifo, payload)

    def test_formal_pause_blocks_501_but_never_changes_smoke(self) -> None:
        class Dataset:
            def __init__(self) -> None:
                self.epochs: list[int] = []

            def set_epoch(self, epoch: int) -> None:
                self.epochs.append(epoch)

        formal = Dataset()
        runner._install_atomic_epoch500_pause(formal, smoke=False)
        formal.set_epoch(500)
        with self.assertRaises(runner.CPHFS2AtomicScreenPause):
            formal.set_epoch(501)
        self.assertEqual(formal.epochs, [500])

        smoke = Dataset()
        runner._install_atomic_epoch500_pause(smoke, smoke=True)
        smoke.set_epoch(501)
        self.assertEqual(smoke.epochs, [501])

    def test_formal_entry_rejects_existing_summary_and_resume_past_500(self) -> None:
        args = self._args(42)
        runs_root = runner.PROJECT_ROOT / "runs"
        with tempfile.TemporaryDirectory(dir=runs_root) as temporary:
            run_dir = Path(temporary)
            paths = {
                "run_dir": run_dir,
                "latest": run_dir / "last_training_state.pth.tar",
                "summary": run_dir / "summary.json",
                "best_mIoU_final": run_dir / "EviSIRST_best_mIoU.pth.tar",
                "best_Pd_final": run_dir / "EviSIRST_best_Pd.pth.tar",
            }
            paths["summary"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "forbids summaries"):
                runner._formal_entry_barrier(args, paths=paths)
            paths["summary"].unlink()
            for epoch in (501, 1000):
                torch.save({"epoch": epoch}, paths["latest"])
                args.resume = True
                with self.assertRaisesRegex(Exception, r"must be in \[1, 500\]"):
                    runner._formal_entry_barrier(args, paths=paths)
            paths["latest"].unlink()
            paths["best_mIoU_final"].write_text("forbidden", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "final checkpoints"):
                runner._formal_entry_barrier(args, paths=paths)

    def test_state_validator_rejects_self_consistent_fixed_kernel_tamper(self) -> None:
        context = SimpleNamespace(smoke_max_train_samples=1)
        with runner._screen_transaction_adapter(context) as screen:
            model, _ = screen._variant_initialize_evisirst(
                runner.DATASET, seed=runner.ARCHITECTURE_SEED, training=True
            )
            tampered = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            tampered["decoder_cp_hf_s2.low_pass_kernel"] = torch.zeros_like(
                tampered["decoder_cp_hf_s2.low_pass_kernel"]
            )
            previous = screen._ACTIVE_VARIANT
            screen._ACTIVE_VARIANT = runner.ARCHITECTURE_VARIANT
            try:
                with self.assertRaisesRegex(Exception, "fixed binomial kernel"):
                    screen._validate_state_dict(tampered, tampered)
            finally:
                screen._ACTIVE_VARIANT = previous

    def test_transaction_adapter_restores_every_patch_after_exception(self) -> None:
        import train_irstd_model_design_screen_v1 as screen

        names = (
            "RULES_PATH",
            "TRAINING_SCHEMA",
            "build_datasets",
            "_variant_run_identity",
            "_validate_state_dict",
            "_variant_atomic_torch_save",
            "_run_process_lock",
        )
        before = {name: getattr(screen, name) for name in names}
        before_resume = screen.r1._load_resume_state
        before_evaluate = screen.r1.evaluate_model
        before_finalize = screen.hf_transaction._finalize_completed

        class Sentinel(RuntimeError):
            pass

        with self.assertRaises(Sentinel):
            with runner._screen_transaction_adapter(
                SimpleNamespace(smoke_max_train_samples=1)
            ):
                self.assertIsNot(screen.RULES_PATH, before["RULES_PATH"])
                self.assertIsNot(screen.build_datasets, before["build_datasets"])
                self.assertIsNot(screen.r1._load_resume_state, before_resume)
                raise Sentinel
        for name, value in before.items():
            self.assertIs(getattr(screen, name), value)
        self.assertIs(screen.r1._load_resume_state, before_resume)
        self.assertIs(screen.r1.evaluate_model, before_evaluate)
        self.assertIs(screen.hf_transaction._finalize_completed, before_finalize)

    def test_actual_resume_frontier_and_finalize_cannot_bypass_stage1(self) -> None:
        import train_irstd_model_design_screen_v1 as screen

        original = screen.r1._load_resume_state
        screen.r1._load_resume_state = lambda *args, **kwargs: (1001, [], [], {})
        try:
            with runner._screen_transaction_adapter(
                SimpleNamespace(smoke_max_train_samples=None)
            ):
                with self.assertRaisesRegex(Exception, "cannot exceed committed epoch 500"):
                    screen.r1._load_resume_state(model=object())
                with self.assertRaisesRegex(Exception, "cannot finalize"):
                    screen.hf_transaction._finalize_completed(
                        SimpleNamespace(smoke_max_train_samples=None), {}
                    )
        finally:
            screen.r1._load_resume_state = original

    def test_process_lock_rechecks_formal_barrier_inside_lock(self) -> None:
        import train_irstd_model_design_screen_v1 as screen

        args = self._args(42)

        @contextmanager
        def fake_lock(_args):
            yield Path("/tmp/fake.lock")

        with mock.patch.object(screen, "_run_process_lock", fake_lock):
            with runner._screen_transaction_adapter(args):
                with mock.patch.object(runner, "_formal_entry_barrier") as barrier:
                    with screen._run_process_lock(args):
                        pass
                    barrier.assert_called_once_with(args)

    def test_execution_is_sealed_to_landed_api_and_not_reference(self) -> None:
        self.assertTrue(runner.EXECUTION_SEALED)
        self.assertEqual(
            runner.ARCHITECTURE_MODULE, "experiments.irstd_cp_hf_s2_v1"
        )
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import EviSIRST_HF_Decoder_V2_reference", source)
        self.assertNotIn("from EviSIRST_HF_Decoder_V2_reference", source)
        self.assertNotIn("EviSIRSTTestDataset", source)

    @unittest.skipUnless(
        os.environ.get("EVISIRST_RUN_CPU_SMOKE") == "1",
        "set EVISIRST_RUN_CPU_SMOKE=1 for the real one-epoch transaction",
    )
    def test_real_cpu_one_epoch_transaction_and_resume(self) -> None:
        dataset_root = os.environ.get("EVISIRST_DATASET_ROOT")
        if not dataset_root:
            self.skipTest("set EVISIRST_DATASET_ROOT to the local dataset parent")
        arguments = [
            "--dataset-root",
            dataset_root,
            "--run-seed",
            "104728269",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--smoke-max-train-samples",
            "1",
            "--smoke-max-val-samples",
            "1",
        ]
        args = runner.parse_args(arguments)
        paths = runner.resolve_run_paths(args)
        if not paths["summary"].exists():
            runner.run(args)
        resumed = runner.run(runner.parse_args([*arguments, "--resume"]))
        self.assertEqual(resumed, paths["best_mIoU_final"])
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["smoke"])
        self.assertFalse(summary["promotion_eligible"])
        self.assertFalse(summary["test_split_accessed"])
        self.assertEqual(len(summary["training_history"]), 1)
        self.assertEqual(len(summary["validation_history"]), 1)


if __name__ == "__main__":
    unittest.main()
