from __future__ import annotations

import ast
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import train_irstd_hf_decoder_v1 as runner


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


class HFDecoderRunnerTest(unittest.TestCase):
    def _parse(self, *extra: str):
        return runner.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--run-seed",
                str(runner.PAIRED_RUN_SEED),
                *extra,
            ]
        )

    def _smoke(self, *extra: str):
        return self._parse(
            "--run-seed",
            "42",
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

    def _contract(self):
        grouping = {
            "mode": "sample_level_fallback",
            "warning": "scene leakage cannot be ruled out",
        }
        return SimpleNamespace(
            dataset=runner.DATASET,
            manifest={
                "schema": "evisirst_v2_train_val_split/v1",
                "seeds": {"split_seed": 20260811},
                "grouping": grouping,
            },
            manifest_sha256=runner.CANONICAL_IRSTD_MANIFEST_SHA256,
            data_tree_sha256=runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
            data_tree_verified=True,
            train_ids=tuple(range(640)),
            val_ids=tuple(range(160)),
        )

    def _state(self) -> dict[str, torch.Tensor]:
        state = {
            f"base.{index}": torch.tensor([float(index)], dtype=torch.float32)
            for index in range(runner.hf_decoder.BASE_STATE_KEY_COUNT)
        }
        state.update(
            {
                key: torch.tensor([float(index + 1)], dtype=torch.float32)
                for index, key in enumerate(runner.hf_decoder.HF_DECODER_STATE_KEYS)
            }
        )
        return state

    def test_cli_freezes_formal_recipe_and_two_named_finals(self) -> None:
        args = self._parse()
        self.assertEqual(args.dataset, "IRSTD-1K")
        self.assertEqual(args.target_mode, "binary")
        self.assertEqual(args.architecture_seed, 42)
        self.assertEqual(args.run_seed, runner.PAIRED_RUN_SEED)
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.base_lr, 1e-3)
        self.assertEqual(args.min_lr, 1e-5)
        self.assertEqual(args.warmup_epochs, 10)
        self.assertEqual(args.val_interval, 1)
        self.assertEqual(args.output_root, runner.DEFAULT_OUTPUT_ROOT)
        paths = runner.resolve_run_paths(args)
        self.assertEqual(Path(paths["best_mIoU_final"]).name, "EviSIRST_best_mIoU.pth.tar")
        self.assertEqual(Path(paths["best_Pd_final"]).name, "EviSIRST_best_Pd.pth.tar")
        self.assertFalse(paths["smoke"])
        self.assertEqual(
            Path(paths["run_dir"]).relative_to(runner.DEFAULT_OUTPUT_ROOT),
            Path("formal/IRSTD-1K/binary/run_seed_1446202191"),
        )
        smoke = runner.resolve_run_paths(self._smoke())
        self.assertTrue(smoke["smoke"])
        self.assertIn("smoke", Path(smoke["run_dir"]).parts)

        invalid = (
            ["--run-seed", "42"],
            ["--epochs", "999"],
            ["--architecture-seed", "43"],
            ["--dataset", "NUAA-SIRST"],
            ["--target-mode", "soft"],
            ["--device", "cpu"],
            ["--split-root", "/tmp/not-canonical"],
            ["--output-root", "/tmp/escape"],
            ["--batch-size", "8"],
            ["--base-lr", "0.01"],
        )
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self._parse(*extra)

    def test_gate_and_protocol_use_zero_threshold_and_reselected_histories(self) -> None:
        gate = runner.promotion_gate()
        self.assertEqual(gate["status"], "TBD")
        self.assertEqual(gate["paired_control"]["schema"], runner._R1_TRAINING_SCHEMA)
        with runner._variant_runtime():
            self.assertEqual(
                runner.promotion_gate()["paired_control"]["schema"],
                runner._R1_TRAINING_SCHEMA,
            )
        self.assertEqual(gate["primary"]["operator"], ">")
        self.assertEqual(gate["primary"]["minimum_delta"], 0.0)
        self.assertIsNone(gate["primary"]["selection_margin_raw"])
        self.assertEqual(
            gate["paired_control"]["zero_margin_selected_epoch"],
            "TBD_after_control_completion",
        )
        self.assertTrue(gate["paired_control"]["control_completion_required"])
        self.assertEqual(
            gate["paired_control"]["comparison_source"],
            "complete_saved_validation_history",
        )
        self.assertFalse(
            gate["paired_control"]["historical_window_summary_selected_is_comparator"]
        )
        self.assertEqual(
            gate["complete_target_comparator"]["comparison_source"],
            "complete_saved_validation_history",
        )
        self.assertFalse(gate["decision"]["public_test_allowed"])
        protocol = runner.PROTOCOL_PATH.read_text(encoding="utf-8")
        self.assertIn("H0", protocol)
        self.assertIn("H1", protocol)
        self.assertIn("GO-1", protocol)
        self.assertIn("GO-2", protocol)
        self.assertIn("STOP", protocol)
        self.assertIn("interim", protocol)
        self.assertIn("complete saved validation history", protocol)
        self.assertIn("Test: unavailable", protocol)

    def test_source_identity_binds_runner_architecture_selector_and_protocol(self) -> None:
        source = runner._variant_source_provenance()
        files = source["files"]
        expected = {
            "variant_runner": "train_irstd_hf_decoder_v1.py",
            "hf_decoder_architecture": "experiments/irstd_hf_decoder_v1.py",
            "zero_margin_selector": "experiments/evisirst_zero_margin_selection.py",
            "frozen_protocol": "experiments/IRSTD_HF_DECODER_V1_PROTOCOL.md",
        }
        for name, relative in expected.items():
            self.assertEqual(files[name]["relative_path"], relative)
            self.assertEqual(len(files[name]["sha256"]), 64)
        self.assertTrue(any(name.startswith("r1/model_internal/") for name in files))
        self.assertEqual(len(source["source_tree_sha256"]), 64)
        identity = runner._variant_determinism_protocol_identity()
        self.assertEqual(identity["schema"], runner.DETERMINISM_SCHEMA)
        self.assertEqual(identity["architecture"]["state_key_count"], 572)
        self.assertEqual(identity["training_data"]["crop_policy"], "clean_R1_unchanged")
        self.assertEqual(identity["selection"]["rule_version"], runner.zero_selection.RULE_VERSION)
        self.assertIsNone(identity["selection"]["margin_raw"])
        self.assertFalse(identity["selection"]["window_applied"])
        self.assertFalse(identity["test_access"]["supported"])

    def test_run_identity_is_single_variable_full_scratch_and_640_160(self) -> None:
        args = self._parse("--allow-sample-level-fallback")
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture",
        }
        identity = runner._variant_run_identity(
            args,
            self._contract(),
            grouping,
            train_count=640,
            val_count=160,
            smoke=False,
        )
        self.assertEqual(identity["architecture_variant"], "IRSTD-HF-Decoder-v1")
        self.assertEqual(identity["initialization"], "full_model_scratch")
        self.assertTrue(identity["all_base_and_hf_parameters_trainable"])
        self.assertEqual(identity["optimizer_parameter_groups"], 1)
        self.assertEqual(identity["loss"], "sum_of_six_BCELoss_mean_terms")
        self.assertEqual(identity["deep_supervision_weights"], [1.0] * 6)
        self.assertEqual(identity["training_crop"], "clean_R1_unchanged")
        self.assertFalse(identity["stacked_complete_target_crop"])
        self.assertFalse(identity["stacked_loss_change"])
        self.assertFalse(identity["stacked_center_head"])
        self.assertEqual(identity["train_count"], 640)
        self.assertEqual(identity["val_count"], 160)
        self.assertEqual(identity["state_contract"]["total_state_key_count"], 572)
        self.assertEqual(identity["selection_roles"], ["best_mIoU", "best_Pd"])
        self.assertIsNone(identity["selection_margin_raw"])
        self.assertFalse(identity["test_split_accessed"])
        unhashed = dict(identity)
        digest = unhashed.pop("identity_sha256")
        self.assertEqual(digest, runner._canonical_sha256(unhashed))

    def test_zero_margin_adapter_uses_complete_keys_and_retains_both_roles(self) -> None:
        history = [
            _record(1, miou=.7000, niou=.80, pd=.80, fa=.02, tiny_pd=.70, loss=.1),
            _record(2, miou=.6999, niou=.60, pd=.99, fa=.01, tiny_pd=.95, loss=.2),
            _record(3, miou=.7000, niou=.75, pd=.90, fa=.03, tiny_pd=.80, loss=.3),
        ]
        provenance = runner.ZERO_MARGIN_SELECTION.select_independent_checkpoint(history)
        self.assertEqual(provenance["primary_selected_epoch"], 3)
        self.assertEqual(provenance["roles"]["best_Pd"]["selected"]["epoch"], 2)
        self.assertFalse(provenance["window_applied"])
        self.assertIsNone(provenance["selection_margin_raw"])
        self.assertEqual(
            runner.ZERO_MARGIN_SELECTION.retention_frontier_epochs(history), (2, 3)
        )
        artifacts = {
            2: {"relative_path": "candidates/epoch_0002.pth.tar", "file_sha256": "2" * 64},
            3: {"relative_path": "candidates/epoch_0003.pth.tar", "file_sha256": "3" * 64},
        }
        with runner._variant_runtime():
            payload = runner.r1.build_selection_payload(history, artifacts)
        self.assertEqual(payload["selected_epoch"], 3)
        self.assertEqual(payload["retention_frontier_epochs"], [2, 3])
        self.assertEqual(payload["selection_provenance"], provenance)
        with self.assertRaisesRegex(Exception, "window"):
            runner.zero_selection.select_checkpoints(history, margin=0.001)

    def test_state_contract_accepts_exact_572_and_rejects_clean_or_extra(self) -> None:
        state = self._state()
        validated = runner._validate_state_dict(state, state)
        self.assertEqual(len(validated), 572)
        self.assertEqual(
            sum(key.startswith(runner.hf_decoder.HF_DECODER_STATE_PREFIX) for key in validated),
            8,
        )
        clean = dict(list(state.items())[:564])
        with self.assertRaisesRegex(runner.HFDecoderRunnerError, "572-key"):
            runner._validate_state_dict(clean, clean)
        extra = dict(state)
        extra["other"] = torch.zeros(1)
        with self.assertRaisesRegex(runner.HFDecoderRunnerError, "572-key"):
            runner._validate_state_dict(extra, extra)
        changed = dict(state)
        changed[next(iter(changed))] = torch.zeros(2)
        with self.assertRaisesRegex(runner.HFDecoderRunnerError, "tensor contract"):
            runner._validate_state_dict(changed, state)

    def test_runtime_restores_every_patch_and_optimizer_scope_is_all_parameters(self) -> None:
        names = (
            "selection",
            "initialize_evisirst",
            "_validate_state_dict",
            "_run_identity",
            "build_datasets",
            "len",
        )
        before = {name: (hasattr(runner.r1, name), getattr(runner.r1, name, None)) for name in names}
        cpu_state = runner.r1.legacy_train._cpu_state
        with runner._variant_runtime():
            self.assertIs(runner.r1.selection, runner.ZERO_MARGIN_SELECTION)
            self.assertIs(runner.r1.initialize_evisirst, runner._variant_initialize_evisirst)
            self.assertIs(runner.r1._validate_state_dict, runner._validate_state_dict)
            self.assertIs(runner.r1.legacy_train._cpu_state, runner._variant_cpu_state)
            state = self._state()
            self.assertEqual(runner.r1.len(state), 564)
            validated = runner.r1._validate_state_dict(state, state)
            self.assertEqual(len(validated), 572)
        self.assertIs(runner.r1.legacy_train._cpu_state, cpu_state)
        for name, (existed, value) in before.items():
            self.assertEqual(hasattr(runner.r1, name), existed)
            if existed:
                self.assertIs(getattr(runner.r1, name), value)

        p0 = torch.nn.Parameter(torch.zeros(1))
        p1 = torch.nn.Parameter(torch.zeros(1))
        model = torch.nn.Module()
        model.register_parameter("base", p0)
        model.register_parameter("hf", p1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.assertEqual(optimizer.param_groups[0]["params"], [p0, p1])

    def test_dual_role_finals_bind_sha_epoch_key_state_and_are_idempotent(self) -> None:
        history = [
            _record(1, miou=.71, niou=.70, pd=.80, fa=.02, tiny_pd=.7, loss=.2),
            _record(2, miou=.70, niou=.69, pd=.95, fa=.01, tiny_pd=.9, loss=.3),
        ]
        provenance = runner.zero_selection.select_checkpoints(history)
        identity = {
            "schema": runner.TRAINING_SCHEMA + "/run_identity",
            "dataset": runner.DATASET,
            "epochs": 2,
            "identity_sha256": "a" * 64,
        }
        state1 = self._state()
        state2 = {key: value + 1 for key, value in state1.items()}
        with tempfile.TemporaryDirectory(dir=runner.PROJECT_ROOT / "runs") as temporary:
            run_dir = Path(temporary)
            candidates = run_dir / "candidates"
            candidates.mkdir()
            artifact_map = {}
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
            primary_payload = {
                "schema": runner.CHECKPOINT_SCHEMA,
                "training": identity,
                "state_dict": state1,
                "selection_provenance": provenance,
                "test_split_accessed": False,
            }
            finals = runner._atomic_write_dual_role_finals(
                run_dir=run_dir,
                args=SimpleNamespace(dataset=runner.DATASET),
                identity=identity,
                history=history,
                candidate_artifacts=artifact_map,
                expected_state=state1,
                primary_payload=primary_payload,
            )
            self.assertEqual(set(finals), {"best_mIoU", "best_Pd"})
            self.assertEqual(finals["best_mIoU"]["epoch"], 1)
            self.assertEqual(finals["best_Pd"]["epoch"], 2)
            for role in ("best_mIoU", "best_Pd"):
                path = run_dir / finals[role]["relative_path"]
                self.assertEqual(runner._sha256_file(path), finals[role]["sha256"])
                payload = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(payload["selection_role"], role)
                self.assertEqual(payload["epoch"], finals[role]["epoch"])
                self.assertEqual(
                    payload["selected_complete_key_sha256"],
                    finals[role]["selected_complete_key_sha256"],
                )
                self.assertFalse(payload["selection_window_applied"])
                self.assertFalse(payload["test_split_accessed"])
            repeated = runner._atomic_write_dual_role_finals(
                run_dir=run_dir,
                args=SimpleNamespace(dataset=runner.DATASET),
                identity=identity,
                history=history,
                candidate_artifacts=artifact_map,
                expected_state=state1,
                primary_payload=primary_payload,
            )
            self.assertEqual(repeated, finals)
            best_pd = run_dir / finals["best_Pd"]["relative_path"]
            corrupted = torch.load(best_pd, map_location="cpu", weights_only=True)
            corrupted["epoch"] = 1
            torch.save(corrupted, best_pd)
            with self.assertRaisesRegex(runner.HFDecoderRunnerError, "failed verification"):
                runner._atomic_write_dual_role_finals(
                    run_dir=run_dir,
                    args=SimpleNamespace(dataset=runner.DATASET),
                    identity=identity,
                    history=history,
                    candidate_artifacts=artifact_map,
                    expected_state=state1,
                    primary_payload=primary_payload,
                )

    def test_run_recovers_completed_transaction_without_reentering_engine(self) -> None:
        args = self._smoke("--resume")
        fake_paths = {
            "run_dir": runner.PROJECT_ROOT / "runs" / "fixture",
            "summary": runner.PROJECT_ROOT / "runs" / "fixture" / "summary.json",
        }
        expected = runner.PROJECT_ROOT / "runs" / "fixture" / "EviSIRST_best_mIoU.pth.tar"
        with mock.patch.object(runner, "resolve_run_paths", return_value=fake_paths), mock.patch.object(
            runner, "_run_process_lock"
        ) as lock, mock.patch.object(
            runner, "_finalize_completed", return_value=expected
        ) as finalize, mock.patch.object(
            runner.r1, "run", side_effect=AssertionError("engine must not reenter")
        ), mock.patch.object(Path, "exists", return_value=True):
            observed = runner.run(args)
        self.assertEqual(observed, expected)
        lock.assert_called_once_with(args)
        finalize.assert_called_once_with(args, fake_paths)

    def test_import_has_no_test_dependency_or_test_dataset_symbol(self) -> None:
        source_path = Path(runner.__file__).resolve()
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("test", imported_roots)
        self.assertNotIn("EviSIRSTTestDataset", source)
        self.assertNotIn("load_models", source)
        with mock.patch.dict("sys.modules", {"test": None, "load_models": None}):
            reloaded = importlib.reload(runner)
        self.assertEqual(reloaded.TRAINING_SCHEMA, runner.TRAINING_SCHEMA)


if __name__ == "__main__":
    unittest.main()
