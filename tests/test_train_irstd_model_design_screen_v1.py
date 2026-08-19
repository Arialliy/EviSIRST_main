from __future__ import annotations

import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import train_irstd_model_design_screen_v1 as runner


def _record(epoch: int = 1) -> dict[str, object]:
    return {
        "epoch": epoch,
        "data_role": "val",
        "mIoU": 0.7,
        "Pd": 0.9,
        "Fa": 1e-5,
        "evaluation_head": "out",
        "metrics": {
            "miou": 0.7,
            "niou": 0.68,
            "pd": 0.9,
            "fa": 1e-5,
            "tiny_pd": 0.8,
            "validation_loss": 0.1,
        },
    }


class ModelDesignScreenRunnerTest(unittest.TestCase):
    def _parse(self, variant: str, seed: int, *extra: str):
        return runner.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--variant",
                variant,
                "--run-seed",
                str(seed),
                *extra,
            ]
        )

    def _smoke(self, variant: str, seed: int, *extra: str):
        return self._parse(
            variant,
            seed,
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
        return SimpleNamespace(
            manifest={"seeds": {"split_seed": 20260811}},
            manifest_sha256=runner.CANONICAL_IRSTD_MANIFEST_SHA256,
            data_tree_sha256=runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
        )

    def _fake_state(self, variant: str) -> dict[str, torch.Tensor]:
        spec = runner.VARIANTS[variant]
        base_count = spec.state_key_count - spec.extension_state_key_count
        state = {
            f"base.{index}": torch.tensor([float(index)], dtype=torch.float32)
            for index in range(base_count)
        }
        if spec.extension_state_prefix is not None:
            state.update(
                {
                    f"{spec.extension_state_prefix}fixture.{index}": torch.tensor(
                        [float(index)], dtype=torch.float32
                    )
                    for index in range(spec.extension_state_key_count)
                }
            )
        return state

    def test_cli_allowlists_formal_recipe_paths_and_smoke_nonpromotion(self) -> None:
        cases = {
            "single_residual_v1": (42, 1446202191),
            "psbfr_v1": (42, 1446202191, 104728269),
        }
        for variant, seeds in cases.items():
            for seed in seeds:
                with self.subTest(variant=variant, seed=seed):
                    args = self._parse(variant, seed)
                    self.assertEqual(args.epochs, 1000)
                    self.assertEqual(args.batch_size, 16)
                    self.assertEqual(args.workers, 0)
                    self.assertEqual(args.base_lr, 1e-3)
                    self.assertEqual(args.min_lr, 1e-5)
                    self.assertEqual(args.warmup_epochs, 10)
                    self.assertEqual(args.val_interval, 1)
                    paths = runner.resolve_run_paths(args)
                    self.assertEqual(
                        Path(paths["run_dir"]).relative_to(args.output_root),
                        Path("formal")
                        / "IRSTD-1K"
                        / "binary"
                        / f"run_seed_{seed}",
                    )
            smoke = self._smoke(variant, seeds[0])
            paths = runner.resolve_run_paths(smoke)
            self.assertTrue(paths["smoke"])
            with runner._hf_adapter(smoke):
                gate = runner.promotion_gate()
            self.assertFalse(gate["promotion_eligible"])
            self.assertEqual(gate["status"], "smoke_test_only_not_promotion_eligible")

        with self.assertRaises(SystemExit):
            self._parse("single_residual_v1", 104728269)
        with self.assertRaises(SystemExit):
            self._parse("psbfr_v1", 7)
        with self.assertRaises(SystemExit):
            self._parse("psbfr_v1", 42, "--epochs", "500")
        with self.assertRaises(SystemExit):
            self._parse("psbfr_v1", 42, "--device", "cpu")

    def test_real_builders_state_counts_and_d0_manifest_disambiguation(self) -> None:
        expected = {
            "single_residual_v1": (564, 10_870_130),
            "psbfr_v1": (569, 10_871_203),
        }
        for variant, (key_count, parameter_count) in expected.items():
            args = self._smoke(variant, runner.VARIANTS[variant].allowed_run_seeds[0])
            with self.subTest(variant=variant), runner._hf_adapter(args):
                model, metadata = runner._variant_initialize_evisirst(
                    runner.DATASET, seed=42, training=True
                )
                self.assertEqual(len(model.state_dict()), key_count)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    parameter_count,
                )
                runner._validate_state_dict(model.state_dict(), model.state_dict())
                self.assertFalse(metadata["public_test_supported"])
                if variant == "single_residual_v1":
                    self.assertTrue(
                        metadata["clean_architecture_interpretation_forbidden"]
                    )
                    self.assertEqual(
                        metadata["architecture_manifest"]["only_graph_change"],
                        "remove_second_skip_addition_at_levels_1_to_4",
                    )
                else:
                    extension = [
                        key
                        for key in model.state_dict()
                        if key.startswith(runner.psbfr.PSBFR_STATE_PREFIX)
                    ]
                    self.assertEqual(len(extension), 5)

    def test_identity_binds_protocol_rules_architecture_and_r1_sources(self) -> None:
        for variant in runner.VARIANTS:
            args = self._parse(
                variant, runner.VARIANTS[variant].allowed_run_seeds[0]
            )
            with self.subTest(variant=variant), runner._hf_adapter(args):
                source = runner._variant_source_provenance()
                files = source["files"]
                self.assertEqual(
                    files["frozen_protocol"]["relative_path"],
                    "experiments/IRSTD_PSBFR_V1_PROTOCOL.md",
                )
                self.assertEqual(
                    files["frozen_screen_rules"]["relative_path"],
                    "experiments/irstd_psbfr_v1_screen_rules.json",
                )
                self.assertEqual(
                    files["selected_architecture"]["relative_path"],
                    runner.VARIANTS[variant]
                    .architecture_source.resolve()
                    .relative_to(runner.PROJECT_ROOT)
                    .as_posix(),
                )
                self.assertTrue(any(name.startswith("r1/") for name in files))
                self.assertTrue(all(len(files[name]["sha256"]) == 64 for name in files))
                identity = runner._variant_run_identity(
                    args,
                    self._contract(),
                    {"mode": "scene_group"},
                    train_count=640,
                    val_count=160,
                    smoke=False,
                )
                self.assertEqual(identity["epochs"], 1000)
                self.assertEqual(identity["formal_configured_total_epochs"], 1000)
                self.assertEqual(identity["loss"], "sum_of_six_BCELoss_mean_terms")
                self.assertEqual(identity["training_crop"], "clean_R1_unchanged")
                self.assertIsNone(identity["selection_margin_raw"])
                self.assertFalse(identity["test_split_accessed"])
                state_contract = identity["state_contract"]
                self.assertEqual(
                    state_contract["architecture_source_sha256"],
                    files["selected_architecture"]["sha256"],
                )
                if variant == "single_residual_v1":
                    self.assertTrue(
                        state_contract["clean_architecture_interpretation_forbidden"]
                    )
                unhashed = dict(identity)
                digest = unhashed.pop("identity_sha256")
                self.assertEqual(digest, runner._canonical_sha256(unhashed))

    def test_fake_one_epoch_transaction_resume_and_tamper_fail_closed(self) -> None:
        args = self._smoke("psbfr_v1", 42, "--resume")
        state = self._fake_state("psbfr_v1")
        history = [_record()]
        identity = {
            "schema": runner.TRAINING_SCHEMA + "/run_identity",
            "variant_key": "psbfr_v1",
            "epochs": 1,
            "smoke": True,
            "test_split_accessed": False,
        }
        with tempfile.TemporaryDirectory(dir=runner.PROJECT_ROOT / "runs") as temp:
            run_dir = Path(temp)
            candidates = run_dir / "candidates"
            candidates.mkdir()
            with runner._hf_adapter(args), runner._variant_runtime():
                candidate = candidates / "epoch_0001.pth.tar"
                runner._variant_atomic_torch_save(
                    candidate,
                    {
                        "schema": runner.CANDIDATE_SCHEMA,
                        "run_identity": identity,
                        "epoch": 1,
                        "validation_record": history[0],
                        "state_dict": state,
                        "test_split_accessed": False,
                    },
                )
                artifacts = {
                    1: {
                        "relative_path": "candidates/epoch_0001.pth.tar",
                        "file_sha256": runner._sha256_file(candidate),
                    }
                }
                provenance = runner.zero_selection.select_checkpoints(history)
                primary = {
                    "schema": runner.CHECKPOINT_SCHEMA,
                    "training": identity,
                    "state_dict": state,
                    "selection_provenance": provenance,
                    "test_split_accessed": False,
                }
                finals = runner._atomic_write_dual_role_finals(
                    run_dir=run_dir,
                    args=args,
                    identity=identity,
                    history=history,
                    candidate_artifacts=artifacts,
                    expected_state=state,
                    primary_payload=primary,
                )
                self.assertEqual(set(finals), {"best_mIoU", "best_Pd"})
                for evidence in finals.values():
                    payload = torch.load(
                        run_dir / evidence["relative_path"],
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assertEqual(payload["model"], "EviSIRST-PSBFR-v1")
                    self.assertFalse(payload["promotion_eligible"])
                    self.assertFalse(payload["test_split_accessed"])

                resume = {
                    "schema": runner.TRAINING_SCHEMA,
                    "run_identity": identity,
                    "epoch": 1,
                    "test_split_accessed": False,
                }
                self.assertIs(
                    runner.r1.validate_resume_identity(resume, identity), resume
                )
                tampered_resume = copy.deepcopy(resume)
                tampered_resume["run_identity"]["variant_key"] = "single_residual_v1"
                with self.assertRaisesRegex(Exception, "resume run identity differs"):
                    runner.r1.validate_resume_identity(tampered_resume, identity)

                final_path = run_dir / finals["best_mIoU"]["relative_path"]
                tampered_final = torch.load(
                    final_path, map_location="cpu", weights_only=True
                )
                tampered_final["test_split_accessed"] = True
                torch.save(tampered_final, final_path)
                with self.assertRaisesRegex(Exception, "failed verification"):
                    runner._atomic_write_dual_role_finals(
                        run_dir=run_dir,
                        args=args,
                        identity=identity,
                        history=history,
                        candidate_artifacts=artifacts,
                        expected_state=state,
                        primary_payload=primary,
                    )

    def test_no_public_test_dependency_or_500_epoch_schedule(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
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
        self.assertNotIn("FORMAL_EPOCHS = 500", source)
        self.assertIn("public_test_supported", source)


if __name__ == "__main__":
    unittest.main()
