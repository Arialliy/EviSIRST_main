from __future__ import annotations

import ast
import builtins
import copy
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import train_validation_selected as trainer


def validation_record(
    epoch: int, miou: float, fa: float, pd: float
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "data_role": "val",
        "mIoU": miou,
        "Fa": fa,
        "Pd": pd,
    }


class FakeNormalizationSpec:
    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "legacy",
            "source": "fixture_frozen_legacy",
            "mean": 1.0,
            "std": 2.0,
        }


class FakeTrainDataset(Dataset):
    def __init__(self, contract: SimpleNamespace, access_log: list[str]) -> None:
        self.contract = contract
        self.access_log = access_log
        self.sample_ids = ("train_fixture",)
        self.normalization = {"mean": 1.0, "std": 2.0}
        self.normalization_spec = FakeNormalizationSpec()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        self.access_log.append("train")
        image = torch.zeros(1, 32, 32, dtype=torch.float32)
        mask = torch.zeros(1, 32, 32, dtype=torch.float32)
        return image, mask


class FakeValDataset(Dataset):
    def __init__(self, contract: SimpleNamespace, access_log: list[str]) -> None:
        self.contract = contract
        self.access_log = access_log
        self.sample_ids = ("val_fixture",)
        self.normalization = {"mean": 1.0, "std": 2.0}
        self.normalization_spec = FakeNormalizationSpec()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        self.access_log.append("val")
        image = torch.zeros(1, 32, 32, dtype=torch.float32)
        mask = torch.zeros(1, 32, 32, dtype=torch.float32)
        return image, mask, (32, 32), "val_fixture"


class Fake564Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros((), dtype=torch.float32)) for _ in range(564)]
        )
        self.mode = "train"

    def forward(self, images: torch.Tensor):
        probability = torch.sigmoid(images * 0.0 + self.weights[0])
        if self.mode == "train":
            return tuple(probability for _ in range(6))
        return probability


class ValidationSelectedTrainingTest(unittest.TestCase):
    def _parse(self, *extra: str):
        return trainer.parse_args(
            [
                "--dataset",
                "NUAA-SIRST",
                "--dataset-root",
                "/tmp/data",
                "--target-mode",
                "binary",
                "--run-seed",
                "123",
                *extra,
            ]
        )

    def _contract(self, mode: str) -> SimpleNamespace:
        grouping: dict[str, object] = {"mode": mode}
        if mode == "sample_level_fallback":
            grouping["warning"] = "scene leakage cannot be ruled out"
        return SimpleNamespace(
            dataset="NUAA-SIRST",
            manifest={
                "schema": "evisirst_v2_train_val_split/v1",
                "seeds": {"split_seed": 20260811},
                "source_index": {
                    "split": "train",
                    "relative_path": (
                        "NUAA-SIRST/img_idx/train_NUAA-SIRST.txt"
                    ),
                    "file_sha256": "1" * 64,
                    "ordered_ids_sha256": "2" * 64,
                    "sample_count": 2,
                },
                "outputs": {
                    "train": {
                        "relative_path": "splits/v2/NUAA-SIRST/train.txt",
                        "file_sha256": "3" * 64,
                        "ordered_ids_sha256": "4" * 64,
                        "sample_count": 1,
                    },
                    "val": {
                        "relative_path": "splits/v2/NUAA-SIRST/val.txt",
                        "file_sha256": "5" * 64,
                        "ordered_ids_sha256": "6" * 64,
                        "sample_count": 1,
                    },
                },
                "grouping": grouping,
            },
            manifest_sha256="a" * 64,
            data_tree_sha256="b" * 64,
            data_tree_verified=True,
            train_ids=("train_fixture",),
            val_ids=("val_fixture",),
        )

    def _write_candidate(
        self,
        path: Path,
        *,
        epoch: int,
        identity: dict[str, object],
        record: dict[str, object],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": trainer.CANDIDATE_SCHEMA,
                "run_identity": identity,
                "epoch": epoch,
                "validation_record": record,
                "state_dict": {"probe": torch.tensor([float(epoch)])},
            },
            path,
        )

    def test_parse_requires_target_and_run_seed_and_freezes_r1(self) -> None:
        args = self._parse()
        self.assertEqual(args.architecture_seed, 42)
        self.assertEqual(args.run_seed, 123)
        self.assertEqual(args.target_mode, "binary")
        self.assertEqual(args.base_lr, 1e-3)
        self.assertEqual(args.min_lr, 1e-5)
        self.assertEqual(args.warmup_epochs, 10)
        self.assertEqual(args.val_interval, 1)
        invalid = (
            ["--dataset", "NUAA-SIRST", "--dataset-root", "/tmp/data"],
            [
                "--dataset",
                "NUAA-SIRST",
                "--dataset-root",
                "/tmp/data",
                "--target-mode",
                "binary",
            ],
            [
                "--dataset",
                "NUAA-SIRST",
                "--dataset-root",
                "/tmp/data",
                "--target-mode",
                "binary",
                "--run-seed",
                "-1",
            ],
            [
                "--dataset",
                "NUAA-SIRST",
                "--dataset-root",
                "/tmp/data",
                "--target-mode",
                "binary",
                "--run-seed",
                "1",
                "--architecture-seed",
                "43",
            ],
        )
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                trainer.parse_args(argv)
        for invalid_seed in (True, -1, trainer.MAX_RUN_SEED + 1, 1.5):
            with self.subTest(seed=invalid_seed), self.assertRaises(
                trainer.ValidationSelectedTrainingError
            ):
                trainer.require_run_seed(invalid_seed)
        self.assertEqual(
            trainer.require_run_seed(trainer.MAX_RUN_SEED),
            trainer.MAX_RUN_SEED,
        )
        trainer.legacy_train.configure_determinism(trainer.MAX_RUN_SEED)

    def test_output_is_repository_bound_and_smoke_is_isolated(self) -> None:
        formal = trainer.resolve_run_paths(self._parse())
        smoke_args = self._parse(
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--smoke-max-train-samples",
            "2",
            "--smoke-max-val-samples",
            "1",
        )
        smoke = trainer.resolve_run_paths(smoke_args)
        self.assertFalse(formal["smoke"])
        self.assertTrue(smoke["smoke"])
        self.assertIn("formal", Path(formal["run_dir"]).parts)
        self.assertIn("smoke", Path(smoke["run_dir"]).parts)
        self.assertNotEqual(formal["run_dir"], smoke["run_dir"])
        forbidden_roots = (
            "/tmp/outside-repository",
            str(trainer.PROJECT_ROOT),
            str(trainer.PROJECT_ROOT / ".git"),
            str(trainer.PROJECT_ROOT / "splits"),
            str(trainer.PROJECT_ROOT / "artifacts"),
            str(trainer.PROJECT_ROOT / "evaluation"),
            str(trainer.PROJECT_ROOT / "runs"),
        )
        for forbidden in forbidden_roots:
            with self.subTest(output_root=forbidden), self.assertRaisesRegex(
                trainer.ValidationSelectedTrainingError,
                "runs directory|too broad",
            ):
                trainer.resolve_run_paths(
                    self._parse("--output-root", forbidden)
                )

        with tempfile.TemporaryDirectory(dir=trainer.PROJECT_ROOT / "runs") as root:
            redirected = Path(root) / "formal"
            redirected.symlink_to(
                trainer.PROJECT_ROOT / "splits", target_is_directory=True
            )
            with self.assertRaisesRegex(
                trainer.ValidationSelectedTrainingError, "symlink"
            ):
                trainer.resolve_run_paths(self._parse("--output-root", root))

        with tempfile.TemporaryDirectory(
            dir=trainer.PROJECT_ROOT / "runs"
        ) as declared, tempfile.TemporaryDirectory(
            dir=trainer.PROJECT_ROOT / "runs"
        ) as horizontal_target:
            (Path(declared) / "formal").symlink_to(
                horizontal_target, target_is_directory=True
            )
            with self.assertRaisesRegex(
                trainer.ValidationSelectedTrainingError, "symlink"
            ):
                trainer.resolve_run_paths(
                    self._parse("--output-root", declared)
                )

        with tempfile.TemporaryDirectory() as temporary:
            fake_project = Path(temporary) / "project"
            external_runs = Path(temporary) / "external-runs"
            fake_project.mkdir()
            external_runs.mkdir()
            (fake_project / "runs").symlink_to(
                external_runs, target_is_directory=True
            )
            args = self._parse(
                "--output-root",
                str(fake_project / "runs" / "validation_selected"),
            )
            with mock.patch.object(
                trainer, "PROJECT_ROOT", fake_project
            ), self.assertRaisesRegex(
                trainer.ValidationSelectedTrainingError,
                "must not be a symlink",
            ):
                trainer.resolve_run_paths(args)

    def test_sample_fallback_requires_explicit_acknowledgement(self) -> None:
        contract = self._contract("sample_level_fallback")
        with self.assertRaisesRegex(
            trainer.ValidationSelectedTrainingError,
            "allow-sample-level-fallback",
        ):
            trainer.enforce_grouping_policy(
                contract, allow_sample_level_fallback=False
            )
        accepted = trainer.enforce_grouping_policy(
            contract, allow_sample_level_fallback=True
        )
        self.assertTrue(accepted["sample_level_fallback_acknowledged"])
        explicit = trainer.enforce_grouping_policy(
            self._contract("explicit_group_mapping"),
            allow_sample_level_fallback=False,
        )
        self.assertFalse(explicit["sample_level_fallback_acknowledged"])

    def test_run_identity_binds_determinism_protocol_and_source_hashes(self) -> None:
        args = self._parse()
        contract = self._contract("explicit_group_mapping")
        grouping = {
            "mode": "explicit_group_mapping",
            "sample_level_fallback_acknowledged": False,
            "warning": None,
        }

        def identity() -> dict[str, object]:
            return trainer._run_identity(
                args,
                contract,
                grouping,
                train_count=1,
                val_count=1,
                smoke=False,
            )

        baseline = identity()
        protocol = baseline["determinism_protocol"]
        self.assertEqual(protocol["training_data"]["patch_size"], 256)
        self.assertEqual(
            protocol["training_data"]["augmentation_version"],
            trainer.v2_data.AUGMENTATION_VERSION,
        )
        evaluation = protocol["validation_evaluation"]
        self.assertEqual(evaluation["prediction_threshold_operator"], ">")
        self.assertEqual(evaluation["match_distance_operator"], "<")
        self.assertEqual(evaluation["connected_component_neighborhood"], "8-connected")
        self.assertIn("Hungarian", evaluation["assignment_algorithm"])
        self.assertEqual(len(protocol["source_tree_sha256"]), 64)
        source_files = protocol["source_files"]
        self.assertTrue(
            {
                "trainer",
                "legacy_train",
                "data",
                "selection",
                "source_protocol",
                "frozen_builder",
                "model_entry",
                "model_package",
            }.issubset(source_files)
        )
        self.assertTrue(
            any(name.startswith("model_internal/") for name in source_files)
        )

        mutations = (
            (trainer.source_protocol, "PATCH_SIZE", 128),
            (trainer.source_protocol, "TRAIN_POSITIVE_CROP_PROBABILITY", 0.25),
            (trainer.v2_data, "AUGMENTATION_VERSION", "changed"),
            (trainer, "PROBABILITY_THRESHOLD", 0.6),
            (trainer, "MATCH_RADIUS", 4.0),
            (trainer, "TINY_AREA", 10),
            (trainer, "MATCH_DISTANCE_OPERATOR", "<="),
        )
        resume_payload = {
            "schema": trainer.TRAINING_SCHEMA,
            "run_identity": baseline,
        }
        for owner, name, changed_value in mutations:
            with self.subTest(field=name), mock.patch.object(
                owner, name, changed_value
            ):
                changed = identity()
                self.assertNotEqual(
                    changed["identity_sha256"], baseline["identity_sha256"]
                )
                with self.assertRaisesRegex(
                    trainer.ValidationSelectedTrainingError,
                    "identity differs",
                ):
                    trainer.validate_resume_identity(resume_payload, changed)

        changed_sources = copy.deepcopy(protocol)
        changed_sources["source_files"]["trainer"]["sha256"] = "f" * 64
        changed_sources["source_tree_sha256"] = "e" * 64
        with mock.patch.object(
            trainer,
            "_determinism_protocol_identity",
            return_value=changed_sources,
        ):
            changed = identity()
        self.assertNotEqual(changed["identity_sha256"], baseline["identity_sha256"])
        with self.assertRaisesRegex(
            trainer.ValidationSelectedTrainingError, "identity differs"
        ):
            trainer.validate_resume_identity(resume_payload, changed)

    def test_build_datasets_constructs_only_v2_train_and_val(self) -> None:
        contract = self._contract("sample_level_fallback")
        fake_train = SimpleNamespace(contract=contract)
        fake_val = SimpleNamespace(contract=contract)
        args = SimpleNamespace(
            dataset="NUAA-SIRST",
            dataset_root=Path("/tmp/data"),
            split_root=Path("/tmp/splits"),
            target_mode="binary",
            run_seed=5,
            allow_sample_level_fallback=True,
        )
        with mock.patch.object(
            trainer, "EviSIRSTV2TrainDataset", return_value=fake_train
        ) as train_constructor, mock.patch.object(
            trainer, "EviSIRSTV2ValDataset", return_value=fake_val
        ) as val_constructor:
            observed_train, observed_val, _ = trainer.build_datasets(args)
        self.assertIs(observed_train, fake_train)
        self.assertIs(observed_val, fake_val)
        train_constructor.assert_called_once()
        val_constructor.assert_called_once()
        self.assertEqual(
            train_constructor.call_args.kwargs["normalization_mode"], "legacy"
        )
        self.assertEqual(train_constructor.call_args.kwargs["target_mode"], "binary")
        self.assertEqual(train_constructor.call_args.kwargs["run_seed"], 5)
        self.assertTrue(train_constructor.call_args.kwargs["verify_data_tree"])

    def test_import_succeeds_with_poisoned_test_dependencies(self) -> None:
        source_path = Path(trainer.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("test", imported_roots)
        self.assertNotIn("EviSIRSTTestDataset", source_path.read_text(encoding="utf-8"))
        with mock.patch.dict("sys.modules", {"test": None, "load_models": None}):
            reloaded = importlib.reload(trainer)
        self.assertEqual(reloaded.TRAINING_SCHEMA, trainer.TRAINING_SCHEMA)

        cold_import = """
import sys
sys.modules['test'] = None
sys.modules['load_models'] = None
import experiments.evisirst_data as legacy_data
class PoisonTestDataset:
    def __init__(self, *args, **kwargs):
        raise AssertionError('test dataset was constructed')
legacy_data.EviSIRSTTestDataset = PoisonTestDataset
import train_validation_selected
print(train_validation_selected.TRAINING_SCHEMA)
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(trainer.PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-c", cold_import],
                cwd=temporary,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(trainer.TRAINING_SCHEMA, completed.stdout)

    def test_frontier_file_plan_and_selection_payload(self) -> None:
        history = [
            validation_record(1, 0.700, 0.050, 0.80),
            validation_record(2, 0.690, 0.060, 0.75),
            validation_record(3, 0.720, 0.070, 0.90),
            validation_record(4, 0.710, 0.040, 0.70),
            validation_record(5, 0.720, 0.070, 0.90),
            validation_record(6, 0.680, 0.030, 0.50),
        ]
        plan = trainer.frontier_file_plan(history, "/tmp/candidates")
        self.assertEqual(plan["frontier_epochs"], (3, 4, 6))
        self.assertEqual(
            [path.name for path in plan["keep"]],
            ["epoch_0003.pth.tar", "epoch_0004.pth.tar", "epoch_0006.pth.tar"],
        )
        artifacts = {
            epoch: {
                "relative_path": f"candidates/epoch_{epoch:04d}.pth.tar",
                "file_sha256": str(epoch) * 64,
            }
            for epoch in (3, 4, 6)
        }
        payload = trainer.build_selection_payload(history, artifacts)
        self.assertEqual(payload["data_role"], "val")
        self.assertIn(payload["selected_epoch"], (3, 4, 6))
        self.assertFalse(payload["selection_is_optimistic"])
        self.assertFalse(payload["optimistic"])
        self.assertEqual(
            payload["selection_provenance"]["rule_version"],
            trainer.selection.INDEPENDENT_RULE_VERSION,
        )

    def test_final_payload_is_clean_and_validation_selected(self) -> None:
        history = [validation_record(3, 0.8, 0.01, 0.9)]
        artifacts = {
            3: {
                "relative_path": "candidates/epoch_0003.pth.tar",
                "file_sha256": "c" * 64,
            }
        }
        selected = trainer.build_selection_payload(history, artifacts)
        split = {
            "split_seed": 20260811,
            "manifest_sha256": "a" * 64,
            "data_tree_sha256": "b" * 64,
            "data_tree_verified": True,
        }
        args = SimpleNamespace(
            dataset="NUAA-SIRST", run_seed=77, target_mode="binary"
        )
        payload = trainer.build_final_checkpoint_payload(
            args=args,
            state_dict={"probe": torch.tensor([1.0])},
            normalization={"mean": 1.0, "std": 2.0},
            normalization_provenance={
                "mode": "legacy",
                "source": "frozen",
                "mean": 1.0,
                "std": 2.0,
            },
            identity={"identity_sha256": "d" * 64},
            split_provenance=split,
            selection_payload=selected,
            model_metadata={},
            smoke=False,
        )
        self.assertEqual(payload["schema"], "evisirst_clean_checkpoint/v1")
        self.assertEqual(payload["checkpoint_role"], "validation_selected")
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["run_seed"], 77)
        self.assertEqual(payload["split_manifest_sha256"], "a" * 64)
        self.assertEqual(payload["normalization_provenance"]["mode"], "legacy")
        self.assertEqual(payload["training_identity_sha256"], "d" * 64)
        self.assertEqual(payload["source_selection"], "evisirst_v2_validation_split")
        self.assertFalse(payload["selection_is_optimistic"])
        self.assertFalse(payload["optimistic"])
        self.assertFalse(payload["test_split_accessed"])

    def test_one_epoch_fake_run_consumes_only_train_val_and_round_trips(self) -> None:
        from experiments import evisirst_data as legacy_data
        from experiments import three_dataset_v2_protocol as source_protocol

        runs_root = trainer.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        access_log: list[str] = []
        opened_paths: list[str] = []
        contract = self._contract("sample_level_fallback")
        fake_train = FakeTrainDataset(contract, access_log)
        fake_val = FakeValDataset(contract, access_log)
        original_open = builtins.open

        def guarded_open(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                identifier = os.fspath(file)
                if isinstance(identifier, bytes):
                    identifier = os.fsdecode(identifier)
                normalized = identifier.replace("\\", "/").lower()
                if "/img_idx/test_" in normalized:
                    raise AssertionError(f"forbidden test-index access: {identifier}")
                opened_paths.append(identifier)
            return original_open(file, *args, **kwargs)

        with tempfile.TemporaryDirectory(
            dir=runs_root, prefix="validation_selected_fixture_"
        ) as temporary:
            args = self._parse(
                "--output-root",
                temporary,
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--warmup-epochs",
                "0",
                "--batch-size",
                "1",
                "--allow-sample-level-fallback",
            )
            poison = mock.Mock(
                side_effect=AssertionError("test dataset must never be constructed")
            )
            with mock.patch.object(
                trainer,
                "build_datasets",
                return_value=(
                    fake_train,
                    fake_val,
                    {
                        "mode": "sample_level_fallback",
                        "sample_level_fallback_acknowledged": True,
                        "warning": "fixture acknowledgement",
                    },
                ),
            ), mock.patch.object(
                trainer,
                "initialize_evisirst",
                return_value=(Fake564Model(), {"fixture": True}),
            ), mock.patch.object(
                source_protocol,
                "load_index",
                side_effect=AssertionError("no index loader is allowed in fake run"),
            ), mock.patch.object(
                legacy_data, "EviSIRSTTestDataset", poison
            ), mock.patch.object(
                builtins, "open", side_effect=guarded_open
            ):
                checkpoint = trainer.run(args)

            self.assertTrue(checkpoint.is_file())
            self.assertEqual(access_log, ["train", "val"])
            poison.assert_not_called()
            self.assertFalse(
                any(
                    "/img_idx/test_" in path.replace("\\", "/").lower()
                    for path in opened_paths
                )
            )

            # This is the actual public custom-checkpoint bridge, exercised with
            # a complete 564-key state rather than a mocked payload validator.
            import test as checkpoint_bridge

            state, payload, checkpoint_sha256 = (
                checkpoint_bridge._state_from_checkpoint(checkpoint)
            )
            self.assertEqual(len(state), 564)
            self.assertEqual(payload["checkpoint_role"], "validation_selected")
            self.assertEqual(payload["split_manifest_sha256"], "a" * 64)
            self.assertEqual(
                payload["training_identity_sha256"],
                payload["training"]["identity_sha256"],
            )
            self.assertFalse(payload["smoke"])
            self.assertEqual(len(checkpoint_sha256), 64)

    def test_resume_identity_is_exact(self) -> None:
        identity = {"run_seed": 1, "target_mode": "binary"}
        payload = {"schema": trainer.TRAINING_SCHEMA, "run_identity": identity}
        self.assertIs(trainer.validate_resume_identity(payload, identity), payload)
        with self.assertRaisesRegex(
            trainer.ValidationSelectedTrainingError, "identity differs"
        ):
            trainer.validate_resume_identity(
                payload, {"run_seed": 2, "target_mode": "binary"}
            )

    def test_resume_rejects_optimizer_rng_and_history_tampering(self) -> None:
        def resume_identity(total_epochs: int) -> dict[str, object]:
            return {
                "identity": "strict",
                "epochs": total_epochs,
                "batch_size": 1,
                "train_count": 1,
                "base_lr": 1e-3,
                "min_lr": 1e-3,
                "warmup_epochs": 0,
            }

        identity = resume_identity(2)
        trainer.legacy_train.configure_determinism(9)
        valid_rng = trainer.legacy_train._capture_rng_state(torch.device("cpu"))

        def initialized_optimizer(model: nn.Module) -> torch.optim.Optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return optimizer

        def payload_for(model: nn.Module, optimizer_state: object, rng: object):
            return {
                "schema": trainer.TRAINING_SCHEMA,
                "run_identity": identity,
                "epoch": 1,
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "optimizer": optimizer_state,
                "training_history": [{"epoch": 1}],
                "validation_history": [],
                "candidate_artifacts": {},
                "rng": rng,
            }

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            candidate_dir = run_dir / "candidates"
            latest = run_dir / "last_training_state.pth.tar"

            model = Fake564Model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            torch.save(payload_for(model, [], valid_rng), latest)
            with self.assertRaisesRegex(
                trainer.ValidationSelectedTrainingError, "optimizer"
            ):
                trainer._load_resume_state(
                    path=latest,
                    run_dir=run_dir,
                    candidate_dir=candidate_dir,
                    identity=identity,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    total_epochs=2,
                    val_interval=2,
                )

            model = Fake564Model()
            saved_optimizer = initialized_optimizer(model)
            torch.save(
                payload_for(
                    model,
                    saved_optimizer.state_dict(),
                    {"device_type": "cpu"},
                ),
                latest,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with self.assertRaisesRegex(ValueError, "NumPy RNG state is malformed"):
                trainer._load_resume_state(
                    path=latest,
                    run_dir=run_dir,
                    candidate_dir=candidate_dir,
                    identity=identity,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    total_epochs=2,
                    val_interval=2,
                )

            final_identity = resume_identity(1)
            model = Fake564Model()
            saved_optimizer = initialized_optimizer(model)
            torch.save(
                {
                    **payload_for(
                        model, saved_optimizer.state_dict(), valid_rng
                    ),
                    "run_identity": final_identity,
                },
                latest,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            start_epoch, train_history, val_history, artifacts = (
                trainer._load_resume_state(
                    path=latest,
                    run_dir=run_dir,
                    candidate_dir=candidate_dir,
                    identity=final_identity,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    total_epochs=1,
                    val_interval=2,
                )
            )
            self.assertEqual(start_epoch, 2)
            self.assertEqual([item["epoch"] for item in train_history], [1])
            self.assertEqual(val_history, [])
            self.assertEqual(artifacts, {})

        with self.assertRaisesRegex(
            trainer.ValidationSelectedTrainingError, "training history"
        ):
            trainer._validate_resume_history(
                completed_epoch=2,
                interval=1,
                training_history=[{"epoch": 1}],
                validation_history=[
                    validation_record(1, 0.7, 0.1, 0.8),
                    validation_record(2, 0.8, 0.1, 0.8),
                ],
            )

    def test_resume_rejects_adam_hyperparameter_and_state_tampering(self) -> None:
        identity = {
            "identity": "strict-adam",
            "epochs": 2,
            "batch_size": 1,
            "train_count": 1,
            "base_lr": 1e-3,
            "min_lr": 1e-3,
            "warmup_epochs": 0,
        }
        trainer.legacy_train.configure_determinism(13)
        valid_rng = trainer.legacy_train._capture_rng_state(torch.device("cpu"))
        source_model = Fake564Model()
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-3)
        for _ in range(2):
            for parameter in source_model.parameters():
                parameter.grad = torch.ones_like(parameter)
            source_optimizer.step()
            source_optimizer.zero_grad(set_to_none=True)
        valid_optimizer_state = source_optimizer.state_dict()
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in source_model.state_dict().items()
        }

        def load(saved_optimizer: object) -> tuple[int, torch.optim.Optimizer]:
            model = Fake564Model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            payload = {
                "schema": trainer.TRAINING_SCHEMA,
                "run_identity": identity,
                "epoch": 2,
                "state_dict": model_state,
                "optimizer": saved_optimizer,
                "training_history": [{"epoch": 1}, {"epoch": 2}],
                "validation_history": [],
                "candidate_artifacts": {},
                "rng": valid_rng,
            }
            torch.save(payload, latest)
            start_epoch, _, _, _ = trainer._load_resume_state(
                path=latest,
                run_dir=run_dir,
                candidate_dir=candidate_dir,
                identity=identity,
                model=model,
                optimizer=optimizer,
                device=torch.device("cpu"),
                total_epochs=2,
                val_interval=3,
            )
            return start_epoch, optimizer

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            candidate_dir = run_dir / "candidates"
            latest = run_dir / "last_training_state.pth.tar"

            start_epoch, loaded_optimizer = load(valid_optimizer_state)
            self.assertEqual(start_epoch, 3)
            self.assertEqual(len(loaded_optimizer.state), 564)

            tampered_states: list[tuple[str, dict[str, object]]] = []
            bad_betas = copy.deepcopy(valid_optimizer_state)
            bad_betas["param_groups"][0]["betas"] = (0.8, 0.999)
            tampered_states.append(("betas", bad_betas))
            bad_decay = copy.deepcopy(valid_optimizer_state)
            bad_decay["param_groups"][0]["weight_decay"] = 0.1
            tampered_states.append(("weight_decay", bad_decay))
            bad_nan = copy.deepcopy(valid_optimizer_state)
            bad_nan["state"][0]["exp_avg"] = torch.tensor(float("nan"))
            tampered_states.append(("non-finite", bad_nan))
            bad_shape = copy.deepcopy(valid_optimizer_state)
            bad_shape["state"][0]["exp_avg_sq"] = torch.zeros(2)
            tampered_states.append(("shape", bad_shape))
            bad_group = copy.deepcopy(valid_optimizer_state)
            parameters = bad_group["param_groups"][0]["params"]
            parameters[0], parameters[1] = parameters[1], parameters[0]
            tampered_states.append(("param IDs", bad_group))
            empty_state = copy.deepcopy(valid_optimizer_state)
            empty_state["state"] = {}
            tampered_states.append(("empty state", empty_state))
            missing_state = copy.deepcopy(valid_optimizer_state)
            missing_state["state"].pop(563)
            tampered_states.append(("missing state", missing_state))
            low_step = copy.deepcopy(valid_optimizer_state)
            low_step["state"][0]["step"] = torch.tensor(1.0)
            tampered_states.append(("low step", low_step))
            aliased_moments = copy.deepcopy(valid_optimizer_state)
            aliased_moments["state"][0]["exp_avg_sq"] = aliased_moments[
                "state"
            ][0]["exp_avg"]
            tampered_states.append(("share storage", aliased_moments))

            for name, tampered in tampered_states:
                with self.subTest(tamper=name), self.assertRaisesRegex(
                    trainer.ValidationSelectedTrainingError, "Adam"
                ):
                    load(tampered)

    def test_frozen_r1_inactive_adam_contract_is_explicit(self) -> None:
        names = trainer.R1_STRUCTURALLY_INACTIVE_PARAMETER_NAMES
        self.assertEqual(len(names), 66)
        self.assertIn("mtc.embeddings_3.position_embeddings", names)
        self.assertIn("mtc.embeddings_4.position_embeddings", names)
        self.assertIn("mtc.encoder.layer.0.channel_attn.q1_attn1", names)
        self.assertIn("mtc.encoder.layer.3.channel_attn.q4_attn4", names)

    @pytest.mark.integration
    def test_real_frozen_model_binds_416_active_adam_states(self) -> None:
        model, _metadata = trainer.initialize_evisirst(
            "NUAA-SIRST", seed=trainer.ARCHITECTURE_SEED, training=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        state = optimizer.state_dict()
        parameter_ids = state["param_groups"][0]["params"]
        live_parameters = optimizer.param_groups[0]["params"]
        active = trainer._expected_adam_state_ids(
            model=model,
            live_parameters=live_parameters,
            parameter_ids=parameter_ids,
            identity={"model": "EviSIRST"},
        )
        name_by_object_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        inactive_names = {
            name_by_object_id[id(parameter)]
            for parameter_id, parameter in zip(parameter_ids, live_parameters)
            if parameter_id not in active
        }
        self.assertEqual(len(parameter_ids), 482)
        self.assertEqual(len(active), 416)
        self.assertEqual(
            inactive_names, trainer.R1_STRUCTURALLY_INACTIVE_PARAMETER_NAMES
        )

    def test_before_commit_orphan_candidate_is_validated_then_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            candidate_dir = run_dir / "candidates"
            identity = {"identity": "run"}
            committed = validation_record(1, 0.70, 0.05, 0.80)
            orphan = validation_record(2, 0.80, 0.04, 0.90)
            first = candidate_dir / "epoch_0001.pth.tar"
            second = candidate_dir / "epoch_0002.pth.tar"
            self._write_candidate(
                first, epoch=1, identity=identity, record=committed
            )
            self._write_candidate(
                second, epoch=2, identity=identity, record=orphan
            )
            artifacts = {
                1: {
                    "relative_path": "candidates/epoch_0001.pth.tar",
                    "file_sha256": trainer._sha256_file(first),
                }
            }
            with mock.patch.object(
                trainer, "_validate_state_dict", return_value={}
            ):
                trainer._validate_resume_candidates(
                    run_dir=run_dir,
                    candidate_dir=candidate_dir,
                    identity=identity,
                    history=[committed],
                    artifacts=artifacts,
                    expected_state={},
                    completed_epoch=1,
                )
            self.assertTrue(first.is_file())
            self.assertFalse(second.exists())

    def test_after_commit_dominated_candidate_is_validated_then_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            candidate_dir = run_dir / "candidates"
            identity = {"identity": "run"}
            dominated = validation_record(1, 0.70, 0.05, 0.80)
            frontier = validation_record(2, 0.80, 0.04, 0.90)
            first = candidate_dir / "epoch_0001.pth.tar"
            second = candidate_dir / "epoch_0002.pth.tar"
            self._write_candidate(
                first, epoch=1, identity=identity, record=dominated
            )
            self._write_candidate(
                second, epoch=2, identity=identity, record=frontier
            )
            artifacts = {
                2: {
                    "relative_path": "candidates/epoch_0002.pth.tar",
                    "file_sha256": trainer._sha256_file(second),
                }
            }
            with mock.patch.object(
                trainer, "_validate_state_dict", return_value={}
            ):
                trainer._validate_resume_candidates(
                    run_dir=run_dir,
                    candidate_dir=candidate_dir,
                    identity=identity,
                    history=[dominated, frontier],
                    artifacts=artifacts,
                    expected_state={},
                    completed_epoch=2,
                )
            self.assertFalse(first.exists())
            self.assertTrue(second.is_file())

    def test_final_prediction_uses_out_head(self) -> None:
        early = torch.zeros(1, 1, 2, 2)
        out = torch.ones(1, 1, 2, 2)
        self.assertIs(trainer.final_prediction((early, out)), out)

    def test_nonempty_target_metrics_are_strict_json_scalars(self) -> None:
        accumulator = trainer.ValidationMetrics(
            trainer.PROBABILITY_THRESHOLD,
            trainer.MATCH_RADIUS,
            trainer.TINY_AREA,
        )
        target = np.zeros((5, 5), dtype=np.float32)
        probability = np.zeros((5, 5), dtype=np.float32)
        target[2, 2] = 1.0
        probability[2, 2] = 0.9

        accumulator.update(probability, target, loss=0.1)
        metrics = accumulator.compute()

        json.dumps(metrics, allow_nan=False, sort_keys=True)
        self.assertIs(type(metrics["tiny_target_count"]), int)
        self.assertIs(type(metrics["matched_tiny_target_count"]), int)
        self.assertIs(type(metrics["tiny_pd"]), float)
        record = trainer.build_validation_record(1, metrics)
        self.assertEqual(record["data_role"], "val")


if __name__ == "__main__":
    unittest.main()
