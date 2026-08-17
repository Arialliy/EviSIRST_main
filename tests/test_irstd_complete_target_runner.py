from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

import train_irstd_complete_target_v1 as runner


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
        self.audit_observation_count = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        self.access_log.append("train")
        self.audit_observation_count += 1
        image = torch.zeros(1, 32, 32, dtype=torch.float32)
        mask = torch.zeros(1, 32, 32, dtype=torch.float32)
        return image, mask

    def crop_audit_state_dict(self) -> dict[str, object]:
        count = self.audit_observation_count
        return {
            "schema": runner.complete_crop.CROP_AUDIT_STATE_SCHEMA,
            "augmentation_version": runner.complete_crop.AUGMENTATION_VERSION,
            "formal_num_workers": 0,
            "observation_count": count,
            "requested_counts": {"complete_target": 0, "uniform": count},
            "realized_counts": {"complete_target": 0, "uniform": count},
            "fallback_count": 0,
            "fallback_reason_counts": {},
            "cut_component_crop_count": 0,
            "total_cut_component_count": 0,
            "max_cut_component_count": 0,
            "test_split_accessed": False,
        }

    def load_crop_audit_state_dict(self, payload: dict[str, object]) -> None:
        verifier = runner.complete_crop.CropAuditAccumulator()
        verifier.load_state_dict(payload)
        self.audit_observation_count = int(payload["observation_count"])

    def crop_audit_summary(self) -> dict[str, object]:
        verifier = runner.complete_crop.CropAuditAccumulator()
        verifier.load_state_dict(self.crop_audit_state_dict())
        return verifier.compute()


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


class CompleteTargetRunnerTest(unittest.TestCase):
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

    def _contract(self) -> SimpleNamespace:
        grouping = {
            "mode": "sample_level_fallback",
            "warning": "scene leakage cannot be ruled out",
        }
        return SimpleNamespace(
            dataset=runner.DATASET,
            manifest={
                "schema": "evisirst_v2_train_val_split/v1",
                "seeds": {"split_seed": 20260811},
                "source_index": {
                    "split": "train",
                    "relative_path": "IRSTD-1K/img_idx/train_IRSTD-1K.txt",
                    "file_sha256": "1" * 64,
                    "ordered_ids_sha256": "2" * 64,
                    "sample_count": 2,
                },
                "outputs": {
                    "train": {
                        "relative_path": "splits/v2/IRSTD-1K/train.txt",
                        "file_sha256": "3" * 64,
                        "ordered_ids_sha256": "4" * 64,
                        "sample_count": 1,
                    },
                    "val": {
                        "relative_path": "splits/v2/IRSTD-1K/val.txt",
                        "file_sha256": "5" * 64,
                        "ordered_ids_sha256": "6" * 64,
                        "sample_count": 1,
                    },
                },
                "grouping": grouping,
            },
            manifest_sha256=runner.CANONICAL_IRSTD_MANIFEST_SHA256,
            data_tree_sha256=runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
            data_tree_verified=True,
            train_ids=("train_fixture",),
            val_ids=("val_fixture",),
        )

    def _assert_nested_equal(self, left: object, right: object) -> None:
        self.assertIs(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self._assert_nested_equal(left_item, right_item)
        else:
            self.assertEqual(left, right)

    def test_cli_freezes_formal_recipe_and_isolates_smoke(self) -> None:
        args = self._parse()
        self.assertEqual(args.dataset, "IRSTD-1K")
        self.assertEqual(args.target_mode, "binary")
        self.assertEqual(args.architecture_seed, 42)
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.base_lr, 1e-3)
        self.assertEqual(args.min_lr, 1e-5)
        self.assertEqual(args.warmup_epochs, 10)
        self.assertEqual(args.val_interval, 1)
        self.assertEqual(args.output_root, runner.DEFAULT_OUTPUT_ROOT)

        formal = runner.resolve_run_paths(args)
        self.assertFalse(formal["smoke"])
        self.assertEqual(
            Path(formal["run_dir"]).relative_to(runner.DEFAULT_OUTPUT_ROOT),
            Path("formal/IRSTD-1K/binary/run_seed_1446202191"),
        )
        smoke_args = self._parse(
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
        )
        smoke = runner.resolve_run_paths(smoke_args)
        self.assertTrue(smoke["smoke"])
        self.assertIn("smoke", Path(smoke["run_dir"]).parts)

        invalid = (
            ["--run-seed", "42"],
            ["--epochs", "999"],
            ["--architecture-seed", "43"],
            ["--dataset", "NUAA-SIRST"],
            ["--target-mode", "soft"],
            ["--device", "cpu"],
            ["--output-root", "/tmp/escape"],
            ["--split-root", "/tmp/not-canonical"],
        )
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self._parse(*extra)

    def test_promotion_gate_is_preregistered_without_invented_results(self) -> None:
        gate = runner.promotion_gate()
        self.assertEqual(gate["status"], "TBD")
        self.assertEqual(gate["primary"]["minimum_delta"], 0.001)
        self.assertEqual(gate["primary"]["operator"], ">=")
        self.assertEqual(gate["safety"]["pd_failure_threshold"], -0.003)
        self.assertEqual(gate["safety"]["fa_failure_threshold"], 0.0)
        self.assertEqual(
            gate["mechanism"]["pass_rule"],
            "at_least_one_metric_strictly_improves",
        )
        self.assertTrue(
            gate["decision"]["single_seed_required_before_expansion"]
        )
        self.assertFalse(gate["decision"]["public_test_allowed"])
        measurement = gate["mechanism"]["measurement_protocol"]
        self.assertEqual(
            measurement["variant_selected_record"],
            "selected_validation_record.metrics."
            "complete_target_mechanism_diagnostics",
        )
        baseline_artifact = measurement["paired_baseline_artifact"]
        self.assertEqual(
            baseline_artifact["status"],
            "TBD_pending_paired_baseline_completion",
        )
        self.assertIn("must not participate", baseline_artifact["generation_rule"])
        self.assertTrue(
            baseline_artifact["output_artifact_template"].endswith(
                "matched_target_diagnostics.json"
            )
        )
        self.assertEqual(baseline_artifact["result"], "TBD")
        self.assertEqual(
            gate["paired_baseline"]["training_schema"],
            runner._R1_TRAINING_SCHEMA,
        )
        self.assertNotEqual(
            gate["paired_baseline"]["training_schema"], runner.TRAINING_SCHEMA
        )

    def test_identity_binds_variant_runner_crop_source_policy_and_recipe(self) -> None:
        args = self._parse()
        contract = self._contract()
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture acknowledgement",
        }
        identity = runner._variant_run_identity(
            args,
            contract,
            grouping,
            train_count=640,
            val_count=160,
            smoke=False,
        )
        self.assertEqual(identity["schema"], runner.TRAINING_SCHEMA + "/run_identity")
        self.assertEqual(identity["deep_supervision_probability_heads"], 6)
        self.assertEqual(identity["deep_supervision_weights"], [1.0] * 6)
        self.assertEqual(identity["evaluation_head"], "out")
        self.assertEqual(identity["runtime_identity"]["requested_device"], "cuda:0")
        self.assertIn("torch_version", identity["runtime_identity"])
        self.assertFalse(identity["test_split_accessed"])
        self.assertEqual(identity["promotion_gate"]["status"], "TBD")
        protocol = identity["determinism_protocol"]
        self.assertEqual(protocol["schema"], runner.DETERMINISM_SCHEMA)
        self.assertEqual(
            protocol["training_data"]["crop_policy_identity_sha256"],
            runner.complete_crop.policy_identity_sha256(),
        )
        source_files = protocol["source_files"]
        self.assertEqual(
            source_files["variant_runner"]["relative_path"],
            "train_irstd_complete_target_v1.py",
        )
        self.assertEqual(
            source_files["complete_target_crop"]["relative_path"],
            "experiments/evisirst_complete_target_crop.py",
        )
        self.assertTrue(any(name.startswith("r1/model_internal/") for name in source_files))
        self.assertEqual(len(protocol["source_tree_sha256"]), 64)

        changed_hash = "f" * 64
        with mock.patch.object(
            runner.complete_crop,
            "policy_identity_sha256",
            return_value=changed_hash,
        ):
            changed = runner._variant_run_identity(
                args,
                contract,
                grouping,
                train_count=640,
                val_count=160,
                smoke=False,
            )
        self.assertNotEqual(identity["identity_sha256"], changed["identity_sha256"])
        with runner._variant_runtime(), self.assertRaisesRegex(
            runner.r1.ValidationSelectedTrainingError, "identity differs"
        ):
            runner.r1.validate_resume_identity(
                {"schema": runner.TRAINING_SCHEMA, "run_identity": identity},
                changed,
            )

    def test_build_datasets_reaches_only_complete_train_and_v2_val(self) -> None:
        args = self._parse(
            "--allow-sample-level-fallback",
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--smoke-max-train-samples",
            "1",
            "--smoke-max-val-samples",
            "1",
        )
        contract = self._contract()
        fake_train = SimpleNamespace(contract=contract)
        fake_val = SimpleNamespace(contract=contract)
        with mock.patch.object(
            runner,
            "EviSIRSTCompleteTargetTrainDataset",
            return_value=fake_train,
        ) as train_constructor, mock.patch.object(
            runner, "EviSIRSTV2ValDataset", return_value=fake_val
        ) as val_constructor:
            observed_train, observed_val, grouping = runner.build_datasets(args)
        self.assertIs(observed_train, fake_train)
        self.assertIs(observed_val, fake_val)
        self.assertTrue(grouping["sample_level_fallback_acknowledged"])
        train_constructor.assert_called_once()
        val_constructor.assert_called_once()
        self.assertEqual(train_constructor.call_args.args, ("IRSTD-1K",))
        self.assertEqual(
            train_constructor.call_args.kwargs["formal_num_workers"], 0
        )
        self.assertEqual(train_constructor.call_args.kwargs["target_mode"], "binary")
        self.assertEqual(
            train_constructor.call_args.kwargs["run_seed"],
            runner.PAIRED_RUN_SEED,
        )
        self.assertTrue(train_constructor.call_args.kwargs["verify_data_tree"])

    def test_runtime_patch_is_bounded_and_exception_safe(self) -> None:
        original = {
            "training_schema": runner.r1.TRAINING_SCHEMA,
            "build_datasets": runner.r1.build_datasets,
            "write_json": runner.r1._write_json,
            "validation_metrics": runner.r1.ValidationMetrics,
        }
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with runner._variant_runtime():
                self.assertEqual(runner.r1.TRAINING_SCHEMA, runner.TRAINING_SCHEMA)
                self.assertIs(
                    runner.r1.build_datasets, runner._engine_build_datasets
                )
                self.assertIs(
                    runner.r1.ValidationMetrics,
                    runner.CompleteTargetValidationMetrics,
                )
                raise RuntimeError("fixture failure")
        self.assertEqual(runner.r1.TRAINING_SCHEMA, original["training_schema"])
        self.assertIs(runner.r1.build_datasets, original["build_datasets"])
        self.assertIs(runner.r1._write_json, original["write_json"])
        self.assertIs(
            runner.r1.ValidationMetrics, original["validation_metrics"]
        )

    def test_run_directory_process_lock_is_nonblocking(self) -> None:
        runs_root = runner.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=runs_root, prefix="complete_target_lock_fixture_"
        ) as temporary:
            isolated_root = Path(temporary) / "complete_target_v1"
            with mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", isolated_root):
                args = self._parse(
                    "--epochs",
                    "1",
                    "--warmup-epochs",
                    "0",
                    "--smoke-max-train-samples",
                    "1",
                    "--smoke-max-val-samples",
                    "1",
                )
                with runner._run_process_lock(args) as lock_path:
                    self.assertTrue(lock_path.is_file())
                    with self.assertRaisesRegex(
                        runner.CompleteTargetRunnerError, "already locked"
                    ):
                        with runner._run_process_lock(args):
                            self.fail("a second process lock must not be acquired")

    def test_supplementary_diagnostics_do_not_change_r1_selection_metrics(self) -> None:
        target = np.zeros((8, 8), dtype=np.float32)
        probability = np.zeros((8, 8), dtype=np.float32)
        target[3:5, 3:5] = 1.0
        probability[3:5, 3:5] = 0.9
        baseline = runner._R1_VALIDATION_METRICS(
            runner.r1.PROBABILITY_THRESHOLD,
            runner.r1.MATCH_RADIUS,
            runner.r1.TINY_AREA,
        )
        variant = runner.CompleteTargetValidationMetrics(
            runner.r1.PROBABILITY_THRESHOLD,
            runner.r1.MATCH_RADIUS,
            runner.r1.TINY_AREA,
        )
        baseline.update(probability, target, 0.1)
        variant.update(probability, target, 0.1)
        expected = baseline.compute()
        observed = variant.compute()
        diagnostics = observed.pop("complete_target_mechanism_diagnostics")
        self.assertEqual(observed, expected)
        self.assertEqual(diagnostics["matched_target_pixel_recall"], 1.0)
        self.assertEqual(diagnostics["matched_component_area_ratio"], 1.0)

    def test_resume_strictly_restores_crop_audit_and_rejects_tampering(self) -> None:
        identity = {"train_count": 1}
        source = FakeTrainDataset(self._contract(), [])
        source.audit_observation_count = 1
        state = source.crop_audit_state_dict()
        summary = source.crop_audit_summary()
        payload = {
            "crop_audit_state": state,
            "crop_audit_summary": summary,
            "crop_audit_history": [{"epoch": 1, "summary": summary}],
            "crop_audit_commit_status": "committed_by_this_atomic_latest",
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "last_training_state.pth.tar"
            torch.save(payload, checkpoint)
            restored = FakeTrainDataset(self._contract(), [])
            runner._RUNTIME_STATE.train_dataset = restored
            runner._RUNTIME_STATE.crop_audit_history = []
            runner._RUNTIME_STATE.run_identity = None
            try:
                with mock.patch.object(
                    runner, "_R1_LOAD_RESUME_STATE", return_value=(2, [], [], {})
                ):
                    result = runner._variant_load_resume_state(
                        path=checkpoint, identity=identity
                    )
                self.assertEqual(result[0], 2)
                self.assertEqual(restored.audit_observation_count, 1)
                self.assertEqual(len(runner._RUNTIME_STATE.crop_audit_history), 1)

                corrupted = copy.deepcopy(payload)
                corrupted["crop_audit_summary"]["fallback_rate"] = 0.5
                torch.save(corrupted, checkpoint)
                restored.audit_observation_count = 0
                with mock.patch.object(
                    runner, "_R1_LOAD_RESUME_STATE", return_value=(2, [], [], {})
                ), self.assertRaisesRegex(
                    runner.CompleteTargetRunnerError, "rates differ"
                ):
                    runner._variant_load_resume_state(
                        path=checkpoint, identity=identity
                    )

                bad_commit = copy.deepcopy(payload)
                bad_commit["crop_audit_commit_status"] = "pre_latest_transaction_view"
                torch.save(bad_commit, checkpoint)
                with mock.patch.object(
                    runner, "_R1_LOAD_RESUME_STATE", return_value=(2, [], [], {})
                ), self.assertRaisesRegex(
                    runner.CompleteTargetRunnerError, "commit status differs"
                ):
                    runner._variant_load_resume_state(
                        path=checkpoint, identity=identity
                    )
            finally:
                for name in (
                    "train_dataset",
                    "crop_audit_history",
                    "run_identity",
                ):
                    if hasattr(runner._RUNTIME_STATE, name):
                        delattr(runner._RUNTIME_STATE, name)

    def test_crop_audit_history_rejects_nonmonotone_adjacent_counts(self) -> None:
        first = FakeTrainDataset(self._contract(), [])
        first.audit_observation_count = 1
        first_summary = first.crop_audit_summary()
        second_state = first.crop_audit_state_dict()
        second_state["observation_count"] = 2
        second_state["requested_counts"] = {"complete_target": 2, "uniform": 0}
        second_state["realized_counts"] = {"complete_target": 2, "uniform": 0}
        verifier = runner.complete_crop.CropAuditAccumulator()
        verifier.load_state_dict(second_state)
        second_summary = verifier.compute()
        with self.assertRaisesRegex(
            runner.CompleteTargetRunnerError, "increment differs|not monotone"
        ):
            runner._validate_crop_audit_history(
                [
                    {"epoch": 1, "summary": first_summary},
                    {"epoch": 2, "summary": second_summary},
                ],
                completed_epoch=2,
                identity={"train_count": 1},
            )

    def test_import_has_no_public_test_dependency(self) -> None:
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

        cold_import = """
import sys
sys.modules['test'] = None
sys.modules['load_models'] = None
import train_irstd_complete_target_v1 as variant
print(variant.TRAINING_SCHEMA)
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(runner.PROJECT_ROOT)
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
        self.assertIn(runner.TRAINING_SCHEMA, completed.stdout)

    def test_two_epoch_precommit_failure_resume_matches_continuous_run(self) -> None:
        runs_root = runner.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        contract = self._contract()
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture acknowledgement",
        }
        common_cli = (
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--warmup-epochs",
            "0",
            "--smoke-max-train-samples",
            "1",
            "--smoke-max-val-samples",
            "1",
            "--allow-sample-level-fallback",
        )

        def fixture_expected_adam_state_ids(
            *, parameter_ids: list[int], **_kwargs: object
        ) -> set[int]:
            return {parameter_ids[0]}

        with tempfile.TemporaryDirectory(
            dir=runs_root, prefix="complete_target_resume_fixture_"
        ) as temporary:
            resumed_root = Path(temporary) / "resumed"
            continuous_root = Path(temporary) / "continuous"
            interrupted_train = FakeTrainDataset(contract, [])
            interrupted_val = FakeValDataset(contract, [])
            original_evaluate = runner.r1.evaluate_model
            evaluation_calls = 0

            def fail_before_second_commit(*args: object, **kwargs: object):
                nonlocal evaluation_calls
                evaluation_calls += 1
                if evaluation_calls == 2:
                    raise RuntimeError("fixture precommit interruption")
                return original_evaluate(*args, **kwargs)

            with mock.patch.object(
                runner, "DEFAULT_OUTPUT_ROOT", resumed_root
            ):
                interrupted_args = self._parse(*common_cli)
                with (
                    mock.patch.object(
                        runner,
                        "build_datasets",
                        return_value=(interrupted_train, interrupted_val, grouping),
                    ),
                    mock.patch.object(
                        runner.r1,
                        "initialize_evisirst",
                        return_value=(Fake564Model(), {"fixture": True}),
                    ),
                    mock.patch.object(
                        runner.r1,
                        "evaluate_model",
                        side_effect=fail_before_second_commit,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "precommit interruption"
                    ),
                ):
                    runner.run(interrupted_args)

                interrupted_paths = runner.resolve_run_paths(interrupted_args)
                interrupted_latest = torch.load(
                    interrupted_paths["latest"],
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(interrupted_latest["epoch"], 1)
                self.assertEqual(
                    interrupted_latest["crop_audit_state"]["observation_count"], 1
                )
                self.assertEqual(interrupted_train.audit_observation_count, 2)

                resumed_train = FakeTrainDataset(contract, [])
                resumed_val = FakeValDataset(contract, [])
                resumed_args = self._parse(*common_cli, "--resume")
                with (
                    mock.patch.object(
                        runner,
                        "build_datasets",
                        return_value=(resumed_train, resumed_val, grouping),
                    ),
                    mock.patch.object(
                        runner.r1,
                        "initialize_evisirst",
                        return_value=(Fake564Model(), {"fixture": True}),
                    ),
                    mock.patch.object(
                        runner.r1,
                        "_expected_adam_state_ids",
                        side_effect=fixture_expected_adam_state_ids,
                    ),
                ):
                    resumed_checkpoint = runner.run(resumed_args)

            continuous_train = FakeTrainDataset(contract, [])
            continuous_val = FakeValDataset(contract, [])
            with mock.patch.object(
                runner, "DEFAULT_OUTPUT_ROOT", continuous_root
            ):
                continuous_args = self._parse(*common_cli)
                with (
                    mock.patch.object(
                        runner,
                        "build_datasets",
                        return_value=(continuous_train, continuous_val, grouping),
                    ),
                    mock.patch.object(
                        runner.r1,
                        "initialize_evisirst",
                        return_value=(Fake564Model(), {"fixture": True}),
                    ),
                ):
                    continuous_checkpoint = runner.run(continuous_args)

            resumed_payload = torch.load(
                resumed_checkpoint, map_location="cpu", weights_only=True
            )
            continuous_payload = torch.load(
                continuous_checkpoint, map_location="cpu", weights_only=True
            )
            self.assertEqual(resumed_payload["search_run_crop_audit_through_epoch"], 2)
            self.assertEqual(
                resumed_payload["selected_checkpoint_crop_audit_through_epoch"],
                resumed_payload["epoch"],
            )
            self.assertEqual(
                resumed_payload["search_run_crop_audit_state"]["observation_count"],
                2,
            )
            self.assertEqual(
                resumed_payload["selected_checkpoint_crop_audit_state"][
                    "observation_count"
                ],
                resumed_payload["epoch"],
            )
            self._assert_nested_equal(
                resumed_payload["state_dict"], continuous_payload["state_dict"]
            )
            resumed_latest = torch.load(
                resumed_checkpoint.parent / "last_training_state.pth.tar",
                map_location="cpu",
                weights_only=True,
            )
            continuous_latest = torch.load(
                continuous_checkpoint.parent / "last_training_state.pth.tar",
                map_location="cpu",
                weights_only=True,
            )
            self._assert_nested_equal(
                resumed_latest["optimizer"], continuous_latest["optimizer"]
            )
            self.assertEqual(
                resumed_latest["crop_audit_state"],
                continuous_latest["crop_audit_state"],
            )
            self.assertEqual(
                resumed_latest["crop_audit_history"],
                continuous_latest["crop_audit_history"],
            )
            self.assertEqual(
                resumed_latest["training_history"],
                continuous_latest["training_history"],
            )
            self.assertEqual(
                resumed_latest["validation_history"],
                continuous_latest["validation_history"],
            )
            self.assertEqual(resumed_train.audit_observation_count, 2)
            self.assertEqual(continuous_train.audit_observation_count, 2)
            resumed_summary = json.loads(
                (resumed_checkpoint.parent / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(resumed_summary["search_run_crop_audit_through_epoch"], 2)
            self.assertEqual(
                resumed_summary["selected_checkpoint_crop_audit_through_epoch"],
                resumed_summary["selected_epoch"],
            )

    def test_one_epoch_fake_smoke_is_validation_only_and_round_trips(self) -> None:
        runs_root = runner.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        access_log: list[str] = []
        contract = self._contract()
        fake_train = FakeTrainDataset(contract, access_log)
        fake_val = FakeValDataset(contract, access_log)

        with tempfile.TemporaryDirectory(
            dir=runs_root, prefix="complete_target_runner_fixture_"
        ) as temporary:
            isolated_root = Path(temporary) / "complete_target_v1"
            with mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", isolated_root):
                args = self._parse(
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
                    "--allow-sample-level-fallback",
                )
                with (
                    mock.patch.object(
                        runner,
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
                    ),
                    mock.patch.object(
                        runner.r1,
                        "initialize_evisirst",
                        return_value=(Fake564Model(), {"fixture": True}),
                    ),
                ):
                    checkpoint = runner.run(args)

            self.assertTrue(checkpoint.is_file())
            self.assertEqual(access_log, ["train", "val"])
            self.assertEqual(checkpoint.name, "EviSIRST.pth.tar")
            self.assertTrue(
                checkpoint.is_relative_to(isolated_root / "smoke")
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(payload["schema"], runner.CHECKPOINT_SCHEMA)
            self.assertEqual(
                payload["checkpoint_role"], "experimental_validation_selected"
            )
            self.assertEqual(payload["experiment_status"], "experimental_validation_only")
            self.assertFalse(payload["public_test_supported"])
            self.assertFalse(payload["test_split_accessed"])
            self.assertTrue(payload["smoke"])
            self.assertEqual(payload["promotion_gate"]["status"], "TBD")
            self.assertEqual(len(payload["state_dict"]), 564)
            self.assertEqual(
                payload["training_identity_sha256"],
                payload["training"]["identity_sha256"],
            )
            self.assertEqual(payload["search_run_crop_audit_through_epoch"], 1)
            self.assertEqual(
                payload["selected_checkpoint_crop_audit_through_epoch"], 1
            )
            self.assertEqual(
                payload["search_run_crop_audit_state"],
                payload["selected_checkpoint_crop_audit_state"],
            )

            summary = json.loads(
                (checkpoint.parent / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema"], runner.TRAINING_SCHEMA + "/summary")
            self.assertEqual(summary["experiment_status"], "experimental_validation_only")
            self.assertFalse(summary["public_test_supported"])
            self.assertFalse(summary["test_split_accessed"])
            self.assertEqual(summary["promotion_gate"]["status"], "TBD")
            self.assertTrue(summary["smoke"])
            selected_records = [
                record
                for record in summary["validation_history"]
                if record["epoch"] == summary["selected_epoch"]
            ]
            self.assertEqual(len(selected_records), 1)
            self.assertEqual(
                summary["selected_validation_record"], selected_records[0]
            )
            self.assertEqual(
                summary["selected_validation_record_sha256"],
                runner._canonical_sha256(selected_records[0]),
            )
            mechanism = summary["selected_validation_record"]["metrics"][
                "complete_target_mechanism_diagnostics"
            ]
            self.assertEqual(mechanism["target_component_count"], 0)
            self.assertIsNone(mechanism["matched_target_pixel_recall"])
            self.assertEqual(
                summary["search_run_crop_audit_summary"]["observation_count"], 1
            )
            self.assertEqual(len(summary["search_run_crop_audit_history"]), 1)
            self.assertEqual(summary["search_run_crop_audit_through_epoch"], 1)
            self.assertEqual(
                summary["selected_checkpoint_crop_audit_through_epoch"], 1
            )
            self.assertEqual(
                summary["crop_audit_commit_status"], "complete"
            )
            self.assertEqual(summary["run_identity"], payload["training"])
            self.assertEqual(
                summary["training_identity_sha256"],
                summary["run_identity"]["identity_sha256"],
            )
            self.assertEqual(
                summary["final_checkpoint_sha256"], runner._sha256_file(checkpoint)
            )

            latest = torch.load(
                checkpoint.parent / "last_training_state.pth.tar",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(latest["schema"], runner.TRAINING_SCHEMA)
            self.assertFalse(latest["public_test_supported"])
            self.assertFalse(latest["test_split_accessed"])
            self.assertEqual(latest["crop_audit_state"]["observation_count"], 1)
            self.assertEqual(len(latest["crop_audit_history"]), 1)
            self.assertEqual(
                latest["crop_audit_commit_status"],
                "committed_by_this_atomic_latest",
            )
            candidates = sorted((checkpoint.parent / "candidates").glob("*.pth.tar"))
            self.assertEqual(len(candidates), 1)
            candidate = torch.load(
                candidates[0], map_location="cpu", weights_only=True
            )
            self.assertEqual(candidate["schema"], runner.CANDIDATE_SCHEMA)
            self.assertEqual(candidate["crop_audit_state"]["observation_count"], 1)
            self.assertEqual(
                summary["selected_candidate_sha256"],
                runner._sha256_file(candidates[0]),
            )
            self.assertEqual(
                payload["selected_candidate_sha256"],
                summary["selected_candidate_sha256"],
            )
            self.assertEqual(
                candidate["crop_audit_commit_status"],
                "candidate_pre_latest_uncommitted",
            )
            self.assertTrue(
                (checkpoint.parent / ".complete_target_v1.lock").is_file()
            )

        # No runtime monkey patch may leak into the baseline process namespace.
        self.assertEqual(runner.r1.TRAINING_SCHEMA, runner._R1_TRAINING_SCHEMA)


if __name__ == "__main__":
    unittest.main()
