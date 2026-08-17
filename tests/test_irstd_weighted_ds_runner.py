from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import train_irstd_weighted_ds_v1 as runner


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
        return (
            torch.zeros(1, 32, 32, dtype=torch.float32),
            torch.zeros(1, 32, 32, dtype=torch.float32),
        )


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
        return (
            torch.zeros(1, 32, 32, dtype=torch.float32),
            torch.zeros(1, 32, 32, dtype=torch.float32),
            (32, 32),
            "val_fixture",
        )


class Fake564Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros((), dtype=torch.float32)) for _ in range(564)]
        )
        self.mode = "train"

    def forward(self, images: torch.Tensor):
        if self.mode == "train":
            return tuple(
                torch.sigmoid(images * 0.0 + self.weights[index])
                for index in range(6)
            )
        return torch.sigmoid(images * 0.0 + self.weights[5])


class WeightedDSRunnerTest(unittest.TestCase):
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

    def _contract(self, *, formal: bool = False) -> SimpleNamespace:
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
                "grouping": {
                    "mode": "sample_level_fallback",
                    "warning": "fixture acknowledgement",
                },
            },
            manifest_sha256=(
                runner.CANONICAL_IRSTD_MANIFEST_SHA256 if formal else "a" * 64
            ),
            data_tree_sha256=(
                runner.CANONICAL_IRSTD_DATA_TREE_SHA256 if formal else "b" * 64
            ),
            data_tree_verified=True,
            train_ids=("train_fixture",),
            val_ids=("val_fixture",),
        )

    def _canonical_payload(self, *, passed: bool) -> dict[str, object]:
        return {
            "schema": runner.CANONICAL_GATE_RESULT_SCHEMA,
            "status": "complete",
            "dataset": runner.DATASET,
            "data_role": "val",
            "architecture_seed": runner.ARCHITECTURE_SEED,
            "run_seed": runner.PAIRED_RUN_SEED,
            "epochs": runner.FORMAL_EPOCHS,
            "inputs": {
                "all_artifact_hashes_verified": True,
                "all_source_hashes_currently_verified": True,
                "canonical_train_val_split_verified": True,
                "both_selections_recomputed": True,
                "gate_evaluator_source": {
                    "relative_path": runner.canonical_gate.GATE_SOURCE_RELATIVE_PATH,
                    "sha256": runner._sha256_file(
                        Path(runner.canonical_gate.__file__)
                    ),
                },
                "paired_baseline": {"fixture": True},
                "complete_target_v1": {"fixture": True},
                "paired_baseline_matched_target_diagnostic": {"fixture": True},
            },
            "primary_gate": {"passed": passed},
            "safety_gate": {"passed": True, "failed": False},
            "mechanism_gate": {"passed": True},
            "decision": {
                "overall_passed": passed,
                "result": "PASS" if passed else "FAIL",
                "three_runtime_seed_validation_expansion_allowed": passed,
                "three_runtime_seed_validation_expansion_status": (
                    "allowed" if passed else "blocked"
                ),
                "public_test_allowed": False,
                "public_test_status": (
                    "blocked_pending_separate_reviewed_gate_extension"
                ),
            },
            "test_split_accessed": False,
            "public_test_allowed": False,
        }

    def _formal_evidence(
        self, *, result_sha256: str = "a" * 64
    ) -> dict[str, object]:
        payload = self._canonical_payload(passed=False)
        return {
            "schema": runner.PREDECESSOR_GATE_EVIDENCE_SCHEMA,
            "status": "complete",
            "authority": (
                "run_irstd_complete_target_promotion_gate."
                "validate_existing_result"
            ),
            "canonical_result_artifact": {
                "relative_path": runner.CANONICAL_GATE_RESULT_RELATIVE_PATH,
                "sha256": result_sha256,
            },
            "canonical_payload_schema": runner.CANONICAL_GATE_RESULT_SCHEMA,
            "canonical_payload_sha256": runner._canonical_sha256(payload),
            "canonical_payload": payload,
            "formal_weighted_ds_allowed": True,
            "public_test_accessed": False,
        }

    def test_loss_policy_is_exact_and_scale_preserving(self) -> None:
        policy = runner.loss_policy_identity()
        self.assertEqual(
            policy["output_order"], ["gt5", "gt4", "gt3", "gt2", "d0", "out"]
        )
        self.assertEqual(policy["weights"], [0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        self.assertEqual(sum(policy["weights"]), 6.0)
        target = torch.zeros(1, 1, 2, 2)
        probabilities = tuple(
            torch.full_like(target, value, requires_grad=True)
            for value in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8)
        )
        criterion = nn.BCELoss(reduction="mean")
        observed = runner.weighted_deep_supervision_loss(
            probabilities, target, criterion
        )
        expected = sum(
            weight * criterion(probability, target)
            for probability, weight in zip(
                probabilities, runner.DEEP_SUPERVISION_WEIGHTS, strict=True
            )
        )
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        observed.backward()
        self.assertTrue(
            all(probability.grad is not None for probability in probabilities)
        )

    def test_weighted_loss_rejects_silent_contract_drift(self) -> None:
        target = torch.zeros(1, 1, 2, 2)
        good = torch.full_like(target, 0.5)
        criterion = nn.BCELoss()
        with self.assertRaisesRegex(RuntimeError, "six probability maps"):
            runner.weighted_deep_supervision_loss((good,) * 5, target, criterion)
        with self.assertRaisesRegex(RuntimeError, "wrong shape"):
            runner.weighted_deep_supervision_loss(
                (good,) * 5 + (torch.zeros(1, 1, 1, 1),), target, criterion
            )
        bad = good.clone()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            runner.weighted_deep_supervision_loss(
                (good,) * 5 + (bad,), target, criterion
            )

    def test_formal_cli_is_frozen_and_smoke_is_isolated(self) -> None:
        formal = self._parse()
        self.assertEqual(formal.epochs, 1000)
        self.assertEqual(formal.workers, 0)
        self.assertEqual(formal.device, "cuda:0")
        self.assertEqual(formal.batch_size, 16)
        self.assertFalse(runner.resolve_run_paths(formal)["smoke"])
        with self.assertRaises(SystemExit):
            self._parse("--epochs", "999")
        with self.assertRaises(SystemExit):
            self._parse("--device", "cpu")
        smoke = self._parse(
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
        self.assertTrue(runner.resolve_run_paths(smoke)["smoke"])

    def test_promotion_gate_is_preregistered_without_results(self) -> None:
        gate = runner.promotion_gate()
        self.assertEqual(gate["status"], "TBD")
        self.assertEqual(gate["primary"]["minimum_delta"], 0.001)
        self.assertEqual(gate["safety"]["pd_failure_threshold"], -0.003)
        self.assertEqual(gate["safety"]["fa_failure_threshold"], 0.0)
        self.assertFalse(gate["decision"]["public_test_allowed"])

    def test_only_canonical_validator_decides_predecessor_gate(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("miou_delta", source)
        self.assertNotIn("safety_failed", source)
        self.assertNotIn("mechanism_passed", source)
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            result_path = repository / runner.CANONICAL_GATE_RESULT_RELATIVE_PATH
            result_path.parent.mkdir(parents=True)
            result_path.write_text("canonical fixture\n", encoding="utf-8")
            payload = self._canonical_payload(passed=False)
            with (
                mock.patch.object(runner, "PROJECT_ROOT", repository),
                mock.patch.object(
                    runner.canonical_gate, "PROJECT_ROOT", repository
                ),
                mock.patch.object(
                    runner.canonical_gate,
                    "validate_existing_result",
                    return_value=payload,
                ) as validator,
            ):
                evidence = runner.validate_canonical_predecessor_failure()
        validator.assert_called_once_with()
        self.assertEqual(
            evidence["canonical_result_artifact"]["relative_path"],
            runner.CANONICAL_GATE_RESULT_RELATIVE_PATH,
        )
        self.assertEqual(
            evidence["canonical_payload_sha256"], runner._canonical_sha256(payload)
        )
        self.assertEqual(evidence["canonical_payload"], payload)
        self.assertTrue(evidence["formal_weighted_ds_allowed"])

    def test_missing_and_tampered_canonical_results_fail_before_engine(self) -> None:
        args = self._parse()
        for error in (
            FileNotFoundError("missing canonical result"),
            runner.canonical_gate.PromotionGateError("result differs"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                mock.patch.object(
                    runner.canonical_gate,
                    "validate_existing_result",
                    side_effect=error,
                ),
                mock.patch.object(runner.r1, "run") as engine,
                self.assertRaisesRegex(
                    runner.WeightedDSRunnerError, "missing or invalid"
                ),
            ):
                runner.run(args)
            engine.assert_not_called()

    def test_pass_result_is_forbidden_before_lock_engine_or_gpu(self) -> None:
        args = self._parse()
        with (
            mock.patch.object(
                runner.canonical_gate,
                "validate_existing_result",
                return_value=self._canonical_payload(passed=True),
            ),
            mock.patch.object(runner, "_run_process_lock") as process_lock,
            mock.patch.object(runner.r1, "run") as engine,
            self.assertRaisesRegex(
                runner.WeightedDSRunnerError, "did not authorize"
            ),
        ):
            runner.run(args)
        process_lock.assert_not_called()
        engine.assert_not_called()

    def test_fail_result_is_the_only_formal_unlock(self) -> None:
        args = self._parse()
        evidence = self._formal_evidence()
        expected = Path("/tmp/weighted-fixture.pth.tar")
        with (
            mock.patch.object(
                runner,
                "validate_canonical_predecessor_failure",
                return_value=evidence,
            ) as validator,
            mock.patch.object(
                runner, "_run_process_lock", return_value=nullcontext(Path("lock"))
            ),
            mock.patch.object(
                runner, "_variant_runtime", return_value=nullcontext()
            ) as runtime,
            mock.patch.object(runner.r1, "run", return_value=expected) as engine,
        ):
            observed = runner.run(args)
        self.assertEqual(observed, expected)
        validator.assert_called_once_with()
        runtime.assert_called_once_with(evidence)
        engine.assert_called_once_with(args)

    def test_gate_result_and_source_are_bound_into_identity_and_resume(self) -> None:
        args = self._parse()
        contract = self._contract(formal=True)
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture acknowledgement",
        }
        first_evidence = self._formal_evidence(result_sha256="a" * 64)
        second_evidence = self._formal_evidence(result_sha256="b" * 64)
        with runner._variant_runtime(first_evidence):
            first = runner._variant_run_identity(
                args,
                contract,
                grouping,
                train_count=runner.CANONICAL_IRSTD_TRAIN_COUNT,
                val_count=runner.CANONICAL_IRSTD_VAL_COUNT,
                smoke=False,
            )
            payload = {"schema": runner.TRAINING_SCHEMA, "run_identity": first}
            runner.r1.validate_resume_identity(payload, first)
        with runner._variant_runtime(second_evidence):
            second = runner._variant_run_identity(
                args,
                contract,
                grouping,
                train_count=runner.CANONICAL_IRSTD_TRAIN_COUNT,
                val_count=runner.CANONICAL_IRSTD_VAL_COUNT,
                smoke=False,
            )
            with self.assertRaisesRegex(
                runner.r1.ValidationSelectedTrainingError, "identity differs"
            ):
                runner.r1.validate_resume_identity(payload, second)
        self.assertNotEqual(first["identity_sha256"], second["identity_sha256"])
        self.assertEqual(
            first["predecessor_gate_evidence"], first_evidence
        )
        sources = first["determinism_protocol"]["source_files"]
        canonical_source = sources["canonical_complete_target_gate"]
        self.assertEqual(
            canonical_source["relative_path"],
            "run_irstd_complete_target_promotion_gate.py",
        )
        self.assertEqual(
            canonical_source["sha256"],
            runner._sha256_file(Path(runner.canonical_gate.__file__)),
        )
        with mock.patch.object(
            runner, "_R1_BUILD_FINAL_CHECKPOINT_PAYLOAD", return_value={}
        ):
            final_payload = runner._variant_final_checkpoint_payload(
                identity=first
            )
        self.assertEqual(
            final_payload["predecessor_gate_evidence"], first_evidence
        )
        self.assertFalse(final_payload["public_test_supported"])
        self.assertFalse(final_payload["test_split_accessed"])

    def test_runtime_patch_is_bounded_and_exception_safe(self) -> None:
        smoke_evidence = {
            "schema": runner.PREDECESSOR_GATE_EVIDENCE_SCHEMA,
            "status": "not_applicable_to_smoke",
            "formal_weighted_ds_allowed": False,
            "public_test_accessed": False,
        }
        original_loss = runner.r1.legacy_train.deep_supervision_loss
        original_schema = runner.r1.TRAINING_SCHEMA
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with runner._variant_runtime(smoke_evidence):
                self.assertIs(
                    runner.r1.legacy_train.deep_supervision_loss,
                    runner.weighted_deep_supervision_loss,
                )
                raise RuntimeError("fixture failure")
        self.assertIs(runner.r1.legacy_train.deep_supervision_loss, original_loss)
        self.assertEqual(runner.r1.TRAINING_SCHEMA, original_schema)

    def test_one_epoch_fake_cpu_smoke_is_validation_only(self) -> None:
        runs_root = runner.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        access_log: list[str] = []
        contract = self._contract()
        fake_train = FakeTrainDataset(contract, access_log)
        fake_val = FakeValDataset(contract, access_log)
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture acknowledgement",
        }
        with tempfile.TemporaryDirectory(
            dir=runs_root, prefix="weighted_ds_runner_fixture_"
        ) as temporary:
            isolated_root = Path(temporary) / "weighted_ds_v1"
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
                        return_value=(fake_train, fake_val, grouping),
                    ),
                    mock.patch.object(
                        runner.r1,
                        "initialize_evisirst",
                        return_value=(Fake564Model(), {"fixture": True}),
                    ),
                ):
                    checkpoint = runner.run(args)
            self.assertEqual(access_log, ["train", "val"])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(payload["schema"], runner.CHECKPOINT_SCHEMA)
            self.assertFalse(payload["public_test_supported"])
            self.assertFalse(payload["test_split_accessed"])
            self.assertEqual(
                payload["predecessor_gate_evidence"]["status"],
                "not_applicable_to_smoke",
            )
            summary = json.loads(
                (checkpoint.parent / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema"], runner.TRAINING_SCHEMA + "/summary")
            self.assertFalse(summary["public_test_supported"])
            self.assertFalse(summary["test_split_accessed"])

    def test_import_has_no_public_test_dependency(self) -> None:
        source_path = Path(runner.__file__).resolve()
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("test", imported_roots)
        self.assertNotIn("EviSIRSTTestDataset", source)
        cold_import = """
import sys
sys.modules['test'] = None
sys.modules['load_models'] = None
import train_irstd_weighted_ds_v1 as variant
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


if __name__ == "__main__":
    unittest.main()
