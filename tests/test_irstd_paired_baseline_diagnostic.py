from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import run_irstd_paired_baseline_diagnostic as diagnostic


class FakeStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for index in range(564):
            self.register_buffer(f"state_{index:03d}", torch.tensor(float(index)))
        self.mode = "test"


class TwoSampleValidationDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = torch.zeros(1, 4, 4)
        target = torch.zeros(1, 4, 4)
        if index == 1:
            target[:, 1:3, 1:3] = 1.0
        return image, target, (4, 4), f"sample_{index}"


class CountingOutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mode = "test"
        self.forward_calls = 0

    def forward(self, images: torch.Tensor):
        self.forward_calls += 1
        outputs = [torch.full_like(images, 0.1) for _ in range(5)]
        outputs.append(torch.full_like(images, 0.75))
        return tuple(outputs)


def _source_provenance() -> dict[str, object]:
    files = {
        "trainer": {
            "relative_path": "train_validation_selected.py",
            "sha256": "a" * 64,
        }
    }
    return {
        "schema": "evisirst_validation_selected_source_set/v2",
        "files": files,
        "source_tree_sha256": diagnostic._sha256_bytes(
            diagnostic._canonical_json_bytes(files)
        ),
    }


def _training_identity(source: dict[str, object]) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema": diagnostic.r1.TRAINING_SCHEMA + "/run_identity",
        "model": "EviSIRST",
        "dataset": diagnostic.DATASET,
        "architecture_seed": diagnostic.ARCHITECTURE_SEED,
        "run_seed": diagnostic.RUN_SEED,
        "target_mode": diagnostic.TARGET_MODE,
        "epochs": diagnostic.EPOCHS,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "val_interval": 1,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "selection_rule": diagnostic.validation_selection.INDEPENDENT_RULE_VERSION,
        "determinism_protocol": {
            "source_files": source["files"],
            "source_tree_sha256": source["source_tree_sha256"],
        },
        "manifest_sha256": "b" * 64,
        "split_seed": 20260811,
        "data_tree_sha256": "c" * 64,
        "grouping_policy": {"mode": "fixture"},
        "train_count": 800,
        "val_count": 200,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = diagnostic._sha256_bytes(
        diagnostic._canonical_json_bytes(identity)
    )
    return identity


def _selection() -> dict[str, object]:
    records = [
        {
            "epoch": epoch,
            "data_role": "val",
            "mIoU": 0.8 if epoch == 7 else 0.1,
            "Fa": 0.0,
            "Pd": 1.0,
        }
        for epoch in range(1, diagnostic.EPOCHS + 1)
    ]
    return diagnostic.validation_selection.select_independent_checkpoint(records)


def _validation_record(epoch: int, selection: dict[str, object]):
    selected = selection["selected"]
    if epoch == selected["epoch"]:
        miou, fa, pd = selected["mIoU"], selected["Fa"], selected["Pd"]
    else:
        miou, fa, pd = 0.1, 0.0, 1.0
    return {
        "epoch": epoch,
        "data_role": "val",
        "mIoU": miou,
        "Fa": fa,
        "Pd": pd,
        "evaluation_head": "out",
        "metrics": {},
    }


class FixedCliContractTest(unittest.TestCase):
    def test_cli_exposes_no_checkpoint_output_or_split_choice(self) -> None:
        args = diagnostic.parse_args(["--dataset-root", "/data", "--device", "cpu"])
        self.assertEqual(args.device, "cpu")
        self.assertFalse(hasattr(args, "checkpoint"))
        self.assertFalse(hasattr(args, "output_json"))
        self.assertFalse(hasattr(args, "split_root"))
        with self.assertRaises(SystemExit):
            diagnostic.parse_args(["--dataset-root", "/data", "--workers", "1"])

    def test_frozen_paths_match_the_preregistered_promotion_gate(self) -> None:
        gate = __import__("train_irstd_complete_target_v1").promotion_gate()
        artifact = gate["mechanism"]["measurement_protocol"][
            "paired_baseline_artifact"
        ]
        self.assertEqual(
            diagnostic.CHECKPOINT_RELATIVE_PATH,
            artifact["input_checkpoint_relative_path"],
        )
        self.assertEqual(
            diagnostic.OUTPUT_RELATIVE_PATH,
            artifact["output_artifact_template"],
        )


class SinglePassEvaluationTest(unittest.TestCase):
    def test_one_out_forward_populates_both_fixed_metric_families(self) -> None:
        model = CountingOutModel()
        loader = DataLoader(TwoSampleValidationDataset(), batch_size=1, shuffle=False)
        r1_metrics, mechanism, digests, count = diagnostic.evaluate_out_once(
            model, loader, torch.device("cpu")
        )

        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(count, 2)
        self.assertEqual(r1_metrics["miou"], 0.125)
        self.assertEqual(r1_metrics["niou"], 0.125)
        self.assertEqual(mechanism["matched_component_count"], 1)
        self.assertEqual(mechanism["matched_target_pixel_recall"], 1.0)
        self.assertEqual(set(digests), {"prediction_sha256", "target_sha256"})
        self.assertTrue(all(len(value) == 64 for value in digests.values()))

    def test_prediction_and_target_digests_are_reproducible(self) -> None:
        loader = DataLoader(TwoSampleValidationDataset(), batch_size=1, shuffle=False)
        first = diagnostic.evaluate_out_once(
            CountingOutModel(), loader, torch.device("cpu")
        )
        second = diagnostic.evaluate_out_once(
            CountingOutModel(), loader, torch.device("cpu")
        )
        self.assertEqual(first, second)


class FinalCheckpointLoadingTest(unittest.TestCase):
    def _write_completed_fixture(self, repository: Path) -> tuple[Path, FakeStateModel]:
        source = _source_provenance()
        identity = _training_identity(source)
        selection = _selection()
        selected_epoch = selection["selected"]["epoch"]
        split = {
            "schema": "fixture/v1",
            "manifest_relative_path": diagnostic.SPLIT_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": identity["manifest_sha256"],
            "split_seed": identity["split_seed"],
            "source_index": {},
            "outputs": {
                "train": {},
                "val": {"file_sha256": "d" * 64},
            },
            "grouping": {},
            "data_tree_sha256": identity["data_tree_sha256"],
            "data_tree_verified": True,
            "train_count": identity["train_count"],
            "val_count": identity["val_count"],
            "test_index_opened": False,
        }
        model = FakeStateModel()
        checkpoint = repository / diagnostic.CHECKPOINT_RELATIVE_PATH
        checkpoint.parent.mkdir(parents=True)
        payload = {
            "schema": diagnostic.r1.CHECKPOINT_SCHEMA,
            "model": "EviSIRST",
            "dataset": diagnostic.DATASET,
            "checkpoint_role": "validation_selected",
            "epoch": selected_epoch,
            "seed": diagnostic.ARCHITECTURE_SEED,
            "architecture_seed": diagnostic.ARCHITECTURE_SEED,
            "run_seed": diagnostic.RUN_SEED,
            "state_dict": model.state_dict(),
            "target_mode": diagnostic.TARGET_MODE,
            "normalization": {"mean": 101.0, "std": 34.0},
            "normalization_provenance": {},
            "training": identity,
            "training_identity_sha256": identity["identity_sha256"],
            "split_provenance": split,
            "split_seed": split["split_seed"],
            "split_manifest_sha256": split["manifest_sha256"],
            "data_tree_sha256": split["data_tree_sha256"],
            "data_tree_verified": True,
            "selection_provenance": selection,
            "source_selection": "evisirst_v2_validation_split",
            "selection_is_optimistic": False,
            "optimistic": False,
            "test_split_accessed": False,
            "model_metadata": {},
            "smoke": False,
        }
        torch.save(payload, checkpoint)
        summary = {
            "schema": diagnostic.r1.TRAINING_SCHEMA + "/summary",
            "status": "complete",
            "dataset": diagnostic.DATASET,
            "checkpoint": diagnostic.CHECKPOINT_RELATIVE_PATH,
            "checkpoint_role": "validation_selected",
            "selected_epoch": selected_epoch,
            "architecture_seed": diagnostic.ARCHITECTURE_SEED,
            "run_seed": diagnostic.RUN_SEED,
            "target_mode": diagnostic.TARGET_MODE,
            "split_provenance": split,
            "selection": {
                "selected_epoch": selected_epoch,
                "selection_provenance": selection,
                "selection_is_optimistic": False,
                "optimistic": False,
            },
            "training_history": [
                {"epoch": epoch} for epoch in range(1, diagnostic.EPOCHS + 1)
            ],
            "validation_history": [
                _validation_record(epoch, selection)
                for epoch in range(1, diagnostic.EPOCHS + 1)
            ],
            "candidate_artifacts": {},
            "normalization": payload["normalization"],
            "source_selection": "evisirst_v2_validation_split",
            "selection_is_optimistic": False,
            "optimistic": False,
            "test_split_accessed": False,
            "smoke": False,
            "elapsed_seconds": 1.0,
        }
        summary_path = repository / diagnostic.SUMMARY_RELATIVE_PATH
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return checkpoint, model

    def test_completed_formal_final_strict_loads_and_binds_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            _checkpoint, fixture_model = self._write_completed_fixture(repository)
            source = _source_provenance()
            with (
                mock.patch.object(diagnostic, "PROJECT_ROOT", repository),
                mock.patch.object(
                    diagnostic.r1,
                    "_protocol_source_provenance",
                    return_value=source,
                ),
                mock.patch.object(
                    diagnostic,
                    "initialize_evisirst",
                    return_value=(FakeStateModel(), {"fixture": True}),
                ),
            ):
                model, metadata = diagnostic.load_fixed_final_model(
                    torch.device("cpu")
                )

            self.assertEqual(metadata["selected_epoch"], 7)
            self.assertEqual(len(metadata["sha256"]), 64)
            self.assertEqual(
                metadata["training_source_tree_sha256"],
                source["source_tree_sha256"],
            )
            self.assertEqual(
                model.state_dict()["state_563"],
                fixture_model.state_dict()["state_563"],
            )

    def test_incomplete_selection_provenance_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            diagnostic.PairedBaselineDiagnosticError, "not completed 1000"
        ):
            diagnostic._validate_selection_provenance(
                {
                    "evaluated_records": [
                        {
                            "epoch": 1,
                            "data_role": "val",
                            "mIoU": 0.1,
                            "Fa": 0.0,
                            "Pd": 1.0,
                        }
                    ]
                },
                selected_epoch=1,
            )


class ResultAndWriteContractTest(unittest.TestCase):
    def _dataset(self):
        return SimpleNamespace(
            contract=SimpleNamespace(
                manifest_sha256="a" * 64,
                data_tree_sha256="b" * 64,
                data_tree_verified=True,
                train_ids=tuple(range(4)),
                val_ids=tuple(range(2)),
            )
        )

    def test_payload_is_descriptive_and_contains_full_r1_plus_mechanism(self) -> None:
        r1_metrics = {
            "validation_loss": 0.1,
            "miou": 0.5,
            "niou": 0.4,
            "pixel_precision": 0.8,
            "pixel_recall": 0.7,
            "pixel_f1": 0.746,
            "pd": 0.9,
            "tiny_pd": 0.5,
            "fa": 0.001,
            "false_objects_per_image": 0.2,
            "target_count": 10,
            "matched_target_count": 9,
            "tiny_target_count": 2,
            "matched_tiny_target_count": 1,
            "predicted_object_count": 10,
            "unmatched_predicted_object_count": 1,
            "valid_pixel_count": 100,
        }
        mechanism = {
            "schema": "evisirst_matched_target_diagnostics/v1/aggregate",
            "matched_target_pixel_recall": 0.8,
            "matched_component_area_ratio": 1.1,
        }
        payload = diagnostic.build_result_payload(
            checkpoint_metadata={
                "sha256": "c" * 64,
                "schema": diagnostic.r1.CHECKPOINT_SCHEMA,
                "selected_epoch": 7,
                "training_identity_sha256": "d" * 64,
                "training_source_tree_sha256": "e" * 64,
                "selection_provenance_sha256": "f" * 64,
                "summary": {
                    "relative_path": diagnostic.SUMMARY_RELATIVE_PATH,
                    "sha256": "1" * 64,
                    "selected_validation_record_sha256": "2" * 64,
                },
                "split_provenance": {
                    "outputs": {"val": {"file_sha256": "3" * 64}}
                },
            },
            dataset=self._dataset(),
            r1_metrics=r1_metrics,
            mechanism_metrics=mechanism,
            digests={"prediction_sha256": "4" * 64, "target_sha256": "5" * 64},
            sample_count=2,
            diagnostic_sources={
                "schema": diagnostic.DIAGNOSTIC_SOURCE_SCHEMA,
                "files": {},
                "source_tree_sha256": "6" * 64,
            },
        )
        self.assertTrue(payload["diagnostic_only"])
        self.assertFalse(payload["selection_allowed"])
        self.assertFalse(payload["test_split_accessed"])
        self.assertEqual(payload["metrics"]["miou"], 0.5)
        self.assertEqual(
            payload["metrics"]["complete_target_mechanism_diagnostics"],
            mechanism,
        )
        self.assertFalse(payload["evaluation"]["prediction_or_target_arrays_written"])
        self.assertNotIn("/home/", json.dumps(payload, sort_keys=True))

    def test_atomic_output_is_fixed_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "runs").mkdir()
            payload = {"schema": "fixture/v1", "test_split_accessed": False}
            with mock.patch.object(diagnostic, "PROJECT_ROOT", repository):
                output = diagnostic._write_fixed_json_atomic(payload)
                self.assertEqual(
                    output.relative_to(repository).as_posix(),
                    diagnostic.OUTPUT_RELATIVE_PATH,
                )
                self.assertEqual(json.loads(output.read_text()), payload)
                with self.assertRaises(FileExistsError):
                    diagnostic._write_fixed_json_atomic(payload)
                self.assertEqual(
                    [path.name for path in output.parent.iterdir()],
                    ["matched_target_diagnostics.json"],
                )

    def test_atomic_output_concurrent_publish_has_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "runs").mkdir()
            barrier = threading.Barrier(8)

            def publish(writer: int):
                barrier.wait(timeout=5.0)
                try:
                    output = diagnostic._write_fixed_json_atomic(
                        {"schema": "fixture/v1", "writer": writer}
                    )
                except FileExistsError:
                    return ("refused", writer, None)
                return ("published", writer, output)

            with mock.patch.object(diagnostic, "PROJECT_ROOT", repository):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    outcomes = list(executor.map(publish, range(8)))

            winners = [outcome for outcome in outcomes if outcome[0] == "published"]
            refused = [outcome for outcome in outcomes if outcome[0] == "refused"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(len(refused), 7)
            output = winners[0][2]
            self.assertIsInstance(output, Path)
            self.assertEqual(json.loads(output.read_text()), {
                "schema": "fixture/v1",
                "writer": winners[0][1],
            })
            self.assertEqual(
                [path.name for path in output.parent.iterdir()],
                ["matched_target_diagnostics.json"],
            )

    def test_preexisting_output_is_rejected_without_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            output = repository / diagnostic.OUTPUT_RELATIVE_PATH
            output.parent.mkdir(parents=True)
            original = b"preexisting immutable bytes\n"
            output.write_bytes(original)
            with (
                mock.patch.object(diagnostic, "PROJECT_ROOT", repository),
                mock.patch.object(
                    diagnostic.os,
                    "replace",
                    side_effect=AssertionError("os.replace must not be used"),
                ),
            ):
                with self.assertRaises(FileExistsError):
                    diagnostic._write_fixed_json_atomic({"writer": "late"})
            self.assertEqual(output.read_bytes(), original)

    def test_cli_imports_no_public_test_dataset_or_evaluator(self) -> None:
        source = Path(diagnostic.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        self.assertNotIn("test", imported_modules)
        self.assertNotIn("experiments.evisirst_data", imported_modules)
        self.assertNotIn("EviSIRSTTestDataset", source)


class ExclusiveRunLockTest(unittest.TestCase):
    def test_direct_invocations_contend_on_one_fixed_run_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "runs").mkdir()
            with (
                mock.patch.object(diagnostic, "PROJECT_ROOT", repository),
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop(diagnostic.RUN_LOCK_FD_ENV, None)
                with diagnostic.exclusive_run_lock():
                    with self.assertRaisesRegex(
                        diagnostic.PairedBaselineDiagnosticError,
                        "holds the run lock",
                    ):
                        with diagnostic.exclusive_run_lock():
                            self.fail("a second direct invocation acquired the lock")

    def test_watcher_inherited_lock_descriptor_does_not_self_deadlock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            lock_path = repository / diagnostic.RUN_LOCK_RELATIVE_PATH
            lock_path.parent.mkdir(parents=True)
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            contender = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    mock.patch.object(diagnostic, "PROJECT_ROOT", repository),
                    mock.patch.dict(
                        os.environ,
                        {diagnostic.RUN_LOCK_FD_ENV: str(descriptor)},
                    ),
                ):
                    with diagnostic.exclusive_run_lock() as inherited:
                        self.assertEqual(inherited, descriptor)

                contender = os.open(lock_path, os.O_RDWR)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if contender is not None:
                    os.close(contender)
                os.close(descriptor)

    def test_inherited_descriptor_must_reference_the_fixed_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            fixed_path = repository / diagnostic.RUN_LOCK_RELATIVE_PATH
            fixed_path.parent.mkdir(parents=True)
            fixed_path.touch()
            wrong_path = repository / "wrong.lock"
            descriptor = os.open(wrong_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                with (
                    mock.patch.object(diagnostic, "PROJECT_ROOT", repository),
                    mock.patch.dict(
                        os.environ,
                        {diagnostic.RUN_LOCK_FD_ENV: str(descriptor)},
                    ),
                ):
                    with self.assertRaisesRegex(
                        diagnostic.PairedBaselineDiagnosticError,
                        "not the fixed run lock",
                    ):
                        with diagnostic.exclusive_run_lock():
                            self.fail("wrong inherited descriptor was accepted")
            finally:
                os.close(descriptor)

    def test_inherited_per_gpu_lock_is_bound_to_full_uuid_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            gpu_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
            lock_path = repository / "runs" / ".gpu_locks" / f"{gpu_uuid}.lock"
            lock_path.parent.mkdir(parents=True)
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            contender = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    mock.patch.object(diagnostic, "PROJECT_ROOT", repository),
                    mock.patch.dict(
                        os.environ,
                        {
                            diagnostic.GPU_LOCK_FD_ENV: str(descriptor),
                            diagnostic.GPU_UUID_ENV: gpu_uuid,
                        },
                    ),
                ):
                    diagnostic.validate_optional_inherited_gpu_lock()
                contender = os.open(lock_path, os.O_RDWR)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if contender is not None:
                    os.close(contender)
                os.close(descriptor)


class WatcherShellContractTest(unittest.TestCase):
    def test_watcher_syntax_and_lock_uuid_retry_contract(self) -> None:
        watcher = (
            Path(diagnostic.__file__).resolve().parent
            / "tools"
            / "run_irstd_paired_baseline_diagnostic_when_ready.sh"
        )
        subprocess.run(["bash", "-n", str(watcher)], check=True)
        source = watcher.read_text(encoding="utf-8")
        self.assertIn('"/proc/${pid}/cmdline"', source)
        self.assertIn("--query-gpu=index,uuid,pci.bus_id,memory.used", source)
        self.assertIn('CUDA_VISIBLE_DEVICES="${selected_gpu_uuid}"', source)
        self.assertIn(diagnostic.RUN_LOCK_FD_ENV, source)
        self.assertIn("GPU_LOCK_ROOT", source)
        self.assertNotIn("exec /usr/bin/time", source)
        self.assertIn("diagnostic failed with exit status", source)


if __name__ == "__main__":
    unittest.main()
