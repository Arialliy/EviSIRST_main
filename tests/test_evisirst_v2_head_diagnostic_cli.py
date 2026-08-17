from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import run_evisirst_v2_head_diagnostic as diagnostic


def _metrics(*, miou: float = 0.5) -> dict[str, float | int | None]:
    return {
        "validation_loss": 0.2,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.75,
        "pixel_recall": 0.6,
        "pixel_f1": 2.0 / 3.0,
        "pd": 1.0,
        "tiny_pd": None,
        "fa": 0.0,
        "false_objects_per_image": 0.0,
        "target_count": 1,
        "matched_target_count": 1,
        "tiny_target_count": 0,
        "matched_tiny_target_count": 0,
        "predicted_object_count": 1,
        "unmatched_predicted_object_count": 0,
        "valid_pixel_count": 4,
    }


def _identity(*, epoch: int = 3, run_seed: int = 7) -> dict[str, object]:
    source_files = {
        name: {
            "relative_path": f"fixture/{name}.py",
            "sha256": str(index + 1) * 64,
        }
        for index, name in enumerate(
            ("trainer", "data", "selection", "source_protocol", "model_entry")
        )
    }
    determinism_protocol = {
        "schema": "evisirst_validation_selected_determinism/v1",
        "training_data": {
            "source_protocol_version": diagnostic.source_protocol.PROTOCOL_VERSION,
            "patch_size": diagnostic.source_protocol.PATCH_SIZE,
            "train_positive_crop_probability": (
                diagnostic.source_protocol.TRAIN_POSITIVE_CROP_PROBABILITY
            ),
            "augmentation_version": diagnostic.AUGMENTATION_VERSION,
        },
        "validation_evaluation": {
            "version": diagnostic.EVALUATION_PROTOCOL_VERSION,
            "evaluation_head": "out",
            "prediction_threshold": diagnostic.PROBABILITY_THRESHOLD,
            "prediction_threshold_operator": (
                diagnostic.PREDICTION_THRESHOLD_OPERATOR
            ),
            "target_threshold": diagnostic.TARGET_THRESHOLD,
            "target_threshold_operator": diagnostic.TARGET_THRESHOLD_OPERATOR,
            "match_radius": diagnostic.MATCH_RADIUS,
            "match_distance_operator": diagnostic.MATCH_DISTANCE_OPERATOR,
            "connected_component_connectivity": (
                diagnostic.CONNECTED_COMPONENT_CONNECTIVITY
            ),
            "connected_component_neighborhood": (
                diagnostic.CONNECTED_COMPONENT_NEIGHBORHOOD
            ),
            "assignment_algorithm": diagnostic.ASSIGNMENT_ALGORITHM,
            "tiny_area": diagnostic.TINY_AREA,
            "tiny_area_operator": diagnostic.TINY_AREA_OPERATOR,
        },
        "selection": {
            "rule_version": diagnostic.validation_selection.INDEPENDENT_RULE_VERSION,
            "miou_candidate_tolerance": (
                diagnostic.validation_selection.MIOU_CANDIDATE_TOLERANCE
            ),
        },
        "source_set_schema": "evisirst_validation_selected_source_set/v2",
        "source_files": source_files,
        "source_tree_sha256": diagnostic._sha256_bytes(
            diagnostic._canonical_json_bytes(source_files)
        ),
    }
    identity: dict[str, object] = {
        "schema": diagnostic.RUN_IDENTITY_SCHEMA,
        "model": "EviSIRST",
        "dataset": diagnostic.DATASET,
        "architecture_seed": 42,
        "run_seed": run_seed,
        "target_mode": "binary",
        "epochs": 10,
        "batch_size": 2,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 1,
        "val_interval": 1,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": diagnostic.EXPECTED_EVALUATION,
        "selection_rule": diagnostic.EXPECTED_SELECTION_RULE,
        "determinism_protocol": determinism_protocol,
        "manifest_sha256": "a" * 64,
        "split_seed": 123,
        "data_tree_sha256": "b" * 64,
        "grouping_policy": {"mode": "fixture"},
        "train_count": 4,
        "val_count": 2,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = diagnostic._sha256_bytes(
        diagnostic._canonical_json_bytes(identity)
    )
    return identity


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
        image = torch.zeros(1, 2, 2)
        target = torch.zeros(1, 2, 2) if index == 0 else torch.ones(1, 2, 2)
        return image, target, (2, 2), f"sample_{index}"


class CountingSixHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mode = "test"
        self.forward_calls = 0

    def forward(self, image: torch.Tensor):
        self.forward_calls += 1
        # Frozen order: gt5, gt4, gt3, gt2, d0, out.  Exactly 0.5 on out
        # also checks that the metric threshold is strict ``>``.
        return tuple(
            torch.full_like(image, value)
            for value in (0.1, 0.2, 0.3, 0.4, 0.75, 0.5)
        )


class FixedHeadEvaluationTest(unittest.TestCase):
    def test_three_heads_share_exactly_one_forward_per_sample(self) -> None:
        model = CountingSixHeadModel()
        loader = DataLoader(TwoSampleValidationDataset(), batch_size=1, shuffle=False)

        metrics, digests, count = diagnostic.evaluate_preregistered_heads(
            model, loader, torch.device("cpu")
        )

        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(count, 2)
        self.assertEqual(tuple(metrics), ("out", "d0", "logit_blend"))
        self.assertEqual(metrics["out"]["miou"], 0.0)
        self.assertEqual(metrics["d0"]["miou"], 0.5)
        self.assertEqual(metrics["logit_blend"]["miou"], 0.5)
        self.assertEqual(set(digests), set(metrics))
        self.assertTrue(all(len(value) == 64 for value in digests.values()))

    def test_prediction_digests_are_reproducible(self) -> None:
        loader = DataLoader(TwoSampleValidationDataset(), batch_size=1, shuffle=False)
        first = diagnostic.evaluate_preregistered_heads(
            CountingSixHeadModel(), loader, torch.device("cpu")
        )
        second = diagnostic.evaluate_preregistered_heads(
            CountingSixHeadModel(), loader, torch.device("cpu")
        )
        self.assertEqual(first, second)


class CandidateLoadingTest(unittest.TestCase):
    def _write_candidate(self, repository: Path) -> tuple[Path, FakeStateModel]:
        model = FakeStateModel()
        epoch = 3
        candidate = (
            repository
            / "runs"
            / "validation_selected"
            / "formal"
            / diagnostic.DATASET
            / "binary"
            / "run_seed_7"
            / "candidates"
            / f"epoch_{epoch:04d}.pth.tar"
        )
        candidate.parent.mkdir(parents=True)
        metrics = _metrics()
        torch.save(
            {
                "schema": diagnostic.CANDIDATE_SCHEMA,
                "model": "EviSIRST",
                "dataset": diagnostic.DATASET,
                "epoch": epoch,
                "run_identity": _identity(epoch=epoch),
                "validation_record": {
                    "epoch": epoch,
                    "data_role": "val",
                    "mIoU": metrics["miou"],
                    "Fa": metrics["fa"],
                    "Pd": metrics["pd"],
                    "evaluation_head": "out",
                    "metrics": metrics,
                },
                "state_dict": model.state_dict(),
                "test_split_accessed": False,
            },
            candidate,
        )
        return candidate, model

    def test_current_candidate_schema_loads_exact_564_key_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            candidate, fixture_model = self._write_candidate(repository)
            with (
                mock.patch.object(diagnostic, "PROJECT_ROOT", repository),
                mock.patch.object(
                    diagnostic,
                    "initialize_evisirst",
                    return_value=(FakeStateModel(), {"fixture": True}),
                ),
            ):
                model, metadata = diagnostic.load_candidate_model(
                    candidate, torch.device("cpu")
                )

            self.assertEqual(len(model.state_dict()), 564)
            self.assertEqual(
                model.state_dict()["state_563"],
                fixture_model.state_dict()["state_563"],
            )
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["run_identity"]["run_seed"], 7)
            self.assertEqual(metadata["validation_record"]["data_role"], "val")
            self.assertFalse(metadata["checkpoint_path"].startswith("/"))
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)

    def test_nonformal_or_test_named_path_is_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            path = repository / "runs" / "public_test" / "candidate.pth.tar"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not a checkpoint")
            with mock.patch.object(diagnostic, "PROJECT_ROOT", repository):
                with self.assertRaisesRegex(
                    diagnostic.EviSIRSTV2HeadDiagnosticError,
                    "formal IRSTD-1K",
                ):
                    diagnostic._repository_relative_regular_checkpoint(path)

    def test_state_contract_rejects_fewer_than_564_keys(self) -> None:
        expected = FakeStateModel().state_dict()
        observed = dict(expected)
        del observed["state_563"]
        with self.assertRaisesRegex(
            diagnostic.EviSIRSTV2HeadDiagnosticError, "564-key"
        ):
            diagnostic._validate_state_dict(observed, expected)


class ResultContractTest(unittest.TestCase):
    def _dataset(self):
        return SimpleNamespace(
            contract=SimpleNamespace(
                manifest_sha256="c" * 64,
                data_tree_sha256="d" * 64,
                data_tree_verified=True,
                train_ids=("train_0", "train_1"),
                val_ids=("val_0", "val_1"),
            )
        )

    def test_bundle_forces_all_three_ledgers_to_diagnostic_only(self) -> None:
        metrics = {spec.head: _metrics() for spec in diagnostic.PREREGISTERED_HEAD_SPECS}
        prediction_sha = {
            spec.head: str(index + 1) * 64
            for index, spec in enumerate(diagnostic.PREREGISTERED_HEAD_SPECS)
        }
        payload = diagnostic.build_result_payload(
            checkpoint_metadata={
                "checkpoint_path": (
                    "runs/validation_selected/formal/IRSTD-1K/binary/"
                    "run_seed_7/candidates/epoch_0003.pth.tar"
                ),
                "checkpoint_sha256": "a" * 64,
                "checkpoint_schema": diagnostic.CANDIDATE_SCHEMA,
                "epoch": 3,
                "run_identity": {"run_seed": 7, "target_mode": "binary"},
            },
            dataset=self._dataset(),
            metrics=metrics,
            prediction_sha256=prediction_sha,
            sample_count=2,
        )

        self.assertEqual(payload["data_role"], "val")
        self.assertFalse(payload["test_split_accessed"])
        self.assertTrue(payload["diagnostic_only"])
        self.assertFalse(payload["selection_allowed"])
        self.assertEqual(
            tuple(payload["head_ledgers"]), ("d0", "logit_blend", "out")
        )
        for ledger in payload["head_ledgers"].values():
            self.assertTrue(ledger["diagnostic_only"])
            self.assertFalse(ledger["selection_allowed"])
            self.assertEqual(ledger["role"], "val")
            self.assertEqual(ledger["split_manifest_sha256"], "c" * 64)
            self.assertFalse(
                ledger["checkpoint_provenance"]["selection_is_optimistic"]
            )

        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertNotIn('"best_head"', encoded.lower())
        self.assertNotIn('"selected_head"', encoded.lower())
        self.assertNotIn("/home/", encoded)
        self.assertTrue(
            all(
                not ledger["protocol_assertions"]["best_api_available"]
                for ledger in payload["head_ledgers"].values()
            )
        )
        self.assertFalse(payload["evaluation"]["prediction_or_target_arrays_written"])

    def test_json_write_is_atomic_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            destination = repository / "runs" / "diagnostics" / "result.json"
            payload = {"schema": "fixture/v1", "test_split_accessed": False}
            with mock.patch.object(diagnostic, "PROJECT_ROOT", repository):
                diagnostic.write_json_atomic(destination, payload)
            self.assertEqual(json.loads(destination.read_text()), payload)
            self.assertEqual(
                [path.name for path in destination.parent.iterdir()],
                ["result.json"],
            )

    def test_cli_source_imports_no_public_test_dataset_or_evaluator(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
