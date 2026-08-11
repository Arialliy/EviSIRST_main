from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import test as public_test
import train as public_train
from experiments import three_dataset_v2_protocol as source_protocol
from experiments.evisirst_data import (
    EviSIRSTTestDataset,
    EviSIRSTTrainDataset,
    SIRST3_NORMALIZATION,
    stable_uint63,
)
from model.EviSIRST import initialize_evisirst


class PublicTrainTestContract(unittest.TestCase):
    def test_learning_rate_endpoints(self) -> None:
        values = {
            epoch: public_train.learning_rate_for_epoch(
                epoch, 1000, 1e-3, 1e-5, 10
            )
            for epoch in (1, 10, 1000)
        }
        self.assertAlmostEqual(values[1], 1e-4)
        self.assertAlmostEqual(values[10], 1e-3)
        self.assertAlmostEqual(values[1000], 1e-5)
        self.assertEqual(
            stable_uint63(42, "SIRST3", "shuffle", 1),
            4199480799363881725,
        )

    def test_rng_state_round_trip(self) -> None:
        public_train.configure_determinism(42)
        state = public_train._capture_rng_state(torch.device("cpu"))
        expected_python = __import__("random").random()
        expected_numpy = float(np.random.random())
        expected_torch = torch.rand(4)
        public_train._restore_rng_state(state, torch.device("cpu"))
        self.assertEqual(__import__("random").random(), expected_python)
        self.assertEqual(float(np.random.random()), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))

    def test_clean_initializer_has_six_training_outputs(self) -> None:
        model, metadata = initialize_evisirst(
            "SIRST3", seed=42, training=True
        )
        self.assertTrue(model.training)
        self.assertEqual(model.mode, "train")
        self.assertEqual(len(model.state_dict()), 564)
        self.assertFalse(hasattr(model, "target_survival"))
        self.assertFalse(metadata["target_survival_registered"])
        with torch.no_grad():
            outputs = model(torch.zeros(1, 1, 32, 32))
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 6)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)

    def test_train_only_dataset_does_not_request_test_index(self) -> None:
        ids = [f"sample_{index}" for index in range(213)]
        observed: list[tuple[str, str]] = []

        def fake_load_index(root: Path, dataset: str, split: str) -> list[str]:
            observed.append((dataset, split))
            return ids

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            source_protocol, "load_index", side_effect=fake_load_index
        ):
            dataset = EviSIRSTTrainDataset(
                "NUAA-SIRST",
                dataset_root=temporary,
            )
        self.assertEqual(len(dataset), 213)
        self.assertEqual(observed, [("NUAA-SIRST", "train")])

    def test_test_only_dataset_requests_selected_test_index(self) -> None:
        ids = [f"sample_{index}" for index in range(664)]
        observed: list[tuple[str, str]] = []

        def fake_load_index(root: Path, dataset: str, split: str) -> list[str]:
            observed.append((dataset, split))
            return ids

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            source_protocol, "load_index", side_effect=fake_load_index
        ):
            dataset = EviSIRSTTestDataset(
                "NUDT-SIRST",
                "NUDT-SIRST",
                dataset_root=temporary,
            )
        self.assertEqual(len(dataset), 664)
        self.assertEqual(observed, [("NUDT-SIRST", "test")])

    def test_sirst3_normalization_is_used_for_all_source_tests(self) -> None:
        counts = {"NUAA-SIRST": 214, "NUDT-SIRST": 664, "IRSTD-1K": 201}

        def fake_load_index(root: Path, dataset: str, split: str) -> list[str]:
            self.assertEqual(split, "test")
            return [f"sample_{index}" for index in range(counts[dataset])]

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            source_protocol, "load_index", side_effect=fake_load_index
        ):
            for evaluation_dataset in counts:
                dataset = EviSIRSTTestDataset(
                    "SIRST3",
                    evaluation_dataset,
                    dataset_root=temporary,
                )
                self.assertEqual(dataset.normalization, SIRST3_NORMALIZATION)

    def test_validation_metrics_perfect_and_false_alarm(self) -> None:
        target = np.zeros((8, 8), dtype=np.float32)
        target[2:4, 2:4] = 1.0
        perfect = public_test.ValidationMetrics(0.5, 3.0, 9)
        perfect.update(target.copy(), target, 0.0)
        metrics = perfect.compute()
        self.assertEqual(metrics["miou"], 1.0)
        self.assertEqual(metrics["niou"], 1.0)
        self.assertEqual(metrics["pixel_f1"], 1.0)
        self.assertEqual(metrics["pd"], 1.0)
        self.assertEqual(metrics["fa"], 0.0)

        probability = target.copy()
        probability[7, 7] = 1.0
        with_false_alarm = public_test.ValidationMetrics(0.5, 3.0, 9)
        with_false_alarm.update(probability, target, 0.0)
        self.assertEqual(with_false_alarm.compute()["fa"], 1 / 64)


if __name__ == "__main__":
    unittest.main()
