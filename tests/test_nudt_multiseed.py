from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from experiments import evisirst_multiseed as multiseed
from experiments import three_dataset_v2_protocol as source_protocol
from experiments.evisirst_data import EviSIRSTTrainDataset
from experiments.evisirst_data import EviSIRSTTestDataset
from experiments.evisirst_multiseed import (
    COMPATIBILITY_SEED,
    CONFIRMATORY_SEEDS,
    NUDT_DATASET,
    NUDTMultiseedTrainDataset,
    initialize_multiseed_evisirst,
)
from experiments.four_dataset_models_seed42_v1 import (
    QFG_TERMINAL_STATE_KEYS,
    state_dict_sha256,
)
from model.EviSIRST import initialize_evisirst
import train_nudt_multiseed_dual_role as runner


class NUDTMultiseedContractTest(unittest.TestCase):
    def test_fresh_confirmatory_seed_contract_is_exact(self) -> None:
        self.assertEqual(
            CONFIRMATORY_SEEDS,
            (1446202191, 104728269, 262620274),
        )
        self.assertNotIn(COMPATIBILITY_SEED, CONFIRMATORY_SEEDS)
        self.assertEqual(len(CONFIRMATORY_SEEDS), len(set(CONFIRMATORY_SEEDS)))

    def test_formal_cli_rejects_seed42_and_recipe_changes(self) -> None:
        with self.assertRaises(SystemExit):
            runner.parse_args(
                ["--dataset-root", "/tmp/data", "--seed", "42"]
            )
        with self.assertRaises(SystemExit):
            runner.parse_args(
                [
                    "--dataset-root",
                    "/tmp/data",
                    "--seed",
                    str(CONFIRMATORY_SEEDS[0]),
                    "--epochs",
                    "2",
                    "--warmup-epochs",
                    "1",
                ]
            )
        args = runner.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--seed",
                str(CONFIRMATORY_SEEDS[0]),
            ]
        )
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.selection_begin, 500)
        self.assertEqual(args.selection_every, 1)
        self.assertEqual(args.output_root, runner.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(args.result_root, runner.DEFAULT_RESULT_ROOT)

    def test_smoke_cli_allows_seed42_but_remains_explicit(self) -> None:
        args = runner.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--seed",
                "42",
                "--smoke",
                "--epochs",
                "2",
                "--warmup-epochs",
                "1",
                "--selection-begin",
                "1",
                "--max-train-samples",
                "1",
                "--max-test-images",
                "1",
            ]
        )
        self.assertTrue(args.smoke)
        self.assertEqual(args.seed, 42)

    def test_seed42_initializer_is_bitwise_identical_to_frozen_oracle(self) -> None:
        oracle, _ = initialize_evisirst(
            NUDT_DATASET, seed=COMPATIBILITY_SEED, training=False
        )
        candidate, metadata = initialize_multiseed_evisirst(
            NUDT_DATASET, seed=COMPATIBILITY_SEED, training=False
        )
        oracle_state = oracle.state_dict()
        candidate_state = candidate.state_dict()
        self.assertEqual(list(candidate_state), list(oracle_state))
        self.assertEqual(
            state_dict_sha256(candidate_state), state_dict_sha256(oracle_state)
        )
        for key in oracle_state:
            self.assertTrue(torch.equal(candidate_state[key], oracle_state[key]), key)
        self.assertEqual(metadata["selected_model_state_key_count"], 564)
        self.assertFalse(hasattr(candidate, "target_survival"))
        del oracle, candidate, oracle_state, candidate_state
        gc.collect()

    def test_confirmatory_seeds_initialize_distinct_clean_graphs(self) -> None:
        hashes: list[str] = []
        for seed in CONFIRMATORY_SEEDS:
            model, metadata = initialize_multiseed_evisirst(
                NUDT_DATASET, seed=seed, training=False
            )
            state = model.state_dict()
            self.assertEqual(len(state), 564)
            self.assertFalse(hasattr(model, "target_survival"))
            self.assertEqual(metadata["training_seed"], seed)
            self.assertTrue(
                all(int(torch.count_nonzero(state[key])) == 0 for key in QFG_TERMINAL_STATE_KEYS)
            )
            hashes.append(state_dict_sha256(state))
            del model, state
            gc.collect()
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_dataset_seed42_parity_and_fresh_seed_transform_difference(self) -> None:
        sample_id = "sample_0"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            yy, xx = np.mgrid[:320, :304]
            image = ((3 * yy + 5 * xx) % 4096).astype(np.uint16)
            mask = np.zeros((320, 304), dtype=np.uint8)
            mask[145:154, 171:180] = 255
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            resolved = source_protocol.ResolvedSample(
                dataset_name=NUDT_DATASET,
                split="train",
                sample_id=sample_id,
                image_path=image_path,
                raw_mask_path=mask_path,
                mask_path=mask_path,
                correction_id=None,
            )
            with mock.patch.object(
                source_protocol, "load_index", return_value=[sample_id]
            ), mock.patch.object(
                source_protocol, "resolve_sample", return_value=resolved
            ):
                oracle = EviSIRSTTrainDataset(
                    NUDT_DATASET,
                    dataset_root=root,
                    seed=COMPATIBILITY_SEED,
                    return_metadata=True,
                )
                compatible = NUDTMultiseedTrainDataset(
                    dataset_root=root,
                    seed=COMPATIBILITY_SEED,
                    return_metadata=True,
                )
                fresh = NUDTMultiseedTrainDataset(
                    dataset_root=root,
                    seed=CONFIRMATORY_SEEDS[0],
                    return_metadata=True,
                )
                for epoch in (0, 1, 500, 1000):
                    oracle.set_epoch(epoch)
                    compatible.set_epoch(epoch)
                    fresh.set_epoch(epoch)
                    expected = oracle[0]
                    observed = compatible[0]
                    changed = fresh[0]
                    self.assertEqual(observed["transform_plan"], expected["transform_plan"])
                    self.assertTrue(torch.equal(observed["image"], expected["image"]))
                    self.assertTrue(torch.equal(observed["mask"], expected["mask"]))
                    self.assertNotEqual(
                        changed["augmentation_seed"], expected["augmentation_seed"]
                    )
                    self.assertNotEqual(changed["transform_plan"], expected["transform_plan"])

    def test_role_keys_are_the_exact_historical_functions(self) -> None:
        self.assertEqual(
            [
                epoch
                for epoch in range(1, 1001)
                if runner.selection_due(epoch, 500, 1)
            ],
            list(range(500, 1001)),
        )
        row = {
            "epoch": 500,
            "test_loss": 0.1,
            "miou": 0.8,
            "niou": 0.7,
            "pd": 0.9,
            "tiny_pd": 0.85,
            "fa": 1e-6,
        }
        self.assertEqual(
            runner.role_key(row, "best_miou"),
            (0.8, 0.9, -1e-6, 0.7, 0.85, -0.1, -500.0),
        )
        self.assertEqual(
            runner.role_key(row, "best_pd"),
            (0.9, -1e-6, 0.85, 0.8, 0.7, -0.1, -500.0),
        )

    def test_recovery_identity_rejects_cross_seed(self) -> None:
        expected_state = {"weight": torch.zeros(1)}
        config = {
            "dataset": NUDT_DATASET,
            "seed": CONFIRMATORY_SEEDS[0],
            "selection_begin_epoch": 500,
            "selection_every": 1,
        }
        payload = {
            "schema": runner.RECOVERY_SCHEMA,
            "model": "EviSIRST",
            "dataset": NUDT_DATASET,
            "epoch": 499,
            "seed": CONFIRMATORY_SEEDS[0],
            "state_dict": expected_state,
            "optimizer": {},
            "training": config,
            "selection_history": [],
            "best_epochs": {"best_miou": None, "best_pd": None},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recovery.pth.tar"
            torch.save(payload, path)
            loaded = runner._load_recovery(
                path, config=config, expected_state=expected_state
            )
            self.assertEqual(loaded["seed"], CONFIRMATORY_SEEDS[0])
            cross_seed = dict(config)
            cross_seed["seed"] = CONFIRMATORY_SEEDS[1]
            with self.assertRaises(ValueError):
                runner._load_recovery(
                    path, config=cross_seed, expected_state=expected_state
                )

    def test_seed_aware_checkpoint_loader_validates_and_strict_loads(self) -> None:
        seed = CONFIRMATORY_SEEDS[0]
        source_model = nn.Linear(2, 1)
        source_model.mode = "test"
        state = {
            key: value.detach().clone()
            for key, value in source_model.state_dict().items()
        }
        payload = {
            "schema": multiseed.MULTISEED_CHECKPOINT_SCHEMA,
            "model": "EviSIRST",
            "dataset": NUDT_DATASET,
            "checkpoint_role": "best_miou",
            "epoch": 500,
            "seed": seed,
            "state_dict": state,
            "training": {"dataset": NUDT_DATASET, "seed": seed},
            "test_selected": True,
            "selection_is_optimistic": True,
            "selection_metrics": {"miou": 0.8, "pd": 0.9},
        }

        def fake_initialize(
            dataset: str, *, seed: int, training: bool
        ) -> tuple[nn.Module, dict[str, object]]:
            self.assertEqual(dataset, NUDT_DATASET)
            self.assertEqual(seed, CONFIRMATORY_SEEDS[0])
            self.assertFalse(training)
            model = nn.Linear(2, 1)
            model.mode = "test"
            return model, {"training_seed": seed}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "EviSIRST_best_mIoU.pth.tar"
            torch.save(payload, path)
            with mock.patch.object(
                multiseed,
                "initialize_multiseed_evisirst",
                side_effect=fake_initialize,
            ):
                loaded, metadata = multiseed.load_multiseed_checkpoint(
                    path, expected_seed=seed
                )
                self.assertEqual(metadata["seed"], seed)
                self.assertEqual(metadata["checkpoint_role"], "best_miou")
                for key, value in state.items():
                    self.assertTrue(torch.equal(loaded.state_dict()[key], value))
                with self.assertRaises(ValueError):
                    multiseed.load_multiseed_checkpoint(
                        path, expected_seed=CONFIRMATORY_SEEDS[1]
                    )

    def test_source_lock_contains_new_and_frozen_dependencies(self) -> None:
        relative = {
            path.relative_to(runner.PROJECT_ROOT).as_posix()
            for path in runner._source_paths()
        }
        self.assertIn("train_nudt_multiseed_dual_role.py", relative)
        self.assertIn("experiments/evisirst_multiseed.py", relative)
        self.assertIn("train_dual_role_test_selected.py", relative)
        self.assertIn("model/_internal/SCTransNet.py", relative)
        self.assertEqual(len(runner._source_tree_sha256()), 64)

    @pytest.mark.integration
    def test_canonical_nudt_train_and_test_pixel_trees_are_frozen(self) -> None:
        root = runner.FORMAL_DATASET_ROOT.resolve(strict=True)
        train_dataset = NUDTMultiseedTrainDataset(
            dataset_root=root, seed=CONFIRMATORY_SEEDS[0]
        )
        test_dataset = EviSIRSTTestDataset(
            NUDT_DATASET, NUDT_DATASET, dataset_root=root
        )
        train_entries, test_entries = runner._data_tree_entries(
            train_dataset, test_dataset
        )
        self.assertEqual(len(train_entries), 2 * 663)
        self.assertEqual(len(test_entries), 2 * 664)
        self.assertEqual(
            runner._file_tree_sha256(root, train_entries),
            runner.EXPECTED_TRAIN_IMAGE_MASK_TREE_SHA256,
        )
        self.assertEqual(
            runner._file_tree_sha256(root, test_entries),
            runner.EXPECTED_TEST_IMAGE_MASK_TREE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
