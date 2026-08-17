from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from experiments import evisirst_v2_data as v2_data
from experiments import evisirst_v2_splits as split_builder
from experiments import three_dataset_v2_protocol as source_protocol


class EviSIRSTV2DataTest(unittest.TestCase):
    def _fixture(
        self, root: Path, *, count: int = 10
    ) -> tuple[Path, Path, split_builder.SplitBundle]:
        dataset = "NUAA-SIRST"
        dataset_root = root / "datasets"
        image_dir = dataset_root / dataset / "images"
        mask_dir = dataset_root / dataset / "masks"
        index_path = dataset_root / dataset / "img_idx" / f"train_{dataset}.txt"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        index_path.parent.mkdir(parents=True)
        sample_ids = [f"fixture_{position:03d}" for position in range(count)]
        # Mirror the historical source files: exact bytes are frozen even when
        # the final line has no newline.  Generated split outputs are canonical.
        index_path.write_text("\n".join(sample_ids), encoding="utf-8")
        for position, sample_id in enumerate(sample_ids):
            image = np.arange(19 * 23, dtype=np.uint16).reshape(19, 23)
            image = np.asarray((image + position * 7) % 256, dtype=np.uint8)
            mask = np.zeros((19, 23), dtype=np.uint8)
            mask[1, 2] = 64
            mask[5:7, 6:8] = 200
            mask[10, 10] = 255
            Image.fromarray(image).save(image_dir / f"{sample_id}.png")
            Image.fromarray(mask).save(mask_dir / f"{sample_id}.png")

        records, identity = split_builder.analyze_fixture_index(
            dataset_root=dataset_root,
            dataset=dataset,
            index_path=index_path,
            image_dir=image_dir,
            mask_dir=mask_dir,
        )
        bundle = split_builder.build_split_bundle(
            dataset=dataset,
            records=records,
            source_index=identity,
            split_seed=20260811,
            run_seed=42,
            val_fraction=0.20,
        )
        split_root = root / "splits" / "v2"
        split_builder.materialize_bundle(bundle, split_root)
        return dataset_root, split_root, bundle

    def test_contract_strictly_validates_hash_union_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, bundle = self._fixture(Path(temporary))
            contract = v2_data.load_v2_split_contract(
                "NUAA-SIRST",
                dataset_root=dataset_root,
                split_root=split_root,
            )
            summary = v2_data.check_v2_splits(
                dataset_root=dataset_root,
                split_root=split_root,
                datasets=("NUAA-SIRST",),
            )

        self.assertEqual(contract.train_ids, bundle.train_ids)
        self.assertEqual(contract.val_ids, bundle.val_ids)
        self.assertFalse(set(contract.train_ids) & set(contract.val_ids))
        self.assertEqual(
            set(contract.train_ids) | set(contract.val_ids),
            set(contract.source_train_ids),
        )
        self.assertEqual(summary["datasets"]["NUAA-SIRST"]["status"], "verified")
        self.assertTrue(contract.data_tree_verified)
        self.assertEqual(
            contract.data_tree_sha256,
            contract.manifest["data_identity"]["ordered_image_mask_tree_sha256"],
        )
        self.assertFalse(
            summary["datasets"]["NUAA-SIRST"]["test_index_opened"]
        )

    def test_changed_source_image_is_rejected_by_data_tree_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, bundle = self._fixture(Path(temporary))
            sample_id = bundle.train_ids[0]
            image_path = dataset_root / "NUAA-SIRST" / "images" / f"{sample_id}.png"
            changed = np.full((19, 23), 17, dtype=np.uint8)
            Image.fromarray(changed).save(image_path)
            with self.assertRaisesRegex(
                v2_data.EviSIRSTV2DataError, "image/mask tree SHA-256"
            ):
                v2_data.load_v2_split_contract(
                    "NUAA-SIRST",
                    dataset_root=dataset_root,
                    split_root=split_root,
                )

    def test_changed_split_file_is_rejected_before_dataset_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, _ = self._fixture(Path(temporary))
            val_path = split_root / "NUAA-SIRST" / "val.txt"
            val_path.write_bytes(val_path.read_bytes() + b"fixture_999\n")
            with self.assertRaisesRegex(
                v2_data.EviSIRSTV2DataError, "SHA-256|count|union"
            ):
                v2_data.load_v2_split_contract(
                    "NUAA-SIRST",
                    dataset_root=dataset_root,
                    split_root=split_root,
                )

    def test_manifest_cannot_redirect_consumer_to_test_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, _ = self._fixture(Path(temporary))
            manifest_path = split_root / "NUAA-SIRST" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_index"]["relative_path"] = (
                "NUAA-SIRST/img_idx/test_NUAA-SIRST.txt"
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
            # No test index exists in this fixture.  The lexical guard must
            # reject the manifest before attempting to resolve any such path.
            with self.assertRaisesRegex(
                v2_data.EviSIRSTV2DataError, "not the train index"
            ):
                v2_data.load_v2_split_contract(
                    "NUAA-SIRST",
                    dataset_root=dataset_root,
                    split_root=split_root,
                )

    def test_train_uses_only_train_membership_and_occurrence_aware_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, bundle = self._fixture(Path(temporary))
            dataset = v2_data.EviSIRSTV2TrainDataset(
                "NUAA-SIRST",
                dataset_root=dataset_root,
                split_root=split_root,
                target_mode="binary",
                run_seed=314159,
                return_metadata=True,
            )
            first = dataset[(0, 7)]
            repeated = dataset[(0, 7)]
            other_occurrence = dataset[(0, 8)]
            dataset.set_epoch(1)
            other_epoch = dataset[(0, 7)]

        self.assertEqual(dataset.sample_ids, bundle.train_ids)
        self.assertFalse(set(dataset.sample_ids) & set(bundle.val_ids))
        self.assertTrue(dataset.metadata["data_tree_verified"])
        self.assertEqual(
            dataset.metadata["data_tree_sha256"],
            dataset.contract.manifest["data_identity"][
                "ordered_image_mask_tree_sha256"
            ],
        )
        self.assertEqual(first["image"].shape, (1, 256, 256))
        self.assertEqual(first["mask"].shape, (1, 256, 256))
        self.assertEqual(float(first["mask"].sum()), 5.0)
        self.assertTrue(torch.equal(first["image"], repeated["image"]))
        self.assertTrue(torch.equal(first["mask"], repeated["mask"]))
        self.assertEqual(first["augmentation_seed"], repeated["augmentation_seed"])
        self.assertNotEqual(
            first["augmentation_seed"], other_occurrence["augmentation_seed"]
        )
        self.assertNotEqual(
            first["augmentation_seed"], other_epoch["augmentation_seed"]
        )
        self.assertEqual(first["mask_audit"]["positive_crop_rule"], "raw_mask>127")
        self.assertEqual(first["mask_audit"]["nonbinary_pixel_count"], 5)

    def test_soft_target_preserves_grayscale_and_records_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, _ = self._fixture(Path(temporary))
            dataset = v2_data.EviSIRSTV2TrainDataset(
                "NUAA-SIRST",
                dataset_root=dataset_root,
                split_root=split_root,
                target_mode="soft",
                run_seed=7,
                return_metadata=True,
            )
            item = dataset[0]

        expected_sum = (64.0 + 4 * 200.0 + 255.0) / 255.0
        self.assertAlmostEqual(float(item["mask"].sum()), expected_sum, places=6)
        self.assertEqual(item["mask_audit"]["encoding"], "grayscale_0_255")
        self.assertEqual(item["mask_audit"]["target_rule"], "raw_mask/255")
        aggregate = dataset.metadata["grayscale_mask_audit_from_manifest"]
        self.assertGreater(aggregate["nonbinary_mask_image_count"], 0)
        self.assertGreater(aggregate["nonbinary_mask_pixel_count"], 0)

    def test_val_is_unaugmented_padded_and_evaluator_tuple_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, bundle = self._fixture(Path(temporary))
            dataset = v2_data.EviSIRSTV2ValDataset(
                "NUAA-SIRST",
                dataset_root=dataset_root,
                split_root=split_root,
                target_mode="binary",
            )
            first = dataset[0]
            repeated = dataset[0]
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

        self.assertEqual(dataset.sample_ids, bundle.val_ids)
        self.assertEqual(len(first), 4)
        image, mask, original_hw, sample_id = first
        self.assertEqual(image.shape, (1, 32, 32))
        self.assertEqual(mask.shape, (1, 32, 32))
        self.assertEqual(original_hw, (19, 23))
        self.assertEqual(sample_id, bundle.val_ids[0])
        self.assertTrue(torch.equal(first[0], repeated[0]))
        self.assertTrue(torch.equal(first[1], repeated[1]))
        self.assertEqual(batch[0].shape, (1, 1, 32, 32))
        self.assertEqual(batch[1].shape, (1, 1, 32, 32))
        self.assertEqual(len(batch), 4)

    def test_r1_normalization_is_explicitly_legacy_and_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, _ = self._fixture(Path(temporary))
            legacy = source_protocol.get_legacy_normalization("NUAA-SIRST")
            accepted = v2_data.EviSIRSTV2ValDataset(
                "NUAA-SIRST",
                dataset_root=dataset_root,
                split_root=split_root,
                target_mode="soft",
                normalization_mode="legacy",
                normalization_values=legacy,
            )
            with self.assertRaisesRegex(
                v2_data.EviSIRSTV2DataError, "differ from frozen legacy"
            ):
                v2_data.EviSIRSTV2ValDataset(
                    "NUAA-SIRST",
                    dataset_root=dataset_root,
                    split_root=split_root,
                    target_mode="soft",
                    normalization_mode="legacy",
                    normalization_values={"mean": 0.0, "std": 1.0},
                )
            with self.assertRaisesRegex(
                v2_data.EviSIRSTV2DataError, "legacy.*only"
            ):
                v2_data.EviSIRSTV2ValDataset(
                    "NUAA-SIRST",
                    dataset_root=dataset_root,
                    split_root=split_root,
                    target_mode="soft",
                    normalization_mode="pooled_train",
                )

        self.assertEqual(accepted.normalization, legacy)
        self.assertEqual(accepted.normalization_spec.mode, "legacy")

    def test_target_mode_has_no_implicit_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root, split_root, _ = self._fixture(Path(temporary))
            with self.assertRaises(TypeError):
                v2_data.EviSIRSTV2TrainDataset(  # type: ignore[call-arg]
                    "NUAA-SIRST",
                    dataset_root=dataset_root,
                    split_root=split_root,
                    run_seed=1,
                )


if __name__ == "__main__":
    unittest.main()
