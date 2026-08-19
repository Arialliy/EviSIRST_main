from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments import evisirst_psbfr_lockbox as lockbox


class EviSIRSTPSBFRLockboxTest(unittest.TestCase):
    def _fixture(
        self, root: Path, *, train_count: int = 40, val_count: int = 10
    ) -> tuple[Path, Path, tuple[str, ...], tuple[str, ...], bytes]:
        parent_dir = root / "repository" / "splits" / "v2" / lockbox.DATASET
        mask_dir = root / "data" / lockbox.DATASET / "masks"
        parent_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        train_ids = tuple(f"train_{position:03d}" for position in range(train_count))
        val_ids = tuple(f"legacy_{position:03d}" for position in range(val_count))
        train_content = ("\n".join(train_ids) + "\n").encode("utf-8")
        # Deliberately use CRLF to prove legacy_dev_val is bound byte-for-byte,
        # not decoded and rewritten with a platform newline.
        val_content = ("\r\n".join(val_ids) + "\r\n").encode("utf-8")
        (parent_dir / "train.txt").write_bytes(train_content)
        (parent_dir / "val.txt").write_bytes(val_content)

        for position, sample_id in enumerate(train_ids):
            mask = np.zeros((24, 24), dtype=np.uint8)
            pattern = position % 4
            if pattern == 0:
                mask[4:6, 4:6] = 255
            elif pattern == 1:
                mask[3:6, 3:6] = 255
            elif pattern == 2:
                mask[2:7, 2:7] = 255
                mask[15:17, 15:17] = 255
            else:
                mask[8:17, 8:17] = 255
            Image.fromarray(mask).save(mask_dir / f"{sample_id}.png")

        fixture_records = lockbox.load_training_mask_records(root / "data", train_ids)
        train_distribution = lockbox._distribution(fixture_records)
        train_distribution["image_shape_histogram"] = train_distribution.pop(
            "mask_shape_histogram"
        )

        parent_manifest = {
            "schema": "evisirst_v2_train_val_split/v1",
            "dataset": lockbox.DATASET,
            "algorithm": {
                "version": lockbox.PARENT_RANKING_ALGORITHM_VERSION,
                "component_connectivity": 8,
                "stratification_fields": [
                    "target_count",
                    "foreground_pixels",
                    "max_target_area",
                ],
            },
            "source_index": {
                "split": "train",
                "relative_path": "fixture/img_idx/train.txt",
                "file_sha256": "3" * 64,
                "ordered_ids_sha256": "4" * 64,
                "sample_count": train_count + val_count,
            },
            "outputs": {
                "train": {
                    "relative_path": f"splits/v2/{lockbox.DATASET}/train.txt",
                    "sample_count": train_count,
                    "file_sha256": hashlib.sha256(train_content).hexdigest(),
                    "ordered_ids_sha256": lockbox._ordered_ids_sha256(train_ids),
                },
                "val": {
                    "relative_path": f"splits/v2/{lockbox.DATASET}/val.txt",
                    "sample_count": val_count,
                    "file_sha256": hashlib.sha256(val_content).hexdigest(),
                    "ordered_ids_sha256": lockbox._ordered_ids_sha256(val_ids),
                },
            },
            "data_identity": {
                "ordered_image_mask_tree_sha256": "1" * 64,
                "ordered_sample_audit_records_sha256": "2" * 64,
            },
            "attribute_distribution": {"train": train_distribution},
            "validation": {
                "train_val_disjoint": True,
                "train_val_union_equals_frozen_train": True,
                "test_was_not_accessed": True,
            },
        }
        (parent_dir / "manifest.json").write_bytes(
            lockbox._canonical_json_bytes(parent_manifest)
        )
        return root / "data", parent_dir, train_ids, val_ids, val_content

    def _fixture_bundle(
        self, root: Path, *, split_seed: int = lockbox.SPLIT_SEED
    ) -> tuple[lockbox.LockboxBundle, Path, Path]:
        dataset_root, parent_dir, _, _, _ = self._fixture(root)
        bundle = lockbox.generate_lockbox_bundle(
            dataset_root=dataset_root,
            parent_split_dir=parent_dir,
            expected_train_count=40,
            expected_legacy_dev_count=10,
            split_seed=split_seed,
            lockbox_count=8,
        )
        return bundle, dataset_root, parent_dir

    def test_fixture_is_deterministic_stratified_and_reads_masks_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, dataset_root, parent_dir = self._fixture_bundle(root)
            repeated = lockbox.generate_lockbox_bundle(
                dataset_root=dataset_root,
                parent_split_dir=parent_dir,
                expected_train_count=40,
                expected_legacy_dev_count=10,
                split_seed=lockbox.SPLIT_SEED,
                lockbox_count=8,
            )
            changed_seed = lockbox.generate_lockbox_bundle(
                dataset_root=dataset_root,
                parent_split_dir=parent_dir,
                expected_train_count=40,
                expected_legacy_dev_count=10,
                split_seed=lockbox.SPLIT_SEED + 1,
                lockbox_count=8,
            )

        self.assertEqual(bundle.manifest_content, repeated.manifest_content)
        self.assertEqual(bundle.confirm_lockbox_ids, repeated.confirm_lockbox_ids)
        self.assertNotEqual(bundle.confirm_lockbox_ids, changed_seed.confirm_lockbox_ids)
        self.assertEqual(len(bundle.train_ids), 32)
        self.assertEqual(len(bundle.confirm_lockbox_ids), 8)
        self.assertFalse(set(bundle.train_ids) & set(bundle.confirm_lockbox_ids))
        self.assertEqual(
            set(bundle.train_ids) | set(bundle.confirm_lockbox_ids),
            {f"train_{position:03d}" for position in range(40)},
        )
        # The fixture intentionally contains no images, test index, test
        # directory, or legacy-validation masks.  Successful construction is
        # an executable guard that none is needed by the generator.
        self.assertTrue(bundle.manifest["lockbox_not_opened_by_model"])
        self.assertFalse(bundle.manifest["generation_audit"]["model_imported_or_executed"])
        self.assertFalse(bundle.manifest["generation_audit"]["metric_computed"])
        self.assertFalse(bundle.manifest["generation_audit"]["test_index_opened"])
        self.assertTrue(
            bundle.manifest["membership_validation"][
                "train_confirm_lockbox_union_equals_parent_v2_train"
            ]
        )

    def test_legacy_dev_val_is_byte_identical_to_parent_val(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root, parent_dir, _, val_ids, val_content = self._fixture(root)
            bundle = lockbox.generate_lockbox_bundle(
                dataset_root=dataset_root,
                parent_split_dir=parent_dir,
                expected_train_count=40,
                expected_legacy_dev_count=10,
                lockbox_count=8,
            )

        self.assertEqual(bundle.legacy_dev_content, val_content)
        self.assertEqual(bundle.legacy_dev_ids, val_ids)
        self.assertEqual(
            bundle.manifest["outputs"]["legacy_dev_val"]["file_sha256"],
            hashlib.sha256(val_content).hexdigest(),
        )
        self.assertTrue(
            bundle.manifest["outputs"]["legacy_dev_val"][
                "byte_identical_to_parent_v2_val"
            ]
        )

    def test_materialize_is_atomic_write_once_and_existing_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, _, _ = self._fixture_bundle(root)
            output_root = root / "output" / "splits" / "psbfr_v1"
            self.assertEqual(lockbox.check_bundle(bundle, output_root), "prospective_only")
            self.assertFalse(output_root.exists())
            self.assertEqual(
                set(lockbox.materialize_bundle(bundle, output_root).values()),
                {"written"},
            )
            self.assertEqual(lockbox.check_bundle(bundle, output_root), "verified_existing")
            self.assertEqual(
                set(lockbox.materialize_bundle(bundle, output_root).values()),
                {"verified_existing"},
            )
            stored_train = output_root / lockbox.DATASET / "train.txt"
            stored_train.write_bytes(stored_train.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(
                lockbox.PSBFRLockboxError, "stored lockbox artifact differs"
            ):
                lockbox.check_bundle(bundle, output_root)

    def test_output_and_parent_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, _, parent_dir = self._fixture_bundle(root)
            real_output = root / "real_output"
            real_output.mkdir()
            linked_output = root / "linked_output"
            linked_output.symlink_to(real_output, target_is_directory=True)
            with self.assertRaisesRegex(lockbox.PSBFRLockboxError, "symlink"):
                lockbox.check_bundle(bundle, linked_output)

            train_path = parent_dir / "train.txt"
            real_train = parent_dir / "real_train.txt"
            train_path.rename(real_train)
            train_path.symlink_to(real_train.name)
            with self.assertRaisesRegex(lockbox.PSBFRLockboxError, "symlink"):
                lockbox.load_parent_snapshot(
                    parent_dir,
                    expected_train_count=40,
                    expected_legacy_dev_count=10,
                )

    def test_parent_manifest_hash_or_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, parent_dir, _, _, _ = self._fixture(root)
            train_path = parent_dir / "train.txt"
            train_path.write_bytes(train_path.read_bytes() + b"extra_member\n")
            with self.assertRaisesRegex(lockbox.PSBFRLockboxError, "count differs"):
                lockbox.load_parent_snapshot(
                    parent_dir,
                    expected_train_count=40,
                    expected_legacy_dev_count=10,
                )

    def test_cli_defaults_to_check_only_and_seed_is_not_configurable(self) -> None:
        args = lockbox.parse_args(["--dataset-root", "/not/opened"])
        self.assertTrue(args.check_only)
        self.assertFalse(args.write)
        self.assertFalse(hasattr(args, "split_seed"))

    def test_direct_script_help_from_outside_repository(self) -> None:
        script = Path(lockbox.__file__).absolute()
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, os.fspath(script), "--help"],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--check-only", completed.stdout)
        self.assertNotIn("--split-seed", completed.stdout)

    def test_repository_frozen_artifacts_are_self_consistent(self) -> None:
        repository_root = Path(lockbox.__file__).absolute().parents[1]
        directory = repository_root / "splits" / "psbfr_v1" / lockbox.DATASET
        expected_names = {
            "train.txt",
            "legacy_dev_val.txt",
            "confirm_lockbox.txt",
            "manifest.json",
        }
        self.assertTrue(directory.is_dir(), "frozen lockbox has not been materialized")
        self.assertEqual({path.name for path in directory.iterdir()}, expected_names)
        manifest_content = (directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_content.decode("utf-8"))
        parent = lockbox.load_parent_snapshot()
        observed: dict[str, tuple[bytes, tuple[str, ...], int]] = {}
        for role, filename, expected_count in (
            ("train", "train.txt", lockbox.NEW_TRAIN_COUNT),
            (
                "legacy_dev_val",
                "legacy_dev_val.txt",
                lockbox.PARENT_LEGACY_DEV_COUNT,
            ),
            ("confirm_lockbox", "confirm_lockbox.txt", lockbox.LOCKBOX_COUNT),
        ):
            content = (directory / filename).read_bytes()
            identifiers = tuple(content.decode("utf-8").splitlines())
            observed[role] = (content, identifiers, expected_count)
            self.assertEqual(len(identifiers), expected_count)
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                manifest["outputs"][role]["file_sha256"],
            )
            self.assertEqual(
                lockbox._ordered_ids_sha256(identifiers),
                manifest["outputs"][role]["ordered_ids_sha256"],
            )
        train_ids = set(observed["train"][1])
        legacy_ids = set(observed["legacy_dev_val"][1])
        confirm_ids = set(observed["confirm_lockbox"][1])
        self.assertFalse(train_ids & legacy_ids)
        self.assertFalse(train_ids & confirm_ids)
        self.assertFalse(legacy_ids & confirm_ids)
        self.assertEqual(train_ids | confirm_ids, set(parent.train_ids))
        self.assertEqual(legacy_ids, set(parent.legacy_dev_ids))
        self.assertEqual(observed["legacy_dev_val"][0], parent.legacy_dev_content)
        self.assertEqual(
            manifest["parent_v2"]["manifest"]["file_sha256"],
            parent.manifest_sha256,
        )
        self.assertTrue(manifest["lockbox_not_opened_by_model"])
        self.assertTrue(
            manifest["historical_training_pool_disclosure"][
                "historical_training_pool_non_pristine"
            ]
        )

    def test_real_rebuild_strictly_matches_frozen_artifacts(self) -> None:
        dataset_root = Path(
            os.environ.get(
                "EVISIRST_DATASET_ROOT", "/home/ly/SCTransNet_main/datasets"
            )
        )
        if not dataset_root.is_dir():
            self.skipTest(f"real dataset root unavailable: {dataset_root}")
        bundle = lockbox.generate_lockbox_bundle(dataset_root=dataset_root)
        self.assertEqual(
            lockbox.check_bundle(bundle, lockbox.DEFAULT_OUTPUT_ROOT),
            "verified_existing",
        )
        self.assertEqual(len(bundle.train_ids), lockbox.NEW_TRAIN_COUNT)
        self.assertEqual(len(bundle.confirm_lockbox_ids), lockbox.LOCKBOX_COUNT)
        self.assertEqual(
            hashlib.sha256(bundle.manifest_content).hexdigest(),
            hashlib.sha256(
                (
                    lockbox.DEFAULT_OUTPUT_ROOT
                    / lockbox.DATASET
                    / "manifest.json"
                ).read_bytes()
            ).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
