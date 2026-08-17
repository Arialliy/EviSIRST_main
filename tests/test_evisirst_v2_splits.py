from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments import evisirst_v2_splits as splits


class EviSIRSTV2SplitTest(unittest.TestCase):
    def test_repository_split_artifacts_are_self_consistent(self) -> None:
        repository_root = Path(splits.__file__).resolve().parents[1]
        expected_counts = {
            "NUAA-SIRST": (170, 43),
            "NUDT-SIRST": (530, 133),
            "IRSTD-1K": (640, 160),
        }
        for dataset, (train_count, val_count) in expected_counts.items():
            with self.subTest(dataset=dataset):
                directory = repository_root / "splits" / "v2" / dataset
                manifest = json.loads(
                    (directory / "manifest.json").read_text(encoding="utf-8")
                )
                train_content = (directory / "train.txt").read_bytes()
                val_content = (directory / "val.txt").read_bytes()
                train_ids = tuple(train_content.decode("utf-8").splitlines())
                val_ids = tuple(val_content.decode("utf-8").splitlines())
                self.assertEqual((len(train_ids), len(val_ids)), (train_count, val_count))
                self.assertFalse(set(train_ids) & set(val_ids))
                self.assertEqual(
                    len(train_ids) + len(val_ids),
                    manifest["source_index"]["sample_count"],
                )
                for name, content, identifiers in (
                    ("train", train_content, train_ids),
                    ("val", val_content, val_ids),
                ):
                    output = manifest["outputs"][name]
                    self.assertEqual(
                        hashlib.sha256(content).hexdigest(), output["file_sha256"]
                    )
                    self.assertEqual(
                        splits._ordered_ids_sha256(identifiers),
                        output["ordered_ids_sha256"],
                    )
                self.assertEqual(
                    manifest["grouping"]["mode"], "sample_level_fallback"
                )
                self.assertTrue(manifest["validation"]["test_was_not_accessed"])

    def _fixture(
        self, root: Path, *, count: int = 30
    ) -> tuple[list[splits.SampleRecord], splits.SourceIndexIdentity, list[str]]:
        image_dir = root / "Fixture-SIRST" / "images"
        mask_dir = root / "Fixture-SIRST" / "masks"
        index_path = root / "Fixture-SIRST" / "img_idx" / "train_fixture.txt"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        index_path.parent.mkdir(parents=True)
        sample_ids = [f"sample_{position:03d}" for position in range(count)]
        index_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
        for position, sample_id in enumerate(sample_ids):
            image = np.full((12, 10), (position * 13) % 256, dtype=np.uint8)
            mask = np.zeros((12, 10), dtype=np.uint8)
            # Every fixture has the same stratum.  This makes seed behaviour and
            # group preservation easy to test without relying on external data.
            mask[4:6, 5:7] = 255
            Image.fromarray(image).save(image_dir / f"{sample_id}.png")
            Image.fromarray(mask).save(mask_dir / f"{sample_id}.png")
        records, identity = splits.analyze_fixture_index(
            dataset_root=root,
            dataset="Fixture-SIRST",
            index_path=index_path,
            image_dir=image_dir,
            mask_dir=mask_dir,
        )
        return records, identity, sample_ids

    def _bundle(
        self,
        records: list[splits.SampleRecord],
        identity: splits.SourceIndexIdentity,
        *,
        split_seed: int = 20260811,
        run_seed: int = 42,
        group_mapping: dict[str, str] | None = None,
    ) -> splits.SplitBundle:
        return splits.build_split_bundle(
            dataset="Fixture-SIRST",
            records=records,
            source_index=identity,
            split_seed=split_seed,
            run_seed=run_seed,
            val_fraction=0.20,
            group_mapping=group_mapping,
            group_mapping_metadata={"mode": "explicit_group_mapping"},
        )

    def test_sample_fallback_is_deterministic_and_run_seed_independent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records, identity, sample_ids = self._fixture(Path(temporary))
            first = self._bundle(records, identity, run_seed=1)
            repeated = self._bundle(records, identity, run_seed=999999)

        self.assertEqual(first.train_ids, repeated.train_ids)
        self.assertEqual(first.val_ids, repeated.val_ids)
        self.assertEqual(first.manifest_content, repeated.manifest_content)
        self.assertEqual(len(first.train_ids), 24)
        self.assertEqual(len(first.val_ids), 6)
        self.assertFalse(set(first.train_ids) & set(first.val_ids))
        self.assertEqual(set(first.train_ids) | set(first.val_ids), set(sample_ids))
        self.assertEqual(
            [item for item in sample_ids if item in set(first.val_ids)],
            list(first.val_ids),
        )
        self.assertEqual(first.manifest["grouping"]["mode"], "sample_level_fallback")
        self.assertIn(
            "leakage cannot be ruled out",
            first.manifest["grouping"]["warning"],
        )
        self.assertIsNone(first.manifest["seeds"]["run_seed"])
        self.assertFalse(first.manifest["seeds"]["run_seed_used_for_membership"])
        self.assertTrue(first.manifest["validation"]["train_val_disjoint"])
        self.assertTrue(first.manifest["validation"]["test_was_not_accessed"])
        self.assertEqual(
            first.manifest["data_identity"]["relative_file_identifier_count"], 60
        )

    def test_split_seed_changes_membership_but_not_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records, identity, sample_ids = self._fixture(Path(temporary))
            first = self._bundle(records, identity, split_seed=100)
            second = self._bundle(records, identity, split_seed=101)

        self.assertNotEqual(first.val_ids, second.val_ids)
        for bundle in (first, second):
            source_positions = [
                sample_ids.index(sample_id) for sample_id in bundle.val_ids
            ]
            self.assertEqual(source_positions, sorted(source_positions))

    def test_explicit_groups_never_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records, identity, sample_ids = self._fixture(Path(temporary))
            mapping = {
                sample_id: f"pair_{position // 2:02d}"
                for position, sample_id in enumerate(sample_ids)
            }
            bundle = self._bundle(records, identity, group_mapping=mapping)

        train_groups = {mapping[sample_id] for sample_id in bundle.train_ids}
        val_groups = {mapping[sample_id] for sample_id in bundle.val_ids}
        self.assertFalse(train_groups & val_groups)
        self.assertEqual(bundle.manifest["grouping"]["mode"], "explicit_group_mapping")
        self.assertEqual(bundle.manifest["grouping"]["group_count"], 15)
        self.assertEqual(bundle.manifest["grouping"]["validation_group_count"], 3)
        self.assertEqual(
            bundle.manifest["grouping"]["actual_validation_sample_count"], 6
        )
        self.assertEqual(
            bundle.manifest["grouping"][
                "sample_count_deviation_due_to_group_indivisibility"
            ],
            0,
        )

    def test_grayscale_mask_is_thresholded_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(np.zeros((5, 6), dtype=np.uint8)).save(image_path)
            mask = np.zeros((5, 6), dtype=np.uint8)
            mask[1, 1] = 64
            mask[3, 4] = 200
            Image.fromarray(mask).save(mask_path)
            record = splits.analyze_sample_file(
                dataset_root=root,
                sample_id="gray",
                index_position=0,
                image_path=image_path,
                mask_path=mask_path,
            )

        self.assertEqual(record.mask_encoding, "grayscale_0_255")
        self.assertEqual(record.nonbinary_pixel_count, 2)
        self.assertEqual(record.foreground_pixels, 1)
        self.assertEqual(record.target_count, 1)

    def test_check_only_does_not_write_and_materialization_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, identity, _ = self._fixture(root / "data")
            bundle = self._bundle(records, identity)
            output_root = root / "outputs" / "splits" / "v2"

            self.assertEqual(
                splits.check_bundle(bundle, output_root), "prospective_only"
            )
            self.assertFalse(output_root.exists())
            self.assertEqual(
                splits.materialize_bundle(bundle, output_root),
                {"train": "written", "val": "written", "manifest": "written"},
            )
            self.assertEqual(
                splits.check_bundle(bundle, output_root), "verified_existing"
            )
            self.assertEqual(
                splits.materialize_bundle(bundle, output_root),
                {
                    "train": "verified_existing",
                    "val": "verified_existing",
                    "manifest": "verified_existing",
                },
            )
            stored = json.loads(
                (output_root / "Fixture-SIRST" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stored["source_index"]["split"], "train")
            self.assertFalse(stored["test_access"]["test_index_opened"])

    def test_group_mapping_must_cover_frozen_index_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping_path = root / "groups.json"
            mapping_path.write_text(
                json.dumps({"groups": {"sample_000": "group_0"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(splits.EviSIRSTSplitError, "cover.*exactly"):
                splits.load_group_mapping(mapping_path, ["sample_000", "sample_001"])

    def test_cli_defaults_to_check_only(self) -> None:
        args = splits.parse_args(["--dataset-root", "/tmp/not-opened"])
        self.assertFalse(args.write)
        self.assertTrue(args.check_only)

    def test_direct_script_imports_from_outside_repository(self) -> None:
        script = Path(splits.__file__).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--check-only", completed.stdout)


if __name__ == "__main__":
    unittest.main()
