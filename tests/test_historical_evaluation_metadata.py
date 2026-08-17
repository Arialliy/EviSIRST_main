from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

from tools import migrate_historical_evaluation_metadata as migration
from tools import verify_checkpoint_artifacts as artifact_checker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation" / "three_folders_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield from _strings(key)
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


class HistoricalEvaluationMetadataTest(unittest.TestCase):
    def _documents(self):
        for spec in migration.RESULT_SPECS:
            path = EVALUATION_ROOT / spec.result_relative_path
            yield spec, json.loads(path.read_text(encoding="utf-8"))

    def test_repository_tree_is_complete_and_canonically_migrated(self) -> None:
        self.assertEqual(len(migration.RESULT_SPECS), 12)
        self.assertEqual(migration.migrate_tree(EVALUATION_ROOT, write=False), ())
        self.assertEqual(migration.main(["--root", str(EVALUATION_ROOT)]), 0)

    def test_measurements_are_frozen_and_machine_paths_are_absent(self) -> None:
        for spec, document in self._documents():
            with self.subTest(result=spec.result_relative_path):
                self.assertEqual(
                    migration.measurement_sha256(document),
                    spec.measurement_sha256,
                )
                self.assertNotIn("dataset_root", document)
                checkpoint = document["checkpoint"]
                self.assertEqual(checkpoint["path"], spec.checkpoint_relative_path)
                self.assertFalse(Path(checkpoint["path"]).is_absolute())
                self.assertFalse(
                    any(value.startswith("/") for value in _strings(document))
                )
                migrated = document["metadata_migration"]
                self.assertEqual(migrated["schema"], migration.MIGRATION_SCHEMA)
                self.assertFalse(migrated["measurement_values_changed"])

    def test_selection_classification_matches_audited_payload_evidence(self) -> None:
        optimistic_count = 0
        fixed_count = 0
        expected_label_basis = {
            "dual_role_test_selected": (
                "normalized_from_checkpoint_role_and_payload_selection_metric"
            ),
            "test_selected_deployment_export": (
                "normalized_from_package_historical_selection_and_"
                "source_checkpoint_role"
            ),
            "fixed_epoch_endpoint": (
                "copied_from_hash_bound_historical_summary_checkpoint_selection_rule"
            ),
        }
        for spec, document in self._documents():
            with self.subTest(result=spec.result_relative_path):
                checkpoint = document["checkpoint"]
                provenance = checkpoint["selection_provenance"]
                self.assertEqual(checkpoint["source_selection"], spec.source_selection)
                self.assertEqual(provenance["schema"], migration.PROVENANCE_SCHEMA)
                self.assertEqual(
                    provenance["source_checkpoint_sha256"],
                    spec.checkpoint_sha256,
                )
                self.assertEqual(
                    provenance["source_selection_label_basis"],
                    expected_label_basis[spec.source_kind],
                )
                if spec.source_kind == "fixed_epoch_endpoint":
                    fixed_count += 1
                    self.assertFalse(checkpoint["selection_is_optimistic"])
                    self.assertFalse(provenance["test_selected"])
                    self.assertEqual(provenance["classification"], "fixed-endpoint")
                    self.assertEqual(
                        provenance["final_test_once_status"],
                        "unsupported_without_execution_ledger",
                    )
                else:
                    optimistic_count += 1
                    self.assertTrue(checkpoint["selection_is_optimistic"])
                    self.assertTrue(provenance["test_selected"])
                    self.assertEqual(
                        provenance["classification"], "historical-test-selected"
                    )
                    self.assertFalse(provenance["unbiased_test_claim_supported"])
        self.assertEqual(optimistic_count, 9)
        self.assertEqual(fixed_count, 3)

    def test_legacy_machine_paths_reimport_to_the_same_documents(self) -> None:
        for spec, current in self._documents():
            with self.subTest(result=spec.result_relative_path):
                legacy = copy.deepcopy(current)
                legacy["dataset_root"] = "/machine/private/datasets"
                checkpoint = legacy["checkpoint"]
                checkpoint["path"] = (
                    f"/machine/private/repository/{spec.checkpoint_relative_path}"
                )
                checkpoint.pop("source_selection")
                checkpoint.pop("selection_is_optimistic")
                checkpoint.pop("selection_provenance")
                legacy.pop("metadata_migration")
                self.assertEqual(
                    migration.migrate_document(
                        legacy,
                        result_relative_path=spec.result_relative_path,
                    ),
                    current,
                )

    def test_artifact_manifest_covers_every_result_checkpoint(self) -> None:
        artifacts = artifact_checker.load_manifest(
            PROJECT_ROOT / "artifacts" / "checkpoints.json"
        )
        by_path = {artifact["relative_path"]: artifact for artifact in artifacts}
        self.assertEqual(len(artifacts), 13)
        for spec in migration.RESULT_SPECS:
            with self.subTest(checkpoint=spec.checkpoint_relative_path):
                artifact = by_path[spec.checkpoint_relative_path]
                self.assertEqual(artifact["dataset"], spec.training_dataset)
                self.assertEqual(artifact["model"], "EviSIRST")
                self.assertEqual(artifact["sha256"], spec.checkpoint_sha256)
                self.assertEqual(artifact["size_bytes"], spec.checkpoint_size_bytes)
                self.assertIsNone(artifact["download_url"])
                self.assertEqual(artifact["download_url_status"], "TBD")

    def test_source_evidence_hashes_match_the_repository_files(self) -> None:
        self.assertEqual(
            _sha256(PROJECT_ROOT / "train_dual_role_test_selected.py"),
            migration.DUAL_ROLE_SOURCE_SHA256,
        )
        self.assertEqual(
            _sha256(PROJECT_ROOT / "run_sirst3_experiment.py"),
            migration.SIRST3_RUNNER_SHA256,
        )
        self.assertEqual(
            _sha256(PROJECT_ROOT / "results/SIRST3/cross_dataset_results.json"),
            migration.SIRST3_SUMMARY_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
