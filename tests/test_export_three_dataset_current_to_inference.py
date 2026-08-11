from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import export_three_dataset_current_to_inference as exporter
from experiments import four_dataset_models_seed42_v1 as registry
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
)


torch.set_num_threads(1)


class ThreeDatasetCurrentInferenceExportTests(unittest.TestCase):
    DATASET = "NUAA-SIRST"

    @classmethod
    def setUpClass(cls) -> None:
        model, metadata = registry.build_paper_model(
            "final",
            cls.DATASET,
            seed=42,
            training=True,
        )
        cls.training_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        cls.model_metadata = copy.deepcopy(metadata)
        cls.model_metadata["formal_three_dataset_scope"] = list(exporter.DATASETS)
        cls.model_metadata["formal_training_objective"] = {
            "authority": "three_dataset_tss_off_seed42_v1_run_recipe",
            "historical_builder_default_is_provenance_only": True,
            "method": "final",
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "tss_heads_registered": True,
            "tss_training_forward_computes_logits": True,
            "tss_loss_consumes_logits": False,
            "tss_survival_target_constructed": False,
        }
        cls.model_metadata["tss_off_control"] = exporter._recipe_identity()
        del model

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.training_state
        del cls.model_metadata

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _make_source_fixture(
        self,
        directory: Path,
    ) -> tuple[Path, Path, dict[str, object]]:
        repository = directory / "repo"
        source_root = repository / "results/three_dataset_tss_off_seed42_v1"
        run_dir = (
            source_root
            / "runs"
            / self.DATASET
            / "final_tss_off"
            / "seed_42"
        )
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        frozen_source = repository / "experiments/frozen_source.py"
        frozen_source.parent.mkdir(parents=True)
        frozen_source.write_text("FROZEN = True\n", encoding="utf-8")
        source_sha = self._sha(frozen_source)

        protocol = {
            "schema": exporter.SOURCE_SCHEMA,
            "dataset": self.DATASET,
            "method": "final",
            "training_seed": 42,
            "epochs": 1000,
            "begin_test": 10,
            "eval_every": 10,
            "smoke": False,
            "test_selected": True,
            "selection_is_optimistic": True,
            "checkpoint_roles": ["best_miou", "best_pd"],
            "recipe": exporter._recipe_identity(),
            "training": {
                "tss_enabled": False,
                "tss_requested_weight": 0.0,
                "tss_ratio_cap_applied": False,
                "tss_survival_target_constructed": False,
                "tss_survival_logits_consumed_by_loss": False,
                "tss_training_forward_computes_logits": True,
            },
            "runtime_sources": {
                "fixture": {
                    "path": str(frozen_source.resolve()),
                    "sha256": source_sha,
                }
            },
        }
        protocol_sha = exporter._canonical_sha256(protocol)
        protocol["protocol_sha256"] = protocol_sha
        protocol_path = run_dir / "protocol.json"
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        checkpoint_path = checkpoint_dir / "best_miou.pth.tar"
        historical_metrics = {
            "miou": 0.75,
            "niou": 0.70,
            "pd": 0.95,
            "fa": 1.25e-5,
            "pixel_f1": 0.85,
            "matched_target_count": 19,
            "target_count": 20,
            "test_loss": 0.001,
        }
        torch.save(
            {
                "schema": exporter.SOURCE_SCHEMA,
                "epoch": 10,
                "dataset": self.DATASET,
                "method": "final",
                "seed": 42,
                "checkpoint_role": "best_miou",
                "selection_source": f"test_{self.DATASET}",
                "test_selected": True,
                "selection_is_optimistic": True,
                "test_metrics": historical_metrics,
                "state_dict": self.training_state,
                "model_metadata": self.model_metadata,
                "protocol_sha256": protocol_sha,
                "recipe": exporter._recipe_identity(),
                "requested_tss_weight": 0.0,
                "tss_enabled": False,
            },
            checkpoint_path,
        )
        checkpoint_sha = self._sha(checkpoint_path)
        summary = {
            "schema": exporter.SOURCE_SCHEMA,
            "status": "complete",
            "dataset": self.DATASET,
            "method": "final",
            "seed": 42,
            "epochs": 1000,
            "test_selected": True,
            "selection_is_optimistic": True,
            "recipe": exporter._recipe_identity(),
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "protocol_sha256": protocol_sha,
            "best_miou": {
                "epoch": 10,
                # Exercise the JSON/NumPy-compatible numeric comparison path.
                "metrics": {
                    key: str(value) if key.endswith("count") else value
                    for key, value in historical_metrics.items()
                },
                "path": str(checkpoint_path.resolve()),
            },
            "checkpoints": {
                "best_miou": {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": checkpoint_sha,
                    "bytes": checkpoint_path.stat().st_size,
                }
            },
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        inference_state = registry.export_final_inference_state(
            self.training_state,
            to_cpu=True,
        )
        source_lock: dict[str, object] = {
            "epoch": 10,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_file_sha256": checkpoint_sha,
            "summary_file_sha256": self._sha(summary_path),
            "protocol_file_sha256": self._sha(protocol_path),
            "protocol_payload_sha256": protocol_sha,
            "training_state_sha256": registry.state_dict_sha256(
                self.training_state
            ),
            "inference_state_sha256": registry.state_dict_sha256(
                inference_state
            ),
        }
        return repository, source_root, source_lock

    def _mock_bundle_contexts(
        self,
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        inference_state = registry.export_final_inference_state(
            self.training_state,
            to_cpu=True,
        )
        training_sha = registry.state_dict_sha256(self.training_state)
        inference_sha = registry.state_dict_sha256(inference_state)
        identity = exporter.assert_synthetic_six_output_bitwise_identity(
            self.training_state,
            dataset=self.DATASET,
        )
        architecture = identity.pop("architecture")
        contexts: dict[str, dict[str, object]] = {}
        locks: dict[str, dict[str, object]] = {}
        epochs = {"NUAA-SIRST": 10, "NUDT-SIRST": 20, "IRSTD-1K": 30}
        for index, dataset in enumerate(exporter.DATASETS, start=1):
            token = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
            summary_sha = hashlib.sha256(f"summary:{dataset}".encode()).hexdigest()
            protocol_file_sha = hashlib.sha256(
                f"protocol-file:{dataset}".encode()
            ).hexdigest()
            protocol_payload_sha = hashlib.sha256(
                f"protocol-payload:{dataset}".encode()
            ).hexdigest()
            lock: dict[str, object] = {
                "epoch": epochs[dataset],
                "checkpoint_bytes": 1000 + index,
                "checkpoint_file_sha256": token,
                "summary_file_sha256": summary_sha,
                "protocol_file_sha256": protocol_file_sha,
                "protocol_payload_sha256": protocol_payload_sha,
                "training_state_sha256": training_sha,
                "inference_state_sha256": inference_sha,
            }
            locks[dataset] = lock
            contexts[dataset] = {
                "dataset": dataset,
                "training_state_dict": self.training_state,
                "inference_state_dict": inference_state,
                "binding": {
                    "run_directory": f"/frozen/{dataset}",
                    "checkpoint_path": f"/frozen/{dataset}/best_miou.pth.tar",
                    "checkpoint_file_sha256": token,
                    "checkpoint_bytes": 1000 + index,
                    "checkpoint_role": "best_miou",
                    "epoch": epochs[dataset],
                    "summary_path": f"/frozen/{dataset}/summary.json",
                    "summary_file_sha256": summary_sha,
                    "protocol_path": f"/frozen/{dataset}/protocol.json",
                    "protocol_file_sha256": protocol_file_sha,
                    "protocol_payload_sha256": protocol_payload_sha,
                    "training_state_hash_algorithm": exporter.STATE_HASH_ALGORITHM,
                    "training_state_sha256": training_sha,
                    "inference_state_hash_algorithm": exporter.STATE_HASH_ALGORITHM,
                    "inference_state_sha256": inference_sha,
                    "state_hash_contract": exporter._state_hash_contract(),
                    "runtime_source_sha256": {"fixture": "e" * 64},
                    "historical_test_metrics_canonical_sha256": "f" * 64,
                    "historical_selection": {
                        "selection_source": f"test_{dataset}",
                        "test_selected": True,
                        "selection_is_optimistic": True,
                        "operational_checkpoint_only": True,
                        "unbiased_test_claim_allowed": False,
                        "metrics_recomputed_during_export": False,
                    },
                    "export_operation_official_access": (
                        exporter._official_false_payload()
                    ),
                    "official_boundary_scope": exporter.OFFICIAL_BOUNDARY_SCOPE,
                    "tss_audit": exporter.require_exact_zero_tss_state(
                        self.training_state
                    ),
                    "synthetic_identity": copy.deepcopy(identity),
                    "architecture": copy.deepcopy(architecture),
                },
            }
        return contexts, locks

    def test_exact_zero_tss_and_six_output_identity(self) -> None:
        audit = exporter.require_exact_zero_tss_state(self.training_state)
        self.assertEqual(audit["state_key_count"], 4)
        self.assertEqual(audit["parameter_count"], 98)
        self.assertTrue(audit["all_exact_zero"])
        identity = exporter.assert_synthetic_six_output_bitwise_identity(
            self.training_state,
            dataset=self.DATASET,
        )
        self.assertEqual(identity["six_output_count"], 6)
        self.assertTrue(identity["all_six_bitwise_equal"])
        self.assertEqual(identity["maximum_absolute_difference"], 0.0)
        self.assertTrue(identity["deployed_single_output_equals_sixth"])

        nonzero = dict(self.training_state)
        key = SURVIVAL_STATE_KEYS[0]
        nonzero[key] = torch.ones_like(nonzero[key])
        with self.assertRaisesRegex(ValueError, "not exact zero"):
            exporter.require_exact_zero_tss_state(nonzero)

    def test_source_validator_uses_only_frozen_files_and_discloses_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            repository, source_root, source_lock = self._make_source_fixture(
                Path(directory_text)
            )
            with mock.patch.object(exporter, "REPO_ROOT", repository), mock.patch.dict(
                exporter.SOURCE_LOCKS,
                {self.DATASET: source_lock},
                clear=True,
            ):
                context = exporter.validate_current_source(
                    source_root,
                    self.DATASET,
                )
        binding = context["binding"]
        self.assertEqual(binding["checkpoint_role"], "best_miou")
        self.assertTrue(binding["historical_selection"]["test_selected"])
        self.assertTrue(
            binding["historical_selection"]["selection_is_optimistic"]
        )
        self.assertFalse(
            binding["historical_selection"]["unbiased_test_claim_allowed"]
        )
        flags = binding["export_operation_official_access"]
        self.assertEqual(set(flags), set(exporter.OFFICIAL_FALSE_FLAGS))
        self.assertEqual(len(flags), 5)
        self.assertTrue(all(value is False for value in flags.values()))

    def test_bundle_is_write_once_idempotent_and_committed_last(self) -> None:
        contexts, locks = self._mock_bundle_contexts()

        def source_side_effect(_root: Path, dataset: str):
            return contexts[dataset]

        with tempfile.TemporaryDirectory() as directory_text, mock.patch.dict(
            exporter.SOURCE_LOCKS,
            locks,
            clear=True,
        ), mock.patch.object(
            exporter,
            "validate_current_source",
            side_effect=source_side_effect,
        ):
            directory = Path(directory_text)
            source_root = directory / "source"
            output_root = directory / "bundle"
            first = exporter.export_current_bundle(source_root, output_root)
            self.assertFalse(first["idempotent_existing_bundle"])
            self.assertTrue((output_root / "COMMITTED").is_file())
            self.assertEqual(
                {path.name for path in output_root.iterdir()},
                {"packages", "manifest.json", "COMMITTED"},
            )
            before = {
                path.relative_to(output_root).as_posix(): self._sha(path)
                for path in output_root.rglob("*")
                if path.is_file()
            }
            second = exporter.export_current_bundle(source_root, output_root)
            after = {
                path.relative_to(output_root).as_posix(): self._sha(path)
                for path in output_root.rglob("*")
                if path.is_file()
            }
            self.assertTrue(second["idempotent_existing_bundle"])
            self.assertEqual(before, after)
            package_path = (
                output_root
                / "packages"
                / exporter.PACKAGE_FILENAMES[self.DATASET]
            )
            weights_only_payload = torch.load(
                package_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                len(weights_only_payload["state_dict"]),
                exporter.INFERENCE_STATE_KEY_COUNT,
            )
            model, metadata = exporter.load_exported_current_model(
                package_path,
                expected_dataset=self.DATASET,
            )
            self.assertFalse(hasattr(model, "target_survival"))
            self.assertFalse(model.training)
            self.assertEqual(model.mode, "test")
            self.assertTrue(metadata["strict_load"])
            self.assertEqual(metadata["output"], "sigmoid(out)")

            tampered_package = directory / "tampered_architecture.pth.tar"
            weights_only_payload["architecture"][
                "architecture_manifest_canonical_json_sha256"
            ] = "0" * 64
            torch.save(weights_only_payload, tampered_package)
            with self.assertRaisesRegex(ValueError, "architecture contract differs"):
                exporter.validate_exported_current_package(
                    tampered_package,
                    expected_dataset=self.DATASET,
                )

            committed_path = output_root / "COMMITTED"
            committed = json.loads(committed_path.read_text(encoding="utf-8"))
            committed["export_operation_official_access"][
                "official_test_accessed"
            ] = True
            committed_path.write_text(
                json.dumps(committed, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "official-access flags differ"):
                exporter.validate_committed_bundle(output_root)

    def test_incomplete_bundle_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            output_root = Path(directory_text) / "partial"
            output_root.mkdir()
            (output_root / "partial.txt").write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "incomplete or foreign"):
                exporter.export_current_bundle(
                    Path(directory_text) / "source",
                    output_root,
                )


if __name__ == "__main__":
    unittest.main()
