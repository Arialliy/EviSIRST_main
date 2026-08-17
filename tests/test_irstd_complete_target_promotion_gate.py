from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import run_irstd_complete_target_promotion_gate as gate


REAL_PROJECT_ROOT = Path(gate.__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _make_sources(repository: Path) -> None:
    all_paths = set(gate._BASELINE_SOURCE_PATHS.values())
    all_paths.update(gate._VARIANT_SOURCE_PATHS.values())
    all_paths.update(gate._DIAGNOSTIC_SOURCE_PATHS.values())
    for relative in sorted(all_paths):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source: {relative}\n", encoding="utf-8")
    shutil.copy2(
        gate.__file__, repository / gate.GATE_SOURCE_RELATIVE_PATH
    )


def _source_protocol(
    repository: Path, *, paths: dict[str, str], schema: str
) -> dict[str, object]:
    files = {
        name: {
            "relative_path": relative,
            "sha256": _sha256(repository / relative),
        }
        for name, relative in sorted(paths.items())
    }
    return {
        "source_set_schema": schema,
        "source_files": files,
        "source_tree_sha256": gate._canonical_sha256(files),
    }


def _copy_canonical_split(repository: Path) -> None:
    for entry in gate.CANONICAL_SPLIT_FILES.values():
        source = REAL_PROJECT_ROOT / entry["relative_path"]
        destination = repository / entry["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _split() -> dict[str, object]:
    return {
        "schema": "evisirst_v2_train_val_split/v1",
        "manifest_relative_path": gate.CANONICAL_SPLIT_FILES["manifest"][
            "relative_path"
        ],
        "manifest_sha256": gate.CANONICAL_MANIFEST_SHA256,
        "split_seed": gate.CANONICAL_SPLIT_SEED,
        "data_tree_sha256": gate.CANONICAL_DATA_TREE_SHA256,
        "data_tree_verified": True,
        "train_count": gate.CANONICAL_TRAIN_COUNT,
        "val_count": gate.CANONICAL_VAL_COUNT,
        "test_index_opened": False,
        "outputs": {
            role: {
                "relative_path": gate.CANONICAL_SPLIT_FILES[role]["relative_path"],
                "file_sha256": gate.CANONICAL_SPLIT_FILES[role]["sha256"],
                "ordered_ids_sha256": gate.CANONICAL_SPLIT_FILES[role][
                    "ordered_ids_sha256"
                ],
                "sample_count": gate.CANONICAL_SPLIT_FILES[role]["sample_count"],
            }
            for role in ("train", "val")
        },
    }


def _mechanism(*, recall_numerator: int, area_ratio: float) -> dict[str, object]:
    return {
        "schema": gate.MECHANISM_SCHEMA,
        "prediction_threshold_rule": "probability>0.5",
        "target_threshold_rule": "target>0.5",
        "component_connectivity": 2,
        "component_neighborhood": "8-connected",
        "assignment_algorithm": "Hungarian/scipy.optimize.linear_sum_assignment",
        "match_rule": "centroid_distance<3",
        "image_count": gate.CANONICAL_VAL_COUNT,
        "target_component_count": 110,
        "predicted_component_count": 105,
        "matched_component_count": 100,
        "matched_overlap_pixel_count": recall_numerator,
        "matched_target_pixel_count": 100,
        "matched_target_pixel_recall": recall_numerator / 100.0,
        "matched_component_area_ratio": area_ratio,
        "centroid_error": 0.5,
    }


def _records(
    *, miou: float, pd: float, fa: float, mechanism: dict[str, object] | None
) -> list[dict[str, object]]:
    output = []
    for epoch in range(1, gate.EPOCHS + 1):
        selected = epoch == 7
        output.append(
            {
                "epoch": epoch,
                "data_role": "val",
                "mIoU": miou if selected else 0.1,
                "Fa": fa if selected else 0.02,
                "Pd": pd if selected else 0.5,
                "evaluation_head": "out",
                "metrics": (
                    {"complete_target_mechanism_diagnostics": mechanism}
                    if selected and mechanism is not None
                    else {}
                ),
            }
        )
    return output


def _selection(records: list[dict[str, object]], candidate_sha: str) -> dict[str, object]:
    provenance = gate.validation_selection.select_independent_checkpoint(records)
    return {
        "schema": "evisirst_validation_selection_payload/v1",
        "data_role": "val",
        "source_selection": "evisirst_v2_validation_split",
        "selected_epoch": provenance["selected"]["epoch"],
        "selected_candidate": {
            "relative_path": "candidates/epoch_0007.pth.tar",
            "file_sha256": candidate_sha,
        },
        "retention_frontier_epochs": [7],
        "selection_provenance": provenance,
        "selection_is_optimistic": False,
        "optimistic": False,
    }


def _baseline_identity(repository: Path) -> dict[str, object]:
    protocol = _source_protocol(
        repository,
        paths=gate._BASELINE_SOURCE_PATHS,
        schema="evisirst_validation_selected_source_set/v2",
    )
    identity: dict[str, object] = {
        "schema": gate.BASELINE_TRAINING_SCHEMA + "/run_identity",
        "model": "EviSIRST",
        "dataset": gate.DATASET,
        "architecture_seed": gate.ARCHITECTURE_SEED,
        "run_seed": gate.RUN_SEED,
        "target_mode": gate.TARGET_MODE,
        "epochs": gate.EPOCHS,
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
        "selection_rule": gate.validation_selection.INDEPENDENT_RULE_VERSION,
        "determinism_protocol": protocol,
        "manifest_sha256": gate.CANONICAL_MANIFEST_SHA256,
        "split_seed": gate.CANONICAL_SPLIT_SEED,
        "data_tree_sha256": gate.CANONICAL_DATA_TREE_SHA256,
        "grouping_policy": {"mode": "fixture"},
        "train_count": gate.CANONICAL_TRAIN_COUNT,
        "val_count": gate.CANONICAL_VAL_COUNT,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = gate._canonical_sha256(identity)
    return identity


def _preregistered_gate() -> dict[str, object]:
    return {
        "schema": gate.PREREGISTERED_GATE_SCHEMA,
        "status": "TBD",
        "paired_baseline": {
            "training_schema": gate.BASELINE_TRAINING_SCHEMA,
            "dataset": gate.DATASET,
            "target_mode": gate.TARGET_MODE,
            "architecture_seed": gate.ARCHITECTURE_SEED,
            "run_seed": gate.RUN_SEED,
            "summary_relative_path": gate.BASELINE_SUMMARY_RELATIVE_PATH,
        },
        "variant": {
            "training_schema": gate.VARIANT_TRAINING_SCHEMA,
            "dataset": gate.DATASET,
            "target_mode": gate.TARGET_MODE,
            "architecture_seed": gate.ARCHITECTURE_SEED,
            "run_seed": gate.RUN_SEED,
        },
        "primary": {
            "metric": "selected_validation_mIoU",
            "comparison": "variant_minus_paired_baseline_raw",
            "operator": ">=",
            "minimum_delta": gate.PRIMARY_MINIMUM_DELTA,
        },
        "safety": {
            "pd_delta_definition": "variant_minus_paired_baseline_raw",
            "pd_failure_operator": "<",
            "pd_failure_threshold": gate.SAFETY_PD_FAILURE_THRESHOLD,
            "fa_delta_definition": "variant_minus_paired_baseline_raw",
            "fa_failure_operator": ">",
            "fa_failure_threshold": gate.SAFETY_FA_FAILURE_THRESHOLD,
        },
        "mechanism": {
            "pass_rule": "at_least_one_metric_strictly_improves",
            "measurement_protocol": {
                "data_role": "same_canonical_validation_split",
                "evaluation_head": "out",
                "selection_allowed": False,
                "paired_baseline_artifact": {
                    "input_checkpoint_relative_path": (
                        gate.BASELINE_CHECKPOINT_RELATIVE_PATH
                    ),
                    "output_artifact_template": (
                        gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH
                    ),
                },
                "variant_selected_record": (
                    "selected_validation_record.metrics."
                    "complete_target_mechanism_diagnostics"
                ),
            },
            "metrics": [
                {
                    "name": "matched_target_pixel_recall",
                    "comparison": "variant_minus_paired_baseline_raw",
                    "operator": ">",
                    "source_schema": gate.MECHANISM_SCHEMA,
                },
                {
                    "name": (
                        "matched_component_area_ratio_closeness_to_one"
                    ),
                    "source_metric": "matched_component_area_ratio",
                    "transform": "-abs(raw_value-1.0)",
                    "comparison": (
                        "variant_minus_paired_baseline_transformed"
                    ),
                    "operator": ">",
                    "source_schema": gate.MECHANISM_SCHEMA,
                },
            ],
        },
        "decision": {
            "single_seed_required_before_expansion": True,
            "expand_to_three_runtime_seeds_only_if_all_gates_pass": True,
            "public_test_allowed": False,
            "public_test_status": (
                "unsupported_until_separate_gate_extension"
            ),
        },
    }


def _variant_identity(repository: Path) -> dict[str, object]:
    protocol = _source_protocol(
        repository,
        paths=gate._VARIANT_SOURCE_PATHS,
        schema="evisirst_irstd_complete_target_source_set/v1",
    )
    crop_policy = {"schema": "fixture-crop/v1", "margin": 8}
    identity: dict[str, object] = {
        "schema": gate.VARIANT_TRAINING_SCHEMA + "/run_identity",
        "experiment": {
            "schema": gate.VARIANT_EXPERIMENT_SCHEMA,
            "name": "IRSTD-1K complete-target crop v1",
            "status": "experimental_validation_only",
            "only_train_crop_policy_differs_from_R1": True,
            "public_test_supported": False,
            "public_test_gate_status": "unsupported_until_separate_gate_extension",
        },
        "model": "EviSIRST",
        "dataset": gate.DATASET,
        "architecture_seed": gate.ARCHITECTURE_SEED,
        "run_seed": gate.RUN_SEED,
        "target_mode": gate.TARGET_MODE,
        "epochs": gate.EPOCHS,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "val_interval": 1,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "optimizer_hyperparameters": {"betas": [0.9, 0.999]},
        "loss": "sum_of_six_BCELoss_mean_terms",
        "deep_supervision_probability_heads": 6,
        "deep_supervision_weights": [1.0] * 6,
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "evaluation_head": "out",
        "evaluation_supplementary_diagnostics": {"selection_allowed": False},
        "execution_contract": {"data_loader_workers": 0},
        "runtime_identity": {"requested_device": "cuda:0"},
        "selection_rule": gate.validation_selection.INDEPENDENT_RULE_VERSION,
        "determinism_protocol": protocol,
        "crop_policy": crop_policy,
        "crop_policy_identity_sha256": gate._canonical_sha256(crop_policy),
        "manifest_sha256": gate.CANONICAL_MANIFEST_SHA256,
        "split_seed": gate.CANONICAL_SPLIT_SEED,
        "data_tree_sha256": gate.CANONICAL_DATA_TREE_SHA256,
        "canonical_split_contract": {
            "split_root_relative_path": "splits/v2",
            "manifest_sha256": gate.CANONICAL_MANIFEST_SHA256,
            "data_tree_sha256": gate.CANONICAL_DATA_TREE_SHA256,
            "train_count": gate.CANONICAL_TRAIN_COUNT,
            "val_count": gate.CANONICAL_VAL_COUNT,
        },
        "grouping_policy": {"mode": "fixture"},
        "train_count": gate.CANONICAL_TRAIN_COUNT,
        "val_count": gate.CANONICAL_VAL_COUNT,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "promotion_gate": _preregistered_gate(),
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = gate._canonical_sha256(identity)
    return identity


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        f"state_{index:03d}": torch.tensor(float(index))
        for index in range(564)
    }


class GateFixture:
    def __init__(
        self,
        repository: Path,
        *,
        baseline_miou: float = 0.700,
        variant_miou: float = 0.702,
        baseline_pd: float = 0.900,
        variant_pd: float = 0.899,
        baseline_fa: float = 0.010,
        variant_fa: float = 0.011,
        baseline_recall: int = 80,
        variant_recall: int = 85,
        baseline_area: float = 1.10,
        variant_area: float = 1.05,
    ) -> None:
        self.repository = repository
        _copy_canonical_split(repository)
        _make_sources(repository)
        self.baseline_identity = _baseline_identity(repository)
        self.variant_identity = _variant_identity(repository)
        self.baseline_mechanism = _mechanism(
            recall_numerator=baseline_recall, area_ratio=baseline_area
        )
        self.variant_mechanism = _mechanism(
            recall_numerator=variant_recall, area_ratio=variant_area
        )
        self.baseline_records = _records(
            miou=baseline_miou,
            pd=baseline_pd,
            fa=baseline_fa,
            mechanism=None,
        )
        self.variant_records = _records(
            miou=variant_miou,
            pd=variant_pd,
            fa=variant_fa,
            mechanism=self.variant_mechanism,
        )
        self.baseline_candidate_sha = "b" * 64
        self.variant_candidate_sha = "c" * 64
        self.baseline_selection = _selection(
            self.baseline_records, self.baseline_candidate_sha
        )
        self.variant_selection = _selection(
            self.variant_records, self.variant_candidate_sha
        )
        self.split = _split()
        self._write_checkpoints()
        self._write_summaries()
        self._write_diagnostic()

    def _base_checkpoint(self, identity: dict[str, object]) -> dict[str, object]:
        return {
            "schema": gate.BASELINE_CHECKPOINT_SCHEMA,
            "model": "EviSIRST",
            "dataset": gate.DATASET,
            "checkpoint_role": "validation_selected",
            "epoch": 7,
            "seed": gate.ARCHITECTURE_SEED,
            "architecture_seed": gate.ARCHITECTURE_SEED,
            "run_seed": gate.RUN_SEED,
            "state_dict": _state_dict(),
            "target_mode": gate.TARGET_MODE,
            "normalization": {"mean": 100.0, "std": 30.0},
            "normalization_provenance": {},
            "training": identity,
            "training_identity_sha256": identity["identity_sha256"],
            "split_provenance": self.split,
            "split_seed": gate.CANONICAL_SPLIT_SEED,
            "split_manifest_sha256": gate.CANONICAL_MANIFEST_SHA256,
            "data_tree_sha256": gate.CANONICAL_DATA_TREE_SHA256,
            "data_tree_verified": True,
            "selection_provenance": self.baseline_selection[
                "selection_provenance"
            ],
            "source_selection": "evisirst_v2_validation_split",
            "selection_is_optimistic": False,
            "optimistic": False,
            "test_split_accessed": False,
            "model_metadata": {},
            "smoke": False,
        }

    def _write_checkpoints(self) -> None:
        baseline = self._base_checkpoint(self.baseline_identity)
        baseline_path = self.repository / gate.BASELINE_CHECKPOINT_RELATIVE_PATH
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(baseline, baseline_path)
        self.baseline_checkpoint_sha = _sha256(baseline_path)

        variant = self._base_checkpoint(self.variant_identity)
        variant.update(
            {
                "schema": gate.VARIANT_CHECKPOINT_SCHEMA,
                "checkpoint_role": "experimental_validation_selected",
                "selection_provenance": self.variant_selection[
                    "selection_provenance"
                ],
                "crop_audit_commit_status": "complete",
                "crop_policy": self.variant_identity["crop_policy"],
                "crop_policy_identity_sha256": self.variant_identity[
                    "crop_policy_identity_sha256"
                ],
                "experiment_schema": gate.VARIANT_EXPERIMENT_SCHEMA,
                "experiment_status": "experimental_validation_only",
                "only_train_crop_policy_differs_from_R1": True,
                "promotion_gate": _preregistered_gate(),
                "public_test_gate_status": "unsupported_until_separate_gate_extension",
                "public_test_supported": False,
                "search_run_crop_audit_history": [],
                "search_run_crop_audit_state": {},
                "search_run_crop_audit_summary": {},
                "search_run_crop_audit_through_epoch": gate.EPOCHS,
                "selected_candidate_sha256": self.variant_candidate_sha,
                "selected_checkpoint_crop_audit_history": [],
                "selected_checkpoint_crop_audit_state": {},
                "selected_checkpoint_crop_audit_summary": {},
                "selected_checkpoint_crop_audit_through_epoch": 7,
            }
        )
        variant_path = self.repository / gate.VARIANT_CHECKPOINT_RELATIVE_PATH
        variant_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(variant, variant_path)
        self.variant_checkpoint_sha = _sha256(variant_path)

    def _base_summary(self) -> dict[str, object]:
        return {
            "schema": gate.BASELINE_TRAINING_SCHEMA + "/summary",
            "status": "complete",
            "dataset": gate.DATASET,
            "checkpoint": gate.BASELINE_CHECKPOINT_RELATIVE_PATH,
            "checkpoint_role": "validation_selected",
            "selected_epoch": 7,
            "architecture_seed": gate.ARCHITECTURE_SEED,
            "run_seed": gate.RUN_SEED,
            "target_mode": gate.TARGET_MODE,
            "split_provenance": self.split,
            "selection": self.baseline_selection,
            "training_history": [
                {"epoch": epoch} for epoch in range(1, gate.EPOCHS + 1)
            ],
            "validation_history": self.baseline_records,
            "candidate_artifacts": {},
            "normalization": {"mean": 100.0, "std": 30.0},
            "source_selection": "evisirst_v2_validation_split",
            "selection_is_optimistic": False,
            "optimistic": False,
            "test_split_accessed": False,
            "smoke": False,
            "elapsed_seconds": 100.0,
        }

    def _write_summaries(self) -> None:
        self.baseline_summary = self._base_summary()
        _write_json(
            self.repository / gate.BASELINE_SUMMARY_RELATIVE_PATH,
            self.baseline_summary,
        )
        self.baseline_summary_sha = _sha256(
            self.repository / gate.BASELINE_SUMMARY_RELATIVE_PATH
        )

        selected = self.variant_records[6]
        variant = self._base_summary()
        variant.update(
            {
                "schema": gate.VARIANT_TRAINING_SCHEMA + "/summary",
                "checkpoint": gate.VARIANT_CHECKPOINT_RELATIVE_PATH,
                "checkpoint_role": "experimental_validation_selected",
                "selection": self.variant_selection,
                "validation_history": self.variant_records,
                "crop_audit_commit_status": "complete",
                "crop_policy": self.variant_identity["crop_policy"],
                "crop_policy_identity_sha256": self.variant_identity[
                    "crop_policy_identity_sha256"
                ],
                "experiment_schema": gate.VARIANT_EXPERIMENT_SCHEMA,
                "experiment_status": "experimental_validation_only",
                "final_checkpoint_sha256": self.variant_checkpoint_sha,
                "only_train_crop_policy_differs_from_R1": True,
                "promotion_gate": _preregistered_gate(),
                "public_test_gate_status": "unsupported_until_separate_gate_extension",
                "public_test_supported": False,
                "run_identity": self.variant_identity,
                "search_run_crop_audit_history": [],
                "search_run_crop_audit_state": {},
                "search_run_crop_audit_summary": {},
                "search_run_crop_audit_through_epoch": gate.EPOCHS,
                "selected_candidate_sha256": self.variant_candidate_sha,
                "selected_checkpoint_crop_audit_history": [],
                "selected_checkpoint_crop_audit_state": {},
                "selected_checkpoint_crop_audit_summary": {},
                "selected_checkpoint_crop_audit_through_epoch": 7,
                "selected_validation_record": selected,
                "selected_validation_record_sha256": gate._canonical_sha256(
                    selected
                ),
                "training_identity_sha256": self.variant_identity[
                    "identity_sha256"
                ],
            }
        )
        self.variant_summary = variant
        _write_json(
            self.repository / gate.VARIANT_SUMMARY_RELATIVE_PATH,
            self.variant_summary,
        )

    def _write_diagnostic(self) -> None:
        diagnostic_protocol = _source_protocol(
            self.repository,
            paths=gate._DIAGNOSTIC_SOURCE_PATHS,
            schema="evisirst_paired_baseline_diagnostic_source_set/v1",
        )
        diagnostic_sources = {
            "schema": diagnostic_protocol["source_set_schema"],
            "files": diagnostic_protocol["source_files"],
            "source_tree_sha256": diagnostic_protocol["source_tree_sha256"],
        }
        selected = self.baseline_records[6]
        self.diagnostic = {
            "schema": gate.BASELINE_DIAGNOSTIC_SCHEMA,
            "status": "complete",
            "dataset": gate.DATASET,
            "data_role": "val",
            "diagnostic_only": True,
            "selection_allowed": False,
            "selected_on_same_validation_split": True,
            "checkpoint": {
                "relative_path": gate.BASELINE_CHECKPOINT_RELATIVE_PATH,
                "sha256": self.baseline_checkpoint_sha,
                "schema": gate.BASELINE_CHECKPOINT_SCHEMA,
                "checkpoint_role": "validation_selected",
                "selected_epoch": 7,
                "architecture_seed": gate.ARCHITECTURE_SEED,
                "run_seed": gate.RUN_SEED,
                "target_mode": gate.TARGET_MODE,
                "training_identity_sha256": self.baseline_identity[
                    "identity_sha256"
                ],
                "selection_provenance_sha256": gate._canonical_sha256(
                    self.baseline_selection["selection_provenance"]
                ),
                "strict_564_key_load": True,
            },
            "completed_run_summary": {
                "relative_path": gate.BASELINE_SUMMARY_RELATIVE_PATH,
                "sha256": self.baseline_summary_sha,
                "selected_validation_record_sha256": gate._canonical_sha256(
                    selected
                ),
            },
            "split": {
                "manifest_relative_path": gate.CANONICAL_SPLIT_FILES[
                    "manifest"
                ]["relative_path"],
                "manifest_sha256": gate.CANONICAL_MANIFEST_SHA256,
                "val_index_relative_path": gate.CANONICAL_SPLIT_FILES["val"][
                    "relative_path"
                ],
                "val_index_sha256": gate.CANONICAL_SPLIT_FILES["val"]["sha256"],
                "data_tree_sha256": gate.CANONICAL_DATA_TREE_SHA256,
                "data_tree_verified": True,
                "train_count": gate.CANONICAL_TRAIN_COUNT,
                "val_count": gate.CANONICAL_VAL_COUNT,
            },
            "sources": {
                "checkpoint_training_source_tree_sha256": self.baseline_identity[
                    "determinism_protocol"
                ]["source_tree_sha256"],
                "checkpoint_training_sources_currently_verified": True,
                "diagnostic": diagnostic_sources,
            },
            "evaluation": {
                "evaluation_head": "out",
                "sample_count": gate.CANONICAL_VAL_COUNT,
                "one_model_forward_per_sample": True,
                "prediction_or_target_arrays_written": False,
            },
            "metrics": {
                "miou": selected["mIoU"],
                "fa": selected["Fa"],
                "pd": selected["Pd"],
                "complete_target_mechanism_diagnostics": self.baseline_mechanism,
            },
            "test_split_accessed": False,
        }
        _write_json(
            self.repository / gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH,
            self.diagnostic,
        )


class FixedCliAndDecisionTest(unittest.TestCase):
    def test_cli_has_no_path_seed_split_metric_or_threshold_surface(self) -> None:
        args = gate.parse_args([])
        self.assertEqual(vars(args), {})
        with self.assertRaises(SystemExit):
            gate.parse_args(["--checkpoint", "/tmp/model"])

    def test_pass_writes_all_rules_but_never_allows_public_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            with mock.patch.object(gate, "PROJECT_ROOT", repository):
                evaluated = gate.evaluate_payload()
                self.assertFalse(
                    (repository / gate.OUTPUT_RELATIVE_PATH).exists()
                )
                output = gate.run(gate.parse_args([]))
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(result, evaluated)
                self.assertEqual(gate.validate_existing_result(), evaluated)
                self.assertEqual(result["schema"], gate.RESULT_SCHEMA)
                self.assertEqual(result["decision"]["result"], "PASS")
                self.assertTrue(result["decision"]["overall_passed"])
                self.assertTrue(
                    result["decision"][
                        "three_runtime_seed_validation_expansion_allowed"
                    ]
                )
                self.assertFalse(result["decision"]["public_test_allowed"])
                self.assertFalse(result["public_test_allowed"])
                self.assertFalse(result["test_split_accessed"])
                self.assertAlmostEqual(result["primary_gate"]["delta_raw"], 0.002)
                self.assertIn("failure_rule", result["safety_gate"])
                self.assertIn("pass_rule", result["mechanism_gate"])
                with self.assertRaises(FileExistsError):
                    gate.run(gate.parse_args([]))

    def test_existing_result_tamper_and_json_number_type_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            with mock.patch.object(gate, "PROJECT_ROOT", repository):
                output = gate.run()
                result = json.loads(output.read_text(encoding="utf-8"))
                result["decision"]["overall_passed"] = False
                _write_json(output, result)
                with self.assertRaisesRegex(
                    gate.PromotionGateError, "fresh canonical evaluation"
                ):
                    gate.validate_existing_result()

        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            with mock.patch.object(gate, "PROJECT_ROOT", repository):
                output = gate.run()
                result = json.loads(output.read_text(encoding="utf-8"))
                result["epochs"] = 1000.0
                _write_json(output, result)
                with self.assertRaisesRegex(
                    gate.PromotionGateError, "fresh canonical evaluation"
                ):
                    gate.validate_existing_result()

    def test_existing_result_duplicate_key_and_nan_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            output = repository / gate.OUTPUT_RELATIVE_PATH
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                '{"schema":"x","schema":"y"}\n', encoding="utf-8"
            )
            with mock.patch.object(gate, "PROJECT_ROOT", repository):
                with self.assertRaisesRegex(
                    gate.PromotionGateError, "strict finite"
                ):
                    gate.validate_existing_result()

        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            output = repository / gate.OUTPUT_RELATIVE_PATH
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"metric":NaN}\n', encoding="utf-8")
            with mock.patch.object(gate, "PROJECT_ROOT", repository):
                with self.assertRaisesRegex(
                    gate.PromotionGateError, "strict finite"
                ):
                    gate.validate_existing_result()

    def test_failed_gate_is_a_valid_immutable_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(
                repository,
                variant_miou=0.7005,
                variant_recall=75,
                variant_area=1.20,
            )
            with mock.patch.object(gate, "PROJECT_ROOT", repository):
                output = gate.run()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["decision"]["result"], "FAIL")
            self.assertFalse(result["primary_gate"]["passed"])
            self.assertFalse(result["mechanism_gate"]["passed"])
            self.assertFalse(
                result["decision"]["three_runtime_seed_validation_expansion_allowed"]
            )

    def test_safety_fails_only_when_both_strict_conditions_hold(self) -> None:
        common = {
            "baseline_artifacts": {"summary": {}, "checkpoint": {}},
            "variant_artifacts": {"summary": {}, "checkpoint": {}},
            "diagnostic_artifact": {},
            "gate_source_artifact": {},
            "baseline_identity": {"identity_sha256": "a" * 64},
            "variant_identity": {"identity_sha256": "b" * 64},
            "baseline_source_tree": "c" * 64,
            "variant_source_tree": "d" * 64,
            "baseline_selection": {},
            "variant_selection": {},
            "baseline_mechanism": _mechanism(
                recall_numerator=80, area_ratio=1.1
            ),
            "variant_mechanism": _mechanism(
                recall_numerator=81, area_ratio=1.1
            ),
        }
        baseline = {"epoch": 7, "mIoU": 0.7, "Pd": 0.9, "Fa": 0.01}
        both = gate.build_result_payload(
            **common,
            baseline_selected=baseline,
            variant_selected={"epoch": 7, "mIoU": 0.702, "Pd": 0.896, "Fa": 0.011},
        )
        pd_only = gate.build_result_payload(
            **common,
            baseline_selected=baseline,
            variant_selected={"epoch": 7, "mIoU": 0.702, "Pd": 0.896, "Fa": 0.009},
        )
        boundary = gate.build_result_payload(
            **common,
            baseline_selected=baseline,
            variant_selected={"epoch": 7, "mIoU": 0.702, "Pd": 0.897, "Fa": 0.011},
        )
        self.assertTrue(both["safety_gate"]["failed"])
        self.assertFalse(pd_only["safety_gate"]["failed"])
        self.assertFalse(boundary["safety_gate"]["failed"])


class FailClosedInputTest(unittest.TestCase):
    def _run(self, repository: Path) -> None:
        with mock.patch.object(gate, "PROJECT_ROOT", repository):
            gate.run()

    def test_tampered_selected_metric_is_rejected_by_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            fixture = GateFixture(repository)
            fixture.variant_summary["validation_history"][6]["mIoU"] = 0.99
            _write_json(
                repository / gate.VARIANT_SUMMARY_RELATIVE_PATH,
                fixture.variant_summary,
            )
            with self.assertRaisesRegex(gate.PromotionGateError, "selection"):
                self._run(repository)

    def test_checkpoint_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            path = repository / gate.VARIANT_CHECKPOINT_RELATIVE_PATH
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(gate.PromotionGateError, "SHA-256"):
                self._run(repository)

    def test_source_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            path = repository / "experiments/evisirst_v2_selection.py"
            path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(gate.PromotionGateError, "source"):
                self._run(repository)

    def test_tbd_selected_metric_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            fixture = GateFixture(repository)
            fixture.variant_summary["validation_history"][6]["mIoU"] = "TBD"
            _write_json(
                repository / gate.VARIANT_SUMMARY_RELATIVE_PATH,
                fixture.variant_summary,
            )
            with self.assertRaisesRegex(gate.PromotionGateError, "numeric scalar"):
                self._run(repository)

    def test_missing_and_incomplete_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            (repository / gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH).unlink()
            with self.assertRaises(FileNotFoundError):
                self._run(repository)
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            fixture = GateFixture(repository)
            fixture.baseline_summary["training_history"].pop()
            _write_json(
                repository / gate.BASELINE_SUMMARY_RELATIVE_PATH,
                fixture.baseline_summary,
            )
            with self.assertRaisesRegex(gate.PromotionGateError, "not complete"):
                self._run(repository)

    def test_nan_and_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            path = repository / gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH
            path.write_text('{"schema":"x","metric":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(gate.PromotionGateError, "strict finite"):
                self._run(repository)
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            GateFixture(repository)
            path = repository / gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH
            path.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
            with self.assertRaisesRegex(gate.PromotionGateError, "strict finite"):
                self._run(repository)

    def test_cli_imports_no_public_test_dataset_or_evaluator(self) -> None:
        source = Path(gate.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules: list[str] = []
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
