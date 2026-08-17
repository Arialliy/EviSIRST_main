from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import run_irstd_paired_baseline_confirmatory_diagnostic as confirmatory


def _pilot_gate_payload(*, passed: bool = True) -> dict[str, object]:
    result = "PASS" if passed else "FAIL"
    status = "allowed" if passed else "blocked"
    return {
        "schema": confirmatory.canonical_gate.RESULT_SCHEMA,
        "status": "complete",
        "run_seed": confirmatory.PILOT_RUN_SEED,
        "public_test_allowed": False,
        "test_split_accessed": False,
        "primary_gate": {"passed": passed},
        "safety_gate": {"passed": passed},
        "mechanism_gate": {"passed": passed},
        "decision": {
            "result": result,
            "overall_passed": passed,
            "three_runtime_seed_validation_expansion_allowed": passed,
            "three_runtime_seed_validation_expansion_status": status,
            "public_test_allowed": False,
        },
    }


def _pilot_authority() -> dict[str, object]:
    return {
        "schema": confirmatory.PILOT_AUTHORITY_SCHEMA,
        "validator": (
            "run_irstd_complete_target_promotion_gate."
            "validate_existing_result"
        ),
        "result_relative_path": (
            confirmatory.canonical_gate.OUTPUT_RELATIVE_PATH
        ),
        "result_sha256": "a" * 64,
        "result_schema": confirmatory.canonical_gate.RESULT_SCHEMA,
        "pilot_run_seed": confirmatory.PILOT_RUN_SEED,
        "result": "PASS",
        "overall_passed": True,
        "three_runtime_seed_validation_expansion_allowed": True,
        "public_test_allowed": False,
        "test_split_accessed": False,
    }


def _source_set(schema: str, fill: str) -> dict[str, object]:
    files = {
        "source": {
            "relative_path": "source.py",
            "sha256": fill * 64,
        }
    }
    return {
        "schema": schema,
        "files": files,
        "source_tree_sha256": hashlib.sha256(
            confirmatory._canonical_json_bytes(files)
        ).hexdigest(),
    }


def _core_payload(
    seed: int, core_sources: dict[str, object]
) -> dict[str, object]:
    paths = confirmatory._relative_paths(seed)
    return {
        "schema": confirmatory.diagnostic_core.DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "dataset": confirmatory.DATASET,
        "data_role": "val",
        "diagnostic_only": True,
        "selection_allowed": False,
        "selected_on_same_validation_split": True,
        "checkpoint": {
            "relative_path": paths["CHECKPOINT_RELATIVE_PATH"],
            "sha256": "b" * 64,
            "run_seed": seed,
        },
        "completed_run_summary": {
            "relative_path": paths["SUMMARY_RELATIVE_PATH"],
            "sha256": "c" * 64,
        },
        "split": {
            "manifest_sha256": "d" * 64,
            "data_tree_sha256": "e" * 64,
        },
        "sources": {
            "checkpoint_training_source_tree_sha256": "f" * 64,
            "checkpoint_training_sources_currently_verified": True,
            "diagnostic": core_sources,
        },
        "evaluation": {
            "prediction_sha256": "1" * 64,
            "target_sha256": "2" * 64,
        },
        "metrics": {
            "complete_target_mechanism_diagnostics": {
                "schema": "evisirst_matched_target_diagnostics/v1/aggregate"
            }
        },
        "test_split_accessed": False,
    }


def _selected_metrics() -> dict[str, object]:
    return {
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


def _mechanism() -> dict[str, object]:
    return {
        "schema": "evisirst_matched_target_diagnostics/v1/aggregate",
        "prediction_threshold_rule": "probability>0.5",
        "target_threshold_rule": "target>0.5",
        "component_connectivity": 2,
        "component_neighborhood": "8-connected",
        "assignment_algorithm": (
            "Hungarian/scipy.optimize.linear_sum_assignment"
        ),
        "match_rule": "centroid_distance<3",
        "image_count": 160,
        "target_component_count": 10,
        "predicted_component_count": 10,
        "matched_component_count": 9,
        "matched_overlap_pixel_count": 80,
        "matched_target_pixel_count": 100,
        "matched_target_pixel_recall": 0.8,
        "matched_component_area_ratio": 1.1,
        "centroid_error": 0.5,
    }


def _strict_core_payload(
    seed: int, core_sources: dict[str, object]
) -> dict[str, object]:
    paths = confirmatory._relative_paths(seed)
    metrics = _selected_metrics()
    metrics["complete_target_mechanism_diagnostics"] = _mechanism()
    return {
        "schema": confirmatory.diagnostic_core.DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "dataset": confirmatory.DATASET,
        "data_role": "val",
        "diagnostic_only": True,
        "selection_allowed": False,
        "selected_on_same_validation_split": True,
        "checkpoint": {
            "relative_path": paths["CHECKPOINT_RELATIVE_PATH"],
            "sha256": "b" * 64,
            "schema": confirmatory.diagnostic_core.r1.CHECKPOINT_SCHEMA,
            "checkpoint_role": "validation_selected",
            "selected_epoch": 7,
            "architecture_seed": 42,
            "run_seed": seed,
            "target_mode": "binary",
            "training_identity_sha256": "6" * 64,
            "selection_provenance_sha256": "7" * 64,
            "strict_564_key_load": True,
        },
        "completed_run_summary": {
            "relative_path": paths["SUMMARY_RELATIVE_PATH"],
            "sha256": "c" * 64,
            "selected_validation_record_sha256": "8" * 64,
        },
        "split": {
            "manifest_relative_path": "splits/v2/IRSTD-1K/manifest.json",
            "manifest_sha256": "d" * 64,
            "val_index_relative_path": "splits/v2/IRSTD-1K/val.txt",
            "val_index_sha256": "9" * 64,
            "data_tree_sha256": "e" * 64,
            "data_tree_verified": True,
            "train_count": 640,
            "val_count": 160,
        },
        "sources": {
            "checkpoint_training_source_tree_sha256": "f" * 64,
            "checkpoint_training_sources_currently_verified": True,
            "diagnostic": core_sources,
        },
        "evaluation": {
            "metrics_contract": (
                confirmatory.diagnostic_core.r1.EVALUATION_PROTOCOL_VERSION
            ),
            "evaluation_head": "out",
            "probability_threshold": 0.5,
            "probability_threshold_operator": ">",
            "target_threshold": 0.5,
            "target_threshold_operator": ">",
            "match_radius": 3.0,
            "match_distance_operator": "<",
            "connected_component_connectivity": 2,
            "connected_component_neighborhood": "8-connected",
            "assignment_algorithm": (
                "Hungarian/scipy.optimize.linear_sum_assignment"
            ),
            "tiny_area": 9,
            "tiny_area_operator": "<=",
            "sample_count": 160,
            "one_model_forward_per_sample": True,
            "prediction_digest_schema": (
                confirmatory.diagnostic_core.PREDICTION_DIGEST_SCHEMA
            ),
            "prediction_sha256": "1" * 64,
            "target_digest_schema": (
                confirmatory.diagnostic_core.TARGET_DIGEST_SCHEMA
            ),
            "target_sha256": "2" * 64,
            "prediction_or_target_arrays_written": False,
        },
        "metrics": metrics,
        "test_split_accessed": False,
    }


@contextmanager
def _temporary_repository(repository: Path):
    with (
        mock.patch.object(confirmatory, "PROJECT_ROOT", repository),
        mock.patch.object(
            confirmatory.diagnostic_core, "PROJECT_ROOT", repository
        ),
        mock.patch.object(
            confirmatory.canonical_gate, "PROJECT_ROOT", repository
        ),
    ):
        yield


class FixedCliAndPathContractTest(unittest.TestCase):
    def test_cli_exposes_only_two_seeds_and_fixed_execution_surface(self) -> None:
        for seed in confirmatory.CONFIRMATORY_RUN_SEEDS:
            args = confirmatory.parse_args(
                ["--run-seed", str(seed), "--dataset-root", "/data"]
            )
            self.assertEqual(args.run_seed, seed)
            self.assertEqual(args.device, "cuda:0")
            self.assertEqual(args.workers, 0)
            self.assertEqual(
                set(vars(args)),
                {"run_seed", "dataset_root", "device", "workers"},
            )
        for rejected in (
            confirmatory.PILOT_RUN_SEED,
            42,
            -1,
        ):
            with self.assertRaises(SystemExit):
                confirmatory.parse_args(
                    ["--run-seed", str(rejected), "--dataset-root", "/data"]
                )
        with self.assertRaises(SystemExit):
            confirmatory.parse_args(
                [
                    "--run-seed",
                    str(confirmatory.CONFIRMATORY_RUN_SEEDS[0]),
                    "--dataset-root",
                    "/data",
                    "--device",
                    "cpu",
                ]
            )
        with self.assertRaises(SystemExit):
            confirmatory.parse_args(
                [
                    "--run-seed",
                    str(confirmatory.CONFIRMATORY_RUN_SEEDS[0]),
                    "--dataset-root",
                    "/data",
                    "--workers",
                    "1",
                ]
            )

    def test_each_seed_has_fixed_disjoint_input_output_and_lock_paths(self) -> None:
        first = confirmatory._relative_paths(
            confirmatory.CONFIRMATORY_RUN_SEEDS[0]
        )
        second = confirmatory._relative_paths(
            confirmatory.CONFIRMATORY_RUN_SEEDS[1]
        )
        self.assertEqual(set(first), set(second))
        self.assertFalse(set(first.values()).intersection(second.values()))
        for seed, paths in zip(confirmatory.CONFIRMATORY_RUN_SEEDS, (first, second)):
            self.assertIn(f"run_seed_{seed}", paths["CHECKPOINT_RELATIVE_PATH"])
            self.assertIn(f"run_seed_{seed}", paths["SUMMARY_RELATIVE_PATH"])
            self.assertIn(f"run_seed_{seed}", paths["OUTPUT_RELATIVE_PATH"])
            self.assertIn(f"run_seed_{seed}", paths["RUN_LOCK_RELATIVE_PATH"])
            self.assertTrue(
                paths["OUTPUT_RELATIVE_PATH"].endswith(
                    "/matched_target_diagnostics.json"
                )
            )

    def test_binding_is_exception_safe_and_restores_single_seed_core(self) -> None:
        before = {
            name: getattr(confirmatory.diagnostic_core, name)
            for name in confirmatory._CORE_PATCH_FIELDS
        }
        seed = confirmatory.CONFIRMATORY_RUN_SEEDS[0]
        with confirmatory._bound_diagnostic_core(seed) as paths:
            self.assertEqual(confirmatory.diagnostic_core.RUN_SEED, seed)
            for name, value in paths.items():
                self.assertEqual(getattr(confirmatory.diagnostic_core, name), value)
        after = {
            name: getattr(confirmatory.diagnostic_core, name)
            for name in confirmatory._CORE_PATCH_FIELDS
        }
        self.assertEqual(after, before)

    def test_cli_imports_no_public_test_data_or_evaluator(self) -> None:
        source = Path(confirmatory.__file__).read_text(encoding="utf-8")
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


class PilotAuthorityTest(unittest.TestCase):
    def test_fresh_canonical_pass_is_bound_to_exact_artifact(self) -> None:
        payload = _pilot_gate_payload()
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            result = (
                repository
                / confirmatory.canonical_gate.OUTPUT_RELATIVE_PATH
            )
            result.parent.mkdir(parents=True)
            result.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
            with (
                _temporary_repository(repository),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "validate_existing_result",
                    return_value=payload,
                ) as validator,
            ):
                authority = confirmatory.validate_pilot_authority()
        validator.assert_called_once_with()
        self.assertEqual(authority["result"], "PASS")
        self.assertTrue(
            authority["three_runtime_seed_validation_expansion_allowed"]
        )
        self.assertFalse(authority["public_test_allowed"])
        self.assertEqual(
            authority["result_sha256"],
            hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )

    def test_fail_or_type_confused_gate_never_authorizes_expansion(self) -> None:
        failed = _pilot_gate_payload(passed=False)
        with self.assertRaises(confirmatory.ConfirmatoryDiagnosticError):
            confirmatory._require_exact_pass_authority(failed)
        confused = _pilot_gate_payload()
        confused["decision"][
            "three_runtime_seed_validation_expansion_allowed"
        ] = 1
        with self.assertRaises(confirmatory.ConfirmatoryDiagnosticError):
            confirmatory._require_exact_pass_authority(confused)

    def test_pilot_failure_occurs_before_lock_or_cuda_path(self) -> None:
        args = SimpleNamespace(
            run_seed=confirmatory.CONFIRMATORY_RUN_SEEDS[0],
            dataset_root=Path("/data"),
            device="cuda:0",
            workers=0,
        )
        with (
            mock.patch.object(
                confirmatory,
                "validate_pilot_authority",
                side_effect=confirmatory.ConfirmatoryDiagnosticError("blocked"),
            ),
            mock.patch.object(
                confirmatory, "_bound_diagnostic_core"
            ) as binder,
            mock.patch.object(
                confirmatory.diagnostic_core,
                "configure_inference_determinism",
            ) as configure,
        ):
            with self.assertRaises(confirmatory.ConfirmatoryDiagnosticError):
                confirmatory.run(args)
        binder.assert_not_called()
        configure.assert_not_called()


class PayloadAndSourceContractTest(unittest.TestCase):
    def test_source_identity_binds_new_cli_old_core_and_pilot_gate(self) -> None:
        provenance = confirmatory.confirmatory_source_provenance()
        files = provenance["files"]
        self.assertEqual(
            set(files),
            {
                "confirmatory_diagnostic_cli",
                "single_seed_diagnostic_core",
                "canonical_promotion_gate",
            },
        )
        for record in files.values():
            path = Path(confirmatory.__file__).resolve().parent / record[
                "relative_path"
            ]
            self.assertEqual(record["sha256"], confirmatory._sha256_file(path))

    def test_payload_is_validation_only_and_identity_bound(self) -> None:
        seed = confirmatory.CONFIRMATORY_RUN_SEEDS[1]
        core_sources = _source_set("core/v1", "3")
        confirmatory_sources = _source_set(
            confirmatory.SOURCE_SET_SCHEMA, "4"
        )
        payload = confirmatory.build_result_payload(
            run_seed=seed,
            paths=confirmatory._relative_paths(seed),
            core_payload=_core_payload(seed, core_sources),
            core_sources=core_sources,
            confirmatory_sources=confirmatory_sources,
            pilot_authority=_pilot_authority(),
        )
        self.assertEqual(payload["schema"], confirmatory.DIAGNOSTIC_SCHEMA)
        self.assertEqual(payload["run_seed"], seed)
        self.assertTrue(payload["diagnostic_only"])
        self.assertFalse(payload["selection_allowed"])
        self.assertFalse(payload["test_split_accessed"])
        self.assertEqual(
            payload["sources"]["confirmatory_diagnostic"],
            confirmatory_sources,
        )
        identity = payload["diagnostic_identity"]
        observed_sha = identity.pop("identity_sha256")
        self.assertEqual(
            observed_sha,
            hashlib.sha256(
                confirmatory._canonical_json_bytes(identity)
            ).hexdigest(),
        )
        self.assertIn(
            f"run_seed_{seed}", identity["output_relative_path"]
        )
        self.assertNotIn("/home/", json.dumps(payload, sort_keys=True))

    def test_pilot_authority_tamper_is_rejected(self) -> None:
        seed = confirmatory.CONFIRMATORY_RUN_SEEDS[0]
        core_sources = _source_set("core/v1", "5")
        confirmatory_sources = _source_set(
            confirmatory.SOURCE_SET_SCHEMA, "6"
        )
        for mutate in ("result_sha256", "validator", "extra"):
            authority = _pilot_authority()
            if mutate == "result_sha256":
                authority[mutate] = "not-a-digest"
            elif mutate == "validator":
                authority[mutate] = "untrusted.validator"
            else:
                authority[mutate] = True
            with self.assertRaises(confirmatory.ConfirmatoryDiagnosticError):
                confirmatory.build_result_payload(
                    run_seed=seed,
                    paths=confirmatory._relative_paths(seed),
                    core_payload=_core_payload(seed, core_sources),
                    core_sources=core_sources,
                    confirmatory_sources=confirmatory_sources,
                    pilot_authority=authority,
                )


class LockPublishAndExecutionTest(unittest.TestCase):
    def test_per_seed_atomic_outputs_are_disjoint_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "runs").mkdir()
            with _temporary_repository(repository):
                outputs = []
                for seed in confirmatory.CONFIRMATORY_RUN_SEEDS:
                    with confirmatory._bound_diagnostic_core(seed):
                        path = confirmatory.diagnostic_core._write_fixed_json_atomic(
                            {"run_seed": seed, "test_split_accessed": False}
                        )
                        outputs.append(path)
                        with self.assertRaises(FileExistsError):
                            confirmatory.diagnostic_core._write_fixed_json_atomic(
                                {"run_seed": seed}
                            )
                self.assertNotEqual(outputs[0], outputs[1])
                self.assertEqual(
                    [
                        json.loads(path.read_text())["run_seed"]
                        for path in outputs
                    ],
                    list(confirmatory.CONFIRMATORY_RUN_SEEDS),
                )

    def test_full_locked_path_calls_one_core_evaluation_and_publishes(self) -> None:
        seed = confirmatory.CONFIRMATORY_RUN_SEEDS[0]
        pilot = _pilot_authority()
        core_sources = _source_set("core/v1", "7")
        confirmatory_sources = _source_set(
            confirmatory.SOURCE_SET_SCHEMA, "8"
        )
        args = SimpleNamespace(
            run_seed=seed,
            dataset_root=Path("/data"),
            device="cuda:0",
            workers=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "runs").mkdir()
            paths = confirmatory._relative_paths(seed)
            checkpoint = repository / paths["CHECKPOINT_RELATIVE_PATH"]
            summary = repository / paths["SUMMARY_RELATIVE_PATH"]
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            summary.write_bytes(b"summary")
            checkpoint_metadata = {
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "summary": {
                    "sha256": hashlib.sha256(b"summary").hexdigest()
                },
                "training_source_tree_sha256": "9" * 64,
            }
            core_payload = _core_payload(seed, core_sources)
            with (
                _temporary_repository(repository),
                confirmatory._bound_diagnostic_core(seed) as bound_paths,
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "validate_optional_inherited_gpu_lock",
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "configure_inference_determinism",
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "require_device",
                    return_value=torch.device("cpu"),
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "diagnostic_source_provenance",
                    return_value=core_sources,
                ),
                mock.patch.object(
                    confirmatory,
                    "confirmatory_source_provenance",
                    return_value=confirmatory_sources,
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "load_fixed_final_model",
                    return_value=(object(), checkpoint_metadata),
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "build_validation_dataset",
                    return_value=[object()],
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "evaluate_out_once",
                    return_value=(
                        {"miou": 0.5},
                        {"matched_target_pixel_recall": 0.8},
                        {
                            "prediction_sha256": "1" * 64,
                            "target_sha256": "2" * 64,
                        },
                        1,
                    ),
                ) as evaluate,
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "build_result_payload",
                    return_value=core_payload,
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core.r1,
                    "_protocol_source_provenance",
                    return_value={"source_tree_sha256": "9" * 64},
                ),
                mock.patch.object(
                    confirmatory,
                    "validate_pilot_authority",
                    return_value=pilot,
                ),
            ):
                output = confirmatory._run_locked(
                    args,
                    paths=bound_paths,
                    pilot_authority=pilot,
                )
            published = json.loads(output.read_text(encoding="utf-8"))
        evaluate.assert_called_once()
        self.assertEqual(published["run_seed"], seed)
        self.assertTrue(published["diagnostic_only"])
        self.assertFalse(published["selection_allowed"])
        self.assertFalse(published["test_split_accessed"])


class ExistingResultValidationTest(unittest.TestCase):
    def _fixture(self):
        seed = confirmatory.CONFIRMATORY_RUN_SEEDS[0]
        paths = confirmatory._relative_paths(seed)
        core_sources = _source_set(
            confirmatory.diagnostic_core.DIAGNOSTIC_SOURCE_SCHEMA, "a"
        )
        confirmatory_sources = _source_set(
            confirmatory.SOURCE_SET_SCHEMA, "b"
        )
        pilot = _pilot_authority()
        core = _strict_core_payload(seed, core_sources)
        selected = {
            "epoch": 7,
            "data_role": "val",
            "mIoU": 0.5,
            "Fa": 0.001,
            "Pd": 0.9,
            "evaluation_head": "out",
            "metrics": _selected_metrics(),
        }
        selection = {"fixture": True}
        core["checkpoint"]["selection_provenance_sha256"] = (
            confirmatory.canonical_gate._canonical_sha256(selection)
        )
        core["completed_run_summary"][
            "selected_validation_record_sha256"
        ] = confirmatory.canonical_gate._canonical_sha256(selected)
        observed = confirmatory.build_result_payload(
            run_seed=seed,
            paths=paths,
            core_payload=core,
            core_sources=core_sources,
            confirmatory_sources=confirmatory_sources,
            pilot_authority=pilot,
        )
        identity = {
            "identity_sha256": "6" * 64,
            "determinism_protocol": {"fixture": True},
        }
        metadata = {
            "diagnostic": {
                "relative_path": paths["OUTPUT_RELATIVE_PATH"],
                "sha256": "5" * 64,
            },
            "summary": {
                "relative_path": paths["SUMMARY_RELATIVE_PATH"],
                "sha256": "c" * 64,
            },
            "checkpoint": {
                "relative_path": paths["CHECKPOINT_RELATIVE_PATH"],
                "sha256": "b" * 64,
            },
        }
        return (
            seed,
            paths,
            core_sources,
            confirmatory_sources,
            pilot,
            observed,
            selected,
            identity,
            metadata,
        )

    def _validate(self, observed: dict[str, object]):
        (
            seed,
            paths,
            core_sources,
            confirmatory_sources,
            pilot,
            _original,
            selected,
            identity,
            metadata,
        ) = self._fixture()
        summary = {"selected_epoch": 7}
        checkpoint = {"fixture": True}

        def load_json(relative_path: str, *, label: str):
            del label
            if relative_path == paths["OUTPUT_RELATIVE_PATH"]:
                return observed, metadata["diagnostic"]
            if relative_path == paths["SUMMARY_RELATIVE_PATH"]:
                return summary, metadata["summary"]
            raise AssertionError(relative_path)

        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "runs").mkdir()
            with (
                _temporary_repository(repository),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_validate_canonical_split_files",
                ),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_load_json",
                    side_effect=load_json,
                ),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_load_checkpoint",
                    return_value=(checkpoint, metadata["checkpoint"]),
                ),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_validate_summary_common",
                    return_value=({"fixture": True}, selected),
                ),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_validate_checkpoint",
                    return_value=(identity, "f" * 64),
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "load_fixed_final_model",
                    return_value=(
                        object(),
                        {
                            "sha256": "b" * 64,
                            "selected_epoch": 7,
                            "training_identity_sha256": "6" * 64,
                        },
                    ),
                ),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_validate_diagnostic",
                    return_value=_mechanism(),
                ),
                mock.patch.object(
                    confirmatory.diagnostic_core,
                    "diagnostic_source_provenance",
                    return_value=core_sources,
                ),
                mock.patch.object(
                    confirmatory,
                    "confirmatory_source_provenance",
                    return_value=confirmatory_sources,
                ),
                mock.patch.object(
                    confirmatory,
                    "validate_pilot_authority",
                    return_value=pilot,
                ),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_assert_inputs_unchanged",
                ),
                mock.patch.object(
                    confirmatory.canonical_gate,
                    "_validate_source_set",
                    return_value="f" * 64,
                ),
            ):
                return confirmatory.validate_existing_result(seed)

    def test_existing_result_is_strictly_reconstructed_without_forward(self) -> None:
        observed = self._fixture()[5]
        with (
            mock.patch.object(
                confirmatory.diagnostic_core, "evaluate_out_once"
            ) as evaluate,
            mock.patch.object(
                confirmatory.torch.cuda,
                "is_available",
                side_effect=AssertionError("validator must remain CPU-only"),
            ) as cuda_probe,
        ):
            validated = self._validate(observed)
        evaluate.assert_not_called()
        cuda_probe.assert_not_called()
        self.assertEqual(validated, observed)
        self.assertFalse(validated["test_split_accessed"])

    def test_metric_tamper_extra_field_nonfinite_and_path_tamper_are_rejected(
        self,
    ) -> None:
        mutations = {}
        metric = json.loads(json.dumps(self._fixture()[5]))
        metric["metrics"]["niou"] = 0.9
        mutations["metric"] = metric
        extra = json.loads(json.dumps(self._fixture()[5]))
        extra["unexpected"] = True
        mutations["extra"] = extra
        nested_extra = json.loads(json.dumps(self._fixture()[5]))
        nested_extra["evaluation"]["unexpected"] = True
        mutations["nested_extra"] = nested_extra
        nonfinite = json.loads(json.dumps(self._fixture()[5]))
        nonfinite["metrics"]["niou"] = float("inf")
        mutations["nonfinite"] = nonfinite
        path = json.loads(json.dumps(self._fixture()[5]))
        path["checkpoint"]["relative_path"] = "runs/redirected/model.pth"
        mutations["path"] = path

        for label, observed in mutations.items():
            with self.subTest(label=label):
                with self.assertRaises(
                    confirmatory.ConfirmatoryDiagnosticError
                ):
                    self._validate(observed)

    def test_artifact_selection_source_authority_and_identity_tamper_rejected(
        self,
    ) -> None:
        mutations = {}
        checkpoint = json.loads(json.dumps(self._fixture()[5]))
        checkpoint["checkpoint"]["sha256"] = "0" * 64
        mutations["checkpoint_hash"] = checkpoint
        summary = json.loads(json.dumps(self._fixture()[5]))
        summary["completed_run_summary"]["sha256"] = "0" * 64
        mutations["summary_hash"] = summary
        selection = json.loads(json.dumps(self._fixture()[5]))
        selection["checkpoint"]["selection_provenance_sha256"] = "0" * 64
        mutations["selection"] = selection
        source = json.loads(json.dumps(self._fixture()[5]))
        source["sources"]["diagnostic"]["source_tree_sha256"] = "0" * 64
        mutations["source"] = source
        authority = json.loads(json.dumps(self._fixture()[5]))
        authority["promotion_prerequisite"]["result_sha256"] = "0" * 64
        mutations["authority"] = authority
        identity = json.loads(json.dumps(self._fixture()[5]))
        identity["diagnostic_identity"]["identity_sha256"] = "0" * 64
        mutations["identity"] = identity

        for label, observed in mutations.items():
            with self.subTest(label=label):
                with self.assertRaises(
                    confirmatory.ConfirmatoryDiagnosticError
                ):
                    self._validate(observed)

    def test_strict_loader_rejects_duplicate_key_nan_and_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "runs").mkdir()
            relative = "runs/fixture.json"
            output = repository / relative
            for content in (
                '{"schema":"x","schema":"y"}\n',
                '{"value":NaN}\n',
                '{"value":1e999}\n',
            ):
                output.write_text(content, encoding="utf-8")
                with mock.patch.object(
                    confirmatory.canonical_gate,
                    "PROJECT_ROOT",
                    repository,
                ):
                    with self.assertRaises(
                        confirmatory.canonical_gate.PromotionGateError
                    ):
                        confirmatory.canonical_gate._load_json(
                            relative, label="fixture"
                        )


if __name__ == "__main__":
    unittest.main()
