from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import test as public_test
from experiments import evisirst_v2_selection as selection


def audit_fields() -> dict[str, object]:
    provenance = selection.select_independent_checkpoint(
        [
            {
                "epoch": 17,
                "data_role": "val",
                "mIoU": 0.8,
                "Fa": 0.01,
                "Pd": 0.9,
            }
        ]
    )
    return {
        "source_selection": public_test._VALIDATION_SELECTED_SOURCE,
        "selection_is_optimistic": False,
        "selection_provenance": provenance,
        "run_seed": 104728269,
        "split_manifest_sha256": "A" * 64,
        "target_mode": "binary",
        "smoke": False,
        "training_identity_sha256": "D" * 64,
    }


def _canonical_identity_sha256(identity: dict[str, object]) -> str:
    canonical = dict(identity)
    canonical.pop("identity_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_identity() -> dict[str, object]:
    identity: dict[str, object] = {
        "schema": "evisirst_validation_selected_training/v1/run_identity",
        "model": "EviSIRST",
        "dataset": "NUAA-SIRST",
        "architecture_seed": 42,
        "run_seed": 104728269,
        "target_mode": "binary",
        "epochs": 1000,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1e-3,
        "min_lr": 1e-5,
        "warmup_epochs": 10,
        "val_interval": 1,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "manifest_sha256": "a" * 64,
        "selection_rule": public_test._VALIDATION_SELECTION_RULE,
        "determinism_protocol": {
            "schema": public_test._VALIDATION_DETERMINISM_SCHEMA,
            "selection": {
                "rule_version": public_test._VALIDATION_SELECTION_RULE,
                "miou_candidate_tolerance": (
                    public_test._VALIDATION_SELECTION_TOLERANCE
                ),
            },
            "validation_evaluation": {
                "version": "evisirst_validation_metrics/v1",
                "evaluation_head": "out",
                "prediction_threshold": 0.5,
                "prediction_threshold_operator": ">",
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
            },
        },
        "split_seed": 20260811,
        "data_tree_sha256": "b" * 64,
        "train_count": 170,
        "val_count": 43,
        "smoke": False,
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = _canonical_identity_sha256(identity)
    return identity


def validation_selected_identity() -> dict[str, object]:
    training = training_identity()
    fields = audit_fields()
    # The public loader already promises safe uppercase SHA normalization.
    fields["training_identity_sha256"] = str(
        training["identity_sha256"]
    ).upper()
    return {
        "checkpoint_role": "validation_selected",
        "dataset": "NUAA-SIRST",
        "epoch": 17,
        "seed": 42,
        "architecture_seed": 42,
        "test_split_accessed": False,
        "split_seed": 20260811,
        "data_tree_sha256": "b" * 64,
        "data_tree_verified": True,
        "training": training,
        "split_provenance": {
            "schema": public_test._VALIDATION_SPLIT_SCHEMA,
            "manifest_relative_path": "splits/v2/NUAA-SIRST/manifest.json",
            "manifest_sha256": "A" * 64,
            "split_seed": 20260811,
            "data_tree_sha256": "b" * 64,
            "data_tree_verified": True,
            "train_count": 170,
            "val_count": 43,
            "source_index": {
                "split": "train",
                "relative_path": (
                    "NUAA-SIRST/img_idx/train_NUAA-SIRST.txt"
                ),
                "sample_count": 213,
                "file_sha256": "c" * 64,
                "ordered_ids_sha256": "d" * 64,
            },
            "outputs": {
                "train": {
                    "relative_path": "splits/v2/NUAA-SIRST/train.txt",
                    "sample_count": 170,
                    "file_sha256": "e" * 64,
                    "ordered_ids_sha256": "f" * 64,
                },
                "val": {
                    "relative_path": "splits/v2/NUAA-SIRST/val.txt",
                    "sample_count": 43,
                    "file_sha256": "1" * 64,
                    "ordered_ids_sha256": "2" * 64,
                },
            },
            "test_index_opened": False,
        },
        **fields,
    }


def _rebind_training_hash(payload: dict[str, object]) -> None:
    training = payload["training"]
    if not isinstance(training, dict):
        raise AssertionError("test fixture training identity must be a dict")
    digest = _canonical_identity_sha256(training)
    training["identity_sha256"] = digest
    payload["training_identity_sha256"] = digest


def clean_checkpoint_payload(*, include_audit: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "evisirst_clean_checkpoint/v1",
        "model": "EviSIRST",
        "dataset": "NUAA-SIRST",
        "checkpoint_role": "final" if not include_audit else "validation_selected",
        "epoch": 17,
        "seed": 42,
        "state_dict": {
            f"fixture.tensor_{index:03d}": torch.zeros((), dtype=torch.float32)
            for index in range(564)
        },
    }
    if include_audit:
        payload.update(validation_selected_identity())
    return payload


class PublicTestCheckpointAuditTest(unittest.TestCase):
    def test_validation_selected_audit_fields_are_normalized_and_json_safe(
        self,
    ) -> None:
        normalized = public_test._checkpoint_audit_metadata(audit_fields())

        self.assertEqual(
            normalized["source_selection"],
            public_test._VALIDATION_SELECTED_SOURCE,
        )
        self.assertFalse(normalized["selection_is_optimistic"])
        self.assertEqual(normalized["run_seed"], 104728269)
        self.assertEqual(normalized["split_manifest_sha256"], "a" * 64)
        self.assertEqual(normalized["target_mode"], "binary")
        self.assertFalse(normalized["smoke"])
        self.assertEqual(normalized["training_identity_sha256"], "d" * 64)
        self.assertEqual(
            normalized["selection_provenance"]["selected"]["epoch"], 17
        )
        json.dumps(normalized, allow_nan=False, sort_keys=True)

    def test_thousand_epoch_flat_provenance_fits_the_bounded_audit(self) -> None:
        records = [
            {
                "epoch": epoch,
                "data_role": "val",
                "mIoU": 0.8,
                "Fa": 0.01,
                "Pd": 0.9,
            }
            for epoch in range(1, 1001)
        ]
        provenance = selection.select_independent_checkpoint(records)

        normalized = public_test._normalize_selection_provenance(provenance)
        self.assertEqual(normalized["selected"]["epoch"], 1)
        self.assertEqual(len(normalized["candidates"]), 1000)

        payload = validation_selected_identity()
        payload["epoch"] = 1
        payload["selection_provenance"] = provenance
        public_test._checkpoint_audit_metadata(payload)

    def test_provenance_beyond_node_limit_remains_rejected(self) -> None:
        oversized = {
            "values": [0] * public_test._MAX_AUDIT_JSON_NODES,
        }
        with self.assertRaisesRegex(ValueError, "too large"):
            public_test._normalize_selection_provenance(oversized)

    def test_clean_checkpoint_loader_preserves_existing_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "validation-selected.pth.tar"
            torch.save(clean_checkpoint_payload(include_audit=True), checkpoint)

            state, payload, observed_sha256 = public_test._state_from_checkpoint(
                checkpoint
            )

        self.assertEqual(len(state), 564)
        self.assertEqual(len(observed_sha256), 64)
        for field in public_test._CHECKPOINT_AUDIT_FIELDS:
            self.assertIn(field, payload)
        self.assertEqual(payload["split_manifest_sha256"], "a" * 64)

    def test_result_checkpoint_block_transmits_the_validated_source(self) -> None:
        metadata = {
            "checkpoint_path": "/fixture/validation-selected.pth.tar",
            "checkpoint_sha256": "b" * 64,
            **validation_selected_identity(),
        }

        result_checkpoint = public_test._checkpoint_result_metadata(metadata)

        self.assertEqual(
            result_checkpoint["path"],
            "<external>/validation-selected.pth.tar",
        )
        self.assertEqual(
            result_checkpoint["path_scope"], "external-path-redacted"
        )
        self.assertEqual(
            result_checkpoint["source_selection"], metadata["source_selection"]
        )
        self.assertEqual(
            result_checkpoint["selection_provenance"],
            public_test._checkpoint_audit_metadata(metadata)["selection_provenance"],
        )
        self.assertEqual(result_checkpoint["split_manifest_sha256"], "a" * 64)
        self.assertFalse(result_checkpoint["smoke"])
        self.assertEqual(
            result_checkpoint["training_identity_sha256"],
            training_identity()["identity_sha256"],
        )
        json.dumps(result_checkpoint, allow_nan=False, sort_keys=True)

    def test_load_model_bridges_clean_payload_audit_fields_into_metadata(self) -> None:
        class EmptyFixtureModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mode = "train"

        payload = validation_selected_identity()
        payload.update(public_test._checkpoint_audit_metadata(payload))
        fake_model = EmptyFixtureModel()
        with mock.patch.object(
            public_test,
            "_state_from_checkpoint",
            return_value=({}, payload, "b" * 64),
        ), mock.patch.object(
            public_test,
            "initialize_evisirst",
            return_value=(fake_model, {"model": "EviSIRST"}),
        ):
            model, metadata = public_test.load_model(
                "evisirst",
                "NUAA-SIRST",
                Path("/fixture/validation-selected.pth.tar"),
            )

        self.assertIs(model, fake_model)
        self.assertFalse(model.training)
        self.assertEqual(model.mode, "test")
        for field in public_test._CHECKPOINT_AUDIT_FIELDS:
            self.assertEqual(metadata[field], payload[field])
        self.assertEqual(
            public_test._checkpoint_result_metadata(metadata)["source_selection"],
            public_test._VALIDATION_SELECTED_SOURCE,
        )

    def test_malformed_or_hostile_audit_fields_are_rejected(self) -> None:
        hostile_values = {
            "source_selection": "validation\nselected",
            "selection_is_optimistic": 0,
            "selection_provenance": {"metric": float("nan")},
            "run_seed": True,
            "split_manifest_sha256": "../manifest.json",
            "target_mode": "auto",
            "smoke": 0,
            "training_identity_sha256": "not-a-hash",
        }
        for field, hostile in hostile_values.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    public_test._checkpoint_audit_metadata({field: hostile})

        with self.assertRaisesRegex(ValueError, "keys"):
            public_test._checkpoint_audit_metadata(
                {"selection_provenance": {1: "stringified-key-collision"}}
            )
        with self.assertRaisesRegex(ValueError, "non-JSON"):
            public_test._checkpoint_audit_metadata(
                {"selection_provenance": {"payload": object()}}
            )

    def test_validation_selected_role_requires_complete_coherent_provenance(
        self,
    ) -> None:
        complete = validation_selected_identity()
        self.assertEqual(
            set(public_test._checkpoint_audit_metadata(complete)),
            set(public_test._CHECKPOINT_AUDIT_FIELDS),
        )

        for missing_field in public_test._CHECKPOINT_AUDIT_FIELDS:
            with self.subTest(missing_field=missing_field):
                incomplete = dict(complete)
                del incomplete[missing_field]
                with self.assertRaisesRegex(ValueError, "missing audit fields"):
                    public_test._checkpoint_audit_metadata(incomplete)

        contradictory = []
        optimistic = dict(complete)
        optimistic["selection_is_optimistic"] = True
        contradictory.append(optimistic)
        wrong_role = dict(complete)
        wrong_role["selection_provenance"] = {
            **wrong_role["selection_provenance"],
            "data_role": "test",
        }
        contradictory.append(wrong_role)
        test_selection = dict(complete)
        test_selection["selection_provenance"] = {
            **test_selection["selection_provenance"],
            "test_selection_supported": True,
        }
        contradictory.append(test_selection)
        missing_test_disclosure = dict(complete)
        missing_test_disclosure["selection_provenance"] = dict(
            missing_test_disclosure["selection_provenance"]
        )
        del missing_test_disclosure["selection_provenance"][
            "test_selection_supported"
        ]
        contradictory.append(missing_test_disclosure)
        smoke_checkpoint = dict(complete)
        smoke_checkpoint["smoke"] = True
        contradictory.append(smoke_checkpoint)
        wrong_source = copy.deepcopy(complete)
        wrong_source["source_selection"] = "historical_test_best_miou"
        contradictory.append(wrong_source)
        for field, value in (
            ("schema", "unknown_selection_schema/v1"),
            ("rule_version", "historical_test_selector/v1"),
            ("selection_kind", "joint_checkpoint"),
            ("candidate_tolerance_raw", 0.01),
        ):
            wrong_contract = copy.deepcopy(complete)
            provenance = wrong_contract["selection_provenance"]
            self.assertIsInstance(provenance, dict)
            provenance[field] = value  # type: ignore[index]
            contradictory.append(wrong_contract)
        for payload in contradictory:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    public_test._checkpoint_audit_metadata(payload)

        wrong_training_rule = copy.deepcopy(complete)
        training = wrong_training_rule["training"]
        self.assertIsInstance(training, dict)
        training["selection_rule"] = "historical_test_selector/v1"
        _rebind_training_hash(wrong_training_rule)
        with self.assertRaisesRegex(ValueError, "selection_rule"):
            public_test._checkpoint_audit_metadata(wrong_training_rule)

        for field, value in (
            ("schema", "unknown_training/v1"),
            ("model", "OtherModel"),
        ):
            wrong_training = copy.deepcopy(complete)
            training = wrong_training["training"]
            self.assertIsInstance(training, dict)
            training[field] = value  # type: ignore[index]
            _rebind_training_hash(wrong_training)
            with self.assertRaisesRegex(ValueError, field):
                public_test._checkpoint_audit_metadata(wrong_training)

        oversized_seed = copy.deepcopy(complete)
        oversized_seed["run_seed"] = public_test._MAX_VALIDATION_RUN_SEED + 1
        training = oversized_seed["training"]
        self.assertIsInstance(training, dict)
        training["run_seed"] = oversized_seed["run_seed"]
        _rebind_training_hash(oversized_seed)
        with self.assertRaisesRegex(ValueError, "uint32"):
            public_test._checkpoint_audit_metadata(oversized_seed)

    def test_validation_selected_training_identity_is_recomputed(self) -> None:
        payload = validation_selected_identity()
        public_test._checkpoint_audit_metadata(payload)

        missing = copy.deepcopy(payload)
        del missing["training"]
        with self.assertRaisesRegex(ValueError, "missing identity fields"):
            public_test._checkpoint_audit_metadata(missing)

        non_mapping = copy.deepcopy(payload)
        non_mapping["training"] = []
        with self.assertRaisesRegex(ValueError, "training must be a mapping"):
            public_test._checkpoint_audit_metadata(non_mapping)

        inner_top_mismatch = copy.deepcopy(payload)
        inner_top_mismatch["training_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "differs from training_identity"):
            public_test._checkpoint_audit_metadata(inner_top_mismatch)

        canonical_tamper = copy.deepcopy(payload)
        canonical_training = canonical_tamper["training"]
        self.assertIsInstance(canonical_training, dict)
        canonical_training["epochs"] = 999  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "does not recompute"):
            public_test._checkpoint_audit_metadata(canonical_tamper)

    def test_validation_selected_training_fields_bind_to_top_level(self) -> None:
        tampered_values = {
            "dataset": "IRSTD-1k",
            "run_seed": 104728270,
            "target_mode": "soft",
            "manifest_sha256": "b" * 64,
            "smoke": True,
            "architecture_seed": 43,
            "test_split_accessed": True,
        }
        for field, value in tampered_values.items():
            with self.subTest(field=field):
                payload = copy.deepcopy(validation_selected_identity())
                training = payload["training"]
                self.assertIsInstance(training, dict)
                training[field] = value  # type: ignore[index]
                _rebind_training_hash(payload)
                with self.assertRaisesRegex(ValueError, field):
                    public_test._checkpoint_audit_metadata(payload)

        architecture_alias_tamper = validation_selected_identity()
        architecture_alias_tamper["architecture_seed"] = 43
        with self.assertRaisesRegex(ValueError, "architecture_seed"):
            public_test._checkpoint_audit_metadata(architecture_alias_tamper)

    def test_validation_selected_training_and_split_protocol_are_bound(self) -> None:
        schedule_cases = (
            ("epochs", 1),
            ("val_interval", 4),
            ("batch_size", 0),
            ("workers", -1),
        )
        for field, value in schedule_cases:
            with self.subTest(field=field):
                payload = copy.deepcopy(validation_selected_identity())
                training = payload["training"]
                self.assertIsInstance(training, dict)
                training[field] = value
                _rebind_training_hash(payload)
                with self.assertRaisesRegex(ValueError, "schedule"):
                    public_test._checkpoint_audit_metadata(payload)

        wrong_evaluator = copy.deepcopy(validation_selected_identity())
        training = wrong_evaluator["training"]
        self.assertIsInstance(training, dict)
        determinism = training["determinism_protocol"]
        self.assertIsInstance(determinism, dict)
        evaluation = determinism["validation_evaluation"]
        self.assertIsInstance(evaluation, dict)
        evaluation["prediction_threshold"] = 0.9
        _rebind_training_hash(wrong_evaluator)
        with self.assertRaisesRegex(ValueError, "prediction_threshold"):
            public_test._checkpoint_audit_metadata(wrong_evaluator)

        wrong_nested_rule = copy.deepcopy(validation_selected_identity())
        training = wrong_nested_rule["training"]
        self.assertIsInstance(training, dict)
        determinism = training["determinism_protocol"]
        self.assertIsInstance(determinism, dict)
        nested_selection = determinism["selection"]
        self.assertIsInstance(nested_selection, dict)
        nested_selection["rule_version"] = "historical_test/v1"
        _rebind_training_hash(wrong_nested_rule)
        with self.assertRaisesRegex(ValueError, "determinism selection"):
            public_test._checkpoint_audit_metadata(wrong_nested_rule)

        redirected_source = copy.deepcopy(validation_selected_identity())
        split = redirected_source["split_provenance"]
        self.assertIsInstance(split, dict)
        source_index = split["source_index"]
        self.assertIsInstance(source_index, dict)
        source_index["split"] = "test"
        source_index["relative_path"] = (
            "NUAA-SIRST/img_idx/test_NUAA-SIRST.txt"
        )
        with self.assertRaisesRegex(ValueError, "source/output roles"):
            public_test._checkpoint_audit_metadata(redirected_source)

        joint_dataset = copy.deepcopy(validation_selected_identity())
        joint_dataset["dataset"] = "SIRST3"
        training = joint_dataset["training"]
        self.assertIsInstance(training, dict)
        training["dataset"] = "SIRST3"
        _rebind_training_hash(joint_dataset)
        with self.assertRaisesRegex(ValueError, "V2 source dataset"):
            public_test._checkpoint_audit_metadata(joint_dataset)

        accessed = copy.deepcopy(validation_selected_identity())
        accessed["test_split_accessed"] = True
        training = accessed["training"]
        self.assertIsInstance(training, dict)
        training["test_split_accessed"] = True  # type: ignore[index]
        _rebind_training_hash(accessed)
        with self.assertRaisesRegex(ValueError, "test_split_accessed=false"):
            public_test._checkpoint_audit_metadata(accessed)

    def test_selection_epoch_and_optional_split_provenance_are_bound(self) -> None:
        valid = validation_selected_identity()
        public_test._checkpoint_audit_metadata(valid)

        selected_epoch_tamper = copy.deepcopy(valid)
        provenance = selected_epoch_tamper["selection_provenance"]
        self.assertIsInstance(provenance, dict)
        selected = provenance["selected"]  # type: ignore[index]
        self.assertIsInstance(selected, dict)
        selected["epoch"] = 18
        with self.assertRaisesRegex(ValueError, "does not exactly recompute"):
            public_test._checkpoint_audit_metadata(selected_epoch_tamper)

        missing_selected = copy.deepcopy(valid)
        provenance = missing_selected["selection_provenance"]
        self.assertIsInstance(provenance, dict)
        del provenance["selected"]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "does not exactly recompute"):
            public_test._checkpoint_audit_metadata(missing_selected)

        contradictory = copy.deepcopy(valid)
        records = [
            {
                "epoch": 1,
                "data_role": "val",
                "mIoU": 0.8,
                "Fa": 0.5,
                "Pd": 0.1,
            },
            {
                "epoch": 2,
                "data_role": "val",
                "mIoU": 0.8,
                "Fa": 0.1,
                "Pd": 0.9,
            },
        ]
        provenance = selection.select_independent_checkpoint(records)
        self.assertEqual(provenance["selected"]["epoch"], 2)
        provenance["selected"] = dict(provenance["candidates"][0])
        contradictory["selection_provenance"] = provenance
        contradictory["epoch"] = 1
        with self.assertRaisesRegex(ValueError, "does not exactly recompute"):
            public_test._checkpoint_audit_metadata(contradictory)

        split_cases = (
            {"manifest_sha256": "b" * 64, "test_index_opened": False},
            {"manifest_sha256": "a" * 64, "test_index_opened": True},
            [],
        )
        for split_provenance in split_cases:
            with self.subTest(split_provenance=split_provenance):
                tampered = copy.deepcopy(valid)
                tampered["split_provenance"] = split_provenance
                with self.assertRaises(ValueError):
                    public_test._checkpoint_audit_metadata(tampered)

        absent_split = copy.deepcopy(valid)
        del absent_split["split_provenance"]
        with self.assertRaisesRegex(ValueError, "missing identity fields"):
            public_test._checkpoint_audit_metadata(absent_split)

    def test_nonvalidation_role_may_honestly_carry_partial_audit_fields(self) -> None:
        self.assertEqual(
            public_test._checkpoint_audit_metadata(
                {
                    "checkpoint_role": "historical_operational",
                    "selection_is_optimistic": True,
                    "source_selection": "historical_test_best_miou",
                    "smoke": False,
                    "training_identity_sha256": "E" * 64,
                }
            ),
            {
                "selection_is_optimistic": True,
                "source_selection": "historical_test_best_miou",
                "smoke": False,
                "training_identity_sha256": "e" * 64,
            },
        )

    def test_smoke_checkpoint_is_rejected_before_any_public_test_result(self) -> None:
        for checkpoint_role in ("validation_selected", "final", "custom"):
            with self.subTest(checkpoint_role=checkpoint_role):
                payload = {
                    "checkpoint_role": checkpoint_role,
                    "smoke": True,
                }
                if checkpoint_role == "validation_selected":
                    payload.update(audit_fields())
                    payload["smoke"] = True
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot enter public test",
                ):
                    public_test._checkpoint_audit_metadata(payload)

    def test_legacy_clean_v1_without_optional_audit_fields_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "legacy-clean-v1.pth.tar"
            torch.save(clean_checkpoint_payload(include_audit=False), checkpoint)

            state, payload, observed_sha256 = public_test._state_from_checkpoint(
                checkpoint
            )

        self.assertEqual(len(state), 564)
        self.assertEqual(len(observed_sha256), 64)
        self.assertEqual(public_test._checkpoint_audit_metadata(payload), {})
        legacy_result = public_test._checkpoint_result_metadata(
            {
                "checkpoint_path": "/fixture/legacy.pth.tar",
                "epoch": 1000,
                "checkpoint_role": "final",
                "checkpoint_sha256": "c" * 64,
            }
        )
        self.assertEqual(
            legacy_result,
            {
                "path": "<external>/legacy.pth.tar",
                "path_scope": "external-path-redacted",
                "epoch": 1000,
                "role": "final",
                "sha256": "c" * 64,
            },
        )

    def test_published_weight_loading_path_is_unchanged(self) -> None:
        fixture_model = torch.nn.Identity()
        fixture_metadata = {"checkpoint_role": "final", "epoch": 850}
        with mock.patch.object(
            public_test,
            "load_evisirst",
            return_value=(fixture_model, fixture_metadata),
        ) as loader:
            observed_model, observed_metadata = public_test.load_model(
                "evisirst", "NUAA-SIRST", None
            )

        loader.assert_called_once_with("NUAA-SIRST")
        self.assertIs(observed_model, fixture_model)
        self.assertIs(observed_metadata, fixture_metadata)


if __name__ == "__main__":
    unittest.main()
