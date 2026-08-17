from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_irstd_complete_target_three_seed_gate as gate


def _record(
    seed: int,
    *,
    miou_delta: float,
    pd_delta: float = 0.0,
    fa_delta: float = 0.0,
    recall_delta: float = 0.01,
    area_closeness_delta: float = 0.01,
) -> dict[str, object]:
    baseline_area = 1.05
    variant_area = 1.05 - area_closeness_delta
    return {
        "run_seed": seed,
        "baseline_selected_epoch": 100,
        "variant_selected_epoch": 200,
        "baseline": {
            "mIoU": 0.60,
            "nIoU": 0.58,
            "Pd": 0.90,
            "Fa": 0.00001,
            "mechanism": {
                "matched_target_pixel_recall": 0.80,
                "matched_component_area_ratio": baseline_area,
            },
        },
        "variant": {
            "mIoU": 0.60 + miou_delta,
            "nIoU": 0.58 + miou_delta,
            "Pd": 0.90 + pd_delta,
            "Fa": 0.00001 + fa_delta,
            "mechanism": {
                "matched_target_pixel_recall": 0.80 + recall_delta,
                "matched_component_area_ratio": variant_area,
            },
        },
        "verified_inputs": {"fixture": str(seed)},
        "test_split_accessed": False,
    }


def _payload(records: list[dict[str, object]]) -> dict[str, object]:
    return gate.build_result_payload(
        seed_evidence=records,
        gate_source_artifact={"relative_path": "gate.py", "sha256": "0" * 64},
        frozen_rules_artifact={
            "relative_path": gate.RULES_RELATIVE_PATH,
            "sha256": "1" * 64,
        },
    )


def _replication_identity_fixture(
    seed: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    with gate._canonical_pilot_gate_bindings():
        authorization = (
            gate.replication_runner.validate_canonical_expansion_authorization()
        )
    source = gate._expected_replication_source_set(authorization)
    determinism = gate._expected_replication_determinism(source)
    pilot_summary = json.loads(
        (
            gate.PROJECT_ROOT
            / gate.SEED_INPUT_PATHS[gate.RUN_SEEDS[0]]["variant_summary"]
        ).read_text("utf-8")
    )
    identity = copy.deepcopy(pilot_summary["run_identity"])
    identity.update(
        {
            "schema": gate.replication_runner.TRAINING_SCHEMA
            + "/run_identity",
            "run_seed": seed,
            "experiment": {
                "schema": gate.replication_runner.EXPERIMENT_SCHEMA,
                "name": (
                    "IRSTD-1K complete-target crop v1 runtime-seed replication"
                ),
                "status": "confirmatory_validation_only",
                "single_variable_from_completed_pilot": "runtime_seed",
                "architecture_seed_fixed": 42,
                "completed_pilot_run_seed": gate.RUN_SEEDS[0],
                "authorized_three_runtime_seeds": list(gate.RUN_SEEDS),
                "public_test_supported": False,
                "public_test_gate_status": "unsupported_pending_separate_review",
            },
            "execution_contract": {
                "single_process_only": True,
                "python_threads_running_variant": 1,
                "data_loader_workers": 0,
                "nonblocking_process_lock": (
                    ".complete_target_replication_v1.lock"
                ),
            },
            "determinism_protocol": determinism,
            "promotion_gate": authorization,
        }
    )
    identity.pop("identity_sha256")
    identity["identity_sha256"] = gate.pilot_gate._canonical_sha256(identity)
    return authorization, identity, pilot_summary


def _completed_replication_evidence_fixture(
    seed: int,
    *,
    authorization: dict[str, object],
    identity: dict[str, object],
    source_tree: str,
    selected: dict[str, object],
    summary: dict[str, object],
) -> dict[str, object]:
    contract = gate.replication_runner.formal_artifact_contract(seed)
    frontier = summary["selection"]["retention_frontier_epochs"]
    retained = [
        {
            "epoch": epoch,
            "relative_path": summary["candidate_artifacts"][str(epoch)][
                "relative_path"
            ],
            "sha256": summary["candidate_artifacts"][str(epoch)][
                "file_sha256"
            ],
            "selected": epoch == selected["epoch"],
        }
        for epoch in frontier
    ]
    return {
        "schema": gate.replication_runner.TRAINING_SCHEMA
        + "/completed_run_evidence",
        "status": "complete",
        "run_seed": seed,
        "summary": {
            "relative_path": contract["summary_relative_path"],
            "sha256": "a" * 64,
        },
        "checkpoint": {
            "relative_path": contract["checkpoint_relative_path"],
            "sha256": "b" * 64,
            "state_key_count": 564,
        },
        "training_identity_sha256": identity["identity_sha256"],
        "source_tree_sha256": source_tree,
        "canonical_authorization_sha256": (
            gate.pilot_gate._canonical_sha256(authorization)
        ),
        "selected_epoch": selected["epoch"],
        "selected_validation_record_sha256": (
            gate.pilot_gate._canonical_sha256(selected)
        ),
        "retention_frontier_epochs": frontier,
        "retained_candidates": retained,
        "retained_candidate_count": len(retained),
        "train_epoch_count": 1000,
        "validation_epoch_count": 1000,
        "test_split_accessed": False,
        "public_test_supported": False,
    }


class ThreeSeedGateTests(unittest.TestCase):
    def test_rules_file_and_embedded_rules_have_frozen_canonical_hash(self) -> None:
        metadata = gate._load_and_validate_frozen_rules()
        observed = json.loads(
            (gate.PROJECT_ROOT / gate.RULES_RELATIVE_PATH).read_text("utf-8")
        )
        canonical = gate._canonical_json_bytes(observed)
        self.assertEqual(observed, gate.PREREGISTERED_RULES)
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            "36b503e9b5906e1118da670313cf2bbf7b131995d176aba70f4936f7de473ef2",
        )
        self.assertEqual(
            gate.PREREGISTERED_RULES_SHA256,
            "36b503e9b5906e1118da670313cf2bbf7b131995d176aba70f4936f7de473ef2",
        )
        self.assertEqual(metadata["relative_path"], gate.RULES_RELATIVE_PATH)

    def test_any_rule_mutation_changes_hash_and_is_rejected(self) -> None:
        changed = copy.deepcopy(gate.PREREGISTERED_RULES)
        changed["primary"]["minimum_positive_seed_count"] = 3
        self.assertNotEqual(
            hashlib.sha256(gate._canonical_json_bytes(changed)).hexdigest(),
            gate.PREREGISTERED_RULES_SHA256,
        )
        with mock.patch.object(gate, "PREREGISTERED_RULES", changed):
            with self.assertRaisesRegex(gate.ThreeSeedGateError, "rule bytes"):
                _payload(
                    [_record(seed, miou_delta=0.01) for seed in gate.RUN_SEEDS]
                )

    def test_fixed_paths_distinguish_pilot_and_replication_variants(self) -> None:
        pilot = gate.SEED_INPUT_PATHS[1446202191]
        first = gate.SEED_INPUT_PATHS[104728269]
        second = gate.SEED_INPUT_PATHS[262620274]
        self.assertEqual(
            pilot["variant_summary"],
            "runs/irstd_performance/complete_target_v1/formal/IRSTD-1K/"
            "binary/run_seed_1446202191/summary.json",
        )
        self.assertEqual(
            first["variant_summary"],
            "runs/irstd_performance/complete_target_v1/"
            "three_runtime_seed_validation/formal/IRSTD-1K/binary/"
            "run_seed_104728269/summary.json",
        )
        self.assertEqual(
            second["variant_checkpoint"],
            "runs/irstd_performance/complete_target_v1/"
            "three_runtime_seed_validation/formal/IRSTD-1K/binary/"
            "run_seed_262620274/EviSIRST.pth.tar",
        )
        with self.assertRaises(gate.ThreeSeedGateError):
            gate._seed_paths(42)

    def test_primary_requires_positive_mean_and_two_strictly_positive_seeds(self) -> None:
        one_positive = _payload(
            [
                _record(gate.RUN_SEEDS[0], miou_delta=0.03),
                _record(gate.RUN_SEEDS[1], miou_delta=-0.01),
                _record(gate.RUN_SEEDS[2], miou_delta=-0.01),
            ]
        )
        self.assertGreater(one_positive["primary_gate"]["mean_delta"], 0)
        self.assertEqual(
            one_positive["primary_gate"]["strictly_positive_seed_count"], 1
        )
        self.assertFalse(one_positive["primary_gate"]["passed"])

        two_positive = _payload(
            [
                _record(gate.RUN_SEEDS[0], miou_delta=0.01),
                _record(gate.RUN_SEEDS[1], miou_delta=0.01),
                _record(gate.RUN_SEEDS[2], miou_delta=-0.005),
            ]
        )
        self.assertTrue(two_positive["primary_gate"]["passed"])

        zero_is_not_positive = _payload(
            [
                _record(gate.RUN_SEEDS[0], miou_delta=0.02),
                _record(gate.RUN_SEEDS[1], miou_delta=0.0),
                _record(gate.RUN_SEEDS[2], miou_delta=-0.001),
            ]
        )
        self.assertEqual(
            zero_is_not_positive["primary_gate"][
                "strictly_positive_seed_count"
            ],
            1,
        )
        self.assertFalse(zero_is_not_positive["primary_gate"]["passed"])

    def test_safety_checks_each_seed_and_aggregate_joint_failure(self) -> None:
        per_seed = _payload(
            [
                _record(
                    gate.RUN_SEEDS[0],
                    miou_delta=0.01,
                    pd_delta=-0.004,
                    fa_delta=0.000001,
                ),
                _record(gate.RUN_SEEDS[1], miou_delta=0.01, pd_delta=0.01),
                _record(gate.RUN_SEEDS[2], miou_delta=0.01, pd_delta=0.01),
            ]
        )
        self.assertEqual(
            per_seed["safety_gate"]["per_seed_joint_failure_seeds"],
            [gate.RUN_SEEDS[0]],
        )
        self.assertFalse(per_seed["safety_gate"]["passed"])

        boundary = _payload(
            [
                _record(
                    seed,
                    miou_delta=0.01,
                    pd_delta=-0.002999,
                    fa_delta=0.000001,
                )
                for seed in gate.RUN_SEEDS
            ]
        )
        self.assertTrue(boundary["safety_gate"]["passed"])

        aggregate = _payload(
            [
                _record(
                    gate.RUN_SEEDS[0],
                    miou_delta=0.01,
                    pd_delta=-0.002,
                    fa_delta=0.000001,
                ),
                _record(
                    gate.RUN_SEEDS[1],
                    miou_delta=0.01,
                    pd_delta=-0.002,
                    fa_delta=0.000001,
                ),
                _record(
                    gate.RUN_SEEDS[2],
                    miou_delta=0.01,
                    pd_delta=-0.006,
                    fa_delta=-0.000001,
                ),
            ]
        )
        self.assertEqual(
            aggregate["safety_gate"]["per_seed_joint_failure_seeds"], []
        )
        self.assertTrue(aggregate["safety_gate"]["aggregate_joint_failure"])
        self.assertFalse(aggregate["safety_gate"]["passed"])

    def test_mechanism_requires_one_same_endpoint_to_pass_both_conditions(self) -> None:
        cross_endpoint_votes = _payload(
            [
                _record(
                    gate.RUN_SEEDS[0],
                    miou_delta=0.01,
                    recall_delta=0.03,
                    area_closeness_delta=-0.02,
                ),
                _record(
                    gate.RUN_SEEDS[1],
                    miou_delta=0.01,
                    recall_delta=-0.02,
                    area_closeness_delta=0.03,
                ),
                _record(
                    gate.RUN_SEEDS[2],
                    miou_delta=0.01,
                    recall_delta=-0.001,
                    area_closeness_delta=-0.001,
                ),
            ]
        )
        self.assertFalse(cross_endpoint_votes["mechanism_gate"]["passed"])
        for endpoint in cross_endpoint_votes["mechanism_gate"]["endpoints"].values():
            self.assertEqual(endpoint["strictly_positive_seed_count"], 1)

        recall_pass = _payload(
            [
                _record(seed, miou_delta=0.01, recall_delta=delta)
                for seed, delta in zip(gate.RUN_SEEDS, (0.02, 0.01, -0.005))
            ]
        )
        self.assertTrue(recall_pass["mechanism_gate"]["passed"])

    def test_reports_descriptive_t_interval_without_using_it_for_gate(self) -> None:
        payload = _payload(
            [
                _record(seed, miou_delta=delta)
                for seed, delta in zip(gate.RUN_SEEDS, (0.01, 0.02, -0.005))
            ]
        )
        stats = payload["paired_delta_statistics"]["mIoU"]
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["degrees_of_freedom"], 2)
        self.assertFalse(stats["decision_use"])
        self.assertEqual(
            stats["interpretation"],
            "descriptive_only_not_a_statistical_significance_claim",
        )
        self.assertTrue(math.isfinite(stats["sample_std_ddof_1"]))
        self.assertEqual(len(stats["descriptive_95pct_t_interval"]), 2)
        self.assertTrue(payload["primary_gate"]["passed"])
        self.assertFalse(payload["decision"]["public_test_allowed"])
        self.assertFalse(payload["public_test_allowed"])
        self.assertFalse(payload["test_split_accessed"])

    def test_cli_has_no_tunable_arguments(self) -> None:
        gate.parse_args([])
        with self.assertRaises(SystemExit):
            gate.parse_args(["--run-seed", "42"])
        with self.assertRaises(SystemExit):
            gate.parse_args(["--threshold", "0"])

    def test_atomic_writer_never_overwrites(self) -> None:
        payload = _payload(
            [_record(seed, miou_delta=0.01) for seed in gate.RUN_SEEDS]
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "runs").mkdir()
            with (
                mock.patch.object(gate, "PROJECT_ROOT", root),
                mock.patch.object(gate, "OUTPUT_RELATIVE_PATH", "runs/gate/result.json"),
            ):
                path = gate._write_fixed_json_atomic_no_replace(payload)
                before = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    gate._write_fixed_json_atomic_no_replace({"changed": True})
                self.assertEqual(path.read_bytes(), before)

    def test_completed_pilot_adapter_strictly_loads_checkpoint_and_metrics(self) -> None:
        record = gate._validate_seed_evidence(gate.RUN_SEEDS[0])
        self.assertEqual(record["run_seed"], gate.RUN_SEEDS[0])
        self.assertEqual(
            record["verified_inputs"]["checkpoint_state_tensor_count"], 564
        )
        self.assertTrue(record["verified_inputs"]["selection_recomputed"])
        self.assertEqual(
            record["verified_inputs"]["history_continuous_through_epoch"],
            1000,
        )
        self.assertFalse(record["test_split_accessed"])

    def test_confirmatory_diagnostic_wrapper_is_exactly_reconstructed(self) -> None:
        seed = gate.RUN_SEEDS[1]
        old_path = (
            gate.PROJECT_ROOT
            / gate.SEED_INPUT_PATHS[gate.RUN_SEEDS[0]]["baseline_diagnostic"]
        )
        core = json.loads(old_path.read_text("utf-8"))
        paths = gate.confirmatory_diagnostic._relative_paths(seed)
        core["checkpoint"]["relative_path"] = paths[
            "CHECKPOINT_RELATIVE_PATH"
        ]
        core["checkpoint"]["run_seed"] = seed
        core["completed_run_summary"]["relative_path"] = paths[
            "SUMMARY_RELATIVE_PATH"
        ]
        core_sources = (
            gate.confirmatory_diagnostic.diagnostic_core
            .diagnostic_source_provenance()
        )
        core["sources"]["diagnostic"] = core_sources
        confirmatory_sources = (
            gate.confirmatory_diagnostic.confirmatory_source_provenance()
        )
        with gate._canonical_pilot_gate_bindings():
            authority = gate.confirmatory_diagnostic.validate_pilot_authority()
        wrapped = gate.confirmatory_diagnostic.build_result_payload(
            run_seed=seed,
            paths=paths,
            core_payload=core,
            core_sources=core_sources,
            confirmatory_sources=confirmatory_sources,
            pilot_authority=authority,
        )
        with mock.patch.object(
            gate.confirmatory_diagnostic,
            "validate_existing_result",
            return_value=wrapped,
        ) as strict_validator:
            recovered = gate._validate_confirmatory_diagnostic_wrapper(
                wrapped,
                seed=seed,
                diagnostic_metadata={
                    "relative_path": paths["OUTPUT_RELATIVE_PATH"],
                    "sha256": "a" * 64,
                },
            )
        strict_validator.assert_called_once_with(seed)
        self.assertEqual(recovered, core)
        with (
            mock.patch.object(
                gate.confirmatory_diagnostic,
                "validate_existing_result",
                return_value=wrapped,
            ),
            self.assertRaisesRegex(
                gate.ThreeSeedGateError, "output path differs"
            ),
        ):
            gate._validate_confirmatory_diagnostic_wrapper(
                wrapped,
                seed=seed,
                diagnostic_metadata={
                    "relative_path": "runs/redirected/result.json",
                    "sha256": "a" * 64,
                },
            )
        tampered = copy.deepcopy(wrapped)
        tampered["diagnostic_identity"]["summary_sha256"] = "0" * 64
        with (
            mock.patch.object(
                gate.confirmatory_diagnostic,
                "validate_existing_result",
                return_value=tampered,
            ),
            self.assertRaisesRegex(gate.ThreeSeedGateError, "identity/hash"),
        ):
            gate._validate_confirmatory_diagnostic_wrapper(
                tampered, seed=seed
            )

    def test_confirmatory_diagnostic_strict_validator_fails_closed(self) -> None:
        seed = gate.RUN_SEEDS[1]
        diagnostic = {"schema": "fixture"}
        for label, outcome in (
            (
                "validator failure",
                gate.confirmatory_diagnostic.ConfirmatoryDiagnosticError(
                    "strict checkpoint/crop evidence rejected"
                ),
            ),
            ("validator returned tampered evidence", {"schema": "changed"}),
        ):
            with self.subTest(label=label):
                kwargs = (
                    {"side_effect": outcome}
                    if isinstance(outcome, BaseException)
                    else {"return_value": outcome}
                )
                with mock.patch.object(
                    gate.confirmatory_diagnostic,
                    "validate_existing_result",
                    **kwargs,
                ) as strict_validator:
                    expected = (
                        "strict confirmatory diagnostic validation failed"
                        if isinstance(outcome, BaseException)
                        else "strict confirmatory diagnostic evidence differs"
                    )
                    with self.assertRaisesRegex(
                        gate.ThreeSeedGateError, expected
                    ):
                        gate._validate_confirmatory_diagnostic_wrapper(
                            diagnostic, seed=seed
                        )
                strict_validator.assert_called_once_with(seed)

    def test_replication_identity_allows_only_frozen_provenance_differences(self) -> None:
        seed = gate.RUN_SEEDS[1]
        authorization, identity, _pilot_summary = _replication_identity_fixture(
            seed
        )
        source = gate._expected_replication_source_set(authorization)
        observed, source_tree = gate._validate_replication_identity(
            identity, seed=seed, authorization=authorization
        )
        self.assertEqual(observed, identity)
        self.assertEqual(source_tree, source["source_tree_sha256"])

        changed = copy.deepcopy(identity)
        changed["base_lr"] = 0.002
        changed.pop("identity_sha256")
        changed["identity_sha256"] = gate.pilot_gate._canonical_sha256(changed)
        with self.assertRaisesRegex(
            gate.ThreeSeedGateError, "base_lr"
        ):
            gate._validate_replication_identity(
                changed, seed=seed, authorization=authorization
            )

    def test_replication_adapter_binds_strict_completed_run_evidence(self) -> None:
        seed = gate.RUN_SEEDS[1]
        authorization, identity, pilot_summary = _replication_identity_fixture(
            seed
        )
        source_tree = identity["determinism_protocol"]["source_tree_sha256"]
        selected = copy.deepcopy(pilot_summary["selected_validation_record"])
        evidence = _completed_replication_evidence_fixture(
            seed,
            authorization=authorization,
            identity=identity,
            source_tree=source_tree,
            selected=selected,
            summary=pilot_summary,
        )
        contract = gate.replication_runner.formal_artifact_contract(seed)
        with mock.patch.object(
            gate.replication_runner,
            "validate_existing_completed_run",
            return_value=evidence,
        ) as strict_validator:
            observed = gate._load_replication_completed_run_evidence(
                seed=seed,
                contract=contract,
                authorization=authorization,
            )
        strict_validator.assert_called_once_with(seed)
        summary = {
            "training_history": [{} for _ in range(1000)],
            "validation_history": [{} for _ in range(1000)],
            "selection": copy.deepcopy(pilot_summary["selection"]),
            "candidate_artifacts": copy.deepcopy(
                pilot_summary["candidate_artifacts"]
            ),
            "selected_candidate_sha256": pilot_summary[
                "selected_candidate_sha256"
            ],
        }
        gate._bind_replication_completed_run_evidence(
            seed=seed,
            evidence=observed,
            summary=summary,
            summary_metadata={
                "relative_path": contract["summary_relative_path"],
                "sha256": "a" * 64,
            },
            checkpoint_metadata={
                "relative_path": contract["checkpoint_relative_path"],
                "sha256": "b" * 64,
            },
            selected=selected,
            identity=identity,
            source_tree=source_tree,
        )

        attacks = {
            "summary content SHA-256": ("summary", "sha256", "c" * 64),
            "checkpoint content SHA-256": (
                "checkpoint",
                "sha256",
                "d" * 64,
            ),
            "training identity SHA-256": (
                None,
                "training_identity_sha256",
                "e" * 64,
            ),
            "source tree SHA-256": (
                None,
                "source_tree_sha256",
                "f" * 64,
            ),
            "selected validation record SHA-256": (
                None,
                "selected_validation_record_sha256",
                "0" * 64,
            ),
            "training history length": (None, "train_epoch_count", 999),
            "validation history length": (
                None,
                "validation_epoch_count",
                999,
            ),
            "candidate frontier binding": (
                None,
                "retention_frontier_epochs",
                observed["retention_frontier_epochs"][:-1],
            ),
        }
        for expected_label, (section, key, changed) in attacks.items():
            with self.subTest(expected_label=expected_label):
                tampered = copy.deepcopy(observed)
                target = tampered if section is None else tampered[section]
                target[key] = changed
                with self.assertRaisesRegex(
                    gate.ThreeSeedGateError, expected_label
                ):
                    gate._bind_replication_completed_run_evidence(
                        seed=seed,
                        evidence=tampered,
                        summary=summary,
                        summary_metadata={
                            "relative_path": contract[
                                "summary_relative_path"
                            ],
                            "sha256": "a" * 64,
                        },
                        checkpoint_metadata={
                            "relative_path": contract[
                                "checkpoint_relative_path"
                            ],
                            "sha256": "b" * 64,
                        },
                        selected=selected,
                        identity=identity,
                        source_tree=source_tree,
                    )

    def test_rehashed_tensor_crop_and_history_attacks_cannot_bypass_runner(self) -> None:
        seed = gate.RUN_SEEDS[1]
        authorization, _identity, _pilot_summary = (
            _replication_identity_fixture(seed)
        )
        contract = gate.replication_runner.formal_artifact_contract(seed)
        attacks = {
            "one_of_564_tensors_replaced_by_scalar": {
                "state_dict": {
                    f"tensor_{index:03d}": (0.0 if index == 0 else index)
                    for index in range(564)
                }
            },
            "crop_audit_history_tampered": {
                "search_run_crop_audit_history": [
                    {"epoch": 1, "summary": {"observation_count": 0}}
                ]
            },
            "training_or_validation_history_tampered": {
                "training_history": [{"epoch": 1}],
                "validation_history": [{"epoch": 1}],
            },
        }
        for attack_name, tampered_payload in attacks.items():
            with self.subTest(attack_name=attack_name):
                # Model an attacker recomputing the enclosing artifact hash.
                # The gate must still rely on the semantic runner validator,
                # never accept a merely self-consistent outer digest.
                recomputed_outer_sha256 = hashlib.sha256(
                    gate._canonical_json_bytes(tampered_payload)
                ).hexdigest()
                self.assertEqual(len(recomputed_outer_sha256), 64)
                rejection = gate.replication_runner.CompleteTargetReplicationError(
                    f"strict validator rejected {attack_name} after outer rehash"
                )
                with (
                    mock.patch.object(
                        gate.replication_runner,
                        "validate_existing_completed_run",
                        side_effect=rejection,
                    ) as strict_validator,
                    self.assertRaisesRegex(
                        gate.ThreeSeedGateError,
                        "strict replication completion validation failed",
                    ),
                ):
                    gate._load_replication_completed_run_evidence(
                        seed=seed,
                        contract=contract,
                        authorization=authorization,
                    )
                strict_validator.assert_called_once_with(seed)

    def test_replication_checkpoint_adapter_loads_and_checks_564_tensors(self) -> None:
        seed = gate.RUN_SEEDS[1]
        authorization, identity, pilot_summary = _replication_identity_fixture(
            seed
        )
        pilot_checkpoint_path = (
            gate.PROJECT_ROOT
            / gate.SEED_INPUT_PATHS[gate.RUN_SEEDS[0]]["variant_checkpoint"]
        )
        checkpoint = gate.pilot_gate.torch.load(
            pilot_checkpoint_path, map_location="cpu", weights_only=True
        )
        summary = copy.deepcopy(pilot_summary)
        summary.update(
            {
                "schema": gate.replication_runner.TRAINING_SCHEMA + "/summary",
                "checkpoint": gate.SEED_INPUT_PATHS[seed]["variant_checkpoint"],
                "run_seed": seed,
                "experiment_schema": gate.replication_runner.EXPERIMENT_SCHEMA,
                "promotion_gate": authorization,
                "run_identity": identity,
                "training_identity_sha256": identity["identity_sha256"],
                "final_checkpoint_sha256": "a" * 64,
            }
        )
        checkpoint = copy.deepcopy(checkpoint)
        checkpoint.update(
            {
                "schema": gate.replication_runner.CHECKPOINT_SCHEMA,
                "run_seed": seed,
                "experiment_schema": gate.replication_runner.EXPERIMENT_SCHEMA,
                "promotion_gate": authorization,
                "training": identity,
                "training_identity_sha256": identity["identity_sha256"],
            }
        )
        selection = summary["selection"]["selection_provenance"]
        identity_observed, source_tree = gate._validate_replication_checkpoint(
            checkpoint,
            {"relative_path": "fixture", "sha256": "a" * 64},
            seed=seed,
            summary=summary,
            selection=selection,
            authorization=authorization,
        )
        self.assertEqual(identity_observed, identity)
        self.assertEqual(
            source_tree,
            identity["determinism_protocol"]["source_tree_sha256"],
        )
        self.assertEqual(len(checkpoint["state_dict"]), 564)

        first_key = next(iter(checkpoint["state_dict"]))
        checkpoint["state_dict"][first_key] = gate.pilot_gate.torch.tensor(
            float("nan")
        )
        with self.assertRaisesRegex(
            gate.pilot_gate.PromotionGateError, "non-finite"
        ):
            gate._validate_replication_checkpoint(
                checkpoint,
                {"relative_path": "fixture", "sha256": "a" * 64},
                seed=seed,
                summary=summary,
                selection=selection,
                authorization=authorization,
            )


if __name__ == "__main__":
    unittest.main()
