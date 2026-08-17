from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import train_irstd_complete_target_replication_v1 as runner


class FakeNormalizationSpec:
    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "legacy",
            "source": "fixture_frozen_legacy",
            "mean": 1.0,
            "std": 2.0,
        }


class FakeFormalTrainDataset(Dataset):
    def __init__(self, contract: SimpleNamespace) -> None:
        self.contract = contract
        self.sample_ids = tuple(
            f"train_{index:04d}"
            for index in range(runner.CANONICAL_IRSTD_TRAIN_COUNT)
        )
        self.normalization = {"mean": 1.0, "std": 2.0}
        self.normalization_spec = FakeNormalizationSpec()
        self.epoch = 0
        self.audit_observation_count = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return runner.CANONICAL_IRSTD_TRAIN_COUNT

    def __getitem__(self, index: int):
        self.audit_observation_count += 1
        return (
            torch.zeros(1, 8, 8, dtype=torch.float32),
            torch.ones(1, 8, 8, dtype=torch.float32),
        )

    def crop_audit_state_dict(self) -> dict[str, object]:
        count = self.audit_observation_count
        return {
            "schema": runner.complete_crop.CROP_AUDIT_STATE_SCHEMA,
            "augmentation_version": runner.complete_crop.AUGMENTATION_VERSION,
            "formal_num_workers": 0,
            "observation_count": count,
            "requested_counts": {"complete_target": 0, "uniform": count},
            "realized_counts": {"complete_target": 0, "uniform": count},
            "fallback_count": 0,
            "fallback_reason_counts": {},
            "cut_component_crop_count": 0,
            "total_cut_component_count": 0,
            "max_cut_component_count": 0,
            "test_split_accessed": False,
        }

    def load_crop_audit_state_dict(self, payload: dict[str, object]) -> None:
        verifier = runner.complete_crop.CropAuditAccumulator()
        verifier.load_state_dict(payload)
        self.audit_observation_count = int(payload["observation_count"])

    def crop_audit_summary(self) -> dict[str, object]:
        verifier = runner.complete_crop.CropAuditAccumulator()
        verifier.load_state_dict(self.crop_audit_state_dict())
        return verifier.compute()


class FakeFormalValDataset(Dataset):
    def __init__(self, contract: SimpleNamespace) -> None:
        self.contract = contract
        self.sample_ids = tuple(
            f"val_{index:04d}"
            for index in range(runner.CANONICAL_IRSTD_VAL_COUNT)
        )
        self.normalization = {"mean": 1.0, "std": 2.0}
        self.normalization_spec = FakeNormalizationSpec()

    def __len__(self) -> int:
        return runner.CANONICAL_IRSTD_VAL_COUNT

    def __getitem__(self, index: int):
        return (
            torch.zeros(1, 8, 8, dtype=torch.float32),
            torch.ones(1, 8, 8, dtype=torch.float32),
            (8, 8),
            self.sample_ids[index],
        )


class Fake564Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros((), dtype=torch.float32))
                for _ in range(564)
            ]
        )
        self.mode = "train"

    def forward(self, images: torch.Tensor):
        probability = torch.sigmoid(images * 0.0 + self.weights[0])
        if self.mode == "train":
            return tuple(probability for _ in range(6))
        return probability


class CompleteTargetReplicationRunnerTest(unittest.TestCase):
    def _parse(self, seed: int = runner.REPLICATION_RUN_SEEDS[0], *extra: str):
        return runner.parse_args(
            [
                "--dataset-root",
                "/tmp/irstd-dataset",
                "--run-seed",
                str(seed),
                *extra,
            ]
        )

    def _authorization(self) -> dict[str, object]:
        payload = {
            "schema": runner.CANONICAL_GATE_RESULT_SCHEMA,
            "status": "complete",
            "dataset": runner.DATASET,
            "data_role": "val",
            "architecture_seed": runner.ARCHITECTURE_SEED,
            "run_seed": runner.PILOT_RUN_SEED,
            "epochs": runner.FORMAL_EPOCHS,
            "inputs": {
                "all_artifact_hashes_verified": True,
                "all_source_hashes_currently_verified": True,
                "canonical_train_val_split_verified": True,
                "both_selections_recomputed": True,
                "gate_evaluator_source": {
                    "relative_path": runner.canonical_gate.GATE_SOURCE_RELATIVE_PATH,
                    "sha256": runner.FROZEN_CANONICAL_GATE_SOURCE_SHA256,
                },
            },
            "decision": {
                "overall_passed": True,
                "result": "PASS",
                "three_runtime_seed_validation_expansion_allowed": True,
                "three_runtime_seed_validation_expansion_status": "allowed",
                "public_test_allowed": False,
            },
            "test_split_accessed": False,
            "public_test_allowed": False,
        }
        parent = runner._frozen_parent_source_set()
        return {
            "schema": runner.AUTHORIZATION_SCHEMA,
            "status": "complete",
            "authority": (
                "run_irstd_complete_target_promotion_gate."
                "validate_existing_result"
            ),
            "canonical_result_artifact": {
                "relative_path": runner.CANONICAL_GATE_RESULT_RELATIVE_PATH,
                "sha256": "a" * 64,
            },
            "canonical_payload_schema": runner.CANONICAL_GATE_RESULT_SCHEMA,
            "canonical_payload_sha256": runner._canonical_sha256(payload),
            "canonical_payload": payload,
            "canonical_gate_evaluator_source": {
                "relative_path": runner.canonical_gate.GATE_SOURCE_RELATIVE_PATH,
                "sha256": runner.FROZEN_CANONICAL_GATE_SOURCE_SHA256,
            },
            "frozen_parent_source_set": parent,
            "frozen_parent_source_tree_sha256": (
                runner.FROZEN_PARENT_SOURCE_TREE_SHA256
            ),
            "completed_pilot_run_seed": runner.PILOT_RUN_SEED,
            "authorized_three_runtime_seeds": list(runner.THREE_RUNTIME_SEEDS),
            "pending_replication_run_seeds": list(
                runner.REPLICATION_RUN_SEEDS
            ),
            "formal_replication_allowed": True,
            "public_test_supported": False,
            "public_test_accessed": False,
        }

    def _contract(self) -> SimpleNamespace:
        grouping = {
            "mode": "sample_level_fallback",
            "warning": "fixture acknowledgement",
        }
        return SimpleNamespace(
            dataset=runner.DATASET,
            manifest_sha256=runner.CANONICAL_IRSTD_MANIFEST_SHA256,
            data_tree_sha256=runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
            data_tree_verified=True,
            manifest={
                "schema": "evisirst_v2_train_val_split/v1",
                "seeds": {"split_seed": 20260811},
                "source_index": {
                    "split": "train",
                    "relative_path": "IRSTD-1K/img_idx/train_IRSTD-1K.txt",
                    "file_sha256": "1" * 64,
                    "ordered_ids_sha256": "2" * 64,
                    "sample_count": 800,
                },
                "outputs": {
                    "train": {
                        "relative_path": "splits/v2/IRSTD-1K/train.txt",
                        "file_sha256": "3" * 64,
                        "ordered_ids_sha256": "4" * 64,
                        "sample_count": runner.CANONICAL_IRSTD_TRAIN_COUNT,
                    },
                    "val": {
                        "relative_path": "splits/v2/IRSTD-1K/val.txt",
                        "file_sha256": "5" * 64,
                        "ordered_ids_sha256": "6" * 64,
                        "sample_count": runner.CANONICAL_IRSTD_VAL_COUNT,
                    },
                },
                "grouping": grouping,
            },
            train_ids=tuple(
                f"train_{index:04d}"
                for index in range(runner.CANONICAL_IRSTD_TRAIN_COUNT)
            ),
            val_ids=tuple(
                f"val_{index:04d}"
                for index in range(runner.CANONICAL_IRSTD_VAL_COUNT)
            ),
        )

    def _identity(
        self, args: object, authorization: dict[str, object]
    ) -> dict[str, object]:
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture acknowledgement",
        }
        with runner._replication_runtime(authorization):
            return runner._replication_run_identity(
                args,
                self._contract(),
                grouping,
                train_count=runner.CANONICAL_IRSTD_TRAIN_COUNT,
                val_count=runner.CANONICAL_IRSTD_VAL_COUNT,
                smoke=False,
            )

    def _crop_audit(self, epoch: int) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        def state_for(count: int) -> dict[str, object]:
            return {
                "schema": runner.complete_crop.CROP_AUDIT_STATE_SCHEMA,
                "augmentation_version": runner.complete_crop.AUGMENTATION_VERSION,
                "formal_num_workers": 0,
                "observation_count": count,
                "requested_counts": {"complete_target": 0, "uniform": count},
                "realized_counts": {"complete_target": 0, "uniform": count},
                "fallback_count": 0,
                "fallback_reason_counts": {},
                "cut_component_crop_count": 0,
                "total_cut_component_count": 0,
                "max_cut_component_count": 0,
                "test_split_accessed": False,
            }

        def summary_for(count: int) -> dict[str, object]:
            verifier = runner.complete_crop.CropAuditAccumulator()
            verifier.load_state_dict(state_for(count))
            return verifier.compute()

        return (
            state_for(epoch),
            summary_for(epoch),
            [
                {"epoch": observed, "summary": summary_for(observed)}
                for observed in range(1, epoch + 1)
            ],
        )

    def _retained_candidate_fixture(self, root: Path) -> dict[str, object]:
        run_dir = root / "run_seed_fixture"
        candidate_dir = run_dir / "candidates"
        candidate_dir.mkdir(parents=True)
        identity = {"train_count": 1, "test_split_accessed": False}
        validation_history = [
            {
                "epoch": 1,
                "data_role": "val",
                "mIoU": 0.6,
                "Fa": 0.2,
                "Pd": 0.5,
            },
            {
                "epoch": 2,
                "data_role": "val",
                "mIoU": 0.5995,
                "Fa": 0.1,
                "Pd": 0.5,
            },
        ]
        self.assertEqual(
            runner.r1.selection.retention_frontier_epochs(validation_history),
            (1, 2),
        )
        self.assertEqual(
            runner.r1.selection.select_independent_checkpoint(
                validation_history
            )["selected"]["epoch"],
            2,
        )
        final_state = {
            f"weights.{index}": torch.tensor(float(index), dtype=torch.float32)
            for index in range(564)
        }
        candidate_artifacts: dict[str, dict[str, str]] = {}
        payloads: dict[int, dict[str, object]] = {}
        for epoch in (1, 2):
            audit_state, audit_summary, audit_history = self._crop_audit(epoch)
            state = {
                key: tensor.clone() + (0.0 if epoch == 2 else 1.0)
                for key, tensor in final_state.items()
            }
            payload: dict[str, object] = {
                "schema": runner.CANDIDATE_SCHEMA,
                "model": "EviSIRST",
                "dataset": runner.DATASET,
                "epoch": epoch,
                "run_identity": identity,
                "validation_record": validation_history[epoch - 1],
                "state_dict": state,
                "test_split_accessed": False,
                "experiment_schema": runner.EXPERIMENT_SCHEMA,
                "experiment_status": "experimental_validation_only",
                "public_test_supported": False,
                "public_test_gate_status": (
                    "unsupported_until_separate_gate_extension"
                ),
                "crop_audit_state": audit_state,
                "crop_audit_summary": audit_summary,
                "crop_audit_history": audit_history,
                "crop_audit_commit_status": (
                    "candidate_pre_latest_uncommitted"
                ),
            }
            path = candidate_dir / f"epoch_{epoch:04d}.pth.tar"
            torch.save(payload, path)
            payloads[epoch] = payload
            candidate_artifacts[str(epoch)] = {
                "relative_path": f"candidates/{path.name}",
                "file_sha256": runner._sha256_file(path),
            }
        selection = {
            "selected_epoch": 2,
            "retention_frontier_epochs": [1, 2],
            "selected_candidate": candidate_artifacts["2"],
        }
        selected = payloads[2]
        return {
            "run_dir": run_dir,
            "identity": identity,
            "validation_history": validation_history,
            "selection": selection,
            "candidate_artifacts": candidate_artifacts,
            "final_state": final_state,
            "selected_audit": {
                "state": selected["crop_audit_state"],
                "summary": selected["crop_audit_summary"],
                "history": selected["crop_audit_history"],
                "through_epoch": 2,
            },
            "selected_candidate_sha256": candidate_artifacts["2"][
                "file_sha256"
            ],
            "payloads": payloads,
        }

    def _refresh_candidate_binding(
        self, fixture: dict[str, object], epoch: int
    ) -> None:
        path = (
            Path(fixture["run_dir"])
            / "candidates"
            / f"epoch_{epoch:04d}.pth.tar"
        )
        digest = runner._sha256_file(path)
        artifacts = fixture["candidate_artifacts"]
        artifacts[str(epoch)]["file_sha256"] = digest
        if epoch == fixture["selection"]["selected_epoch"]:
            fixture["selection"]["selected_candidate"] = artifacts[str(epoch)]
            fixture["selected_candidate_sha256"] = digest

    def _candidate_temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        runs_root = runner.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(
            dir=runs_root, prefix="retained_candidate_fixture_"
        )

    def test_cli_and_paths_freeze_two_remaining_runtime_seeds(self) -> None:
        for seed in runner.REPLICATION_RUN_SEEDS:
            with self.subTest(seed=seed):
                args = self._parse(seed)
                self.assertEqual(args.epochs, 1000)
                self.assertEqual(args.workers, 0)
                self.assertEqual(args.architecture_seed, 42)
                self.assertEqual(args.device, "cuda:0")
                self.assertEqual(args.output_root, runner.DEFAULT_OUTPUT_ROOT)
                paths = runner.resolve_run_paths(args)
                self.assertFalse(paths["smoke"])
                self.assertEqual(
                    Path(paths["run_dir"]).relative_to(
                        runner.DEFAULT_OUTPUT_ROOT
                    ),
                    Path(
                        f"formal/IRSTD-1K/binary/run_seed_{seed}"
                    ),
                )
                artifact = runner.formal_artifact_contract(seed)
                self.assertEqual(
                    artifact["summary_schema"],
                    runner.TRAINING_SCHEMA + "/summary",
                )
                self.assertEqual(
                    artifact["checkpoint_schema"], runner.CHECKPOINT_SCHEMA
                )
        for forbidden in (
            runner.PILOT_RUN_SEED,
            42,
            -1,
        ):
            with self.subTest(forbidden=forbidden), self.assertRaises(
                SystemExit
            ):
                self._parse(forbidden)
        for extra in (
            ("--epochs", "999"),
            ("--device", "cpu"),
            ("--split-root", "/tmp/not-canonical"),
            ("--architecture-seed", "43"),
            ("--target-mode", "soft"),
        ):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self._parse(runner.REPLICATION_RUN_SEEDS[0], *extra)

    def test_real_canonical_authorization_is_pass_and_source_bound(self) -> None:
        authorization = runner.validate_canonical_expansion_authorization()
        self.assertEqual(authorization["schema"], runner.AUTHORIZATION_SCHEMA)
        self.assertTrue(authorization["formal_replication_allowed"])
        self.assertFalse(authorization["public_test_supported"])
        self.assertFalse(authorization["public_test_accessed"])
        self.assertEqual(
            authorization["canonical_payload"]["decision"]["result"],
            "PASS",
        )
        self.assertTrue(
            authorization["canonical_payload"]["decision"][
                "three_runtime_seed_validation_expansion_allowed"
            ]
        )
        self.assertEqual(
            authorization["frozen_parent_source_tree_sha256"],
            runner.FROZEN_PARENT_SOURCE_TREE_SHA256,
        )
        self.assertEqual(
            authorization["frozen_parent_source_set"]["files"][
                "variant_runner"
            ]["sha256"],
            runner.FROZEN_PILOT_RUNNER_SHA256,
        )

    def test_gate_failure_precedes_lock_output_and_cuda_engine(self) -> None:
        args = self._parse()
        with tempfile.TemporaryDirectory() as temporary:
            isolated_root = Path(temporary) / "must_not_exist"
            args.output_root = isolated_root
            with (
                mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", isolated_root),
                mock.patch.object(
                    runner,
                    "validate_canonical_expansion_authorization",
                    side_effect=runner.CompleteTargetReplicationError(
                        "fixture gate failure"
                    ),
                ),
                mock.patch.object(runner, "_run_process_lock") as lock,
                mock.patch.object(runner.r1, "run") as engine,
                self.assertRaisesRegex(
                    runner.CompleteTargetReplicationError,
                    "fixture gate failure",
                ),
            ):
                runner.run(args)
            self.assertFalse(isolated_root.exists())
            lock.assert_not_called()
            engine.assert_not_called()

    def test_runtime_patch_is_exception_safe_and_nonreentrant(self) -> None:
        args = self._parse()
        authorization = self._authorization()
        original_pilot = {
            "root": runner.pilot.DEFAULT_OUTPUT_ROOT,
            "schema": runner.pilot.TRAINING_SCHEMA,
            "require": runner.pilot._require_variant_args,
        }
        original_r1 = {
            "schema": runner.r1.TRAINING_SCHEMA,
            "identity": runner.r1._run_identity,
            "write": runner.r1._write_json,
        }
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with runner._replication_runtime(authorization):
                self.assertEqual(
                    runner.pilot.DEFAULT_OUTPUT_ROOT,
                    runner.DEFAULT_OUTPUT_ROOT,
                )
                self.assertEqual(runner.r1.TRAINING_SCHEMA, runner.TRAINING_SCHEMA)
                with self.assertRaisesRegex(
                    runner.CompleteTargetReplicationError, "already active"
                ):
                    with runner._replication_runtime(authorization):
                        self.fail("nested replication runtime must not enter")
                raise RuntimeError("fixture failure")
        self.assertEqual(runner.pilot.DEFAULT_OUTPUT_ROOT, original_pilot["root"])
        self.assertEqual(runner.pilot.TRAINING_SCHEMA, original_pilot["schema"])
        self.assertIs(runner.pilot._require_variant_args, original_pilot["require"])
        self.assertEqual(runner.r1.TRAINING_SCHEMA, original_r1["schema"])
        self.assertIs(runner.r1._run_identity, original_r1["identity"])
        self.assertIs(runner.r1._write_json, original_r1["write"])
        self.assertFalse(hasattr(runner._RUNTIME_STATE, "authorization"))

        # The exception path must also release both locks for a later entry.
        with runner._replication_runtime(authorization):
            identity = runner._replication_run_identity(
                args,
                self._contract(),
                {
                    "mode": "sample_level_fallback",
                    "sample_level_fallback_acknowledged": True,
                    "warning": "fixture acknowledgement",
                },
                train_count=runner.CANONICAL_IRSTD_TRAIN_COUNT,
                val_count=runner.CANONICAL_IRSTD_VAL_COUNT,
                smoke=False,
            )
        self.assertEqual(identity["schema"], runner.TRAINING_SCHEMA + "/run_identity")

    def test_run_lock_is_nonblocking_and_seed_isolated(self) -> None:
        runs_root = runner.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=runs_root, prefix="replication_lock_fixture_"
        ) as temporary:
            isolated_root = Path(temporary) / "replication"
            with mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", isolated_root):
                first = self._parse(runner.REPLICATION_RUN_SEEDS[0])
                second = self._parse(runner.REPLICATION_RUN_SEEDS[1])
                with runner._run_process_lock(first) as first_lock:
                    self.assertEqual(
                        first_lock.name,
                        ".complete_target_replication_v1.lock",
                    )
                    with self.assertRaisesRegex(
                        runner.CompleteTargetReplicationError,
                        "already locked",
                    ):
                        with runner._run_process_lock(first):
                            self.fail("same seed acquired twice")
                    with runner._run_process_lock(second) as second_lock:
                        self.assertNotEqual(first_lock.parent, second_lock.parent)

    def test_identity_binds_authorization_and_all_parent_sources(self) -> None:
        args = self._parse()
        authorization = self._authorization()
        identity = self._identity(args, authorization)
        self.assertEqual(identity["run_seed"], args.run_seed)
        self.assertEqual(identity["promotion_gate"], authorization)
        self.assertEqual(identity["experiment"]["schema"], runner.EXPERIMENT_SCHEMA)
        self.assertEqual(
            identity["experiment"]["single_variable_from_completed_pilot"],
            "runtime_seed",
        )
        determinism = identity["determinism_protocol"]
        self.assertEqual(determinism["schema"], runner.DETERMINISM_SCHEMA)
        self.assertEqual(
            determinism["frozen_parent_source_tree_sha256"],
            runner.FROZEN_PARENT_SOURCE_TREE_SHA256,
        )
        self.assertEqual(
            determinism["source_files"]["parent/variant_runner"]["sha256"],
            runner.FROZEN_PILOT_RUNNER_SHA256,
        )
        self.assertEqual(
            determinism["source_files"]["parent/complete_target_crop"][
                "sha256"
            ],
            runner.FROZEN_COMPLETE_TARGET_CROP_SHA256,
        )
        unhashed = dict(identity)
        observed = unhashed.pop("identity_sha256")
        self.assertEqual(observed, runner._canonical_sha256(unhashed))

        changed = copy.deepcopy(authorization)
        changed["canonical_result_artifact"]["sha256"] = "b" * 64
        changed_identity = self._identity(args, changed)
        self.assertNotEqual(
            identity["identity_sha256"], changed_identity["identity_sha256"]
        )

    def test_pilot_writers_emit_replication_schemas_and_authorization(self) -> None:
        args = self._parse()
        authorization = self._authorization()
        identity = self._identity(args, authorization)
        contract = runner.formal_artifact_contract(args.run_seed)
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            observed_json: list[tuple[Path, dict[str, object]]] = []
            observed_torch: list[tuple[Path, dict[str, object]]] = []
            runner.pilot._RUNTIME_STATE.run_identity = identity
            try:
                with (
                    runner._replication_runtime(authorization),
                    mock.patch.object(
                        runner.pilot,
                        "_R1_WRITE_JSON",
                        side_effect=lambda path, payload: observed_json.append(
                            (path, dict(payload))
                        ),
                    ),
                    mock.patch.object(
                        runner.pilot,
                        "_R1_ATOMIC_TORCH_SAVE",
                        side_effect=lambda path, payload: observed_torch.append(
                            (path, dict(payload))
                        ),
                    ),
                    mock.patch.object(
                        runner.pilot,
                        "_current_crop_audit",
                        return_value=({}, {}),
                    ),
                    mock.patch.object(
                        runner.pilot,
                        "_prospective_crop_audit_history",
                        return_value=[],
                    ),
                ):
                    # An unrelated strict JSON payload proves delegation while
                    # the checkpoint branch below proves replication stamping.
                    runner.pilot._variant_write_json(
                        temporary_root / "validation_history.json",
                        {
                            "schema": "fixture/non_artifact",
                            "data_role": "val",
                            "test_split_accessed": False,
                            "run_identity": identity,
                            "training_history": [{"epoch": 1}],
                            "validation_history": [],
                            "retention_frontier_epochs": [],
                            "candidate_artifacts": {},
                        },
                    )
                    runner.pilot._variant_atomic_torch_save(
                        temporary_root / "candidate.pth.tar",
                        {
                            "schema": runner.CANDIDATE_SCHEMA,
                            "run_identity": identity,
                            "epoch": 1,
                            "test_split_accessed": False,
                        },
                    )
            finally:
                if hasattr(runner.pilot._RUNTIME_STATE, "run_identity"):
                    delattr(runner.pilot._RUNTIME_STATE, "run_identity")
        self.assertEqual(observed_json[0][1]["schema"], "fixture/non_artifact")
        self.assertEqual(
            observed_torch[0][1]["schema"], runner.CANDIDATE_SCHEMA
        )
        self.assertEqual(
            observed_torch[0][1]["run_identity"]["promotion_gate"],
            authorization,
        )
        self.assertFalse(observed_torch[0][1]["public_test_supported"])
        self.assertFalse(observed_torch[0][1]["test_split_accessed"])
        self.assertTrue(
            contract["checkpoint_relative_path"].endswith("EviSIRST.pth.tar")
        )

    def test_completed_validator_is_cpu_only_and_fails_closed(self) -> None:
        seed = runner.REPLICATION_RUN_SEEDS[0]
        contract = runner.formal_artifact_contract(seed)
        # The formal directory does not exist yet.  Validation must fail before
        # touching any dataset, model builder, or device API.
        summary = runner.PROJECT_ROOT / contract["summary_relative_path"]
        self.assertFalse(summary.exists())
        with (
            mock.patch.object(
                runner.r1.legacy_train,
                "require_device",
                side_effect=AssertionError("CUDA must not be touched"),
            ) as require_device,
            mock.patch.object(
                runner.pilot,
                "build_datasets",
                side_effect=AssertionError("dataset must not be touched"),
            ) as datasets,
            self.assertRaises(runner.CompleteTargetReplicationError),
        ):
            runner.validate_existing_completed_run(seed)
        require_device.assert_not_called()
        datasets.assert_not_called()

    def test_public_test_markers_are_recursively_rejected(self) -> None:
        safe = {
            "test_split_accessed": False,
            "nested": {
                "test_index_opened": False,
                "public_test_supported": False,
            },
        }
        runner._require_no_public_test_access(safe)
        for key in (
            "test_split_accessed",
            "test_index_opened",
            "public_test_accessed",
            "public_test_allowed",
            "public_test_supported",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "public-test marker",
            ):
                runner._require_no_public_test_access({"nested": {key: True}})

    def test_state_contract_rejects_renamed_reshaped_and_retyped_tensors(self) -> None:
        authorization = runner.validate_canonical_expansion_authorization()
        pilot_artifact = authorization["canonical_payload"]["inputs"][
            "complete_target_v1"
        ]["final_checkpoint"]
        pilot_checkpoint = torch.load(
            runner.PROJECT_ROOT / pilot_artifact["relative_path"],
            map_location="cpu",
            weights_only=True,
        )
        state = pilot_checkpoint["state_dict"]
        self.assertEqual(
            runner.validate_state_dict_contract_for_gate(state, authorization),
            564,
        )
        first_key = next(iter(state))
        renamed = dict(state)
        renamed[first_key + ".tampered"] = renamed.pop(first_key)
        with self.assertRaisesRegex(
            runner.CompleteTargetReplicationError, "key names differ"
        ):
            runner.validate_state_dict_contract_for_gate(renamed, authorization)

        reshaped = dict(state)
        reshaped[first_key] = state[first_key].reshape(-1)
        if reshaped[first_key].shape == state[first_key].shape:
            reshaped[first_key] = state[first_key].unsqueeze(0)
        with self.assertRaisesRegex(
            runner.CompleteTargetReplicationError, "tensor contract differs"
        ):
            runner.validate_state_dict_contract_for_gate(reshaped, authorization)

        retyped = dict(state)
        replacement_dtype = (
            torch.float64
            if state[first_key].dtype != torch.float64
            else torch.float32
        )
        retyped[first_key] = state[first_key].to(replacement_dtype)
        with self.assertRaisesRegex(
            runner.CompleteTargetReplicationError, "tensor contract differs"
        ):
            runner.validate_state_dict_contract_for_gate(retyped, authorization)

    def test_retained_candidate_frontier_and_directory_are_strict(self) -> None:
        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            evidence = runner._validate_retained_candidates(
                **{
                    key: value
                    for key, value in fixture.items()
                    if key != "payloads"
                }
            )
            self.assertEqual(
                evidence,
                [
                    {
                        "epoch": 1,
                        "relative_path": "candidates/epoch_0001.pth.tar",
                        "sha256": fixture["candidate_artifacts"]["1"][
                            "file_sha256"
                        ],
                        "selected": False,
                    },
                    {
                        "epoch": 2,
                        "relative_path": "candidates/epoch_0002.pth.tar",
                        "sha256": fixture["candidate_artifacts"]["2"][
                            "file_sha256"
                        ],
                        "selected": True,
                    },
                ],
            )

            fixture["selection"]["retention_frontier_epochs"] = [2]
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "retention frontier differs",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

    def test_retained_candidate_missing_extra_and_hash_attacks_fail(self) -> None:
        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            candidate_dir = Path(fixture["run_dir"]) / "candidates"
            missing = candidate_dir / "epoch_0001.pth.tar"
            missing.unlink()
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "missing or extra files",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            candidate_dir = Path(fixture["run_dir"]) / "candidates"
            torch.save({}, candidate_dir / "epoch_0999.pth.tar")
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "missing or extra files",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            fixture["candidate_artifacts"]["1"]["file_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "SHA-256 differs",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

    def test_retained_candidate_payload_state_and_crop_attacks_fail(self) -> None:
        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            path = (
                Path(fixture["run_dir"])
                / "candidates"
                / "epoch_0001.pth.tar"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["unexpected"] = True
            torch.save(payload, path)
            self._refresh_candidate_binding(fixture, 1)
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "candidate keys differ",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

    def test_final_crop_audit_bundle_rejects_synchronized_state_summary_tamper(self) -> None:
        identity = {"train_count": 1}
        state, summary, history = self._crop_audit(2)
        observed = runner._validate_final_crop_audit_bundle(
            state=state,
            summary=summary,
            history=history,
            through_epoch=2,
            expected_epoch=2,
            identity=identity,
            label="search-run fixture",
        )
        self.assertEqual(observed["summary"], summary)

        # An attacker synchronizes state+summary but leaves the committed
        # history unchanged.  The validator must close all three views.
        tampered_state = copy.deepcopy(state)
        tampered_state["requested_counts"] = {
            "complete_target": 1,
            "uniform": 1,
        }
        tampered_state["realized_counts"] = {
            "complete_target": 1,
            "uniform": 1,
        }
        verifier = runner.complete_crop.CropAuditAccumulator()
        verifier.load_state_dict(tampered_state)
        tampered_summary = verifier.compute()
        with self.assertRaisesRegex(
            runner.CompleteTargetReplicationError,
            "state/summary/history differ",
        ):
            runner._validate_final_crop_audit_bundle(
                state=tampered_state,
                summary=tampered_summary,
                history=history,
                through_epoch=2,
                expected_epoch=2,
                identity=identity,
                label="search-run fixture",
            )

        # A top-level summary changed without its state is independently
        # rejected even when the history tail is changed to match it.
        tampered_history = copy.deepcopy(history)
        tampered_history[-1]["summary"] = tampered_summary
        with self.assertRaisesRegex(
            runner.CompleteTargetReplicationError,
            "bundle differs|state/summary/history differ",
        ):
            runner._validate_final_crop_audit_bundle(
                state=state,
                summary=tampered_summary,
                history=tampered_history,
                through_epoch=2,
                expected_epoch=2,
                identity=identity,
                label="search-run fixture",
            )

        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            path = (
                Path(fixture["run_dir"])
                / "candidates"
                / "epoch_0002.pth.tar"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            first_key = next(iter(payload["state_dict"]))
            payload["state_dict"][first_key] = (
                payload["state_dict"][first_key] + 1.0
            )
            torch.save(payload, path)
            self._refresh_candidate_binding(fixture, 2)
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "tensor values differ",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            path = (
                Path(fixture["run_dir"])
                / "candidates"
                / "epoch_0001.pth.tar"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["crop_audit_summary"]["fallback_rate"] = 0.5
            torch.save(payload, path)
            self._refresh_candidate_binding(fixture, 1)
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "crop audit differs",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

        with self._candidate_temporary_directory() as temporary:
            fixture = self._retained_candidate_fixture(Path(temporary))
            path = (
                Path(fixture["run_dir"])
                / "candidates"
                / "epoch_0002.pth.tar"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["crop_audit_state"]["requested_counts"] = {
                "complete_target": 1,
                "uniform": 1,
            }
            payload["crop_audit_state"]["realized_counts"] = {
                "complete_target": 1,
                "uniform": 1,
            }
            verifier = runner.complete_crop.CropAuditAccumulator()
            verifier.load_state_dict(payload["crop_audit_state"])
            payload["crop_audit_summary"] = verifier.compute()
            payload["crop_audit_history"][-1]["summary"] = verifier.compute()
            torch.save(payload, path)
            self._refresh_candidate_binding(fixture, 2)
            with self.assertRaisesRegex(
                runner.CompleteTargetReplicationError,
                "selected candidate crop audit differs",
            ):
                runner._validate_retained_candidates(
                    **{
                        key: value
                        for key, value in fixture.items()
                        if key != "payloads"
                    }
                )

    def test_one_epoch_fake_engine_executes_full_replication_schema_bridge(self) -> None:
        """Exercise the real R1 transaction engine through the new wrapper.

        Interruption/resume semantics themselves are inherited unchanged from
        the pilot functions and are covered by
        ``test_irstd_complete_target_runner.py``'s two-epoch bitwise resume
        test.  This test covers the new layer: authorization ordering, formal
        path, schema monkeypatch, identity binding, candidate/latest/final
        writers, crop audit, and summary publication.
        """

        runs_root = runner.PROJECT_ROOT / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        authorization = self._authorization()
        contract = self._contract()
        train_data = FakeFormalTrainDataset(contract)
        val_data = FakeFormalValDataset(contract)
        grouping = {
            "mode": "sample_level_fallback",
            "sample_level_fallback_acknowledged": True,
            "warning": "fixture acknowledgement",
        }
        args = self._parse()
        args.epochs = 1
        args.warmup_epochs = 0
        args.device = "cpu"
        with tempfile.TemporaryDirectory(
            dir=runs_root, prefix="replication_engine_fixture_"
        ) as temporary:
            isolated_root = Path(temporary) / "replication"
            args.output_root = isolated_root
            with (
                mock.patch.object(runner, "DEFAULT_OUTPUT_ROOT", isolated_root),
                mock.patch.object(
                    runner,
                    "_require_replication_args",
                    return_value=None,
                ),
                mock.patch.object(
                    runner,
                    "validate_canonical_expansion_authorization",
                    return_value=authorization,
                ),
                mock.patch.object(
                    runner.pilot,
                    "build_datasets",
                    return_value=(train_data, val_data, grouping),
                ),
                mock.patch.object(
                    runner.r1,
                    "initialize_evisirst",
                    return_value=(Fake564Model(), {"fixture": True}),
                ),
            ):
                checkpoint_path = runner.run(args)

            self.assertEqual(
                checkpoint_path,
                isolated_root
                / "formal"
                / runner.DATASET
                / runner.TARGET_MODE
                / f"run_seed_{args.run_seed}"
                / "EviSIRST.pth.tar",
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            latest = torch.load(
                checkpoint_path.parent / "last_training_state.pth.tar",
                map_location="cpu",
                weights_only=True,
            )
            summary = json.loads(
                (checkpoint_path.parent / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            candidate_paths = list(
                (checkpoint_path.parent / "candidates").glob("*.pth.tar")
            )
            self.assertEqual(len(candidate_paths), 1)
            candidate = torch.load(
                candidate_paths[0], map_location="cpu", weights_only=True
            )

            self.assertEqual(checkpoint["schema"], runner.CHECKPOINT_SCHEMA)
            self.assertEqual(latest["schema"], runner.TRAINING_SCHEMA)
            self.assertEqual(candidate["schema"], runner.CANDIDATE_SCHEMA)
            self.assertEqual(
                summary["schema"], runner.TRAINING_SCHEMA + "/summary"
            )
            identity = checkpoint["training"]
            self.assertEqual(
                identity["schema"], runner.TRAINING_SCHEMA + "/run_identity"
            )
            self.assertEqual(identity["promotion_gate"], authorization)
            self.assertEqual(latest["run_identity"], identity)
            self.assertEqual(candidate["run_identity"], identity)
            self.assertEqual(summary["run_identity"], identity)
            self.assertEqual(checkpoint["promotion_gate"], authorization)
            self.assertEqual(summary["promotion_gate"], authorization)
            self.assertFalse(checkpoint["public_test_supported"])
            self.assertFalse(summary["public_test_supported"])
            self.assertFalse(checkpoint["test_split_accessed"])
            self.assertFalse(latest["test_split_accessed"])
            self.assertFalse(candidate["test_split_accessed"])
            self.assertFalse(summary["test_split_accessed"])
            self.assertEqual(len(checkpoint["state_dict"]), 564)
            self.assertEqual(train_data.audit_observation_count, 640)
            self.assertEqual(
                checkpoint["search_run_crop_audit_through_epoch"], 1
            )
            self.assertEqual(
                summary["search_run_crop_audit_through_epoch"], 1
            )
            self.assertEqual(
                checkpoint["selected_checkpoint_crop_audit_through_epoch"],
                checkpoint["epoch"],
            )
            self.assertEqual(len(summary["training_history"]), 1)
            self.assertEqual(len(summary["validation_history"]), 1)


if __name__ == "__main__":
    unittest.main()
