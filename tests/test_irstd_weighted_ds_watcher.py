from __future__ import annotations

import ast
import fcntl
import importlib
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


watcher = importlib.import_module("tools.run_irstd_weighted_ds_when_unlocked")


class FixedContractTest(unittest.TestCase):
    def test_fixed_paths_and_command(self) -> None:
        self.assertEqual(
            watcher.WATCHER_LOCK,
            watcher.OUTPUT_ROOT / "watcher.run.lock",
        )
        self.assertEqual(
            watcher.CANONICAL_RESULT.name,
            "result.json",
        )
        self.assertEqual(
            watcher.FORMAL_RUN_DIR.relative_to(watcher.OUTPUT_ROOT).as_posix(),
            "formal/IRSTD-1K/binary/run_seed_1446202191",
        )
        expected = [
            os.fspath(watcher.TIME_BIN),
            "-v",
            os.fspath(watcher.PYTHON_BIN),
            os.fspath(watcher.TRAINER),
            "--dataset-root",
            os.fspath(watcher.DATASET_ROOT),
            "--split-root",
            os.fspath(watcher.SPLIT_ROOT),
            "--dataset",
            "IRSTD-1K",
            "--target-mode",
            "binary",
            "--architecture-seed",
            "42",
            "--run-seed",
            "1446202191",
            "--device",
            "cuda:0",
            "--epochs",
            "1000",
            "--warmup-epochs",
            "10",
            "--allow-sample-level-fallback",
        ]
        self.assertEqual(watcher.fixed_training_command(resume=False), expected)
        self.assertEqual(
            watcher.fixed_training_command(resume=True), [*expected, "--resume"]
        )

    def test_watcher_import_is_stdlib_only_and_validator_binds_runner(self) -> None:
        source = Path(watcher.__file__).read_text(encoding="utf-8")
        imported_roots: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported_roots)
        self.assertIn(
            "runner.validate_canonical_predecessor_failure()",
            watcher._VALIDATOR_PROGRAM,
        )
        self.assertIn("gate.validate_existing_result()", watcher._VALIDATOR_PROGRAM)
        self.assertIn('emit("PASS")', watcher._VALIDATOR_PROGRAM)
        self.assertIn('emit("FAIL")', watcher._VALIDATOR_PROGRAM)

    def test_strict_json_rejects_overflow_to_infinity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "overflow.json"
            for payload in ('{"nested":{"value":1e999}}\n', '{"value":NaN}\n'):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        watcher._strict_json_file(path)

    def test_exact_schemas_reject_extra_top_history_and_metric_fields(self) -> None:
        for label, required in (
            ("summary", watcher._SUMMARY_KEYS),
            ("training", watcher._TRAINING_RECORD_KEYS),
            ("validation", watcher._VALIDATION_RECORD_KEYS),
            ("metrics", watcher._VALIDATION_METRIC_KEYS),
            ("candidate metadata", watcher._CANDIDATE_METADATA_KEYS),
            ("candidate checkpoint", watcher._CANDIDATE_CHECKPOINT_KEYS),
            ("final checkpoint", watcher._FINAL_CHECKPOINT_KEYS),
        ):
            with self.subTest(label=label):
                fixture = dict.fromkeys(required)
                watcher._require_exact_keys(fixture, required, label=label)
                fixture["tampered"] = 1
                with self.assertRaisesRegex(ValueError, "keys differ"):
                    watcher._require_exact_keys(fixture, required, label=label)

    def test_candidate_schema_matches_real_weighted_runner_writer(self) -> None:
        import torch
        import train_irstd_weighted_ds_v1 as runner

        base_payload = {
            "schema": runner.CANDIDATE_SCHEMA,
            "model": "EviSIRST",
            "dataset": runner.DATASET,
            "epoch": 1,
            "run_identity": {"fixture": True},
            "validation_record": {"epoch": 1},
            "state_dict": {},
            "test_split_accessed": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate.pth.tar"
            runner._variant_atomic_torch_save(output, base_payload)
            observed = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(set(observed), set(watcher._CANDIDATE_CHECKPOINT_KEYS))
        self.assertNotIn("predecessor_gate_evidence", observed)

    def test_final_schema_matches_real_weighted_runner_builder(self) -> None:
        import train_irstd_weighted_ds_v1 as runner

        identity = {
            "identity_sha256": "a" * 64,
            "predecessor_gate_evidence": {"fixture": True},
        }
        payload = runner._variant_final_checkpoint_payload(
            args=SimpleNamespace(
                dataset=runner.DATASET,
                run_seed=runner.PAIRED_RUN_SEED,
                target_mode=runner.TARGET_MODE,
            ),
            state_dict={},
            normalization={"mean": 1.0, "std": 2.0},
            normalization_provenance={"fixture": True},
            identity=identity,
            split_provenance={
                "split_seed": 1,
                "manifest_sha256": "b" * 64,
                "data_tree_sha256": "c" * 64,
                "data_tree_verified": True,
            },
            selection_payload={
                "data_role": "val",
                "selection_is_optimistic": False,
                "optimistic": False,
                "selected_epoch": 1,
                "selection_provenance": {"fixture": True},
                "source_selection": "evisirst_v2_validation_split",
            },
            model_metadata={"fixture": True},
            smoke=False,
        )
        self.assertEqual(set(payload), set(watcher._FINAL_CHECKPOINT_KEYS))

    def test_real_fixed_executables_pass_main_preflight(self) -> None:
        self.assertTrue(watcher.PYTHON_BIN.is_symlink())
        self.assertEqual(
            watcher.PYTHON_BIN.resolve(strict=True), watcher.PYTHON_RESOLVED_TARGET
        )
        self.assertTrue(watcher.PYTHON_RESOLVED_TARGET.is_file())
        self.assertFalse(watcher.PYTHON_RESOLVED_TARGET.is_symlink())
        self.assertTrue(os.access(watcher.PYTHON_BIN, os.X_OK))
        for path in (watcher.TIME_BIN, watcher.NVIDIA_SMI_BIN, watcher.TRAINER):
            self.assertTrue(path.is_file(), path)
            self.assertFalse(path.is_symlink(), path)
        # This is exactly the dependency portion of main(), but has no lock,
        # run-directory, gate, GPU, or process side effect.
        watcher.validate_fixed_dependencies()

    def test_python_symlink_must_resolve_to_frozen_regular_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interpreter = root / "python-real"
            interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
            interpreter.chmod(0o700)
            launcher = root / "python"
            launcher.symlink_to(interpreter.name)
            self.assertEqual(
                watcher._require_fixed_dependency(
                    launcher,
                    label="fixture Python",
                    allow_symlink=True,
                    expected_resolved=interpreter,
                    require_executable=True,
                ),
                interpreter,
            )
            wrong = root / "wrong-python"
            wrong.write_text("#!/bin/sh\n", encoding="utf-8")
            wrong.chmod(0o700)
            with self.assertRaisesRegex(
                watcher.WeightedDSWatcherError, "fixture Python"
            ):
                watcher._require_fixed_dependency(
                    launcher,
                    label="fixture Python",
                    allow_symlink=True,
                    expected_resolved=wrong,
                    require_executable=True,
                )

    def test_main_refuses_configuration(self) -> None:
        with self.assertRaisesRegex(
            watcher.WeightedDSWatcherError, "accepts no arguments"
        ):
            watcher.main(["--poll-seconds", "1"])


class GateProbeTest(unittest.TestCase):
    def test_missing_result_waits_without_starting_validator(self) -> None:
        with (
            mock.patch.object(
                watcher, "_require_missing_or_regular", return_value=False
            ),
            mock.patch.object(watcher, "_run_validation_probe") as probe,
        ):
            self.assertIs(watcher.probe_canonical_decision(), watcher.GateDecision.WAIT)
        probe.assert_not_called()

    def test_only_exact_validated_tokens_authorize_a_decision(self) -> None:
        for token, expected in (
            ("PASS", watcher.GateDecision.PASS),
            ("FAIL", watcher.GateDecision.FAIL),
            ("pass", watcher.GateDecision.INVALID),
            (None, watcher.GateDecision.INVALID),
        ):
            with self.subTest(token=token):
                with (
                    mock.patch.object(
                        watcher, "_require_missing_or_regular", return_value=True
                    ),
                    mock.patch.object(
                        watcher, "_run_validation_probe", return_value=token
                    ),
                ):
                    self.assertIs(watcher.probe_canonical_decision(), expected)

    def test_validation_probe_hides_all_gpu_and_rejects_ambiguous_stdout(self) -> None:
        completed = subprocess.CompletedProcess(
            args=(), returncode=0, stdout="EVISIRST_WEIGHTED_DS_WATCHER_PROBE:FAIL\n", stderr=""
        )
        runner = mock.Mock(return_value=completed)
        self.assertEqual(
            watcher._run_validation_probe("gate", process_runner=runner), "FAIL"
        )
        args, kwargs = runner.call_args
        self.assertEqual(args[0][-1], "gate")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertTrue(kwargs["capture_output"])
        self.assertFalse(kwargs["check"])

        runner.return_value = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=(
                "EVISIRST_WEIGHTED_DS_WATCHER_PROBE:FAIL\n"
                "EVISIRST_WEIGHTED_DS_WATCHER_PROBE:PASS\n"
            ),
            stderr="",
        )
        self.assertIsNone(
            watcher._run_validation_probe("gate", process_runner=runner)
        )

    def test_pass_is_terminal_before_run_dir_or_gpu_work(self) -> None:
        gate_probe = mock.Mock(return_value=watcher.GateDecision.PASS)
        with (
            mock.patch.object(watcher, "inspect_run_mode") as inspect_mode,
            mock.patch.object(watcher, "list_gpu_identities") as list_gpus,
        ):
            self.assertEqual(
                watcher.watch_forever(
                    sleep=lambda _seconds: None, gate_probe=gate_probe
                ),
                0,
            )
        inspect_mode.assert_not_called()
        list_gpus.assert_not_called()

    def test_missing_or_invalid_gate_only_waits(self) -> None:
        stop = RuntimeError("test stop")
        probes = iter(
            [
                watcher.GateDecision.WAIT,
                watcher.GateDecision.INVALID,
                stop,
            ]
        )

        def gate_probe():
            value = next(probes)
            if isinstance(value, BaseException):
                raise value
            return value

        sleeps: list[float] = []
        with (
            mock.patch.object(watcher, "inspect_run_mode") as inspect_mode,
            mock.patch.object(watcher, "list_gpu_identities") as list_gpus,
            self.assertRaisesRegex(RuntimeError, "test stop"),
        ):
            watcher.watch_forever(sleep=sleeps.append, gate_probe=gate_probe)
        self.assertEqual(sleeps, [watcher.POLL_SECONDS, watcher.POLL_SECONDS])
        inspect_mode.assert_not_called()
        list_gpus.assert_not_called()


class RunStateTest(unittest.TestCase):
    def test_fresh_resume_complete_and_unsafe_partial_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "formal"
            summary = run_dir / "summary.json"
            last = run_dir / "last_training_state.pth.tar"
            with (
                mock.patch.object(watcher, "FORMAL_RUN_DIR", run_dir),
                mock.patch.object(watcher, "SUMMARY", summary),
                mock.patch.object(watcher, "LAST_STATE", last),
            ):
                self.assertIs(watcher.inspect_run_mode(), watcher.RunMode.FRESH)
                run_dir.mkdir()
                (run_dir / ".weighted_ds_v1.lock").touch()
                self.assertIs(watcher.inspect_run_mode(), watcher.RunMode.FRESH)
                last.touch()
                self.assertIs(watcher.inspect_run_mode(), watcher.RunMode.RESUME)
                last.unlink()
                (run_dir / "unexpected.bin").touch()
                self.assertIs(watcher.inspect_run_mode(), watcher.RunMode.WAIT)
                summary.touch()
                self.assertIs(watcher.inspect_run_mode(), watcher.RunMode.COMPLETE)

    def test_complete_summary_must_validate_before_exit(self) -> None:
        probes = iter(
            [
                watcher.GateDecision.FAIL,
                watcher.GateDecision.FAIL,
                RuntimeError("stop"),
            ]
        )

        def gate_probe():
            value = next(probes)
            if isinstance(value, BaseException):
                raise value
            return value

        validations = iter([False, True])
        sleeps: list[float] = []
        with (
            mock.patch.object(
                watcher, "inspect_run_mode", return_value=watcher.RunMode.COMPLETE
            ),
            mock.patch.object(
                watcher,
                "validate_complete_summary",
                side_effect=lambda: next(validations),
            ),
        ):
            self.assertEqual(
                watcher.watch_forever(sleep=sleeps.append, gate_probe=gate_probe),
                0,
            )
        self.assertEqual(sleeps, [watcher.POLL_SECONDS])


class ProcRecognitionTest(unittest.TestCase):
    def test_exact_formal_tokens_accept_and_smoke_or_decoys_reject(self) -> None:
        formal = (
            os.fspath(watcher.PYTHON_BIN),
            os.fspath(watcher.TRAINER),
            "--run-seed=1446202191",
            "--epochs",
            "1000",
        )
        self.assertTrue(
            watcher.argv_is_formal_weighted_trainer(
                formal, cwd=watcher.PROJECT_ROOT
            )
        )
        self.assertFalse(
            watcher.argv_is_formal_weighted_trainer(
                (*formal, "--smoke-max-train-samples=1"), cwd=watcher.PROJECT_ROOT
            )
        )
        self.assertFalse(
            watcher.argv_is_formal_weighted_trainer(
                (
                    os.fspath(watcher.PYTHON_BIN),
                    os.fspath(watcher.TRAINER) + ".decoy",
                    "--run-seed",
                    "1446202191",
                ),
                cwd=watcher.PROJECT_ROOT,
            )
        )
        self.assertFalse(
            watcher.argv_is_formal_weighted_trainer(
                (
                    os.fspath(watcher.PYTHON_BIN),
                    os.fspath(watcher.TRAINER),
                    "--run-seed",
                    "14462021910",
                ),
                cwd=watcher.PROJECT_ROOT,
            )
        )

    def test_nul_cmdline_and_relative_script_are_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            trainer = project / "trainer.py"
            trainer.write_text("# fixture\n", encoding="utf-8")
            proc = root / "proc"
            pid_dir = proc / "321"
            pid_dir.mkdir(parents=True)
            (pid_dir / "cmdline").write_bytes(
                b"python\x00trainer.py\x00--run-seed\x001446202191\x00"
            )
            (pid_dir / "cwd").symlink_to(project, target_is_directory=True)
            (pid_dir / "exe").symlink_to(Path(sys.executable).resolve())
            with mock.patch.object(watcher, "TRAINER", trainer.resolve()):
                self.assertEqual(
                    watcher.matching_formal_trainers(
                        proc_root=proc, exclude_pid=None
                    ),
                    (321,),
                )


class GPUIdentityTest(unittest.TestCase):
    GPU = watcher.GPUIdentity(
        index=2,
        uuid="GPU-12345678-1234-1234-1234-123456789abc",
        bus_id="00000000:65:00.0",
        memory_used_mib=200,
    )

    def _completed(self, stdout: str, returncode: int = 0):
        return subprocess.CompletedProcess((), returncode, stdout, "")

    def test_full_uuid_bus_index_rows_and_duplicates(self) -> None:
        row = f"2, {self.GPU.uuid}, 00000000:65:00.0, 200\n"
        with mock.patch.object(
            watcher, "_run_capture", return_value=self._completed(row)
        ):
            self.assertEqual(watcher.list_gpu_identities(), (self.GPU,))
        duplicate = row + row.replace("2,", "3,", 1)
        with mock.patch.object(
            watcher, "_run_capture", return_value=self._completed(duplicate)
        ):
            self.assertEqual(watcher.list_gpu_identities(), ())

    def test_idle_rechecks_full_identity_memory_and_compute_count(self) -> None:
        identity = f"2, {self.GPU.uuid}, 00000000:65:00.0, 200\n"
        with mock.patch.object(
            watcher,
            "_run_capture",
            side_effect=[self._completed(identity), self._completed("")],
        ):
            self.assertTrue(watcher.gpu_identity_is_idle(self.GPU))
        with mock.patch.object(
            watcher,
            "_run_capture",
            side_effect=[
                self._completed(identity),
                self._completed(self.GPU.uuid + "\n"),
            ],
        ):
            self.assertFalse(watcher.gpu_identity_is_idle(self.GPU))
        high_memory = identity.rsplit("200", 1)[0] + "1025\n"
        with mock.patch.object(
            watcher, "_run_capture", return_value=self._completed(high_memory)
        ):
            self.assertFalse(watcher.gpu_identity_is_idle(self.GPU))

    def test_gpu_lock_is_uuid_named_inheritable_and_performs_third_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_root = Path(temporary) / "runs/.gpu_locks"
            (Path(temporary) / "runs").mkdir()
            with (
                mock.patch.object(watcher, "PROJECT_ROOT", Path(temporary)),
                mock.patch.object(watcher, "GPU_LOCK_ROOT", lock_root),
                mock.patch.object(watcher, "gpu_identity_is_idle", return_value=True) as idle,
            ):
                with watcher.claimed_gpu_lock(self.GPU) as descriptor:
                    self.assertIsInstance(descriptor, int)
                    self.assertTrue(os.get_inheritable(descriptor))
                    self.assertTrue(
                        (lock_root / f"{self.GPU.uuid}.lock").is_file()
                    )
                    with watcher.claimed_gpu_lock(self.GPU) as second:
                        self.assertIsNone(second)
                self.assertEqual(idle.call_count, 1)

    def test_trainer_inherits_lock_and_full_uuid_mapping(self) -> None:
        completed = mock.Mock(returncode=7)
        with mock.patch.object(
            watcher.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(
                watcher.run_trainer_once(
                    resume=True, gpu=self.GPU, gpu_lock_fd=123
                ),
                7,
            )
        args, kwargs = run.call_args
        self.assertEqual(args[0], watcher.fixed_training_command(resume=True))
        self.assertEqual(kwargs["pass_fds"], (123,))
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], self.GPU.uuid)
        self.assertEqual(kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(kwargs["env"][watcher.GPU_LOCK_FD_ENV], "123")
        self.assertEqual(kwargs["env"][watcher.GPU_UUID_ENV], self.GPU.uuid)

    def test_two_observations_then_lock_and_launch_boundary_checks(self) -> None:
        sleeps: list[float] = []
        gate_probe = mock.Mock(
            side_effect=[
                watcher.GateDecision.FAIL,
                watcher.GateDecision.FAIL,
                watcher.GateDecision.FAIL,
            ]
        )
        modes = mock.Mock(
            side_effect=[
                watcher.RunMode.FRESH,
                watcher.RunMode.FRESH,
                watcher.RunMode.FRESH,
                watcher.RunMode.COMPLETE,
            ]
        )
        trainer = mock.Mock(return_value=0)
        with (
            mock.patch.object(watcher, "inspect_run_mode", modes),
            mock.patch.object(watcher, "validate_complete_summary", return_value=True),
            mock.patch.object(watcher, "matching_formal_trainers", return_value=()),
            mock.patch.object(watcher, "list_gpu_identities", return_value=(self.GPU,)),
            mock.patch.object(watcher, "gpu_identity_is_idle", return_value=True) as idle,
        ):
            self.assertEqual(
                watcher.watch_forever(
                    sleep=sleeps.append,
                    gate_probe=gate_probe,
                    trainer_runner=trainer,
                ),
                0,
            )
        # One idle check in each observation, one inside the claimed lock, and
        # one at the final launch boundary.
        self.assertEqual(idle.call_count, 4)
        self.assertEqual(sleeps, [watcher.POLL_SECONDS])
        trainer.assert_called_once()
        self.assertFalse(trainer.call_args.kwargs["resume"])


class WatcherLockTest(unittest.TestCase):
    def test_second_watcher_cannot_acquire_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            runs = project / "runs"
            runs.mkdir()
            lock = runs / "irstd_performance/weighted_ds_v1/watcher.run.lock"
            with mock.patch.object(watcher, "PROJECT_ROOT", project):
                with watcher.exclusive_watcher_lock(lock) as first:
                    self.assertIsInstance(first, int)
                    with watcher.exclusive_watcher_lock(lock) as second:
                        self.assertIsNone(second)
                    fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == "__main__":
    unittest.main()
