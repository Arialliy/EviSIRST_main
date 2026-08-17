from __future__ import annotations

import fcntl
import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


watcher = importlib.import_module(
    "tools.run_irstd_complete_target_promotion_when_ready"
)


class FixedContractTest(unittest.TestCase):
    def test_five_fixed_inputs_and_cpu_only_gate_command(self) -> None:
        self.assertEqual(len(watcher.FIXED_INPUTS), 5)
        self.assertEqual(
            watcher.FIXED_INPUTS[-1], watcher.BASELINE_DIAGNOSTIC_OUTPUT
        )
        self.assertEqual(watcher.RESULT.name, "result.json")
        self.assertEqual(watcher.RUN_LOCK.name, "watcher.run.lock")

        source = Path(watcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("import torch", source)
        self.assertIn("/proc", source)

    def test_gate_launch_has_no_arguments_or_gpu_environment(self) -> None:
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "GPU-fixture",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "NVIDIA_VISIBLE_DEVICES": "all",
                },
            ),
            mock.patch.object(
                watcher.subprocess, "run", return_value=completed
            ) as run,
        ):
            self.assertEqual(watcher.run_gate_once(), 0)
        args, kwargs = run.call_args
        self.assertEqual(
            args[0],
            [os.fspath(watcher.PYTHON_BIN), os.fspath(watcher.PROMOTION_GATE)],
        )
        self.assertEqual(kwargs["cwd"], watcher.PROJECT_ROOT)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", kwargs["env"])
        self.assertNotIn("CUDA_DEVICE_ORDER", kwargs["env"])
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", kwargs["env"])
        self.assertFalse(kwargs["check"])

    def test_main_refuses_configuration(self) -> None:
        with self.assertRaisesRegex(
            watcher.PromotionWatcherError, "accepts no arguments"
        ):
            watcher.main(["--poll-seconds", "1"])


class ProcArgvRecognitionTest(unittest.TestCase):
    def test_exact_tokens_accept_formal_and_reject_smoke_or_substrings(self) -> None:
        spec = watcher.PROCESS_SPECS[1]
        cwd = watcher.PROJECT_ROOT
        formal = (
            os.fspath(watcher.PYTHON_BIN),
            os.fspath(watcher.VARIANT_TRAINER),
            "--run-seed=1446202191",
        )
        self.assertTrue(watcher.argv_matches_spec(formal, cwd=cwd, spec=spec))
        self.assertFalse(
            watcher.argv_matches_spec(
                (*formal, "--smoke-max-train-samples=1"), cwd=cwd, spec=spec
            )
        )
        self.assertFalse(
            watcher.argv_matches_spec(
                (
                    os.fspath(watcher.PYTHON_BIN),
                    os.fspath(watcher.VARIANT_TRAINER) + ".decoy",
                    "--run-seed",
                    "1446202191",
                ),
                cwd=cwd,
                spec=spec,
            )
        )

    def test_fake_proc_cmdline_and_relative_script_are_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            script = project / "trainer.py"
            script.write_text("# fixture\n", encoding="utf-8")
            proc_root = root / "proc"
            pid_directory = proc_root / "123"
            pid_directory.mkdir(parents=True)
            (pid_directory / "cmdline").write_bytes(
                b"python\x00trainer.py\x00--seed\x0042\x00"
            )
            (pid_directory / "cwd").symlink_to(project, target_is_directory=True)
            (pid_directory / "exe").symlink_to(Path(sys.executable).resolve())
            spec = watcher.ProcessSpec(
                name="fixture",
                script=script.resolve(),
                required_option_values=(("--seed", "42"),),
            )
            with mock.patch.object(watcher, "PROCESS_SPECS", (spec,)):
                matches = watcher.matching_processes(
                    proc_root=proc_root, exclude_pid=None
                )
            self.assertEqual(matches, {"fixture": (123,)})


class LockAndRetryTest(unittest.TestCase):
    def test_second_watcher_cannot_acquire_fixed_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "watcher.run.lock"
            with watcher.exclusive_watcher_lock(lock) as first:
                self.assertIsInstance(first, int)
                with watcher.exclusive_watcher_lock(lock) as second:
                    self.assertIsNone(second)
                fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_failed_gate_without_result_retries_with_bounded_backoff(self) -> None:
        empty_processes = {
            spec.name: () for spec in watcher.PROCESS_SPECS
        }
        sleeps: list[float] = []
        gate_runner = mock.Mock(side_effect=[9, 0])
        with (
            mock.patch.object(
                watcher,
                "_validate_result_path_state",
                side_effect=[False, False, False, False, False, True],
            ),
            mock.patch.object(
                watcher, "fixed_inputs_are_ready", return_value=True
            ),
            mock.patch.object(
                watcher, "matching_processes", return_value=empty_processes
            ),
        ):
            status = watcher.watch_forever(
                sleep=sleeps.append, gate_runner=gate_runner
            )
        self.assertEqual(status, 0)
        self.assertEqual(gate_runner.call_count, 2)
        self.assertEqual(sleeps, [watcher.INITIAL_FAILURE_BACKOFF_SECONDS])

    def test_existing_result_exits_without_launch(self) -> None:
        gate_runner = mock.Mock()
        with mock.patch.object(
            watcher, "_validate_result_path_state", return_value=True
        ):
            status = watcher.watch_forever(
                sleep=lambda _seconds: None, gate_runner=gate_runner
            )
        self.assertEqual(status, 0)
        gate_runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
