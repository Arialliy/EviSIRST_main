from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


watcher = importlib.import_module(
    "tools.run_irstd_three_seed_replications_when_ready"
)


class FixedQueueContractTest(unittest.TestCase):
    def test_fixed_task_order_and_seeds(self) -> None:
        self.assertEqual(
            [(task.name, task.kind, task.run_seed) for task in watcher.TASKS],
            [
                ("b104", "baseline", 104728269),
                ("v104", "variant", 104728269),
                ("b262", "baseline", 262620274),
                ("v262", "variant", 262620274),
            ],
        )

    def test_baseline_command_is_fully_fixed_and_validation_only(self) -> None:
        command = watcher.training_argv(watcher.TASKS[0], resume=False)
        self.assertEqual(command[0], os.fspath(watcher.PYTHON_BIN))
        self.assertEqual(command[1], os.fspath(watcher.BASELINE_TRAINER))
        expected_pairs = {
            "--dataset": "IRSTD-1K",
            "--dataset-root": os.fspath(watcher.DATASET_ROOT),
            "--split-root": os.fspath(watcher.SPLIT_ROOT),
            "--output-root": os.fspath(watcher.BASELINE_OUTPUT_ROOT),
            "--target-mode": "binary",
            "--architecture-seed": "42",
            "--run-seed": "104728269",
            "--device": "cuda:0",
            "--epochs": "1000",
            "--batch-size": "16",
            "--workers": "0",
            "--base-lr": "0.001",
            "--min-lr": "0.00001",
            "--warmup-epochs": "10",
            "--val-interval": "1",
        }
        for option, value in expected_pairs.items():
            position = command.index(option)
            self.assertEqual(command[position + 1], value)
        self.assertNotIn("--resume", command)
        self.assertIn("--allow-sample-level-fallback", command)
        self._assert_no_test_tokens(command)

    def test_variant_and_diagnostic_commands_are_fixed(self) -> None:
        variant = watcher.training_argv(watcher.TASKS[3], resume=True)
        self.assertEqual(variant[-1], "--resume")
        self.assertIn("--allow-sample-level-fallback", variant)
        self.assertEqual(variant[1], os.fspath(watcher.VARIANT_TRAINER))
        self.assertEqual(variant[variant.index("--run-seed") + 1], "262620274")
        self.assertEqual(variant[variant.index("--epochs") + 1], "1000")
        diagnostic = watcher.diagnostic_argv(262620274)
        self.assertEqual(diagnostic[1], os.fspath(watcher.DIAGNOSTIC_RUNNER))
        self.assertEqual(diagnostic[diagnostic.index("--workers") + 1], "0")
        self._assert_no_test_tokens(variant)
        self._assert_no_test_tokens(diagnostic)
        self.assertEqual(
            watcher.gate_argv(),
            (os.fspath(watcher.PYTHON_BIN), os.fspath(watcher.THREE_SEED_GATE)),
        )

    def test_source_is_stdlib_only_no_torch_and_no_cli_configuration(self) -> None:
        source = Path(watcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertIn("pass_fds=", source)
        with self.assertRaisesRegex(
            watcher.ReplicationWatcherError, "accepts no arguments"
        ):
            watcher.main(["--seed", "1"])

    def _assert_no_test_tokens(self, command: tuple[str, ...]) -> None:
        lowered = tuple(token.lower() for token in command)
        self.assertNotIn("--test", lowered)
        self.assertNotIn("--test-root", lowered)
        self.assertNotIn("test", lowered)
        self.assertTrue(all("smoke" not in token for token in lowered))


class ExactProcessRecognitionTest(unittest.TestCase):
    def test_only_exact_argv_matches(self) -> None:
        expected = ("python", "/fixed/train.py", "--run-seed", "104728269")
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            for pid, argv in {
                "101": expected,
                "102": (*expected, "--extra"),
                "103": ("python", "/fixed/train.py.decoy", "--run-seed", "104728269"),
            }.items():
                directory = proc / pid
                directory.mkdir()
                (directory / "cmdline").write_bytes(
                    b"\0".join(token.encode() for token in argv) + b"\0"
                )
            self.assertEqual(
                watcher.exact_argv_processes(expected, proc_root=proc), (101,)
            )

    def test_timed_wrapper_and_python_child_are_both_deduplicated(self) -> None:
        argv = ("python", "train.py", "--run-seed", "104728269")
        with mock.patch.object(
            watcher,
            "exact_argv_processes",
            side_effect=[(22,), (21,)],
        ) as exact:
            self.assertEqual(watcher.work_processes(argv), (21, 22))
        self.assertEqual(exact.call_args_list[0].args[0], argv)
        self.assertEqual(exact.call_args_list[1].args[0], watcher.timed_argv(argv))


class TaskStateTest(unittest.TestCase):
    def _task(self, root: Path) -> watcher.Task:
        return watcher.Task("fixture", "baseline", 104728269, root / "train.py", root / "run")

    def test_fresh_resume_complete_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(root)
            with (
                mock.patch.object(watcher, "task_is_complete", return_value=False),
                mock.patch.object(watcher, "exact_argv_processes", return_value=()),
            ):
                self.assertEqual(watcher.task_state(task), "fresh")
                task.run_dir.mkdir()
                (task.run_dir / "last_training_state.pth.tar").write_bytes(b"resume")
                self.assertEqual(watcher.task_state(task), "resume")
                (task.run_dir / "summary.json").write_text("{}", encoding="utf-8")
                self.assertEqual(watcher.task_state(task), "conflict")
            with mock.patch.object(watcher, "task_is_complete", return_value=True):
                self.assertEqual(watcher.task_state(task), "complete")

    def test_exact_live_process_is_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = self._task(Path(temporary))
            with (
                mock.patch.object(watcher, "task_is_complete", return_value=False),
                mock.patch.object(watcher, "exact_argv_processes", return_value=(321,)),
            ):
                self.assertEqual(watcher.task_state(task), "conflict")


class GPUIdentityTest(unittest.TestCase):
    def test_full_uuid_index_bus_and_memory_are_parsed(self) -> None:
        parsed = watcher._parse_gpu_rows(
            "0, GPU-aaaa-bbbb, 00000000:17:00.0, 512\n"
            "2, GPU-cccc-dddd, 00000000:65:00.0, 10900\n"
        )
        self.assertEqual(parsed["GPU-aaaa-bbbb"].index, 0)
        self.assertEqual(parsed["GPU-aaaa-bbbb"].bus_id, "00000000:17:00.0")
        self.assertEqual(parsed["GPU-cccc-dddd"].memory_used_mib, 10900)

    def test_duplicate_or_changed_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(watcher.ReplicationWatcherError, "ambiguous"):
            watcher._parse_gpu_rows(
                "0, GPU-aaaa, 00000000:17:00.0, 1\n"
                "0, GPU-bbbb, 00000000:65:00.0, 1\n"
            )
        a = {"GPU-aaaa": watcher.GPU(0, "GPU-aaaa", "bus-a", 1)}
        b = {"GPU-aaaa": watcher.GPU(1, "GPU-aaaa", "bus-a", 1)}
        self.assertEqual(watcher._stable_intersection((a, b)), {})

    def test_idle_requires_low_memory_and_no_compute_application(self) -> None:
        query_gpu = mock.Mock(
            returncode=0,
            stdout=(
                "0, GPU-idle, 00000000:17:00.0, 100\n"
                "1, GPU-app, 00000000:18:00.0, 100\n"
                "2, GPU-memory, 00000000:19:00.0, 1025\n"
            ),
            stderr="",
        )
        query_apps = mock.Mock(
            returncode=0, stdout="GPU-app, 777\n", stderr=""
        )
        with mock.patch.object(
            watcher.subprocess, "run", side_effect=[query_gpu, query_apps]
        ):
            idle = watcher.sample_idle_gpus()
        self.assertEqual(tuple(idle), ("GPU-idle",))

    def test_four_samples_and_post_lock_identity_confirmation(self) -> None:
        gpu = watcher.GPU(0, "GPU-aaaa", "bus-a", 10)
        samples = [{gpu.uuid: gpu} for _ in range(4)]
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        sleeps: list[float] = []
        try:
            with (
                mock.patch.object(watcher, "sample_idle_gpus", side_effect=samples) as sample,
                mock.patch.object(watcher, "_try_lock", return_value=read_fd),
            ):
                claimed = watcher.claim_idle_gpus(sleep=sleeps.append)
            self.assertEqual(sample.call_count, 4)
            self.assertEqual(sleeps, [watcher.GPU_CONFIRM_SECONDS] * 2)
            self.assertEqual(claimed, [(gpu, read_fd)])
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    def test_launch_rechecks_idle_after_task_lock_and_releases_when_busy(self) -> None:
        gpu = watcher.GPU(0, "GPU-aaaa", "bus-a", 10)
        gpu_read, gpu_write = os.pipe()
        task_read, task_write = os.pipe()
        os.close(gpu_write)
        os.close(task_write)
        popen = mock.Mock()
        with (
            mock.patch.object(watcher, "_try_lock", return_value=task_read),
            mock.patch.object(watcher, "work_processes", return_value=()),
            mock.patch.object(watcher, "authorization_is_valid", return_value=True) as authorization,
            mock.patch.object(watcher, "sample_idle_gpus", return_value={}) as sample,
            mock.patch.object(watcher.subprocess, "Popen", popen),
        ):
            self.assertIsNone(
                watcher.launch_gpu_work("fixture", ("python", "train.py"), gpu, gpu_read)
            )
        sample.assert_called_once_with()
        authorization.assert_called_once_with()
        popen.assert_not_called()
        with self.assertRaises(OSError):
            os.fstat(gpu_read)
        with self.assertRaises(OSError):
            os.fstat(task_read)

    def test_launch_fails_closed_if_fresh_authorization_disappears(self) -> None:
        gpu = watcher.GPU(0, "GPU-aaaa", "bus-a", 10)
        gpu_read, gpu_write = os.pipe()
        task_read, task_write = os.pipe()
        os.close(gpu_write)
        os.close(task_write)
        with (
            mock.patch.object(watcher, "_try_lock", return_value=task_read),
            mock.patch.object(watcher, "work_processes", return_value=()),
            mock.patch.object(watcher, "authorization_is_valid", return_value=False),
            mock.patch.object(watcher, "sample_idle_gpus") as sample,
            mock.patch.object(watcher.subprocess, "Popen") as popen,
        ):
            self.assertIsNone(
                watcher.launch_gpu_work("fixture", ("python", "train.py"), gpu, gpu_read)
            )
        sample.assert_not_called()
        popen.assert_not_called()
        with self.assertRaises(OSError):
            os.fstat(gpu_read)
        with self.assertRaises(OSError):
            os.fstat(task_read)


class PreflightPathTest(unittest.TestCase):
    def test_existing_symlink_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            real = root / "real"
            real.mkdir()
            (runs / "redirect").symlink_to(real, target_is_directory=True)
            with (
                mock.patch.object(watcher, "PROJECT_ROOT", root),
            ):
                with self.assertRaisesRegex(
                    watcher.ReplicationWatcherError, "symlink component"
                ):
                    watcher._require_safe_directory_chain(runs / "redirect/leaf")

    def test_repository_runs_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            (root / "runs").symlink_to(real, target_is_directory=True)
            with mock.patch.object(watcher, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(
                    watcher.ReplicationWatcherError, "runs root"
                ):
                    watcher._require_safe_directory_chain(root / "runs/leaf")


class ValidatorProbeTest(unittest.TestCase):
    def test_real_completed_single_seed_authorization_probe_passes(self) -> None:
        self.assertTrue(watcher.authorization_is_valid())

    def test_product_validators_are_run_in_fixed_venv_subprocesses(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch.object(watcher.subprocess, "run", return_value=completed) as run:
            self.assertTrue(watcher.task_is_complete(watcher.TASKS[1]))
            self.assertTrue(watcher.diagnostic_is_complete(104728269))
            self.assertTrue(watcher.gate_is_complete())
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[0], os.fspath(watcher.PYTHON_BIN))
            self.assertEqual(command[1], "-c")
            self.assertEqual(call.kwargs["cwd"], watcher.PROJECT_ROOT)
            self.assertFalse(call.kwargs["check"])

    def test_cpu_gate_has_own_lock_dedup_and_fresh_authorization(self) -> None:
        gate_read, gate_write = os.pipe()
        os.close(gate_write)
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(watcher, "_try_lock", return_value=gate_read),
            mock.patch.object(watcher, "exact_argv_processes", return_value=()),
            mock.patch.object(watcher, "authorization_is_valid", return_value=True) as authorization,
            mock.patch.object(watcher.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual(watcher._run_gate_once(), 0)
        authorization.assert_called_once_with()
        self.assertEqual(run.call_args.args[0], list(watcher.gate_argv()))
        self.assertEqual(run.call_args.kwargs["pass_fds"], (gate_read,))
        with self.assertRaises(OSError):
            os.fstat(gate_read)


if __name__ == "__main__":
    unittest.main()
