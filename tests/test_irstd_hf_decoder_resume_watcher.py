from __future__ import annotations

import ast
import importlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


watcher = importlib.import_module(
    "tools.run_irstd_hf_decoder_resume_when_three_seed_terminal"
)


class FixedContractTest(unittest.TestCase):
    def test_fixed_resume_command_and_paths(self) -> None:
        self.assertEqual(watcher.EXPECTED_RESUME_EPOCH, 780)
        self.assertEqual(watcher.FINAL_EPOCH, 1000)
        self.assertEqual(
            watcher.RUN_DIR.relative_to(watcher.PROJECT_ROOT).as_posix(),
            "runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/binary/"
            "run_seed_1446202191",
        )
        expected = (
            os.fspath(watcher.PYTHON_BIN), os.fspath(watcher.HF_TRAINER),
            "--dataset-root", os.fspath(watcher.DATASET_ROOT),
            "--split-root", os.fspath(watcher.SPLIT_ROOT),
            "--dataset", "IRSTD-1K", "--target-mode", "binary",
            "--architecture-seed", "42", "--run-seed", "1446202191",
            "--device", "cuda:0", "--epochs", "1000",
            "--warmup-epochs", "10", "--allow-sample-level-fallback",
            "--resume",
        )
        self.assertEqual(watcher.resume_argv(), expected)
        self.assertEqual(
            watcher.timed_argv(expected),
            (os.fspath(watcher.TIME_BIN), "-v", *expected),
        )
        self.assertEqual(expected.count("--resume"), 1)

    def test_import_is_stdlib_only_and_probe_is_validation_only(self) -> None:
        source = Path(watcher.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", roots)
        self.assertIn("gate.validate_existing_result()", watcher._PROBE_PROGRAM)
        self.assertNotIn("_load_resume_state(", watcher._PROBE_PROGRAM)
        self.assertNotIn("_finalize_completed(", watcher._PROBE_PROGRAM)
        self.assertIn('emit("RESUME")', watcher._PROBE_PROGRAM)
        self.assertIn('emit("FINALIZE")', watcher._PROBE_PROGRAM)
        self.assertIn('emit("COMPLETE")', watcher._PROBE_PROGRAM)
        self.assertIn('minimum_epoch=780, maximum_epoch=999', watcher._PROBE_PROGRAM)
        self.assertIn('minimum_epoch=1000, maximum_epoch=1000', watcher._PROBE_PROGRAM)
        self.assertIn('latest.get("schema") != r.TRAINING_SCHEMA', watcher._PROBE_PROGRAM)
        self.assertIn('set(latest) != latest_keys', watcher._PROBE_PROGRAM)

    def test_real_dependencies_and_current_transaction_probe(self) -> None:
        watcher.validate_fixed_dependencies()
        # Read-only and GPU-hidden.  It remains valid after later legal progress.
        self.assertIn(watcher.run_state(), {"RESUME", "FINALIZE", "COMPLETE"})

    def test_no_arguments(self) -> None:
        with self.assertRaisesRegex(watcher.HFResumeWatcherError, "no arguments"):
            watcher.main(["--device", "cuda:1"])


class ProbeTest(unittest.TestCase):
    def test_probe_hides_gpu_and_requires_one_exact_sentinel(self) -> None:
        good = subprocess.CompletedProcess(
            args=(), returncode=0,
            stdout=watcher.PROBE_PREFIX + "TERMINAL\n", stderr="",
        )
        runner = mock.Mock(return_value=good)
        with mock.patch.object(subprocess, "run", runner):
            self.assertEqual(watcher._run_probe("gate"), "TERMINAL")
        _args, kwargs = runner.call_args
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", kwargs["env"])
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            self.assertNotIn(name, kwargs["env"])
        runner.return_value = subprocess.CompletedProcess(
            args=(), returncode=0,
            stdout=(watcher.PROBE_PREFIX + "RESUME\n" + watcher.PROBE_PREFIX + "COMPLETE\n"),
            stderr="",
        )
        with mock.patch.object(subprocess, "run", runner):
            self.assertIsNone(watcher._run_probe("state"))

    def test_gate_and_state_accept_only_fixed_tokens(self) -> None:
        for token, terminal in (("TERMINAL", True), ("PASS", False), (None, False)):
            with self.subTest(token=token), mock.patch.object(
                watcher, "_run_probe", return_value=token
            ):
                self.assertIs(watcher.terminal_gate_is_valid(), terminal)
        for token, expected in (
            ("RESUME", "RESUME"), ("FINALIZE", "FINALIZE"),
            ("COMPLETE", "COMPLETE"), ("WAIT", "CONFLICT"), (None, "CONFLICT"),
        ):
            with self.subTest(token=token), mock.patch.object(
                watcher, "_run_probe", return_value=token
            ):
                self.assertEqual(watcher.run_state(), expected)


class ProcessAndGPUTest(unittest.TestCase):
    def _fake_proc(self, root: Path, pid: int, argv: tuple[str, ...]) -> None:
        directory = root / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(
            b"\0".join(part.encode("utf-8") for part in argv) + b"\0"
        )

    def test_exact_process_recognition_rejects_decoys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fake_proc(root, 10, watcher.resume_argv())
            self._fake_proc(root, 11, watcher.timed_argv(watcher.resume_argv()))
            self._fake_proc(root, 12, (*watcher.resume_argv(), "--smoke-max-train-samples", "1"))
            self.assertEqual(
                watcher.exact_argv_processes(watcher.resume_argv(), proc_root=root),
                (10,),
            )
            with mock.patch.object(watcher, "PROC_ROOT", root):
                self.assertEqual(watcher.work_processes(), (10, 11))

    def test_gpu_rows_require_unique_full_identity(self) -> None:
        parsed = watcher._parse_gpu_rows(
            "0, GPU-aaaa-bbbb, 00000000:16:00.0, 20\n"
            "1, GPU-cccc-dddd, 00000000:27:00.0, 30\n"
        )
        self.assertEqual(parsed["GPU-aaaa-bbbb"].index, 0)
        self.assertEqual(parsed["GPU-aaaa-bbbb"].bus_id, "00000000:16:00.0")
        with self.assertRaises(watcher.HFResumeWatcherError):
            watcher._parse_gpu_rows(
                "0, GPU-aaaa-bbbb, 00000000:16:00.0, 20\n"
                "0, GPU-cccc-dddd, 00000000:27:00.0, 30\n"
            )
        with self.assertRaises(watcher.HFResumeWatcherError):
            watcher._parse_compute_uuids("not-a-uuid, 123\n")

    def test_log_open_refuses_symlink_and_accepts_regular(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.log"
            target.write_bytes(b"untouched")
            link = root / "resume.log"
            link.symlink_to(target)
            with mock.patch.object(watcher, "WATCH_ROOT", root), mock.patch.object(
                watcher, "LOG_PATH", link
            ), self.assertRaises(watcher.HFResumeWatcherError):
                watcher._open_log()
            self.assertEqual(target.read_bytes(), b"untouched")
            link.unlink()
            with mock.patch.object(watcher, "WATCH_ROOT", root), mock.patch.object(
                watcher, "LOG_PATH", link
            ):
                handle = watcher._open_log()
                handle.write(b"ok"); handle.close()
            self.assertEqual(link.read_bytes(), b"ok")

    def test_claim_requires_two_stable_samples_and_shared_lock(self) -> None:
        first = watcher.GPU(0, "GPU-aaaa", "bus0", 10)
        changed = watcher.GPU(1, "GPU-aaaa", "bus1", 10)
        with mock.patch.object(
            watcher, "sample_idle_gpus", side_effect=[{first.uuid: first}, {changed.uuid: changed}]
        ), mock.patch.object(watcher, "_try_lock") as lock:
            self.assertIsNone(watcher.claim_one_idle_gpu(sleep=lambda _seconds: None))
        lock.assert_not_called()
        with mock.patch.object(
            watcher, "sample_idle_gpus", side_effect=[
                {first.uuid: first}, {first.uuid: first}, {first.uuid: first}
            ]
        ), mock.patch.object(watcher, "_try_lock", return_value=91) as lock:
            self.assertEqual(
                watcher.claim_one_idle_gpu(sleep=lambda _seconds: None),
                (first, 91),
            )
        self.assertEqual(
            lock.call_args.args[0], watcher.GPU_LOCK_ROOT / "GPU-aaaa.lock"
        )
        self.assertTrue(lock.call_args.kwargs["inheritable"])

    def test_claim_releases_lock_if_lock_inside_third_sample_changes(self) -> None:
        first = watcher.GPU(0, "GPU-aaaa", "bus0", 10)
        changed = watcher.GPU(1, "GPU-aaaa", "bus1", 10)
        with mock.patch.object(
            watcher, "sample_idle_gpus", side_effect=[
                {first.uuid: first}, {first.uuid: first}, {changed.uuid: changed}
            ]
        ), mock.patch.object(watcher, "_try_lock", return_value=91), mock.patch.object(
            os, "close"
        ) as close:
            self.assertIsNone(watcher.claim_one_idle_gpu(sleep=lambda _seconds: None))
        close.assert_called_once_with(91)


class StateMachineTest(unittest.TestCase):
    def test_gate_and_existing_process_precede_artifact_or_gpu_probe(self) -> None:
        stopped = RuntimeError("stop")
        with mock.patch.object(
            watcher, "terminal_gate_is_valid", side_effect=[False, stopped]
        ), mock.patch.object(watcher, "work_processes") as processes, mock.patch.object(
            watcher, "run_state"
        ) as state, mock.patch.object(watcher, "claim_one_idle_gpu") as claim, self.assertRaisesRegex(
            RuntimeError, "stop"
        ):
            watcher.supervise(sleep=lambda _seconds: None)
        processes.assert_not_called(); state.assert_not_called(); claim.assert_not_called()

        with mock.patch.object(watcher, "terminal_gate_is_valid", return_value=True), mock.patch.object(
            watcher, "work_processes", side_effect=[(123,), stopped]
        ), mock.patch.object(watcher, "run_state") as state, mock.patch.object(
            watcher, "claim_one_idle_gpu"
        ) as claim, self.assertRaisesRegex(RuntimeError, "stop"):
            watcher.supervise(sleep=lambda _seconds: None)
        state.assert_not_called(); claim.assert_not_called()

    def test_complete_exits_and_conflict_fails_closed_without_gpu(self) -> None:
        with mock.patch.object(watcher, "terminal_gate_is_valid", return_value=True), mock.patch.object(
            watcher, "work_processes", return_value=()
        ), mock.patch.object(watcher, "run_state", return_value="COMPLETE"), mock.patch.object(
            watcher, "claim_one_idle_gpu"
        ) as claim:
            self.assertEqual(watcher.supervise(sleep=lambda _seconds: None), 0)
        claim.assert_not_called()

    def test_nonzero_worker_reenters_strict_state_machine(self) -> None:
        gpu = watcher.GPU(0, "GPU-fixed", "bus", 10)
        process = mock.Mock(returncode=9)
        process.poll.return_value = 9
        log = mock.Mock()
        states = iter(["RESUME", "CONFLICT"])
        sleeps: list[float] = []
        with mock.patch.object(watcher, "terminal_gate_is_valid", return_value=True), mock.patch.object(
            watcher, "work_processes", return_value=()
        ), mock.patch.object(watcher, "run_state", side_effect=lambda: next(states)), mock.patch.object(
            watcher, "claim_one_idle_gpu", return_value=(gpu, 91)
        ), mock.patch.object(watcher, "launch_once", return_value=(process, 92, log)), mock.patch.object(
            os, "close"
        ), self.assertRaisesRegex(watcher.HFResumeWatcherError, "not safely"):
            watcher.supervise(sleep=sleeps.append)
        self.assertEqual(sleeps, [watcher.INITIAL_RETRY_SECONDS])
        log.close.assert_called_once_with()
        with mock.patch.object(watcher, "terminal_gate_is_valid", return_value=True), mock.patch.object(
            watcher, "work_processes", return_value=()
        ), mock.patch.object(watcher, "run_state", return_value="CONFLICT"), mock.patch.object(
            watcher, "claim_one_idle_gpu"
        ) as claim, self.assertRaisesRegex(watcher.HFResumeWatcherError, "not safely"):
            watcher.supervise(sleep=lambda _seconds: None)
        claim.assert_not_called()

    def test_launch_boundary_rechecks_gate_process_state_and_physical_gpu(self) -> None:
        gpu = watcher.GPU(2, "GPU-fixed", "bus-fixed", 10)
        popen = mock.Mock()
        popen.pid = 777
        fake_log = mock.Mock()
        order: list[str] = []
        with mock.patch.object(watcher, "_try_lock", return_value=92), mock.patch.object(
            watcher, "terminal_gate_is_valid", side_effect=lambda: order.append("gate") or True
        ), mock.patch.object(
            watcher, "work_processes", side_effect=lambda: order.append("process") or ()
        ), mock.patch.object(
            watcher, "run_state", side_effect=lambda: order.append("state") or "RESUME"
        ), mock.patch.object(
            watcher, "sample_idle_gpus", side_effect=lambda: order.append("idle") or {gpu.uuid: gpu}
        ), mock.patch.object(watcher, "_open_log", return_value=fake_log), mock.patch.object(
            subprocess, "Popen", return_value=popen
        ) as spawn:
            result = watcher.launch_once(gpu, 91)
        self.assertEqual(order, ["gate", "process", "state", "idle"])
        self.assertEqual(result, (popen, 92, fake_log))
        kwargs = spawn.call_args.kwargs
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], gpu.uuid)
        self.assertEqual(kwargs["pass_fds"], (91, 92))
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(tuple(spawn.call_args.args[0]), watcher.timed_argv(watcher.resume_argv()))


if __name__ == "__main__":
    unittest.main()
