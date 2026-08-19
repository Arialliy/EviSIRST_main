from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_irstd_model_design_screen_when_idle as watcher


class BarrierPause(Exception):
    pass


class FakeProcess:
    def __init__(self, pid: int = 321) -> None:
        self.pid = pid
        self.returncode = None
        self.wait_calls: list[int] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout: int):
        self.wait_calls.append(timeout)
        self.returncode = -signal.SIGTERM
        return self.returncode


class ModelDesignScreenWatcherTest(unittest.TestCase):
    def test_fixed_six_tasks_and_exact_1000_epoch_argv(self) -> None:
        observed = [(task.kind, task.variant, task.run_seed) for task in watcher.TASKS]
        self.assertEqual(
            observed,
            [
                ("clean_baseline", "clean_evisirst", 104728269),
                ("model_design", "single_residual_v1", 42),
                ("model_design", "single_residual_v1", 1446202191),
                ("model_design", "psbfr_v1", 42),
                ("model_design", "psbfr_v1", 1446202191),
                ("model_design", "psbfr_v1", 104728269),
            ],
        )
        baseline = watcher.TASKS[0]
        argv = watcher.formal_training_argv(baseline, resume=True)
        self.assertEqual(Path(argv[1]).name, "train_validation_selected.py")
        self.assertEqual(argv.count("--resume"), 1)
        self.assertEqual(argv[argv.index("--epochs") + 1], "1000")
        self.assertEqual(argv[argv.index("--run-seed") + 1], "104728269")
        with self.assertRaisesRegex(watcher.ModelDesignWatcherError, "resume-only"):
            watcher.formal_training_argv(baseline, resume=False)

        for task in watcher.TASKS[1:]:
            fresh = watcher.formal_training_argv(task, resume=False)
            resumed = watcher.formal_training_argv(task, resume=True)
            self.assertEqual(Path(fresh[1]).name, "train_irstd_model_design_screen_v1.py")
            self.assertNotIn("--resume", fresh)
            self.assertEqual(resumed[-1], "--resume")
            self.assertEqual(fresh[fresh.index("--epochs") + 1], "1000")
            self.assertEqual(fresh[fresh.index("--variant") + 1], task.variant)
        self.assertNotIn("262620274", json.dumps(watcher.task_manifest()))

    def test_smoke_is_separate_and_formal_is_disabled_without_fixed_auth(self) -> None:
        self.assertIsNone(watcher.FORMAL_AUTHORIZATION_SHA256)
        self.assertFalse(watcher.authorization_is_valid())
        for task in watcher.TASKS[1:]:
            argv = watcher.smoke_training_argv(task)
            self.assertEqual(argv[argv.index("--epochs") + 1], "1")
            self.assertEqual(argv[argv.index("--device") + 1], "cpu")
            self.assertIn("--smoke-max-train-samples", argv)
        with self.assertRaisesRegex(watcher.ModelDesignWatcherError, "no smoke"):
            watcher.smoke_training_argv(watcher.TASKS[0])
        with self.assertRaisesRegex(watcher.ModelDesignWatcherError, "disabled"):
            watcher.main(["--formal"])

    def test_authorization_binds_file_hash_and_exact_task_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authorization.json"
            payload = {
                "schema": watcher.AUTHORIZATION_SCHEMA,
                "status": "AUTHORIZED",
                "task_manifest_sha256": watcher._canonical_sha256(
                    watcher.task_manifest()
                ),
                "public_test_allowed": False,
                "test_split_accessed": False,
            }
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(watcher, "AUTHORIZATION_PATH", path), mock.patch.object(
                watcher, "FORMAL_AUTHORIZATION_SHA256", digest
            ):
                self.assertTrue(watcher.authorization_is_valid())
                payload["task_manifest_sha256"] = "0" * 64
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertFalse(watcher.authorization_is_valid())

    def test_cleanup_barrier_runs_cleanup_then_blocks_and_forbids_501(self) -> None:
        events: list[str] = []

        def cleanup(*_args, **_kwargs):
            events.append("cleanup")
            return {"clean": True}

        def announce():
            events.append("announce")

        def pause():
            events.append("pause")
            raise BarrierPause

        wrapped = watcher.make_cleanup_barrier(
            cleanup, pause=pause, announce=announce
        )
        self.assertEqual(wrapped(completed_epoch=499), {"clean": True})
        self.assertEqual(events, ["cleanup"])
        events.clear()
        with self.assertRaises(BarrierPause):
            wrapped(completed_epoch=500)
        self.assertEqual(events, ["cleanup", "announce", "pause"])
        events.clear()
        with self.assertRaisesRegex(watcher.ModelDesignWatcherError, "501"):
            wrapped(completed_epoch=501)
        self.assertEqual(events, ["cleanup"])

    def test_full_uuid_idle_intersection_and_four_sample_shared_flock(self) -> None:
        rows = watcher._parse_gpu_rows(
            "0, GPU-aaaaaaaa-bbbb, 0000:01:00.0, 12\n"
            "1, GPU-cccccccc-dddd, 0000:02:00.0, 34\n"
        )
        self.assertEqual(rows["GPU-aaaaaaaa-bbbb"].bus_id, "0000:01:00.0")
        self.assertEqual(
            watcher._parse_compute_uuids("GPU-cccccccc-dddd, 9001\n"),
            {"GPU-cccccccc-dddd"},
        )
        gpu = rows["GPU-aaaaaaaa-bbbb"]
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            watcher, "GPU_LOCK_ROOT", Path(temporary)
        ), mock.patch.object(
            watcher,
            "sample_idle_gpus",
            side_effect=[{gpu.uuid: gpu}, {gpu.uuid: gpu}, {gpu.uuid: gpu}, {gpu.uuid: gpu}],
        ) as samples:
            claimed = watcher.claim_idle_gpus(sleep=lambda _seconds: None)
            self.assertEqual(len(claimed), 1)
            self.assertEqual(samples.call_count, 4)
            self.assertEqual(Path(temporary, f"{gpu.uuid}.lock").name, f"{gpu.uuid}.lock")
            os.close(claimed[0][1])

    def test_exact_argv_process_dedup_fixture(self) -> None:
        task = watcher.TASKS[2]
        expected = watcher.worker_argv(task, resume=False)
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            for pid, argv in (
                ("10", expected),
                ("11", watcher.timed_argv(expected)),
                ("12", (*expected, "--resume")),
            ):
                directory = proc / pid
                directory.mkdir()
                (directory / "cmdline").write_bytes(
                    b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
                )
            self.assertEqual(
                watcher.exact_argv_processes(expected, proc_root=proc), (10,)
            )
            self.assertEqual(
                watcher.task_processes(task, resume=False, proc_root=proc), (10, 11)
            )

    def test_fresh_resume_conflict_complete_state_fixture(self) -> None:
        task = watcher.TASKS[1]
        with mock.patch.object(watcher, "task_processes", return_value=()):
            with mock.patch.object(watcher, "_probe_task_state", return_value="fresh"):
                self.assertEqual(watcher.task_state(task), "fresh")
            with mock.patch.object(watcher, "_probe_task_state", return_value="resume"):
                self.assertEqual(watcher.task_state(task), "resume")
            with mock.patch.object(watcher, "_probe_task_state", return_value="conflict"):
                self.assertEqual(watcher.task_state(task), "conflict")
            with mock.patch.object(watcher, "_probe_task_state", return_value="ready"), mock.patch.object(
                watcher, "_completion_is_valid", return_value=True
            ):
                self.assertEqual(watcher.task_state(task), "complete")
        baseline = watcher.TASKS[0]
        with mock.patch.object(watcher, "task_processes", return_value=()), mock.patch.object(
            watcher, "_probe_task_state", return_value="fresh"
        ):
            self.assertEqual(watcher.task_state(baseline), "conflict")

    def test_ready_job_terms_independent_process_group_before_marker(self) -> None:
        task = watcher.TASKS[1]
        process = FakeProcess()
        job = watcher.RunningJob(task, process, watcher.GPU(0, "GPU-aa", "01", 0), 1, 2, mock.Mock())
        events: list[str] = []
        with mock.patch.object(
            watcher, "_probe_task_state", side_effect=["ready", "ready"]
        ), mock.patch.object(
            watcher.os, "killpg", side_effect=lambda pid, sig: events.append(f"term:{pid}:{sig}")
        ) as killpg, mock.patch.object(
            watcher, "_atomic_write_completion", side_effect=lambda _task: events.append("marker")
        ), mock.patch.object(watcher, "_completion_is_valid", return_value=True):
            self.assertTrue(watcher.stop_ready_job(job))
        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertEqual(events[-1], "marker")
        self.assertEqual(process.wait_calls, [watcher.TERM_TIMEOUT_SECONDS])

    def test_source_is_stdlib_only_and_worker_uses_post_cleanup_barrier(self) -> None:
        source = Path(watcher.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported_roots)
        self.assertIn("make_cleanup_barrier", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("signal.pause", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("epoch 501 is forbidden", source)
        self.assertNotIn("EviSIRSTTestDataset", source)


if __name__ == "__main__":
    unittest.main()
