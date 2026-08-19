from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import run_irstd_ab_model_screen_when_idle as watcher


class BarrierPause(Exception):
    pass


class FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = None
        self.wait_calls: list[int] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout: int):
        self.wait_calls.append(timeout)
        self.returncode = -signal.SIGTERM
        return self.returncode


class ABModelScreenWatcherTest(unittest.TestCase):
    def test_exact_seven_tasks_and_two_independent_candidate_arms(self) -> None:
        observed = [
            (task.kind, task.variant, task.run_seed) for task in watcher.TASKS
        ]
        self.assertEqual(
            observed,
            [
                ("clean", "clean_evisirst", 104728269),
                ("psbfr", "psbfr_v1", 42),
                ("psbfr", "psbfr_v1", 1446202191),
                ("psbfr", "psbfr_v1", 104728269),
                ("cp_hf_s2", "cp_hf_s2_v1", 42),
                ("cp_hf_s2", "cp_hf_s2_v1", 1446202191),
                ("cp_hf_s2", "cp_hf_s2_v1", 104728269),
            ],
        )
        clean = watcher.TASKS[0]
        self.assertEqual(clean.minimum_resume_epoch, 464)
        self.assertFalse(clean.fresh_allowed)
        with self.assertRaisesRegex(
            watcher.ABModelScreenWatcherError, "resume-only"
        ):
            watcher.formal_training_argv(clean, resume=False)
        clean_argv = watcher.formal_training_argv(clean, resume=True)
        self.assertEqual(Path(clean_argv[1]).name, "train_validation_selected.py")
        self.assertEqual(clean_argv[-1], "--resume")

        for task in watcher.TASKS[1:]:
            fresh = watcher.formal_training_argv(task, resume=False)
            resumed = watcher.formal_training_argv(task, resume=True)
            self.assertNotIn("--resume", fresh)
            self.assertEqual(resumed[-1], "--resume")
            self.assertEqual(fresh[fresh.index("--epochs") + 1], "1000")
            self.assertEqual(fresh[fresh.index("--run-seed") + 1], str(task.run_seed))
            if task.kind == "psbfr":
                self.assertEqual(
                    Path(fresh[1]).name, "train_irstd_model_design_screen_v1.py"
                )
                self.assertEqual(fresh[fresh.index("--variant") + 1], "psbfr_v1")
            else:
                self.assertEqual(
                    Path(fresh[1]).name,
                    "train_irstd_cp_hf_s2_legacy_screen_v1.py",
                )
                self.assertNotIn("--variant", fresh)
                self.assertNotIn("--split-root", fresh)

        manifest = watcher.task_manifest()
        self.assertEqual(
            watcher.STAGE1_TASK_NAMES, frozenset({"psbfr_s42", "cp_hf_s2_s42"})
        )
        self.assertEqual(manifest["launcher_stage"], "S1_seed42_only")
        self.assertFalse(manifest["s2_unlock_implemented"])
        authorized = {
            entry["name"]
            for entry in manifest["tasks"]
            if entry["launch_authorized_by_this_watcher"]
        }
        self.assertEqual(authorized, set(watcher.STAGE1_TASK_NAMES))
        self.assertTrue(manifest["candidate_decisions_are_independent"])
        self.assertFalse(manifest["candidate_ranking_or_winner_selection"])
        self.assertFalse(manifest["lockbox_accessed"])
        self.assertFalse(manifest["public_test_allowed"])
        self.assertFalse(manifest["test_split_accessed"])

        for task in watcher.TASKS:
            if task.name not in watcher.STAGE1_TASK_NAMES:
                with self.assertRaisesRegex(
                    watcher.ABModelScreenWatcherError, "locked"
                ):
                    watcher.worker_argv(task, resume=not task.fresh_allowed)

    def test_source_allowlist_is_exact_and_mismatch_fails_closed(self) -> None:
        self.assertEqual(len(watcher.FROZEN_SOURCE_SHA256), 56)
        self.assertTrue(
            all(
                re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in watcher.FROZEN_SOURCE_SHA256.values()
            )
        )
        self.assertTrue(watcher.frozen_sources_are_valid())
        self.assertEqual(
            watcher.FROZEN_SOURCE_SHA256[
                "train_irstd_cp_hf_s2_legacy_screen_v1.py"
            ],
            "0383b5f2769b510b7040639a8cfdd0209cc91b901062b2f3d9eb897eb1efedab",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runner.py"
            source.write_text("frozen\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with mock.patch.object(watcher, "PROJECT_ROOT", root), mock.patch.object(
                watcher, "FROZEN_SOURCE_SHA256", {"runner.py": digest}
            ):
                watcher.assert_frozen_sources()
                source.write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    watcher.ABModelScreenWatcherError, "SHA-256 differs"
                ):
                    watcher.assert_frozen_sources()

    def test_stage1_clean_control_artifact_hashes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            latest = run / "last_training_state.pth.tar"
            history = run / "validation_history.json"
            latest.write_bytes(b"latest")
            history.write_bytes(b"history")
            expected = {
                latest.name: hashlib.sha256(latest.read_bytes()).hexdigest(),
                history.name: hashlib.sha256(history.read_bytes()).hexdigest(),
            }
            with mock.patch.object(
                watcher, "STAGE1_CLEAN_CONTROL_RUN_DIR", run
            ), mock.patch.object(
                watcher, "FROZEN_STAGE1_CLEAN_CONTROL_SHA256", expected
            ):
                watcher.assert_frozen_stage1_clean_control_artifacts()
                history.write_bytes(b"changed")
                with self.assertRaisesRegex(
                    watcher.ABModelScreenWatcherError, "SHA differs"
                ):
                    watcher.assert_frozen_stage1_clean_control_artifacts()

            latest.write_bytes(b"locked-latest")
            history.write_bytes(b"locked-history")
            locked = watcher.Task(
                "clean_r1_s104728269_to500",
                "clean",
                "clean_evisirst",
                104728269,
                run,
                464,
                False,
            )
            locked_hashes = {
                latest.name: hashlib.sha256(latest.read_bytes()).hexdigest(),
                history.name: hashlib.sha256(history.read_bytes()).hexdigest(),
            }
            with mock.patch.object(watcher, "PROJECT_ROOT", run.parent), mock.patch.object(
                watcher, "TASKS", (locked,)
            ), mock.patch.object(
                watcher, "FROZEN_STAGE2_CLEAN_RESUME_SHA256", locked_hashes
            ), mock.patch.object(
                watcher, "FROZEN_STAGE2_CLEAN_RESUME_EPOCH", 464
            ), mock.patch.object(
                watcher, "_assert_safe_task_run_chain"
            ):
                watcher.assert_frozen_stage2_clean_resume_artifacts()
                latest.write_bytes(b"externally-advanced")
                with self.assertRaisesRegex(
                    watcher.ABModelScreenWatcherError, "SHA differs"
                ):
                    watcher.assert_frozen_stage2_clean_resume_artifacts()

    def test_authorization_binds_manifest_watcher_and_source_allowlist(self) -> None:
        self.assertRegex(watcher.FORMAL_AUTHORIZATION_SHA256 or "", r"^[0-9a-f]{64}$")
        self.assertTrue(watcher.authorization_is_valid())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authorization.json"
            watcher_sha = "a" * 64
            payload = {
                "schema": watcher.AUTHORIZATION_SCHEMA,
                "status": "AUTHORIZED",
                "task_manifest_sha256": watcher._canonical_sha256(
                    watcher.task_manifest()
                ),
                "watcher_source_sha256": watcher_sha,
                "source_allowlist_sha256": watcher.frozen_source_allowlist_sha256(),
                "lockbox_accessed": False,
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
            ), mock.patch.object(
                watcher, "assert_frozen_sources"
            ), mock.patch.object(
                watcher, "assert_frozen_stage1_clean_control_artifacts"
            ), mock.patch.object(
                watcher, "assert_frozen_stage2_clean_resume_artifacts"
            ), mock.patch.object(
                watcher, "_watcher_source_sha256", return_value=watcher_sha
            ):
                self.assertTrue(watcher.authorization_is_valid())
                payload["watcher_source_sha256"] = "b" * 64
                path.write_text(json.dumps(payload), encoding="utf-8")
                changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with mock.patch.object(
                    watcher, "FORMAL_AUTHORIZATION_SHA256", changed_digest
                ):
                    self.assertFalse(watcher.authorization_is_valid())

    def test_watcher_hash_normalizes_only_authorization_literal(self) -> None:
        observed = watcher._watcher_source_sha256()
        self.assertRegex(observed, r"^[0-9a-f]{64}$")
        source = Path(watcher.__file__).read_text(encoding="utf-8")
        declarations = re.findall(
            r"^FORMAL_AUTHORIZATION_SHA256: str \| None = .+$",
            source,
            flags=re.MULTILINE,
        )
        self.assertEqual(len(declarations), 1)

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
        with self.assertRaisesRegex(watcher.ABModelScreenWatcherError, "501"):
            wrapped(completed_epoch=501)
        self.assertEqual(events, ["cleanup"])

    def test_full_uuid_pci_four_samples_and_shared_gpu_flock(self) -> None:
        rows = watcher._parse_gpu_rows(
            "0, GPU-aaaaaaaa-bbbb, 0000:01:00.0, 12\n"
            "1, GPU-cccccccc-dddd, 0000:02:00.0, 34\n"
        )
        gpu = rows["GPU-aaaaaaaa-bbbb"]
        self.assertEqual(gpu.bus_id, "0000:01:00.0")
        self.assertEqual(
            watcher._parse_compute_uuids("GPU-cccccccc-dddd, 9001\n"),
            {"GPU-cccccccc-dddd"},
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            watcher, "GPU_LOCK_ROOT", Path(temporary)
        ), mock.patch.object(
            watcher,
            "sample_idle_gpus",
            side_effect=[
                {gpu.uuid: gpu},
                {gpu.uuid: gpu},
                {gpu.uuid: gpu},
                {gpu.uuid: gpu},
            ],
        ) as samples:
            claimed = watcher.claim_idle_gpus(sleep=lambda _seconds: None)
            self.assertEqual(samples.call_count, 4)
            self.assertEqual(len(claimed), 1)
            self.assertTrue(Path(temporary, f"{gpu.uuid}.lock").exists())
            os.close(claimed[0][1])

    def test_exact_argv_detects_worker_and_direct_runner_without_prefix_match(self) -> None:
        task = watcher.TASKS[4]
        worker = watcher.worker_argv(task, resume=False)
        trainer = watcher.formal_training_argv(task, resume=False)
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            fixtures = (
                ("10", worker),
                ("11", watcher.timed_argv(worker)),
                ("12", trainer),
                ("13", watcher.timed_argv(trainer)),
                ("14", (*worker, "--extra")),
            )
            for pid, argv in fixtures:
                directory = proc / pid
                directory.mkdir()
                (directory / "cmdline").write_bytes(
                    b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
                )
            self.assertEqual(
                watcher.task_processes(task, resume=False, proc_root=proc),
                (10, 11, 12, 13),
            )

    def test_fresh_resume_conflict_and_complete_states_fail_closed(self) -> None:
        candidate = watcher.TASKS[1]
        with mock.patch.object(watcher, "task_processes", return_value=()):
            with mock.patch.object(watcher, "_probe_task_state", return_value="fresh"):
                self.assertEqual(watcher.task_state(candidate), "fresh")
            with mock.patch.object(watcher, "_probe_task_state", return_value="resume"):
                self.assertEqual(watcher.task_state(candidate), "resume")
            with mock.patch.object(
                watcher, "_probe_task_state", return_value="conflict"
            ):
                self.assertEqual(watcher.task_state(candidate), "conflict")
            with tempfile.TemporaryDirectory() as temporary:
                marker = Path(temporary) / "complete.json"
                marker.write_text("{}\n", encoding="utf-8")
                with mock.patch.object(
                    watcher, "_probe_task_state", return_value="ready"
                ), mock.patch.object(
                    watcher, "_completion_path", return_value=marker
                ), mock.patch.object(
                    watcher, "_completion_is_valid", return_value=True
                ):
                    self.assertEqual(watcher.task_state(candidate), "complete")
            with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                watcher, "_probe_task_state", return_value="ready"
            ), mock.patch.object(
                watcher,
                "_completion_path",
                return_value=Path(temporary) / "missing.json",
            ):
                self.assertEqual(watcher.task_state(candidate), "resume")
            with tempfile.TemporaryDirectory() as temporary:
                invalid = Path(temporary) / "invalid.json"
                invalid.write_text("{}\n", encoding="utf-8")
                with mock.patch.object(
                    watcher, "_probe_task_state", return_value="ready"
                ), mock.patch.object(
                    watcher, "_completion_path", return_value=invalid
                ), mock.patch.object(
                    watcher, "_completion_is_valid", return_value=False
                ):
                    self.assertEqual(watcher.task_state(candidate), "conflict")
        clean = watcher.TASKS[0]
        with mock.patch.object(watcher, "task_processes", return_value=()), mock.patch.object(
            watcher, "_probe_task_state", return_value="fresh"
        ):
            self.assertEqual(watcher.task_state(clean), "conflict")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runs").mkdir()
            outside = root / "outside"
            outside.mkdir()
            (root / "runs" / "linked").symlink_to(
                outside, target_is_directory=True
            )
            linked_task = watcher.Task(
                "linked",
                "psbfr",
                "psbfr_v1",
                42,
                root / "runs" / "linked" / "run",
                1,
                True,
            )
            with mock.patch.object(watcher, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(
                    watcher.ABModelScreenWatcherError, "contains a symlink"
                ):
                    watcher._assert_safe_task_run_chain(linked_task)

    def test_completion_marker_is_atomic_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "last_training_state.pth.tar").write_bytes(b"latest")
            (run / "validation_history.json").write_text("{}\n", encoding="utf-8")
            task = watcher.Task(
                "fixture", "psbfr", "psbfr_v1", 42, run, 1, True
            )
            completion = root / "completion"
            with mock.patch.object(watcher, "COMPLETION_ROOT", completion), mock.patch.object(
                watcher, "FORMAL_AUTHORIZATION_SHA256", "c" * 64
            ):
                path = watcher._atomic_write_completion(task)
                original = path.read_bytes()
                self.assertFalse(json.loads(original)["lockbox_accessed"])
                self.assertEqual(watcher._atomic_write_completion(task), path)
                self.assertEqual(path.read_bytes(), original)
                path.write_text("different\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    watcher.ABModelScreenWatcherError, "no-clobber"
                ):
                    watcher._atomic_write_completion(task)

    def test_ready_job_terms_process_group_only_after_two_ready_probes(self) -> None:
        task = watcher.TASKS[1]
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "job.log"
            stale = b"EVISIRST_AB_MODEL_SCREEN_WORKER:EPOCH500_CLEAN\n"
            log.write_bytes(stale)
            job = watcher.RunningJob(
                task,
                process,
                watcher.GPU(0, "GPU-aa", "0000:01:00.0", 0),
                1,
                2,
                mock.Mock(),
                log,
                len(stale),
            )
            self.assertFalse(watcher._job_announced_post_cleanup(job))
            with log.open("ab") as handle:
                handle.write(stale)
            self.assertTrue(watcher._job_announced_post_cleanup(job))
            events: list[str] = []
            with mock.patch.object(
                watcher, "_probe_task_state", side_effect=["ready", "ready"]
            ), mock.patch.object(
                watcher.os,
                "killpg",
                side_effect=lambda pid, sig: events.append(f"term:{pid}:{sig}"),
            ), mock.patch.object(
                watcher,
                "_atomic_write_completion",
                side_effect=lambda _task: events.append("marker"),
            ), mock.patch.object(watcher, "_completion_is_valid", return_value=True):
                self.assertTrue(watcher.stop_ready_job(job))
            self.assertEqual(events[-1], "marker")
            self.assertEqual(process.wait_calls, [watcher.TERM_TIMEOUT_SECONDS])

    def test_launch_holds_shared_gpu_and_task_locks_in_new_session(self) -> None:
        gpu = watcher.GPU(0, "GPU-aaaaaaaa", "0000:01:00.0", 0)
        task = next(task for task in watcher.TASKS if task.name == "psbfr_s42")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu_lock_path = root / "gpu.lock"
            gpu_fd = os.open(gpu_lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            fake = FakeProcess()
            with mock.patch.object(
                watcher, "TASK_LOCK_ROOT", root / "task_locks"
            ), mock.patch.object(
                watcher, "LOG_ROOT", root / "logs"
            ), mock.patch.object(
                watcher, "authorization_is_valid", return_value=True
            ), mock.patch.object(
                watcher, "task_processes", return_value=()
            ), mock.patch.object(
                watcher, "sample_idle_gpus", return_value={gpu.uuid: gpu}
            ), mock.patch.object(
                watcher.subprocess, "Popen", return_value=fake
            ) as popen:
                job = watcher.launch_task(
                    task, resume=False, gpu=gpu, gpu_lock_fd=gpu_fd
                )
                self.assertIsNotNone(job)
                kwargs = popen.call_args.kwargs
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(
                    set(kwargs["pass_fds"]), {job.gpu_lock_fd, job.task_lock_fd}
                )
                self.assertTrue((root / "task_locks/psbfr_s42.lock").is_file())
                watcher._release_job(job)

            locked = next(
                task for task in watcher.TASKS if task.name == "psbfr_s1446202191"
            )
            locked_fd = os.open(root / "locked_gpu.lock", os.O_RDWR | os.O_CREAT, 0o600)
            with self.assertRaisesRegex(
                watcher.ABModelScreenWatcherError, "Stage-2"
            ):
                watcher.launch_task(
                    locked, resume=False, gpu=gpu, gpu_lock_fd=locked_fd
                )
            with self.assertRaises(OSError):
                os.fstat(locked_fd)

            abandoned = []
            for index in range(2):
                gpu_fd = os.open(root / f"abandon_gpu_{index}.lock", os.O_RDWR | os.O_CREAT, 0o600)
                task_fd = os.open(root / f"abandon_task_{index}.lock", os.O_RDWR | os.O_CREAT, 0o600)
                abandoned.append(
                    watcher.RunningJob(
                        task,
                        FakeProcess(pid=5000 + index),
                        gpu,
                        gpu_fd,
                        task_fd,
                        mock.Mock(),
                        root / f"abandon_{index}.log",
                        0,
                    )
                )
            with mock.patch.object(watcher.os, "killpg") as killpg:
                for job in abandoned:
                    watcher._abort_job(job)
            self.assertEqual(killpg.call_count, 2)
            for job in abandoned:
                self.assertEqual(job.gpu_lock_fd, -1)
                self.assertEqual(job.task_lock_fd, -1)
                self.assertEqual(job.process.wait_calls, [watcher.TERM_TIMEOUT_SECONDS])

    def test_stdlib_parent_cp_adapter_and_no_test_dataset_entrypoint(self) -> None:
        source = Path(watcher.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("numpy", imported_roots)
        self.assertIn("train_irstd_cp_hf_s2_legacy_screen_v1", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("outside the Stage-1 authorization", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("launch state changed after scheduling", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("runner lock was acquired", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("_finalize_completed=forbid_completed_finalization", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("shared.r1._validate_resume_candidates", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("signal.pause", watcher._FORMAL_WORKER_PROGRAM)
        self.assertIn("candidate cleanup is incomplete", watcher._STATE_PROBE_PROGRAM)
        self.assertIn("duplicate JSON key", watcher._STATE_PROBE_PROGRAM)
        self.assertIn(
            "run directory chain contains a symlink", watcher._STATE_PROBE_PROGRAM
        )
        self.assertIn('"lockbox_accessed"', watcher._STATE_PROBE_PROGRAM)
        self.assertIn(
            'expected_states={"resume","ready"}', watcher._FORMAL_WORKER_PROGRAM
        )
        self.assertLess(
            watcher._STATE_PROBE_PROGRAM.index("if run.is_symlink()"),
            watcher._STATE_PROBE_PROGRAM.index("if not run.exists()"),
        )
        self.assertNotIn("EviSIRSTTestDataset", source)

    def test_formal_mode_requires_valid_authorization_and_never_spawns(self) -> None:
        with mock.patch.object(
            watcher, "authorization_is_valid", return_value=False
        ), mock.patch.object(watcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(watcher.ABModelScreenWatcherError, "disabled"):
                watcher.main(["--formal"])
        popen.assert_not_called()

    def test_supervisor_finishes_stage1_without_touching_locked_stage2(self) -> None:
        def clean_stage1_state(task):
            if task.name in watcher.STAGE1_TASK_NAMES:
                return "complete"
            return "resume" if task.kind == "clean" else "fresh"

        with mock.patch.object(
            watcher, "authorization_is_valid", return_value=True
        ), mock.patch.object(
            watcher, "stage1_clean_control_is_valid", return_value=True
        ), mock.patch.object(
            watcher, "task_state", side_effect=clean_stage1_state
        ), mock.patch.object(watcher, "claim_idle_gpus") as claim:
            self.assertEqual(watcher.supervise(sleep=lambda _seconds: None), 0)
        claim.assert_not_called()

        def violated_lock(task):
            if task.name in watcher.STAGE1_TASK_NAMES:
                return "complete"
            if task.name == "psbfr_s1446202191":
                return "resume"
            return "resume" if task.kind == "clean" else "fresh"

        with mock.patch.object(
            watcher, "authorization_is_valid", return_value=True
        ), mock.patch.object(
            watcher, "stage1_clean_control_is_valid", return_value=True
        ), mock.patch.object(
            watcher, "task_state", side_effect=violated_lock
        ):
            with self.assertRaisesRegex(
                watcher.ABModelScreenWatcherError, "locked Stage-2"
            ):
                watcher.supervise(sleep=lambda _seconds: None)


if __name__ == "__main__":
    unittest.main()
