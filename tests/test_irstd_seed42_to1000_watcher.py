from __future__ import annotations

import hashlib
import fcntl
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import run_irstd_seed42_to1000_when_idle as watcher


class Seed42ContinuationWatcherTests(unittest.TestCase):
    def test_manifest_authorizes_only_two_seed42_routes(self) -> None:
        manifest = watcher.task_manifest()
        self.assertEqual(manifest["task_count"], 2)
        self.assertEqual(
            {item["route"] for item in manifest["tasks"]},
            {"psbfr", "cp_hf_s2"},
        )
        self.assertEqual({item["run_seed"] for item in manifest["tasks"]}, {42})
        self.assertEqual(
            {item["architecture_seed"] for item in manifest["tasks"]}, {42}
        )
        self.assertEqual(
            {item["configured_total_epochs"] for item in manifest["tasks"]},
            {1000},
        )
        self.assertFalse(manifest["fresh_training_allowed"])
        self.assertFalse(manifest["other_run_seeds_allowed"])

    def test_manifest_pins_original_full_gpu_identities(self) -> None:
        entries = {item["route"]: item for item in watcher.task_manifest()["tasks"]}
        self.assertEqual(
            entries["psbfr"]["pinned_gpu_uuid"],
            "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
        )
        self.assertEqual(entries["psbfr"]["pinned_gpu_bus_id"], "00000000:16:00.0")
        self.assertEqual(
            entries["cp_hf_s2"]["pinned_gpu_uuid"],
            "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
        )
        self.assertEqual(entries["cp_hf_s2"]["pinned_gpu_bus_id"], "00000000:27:00.0")

    def test_post_interim_disclosure_is_explicit_and_not_claim_eligible(self) -> None:
        manifest = watcher.task_manifest()
        for key, value in watcher.POST_INTERIM_DISCLOSURE.items():
            self.assertIs(manifest[key], value)
        self.assertTrue(manifest["epoch500_results_disclosed_before_continuation"])
        self.assertTrue(manifest["user_requested_both_routes_to1000"])
        self.assertFalse(manifest["stable_over_baseline_claim_eligible"])

    def test_adapter_argv_is_exact_and_rejects_other_route(self) -> None:
        self.assertEqual(
            watcher.adapter_argv("psbfr"),
            (
                "/home/ly/BasicIRSTD/infrarenet/bin/python",
                "/home/ly/EviSIRST_main/run_irstd_seed42_to1000_v1.py",
                "--route",
                "psbfr",
            ),
        )
        with self.assertRaises(watcher.Seed42ContinuationWatcherError):
            watcher.adapter_argv("psbfr_s1446202191")

    def test_normalized_hash_masks_only_authorization_literal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.py"
            second = Path(directory) / "second.py"
            first.write_text(
                'FORMAL_AUTHORIZATION_SHA256: str | None = None\nVALUE=1\n',
                encoding="utf-8",
            )
            second.write_text(
                'FORMAL_AUTHORIZATION_SHA256: str | None = "' + "a" * 64 + '"\nVALUE=1\n',
                encoding="utf-8",
            )
            self.assertEqual(
                watcher._normalized_authorization_source_sha256(first),
                watcher._normalized_authorization_source_sha256(second),
            )
            second.write_text(
                'FORMAL_AUTHORIZATION_SHA256: str | None = "' + "a" * 64 + '"\nVALUE=2\n',
                encoding="utf-8",
            )
            self.assertNotEqual(
                watcher._normalized_authorization_source_sha256(first),
                watcher._normalized_authorization_source_sha256(second),
            )

    def test_formal_authorization_literal_is_none_or_a_full_sha256(self) -> None:
        value = watcher.FORMAL_AUTHORIZATION_SHA256
        self.assertTrue(
            value is None
            or (
                isinstance(value, str)
                and watcher._SHA256_RE.fullmatch(value) is not None
            )
        )

    def test_real_frozen_epoch500_ledgers_bind_exact_snapshots(self) -> None:
        for task in watcher.TASKS:
            payload = watcher.assert_frozen_epoch500_ledger(task.route)
            self.assertEqual(payload["stopped_after_epoch"], 500)
            self.assertEqual(payload["latest_sha256"], task.epoch500_latest_sha256)
            self.assertEqual(payload["history_sha256"], task.epoch500_history_sha256)
            self.assertFalse(payload["test_split_accessed"])

    def test_probe_maps_all_adapter_states_and_fails_closed(self) -> None:
        task = watcher.TASKS[0]
        mapping = {
            "epoch500_ready": "ready",
            "resume_501_999": "resume",
            "finalize_1000": "finalize",
            "complete": "complete_artifacts",
            "unexpected": "conflict",
        }
        with mock.patch.object(watcher, "assert_frozen_sources"), mock.patch.object(
            watcher, "assert_frozen_epoch500_ledger"
        ):
            for raw, expected in mapping.items():
                with self.subTest(raw=raw), mock.patch.object(
                    watcher, "_adapter_action", return_value={"state": raw}
                ):
                    self.assertEqual(watcher._probe_task_state(task), expected)

    def test_create_snapshot_only_at_exact_epoch500(self) -> None:
        with mock.patch.object(
            watcher, "_probe_task_state", return_value="ready"
        ), mock.patch.object(watcher, "assert_epoch500_start") as start, mock.patch.object(
            watcher, "_adapter_action"
        ) as action:
            action.side_effect = [
                {"created": True},
                {"snapshot_sha256": "a" * 64},
            ]
            result = watcher.create_or_validate_start_snapshot("psbfr")
        start.assert_called_once_with("psbfr")
        self.assertEqual(action.call_args_list[0].args, ("psbfr", "create_snapshot"))
        self.assertEqual(action.call_args_list[1].args, ("psbfr", "validate_snapshot"))
        self.assertEqual(result["snapshot_sha256"], "a" * 64)

    def test_resume_validates_but_does_not_recreate_start_snapshot(self) -> None:
        with mock.patch.object(
            watcher, "_probe_task_state", return_value="resume"
        ), mock.patch.object(watcher, "assert_epoch500_start") as start, mock.patch.object(
            watcher,
            "_adapter_action",
            return_value={"snapshot_sha256": "b" * 64},
        ) as action:
            watcher.create_or_validate_start_snapshot("cp_hf_s2")
        start.assert_not_called()
        action.assert_called_once_with("cp_hf_s2", "validate_snapshot")

    def test_continuation_entry_accepts_only_noncomplete_recovery_states(self) -> None:
        with mock.patch.object(watcher, "assert_frozen_sources"), mock.patch.object(
            watcher, "assert_frozen_epoch500_ledger"
        ):
            for state in ("epoch500_ready", "resume_501_999", "finalize_1000"):
                with self.subTest(state=state), mock.patch.object(
                    watcher, "_adapter_action", return_value={"state": state}
                ):
                    self.assertEqual(
                        watcher.assert_continuation_entry("psbfr")["state"], state
                    )
            with mock.patch.object(
                watcher, "_adapter_action", return_value={"state": "complete"}
            ), self.assertRaises(watcher.Seed42ContinuationWatcherError):
                watcher.assert_continuation_entry("psbfr")

    def test_completed_validator_requires_complete(self) -> None:
        with mock.patch.object(watcher, "assert_frozen_sources"), mock.patch.object(
            watcher, "assert_frozen_epoch500_ledger"
        ), mock.patch.object(
            watcher, "_adapter_action", return_value={"state": "resume_501_999"}
        ), self.assertRaises(watcher.Seed42ContinuationWatcherError):
            watcher.assert_completed_state("psbfr")

    def _make_proc_entry(
        self, root: Path, pid: int, argv: tuple[str, ...], ppid: int
    ) -> None:
        directory = root / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(
            b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
        )
        (directory / "status").write_text(
            f"Name:\ttest\nPPid:\t{ppid}\n", encoding="utf-8"
        )

    def test_exact_timed_parent_child_pair_is_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_proc_entry(root, 100, watcher.timed_adapter_argv("psbfr"), 1)
            self._make_proc_entry(root, 101, watcher.adapter_argv("psbfr"), 100)
            self.assertTrue(
                watcher._authorized_route_process_is_running(
                    "psbfr", proc_root=root
                )
            )
            self._make_proc_entry(root, 102, watcher.adapter_argv("cp_hf_s2"), 1)
            self.assertFalse(
                watcher._authorized_route_process_is_running(
                    "cp_hf_s2", proc_root=root
                )
            )

    def test_forbidden_old_worker_process_fails_closed(self) -> None:
        with mock.patch.object(
            watcher.stage1, "task_processes", return_value=(123,)
        ), self.assertRaises(watcher.Seed42ContinuationWatcherError):
            watcher.assert_no_forbidden_processes("psbfr")

    def test_external_duplicate_adapter_pair_is_rejected(self) -> None:
        with mock.patch.object(
            watcher.stage1, "task_processes", return_value=()
        ), mock.patch.object(
            watcher, "_route_process_pair", return_value=((101,), (100,))
        ), mock.patch.object(
            watcher, "_authorized_route_process_is_running", return_value=True
        ), mock.patch.object(os, "getpid", return_value=999), mock.patch.object(
            os, "getppid", return_value=998
        ), self.assertRaises(watcher.Seed42ContinuationWatcherError):
            watcher.assert_no_forbidden_processes("psbfr")

    def test_launch_capability_requires_three_distinct_locked_exact_files(self) -> None:
        task = watcher.TASKS[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "tasks"
            gpu_root = root / "gpus"
            task_root.mkdir()
            gpu_root.mkdir()
            old_lock = root / "old.lock"
            paths = (
                task_root / f"{task.name}.lock",
                gpu_root / f"{task.gpu_uuid}.lock",
                old_lock,
            )
            descriptors = []
            try:
                for path in paths:
                    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    descriptors.append(descriptor)
                env = {
                    "EVISIRST_SEED42_TO1000_ROUTE": task.route,
                    "EVISIRST_SEED42_TO1000_GPU_UUID": task.gpu_uuid,
                    "EVISIRST_SEED42_TO1000_TASK_LOCK_FD": str(descriptors[0]),
                    "EVISIRST_SEED42_TO1000_GPU_LOCK_FD": str(descriptors[1]),
                    "EVISIRST_SEED42_TO1000_OLD_WATCHER_LOCK_FD": str(
                        descriptors[2]
                    ),
                    "CUDA_VISIBLE_DEVICES": task.gpu_uuid,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                }
                with mock.patch.object(watcher, "TASK_LOCK_ROOT", task_root), mock.patch.object(
                    watcher, "GPU_LOCK_ROOT", gpu_root
                ), mock.patch.object(watcher, "OLD_WATCHER_LOCK", old_lock), mock.patch.dict(
                    os.environ, env, clear=True
                ):
                    watcher.assert_launch_capability(task.route)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    with self.assertRaises(watcher.Seed42ContinuationWatcherError):
                        watcher.assert_launch_capability(task.route)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_worker_environment_exports_full_uuid_and_lock_fds(self) -> None:
        task = watcher.TASKS[0]
        gpu = watcher.stage1.GPU(0, task.gpu_uuid, task.gpu_bus_id, 0)
        env = watcher._worker_env(
            task,
            gpu,
            gpu_lock_fd=11,
            task_lock_fd=12,
            old_watcher_lock_fd=13,
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], task.gpu_uuid)
        self.assertEqual(env["EVISIRST_SEED42_TO1000_GPU_UUID"], task.gpu_uuid)
        self.assertEqual(env["EVISIRST_SEED42_TO1000_GPU_LOCK_FD"], "11")
        self.assertEqual(env["EVISIRST_SEED42_TO1000_TASK_LOCK_FD"], "12")
        self.assertEqual(env["EVISIRST_SEED42_TO1000_OLD_WATCHER_LOCK_FD"], "13")

    def test_abort_waits_for_group_exit_before_locks_release_and_logs_recovery(self) -> None:
        task = watcher.TASKS[0]
        gpu = watcher.stage1.GPU(0, task.gpu_uuid, task.gpu_bus_id, 0)
        gpu_read, gpu_fd = os.pipe()
        task_read, task_fd = os.pipe()
        os.close(gpu_read)
        os.close(task_read)
        process = mock.Mock()
        process.pid = 456
        process.poll.return_value = None
        process.wait.return_value = 0
        log_handle = mock.Mock()
        job = watcher.RunningJob(
            task=task,
            process=process,
            gpu=gpu,
            gpu_lock_fd=gpu_fd,
            task_lock_fd=task_fd,
            log_handle=log_handle,
            log_path=Path("/tmp/not-read-by-test.log"),
        )
        with mock.patch.object(os, "killpg") as killpg, mock.patch.object(
            watcher, "_log"
        ) as log:
            watcher._abort_job(job)
        killpg.assert_called_once_with(456, watcher.signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=watcher.TERM_TIMEOUT_SECONDS)
        log_handle.close.assert_called_once()
        self.assertEqual(job.gpu_lock_fd, -1)
        self.assertEqual(job.task_lock_fd, -1)
        messages = " ".join(call.args[0] for call in log.call_args_list)
        self.assertIn("waiting for process-group exit", messages)
        self.assertIn("recover fixed transaction temporaries", messages)

    def test_claim_pinned_gpu_checks_idle_four_times_around_lock(self) -> None:
        task = watcher.TASKS[0]
        gpu = watcher.stage1.GPU(0, task.gpu_uuid, task.gpu_bus_id, 0)
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        try:
            with mock.patch.object(
                watcher.stage1,
                "sample_idle_gpus",
                side_effect=[
                    {task.gpu_uuid: gpu},
                    {task.gpu_uuid: gpu},
                    {task.gpu_uuid: gpu},
                    {task.gpu_uuid: gpu},
                ],
            ) as sample, mock.patch.object(
                watcher.stage1, "_try_lock", return_value=write_fd
            ) as lock:
                claimed = watcher.claim_pinned_gpu(task, sleep=lambda _seconds: None)
            self.assertEqual(claimed, (gpu, write_fd))
            self.assertEqual(sample.call_count, 4)
            lock.assert_called_once_with(
                watcher.GPU_LOCK_ROOT / f"{task.gpu_uuid}.lock"
            )
        finally:
            try:
                os.close(write_fd)
            except OSError:
                pass

    def test_claim_pinned_gpu_rejects_wrong_bus_before_lock(self) -> None:
        task = watcher.TASKS[0]
        wrong = watcher.stage1.GPU(0, task.gpu_uuid, "00000000:99:00.0", 0)
        with mock.patch.object(
            watcher.stage1,
            "sample_idle_gpus",
            side_effect=[{task.gpu_uuid: wrong}, {task.gpu_uuid: wrong}],
        ), mock.patch.object(watcher.stage1, "_try_lock") as lock:
            self.assertIsNone(
                watcher.claim_pinned_gpu(task, sleep=lambda _seconds: None)
            )
        lock.assert_not_called()

    def test_task_state_distinguishes_unledgered_and_ledgered_complete(self) -> None:
        task = watcher.TASKS[0]
        with mock.patch.object(
            watcher, "_route_process_pair", return_value=((), ())
        ), mock.patch.object(
            watcher, "_probe_task_state", return_value="complete_artifacts"
        ), mock.patch.object(Path, "exists", return_value=False), mock.patch.object(
            Path, "is_symlink", return_value=False
        ):
            self.assertEqual(watcher.task_state(task), "complete_unledgered")

    def test_launch_rejects_invalid_authorization_and_closes_gpu_fd(self) -> None:
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        task = watcher.TASKS[0]
        gpu = watcher.stage1.GPU(0, task.gpu_uuid, task.gpu_bus_id, 0)
        with mock.patch.object(watcher, "authorization_is_valid", return_value=False):
            self.assertIsNone(
                watcher.launch_task(
                    task,
                    expected_state="ready",
                    gpu=gpu,
                    gpu_lock_fd=write_fd,
                    old_watcher_lock_fd=123,
                )
            )
        with self.assertRaises(OSError):
            os.fstat(write_fd)

    def test_launch_passes_all_capability_fds_and_exact_pinned_environment(self) -> None:
        task = watcher.TASKS[0]
        gpu = watcher.stage1.GPU(0, task.gpu_uuid, task.gpu_bus_id, 0)
        gpu_read, gpu_fd = os.pipe()
        task_read, task_fd = os.pipe()
        os.close(gpu_read)
        os.close(task_read)
        fake_process = SimpleNamespace(pid=321)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            watcher, "LOG_ROOT", Path(directory)
        ), mock.patch.object(
            watcher, "authorization_is_valid", return_value=True
        ), mock.patch.object(
            watcher.stage1, "_try_lock", return_value=task_fd
        ), mock.patch.object(
            watcher, "task_state", return_value="ready"
        ), mock.patch.object(
            watcher, "assert_no_forbidden_processes"
        ), mock.patch.object(
            watcher, "create_or_validate_start_snapshot"
        ), mock.patch.object(
            watcher, "assert_continuation_entry"
        ), mock.patch.object(
            watcher.stage1,
            "sample_idle_gpus",
            return_value={task.gpu_uuid: gpu},
        ), mock.patch.object(
            subprocess, "Popen", return_value=fake_process
        ) as popen:
            job = watcher.launch_task(
                task,
                expected_state="ready",
                gpu=gpu,
                gpu_lock_fd=gpu_fd,
                old_watcher_lock_fd=77,
            )
            self.assertIsNotNone(job)
            keywords = popen.call_args.kwargs
            self.assertEqual(keywords["pass_fds"], (gpu_fd, task_fd, 77))
            self.assertEqual(keywords["env"]["CUDA_VISIBLE_DEVICES"], task.gpu_uuid)
            self.assertEqual(
                keywords["env"]["EVISIRST_SEED42_TO1000_TASK_LOCK_FD"],
                str(task_fd),
            )
            self.assertTrue(keywords["start_new_session"])
            watcher._release_job(job)

    def test_sigterm_contains_running_job_before_singleton_lock_release(self) -> None:
        task = watcher.TASKS[0]
        gpu = watcher.stage1.GPU(0, task.gpu_uuid, task.gpu_bus_id, 0)
        job = SimpleNamespace(task=task)
        events: list[object] = []
        active_handlers: dict[int, object] = {}
        original_handlers = {
            watcher.signal.SIGTERM: object(),
            watcher.signal.SIGINT: object(),
        }

        def fake_signal(signum: int, handler: object) -> object:
            previous = active_handlers.get(signum, original_handlers[signum])
            active_handlers[signum] = handler
            return previous

        def launch_with_pending_signal(*_args, **_kwargs) -> object:
            handler = active_handlers[watcher.signal.SIGTERM]
            self.assertTrue(callable(handler))
            handler(watcher.signal.SIGTERM, None)
            events.append("signal-recorded-before-launch-return")
            return job

        def abort(observed_job: object) -> None:
            self.assertIs(observed_job, job)
            events.append("abort")

        with mock.patch.object(
            watcher, "authorization_is_valid", return_value=True
        ), mock.patch.object(
            watcher, "_assert_safe_directory_chain"
        ), mock.patch.object(
            watcher.stage1, "_try_lock", return_value=101
        ), mock.patch.object(
            watcher, "_try_existing_lock", return_value=102
        ), mock.patch.object(
            watcher,
            "task_state",
            side_effect=lambda observed: (
                "ready" if observed is task else "waiting"
            ),
        ), mock.patch.object(
            watcher, "claim_pinned_gpu", return_value=(gpu, 103)
        ), mock.patch.object(
            watcher, "launch_task", side_effect=launch_with_pending_signal
        ), mock.patch.object(
            watcher, "_abort_job", side_effect=abort
        ), mock.patch.object(
            watcher.signal, "signal", side_effect=fake_signal
        ), mock.patch.object(
            watcher.os, "close", side_effect=lambda descriptor: events.append(
                ("close", descriptor)
            )
        ), mock.patch.object(watcher, "_log") as log:
            self.assertEqual(watcher.main(["--formal"]), 143)

        self.assertEqual(
            events,
            [
                "signal-recorded-before-launch-return",
                "abort",
                ("close", 102),
                ("close", 101),
            ],
        )
        self.assertIs(
            active_handlers[watcher.signal.SIGTERM],
            original_handlers[watcher.signal.SIGTERM],
        )
        self.assertIs(
            active_handlers[watcher.signal.SIGINT],
            original_handlers[watcher.signal.SIGINT],
        )
        self.assertIn(
            "all supervised process groups contained before lock release",
            " ".join(call.args[0] for call in log.call_args_list),
        )

    def test_main_formal_is_blocked_without_frozen_authorization(self) -> None:
        with mock.patch.object(
            watcher, "authorization_is_valid", return_value=False
        ), self.assertRaises(watcher.Seed42ContinuationWatcherError):
            watcher.main(["--formal"])


if __name__ == "__main__":
    unittest.main()
