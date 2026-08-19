from __future__ import annotations

import hashlib
import json
import os
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

import run_irstd_seed42_to1000_v1 as adapter


FAKE_SHA = "a" * 64


class Seed42ContinuationTests(unittest.TestCase):
    def _snapshot_fixture(self):
        temporary = tempfile.TemporaryDirectory(
            dir=adapter.PROJECT_ROOT / "runs"
        )
        root = Path(temporary.name)
        run_dir = root / "live"
        (run_dir / "candidates").mkdir(parents=True)
        contents = {
            "last_training_state.pth.tar": b"latest-500",
            "validation_history.json": b"history-500",
            "candidates/epoch_0007.pth.tar": b"candidate-7",
        }
        digests = {}
        for relative, content in contents.items():
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            digests[relative] = hashlib.sha256(content).hexdigest()
        stage1 = root / "stage1"
        ledger_dir = stage1 / "epoch500_commits"
        ledger_dir.mkdir(parents=True)
        ledger = {
            "schema": "evisirst_irstd_ab_model_screen_epoch500_stop/v1",
            "task": "unit_s42",
            "kind": "unit",
            "variant": "unit_v1",
            "run_seed": 42,
            "configured_total_epochs": 1000,
            "stopped_after_epoch": 500,
            "latest_sha256": digests["last_training_state.pth.tar"],
            "history_sha256": digests["validation_history.json"],
            "candidate_cleanup_validated": True,
            "lockbox_accessed": False,
            "public_test_allowed": False,
            "test_split_accessed": False,
        }
        ledger_path = ledger_dir / "unit_s42.json"
        ledger_path.write_text(
            json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        spec = adapter.RouteSpec(
            "unit",
            "unit_s42",
            "unit",
            "unit_v1",
            run_dir,
            adapter._sha256(ledger_path),
            digests["last_training_state.pth.tar"],
            digests["validation_history.json"],
            {"candidates/epoch_0007.pth.tar": digests["candidates/epoch_0007.pth.tar"]},
        )
        patches = (
            mock.patch.object(adapter, "STAGE1_WATCH_ROOT", stage1),
            mock.patch.object(adapter, "START_SNAPSHOT_ROOT", root / "snapshots"),
            mock.patch.dict(adapter.ROUTE_SPECS, {"unit": spec}, clear=True),
            mock.patch.object(adapter, "_authorization", return_value=(FAKE_SHA, "b" * 64)),
        )
        return temporary, spec, digests, patches

    def test_snapshot_is_hardlinked_manifest_last_and_survives_source_replace(self):
        temporary, spec, digests, patches = self._snapshot_fixture()
        try:
            with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
                adapter,
                "inspect_route_state",
                return_value={
                    "state": "epoch500_ready",
                    "latest_sha256": spec.latest_sha256,
                    "history_sha256": spec.history_sha256,
                    "candidate_sha256": dict(spec.candidates),
                },
            ):
                manifest_path = adapter.create_start_snapshot("unit")
                manifest = adapter.validate_start_snapshot("unit")
                self.assertEqual(manifest["schema"], adapter.SNAPSHOT_SCHEMA)
                for key, value in adapter.POST_INTERIM_DISCLOSURE.items():
                    self.assertIs(manifest[key], value)
                for relative in digests:
                    source = spec.run_dir / relative
                    snapshot = spec.snapshot_dir / relative
                    self.assertEqual(
                        (source.stat().st_dev, source.stat().st_ino),
                        (snapshot.stat().st_dev, snapshot.stat().st_ino),
                    )
                replacement = spec.run_dir / ".replacement"
                replacement.write_bytes(b"new-live-latest")
                os.replace(replacement, spec.run_dir / "last_training_state.pth.tar")
                self.assertEqual(
                    adapter._sha256(spec.snapshot_dir / "last_training_state.pth.tar"),
                    spec.latest_sha256,
                )
                self.assertTrue(manifest_path.is_file())
        finally:
            temporary.cleanup()

    def test_snapshot_tamper_and_existing_conflict_fail_closed(self):
        temporary, spec, _digests, patches = self._snapshot_fixture()
        try:
            report = {
                "state": "epoch500_ready",
                "latest_sha256": spec.latest_sha256,
                "history_sha256": spec.history_sha256,
                "candidate_sha256": dict(spec.candidates),
            }
            with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
                adapter, "inspect_route_state", return_value=report
            ):
                adapter._safe_snapshot_dir(spec.snapshot_dir, True)
                (spec.snapshot_dir / "last_training_state.pth.tar").write_bytes(b"conflict")
                with self.assertRaises(adapter.Seed42ContinuationError):
                    adapter.create_start_snapshot("unit")
                (spec.snapshot_dir / "last_training_state.pth.tar").unlink()
                adapter.create_start_snapshot("unit")
                source = spec.run_dir / "last_training_state.pth.tar"
                replacement = spec.run_dir / ".new-live"
                replacement.write_bytes(source.read_bytes())
                os.replace(replacement, source)
                (spec.snapshot_dir / "last_training_state.pth.tar").write_bytes(b"tamper")
                with self.assertRaises(adapter.Seed42ContinuationError):
                    adapter.validate_start_snapshot("unit")
        finally:
            temporary.cleanup()

    def test_incomplete_snapshot_manifest_temp_recovers_but_completed_or_symlink_fails(self):
        temporary, spec, _digests, patches = self._snapshot_fixture()
        try:
            report = {
                "state": "epoch500_ready",
                "latest_sha256": spec.latest_sha256,
                "history_sha256": spec.history_sha256,
                "candidate_sha256": dict(spec.candidates),
            }
            with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
                adapter, "inspect_route_state", return_value=report
            ):
                adapter._safe_snapshot_dir(spec.snapshot_dir, True)
                stale = spec.snapshot_dir / ".manifest.abcdefgh.tmp"
                stale.write_bytes(b"uncommitted")
                adapter.create_start_snapshot("unit")
                self.assertFalse(stale.exists())
                os.link(spec.snapshot_dir / "manifest.json", stale)
                adapter.create_start_snapshot("unit")
                self.assertFalse(stale.exists())
                stale.write_bytes(b"unexpected-after-commit")
                with self.assertRaises(adapter.Seed42ContinuationError):
                    adapter.create_start_snapshot("unit")
        finally:
            temporary.cleanup()
        temporary, spec, _digests, patches = self._snapshot_fixture()
        try:
            report = {
                "state": "epoch500_ready",
                "latest_sha256": spec.latest_sha256,
                "history_sha256": spec.history_sha256,
                "candidate_sha256": dict(spec.candidates),
            }
            with patches[0], patches[1], patches[2], patches[3], mock.patch.object(
                adapter, "inspect_route_state", return_value=report
            ):
                adapter._safe_snapshot_dir(spec.snapshot_dir, True)
                target = spec.snapshot_dir / "target"
                target.write_bytes(b"x")
                (spec.snapshot_dir / ".manifest.abcdefgh.tmp").symlink_to(target)
                with self.assertRaises(adapter.Seed42ContinuationError):
                    adapter.create_start_snapshot("unit")
        finally:
            temporary.cleanup()

    def test_entry_accepts_restart_and_finalize_but_not_complete(self):
        for state in ("resume_501_999", "finalize_1000"):
            with self.subTest(state=state), mock.patch.object(
                adapter, "validate_start_snapshot"
            ), mock.patch.object(
                adapter, "inspect_route_state", return_value={"state": state}
            ):
                self.assertEqual(adapter.assert_continuation_entry("psbfr")["state"], state)
        with mock.patch.object(adapter, "validate_start_snapshot"), mock.patch.object(
            adapter, "inspect_route_state", return_value={"state": "complete"}
        ):
            with self.assertRaises(adapter.Seed42ContinuationError):
                adapter.assert_continuation_entry("psbfr")

    def test_completed_sidecar_is_explicitly_exploratory(self):
        report = {"state": "complete", "committed_epoch": 1000}
        with mock.patch.object(adapter, "validate_start_snapshot"), mock.patch.object(
            adapter, "inspect_route_state", return_value=report
        ):
            evidence = adapter.assert_completed_state("cp_hf_s2")
        for key, value in adapter.POST_INTERIM_DISCLOSURE.items():
            self.assertIs(evidence[key], value)
        self.assertIs(evidence["stable_over_baseline_claim_eligible"], False)

    def test_cp_context_accepts_post_501_loader_and_restores_every_patch(self):
        trusted_finalize = object()
        blocked_finalize = object()
        blocked_loader = object()
        trusted_atomic = mock.Mock()
        active = types.SimpleNamespace()
        active.r1 = types.SimpleNamespace(
            _load_resume_state=lambda *a, **k: (777, [], [], {})
        )
        trusted_loader = active.r1._load_resume_state
        active.hf_transaction = types.SimpleNamespace(_finalize_completed=trusted_finalize)
        active._variant_atomic_torch_save = trusted_atomic
        active.resolve_run_paths = lambda args: {"final": Path("/nonexistent/generic")}
        @contextmanager
        def run_lock(args):
            yield Path("/nonexistent/lock")
        active._run_process_lock = run_lock

        @contextmanager
        def original_context(args):
            active.r1._load_resume_state = blocked_loader
            active.hf_transaction._finalize_completed = blocked_finalize
            try:
                yield active
            finally:
                active.r1._load_resume_state = trusted_loader
                active.hf_transaction._finalize_completed = trusted_finalize

        original_barrier = object()
        original_pause = object()
        validator = mock.Mock(
            return_value={
                "insertion_point": "after_up_decoder2_finish_before_gt2_and_up_decoder1"
            }
        )
        cp = types.SimpleNamespace(
            _screen_transaction_adapter=original_context,
            _formal_entry_barrier=original_barrier,
            _install_atomic_epoch500_pause=original_pause,
            architecture_api=lambda: {"validate_irstd_cp_hf_s2_v1": validator},
        )
        args = types.SimpleNamespace(run_seed=42, epochs=1000, resume=True)
        with adapter._cp_entry_guard(cp, active):
            self.assertIsNot(cp._screen_transaction_adapter, original_context)
            with cp._screen_transaction_adapter(args) as entered:
                restored = entered.r1._load_resume_state(model=object())
                self.assertEqual(restored[0], 777)
                self.assertIs(entered.hf_transaction._finalize_completed, trusted_finalize)
                validator.assert_called_once()
            self.assertIs(active.r1._load_resume_state, trusted_loader)
        self.assertIs(cp._screen_transaction_adapter, original_context)
        self.assertIs(cp._formal_entry_barrier, original_barrier)
        self.assertIs(cp._install_atomic_epoch500_pause, original_pause)

    def test_cp_context_rejects_start_after_1000(self):
        active = types.SimpleNamespace()
        active.r1 = types.SimpleNamespace(_load_resume_state=lambda *a, **k: (1002, [], [], {}))
        active.hf_transaction = types.SimpleNamespace(_finalize_completed=object())
        active._variant_atomic_torch_save = mock.Mock()
        active.resolve_run_paths = lambda args: {"final": Path("/nonexistent/generic")}
        @contextmanager
        def run_lock(args):
            yield Path("/nonexistent/lock")
        active._run_process_lock = run_lock
        @contextmanager
        def context(args):
            yield active
        cp = types.SimpleNamespace(
            _screen_transaction_adapter=context,
            _formal_entry_barrier=object(),
            _install_atomic_epoch500_pause=object(),
            architecture_api=lambda: {"validate_irstd_cp_hf_s2_v1": mock.Mock()},
        )
        args = types.SimpleNamespace(run_seed=42, epochs=1000, resume=True)
        with adapter._cp_entry_guard(cp, active):
            with cp._screen_transaction_adapter(args) as entered:
                with self.assertRaises(adapter.Seed42ContinuationError):
                    entered.r1._load_resume_state(model=object())

    def test_recoverable_history_and_candidate_windows(self):
        self.assertTrue(adapter._recoverable_history_epoch(700, 699))
        self.assertTrue(adapter._recoverable_history_epoch(700, 700))
        self.assertTrue(adapter._recoverable_history_epoch(700, 701))
        self.assertFalse(adapter._recoverable_history_epoch(700, 702))
        self.assertTrue(
            adapter._recoverable_extra_candidate(
                700, 650, has_committed_record=True
            )
        )
        self.assertTrue(
            adapter._recoverable_extra_candidate(
                700, 701, has_committed_record=False
            )
        )
        self.assertFalse(
            adapter._recoverable_extra_candidate(
                700, 650, has_committed_record=False
            )
        )
        self.assertFalse(
            adapter._recoverable_extra_candidate(
                700, 702, has_committed_record=False
            )
        )
        self.assertFalse(
            adapter._recoverable_extra_candidate(
                1000, 1001, has_committed_record=False
            )
        )
        adapter._assert_sidecar_crash_window(
            committed_epoch=1000,
            history_epoch=1000,
            extra_candidate_count=0,
            final_artifact_exists=True,
        )
        with self.assertRaises(adapter.Seed42ContinuationError):
            adapter._assert_sidecar_crash_window(
                committed_epoch=1000,
                history_epoch=999,
                extra_candidate_count=0,
                final_artifact_exists=True,
            )
        with self.assertRaises(adapter.Seed42ContinuationError):
            adapter._assert_sidecar_crash_window(
                committed_epoch=1000,
                history_epoch=1000,
                extra_candidate_count=1,
                final_artifact_exists=True,
            )

    def test_summary_evidence_rejects_training_candidate_or_selected_record_tamper(self):
        record = {"epoch": 9, "mIoU": 0.7}
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        expected = {
            "training_history": [{"epoch": 1}],
            "validation_history": [record],
            "candidate_artifacts": {"9": {"relative_path": "candidates/x"}},
            "selected_validation_record": record,
            "selected_validation_record_sha256": digest,
        }
        kwargs = {
            "training_history": expected["training_history"],
            "validation_history": expected["validation_history"],
            "candidate_artifacts": expected["candidate_artifacts"],
            "selected_record": record,
            "selected_record_sha256": digest,
        }
        adapter._assert_summary_evidence(expected, **kwargs)
        for key, bad in (
            ("training_history", []),
            ("candidate_artifacts", {}),
            ("selected_validation_record", {"epoch": 8}),
            ("selected_validation_record_sha256", "0" * 64),
        ):
            with self.subTest(key=key), self.assertRaises(
                adapter.Seed42ContinuationError
            ):
                adapter._assert_summary_evidence({**expected, key: bad}, **kwargs)

    def test_transaction_window_classification_covers_restart_and_finalize(self):
        classify = adapter._classify_transaction_window
        base = dict(
            summary_exists=False,
            role_map_present=False,
            generic_final_exists=False,
            role_file_count=0,
            primary_role_exists=False,
        )
        self.assertEqual(classify(committed_epoch=500, **base), "epoch500_ready")
        self.assertEqual(classify(committed_epoch=777, **base), "resume_501_999")
        self.assertEqual(classify(committed_epoch=1000, **base), "finalize_1000")
        self.assertEqual(
            classify(
                committed_epoch=1000,
                **{**base, "summary_exists": True, "generic_final_exists": True},
            ),
            "finalize_1000",
        )
        self.assertEqual(
            classify(
                committed_epoch=1000,
                **{
                    **base,
                    "summary_exists": True,
                    "role_map_present": True,
                    "generic_final_exists": True,
                    "role_file_count": 2,
                    "primary_role_exists": True,
                },
            ),
            "finalize_1000",
        )
        self.assertEqual(
            classify(
                committed_epoch=1000,
                **{
                    **base,
                    "summary_exists": True,
                    "role_map_present": True,
                    "role_file_count": 2,
                    "primary_role_exists": True,
                },
            ),
            "complete",
        )
        self.assertEqual(
            classify(
                committed_epoch=1000,
                **{
                    **base,
                    "summary_exists": True,
                    "role_map_present": True,
                    "role_file_count": 2,
                    "primary_role_exists": True,
                    "recoverable_temp_count": 1,
                },
            ),
            "finalize_1000",
        )
        with self.assertRaises(adapter.Seed42ContinuationError):
            classify(committed_epoch=700, **{**base, "generic_final_exists": True})
        with self.assertRaises(adapter.Seed42ContinuationError):
            classify(
                committed_epoch=1000,
                **{**base, "summary_exists": True, "role_file_count": 1},
            )
        with self.assertRaises(adapter.Seed42ContinuationError):
            classify(committed_epoch=1000, **{**base, "role_file_count": 1})

    def test_semantic_comparator_rejects_any_tensor_or_metadata_change(self):
        import torch
        expected = {"state": {"weight": torch.tensor([1.0, 2.0])}, "epoch": 1000}
        self.assertTrue(adapter._semantic_equal(expected, expected))
        self.assertFalse(
            adapter._semantic_equal(
                expected,
                {"state": {"weight": torch.tensor([1.0, 3.0])}, "epoch": 1000},
            )
        )
        provenance = {"rule": "frozen"}
        legal_generic = {"epoch": 1000, "selection_provenance": provenance}
        adapter._assert_generic_selection_metadata(
            legal_generic, selected_epoch=1000, provenance=provenance
        )
        with self.assertRaises(adapter.Seed42ContinuationError):
            adapter._assert_generic_selection_metadata(
                {**legal_generic, "selected_validation_record": {}},
                selected_epoch=1000,
                provenance=provenance,
            )

    def test_locked_cleanup_may_expose_complete_for_trusted_finalizer(self):
        report = {"state": "complete", "recoverable_temporary_files": []}
        with mock.patch.object(adapter, "validate_start_snapshot"), mock.patch.object(
            adapter, "assert_continuation_entry"
        ) as continuation:
            self.assertEqual(
                adapter._accept_locked_clean_state("psbfr", report), report
            )
            continuation.assert_not_called()

    def test_fixed_writer_temp_cleanup_and_unknown_or_symlink_rejection(self):
        with tempfile.TemporaryDirectory(dir=adapter.PROJECT_ROOT / "runs") as name:
            run_dir = Path(name)
            temp = run_dir / ".last_training_state.pth.tar.abc12345.tmp"
            temp.write_bytes(b"uncommitted")
            spec = adapter.RouteSpec(
                "unit", "unit", "unit", "unit", run_dir,
                FAKE_SHA, FAKE_SHA, FAKE_SHA, {},
            )
            reports = [
                {"recoverable_temporary_files": [temp.name]},
                {"recoverable_temporary_files": []},
            ]
            with mock.patch.dict(adapter.ROUTE_SPECS, {"unit": spec}, clear=True), mock.patch.object(
                adapter, "inspect_route_state", side_effect=reports
            ):
                adapter._cleanup_recoverable_temps_under_run_lock("unit")
            self.assertFalse(temp.exists())
            with mock.patch.dict(adapter.ROUTE_SPECS, {"unit": spec}, clear=True), mock.patch.object(
                adapter,
                "inspect_route_state",
                return_value={"recoverable_temporary_files": ["unknown.tmp"]},
            ):
                with self.assertRaises(adapter.Seed42ContinuationError):
                    adapter._cleanup_recoverable_temps_under_run_lock("unit")
            target = run_dir / "target"
            target.write_bytes(b"x")
            link = run_dir / ".summary.json.abc12345.tmp"
            link.symlink_to(target)
            with mock.patch.dict(adapter.ROUTE_SPECS, {"unit": spec}, clear=True), mock.patch.object(
                adapter,
                "inspect_route_state",
                return_value={"recoverable_temporary_files": [link.name]},
            ):
                with self.assertRaises(adapter.Seed42ContinuationError):
                    adapter._cleanup_recoverable_temps_under_run_lock("unit")

    def test_cp_generic_final_idempotence_requires_full_expected_payload(self):
        import torch
        with tempfile.TemporaryDirectory(dir=adapter.PROJECT_ROOT / "runs") as name:
            generic = Path(name) / "EviSIRST.pth.tar"
            raw = {"schema": "checkpoint", "tensor": torch.tensor([1.0]), "epoch": 1000}
            expected = {**raw, "cp_bound": True}
            torch.save(expected, generic)
            active = types.SimpleNamespace()
            active.r1 = types.SimpleNamespace(_load_resume_state=lambda *a, **k: (1001, [], [], {}))
            trusted_loader = active.r1._load_resume_state
            active.hf_transaction = types.SimpleNamespace(_finalize_completed=object())
            trusted_finalize = active.hf_transaction._finalize_completed
            def atomic(path, payload):
                torch.save({**payload, "cp_bound": True}, path)
            active._variant_atomic_torch_save = atomic
            active.resolve_run_paths = lambda args: {"final": generic}
            @contextmanager
            def run_lock(args):
                yield Path(name) / "lock"
            active._run_process_lock = run_lock
            @contextmanager
            def original_context(args):
                yield active
            cp = types.SimpleNamespace(
                _screen_transaction_adapter=original_context,
                _formal_entry_barrier=object(),
                _install_atomic_epoch500_pause=object(),
                architecture_api=lambda: {
                    "validate_irstd_cp_hf_s2_v1": mock.Mock(
                        return_value={"insertion_point": "after_up_decoder2_finish_before_gt2_and_up_decoder1"}
                    )
                },
            )
            args = types.SimpleNamespace(run_seed=42, epochs=1000, resume=True)
            with mock.patch.object(adapter, "assert_continuation_entry", return_value={"state": "finalize_1000"}):
                with adapter._cp_entry_guard(cp, active):
                    with cp._screen_transaction_adapter(args) as entered:
                        entered._variant_atomic_torch_save(generic, raw)
                        torch.save({**expected, "epoch": 999}, generic)
                        with self.assertRaises(adapter.Seed42ContinuationError):
                            entered._variant_atomic_torch_save(generic, raw)
            self.assertIs(active.r1._load_resume_state, trusted_loader)
            self.assertIs(active.hf_transaction._finalize_completed, trusted_finalize)
        self.assertFalse(
            adapter._semantic_equal(
                expected,
                {"state": {"weight": torch.tensor([1.0, 2.0])}, "epoch": 999},
            )
        )

    def test_frozen_selector_tie_break_is_not_plain_miou_max(self):
        from experiments import evisirst_zero_margin_selection as selector
        def record(epoch, pd, fa):
            return {
                "epoch": epoch,
                "data_role": "val",
                "mIoU": 0.7,
                "nIoU": 0.6,
                "Pd": pd,
                "Fa": fa,
                "tinyPd": 0.5,
                "loss": 0.4,
                "test_split_accessed": False,
                "test_index_opened": False,
                "test_selected": False,
                "test_selection_supported": False,
            }
        provenance = selector.select_checkpoints(
            [record(1, 0.8, 1e-5), record(2, 0.9, 2e-5)], margin=None
        )
        self.assertEqual(provenance["primary_selected_epoch"], 2)
        self.assertIn("select_checkpoints", adapter._INSPECTION_PROGRAM)

    @pytest.mark.integration
    def test_live_completed_inspection_is_strict_and_read_only(self):
        for route in adapter.ROUTES:
            with self.subTest(route=route):
                report = adapter.inspect_route_state(route)
                self.assertEqual(report["state"], "complete")
                self.assertEqual(report["committed_epoch"], 1000)
                self.assertEqual(report["recoverable_temporary_files"], [])
                self.assertIs(report["test_split_accessed"], False)


if __name__ == "__main__":
    unittest.main()
