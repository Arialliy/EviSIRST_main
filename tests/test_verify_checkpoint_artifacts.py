from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools import verify_checkpoint_artifacts as checker


class VerifyCheckpointArtifactsTest(unittest.TestCase):
    @staticmethod
    def _manifest(path: Path, *, relative_path: str, payload: bytes) -> Path:
        manifest = {
            "schema": checker.MANIFEST_SCHEMA,
            "artifacts": [
                {
                    "relative_path": relative_path,
                    "dataset": "fixture-dataset",
                    "model": "fixture-model",
                    "checkpoint_role": "fixture",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "download_url": None,
                    "download_url_status": "TBD",
                }
            ],
        }
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    @staticmethod
    def _run(*arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = checker.main(list(arguments))
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_valid_temporary_artifact_passes_required_mode(self) -> None:
        payload = b"temporary checkpoint fixture"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "weights" / "fixture.pth.tar"
            artifact.parent.mkdir()
            artifact.write_bytes(payload)
            manifest = self._manifest(
                root / "manifest.json",
                relative_path="weights/fixture.pth.tar",
                payload=payload,
            )
            code, stdout, stderr = self._run(
                "--manifest",
                str(manifest),
                "--root",
                str(root),
                "--require",
            )
        self.assertEqual(code, 0)
        self.assertIn("OK weights/fixture.pth.tar", stdout)
        self.assertIn("ok=1 missing=0 invalid=0", stdout)
        self.assertEqual(stderr, "")

    def test_missing_is_allowed_by_default_and_fails_when_required(self) -> None:
        payload = b"artifact that is intentionally absent"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(
                root / "manifest.json",
                relative_path="weights/missing.pth.tar",
                payload=payload,
            )
            default_code, default_stdout, _ = self._run(
                "--manifest", str(manifest), "--root", str(root)
            )
            allowed_code, _, _ = self._run(
                "--manifest", str(manifest), "--root", str(root), "--allow-missing"
            )
            required_code, required_stdout, _ = self._run(
                "--manifest", str(manifest), "--root", str(root), "--require"
            )
        self.assertEqual(default_code, 0)
        self.assertEqual(allowed_code, 0)
        self.assertEqual(required_code, 1)
        self.assertIn("MISSING weights/missing.pth.tar", default_stdout)
        self.assertIn("missing=1", required_stdout)

    def test_existing_size_mismatch_always_fails(self) -> None:
        expected = b"expected artifact bytes"
        observed = b"short"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "weights" / "fixture.pth.tar"
            artifact.parent.mkdir()
            artifact.write_bytes(observed)
            manifest = self._manifest(
                root / "manifest.json",
                relative_path="weights/fixture.pth.tar",
                payload=expected,
            )
            code, stdout, _ = self._run(
                "--manifest", str(manifest), "--root", str(root), "--allow-missing"
            )
        self.assertEqual(code, 1)
        self.assertIn("ERROR weights/fixture.pth.tar: size mismatch", stdout)
        self.assertIn("invalid=1", stdout)

    def test_existing_hash_mismatch_always_fails(self) -> None:
        expected = b"expected"
        observed = b"observed"
        self.assertEqual(len(expected), len(observed))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "weights" / "fixture.pth.tar"
            artifact.parent.mkdir()
            artifact.write_bytes(observed)
            manifest = self._manifest(
                root / "manifest.json",
                relative_path="weights/fixture.pth.tar",
                payload=expected,
            )
            code, stdout, _ = self._run(
                "--manifest", str(manifest), "--root", str(root), "--allow-missing"
            )
        self.assertEqual(code, 1)
        self.assertIn("ERROR weights/fixture.pth.tar: sha256 mismatch", stdout)
        self.assertIn("invalid=1", stdout)

    def test_manifest_rejects_parent_traversal(self) -> None:
        payload = b"fixture"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(
                root / "manifest.json",
                relative_path="../outside.pth.tar",
                payload=payload,
            )
            code, _, stderr = self._run(
                "--manifest", str(manifest), "--root", str(root)
            )
        self.assertEqual(code, 2)
        self.assertIn("normalized relative path", stderr)

    def test_parent_symlink_cannot_escape_artifact_root(self) -> None:
        payload = b"outside artifact"
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "fixture.pth.tar").write_bytes(payload)
            (root / "weights").symlink_to(outside, target_is_directory=True)
            manifest = self._manifest(
                root / "manifest.json",
                relative_path="weights/fixture.pth.tar",
                payload=payload,
            )
            code, stdout, _ = self._run(
                "--manifest", str(manifest), "--root", str(root)
            )
        self.assertEqual(code, 1)
        self.assertIn("resolved path escapes", stdout)


if __name__ == "__main__":
    unittest.main()
