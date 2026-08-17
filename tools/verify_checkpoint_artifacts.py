"""Read-only verification for the published checkpoint artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "artifacts" / "checkpoints.json"
MANIFEST_SCHEMA = "evisirst/checkpoint-artifacts/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """Raised when a checkpoint manifest does not satisfy the frozen schema."""


@dataclass(frozen=True)
class ArtifactCheck:
    relative_path: str
    status: str
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_string(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"artifacts[{index}].{field} must be a non-empty string")
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"manifest does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {path}: {error}") from error

    if not isinstance(document, dict) or document.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("manifest artifacts must be a non-empty list")

    validated: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    required_fields = {
        "relative_path",
        "dataset",
        "model",
        "checkpoint_role",
        "sha256",
        "size_bytes",
        "download_url",
        "download_url_status",
    }
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, dict):
            raise ManifestError(f"artifacts[{index}] must be an object")
        missing_fields = sorted(required_fields.difference(raw))
        if missing_fields:
            raise ManifestError(
                f"artifacts[{index}] is missing fields: {', '.join(missing_fields)}"
            )

        relative_path = _nonempty_string(raw["relative_path"], "relative_path", index)
        if "\\" in relative_path:
            raise ManifestError(
                f"artifacts[{index}].relative_path must use POSIX separators"
            )
        parsed_path = PurePosixPath(relative_path)
        if (
            parsed_path.is_absolute()
            or parsed_path == PurePosixPath(".")
            or ".." in parsed_path.parts
            or parsed_path.as_posix() != relative_path
        ):
            raise ManifestError(
                f"artifacts[{index}].relative_path must be a normalized relative path"
            )
        if relative_path in observed_paths:
            raise ManifestError(f"duplicate artifact path: {relative_path}")
        observed_paths.add(relative_path)

        for field in ("dataset", "model", "checkpoint_role", "download_url_status"):
            _nonempty_string(raw[field], field, index)
        sha256 = _nonempty_string(raw["sha256"], "sha256", index)
        if SHA256_PATTERN.fullmatch(sha256) is None:
            raise ManifestError(f"artifacts[{index}].sha256 must be lowercase SHA-256")
        size_bytes = raw["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ManifestError(f"artifacts[{index}].size_bytes must be a positive integer")
        download_url = raw["download_url"]
        if download_url is not None and (
            not isinstance(download_url, str) or not download_url.strip()
        ):
            raise ManifestError(
                f"artifacts[{index}].download_url must be null or a non-empty string"
            )
        validated.append(dict(raw))
    return validated


def verify_artifacts(
    artifacts: Sequence[dict[str, Any]],
    *,
    root: Path,
) -> list[ArtifactCheck]:
    checks: list[ArtifactCheck] = []
    root = root.resolve()
    for artifact in artifacts:
        relative_path = str(artifact["relative_path"])
        path = root.joinpath(*PurePosixPath(relative_path).parts)
        if not path.exists():
            checks.append(ArtifactCheck(relative_path, "missing", "file is absent"))
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            checks.append(
                ArtifactCheck(
                    relative_path,
                    "invalid",
                    "resolved path escapes the declared artifact root",
                )
            )
            continue
        if resolved != path.absolute() or path.is_symlink() or not path.is_file():
            checks.append(
                ArtifactCheck(
                    relative_path,
                    "invalid",
                    "path is not a direct regular file (symlinks are forbidden)",
                )
            )
            continue

        observed_size = path.stat().st_size
        expected_size = int(artifact["size_bytes"])
        if observed_size != expected_size:
            checks.append(
                ArtifactCheck(
                    relative_path,
                    "invalid",
                    f"size mismatch: expected {expected_size}, observed {observed_size}",
                )
            )
            continue

        observed_sha256 = sha256_file(path)
        expected_sha256 = str(artifact["sha256"])
        if observed_sha256 != expected_sha256:
            checks.append(
                ArtifactCheck(
                    relative_path,
                    "invalid",
                    "sha256 mismatch: "
                    f"expected {expected_sha256}, observed {observed_sha256}",
                )
            )
            continue
        checks.append(ArtifactCheck(relative_path, "ok", "size and SHA-256 match"))
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify checkpoint size and SHA-256 without modifying any files."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"manifest path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help=f"repository/artifact root (default: {REPOSITORY_ROOT})",
    )
    missing_mode = parser.add_mutually_exclusive_group()
    missing_mode.add_argument(
        "--require",
        action="store_true",
        help="fail when any declared artifact is missing",
    )
    missing_mode.add_argument(
        "--allow-missing",
        dest="require",
        action="store_false",
        help="allow missing files (the default); existing files remain strictly checked",
    )
    parser.set_defaults(require=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifacts = load_manifest(args.manifest)
        checks = verify_artifacts(artifacts, root=args.root)
    except ManifestError as error:
        print(f"MANIFEST ERROR: {error}", file=sys.stderr)
        return 2

    for check in checks:
        if check.status == "ok":
            label = "OK"
        elif check.status == "missing":
            label = "MISSING"
        else:
            label = "ERROR"
        print(f"{label} {check.relative_path}: {check.detail}")

    ok_count = sum(check.status == "ok" for check in checks)
    missing_count = sum(check.status == "missing" for check in checks)
    invalid_count = sum(check.status == "invalid" for check in checks)
    print(
        "SUMMARY "
        f"total={len(checks)} ok={ok_count} missing={missing_count} "
        f"invalid={invalid_count} require_missing={'yes' if args.require else 'no'}"
    )
    if invalid_count or (args.require and missing_count):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
