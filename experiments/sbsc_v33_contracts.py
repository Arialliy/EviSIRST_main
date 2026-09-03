"""Fail-closed artifact contracts shared by C3-SBSC V3.3 tools.

This module contains no training or evaluation code.  It verifies repository-
relative regular paths before resolving them, hashes physical authorities, and
provides atomic write-once JSON for configs and launch artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[1]
BASELINE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sbsc_v33_baseline_authority_manifest.json"
)
GRADIENT_AUTHORIZATION_PATH = (
    PROJECT_ROOT / "experiments" / "sbsc_v33_gradient_authorization.json"
)
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
EXPECTED_SAMPLE_COUNTS = {
    "NUAA-SIRST": 214,
    "NUDT-SIRST": 664,
    "IRSTD-1K": 201,
}
EXPECTED_NORMALIZATION = {
    "NUAA-SIRST": {
        "mean": 101.06385040283203,
        "std": 34.619606018066406,
    },
    "NUDT-SIRST": {
        "mean": 107.80905151367188,
        "std": 33.02274703979492,
    },
    "IRSTD-1K": {
        "mean": 87.4661865234375,
        "std": 39.71953201293945,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _require_finite_json_tree(value: Any, *, path: str = "$") -> None:
    """Reject numeric overflow that the stdlib decoder materializes as inf."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_json_tree(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_json_tree(nested, path=f"{path}[{index}]")


def load_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"JSON path is not a regular file: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError("JSON artifact must contain an object")
    _require_finite_json_tree(value)
    return value


def _require_exact_authority_file(path: Path, expected: Path, *, name: str) -> Path:
    """Reject aliases and symlinks before resolving a frozen authority path."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"{name} is not a regular non-symlink file")
    if candidate != expected:
        raise ValueError(f"{name} path differs from authority")
    resolved = candidate.resolve(strict=True)
    if resolved != expected.resolve(strict=True):
        raise ValueError(f"{name} resolves away from authority")
    return candidate


def require_repository_relative_regular_file(relative: str) -> Path:
    if type(relative) is not str or not relative:
        raise TypeError("repository path must be a nonempty string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError("repository path must be canonical and relative")
    if pure.as_posix() != relative or any(part in ("", "/") for part in pure.parts):
        raise ValueError("repository path must be canonical POSIX text")
    current = PROJECT_ROOT
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"repository path traverses a symlink: {relative}")
    if not current.is_file():
        raise FileNotFoundError(f"repository file is unavailable: {relative}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("repository file resolves outside project root") from exc
    return resolved


def repository_relative_path(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("artifact is outside project root") from exc


def validated_output_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        relative = candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("output path must remain below project root") from exc
    if ".." in relative.parts or not relative.parts:
        raise ValueError("output path is malformed")
    current = PROJECT_ROOT
    for part in relative.parent.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("output parent traverses a symlink")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.is_symlink():
        raise ValueError("output destination cannot be a symlink")
    return candidate


def write_once_json(path: Path, payload: Mapping[str, Any]) -> str:
    destination = validated_output_path(path)
    content = canonical_json_bytes(dict(payload))
    digest = hashlib.sha256(content).hexdigest()
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError("existing artifact is not a regular file")
        if destination.read_bytes() != content:
            raise FileExistsError(f"write-once artifact differs: {destination}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def _nonnegative(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_baseline_evaluation(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    expected_checkpoint_path: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    exact = {
        "schema": "evisirst_public_evaluation/v1",
        "model": "baseline",
        "training_dataset": dataset,
        "evaluation_dataset": dataset,
        "dataset": dataset,
        "sample_count": EXPECTED_SAMPLE_COUNTS[dataset],
        "threshold": 0.5,
        "threshold_operator": ">",
        "match_radius": 3.0,
        "match_radius_operator": "<",
        "tiny_area": 9,
        "protocol": "evisirst_public_common_evaluator_v1",
        "normalization_dataset": dataset,
        "normalization": EXPECTED_NORMALIZATION[dataset],
    }
    for key, value in exact.items():
        if payload.get(key) != value:
            raise ValueError(f"baseline {dataset} field {key!r} differs")
    source = payload.get("dataset_source")
    if source != {
        "kind": "external_cli_dataset_root",
        "root_recorded": False,
        "dataset": dataset,
    }:
        raise ValueError("baseline dataset source differs")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("baseline checkpoint record is missing")
    if (
        checkpoint.get("path") != expected_checkpoint_path
        or checkpoint.get("path_scope") != "repository-relative"
        or checkpoint.get("role") != "baseline"
        or checkpoint.get("sha256") != expected_checkpoint_sha256
        or type(checkpoint.get("epoch")) is not int
        or checkpoint["epoch"] < 1
        or type(checkpoint.get("source_selection")) is not str
        or type(checkpoint.get("selection_is_optimistic")) is not bool
    ):
        raise ValueError("baseline checkpoint identity differs")
    provenance = checkpoint.get("selection_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema")
        != "evisirst_historical_checkpoint_selection/v1"
        or provenance.get("source_checkpoint_sha256")
        != expected_checkpoint_sha256
        or provenance.get("unbiased_test_claim_supported") is not False
    ):
        raise ValueError("baseline selection provenance differs")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("baseline metrics are missing")
    probabilities = (
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
    )
    validated_metrics = {
        name: _probability(metrics.get(name), name=f"metrics.{name}")
        for name in probabilities
    }
    validated_metrics["test_loss"] = _nonnegative(
        metrics.get("test_loss"),
        name="metrics.test_loss",
    )
    validated_metrics["fa"] = _nonnegative(
        metrics.get("fa"),
        name="metrics.fa",
    )
    validated_metrics["false_objects_per_image"] = _nonnegative(
        metrics.get("false_objects_per_image"),
        name="metrics.false_objects_per_image",
    )
    count_names = (
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    )
    for name in count_names:
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"metrics.{name} must be a nonnegative int")
        validated_metrics[name] = value
    if (
        validated_metrics["matched_target_count"]
        > validated_metrics["target_count"]
        or validated_metrics["matched_tiny_target_count"]
        > validated_metrics["tiny_target_count"]
        or validated_metrics["unmatched_predicted_object_count"]
        > validated_metrics["predicted_object_count"]
    ):
        raise ValueError("baseline metric counts are inconsistent")
    return {
        "dataset": dataset,
        "sample_count": EXPECTED_SAMPLE_COUNTS[dataset],
        "protocol": payload["protocol"],
        "threshold": payload["threshold"],
        "threshold_operator": payload["threshold_operator"],
        "match_radius": payload["match_radius"],
        "match_radius_operator": payload["match_radius_operator"],
        "tiny_area": payload["tiny_area"],
        "normalization": dict(payload["normalization"]),
        "checkpoint": {
            "path": checkpoint["path"],
            "epoch": checkpoint["epoch"],
            "sha256": checkpoint["sha256"],
            "source_selection": checkpoint["source_selection"],
            "selection_is_optimistic": checkpoint[
                "selection_is_optimistic"
            ],
            "selection_provenance": dict(provenance),
        },
        "metrics": validated_metrics,
    }


def load_baseline_authority_manifest(
    path: Path = BASELINE_MANIFEST_PATH,
) -> dict[str, Any]:
    path = _require_exact_authority_file(
        path,
        BASELINE_MANIFEST_PATH,
        name="baseline manifest",
    )
    manifest = load_strict_json(path)
    exact = {
        "schema": "sctransnet_sbsc_v33/baseline_authority_manifest/v1",
        "status": "frozen",
        "kind": "index_only_no_duplicated_metrics",
        "repository_root_scope": "repository-relative",
        "authority_root": (
            "baseline/evaluation/common_evaluator_v1_recheck_20260818"
        ),
        "dataset_order": list(DATASETS),
        "write_once": True,
    }
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise ValueError(f"baseline manifest field {key!r} differs")
    entries = manifest.get("datasets")
    if not isinstance(entries, Mapping) or set(entries) != set(DATASETS):
        raise ValueError("baseline manifest dataset set differs")
    authorities: dict[str, Any] = {}
    for dataset in DATASETS:
        entry = entries[dataset]
        if not isinstance(entry, Mapping) or set(entry) != {
            "evaluation_path",
            "evaluation_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
        }:
            raise ValueError(f"baseline manifest entry differs for {dataset}")
        expected_evaluation_path = (
            "baseline/evaluation/common_evaluator_v1_recheck_20260818/"
            + dataset
            + ".json"
        )
        expected_checkpoint_path = (
            f"baseline/checkpoints/{dataset}/SCTransNet.pth.tar"
        )
        if (
            entry["evaluation_path"] != expected_evaluation_path
            or entry["checkpoint_path"] != expected_checkpoint_path
        ):
            raise ValueError("baseline authority path differs")
        evaluation_path = require_repository_relative_regular_file(
            entry["evaluation_path"]
        )
        checkpoint_path = require_repository_relative_regular_file(
            entry["checkpoint_path"]
        )
        if sha256_file(evaluation_path) != entry["evaluation_sha256"]:
            raise ValueError("baseline evaluation SHA-256 differs")
        if sha256_file(checkpoint_path) != entry["checkpoint_sha256"]:
            raise ValueError("baseline physical checkpoint SHA-256 differs")
        evaluation = _validate_baseline_evaluation(
            load_strict_json(evaluation_path),
            dataset=dataset,
            expected_checkpoint_path=entry["checkpoint_path"],
            expected_checkpoint_sha256=entry["checkpoint_sha256"],
        )
        evaluation.update(
            {
                "evaluation_path": entry["evaluation_path"],
                "evaluation_sha256": entry["evaluation_sha256"],
                "physical_checkpoint_sha256_verified": True,
            }
        )
        authorities[dataset] = evaluation
    return {
        "schema": manifest["schema"],
        "status": manifest["status"],
        "manifest_path": repository_relative_path(path),
        "manifest_sha256": sha256_file(path),
        "dataset_order": list(DATASETS),
        "authorities": authorities,
    }


def load_gradient_authorization(
    path: Path = GRADIENT_AUTHORIZATION_PATH,
) -> dict[str, Any]:
    path = _require_exact_authority_file(
        path,
        GRADIENT_AUTHORIZATION_PATH,
        name="gradient authorization",
    )
    authorization = load_strict_json(path)
    exact = {
        "schema": "sctransnet_sbsc_v33/gradient_authorization/v1",
        "status": "frozen",
        "scope": "global_three_dataset_v33",
        "architecture": "SCTransNet-C3-SBSC-V3.3",
        "architecture_seed": 42,
        "run_seed": 42,
        "dataset_role": "train",
        "test_loader_constructed": False,
        "optimizer_steps": 0,
        "write_once": True,
    }
    for key, value in exact.items():
        if authorization.get(key) != value:
            raise ValueError(f"gradient authorization field {key!r} differs")
    mode = authorization.get("authorized_router_value_gradient_mode")
    if mode not in ("live", "detached"):
        raise ValueError("authorized gradient mode differs")
    for name in ("report_path", "rules_path"):
        artifact = require_repository_relative_regular_file(authorization[name])
        expected_sha = authorization[name.replace("path", "sha256")]
        if sha256_file(artifact) != expected_sha:
            raise ValueError(f"gradient {name} SHA-256 differs")
    report_path = require_repository_relative_regular_file(
        authorization["report_path"]
    )
    report = load_strict_json(report_path)
    if (
        report.get("schema") != "sctransnet_sbsc_v33/gradient_diagnostic/v1"
        or report.get("status") != "complete"
        or report.get("decision") != authorization.get("decision")
        or report.get("decision", {}).get(
            "authorized_router_value_gradient_mode"
        )
        != mode
        or report.get("test_loader_constructed") is not False
        or report.get("optimizer_steps") != 0
    ):
        raise ValueError("gradient diagnostic report differs from authorization")
    source_manifest = report.get("source_manifest")
    if (
        not isinstance(source_manifest, Mapping)
        or source_manifest.get("sha256")
        != authorization.get("source_manifest_sha256")
        or not isinstance(source_manifest.get("files"), list)
    ):
        raise ValueError("gradient source manifest differs")
    records = []
    for record in source_manifest["files"]:
        if not isinstance(record, Mapping):
            raise TypeError("gradient source record is malformed")
        source_path = require_repository_relative_regular_file(record["path"])
        if (
            sha256_file(source_path) != record.get("sha256")
            or source_path.stat().st_size != record.get("size_bytes")
        ):
            raise ValueError("authorized gradient source file drifted")
        records.append(dict(record))
    if canonical_sha256(records) != source_manifest["sha256"]:
        raise ValueError("gradient source manifest digest differs")
    return {
        "schema": authorization["schema"],
        "status": authorization["status"],
        "authorization_path": repository_relative_path(path),
        "authorization_sha256": sha256_file(path),
        "authorized_router_value_gradient_mode": mode,
        "report_path": authorization["report_path"],
        "report_sha256": authorization["report_sha256"],
        "rules_path": authorization["rules_path"],
        "rules_sha256": authorization["rules_sha256"],
        "source_manifest_sha256": source_manifest["sha256"],
        "decision": authorization["decision"],
    }


def build_source_manifest(relative_paths: Sequence[str]) -> dict[str, Any]:
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("source manifest paths must be unique and nonempty")
    records = []
    for relative in relative_paths:
        path = require_repository_relative_regular_file(relative)
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema": "sctransnet_sbsc_v33/source_manifest/v1",
        "files": records,
        "sha256": canonical_sha256(records),
    }


__all__ = [
    "BASELINE_MANIFEST_PATH",
    "DATASETS",
    "GRADIENT_AUTHORIZATION_PATH",
    "PROJECT_ROOT",
    "build_source_manifest",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_baseline_authority_manifest",
    "load_gradient_authorization",
    "load_strict_json",
    "repository_relative_path",
    "require_repository_relative_regular_file",
    "sha256_file",
    "validated_output_path",
    "write_once_json",
]
