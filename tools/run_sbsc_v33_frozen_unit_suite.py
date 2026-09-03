#!/usr/bin/env python3
"""Run the frozen C3-SBSC V3.1/V3.2/V3.3 regression whitelist once.

The command writes one immutable machine-readable report.  It never turns a
missing test, an unparseable pytest summary, or a non-zero pytest exit into a
PASS.  Pytest output is not copied into the report; its exact bytes are bound
by SHA-256.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


ENTRY = Path(__file__)
if ENTRY.is_symlink() or not ENTRY.is_file():
    raise RuntimeError("unit-suite entry point must be a regular file")
PROJECT_ROOT = ENTRY.resolve(strict=True).parents[1]
if ENTRY.resolve(strict=True) != PROJECT_ROOT / "tools" / ENTRY.name:
    raise RuntimeError("unit-suite entry point resolves away from tools")
sys.path[:] = [str(PROJECT_ROOT), *[item for item in sys.path if item not in ("", str(PROJECT_ROOT))]]

import torch
import pytest

from experiments import sbsc_v33_contracts as contracts


REPORT_SCHEMA = "sctransnet_sbsc_v33/unit_test_report/v1"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "sbsc_v33_preflight" / "unit_test_report.json"
PYTHONHASHSEED = "42"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_PYTEST_VERSION = "9.0.2"
# Frozen from the exact TEST_FILES argv with the isolated environment and
# pytest 9.0.2: ``--collect-only`` observed 607 tests on 2026-08-31 after
# adding the fail-closed post-write, formal-launch closure, and frozen
# Resource-V2 physical-digest regressions.
EXPECTED_PASSED_COUNT = 607

# This is deliberately a file whitelist, not a directory or discovery glob.
# A renamed/deleted regression therefore blocks the preflight rather than
# silently reducing coverage.
TEST_FILES = (
    # V3.1 core, diagnostic, trainer, and selector.
    "tests/test_sctransnet_sbsc_v31.py",
    "tests/test_diagnose_sctransnet_sbsc_v31.py",
    "tests/test_train_sctransnet_sbsc_v31_validation.py",
    "tests/test_sbsc_v31_selection.py",
    # V3.2 core, train-only smoke, both trainers, and both selectors.
    "tests/test_sctransnet_sbsc_v32.py",
    "tests/test_smoke_sctransnet_sbsc_v32_train_only.py",
    "tests/test_train_sctransnet_sbsc_v32_validation.py",
    "tests/test_train_sctransnet_sbsc_v32_img_idx_test_selected.py",
    "tests/test_sbsc_v32_selection.py",
    "tests/test_sbsc_v32_test_selection.py",
    # Complete V3.3 architecture and preflight/formal artifact chain.
    "tests/test_sctransnet_sbsc_v33.py",
    "tests/test_sbsc_v33_contracts.py",
    "tests/test_sbsc_v33_gradient_authorization.py",
    "tests/test_sbsc_v33_canary.py",
    "tests/test_sbsc_v33_canary_v2.py",
    "tests/test_sbsc_v33_test_selection.py",
    "tests/test_train_sctransnet_sbsc_v33_img_idx_test_selected.py",
    "tests/test_finalize_sbsc_v33_results.py",
    "tests/test_sbsc_v33_resource_benchmark.py",
    "tests/test_sbsc_v33_resource_benchmark_v2.py",
    "tests/test_run_sbsc_v33_frozen_unit_suite.py",
    "tests/test_authorize_sbsc_v33_formal_launch.py",
)

RELATED_SOURCE_FILES = (
    "tools/run_sbsc_v33_frozen_unit_suite.py",
    "tools/authorize_sbsc_v33_formal_launch.py",
    "tools/benchmark_sbsc_v33_resources.py",
    "tools/benchmark_sbsc_v33_resources_v2.py",
    "tools/diagnose_sctransnet_sbsc_v31.py",
    "tools/smoke_sctransnet_sbsc_v32_train_only.py",
    "tools/diagnose_sbsc_v33_gradient_conflict.py",
    "tools/run_sbsc_v33_canary.py",
    "tools/run_sbsc_v33_canary_v2.py",
    "tools/freeze_sbsc_v33_irstd_formal_method_config.py",
    "tools/finalize_sbsc_v33_results.py",
    "train_sctransnet_sbsc_v31_validation.py",
    "train_sctransnet_sbsc_v32_validation.py",
    "train_sctransnet_sbsc_v32_img_idx_test_selected.py",
    "train_sctransnet_sbsc_v33_img_idx_test_selected.py",
    "train.py",
    "test.py",
    "experiments/sbsc_v31_selection.py",
    "experiments/sbsc_v32_selection.py",
    "experiments/sbsc_v32_test_selection.py",
    "experiments/sbsc_v33_test_selection.py",
    "experiments/sbsc_v33_contracts.py",
    "experiments/sctransnet_sbsc_v31.py",
    "experiments/sctransnet_sbsc_v32.py",
    "experiments/sctransnet_sbsc_v33.py",
    "experiments/evisirst_data.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/four_dataset_models_seed42_v1.py",
    "experiments/sbsc_v33_baseline_authority_manifest.json",
    "baseline/evaluation/common_evaluator_v1_recheck_20260818/NUAA-SIRST.json",
    "baseline/evaluation/common_evaluator_v1_recheck_20260818/NUDT-SIRST.json",
    "baseline/evaluation/common_evaluator_v1_recheck_20260818/IRSTD-1K.json",
    "experiments/sbsc_v33_gradient_authorization_rules.json",
    "experiments/sbsc_v33_gradient_authorization.json",
    "experiments/sbsc_v33_canary_rules.json",
    "experiments/sbsc_v33_canary_rules_v2.json",
    "experiments/sbsc_v33_resource_benchmark_rules.json",
    "experiments/sbsc_v33_resource_benchmark_rules_v2.json",
    "experiments/sbsc_v33_methods/sbsc_v33_third_irstd_formal.json",
    "artifacts/sbsc_v33_preflight/gradient_diagnostic.json",
    "runs/sbsc_v33_third/canary/IRSTD-1K/manifest.json",
    "runs/sbsc_v33_third/canary/IRSTD-1K/report.json",
    "runs/sbsc_v33_third/canary_v2/IRSTD-1K/manifest.json",
    "runs/sbsc_v33_third/canary_v2/IRSTD-1K/report.json",
    "artifacts/sbsc_v33_preflight/paired_resource_benchmark.json",
    "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json",
    "pytest.ini",
)

_COUNT_PATTERN = re.compile(
    r"(?P<count>[0-9]+) (?P<kind>passed|failed|skipped|warnings?|errors?|xfailed|xpassed|deselected)"
)
_FINAL_SUMMARY_PATTERN = re.compile(
    r"^(?:=+\s*)?"
    r"(?P<body>[0-9]+ (?:passed|failed|skipped|warnings?|errors?|xfailed|xpassed|deselected)"
    r"(?:, [0-9]+ (?:passed|failed|skipped|warnings?|errors?|xfailed|xpassed|deselected))*)"
    r" in [0-9]+(?:\.[0-9]+)?s"
    r"(?: \([0-9]+:[0-5][0-9]:[0-5][0-9]\))?"
    r"(?:\s*=+)?$"
)


class FrozenUnitSuiteError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lower_sha(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FrozenUnitSuiteError(f"{label} is not a lowercase SHA-256")
    return value


def _regular_record(relative: str) -> dict[str, Any]:
    path = contracts.require_repository_relative_regular_file(relative)
    return {
        "path": relative,
        "sha256": contracts.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _source_snapshot(expected_paths: Sequence[str]) -> dict[str, Any]:
    if not expected_paths or len(expected_paths) != len(set(expected_paths)):
        raise FrozenUnitSuiteError("source snapshot path set is malformed")
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in expected_paths:
        candidate = PROJECT_ROOT / relative
        if candidate.is_symlink() or not candidate.is_file():
            missing.append(relative)
        else:
            records.append(_regular_record(relative))
    body = {
        "expected_files": list(expected_paths),
        "existing_files": records,
        "missing_files": missing,
    }
    return {
        "schema": "sctransnet_sbsc_v33/unit_source_snapshot/v1",
        **body,
        "sha256": contracts.canonical_sha256(body),
    }


_IGNORED_SOURCE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "env",
    "venv",
}


def _repository_python_sources() -> tuple[str, ...]:
    """Return every normal repository Python source, not a guessed import set."""

    return tuple(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted(PROJECT_ROOT.rglob("*.py"))
        if path.is_file()
        and not path.is_symlink()
        and not any(part in _IGNORED_SOURCE_PARTS for part in path.parts)
    )


def expected_source_files(
    test_files: Sequence[str] = TEST_FILES,
    related_source_files: Sequence[str] = RELATED_SOURCE_FILES,
) -> tuple[str, ...]:
    if len(test_files) != len(set(test_files)) or len(related_source_files) != len(
        set(related_source_files)
    ):
        raise FrozenUnitSuiteError("source seed paths must be unique")
    return tuple(
        sorted({*test_files, *related_source_files, *_repository_python_sources()})
    )


def pytest_argv(test_files: Sequence[str] = TEST_FILES) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "--color=no",
        "--strict-config",
        "--strict-markers",
        "-p",
        "no:cacheprovider",
        "-ra",
        "-q",
        *test_files,
    ]


def frozen_subprocess_environment() -> tuple[dict[str, str], dict[str, Any]]:
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment.pop("PYTHONSTARTUP", None)
    environment.pop("PYTHONUSERBASE", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONHASHSEED"] = PYTHONHASHSEED
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONWARNINGS"] = "default"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    selected = {
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"],
        "PYTHONHASHSEED": environment["PYTHONHASHSEED"],
        "CUBLAS_WORKSPACE_CONFIG": environment["CUBLAS_WORKSPACE_CONFIG"],
        "PYTHONPATH": environment["PYTHONPATH"],
        "PYTHONNOUSERSITE": environment["PYTHONNOUSERSITE"],
        "PYTHONDONTWRITEBYTECODE": environment["PYTHONDONTWRITEBYTECODE"],
        "PYTHONWARNINGS": environment["PYTHONWARNINGS"],
        "PYTHONUTF8": environment["PYTHONUTF8"],
        "PYTHONIOENCODING": environment["PYTHONIOENCODING"],
        "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
    }
    return environment, selected


def runtime_identity(selected_environment: Mapping[str, Any]) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                }
            )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pytest": pytest.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": len(devices),
        "cuda_devices": devices,
        "device_type": "cuda" if cuda_available else "cpu",
        "cublas_workspace_config": selected_environment["CUBLAS_WORKSPACE_CONFIG"],
        "selected_environment": dict(selected_environment),
    }


def parse_pytest_summary(stdout: bytes) -> dict[str, Any]:
    text = stdout.decode("utf-8", errors="replace")
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = [
        line for line in nonempty_lines if _FINAL_SUMMARY_PATTERN.fullmatch(line)
    ]
    summary_line = candidates[0] if len(candidates) == 1 else None
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "warnings": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
    }
    if summary_line is not None and nonempty_lines[-1] == summary_line:
        matches = list(_COUNT_PATTERN.finditer(summary_line))
        normalized_kinds: list[str] = []
        for match in matches:
            kind = match.group("kind")
            if kind == "warning":
                kind = "warnings"
            elif kind == "error":
                kind = "errors"
            normalized_kinds.append(kind)
            counts[kind] = int(match.group("count"))
        if len(normalized_kinds) != len(set(normalized_kinds)):
            summary_line = None
            counts = {key: 0 for key in counts}
    else:
        summary_line = None
    return {
        **counts,
        "summary_parse_valid": summary_line is not None,
        "summary_candidate_count": len(candidates),
        "summary_line_sha256": (
            _sha256_bytes(summary_line.encode("utf-8"))
            if summary_line is not None
            else None
        ),
    }


def _default_command_runner(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=False,
    )


def validate_unit_report(
    report: Mapping[str, Any],
    *,
    expected_tests: Sequence[str] = TEST_FILES,
    expected_related_sources: Sequence[str] = RELATED_SOURCE_FILES,
    expected_passed_count: int = EXPECTED_PASSED_COUNT,
    replay_physical_sources: bool = True,
) -> None:
    expected_fields = {
        "schema",
        "status",
        "write_once",
        "scope",
        "argv",
        "working_directory",
        "execution_started",
        "exit_code",
        "elapsed_seconds",
        "test_files",
        "expected_test_files",
        "missing_files",
        "expected_source_files",
        "source_snapshot",
        "source_manifest",
        "source_manifest_sha256",
        "source_snapshot_phase",
        "post_source_snapshot_sha256",
        "post_source_file_count",
        "source_unchanged_during_execution",
        "runtime_identity",
        "runtime_identity_sha256",
        "selected_environment",
        "counts",
        "expected_collected_count",
        "expected_passed_count",
        "collected_count",
        "summary_parse_valid",
        "summary_candidate_count",
        "summary_line_sha256",
        "stdout_sha256",
        "stdout_size_bytes",
        "stdout_base64",
        "execution_error",
    }
    if set(report) != expected_fields:
        raise FrozenUnitSuiteError("unit report field set differs")
    if report.get("schema") != REPORT_SCHEMA or report.get("write_once") is not True:
        raise FrozenUnitSuiteError("unit report identity differs")
    if (
        report.get("scope") != "frozen_v31_v32_v33_regression_whitelist"
        or report.get("working_directory") != "."
        or report.get("expected_test_files") != list(expected_tests)
    ):
        raise FrozenUnitSuiteError("unit report frozen scope differs")
    if report.get("status") not in ("PASS", "FAIL"):
        raise FrozenUnitSuiteError("unit report status differs")
    if report.get("argv") != pytest_argv(expected_tests):
        raise FrozenUnitSuiteError("unit report argv differs")
    expected_sources = expected_source_files(expected_tests, expected_related_sources)
    if report.get("expected_source_files") != list(expected_sources):
        raise FrozenUnitSuiteError("unit report expected source set differs")
    snapshot = report.get("source_snapshot")
    if not isinstance(snapshot, Mapping) or set(snapshot) != {
        "schema",
        "expected_files",
        "existing_files",
        "missing_files",
        "sha256",
    }:
        raise FrozenUnitSuiteError("unit report source snapshot is malformed")
    snapshot_body = {
        "expected_files": snapshot.get("expected_files"),
        "existing_files": snapshot.get("existing_files"),
        "missing_files": snapshot.get("missing_files"),
    }
    if (
        snapshot.get("schema")
        != "sctransnet_sbsc_v33/unit_source_snapshot/v1"
        or snapshot.get("expected_files") != list(expected_sources)
        or not isinstance(snapshot.get("existing_files"), list)
        or not isinstance(snapshot.get("missing_files"), list)
        or contracts.canonical_sha256(snapshot_body) != snapshot.get("sha256")
        or report.get("missing_files") != snapshot.get("missing_files")
    ):
        raise FrozenUnitSuiteError("unit report source snapshot identity differs")
    missing_set = set(snapshot["missing_files"])
    if (
        len(missing_set) != len(snapshot["missing_files"])
        or not missing_set.issubset(expected_sources)
        or [record.get("path") for record in snapshot["existing_files"]]
        != [path for path in expected_sources if path not in missing_set]
    ):
        raise FrozenUnitSuiteError("unit report source snapshot partition differs")
    records = report.get("test_files")
    expected_existing_tests = [path for path in expected_tests if path not in missing_set]
    if (
        not isinstance(records, list)
        or [item.get("path") for item in records] != expected_existing_tests
    ):
        raise FrozenUnitSuiteError("unit report test whitelist differs")
    for record in snapshot["existing_files"]:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise FrozenUnitSuiteError("unit report source record differs")
        _lower_sha(record.get("sha256"), label="test file SHA-256")
        if type(record.get("size_bytes")) is not int or record["size_bytes"] < 0:
            raise FrozenUnitSuiteError("unit report source size differs")
        if replay_physical_sources:
            physical = contracts.require_repository_relative_regular_file(record["path"])
            if (
                contracts.sha256_file(physical) != record["sha256"]
                or physical.stat().st_size != record["size_bytes"]
            ):
                raise FrozenUnitSuiteError("unit report source file drifted")
    record_by_path = {record["path"]: record for record in snapshot["existing_files"]}
    if records != [record_by_path[path] for path in expected_existing_tests]:
        raise FrozenUnitSuiteError("unit report test/source records differ")
    if replay_physical_sources and _source_snapshot(expected_sources) != snapshot:
        raise FrozenUnitSuiteError("unit report physical source snapshot drifted")
    source = report.get("source_manifest")
    if not missing_set:
        if (
            not isinstance(source, Mapping)
            or set(source) != {"schema", "files", "sha256"}
            or source.get("schema") != "sctransnet_sbsc_v33/source_manifest/v1"
            or source.get("files") != snapshot["existing_files"]
            or source.get("sha256")
            != contracts.canonical_sha256(source["files"])
            or report.get("source_manifest_sha256") != source.get("sha256")
        ):
            raise FrozenUnitSuiteError("unit report complete source manifest differs")
    elif source is not None or report.get("source_manifest_sha256") is not None:
        raise FrozenUnitSuiteError("incomplete source snapshot invented a manifest")
    post_sha = _lower_sha(
        report.get("post_source_snapshot_sha256"),
        label="post source snapshot SHA-256",
    )
    unchanged = report.get("source_unchanged_during_execution")
    if (
        report.get("source_snapshot_phase") != "before_pytest_execution"
        or type(report.get("post_source_file_count")) is not int
        or report["post_source_file_count"] < 0
        or type(unchanged) is not bool
        or unchanged
        is not (
            post_sha == snapshot["sha256"]
            and report["post_source_file_count"]
            == len(snapshot["existing_files"])
        )
    ):
        raise FrozenUnitSuiteError("unit report pre/post source replay differs")
    counts = report.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "passed",
        "failed",
        "skipped",
        "warnings",
        "errors",
        "xfailed",
        "xpassed",
        "deselected",
    } or any(type(value) is not int or value < 0 for value in counts.values()):
        raise FrozenUnitSuiteError("unit report pytest counts differ")
    if (
        type(expected_passed_count) is not int
        or report.get("expected_passed_count") != expected_passed_count
        or report.get("expected_collected_count") != expected_passed_count
    ):
        raise FrozenUnitSuiteError("unit report frozen expected count differs")
    collected = sum(
        counts[key]
        for key in (
            "passed",
            "failed",
            "skipped",
            "xfailed",
            "xpassed",
            "deselected",
        )
    )
    if report.get("collected_count") != collected:
        raise FrozenUnitSuiteError("unit report collected count differs")
    stdout_base64 = report.get("stdout_base64")
    if type(stdout_base64) is not str:
        raise FrozenUnitSuiteError("unit report stdout encoding differs")
    try:
        stdout = base64.b64decode(stdout_base64.encode("ascii"), validate=True)
    except Exception as exc:
        raise FrozenUnitSuiteError("unit report stdout is not strict base64") from exc
    stdout_sha = _lower_sha(report.get("stdout_sha256"), label="stdout_sha256")
    parsed = parse_pytest_summary(stdout)
    parsed_counts = {key: parsed[key] for key in counts}
    if (
        _sha256_bytes(stdout) != stdout_sha
        or len(stdout) != report.get("stdout_size_bytes")
        or parsed_counts != dict(counts)
        or parsed["summary_parse_valid"] != report.get("summary_parse_valid")
        or parsed["summary_candidate_count"]
        != report.get("summary_candidate_count")
        or parsed["summary_line_sha256"] != report.get("summary_line_sha256")
    ):
        raise FrozenUnitSuiteError("unit report stdout replay differs")
    elapsed = report.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
        or type(report.get("stdout_size_bytes")) is not int
        or report["stdout_size_bytes"] < 0
    ):
        raise FrozenUnitSuiteError("unit report execution accounting differs")
    selected = report.get("selected_environment")
    identity = report.get("runtime_identity")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(identity, Mapping)
        or identity.get("selected_environment") != selected
        or report.get("runtime_identity_sha256")
        != contracts.canonical_sha256(identity)
        or selected.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1"
        or selected.get("PYTHONHASHSEED") != PYTHONHASHSEED
        or selected.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG
        or selected.get("PYTHONPATH") != str(PROJECT_ROOT)
        or selected.get("PYTHONNOUSERSITE") != "1"
        or selected.get("PYTHONDONTWRITEBYTECODE") != "1"
        or selected.get("PYTHONWARNINGS") != "default"
        or selected.get("PYTHONUTF8") != "1"
        or selected.get("PYTHONIOENCODING") != "utf-8"
        or identity.get("pytest") != EXPECTED_PYTEST_VERSION
    ):
        raise FrozenUnitSuiteError("unit report runtime identity differs")
    if report.get("execution_started") is True:
        error = report.get("execution_error")
        if error is None:
            if type(report.get("exit_code")) is not int:
                raise FrozenUnitSuiteError("unit report exit code differs")
        elif (
            not isinstance(error, Mapping)
            or set(error) != {"exception_type", "message"}
            or type(error.get("exception_type")) is not str
            or type(error.get("message")) is not str
            or report.get("exit_code") is not None
        ):
            raise FrozenUnitSuiteError("unit report execution error differs")
    elif report.get("execution_started") is False:
        if report.get("exit_code") is not None:
            raise FrozenUnitSuiteError("non-executed unit report invented an exit code")
        if report.get("execution_error") is not None:
            raise FrozenUnitSuiteError("non-executed unit report invented an error")
    else:
        raise FrozenUnitSuiteError("unit report execution flag differs")
    pass_condition = (
        report["exit_code"] == 0
        and report.get("execution_error") is None
        and report.get("summary_parse_valid") is True
        and report.get("summary_candidate_count") == 1
        and expected_passed_count > 0
        and counts["passed"] == expected_passed_count
        and collected == expected_passed_count
        and all(
            counts[key] == 0
            for key in (
                "failed",
                "skipped",
                "errors",
                "xfailed",
                "xpassed",
                "deselected",
            )
        )
        and report.get("missing_files") == []
        and report.get("source_unchanged_during_execution") is True
    )
    if (report.get("status") == "PASS") is not pass_condition:
        raise FrozenUnitSuiteError("unit report PASS decision differs")


def run_suite(
    *,
    output_path: Path = REPORT_PATH,
    test_files: Sequence[str] = TEST_FILES,
    related_source_files: Sequence[str] = RELATED_SOURCE_FILES,
    expected_passed_count: int = EXPECTED_PASSED_COUNT,
    command_runner: Callable[..., subprocess.CompletedProcess[bytes]] = _default_command_runner,
) -> dict[str, Any]:
    destination = contracts.validated_output_path(output_path)
    if destination.exists():
        raise FileExistsError("frozen unit-test report already exists")
    expected_sources = expected_source_files(test_files, related_source_files)
    source_snapshot = _source_snapshot(expected_sources)
    missing = list(source_snapshot["missing_files"])
    source_manifest = None
    if not missing:
        source_manifest = {
            "schema": "sctransnet_sbsc_v33/source_manifest/v1",
            "files": list(source_snapshot["existing_files"]),
            "sha256": contracts.canonical_sha256(
                source_snapshot["existing_files"]
            ),
        }
    environment, selected_environment = frozen_subprocess_environment()
    identity = runtime_identity(selected_environment)
    argv = pytest_argv(test_files)
    started = time.monotonic()
    stdout = b""
    exit_code: int | None = None
    execution_started = False
    execution_error: dict[str, str] | None = None
    if not missing:
        execution_started = True
        try:
            completed = command_runner(argv, cwd=PROJECT_ROOT, env=environment)
            if type(completed.returncode) is not int or not isinstance(
                completed.stdout, bytes
            ):
                raise FrozenUnitSuiteError(
                    "pytest command runner returned malformed data"
                )
            exit_code = completed.returncode
            stdout = completed.stdout
        except Exception as exc:
            execution_error = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
    elapsed = time.monotonic() - started
    post_expected_sources = expected_source_files(test_files, related_source_files)
    post_source_snapshot = _source_snapshot(post_expected_sources)
    source_unchanged = post_source_snapshot == source_snapshot
    parsed = parse_pytest_summary(stdout)
    source_records = {
        record["path"]: record for record in source_snapshot["existing_files"]
    }
    test_records = [
        source_records[relative] for relative in test_files if relative in source_records
    ]
    counts = {
        key: parsed[key]
        for key in (
            "passed",
            "failed",
            "skipped",
            "warnings",
            "errors",
            "xfailed",
            "xpassed",
            "deselected",
        )
    }
    pass_condition = (
        not missing
        and execution_error is None
        and exit_code == 0
        and parsed["summary_parse_valid"] is True
        and parsed["summary_candidate_count"] == 1
        and type(expected_passed_count) is int
        and expected_passed_count > 0
        and counts["passed"] == expected_passed_count
        and all(
            counts[key] == 0
            for key in (
                "failed",
                "skipped",
                "errors",
                "xfailed",
                "xpassed",
                "deselected",
            )
        )
        and source_unchanged
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if pass_condition else "FAIL",
        "write_once": True,
        "scope": "frozen_v31_v32_v33_regression_whitelist",
        "argv": argv,
        "working_directory": ".",
        "execution_started": execution_started,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "test_files": test_records,
        "expected_test_files": list(test_files),
        "missing_files": missing,
        "expected_source_files": list(expected_sources),
        "source_snapshot": source_snapshot,
        "source_manifest": source_manifest,
        "source_manifest_sha256": (
            source_manifest["sha256"] if source_manifest is not None else None
        ),
        "source_snapshot_phase": "before_pytest_execution",
        "post_source_snapshot_sha256": post_source_snapshot["sha256"],
        "post_source_file_count": len(post_source_snapshot["existing_files"]),
        "source_unchanged_during_execution": source_unchanged,
        "runtime_identity": identity,
        "runtime_identity_sha256": contracts.canonical_sha256(identity),
        "selected_environment": dict(selected_environment),
        "counts": counts,
        "expected_collected_count": expected_passed_count,
        "expected_passed_count": expected_passed_count,
        "collected_count": sum(
            counts[key]
            for key in (
                "passed",
                "failed",
                "skipped",
                "xfailed",
                "xpassed",
                "deselected",
            )
        ),
        "summary_parse_valid": parsed["summary_parse_valid"],
        "summary_candidate_count": parsed["summary_candidate_count"],
        "summary_line_sha256": parsed["summary_line_sha256"],
        "stdout_sha256": _sha256_bytes(stdout),
        "stdout_size_bytes": len(stdout),
        "stdout_base64": base64.b64encode(stdout).decode("ascii"),
        "execution_error": execution_error,
    }
    expected_bytes = contracts.canonical_json_bytes(report)
    expected_sha256 = contracts.write_once_json(destination, report)
    if (
        contracts.sha256_file(destination) != expected_sha256
        or destination.read_bytes() != expected_bytes
    ):
        raise FrozenUnitSuiteError(
            "write-once unit report physical bytes differ after publication"
        )
    reopened = contracts.load_strict_json(destination)
    if reopened != report:
        raise FrozenUnitSuiteError(
            "write-once unit report strict reload differs from memory"
        )
    validate_unit_report(
        reopened,
        expected_tests=test_files,
        expected_related_sources=related_source_files,
        expected_passed_count=expected_passed_count,
        replay_physical_sources=True,
    )
    return reopened


def main() -> int:
    report = run_suite()
    print(contracts.canonical_json_bytes(report).decode("utf-8"), end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
