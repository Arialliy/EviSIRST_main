from __future__ import annotations

import hashlib
import copy
import subprocess
import tempfile
from pathlib import Path

import pytest

from experiments import sbsc_v33_contracts as contracts
from tools import run_sbsc_v33_frozen_unit_suite as suite


def _fixture_files(directory: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    test_path = directory / "test_fixture.py"
    source_path = directory / "fixture_source.py"
    test_path.write_text("def test_fixture():\n    assert True\n", encoding="utf-8")
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    return (
        (test_path.relative_to(suite.PROJECT_ROOT).as_posix(),),
        (source_path.relative_to(suite.PROJECT_ROOT).as_posix(),),
    )


def _completed(stdout: bytes, returncode: int = 0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=["fixture"], returncode=returncode, stdout=stdout, stderr=None
    )


def test_frozen_whitelist_covers_v31_v32_and_complete_v33_chain() -> None:
    assert len(suite.TEST_FILES) == len(set(suite.TEST_FILES))
    required = {
        "tests/test_sctransnet_sbsc_v31.py",
        "tests/test_diagnose_sctransnet_sbsc_v31.py",
        "tests/test_train_sctransnet_sbsc_v31_validation.py",
        "tests/test_sctransnet_sbsc_v32.py",
        "tests/test_smoke_sctransnet_sbsc_v32_train_only.py",
        "tests/test_train_sctransnet_sbsc_v32_validation.py",
        "tests/test_sctransnet_sbsc_v33.py",
        "tests/test_sbsc_v33_contracts.py",
        "tests/test_sbsc_v33_gradient_authorization.py",
        "tests/test_sbsc_v33_canary_v2.py",
        "tests/test_sbsc_v33_test_selection.py",
        "tests/test_train_sctransnet_sbsc_v33_img_idx_test_selected.py",
        "tests/test_finalize_sbsc_v33_results.py",
        "tests/test_sbsc_v33_resource_benchmark.py",
        "tests/test_sbsc_v33_resource_benchmark_v2.py",
        "tests/test_authorize_sbsc_v33_formal_launch.py",
    }
    assert required <= set(suite.TEST_FILES)
    argv = suite.pytest_argv()
    assert argv[:10] == [
        suite.sys.executable,
        "-m",
        "pytest",
        "--color=no",
        "--strict-config",
        "--strict-markers",
        "-p",
        "no:cacheprovider",
        "-ra",
        "-q",
    ]
    assert argv[10:] == list(suite.TEST_FILES)
    sources = set(suite.expected_source_files())
    assert {
        "experiments/__init__.py",
        "tools/__init__.py",
        "experiments/evisirst_v2_data.py",
        "experiments/sctransnet_sbsc_v2.py",
        "experiments/sctransnet_sbsc_v21.py",
        "train_validation_selected.py",
        "load_models.py",
        "pytest.ini",
        "experiments/sbsc_v33_methods/sbsc_v33_third_irstd_formal.json",
        "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json",
    } <= sources


def test_environment_removes_user_pytest_overrides_and_freezes_seed(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "--lf")
    monkeypatch.setenv("PYTEST_PLUGINS", "hostile")
    environment, selected = suite.frozen_subprocess_environment()
    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_PLUGINS" not in environment
    assert selected["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert selected["PYTHONHASHSEED"] == "42"
    assert selected["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert selected["PYTHONPATH"] == str(suite.PROJECT_ROOT)
    assert selected["PYTHONNOUSERSITE"] == "1"
    assert selected["PYTHONDONTWRITEBYTECODE"] == "1"
    assert selected["PYTHONWARNINGS"] == "default"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            b"607 passed, 2 warnings in 75.27s (0:01:15)\n",
            {"passed": 607, "skipped": 0, "warnings": 2, "failed": 0},
        ),
        (
            b"F.\n1 failed, 1 passed, 2 warnings in 0.50s\n",
            {"passed": 1, "skipped": 0, "warnings": 2, "failed": 1},
        ),
    ],
)
def test_summary_parser_uses_only_observed_pytest_counts(line, expected) -> None:
    parsed = suite.parse_pytest_summary(line)
    assert parsed["summary_parse_valid"] is True
    for key, value in expected.items():
        assert parsed[key] == value


def test_pass_report_binds_exact_command_files_sources_environment_and_stdout() -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_pass_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        stdout = b"..\n2 passed, 3 warnings in 0.12s\n"
        calls = []

        def command_runner(argv, *, cwd, env):
            calls.append((list(argv), cwd, dict(env)))
            return _completed(stdout)

        report = suite.run_suite(
            output_path=directory / "report.json",
            test_files=tests,
            related_source_files=sources,
            expected_passed_count=2,
            command_runner=command_runner,
        )
        assert report["status"] == "PASS"
        assert report["exit_code"] == 0
        assert report["counts"] == {
            "passed": 2,
            "failed": 0,
            "skipped": 0,
            "warnings": 3,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "deselected": 0,
        }
        assert report["stdout_sha256"] == hashlib.sha256(stdout).hexdigest()
        assert report["stdout_size_bytes"] == len(stdout)
        assert report["source_unchanged_during_execution"] is True
        assert report["argv"] == suite.pytest_argv(tests)
        assert report["source_manifest_sha256"] == report["source_manifest"]["sha256"]
        assert report["runtime_identity_sha256"] == contracts.canonical_sha256(
            report["runtime_identity"]
        )
        assert len(calls) == 1 and calls[0][1] == suite.PROJECT_ROOT
        suite.validate_unit_report(
            report,
            expected_tests=tests,
            expected_related_sources=sources,
            expected_passed_count=2,
        )


def test_nonzero_exit_is_recorded_as_fail_without_inventing_counts() -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_fail_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        stdout = b"F\n1 failed in 0.10s\n"
        report = suite.run_suite(
            output_path=directory / "report.json",
            test_files=tests,
            related_source_files=sources,
            expected_passed_count=2,
            command_runner=lambda *args, **kwargs: _completed(stdout, returncode=1),
        )
        assert report["status"] == "FAIL"
        assert report["exit_code"] == 1
        assert report["counts"]["failed"] == 1
        assert report["counts"]["passed"] == 0


def test_missing_file_fails_before_pytest_and_records_null_exit_code() -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_missing_", dir=runs) as temporary:
        directory = Path(temporary)
        source = directory / "source.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        missing = (directory / "missing_test.py").relative_to(
            suite.PROJECT_ROOT
        ).as_posix()
        source_relative = source.relative_to(suite.PROJECT_ROOT).as_posix()
        called = False

        def command_runner(*args, **kwargs):
            nonlocal called
            called = True
            return _completed(b"1 passed in 0.01s\n")

        report = suite.run_suite(
            output_path=directory / "report.json",
            test_files=(missing,),
            related_source_files=(source_relative,),
            expected_passed_count=1,
            command_runner=command_runner,
        )
        assert called is False
        assert report["status"] == "FAIL"
        assert report["execution_started"] is False
        assert report["exit_code"] is None
        assert report["missing_files"] == [missing]
        assert report["source_manifest"] is None
        assert report["stdout_sha256"] == hashlib.sha256(b"").hexdigest()
        suite.validate_unit_report(
            report,
            expected_tests=(missing,),
            expected_related_sources=(source_relative,),
            expected_passed_count=1,
        )


def test_report_destination_is_strictly_write_once() -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_once_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        output = directory / "report.json"
        runner = lambda *args, **kwargs: _completed(b"1 passed in 0.01s\n")
        suite.run_suite(
            output_path=output,
            test_files=tests,
            related_source_files=sources,
            expected_passed_count=1,
            command_runner=runner,
        )
        with pytest.raises(FileExistsError):
            suite.run_suite(
                output_path=output,
                test_files=tests,
                related_source_files=sources,
                expected_passed_count=1,
                command_runner=runner,
            )


@pytest.mark.parametrize(
    "stdout",
    [
        b"1 passed in 0.01s\ntrailing injected text\n",
        b"1 passed in 0.01s\n1 passed in 0.02s\n",
        b".s\n1 passed, 1 skipped in 0.02s\n",
    ],
)
def test_ambiguous_trailing_or_nonpass_summary_cannot_authorize(stdout: bytes) -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_summary_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        report = suite.run_suite(
            output_path=directory / "report.json",
            test_files=tests,
            related_source_files=sources,
            expected_passed_count=1,
            command_runner=lambda *args, **kwargs: _completed(stdout),
        )
        assert report["status"] == "FAIL"


def test_source_change_during_pytest_is_a_hard_failure() -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_drift_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        source = suite.PROJECT_ROOT / sources[0]

        def mutating_runner(*args, **kwargs):
            source.write_text("VALUE = 2\n", encoding="utf-8")
            return _completed(b"1 passed in 0.01s\n")

        output = directory / "report.json"
        with pytest.raises(suite.FrozenUnitSuiteError, match="source file drifted"):
            suite.run_suite(
                output_path=output,
                test_files=tests,
                related_source_files=sources,
                expected_passed_count=1,
                command_runner=mutating_runner,
            )
        report = contracts.load_strict_json(output)
        assert report["status"] == "FAIL"
        assert report["source_unchanged_during_execution"] is False
        assert report["post_source_snapshot_sha256"] != report["source_snapshot"][
            "sha256"
        ]


def test_subprocess_exception_is_recorded_without_fabricated_exit_or_counts() -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_error_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)

        def broken_runner(*args, **kwargs):
            raise OSError("fixture spawn failure")

        report = suite.run_suite(
            output_path=directory / "report.json",
            test_files=tests,
            related_source_files=sources,
            expected_passed_count=1,
            command_runner=broken_runner,
        )
        assert report["status"] == "FAIL"
        assert report["execution_started"] is True
        assert report["exit_code"] is None
        assert report["counts"]["passed"] == 0
        assert report["execution_error"] == {
            "exception_type": "OSError",
            "message": "fixture spawn failure",
        }


def test_stdout_bytes_and_summary_sha_are_independently_replayed() -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_stdout_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        report = suite.run_suite(
            output_path=directory / "report.json",
            test_files=tests,
            related_source_files=sources,
            expected_passed_count=1,
            command_runner=lambda *args, **kwargs: _completed(
                b"1 passed in 0.01s\n"
            ),
        )
        forged = copy.deepcopy(report)
        forged["summary_line_sha256"] = "0" * 64
        with pytest.raises(suite.FrozenUnitSuiteError, match="stdout replay"):
            suite.validate_unit_report(
                forged,
                expected_tests=tests,
                expected_related_sources=sources,
                expected_passed_count=1,
            )


def test_unit_runner_returns_only_after_strict_disk_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_reload_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        output = directory / "report.json"
        real_load = contracts.load_strict_json
        reloads = 0

        def tracking_load(path: Path):
            nonlocal reloads
            if path == output:
                reloads += 1
            return real_load(path)

        monkeypatch.setattr(contracts, "load_strict_json", tracking_load)
        report = suite.run_suite(
            output_path=output,
            test_files=tests,
            related_source_files=sources,
            expected_passed_count=1,
            command_runner=lambda *args, **kwargs: _completed(
                b"1 passed in 0.01s\n"
            ),
        )
        assert reloads == 1
        assert report["status"] == "PASS"


def test_unit_runner_rejects_tampered_strict_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_bad_reload_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        output = directory / "report.json"
        real_load = contracts.load_strict_json

        def forged_load(path: Path):
            value = real_load(path)
            if path == output:
                value["status"] = "FAIL"
            return value

        monkeypatch.setattr(contracts, "load_strict_json", forged_load)
        with pytest.raises(suite.FrozenUnitSuiteError, match="strict reload differs"):
            suite.run_suite(
                output_path=output,
                test_files=tests,
                related_source_files=sources,
                expected_passed_count=1,
                command_runner=lambda *args, **kwargs: _completed(
                    b"1 passed in 0.01s\n"
                ),
            )
        assert output.is_file()


def test_unit_runner_detects_physical_byte_change_after_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = suite.PROJECT_ROOT / "runs"
    runs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unit_suite_bad_bytes_", dir=runs) as temporary:
        directory = Path(temporary)
        tests, sources = _fixture_files(directory)
        output = directory / "report.json"
        real_write = contracts.write_once_json

        def corrupting_write(path: Path, payload):
            digest = real_write(path, payload)
            path.write_bytes(path.read_bytes() + b" ")
            return digest

        monkeypatch.setattr(contracts, "write_once_json", corrupting_write)
        with pytest.raises(suite.FrozenUnitSuiteError, match="physical bytes differ"):
            suite.run_suite(
                output_path=output,
                test_files=tests,
                related_source_files=sources,
                expected_passed_count=1,
                command_runner=lambda *args, **kwargs: _completed(
                    b"1 passed in 0.01s\n"
                ),
            )
