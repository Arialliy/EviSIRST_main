from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments import sbsc_v33_contracts as contracts


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "duplicate.json"
    artifact.write_text(
        '{"outer":{"value":1,"value":2}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        contracts.load_strict_json(artifact)


@pytest.mark.parametrize(
    "document",
    (
        '{"value":NaN}\n',
        '{"value":Infinity}\n',
        '{"value":-Infinity}\n',
        '{"value":1e999}\n',
    ),
)
def test_strict_json_rejects_every_nonfinite_number(
    tmp_path: Path,
    document: str,
) -> None:
    artifact = tmp_path / "nonfinite.json"
    artifact.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        contracts.load_strict_json(artifact)


@pytest.mark.parametrize(
    ("loader_name", "authority_name"),
    (
        ("load_baseline_authority_manifest", "BASELINE_MANIFEST_PATH"),
        ("load_gradient_authorization", "GRADIENT_AUTHORIZATION_PATH"),
    ),
)
def test_authority_loaders_reject_raw_symlink_even_to_exact_authority(
    tmp_path: Path,
    loader_name: str,
    authority_name: str,
) -> None:
    authority = getattr(contracts, authority_name)
    alias = tmp_path / authority.name
    alias.symlink_to(authority)

    with pytest.raises(FileNotFoundError, match="regular.*symlink file"):
        getattr(contracts, loader_name)(alias)


@pytest.mark.parametrize(
    "relative",
    (
        "../outside.json",
        "experiments/../outside.json",
        "/etc/passwd",
        "experiments//sbsc_v33_contracts.py",
    ),
)
def test_repository_input_path_rejects_escape_or_noncanonical_text(
    relative: str,
) -> None:
    with pytest.raises(ValueError, match="canonical|relative"):
        contracts.require_repository_relative_regular_file(relative)


def test_output_path_rejects_project_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="below project root"):
        contracts.validated_output_path(tmp_path / "escaped.json")

    lexical_escape = (
        contracts.PROJECT_ROOT
        / "artifacts"
        / ".."
        / "escaped.json"
    )
    with pytest.raises(ValueError, match="malformed"):
        contracts.validated_output_path(lexical_escape)


def test_real_baseline_manifest_loads_all_physical_authorities() -> None:
    loaded = contracts.load_baseline_authority_manifest()

    assert loaded["status"] == "frozen"
    assert loaded["dataset_order"] == list(contracts.DATASETS)
    assert loaded["manifest_sha256"] == contracts.sha256_file(
        contracts.BASELINE_MANIFEST_PATH
    )
    assert set(loaded["authorities"]) == set(contracts.DATASETS)
    for dataset in contracts.DATASETS:
        authority = loaded["authorities"][dataset]
        assert authority["dataset"] == dataset
        assert authority["sample_count"] == contracts.EXPECTED_SAMPLE_COUNTS[
            dataset
        ]
        assert authority["physical_checkpoint_sha256_verified"] is True


def test_baseline_manifest_hash_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = contracts.load_strict_json
    tampered = copy.deepcopy(real_load(contracts.BASELINE_MANIFEST_PATH))
    tampered["datasets"]["NUAA-SIRST"]["evaluation_sha256"] = "0" * 64

    def load_with_tampered_manifest(path: Path) -> dict:
        if path.resolve(strict=True) == contracts.BASELINE_MANIFEST_PATH.resolve(
            strict=True
        ):
            return copy.deepcopy(tampered)
        return real_load(path)

    monkeypatch.setattr(
        contracts,
        "load_strict_json",
        load_with_tampered_manifest,
    )

    with pytest.raises(ValueError, match="baseline evaluation SHA-256 differs"):
        contracts.load_baseline_authority_manifest()


def test_real_gradient_authorization_loads_and_binds_sources() -> None:
    loaded = contracts.load_gradient_authorization()

    assert loaded["status"] == "frozen"
    assert loaded["authorized_router_value_gradient_mode"] in {
        "live",
        "detached",
    }
    assert loaded["authorization_sha256"] == contracts.sha256_file(
        contracts.GRADIENT_AUTHORIZATION_PATH
    )
    assert len(loaded["source_manifest_sha256"]) == 64


def test_gradient_authorization_artifact_hash_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = contracts.load_strict_json
    tampered = copy.deepcopy(real_load(contracts.GRADIENT_AUTHORIZATION_PATH))
    tampered["report_sha256"] = "0" * 64

    def load_with_tampered_authorization(path: Path) -> dict:
        if path.resolve(strict=True) == (
            contracts.GRADIENT_AUTHORIZATION_PATH.resolve(strict=True)
        ):
            return copy.deepcopy(tampered)
        return real_load(path)

    monkeypatch.setattr(
        contracts,
        "load_strict_json",
        load_with_tampered_authorization,
    )

    with pytest.raises(ValueError, match="gradient report_path SHA-256 differs"):
        contracts.load_gradient_authorization()


def test_gradient_authorization_source_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = contracts.load_strict_json(
        contracts.GRADIENT_AUTHORIZATION_PATH
    )
    report_path = contracts.require_repository_relative_regular_file(
        authorization["report_path"]
    )
    report = contracts.load_strict_json(report_path)
    source_record = report["source_manifest"]["files"][0]
    drifted_source = contracts.require_repository_relative_regular_file(
        source_record["path"]
    )
    real_sha256_file = contracts.sha256_file

    def sha256_with_source_drift(path: Path) -> str:
        if path.resolve(strict=True) == drifted_source:
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(
        contracts,
        "sha256_file",
        sha256_with_source_drift,
    )

    with pytest.raises(ValueError, match="authorized gradient source file drifted"):
        contracts.load_gradient_authorization()


def test_write_once_json_is_idempotent_and_rejects_different_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contracts, "PROJECT_ROOT", tmp_path.resolve())
    destination = tmp_path / "nested" / "artifact.json"
    first_payload = {
        "schema": "test/sbsc_v33_write_once/v1",
        "value": 1,
    }

    first_sha = contracts.write_once_json(destination, first_payload)
    original_bytes = destination.read_bytes()

    assert first_sha == contracts.sha256_file(destination)
    assert first_sha == contracts.write_once_json(
        destination,
        {"value": 1, "schema": "test/sbsc_v33_write_once/v1"},
    )
    assert destination.read_bytes() == original_bytes

    with pytest.raises(FileExistsError, match="write-once artifact differs"):
        contracts.write_once_json(
            destination,
            {
                "schema": "test/sbsc_v33_write_once/v1",
                "value": 2,
            },
        )

    assert destination.read_bytes() == original_bytes
