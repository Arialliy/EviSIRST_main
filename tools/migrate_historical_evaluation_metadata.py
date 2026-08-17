"""Rebase and classify the frozen ``three_folders_v1`` evaluation JSON files.

The migration is deliberately metadata-only.  It validates a frozen SHA-256
over ``metrics`` and ``metrics_display`` before replacing machine-local paths
and attaching checkpoint-selection provenance derived from the audited
checkpoint payloads.  No model, dataset, or PyTorch import is required.

Run without flags (or with ``--check``) for a read-only consistency check.
Use ``--write`` only when importing the original historical JSON files again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_ROOT = PROJECT_ROOT / "evaluation" / "three_folders_v1"
MIGRATION_SCHEMA = "evisirst_historical_evaluation_metadata_migration/v1"
PROVENANCE_SCHEMA = "evisirst_historical_checkpoint_selection/v1"
DUAL_ROLE_SOURCE_SHA256 = (
    "e8bad14bb1a5db04d9c74eda78fd7203ad01af637bc0e47bfc9219374074d7e9"
)
DUAL_ROLE_SOURCE_TREE_SHA256 = (
    "0e8d6367cd1ef879b106e339d10f5bbfcbdb2097e41ff924081a6c84f4dd2b6d"
)
SIRST3_RUNNER_SHA256 = (
    "e16ab58bd9cd06083e1d428ae98ce282ab5905f138b1f2eb8edf0ba44bd15e71"
)
SIRST3_SUMMARY_SHA256 = (
    "b6fc4de06b07a5ea68486ea277fb70f6888546c380b07fea00a7d44ef729a6af"
)


class HistoricalEvaluationMigrationError(ValueError):
    """A frozen historical result differs from its audited identity."""


@dataclass(frozen=True)
class ResultSpec:
    result_relative_path: str
    dataset: str
    training_dataset: str
    checkpoint_relative_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    checkpoint_epoch: int
    checkpoint_role: str
    source_kind: str
    source_selection: str
    selection_metric: str | None
    measurement_sha256: str


def _specs() -> tuple[ResultSpec, ...]:
    independent = {
        "NUAA-SIRST": (
            850,
            "01031f35811ec6ca23643da32d8acdf89ddd7d5cda01f1b525a37f3c8299bf14",
            43706327,
            "5ef31816045745d62c317357a58eb1eb9ac2cdd3787d403e9684cbebbe350a1e",
        ),
        "NUDT-SIRST": (
            420,
            "497b079c89c23b26c18c8812672bf77b4a99aeabd246c77a05cea9f67fe31610",
            43706327,
            "02c95c0fce81f37d4e5b38f94c538a18cc69b9b088e8c100877482fc9e4da068",
        ),
        "IRSTD-1K": (
            830,
            "87bb8bb333e28cbb053448b90d50b64eeb3bb905667451d46151079159d24bed",
            43706263,
            "c5d8abb06d918d82a7881f312cacb4e4fb9540a89d9e712ad4fe45f5b518be5c",
        ),
    }
    best = {
        ("NUAA-SIRST", "best_miou"): (
            556,
            "7882ba116f327b794ea8b347100d0bc44394a28dc7958b2aa9e0a074a6c72615",
            43701289,
            "5ef1d3bfdd1413d3ebbc0d9ba3f2541215953f706aa026b6c28816db430bb2ce",
        ),
        ("NUDT-SIRST", "best_miou"): (
            626,
            "d4c24378e30c6f028f70c2fef182929a4fb2fd2eee474925495a95e1fc34312d",
            43701225,
            "d72257bc0e5571e0c1ca05c2657dd8fde8225abed4d69057e0e0fae2ca94ae1b",
        ),
        ("IRSTD-1K", "best_miou"): (
            833,
            "a0842eecaac372482b6ff25dad32afa787b828fe018be29250e2dc8f79d05eb2",
            43701225,
            "618103587c1560d03d9f29030205e5d7a23004d75c45e9055ead068f8ecd170b",
        ),
        ("NUAA-SIRST", "best_pd"): (
            557,
            "7cd6beccc39c147faf4fae4a4df9bd4aba17232d5ceefd5b8997e9b09cf07714",
            43700085,
            "4307c61adbbbeaee4635dfafb2ff8f844aa02a65e2086c43a9ddbcb139f7eb74",
        ),
        ("NUDT-SIRST", "best_pd"): (
            626,
            "7443d84b9c99980312bc15e815c0f121628a5018a16fbaae142fba0c0f78e07e",
            43700085,
            "d72257bc0e5571e0c1ca05c2657dd8fde8225abed4d69057e0e0fae2ca94ae1b",
        ),
        ("IRSTD-1K", "best_pd"): (
            552,
            "4b58f2fb7bed2ccf377f597c9a2b0ea38c0d899f62c05b49b8cdf952dcb66d92",
            43700021,
            "4fc3ecd21249e5252b145d442ad2b39264cab7f9e98409edbe5d37b51e3f18c9",
        ),
    }
    sirst3_measurements = {
        "NUAA-SIRST": (
            "e4abea4ea1da206b3c1468f0ed033377efe6fd426d63640758435615dc0faeb3"
        ),
        "NUDT-SIRST": (
            "c9ab21f7255ad013356820887f10390df56cd3c3a420423dae50b125b4438f5d"
        ),
        "IRSTD-1K": (
            "c2751b5dcb8de0acdadac344430d7ded243697d42bf61afb9b996a90becc900f"
        ),
    }

    records: list[ResultSpec] = []
    for (dataset, role), (epoch, digest, size, measurement) in best.items():
        filename = (
            "EviSIRST_best_mIoU.pth.tar"
            if role == "best_miou"
            else "EviSIRST_best_Pd.pth.tar"
        )
        records.append(
            ResultSpec(
                result_relative_path=f"result_{role}/{dataset}.json",
                dataset=dataset,
                training_dataset=dataset,
                checkpoint_relative_path=f"result/{dataset}/{filename}",
                checkpoint_sha256=digest,
                checkpoint_size_bytes=size,
                checkpoint_epoch=epoch,
                checkpoint_role=role,
                source_kind="dual_role_test_selected",
                source_selection=f"historical_test_{role}",
                selection_metric=(
                    "global_foreground_mIoU" if role == "best_miou" else "Pd"
                ),
                measurement_sha256=measurement,
            )
        )
    for dataset, (epoch, digest, size, measurement) in independent.items():
        records.append(
            ResultSpec(
                result_relative_path=f"results_independent/{dataset}.json",
                dataset=dataset,
                training_dataset=dataset,
                checkpoint_relative_path=f"results/{dataset}/EviSIRST.pth.tar",
                checkpoint_sha256=digest,
                checkpoint_size_bytes=size,
                checkpoint_epoch=epoch,
                checkpoint_role="final",
                source_kind="test_selected_deployment_export",
                source_selection="historical_best_miou",
                selection_metric="global_foreground_mIoU",
                measurement_sha256=measurement,
            )
        )
    for dataset, measurement in sirst3_measurements.items():
        records.append(
            ResultSpec(
                result_relative_path=f"results_sirst3/{dataset}.json",
                dataset=dataset,
                training_dataset="SIRST3",
                checkpoint_relative_path="results/SIRST3/EviSIRST.pth.tar",
                checkpoint_sha256=(
                    "37673d0e376d80c383007aa59315b745461b6cd3e6c8a52d57848822ce4a3137"
                ),
                checkpoint_size_bytes=43692709,
                checkpoint_epoch=1000,
                checkpoint_role="final",
                source_kind="fixed_epoch_endpoint",
                source_selection="fixed_epoch_1000_endpoint",
                selection_metric=None,
                measurement_sha256=measurement,
            )
        )
    return tuple(sorted(records, key=lambda item: item.result_relative_path))


RESULT_SPECS = _specs()
RESULT_SPEC_BY_PATH = {item.result_relative_path: item for item in RESULT_SPECS}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def measurement_sha256(document: Mapping[str, Any]) -> str:
    try:
        projection = {
            "metrics": document["metrics"],
            "metrics_display": document["metrics_display"],
        }
    except KeyError as exc:
        raise HistoricalEvaluationMigrationError(
            "historical result is missing metrics or metrics_display"
        ) from exc
    return _canonical_sha256(projection)


def _path_has_expected_suffix(value: str, expected: str) -> bool:
    observed_parts = PurePosixPath(value.replace("\\", "/")).parts
    expected_parts = PurePosixPath(expected).parts
    return len(observed_parts) >= len(expected_parts) and (
        observed_parts[-len(expected_parts) :] == expected_parts
    )


def _selection_provenance(spec: ResultSpec) -> dict[str, Any]:
    common = {
        "schema": PROVENANCE_SCHEMA,
        "source_checkpoint_sha256": spec.checkpoint_sha256,
    }
    if spec.source_kind == "dual_role_test_selected":
        return {
            **common,
            "classification": "historical-test-selected",
            "source_selection_label_basis": (
                "normalized_from_checkpoint_role_and_payload_selection_metric"
            ),
            "data_role": "test",
            "test_selected": True,
            "selection_is_optimistic": True,
            "unbiased_test_claim_supported": False,
            "checkpoint_payload_evidence": {
                "schema": "evisirst_clean_checkpoint/v1",
                "training_protocol": "evisirst_four_regime_dual_role_test_selected_v1",
                "selection_split": f"{spec.dataset}_test",
                "selection_metric": spec.selection_metric,
                "test_split_accessed": True,
                "this_checkpoint_selected_by_test": True,
            },
            "source_code_evidence": {
                "relative_path": "train_dual_role_test_selected.py",
                "file_sha256": DUAL_ROLE_SOURCE_SHA256,
                "training_source_tree_sha256": DUAL_ROLE_SOURCE_TREE_SHA256,
            },
        }
    if spec.source_kind == "test_selected_deployment_export":
        return {
            **common,
            "classification": "historical-test-selected",
            "source_selection_label_basis": (
                "normalized_from_package_historical_selection_and_"
                "source_checkpoint_role"
            ),
            "data_role": "test",
            "test_selected": True,
            "selection_is_optimistic": True,
            "unbiased_test_claim_supported": False,
            "checkpoint_payload_evidence": {
                "schema": "sctransnet_three_component_current_inference_package/v1",
                "historical_selection_source": f"test_{spec.dataset}",
                "source_checkpoint_role": "best_miou",
                "source_checkpoint_epoch": spec.checkpoint_epoch,
                "operational_checkpoint_only": True,
                "unbiased_test_claim_allowed": False,
                "metrics_recomputed_during_export": False,
            },
        }
    if spec.source_kind == "fixed_epoch_endpoint":
        return {
            **common,
            "classification": "fixed-endpoint",
            "source_selection_label_basis": (
                "copied_from_hash_bound_historical_summary_checkpoint_selection_rule"
            ),
            "data_role": "none",
            "test_selected": False,
            "selection_is_optimistic": False,
            "checkpoint_payload_evidence": {
                "schema": "evisirst_clean_checkpoint/v1",
                "checkpoint_role": "final",
                "epoch": 1000,
                "test_split_accessed": False,
                "training_data_protocol": "evisirst_public_joint_training_v1",
                "training_test_split_accessed": False,
            },
            "source_code_evidence": {
                "relative_path": "run_sirst3_experiment.py",
                "file_sha256": SIRST3_RUNNER_SHA256,
                "checkpoint_selection_rule": "fixed_epoch_1000_endpoint",
                "test_selected": False,
                "target_test_used_for_training_or_selection": False,
                "binding_scope": (
                    "repository_script_contract_not_cryptographically_embedded"
                ),
            },
            "historical_summary_evidence": {
                "relative_path": "results/SIRST3/cross_dataset_results.json",
                "file_sha256": SIRST3_SUMMARY_SHA256,
                "schema": (
                    "evisirst_sirst3_fixed_endpoint_three_source_evaluation/v1"
                ),
                "checkpoint_selection_rule": "fixed_epoch_1000_endpoint",
                "test_selected": False,
                "target_test_used_for_training_or_selection": False,
                "threshold_search_performed": False,
            },
            "final_test_once_status": "unsupported_without_execution_ledger",
        }
    raise AssertionError(f"unsupported source kind: {spec.source_kind}")


def migrate_document(
    document: Mapping[str, Any], *, result_relative_path: str
) -> dict[str, Any]:
    """Return the canonical metadata-migrated form of one frozen result."""

    try:
        spec = RESULT_SPEC_BY_PATH[result_relative_path]
    except KeyError as exc:
        raise HistoricalEvaluationMigrationError(
            f"unexpected historical result path: {result_relative_path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise HistoricalEvaluationMigrationError(
            "historical result root is not an object"
        )
    migrated = dict(document)
    identity = {
        "schema": "evisirst_public_evaluation/v1",
        "model": "evisirst",
        "training_dataset": spec.training_dataset,
        "evaluation_dataset": spec.dataset,
        "dataset": spec.dataset,
    }
    for field, expected in identity.items():
        if migrated.get(field) != expected:
            raise HistoricalEvaluationMigrationError(
                f"{result_relative_path}: {field} differs from the frozen identity"
            )
    observed_measurement = measurement_sha256(migrated)
    if observed_measurement != spec.measurement_sha256:
        raise HistoricalEvaluationMigrationError(
            f"{result_relative_path}: metric payload SHA-256 differs"
        )

    checkpoint = migrated.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise HistoricalEvaluationMigrationError(
            f"{result_relative_path}: checkpoint metadata is missing"
        )
    checkpoint = dict(checkpoint)
    expected_checkpoint = {
        "epoch": spec.checkpoint_epoch,
        "role": spec.checkpoint_role,
        "sha256": spec.checkpoint_sha256,
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint.get(field) != expected:
            raise HistoricalEvaluationMigrationError(
                f"{result_relative_path}: checkpoint {field} differs"
            )
    raw_checkpoint_path = checkpoint.get("path")
    if not isinstance(raw_checkpoint_path, str) or not _path_has_expected_suffix(
        raw_checkpoint_path, spec.checkpoint_relative_path
    ):
        raise HistoricalEvaluationMigrationError(
            f"{result_relative_path}: checkpoint path differs"
        )

    # Dataset locations are deliberately external and are not an identity.
    migrated.pop("dataset_root", None)
    checkpoint["path"] = spec.checkpoint_relative_path
    checkpoint["source_selection"] = spec.source_selection
    checkpoint["selection_is_optimistic"] = (
        spec.source_kind != "fixed_epoch_endpoint"
    )
    checkpoint["selection_provenance"] = _selection_provenance(spec)
    migrated["checkpoint"] = checkpoint
    migrated["metadata_migration"] = {
        "schema": MIGRATION_SCHEMA,
        "measurement_sha256": spec.measurement_sha256,
        "measurement_fields": ["metrics", "metrics_display"],
        "measurement_values_changed": False,
        "machine_local_dataset_root_removed": True,
        "checkpoint_path_rebased_to_manifest_relative_path": True,
        "checkpoint_size_bytes": spec.checkpoint_size_bytes,
    }
    return migrated


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def migrate_tree(root: Path, *, write: bool) -> tuple[str, ...]:
    """Check or rewrite all twelve frozen results and return changed paths."""

    root = root.resolve(strict=True)
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*.json")
    }
    expected = set(RESULT_SPEC_BY_PATH)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise HistoricalEvaluationMigrationError(
            f"historical result set differs; missing={missing}, extra={extra}"
        )
    changed: list[str] = []
    for relative_path in sorted(expected):
        path = root / relative_path
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HistoricalEvaluationMigrationError(
                f"cannot read {relative_path}: {exc}"
            ) from exc
        migrated = migrate_document(document, result_relative_path=relative_path)
        content = _json_bytes(migrated)
        if path.read_bytes() != content:
            changed.append(relative_path)
            if write:
                _write_atomic(path, content)
    return tuple(changed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only check (default)")
    mode.add_argument("--write", action="store_true", help="atomically apply migration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        changed = migrate_tree(args.root, write=args.write)
    except HistoricalEvaluationMigrationError as exc:
        print(f"MIGRATION ERROR: {exc}")
        return 2
    if changed and not args.write:
        for relative_path in changed:
            print(f"NEEDS_MIGRATION {relative_path}")
        return 1
    verb = "MIGRATED" if args.write else "VERIFIED"
    print(f"{verb} historical_results={len(RESULT_SPECS)} changed={len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
