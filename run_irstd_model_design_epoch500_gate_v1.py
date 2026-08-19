#!/usr/bin/env python3
"""Immutable validation-only epoch-500 gate for IRSTD PSBFR V1 and D0.

The gate has no path, seed, threshold, or metric CLI options.  It reads only
the first 500 validation records, directly invokes the frozen zero-margin
selector, and requires variant transactions to be atomically paused at epoch
500 while retaining a 1000-epoch schedule identity.  Missing prerequisites
produce a non-writing WAIT; only a complete terminal decision can be written.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

from experiments import evisirst_zero_margin_selection as zero_selection
import train_irstd_model_design_screen_v1 as screen_runner


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_SCHEMA = "evisirst_irstd_model_design_epoch500_gate/v1"
DIAGNOSTIC_SCHEMA = "evisirst_irstd_psbfr_correction_diagnostic/v1"
OUTPUT_RELATIVE_PATH = "runs/irstd_model_design/epoch500_gate/result.json"
DIAGNOSTIC_RELATIVE_PATH = (
    "runs/irstd_model_design/psbfr_v1/legacy_screen/diagnostic/"
    "run_seed_42/result.json"
)
GATE_SOURCE_RELATIVE_PATH = "run_irstd_model_design_epoch500_gate_v1.py"
PROTOCOL_RELATIVE_PATH = "experiments/IRSTD_PSBFR_V1_PROTOCOL.md"
RULES_RELATIVE_PATH = "experiments/irstd_psbfr_v1_screen_rules.json"
RUNNER_RELATIVE_PATH = "train_irstd_model_design_screen_v1.py"
SELECTOR_RELATIVE_PATH = "experiments/evisirst_zero_margin_selection.py"
SPLIT_RELATIVE_PATHS = (
    "splits/v2/IRSTD-1K/manifest.json",
    "splits/v2/IRSTD-1K/train.txt",
    "splits/v2/IRSTD-1K/val.txt",
)

SCREEN_EPOCH = 500
FORMAL_EPOCHS = 1000
ARCHITECTURE_SEED = 42
DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
RUN_SEEDS = {
    "single_residual_v1": (42, 1_446_202_191),
    "psbfr_v1": (42, 1_446_202_191, 104_728_269),
}
VARIANT_ROOTS = {
    key: (
        PROJECT_ROOT
        / "runs"
        / "irstd_model_design"
        / key
        / "legacy_screen"
        / "formal"
        / DATASET
        / TARGET_MODE
    )
    for key in RUN_SEEDS
}
CLEAN_ROOT = (
    PROJECT_ROOT / "runs" / "validation_selected" / "formal" / DATASET / TARGET_MODE
)


class ModelDesignEpoch500GateError(ValueError):
    """Available evidence violates the immutable screen contract."""


class ModelDesignEpoch500GateNotReady(RuntimeError):
    """At least one fixed prerequisite has not reached epoch 500."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelDesignEpoch500GateError("non-canonical evidence") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ModelDesignEpoch500GateError("artifact escaped repository") from exc
    return {
        "relative_path": relative,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _artifact(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=_finite_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelDesignEpoch500GateError(f"malformed JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ModelDesignEpoch500GateError(f"JSON root is not an object: {path}")
    _canonical_bytes(value)
    return dict(value), metadata


def _false_disclosures(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {
                "test_split_accessed",
                "public_test_allowed",
                "public_test_supported",
                "test_index_opened",
            } and nested is not False:
                raise ModelDesignEpoch500GateError(f"{label}.{key} is not false")
            _false_disclosures(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _false_disclosures(nested, label=f"{label}[{index}]")


def _validate_identity(
    value: Any,
    *,
    seed: int,
    variant: str | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelDesignEpoch500GateError("run identity is missing")
    identity = dict(value)
    digest = identity.get("identity_sha256")
    unhashed = dict(identity)
    unhashed.pop("identity_sha256", None)
    if not isinstance(digest, str) or _canonical_sha256(unhashed) != digest:
        raise ModelDesignEpoch500GateError("run identity SHA-256 differs")
    required = {
        "dataset": DATASET,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": seed,
        "target_mode": TARGET_MODE,
        "epochs": FORMAL_EPOCHS,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1e-3,
        "min_lr": 1e-5,
        "warmup_epochs": 10,
        "val_interval": 1,
        "manifest_sha256": screen_runner.CANONICAL_IRSTD_MANIFEST_SHA256,
        "data_tree_sha256": screen_runner.CANONICAL_IRSTD_DATA_TREE_SHA256,
        "smoke": False,
        "test_split_accessed": False,
    }
    if any(identity.get(key) != expected for key, expected in required.items()):
        raise ModelDesignEpoch500GateError("formal 1000-epoch identity differs")
    if variant is not None:
        if (
            identity.get("schema") != screen_runner.TRAINING_SCHEMA + "/run_identity"
            or identity.get("variant_key") != variant
            or identity.get("formal_configured_total_epochs") != FORMAL_EPOCHS
            or identity.get("selection_rule") != zero_selection.RULE_VERSION
            or identity.get("selection_margin_raw") is not None
            or identity.get("selection_window_applied") is not False
            or identity.get("promotion_eligible") is not True
        ):
            raise ModelDesignEpoch500GateError("variant screen identity differs")
        spec = screen_runner.VARIANTS[variant]
        state_contract = identity.get("state_contract")
        if not isinstance(state_contract, Mapping) or (
            state_contract.get("architecture_source_sha256")
            != _sha256_file(spec.architecture_source)
            or state_contract.get("total_state_key_count") != spec.state_key_count
            or state_contract.get("total_parameter_count") != spec.parameter_count
        ):
            raise ModelDesignEpoch500GateError("variant architecture hash differs")
        source_files = identity.get("determinism_protocol", {}).get("source_files", {})
        expected_sources = {
            "frozen_protocol": PROJECT_ROOT / PROTOCOL_RELATIVE_PATH,
            "frozen_screen_rules": PROJECT_ROOT / RULES_RELATIVE_PATH,
            "screen_runner": PROJECT_ROOT / RUNNER_RELATIVE_PATH,
            "selected_architecture": spec.architecture_source,
            "zero_margin_selector": PROJECT_ROOT / SELECTOR_RELATIVE_PATH,
        }
        for name, path in expected_sources.items():
            entry = source_files.get(name) if isinstance(source_files, Mapping) else None
            if not isinstance(entry, Mapping) or entry.get("sha256") != _sha256_file(path):
                raise ModelDesignEpoch500GateError(f"variant source hash differs: {name}")
    _false_disclosures(identity, label="run_identity")
    return json.loads(_canonical_bytes(identity).decode("ascii"))


def _first500_selection(records: Any) -> tuple[dict[str, Any], dict[str, float | int]]:
    if not isinstance(records, list) or len(records) < SCREEN_EPOCH:
        raise ModelDesignEpoch500GateNotReady(
            f"validation history has {len(records) if isinstance(records, list) else 0}/500"
        )
    prefix = records[:SCREEN_EPOCH]
    if [record.get("epoch") for record in prefix if isinstance(record, Mapping)] != list(
        range(1, SCREEN_EPOCH + 1)
    ):
        raise ModelDesignEpoch500GateError("first-500 history is not continuous")
    for record in prefix:
        if (
            not isinstance(record, Mapping)
            or record.get("data_role") != "val"
            or record.get("evaluation_head") != "out"
        ):
            raise ModelDesignEpoch500GateError("first-500 validation record differs")
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ModelDesignEpoch500GateError("validation metrics are missing")
        for outer, inner in (("mIoU", "miou"), ("Pd", "pd"), ("Fa", "fa")):
            if record.get(outer) != metrics.get(inner):
                raise ModelDesignEpoch500GateError(f"stored {outer} copies differ")
    fresh = zero_selection.select_checkpoints(
        prefix,
        primary_role=zero_selection.PRIMARY_ROLE,
        margin=None,
    )
    selected = fresh["roles"][zero_selection.PRIMARY_ROLE]["selected"]
    epoch = selected["epoch"]
    record = prefix[epoch - 1]
    metrics = {
        "epoch": int(epoch),
        "mIoU": float(record["mIoU"]),
        "Pd": float(record["Pd"]),
        "Fa": float(record["Fa"]),
    }
    return json.loads(_canonical_bytes(fresh).decode("ascii")), metrics


def _candidate_evidence(
    latest: Mapping[str, Any],
    *,
    run_dir: Path,
    epoch: int,
    identity: Mapping[str, Any],
    expected_state_count: int,
) -> dict[str, Any]:
    artifacts = latest.get("candidate_artifacts")
    artifact = None
    if isinstance(artifacts, Mapping):
        artifact = artifacts.get(epoch, artifacts.get(str(epoch)))
    if not isinstance(artifact, Mapping):
        raise ModelDesignEpoch500GateError("selected first-500 candidate is absent")
    expected_relative = f"candidates/epoch_{epoch:04d}.pth.tar"
    if artifact.get("relative_path") != expected_relative:
        raise ModelDesignEpoch500GateError("selected candidate path differs")
    path = run_dir / expected_relative
    metadata = _artifact(path)
    if metadata["sha256"] != artifact.get("file_sha256"):
        raise ModelDesignEpoch500GateError("selected candidate SHA-256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("epoch") != epoch
        or payload.get("run_identity") != identity
        or payload.get("test_split_accessed") is not False
        or payload.get("validation_record")
        != latest["validation_history"][epoch - 1]
    ):
        raise ModelDesignEpoch500GateError("selected candidate payload differs")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or len(state) != expected_state_count:
        raise ModelDesignEpoch500GateError("selected candidate state contract differs")
    return {
        "artifact": metadata,
        "state_key_count": len(state),
        "validation_record_sha256": _canonical_sha256(payload["validation_record"]),
    }


def _read_arm(
    *,
    seed: int,
    variant: str | None,
) -> dict[str, Any]:
    run_dir = (
        CLEAN_ROOT / f"run_seed_{seed}"
        if variant is None
        else VARIANT_ROOTS[variant] / f"run_seed_{seed}"
    )
    latest_path = run_dir / "last_training_state.pth.tar"
    history_path = run_dir / "validation_history.json"
    if not latest_path.is_file() or not history_path.is_file():
        raise ModelDesignEpoch500GateNotReady(
            f"missing transaction artifacts: {run_dir.relative_to(PROJECT_ROOT)}"
        )
    history_payload, history_artifact = _load_json(history_path)
    latest_artifact = _artifact(latest_path)
    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
    if not isinstance(latest, Mapping):
        raise ModelDesignEpoch500GateError("latest transaction is malformed")
    identity = _validate_identity(
        latest.get("run_identity"), seed=seed, variant=variant
    )
    if history_payload.get("run_identity") != identity:
        raise ModelDesignEpoch500GateError("latest/history identities differ")
    if latest.get("test_split_accessed") is not False or history_payload.get(
        "test_split_accessed"
    ) is not False:
        raise ModelDesignEpoch500GateError("test disclosure is not false")
    latest_epoch = latest.get("epoch")
    validation = latest.get("validation_history")
    training = latest.get("training_history")
    if not isinstance(latest_epoch, int) or latest_epoch < SCREEN_EPOCH:
        raise ModelDesignEpoch500GateNotReady(
            f"{variant or 'clean'} seed {seed} is at epoch {latest_epoch}/500"
        )
    if variant is not None and (
        latest_epoch != SCREEN_EPOCH
        or not isinstance(validation, list)
        or len(validation) != SCREEN_EPOCH
        or not isinstance(training, list)
        or len(training) != SCREEN_EPOCH
    ):
        raise ModelDesignEpoch500GateError(
            "variant is not atomically paused at exactly epoch 500"
        )
    if (
        not isinstance(validation, list)
        or not isinstance(history_payload.get("validation_history"), list)
        or validation[:SCREEN_EPOCH]
        != history_payload["validation_history"][:SCREEN_EPOCH]
    ):
        raise ModelDesignEpoch500GateError("latest/history first-500 records differ")
    fresh, metrics = _first500_selection(validation)
    evidence: dict[str, Any] = {
        "role": "clean_control" if variant is None else variant,
        "run_seed": seed,
        "configured_total_epochs": identity["epochs"],
        "committed_epoch": latest_epoch,
        "screen_prefix_epochs": SCREEN_EPOCH,
        "history": history_artifact,
        "latest": latest_artifact,
        "run_identity_sha256": identity["identity_sha256"],
        "fresh_zero_margin_selection": fresh,
        "fresh_zero_margin_selection_sha256": _canonical_sha256(fresh),
        "selected_metrics": metrics,
        "test_split_accessed": False,
    }
    if variant is not None:
        evidence["atomic_pause_at_exactly_500"] = True
        evidence["selected_candidate"] = _candidate_evidence(
            latest,
            run_dir=run_dir,
            epoch=int(metrics["epoch"]),
            identity=identity,
            expected_state_count=screen_runner.VARIANTS[variant].state_key_count,
        )
    else:
        evidence["atomic_pause_at_exactly_500"] = False
        evidence["candidate_evidence"] = (
            "not_required_for_preexisting_clean_history_control"
        )
    return evidence


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelDesignEpoch500GateError("decision metric is not numeric")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ModelDesignEpoch500GateError("decision metric is not finite")
    return result


def _decision(
    variant: str,
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    rules: Mapping[str, Any],
    diagnostic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    deltas: list[Decimal] = []
    safety_failures: list[int] = []
    for clean, candidate in pairs:
        c = clean["selected_metrics"]
        v = candidate["selected_metrics"]
        delta_miou = _decimal(v["mIoU"]) - _decimal(c["mIoU"])
        delta_pd = _decimal(v["Pd"]) - _decimal(c["Pd"])
        delta_fa = _decimal(v["Fa"]) - _decimal(c["Fa"])
        deltas.append(delta_miou)
        safety = delta_pd < Decimal("-0.003") and delta_fa > Decimal("0")
        if safety:
            safety_failures.append(int(candidate["run_seed"]))
        comparisons.append(
            {
                "run_seed": candidate["run_seed"],
                "clean": dict(c),
                "variant": dict(v),
                "delta_mIoU_decimal": format(delta_miou, "f"),
                "delta_Pd_decimal": format(delta_pd, "f"),
                "delta_Fa_decimal": format(delta_fa, "f"),
                "safety_failure": safety,
            }
        )
    mean = sum(deltas, Decimal("0")) / Decimal(len(deltas))
    passed = (
        all(delta > 0 for delta in deltas)
        and mean >= _decimal(rules["mean_delta_mIoU_minimum"])
        and not safety_failures
    )
    diagnostic_checks: dict[str, Any] | None = None
    if variant == "psbfr_v1":
        if diagnostic is None:
            raise ModelDesignEpoch500GateError("PSBFR diagnostic is missing")
        summary = diagnostic.get("summary")
        if not isinstance(summary, Mapping):
            raise ModelDesignEpoch500GateError("PSBFR diagnostic summary is missing")
        diagnostic_checks = {
            "bound_holds": summary.get("bound_violation_count") == 0,
            "corrector_nonzero": summary.get("nonzero_correction_count", 0) > 0,
            "saturation_fraction_below_limit": (
                _decimal(summary.get("tanh_saturation_fraction"))
                < _decimal(rules["tanh_absolute_saturation_fraction_maximum"])
            ),
        }
        passed = passed and all(diagnostic_checks.values())
    return {
        "variant": variant,
        "paired_comparisons": comparisons,
        "mean_delta_mIoU_decimal": format(mean, "f"),
        "safety_failure_run_seeds": safety_failures,
        "diagnostic_checks": diagnostic_checks,
        "result": "GO" if passed else "STOP",
    }


def _fixed_sources() -> dict[str, Any]:
    paths = {
        "gate": PROJECT_ROOT / GATE_SOURCE_RELATIVE_PATH,
        "protocol": PROJECT_ROOT / PROTOCOL_RELATIVE_PATH,
        "rules": PROJECT_ROOT / RULES_RELATIVE_PATH,
        "runner": PROJECT_ROOT / RUNNER_RELATIVE_PATH,
        "selector": PROJECT_ROOT / SELECTOR_RELATIVE_PATH,
    }
    paths.update(
        {f"split/{Path(relative).name}": PROJECT_ROOT / relative for relative in SPLIT_RELATIVE_PATHS}
    )
    return {name: _artifact(path) for name, path in paths.items()}


def evaluate_payload() -> dict[str, Any]:
    sources = _fixed_sources()
    rules_payload, rules_artifact = _load_json(PROJECT_ROOT / RULES_RELATIVE_PATH)
    if rules_artifact != sources["rules"]:
        raise ModelDesignEpoch500GateError("rules changed during evaluation")
    waiting: list[str] = []
    arms: dict[str, dict[int, dict[str, Any]]] = {"clean": {}}
    for seed in sorted(set(RUN_SEEDS["psbfr_v1"])):
        try:
            arms["clean"][seed] = _read_arm(seed=seed, variant=None)
        except ModelDesignEpoch500GateNotReady as exc:
            waiting.append(str(exc))
    for variant, seeds in RUN_SEEDS.items():
        arms[variant] = {}
        for seed in seeds:
            try:
                arms[variant][seed] = _read_arm(seed=seed, variant=variant)
            except ModelDesignEpoch500GateNotReady as exc:
                waiting.append(str(exc))
    if waiting:
        return {
            "schema": RESULT_SCHEMA,
            "status": "waiting_for_epoch_500",
            "waiting_reasons": sorted(waiting),
            "immutable_result_written": False,
            "source_identity": sources,
            "test_split_accessed": False,
            "public_test_allowed": False,
        }
    diagnostic, diagnostic_artifact = _load_json(PROJECT_ROOT / DIAGNOSTIC_RELATIVE_PATH)
    if (
        diagnostic.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostic.get("status") != "complete"
        or diagnostic.get("selection_allowed") is not False
        or diagnostic.get("test_split_accessed") is not False
        or diagnostic.get("public_test_allowed") is not False
    ):
        raise ModelDesignEpoch500GateError("PSBFR diagnostic identity differs")
    psbfr_candidate = arms["psbfr_v1"][42]["selected_candidate"]["artifact"]
    if diagnostic.get("selected_candidate") != psbfr_candidate:
        raise ModelDesignEpoch500GateError("diagnostic candidate binding differs")
    d0_rules = rules_payload["d0_diagnostic"]
    psbfr_rules = rules_payload["legacy_screen"]
    decisions = {
        "single_residual_v1": _decision(
            "single_residual_v1",
            [(arms["clean"][seed], arms["single_residual_v1"][seed]) for seed in RUN_SEEDS["single_residual_v1"]],
            d0_rules,
            None,
        ),
        "psbfr_v1": _decision(
            "psbfr_v1",
            [(arms["clean"][seed], arms["psbfr_v1"][seed]) for seed in RUN_SEEDS["psbfr_v1"]],
            psbfr_rules,
            diagnostic,
        ),
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "screen_epoch": SCREEN_EPOCH,
        "configured_total_epochs": FORMAL_EPOCHS,
        "data_role": "legacy_dev_val",
        "selection_rule": zero_selection.RULE_VERSION,
        "selection_margin_raw": None,
        "arms": arms,
        "diagnostic": {"artifact": diagnostic_artifact, "summary": diagnostic["summary"]},
        "decisions": decisions,
        "source_identity": sources,
        "all_variant_runs_atomically_paused_at_exactly_500": True,
        "records_after_epoch_500_used": False,
        "selection_allowed_from_diagnostic": False,
        "test_split_accessed": False,
        "public_test_allowed": False,
        "lockbox_accessed": False,
    }
    return json.loads(_canonical_bytes(payload).decode("ascii"))


def _write_no_replace(payload: Mapping[str, Any]) -> Path:
    if payload.get("status") != "complete":
        raise ModelDesignEpoch500GateNotReady("fixed gate prerequisites are incomplete")
    path = PROJECT_ROOT / OUTPUT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=".result.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(f"immutable gate already exists: {path}") from exc
            raise
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def validate_existing_result() -> dict[str, Any]:
    observed, _ = _load_json(PROJECT_ROOT / OUTPUT_RELATIVE_PATH)
    expected = evaluate_payload()
    if expected.get("status") != "complete" or _canonical_bytes(observed) != _canonical_bytes(expected):
        raise ModelDesignEpoch500GateError("existing gate differs from fresh evaluation")
    return expected


def run(_args: argparse.Namespace | None = None) -> Path:
    return _write_no_replace(evaluate_payload())


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "ModelDesignEpoch500GateError",
    "ModelDesignEpoch500GateNotReady",
    "OUTPUT_RELATIVE_PATH",
    "RESULT_SCHEMA",
    "evaluate_payload",
    "parse_args",
    "run",
    "validate_existing_result",
]
