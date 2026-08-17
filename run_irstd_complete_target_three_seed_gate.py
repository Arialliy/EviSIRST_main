#!/usr/bin/env python3
"""Evaluate the fixed three-runtime-seed IRSTD-1K validation confirmation gate.

This command has no configurable paths, seeds, metrics, or thresholds.  It
consumes the paired baseline and complete-target final summaries/checkpoints
and one paired-baseline matched-target diagnostic for each of the three
pre-registered runtime seeds.  All inputs are validation-only artifacts.

The 95% paired t intervals in the result are descriptive uncertainty summaries
for only three fixed-initialization runtime repetitions; they are not a claim
of statistical significance or three independent model initializations.
Public-test evaluation remains unconditionally blocked.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import hashlib
import json
import math
import os
import stat
import statistics
import tempfile
import threading
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

import run_irstd_complete_target_promotion_gate as pilot_gate
import run_irstd_paired_baseline_confirmatory_diagnostic as confirmatory_diagnostic
import train_irstd_complete_target_replication_v1 as replication_runner


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET = "IRSTD-1K"
DATA_ROLE = "val"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
EPOCHS = 1000
RUN_SEEDS = (1446202191, 104728269, 262620274)

SAFETY_PD_FAILURE_THRESHOLD = Decimal("-0.003")
SAFETY_FA_FAILURE_THRESHOLD = Decimal("0")
REQUIRED_POSITIVE_SEEDS = 2
T_CRITICAL_95_DF2 = 4.302652729911275

RESULT_SCHEMA = (
    "evisirst_irstd_complete_target_three_runtime_seed_confirmation_gate_result/v1"
)
OUTPUT_RELATIVE_PATH = (
    "runs/irstd_performance/complete_target_v1/"
    "three_seed_confirmation_gate/result.json"
)
GATE_SOURCE_RELATIVE_PATH = (
    "run_irstd_complete_target_three_seed_gate.py"
)
RULES_RELATIVE_PATH = (
    "experiments/irstd_complete_target_three_seed_rules_v1.json"
)


def _seed_paths(seed: int) -> dict[str, str]:
    if seed not in RUN_SEEDS:
        raise ThreeSeedGateError(f"seed {seed!r} is not pre-registered")
    variant_stage = (
        "formal"
        if seed == RUN_SEEDS[0]
        else "three_runtime_seed_validation/formal"
    )
    return {
        "baseline_summary": (
            "runs/validation_selected/formal/IRSTD-1K/binary/"
            f"run_seed_{seed}/summary.json"
        ),
        "baseline_checkpoint": (
            "runs/validation_selected/formal/IRSTD-1K/binary/"
            f"run_seed_{seed}/EviSIRST.pth.tar"
        ),
        "variant_summary": (
            f"runs/irstd_performance/complete_target_v1/{variant_stage}/"
            f"IRSTD-1K/binary/run_seed_{seed}/summary.json"
        ),
        "variant_checkpoint": (
            f"runs/irstd_performance/complete_target_v1/{variant_stage}/"
            f"IRSTD-1K/binary/run_seed_{seed}/EviSIRST.pth.tar"
        ),
        "baseline_diagnostic": (
            "runs/irstd_performance/complete_target_v1/"
            f"paired_baseline_diagnostics/run_seed_{seed}/"
            "matched_target_diagnostics.json"
        ),
    }


SEED_INPUT_PATHS = {seed: _seed_paths(seed) for seed in RUN_SEEDS}

PREREGISTERED_RULES = {
    "schema": "evisirst_irstd_complete_target_three_seed_gate_rules/v1",
    "registered_before_confirmatory_training": True,
    "runtime_seeds": list(RUN_SEEDS),
    "replication_scope": (
        "fixed_architecture_seed_42_runtime_randomness_repetitions;"
        "not_independent_initializations"
    ),
    "primary": {
        "metric": "selected_validation_mIoU",
        "delta": "complete_target_minus_paired_baseline",
        "pass_rule": "mean_delta>0 AND strictly_positive_seed_count>=2",
        "mean_operator": ">",
        "mean_threshold": 0.0,
        "seed_operator": ">",
        "seed_threshold": 0.0,
        "minimum_positive_seed_count": REQUIRED_POSITIVE_SEEDS,
    },
    "safety": {
        "per_seed_failure_rule": "Pd_delta<-0.003 AND Fa_delta>0",
        "aggregate_failure_rule": "mean_Pd_delta<-0.003 AND mean_Fa_delta>0",
        "final_failure_rule": (
            "any_per_seed_joint_failure OR aggregate_joint_failure"
        ),
        "pd_failure_operator": "<",
        "pd_failure_threshold": float(SAFETY_PD_FAILURE_THRESHOLD),
        "fa_failure_operator": ">",
        "fa_failure_threshold": float(SAFETY_FA_FAILURE_THRESHOLD),
    },
    "mechanism": {
        "pass_rule": (
            "at_least_one_same_endpoint_has_mean_delta>0_AND_"
            "strictly_positive_seed_count>=2"
        ),
        "endpoint_combination": "no_cross_endpoint_vote_pooling",
        "endpoints": [
            {
                "name": "matched_target_pixel_recall",
                "transform": "identity",
            },
            {
                "name": "matched_component_area_ratio_closeness_to_one",
                "source_metric": "matched_component_area_ratio",
                "transform": "-abs(raw_value-1.0)",
            },
        ],
        "mean_operator": ">",
        "mean_threshold": 0.0,
        "seed_operator": ">",
        "seed_threshold": 0.0,
        "minimum_positive_seed_count": REQUIRED_POSITIVE_SEEDS,
    },
    "uncertainty": {
        "summary": "paired_delta_mean_sample_std_and_two_sided_95pct_t_interval",
        "sample_std_ddof": 1,
        "degrees_of_freedom": 2,
        "t_critical": T_CRITICAL_95_DF2,
        "decision_use": False,
        "interpretation": (
            "descriptive_only_not_a_statistical_significance_claim"
        ),
    },
    "decision": {
        "overall_pass_rule": "primary AND safety AND mechanism",
        "validation_improvement_confirmation_only": True,
        "public_test_allowed": False,
        "public_test_status": "blocked_pending_separate_reviewed_gate_extension",
    },
}
PREREGISTERED_RULES_SHA256 = (
    "36b503e9b5906e1118da670313cf2bbf7b131995d176aba70f4936f7de473ef2"
)


class ThreeSeedGateError(ValueError):
    """A fixed input does not prove the pre-registered confirmation gate."""


_PILOT_GATE_PATCH_LOCK = threading.Lock()
_PILOT_GATE_PATCH_FIELDS = (
    "RUN_SEED",
    "BASELINE_SUMMARY_RELATIVE_PATH",
    "BASELINE_CHECKPOINT_RELATIVE_PATH",
    "VARIANT_SUMMARY_RELATIVE_PATH",
    "VARIANT_CHECKPOINT_RELATIVE_PATH",
    "BASELINE_DIAGNOSTIC_RELATIVE_PATH",
    "VARIANT_TRAINING_SCHEMA",
    "VARIANT_CHECKPOINT_SCHEMA",
    "VARIANT_EXPERIMENT_SCHEMA",
)


@contextlib.contextmanager
def _bound_pilot_validator(seed: int):
    """Bind the audited single-seed validator to one fixed path set."""

    if seed not in RUN_SEEDS:
        raise ThreeSeedGateError("validator seed is not pre-registered")
    if not _PILOT_GATE_PATCH_LOCK.acquire(blocking=False):
        raise ThreeSeedGateError("single-seed validator is already bound")
    paths = SEED_INPUT_PATHS[seed]
    previous = {
        name: getattr(pilot_gate, name) for name in _PILOT_GATE_PATCH_FIELDS
    }
    try:
        pilot_gate.RUN_SEED = seed
        pilot_gate.BASELINE_SUMMARY_RELATIVE_PATH = paths["baseline_summary"]
        pilot_gate.BASELINE_CHECKPOINT_RELATIVE_PATH = paths[
            "baseline_checkpoint"
        ]
        pilot_gate.VARIANT_SUMMARY_RELATIVE_PATH = paths["variant_summary"]
        pilot_gate.VARIANT_CHECKPOINT_RELATIVE_PATH = paths[
            "variant_checkpoint"
        ]
        pilot_gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH = paths[
            "baseline_diagnostic"
        ]
        if seed != RUN_SEEDS[0]:
            pilot_gate.VARIANT_TRAINING_SCHEMA = (
                replication_runner.TRAINING_SCHEMA
            )
            pilot_gate.VARIANT_CHECKPOINT_SCHEMA = (
                replication_runner.CHECKPOINT_SCHEMA
            )
            pilot_gate.VARIANT_EXPERIMENT_SCHEMA = (
                replication_runner.EXPERIMENT_SCHEMA
            )
        yield paths
    finally:
        for name, value in previous.items():
            setattr(pilot_gate, name, value)
        _PILOT_GATE_PATCH_LOCK.release()


@contextlib.contextmanager
def _canonical_pilot_gate_bindings():
    """Temporarily restore the completed-pilot constants for authority checks."""

    previous = {
        name: getattr(pilot_gate, name) for name in _PILOT_GATE_PATCH_FIELDS
    }
    pilot_paths = SEED_INPUT_PATHS[RUN_SEEDS[0]]
    try:
        pilot_gate.RUN_SEED = RUN_SEEDS[0]
        pilot_gate.BASELINE_SUMMARY_RELATIVE_PATH = pilot_paths[
            "baseline_summary"
        ]
        pilot_gate.BASELINE_CHECKPOINT_RELATIVE_PATH = pilot_paths[
            "baseline_checkpoint"
        ]
        pilot_gate.VARIANT_SUMMARY_RELATIVE_PATH = pilot_paths[
            "variant_summary"
        ]
        pilot_gate.VARIANT_CHECKPOINT_RELATIVE_PATH = pilot_paths[
            "variant_checkpoint"
        ]
        pilot_gate.BASELINE_DIAGNOSTIC_RELATIVE_PATH = pilot_paths[
            "baseline_diagnostic"
        ]
        pilot_gate.VARIANT_TRAINING_SCHEMA = (
            "evisirst_irstd_complete_target_training/v1"
        )
        pilot_gate.VARIANT_CHECKPOINT_SCHEMA = (
            "evisirst_irstd_complete_target_checkpoint/v1"
        )
        pilot_gate.VARIANT_EXPERIMENT_SCHEMA = (
            "evisirst_irstd_complete_target_experiment/v1"
        )
        yield
    finally:
        for name, value in previous.items():
            setattr(pilot_gate, name, value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ThreeSeedGateError("value is not strict finite JSON data") from exc


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_path(relative_path: str, *, must_exist: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ThreeSeedGateError("internal artifact path is unsafe")
    root = PROJECT_ROOT.resolve(strict=True)
    path = root.joinpath(*pure.parts)
    current = root
    for component in pure.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise ThreeSeedGateError(
                f"artifact path contains a symlink: {relative_path}"
            )
    if must_exist:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.resolve(strict=True) != path:
            raise ThreeSeedGateError(
                f"artifact path was redirected: {relative_path}"
            )
    return path


def _artifact_metadata(relative_path: str) -> dict[str, str]:
    path = _fixed_path(relative_path, must_exist=True)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ThreeSeedGateError("gate source is not a regular file")
        content = handle.read()
        after = os.fstat(handle.fileno())
    if (
        not content
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ThreeSeedGateError("gate source is empty or changed while read")
    return {"relative_path": relative_path, "sha256": _sha256_bytes(content)}


def _load_and_validate_frozen_rules() -> dict[str, str]:
    metadata = _artifact_metadata(RULES_RELATIVE_PATH)
    path = _fixed_path(RULES_RELATIVE_PATH, must_exist=True)
    try:
        observed = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ThreeSeedGateError("frozen rules file is not strict JSON") from exc
    canonical = _canonical_json_bytes(observed)
    if observed != PREREGISTERED_RULES:
        raise ThreeSeedGateError("frozen rules file differs from embedded rules")
    if _sha256_bytes(canonical) != PREREGISTERED_RULES_SHA256:
        raise ThreeSeedGateError("frozen rules canonical SHA-256 differs")
    return metadata


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThreeSeedGateError(f"{label} must be a numeric scalar")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ThreeSeedGateError(f"{label} must be finite")
    return number


def _paired_summary(values: Sequence[Decimal]) -> dict[str, Any]:
    if len(values) != len(RUN_SEEDS):
        raise ThreeSeedGateError("paired statistic requires exactly three values")
    floats = [float(value) for value in values]
    mean = statistics.fmean(floats)
    sample_std = statistics.stdev(floats)
    margin = T_CRITICAL_95_DF2 * sample_std / math.sqrt(len(floats))
    return {
        "n": len(floats),
        "mean": mean,
        "sample_std_ddof_1": sample_std,
        "descriptive_95pct_t_interval": [mean - margin, mean + margin],
        "degrees_of_freedom": 2,
        "t_critical": T_CRITICAL_95_DF2,
        "decision_use": False,
        "interpretation": "descriptive_only_not_a_statistical_significance_claim",
    }


def _strictly_positive_count(values: Sequence[Decimal]) -> int:
    return sum(value > Decimal("0") for value in values)


def _decimal_mean(values: Sequence[Decimal]) -> Decimal:
    if len(values) != len(RUN_SEEDS):
        raise ThreeSeedGateError("gate mean requires exactly three values")
    return sum(values, Decimal("0")) / Decimal(len(values))


def _metric_triplet(
    baseline: Any, variant: Any, *, label: str
) -> tuple[float, float, Decimal]:
    baseline_decimal = _decimal(baseline, label=f"{label}.baseline")
    variant_decimal = _decimal(variant, label=f"{label}.variant")
    return (
        float(baseline_decimal),
        float(variant_decimal),
        variant_decimal - baseline_decimal,
    )


def _selected_metric_view(
    selected: Mapping[str, Any], mechanism: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    metrics = selected.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ThreeSeedGateError(f"{label} selected metrics are missing")
    fixed_pairs = (("mIoU", "miou"), ("Pd", "pd"), ("Fa", "fa"))
    output: dict[str, Any] = {}
    for top_name, nested_name in fixed_pairs:
        top = _decimal(selected.get(top_name), label=f"{label}.{top_name}")
        nested = _decimal(
            metrics.get(nested_name), label=f"{label}.metrics.{nested_name}"
        )
        if top != nested:
            raise ThreeSeedGateError(
                f"{label} top-level and nested {top_name} differ"
            )
        output[top_name] = float(top)
    niou = _decimal(metrics.get("niou"), label=f"{label}.metrics.niou")
    if not Decimal("0") <= niou <= Decimal("1"):
        raise ThreeSeedGateError(f"{label}.metrics.niou is outside [0,1]")
    output["nIoU"] = float(niou)
    output["mechanism"] = {
        "matched_target_pixel_recall": float(
            _decimal(
                mechanism.get("matched_target_pixel_recall"),
                label=f"{label}.mechanism.recall",
            )
        ),
        "matched_component_area_ratio": float(
            _decimal(
                mechanism.get("matched_component_area_ratio"),
                label=f"{label}.mechanism.area_ratio",
            )
        ),
    }
    return output


def _validate_confirmatory_diagnostic_wrapper(
    diagnostic: Mapping[str, Any],
    *,
    seed: int,
    diagnostic_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate all confirmatory wrapper fields and recover its core payload."""

    try:
        with _canonical_pilot_gate_bindings():
            strictly_validated = (
                confirmatory_diagnostic.validate_existing_result(seed)
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise ThreeSeedGateError(
            f"seed {seed} strict confirmatory diagnostic validation failed"
        ) from exc
    if not isinstance(strictly_validated, Mapping) or _canonical_json_bytes(
        strictly_validated
    ) != _canonical_json_bytes(diagnostic):
        raise ThreeSeedGateError(
            f"seed {seed} strict confirmatory diagnostic evidence differs"
        )
    if diagnostic_metadata is not None:
        paths = confirmatory_diagnostic._relative_paths(seed)
        expected_relative_path = paths["OUTPUT_RELATIVE_PATH"]
        if (
            set(diagnostic_metadata) != {"relative_path", "sha256"}
            or diagnostic_metadata.get("relative_path")
            != expected_relative_path
        ):
            raise ThreeSeedGateError(
                f"seed {seed} strict confirmatory diagnostic output path differs"
            )
        pilot_gate._require_sha256(
            diagnostic_metadata.get("sha256"),
            label=f"seed {seed} strict confirmatory diagnostic output",
        )

    expected_top = set(pilot_gate._DIAGNOSTIC_KEYS) | {
        "run_seed",
        "promotion_prerequisite",
        "diagnostic_identity",
    }
    if set(diagnostic) != expected_top:
        raise ThreeSeedGateError(
            f"seed {seed} confirmatory diagnostic top-level keys differ"
        )
    if (
        diagnostic.get("schema") != confirmatory_diagnostic.DIAGNOSTIC_SCHEMA
        or diagnostic.get("run_seed") != seed
        or diagnostic.get("test_split_accessed") is not False
    ):
        raise ThreeSeedGateError(
            f"seed {seed} confirmatory diagnostic identity differs"
        )
    sources = diagnostic.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "checkpoint_training_source_tree_sha256",
        "checkpoint_training_sources_currently_verified",
        "diagnostic",
        "confirmatory_diagnostic",
    }:
        raise ThreeSeedGateError(
            f"seed {seed} confirmatory diagnostic sources differ"
        )
    core_sources = sources.get("diagnostic")
    confirmatory_sources = sources.get("confirmatory_diagnostic")
    current_core_sources = (
        confirmatory_diagnostic.diagnostic_core.diagnostic_source_provenance()
    )
    current_confirmatory_sources = (
        confirmatory_diagnostic.confirmatory_source_provenance()
    )
    if (
        core_sources != current_core_sources
        or confirmatory_sources != current_confirmatory_sources
    ):
        raise ThreeSeedGateError(
            f"seed {seed} confirmatory diagnostic source hashes differ"
        )
    authority = diagnostic.get("promotion_prerequisite")
    try:
        with _canonical_pilot_gate_bindings():
            expected_authority = (
                confirmatory_diagnostic.validate_pilot_authority()
            )
    except (OSError, ValueError) as exc:
        raise ThreeSeedGateError(
            "pilot promotion authority failed fresh validation"
        ) from exc
    if authority != expected_authority:
        raise ThreeSeedGateError(
            f"seed {seed} confirmatory promotion authority differs"
        )

    core = copy.deepcopy(dict(diagnostic))
    del core["run_seed"]
    del core["promotion_prerequisite"]
    del core["diagnostic_identity"]
    core["schema"] = pilot_gate.BASELINE_DIAGNOSTIC_SCHEMA
    core["sources"] = {
        name: copy.deepcopy(value)
        for name, value in sources.items()
        if name != "confirmatory_diagnostic"
    }
    paths = confirmatory_diagnostic._relative_paths(seed)
    try:
        expected = confirmatory_diagnostic.build_result_payload(
            run_seed=seed,
            paths=paths,
            core_payload=core,
            core_sources=current_core_sources,
            confirmatory_sources=current_confirmatory_sources,
            pilot_authority=expected_authority,
        )
    except (OSError, ValueError) as exc:
        raise ThreeSeedGateError(
            f"seed {seed} confirmatory diagnostic reconstruction failed"
        ) from exc
    if _canonical_json_bytes(diagnostic) != _canonical_json_bytes(expected):
        raise ThreeSeedGateError(
            f"seed {seed} confirmatory diagnostic identity/hash differs"
        )
    return core


def _validate_baseline_and_diagnostic(
    seed: int,
) -> dict[str, Any]:
    paths = SEED_INPUT_PATHS[seed]
    baseline_summary, baseline_summary_meta = pilot_gate._load_json(
        paths["baseline_summary"], label=f"seed {seed} paired baseline summary"
    )
    baseline_checkpoint, baseline_checkpoint_meta = pilot_gate._load_checkpoint(
        paths["baseline_checkpoint"],
        label=f"seed {seed} paired baseline checkpoint",
    )
    diagnostic, diagnostic_meta = pilot_gate._load_json(
        paths["baseline_diagnostic"],
        label=f"seed {seed} paired baseline diagnostic",
    )
    baseline_selection, baseline_selected = pilot_gate._validate_summary_common(
        baseline_summary, variant=False
    )
    baseline_identity, baseline_source_tree = pilot_gate._validate_checkpoint(
        baseline_checkpoint,
        baseline_checkpoint_meta,
        summary=baseline_summary,
        selection=baseline_selection,
        variant=False,
    )
    if seed == RUN_SEEDS[0]:
        if diagnostic.get("schema") != pilot_gate.BASELINE_DIAGNOSTIC_SCHEMA:
            raise ThreeSeedGateError("pilot diagnostic schema differs")
        core_diagnostic = diagnostic
    else:
        core_diagnostic = _validate_confirmatory_diagnostic_wrapper(
            diagnostic, seed=seed, diagnostic_metadata=diagnostic_meta
        )
    baseline_mechanism = pilot_gate._validate_diagnostic(
        core_diagnostic,
        diagnostic_meta,
        baseline_summary=baseline_summary,
        baseline_summary_metadata=baseline_summary_meta,
        baseline_checkpoint_metadata=baseline_checkpoint_meta,
        baseline_training_identity_sha256=baseline_identity["identity_sha256"],
        baseline_source_tree=baseline_source_tree,
        baseline_selection=baseline_selection,
        baseline_selected_record=baseline_selected,
    )
    return {
        "summary": baseline_summary,
        "summary_meta": baseline_summary_meta,
        "checkpoint_meta": baseline_checkpoint_meta,
        "diagnostic_meta": diagnostic_meta,
        "selection": baseline_selection,
        "selected": baseline_selected,
        "identity": baseline_identity,
        "source_tree": baseline_source_tree,
        "mechanism": baseline_mechanism,
        "diagnostic": diagnostic,
        "core_diagnostic": core_diagnostic,
    }


def _revalidate_diagnostic_sources(
    seed: int, baseline: Mapping[str, Any]
) -> None:
    core = baseline.get("core_diagnostic")
    if not isinstance(core, Mapping):
        raise ThreeSeedGateError(f"seed {seed} diagnostic core is missing")
    sources = core.get("sources")
    diagnostic_sources = (
        sources.get("diagnostic") if isinstance(sources, Mapping) else None
    )
    if not isinstance(diagnostic_sources, Mapping):
        raise ThreeSeedGateError(f"seed {seed} diagnostic sources are missing")
    observed_tree = pilot_gate._validate_source_set(
        {
            "source_set_schema": diagnostic_sources.get("schema"),
            "source_files": diagnostic_sources.get("files"),
            "source_tree_sha256": diagnostic_sources.get("source_tree_sha256"),
        },
        expected_schema="evisirst_paired_baseline_diagnostic_source_set/v1",
        expected_paths=pilot_gate._DIAGNOSTIC_SOURCE_PATHS,
        label=f"seed {seed} diagnostic final source check",
    )
    if diagnostic_sources.get("source_tree_sha256") != observed_tree:
        raise ThreeSeedGateError(f"seed {seed} diagnostic source tree changed")
    if seed != RUN_SEEDS[0]:
        diagnostic = baseline.get("diagnostic")
        wrapped_sources = (
            diagnostic.get("sources")
            if isinstance(diagnostic, Mapping)
            else None
        )
        recorded = (
            wrapped_sources.get("confirmatory_diagnostic")
            if isinstance(wrapped_sources, Mapping)
            else None
        )
        if recorded != confirmatory_diagnostic.confirmatory_source_provenance():
            raise ThreeSeedGateError(
                f"seed {seed} confirmatory diagnostic sources changed"
            )


def build_result_payload(
    *,
    seed_evidence: Sequence[Mapping[str, Any]],
    gate_source_artifact: Mapping[str, str],
    frozen_rules_artifact: Mapping[str, str],
) -> dict[str, Any]:
    """Aggregate three already-strictly-validated paired seed records."""

    if len(seed_evidence) != len(RUN_SEEDS):
        raise ThreeSeedGateError("exactly three seed evidence records are required")
    observed_rules_sha256 = _sha256_bytes(
        _canonical_json_bytes(PREREGISTERED_RULES)
    )
    if observed_rules_sha256 != PREREGISTERED_RULES_SHA256:
        raise ThreeSeedGateError("pre-registered rule bytes changed")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for record in seed_evidence:
        seed = record.get("run_seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in RUN_SEEDS
            or seed in by_seed
        ):
            raise ThreeSeedGateError("seed evidence set/order/uniqueness differs")
        by_seed[int(seed)] = record
    if set(by_seed) != set(RUN_SEEDS):
        raise ThreeSeedGateError("seed evidence set differs")

    delta_series: dict[str, list[Decimal]] = {
        name: []
        for name in (
            "mIoU",
            "nIoU",
            "Pd",
            "Fa",
            "matched_target_pixel_recall",
            "matched_component_area_ratio_closeness_to_one",
        )
    }
    per_seed: list[dict[str, Any]] = []
    per_seed_safety_failures: list[int] = []

    for seed in RUN_SEEDS:
        record = by_seed[seed]
        if record.get("test_split_accessed") is not False:
            raise ThreeSeedGateError(f"seed {seed} test ledger is not false")
        for epoch_name in (
            "baseline_selected_epoch",
            "variant_selected_epoch",
        ):
            epoch = record.get(epoch_name)
            if (
                isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or not 1 <= epoch <= EPOCHS
            ):
                raise ThreeSeedGateError(
                    f"seed {seed} {epoch_name} is outside completed history"
                )
        baseline = record.get("baseline")
        variant = record.get("variant")
        if not isinstance(baseline, Mapping) or not isinstance(variant, Mapping):
            raise ThreeSeedGateError(f"seed {seed} metrics are missing")
        row_metrics: dict[str, Any] = {}
        for name in ("mIoU", "nIoU", "Pd", "Fa"):
            base, changed, delta = _metric_triplet(
                baseline.get(name), variant.get(name), label=f"seed {seed}.{name}"
            )
            delta_series[name].append(delta)
            row_metrics[name] = {
                "baseline": base,
                "variant": changed,
                "delta": float(delta),
            }

        baseline_mechanism = baseline.get("mechanism")
        variant_mechanism = variant.get("mechanism")
        if not isinstance(baseline_mechanism, Mapping) or not isinstance(
            variant_mechanism, Mapping
        ):
            raise ThreeSeedGateError(f"seed {seed} mechanism metrics are missing")
        recall_base, recall_variant, recall_delta = _metric_triplet(
            baseline_mechanism.get("matched_target_pixel_recall"),
            variant_mechanism.get("matched_target_pixel_recall"),
            label=f"seed {seed}.matched_target_pixel_recall",
        )
        baseline_area = _decimal(
            baseline_mechanism.get("matched_component_area_ratio"),
            label=f"seed {seed}.baseline area ratio",
        )
        variant_area = _decimal(
            variant_mechanism.get("matched_component_area_ratio"),
            label=f"seed {seed}.variant area ratio",
        )
        baseline_closeness = -abs(baseline_area - Decimal("1"))
        variant_closeness = -abs(variant_area - Decimal("1"))
        closeness_delta = variant_closeness - baseline_closeness
        delta_series["matched_target_pixel_recall"].append(recall_delta)
        delta_series[
            "matched_component_area_ratio_closeness_to_one"
        ].append(closeness_delta)

        pd_delta = delta_series["Pd"][-1]
        fa_delta = delta_series["Fa"][-1]
        safety_failed = (
            pd_delta < SAFETY_PD_FAILURE_THRESHOLD
            and fa_delta > SAFETY_FA_FAILURE_THRESHOLD
        )
        if safety_failed:
            per_seed_safety_failures.append(seed)

        per_seed.append(
            {
                "run_seed": seed,
                "baseline_selected_epoch": record.get("baseline_selected_epoch"),
                "variant_selected_epoch": record.get("variant_selected_epoch"),
                "metrics": row_metrics,
                "mechanism": {
                    "matched_target_pixel_recall": {
                        "baseline": recall_base,
                        "variant": recall_variant,
                        "delta": float(recall_delta),
                    },
                    "matched_component_area_ratio_closeness_to_one": {
                        "baseline_raw_area_ratio": float(baseline_area),
                        "variant_raw_area_ratio": float(variant_area),
                        "baseline_transformed": float(baseline_closeness),
                        "variant_transformed": float(variant_closeness),
                        "delta_transformed": float(closeness_delta),
                    },
                },
                "safety_joint_failure": safety_failed,
                "verified_inputs": record.get("verified_inputs"),
                "test_split_accessed": False,
            }
        )

    statistics_payload = {
        name: _paired_summary(values) for name, values in delta_series.items()
    }
    miou_mean = _decimal_mean(delta_series["mIoU"])
    miou_positive_count = _strictly_positive_count(delta_series["mIoU"])
    primary_passed = (
        miou_mean > Decimal("0")
        and miou_positive_count >= REQUIRED_POSITIVE_SEEDS
    )

    mean_pd_delta = _decimal_mean(delta_series["Pd"])
    mean_fa_delta = _decimal_mean(delta_series["Fa"])
    aggregate_safety_failed = (
        mean_pd_delta < SAFETY_PD_FAILURE_THRESHOLD
        and mean_fa_delta > SAFETY_FA_FAILURE_THRESHOLD
    )
    safety_failed = bool(per_seed_safety_failures) or aggregate_safety_failed

    mechanism_endpoints: dict[str, Any] = {}
    for endpoint in (
        "matched_target_pixel_recall",
        "matched_component_area_ratio_closeness_to_one",
    ):
        values = delta_series[endpoint]
        positive_count = _strictly_positive_count(values)
        mean_delta = _decimal_mean(values)
        passed = (
            mean_delta > Decimal("0")
            and positive_count >= REQUIRED_POSITIVE_SEEDS
        )
        mechanism_endpoints[endpoint] = {
            "mean_delta": float(mean_delta),
            "strictly_positive_seed_count": positive_count,
            "minimum_positive_seed_count": REQUIRED_POSITIVE_SEEDS,
            "passed": passed,
        }
    mechanism_passed = any(
        endpoint["passed"] for endpoint in mechanism_endpoints.values()
    )
    overall_passed = primary_passed and not safety_failed and mechanism_passed

    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": DATA_ROLE,
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "epochs": EPOCHS,
        "runtime_seeds": list(RUN_SEEDS),
        "rules": PREREGISTERED_RULES,
        "rules_sha256": PREREGISTERED_RULES_SHA256,
        "inputs": {
            "fixed_paths": {
                str(seed): SEED_INPUT_PATHS[seed] for seed in RUN_SEEDS
            },
            "gate_evaluator_source": dict(gate_source_artifact),
            "frozen_rules_artifact": dict(frozen_rules_artifact),
            "all_three_seed_records_strictly_verified": True,
            "all_artifact_hashes_verified": True,
            "all_source_hashes_currently_verified": True,
            "all_selections_recomputed": True,
            "all_histories_continuous_through_epoch_1000": True,
            "canonical_train_val_split_verified": True,
        },
        "per_seed": per_seed,
        "paired_delta_statistics": statistics_payload,
        "primary_gate": {
            "metric": "selected_validation_mIoU",
            "mean_delta": float(miou_mean),
            "mean_operator": ">",
            "mean_threshold": 0.0,
            "strictly_positive_seed_count": miou_positive_count,
            "minimum_positive_seed_count": REQUIRED_POSITIVE_SEEDS,
            "passed": primary_passed,
        },
        "safety_gate": {
            "per_seed_joint_failure_seeds": per_seed_safety_failures,
            "aggregate_mean_pd_delta": float(mean_pd_delta),
            "aggregate_mean_fa_delta": float(mean_fa_delta),
            "aggregate_joint_failure": aggregate_safety_failed,
            "failed": safety_failed,
            "passed": not safety_failed,
        },
        "mechanism_gate": {
            "endpoint_combination": "no_cross_endpoint_vote_pooling",
            "endpoints": mechanism_endpoints,
            "passed": mechanism_passed,
        },
        "decision": {
            "overall_passed": overall_passed,
            "result": "PASS" if overall_passed else "FAIL",
            "validation_improvement_confirmed": overall_passed,
            "public_test_allowed": False,
            "public_test_status": (
                "blocked_pending_separate_reviewed_gate_extension"
            ),
        },
        "test_split_accessed": False,
        "public_test_allowed": False,
    }
    return json.loads(_canonical_json_bytes(payload).decode("ascii"))


def _validate_seed_evidence(seed: int) -> dict[str, Any]:
    """Strictly validate one fixed seed and return normalized evidence.

    The adapters are deliberately seed/schema-specific.  The pilot adapter and
    confirmatory adapter are completed below; no generic or permissive schema
    fallback is allowed here.
    """

    with _bound_pilot_validator(seed) as paths:
        try:
            pilot_gate._validate_canonical_split_files()
            baseline = _validate_baseline_and_diagnostic(seed)
            if seed != RUN_SEEDS[0]:
                return _validate_replication_variant(
                    seed=seed, paths=paths, baseline=baseline
                )

            variant_summary, variant_summary_meta = pilot_gate._load_json(
                paths["variant_summary"], label="pilot complete-target summary"
            )
            variant_checkpoint, variant_checkpoint_meta = (
                pilot_gate._load_checkpoint(
                    paths["variant_checkpoint"],
                    label="pilot complete-target final checkpoint",
                )
            )
            variant_selection, variant_selected = (
                pilot_gate._validate_summary_common(
                    variant_summary, variant=True
                )
            )
            variant_identity, variant_source_tree = (
                pilot_gate._validate_checkpoint(
                    variant_checkpoint,
                    variant_checkpoint_meta,
                    summary=variant_summary,
                    selection=variant_selection,
                    variant=True,
                )
            )
            variant_metrics = variant_selected.get("metrics")
            if not isinstance(variant_metrics, Mapping):
                raise ThreeSeedGateError("pilot variant metrics are missing")
            variant_mechanism = pilot_gate._validate_mechanism(
                variant_metrics.get(
                    "complete_target_mechanism_diagnostics"
                ),
                label="pilot variant mechanism",
            )
            try:
                with _canonical_pilot_gate_bindings():
                    pilot_result = pilot_gate.validate_existing_result()
            except (OSError, ValueError) as exc:
                raise ThreeSeedGateError(
                    "pilot promotion result failed fresh validation"
                ) from exc
            if (
                pilot_result.get("decision", {}).get("result") != "PASS"
                or pilot_result.get("decision", {}).get(
                    "three_runtime_seed_validation_expansion_allowed"
                )
                is not True
                or pilot_result.get("test_split_accessed") is not False
            ):
                raise ThreeSeedGateError(
                    "pilot result does not authorize confirmation"
                )
            observed = [
                baseline["summary_meta"],
                baseline["checkpoint_meta"],
                baseline["diagnostic_meta"],
                variant_summary_meta,
                variant_checkpoint_meta,
            ]
            pilot_gate._assert_inputs_unchanged(observed)
            pilot_gate._validate_canonical_split_files()
            if pilot_gate._validate_source_set(
                baseline["identity"]["determinism_protocol"],
                expected_schema="evisirst_validation_selected_source_set/v2",
                expected_paths=pilot_gate._BASELINE_SOURCE_PATHS,
                label="pilot baseline final source check",
            ) != baseline["source_tree"]:
                raise ThreeSeedGateError("pilot baseline source tree changed")
            _revalidate_diagnostic_sources(seed, baseline)
            if pilot_gate._validate_source_set(
                variant_identity["determinism_protocol"],
                expected_schema="evisirst_irstd_complete_target_source_set/v1",
                expected_paths=pilot_gate._VARIANT_SOURCE_PATHS,
                label="pilot variant final source check",
            ) != variant_source_tree:
                raise ThreeSeedGateError("pilot variant source tree changed")
            return _normalized_seed_record(
                seed=seed,
                baseline=baseline,
                variant_summary_meta=variant_summary_meta,
                variant_checkpoint_meta=variant_checkpoint_meta,
                variant_selection=variant_selection,
                variant_selected=variant_selected,
                variant_identity=variant_identity,
                variant_source_tree=variant_source_tree,
                variant_mechanism=variant_mechanism,
            )
        except ThreeSeedGateError:
            raise
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise ThreeSeedGateError(
                f"seed {seed} failed strict artifact validation"
            ) from exc


def _normalized_seed_record(
    *,
    seed: int,
    baseline: Mapping[str, Any],
    variant_summary_meta: Mapping[str, str],
    variant_checkpoint_meta: Mapping[str, str],
    variant_selection: Mapping[str, Any],
    variant_selected: Mapping[str, Any],
    variant_identity: Mapping[str, Any],
    variant_source_tree: str,
    variant_mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_selected = baseline["selected"]
    return {
        "run_seed": seed,
        "baseline_selected_epoch": baseline_selected["epoch"],
        "variant_selected_epoch": variant_selected["epoch"],
        "baseline": _selected_metric_view(
            baseline_selected,
            baseline["mechanism"],
            label=f"seed {seed} baseline selected record",
        ),
        "variant": _selected_metric_view(
            variant_selected,
            variant_mechanism,
            label=f"seed {seed} variant selected record",
        ),
        "verified_inputs": {
            "paired_baseline": {
                "summary": dict(baseline["summary_meta"]),
                "final_checkpoint": dict(baseline["checkpoint_meta"]),
                "matched_target_diagnostic": dict(
                    baseline["diagnostic_meta"]
                ),
                "training_identity_sha256": baseline["identity"][
                    "identity_sha256"
                ],
                "training_source_tree_sha256": baseline["source_tree"],
                "selection_provenance_sha256": pilot_gate._canonical_sha256(
                    baseline["selection"]
                ),
                "selected_validation_record_sha256": (
                    pilot_gate._canonical_sha256(baseline_selected)
                ),
            },
            "complete_target": {
                "summary": dict(variant_summary_meta),
                "final_checkpoint": dict(variant_checkpoint_meta),
                "training_identity_sha256": variant_identity[
                    "identity_sha256"
                ],
                "training_source_tree_sha256": variant_source_tree,
                "selection_provenance_sha256": pilot_gate._canonical_sha256(
                    variant_selection
                ),
                "selected_validation_record_sha256": (
                    pilot_gate._canonical_sha256(variant_selected)
                ),
            },
            "checkpoint_state_tensor_count": 564,
            "selection_recomputed": True,
            "history_continuous_through_epoch": EPOCHS,
            "source_hashes_currently_verified": True,
            "artifact_hashes_verified": True,
        },
        "test_split_accessed": False,
    }


def _expected_replication_source_set(
    authorization: Mapping[str, Any]
) -> dict[str, Any]:
    parent = authorization.get("frozen_parent_source_set")
    if (
        not isinstance(parent, Mapping)
        or parent.get("schema") != replication_runner.PARENT_SOURCE_SET_SCHEMA
        or parent.get("source_tree_sha256")
        != replication_runner.FROZEN_PARENT_SOURCE_TREE_SHA256
        or not isinstance(parent.get("files"), Mapping)
    ):
        raise ThreeSeedGateError("replication parent source evidence differs")
    files = {
        f"parent/{name}": copy.deepcopy(artifact)
        for name, artifact in sorted(parent["files"].items())
    }
    runner_path = _fixed_path(
        "train_irstd_complete_target_replication_v1.py", must_exist=True
    )
    files["replication_runner"] = {
        "relative_path": "train_irstd_complete_target_replication_v1.py",
        "sha256": _sha256_file(runner_path),
    }
    for name, artifact in files.items():
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "relative_path",
            "sha256",
        }:
            raise ThreeSeedGateError(
                f"replication source artifact {name!r} is malformed"
            )
        path = _fixed_path(str(artifact["relative_path"]), must_exist=True)
        if _sha256_file(path) != artifact["sha256"]:
            raise ThreeSeedGateError(
                f"replication source artifact {name!r} changed"
            )
    return {
        "schema": replication_runner.SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": pilot_gate._canonical_sha256(files),
        "frozen_parent_source_tree_sha256": (
            replication_runner.FROZEN_PARENT_SOURCE_TREE_SHA256
        ),
    }


def _expected_replication_determinism(
    source: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        identity = copy.deepcopy(
            replication_runner._PILOT_DETERMINISM_PROTOCOL_IDENTITY()
        )
    except (OSError, ValueError) as exc:
        raise ThreeSeedGateError(
            "pilot determinism identity failed fresh reconstruction"
        ) from exc
    identity["schema"] = replication_runner.DETERMINISM_SCHEMA
    identity["replication_contract"] = {
        "schema": replication_runner.EXPERIMENT_SCHEMA
        + "/runtime_seed_contract",
        "architecture_seed": ARCHITECTURE_SEED,
        "completed_pilot_run_seed": RUN_SEEDS[0],
        "authorized_three_runtime_seeds": list(RUN_SEEDS),
        "pending_replication_run_seeds": list(RUN_SEEDS[1:]),
        "single_variable_from_pilot": "runtime_seed",
        "public_test_supported": False,
    }
    identity["source_set_schema"] = source["schema"]
    identity["source_files"] = source["files"]
    identity["source_tree_sha256"] = source["source_tree_sha256"]
    identity["frozen_parent_source_tree_sha256"] = (
        replication_runner.FROZEN_PARENT_SOURCE_TREE_SHA256
    )
    return json.loads(_canonical_json_bytes(identity).decode("ascii"))


def _validate_replication_identity(
    raw: Any, *, seed: int, authorization: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    identity = dict(
        pilot_gate._require_exact_keys(
            raw,
            pilot_gate._VARIANT_IDENTITY_KEYS,
            label=f"seed {seed} replication run identity",
        )
    )
    observed_sha = pilot_gate._require_sha256(
        identity.get("identity_sha256"),
        label=f"seed {seed} replication identity SHA-256",
    )
    unhashed = dict(identity)
    del unhashed["identity_sha256"]
    if pilot_gate._canonical_sha256(unhashed) != observed_sha:
        raise ThreeSeedGateError(
            f"seed {seed} replication identity SHA-256 differs"
        )
    fixed = {
        "schema": replication_runner.TRAINING_SCHEMA + "/run_identity",
        "model": "EviSIRST",
        "dataset": DATASET,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": seed,
        "target_mode": TARGET_MODE,
        "epochs": EPOCHS,
        "batch_size": 16,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "val_interval": 1,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "evaluation_head": "out",
        "deep_supervision_probability_heads": 6,
        "deep_supervision_weights": [1.0] * 6,
        "selection_rule": pilot_gate.validation_selection.INDEPENDENT_RULE_VERSION,
        "manifest_sha256": pilot_gate.CANONICAL_MANIFEST_SHA256,
        "split_seed": pilot_gate.CANONICAL_SPLIT_SEED,
        "data_tree_sha256": pilot_gate.CANONICAL_DATA_TREE_SHA256,
        "train_count": pilot_gate.CANONICAL_TRAIN_COUNT,
        "val_count": pilot_gate.CANONICAL_VAL_COUNT,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    for name, expected in fixed.items():
        if type(identity.get(name)) is not type(expected) or identity.get(name) != expected:
            raise ThreeSeedGateError(
                f"seed {seed} replication identity.{name} differs"
            )
    expected_experiment = {
        "schema": replication_runner.EXPERIMENT_SCHEMA,
        "name": "IRSTD-1K complete-target crop v1 runtime-seed replication",
        "status": "confirmatory_validation_only",
        "single_variable_from_completed_pilot": "runtime_seed",
        "architecture_seed_fixed": ARCHITECTURE_SEED,
        "completed_pilot_run_seed": RUN_SEEDS[0],
        "authorized_three_runtime_seeds": list(RUN_SEEDS),
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_pending_separate_review",
    }
    if identity.get("experiment") != expected_experiment:
        raise ThreeSeedGateError(
            f"seed {seed} replication experiment identity differs"
        )
    expected_execution = {
        "single_process_only": True,
        "python_threads_running_variant": 1,
        "data_loader_workers": 0,
        "nonblocking_process_lock": ".complete_target_replication_v1.lock",
    }
    if identity.get("execution_contract") != expected_execution:
        raise ThreeSeedGateError(
            f"seed {seed} replication execution contract differs"
        )
    if identity.get("promotion_gate") != authorization:
        raise ThreeSeedGateError(
            f"seed {seed} replication authorization binding differs"
        )
    source = _expected_replication_source_set(authorization)
    expected_determinism = _expected_replication_determinism(source)
    if identity.get("determinism_protocol") != expected_determinism:
        raise ThreeSeedGateError(
            f"seed {seed} replication determinism/source identity differs"
        )
    canonical = identity.get("canonical_split_contract")
    if canonical != {
        "split_root_relative_path": "splits/v2",
        "manifest_sha256": pilot_gate.CANONICAL_MANIFEST_SHA256,
        "data_tree_sha256": pilot_gate.CANONICAL_DATA_TREE_SHA256,
        "train_count": pilot_gate.CANONICAL_TRAIN_COUNT,
        "val_count": pilot_gate.CANONICAL_VAL_COUNT,
    }:
        raise ThreeSeedGateError(
            f"seed {seed} replication canonical split contract differs"
        )
    crop_sha = pilot_gate._require_sha256(
        identity.get("crop_policy_identity_sha256"),
        label=f"seed {seed} crop policy identity",
    )
    if pilot_gate._canonical_sha256(identity.get("crop_policy")) != crop_sha:
        raise ThreeSeedGateError(
            f"seed {seed} replication crop-policy SHA-256 differs"
        )

    pilot_summary, _pilot_summary_meta = pilot_gate._load_json(
        SEED_INPUT_PATHS[RUN_SEEDS[0]]["variant_summary"],
        label="completed pilot variant summary for replication comparison",
    )
    pilot_identity = pilot_summary.get("run_identity")
    if not isinstance(pilot_identity, Mapping):
        raise ThreeSeedGateError("completed pilot run identity is missing")
    replication_stable = copy.deepcopy(identity)
    pilot_stable = copy.deepcopy(dict(pilot_identity))
    permitted_differences = {
        "schema",
        "run_seed",
        "experiment",
        "execution_contract",
        "runtime_identity",
        "determinism_protocol",
        "promotion_gate",
        "identity_sha256",
    }
    for name in permitted_differences:
        replication_stable.pop(name, None)
        pilot_stable.pop(name, None)
    if replication_stable != pilot_stable:
        raise ThreeSeedGateError(
            f"seed {seed} changes more than runtime/provenance identity"
        )
    runtime = identity.get("runtime_identity")
    pilot_runtime = pilot_identity.get("runtime_identity")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(pilot_runtime, Mapping)
        or set(runtime) != set(pilot_runtime)
        or runtime.get("requested_device") != "cuda:0"
        or runtime.get("physical_device_mapping")
        != "external CUDA_VISIBLE_DEVICES; formal process uses logical cuda:0"
    ):
        raise ThreeSeedGateError(
            f"seed {seed} replication runtime identity differs"
        )
    return identity, str(source["source_tree_sha256"])


def _validate_replication_checkpoint(
    checkpoint: Mapping[str, Any],
    metadata: Mapping[str, str],
    *,
    seed: int,
    summary: Mapping[str, Any],
    selection: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    pilot_gate._require_exact_keys(
        checkpoint,
        pilot_gate._VARIANT_CHECKPOINT_KEYS,
        label=f"seed {seed} replication checkpoint",
    )
    fixed = {
        "schema": replication_runner.CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "checkpoint_role": "experimental_validation_selected",
        "epoch": summary["selected_epoch"],
        "seed": ARCHITECTURE_SEED,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": seed,
        "target_mode": TARGET_MODE,
        "source_selection": "evisirst_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "test_split_accessed": False,
        "smoke": False,
        "data_tree_verified": True,
        "split_seed": pilot_gate.CANONICAL_SPLIT_SEED,
        "split_manifest_sha256": pilot_gate.CANONICAL_MANIFEST_SHA256,
        "data_tree_sha256": pilot_gate.CANONICAL_DATA_TREE_SHA256,
        "experiment_schema": replication_runner.EXPERIMENT_SCHEMA,
        "experiment_status": "experimental_validation_only",
        "only_train_crop_policy_differs_from_R1": True,
        "public_test_supported": False,
        "public_test_gate_status": "unsupported_until_separate_gate_extension",
        "crop_audit_commit_status": "complete",
        "search_run_crop_audit_through_epoch": EPOCHS,
        "selected_checkpoint_crop_audit_through_epoch": summary[
            "selected_epoch"
        ],
        "selected_candidate_sha256": summary["selected_candidate_sha256"],
    }
    for name, expected in fixed.items():
        if type(checkpoint.get(name)) is not type(expected) or checkpoint.get(name) != expected:
            raise ThreeSeedGateError(
                f"seed {seed} replication checkpoint.{name} differs"
            )
    if checkpoint.get("selection_provenance") != selection:
        raise ThreeSeedGateError(
            f"seed {seed} checkpoint selector provenance differs"
        )
    if checkpoint.get("split_provenance") != summary.get("split_provenance"):
        raise ThreeSeedGateError(
            f"seed {seed} checkpoint split provenance differs"
        )
    if checkpoint.get("normalization") != summary.get("normalization"):
        raise ThreeSeedGateError(
            f"seed {seed} checkpoint normalization differs"
        )
    pilot_gate._validate_state_dict(
        checkpoint.get("state_dict"),
        label=f"seed {seed} replication checkpoint state_dict",
    )
    identity, source_tree = _validate_replication_identity(
        checkpoint.get("training"), seed=seed, authorization=authorization
    )
    if (
        checkpoint.get("training_identity_sha256")
        != identity["identity_sha256"]
        or summary.get("training_identity_sha256")
        != identity["identity_sha256"]
        or summary.get("run_identity") != identity
    ):
        raise ThreeSeedGateError(
            f"seed {seed} replication training identity binding differs"
        )
    if (
        checkpoint.get("promotion_gate") != authorization
        or summary.get("promotion_gate") != authorization
    ):
        raise ThreeSeedGateError(
            f"seed {seed} replication promotion evidence differs"
        )
    if (
        checkpoint.get("crop_policy") != identity.get("crop_policy")
        or summary.get("crop_policy") != identity.get("crop_policy")
        or checkpoint.get("crop_policy_identity_sha256")
        != identity.get("crop_policy_identity_sha256")
        or summary.get("crop_policy_identity_sha256")
        != identity.get("crop_policy_identity_sha256")
    ):
        raise ThreeSeedGateError(
            f"seed {seed} replication crop policy binding differs"
        )
    if metadata["sha256"] != summary.get("final_checkpoint_sha256"):
        raise ThreeSeedGateError(
            f"seed {seed} replication final checkpoint SHA-256 differs"
        )
    return identity, source_tree


def _load_replication_completed_run_evidence(
    *,
    seed: int,
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Obtain the replication runner's stronger, CPU-only completion proof.

    This is intentionally not replaced by the aggregate gate's local adapter.
    The replication runner owns validation of the exact 564-key tensor
    structure (names/shapes/dtypes/layouts/finiteness), both 1000-epoch
    histories, and the complete crop-audit histories.  The aggregate gate
    independently reads the same fixed artifacts and binds every decision
    input to the hashes and identities returned here.
    """

    try:
        # ``canonical_gate`` inside the replication module is the same module
        # object as ``pilot_gate``.  The surrounding seed adapter temporarily
        # binds that object to confirmatory paths, so restore the completed
        # pilot bindings while the runner reconstructs its authorization.
        with _canonical_pilot_gate_bindings():
            raw = replication_runner.validate_existing_completed_run(seed)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise ThreeSeedGateError(
            f"seed {seed} strict replication completion validation failed"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ThreeSeedGateError(
            f"seed {seed} strict replication completion evidence is malformed"
        )
    evidence = copy.deepcopy(dict(raw))
    fixed = {
        "schema": replication_runner.TRAINING_SCHEMA
        + "/completed_run_evidence",
        "status": "complete",
        "run_seed": seed,
        "canonical_authorization_sha256": pilot_gate._canonical_sha256(
            authorization
        ),
        "train_epoch_count": EPOCHS,
        "validation_epoch_count": EPOCHS,
        "test_split_accessed": False,
        "public_test_supported": False,
    }
    for name, expected in fixed.items():
        if type(evidence.get(name)) is not type(expected) or evidence.get(
            name
        ) != expected:
            raise ThreeSeedGateError(
                f"seed {seed} strict replication evidence.{name} differs"
            )
    summary = evidence.get("summary")
    checkpoint = evidence.get("checkpoint")
    if (
        not isinstance(summary, Mapping)
        or set(summary) != {"relative_path", "sha256"}
        or summary.get("relative_path") != contract["summary_relative_path"]
        or not isinstance(checkpoint, Mapping)
        or not {"relative_path", "sha256", "state_key_count"}.issubset(
            checkpoint
        )
        or checkpoint.get("relative_path")
        != contract["checkpoint_relative_path"]
        or checkpoint.get("state_key_count") != 564
    ):
        raise ThreeSeedGateError(
            f"seed {seed} strict replication fixed paths/state evidence differs"
        )
    pilot_gate._require_sha256(
        summary.get("sha256"),
        label=f"seed {seed} strict replication summary SHA-256",
    )
    pilot_gate._require_sha256(
        checkpoint.get("sha256"),
        label=f"seed {seed} strict replication checkpoint SHA-256",
    )
    for name in (
        "training_identity_sha256",
        "source_tree_sha256",
        "selected_validation_record_sha256",
    ):
        pilot_gate._require_sha256(
            evidence.get(name),
            label=f"seed {seed} strict replication {name}",
        )
    selected_epoch = evidence.get("selected_epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not 1 <= selected_epoch <= EPOCHS
    ):
        raise ThreeSeedGateError(
            f"seed {seed} strict replication selected epoch differs"
        )
    frontier = evidence.get("retention_frontier_epochs")
    retained = evidence.get("retained_candidates")
    retained_count = evidence.get("retained_candidate_count")
    if (
        not isinstance(frontier, list)
        or not frontier
        or any(
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= EPOCHS
            for epoch in frontier
        )
        or frontier != sorted(set(frontier))
        or not isinstance(retained, list)
        or isinstance(retained_count, bool)
        or not isinstance(retained_count, int)
        or retained_count != len(retained)
        or retained_count != len(frontier)
    ):
        raise ThreeSeedGateError(
            f"seed {seed} strict replication retained-candidate evidence differs"
        )
    selected_candidates = 0
    for expected_epoch, artifact in zip(frontier, retained):
        if (
            not isinstance(artifact, Mapping)
            or set(artifact)
            != {"epoch", "relative_path", "sha256", "selected"}
            or artifact.get("epoch") != expected_epoch
            or artifact.get("relative_path")
            != f"candidates/epoch_{expected_epoch:04d}.pth.tar"
            or type(artifact.get("selected")) is not bool
        ):
            raise ThreeSeedGateError(
                f"seed {seed} strict replication retained candidate differs"
            )
        pilot_gate._require_sha256(
            artifact.get("sha256"),
            label=(
                f"seed {seed} strict replication retained candidate "
                f"epoch {expected_epoch}"
            ),
        )
        if artifact["selected"]:
            selected_candidates += 1
            if expected_epoch != selected_epoch:
                raise ThreeSeedGateError(
                    f"seed {seed} strict replication selected candidate differs"
                )
    if selected_candidates != 1 or selected_epoch not in frontier:
        raise ThreeSeedGateError(
            f"seed {seed} strict replication selected candidate differs"
        )
    return evidence


def _bind_replication_completed_run_evidence(
    *,
    seed: int,
    evidence: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_metadata: Mapping[str, str],
    checkpoint_metadata: Mapping[str, str],
    selected: Mapping[str, Any],
    identity: Mapping[str, Any],
    source_tree: str,
) -> None:
    """Bind the strong runner proof to the aggregate adapter's own reads."""

    summary_evidence = evidence.get("summary")
    checkpoint_evidence = evidence.get("checkpoint")
    if not isinstance(summary_evidence, Mapping) or not isinstance(
        checkpoint_evidence, Mapping
    ):
        raise ThreeSeedGateError(
            f"seed {seed} strict replication artifact evidence is missing"
        )
    bindings = (
        (
            summary_evidence.get("relative_path"),
            summary_metadata.get("relative_path"),
            "summary fixed path",
        ),
        (
            summary_evidence.get("sha256"),
            summary_metadata.get("sha256"),
            "summary content SHA-256",
        ),
        (
            checkpoint_evidence.get("relative_path"),
            checkpoint_metadata.get("relative_path"),
            "checkpoint fixed path",
        ),
        (
            checkpoint_evidence.get("sha256"),
            checkpoint_metadata.get("sha256"),
            "checkpoint content SHA-256",
        ),
        (
            evidence.get("training_identity_sha256"),
            identity.get("identity_sha256"),
            "training identity SHA-256",
        ),
        (
            evidence.get("source_tree_sha256"),
            source_tree,
            "source tree SHA-256",
        ),
        (
            evidence.get("selected_epoch"),
            selected.get("epoch"),
            "selected epoch",
        ),
        (
            evidence.get("selected_validation_record_sha256"),
            pilot_gate._canonical_sha256(selected),
            "selected validation record SHA-256",
        ),
        (
            evidence.get("train_epoch_count"),
            len(summary.get("training_history", [])),
            "training history length",
        ),
        (
            evidence.get("validation_epoch_count"),
            len(summary.get("validation_history", [])),
            "validation history length",
        ),
    )
    for proved, observed, label in bindings:
        if type(proved) is not type(observed) or proved != observed:
            raise ThreeSeedGateError(
                f"seed {seed} strict replication {label} binding differs"
            )
    selection = summary.get("selection")
    candidate_artifacts = summary.get("candidate_artifacts")
    retained = evidence.get("retained_candidates")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(candidate_artifacts, Mapping)
        or not isinstance(retained, list)
        or evidence.get("retention_frontier_epochs")
        != selection.get("retention_frontier_epochs")
        or evidence.get("retained_candidate_count") != len(retained)
        or set(candidate_artifacts)
        != {str(epoch) for epoch in evidence["retention_frontier_epochs"]}
    ):
        raise ThreeSeedGateError(
            f"seed {seed} strict replication candidate frontier binding differs"
        )
    selected_evidence: list[Mapping[str, Any]] = []
    for artifact in retained:
        epoch = artifact["epoch"]
        summary_artifact = candidate_artifacts.get(str(epoch))
        if summary_artifact != {
            "relative_path": artifact["relative_path"],
            "file_sha256": artifact["sha256"],
        }:
            raise ThreeSeedGateError(
                f"seed {seed} strict replication candidate artifact binding differs"
            )
        if artifact["selected"]:
            selected_evidence.append(artifact)
    if (
        len(selected_evidence) != 1
        or selection.get("selected_candidate")
        != {
            "relative_path": selected_evidence[0]["relative_path"],
            "file_sha256": selected_evidence[0]["sha256"],
        }
        or summary.get("selected_candidate_sha256")
        != selected_evidence[0]["sha256"]
    ):
        raise ThreeSeedGateError(
            f"seed {seed} strict replication selected candidate binding differs"
        )


def _validate_replication_variant(
    *, seed: int, paths: Mapping[str, str], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    if seed not in RUN_SEEDS[1:]:
        raise ThreeSeedGateError("replication adapter received the pilot seed")
    contract = replication_runner.formal_artifact_contract(seed)
    if (
        contract["summary_relative_path"] != paths["variant_summary"]
        or contract["checkpoint_relative_path"] != paths["variant_checkpoint"]
        or contract["test_split_accessed"] is not False
        or contract["public_test_supported"] is not False
    ):
        raise ThreeSeedGateError(
            f"seed {seed} replication fixed artifact contract differs"
        )
    try:
        with _canonical_pilot_gate_bindings():
            authorization = (
                replication_runner.validate_canonical_expansion_authorization()
            )
    except (OSError, ValueError) as exc:
        raise ThreeSeedGateError(
            f"seed {seed} replication authorization failed fresh validation"
        ) from exc
    completed_evidence = _load_replication_completed_run_evidence(
        seed=seed,
        contract=contract,
        authorization=authorization,
    )
    variant_summary, variant_summary_meta = pilot_gate._load_json(
        paths["variant_summary"],
        label=f"seed {seed} complete-target replication summary",
    )
    variant_checkpoint, variant_checkpoint_meta = pilot_gate._load_checkpoint(
        paths["variant_checkpoint"],
        label=f"seed {seed} complete-target replication checkpoint",
    )
    variant_selection, variant_selected = pilot_gate._validate_summary_common(
        variant_summary, variant=True
    )
    if (
        variant_summary.get("promotion_gate") != authorization
        or variant_summary.get("run_seed") != seed
    ):
        raise ThreeSeedGateError(
            f"seed {seed} replication summary authorization differs"
        )
    variant_identity, variant_source_tree = _validate_replication_checkpoint(
        variant_checkpoint,
        variant_checkpoint_meta,
        seed=seed,
        summary=variant_summary,
        selection=variant_selection,
        authorization=authorization,
    )
    metrics = variant_selected.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ThreeSeedGateError(
            f"seed {seed} replication selected metrics are missing"
        )
    variant_mechanism = pilot_gate._validate_mechanism(
        metrics.get("complete_target_mechanism_diagnostics"),
        label=f"seed {seed} replication mechanism",
    )
    _bind_replication_completed_run_evidence(
        seed=seed,
        evidence=completed_evidence,
        summary=variant_summary,
        summary_metadata=variant_summary_meta,
        checkpoint_metadata=variant_checkpoint_meta,
        selected=variant_selected,
        identity=variant_identity,
        source_tree=variant_source_tree,
    )
    observed = [
        baseline["summary_meta"],
        baseline["checkpoint_meta"],
        baseline["diagnostic_meta"],
        variant_summary_meta,
        variant_checkpoint_meta,
    ]
    pilot_gate._assert_inputs_unchanged(observed)
    pilot_gate._validate_canonical_split_files()
    if pilot_gate._validate_source_set(
        baseline["identity"]["determinism_protocol"],
        expected_schema="evisirst_validation_selected_source_set/v2",
        expected_paths=pilot_gate._BASELINE_SOURCE_PATHS,
        label=f"seed {seed} baseline final source check",
    ) != baseline["source_tree"]:
        raise ThreeSeedGateError(
            f"seed {seed} baseline source tree changed"
        )
    _revalidate_diagnostic_sources(seed, baseline)
    if _expected_replication_source_set(authorization)[
        "source_tree_sha256"
    ] != variant_source_tree:
        raise ThreeSeedGateError(
            f"seed {seed} replication source tree changed"
        )
    record = _normalized_seed_record(
        seed=seed,
        baseline=baseline,
        variant_summary_meta=variant_summary_meta,
        variant_checkpoint_meta=variant_checkpoint_meta,
        variant_selection=variant_selection,
        variant_selected=variant_selected,
        variant_identity=variant_identity,
        variant_source_tree=variant_source_tree,
        variant_mechanism=variant_mechanism,
    )
    complete_target_inputs = record["verified_inputs"]["complete_target"]
    complete_target_inputs["strict_completed_run_validator"] = (
        "train_irstd_complete_target_replication_v1."
        "validate_existing_completed_run"
    )
    complete_target_inputs["strict_completed_run_evidence"] = copy.deepcopy(
        completed_evidence
    )
    complete_target_inputs["strict_completed_run_evidence_sha256"] = (
        pilot_gate._canonical_sha256(completed_evidence)
    )
    return record


def evaluate_payload() -> dict[str, Any]:
    gate_source = _artifact_metadata(GATE_SOURCE_RELATIVE_PATH)
    frozen_rules = _load_and_validate_frozen_rules()
    records = [_validate_seed_evidence(seed) for seed in RUN_SEEDS]
    if _sha256_file(_fixed_path(GATE_SOURCE_RELATIVE_PATH, must_exist=True)) != (
        gate_source["sha256"]
    ):
        raise ThreeSeedGateError("gate source changed during evaluation")
    if _artifact_metadata(RULES_RELATIVE_PATH) != frozen_rules:
        raise ThreeSeedGateError("frozen rules file changed during evaluation")
    return build_result_payload(
        seed_evidence=records,
        gate_source_artifact=gate_source,
        frozen_rules_artifact=frozen_rules,
    )


def _write_fixed_json_atomic_no_replace(payload: Mapping[str, Any]) -> Path:
    path = _fixed_path(OUTPUT_RELATIVE_PATH, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ThreeSeedGateError("gate output directory is a symlink")
    parent = path.parent.resolve(strict=True)
    runs_root = (PROJECT_ROOT.resolve(strict=True) / "runs").resolve(strict=True)
    try:
        parent.relative_to(runs_root)
    except ValueError as exc:
        raise ThreeSeedGateError("gate output escaped runs/") from exc
    content = json.dumps(
        dict(payload), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(
                    f"three-seed gate result already exists and is immutable: {path}"
                ) from exc
            raise
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def validate_existing_result() -> dict[str, Any]:
    path = _fixed_path(OUTPUT_RELATIVE_PATH, must_exist=True)
    metadata_before = _artifact_metadata(OUTPUT_RELATIVE_PATH)
    content = path.read_bytes()
    if _artifact_metadata(OUTPUT_RELATIVE_PATH) != metadata_before:
        raise ThreeSeedGateError("existing result changed while being read")
    try:
        observed = json.loads(
            content.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ThreeSeedGateError("existing result is not strict JSON") from exc
    expected = evaluate_payload()
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
        raise ThreeSeedGateError("existing result differs from fresh evaluation")
    if _artifact_metadata(OUTPUT_RELATIVE_PATH) != metadata_before:
        raise ThreeSeedGateError("existing result changed during validation")
    return expected


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def run(_args: argparse.Namespace | None = None) -> Path:
    return _write_fixed_json_atomic_no_replace(evaluate_payload())


def main(argv: Sequence[str] | None = None) -> None:
    output = run(parse_args(argv))
    print(output.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "OUTPUT_RELATIVE_PATH",
    "PREREGISTERED_RULES",
    "PREREGISTERED_RULES_SHA256",
    "RULES_RELATIVE_PATH",
    "RESULT_SCHEMA",
    "RUN_SEEDS",
    "SEED_INPUT_PATHS",
    "ThreeSeedGateError",
    "build_result_payload",
    "evaluate_payload",
    "main",
    "parse_args",
    "run",
    "validate_existing_result",
]
