#!/usr/bin/env python3
"""Fixed clean/PSBFR/CP-HF first-500 evidence reader and paired gate.

The gate reselects each first-500 history with the frozen zero-margin
selector, emits both candidate rows, applies identical paired rules, and
never ranks or picks a candidate.  Missing evidence stays non-terminal and no
immutable result is written.  Its combined S2 writer is deliberately sealed;
the independent Seed-42 route gate is the only authorization entrypoint in
this amendment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from experiments import evisirst_zero_margin_selection as zero_selection
import train_irstd_cp_hf_s2_legacy_screen_v1 as runner


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_SCHEMA = "evisirst_irstd_cp_hf_s2_first500_ab_gate/v1"
OUTPUT_PATH = (
    runner.DEFAULT_OUTPUT_ROOT
    / "comparison"
    / "first500_clean_psbfr_cp_hf_s2_v1.json"
)
RUN_SEEDS = runner.FORMAL_RUN_SEEDS
CANDIDATE_REGISTRY = ("psbfr_v1", "cp_hf_s2_v1")
SCREEN_EPOCH = runner.SCREEN_EPOCH
FORMAL_EPOCHS = runner.FORMAL_EPOCHS
S2_ROUTE_LEDGER_EXECUTION_SEALED = False
CP_MECHANISM_DIAGNOSTIC_PATH = (
    runner.DEFAULT_OUTPUT_ROOT / "comparison" / "mechanism"
)
CP_MECHANISM_DIAGNOSTIC_SCHEMA = (
    "evisirst_irstd_cp_hf_s2_selected_mechanism_diagnostic/v1"
)


class CPHFS2First500GateError(ValueError):
    """Available evidence violates the preregistered comparison contract."""


class CPHFS2First500GateNotReady(RuntimeError):
    """One or more fixed epoch-500 transactions is unavailable."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CPHFS2First500GateError(f"{label} is not numeric")
    try:
        normalized = Decimal(str(value))
    except Exception as exc:
        raise CPHFS2First500GateError(f"{label} is not numeric") from exc
    if not normalized.is_finite():
        raise CPHFS2First500GateError(f"{label} is non-finite")
    return normalized


def select_first500(records: Any) -> dict[str, Any]:
    if not isinstance(records, list) or len(records) < SCREEN_EPOCH:
        raise CPHFS2First500GateNotReady("validation history has fewer than 500 records")
    prefix = records[:SCREEN_EPOCH]
    if [record.get("epoch") for record in prefix if isinstance(record, Mapping)] != list(
        range(1, SCREEN_EPOCH + 1)
    ):
        raise CPHFS2First500GateError("first-500 epochs are not continuous")
    for record in prefix:
        if (
            not isinstance(record, Mapping)
            or record.get("data_role") != "val"
            or record.get("evaluation_head") != "out"
        ):
            raise CPHFS2First500GateError("validation record identity differs")
        runner.require_test_false(record, label="validation_record")
    provenance = zero_selection.select_checkpoints(
        prefix,
        primary_role=zero_selection.PRIMARY_ROLE,
        margin=None,
    )
    selected = provenance["roles"][zero_selection.PRIMARY_ROLE]["selected"]
    record = prefix[int(selected["epoch"]) - 1]
    decision = {
        "selected_epoch": int(selected["epoch"]),
        "selected_mIoU": float(record["mIoU"]),
        "selected_Pd": float(record["Pd"]),
        "selected_Fa": float(record["Fa"]),
        "selection_provenance": provenance,
        "selection_provenance_sha256": runner._canonical_sha256(provenance),
        "records_used": SCREEN_EPOCH,
        "records_after_epoch_500_used": False,
        "test_split_accessed": False,
    }
    return decision


def _require_seed_histories(value: Any, *, label: str) -> Mapping[int, list[Any]]:
    if not isinstance(value, Mapping) or set(value) != set(RUN_SEEDS):
        raise CPHFS2First500GateNotReady(
            f"{label} must contain exactly the three preregistered seeds"
        )
    normalized: dict[int, list[Any]] = {}
    for seed in RUN_SEEDS:
        history = value.get(seed)
        if not isinstance(history, list):
            raise CPHFS2First500GateNotReady(f"{label} seed {seed} is missing")
        normalized[seed] = history
    return normalized


def _mechanism_valid(value: Any, *, candidate: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CPHFS2First500GateNotReady(
            f"{candidate} mechanism checks are unavailable"
        )
    checks = dict(value)
    required = (
        ("bound_holds", "corrector_nonzero", "saturation_below_limit")
        if candidate == "psbfr_v1"
        else ("adapter_executed", "adapter_departed_identity", "finite")
    )
    if set(checks) != set(required) or any(type(checks[name]) is not bool for name in required):
        raise CPHFS2First500GateError(f"{candidate} mechanism checks differ")
    return checks


def candidate_decision(
    *,
    candidate: str,
    clean_histories: Mapping[int, list[Any]],
    candidate_histories: Mapping[int, list[Any]],
    mechanism_checks: Mapping[str, Any],
) -> dict[str, Any]:
    if candidate not in CANDIDATE_REGISTRY:
        raise CPHFS2First500GateError("candidate is outside the frozen registry")
    clean = _require_seed_histories(clean_histories, label="clean")
    variant = _require_seed_histories(candidate_histories, label=candidate)
    checks = _mechanism_valid(mechanism_checks, candidate=candidate)
    comparison_rules = runner.load_rules()["comparison"]
    mean_minimum = _decimal(
        comparison_rules["mean_delta_mIoU_minimum"], label="mean minimum"
    )
    pairs: list[dict[str, Any]] = []
    delta_mious: list[Decimal] = []
    safety_failures: list[int] = []
    for seed in RUN_SEEDS:
        a = select_first500(clean[seed])
        b = select_first500(variant[seed])
        delta_miou = _decimal(b["selected_mIoU"], label="candidate mIoU") - _decimal(
            a["selected_mIoU"], label="clean mIoU"
        )
        delta_pd = _decimal(b["selected_Pd"], label="candidate Pd") - _decimal(
            a["selected_Pd"], label="clean Pd"
        )
        delta_fa = _decimal(b["selected_Fa"], label="candidate Fa") - _decimal(
            a["selected_Fa"], label="clean Fa"
        )
        safety_failure = delta_pd < Decimal("-0.003") and delta_fa > 0
        if safety_failure:
            safety_failures.append(seed)
        delta_mious.append(delta_miou)
        pairs.append(
            {
                "architecture_seed": runner.ARCHITECTURE_SEED,
                "run_seed": seed,
                "clean": a,
                "candidate": b,
                "delta_mIoU_decimal": format(delta_miou, "f"),
                "delta_Pd_decimal": format(delta_pd, "f"),
                "delta_Fa_decimal": format(delta_fa, "f"),
                "safety_failure": safety_failure,
            }
        )
    mean_delta = sum(delta_mious, Decimal("0")) / Decimal(len(delta_mious))
    metric_pass = (
        all(delta > 0 for delta in delta_mious)
        and mean_delta >= mean_minimum
        and not safety_failures
    )
    mechanism_pass = all(checks.values())
    decision = {
        "candidate": candidate,
        "paired_comparisons": pairs,
        "all_three_delta_mIoU_positive": all(delta > 0 for delta in delta_mious),
        "mean_delta_mIoU_decimal": format(mean_delta, "f"),
        "mean_delta_mIoU_minimum_decimal": format(mean_minimum, "f"),
        "safety_failure_run_seeds": safety_failures,
        "mechanism_checks": checks,
        "metric_screen_passed": metric_pass,
        "mechanism_screen_passed": mechanism_pass,
        "result": "GO" if metric_pass and mechanism_pass else "STOP",
    }
    decision.update(runner._strict_false_disclosures())
    return decision


def evaluate_payload(
    *,
    clean_histories: Mapping[int, list[Any]],
    candidate_histories: Mapping[str, Mapping[int, list[Any]]],
    mechanism_checks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    observed_candidates = tuple(candidate_histories)
    if set(observed_candidates) != set(CANDIDATE_REGISTRY):
        expected = set(CANDIDATE_REGISTRY)
        observed = set(observed_candidates)
        status = (
            "INCOMPLETE"
            if observed < expected
            else "INVALID_MULTIPLE_CANDIDATES"
        )
        incomplete = {
            "schema": RESULT_SCHEMA,
            "status": status,
            "expected_candidate_registry": list(CANDIDATE_REGISTRY),
            "observed_candidate_registry": list(observed_candidates),
            "candidate_ranking_performed": False,
            "immutable_result_written": False,
            "public_test_allowed": False,
        }
        incomplete.update(runner._strict_false_disclosures())
        return incomplete
    if not isinstance(mechanism_checks, Mapping) or set(mechanism_checks) != set(
        CANDIDATE_REGISTRY
    ):
        raise CPHFS2First500GateNotReady("both mechanism-check ledgers are required")
    decisions = {
        candidate: candidate_decision(
            candidate=candidate,
            clean_histories=clean_histories,
            candidate_histories=candidate_histories[candidate],
            mechanism_checks=mechanism_checks[candidate],
        )
        for candidate in CANDIDATE_REGISTRY
    }
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "screen_stage": "legacy_dev_validation_first500",
        "configured_total_epochs": FORMAL_EPOCHS,
        "operational_pause_after_atomic_epoch": SCREEN_EPOCH,
        "run_seeds": list(RUN_SEEDS),
        "candidate_registry": list(CANDIDATE_REGISTRY),
        "candidate_decisions_are_independent": True,
        "candidate_ranking_performed": False,
        "winner_selected": None,
        "selection_rule": zero_selection.RULE_VERSION,
        "selection_margin_raw": None,
        "selection_window_applied": False,
        "decisions": decisions,
        "stability_claim_allowed": False,
        "five_pair_confirmation_implemented": False,
        "records_after_epoch_500_used": False,
        "test_split_accessed": False,
        "public_test_allowed": False,
    }
    payload.update(runner._strict_false_disclosures())
    runner.require_test_false(payload, label="gate")
    return json.loads(runner._canonical_bytes(payload).decode("ascii"))


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CPHFS2First500GateNotReady(f"missing fixed artifact: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise CPHFS2First500GateError("fixed artifact escapes repository") from exc
    return {
        "relative_path": relative,
        "sha256": runner._sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _run_dir(role: str, seed: int) -> Path:
    if role == "clean":
        return (
            PROJECT_ROOT
            / "runs"
            / "validation_selected"
            / "formal"
            / runner.DATASET
            / runner.TARGET_MODE
            / f"run_seed_{seed}"
        )
    return (
        PROJECT_ROOT
        / "runs"
        / "irstd_model_design"
        / role
        / "legacy_screen"
        / "formal"
        / runner.DATASET
        / runner.TARGET_MODE
        / f"run_seed_{seed}"
    )


def _validate_identity_source_tree(identity: Mapping[str, Any], *, role: str) -> None:
    determinism = identity.get("determinism_protocol")
    files = determinism.get("source_files") if isinstance(determinism, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise CPHFS2First500GateError(f"{role} source manifest is missing")
    normalized: dict[str, Any] = {}
    for name, artifact in files.items():
        if not isinstance(name, str) or not isinstance(artifact, Mapping):
            raise CPHFS2First500GateError(f"{role} source entry is malformed")
        relative = artifact.get("relative_path")
        sha256 = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(sha256, str):
            raise CPHFS2First500GateError(f"{role} source entry is incomplete")
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise CPHFS2First500GateError(f"{role} source file is unavailable")
        try:
            path.resolve(strict=True).relative_to(PROJECT_ROOT.resolve(strict=True))
        except ValueError as exc:
            raise CPHFS2First500GateError(f"{role} source file escapes repository") from exc
        observed = runner._sha256_file(path)
        if observed != sha256:
            raise CPHFS2First500GateError(f"{role} source hash differs: {name}")
        normalized[name] = {"relative_path": relative, "sha256": observed}
    if (
        normalized != dict(files)
        or determinism.get("source_tree_sha256")
        != runner._canonical_sha256(normalized)
    ):
        raise CPHFS2First500GateError(f"{role} source tree differs")


def _history_from_latest(
    torch: Any,
    *,
    role: str,
    seed: int,
    expected_latest: Mapping[str, Any],
    expected_identity_sha256: str,
) -> list[Any]:
    path = _run_dir(role, seed) / "last_training_state.pth.tar"
    before = _artifact(path)
    if before != expected_latest:
        raise CPHFS2First500GateError(
            f"{role} seed {seed} latest changed after arm validation"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if _artifact(path) != before:
        raise CPHFS2First500GateError(
            f"{role} seed {seed} latest changed during snapshot read"
        )
    if not isinstance(payload, Mapping):
        raise CPHFS2First500GateError("latest transaction is malformed")
    runner.require_test_false(payload, label=f"{role}.{seed}.latest")
    identity = payload.get("run_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("identity_sha256") != expected_identity_sha256
    ):
        raise CPHFS2First500GateError("latest training identity differs from arm")
    _validate_identity_source_tree(identity, role=role)
    if role == runner.ARCHITECTURE_VARIANT:
        unhashed = dict(identity)
        digest = unhashed.pop("identity_sha256", None)
        if digest != runner._canonical_sha256(unhashed):
            raise CPHFS2First500GateError("CP-HF training identity SHA differs")
        if any(
            identity.get(name) is not False
            for name in runner._strict_false_disclosures()
        ):
            raise CPHFS2First500GateError(
                "CP-HF training disclosure fields are incomplete"
            )
        runner._validate_current_source_tree(identity)
    history = payload.get("validation_history")
    if not isinstance(history, list):
        raise CPHFS2First500GateError("latest validation history is missing")
    return history


@contextmanager
def _cp_reader_contract(base_gate: Any) -> Iterator[Any]:
    # The runner context patches the shared screen module to exactly the same
    # schema/spec used when writing CP-HF transactions.  The existing audited
    # epoch-500 reader can then validate CP-HF without modifying that reader.
    with runner._screen_transaction_adapter(
        SimpleNamespace(smoke_max_train_samples=None)
    ):
        additions = {
            runner.ARCHITECTURE_VARIANT: (
                PROJECT_ROOT
                / "runs"
                / "irstd_model_design"
                / runner.ARCHITECTURE_VARIANT
                / "legacy_screen"
                / "formal"
                / runner.DATASET
                / runner.TARGET_MODE
            )
        }
        previous_roots = base_gate.VARIANT_ROOTS
        previous_protocol = base_gate.PROTOCOL_RELATIVE_PATH
        previous_rules = base_gate.RULES_RELATIVE_PATH
        try:
            base_gate.VARIANT_ROOTS = dict(previous_roots) | additions
            base_gate.PROTOCOL_RELATIVE_PATH = PROTOCOL_RELATIVE_PATH = (
                runner.PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix()
            )
            base_gate.RULES_RELATIVE_PATH = RULES_RELATIVE_PATH = (
                runner.RULES_PATH.relative_to(PROJECT_ROOT).as_posix()
            )
            # Local names make accidental refactors visibly fail lint/tests.
            if not PROTOCOL_RELATIVE_PATH or not RULES_RELATIVE_PATH:
                raise AssertionError("CP-HF fixed source paths are empty")
            yield base_gate
        finally:
            base_gate.VARIANT_ROOTS = previous_roots
            base_gate.PROTOCOL_RELATIVE_PATH = previous_protocol
            base_gate.RULES_RELATIVE_PATH = previous_rules


def _cp_diagnostic_path(seed: int) -> Path:
    if type(seed) is not int or seed not in RUN_SEEDS:
        raise CPHFS2First500GateError("CP-HF diagnostic seed differs")
    return CP_MECHANISM_DIAGNOSTIC_PATH / f"run_seed_{seed}" / "result.json"


def _expected_cp_diagnostic_sources() -> dict[str, Any]:
    paths = {
        "diagnostic": PROJECT_ROOT / "run_irstd_cp_hf_s2_mechanism_diagnostic_v1.py",
        "comparison_gate": Path(__file__),
        "epoch500_reader": PROJECT_ROOT / "run_irstd_model_design_epoch500_gate_v1.py",
        "runner": Path(runner.__file__),
        "protocol": runner.PROTOCOL_PATH,
        "rules": runner.RULES_PATH,
        "architecture": runner.ARCHITECTURE_PATH,
        "architecture_tests": runner.ARCHITECTURE_TEST_PATH,
        "validation_dataset": PROJECT_ROOT / "experiments/evisirst_v2_data.py",
        "zero_margin_selector": Path(zero_selection.__file__),
    }
    return {name: _artifact(path) for name, path in paths.items()}


def _cp_seed_mechanism_check(
    base_gate: Any, *, seed: int, arm: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    diagnostic, artifact = base_gate._load_json(_cp_diagnostic_path(seed))
    result = diagnostic.get("result")
    required_false = {
        "selection_allowed",
        "threshold_tuning_allowed",
        "lockbox_accessed",
        "test_selected",
        "test_selection_supported",
        "public_test_supported",
        "public_test_allowed",
        "test_split_accessed",
    }
    if (
        diagnostic.get("schema") != CP_MECHANISM_DIAGNOSTIC_SCHEMA
        or diagnostic.get("status") != "complete"
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("device") != "cuda:0"
        or diagnostic.get("validation_passes_per_selected_checkpoint") != 1
        or diagnostic.get("run_seed") != seed
        or any(diagnostic.get(name) is not False for name in required_false)
        or not isinstance(result, Mapping)
    ):
        raise CPHFS2First500GateError("CP-HF mechanism diagnostic differs")
    runner.require_test_false(diagnostic, label=f"cp_hf_diagnostic.{seed}")
    sources = diagnostic.get("source_identity")
    expected_sources = _expected_cp_diagnostic_sources()
    if (
        not isinstance(sources, Mapping)
        or sources.get("files") != expected_sources
        or sources.get("source_tree_sha256")
        != runner._canonical_sha256(expected_sources)
    ):
        raise CPHFS2First500GateError("CP-HF diagnostic source identity differs")
    selected = arm.get("selected_candidate")
    metadata = selected.get("artifact") if isinstance(selected, Mapping) else None
    summary = result.get("summary") if isinstance(result, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or result.get("run_seed") != seed
        or result.get("architecture_seed") != runner.ARCHITECTURE_SEED
        or result.get("selected_candidate") != metadata
        or result.get("selected_epoch") != arm["selected_metrics"]["epoch"]
        or result.get("one_full_validation_pass") is not True
        or not isinstance(summary, Mapping)
    ):
        raise CPHFS2First500GateError(
            f"CP-HF diagnostic binding differs for seed {seed}"
        )
    executed = (
        summary.get("validation_sample_count") == 160
        and summary.get("adapter_hook_call_count") == 160
    )
    departed = summary.get("nonzero_correction_count", 0) > 0
    finite = summary.get("correction_finite") is True
    bound = (
        summary.get("absolute_correction_bound") == 0.25
        and summary.get("bound_violation_count") == 0
    )
    return (
        {
            "adapter_executed": executed,
            "adapter_departed_identity": departed,
            "finite": finite and bound,
        },
        artifact,
    )


def _cp_mechanism_checks(
    base_gate: Any, cp_arms: Mapping[int, Mapping[str, Any]]
) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: list[dict[str, bool]] = []
    artifacts: dict[str, Any] = {}
    for seed in RUN_SEEDS:
        check, artifact = _cp_seed_mechanism_check(
            base_gate, seed=seed, arm=cp_arms[seed]
        )
        checks.append(check)
        artifacts[str(seed)] = artifact
    return (
        {
            name: all(check[name] for check in checks)
            for name in ("adapter_executed", "adapter_departed_identity", "finite")
        },
        artifacts,
    )


def _psbfr_mechanism_checks(
    base_gate: Any, seed42_arm: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    path = PROJECT_ROOT / base_gate.DIAGNOSTIC_RELATIVE_PATH
    diagnostic, artifact = base_gate._load_json(path)
    summary = diagnostic.get("summary")
    if (
        diagnostic.get("schema") != base_gate.DIAGNOSTIC_SCHEMA
        or diagnostic.get("status") != "complete"
        or diagnostic.get("run_seed") != 42
        or diagnostic.get("selected_epoch")
        != seed42_arm.get("selected_metrics", {}).get("epoch")
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("threshold_tuning_allowed") is not False
        or diagnostic.get("test_split_accessed") is not False
        or diagnostic.get("selection_allowed") is not False
        or diagnostic.get("public_test_allowed") is not False
        or diagnostic.get("lockbox_accessed") is not False
        or not isinstance(summary, Mapping)
        or summary.get("validation_sample_count") != 160
    ):
        raise CPHFS2First500GateError("PSBFR mechanism diagnostic differs")
    runner.require_test_false(diagnostic, label="psbfr_diagnostic.seed42")
    selected = seed42_arm.get("selected_candidate")
    selected_artifact = (
        selected.get("artifact") if isinstance(selected, Mapping) else None
    )
    if diagnostic.get("selected_candidate") != selected_artifact:
        raise CPHFS2First500GateError(
            "PSBFR diagnostic is not bound to the selected Seed-42 candidate"
        )
    expected_psbfr_sources = {
        "diagnostic": _artifact(
            PROJECT_ROOT / "run_irstd_psbfr_correction_diagnostic_v1.py"
        ),
        "architecture": _artifact(PROJECT_ROOT / "experiments/irstd_psbfr_v1.py"),
        "protocol": _artifact(PROJECT_ROOT / base_gate.PROTOCOL_RELATIVE_PATH),
        "rules": _artifact(PROJECT_ROOT / base_gate.RULES_RELATIVE_PATH),
        "selector": _artifact(PROJECT_ROOT / base_gate.SELECTOR_RELATIVE_PATH),
    }
    if diagnostic.get("source_identity") != expected_psbfr_sources:
        raise CPHFS2First500GateError("PSBFR diagnostic source identity differs")
    psbfr_rules, _ = base_gate._load_json(
        PROJECT_ROOT / "experiments/irstd_psbfr_v1_screen_rules.json"
    )
    saturation_limit = psbfr_rules.get("legacy_screen", {}).get(
        "tanh_absolute_saturation_fraction_maximum"
    )
    if saturation_limit != 0.1:
        raise CPHFS2First500GateError("PSBFR frozen saturation limit differs")
    if summary.get("tanh_saturation_threshold") != psbfr_rules.get(
        "legacy_screen", {}
    ).get("tanh_absolute_saturation_threshold"):
        raise CPHFS2First500GateError("PSBFR saturation threshold differs")
    return (
        {
            "bound_holds": summary.get("bound_violation_count") == 0,
            "corrector_nonzero": summary.get("nonzero_correction_count", 0) > 0,
            "saturation_below_limit": (
                _decimal(
                    summary.get("tanh_saturation_fraction"),
                    label="PSBFR saturation",
                )
                < _decimal(saturation_limit, label="PSBFR saturation limit")
            ),
        },
        artifact,
    )


def _source_identity() -> dict[str, Any]:
    paths = {
        "gate": Path(__file__),
        "cp_hf_runner": Path(runner.__file__),
        "cp_hf_protocol": runner.PROTOCOL_PATH,
        "cp_hf_rules": runner.RULES_PATH,
        "cp_hf_architecture": runner.ARCHITECTURE_PATH,
        "cp_hf_architecture_tests": runner.ARCHITECTURE_TEST_PATH,
        "cp_hf_mechanism_diagnostic": PROJECT_ROOT
        / "run_irstd_cp_hf_s2_mechanism_diagnostic_v1.py",
        "zero_margin_selector": Path(zero_selection.__file__),
        "shared_epoch500_reader": PROJECT_ROOT
        / "run_irstd_model_design_epoch500_gate_v1.py",
        "psbfr_protocol": PROJECT_ROOT / "experiments/IRSTD_PSBFR_V1_PROTOCOL.md",
        "psbfr_rules": PROJECT_ROOT / "experiments/irstd_psbfr_v1_screen_rules.json",
        "psbfr_architecture": PROJECT_ROOT / "experiments/irstd_psbfr_v1.py",
        "psbfr_runner": PROJECT_ROOT / "train_irstd_model_design_screen_v1.py",
        "psbfr_mechanism_diagnostic": PROJECT_ROOT
        / "run_irstd_psbfr_correction_diagnostic_v1.py",
    }
    paths.update(
        {
            f"split/{name}": runner.CANONICAL_SPLIT_ROOT / runner.DATASET / name
            for name in ("manifest.json", "train.txt", "val.txt")
        }
    )
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    return {
        "schema": "evisirst_irstd_cp_hf_s2_first500_gate_sources/v1",
        "files": artifacts,
        "source_tree_sha256": runner._canonical_sha256(artifacts),
    }


def evaluate_fixed_payload() -> dict[str, Any]:
    if not runner.EXECUTION_SEALED:
        raise CPHFS2First500GateNotReady("CP-HF transaction adapter is not sealed")
    try:
        torch = importlib.import_module("torch")
        base_gate = importlib.import_module("run_irstd_model_design_epoch500_gate_v1")
    except (ImportError, ModuleNotFoundError) as exc:
        raise CPHFS2First500GateNotReady("PyTorch/evidence reader unavailable") from exc

    arm_evidence: dict[str, dict[int, Any]] = {
        "clean": {},
        "psbfr_v1": {},
        "cp_hf_s2_v1": {},
    }
    histories: dict[str, dict[int, list[Any]]] = {
        "clean": {},
        "psbfr_v1": {},
        "cp_hf_s2_v1": {},
    }
    for seed in RUN_SEEDS:
        try:
            arm_evidence["clean"][seed] = base_gate._read_arm(
                seed=seed, variant=None
            )
            arm_evidence["psbfr_v1"][seed] = base_gate._read_arm(
                seed=seed, variant="psbfr_v1"
            )
        except base_gate.ModelDesignEpoch500GateNotReady as exc:
            raise CPHFS2First500GateNotReady(str(exc)) from exc
        histories["clean"][seed] = _history_from_latest(
            torch,
            role="clean",
            seed=seed,
            expected_latest=arm_evidence["clean"][seed]["latest"],
            expected_identity_sha256=arm_evidence["clean"][seed][
                "run_identity_sha256"
            ],
        )
        histories["psbfr_v1"][seed] = _history_from_latest(
            torch,
            role="psbfr_v1",
            seed=seed,
            expected_latest=arm_evidence["psbfr_v1"][seed]["latest"],
            expected_identity_sha256=arm_evidence["psbfr_v1"][seed][
                "run_identity_sha256"
            ],
        )

    with _cp_reader_contract(base_gate) as patched_gate:
        for seed in RUN_SEEDS:
            try:
                arm_evidence["cp_hf_s2_v1"][seed] = patched_gate._read_arm(
                    seed=seed, variant="cp_hf_s2_v1"
                )
            except patched_gate.ModelDesignEpoch500GateNotReady as exc:
                raise CPHFS2First500GateNotReady(str(exc)) from exc
            histories["cp_hf_s2_v1"][seed] = _history_from_latest(
                torch,
                role="cp_hf_s2_v1",
                seed=seed,
                expected_latest=arm_evidence["cp_hf_s2_v1"][seed]["latest"],
                expected_identity_sha256=arm_evidence["cp_hf_s2_v1"][seed][
                    "run_identity_sha256"
                ],
            )

    psbfr_checks, psbfr_diagnostic_artifact = _psbfr_mechanism_checks(
        base_gate, arm_evidence["psbfr_v1"][42]
    )
    cp_checks, cp_diagnostic_artifact = _cp_mechanism_checks(
        base_gate, arm_evidence["cp_hf_s2_v1"]
    )
    payload = evaluate_payload(
        clean_histories=histories["clean"],
        candidate_histories={
            "psbfr_v1": histories["psbfr_v1"],
            "cp_hf_s2_v1": histories["cp_hf_s2_v1"],
        },
        mechanism_checks={
            "psbfr_v1": psbfr_checks,
            "cp_hf_s2_v1": cp_checks,
        },
    )
    payload["arm_evidence"] = arm_evidence
    payload["psbfr_diagnostic_artifact"] = psbfr_diagnostic_artifact
    payload["cp_hf_s2_diagnostic_artifact"] = cp_diagnostic_artifact
    payload["source_identity"] = _source_identity()
    payload["split_provenance"] = runner.split_provenance()
    runner.require_test_false(payload, label="fixed_gate")
    return json.loads(runner._canonical_bytes(payload).decode("ascii"))


def run(_args: argparse.Namespace | None = None) -> Path:
    raise CPHFS2First500GateNotReady(
        "combined S2 authorization is sealed until route-local S1 ledgers are terminal"
    )


def write_complete_payload(payload: Mapping[str, Any]) -> Path:
    del payload
    raise CPHFS2First500GateNotReady(
        "combined S2 writer is sealed; route-local terminal ledgers are not implemented"
    )


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "CANDIDATE_REGISTRY",
    "CPHFS2First500GateError",
    "CPHFS2First500GateNotReady",
    "OUTPUT_PATH",
    "RESULT_SCHEMA",
    "RUN_SEEDS",
    "S2_ROUTE_LEDGER_EXECUTION_SEALED",
    "candidate_decision",
    "evaluate_payload",
    "parse_args",
    "run",
    "select_first500",
    "write_complete_payload",
]
