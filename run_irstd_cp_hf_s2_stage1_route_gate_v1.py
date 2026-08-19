#!/usr/bin/env python3
"""Independent Seed-42 route gates for PSBFR and CP-HF-S2.

Both fixed routes are evaluated, never ranked.  A route may authorize only
its own preregistered run seeds 1446202191 and 104728269 after its Seed-42
epoch-500 transaction and mechanism diagnostic pass this gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import run_irstd_cp_hf_s2_first500_ab_gate_v1 as shared
import train_irstd_cp_hf_s2_legacy_screen_v1 as runner


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_SCHEMA = "evisirst_irstd_model_route_stage1_seed42/v1"
CANDIDATE_REGISTRY = ("psbfr_v1", "cp_hf_s2_v1")
SEED = 42
CONTINUATION_SEEDS = (1_446_202_191, 104_728_269)
OUTPUT_ROOT = runner.DEFAULT_OUTPUT_ROOT / "comparison" / "stage1"


class Stage1RouteGateError(ValueError):
    """Available Stage-1 evidence violates the frozen route contract."""


class Stage1RouteGateNotReady(RuntimeError):
    """A fixed route is missing its Seed-42 evidence."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def output_path(candidate: str) -> Path:
    if candidate not in CANDIDATE_REGISTRY:
        raise Stage1RouteGateError("candidate is outside the frozen registry")
    return OUTPUT_ROOT / candidate / "seed42_interim.json"


def _require_reserved_s2_absent() -> None:
    if shared.OUTPUT_PATH.exists() or shared.OUTPUT_PATH.is_symlink():
        raise Stage1RouteGateError(
            "reserved combined S2 path must remain absent during Stage 1"
        )


def _arm_and_history(
    *, base_gate: Any, torch: Any, candidate: str
) -> tuple[dict[str, Any], dict[str, Any], list[Any], list[Any]]:
    try:
        clean = base_gate._read_arm(seed=SEED, variant=None)
        if candidate == "psbfr_v1":
            variant = base_gate._read_arm(seed=SEED, variant=candidate)
        else:
            with shared._cp_reader_contract(base_gate) as patched:
                variant = patched._read_arm(seed=SEED, variant=candidate)
    except base_gate.ModelDesignEpoch500GateNotReady as exc:
        raise Stage1RouteGateNotReady(str(exc)) from exc
    clean_history = shared._history_from_latest(
        torch,
        role="clean",
        seed=SEED,
        expected_latest=clean["latest"],
        expected_identity_sha256=clean["run_identity_sha256"],
    )
    variant_history = shared._history_from_latest(
        torch,
        role=candidate,
        seed=SEED,
        expected_latest=variant["latest"],
        expected_identity_sha256=variant["run_identity_sha256"],
    )
    return clean, variant, clean_history, variant_history


def _route_mechanism(
    *, base_gate: Any, candidate: str, variant_arm: Mapping[str, Any]
) -> tuple[dict[str, bool], Any]:
    try:
        if candidate == "psbfr_v1":
            return shared._psbfr_mechanism_checks(base_gate, variant_arm)
        return shared._cp_seed_mechanism_check(
            base_gate, seed=SEED, arm=variant_arm
        )
    except (FileNotFoundError, shared.CPHFS2First500GateNotReady) as exc:
        raise Stage1RouteGateNotReady(str(exc)) from exc


def route_decision(
    *,
    candidate: str,
    clean: Mapping[str, Any],
    variant: Mapping[str, Any],
    mechanism_checks: Mapping[str, Any],
) -> dict[str, Any]:
    if candidate not in CANDIDATE_REGISTRY:
        raise Stage1RouteGateError("candidate is outside the frozen registry")
    required = (
        ("bound_holds", "corrector_nonzero", "saturation_below_limit")
        if candidate == "psbfr_v1"
        else ("adapter_executed", "adapter_departed_identity", "finite")
    )
    if set(mechanism_checks) != set(required) or any(
        type(mechanism_checks[name]) is not bool for name in required
    ):
        raise Stage1RouteGateError("route mechanism checks differ")
    delta_miou = Decimal(str(variant["selected_mIoU"])) - Decimal(
        str(clean["selected_mIoU"])
    )
    delta_pd = Decimal(str(variant["selected_Pd"])) - Decimal(
        str(clean["selected_Pd"])
    )
    delta_fa = Decimal(str(variant["selected_Fa"])) - Decimal(
        str(clean["selected_Fa"])
    )
    if not all(value.is_finite() for value in (delta_miou, delta_pd, delta_fa)):
        raise Stage1RouteGateError("route metric delta is non-finite")
    safety_failure = delta_pd < Decimal("-0.003") and delta_fa > 0
    metric_pass = delta_miou > 0 and not safety_failure
    mechanism_pass = all(mechanism_checks.values())
    result = "GO" if metric_pass and mechanism_pass else "STOP"
    return {
        "delta_mIoU_decimal": format(delta_miou, "f"),
        "delta_Pd_decimal": format(delta_pd, "f"),
        "delta_Fa_decimal": format(delta_fa, "f"),
        "delta_mIoU_positive": delta_miou > 0,
        "safety_failure": safety_failure,
        "mechanism_checks": dict(mechanism_checks),
        "metric_screen_passed": metric_pass,
        "mechanism_screen_passed": mechanism_pass,
        "result": result,
        "route_continuation_authorized": result == "GO",
        "authorized_remaining_run_seeds": (
            list(CONTINUATION_SEEDS) if result == "GO" else []
        ),
    }


def evaluate_route(candidate: str) -> dict[str, Any]:
    if candidate not in CANDIDATE_REGISTRY:
        raise Stage1RouteGateError("candidate is outside the frozen registry")
    _require_reserved_s2_absent()
    runner.load_rules()
    try:
        torch = importlib.import_module("torch")
        base_gate = importlib.import_module("run_irstd_model_design_epoch500_gate_v1")
    except (ImportError, ModuleNotFoundError) as exc:
        raise Stage1RouteGateNotReady("fixed evidence reader is unavailable") from exc
    clean_arm, variant_arm, clean_history, variant_history = _arm_and_history(
        base_gate=base_gate, torch=torch, candidate=candidate
    )
    clean = shared.select_first500(clean_history)
    variant = shared.select_first500(variant_history)
    if (
        clean["selected_epoch"] != clean_arm["selected_metrics"]["epoch"]
        or variant["selected_epoch"]
        != variant_arm["selected_metrics"]["epoch"]
    ):
        raise Stage1RouteGateError("fresh selection differs from arm evidence")
    checks, mechanism_artifact = _route_mechanism(
        base_gate=base_gate, candidate=candidate, variant_arm=variant_arm
    )
    decision = route_decision(
        candidate=candidate,
        clean=clean,
        variant=variant,
        mechanism_checks=checks,
    )
    source_paths = {
        "stage1_gate": Path(__file__),
        "shared_gate": Path(shared.__file__),
        "runner": Path(runner.__file__),
        "protocol": runner.PROTOCOL_PATH,
        "rules": runner.RULES_PATH,
        "zero_margin_selector": PROJECT_ROOT
        / "experiments/evisirst_zero_margin_selection.py",
        "epoch500_reader": PROJECT_ROOT
        / "run_irstd_model_design_epoch500_gate_v1.py",
    }
    sources = {name: shared._artifact(path) for name, path in source_paths.items()}
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "stage": "S1_seed42_interim_route",
        "candidate": candidate,
        "architecture_seed": runner.ARCHITECTURE_SEED,
        "run_seed": SEED,
        "clean_arm": clean_arm,
        "candidate_arm": variant_arm,
        "clean_selection": clean,
        "candidate_selection": variant,
        "mechanism_artifact": mechanism_artifact,
        "other_candidate_considered": False,
        "candidate_ranking_performed": False,
        "winner_selected": None,
        "source_identity": {
            "files": sources,
            "source_tree_sha256": runner._canonical_sha256(sources),
        },
        "public_test_allowed": False,
    }
    payload.update(decision)
    payload.update(runner._strict_false_disclosures())
    runner.require_test_false(payload, label="stage1_route")
    return json.loads(runner._canonical_bytes(payload).decode("ascii"))


def write_route_payload(candidate: str, payload: Mapping[str, Any]) -> Path:
    fresh = evaluate_route(candidate)
    if runner._canonical_bytes(payload) != runner._canonical_bytes(fresh):
        raise Stage1RouteGateError("route payload differs from fresh fixed evidence")
    if (
        payload.get("schema") != RESULT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("candidate") != candidate
        or any(
            payload.get(name) is not False
            for name in runner._strict_false_disclosures()
        )
        or payload.get("public_test_allowed") is not False
    ):
        raise Stage1RouteGateError("route payload contract is incomplete")
    return runner.write_json_no_clobber(output_path(candidate), payload)


def run(_args: argparse.Namespace | None = None) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for candidate in CANDIDATE_REGISTRY:
        try:
            payload = evaluate_route(candidate)
            outcomes[candidate] = str(write_route_payload(candidate, payload))
        except Stage1RouteGateNotReady as exc:
            outcomes[candidate] = f"WAIT: {exc}"
    return outcomes


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CANDIDATE_REGISTRY",
    "RESULT_SCHEMA",
    "Stage1RouteGateError",
    "Stage1RouteGateNotReady",
    "evaluate_route",
    "output_path",
    "parse_args",
    "run",
    "route_decision",
    "write_route_payload",
]
