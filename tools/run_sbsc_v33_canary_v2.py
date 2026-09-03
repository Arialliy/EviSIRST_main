#!/usr/bin/env python3
"""Run the immutable SCTransNet C3-SBSC V3.3 train-only canary V2.

This is a mechanism-health transaction, not a benchmark run.  Its only data
role is the frozen IRSTD-1K training split.  It has no evaluation, selection,
resume, or model-serialization path, and its trained state is deliberately
discarded when the process exits.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_ENTRY_SOURCE = Path(__file__)
if _ENTRY_SOURCE.is_symlink() or not _ENTRY_SOURCE.is_file():
    raise RuntimeError("canary entry point must be a regular non-symlink file")
PROJECT_ROOT = _ENTRY_SOURCE.resolve(strict=True).parents[1]
if _ENTRY_SOURCE.resolve(strict=True) != PROJECT_ROOT / "tools" / _ENTRY_SOURCE.name:
    raise RuntimeError("canary entry point resolves away from the project tool path")
_PROJECT_ROOT_TEXT = str(PROJECT_ROOT)
sys.path[:] = [
    _PROJECT_ROOT_TEXT,
    *[
        entry
        for entry in sys.path
        if entry not in ("", _PROJECT_ROOT_TEXT)
    ],
]

FROZEN_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", FROZEN_CUBLAS_WORKSPACE_CONFIG)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments import sbsc_v33_contracts as contracts
from experiments import sctransnet_sbsc_v33 as core
from experiments import three_dataset_v2_protocol as source_protocol
from experiments.evisirst_data import EviSIRSTTrainDataset, stable_uint63
from experiments.four_dataset_models_seed42_v1 import state_dict_sha256
from train import configure_determinism, learning_rate_for_epoch, require_device


CANARY_RULES_PATH = PROJECT_ROOT / "experiments" / "sbsc_v33_canary_rules_v2.json"
EXPECTED_RULES_SHA256 = (
    "b2fc34a711cd8a6fc8ce0e69b4015b8a300c040acaef37d1922354627ae0f2ba"
)
EXPECTED_METHOD_CONFIG_SHA256 = (
    "73e86c18ba5324240513916d30bad7e50bd448b7686f175dde79d4694665f7ab"
)
RULES_SCHEMA = "sctransnet_sbsc_v33/canary_rules/v2"
METHOD_CONFIG_SCHEMA = "sctransnet_sbsc_v33/canary_method_config/v2"
MANIFEST_SCHEMA = "sctransnet_sbsc_v33/canary_manifest/v2"
REPORT_SCHEMA = "sctransnet_sbsc_v33/canary_report/v2"

DATASET = "IRSTD-1K"
METHOD = "sbsc_v33_third"
ARCHITECTURE_SEED = 42
RUN_SEED = 42
EPOCHS = 20
SAMPLE_COUNT = 800
BATCH_SIZE = 16
WORKERS = 0
PATCH_SIZE = 256
STEPS_PER_EPOCH = 50
TOTAL_SAMPLES = 16_000
TOTAL_STEPS = 1_000
GRADIENT_MODE = "live"
BALANCE_MODE = "one_third_two_thirds"
ROUTER_LOSS_WEIGHT = 1.0
OUTPUT_NAMESPACE = "runs/sbsc_v33_third/canary_v2/IRSTD-1K"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / OUTPUT_NAMESPACE
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.json"
ROLE_NAMES = ("C", "H", "B")
MODE_NAMES = core.MODE_NAMES
HARD_GATE_ORDER = (
    "finite",
    "sample_step_counts",
    "role_probability",
    "availability",
    "certificate",
    "combined_backward",
    "gain_bounds",
)
COMPACT_COUNTER_FIELDS = (
    "projection_rows",
    "nonidentity_rows",
    "solver_attempt_rows",
    "solver_accepted_rows",
    "solver_fallback_rows",
    "emission_checked_rows",
    "emission_accepted_rows",
    "emission_fallback_rows",
    "mode_identity_rows",
    "mode_dual_rows",
    "mode_hard_only_rows",
    "mode_background_only_rows",
)


class CanaryError(RuntimeError):
    """A fail-closed V3.3 canary contract violation."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def _fail(category: str, message: str) -> None:
    raise CanaryError(category, message)


def _source_paths() -> tuple[str, ...]:
    fixed = (
        "tools/run_sbsc_v33_canary_v2.py",
        "experiments/sbsc_v33_canary_rules_v2.json",
        "experiments/sbsc_v33_contracts.py",
        "experiments/sctransnet_sbsc_v33.py",
        "experiments/sctransnet_sbsc_v32.py",
        "experiments/sctransnet_sbsc_v31.py",
        "experiments/four_dataset_models_seed42_v1.py",
        "experiments/evisirst_data.py",
        "experiments/three_dataset_v2_protocol.py",
        "experiments/__init__.py",
        "train.py",
    )
    model_sources = tuple(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted((PROJECT_ROOT / "model").rglob("*.py"))
        if path.is_file() and not path.is_symlink()
    )
    paths = tuple(sorted({*fixed, *model_sources}))
    if len(paths) != len(fixed) + len(model_sources):
        _fail("execution_contract", "training source paths are not unique")
    return paths


def build_source_manifest() -> dict[str, Any]:
    return contracts.build_source_manifest(_source_paths())


def load_frozen_rules() -> tuple[dict[str, Any], str]:
    if CANARY_RULES_PATH.is_symlink() or not CANARY_RULES_PATH.is_file():
        _fail("execution_contract", "canary rules are not a regular file")
    observed_sha = contracts.sha256_file(CANARY_RULES_PATH)
    if observed_sha != EXPECTED_RULES_SHA256:
        _fail("execution_contract", "canary rules physical SHA-256 drifted")
    rules = contracts.load_strict_json(CANARY_RULES_PATH)
    if (
        rules.get("schema") != RULES_SCHEMA
        or rules.get("status") != "frozen"
        or rules.get("write_once") is not True
        or tuple(rules.get("hard_gate_order", ())) != HARD_GATE_ORDER
        or rules.get("warning_diagnostics", {}).get("affects_verdict") is not False
    ):
        _fail("execution_contract", "canary rule identity differs")
    v1_evidence = rules.get("superseded_v1_write_once_evidence")
    if not isinstance(v1_evidence, Mapping):
        _fail("execution_contract", "superseded V1 evidence binding is absent")
    expected_v1_evidence = {
        "runner_path": "tools/run_sbsc_v33_canary.py",
        "runner_sha256": "c67478f09ade05dcf35e3dd10ab65fd6318f0d8cca7a6abe2cd003a39edd9f3f",
        "rules_path": "experiments/sbsc_v33_canary_rules.json",
        "rules_sha256": "0f5df664a03e99087ce426f700a32081495a439ae6f2cc41b23bbda3f829815f",
        "manifest_path": "runs/sbsc_v33_third/canary/IRSTD-1K/manifest.json",
        "manifest_sha256": "f18d20a948c0770a3472a67a28644676ca6e74690ebafcfcc5bdfbdb2356dc76",
        "report_path": "runs/sbsc_v33_third/canary/IRSTD-1K/report.json",
        "report_sha256": "555d29b54b23777aac48f8fd14ad48dac6a27459a76578a9bf8bb687923f2986",
        "observed_failure_batch": 10,
        "observed_nonidentity_rows": 0,
        "observed_emission_accepted_rows": 0,
        "preceding_gain_gradient_finite_batches": 9,
        "v2_change_scope": "condition_only_gain_gradient_presence_on_emission_acceptance",
    }
    if dict(v1_evidence) != expected_v1_evidence:
        _fail("execution_contract", "superseded V1 evidence identity differs")
    for kind in ("runner", "rules", "manifest", "report"):
        path = contracts.require_repository_relative_regular_file(
            v1_evidence[f"{kind}_path"]
        )
        if contracts.sha256_file(path) != v1_evidence[f"{kind}_sha256"]:
            _fail("execution_contract", f"superseded V1 {kind} bytes changed")
    method_config = rules.get("method_config")
    if not isinstance(method_config, Mapping):
        _fail("execution_contract", "embedded method config is absent")
    method_config_sha = contracts.canonical_sha256(method_config)
    if method_config_sha != EXPECTED_METHOD_CONFIG_SHA256:
        _fail("execution_contract", "embedded method config SHA-256 drifted")
    exact = {
        "schema": METHOD_CONFIG_SCHEMA,
        "status": "frozen",
        "run_kind": "canary",
        "method": METHOD,
        "dataset": DATASET,
        "data_role": "train",
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "epochs": EPOCHS,
        "train_sample_count": SAMPLE_COUNT,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "router_value_gradient_mode": GRADIENT_MODE,
        "role_loss_balance_mode": BALANCE_MODE,
        "router_level_count": 4,
        "router_level_reduction": "mean",
        "router_loss_weight": ROUTER_LOSS_WEIGHT,
        "output_namespace": OUTPUT_NAMESPACE,
        "validation_dataset_constructed": False,
        "validation_loader_constructed": False,
        "test_dataset_constructed": False,
        "test_loader_constructed": False,
        "checkpoint_written": False,
        "formal_weights_reusable": False,
    }
    for key, expected in exact.items():
        if method_config.get(key) != expected:
            _fail("execution_contract", f"method config field {key!r} differs")
    expected_execution = rules.get("expected_execution")
    if expected_execution != {
        "epochs": EPOCHS,
        "samples_per_epoch": SAMPLE_COUNT,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "total_processed_samples": TOTAL_SAMPLES,
        "total_optimizer_steps": TOTAL_STEPS,
        "combined_backward_calls_per_step": 1,
    }:
        _fail("execution_contract", "canary execution count contract differs")
    if method_config.get("gain_gradient_contract") != {
        "conditioning_counter": "emission_accepted_rows",
        "when_positive": "present_and_finite",
        "when_zero": "none_or_finite_all_zero",
        "minimum_gain_active_batches_over_complete_run": 1,
        "nonfinite_gain_gradient_batches": 0,
    }:
        _fail("execution_contract", "conditional gain-gradient contract differs")
    if rules.get("report_contract") != {
        "success_status": "PASS",
        "failure_status": "FAIL",
        "success_hard_gate_pass": True,
        "failure_hard_gate_pass": False,
        "hard_gate_success_verdict": "GO",
        "hard_gate_failure_verdict": "NO-GO",
        "partial_progress_fields": [
            "current_epoch",
            "started_batches",
            "completed_batches",
            "batch_count",
            "combined_backward_calls",
            "processed_samples",
            "optimizer_steps",
        ],
        "partial_progress_semantics": {
            "current_epoch": "one_based_epoch_entered_or_zero_before_training",
            "started_batches": "batches_entering_the_training_loop",
            "completed_batches": "batches_passing_post_step_state_and_gain_audit",
            "batch_count": "exact_alias_of_completed_batches",
            "combined_backward_calls": "combined_backward_calls_returning_without_exception",
            "processed_samples": "samples_consumed_by_applied_optimizer_steps",
            "optimizer_steps": "optimizer_step_calls_returning_without_exception",
        },
    }:
        _fail("execution_contract", "canary report contract differs")
    return rules, method_config_sha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed V3.3 IRSTD-1K train-only canary V2"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _require_runtime_environment() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != FROZEN_CUBLAS_WORKSPACE_CONFIG:
        _fail("execution_contract", "CUBLAS_WORKSPACE_CONFIG conflicts")
    if PROJECT_ROOT != contracts.PROJECT_ROOT:
        _fail("execution_contract", "contracts module was loaded from another root")


def _canonical_dataset_root() -> Path:
    root = PROJECT_ROOT / "datasets"
    if root.is_symlink() or not root.is_dir():
        _fail("execution_contract", "canonical datasets directory is unavailable")
    if root.resolve(strict=True) != root:
        _fail("execution_contract", "canonical datasets directory resolves elsewhere")
    return root


def _epoch_order(epoch: int) -> list[int]:
    if type(epoch) is not int or not 1 <= epoch <= EPOCHS:
        _fail("sample_step_counts", "epoch is outside the frozen canary range")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_uint63(
            "sctransnet_sbsc_v33_canary_epoch_order",
            DATASET,
            METHOD,
            RUN_SEED,
            epoch,
            SAMPLE_COUNT,
        )
    )
    order = torch.randperm(SAMPLE_COUNT, generator=generator).tolist()
    if sorted(order) != list(range(SAMPLE_COUNT)):
        _fail("sample_step_counts", "epoch order is not a full permutation")
    return order


def _learning_rate(epoch: int, method_config: Mapping[str, Any]) -> float:
    schedule = method_config["learning_rate_schedule"]
    value = learning_rate_for_epoch(
        epoch,
        int(schedule["total_epochs"]),
        float(schedule["base_lr"]),
        float(schedule["min_lr"]),
        int(schedule["warmup_epochs"]),
    )
    if not math.isfinite(value) or value <= 0.0:
        _fail("finite", "learning rate is invalid")
    return value


def _require_finite_tensor(value: Any, *, category: str, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not bool(
        torch.isfinite(value.detach()).all()
    ):
        _fail(category, f"{name} is absent or non-finite")
    return value


def _require_nonnegative_int(value: Any, *, category: str, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail(category, f"{name} must be a non-negative integer")
    return value


def audit_capture_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one captured forward and return compact JSON diagnostics."""

    if not isinstance(record, Mapping):
        _fail("role_probability", "router capture is not a mapping")
    expected_schema = core.SBSC_V33_SCHEMA + "/training_router_capture/v1"
    if record.get("schema") != expected_schema:
        _fail("role_probability", "router capture schema differs")
    batch = record.get("batch_size")
    token_hw = record.get("token_hw")
    if (
        type(batch) is not int
        or batch <= 0
        or not isinstance(token_hw, tuple)
        or len(token_hw) != 2
        or any(type(value) is not int or value <= 0 for value in token_hw)
    ):
        _fail("role_probability", "router capture geometry is malformed")
    token_count = int(token_hw[0] * token_hw[1])
    names = (
        "logits",
        "supports",
        "availability",
        "role_mass",
        "integrated_winning_evidence",
        "intended_mode_codes",
        "effective_mode_codes",
    )
    fields: dict[str, tuple[Any, ...]] = {}
    for name in names:
        value = record.get(name)
        if not isinstance(value, tuple) or len(value) != 4:
            _fail("role_probability", f"capture field {name!r} is not four-level")
        fields[name] = value
    if record.get("router_value_gradient_mode") != GRADIENT_MODE:
        _fail("combined_backward", "capture gradient mode is not live")

    probability_min = 1.0
    probability_max = 0.0
    simplex_max_error = 0.0
    availability_counts = {role: 0 for role in ROLE_NAMES}
    availability_opportunities = {role: 0 for role in ROLE_NAMES}
    intended_counts = {name: 0 for name in MODE_NAMES}
    effective_counts = {name: 0 for name in MODE_NAMES}
    projection_rows = 0
    nonidentity_rows = 0

    for level in range(4):
        logits = _require_finite_tensor(
            fields["logits"][level],
            category="role_probability",
            name=f"level {level} logits",
        )
        probability = _require_finite_tensor(
            fields["supports"][level],
            category="role_probability",
            name=f"level {level} probability",
        )
        expected_shape = (batch, 3, token_hw[0], token_hw[1])
        if tuple(logits.shape) != expected_shape or tuple(probability.shape) != expected_shape:
            _fail("role_probability", f"level {level} role geometry differs")
        detached_probability = probability.detach().float()
        lower = float(detached_probability.amin().item())
        upper = float(detached_probability.amax().item())
        error = float(
            detached_probability.sum(dim=1).sub(1.0).abs().amax().item()
        )
        if lower < 0.0 or upper > 1.0 or error > 2.0e-6:
            _fail("role_probability", f"level {level} role simplex differs")
        if not torch.allclose(
            detached_probability,
            torch.softmax(logits.detach().float(), dim=1),
            rtol=0.0,
            atol=2.0e-6,
        ):
            _fail("role_probability", f"level {level} support differs from logits")
        probability_min = min(probability_min, lower)
        probability_max = max(probability_max, upper)
        simplex_max_error = max(simplex_max_error, error)

        availability = fields["availability"][level]
        mass = _require_finite_tensor(
            fields["role_mass"][level],
            category="availability",
            name=f"level {level} role mass",
        )
        integrated = _require_finite_tensor(
            fields["integrated_winning_evidence"][level],
            category="availability",
            name=f"level {level} integrated evidence",
        )
        if (
            not isinstance(availability, torch.Tensor)
            or availability.dtype is not torch.bool
            or tuple(availability.shape) != (batch, 3)
            or tuple(mass.shape) != (batch, 3)
            or tuple(integrated.shape) != (batch, 3)
            or bool((mass.detach() < 0.0).any())
            or bool((integrated.detach() < 0.0).any())
        ):
            _fail("availability", f"level {level} availability geometry differs")
        expected_availability = mass.detach().ge(1.0) & integrated.detach().ge(
            1.0 / float(token_count)
        )
        if not torch.equal(availability.detach(), expected_availability):
            _fail("availability", f"level {level} availability semantics differ")
        for role_index, role in enumerate(ROLE_NAMES):
            availability_counts[role] += int(
                availability[:, role_index].sum().item()
            )
            availability_opportunities[role] += batch

        intended = fields["intended_mode_codes"][level]
        effective = fields["effective_mode_codes"][level]
        if (
            not isinstance(intended, torch.Tensor)
            or not isinstance(effective, torch.Tensor)
            or intended.dtype is not torch.int64
            or effective.dtype is not torch.int64
            or intended.shape != effective.shape
            or intended.ndim != 4
            or intended.shape[0] != batch
            or intended.shape[-1] != 1
            or bool(((intended < 0) | (intended > 3)).any())
            or bool(((effective < 0) | (effective > 3)).any())
        ):
            _fail("certificate", f"level {level} mode code geometry differs")
        projection_rows += int(intended.numel())
        nonidentity_rows += int(intended.ne(core.MODE_IDENTITY).sum().item())
        for code, name in enumerate(MODE_NAMES):
            intended_counts[name] += int(intended.eq(code).sum().item())
            effective_counts[name] += int(effective.eq(code).sum().item())

    compact = record.get("compact_diagnostics")
    if not isinstance(compact, Mapping) or set(compact) != set(COMPACT_COUNTER_FIELDS):
        _fail("certificate", "compact certificate field set differs")
    counters = {
        name: _require_nonnegative_int(
            compact[name], category="certificate", name=name
        )
        for name in COMPACT_COUNTER_FIELDS
    }
    if counters["projection_rows"] != projection_rows:
        _fail("certificate", "projection row count differs")
    if counters["nonidentity_rows"] != nonidentity_rows:
        _fail("certificate", "non-identity row count differs")
    if (
        counters["solver_accepted_rows"] + counters["solver_fallback_rows"]
        != counters["solver_attempt_rows"]
        or counters["solver_attempt_rows"] > counters["nonidentity_rows"]
        or counters["emission_checked_rows"] != counters["nonidentity_rows"]
        or counters["emission_accepted_rows"] + counters["emission_fallback_rows"]
        != counters["emission_checked_rows"]
    ):
        _fail("certificate", "solver/emission certificate arithmetic differs")
    for name in MODE_NAMES:
        if counters[f"mode_{name}_rows"] != effective_counts[name]:
            _fail("certificate", f"effective mode counter {name!r} differs")
    if sum(effective_counts.values()) != projection_rows:
        _fail("certificate", "effective mode counts do not cover projection rows")
    if (
        sum(effective_counts[name] for name in MODE_NAMES[1:])
        != counters["emission_accepted_rows"]
    ):
        _fail("certificate", "effective non-identity/emission acceptance differs")
    return {
        "batch_size": batch,
        "level_count": 4,
        "probability_min": probability_min,
        "probability_max": probability_max,
        "simplex_max_abs_error": simplex_max_error,
        "availability_counts": availability_counts,
        "availability_opportunities": availability_opportunities,
        "intended_mode_counts": intended_counts,
        "effective_mode_counts": effective_counts,
        "certificate": counters,
    }


def _audit_outputs(outputs: Any, masks: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        _fail("finite", "training forward must return six outputs")
    result = tuple(outputs)
    for index, output in enumerate(result):
        _require_finite_tensor(output, category="finite", name=f"output {index}")
        if tuple(output.shape) != tuple(masks.shape) or bool(
            ((output.detach() < 0.0) | (output.detach() > 1.0)).any()
        ):
            _fail("finite", f"output {index} geometry/range differs")
    return result


def _audit_losses(
    total: torch.Tensor,
    segmentation: torch.Tensor,
    router: torch.Tensor,
    per_level: Sequence[torch.Tensor],
) -> None:
    values = (total, segmentation, router, *per_level)
    for index, value in enumerate(values):
        _require_finite_tensor(value, category="finite", name=f"loss {index}")
        if value.ndim != 0 or float(value.detach().item()) < 0.0:
            _fail("finite", f"loss {index} is not a non-negative scalar")


def _audit_combined_backward(
    model: nn.Module,
    active_names: Sequence[str],
    value_spatial: torch.Tensor,
    *,
    emission_accepted_rows: int,
) -> dict[str, Any]:
    """Audit one backward with gain presence conditioned on routed emission."""

    if type(emission_accepted_rows) is not int or emission_accepted_rows < 0:
        _fail("combined_backward", "emission acceptance count is malformed")
    named = dict(model.named_parameters())
    gain_name = core.SBSC_V33_GAIN_STATE_KEY
    if gain_name not in active_names or gain_name not in named:
        _fail("combined_backward", "conditional gain parameter is absent")
    non_gain_names = tuple(name for name in active_names if name != gain_name)
    missing = []
    nonfinite = []
    nonzero_elements = 0
    for name in non_gain_names:
        gradient = named[name].grad
        if gradient is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(gradient.detach()).all()):
            nonfinite.append(name)
        nonzero_elements += int(torch.count_nonzero(gradient.detach()).item())
    router_names = tuple(core.SBSC_V33_ROUTER_STATE_KEYS)
    router_ok = all(
        name in named
        and named[name].grad is not None
        and bool(torch.isfinite(named[name].grad.detach()).all())
        for name in router_names
    )
    gain_gradient = named[gain_name].grad
    gain_active = emission_accepted_rows > 0
    gain_nonfinite = False
    if gain_active:
        if gain_gradient is None:
            _fail(
                "combined_backward",
                "gain gradient is absent for a gain-active emission batch",
            )
        gain_nonfinite = not bool(torch.isfinite(gain_gradient.detach()).all())
        if gain_nonfinite:
            _fail(
                "combined_backward",
                "gain gradient is non-finite for a gain-active emission batch",
            )
        gain_state = "present_finite_required"
        nonzero_elements += int(torch.count_nonzero(gain_gradient.detach()).item())
    elif gain_gradient is None:
        gain_state = "absent_allowed"
    else:
        gain_nonfinite = not bool(torch.isfinite(gain_gradient.detach()).all())
        if gain_nonfinite or int(torch.count_nonzero(gain_gradient.detach()).item()) != 0:
            _fail(
                "combined_backward",
                "gain-inactive batch requires None or an exactly-zero finite gradient",
            )
        gain_state = "finite_zero_allowed"
    value_gradient = value_spatial.grad
    value_ok = (
        value_gradient is not None
        and bool(torch.isfinite(value_gradient.detach()).all())
    )
    if missing or nonfinite or nonzero_elements <= 0 or not router_ok or not value_ok:
        _fail(
            "combined_backward",
            "combined backward gradient contract differs: "
            f"missing={missing[:3]}, nonfinite={nonfinite[:3]}",
        )
    return {
        "all_non_gain_active_present": True,
        "all_non_gain_active_finite": True,
        "non_gain_active_parameter_count": len(non_gain_names),
        "nonzero_gradient_elements": nonzero_elements,
        "router_present_and_finite": True,
        "live_value_spatial_gradient_present_and_finite": True,
        "gain_conditioning_emission_accepted_rows": emission_accepted_rows,
        "gain_active_batch": gain_active,
        "gain_gradient_state": gain_state,
        "gain_gradient_nonfinite": gain_nonfinite,
        "gain_contract_pass": True,
    }


def _audit_model_optimizer_state(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[float]:
    for name, value in model.state_dict().items():
        _require_finite_tensor(value, category="finite", name=f"state {name}")
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor) or not isinstance(state, Mapping):
            _fail("finite", "optimizer state identity differs")
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                _require_finite_tensor(
                    value, category="finite", name=f"optimizer state {name}"
                )
    gain = dict(model.named_parameters()).get(core.SBSC_V33_GAIN_STATE_KEY)
    if gain is None:
        _fail("gain_bounds", "gain parameter is absent")
    values = [float(value) for value in gain.detach().cpu().tolist()]
    if (
        len(values) != 4
        or any(not math.isfinite(value) for value in values)
        or any(
            value < core.SBSC_V33_GAIN_MIN or value > core.SBSC_V33_GAIN_MAX
            for value in values
        )
    ):
        _fail("gain_bounds", "gain is non-finite or outside [0,0.25]")
    return values


def _empty_epoch_accumulator(epoch: int, order: Sequence[int]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "order_sha256": contracts.canonical_sha256(list(order)),
        "processed_samples": 0,
        "optimizer_steps": 0,
        "combined_backward_calls": 0,
        "loss_sums": {"total": 0.0, "segmentation": 0.0, "router": 0.0},
        "router_level_loss_sums": [0.0, 0.0, 0.0, 0.0],
        "probability_min": 1.0,
        "probability_max": 0.0,
        "simplex_max_abs_error": 0.0,
        "availability_counts": {role: 0 for role in ROLE_NAMES},
        "availability_opportunities": {role: 0 for role in ROLE_NAMES},
        "intended_mode_counts": {name: 0 for name in MODE_NAMES},
        "effective_mode_counts": {name: 0 for name in MODE_NAMES},
        "certificate": {name: 0 for name in COMPACT_COUNTER_FIELDS},
        "gradient": {
            "all_non_gain_active_present": True,
            "all_non_gain_active_finite": True,
            "non_gain_active_parameter_count": None,
            "nonzero_gradient_elements": 0,
            "router_present_and_finite": True,
            "live_value_spatial_gradient_present_and_finite": True,
            "gain_active_batches": 0,
            "gain_inactive_batches": 0,
            "gain_gradient_nonfinite_batches": 0,
            "gain_gradient_state_counts": {
                "present_finite_required": 0,
                "absent_allowed": 0,
                "finite_zero_allowed": 0,
            },
            "gain_contract_pass": True,
        },
        "gain": None,
    }


def _accumulate_capture(accumulator: dict[str, Any], audit: Mapping[str, Any]) -> None:
    accumulator["probability_min"] = min(
        accumulator["probability_min"], audit["probability_min"]
    )
    accumulator["probability_max"] = max(
        accumulator["probability_max"], audit["probability_max"]
    )
    accumulator["simplex_max_abs_error"] = max(
        accumulator["simplex_max_abs_error"], audit["simplex_max_abs_error"]
    )
    for role in ROLE_NAMES:
        accumulator["availability_counts"][role] += audit["availability_counts"][role]
        accumulator["availability_opportunities"][role] += audit[
            "availability_opportunities"
        ][role]
    for name in MODE_NAMES:
        accumulator["intended_mode_counts"][name] += audit[
            "intended_mode_counts"
        ][name]
        accumulator["effective_mode_counts"][name] += audit[
            "effective_mode_counts"
        ][name]
    for name in COMPACT_COUNTER_FIELDS:
        accumulator["certificate"][name] += audit["certificate"][name]


def _accumulate_gradient(
    accumulator: dict[str, Any], gradient: Mapping[str, Any]
) -> None:
    target = accumulator["gradient"]
    count = gradient["non_gain_active_parameter_count"]
    if target["non_gain_active_parameter_count"] is None:
        target["non_gain_active_parameter_count"] = count
    elif target["non_gain_active_parameter_count"] != count:
        _fail("combined_backward", "non-gain active parameter count changed")
    for name in (
        "all_non_gain_active_present",
        "all_non_gain_active_finite",
        "router_present_and_finite",
        "live_value_spatial_gradient_present_and_finite",
        "gain_contract_pass",
    ):
        target[name] = bool(target[name] and gradient[name])
    target["nonzero_gradient_elements"] += gradient["nonzero_gradient_elements"]
    if gradient["gain_active_batch"]:
        target["gain_active_batches"] += 1
    else:
        target["gain_inactive_batches"] += 1
    if gradient["gain_gradient_nonfinite"]:
        target["gain_gradient_nonfinite_batches"] += 1
    target["gain_gradient_state_counts"][gradient["gain_gradient_state"]] += 1


def _finalize_epoch(accumulator: Mapping[str, Any], learning_rate: float) -> dict[str, Any]:
    samples = int(accumulator["processed_samples"])
    if samples <= 0:
        _fail("sample_step_counts", "epoch processed no samples")
    availability_fraction = {
        role: accumulator["availability_counts"][role]
        / accumulator["availability_opportunities"][role]
        for role in ROLE_NAMES
    }
    projection_rows = accumulator["certificate"]["projection_rows"]
    effective_mode_fraction = {
        name: accumulator["effective_mode_counts"][name] / projection_rows
        for name in MODE_NAMES
    }
    return {
        "epoch": accumulator["epoch"],
        "order_sha256": accumulator["order_sha256"],
        "learning_rate": learning_rate,
        "processed_samples": samples,
        "optimizer_steps": accumulator["optimizer_steps"],
        "combined_backward_calls": accumulator["combined_backward_calls"],
        "loss": {
            name: accumulator["loss_sums"][name] / samples
            for name in ("total", "segmentation", "router")
        },
        "router_level_loss": [value / samples for value in accumulator["router_level_loss_sums"]],
        "role_probability": {
            "minimum": accumulator["probability_min"],
            "maximum": accumulator["probability_max"],
            "simplex_max_abs_error": accumulator["simplex_max_abs_error"],
        },
        "availability": {
            "counts": dict(accumulator["availability_counts"]),
            "opportunities": dict(accumulator["availability_opportunities"]),
            "fraction": availability_fraction,
        },
        "modes": {
            "intended_counts": dict(accumulator["intended_mode_counts"]),
            "effective_counts": dict(accumulator["effective_mode_counts"]),
            "effective_fraction": effective_mode_fraction,
        },
        "certificate": dict(accumulator["certificate"]),
        "combined_backward": dict(accumulator["gradient"]),
        "gain": list(accumulator["gain"]),
        "hard_gate_violations": {name: 0 for name in HARD_GATE_ORDER},
    }


def _json_tree_is_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_json_tree_is_finite(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return all(_json_tree_is_finite(nested) for nested in value)
    return False


def evaluate_canary_gate(
    epochs: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
    *,
    execution_complete: bool,
) -> dict[str, Any]:
    """Re-evaluate only the seven objective hard gates from epoch records."""

    checks = {name: True for name in HARD_GATE_ORDER}
    reasons: list[str] = []
    if tuple(rules.get("hard_gate_order", ())) != HARD_GATE_ORDER:
        checks = {name: False for name in HARD_GATE_ORDER}
        reasons.append("hard gate rule order differs")
    if not execution_complete or len(epochs) != EPOCHS:
        checks["sample_step_counts"] = False
        reasons.append("20 epochs did not complete")
    expected_epoch = 1
    total_samples = 0
    total_steps = 0
    total_backwards = 0
    total_gain_active_batches = 0
    total_gain_nonfinite_batches = 0
    for record in epochs:
        if not isinstance(record, Mapping):
            checks["finite"] = False
            reasons.append("epoch record is malformed")
            continue
        if not _json_tree_is_finite(record):
            checks["finite"] = False
            reasons.append("epoch record contains non-finite data")
        violations = record.get("hard_gate_violations")
        if not isinstance(violations, Mapping) or set(violations) != set(HARD_GATE_ORDER):
            checks = {name: False for name in HARD_GATE_ORDER}
            reasons.append("hard gate violation ledger differs")
        else:
            for name in HARD_GATE_ORDER:
                if violations[name] != 0:
                    checks[name] = False
                    reasons.append(f"{name} violation count is nonzero")
        if (
            record.get("epoch") != expected_epoch
            or record.get("processed_samples") != SAMPLE_COUNT
            or record.get("optimizer_steps") != STEPS_PER_EPOCH
            or record.get("combined_backward_calls") != STEPS_PER_EPOCH
        ):
            checks["sample_step_counts"] = False
            reasons.append(f"epoch {expected_epoch} sample/step count differs")
        expected_epoch += 1
        processed = record.get("processed_samples")
        steps = record.get("optimizer_steps")
        backwards = record.get("combined_backward_calls")
        total_samples += processed if type(processed) is int else 0
        total_steps += steps if type(steps) is int else 0
        total_backwards += backwards if type(backwards) is int else 0
        probability = record.get("role_probability", {})
        if (
            probability.get("minimum", -1.0) < 0.0
            or probability.get("maximum", 2.0) > 1.0
            or probability.get("simplex_max_abs_error", math.inf) > 2.0e-6
        ):
            checks["role_probability"] = False
            reasons.append(f"epoch {record.get('epoch')} probability audit differs")
        availability = record.get("availability", {})
        counts = availability.get("counts", {})
        opportunities = availability.get("opportunities", {})
        if any(
            type(counts.get(role)) is not int
            or type(opportunities.get(role)) is not int
            or opportunities.get(role) != SAMPLE_COUNT * 4
            or not 0 <= counts.get(role) <= opportunities.get(role)
            for role in ROLE_NAMES
        ):
            checks["availability"] = False
            reasons.append(f"epoch {record.get('epoch')} availability audit differs")
        certificate = record.get("certificate", {})
        if (
            certificate.get("solver_accepted_rows", -1)
            + certificate.get("solver_fallback_rows", -1)
            != certificate.get("solver_attempt_rows", -3)
            or certificate.get("solver_attempt_rows", 1)
            > certificate.get("nonidentity_rows", 0)
            or certificate.get("emission_checked_rows", -1)
            != certificate.get("nonidentity_rows", -2)
            or certificate.get("emission_accepted_rows", -1)
            + certificate.get("emission_fallback_rows", -1)
            != certificate.get("emission_checked_rows", -3)
            or sum(
                certificate.get(f"mode_{name}_rows", -1) for name in MODE_NAMES
            )
            != certificate.get("projection_rows", -5)
            or sum(
                certificate.get(f"mode_{name}_rows", -1)
                for name in MODE_NAMES[1:]
            )
            != certificate.get("emission_accepted_rows", -4)
        ):
            checks["certificate"] = False
            reasons.append(f"epoch {record.get('epoch')} certificate differs")
        backward = record.get("combined_backward", {})
        if not all(
            backward.get(name) is True
            for name in (
                "all_non_gain_active_present",
                "all_non_gain_active_finite",
                "router_present_and_finite",
                "live_value_spatial_gradient_present_and_finite",
                "gain_contract_pass",
            )
        ) or backward.get("nonzero_gradient_elements", 0) <= 0:
            checks["combined_backward"] = False
            reasons.append(f"epoch {record.get('epoch')} backward audit differs")
        gain_active_batches = backward.get("gain_active_batches")
        gain_inactive_batches = backward.get("gain_inactive_batches")
        gain_nonfinite_batches = backward.get("gain_gradient_nonfinite_batches")
        state_counts = backward.get("gain_gradient_state_counts", {})
        if (
            type(gain_active_batches) is not int
            or type(gain_inactive_batches) is not int
            or type(gain_nonfinite_batches) is not int
            or gain_active_batches < 0
            or gain_inactive_batches < 0
            or gain_active_batches + gain_inactive_batches != STEPS_PER_EPOCH
            or gain_nonfinite_batches != 0
            or not isinstance(state_counts, Mapping)
            or set(state_counts)
            != {
                "present_finite_required",
                "absent_allowed",
                "finite_zero_allowed",
            }
            or any(type(value) is not int or value < 0 for value in state_counts.values())
            or sum(state_counts.values()) != STEPS_PER_EPOCH
            or state_counts.get("present_finite_required") != gain_active_batches
            or state_counts.get("absent_allowed", 0)
            + state_counts.get("finite_zero_allowed", 0)
            != gain_inactive_batches
        ):
            checks["combined_backward"] = False
            reasons.append(
                f"epoch {record.get('epoch')} conditional gain-gradient audit differs"
            )
        if type(gain_active_batches) is int:
            total_gain_active_batches += gain_active_batches
        if type(gain_nonfinite_batches) is int:
            total_gain_nonfinite_batches += gain_nonfinite_batches
        gain = record.get("gain")
        if (
            not isinstance(gain, list)
            or len(gain) != 4
            or any(value < 0.0 or value > 0.25 for value in gain)
        ):
            checks["gain_bounds"] = False
            reasons.append(f"epoch {record.get('epoch')} gain differs")
    if (
        total_samples != TOTAL_SAMPLES
        or total_steps != TOTAL_STEPS
        or total_backwards != TOTAL_STEPS
    ):
        checks["sample_step_counts"] = False
        reasons.append("total sample/step/backward count differs")
    if total_gain_active_batches < 1 or total_gain_nonfinite_batches != 0:
        checks["combined_backward"] = False
        reasons.append(
            "complete run requires at least one gain-active batch and no non-finite gain gradient"
        )
    verdict = "GO" if execution_complete and all(checks.values()) else "NO-GO"
    return {
        "schema": REPORT_SCHEMA + "/hard_gate/v2",
        "hard_gate_order": list(HARD_GATE_ORDER),
        "checks": checks,
        "execution_complete": execution_complete,
        "verdict": verdict,
        "reasons": reasons,
    }


def warning_diagnostics(
    epochs: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
) -> dict[str, Any]:
    """Return distribution/trend warnings that cannot change the verdict."""

    warnings: list[dict[str, Any]] = []
    if not epochs:
        return {"affects_verdict": False, "warnings": []}
    role_counts = {role: 0 for role in ROLE_NAMES}
    role_opportunities = {role: 0 for role in ROLE_NAMES}
    mode_counts = {name: 0 for name in MODE_NAMES}
    certificate = {name: 0 for name in COMPACT_COUNTER_FIELDS}
    for record in epochs:
        for role in ROLE_NAMES:
            role_counts[role] += record["availability"]["counts"][role]
            role_opportunities[role] += record["availability"]["opportunities"][role]
        for name in MODE_NAMES:
            mode_counts[name] += record["modes"]["effective_counts"][name]
        for name in COMPACT_COUNTER_FIELDS:
            certificate[name] += record["certificate"][name]
    role_fraction = {
        role: role_counts[role] / role_opportunities[role] for role in ROLE_NAMES
    }
    mode_total = sum(mode_counts.values())
    mode_fraction = {name: mode_counts[name] / mode_total for name in MODE_NAMES}
    for role, value in role_fraction.items():
        if value in (0.0, 1.0):
            warnings.append({"kind": "role_availability_extreme", "role": role, "fraction": value})
    for name, value in mode_fraction.items():
        if value in (0.0, 1.0):
            warnings.append({"kind": "mode_distribution_extreme", "mode": name, "fraction": value})
    warning_rules = rules["warning_diagnostics"]
    if len(epochs) >= 20:
        for level in range(4):
            early = sum(record["router_level_loss"][level] for record in epochs[:5]) / 5.0
            late = sum(record["router_level_loss"][level] for record in epochs[-5:]) / 5.0
            ratio = late / early if early > 0.0 else (1.0 if late == 0.0 else None)
            if ratio is None or ratio > warning_rules["router_ce_late_to_early_max_ratio"]:
                warnings.append(
                    {
                        "kind": "router_ce_trend",
                        "level": level,
                        "late_to_early_ratio": ratio,
                        "early_mean": early,
                        "late_mean": late,
                    }
                )
    nonidentity = certificate["nonidentity_rows"]
    if nonidentity > 0:
        emission_fraction = certificate["emission_fallback_rows"] / nonidentity
        solver_fraction = certificate["solver_fallback_rows"] / nonidentity
        if emission_fraction > warning_rules["emission_fallback_fraction"]:
            warnings.append({"kind": "emission_fallback_fraction", "fraction": emission_fraction})
        if solver_fraction > warning_rules["solver_fallback_fraction"]:
            warnings.append({"kind": "solver_fallback_fraction", "fraction": solver_fraction})
    else:
        emission_fraction = None
        solver_fraction = None
    if all(value == 0.0 for value in epochs[-1]["gain"]):
        warnings.append({"kind": "final_gain_all_zero"})
    return {
        "affects_verdict": False,
        "role_availability_fraction": role_fraction,
        "effective_mode_fraction": mode_fraction,
        "emission_fallback_fraction": emission_fraction,
        "solver_fallback_fraction": solver_fraction,
        "warnings": warnings,
    }


def _prepare_canary(device_value: str) -> dict[str, Any]:
    _require_runtime_environment()
    rules, method_config_sha = load_frozen_rules()
    method_config = rules["method_config"]
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    if (
        method_config["baseline_authority_sha256"] != baseline["manifest_sha256"]
        or method_config["gradient_authorization_sha256"]
        != gradient["authorization_sha256"]
        or gradient["authorized_router_value_gradient_mode"] != GRADIENT_MODE
    ):
        _fail("execution_contract", "method config authority binding differs")
    source_manifest = build_source_manifest()
    dataset_root = _canonical_dataset_root()
    dataset = EviSIRSTTrainDataset(
        DATASET,
        dataset_root=dataset_root,
        patch_size=PATCH_SIZE,
        seed=RUN_SEED,
        return_metadata=False,
    )
    if len(dataset) != SAMPLE_COUNT:
        _fail("sample_step_counts", "IRSTD-1K train split is not exactly 800")
    configure_determinism(RUN_SEED)
    model, builder_metadata = core.build_sctransnet_sbsc_v33_method(
        METHOD,
        DATASET,
        architecture_seed=ARCHITECTURE_SEED,
        training=True,
        router_value_gradient_mode=GRADIENT_MODE,
    )
    validation = core.validate_sctransnet_sbsc_v33(
        model,
        method=METHOD,
        router_value_gradient_mode=GRADIENT_MODE,
        require_zero_gain=True,
    )
    inactive_names = core.structurally_inactive_parameter_names(model)
    named_parameters = dict(model.named_parameters())
    if any(name not in named_parameters for name in inactive_names):
        _fail("execution_contract", "structurally inactive parameter set differs")
    for name in inactive_names:
        named_parameters[name].requires_grad_(False)
    active_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not active_names or set(active_names) & set(inactive_names):
        _fail("execution_contract", "active optimizer parameter scope differs")
    initial_state_sha = state_dict_sha256(model.state_dict())
    device = require_device(device_value)
    split = source_protocol.EXPECTED_SPLITS[DATASET]["train"]
    return {
        "rules": rules,
        "rules_sha256": EXPECTED_RULES_SHA256,
        "method_config_sha256": method_config_sha,
        "baseline": baseline,
        "gradient": gradient,
        "source_manifest": source_manifest,
        "dataset": dataset,
        "model": model,
        "builder_metadata": builder_metadata,
        "builder_metadata_sha256": contracts.canonical_sha256(builder_metadata),
        "model_validation": validation,
        "initial_state_sha256": initial_state_sha,
        "inactive_names": inactive_names,
        "active_names": active_names,
        "device": device,
        "device_requested": device_value,
        "split": dict(split),
    }


def build_manifest(prepared: Mapping[str, Any]) -> dict[str, Any]:
    baseline_irstd = prepared["baseline"]["authorities"][DATASET]
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "scope": "train_only_mechanism_health_not_performance",
        "write_once": True,
        "rules_path": "experiments/sbsc_v33_canary_rules_v2.json",
        "rules_sha256": prepared["rules_sha256"],
        "method_config": dict(prepared["rules"]["method_config"]),
        "method_config_sha256": prepared["method_config_sha256"],
        "superseded_v1_write_once_evidence": dict(
            prepared["rules"]["superseded_v1_write_once_evidence"]
        ),
        "authority_bindings": {
            "baseline": {
                "path": prepared["baseline"]["manifest_path"],
                "sha256": prepared["baseline"]["manifest_sha256"],
                "irstd_evaluation_path": baseline_irstd["evaluation_path"],
                "irstd_evaluation_sha256": baseline_irstd["evaluation_sha256"],
                "irstd_checkpoint_path": baseline_irstd["checkpoint"]["path"],
                "irstd_checkpoint_sha256": baseline_irstd["checkpoint"]["sha256"],
            },
            "gradient": {
                "path": prepared["gradient"]["authorization_path"],
                "sha256": prepared["gradient"]["authorization_sha256"],
                "authorized_router_value_gradient_mode": prepared["gradient"][
                    "authorized_router_value_gradient_mode"
                ],
                "diagnostic_report_path": prepared["gradient"]["report_path"],
                "diagnostic_report_sha256": prepared["gradient"]["report_sha256"],
                "diagnostic_rules_path": prepared["gradient"]["rules_path"],
                "diagnostic_rules_sha256": prepared["gradient"]["rules_sha256"],
                "diagnostic_source_manifest_sha256": prepared["gradient"][
                    "source_manifest_sha256"
                ],
            },
        },
        "training_source_manifest": dict(prepared["source_manifest"]),
        "data_identity": {
            "dataset_root": "datasets",
            "dataset": DATASET,
            "data_role": "train",
            "index_relpath": prepared["split"]["index_relpath"],
            "index_file_sha256": prepared["split"]["file_sha256"],
            "ordered_ids_sha256": prepared["split"]["ordered_ids_sha256"],
            "sample_count": prepared["split"]["count"],
            "normalization": dict(source_protocol.LEGACY_NORMALIZATION[DATASET]),
            "patch_size": PATCH_SIZE,
        },
        "model_identity": {
            "model": core.SBSC_V33_MODEL_NAME,
            "model_schema": core.SBSC_V33_SCHEMA,
            "method": METHOD,
            "builder_metadata_sha256": prepared["builder_metadata_sha256"],
            "initial_state_sha256": prepared["initial_state_sha256"],
            "state_key_count": prepared["model_validation"]["state_key_count"],
            "parameter_count": prepared["model_validation"]["parameter_count"],
            "inactive_parameter_names": list(prepared["inactive_names"]),
            "active_parameter_count": len(prepared["active_names"]),
        },
        "runtime_identity": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_requested": prepared["device_requested"],
            "device_type": prepared["device"].type,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "expected_execution": dict(prepared["rules"]["expected_execution"]),
        "artifact_boundaries": {
            "validation_dataset_constructed": False,
            "validation_loader_constructed": False,
            "test_dataset_constructed": False,
            "test_loader_constructed": False,
            "performance_evaluation": False,
            "model_selection": None,
            "checkpoint_written": False,
            "formal_weights_reusable": False,
            "manifest_path": f"{OUTPUT_NAMESPACE}/{MANIFEST_NAME}",
            "report_path": f"{OUTPUT_NAMESPACE}/{REPORT_NAME}",
        },
    }


def new_progress_ledger() -> dict[str, int]:
    """Create the mutable, reportable batch-level execution ledger."""

    return {
        "current_epoch": 0,
        "completed_epochs": 0,
        "started_batches": 0,
        "completed_batches": 0,
        "batch_count": 0,
        "combined_backward_calls": 0,
        "processed_samples": 0,
        "optimizer_steps": 0,
    }


def _execute_training(
    prepared: Mapping[str, Any],
    progress: dict[str, int],
    epoch_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if progress != new_progress_ledger() or epoch_records:
        _fail("execution_contract", "canary V2 progress ledger must start empty")
    rules = prepared["rules"]
    method_config = rules["method_config"]
    dataset = prepared["dataset"]
    model = prepared["model"]
    device = prepared["device"]
    active_names = prepared["active_names"]
    configure_determinism(RUN_SEED)
    model.to(device)
    model.train()
    model.mode = "train"
    criterion = nn.BCELoss(reduction="mean")
    optimizer_config = method_config["optimizer"]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimizer_config["base_lr"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
    )
    for epoch in range(1, EPOCHS + 1):
        progress["current_epoch"] = epoch
        dataset.set_epoch(epoch)
        order = _epoch_order(epoch)
        loader = DataLoader(
            Subset(dataset, order),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        learning_rate = _learning_rate(epoch, method_config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        accumulator = _empty_epoch_accumulator(epoch, order)
        for images, masks in loader:
            progress["started_batches"] += 1
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            _require_finite_tensor(images, category="finite", name="input images")
            _require_finite_tensor(masks, category="finite", name="input masks")
            if (
                tuple(images.shape[1:]) != (1, PATCH_SIZE, PATCH_SIZE)
                or tuple(masks.shape) != tuple(images.shape)
                or bool(((masks < 0.0) | (masks > 1.0)).any())
            ):
                _fail("finite", "training batch geometry/range differs")
            optimizer.zero_grad(set_to_none=True)
            with core.capture_c3_v33_training_router(model) as capture:
                outputs = _audit_outputs(model(images), masks)
            if len(capture.records) != 1:
                _fail("role_probability", "forward did not emit exactly one capture")
            capture_audit = audit_capture_record(capture.records[0])
            breakdown = core.router_loss_v33(
                capture,
                outputs[-1].detach(),
                masks,
                balance_mode=BALANCE_MODE,
            )
            segmentation_loss = sum(criterion(output, masks) for output in outputs)
            router_loss = breakdown.total
            total_loss = segmentation_loss + ROUTER_LOSS_WEIGHT * router_loss
            _audit_losses(
                total_loss,
                segmentation_loss,
                router_loss,
                breakdown.per_level,
            )
            value_spatial = capture.records[0]["value_spatial"]
            if not value_spatial.requires_grad:
                _fail("combined_backward", "live value-spatial tensor is detached")
            value_spatial.retain_grad()
            total_loss.backward()
            progress["combined_backward_calls"] += 1
            gradient = _audit_combined_backward(
                model,
                active_names,
                value_spatial,
                emission_accepted_rows=capture_audit["certificate"][
                    "emission_accepted_rows"
                ],
            )
            count = int(images.shape[0])
            optimizer.step()
            progress["optimizer_steps"] += 1
            progress["processed_samples"] += count
            core.project_sbsc_v33_constraints_(model)
            gains = _audit_model_optimizer_state(model, optimizer)
            progress["completed_batches"] += 1
            progress["batch_count"] = progress["completed_batches"]

            accumulator["processed_samples"] += count
            accumulator["optimizer_steps"] += 1
            accumulator["combined_backward_calls"] += 1
            accumulator["loss_sums"]["total"] += float(total_loss.detach().item()) * count
            accumulator["loss_sums"]["segmentation"] += float(
                segmentation_loss.detach().item()
            ) * count
            accumulator["loss_sums"]["router"] += float(router_loss.detach().item()) * count
            for level, value in enumerate(breakdown.per_level):
                accumulator["router_level_loss_sums"][level] += float(
                    value.detach().item()
                ) * count
            _accumulate_capture(accumulator, capture_audit)
            _accumulate_gradient(accumulator, gradient)
            accumulator["gain"] = gains
        epoch_record = _finalize_epoch(accumulator, learning_rate)
        if (
            epoch_record["processed_samples"] != SAMPLE_COUNT
            or epoch_record["optimizer_steps"] != STEPS_PER_EPOCH
            or epoch_record["combined_backward_calls"] != STEPS_PER_EPOCH
        ):
            _fail("sample_step_counts", f"epoch {epoch} count contract differs")
        epoch_records.append(epoch_record)
        progress["completed_epochs"] = len(epoch_records)
        print(
            "canary "
            f"epoch={epoch}/{EPOCHS} "
            f"loss={epoch_record['loss']['total']:.6f} "
            f"seg={epoch_record['loss']['segmentation']:.6f} "
            f"router={epoch_record['loss']['router']:.6f} "
            f"samples={epoch_record['processed_samples']} "
            f"steps={epoch_record['optimizer_steps']}",
            flush=True,
        )
    return epoch_records


def _report_common(manifest_sha: str, prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "write_once": True,
        "scope": "train_only_mechanism_health_not_performance",
        "manifest_path": f"{OUTPUT_NAMESPACE}/{MANIFEST_NAME}",
        "manifest_sha256": manifest_sha,
        "rules_path": "experiments/sbsc_v33_canary_rules_v2.json",
        "rules_sha256": prepared["rules_sha256"],
        "method_config_sha256": prepared["method_config_sha256"],
        "baseline_authority_sha256": prepared["baseline"]["manifest_sha256"],
        "gradient_authorization_sha256": prepared["gradient"][
            "authorization_sha256"
        ],
        "training_source_manifest_sha256": prepared["source_manifest"]["sha256"],
        "superseded_v1_report_sha256": prepared["rules"][
            "superseded_v1_write_once_evidence"
        ]["report_sha256"],
        "dataset": DATASET,
        "data_role": "train",
        "method": METHOD,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "router_value_gradient_mode": GRADIENT_MODE,
        "validation_dataset_constructed": False,
        "validation_loader_constructed": False,
        "test_dataset_constructed": False,
        "test_loader_constructed": False,
        "performance_evaluation": False,
        "model_selection": None,
        "checkpoint_written": False,
        "formal_weights_reusable": False,
    }


def _report_progress(progress: Mapping[str, int]) -> dict[str, Any]:
    expected_keys = set(new_progress_ledger())
    if set(progress) != expected_keys or any(
        type(value) is not int or value < 0 for value in progress.values()
    ):
        _fail("execution_contract", "batch-level progress ledger is malformed")
    if (
        progress["batch_count"] != progress["completed_batches"]
        or progress["completed_batches"] > progress["started_batches"]
        or progress["completed_batches"] > progress["optimizer_steps"]
        or progress["optimizer_steps"] > progress["combined_backward_calls"]
        or progress["combined_backward_calls"] > progress["started_batches"]
        or progress["processed_samples"]
        != progress["optimizer_steps"] * BATCH_SIZE
        or progress["completed_epochs"] * STEPS_PER_EPOCH
        > progress["completed_batches"]
        or progress["completed_epochs"] > progress["current_epoch"]
        or progress["current_epoch"] > EPOCHS
        or progress["started_batches"] > TOTAL_STEPS
    ):
        _fail("execution_contract", "batch-level progress arithmetic differs")
    return {
        "current_epoch": progress["current_epoch"],
        "completed_epochs": progress["completed_epochs"],
        "started_batches": progress["started_batches"],
        "completed_batches": progress["completed_batches"],
        "batch_count": progress["batch_count"],
        "combined_backward_calls": progress["combined_backward_calls"],
        "processed_samples": progress["processed_samples"],
        "optimizer_steps": progress["optimizer_steps"],
    }


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = DEFAULT_OUTPUT_DIR
    manifest_path = contracts.validated_output_path(output_dir / MANIFEST_NAME)
    report_path = contracts.validated_output_path(output_dir / REPORT_NAME)
    existing = tuple(output_dir.iterdir())
    if manifest_path.exists() or report_path.exists() or existing:
        raise FileExistsError("canary manifest/report transaction already exists")
    prepared = _prepare_canary(args.device)
    manifest = build_manifest(prepared)
    manifest_sha = contracts.write_once_json(manifest_path, manifest)
    started = time.monotonic()
    epochs: list[dict[str, Any]] = []
    progress = new_progress_ledger()
    try:
        _execute_training(prepared, progress, epochs)
        if progress != {
            "current_epoch": EPOCHS,
            "completed_epochs": EPOCHS,
            "started_batches": TOTAL_STEPS,
            "completed_batches": TOTAL_STEPS,
            "batch_count": TOTAL_STEPS,
            "combined_backward_calls": TOTAL_STEPS,
            "processed_samples": TOTAL_SAMPLES,
            "optimizer_steps": TOTAL_STEPS,
        }:
            _fail("sample_step_counts", "complete-run progress ledger differs")
        if build_source_manifest() != prepared["source_manifest"]:
            _fail("execution_contract", "training source changed during canary")
        gate = evaluate_canary_gate(epochs, prepared["rules"], execution_complete=True)
        warnings = warning_diagnostics(epochs, prepared["rules"])
        report = {
            **_report_common(manifest_sha, prepared),
            **_report_progress(progress),
            "status": "PASS" if gate["verdict"] == "GO" else "FAIL",
            "hard_gate_pass": gate["verdict"] == "GO",
            "elapsed_seconds": time.monotonic() - started,
            "execution_progress": _report_progress(progress),
            "epoch_records": epochs,
            "hard_gate": gate,
            "warning_diagnostics": warnings,
        }
    except Exception as exc:
        category = exc.category if isinstance(exc, CanaryError) else "runtime_failure"
        progress_payload = _report_progress(progress)
        report = {
            **_report_common(manifest_sha, prepared),
            **progress_payload,
            "status": "FAIL",
            "hard_gate_pass": False,
            "elapsed_seconds": time.monotonic() - started,
            "execution_progress": progress_payload,
            "epoch_records": epochs,
            "hard_gate": {
                "schema": REPORT_SCHEMA + "/hard_gate/v2",
                "hard_gate_order": list(HARD_GATE_ORDER),
                "checks": {name: False for name in HARD_GATE_ORDER},
                "execution_complete": False,
                "verdict": "NO-GO",
                "reasons": [f"{category}: {type(exc).__name__}: {exc}"],
            },
            "warning_diagnostics": {"affects_verdict": False, "warnings": []},
            "failure": {
                "category": category,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        }
    contracts.write_once_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = run_canary(parse_args(argv))
    print(contracts.canonical_json_bytes(report).decode("utf-8"), end="")
    return 0 if report["hard_gate"]["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
