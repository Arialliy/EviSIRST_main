#!/usr/bin/env python3
"""Fixed one-pass CP-HF-S2 mechanism diagnostic on legacy validation only.

For each preregistered run seed this program reloads the exact first-500
zero-margin-selected candidate, validates the complete architecture, and
observes the installed decoder adapter during one full 160-sample validation
pass.  It cannot select a checkpoint, tune a threshold, or access test data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from experiments import irstd_cp_hf_s2_v1 as architecture
from experiments.evisirst_v2_data import EviSIRSTV2ValDataset
import run_irstd_cp_hf_s2_first500_ab_gate_v1 as comparison_gate
import run_irstd_model_design_epoch500_gate_v1 as epoch500_gate
import train_irstd_cp_hf_s2_legacy_screen_v1 as runner


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_SCHEMA = "evisirst_irstd_cp_hf_s2_selected_mechanism_diagnostic/v1"
OUTPUT_ROOT = runner.DEFAULT_OUTPUT_ROOT / "comparison" / "mechanism"
RUN_SEEDS = runner.FORMAL_RUN_SEEDS
VALIDATION_SAMPLE_COUNT = 160


class CPHFS2MechanismDiagnosticError(ValueError):
    """The fixed checkpoint, validation pass, or correction differs."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-seed", type=int, choices=RUN_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.device != "cuda:0":
        parser.error("mechanism diagnostic requires logical cuda:0")
    return args


def resolve_output_path(run_seed: int) -> Path:
    if type(run_seed) is not int or run_seed not in RUN_SEEDS:
        raise CPHFS2MechanismDiagnosticError("run seed is outside the registry")
    return OUTPUT_ROOT / f"run_seed_{run_seed}" / "result.json"


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CPHFS2MechanismDiagnosticError(f"artifact is missing: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise CPHFS2MechanismDiagnosticError("artifact escaped repository") from exc
    return {
        "relative_path": relative,
        "sha256": runner._sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


@torch.inference_mode()
def diagnose_selected_checkpoint(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, Any]:
    extension = getattr(model, architecture.MODULE_NAME, None)
    if not isinstance(extension, architecture.ContextPurifiedHighFrequencyS2):
        raise CPHFS2MechanismDiagnosticError("CP-HF-S2 adapter is absent")
    architecture.validate_irstd_cp_hf_s2_v1(
        model, require_identity_initialization=False
    )
    model.to(device)
    model.eval()
    model.mode = "test"
    hook_calls = 0
    sample_count = 0
    pixel_count = 0
    nonzero_count = 0
    bound_violation_count = 0
    max_absolute_correction = 0.0
    max_bound_excess = -math.inf
    correction_square_sum = 0.0

    def observe(
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        nonlocal hook_calls, pixel_count, nonzero_count
        nonlocal bound_violation_count, max_absolute_correction
        nonlocal max_bound_excess, correction_square_sum
        if (
            module is not extension
            or len(inputs) != 2
            or not all(isinstance(item, torch.Tensor) for item in inputs)
            or not isinstance(output, torch.Tensor)
        ):
            raise CPHFS2MechanismDiagnosticError("adapter hook contract differs")
        feature, skip = inputs
        components = extension.refinement_components(feature, skip)
        expected = components["correction"]
        observed = output - feature
        tolerance = 16.0 * torch.finfo(observed.dtype).eps
        if not bool(torch.isfinite(observed).all()) or not bool(
            torch.isfinite(expected).all()
        ):
            raise CPHFS2MechanismDiagnosticError("adapter correction is non-finite")
        if not torch.allclose(observed, expected, rtol=1e-5, atol=tolerance):
            raise CPHFS2MechanismDiagnosticError(
                "hooked feature delta differs from adapter components"
            )
        bound = torch.full_like(observed, architecture.MAX_FEATURE_DELTA)
        numeric_slack = 16.0 * torch.finfo(observed.dtype).eps
        excess = observed.abs() - bound
        bound_violation_count += int(
            torch.count_nonzero(excess > numeric_slack).item()
        )
        max_bound_excess = max(max_bound_excess, float(excess.max().item()))
        nonzero_count += int(torch.count_nonzero(observed).item())
        pixel_count += observed.numel()
        max_absolute_correction = max(
            max_absolute_correction, float(observed.abs().max().item())
        )
        correction_square_sum += float(observed.double().square().sum().item())
        hook_calls += 1

    handle = extension.register_forward_hook(observe)
    try:
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise CPHFS2MechanismDiagnosticError(
                    "validation batch contract differs"
                )
            images = batch[0]
            if not isinstance(images, torch.Tensor) or images.shape[0] != 1:
                raise CPHFS2MechanismDiagnosticError(
                    "diagnostic requires batch size one"
                )
            model(images.to(device, non_blocking=True))
            sample_count += int(images.shape[0])
    finally:
        handle.remove()
    if (
        sample_count != VALIDATION_SAMPLE_COUNT
        or hook_calls != sample_count
        or pixel_count < 1
    ):
        raise CPHFS2MechanismDiagnosticError(
            "adapter did not execute exactly once per validation sample"
        )
    return {
        "validation_sample_count": sample_count,
        "adapter_hook_call_count": hook_calls,
        "evaluated_feature_element_count": pixel_count,
        "nonzero_correction_count": nonzero_count,
        "nonzero_correction_fraction": nonzero_count / pixel_count,
        "correction_finite": True,
        "absolute_correction_bound": architecture.MAX_FEATURE_DELTA,
        "bound_violation_count": bound_violation_count,
        "max_bound_excess": max_bound_excess,
        "correction_absolute_max": max_absolute_correction,
        "correction_rms": math.sqrt(correction_square_sum / pixel_count),
    }


def _selected_arm(run_seed: int) -> dict[str, Any]:
    if run_seed not in RUN_SEEDS:
        raise CPHFS2MechanismDiagnosticError("run seed is outside the registry")
    with comparison_gate._cp_reader_contract(epoch500_gate) as patched_gate:
        try:
            arm = patched_gate._read_arm(
                seed=run_seed, variant=runner.ARCHITECTURE_VARIANT
            )
        except patched_gate.ModelDesignEpoch500GateNotReady as exc:
            raise CPHFS2MechanismDiagnosticError(str(exc)) from exc
    history = comparison_gate._history_from_latest(
        torch,
        role=runner.ARCHITECTURE_VARIANT,
        seed=run_seed,
        expected_latest=arm["latest"],
        expected_identity_sha256=arm["run_identity_sha256"],
    )
    fresh = comparison_gate.select_first500(history)
    if fresh["selected_epoch"] != arm["selected_metrics"]["epoch"]:
        raise CPHFS2MechanismDiagnosticError(
            "fresh selected checkpoint differs from arm evidence"
        )
    return arm


def evaluate_payload(args: argparse.Namespace) -> dict[str, Any]:
    if (
        getattr(args, "device", None) != "cuda:0"
        or type(getattr(args, "run_seed", None)) is not int
        or args.run_seed not in RUN_SEEDS
        or not isinstance(getattr(args, "dataset_root", None), Path)
    ):
        raise CPHFS2MechanismDiagnosticError(
            "diagnostic requires the frozen seed/device/dataset-root contract"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CPHFS2MechanismDiagnosticError("requested CUDA is unavailable")
    dataset = EviSIRSTV2ValDataset(
        runner.DATASET,
        dataset_root=args.dataset_root,
        target_mode=runner.TARGET_MODE,
        split_root=runner.CANONICAL_SPLIT_ROOT,
        normalization_mode="legacy",
        return_metadata=False,
        verify_data_tree=True,
    )
    if len(dataset) != VALIDATION_SAMPLE_COUNT:
        raise CPHFS2MechanismDiagnosticError("validation sample count differs")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    arm = _selected_arm(args.run_seed)
    selected = arm.get("selected_candidate")
    metadata = selected.get("artifact") if isinstance(selected, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise CPHFS2MechanismDiagnosticError("selected candidate is missing")
    path = PROJECT_ROOT / str(metadata.get("relative_path"))
    if _artifact(path) != metadata:
        raise CPHFS2MechanismDiagnosticError("selected candidate changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    runner.require_test_false(payload, label="selected_candidate")
    if any(
        payload.get(name) is not False for name in runner._strict_false_disclosures()
    ):
        raise CPHFS2MechanismDiagnosticError(
            "selected candidate disclosure fields are incomplete"
        )
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise CPHFS2MechanismDiagnosticError("candidate state is missing")
    model, _ = architecture.build_irstd_cp_hf_s2_v1(
        runner.DATASET, seed=runner.ARCHITECTURE_SEED, training=True
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CPHFS2MechanismDiagnosticError("candidate strict load failed")
    architecture.validate_irstd_cp_hf_s2_v1(
        model, require_identity_initialization=False
    )
    summary = diagnose_selected_checkpoint(model, loader, device)
    if _artifact(path) != metadata:
        raise CPHFS2MechanismDiagnosticError(
            "selected candidate changed during diagnostic"
        )
    result = {
        "architecture_seed": runner.ARCHITECTURE_SEED,
        "run_seed": args.run_seed,
        "selected_epoch": arm["selected_metrics"]["epoch"],
        "selected_candidate": dict(metadata),
        "one_full_validation_pass": True,
        "summary": summary,
    }

    source_paths = {
        "diagnostic": Path(__file__),
        "comparison_gate": Path(comparison_gate.__file__),
        "epoch500_reader": Path(epoch500_gate.__file__),
        "runner": Path(runner.__file__),
        "protocol": runner.PROTOCOL_PATH,
        "rules": runner.RULES_PATH,
        "architecture": runner.ARCHITECTURE_PATH,
        "architecture_tests": runner.ARCHITECTURE_TEST_PATH,
        "validation_dataset": PROJECT_ROOT / "experiments/evisirst_v2_data.py",
        "zero_margin_selector": PROJECT_ROOT
        / "experiments/evisirst_zero_margin_selection.py",
    }
    sources = {name: _artifact(path) for name, path in source_paths.items()}
    output = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "data_role": "legacy_dev_val",
        "device": "cuda:0",
        "run_seed": args.run_seed,
        "validation_passes_per_selected_checkpoint": 1,
        "result": result,
        "diagnostic_only": True,
        "selection_allowed": False,
        "threshold_tuning_allowed": False,
        "source_identity": {
            "files": sources,
            "source_tree_sha256": runner._canonical_sha256(sources),
        },
        "public_test_allowed": False,
    }
    output.update(runner._strict_false_disclosures())
    runner.require_test_false(output, label="cp_hf_s2_mechanism_diagnostic")
    return json.loads(runner._canonical_bytes(output).decode("ascii"))


def run(args: argparse.Namespace) -> Path:
    return runner.write_json_no_clobber(
        resolve_output_path(args.run_seed), evaluate_payload(args)
    )


def main(argv: Sequence[str] | None = None) -> None:
    print(run(parse_args(argv)))


if __name__ == "__main__":
    main()


__all__ = [
    "CPHFS2MechanismDiagnosticError",
    "OUTPUT_ROOT",
    "RESULT_SCHEMA",
    "diagnose_selected_checkpoint",
    "evaluate_payload",
    "parse_args",
    "resolve_output_path",
    "run",
]
