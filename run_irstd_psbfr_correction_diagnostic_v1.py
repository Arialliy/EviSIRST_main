#!/usr/bin/env python3
"""One-pass legacy-validation correction diagnostic for PSBFR V1.

This diagnostic reads the fixed Seed-42 first-500 best-mIoU candidate and the
validation split only.  Its region summaries cannot select a checkpoint,
change a gate threshold, or authorize lockbox/public-test access.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments import irstd_psbfr_v1 as psbfr
from experiments.evisirst_v2_data import EviSIRSTV2ValDataset
import run_irstd_model_design_epoch500_gate_v1 as epoch500_gate
import train_irstd_model_design_screen_v1 as screen_runner


PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_SCHEMA = "evisirst_irstd_psbfr_correction_diagnostic/v1"
OUTPUT_RELATIVE_PATH = epoch500_gate.DIAGNOSTIC_RELATIVE_PATH
SOURCE_RELATIVE_PATH = "run_irstd_psbfr_correction_diagnostic_v1.py"
RUN_SEED = 42
BOUNDARY_KERNEL_SIZE = 7
SATURATION_THRESHOLD = 0.99
NONZERO_THRESHOLD = 0.0


class PSBFRCorrectionDiagnosticError(ValueError):
    """The fixed validation-only diagnostic contract was violated."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


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
        raise PSBFRCorrectionDiagnosticError("diagnostic is not finite JSON") from exc


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
        raise PSBFRCorrectionDiagnosticError("artifact escaped repository") from exc
    return {
        "relative_path": relative,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


class _SignedRegionAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.positive = 0
        self.negative = 0
        self.zero = 0
        self.total = 0.0
        self.absolute_total = 0.0

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().double().reshape(-1).cpu()
        self.count += flat.numel()
        self.positive += int(torch.count_nonzero(flat > 0).item())
        self.negative += int(torch.count_nonzero(flat < 0).item())
        self.zero += int(torch.count_nonzero(flat == 0).item())
        self.total += float(flat.sum().item())
        self.absolute_total += float(flat.abs().sum().item())

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean_signed_correction": None,
                "mean_absolute_correction": None,
                "positive_fraction": None,
                "negative_fraction": None,
                "zero_fraction": None,
            }
        return {
            "count": self.count,
            "mean_signed_correction": self.total / self.count,
            "mean_absolute_correction": self.absolute_total / self.count,
            "positive_fraction": self.positive / self.count,
            "negative_fraction": self.negative / self.count,
            "zero_fraction": self.zero / self.count,
        }


@torch.inference_mode()
def diagnose_validation_pass(
    model: torch.nn.Module,
    batches: Iterable[Any],
    device: torch.device,
) -> dict[str, Any]:
    """Aggregate one complete validation pass without retaining tensors."""

    refinement = getattr(model, psbfr.PSBFR_MODULE_NAME, None)
    if not isinstance(
        refinement, psbfr.PredictionSupportedBoundedFrequencyRefinement
    ):
        raise PSBFRCorrectionDiagnosticError("model is not formal PSBFR")
    model.eval()
    model.mode = "test"
    regions = {
        "target": _SignedRegionAccumulator(),
        "boundary": _SignedRegionAccumulator(),
        "background": _SignedRegionAccumulator(),
    }
    sample_count = 0
    pixel_count = 0
    nonzero_count = 0
    saturation_count = 0
    bound_violation_count = 0
    max_bound_excess = -math.inf
    correction_abs_max = 0.0
    correction_square_sum = 0.0
    captured: dict[str, tuple[torch.Tensor, ...]] = {}

    def capture(
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        _output: Any,
    ) -> None:
        if len(inputs) != 2 or not all(isinstance(value, torch.Tensor) for value in inputs):
            raise PSBFRCorrectionDiagnosticError("PSBFR hook inputs differ")
        captured["components"] = refinement.refinement_components(inputs[0], inputs[1])

    handle = refinement.register_forward_hook(capture)
    try:
        for batch in batches:
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise PSBFRCorrectionDiagnosticError("validation batch contract differs")
            images, masks, sizes, sample_ids = batch
            if images.shape[0] != 1 or masks.shape[0] != 1:
                raise PSBFRCorrectionDiagnosticError("diagnostic requires batch size one")
            if isinstance(sizes, (tuple, list)) and len(sizes) == 2:
                height = int(torch.as_tensor(sizes[0]).reshape(-1)[0])
                width = int(torch.as_tensor(sizes[1]).reshape(-1)[0])
            else:
                flattened = torch.as_tensor(sizes).reshape(-1)
                height, width = int(flattened[0]), int(flattened[1])
            captured.clear()
            model(images.to(device, non_blocking=True))
            components = captured.get("components")
            if components is None:
                raise PSBFRCorrectionDiagnosticError("PSBFR components were not captured")
            _low, _high, raw, support, correction = components
            raw = raw[:, :, :height, :width]
            support = support[:, :, :height, :width]
            correction = correction[:, :, :height, :width]
            target = masks[:, :, :height, :width].to(device) > 0.5
            dilated = F.max_pool2d(
                target.float(),
                kernel_size=BOUNDARY_KERNEL_SIZE,
                stride=1,
                padding=BOUNDARY_KERNEL_SIZE // 2,
            ) > 0.5
            boundary = dilated & ~target
            background = ~dilated
            regions["target"].update(correction[target])
            regions["boundary"].update(correction[boundary])
            regions["background"].update(correction[background])
            bound = psbfr.PSBFR_MAX_LOGIT_DELTA * support
            tolerance = 8.0 * torch.finfo(correction.dtype).eps * torch.maximum(
                torch.ones_like(bound), bound
            )
            excess = correction.abs() - bound
            bound_violation_count += int(torch.count_nonzero(excess > tolerance).item())
            max_bound_excess = max(max_bound_excess, float(excess.max().item()))
            nonzero_count += int(
                torch.count_nonzero(correction.abs() > NONZERO_THRESHOLD).item()
            )
            saturation_count += int(
                torch.count_nonzero(torch.tanh(raw).abs() >= SATURATION_THRESHOLD).item()
            )
            pixel_count += correction.numel()
            correction_abs_max = max(
                correction_abs_max, float(correction.abs().max().item())
            )
            correction_square_sum += float(correction.double().square().sum().item())
            sample_count += int(images.shape[0])
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            if len(sample_ids) != 1:
                raise PSBFRCorrectionDiagnosticError("sample id batch differs")
    finally:
        handle.remove()
    if sample_count < 1 or pixel_count < 1:
        raise PSBFRCorrectionDiagnosticError("validation pass is empty")
    return {
        "validation_sample_count": sample_count,
        "evaluated_pixel_count": pixel_count,
        "bound_violation_count": bound_violation_count,
        "max_bound_excess": max_bound_excess,
        "nonzero_correction_count": nonzero_count,
        "nonzero_correction_fraction": nonzero_count / pixel_count,
        "tanh_saturation_threshold": SATURATION_THRESHOLD,
        "tanh_saturation_count": saturation_count,
        "tanh_saturation_fraction": saturation_count / pixel_count,
        "correction_absolute_max": correction_abs_max,
        "correction_rms": math.sqrt(correction_square_sum / pixel_count),
        "signed_region_diagnostics": {
            name: accumulator.result() for name, accumulator in regions.items()
        },
    }


def evaluate_payload(args: argparse.Namespace) -> dict[str, Any]:
    arm = epoch500_gate._read_arm(seed=RUN_SEED, variant="psbfr_v1")
    candidate_artifact = arm["selected_candidate"]["artifact"]
    candidate_path = PROJECT_ROOT / candidate_artifact["relative_path"]
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if not isinstance(candidate, Mapping) or candidate.get("test_split_accessed") is not False:
        raise PSBFRCorrectionDiagnosticError("candidate payload differs")
    model, _ = psbfr.build_irstd_psbfr_v1("IRSTD-1K", seed=42, training=True)
    model.load_state_dict(candidate["state_dict"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PSBFRCorrectionDiagnosticError("requested CUDA is unavailable")
    model.to(device)
    dataset = EviSIRSTV2ValDataset(
        "IRSTD-1K",
        dataset_root=args.dataset_root,
        target_mode="binary",
        split_root=screen_runner.CANONICAL_SPLIT_ROOT,
        normalization_mode="legacy",
        return_metadata=False,
        verify_data_tree=True,
    )
    if len(dataset) != screen_runner.CANONICAL_IRSTD_VAL_COUNT:
        raise PSBFRCorrectionDiagnosticError("validation sample count differs")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    summary = diagnose_validation_pass(model, loader, device)
    sources = {
        "diagnostic": _artifact(PROJECT_ROOT / SOURCE_RELATIVE_PATH),
        "architecture": _artifact(Path(psbfr.__file__)),
        "protocol": _artifact(PROJECT_ROOT / epoch500_gate.PROTOCOL_RELATIVE_PATH),
        "rules": _artifact(PROJECT_ROOT / epoch500_gate.RULES_RELATIVE_PATH),
        "selector": _artifact(PROJECT_ROOT / epoch500_gate.SELECTOR_RELATIVE_PATH),
    }
    return json.loads(
        _canonical_bytes(
            {
                "schema": RESULT_SCHEMA,
                "status": "complete",
                "data_role": "legacy_dev_val",
                "run_seed": RUN_SEED,
                "selected_candidate": candidate_artifact,
                "selected_epoch": arm["selected_metrics"]["epoch"],
                "summary": summary,
                "diagnostic_only": True,
                "selection_allowed": False,
                "threshold_tuning_allowed": False,
                "lockbox_accessed": False,
                "test_split_accessed": False,
                "public_test_allowed": False,
                "source_identity": sources,
            }
        ).decode("ascii")
    )


def _write_no_replace(payload: Mapping[str, Any]) -> Path:
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
                raise FileExistsError(f"immutable diagnostic exists: {path}") from exc
            raise
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def validate_existing_result(args: argparse.Namespace) -> dict[str, Any]:
    path = PROJECT_ROOT / OUTPUT_RELATIVE_PATH
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = evaluate_payload(args)
    if _canonical_bytes(observed) != _canonical_bytes(expected):
        raise PSBFRCorrectionDiagnosticError(
            "existing diagnostic differs from fresh validation"
        )
    return expected


def run(args: argparse.Namespace) -> Path:
    return _write_no_replace(evaluate_payload(args))


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "OUTPUT_RELATIVE_PATH",
    "PSBFRCorrectionDiagnosticError",
    "RESULT_SCHEMA",
    "diagnose_validation_pass",
    "evaluate_payload",
    "parse_args",
    "run",
    "validate_existing_result",
]
