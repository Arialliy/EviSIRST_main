#!/usr/bin/env python3
"""Fresh-state, train-only gradient authorization for C3-SBSC V3.3.

The diagnostic never constructs a test dataset, never takes an optimizer step,
and restores the exact seed-42 initial model state before every batch.  It
reports the detach-boundary activation, the direct L1 V parameters, and a
frozen upstream L0 V parameter group.  A separate write-once authorization is
materialized only after the raw records are replayed against the preregistered
rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[1]
if str(_BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments import sctransnet_sbsc_v33 as core
from experiments.evisirst_data import EviSIRSTTrainDataset, stable_uint63
from experiments.four_dataset_models_seed42_v1 import state_dict_sha256
from train import configure_determinism


PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[1]
RULES_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sbsc_v33_gradient_authorization_rules.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "sbsc_v33_preflight"
    / "gradient_diagnostic.json"
)
DEFAULT_AUTHORIZATION_PATH = (
    PROJECT_ROOT / "experiments" / "sbsc_v33_gradient_authorization.json"
)
REPORT_SCHEMA = "sctransnet_sbsc_v33/gradient_diagnostic/v1"
AUTHORIZATION_SCHEMA = "sctransnet_sbsc_v33/gradient_authorization/v1"
EXPECTED_TRAIN_COUNTS = {
    "NUAA-SIRST": 213,
    "NUDT-SIRST": 663,
    "IRSTD-1K": 800,
}
SOURCE_FILES = (
    "experiments/sctransnet_sbsc_v31.py",
    "experiments/sctransnet_sbsc_v32.py",
    "experiments/sctransnet_sbsc_v33.py",
    "experiments/evisirst_data.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/sbsc_v33_gradient_authorization_rules.json",
    "tools/diagnose_sbsc_v33_gradient_conflict.py",
)


@dataclass(frozen=True)
class OrderedParameterSet:
    name: str
    names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"JSON authority is not a regular file: {path}")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise TypeError("JSON authority must contain one object")
    return payload


def _relative_regular_path(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"path is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("artifact path escapes project root") from exc
    return relative.as_posix()


def _validated_output_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    parent = candidate.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError("output parent cannot be a symlink")
    resolved_parent = parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("output path escapes project root") from exc
    resolved = resolved_parent / candidate.name
    if resolved.is_symlink():
        raise ValueError("output artifact cannot be a symlink")
    return resolved


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> str:
    destination = _validated_output_path(path)
    content = _canonical_bytes(dict(payload))
    digest = hashlib.sha256(content).hexdigest()
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise FileExistsError("existing artifact is not a regular file")
        if destination.read_bytes() != content:
            raise FileExistsError(f"write-once artifact differs: {destination}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    rules = _load_strict_json(path)
    expected = {
        "schema": "sctransnet_sbsc_v33/gradient_authorization_rules/v1",
        "status": "preregistered",
        "architecture_seed": 42,
        "run_seed": 42,
        "diagnostic_router_value_gradient_mode": "live",
        "loss_profile": "one_third_two_thirds",
        "router_level_count": 4,
        "router_level_reduction": "mean",
        "dataset_role": "train",
        "test_loader_constructed": False,
        "datasets": list(core.SUPPORTED_DATASETS),
        "dataset_epoch": 0,
        "sample_selection": "evenly_spaced_over_frozen_ordered_train_ids",
        "samples_per_dataset": 64,
        "batch_size": 4,
        "batches_per_dataset": 16,
        "drop_last": False,
        "fresh_state_per_batch": True,
        "fresh_rng_substream_per_batch": True,
        "repeatability_batches_per_dataset": 1,
        "zero_norm_policy": "invalid_batch",
        "required_valid_batches_per_dataset_per_group": 16,
        "otherwise": "live",
        "write_once": True,
    }
    for key, value in expected.items():
        if rules.get(key) != value:
            raise ValueError(f"gradient rule {key!r} differs")
    parameter_groups = rules.get("parameter_groups")
    if parameter_groups != {
        "direct_v": [
            "mtc.encoder.layer.1.channel_attn.mheadv.weight",
            "mtc.encoder.layer.1.channel_attn.v.weight",
        ],
        "upstream_shared": [
            "mtc.encoder.layer.0.channel_attn.mheadv.weight",
            "mtc.encoder.layer.0.channel_attn.v.weight",
        ],
    }:
        raise ValueError("gradient parameter groups differ")
    if rules.get("activation_groups") != ["value_spatial"]:
        raise ValueError("gradient activation groups differ")
    bootstrap = rules.get("bootstrap")
    if bootstrap != {
        "seed": 42,
        "resamples": 10000,
        "unit": "batch",
        "statistic": "median",
        "interval": "percentile",
        "confidence": 0.95,
    }:
        raise ValueError("gradient bootstrap rules differ")
    threshold = rules.get("detached_only_if_every_dataset_and_group_passes")
    if threshold != {
        "median_cosine_lt": 0.0,
        "bootstrap_cosine_ci_upper_lt": 0.0,
        "median_router_to_segmentation_norm_ratio_gte": 0.1,
    }:
        raise ValueError("gradient decision thresholds differ")
    return rules


def select_ordered_parameter_sets(
    model: nn.Module,
    rules: Mapping[str, Any],
) -> dict[str, OrderedParameterSet]:
    by_name = dict(model.named_parameters())
    result: dict[str, OrderedParameterSet] = {}
    groups = rules["parameter_groups"]
    for group_name in ("direct_v", "upstream_shared"):
        names = tuple(groups[group_name])
        if tuple(name for name in names if name in by_name) != names:
            missing = [name for name in names if name not in by_name]
            raise RuntimeError(f"gradient group {group_name} missing {missing}")
        parameters = tuple(by_name[name] for name in names)
        if any(not parameter.requires_grad for parameter in parameters):
            raise RuntimeError(f"gradient group {group_name} contains frozen state")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError(f"gradient group {group_name} aliases parameters")
        result[group_name] = OrderedParameterSet(
            name=group_name,
            names=names,
            parameters=parameters,
        )
    return result


def aligned_gradient_tensors(
    ordered: OrderedParameterSet,
    gradients: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    """Preserve every parameter position and replace None by exact zeros."""

    if len(gradients) != len(ordered.parameters):
        raise ValueError("gradient count differs from ordered parameter count")
    aligned: list[torch.Tensor] = []
    for name, parameter, gradient in zip(
        ordered.names,
        ordered.parameters,
        gradients,
        strict=True,
    ):
        if gradient is None:
            aligned.append(
                torch.zeros_like(
                    parameter,
                    dtype=torch.float32,
                    memory_format=torch.preserve_format,
                )
            )
            continue
        if gradient.shape != parameter.shape or gradient.device != parameter.device:
            raise RuntimeError(f"gradient geometry differs for {name!r}")
        if not bool(torch.isfinite(gradient.detach()).all()):
            raise RuntimeError(f"non-finite gradient for {name!r}")
        aligned.append(gradient.detach().float())
    return tuple(aligned)


def _gradient_dot_norms(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(left) != len(right) or not left:
        raise ValueError("aligned gradient collections differ or are empty")
    device = left[0].device
    dot = torch.zeros((), device=device, dtype=torch.float64)
    left_sq = torch.zeros_like(dot)
    right_sq = torch.zeros_like(dot)
    for left_tensor, right_tensor in zip(left, right, strict=True):
        if left_tensor.shape != right_tensor.shape:
            raise RuntimeError("aligned gradient tensor shapes differ")
        left64 = left_tensor.double()
        right64 = right_tensor.double()
        dot = dot + (left64 * right64).sum()
        left_sq = left_sq + left64.square().sum()
        right_sq = right_sq + right64.square().sum()
    return dot, left_sq.sqrt(), right_sq.sqrt()


def _statistics_from_aligned(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> dict[str, Any]:
    dot, segmentation_norm, router_norm = _gradient_dot_norms(left, right)
    segmentation_value = float(segmentation_norm.detach().cpu())
    router_value = float(router_norm.detach().cpu())
    valid = (
        math.isfinite(segmentation_value)
        and math.isfinite(router_value)
        and segmentation_value > 0.0
        and router_value > 0.0
    )
    if not valid:
        return {
            "valid": False,
            "invalid_reason": "zero_or_nonfinite_norm",
            "cosine": None,
            "router_to_segmentation_norm": None,
            "segmentation_norm": segmentation_value,
            "router_norm": router_value,
        }
    cosine = float((dot / (segmentation_norm * router_norm)).detach().cpu())
    ratio = router_value / segmentation_value
    if not math.isfinite(cosine) or not math.isfinite(ratio):
        raise RuntimeError("gradient statistic is non-finite")
    return {
        "valid": True,
        "invalid_reason": None,
        "cosine": cosine,
        "router_to_segmentation_norm": ratio,
        "segmentation_norm": segmentation_value,
        "router_norm": router_value,
    }


def gradient_conflict_statistics(
    *,
    segmentation_loss: torch.Tensor,
    router_loss: torch.Tensor,
    ordered: OrderedParameterSet,
) -> dict[str, Any]:
    if segmentation_loss.ndim != 0 or router_loss.ndim != 0:
        raise ValueError("losses must be scalar")
    segmentation_gradients = torch.autograd.grad(
        segmentation_loss,
        ordered.parameters,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    router_gradients = torch.autograd.grad(
        router_loss,
        ordered.parameters,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    result = _statistics_from_aligned(
        aligned_gradient_tensors(ordered, segmentation_gradients),
        aligned_gradient_tensors(ordered, router_gradients),
    )
    result.update(
        {
            "group": ordered.name,
            "parameter_names": list(ordered.names),
            "parameter_names_sha256": _canonical_sha256(list(ordered.names)),
            "parameter_count": sum(
                int(parameter.numel()) for parameter in ordered.parameters
            ),
        }
    )
    return result


def activation_gradient_conflict_statistics(
    *,
    segmentation_loss: torch.Tensor,
    router_loss: torch.Tensor,
    activation: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(activation, torch.Tensor) or not activation.requires_grad:
        raise ValueError("boundary activation must require gradients")
    gradients: list[torch.Tensor | None] = []
    for loss in (segmentation_loss, router_loss):
        gradient = torch.autograd.grad(
            loss,
            activation,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )[0]
        gradients.append(gradient)
    zero = torch.zeros_like(activation, dtype=torch.float32)
    aligned = tuple(
        zero if gradient is None else gradient.detach().float()
        for gradient in gradients
    )
    if any(
        gradient is not None and not bool(torch.isfinite(gradient).all())
        for gradient in gradients
    ):
        raise RuntimeError("boundary activation gradient is non-finite")
    result = _statistics_from_aligned((aligned[0],), (aligned[1],))
    result.update(
        {
            "group": "value_spatial",
            "activation_shape": list(activation.shape),
            "activation_numel": int(activation.numel()),
        }
    )
    return result


def _evenly_spaced_indices(count: int, sample_count: int) -> tuple[int, ...]:
    if count < sample_count or sample_count < 2:
        raise ValueError("cannot select requested evenly spaced samples")
    indices = tuple(
        (index * (count - 1)) // (sample_count - 1)
        for index in range(sample_count)
    )
    if len(indices) != len(set(indices)) or indices[0] != 0 or indices[-1] != count - 1:
        raise RuntimeError("evenly spaced sample indices are malformed")
    return indices


def _source_manifest() -> dict[str, Any]:
    records = []
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"source file is not regular: {relative}")
        records.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema": REPORT_SCHEMA + "/source_manifest/v1",
        "files": records,
        "sha256": _canonical_sha256(records),
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("git HEAD is malformed")
    return value


def _device_record(device: torch.device) -> dict[str, Any]:
    if device.type == "cpu":
        return {"type": "cpu", "torch_num_threads": torch.get_num_threads()}
    properties = torch.cuda.get_device_properties(device)
    return {
        "type": "cuda",
        "index": int(device.index or 0),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
    }


def _batch_statistics(
    *,
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    parameter_sets: Mapping[str, OrderedParameterSet],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    with core.capture_c3_v33_training_router(model) as capture:
        outputs = model(images)
    if not isinstance(outputs, tuple) or len(outputs) != 6:
        raise RuntimeError("gradient diagnostic requires six model outputs")
    breakdown = core.router_loss_v33(
        capture,
        outputs[-1].detach(),
        masks,
        balance_mode="one_third_two_thirds",
    )
    segmentation_loss = sum(criterion(output, masks) for output in outputs)
    record = capture.records[0]
    value_spatial = record.get("value_spatial")
    if not isinstance(value_spatial, torch.Tensor):
        raise RuntimeError("V3.3 capture omitted value_spatial")
    groups = {
        "value_spatial": activation_gradient_conflict_statistics(
            segmentation_loss=segmentation_loss,
            router_loss=breakdown.total,
            activation=value_spatial,
        )
    }
    for group_name in ("direct_v", "upstream_shared"):
        groups[group_name] = gradient_conflict_statistics(
            segmentation_loss=segmentation_loss,
            router_loss=breakdown.total,
            ordered=parameter_sets[group_name],
        )
    return {
        "segmentation_loss": float(segmentation_loss.detach().cpu()),
        "router_loss": float(breakdown.total.detach().cpu()),
        "router_level_losses": [
            float(value.detach().cpu()) for value in breakdown.per_level
        ],
        "groups": groups,
        "compact_diagnostics": record["compact_diagnostics"],
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile input is malformed")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _bootstrap_median_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = random.Random(seed)
    count = len(values)
    medians = []
    for _ in range(resamples):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        medians.append(float(torch.tensor(sample, dtype=torch.float64).median()))
    medians.sort()
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def decide_authorized_mode(
    records_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay all raw batches and apply the preregistered conservative rule."""

    required = int(rules["required_valid_batches_per_dataset_per_group"])
    resamples = int(rules["bootstrap"]["resamples"])
    base_seed = int(rules["bootstrap"]["seed"])
    group_names = ("value_spatial", "direct_v", "upstream_shared")
    aggregates: dict[str, Any] = {}
    all_pass = True
    for dataset_index, dataset in enumerate(rules["datasets"]):
        records = list(records_by_dataset.get(dataset, ()))
        if len(records) != int(rules["batches_per_dataset"]):
            raise ValueError(f"diagnostic batch count differs for {dataset}")
        dataset_aggregate: dict[str, Any] = {}
        for group_index, group_name in enumerate(group_names):
            values = [record["groups"][group_name] for record in records]
            valid = [value for value in values if value.get("valid") is True]
            if len(valid) != required:
                group_pass = False
                median_cosine = None
                median_ratio = None
                ci_lower = None
                ci_upper = None
            else:
                cosines = [float(value["cosine"]) for value in valid]
                ratios = [
                    float(value["router_to_segmentation_norm"]) for value in valid
                ]
                median_cosine = float(
                    torch.tensor(cosines, dtype=torch.float64).median()
                )
                median_ratio = float(
                    torch.tensor(ratios, dtype=torch.float64).median()
                )
                ci_lower, ci_upper = _bootstrap_median_ci(
                    cosines,
                    seed=stable_uint63(
                        base_seed,
                        dataset_index,
                        group_index,
                        dataset,
                        group_name,
                    ),
                    resamples=resamples,
                )
                group_pass = (
                    median_cosine < 0.0
                    and ci_upper < 0.0
                    and median_ratio >= 0.10
                )
            dataset_aggregate[group_name] = {
                "valid_batch_count": len(valid),
                "required_valid_batch_count": required,
                "median_cosine": median_cosine,
                "bootstrap_cosine_ci_95": [ci_lower, ci_upper],
                "median_router_to_segmentation_norm_ratio": median_ratio,
                "detached_conditions_pass": group_pass,
            }
            all_pass = all_pass and group_pass
        aggregates[dataset] = dataset_aggregate
    mode = "detached" if all_pass else "live"
    return {
        "authorized_router_value_gradient_mode": mode,
        "all_dataset_group_detached_conditions_pass": all_pass,
        "aggregates": aggregates,
    }


def run_diagnostic(
    *,
    dataset_root: Path,
    device: torch.device,
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    configure_determinism(42)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    source_manifest = _source_manifest()
    rules_sha = _sha256_file(RULES_PATH)
    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    dataset_manifests: dict[str, Any] = {}
    initial_state_hashes: dict[str, str] = {}
    criterion = nn.BCELoss(reduction="mean")
    for dataset in rules["datasets"]:
        full_train = EviSIRSTTrainDataset(
            dataset,
            dataset_root=dataset_root,
            return_metadata=False,
        )
        full_train.set_epoch(int(rules["dataset_epoch"]))
        if len(full_train) != EXPECTED_TRAIN_COUNTS[dataset]:
            raise RuntimeError(f"train count differs for {dataset}")
        indices = _evenly_spaced_indices(
            len(full_train),
            int(rules["samples_per_dataset"]),
        )
        sample_ids = [full_train.sample_ids[index] for index in indices]
        batches = [
            sample_ids[offset : offset + int(rules["batch_size"])]
            for offset in range(0, len(sample_ids), int(rules["batch_size"]))
        ]
        if len(batches) != int(rules["batches_per_dataset"]):
            raise RuntimeError("diagnostic batch manifest count differs")
        dataset_manifests[dataset] = {
            "full_train_count": len(full_train),
            "selected_indices": list(indices),
            "selected_sample_ids": sample_ids,
            "selected_sample_ids_sha256": _canonical_sha256(sample_ids),
            "batch_sample_ids": batches,
            "batch_sample_ids_sha256": _canonical_sha256(batches),
            "dataset_epoch": int(rules["dataset_epoch"]),
        }
        loader = DataLoader(
            Subset(full_train, indices),
            batch_size=int(rules["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        model, metadata = core.build_sctransnet_sbsc_v33(
            dataset=dataset,
            architecture_seed=42,
            training=True,
            router_value_gradient_mode="live",
        )
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        initial_hash = state_dict_sha256(initial_state)
        if initial_hash != metadata["state_sha256"]:
            raise RuntimeError("fresh CPU model-state hash differs")
        initial_state_hashes[dataset] = initial_hash
        model.to(device)
        model.train()
        model.mode = "train"
        parameter_sets = select_ordered_parameter_sets(model, rules)
        dataset_records: list[dict[str, Any]] = []
        for batch_index, (images, masks) in enumerate(loader):
            incompatible = model.load_state_dict(initial_state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("fresh-state reload differs")
            if state_dict_sha256(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            ) != initial_hash:
                raise RuntimeError("pre-batch model state is not fresh")
            batch_seed = stable_uint63(
                42,
                "sbsc_v33_gradient_diagnostic",
                dataset,
                batch_index,
            )
            fork_devices = [int(device.index or 0)] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(batch_seed)
                images_device = images.to(device, non_blocking=True)
                masks_device = masks.to(device, non_blocking=True)
                statistics = _batch_statistics(
                    model=model,
                    images=images_device,
                    masks=masks_device,
                    criterion=criterion,
                    parameter_sets=parameter_sets,
                )
            statistics.update(
                {
                    "batch_index": batch_index,
                    "sample_ids": batches[batch_index],
                    "batch_seed": batch_seed,
                    "fresh_initial_state_sha256": initial_hash,
                }
            )
            if batch_index < int(rules["repeatability_batches_per_dataset"]):
                incompatible = model.load_state_dict(initial_state, strict=True)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError("repeatability fresh-state reload differs")
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(batch_seed)
                    repeated = _batch_statistics(
                        model=model,
                        images=images.to(device, non_blocking=True),
                        masks=masks.to(device, non_blocking=True),
                        criterion=criterion,
                        parameter_sets=parameter_sets,
                    )
                if _canonical_sha256(repeated) != _canonical_sha256(
                    {
                        key: value
                        for key, value in statistics.items()
                        if key
                        not in {
                            "batch_index",
                            "sample_ids",
                            "batch_seed",
                            "fresh_initial_state_sha256",
                        }
                    }
                ):
                    raise RuntimeError("fresh-state batch diagnostic is not repeatable")
                statistics["repeatability_check"] = "PASS"
            else:
                statistics["repeatability_check"] = "not_repeated"
            dataset_records.append(statistics)
        if len(dataset_records) != int(rules["batches_per_dataset"]):
            raise RuntimeError("processed diagnostic batch count differs")
        records_by_dataset[dataset] = dataset_records
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    decision = decide_authorized_mode(records_by_dataset, rules)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": "fresh_state_train_only_three_dataset/v1",
        "git_head": _git_head(),
        "source_manifest": source_manifest,
        "rules_path": _relative_regular_path(RULES_PATH),
        "rules_sha256": rules_sha,
        "rules_canonical_sha256": _canonical_sha256(dict(rules)),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": _device_record(device),
        },
        "architecture_seed": 42,
        "run_seed": 42,
        "dataset_role": "train",
        "test_loader_constructed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "diagnostic_router_value_gradient_mode": "live",
        "loss_profile": "one_third_two_thirds",
        "dataset_manifests": dataset_manifests,
        "initial_state_sha256": initial_state_hashes,
        "records_by_dataset": records_by_dataset,
        "decision": decision,
    }


def freeze_authorization(
    *,
    report_path: Path,
    authorization_path: Path,
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    report = _load_strict_json(report_path)
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("dataset_role") != "train"
        or report.get("test_loader_constructed") is not False
        or report.get("optimizer_constructed") is not False
        or report.get("optimizer_steps") != 0
        or report.get("rules_sha256") != _sha256_file(RULES_PATH)
        or report.get("source_manifest") != _source_manifest()
    ):
        raise ValueError("gradient diagnostic report authority differs")
    replayed = decide_authorized_mode(report["records_by_dataset"], rules)
    if report.get("decision") != replayed:
        raise ValueError("stored gradient decision does not replay")
    payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "frozen",
        "scope": "global_three_dataset_v33",
        "architecture": core.SBSC_V33_MODEL_NAME,
        "architecture_seed": 42,
        "run_seed": 42,
        "authorized_router_value_gradient_mode": replayed[
            "authorized_router_value_gradient_mode"
        ],
        "decision": replayed,
        "report_path": _relative_regular_path(report_path),
        "report_sha256": _sha256_file(report_path),
        "rules_path": _relative_regular_path(RULES_PATH),
        "rules_sha256": _sha256_file(RULES_PATH),
        "source_manifest_sha256": report["source_manifest"]["sha256"],
        "git_head": report["git_head"],
        "dataset_role": "train",
        "test_loader_constructed": False,
        "optimizer_steps": 0,
        "write_once": True,
    }
    _write_once_json(authorization_path, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=DEFAULT_AUTHORIZATION_PATH,
    )
    parser.add_argument("--freeze-authorization", action="store_true")
    parser.add_argument(
        "--freeze-existing-report-only",
        action="store_true",
        help="do not run the model; replay and freeze an existing report",
    )
    args = parser.parse_args(argv)
    if args.freeze_existing_report_only and not args.freeze_authorization:
        parser.error("--freeze-existing-report-only requires --freeze-authorization")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rules = load_rules()
    report_path = _validated_output_path(args.report)
    if not args.freeze_existing_report_only:
        dataset_root = args.dataset_root.resolve(strict=True)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostic requested but CUDA is unavailable")
        report = run_diagnostic(
            dataset_root=dataset_root,
            device=device,
            rules=rules,
        )
        report_sha = _write_once_json(report_path, report)
        print(
            json.dumps(
                {
                    "report": _relative_regular_path(report_path),
                    "report_sha256": report_sha,
                    "decision": report["decision"][
                        "authorized_router_value_gradient_mode"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    if args.freeze_authorization:
        authorization = freeze_authorization(
            report_path=report_path,
            authorization_path=args.authorization,
            rules=rules,
        )
        print(
            json.dumps(
                {
                    "authorization": _relative_regular_path(
                        _validated_output_path(args.authorization)
                    ),
                    "authorized_router_value_gradient_mode": authorization[
                        "authorized_router_value_gradient_mode"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
