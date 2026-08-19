#!/usr/bin/env python3
"""Unified validation-only legacy screen for the frozen IRSTD model designs.

The two allowed variants share the audited R1 transactional engine and the
zero-margin dual-role selector, but have independent schemas, identities,
output roots, and process locks.  Formal runs always retain the 1000-epoch
schedule; an external watcher may pause only after the atomic epoch-500
commit.  This module has no public-test dataset or evaluation entry point.
"""

from __future__ import annotations

import argparse
import builtins
import copy
import errno
import fcntl
import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from experiments import evisirst_zero_margin_selection as zero_selection
from experiments import irstd_psbfr_v1 as psbfr
from experiments import irstd_single_residual_v1 as single_residual
from experiments.evisirst_v2_data import DEFAULT_SPLIT_ROOT, V2SplitContract

import train_irstd_hf_decoder_v1 as hf_transaction
import train_validation_selected as r1


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "irstd_model_design"
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "IRSTD_PSBFR_V1_PROTOCOL.md"
RULES_PATH = PROJECT_ROOT / "experiments" / "irstd_psbfr_v1_screen_rules.json"

DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
FORMAL_EPOCHS = 1000
FORMAL_BATCH_SIZE = 16
FORMAL_WORKERS = 0
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_VAL_INTERVAL = 1
CANONICAL_SPLIT_ROOT = PROJECT_ROOT / "splits" / "v2"
CANONICAL_IRSTD_MANIFEST_SHA256 = (
    "5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596"
)
CANONICAL_IRSTD_DATA_TREE_SHA256 = (
    "ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30"
)
CANONICAL_IRSTD_TRAIN_COUNT = 640
CANONICAL_IRSTD_VAL_COUNT = 160

TRAINING_SCHEMA = "evisirst_irstd_model_design_screen_training/v1"
CANDIDATE_SCHEMA = "evisirst_irstd_model_design_screen_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_irstd_model_design_screen_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_irstd_model_design_screen_history/v1"
SELECTION_PAYLOAD_SCHEMA = "evisirst_irstd_model_design_screen_selection/v1"
SOURCE_SET_SCHEMA = "evisirst_irstd_model_design_screen_source_set/v1"
DETERMINISM_SCHEMA = "evisirst_irstd_model_design_screen_determinism/v1"
EXPERIMENT_SCHEMA = "evisirst_irstd_model_design_legacy_screen/v1"
PROMOTION_GATE_SCHEMA = "evisirst_irstd_model_design_screen_gate/v1"


class ModelDesignScreenError(r1.ValidationSelectedTrainingError):
    """The request violates the frozen model-design screen contract."""


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    allowed_run_seeds: tuple[int, ...]
    architecture_source: Path
    state_key_count: int
    parameter_count: int
    extension_state_key_count: int
    extension_state_prefix: str | None
    architecture_schema: str
    single_variable: str


VARIANTS = {
    "single_residual_v1": VariantSpec(
        key="single_residual_v1",
        label="EviSIRST-Single-Residual-v1",
        allowed_run_seeds=(42, 1446202191),
        architecture_source=Path(single_residual.__file__),
        state_key_count=564,
        parameter_count=10_870_130,
        extension_state_key_count=0,
        extension_state_prefix=None,
        architecture_schema=single_residual.ARCHITECTURE_SCHEMA,
        single_variable="remove_second_encoder_skip_addition_at_levels_1_to_4",
    ),
    "psbfr_v1": VariantSpec(
        key="psbfr_v1",
        label="EviSIRST-PSBFR-v1",
        allowed_run_seeds=(42, 1446202191, 104728269),
        architecture_source=Path(psbfr.__file__),
        state_key_count=569,
        parameter_count=10_871_203,
        extension_state_key_count=5,
        extension_state_prefix=psbfr.PSBFR_STATE_PREFIX,
        architecture_schema=psbfr.ARCHITECTURE_SCHEMA,
        single_variable="prediction_supported_bounded_frequency_logit_refinement",
    ),
}

ZERO_MARGIN_SELECTION = hf_transaction.ZERO_MARGIN_SELECTION
_R1_BUILD_DATASETS = hf_transaction._R1_BUILD_DATASETS
_R1_BUILD_FINAL_CHECKPOINT_PAYLOAD = hf_transaction._R1_BUILD_FINAL_CHECKPOINT_PAYLOAD
_R1_WRITE_JSON = hf_transaction._R1_WRITE_JSON
_R1_ATOMIC_TORCH_SAVE = hf_transaction._R1_ATOMIC_TORCH_SAVE
_BASE_R1_SOURCE_PROVENANCE = copy.deepcopy(
    hf_transaction._BASE_R1_SOURCE_PROVENANCE
)
_BASE_R1_DETERMINISM_IDENTITY = copy.deepcopy(
    hf_transaction._BASE_R1_DETERMINISM_IDENTITY
)
_HF_RUN = hf_transaction.run
_HF_ATOMIC_WRITE_DUAL_ROLE_FINALS = (
    hf_transaction._atomic_write_dual_role_finals
)

_PATCH_LOCK = threading.Lock()
_ACTIVE_VARIANT: str | None = None
_ACTIVE_SMOKE: bool | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelDesignScreenError("protocol metadata is not strict JSON") from exc


def _load_rules() -> dict[str, Any]:
    if RULES_PATH.is_symlink() or not RULES_PATH.is_file():
        raise ModelDesignScreenError("frozen screen rules are unavailable")
    try:
        rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelDesignScreenError("frozen screen rules are malformed") from exc
    if (
        not isinstance(rules, dict)
        or rules.get("schema") != "evisirst_irstd_psbfr_v1_rules/v1"
        or rules.get("protocol_relative_path")
        != "experiments/IRSTD_PSBFR_V1_PROTOCOL.md"
        or rules.get("execution", {}).get("configured_total_epochs")
        != FORMAL_EPOCHS
        or tuple(rules.get("d0_diagnostic", {}).get("run_seeds", ()))
        != VARIANTS["single_residual_v1"].allowed_run_seeds
        or tuple(rules.get("legacy_screen", {}).get("run_seeds", ()))
        != VARIANTS["psbfr_v1"].allowed_run_seeds
    ):
        raise ModelDesignScreenError("frozen screen rules identity differs")
    return rules


def _active_spec(variant: str | None = None) -> VariantSpec:
    key = variant if variant is not None else _ACTIVE_VARIANT
    try:
        return VARIANTS[key]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ModelDesignScreenError("no model-design variant is active") from exc


def _is_active_smoke() -> bool:
    if type(_ACTIVE_SMOKE) is not bool:
        raise ModelDesignScreenError("screen smoke/formal role is not active")
    return _ACTIVE_SMOKE


def promotion_gate(
    variant: str | None = None,
    *,
    smoke: bool | None = None,
) -> dict[str, Any]:
    spec = _active_spec(variant)
    is_smoke = _is_active_smoke() if smoke is None else smoke
    rules = _load_rules()
    section_name = "d0_diagnostic" if spec.key == "single_residual_v1" else "legacy_screen"
    section = rules[section_name]
    return {
        "schema": PROMOTION_GATE_SCHEMA,
        "status": "smoke_test_only_not_promotion_eligible" if is_smoke else "TBD",
        "variant": spec.key,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seeds": list(spec.allowed_run_seeds),
        "configured_total_epochs": FORMAL_EPOCHS,
        "operational_pause_after_atomic_epoch": 500,
        "screen_records_after_epoch_500_allowed": False,
        "selection_rule": zero_selection.RULE_VERSION,
        "selection_margin_raw": None,
        "mean_delta_mIoU_minimum": section["mean_delta_mIoU_minimum"],
        "positive_delta_required_for_every_pair": section[
            "positive_delta_required_for_every_pair"
        ],
        "safety_failure": _json_clone(section["safety_failure"]),
        "promotion_eligible": not is_smoke,
        "public_test_allowed": False,
        "test_split_accessed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument("--target-mode", choices=(TARGET_MODE,), default=TARGET_MODE)
    parser.add_argument(
        "--architecture-seed",
        type=int,
        choices=(ARCHITECTURE_SEED,),
        default=ARCHITECTURE_SEED,
    )
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-sample-level-fallback", action="store_true")
    parser.add_argument("--smoke-max-train-samples", type=int)
    parser.add_argument("--smoke-max-val-samples", type=int)
    args = parser.parse_args(argv)

    if args.run_seed not in VARIANTS[args.variant].allowed_run_seeds:
        parser.error(
            f"{args.variant} permits --run-seed only from "
            f"{VARIANTS[args.variant].allowed_run_seeds}"
        )
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for name in ("smoke_max_train_samples", "smoke_max_val_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    smoke = r1._is_smoke(args)
    if not smoke:
        if args.epochs != FORMAL_EPOCHS:
            parser.error(f"formal screen requires --epochs {FORMAL_EPOCHS}")
        if args.warmup_epochs not in (None, FORMAL_WARMUP_EPOCHS):
            parser.error(
                f"formal screen requires --warmup-epochs {FORMAL_WARMUP_EPOCHS}"
            )
        args.warmup_epochs = FORMAL_WARMUP_EPOCHS
    else:
        if args.warmup_epochs is None:
            args.warmup_epochs = min(FORMAL_WARMUP_EPOCHS, args.epochs)
        if not 0 <= args.warmup_epochs <= args.epochs:
            parser.error("smoke --warmup-epochs must be in [0, epochs]")

    args.output_root = DEFAULT_OUTPUT_ROOT / args.variant / "legacy_screen"
    args.batch_size = FORMAL_BATCH_SIZE
    args.workers = FORMAL_WORKERS
    args.base_lr = FORMAL_BASE_LR
    args.min_lr = FORMAL_MIN_LR
    args.val_interval = FORMAL_VAL_INTERVAL
    try:
        _require_variant_args(args)
    except ModelDesignScreenError as exc:
        parser.error(str(exc))
    return args


def _require_variant_args(args: argparse.Namespace) -> None:
    spec = _active_spec(args.variant)
    exact = {
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "val_interval": FORMAL_VAL_INTERVAL,
    }
    for name, expected in exact.items():
        if getattr(args, name, None) != expected:
            raise ModelDesignScreenError(f"screen freezes {name}={expected!r}")
    if args.run_seed not in spec.allowed_run_seeds:
        raise ModelDesignScreenError(f"run seed is not allowed for {spec.key}")
    expected_root = DEFAULT_OUTPUT_ROOT / spec.key / "legacy_screen"
    if Path(args.output_root).resolve() != expected_root.resolve():
        raise ModelDesignScreenError("screen output root differs")
    smoke = r1._is_smoke(args)
    if not smoke and (
        args.epochs != FORMAL_EPOCHS
        or args.warmup_epochs != FORMAL_WARMUP_EPOCHS
    ):
        raise ModelDesignScreenError("formal screen requires the 1000-epoch recipe")
    if not smoke and args.device != "cuda:0":
        raise ModelDesignScreenError(
            "formal screen uses logical cuda:0; map a free GPU with CUDA_VISIBLE_DEVICES"
        )
    if not smoke:
        supplied = Path(os.path.abspath(args.split_root))
        canonical = Path(os.path.abspath(CANONICAL_SPLIT_ROOT))
        if supplied != canonical:
            raise ModelDesignScreenError("formal screen requires repository splits/v2")
        manifest = canonical / DATASET / "manifest.json"
        if (
            canonical.is_symlink()
            or not canonical.is_dir()
            or manifest.is_symlink()
            or not manifest.is_file()
            or _sha256_file(manifest) != CANONICAL_IRSTD_MANIFEST_SHA256
        ):
            raise ModelDesignScreenError("canonical IRSTD split identity differs")
    _load_rules()


def _variant_source_provenance() -> dict[str, Any]:
    spec = _active_spec()
    baseline_files = _BASE_R1_SOURCE_PROVENANCE.get("files")
    if not isinstance(baseline_files, Mapping):
        raise ModelDesignScreenError("R1 source provenance is malformed")
    files = {
        f"r1/{name}": copy.deepcopy(entry)
        for name, entry in sorted(baseline_files.items())
    }
    additions = {
        "screen_runner": Path(__file__),
        "transaction_adapter": Path(hf_transaction.__file__),
        "selected_architecture": spec.architecture_source,
        "zero_margin_selector": Path(zero_selection.__file__),
        "frozen_protocol": PROTOCOL_PATH,
        "frozen_screen_rules": RULES_PATH,
    }
    for name, path in additions.items():
        if path.is_symlink() or not path.is_file():
            raise ModelDesignScreenError(f"{name} is not a regular source file")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise ModelDesignScreenError(f"{name} is outside repository") from exc
        files[name] = {"relative_path": relative, "sha256": _sha256_file(resolved)}
    return {
        "schema": SOURCE_SET_SCHEMA,
        "variant": spec.key,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


def _state_contract(spec: VariantSpec) -> dict[str, Any]:
    return {
        "architecture_schema": spec.architecture_schema,
        "architecture_source_relative_path": spec.architecture_source.resolve(
            strict=True
        ).relative_to(PROJECT_ROOT).as_posix(),
        "architecture_source_sha256": _sha256_file(spec.architecture_source),
        "base_state_key_count": 564,
        "extension_state_key_count": spec.extension_state_key_count,
        "total_state_key_count": spec.state_key_count,
        "total_parameter_count": spec.parameter_count,
        "extension_state_prefix": spec.extension_state_prefix,
        "d0_state_is_clean_load_compatible": spec.key == "single_residual_v1",
        "clean_architecture_interpretation_forbidden": (
            spec.key == "single_residual_v1"
        ),
        "required_forward_binding": (
            "single_residual_forward_with_relay"
            if spec.key == "single_residual_v1"
            else None
        ),
    }


def _variant_determinism_protocol_identity() -> dict[str, Any]:
    spec = _active_spec()
    source = _variant_source_provenance()
    identity = copy.deepcopy(_BASE_R1_DETERMINISM_IDENTITY)
    identity["schema"] = DETERMINISM_SCHEMA
    identity["architecture"] = {
        "variant": spec.key,
        "name": spec.label,
        "initialization": "full_model_scratch",
        "single_variable_vs_clean_R1": spec.single_variable,
        "all_parameters_trainable": True,
        "state_contract": _state_contract(spec),
    }
    identity["training_data"]["crop_policy"] = "clean_R1_unchanged"
    identity["selection"] = {
        "rule_version": zero_selection.RULE_VERSION,
        "primary_role": zero_selection.PRIMARY_ROLE,
        "secondary_role": zero_selection.SECONDARY_ROLE,
        "margin_raw": None,
        "window_applied": False,
        "strict_complete_key": True,
    }
    identity["schedule"] = {
        "configured_total_epochs": FORMAL_EPOCHS,
        "operational_pause_after_atomic_epoch": 500,
        "runner_contains_500_epoch_schedule": False,
    }
    identity["test_access"] = {"supported": False, "test_split_accessed": False}
    identity["source_set_schema"] = source["schema"]
    identity["source_files"] = source["files"]
    identity["source_tree_sha256"] = source["source_tree_sha256"]
    return _json_clone(identity)


def build_datasets(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    _require_variant_args(args)
    train_dataset, val_dataset, grouping = _R1_BUILD_DATASETS(args)
    contract = train_dataset.contract
    if contract is not val_dataset.contract and (
        contract.manifest_sha256 != val_dataset.contract.manifest_sha256
        or contract.data_tree_sha256 != val_dataset.contract.data_tree_sha256
    ):
        raise ModelDesignScreenError("train/validation contracts differ")
    if not r1._is_smoke(args) and (
        contract.manifest_sha256 != CANONICAL_IRSTD_MANIFEST_SHA256
        or contract.data_tree_sha256 != CANONICAL_IRSTD_DATA_TREE_SHA256
        or len(train_dataset) != CANONICAL_IRSTD_TRAIN_COUNT
        or len(val_dataset) != CANONICAL_IRSTD_VAL_COUNT
    ):
        raise ModelDesignScreenError("formal 640/160 split identity differs")
    return train_dataset, val_dataset, grouping


def _variant_run_identity(
    args: argparse.Namespace,
    contract: V2SplitContract,
    grouping: Mapping[str, Any],
    *,
    train_count: int,
    val_count: int,
    smoke: bool,
) -> dict[str, Any]:
    _require_variant_args(args)
    spec = _active_spec(args.variant)
    if not smoke and (
        contract.manifest_sha256 != CANONICAL_IRSTD_MANIFEST_SHA256
        or contract.data_tree_sha256 != CANONICAL_IRSTD_DATA_TREE_SHA256
        or train_count != CANONICAL_IRSTD_TRAIN_COUNT
        or val_count != CANONICAL_IRSTD_VAL_COUNT
    ):
        raise ModelDesignScreenError("formal run identity differs from frozen split")
    identity = {
        "schema": TRAINING_SCHEMA + "/run_identity",
        "model": "EviSIRST",
        "architecture_variant": spec.label,
        "variant_key": spec.key,
        "experiment": {
            "schema": EXPERIMENT_SCHEMA,
            "status": "smoke_test_only" if smoke else "legacy_dev_validation_only",
            "single_variable_vs_clean_R1": spec.single_variable,
            "protocol_relative_path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "rules_relative_path": RULES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "dataset": DATASET,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": args.run_seed,
        "target_mode": TARGET_MODE,
        "epochs": args.epochs,
        "formal_configured_total_epochs": FORMAL_EPOCHS,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "val_interval": args.val_interval,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "optimizer_hyperparameters": _json_clone(r1.R1_ADAM_GROUP_HYPERPARAMETERS),
        "optimizer_parameter_groups": 1,
        "all_parameters_trainable": True,
        "initialization": "full_model_scratch",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "deep_supervision_probability_heads": 6,
        "deep_supervision_weights": [1.0] * 6,
        "training_crop": "clean_R1_unchanged",
        "stacked_complete_target_crop": False,
        "stacked_loss_change": False,
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "evaluation_head": "out",
        "selection_rule": zero_selection.RULE_VERSION,
        "selection_margin_raw": None,
        "selection_window_applied": False,
        "selection_roles": list(zero_selection.VALID_ROLES),
        "determinism_protocol": _variant_determinism_protocol_identity(),
        "state_contract": _state_contract(spec),
        "manifest_sha256": contract.manifest_sha256,
        "split_seed": contract.manifest["seeds"]["split_seed"],
        "data_tree_sha256": contract.data_tree_sha256,
        "canonical_split_contract": {
            "split_root_relative_path": "splits/v2",
            "manifest_sha256": CANONICAL_IRSTD_MANIFEST_SHA256,
            "data_tree_sha256": CANONICAL_IRSTD_DATA_TREE_SHA256,
            "train_count": CANONICAL_IRSTD_TRAIN_COUNT,
            "val_count": CANONICAL_IRSTD_VAL_COUNT,
        },
        "grouping_policy": dict(grouping),
        "train_count": train_count,
        "val_count": val_count,
        "smoke": smoke,
        "smoke_max_train_samples": args.smoke_max_train_samples,
        "smoke_max_val_samples": args.smoke_max_val_samples,
        "promotion_gate": promotion_gate(spec.key, smoke=smoke),
        "promotion_eligible": not smoke,
        "public_test_supported": False,
        "test_split_accessed": False,
    }
    normalized = _json_clone(identity)
    normalized["identity_sha256"] = _canonical_sha256(normalized)
    return normalized


def _validate_state_dict(
    value: Any,
    expected: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    spec = _active_spec()
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ModelDesignScreenError("checkpoint state keys differ")
    keys = set(expected)
    extension_keys = (
        {key for key in keys if key.startswith(spec.extension_state_prefix)}
        if spec.extension_state_prefix is not None
        else set()
    )
    if (
        builtins.len(keys) != spec.state_key_count
        or builtins.len(extension_keys) != spec.extension_state_key_count
        or any(key.startswith("target_survival") for key in keys)
    ):
        raise ModelDesignScreenError(
            f"model is not the frozen {spec.state_key_count}-key {spec.key} graph"
        )
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ModelDesignScreenError("checkpoint state is malformed")
        reference = expected[key]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ModelDesignScreenError(f"tensor contract differs for {key!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ModelDesignScreenError(f"checkpoint tensor is non-finite: {key!r}")
        state[key] = tensor.detach().cpu()
    return state


def _variant_cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    raw = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return _validate_state_dict(raw, model.state_dict())


def _variant_initialize_evisirst(
    dataset: str,
    *,
    seed: int = ARCHITECTURE_SEED,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    spec = _active_spec()
    if dataset != DATASET or seed != ARCHITECTURE_SEED or training is not True:
        raise ModelDesignScreenError("formal constructor contract differs")
    if spec.key == "single_residual_v1":
        model, metadata = single_residual.build_irstd_single_residual_v1(
            dataset, seed=seed, training=True
        )
        manifest = single_residual.validate_irstd_single_residual_v1(model)
        if (
            manifest.get("schema") != spec.architecture_schema
            or manifest.get("only_graph_change")
            != "remove_second_skip_addition_at_levels_1_to_4"
        ):
            raise ModelDesignScreenError("D0 architecture manifest differs")
    else:
        model, metadata = psbfr.build_irstd_psbfr_v1(
            dataset, seed=seed, training=True
        )
        manifest = psbfr.validate_irstd_psbfr_v1(
            model,
            require_identity_initialization=True,
            require_all_trainable=True,
        )
    if (
        len(model.state_dict()) != spec.state_key_count
        or sum(parameter.numel() for parameter in model.parameters())
        != spec.parameter_count
        or any(not parameter.requires_grad for parameter in model.parameters())
    ):
        raise ModelDesignScreenError("constructed architecture contract differs")
    _validate_state_dict(model.state_dict(), model.state_dict())
    ready = dict(metadata)
    ready.update(
        {
            "public_model": spec.label,
            "variant_key": spec.key,
            "architecture_manifest": _json_clone(manifest),
            "architecture_manifest_sha256": _canonical_sha256(manifest),
            "full_model_scratch": True,
            "all_parameters_trainable": True,
            "state_key_count": spec.state_key_count,
            "parameter_count": spec.parameter_count,
            "clean_architecture_interpretation_forbidden": (
                spec.key == "single_residual_v1"
            ),
            "public_test_supported": False,
        }
    )
    return model, ready


def _variant_len(value: Any) -> int:
    """Bridge R1's single legacy 564-key constructor assertion."""

    observed = builtins.len(value)
    spec = _active_spec()
    if isinstance(value, Mapping) and observed == spec.state_key_count:
        if spec.extension_state_prefix is None:
            return 564
        count = sum(
            isinstance(key, str) and key.startswith(spec.extension_state_prefix)
            for key in value
        )
        if count == spec.extension_state_key_count:
            return 564
    return observed


def _variant_final_checkpoint_payload(**kwargs: Any) -> dict[str, Any]:
    spec = _active_spec()
    state = _validate_state_dict(kwargs.get("state_dict"), kwargs.get("state_dict"))
    adjusted = dict(kwargs)
    adjusted["state_dict"] = state
    payload = _R1_BUILD_FINAL_CHECKPOINT_PAYLOAD(**adjusted)
    identity = kwargs.get("identity")
    if not isinstance(identity, Mapping):
        raise ModelDesignScreenError("final training identity is missing")
    payload.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": spec.label,
            "checkpoint_role": "legacy_validation_selected_best_mIoU",
            "experiment_schema": EXPERIMENT_SCHEMA,
            "experiment_status": (
                "smoke_test_only" if _is_active_smoke() else "legacy_dev_validation_only"
            ),
            "architecture_variant": spec.label,
            "variant_key": spec.key,
            "state_contract": _json_clone(identity.get("state_contract")),
            "selection_margin_raw": None,
            "selection_window_applied": False,
            "promotion_gate": promotion_gate(),
            "promotion_eligible": not _is_active_smoke(),
            "public_test_supported": False,
            "test_split_accessed": False,
        }
    )
    return payload


def _variant_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    spec = _active_spec()
    output = dict(payload)
    if output.get("schema") in {HISTORY_SCHEMA, TRAINING_SCHEMA + "/summary"}:
        output.update(
            {
                "experiment_schema": EXPERIMENT_SCHEMA,
                "architecture_variant": spec.label,
                "variant_key": spec.key,
                "selection_margin_raw": None,
                "selection_window_applied": False,
                "promotion_eligible": not _is_active_smoke(),
                "public_test_supported": False,
                "test_split_accessed": False,
            }
        )
    if output.get("schema") == TRAINING_SCHEMA + "/summary":
        history = output.get("validation_history")
        selection_payload = output.get("selection")
        if not isinstance(history, list) or not isinstance(selection_payload, Mapping):
            raise ModelDesignScreenError("final summary evidence is incomplete")
        selected_epoch = selection_payload.get("selected_epoch")
        records = [
            record
            for record in history
            if isinstance(record, Mapping) and record.get("epoch") == selected_epoch
        ]
        if len(records) != 1:
            raise ModelDesignScreenError("selected validation record is ambiguous")
        provenance = selection_payload.get("selection_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("rule_version") != zero_selection.RULE_VERSION
            or provenance.get("window_applied") is not False
        ):
            raise ModelDesignScreenError("selector is not strict zero-margin")
        output["selected_validation_record"] = _json_clone(records[0])
        output["selected_validation_record_sha256"] = _canonical_sha256(records[0])
        output["selection_roles"] = _json_clone(provenance.get("roles"))
        output["promotion_gate"] = promotion_gate()
    _R1_WRITE_JSON(path, output)


def _variant_atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    spec = _active_spec()
    output = dict(payload)
    if output.get("schema") in {TRAINING_SCHEMA, CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA}:
        output.update(
            {
                "experiment_schema": EXPERIMENT_SCHEMA,
                "architecture_variant": spec.label,
                "variant_key": spec.key,
                "selection_margin_raw": None,
                "selection_window_applied": False,
                "promotion_eligible": not _is_active_smoke(),
                "public_test_supported": False,
                "test_split_accessed": False,
            }
        )
    _R1_ATOMIC_TORCH_SAVE(path, output)


@contextmanager
def _variant_runtime() -> Iterator[None]:
    overrides = {
        "TRAINING_SCHEMA": TRAINING_SCHEMA,
        "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "HISTORY_SCHEMA": HISTORY_SCHEMA,
        "SELECTION_PAYLOAD_SCHEMA": SELECTION_PAYLOAD_SCHEMA,
        "selection": ZERO_MARGIN_SELECTION,
        "build_datasets": build_datasets,
        "initialize_evisirst": _variant_initialize_evisirst,
        "_protocol_source_provenance": _variant_source_provenance,
        "_determinism_protocol_identity": _variant_determinism_protocol_identity,
        "_run_identity": _variant_run_identity,
        "_validate_state_dict": _validate_state_dict,
        "build_final_checkpoint_payload": _variant_final_checkpoint_payload,
        "_write_json": _variant_write_json,
        "_atomic_torch_save": _variant_atomic_torch_save,
        "len": _variant_len,
    }
    previous = {
        name: (hasattr(r1, name), getattr(r1, name, None)) for name in overrides
    }
    previous_cpu_state = r1.legacy_train._cpu_state
    try:
        for name, value in overrides.items():
            setattr(r1, name, value)
        r1.legacy_train._cpu_state = _variant_cpu_state
        yield
    finally:
        r1.legacy_train._cpu_state = previous_cpu_state
        for name, (existed, value) in previous.items():
            if existed:
                setattr(r1, name, value)
            elif hasattr(r1, name):
                delattr(r1, name)


@contextmanager
def _run_process_lock(args: argparse.Namespace) -> Iterator[Path]:
    paths = resolve_run_paths(args)
    run_dir = paths["run_dir"]
    if not isinstance(run_dir, Path):
        raise ModelDesignScreenError("run directory is malformed")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".model_design_screen_v1.lock"
    if lock_path.is_symlink():
        raise ModelDesignScreenError("run lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ModelDesignScreenError(
                    f"model-design run is already locked: {run_dir}"
                ) from exc
            raise
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path | bool]:
    _require_variant_args(args)
    paths = dict(r1.resolve_run_paths(args))
    run_dir = paths["run_dir"]
    if not isinstance(run_dir, Path):
        raise ModelDesignScreenError("run directory is malformed")
    paths["best_mIoU_final"] = run_dir / "EviSIRST_best_mIoU.pth.tar"
    paths["best_Pd_final"] = run_dir / "EviSIRST_best_Pd.pth.tar"
    expected_root = DEFAULT_OUTPUT_ROOT / args.variant / "legacy_screen"
    try:
        relative = run_dir.relative_to(expected_root)
    except ValueError as exc:
        raise ModelDesignScreenError("resolved run directory escapes output root") from exc
    if not r1._is_smoke(args):
        expected = Path("formal") / DATASET / TARGET_MODE / f"run_seed_{args.run_seed}"
        if relative != expected:
            raise ModelDesignScreenError("formal output path differs")
    return paths


def _atomic_write_dual_role_finals(**kwargs: Any) -> dict[str, Any]:
    """Use the audited writer, then reject its obsolete HF model label.

    The transaction implementation is reused verbatim.  Its payload template
    has one historical model-name literal, so this adapter stages into an
    isolated temporary call, rewrites both files atomically with the active
    identity, and recomputes their hashes before any summary can bind them.
    """

    spec = _active_spec()
    artifacts = _HF_ATOMIC_WRITE_DUAL_ROLE_FINALS(**kwargs)
    run_dir = kwargs.get("run_dir")
    if not isinstance(run_dir, Path):
        raise ModelDesignScreenError("dual-role run directory is malformed")
    normalized: dict[str, Any] = {}
    for role, evidence in artifacts.items():
        if not isinstance(evidence, Mapping):
            raise ModelDesignScreenError("dual-role evidence is malformed")
        path = run_dir / str(evidence.get("relative_path"))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ModelDesignScreenError("dual-role checkpoint is malformed")
        required = {
            "model": spec.label,
            "architecture_variant": spec.label,
            "variant_key": spec.key,
            "experiment_schema": EXPERIMENT_SCHEMA,
            "promotion_eligible": not _is_active_smoke(),
            "public_test_supported": False,
            "test_split_accessed": False,
        }
        updated = dict(payload)
        updated.update(required)
        if all(payload.get(name) == value for name, value in required.items()):
            item = _json_clone(dict(evidence))
            item["sha256"] = _sha256_file(path)
            normalized[role] = item
            continue
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=run_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(updated, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        item = _json_clone(dict(evidence))
        item["sha256"] = _sha256_file(path)
        normalized[role] = item
    return normalized


@contextmanager
def _hf_adapter(args: argparse.Namespace) -> Iterator[None]:
    global _ACTIVE_VARIANT, _ACTIVE_SMOKE
    if not _PATCH_LOCK.acquire(blocking=False):
        raise ModelDesignScreenError("model-design adapter is already active")
    _ACTIVE_VARIANT = args.variant
    _ACTIVE_SMOKE = r1._is_smoke(args)
    spec = _active_spec()
    overrides = {
        "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT / spec.key / "legacy_screen",
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "TRAINING_SCHEMA": TRAINING_SCHEMA,
        "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "HISTORY_SCHEMA": HISTORY_SCHEMA,
        "SELECTION_PAYLOAD_SCHEMA": SELECTION_PAYLOAD_SCHEMA,
        "SOURCE_SET_SCHEMA": SOURCE_SET_SCHEMA,
        "DETERMINISM_SCHEMA": DETERMINISM_SCHEMA,
        "EXPERIMENT_SCHEMA": EXPERIMENT_SCHEMA,
        "PROMOTION_GATE_SCHEMA": PROMOTION_GATE_SCHEMA,
        "HFDecoderRunnerError": ModelDesignScreenError,
        "promotion_gate": promotion_gate,
        "_require_variant_args": _require_variant_args,
        "_variant_source_provenance": _variant_source_provenance,
        "_variant_determinism_protocol_identity": _variant_determinism_protocol_identity,
        "build_datasets": build_datasets,
        "_variant_run_identity": _variant_run_identity,
        "_validate_state_dict": _validate_state_dict,
        "_variant_cpu_state": _variant_cpu_state,
        "_variant_initialize_evisirst": _variant_initialize_evisirst,
        "_variant_len": _variant_len,
        "_variant_final_checkpoint_payload": _variant_final_checkpoint_payload,
        "_variant_write_json": _variant_write_json,
        "_variant_atomic_torch_save": _variant_atomic_torch_save,
        "_variant_runtime": _variant_runtime,
        "_run_process_lock": _run_process_lock,
        "resolve_run_paths": resolve_run_paths,
        "_atomic_write_dual_role_finals": _atomic_write_dual_role_finals,
    }
    previous = {name: getattr(hf_transaction, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(hf_transaction, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(hf_transaction, name, value)
        _ACTIVE_VARIANT = None
        _ACTIVE_SMOKE = None
        _PATCH_LOCK.release()


def run(args: argparse.Namespace) -> Path:
    """Execute the selected design through the audited R1 transaction."""

    _require_variant_args(args)
    with _hf_adapter(args):
        return _HF_RUN(args)


def main(argv: list[str] | None = None) -> None:
    checkpoint = run(parse_args(argv))
    print(checkpoint)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "DATASET",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAL_EPOCHS",
    "HISTORY_SCHEMA",
    "ModelDesignScreenError",
    "RULES_PATH",
    "SELECTION_PAYLOAD_SCHEMA",
    "TARGET_MODE",
    "TRAINING_SCHEMA",
    "VARIANTS",
    "build_datasets",
    "parse_args",
    "promotion_gate",
    "resolve_run_paths",
    "run",
]
