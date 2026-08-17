#!/usr/bin/env python3
"""Run the preregistered IRSTD-1K HF-decoder Stage-A validation pilot.

This entry point is a bounded adapter around ``train_validation_selected``.
It preserves that runner's train/validation transaction and resume engine but
replaces the clean constructor with the 572-key HF-decoder graph and replaces
the historical 0.001-window selector with an exact, zero-margin dual-role
selector.  Public test data is deliberately unreachable from this module.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from experiments import evisirst_zero_margin_selection as zero_selection
from experiments import irstd_hf_decoder_v1 as hf_decoder
from experiments.evisirst_v2_data import DEFAULT_SPLIT_ROOT, V2SplitContract

import train_validation_selected as r1


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "irstd_performance" / "hf_decoder_v1"
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "IRSTD_HF_DECODER_V1_PROTOCOL.md"

DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
PAIRED_RUN_SEED = 1446202191
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

TRAINING_SCHEMA = "evisirst_irstd_hf_decoder_training/v1"
CANDIDATE_SCHEMA = "evisirst_irstd_hf_decoder_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_irstd_hf_decoder_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_irstd_hf_decoder_history/v1"
SELECTION_PAYLOAD_SCHEMA = "evisirst_irstd_hf_decoder_selection_payload/v1"
SOURCE_SET_SCHEMA = "evisirst_irstd_hf_decoder_source_set/v1"
DETERMINISM_SCHEMA = "evisirst_irstd_hf_decoder_determinism/v1"
EXPERIMENT_SCHEMA = "evisirst_irstd_hf_decoder_experiment/v1"
PROMOTION_GATE_SCHEMA = "evisirst_irstd_hf_decoder_promotion_gate/v1"


class HFDecoderRunnerError(r1.ValidationSelectedTrainingError):
    """The requested run violates the frozen HF-decoder-v1 contract."""


# Capture trusted engine entry points before the runtime adapter is installed.
_R1_PROTOCOL_SOURCE_PROVENANCE = r1._protocol_source_provenance
_R1_DETERMINISM_PROTOCOL_IDENTITY = r1._determinism_protocol_identity
_R1_BUILD_DATASETS = r1.build_datasets
_R1_BUILD_FINAL_CHECKPOINT_PAYLOAD = r1.build_final_checkpoint_payload
_R1_WRITE_JSON = r1._write_json
_R1_ATOMIC_TORCH_SAVE = r1._atomic_torch_save
_R1_TRAINING_SCHEMA = r1.TRAINING_SCHEMA
_BASE_R1_SOURCE_PROVENANCE = _R1_PROTOCOL_SOURCE_PROVENANCE()
_BASE_R1_DETERMINISM_IDENTITY = _R1_DETERMINISM_PROTOCOL_IDENTITY()
_PATCH_LOCK = threading.Lock()


class _ZeroMarginSelectionAdapter(SimpleNamespace):
    """Expose the exact selector through the interface consumed by R1."""

    __file__ = zero_selection.__file__
    INDEPENDENT_RULE_VERSION = zero_selection.RULE_VERSION
    MIOU_CANDIDATE_TOLERANCE = None

    @staticmethod
    def select_independent_checkpoint(records: Any) -> dict[str, Any]:
        return zero_selection.select_checkpoints(
            records,
            primary_role=zero_selection.PRIMARY_ROLE,
            margin=None,
        )

    @staticmethod
    def retention_frontier_epochs(records: Any) -> tuple[int, ...]:
        return zero_selection.retention_frontier_epochs(
            records,
            retain_roles=zero_selection.VALID_ROLES,
            margin=None,
        )


ZERO_MARGIN_SELECTION = _ZeroMarginSelectionAdapter()


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
        raise HFDecoderRunnerError("protocol metadata is not strict JSON") from exc


def promotion_gate() -> dict[str, Any]:
    """Return the preregistered zero-threshold paired-validation gate."""

    return {
        "schema": PROMOTION_GATE_SCHEMA,
        "status": "TBD",
        "hypotheses": {
            "H0": "variant_selected_validation_mIoU <= paired_control",
            "H1": "variant_selected_validation_mIoU > paired_control",
        },
        "paired_control": {
            "schema": _R1_TRAINING_SCHEMA,
            "dataset": DATASET,
            "target_mode": TARGET_MODE,
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": PAIRED_RUN_SEED,
            "summary_relative_path": (
                "runs/validation_selected/formal/IRSTD-1K/binary/"
                "run_seed_1446202191/summary.json"
            ),
            "comparison_source": "complete_saved_validation_history",
            "offline_reselection_rule": zero_selection.RULE_VERSION,
            "historical_window_summary_selected_is_comparator": False,
            "control_completion_required": True,
            "zero_margin_selected_epoch": "TBD_after_control_completion",
            "selected_validation_metrics": "TBD",
        },
        "complete_target_comparator": {
            "summary_relative_path": (
                "runs/irstd_performance/complete_target_v1/formal/IRSTD-1K/"
                "binary/run_seed_1446202191/summary.json"
            ),
            "comparison_source": "complete_saved_validation_history",
            "offline_reselection_rule": zero_selection.RULE_VERSION,
            "historical_window_summary_selected_is_comparator": False,
            "selected_validation_metrics": "TBD",
        },
        "variant": {
            "schema": TRAINING_SCHEMA,
            "selected_validation_metrics": "TBD",
        },
        "primary": {
            "metric": "selected_validation_mIoU",
            "comparison": "variant_minus_paired_control_raw",
            "operator": ">",
            "minimum_delta": 0.0,
            "selection_margin_raw": None,
            "baseline_value": "TBD",
            "variant_value": "TBD",
            "delta": "TBD",
            "passed": "TBD",
        },
        "reporting": {
            "complete_metric_key_required": [
                "mIoU",
                "nIoU",
                "F1",
                "Pd",
                "Fa",
                "tinyPd",
                "loss",
                "epoch",
            ],
            "best_Pd_is_secondary_operating_point": True,
        },
        "decision": {
            "GO_1": "strict raw mIoU improvement over reselected clean history",
            "GO_2": (
                "strict raw mIoU improvement over reselected complete-target history"
            ),
            "STOP": "zero or negative raw selected-validation mIoU change",
            "single_seed_screen_only": True,
            "public_test_allowed": False,
            "test_split_accessed": False,
            "result": "TBD",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
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

    try:
        r1.require_run_seed(args.run_seed)
    except r1.ValidationSelectedTrainingError as exc:
        parser.error(str(exc))
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for name in ("smoke_max_train_samples", "smoke_max_val_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    smoke = r1._is_smoke(args)
    if not smoke:
        if args.run_seed != PAIRED_RUN_SEED:
            parser.error(
                f"formal HF-decoder-v1 requires --run-seed {PAIRED_RUN_SEED}"
            )
        if args.epochs != FORMAL_EPOCHS:
            parser.error(f"formal HF-decoder-v1 requires --epochs {FORMAL_EPOCHS}")
        if args.warmup_epochs not in (None, FORMAL_WARMUP_EPOCHS):
            parser.error(
                "formal HF-decoder-v1 requires --warmup-epochs "
                f"{FORMAL_WARMUP_EPOCHS}"
            )
        args.warmup_epochs = FORMAL_WARMUP_EPOCHS
    else:
        if args.warmup_epochs is None:
            args.warmup_epochs = min(FORMAL_WARMUP_EPOCHS, args.epochs)
        if not 0 <= args.warmup_epochs <= args.epochs:
            parser.error("smoke --warmup-epochs must be in [0, epochs]")

    # Deliberately absent from the CLI: these are immutable comparison fields.
    args.output_root = DEFAULT_OUTPUT_ROOT
    args.batch_size = FORMAL_BATCH_SIZE
    args.workers = FORMAL_WORKERS
    args.base_lr = FORMAL_BASE_LR
    args.min_lr = FORMAL_MIN_LR
    args.val_interval = FORMAL_VAL_INTERVAL
    try:
        _require_variant_args(args)
    except HFDecoderRunnerError as exc:
        parser.error(str(exc))
    return args


def _require_variant_args(args: argparse.Namespace) -> None:
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
            raise HFDecoderRunnerError(
                f"HF-decoder-v1 freezes {name}={expected!r}"
            )
    if Path(args.output_root).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise HFDecoderRunnerError(
            "HF-decoder-v1 output is fixed under "
            "runs/irstd_performance/hf_decoder_v1"
        )
    smoke = r1._is_smoke(args)
    if not smoke and (
        args.run_seed != PAIRED_RUN_SEED
        or args.epochs != FORMAL_EPOCHS
        or args.warmup_epochs != FORMAL_WARMUP_EPOCHS
    ):
        raise HFDecoderRunnerError(
            "formal HF-decoder-v1 requires the paired seed and 1000-epoch recipe"
        )
    if not smoke and args.device != "cuda:0":
        raise HFDecoderRunnerError(
            "formal HF-decoder-v1 uses logical cuda:0; map a free physical GPU "
            "with CUDA_VISIBLE_DEVICES"
        )
    if not smoke:
        supplied = Path(os.path.abspath(args.split_root))
        canonical = Path(os.path.abspath(CANONICAL_SPLIT_ROOT))
        if supplied != canonical:
            raise HFDecoderRunnerError(
                "formal HF-decoder-v1 requires repository splits/v2"
            )
        if canonical.is_symlink() or not canonical.is_dir():
            raise HFDecoderRunnerError("canonical splits/v2 is not a real directory")
        manifest = canonical / DATASET / "manifest.json"
        if (
            manifest.is_symlink()
            or not manifest.is_file()
            or _sha256_file(manifest) != CANONICAL_IRSTD_MANIFEST_SHA256
        ):
            raise HFDecoderRunnerError("canonical IRSTD-1K manifest SHA-256 differs")


def _variant_source_provenance() -> dict[str, Any]:
    baseline_files = _BASE_R1_SOURCE_PROVENANCE.get("files")
    if not isinstance(baseline_files, Mapping):
        raise HFDecoderRunnerError("R1 source provenance is malformed")
    files = {
        f"r1/{name}": copy.deepcopy(entry)
        for name, entry in sorted(baseline_files.items())
    }
    additions = {
        "variant_runner": Path(__file__),
        "hf_decoder_architecture": Path(hf_decoder.__file__),
        "zero_margin_selector": Path(zero_selection.__file__),
        "frozen_protocol": PROTOCOL_PATH,
    }
    for name, path in additions.items():
        if path.is_symlink() or not path.is_file():
            raise HFDecoderRunnerError(f"{name} is not a regular source file")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise HFDecoderRunnerError(f"{name} is outside the repository") from exc
        files[name] = {
            "relative_path": relative,
            "sha256": _sha256_file(resolved),
        }
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


def _variant_determinism_protocol_identity() -> dict[str, Any]:
    identity = copy.deepcopy(_BASE_R1_DETERMINISM_IDENTITY)
    source = _variant_source_provenance()
    identity["schema"] = DETERMINISM_SCHEMA
    identity["architecture"] = {
        "name": "IRSTD-HF-Decoder-v1",
        "initialization": "full_model_scratch_identity_starting_hf",
        "single_variable_vs_R1": "final_decoder_HF_residual_only",
        "all_base_and_hf_parameters_trainable": True,
        "state_key_count": hf_decoder.FORMAL_STATE_KEY_COUNT,
        "hf_state_prefix": hf_decoder.HF_DECODER_STATE_PREFIX,
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
    identity["test_access"] = {
        "supported": False,
        "test_split_accessed": False,
    }
    identity["source_set_schema"] = source["schema"]
    identity["source_files"] = source["files"]
    identity["source_tree_sha256"] = source["source_tree_sha256"]
    return _json_clone(identity)


def build_datasets(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Construct only the unchanged clean R1 train and validation datasets."""

    _require_variant_args(args)
    train_dataset, val_dataset, grouping = _R1_BUILD_DATASETS(args)
    contract = train_dataset.contract
    if contract is not val_dataset.contract and (
        contract.manifest_sha256 != val_dataset.contract.manifest_sha256
        or contract.data_tree_sha256 != val_dataset.contract.data_tree_sha256
    ):
        raise HFDecoderRunnerError("train/validation contracts differ")
    if not r1._is_smoke(args) and (
        contract.manifest_sha256 != CANONICAL_IRSTD_MANIFEST_SHA256
        or contract.data_tree_sha256 != CANONICAL_IRSTD_DATA_TREE_SHA256
        or len(train_dataset) != CANONICAL_IRSTD_TRAIN_COUNT
        or len(val_dataset) != CANONICAL_IRSTD_VAL_COUNT
    ):
        raise HFDecoderRunnerError(
            "formal IRSTD-1K manifest/data-tree/count identity differs"
        )
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
    if not smoke and (
        contract.manifest_sha256 != CANONICAL_IRSTD_MANIFEST_SHA256
        or contract.data_tree_sha256 != CANONICAL_IRSTD_DATA_TREE_SHA256
        or train_count != CANONICAL_IRSTD_TRAIN_COUNT
        or val_count != CANONICAL_IRSTD_VAL_COUNT
    ):
        raise HFDecoderRunnerError("formal run identity differs from frozen split")
    identity = {
        "schema": TRAINING_SCHEMA + "/run_identity",
        # Keep the trusted R1 optimizer's inactive-base-parameter contract;
        # architecture_variant below disambiguates the actual 572-key graph.
        "model": "EviSIRST",
        "architecture_variant": "IRSTD-HF-Decoder-v1",
        "experiment": {
            "schema": EXPERIMENT_SCHEMA,
            "status": "experimental_validation_only",
            "single_variable_vs_R1": "final_decoder_HF_residual_only",
            "protocol_relative_path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "dataset": DATASET,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": args.run_seed,
        "target_mode": TARGET_MODE,
        "epochs": args.epochs,
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
        "all_base_and_hf_parameters_trainable": True,
        "initialization": "full_model_scratch",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "deep_supervision_probability_heads": 6,
        "deep_supervision_weights": [1.0] * 6,
        "training_crop": "clean_R1_unchanged",
        "stacked_complete_target_crop": False,
        "stacked_loss_change": False,
        "stacked_center_head": False,
        "evaluation": "evisirst_common_evaluate_model_out_head/v1",
        "evaluation_head": "out",
        "selection_rule": zero_selection.RULE_VERSION,
        "selection_margin_raw": None,
        "selection_window_applied": False,
        "selection_roles": list(zero_selection.VALID_ROLES),
        "determinism_protocol": _variant_determinism_protocol_identity(),
        "state_contract": {
            "base_state_key_count": hf_decoder.BASE_STATE_KEY_COUNT,
            "hf_state_key_count": hf_decoder.HF_DECODER_STATE_KEY_COUNT,
            "total_state_key_count": hf_decoder.FORMAL_STATE_KEY_COUNT,
            "hf_state_prefix": hf_decoder.HF_DECODER_STATE_PREFIX,
        },
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
        "promotion_gate": promotion_gate(),
        "public_test_supported": False,
        "test_split_accessed": False,
    }
    normalized = _json_clone(identity)
    normalized["identity_sha256"] = _canonical_sha256(normalized)
    return normalized


def _validate_state_dict(
    value: Any, expected: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Validate the exact 564-base + 8-HF tensor contract."""

    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise HFDecoderRunnerError("HF checkpoint state keys differ")
    expected_keys = set(expected)
    hf_keys = {
        key for key in expected_keys if key.startswith(hf_decoder.HF_DECODER_STATE_PREFIX)
    }
    base_keys = expected_keys - hf_keys
    if (
        builtins.len(expected_keys) != hf_decoder.FORMAL_STATE_KEY_COUNT
        or builtins.len(base_keys) != hf_decoder.BASE_STATE_KEY_COUNT
        or builtins.len(hf_keys) != hf_decoder.HF_DECODER_STATE_KEY_COUNT
        or any(key.startswith("target_survival") for key in expected_keys)
    ):
        raise HFDecoderRunnerError("model is not the frozen 572-key HF graph")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise HFDecoderRunnerError("HF checkpoint state is malformed")
        reference = expected[key]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise HFDecoderRunnerError(f"tensor contract differs for {key!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise HFDecoderRunnerError(f"checkpoint tensor is non-finite: {key!r}")
        state[key] = tensor.detach().cpu()
    return state


def _variant_cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    raw = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return _validate_state_dict(raw, model.state_dict())


def _variant_initialize_evisirst(
    dataset: str, *, seed: int = ARCHITECTURE_SEED, training: bool = True
) -> tuple[nn.Module, dict[str, Any]]:
    if dataset != DATASET or seed != ARCHITECTURE_SEED or training is not True:
        raise HFDecoderRunnerError("HF formal constructor contract differs")
    model, metadata = hf_decoder.build_irstd_hf_decoder_v1(
        dataset, seed=seed, training=training
    )
    manifest = hf_decoder.validate_irstd_hf_decoder_v1(
        model,
        require_identity_initialization=True,
        require_all_trainable=True,
    )
    parameters = list(model.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise HFDecoderRunnerError("all base and HF parameters must be trainable")
    _validate_state_dict(model.state_dict(), model.state_dict())
    ready = dict(metadata)
    ready.update(
        {
            "public_model": "EviSIRST-HF-Decoder-v1",
            "architecture_manifest": _json_clone(manifest),
            "full_model_scratch": True,
            "all_base_and_hf_parameters_trainable": True,
            "state_key_count": hf_decoder.FORMAL_STATE_KEY_COUNT,
            "public_test_supported": False,
        }
    )
    return model, ready


def _variant_len(value: Any) -> int:
    """Bridge R1's one legacy 564-key constructor assertion, and nothing else."""

    observed = builtins.len(value)
    if isinstance(value, Mapping) and observed == hf_decoder.FORMAL_STATE_KEY_COUNT:
        hf_count = sum(
            isinstance(key, str)
            and key.startswith(hf_decoder.HF_DECODER_STATE_PREFIX)
            for key in value
        )
        if hf_count == hf_decoder.HF_DECODER_STATE_KEY_COUNT:
            # R1 has one hard-coded clean-graph assertion immediately after its
            # constructor.  Every actual save/load contract uses the strict
            # variant validator above and therefore observes all 572 tensors.
            return hf_decoder.BASE_STATE_KEY_COUNT
    return observed


def _role_final_path(run_dir: Path, role: str) -> Path:
    names = {
        zero_selection.PRIMARY_ROLE: "EviSIRST_best_mIoU.pth.tar",
        zero_selection.SECONDARY_ROLE: "EviSIRST_best_Pd.pth.tar",
    }
    try:
        return run_dir / names[role]
    except KeyError as exc:
        raise HFDecoderRunnerError(f"unsupported final checkpoint role: {role}") from exc


def _validate_zero_margin_provenance(
    value: Any, *, history: list[Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HFDecoderRunnerError("zero-margin selection provenance is missing")
    provenance = _json_clone(dict(value))
    if (
        provenance.get("schema") != zero_selection.PROVENANCE_SCHEMA
        or provenance.get("rule_version") != zero_selection.RULE_VERSION
        or provenance.get("window_applied") is not False
        or provenance.get("selection_margin_raw") is not None
        or provenance.get("primary_role") != zero_selection.PRIMARY_ROLE
        or set(provenance.get("roles", {})) != set(zero_selection.VALID_ROLES)
    ):
        raise HFDecoderRunnerError("selection is not the frozen zero-margin rule")
    recomputed = zero_selection.select_checkpoints(
        history,
        primary_role=zero_selection.PRIMARY_ROLE,
        margin=None,
    )
    if recomputed != provenance:
        raise HFDecoderRunnerError("selection provenance cannot be recomputed")
    return provenance


def _candidate_for_role(
    *,
    run_dir: Path,
    role: str,
    provenance: Mapping[str, Any],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
    identity: Mapping[str, Any],
    history: list[Any],
    expected_state: Mapping[str, torch.Tensor],
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor], Path, str]:
    selected = provenance["roles"][role]["selected"]
    epoch = selected.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise HFDecoderRunnerError(f"{role} selected epoch is malformed")
    artifact = candidate_artifacts.get(epoch)
    if not isinstance(artifact, Mapping):
        raise HFDecoderRunnerError(f"{role} candidate artifact is missing")
    relative = artifact.get("relative_path")
    if relative != f"candidates/epoch_{epoch:04d}.pth.tar":
        raise HFDecoderRunnerError(f"{role} candidate path differs")
    path = run_dir / relative
    if path.is_symlink() or not path.is_file():
        raise HFDecoderRunnerError(f"{role} candidate is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise HFDecoderRunnerError(f"{role} candidate escapes run directory") from exc
    candidate_sha256 = _sha256_file(resolved)
    if artifact.get("file_sha256") != candidate_sha256:
        raise HFDecoderRunnerError(f"{role} candidate SHA-256 differs")
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    records = [
        record
        for record in history
        if isinstance(record, Mapping) and record.get("epoch") == epoch
    ]
    if builtins.len(records) != 1:
        raise HFDecoderRunnerError(f"{role} validation record is ambiguous")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CANDIDATE_SCHEMA
        or payload.get("run_identity") != identity
        or payload.get("epoch") != epoch
        or payload.get("validation_record") != records[0]
        or payload.get("test_split_accessed") is not False
    ):
        raise HFDecoderRunnerError(f"{role} candidate identity differs")
    state = _validate_state_dict(payload.get("state_dict"), expected_state)
    return epoch, dict(records[0]), state, resolved, candidate_sha256


def _atomic_write_dual_role_finals(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    history: list[Any],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
    expected_state: Mapping[str, torch.Tensor],
    primary_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Write and verify permanent best-mIoU and best-Pd final checkpoints."""

    provenance = _validate_zero_margin_provenance(
        primary_payload.get("selection_provenance"), history=history
    )
    staged: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    expected_by_role: dict[str, tuple[Path, dict[str, Any]]] = {}
    try:
        for role in zero_selection.VALID_ROLES:
            epoch, record, state, candidate_path, candidate_sha256 = _candidate_for_role(
                run_dir=run_dir,
                role=role,
                provenance=provenance,
                candidate_artifacts=candidate_artifacts,
                identity=identity,
                history=history,
                expected_state=expected_state,
            )
            final_path = _role_final_path(run_dir, role)
            role_payload = copy.deepcopy(dict(primary_payload))
            role_payload.update(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "model": "EviSIRST-HF-Decoder-v1",
                    "checkpoint_role": (
                        "experimental_validation_selected_" + role
                    ),
                    "selection_role": role,
                    "epoch": epoch,
                    "state_dict": state,
                    "selected_validation_record": _json_clone(record),
                    "selected_validation_record_sha256": _canonical_sha256(record),
                    "selected_complete_key": _json_clone(
                        provenance["roles"][role]["selected"]
                    ),
                    "selected_complete_key_sha256": _canonical_sha256(
                        provenance["roles"][role]["selected"]
                    ),
                    "source_candidate": {
                        "relative_path": candidate_path.relative_to(run_dir).as_posix(),
                        "sha256": candidate_sha256,
                    },
                    "selection_provenance": provenance,
                    "selection_margin_raw": None,
                    "selection_window_applied": False,
                    "public_test_supported": False,
                    "test_split_accessed": False,
                }
            )
            expected_by_role[role] = (final_path, role_payload)
            if final_path.is_symlink():
                raise HFDecoderRunnerError(f"{role} final must not be a symlink")
            if not final_path.exists():
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{final_path.name}.", suffix=".tmp", dir=run_dir
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                torch.save(role_payload, temporary)
                staged[role] = (temporary, final_path, role_payload)

        for temporary, final_path, _payload in staged.values():
            os.replace(temporary, final_path)

        artifacts: dict[str, Any] = {}
        for role, (final_path, expected_payload) in expected_by_role.items():
            if final_path.is_symlink() or not final_path.is_file():
                raise HFDecoderRunnerError(f"{role} final is unavailable")
            observed = torch.load(final_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(observed, Mapping)
                or observed.get("schema") != CHECKPOINT_SCHEMA
                or observed.get("training") != identity
                or observed.get("selection_role") != role
                or observed.get("epoch") != expected_payload["epoch"]
                or observed.get("selected_complete_key")
                != expected_payload["selected_complete_key"]
                or observed.get("selected_complete_key_sha256")
                != expected_payload["selected_complete_key_sha256"]
                or observed.get("source_candidate")
                != expected_payload["source_candidate"]
                or observed.get("selection_provenance") != provenance
                or observed.get("selection_margin_raw") is not None
                or observed.get("selection_window_applied") is not False
                or observed.get("public_test_supported") is not False
                or observed.get("test_split_accessed") is not False
            ):
                raise HFDecoderRunnerError(f"written {role} final failed verification")
            observed_state = _validate_state_dict(
                observed.get("state_dict"), expected_state
            )
            expected_role_state = expected_payload["state_dict"]
            if any(
                not torch.equal(observed_state[key], expected_role_state[key])
                for key in observed_state
            ):
                raise HFDecoderRunnerError(f"written {role} state differs")
            artifacts[role] = {
                "relative_path": final_path.relative_to(run_dir).as_posix(),
                "sha256": _sha256_file(final_path),
                "epoch": expected_payload["epoch"],
                "selection_role": role,
                "selected_complete_key": _json_clone(
                    expected_payload["selected_complete_key"]
                ),
                "selected_complete_key_sha256": expected_payload[
                    "selected_complete_key_sha256"
                ],
            }
        return artifacts
    finally:
        for temporary, _final, _payload in staged.values():
            if temporary.exists():
                temporary.unlink()


def _validate_committed_role_map(
    *,
    run_dir: Path,
    role_map: Any,
    identity: Mapping[str, Any],
    history: list[Any],
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Verify already-committed role files solely from summary evidence."""

    if not isinstance(role_map, Mapping) or set(role_map) != set(
        zero_selection.VALID_ROLES
    ):
        raise HFDecoderRunnerError("summary role-final map is malformed")
    provenance = zero_selection.select_checkpoints(
        history, primary_role=zero_selection.PRIMARY_ROLE, margin=None
    )
    normalized: dict[str, Any] = {}
    for role in zero_selection.VALID_ROLES:
        evidence = role_map.get(role)
        if not isinstance(evidence, Mapping):
            raise HFDecoderRunnerError(f"summary {role} evidence is malformed")
        selected = provenance["roles"][role]["selected"]
        expected_path = _role_final_path(run_dir, role)
        required = {
            "relative_path": expected_path.relative_to(run_dir).as_posix(),
            "epoch": selected["epoch"],
            "selection_role": role,
            "selected_complete_key": selected,
            "selected_complete_key_sha256": _canonical_sha256(selected),
        }
        if any(evidence.get(name) != value for name, value in required.items()):
            raise HFDecoderRunnerError(f"summary {role} evidence differs")
        expected_sha = evidence.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or builtins.len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
            or expected_path.is_symlink()
            or not expected_path.is_file()
            or _sha256_file(expected_path) != expected_sha
        ):
            raise HFDecoderRunnerError(f"committed summary {role} SHA/file differs")
        payload = torch.load(expected_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("training") != identity
            or payload.get("selection_role") != role
            or payload.get("epoch") != selected["epoch"]
            or payload.get("selected_complete_key") != selected
            or payload.get("selected_complete_key_sha256")
            != required["selected_complete_key_sha256"]
            or payload.get("selection_provenance") != provenance
            or payload.get("selection_margin_raw") is not None
            or payload.get("selection_window_applied") is not False
            or payload.get("public_test_supported") is not False
            or payload.get("test_split_accessed") is not False
        ):
            raise HFDecoderRunnerError(f"committed summary {role} payload differs")
        _validate_state_dict(payload.get("state_dict"), expected_state)
        normalized[role] = _json_clone(dict(evidence))
    return normalized


def _variant_final_checkpoint_payload(**kwargs: Any) -> dict[str, Any]:
    state = _validate_state_dict(
        kwargs.get("state_dict"), kwargs.get("state_dict")
    )
    adjusted = dict(kwargs)
    adjusted["state_dict"] = state
    payload = _R1_BUILD_FINAL_CHECKPOINT_PAYLOAD(**adjusted)
    identity = kwargs.get("identity")
    if not isinstance(identity, Mapping):
        raise HFDecoderRunnerError("final HF training identity is missing")
    payload.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": "EviSIRST-HF-Decoder-v1",
            "checkpoint_role": "experimental_validation_selected_best_mIoU",
            "experiment_schema": EXPERIMENT_SCHEMA,
            "experiment_status": "experimental_validation_only",
            "architecture_variant": identity.get("architecture_variant"),
            "state_contract": _json_clone(identity.get("state_contract")),
            "selection_margin_raw": None,
            "selection_window_applied": False,
            "promotion_gate": promotion_gate(),
            "public_test_supported": False,
            "test_split_accessed": False,
        }
    )
    return payload


def _variant_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = dict(payload)
    if output.get("schema") in {HISTORY_SCHEMA, TRAINING_SCHEMA + "/summary"}:
        output.update(
            {
                "experiment_schema": EXPERIMENT_SCHEMA,
                "architecture_variant": "IRSTD-HF-Decoder-v1",
                "selection_margin_raw": None,
                "selection_window_applied": False,
                "public_test_supported": False,
                "test_split_accessed": False,
            }
        )
    if output.get("schema") == TRAINING_SCHEMA + "/summary":
        history = output.get("validation_history")
        selection_payload = output.get("selection")
        if not isinstance(history, list) or not isinstance(selection_payload, Mapping):
            raise HFDecoderRunnerError("final HF summary evidence is incomplete")
        selected_epoch = selection_payload.get("selected_epoch")
        records = [
            record
            for record in history
            if isinstance(record, Mapping) and record.get("epoch") == selected_epoch
        ]
        if builtins.len(records) != 1:
            raise HFDecoderRunnerError("selected validation record is ambiguous")
        provenance = selection_payload.get("selection_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("rule_version") != zero_selection.RULE_VERSION
            or provenance.get("window_applied") is not False
        ):
            raise HFDecoderRunnerError("final selector is not strict zero-margin")
        output["selected_validation_record"] = _json_clone(records[0])
        output["selected_validation_record_sha256"] = _canonical_sha256(records[0])
        output["selection_roles"] = _json_clone(provenance.get("roles"))
        output["promotion_gate"] = promotion_gate()
    _R1_WRITE_JSON(path, output)


def _variant_atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    output = dict(payload)
    if output.get("schema") in {TRAINING_SCHEMA, CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA}:
        output.update(
            {
                "experiment_schema": EXPERIMENT_SCHEMA,
                "architecture_variant": "IRSTD-HF-Decoder-v1",
                "selection_margin_raw": None,
                "selection_window_applied": False,
                "public_test_supported": False,
                "test_split_accessed": False,
            }
        )
    _R1_ATOMIC_TORCH_SAVE(path, output)


@contextmanager
def _variant_runtime() -> Iterator[None]:
    """Exception-safely route the R1 transaction engine to this variant."""

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
        # See _variant_len: this is scoped to this module's runtime only.
        "len": _variant_len,
    }
    if not _PATCH_LOCK.acquire(blocking=False):
        raise HFDecoderRunnerError("HF-decoder runtime is already active")
    previous: dict[str, tuple[bool, Any]] = {
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
        _PATCH_LOCK.release()


@contextmanager
def _run_process_lock(args: argparse.Namespace) -> Iterator[Path]:
    paths = resolve_run_paths(args)
    run_dir = paths["run_dir"]
    if not isinstance(run_dir, Path):
        raise HFDecoderRunnerError("run directory is malformed")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".hf_decoder_v1.lock"
    if lock_path.is_symlink():
        raise HFDecoderRunnerError("run lock must not be a symlink")
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
                raise HFDecoderRunnerError(
                    f"HF-decoder run is already locked: {run_dir}"
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
        raise HFDecoderRunnerError("run directory is malformed")
    paths["best_mIoU_final"] = _role_final_path(
        run_dir, zero_selection.PRIMARY_ROLE
    )
    paths["best_Pd_final"] = _role_final_path(
        run_dir, zero_selection.SECONDARY_ROLE
    )
    return paths


def _current_identity_for_args(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild the immutable train/validation identity for finalization."""

    full_train, full_val, grouping = build_datasets(args)
    train_count = len(full_train)
    val_count = len(full_val)
    if args.smoke_max_train_samples is not None:
        train_count = min(args.smoke_max_train_samples, train_count)
    if args.smoke_max_val_samples is not None:
        val_count = min(args.smoke_max_val_samples, val_count)
    if train_count < 1 or val_count < 1:
        raise HFDecoderRunnerError("train/validation data is empty")
    return _variant_run_identity(
        args,
        full_train.contract,
        grouping,
        train_count=train_count,
        val_count=val_count,
        smoke=r1._is_smoke(args),
    )


def _load_completed_transaction(
    args: argparse.Namespace, paths: Mapping[str, Path | bool]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary_path = paths["summary"]
    latest_path = paths["latest"]
    assert isinstance(summary_path, Path)
    assert isinstance(latest_path, Path)
    if (
        summary_path.is_symlink()
        or latest_path.is_symlink()
        or not summary_path.is_file()
        or not latest_path.is_file()
    ):
        raise HFDecoderRunnerError("completed HF transaction artifacts are missing")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HFDecoderRunnerError("completed HF summary is malformed") from exc
    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != TRAINING_SCHEMA + "/summary"
        or summary.get("status") != "complete"
        or summary.get("test_split_accessed") is not False
        or not isinstance(latest, dict)
        or latest.get("schema") != TRAINING_SCHEMA
        or latest.get("epoch") != args.epochs
        or latest.get("test_split_accessed") is not False
    ):
        raise HFDecoderRunnerError("completed HF transaction identity differs")
    current_identity = _current_identity_for_args(args)
    if (
        latest.get("run_identity") != current_identity
        or summary.get("run_seed") != args.run_seed
        or summary.get("dataset") != DATASET
        or summary.get("target_mode") != TARGET_MODE
    ):
        raise HFDecoderRunnerError("completed HF run identity differs")
    training_history = latest.get("training_history")
    validation_history = latest.get("validation_history")
    if (
        not isinstance(training_history, list)
        or builtins.len(training_history) != args.epochs
        or not isinstance(validation_history, list)
        or validation_history != summary.get("validation_history")
        or not validation_history
    ):
        raise HFDecoderRunnerError("completed HF histories differ")
    provenance = zero_selection.select_checkpoints(
        validation_history,
        primary_role=zero_selection.PRIMARY_ROLE,
        margin=None,
    )
    summary_selection = summary.get("selection")
    if (
        not isinstance(summary_selection, Mapping)
        or summary_selection.get("selection_provenance") != provenance
        or summary_selection.get("selected_epoch")
        != provenance["primary_selected_epoch"]
    ):
        raise HFDecoderRunnerError("completed HF selection differs")
    return summary, latest, current_identity


def _finalize_completed(
    args: argparse.Namespace, paths: Mapping[str, Path | bool]
) -> Path:
    """Crash-idempotently materialize and bind both role checkpoints."""

    run_dir = paths["run_dir"]
    summary_path = paths["summary"]
    primary_engine_path = paths["final"]
    assert isinstance(run_dir, Path)
    assert isinstance(summary_path, Path)
    assert isinstance(primary_engine_path, Path)
    summary, latest, identity = _load_completed_transaction(args, paths)

    existing_role_map = summary.get("role_final_checkpoints")
    template_path = primary_engine_path
    if not template_path.is_file() or template_path.is_symlink():
        template_path = _role_final_path(run_dir, zero_selection.PRIMARY_ROLE)
    if template_path.is_symlink() or not template_path.is_file():
        raise HFDecoderRunnerError("no trusted primary finalization template exists")
    primary = torch.load(template_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(primary, Mapping)
        or primary.get("schema") != CHECKPOINT_SCHEMA
        or primary.get("training") != identity
        or primary.get("test_split_accessed") is not False
    ):
        raise HFDecoderRunnerError("primary finalization template identity differs")
    expected_state = _validate_state_dict(
        primary.get("state_dict"), primary.get("state_dict")
    )

    if existing_role_map is not None:
        role_finals = _validate_committed_role_map(
            run_dir=run_dir,
            role_map=existing_role_map,
            identity=identity,
            history=summary["validation_history"],
            expected_state=expected_state,
        )
        if primary_engine_path.exists():
            if primary_engine_path.is_symlink() or not primary_engine_path.is_file():
                raise HFDecoderRunnerError("legacy primary checkpoint is not regular")
            primary_engine_path.unlink()
        return _role_final_path(run_dir, zero_selection.PRIMARY_ROLE)

    raw_artifacts = latest.get("candidate_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise HFDecoderRunnerError("candidate artifact map is missing")
    artifacts: dict[int, Mapping[str, Any]] = {}
    for epoch, artifact in raw_artifacts.items():
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise HFDecoderRunnerError("candidate artifact epoch is malformed")
        if not isinstance(artifact, Mapping):
            raise HFDecoderRunnerError("candidate artifact is malformed")
        artifacts[epoch] = artifact
    role_finals = _atomic_write_dual_role_finals(
        run_dir=run_dir,
        args=args,
        identity=identity,
        history=summary["validation_history"],
        candidate_artifacts=artifacts,
        expected_state=expected_state,
        primary_payload=primary,
    )
    expected_summary = dict(summary)
    expected_summary["role_final_checkpoints"] = role_finals
    expected_summary["primary_checkpoint_role"] = zero_selection.PRIMARY_ROLE
    expected_summary["checkpoint"] = str(
        _role_final_path(run_dir, zero_selection.PRIMARY_ROLE).relative_to(
            PROJECT_ROOT
        )
    )
    expected_summary["legacy_engine_primary_checkpoint"] = str(
        primary_engine_path.relative_to(PROJECT_ROOT)
    )
    if summary != expected_summary:
        _variant_write_json(summary_path, expected_summary)
    # Re-read the commit before removing the redundant generic primary file.
    committed = json.loads(summary_path.read_text(encoding="utf-8"))
    if committed != expected_summary:
        raise HFDecoderRunnerError("dual-role summary commit verification failed")
    if primary_engine_path.exists():
        if primary_engine_path.is_symlink() or not primary_engine_path.is_file():
            raise HFDecoderRunnerError("legacy primary checkpoint is not regular")
        primary_engine_path.unlink()
    return _role_final_path(run_dir, zero_selection.PRIMARY_ROLE)


def run(args: argparse.Namespace) -> Path:
    """Execute through R1's transactional, validation-only engine."""

    _require_variant_args(args)
    paths = resolve_run_paths(args)
    summary_path = paths["summary"]
    assert isinstance(summary_path, Path)
    with _run_process_lock(args):
        if not summary_path.exists():
            with _variant_runtime():
                r1.run(args)
        # A completed R1 transaction is deliberately finalized outside the
        # engine call so --resume can recover any crash between generic final,
        # the two role finals, summary binding, and generic-file cleanup.
        return _finalize_completed(args, paths)


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
    "HFDecoderRunnerError",
    "HISTORY_SCHEMA",
    "PAIRED_RUN_SEED",
    "SELECTION_PAYLOAD_SCHEMA",
    "TARGET_MODE",
    "TRAINING_SCHEMA",
    "build_datasets",
    "parse_args",
    "promotion_gate",
    "resolve_run_paths",
    "run",
]
