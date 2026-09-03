#!/usr/bin/env python3
"""Train SCTransNet-C3-SBSC-V3.3 on the frozen full ``img_idx`` protocol.

Each of NUAA-SIRST, NUDT-SIRST, and IRSTD-1K is trained independently from its
original ``img_idx/train_*.txt`` split.  Epochs 1--499 are training-only and
epochs 500--1000 evaluate the complete original ``img_idx/test_*.txt`` split
after every epoch.  Two independent physical checkpoints retain best-mIoU and
best-Pd operating points.

This is deliberately an optimistic, test-selected protocol.  It cannot support
an unbiased held-out-test claim.  The SCTransNet baseline remains a read-only
historical reference and is never trained or evaluated by this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments.evisirst_data import (
    EviSIRSTTestDataset,
    EviSIRSTTrainDataset,
    SOURCE_DATASETS,
    stable_uint63,
)
from experiments import sbsc_v33_test_selection as selection
from experiments import sbsc_v33_contracts as contracts
from experiments import three_dataset_v2_protocol as data_protocol
from experiments import sctransnet_sbsc_v33 as core
from test import MATCH_RADIUS, PROBABILITY_THRESHOLD, TINY_AREA, evaluate_model
from train import (
    _atomic_torch_save,
    _capture_rng_state,
    _index_sha,
    _restore_rng_state,
    _write_json,
    configure_determinism,
    learning_rate_for_epoch,
    require_device,
)


_ENTRY_SOURCE = Path(__file__)
if _ENTRY_SOURCE.is_symlink() or not _ENTRY_SOURCE.is_file():
    raise RuntimeError("formal V3.3 runner must be a regular non-symlink file")
PROJECT_ROOT = _ENTRY_SOURCE.resolve(strict=True).parent
RECOVERY_SCHEMA = "sctransnet_c3_sbsc_v33_img_idx_test_selected_recovery/v1"
CHECKPOINT_SCHEMA = "sctransnet_c3_sbsc_v33_img_idx_test_selected_checkpoint/v1"
SUMMARY_SCHEMA = "sctransnet_c3_sbsc_v33_img_idx_test_selected_summary/v1"
PROTOCOL_NAME = "sctransnet_c3_sbsc_v33_three_dataset_img_idx_test_selected/v1"
MODEL_NAME = "SCTransNet-C3-SBSC-V3.3"
METHOD_NAME = "sbsc_v33_third"
BALANCE_MODE = "one_third_two_thirds"
ROUTER_VALUE_GRADIENT_MODE = "live"
ARCHITECTURE_SEED = 42
RUN_SEED = 42
SHUFFLE_STREAM = "sctransnet_sbsc_v33_pair"
TRAINING_TARGET_RULE = "raw_mask_div_255"
EXPECTED_COUNTS = {
    "NUAA-SIRST": {"train": 213, "test": 214},
    "NUDT-SIRST": {"train": 663, "test": 664},
    "IRSTD-1K": {"train": 800, "test": 201},
}
FORMAL_EPOCHS = 1000
FORMAL_SELECTION_BEGIN = 500
FORMAL_SELECTION_EVERY = 1
EXPECTED_STATE_KEY_COUNT = 513
EXPECTED_PARAMETER_COUNT = 11_330_188
METHOD_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sbsc_v33_methods"
    / "sbsc_v33_third_irstd_formal.json"
)
FORMAL_LAUNCH_AUTHORIZATION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "sbsc_v33_formal_launch_authorization.json"
)
FORMAL_LAUNCH_FIELDS = frozenset(
    {
        "schema",
        "status",
        "write_once",
        "authorized_run",
        "method_config_path",
        "method_config_sha256",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_sha256",
        "training_source_manifest_sha256",
        "evidence",
        "environment_identity",
        "environment_identity_sha256",
        "source_bindings",
        "replay_checks",
    }
)
FORMAL_EVIDENCE_PATHS = {
    "unit_tests": "artifacts/sbsc_v33_preflight/unit_test_report.json",
    "canary": "runs/sbsc_v33_third/canary_v2/IRSTD-1K/report.json",
    "paired_resource_benchmark": (
        "artifacts/sbsc_v33_preflight/paired_resource_benchmark_v2.json"
    ),
}
EXPECTED_RESOURCE_V2_REPORT_SHA256 = (
    "ec6b04abcc61454a0bbc46bd84167b0bf6770d4ad7ed4397018a4f551e39a6b1"
)
EXPECTED_CANARY_V2_REPORT_SHA256 = (
    "5226e11896c241cd235c8e1689a4c5f7678b86a143823a1fd4c6bf2efc04a34c"
)
EXPECTED_CANARY_V2_MANIFEST_SHA256 = (
    "c986f19557415b6808802cd3f7608a06de70004035b8bb6748032930e80c9810"
)
FROZEN_FORMAL_EVIDENCE_SHA256 = {
    "canary": EXPECTED_CANARY_V2_REPORT_SHA256,
    "paired_resource_benchmark": EXPECTED_RESOURCE_V2_REPORT_SHA256,
}
EXPECTED_INACTIVE_PARAMETER_NAMES = frozenset(
    {
        f"mtc.embeddings_{index}.position_embeddings"
        for index in range(1, 5)
    }
    | {
        f"mtc.encoder.layer.{layer}.channel_attn.q{query}_attn{attention}"
        for layer in range(4)
        for query in range(1, 5)
        for attention in range(1, 5)
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCE_DATASETS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    # These values are not CLI knobs.  They are repeated here only so the
    # inherited training loop can serialize the exact frozen config.
    args.output_root = PROJECT_ROOT / "runs" / METHOD_NAME / "formal"
    args.seed = RUN_SEED
    args.architecture_seed = ARCHITECTURE_SEED
    args.epochs = FORMAL_EPOCHS
    args.batch_size = 16
    args.patch_size = 256
    args.workers = 0
    args.base_lr = 1.0e-3
    args.min_lr = 1.0e-5
    args.warmup_epochs = 10
    args.selection_begin = FORMAL_SELECTION_BEGIN
    args.selection_every = FORMAL_SELECTION_EVERY
    args.max_train_samples = None
    args.max_test_images = None
    args.smoke = False
    return args


def selection_due(epoch: int, begin: int, every: int) -> bool:
    return epoch >= begin and (epoch - begin) % every == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return contracts.canonical_sha256(value)


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    value = core.state_dict_sha256(state)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError("core returned a malformed state SHA-256")
    return value


def _validate_optimizer_state_finite(value: Any) -> None:
    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all()
            ):
                raise ValueError(f"optimizer contains a non-finite tensor at {path}")
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"optimizer contains a non-finite scalar at {path}")
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                raise ValueError("optimizer state contains a container cycle")
            seen.add(identity)
            for key, nested in item.items():
                visit(nested, f"{path}.{key}")
            seen.remove(identity)
            return
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                raise ValueError("optimizer state contains a container cycle")
            seen.add(identity)
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
            seen.remove(identity)

    visit(value, "optimizer")


def _training_source_manifest() -> dict[str, Any]:
    relative_paths = [
        "train_sctransnet_sbsc_v33_img_idx_test_selected.py",
        "train.py",
        "test.py",
        "experiments/evisirst_data.py",
        "experiments/three_dataset_v2_protocol.py",
        "experiments/four_dataset_models_seed42_v1.py",
        "experiments/sbsc_v33_contracts.py",
        "experiments/sbsc_v33_test_selection.py",
        "experiments/sctransnet_sbsc_v33.py",
        "experiments/sctransnet_sbsc_v32.py",
        "experiments/sctransnet_sbsc_v31.py",
        *sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "model").rglob("*.py")
        ),
    ]
    return contracts.build_source_manifest(relative_paths)


def _source_tree_sha256() -> str:
    return str(_training_source_manifest()["sha256"])


def _lower_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _load_bound_json(record: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{name} reference must contain exact path/SHA fields")
    path = contracts.require_repository_relative_regular_file(record["path"])
    expected = _lower_sha256(record["sha256"], name=f"{name}.sha256")
    if contracts.sha256_file(path) != expected:
        raise ValueError(f"{name} physical SHA-256 differs")
    return contracts.load_strict_json(path)


def _load_frozen_bound_json(
    record: Any,
    *,
    name: str,
    expected_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load evidence only when its reference matches an independent freeze."""

    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256"}
        or record.get("path") != expected_path
        or record.get("sha256") != expected_sha256
    ):
        raise ValueError(f"{name} reference differs from frozen path/SHA-256")
    return _load_bound_json(record, name=name)


def _load_frozen_json_path(
    path: Path,
    *,
    name: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load a fixed-path JSON evidence file after checking its physical digest."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} is unavailable")
    if contracts.sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} physical SHA-256 differs from frozen digest")
    return contracts.load_strict_json(path)


def _validate_formal_launch_closure(
    launch: Mapping[str, Any],
    *,
    dataset: str,
    method_config_sha256: str,
    baseline: Mapping[str, Any],
    gradient: Mapping[str, Any],
    current_source: Mapping[str, Any],
    unit_record: Mapping[str, Any],
    canary_record: Mapping[str, Any],
    resource_record: Mapping[str, Any],
) -> None:
    """Close every launch field against independently replayed evidence."""

    from tools import authorize_sbsc_v33_formal_launch as authorizer

    if set(launch) != FORMAL_LAUNCH_FIELDS:
        raise ValueError("formal-launch authorization field set differs")
    if (
        launch.get("schema") != authorizer.AUTHORIZATION_SCHEMA
        or launch.get("status") != "PASS"
        or launch.get("write_once") is not True
        or launch.get("authorized_run")
        != {
            "method": METHOD_NAME,
            "dataset": dataset,
            "run_kind": "formal",
            "output_namespace": "runs/sbsc_v33_third/formal/IRSTD-1K",
        }
        or launch.get("method_config_path")
        != contracts.repository_relative_path(METHOD_CONFIG_PATH)
        or launch.get("method_config_sha256") != method_config_sha256
        or launch.get("gradient_authorization_sha256")
        != gradient["authorization_sha256"]
        or launch.get("baseline_authority_manifest_sha256")
        != baseline["manifest_sha256"]
        or launch.get("training_source_manifest_sha256")
        != current_source["sha256"]
    ):
        raise ValueError("formal-launch identity or authority binding differs")

    evidence = launch.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        FORMAL_EVIDENCE_PATHS
    ):
        raise ValueError("formal-launch evidence set differs")
    for name, expected_path in FORMAL_EVIDENCE_PATHS.items():
        reference = evidence.get(name)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or reference.get("path") != expected_path
        ):
            raise ValueError(f"formal-launch evidence reference {name!r} differs")
        _lower_sha256(reference.get("sha256"), name=f"{name}.sha256")
        frozen_sha256 = FROZEN_FORMAL_EVIDENCE_SHA256.get(name)
        if frozen_sha256 is not None and reference.get("sha256") != frozen_sha256:
            raise ValueError(
                f"formal-launch evidence reference {name!r} SHA-256 differs"
            )

    source_bindings = launch.get("source_bindings")
    expected_source_bindings = {
        "formal_training": current_source["sha256"],
        "unit_suite": unit_record["source_manifest"]["sha256"],
        "canary_v2": canary_record["source_manifest"]["sha256"],
        "paired_resource_benchmark": resource_record["source_manifest"][
            "sha256"
        ],
    }
    if (
        not isinstance(source_bindings, Mapping)
        or set(source_bindings) != set(expected_source_bindings)
        or dict(source_bindings) != expected_source_bindings
        or launch.get("replay_checks") != authorizer.REPLAY_CHECKS
    ):
        raise ValueError("formal-launch source or replay-check binding differs")
    for name, value in source_bindings.items():
        _lower_sha256(value, name=f"source_bindings.{name}")

    environment = authorizer.build_environment_identity(
        unit=unit_record,
        canary=canary_record,
        resource_record=resource_record,
    )
    embedded_environment = launch.get("environment_identity")
    if not isinstance(embedded_environment, Mapping):
        raise ValueError("formal-launch environment identity is missing")
    validated_environment = authorizer.validate_environment_identity(
        embedded_environment
    )
    if (
        validated_environment != environment
        or launch.get("environment_identity_sha256")
        != contracts.canonical_sha256(environment)
    ):
        raise ValueError("formal-launch environment identity binding differs")


def _load_formal_preflight(dataset: str) -> dict[str, Any]:
    """Verify every launch authority before data construction or GPU access."""

    if dataset != "IRSTD-1K":
        raise RuntimeError(
            "the write-once V3.3 launch sequence currently authorizes IRSTD-1K only"
        )
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    if gradient["authorized_router_value_gradient_mode"] != ROUTER_VALUE_GRADIENT_MODE:
        raise RuntimeError("formal gradient mode differs from frozen authorization")

    if METHOD_CONFIG_PATH.is_symlink() or not METHOD_CONFIG_PATH.is_file():
        raise FileNotFoundError("frozen V3.3 formal method config is unavailable")
    method_config = contracts.load_strict_json(METHOD_CONFIG_PATH)
    required_method_values = {
        "schema": "sctransnet_sbsc_v33/method_config/v1",
        "status": "frozen",
        "write_once": True,
        "run_kind": "formal",
        "protocol": PROTOCOL_NAME,
        "model": MODEL_NAME,
        "builder": (
            "experiments.sctransnet_sbsc_v33."
            "build_sctransnet_sbsc_v33_method"
        ),
        "method": METHOD_NAME,
        "dataset": dataset,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "router_value_gradient_mode": ROUTER_VALUE_GRADIENT_MODE,
        "balance_mode": BALANCE_MODE,
        "loss_schema": core.SBSC_V33_LOSS_SCHEMA,
        "segmentation_loss": "sum_of_six_BCELoss_mean_terms",
        "router_level_count": 4,
        "router_loss_weight": 1.0,
        "level_reduction": "mean",
        "optimizer": "Adam",
        "epochs": FORMAL_EPOCHS,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "amp": False,
        "selection_begin_epoch": FORMAL_SELECTION_BEGIN,
        "selection_end_epoch": FORMAL_EPOCHS,
        "selection_every": FORMAL_SELECTION_EVERY,
        "output_namespace": "runs/sbsc_v33_third/formal/IRSTD-1K",
        "dataset_root_recorded": False,
        "selection_roles": {
            "best_miou": [
                "miou:max",
                "pd:max",
                "fa:min",
                "niou:max",
                "tiny_pd:max",
                "test_loss:min",
                "epoch:min",
            ],
            "best_pd": [
                "pd:max",
                "fa:min",
                "tiny_pd:max",
                "miou:max",
                "niou:max",
                "test_loss:min",
                "epoch:min",
            ],
        },
        "evaluator": "evisirst_public_common_evaluator_v1",
        "probability_threshold": 0.5,
        "probability_comparison": ">",
        "component_match_radius": 3.0,
        "component_match_comparison": "<",
        "tiny_area_max": 9,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
    }
    for key, expected in required_method_values.items():
        if method_config.get(key) != expected:
            raise ValueError(f"formal method config field {key!r} differs")
    expected_method_fields = set(required_method_values) | {
        "split_contract",
        "split_contract_sha256",
        "training_source_manifest",
        "training_source_manifest_sha256",
        "gradient_authorization_path",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_path",
        "baseline_authority_manifest_sha256",
    }
    if set(method_config) != expected_method_fields:
        raise ValueError("formal method config field set differs")
    if (
        method_config.get("gradient_authorization_path")
        != gradient["authorization_path"]
        or method_config.get("gradient_authorization_sha256")
        != gradient["authorization_sha256"]
        or method_config.get("baseline_authority_manifest_path")
        != baseline["manifest_path"]
        or method_config.get("baseline_authority_manifest_sha256")
        != baseline["manifest_sha256"]
    ):
        raise ValueError("formal method config authority binding differs")
    current_source = _training_source_manifest()
    if (
        method_config.get("training_source_manifest") != current_source
        or method_config.get("training_source_manifest_sha256")
        != current_source["sha256"]
    ):
        raise ValueError("formal training source manifest drifted")

    # Rebuild and replay the exact method config without constructing a dataset.
    from tools import authorize_sbsc_v33_formal_launch as authorizer
    from tools import benchmark_sbsc_v33_resources_v2 as resource

    expected_method_config = authorizer.freezer.build_config(PROJECT_ROOT / "datasets")
    authorizer.validate_method_config(
        method_config, expected_config=expected_method_config
    )

    if (
        FORMAL_LAUNCH_AUTHORIZATION_PATH.is_symlink()
        or not FORMAL_LAUNCH_AUTHORIZATION_PATH.is_file()
    ):
        raise FileNotFoundError("formal-launch authorization is unavailable")
    launch = contracts.load_strict_json(FORMAL_LAUNCH_AUTHORIZATION_PATH)
    evidence = launch.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        FORMAL_EVIDENCE_PATHS
    ):
        raise ValueError("formal-launch evidence set differs")
    for name, expected_path in FORMAL_EVIDENCE_PATHS.items():
        reference = evidence.get(name)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or reference.get("path") != expected_path
        ):
            raise ValueError(f"formal-launch evidence reference {name!r} differs")
    unit_report = _load_bound_json(evidence["unit_tests"], name="unit tests")
    canary_report = _load_frozen_bound_json(
        evidence["canary"],
        name="canary",
        expected_path=FORMAL_EVIDENCE_PATHS["canary"],
        expected_sha256=EXPECTED_CANARY_V2_REPORT_SHA256,
    )
    benchmark_report = _load_frozen_bound_json(
        evidence["paired_resource_benchmark"],
        name="paired resource benchmark",
        expected_path=FORMAL_EVIDENCE_PATHS["paired_resource_benchmark"],
        expected_sha256=EXPECTED_RESOURCE_V2_REPORT_SHA256,
    )
    unit_record = authorizer.validate_unit_evidence(unit_report)

    canary_manifest = _load_frozen_json_path(
        authorizer.CANARY_MANIFEST_PATH,
        name="canary V2 manifest",
        expected_sha256=EXPECTED_CANARY_V2_MANIFEST_SHA256,
    )
    canary_rules = contracts.load_strict_json(authorizer.CANARY_RULES_PATH)
    canary_record = authorizer.validate_canary_v2_evidence(
        canary_report,
        canary_manifest,
        canary_rules,
        baseline=baseline,
        gradient=gradient,
    )

    resource_rules = resource.load_frozen_rules()
    resource_evidence = resource.load_bound_evidence(resource_rules)
    resource_source = resource.build_source_manifest()
    resource_plan = resource.v1.expected_batch_plan()
    benchmark_batch_plan = benchmark_report.get("batch_plan")
    if not isinstance(benchmark_batch_plan, Mapping) or not isinstance(
        benchmark_batch_plan.get("input_pair_sha256_by_step"), list
    ):
        raise ValueError("paired resource benchmark input hashes are missing")
    resource_input_hashes = list(
        benchmark_batch_plan["input_pair_sha256_by_step"]
    )
    resource_record = authorizer.validate_resource_evidence(
        benchmark_report,
        resource_rules,
        expected_evidence=resource_evidence,
        expected_source_manifest=resource_source,
        batch_plan=resource_plan,
        input_hashes=resource_input_hashes,
    )

    method_config_sha256 = contracts.sha256_file(METHOD_CONFIG_PATH)
    _validate_formal_launch_closure(
        launch,
        dataset=dataset,
        method_config_sha256=method_config_sha256,
        baseline=baseline,
        gradient=gradient,
        current_source=current_source,
        unit_record=unit_record,
        canary_record=canary_record,
        resource_record=resource_record,
    )
    return {
        "baseline": baseline,
        "gradient": gradient,
        "method_config": method_config,
        "method_config_sha256": method_config_sha256,
        "launch": launch,
        "launch_authorization_sha256": contracts.sha256_file(
            FORMAL_LAUNCH_AUTHORIZATION_PATH
        ),
        "training_source_manifest": current_source,
    }


def _formal_config(
    args: argparse.Namespace,
    *,
    train_count: int,
    test_count: int,
    train_index_sha256: str,
    test_index_sha256: str,
    normalization: Mapping[str, float],
    source_tree_sha256: str,
    authorities: Mapping[str, Any],
) -> dict[str, Any]:
    split_manifest = {
        "dataset": args.dataset,
        "train_count": train_count,
        "test_count": test_count,
        "train_index_order_sha256": train_index_sha256,
        "test_index_order_sha256": test_index_sha256,
        "normalization": dict(normalization),
        "training_target_rule": TRAINING_TARGET_RULE,
    }
    split_manifest_sha256 = _canonical_sha256(split_manifest)
    run_identity_basis = {
        "protocol": PROTOCOL_NAME,
        "model": MODEL_NAME,
        "method": METHOD_NAME,
        "dataset": args.dataset,
        "architecture_seed": args.architecture_seed,
        "run_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "workers": args.workers,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "selection_begin_epoch": args.selection_begin,
        "selection_every": args.selection_every,
        "split_manifest_sha256": split_manifest_sha256,
        "training_source_tree_sha256": source_tree_sha256,
        "method_config_sha256": authorities["method_config_sha256"],
        "gradient_authorization_sha256": authorities["gradient"][
            "authorization_sha256"
        ],
        "baseline_authority_manifest_sha256": authorities["baseline"][
            "manifest_sha256"
        ],
        "formal_launch_authorization_sha256": authorities[
            "launch_authorization_sha256"
        ],
        "smoke": bool(args.smoke),
    }
    selection_identity = {
        "method": METHOD_NAME,
        "dataset": args.dataset,
        "architecture_seed": args.architecture_seed,
        "run_seed": args.seed,
        "split_manifest_sha256": split_manifest_sha256,
        "run_identity_sha256": _canonical_sha256(run_identity_basis),
        "method_config_sha256": authorities["method_config_sha256"],
        "gradient_authorization_sha256": authorities["gradient"][
            "authorization_sha256"
        ],
        "baseline_authority_manifest_sha256": authorities["baseline"][
            "manifest_sha256"
        ],
        "formal_launch_authorization_sha256": authorities[
            "launch_authorization_sha256"
        ],
        "training_source_manifest_sha256": source_tree_sha256,
    }
    baseline_evaluation = authorities["baseline"]["authorities"][args.dataset]
    evaluation_contract = {
        "dataset": args.dataset,
        "sample_count": baseline_evaluation["sample_count"],
        "protocol": baseline_evaluation["protocol"],
        "threshold": baseline_evaluation["threshold"],
        "threshold_operator": baseline_evaluation["threshold_operator"],
        "match_radius": baseline_evaluation["match_radius"],
        "match_radius_operator": baseline_evaluation["match_radius_operator"],
        "tiny_area": baseline_evaluation["tiny_area"],
        "normalization": dict(baseline_evaluation["normalization"]),
    }
    evaluation_contract_sha256 = contracts.canonical_sha256(evaluation_contract)
    run_identity_basis["evaluation_contract_sha256"] = (
        evaluation_contract_sha256
    )
    selection_identity["run_identity_sha256"] = contracts.canonical_sha256(
        run_identity_basis
    )
    selection_identity["evaluation_contract_sha256"] = (
        evaluation_contract_sha256
    )
    return {
        "schema": "sctransnet_sbsc_v33/formal_training_config/v1",
        "run_kind": "formal",
        "protocol": PROTOCOL_NAME,
        "model": MODEL_NAME,
        "method": METHOD_NAME,
        "model_components": [
            "C3-SBSC-V3.3",
            "dual-risk evidence estimator",
            "train-only tri-router supervision",
        ],
        "model_graph": "sctransnet_c3_sbsc_v33_513_state_keys",
        "initialization": "scratch_seed42",
        "architecture_seed": args.architecture_seed,
        "run_seed": args.seed,
        "baseline_checkpoint_loaded": False,
        "baseline_trained_by_runner": False,
        "baseline_reference_kind": "historical_existing",
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "workers": args.workers,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "optimizer": "Adam",
        "loss": core.SBSC_V33_LOSS_SCHEMA,
        "segmentation_loss": "sum_of_six_BCELoss_mean_terms",
        "router_auxiliary_loss_weight": 1.0,
        "router_loss_weight": 1.0,
        "router_level_count": 4,
        "amp": False,
        "training_target_rule": TRAINING_TARGET_RULE,
        "train_count": train_count,
        "test_count": test_count,
        "train_index_order_sha256": train_index_sha256,
        "test_index_order_sha256": test_index_sha256,
        "split_manifest": split_manifest,
        "split_manifest_sha256": split_manifest_sha256,
        "run_identity_sha256": selection_identity["run_identity_sha256"],
        "selection_identity": selection_identity,
        "normalization": dict(normalization),
        "shuffle_seed_algorithm": (
            "sha256_length_prefixed_str_parts_uint63("
            "42,sctransnet_sbsc_v33_pair,dataset,shuffle,epoch)"
        ),
        "shuffle_stream": SHUFFLE_STREAM,
        "crop_seed_algorithm": "historical_source_namespaced_sha256_uint64",
        "selection_begin_epoch": args.selection_begin,
        "selection_end_epoch": args.epochs,
        "selection_every": args.selection_every,
        "selection_roles": {
            "best_miou": [
                "miou:max",
                "pd:max",
                "fa:min",
                "niou:max",
                "tiny_pd:max",
                "test_loss:min",
                "epoch:min",
            ],
            "best_pd": [
                "pd:max",
                "fa:min",
                "tiny_pd:max",
                "miou:max",
                "niou:max",
                "test_loss:min",
                "epoch:min",
            ],
        },
        "selection_comparison": "strict_lexicographic_greater_than",
        "selection_tie_rule": "earliest_exact_role_key_wins",
        "selection_split": f"{args.dataset}_test",
        "data_role": "test",
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": ">",
        "component_connectivity": 8,
        "component_match_radius": MATCH_RADIUS,
        "component_match_comparison": "<",
        "tiny_area_max": TINY_AREA,
        "evaluator": "evisirst_public_common_evaluator_v1",
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "evaluation_source_sha256": _sha256(PROJECT_ROOT / "test.py"),
        "evaluation_sample_count": test_count,
        "full_metric_vector_evaluated": True,
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
        "training_source_tree_sha256": source_tree_sha256,
        "training_source_manifest": authorities["training_source_manifest"],
        "training_source_manifest_sha256": source_tree_sha256,
        "method_config_path": contracts.repository_relative_path(
            METHOD_CONFIG_PATH
        ),
        "method_config_sha256": authorities["method_config_sha256"],
        "gradient_authorization_path": authorities["gradient"][
            "authorization_path"
        ],
        "gradient_authorization_sha256": authorities["gradient"][
            "authorization_sha256"
        ],
        "baseline_authority_manifest_path": authorities["baseline"][
            "manifest_path"
        ],
        "baseline_authority_manifest_sha256": authorities["baseline"][
            "manifest_sha256"
        ],
        "formal_launch_authorization_path": contracts.repository_relative_path(
            FORMAL_LAUNCH_AUTHORIZATION_PATH
        ),
        "formal_launch_authorization_sha256": authorities[
            "launch_authorization_sha256"
        ],
        "router_value_gradient_mode": ROUTER_VALUE_GRADIENT_MODE,
        "balance_mode": BALANCE_MODE,
        "level_reduction": "mean",
        "smoke": bool(args.smoke),
    }


def _validate_state(
    value: Any, expected: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    try:
        core.validate_sbsc_v33_state_dict(value, METHOD_NAME)
    except Exception as exc:
        raise ValueError("checkpoint violates the frozen SBSC-V3.3 schema") from exc
    if not isinstance(value, Mapping) or list(value) != list(expected):
        raise ValueError("checkpoint state key/order differs from SBSC-V3.3")
    for key, tensor in value.items():
        reference = expected[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint tensor {key!r} is not a tensor")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"checkpoint tensor contract differs for {key!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint tensor {key!r} is non-finite")
    return value


def _cpu_model_state(
    model: nn.Module, expected: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }
    _validate_state(state, expected)
    return state


def _freeze_structurally_inactive(model: nn.Module) -> tuple[str, ...]:
    names = tuple(core.structurally_inactive_parameter_names(model))
    if set(names) != EXPECTED_INACTIVE_PARAMETER_NAMES or len(names) != len(
        EXPECTED_INACTIVE_PARAMETER_NAMES
    ):
        raise RuntimeError("structurally inactive parameter contract differs")
    named = dict(model.named_parameters())
    if not set(names).issubset(named):
        raise RuntimeError("structurally inactive parameters are missing")
    for name in names:
        named[name].requires_grad_(False)
    if any(named[name].requires_grad for name in names):
        raise RuntimeError("structurally inactive parameter freeze failed")
    return tuple(sorted(names))


def _build_model(dataset: str) -> tuple[nn.Module, dict[str, Any], tuple[str, ...]]:
    expected_core = (
        PROJECT_ROOT / "experiments" / "sctransnet_sbsc_v33.py"
    ).resolve(strict=True)
    if Path(core.__file__ or "").resolve(strict=True) != expected_core:
        raise RuntimeError("SBSC-V3.3 core was not imported from this repository")
    configure_determinism(ARCHITECTURE_SEED)
    model, raw_metadata = core.build_sctransnet_sbsc_v33_method(
        method=METHOD_NAME,
        dataset=dataset,
        architecture_seed=ARCHITECTURE_SEED,
        training=True,
        router_value_gradient_mode=ROUTER_VALUE_GRADIENT_MODE,
    )
    if not isinstance(model, nn.Module) or not isinstance(raw_metadata, Mapping):
        raise RuntimeError("SBSC-V3.3 builder returned malformed data")
    metadata = json.loads(
        json.dumps(
            dict(raw_metadata),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    pair = metadata.get("pair")
    if (
        metadata.get("method") != METHOD_NAME
        or metadata.get("model") != MODEL_NAME
        or metadata.get("dataset") != dataset
        or metadata.get("architecture_seed") != ARCHITECTURE_SEED
        or metadata.get("test_split_accessed") is not False
        or not isinstance(pair, Mapping)
        or pair.get("parent_checkpoint") is not None
        or pair.get("warm_start_used") is not False
        or pair.get("predecessor_checkpoint_used") is not False
    ):
        raise RuntimeError("SBSC-V3.3 builder identity differs")
    if len(model.state_dict()) != EXPECTED_STATE_KEY_COUNT:
        raise RuntimeError("SBSC-V3.3 state-key count differs")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("SBSC-V3.3 parameter count differs")
    core.validate_sctransnet_sbsc_v33(
        model,
        method=METHOD_NAME,
        router_value_gradient_mode=ROUTER_VALUE_GRADIENT_MODE,
        require_zero_gain=True,
    )
    _validate_state(model.state_dict(), model.state_dict())
    inactive = _freeze_structurally_inactive(model)
    return model, metadata, inactive


def _training_losses(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        total_loss, segmentation_loss, router_loss, _breakdown = (
            core.training_losses_v33(
                model,
                images,
                masks,
                criterion,
                balance_mode=BALANCE_MODE,
                router_loss_weight=1.0,
            )
        )
    except Exception as exc:
        raise RuntimeError("SBSC-V3.3 router supervision failed") from exc
    for label, loss in (
        ("total", total_loss),
        ("segmentation", segmentation_loss),
        ("router", router_loss),
    ):
        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not bool(torch.isfinite(loss))
            or float(loss.detach().item()) < 0.0
        ):
            raise RuntimeError(f"{label} loss is malformed")
    return total_loss, segmentation_loss, router_loss


def _metric_row(
    metrics: Mapping[str, Any],
    epoch: int,
    *,
    selection_identity: Mapping[str, Any],
    model_state_sha256: str,
    sample_count: int,
) -> dict[str, Any]:
    required = (
        "test_loss",
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
        "fa",
        "false_objects_per_image",
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    )
    if set(metrics) != set(required):
        raise ValueError("evaluator returned an unexpected metric field set")
    row = {"epoch": epoch}
    for key in required:
        value = metrics[key]
        if value is None:
            if key != "tiny_pd":
                raise ValueError(f"metric {key!r} is unexpectedly null")
            row[key] = None
            row["tiny_pd_was_undefined"] = True
        elif isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"metric {key!r} is not numeric")
        elif not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} is non-finite")
        else:
            if key.endswith("_count") or key == "valid_pixel_count":
                if not isinstance(value, numbers.Integral) or int(value) < 0:
                    raise ValueError(f"metric {key!r} must be a nonnegative int")
                row[key] = int(value)
            else:
                row[key] = float(value)
    for key in (
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
    ):
        if not 0.0 <= float(row[key]) <= 1.0:
            raise ValueError(f"selection metric {key!r} is outside [0,1]")
    if row["tiny_pd"] is not None and not 0.0 <= float(row["tiny_pd"]) <= 1.0:
        raise ValueError("selection tiny_pd is outside [0,1]")
    for key in ("test_loss", "fa", "false_objects_per_image"):
        if float(row[key]) < 0.0:
            raise ValueError(f"selection metric {key!r} must be nonnegative")
    if (
        row["matched_target_count"] > row["target_count"]
        or row["matched_tiny_target_count"] > row["tiny_target_count"]
        or row["unmatched_predicted_object_count"]
        > row["predicted_object_count"]
    ):
        raise ValueError("selection metric counts are inconsistent")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("selection sample_count must be a positive int")
    if "tiny_pd_was_undefined" not in row:
        row["tiny_pd_was_undefined"] = False
    identity = dict(selection_identity)
    if set(identity) != {
        "method",
        "dataset",
        "architecture_seed",
        "run_seed",
        "split_manifest_sha256",
        "run_identity_sha256",
        "evaluation_contract_sha256",
        "method_config_sha256",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_sha256",
        "formal_launch_authorization_sha256",
        "training_source_manifest_sha256",
    }:
        raise ValueError("selection identity field set differs")
    row.update(
        {
            "schema": selection.RECORD_SCHEMA,
            "data_role": "test",
            "test_split_accessed": True,
            "test_selected": True,
            "selection_is_optimistic": True,
            "unbiased_test_claim_supported": False,
            "model_state_sha256": model_state_sha256,
            "model": MODEL_NAME,
            "seed": RUN_SEED,
            "sample_count": sample_count,
            "evaluation_source_sha256": _sha256(PROJECT_ROOT / "test.py"),
            "mIoU": float(row["miou"]),
            "nIoU": float(row["niou"]),
            "Pd": float(row["pd"]),
            "Fa": float(row["fa"]),
            "tinyPd": (
                None if row["tiny_pd"] is None else float(row["tiny_pd"])
            ),
            "loss": float(row["test_loss"]),
            **identity,
        }
    )
    return row


def role_key(row: Mapping[str, Any], role: str) -> tuple[float, ...]:
    tiny = float("-inf") if row.get("tiny_pd") is None else float(row["tiny_pd"])
    epoch = int(row["epoch"])
    if role == "best_miou":
        return (
            float(row["miou"]),
            float(row["pd"]),
            -float(row["fa"]),
            float(row["niou"]),
            tiny,
            -float(row["test_loss"]),
            -float(epoch),
        )
    if role == "best_pd":
        return (
            float(row["pd"]),
            -float(row["fa"]),
            tiny,
            float(row["miou"]),
            float(row["niou"]),
            -float(row["test_loss"]),
            -float(epoch),
        )
    raise ValueError(f"unsupported checkpoint role: {role!r}")


def role_key_record(row: Mapping[str, Any], role: str) -> list[float | None]:
    return [value if math.isfinite(value) else None for value in role_key(row, role)]


def _validate_history(
    history: Any,
    *,
    completed_epoch: int,
    config: Mapping[str, Any],
) -> dict[str, int | None]:
    if not isinstance(history, list) or any(not isinstance(row, Mapping) for row in history):
        raise ValueError("selection history is malformed")
    begin = int(config["selection_begin_epoch"])
    every = int(config["selection_every"])
    expected_epochs = [
        epoch
        for epoch in range(begin, completed_epoch + 1)
        if selection_due(epoch, begin, every)
    ]
    observed_epochs = [int(row.get("epoch", -1)) for row in history]
    if observed_epochs != expected_epochs:
        raise ValueError("selection history cadence differs")
    if not bool(config["smoke"]):
        try:
            payload = selection.select_prefix(
                history,
                completed_epoch=completed_epoch,
                expected_identity=config["selection_identity"],
            )
        except selection.SBSCV33TestSelectionError as exc:
            raise ValueError("formal test-selection history differs") from exc
        role_map = {"best_mIoU": "best_miou", "best_Pd": "best_pd"}
        if set(selection.VALID_ROLES) != set(role_map):
            raise ValueError("formal selector role names differ")
        return {
            internal: (
                None
                if external not in payload["roles"]
                else int(payload["roles"][external]["selected"]["epoch"])
            )
            for external, internal in role_map.items()
        }
    return {
        role: (
            None if not history else int(max(history, key=lambda row: role_key(row, role))["epoch"])
        )
        for role in ("best_miou", "best_pd")
    }


def _load_recovery(
    path: Path,
    *,
    config: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != RECOVERY_SCHEMA
        or payload.get("model") != MODEL_NAME
        or payload.get("method") != METHOD_NAME
        or payload.get("dataset") != config["dataset"]
        or payload.get("seed") != 42
        or payload.get("training") != config
    ):
        raise ValueError("recovery identity differs")
    _validate_state(payload.get("state_dict"), expected_state)
    if not isinstance(payload.get("optimizer"), Mapping):
        raise ValueError("recovery optimizer state is malformed")
    _validate_optimizer_state_finite(payload["optimizer"])
    completed = int(payload.get("epoch", -1))
    best_epochs = _validate_history(
        payload.get("selection_history"),
        completed_epoch=completed,
        config=config,
    )
    stored_epochs = payload.get("best_epochs")
    if not isinstance(stored_epochs, Mapping) or {
        role: (None if stored_epochs.get(role) is None else int(stored_epochs[role]))
        for role in ("best_miou", "best_pd")
    } != best_epochs:
        raise ValueError("recovery best selection differs from history")
    return dict(payload)


def _recovery_matches_winner(
    payload: Mapping[str, Any],
    *,
    role: str,
    winner_epoch: int,
) -> bool:
    if type(payload.get("epoch")) is not int or payload["epoch"] != winner_epoch:
        return False
    best_epochs = payload.get("best_epochs")
    if (
        not isinstance(best_epochs, Mapping)
        or type(best_epochs.get(role)) is not int
        or best_epochs[role] != winner_epoch
    ):
        return False
    history = payload.get("selection_history")
    if not isinstance(history, list):
        return False
    row = next(
        (
            candidate
            for candidate in history
            if isinstance(candidate, Mapping)
            and type(candidate.get("epoch")) is int
            and candidate["epoch"] == winner_epoch
        ),
        None,
    )
    if not isinstance(row, Mapping):
        return False
    expected_state_sha = row.get("model_state_sha256")
    try:
        observed_state_sha = _state_dict_sha256(payload.get("state_dict"))
    except Exception:
        return False
    return observed_state_sha == expected_state_sha


def _repair_missing_winner_recoveries(
    *,
    candidate_paths: list[tuple[int, Path]],
    best_epochs: Mapping[str, int | None],
    best_recovery_paths: Mapping[str, Path],
    config: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
) -> None:
    """Repair an interrupted multi-role frontier without guessing a state.

    A recovery can be copied only when its own epoch, selected-role epoch and
    tensor hash all match the frozen history record.  This covers a crash
    between the two best-role writes while refusing an unreconstructable
    winner.
    """

    if set(best_epochs) != {"best_miou", "best_pd"} or set(
        best_recovery_paths
    ) != {"best_miou", "best_pd"}:
        raise ValueError("winner-recovery role set differs")
    for role in ("best_miou", "best_pd"):
        winner_epoch = best_epochs[role]
        if winner_epoch is None:
            continue
        if type(winner_epoch) is not int:
            raise ValueError(f"{role} winner epoch must be an int")
        target = best_recovery_paths[role]
        if target.is_symlink():
            raise ValueError(f"{role} recovery cannot be a symlink")
        if target.exists() and not target.is_file():
            raise ValueError(f"{role} recovery must be a regular file")
        if target.is_file():
            current = _load_recovery(
                target,
                config=config,
                expected_state=expected_state,
            )
            if _recovery_matches_winner(
                current,
                role=role,
                winner_epoch=winner_epoch,
            ):
                continue

        source: dict[str, Any] | None = None
        for candidate_epoch, candidate_path in candidate_paths:
            if candidate_epoch != winner_epoch:
                continue
            candidate = _load_recovery(
                candidate_path,
                config=config,
                expected_state=expected_state,
            )
            if _recovery_matches_winner(
                candidate,
                role=role,
                winner_epoch=winner_epoch,
            ):
                source = candidate
                break
        if source is None:
            raise RuntimeError(
                f"cannot reconstruct interrupted {role} winner recovery"
            )
        _atomic_torch_save(target, source)
        replay = _load_recovery(
            target,
            config=config,
            expected_state=expected_state,
        )
        if not _recovery_matches_winner(
            replay,
            role=role,
            winner_epoch=winner_epoch,
        ):
            raise RuntimeError(f"reconstructed {role} recovery replay failed")


def _slim_checkpoint(
    *,
    state: Mapping[str, torch.Tensor],
    dataset: str,
    role: str,
    epoch: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if role not in ("best_miou", "best_pd"):
        raise ValueError(f"unsupported checkpoint role: {role!r}")
    metric_name = "global_foreground_mIoU" if role == "best_miou" else "Pd"
    score = float(metrics["miou"] if role == "best_miou" else metrics["pd"])
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": MODEL_NAME,
        "method": METHOD_NAME,
        "dataset": dataset,
        "checkpoint_role": role,
        "epoch": epoch,
        "seed": 42,
        "state_dict": dict(state),
        "training": dict(config),
        "train_index_order_sha256": config["train_index_order_sha256"],
        "normalization": dict(config["normalization"]),
        "test_split_accessed": True,
        "data_role": "test",
        "run_used_test_for_selection": True,
        "run_selection_is_optimistic": True,
        "test_selected": True,
        "this_checkpoint_selected_by_test": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
        "baseline_reference_kind": "historical_existing",
        "training_target_rule": TRAINING_TARGET_RULE,
        "selection_metric": metric_name,
        "selection_score": score,
        "selection_epoch": epoch,
        "model_state_sha256": _state_dict_sha256(state),
        "selection_identity": dict(config["selection_identity"]),
        "selection_role_key": role_key_record(metrics, role),
        "selection_metrics": dict(metrics),
        "role_key": role_key_record(metrics, role),
        "metrics": dict(metrics),
        "method_config_path": config["method_config_path"],
        "method_config_sha256": config["method_config_sha256"],
        "gradient_authorization_path": config[
            "gradient_authorization_path"
        ],
        "gradient_authorization_sha256": config[
            "gradient_authorization_sha256"
        ],
        "baseline_authority_manifest_path": config[
            "baseline_authority_manifest_path"
        ],
        "baseline_authority_manifest_sha256": config[
            "baseline_authority_manifest_sha256"
        ],
        "formal_launch_authorization_path": config[
            "formal_launch_authorization_path"
        ],
        "formal_launch_authorization_sha256": config[
            "formal_launch_authorization_sha256"
        ],
        "training_source_manifest_sha256": config[
            "training_source_tree_sha256"
        ],
    }


def _canonical_json_text(value: Any, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite strict JSON") from exc


def _validate_checkpoint_payload_exact(
    observed: Any,
    expected: Mapping[str, Any],
    *,
    expected_state: Mapping[str, torch.Tensor] | None,
    label: str,
) -> dict[str, Any]:
    """Replay every checkpoint field and tensor, not only its role key."""

    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        raise ValueError(f"{label} checkpoint field set differs")
    observed_state = observed.get("state_dict")
    expected_payload_state = expected.get("state_dict")
    if not isinstance(observed_state, Mapping) or not isinstance(
        expected_payload_state, Mapping
    ):
        raise ValueError(f"{label} checkpoint state_dict is missing")
    if expected_state is None:
        try:
            core.validate_sbsc_v33_state_dict(observed_state, METHOD_NAME)
        except Exception as exc:
            raise ValueError(
                f"{label} checkpoint violates the frozen V3.3 state schema"
            ) from exc
        if len(observed_state) != EXPECTED_STATE_KEY_COUNT or any(
            key.startswith("module.") for key in observed_state
        ):
            raise ValueError(
                f"{label} checkpoint must use the exact unwrapped V3.3 state"
            )
    else:
        _validate_state(observed_state, expected_state)
    if list(observed_state) != list(expected_payload_state):
        raise ValueError(f"{label} checkpoint state key/order differs")
    for key, expected_tensor in expected_payload_state.items():
        observed_tensor = observed_state[key]
        if not isinstance(observed_tensor, torch.Tensor) or not isinstance(
            expected_tensor, torch.Tensor
        ):
            raise ValueError(f"{label} checkpoint tensor {key!r} is malformed")
        if not torch.equal(observed_tensor, expected_tensor):
            raise ValueError(f"{label} checkpoint tensor differs: {key!r}")
    observed_metadata = {
        key: value for key, value in observed.items() if key != "state_dict"
    }
    expected_metadata = {
        key: value for key, value in expected.items() if key != "state_dict"
    }
    if _canonical_json_text(
        observed_metadata, label=f"{label} checkpoint metadata"
    ) != _canonical_json_text(
        expected_metadata, label=f"expected {label} checkpoint metadata"
    ):
        raise ValueError(f"{label} checkpoint metadata differs")
    observed_state_sha = _state_dict_sha256(observed_state)
    if (
        observed.get("model_state_sha256") != observed_state_sha
        or expected.get("model_state_sha256") != observed_state_sha
    ):
        raise ValueError(f"{label} checkpoint state SHA-256 differs")
    return dict(observed)


def _publish_once(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"final result checkpoint already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _validate_completed_training_config(
    value: Any,
    *,
    dataset: str,
    authorities: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("completed training config is missing")
    method_split = authorities["method_config"].get("split_contract")
    if not isinstance(method_split, Mapping):
        raise ValueError("completed method split contract is missing")
    train_split = method_split.get("train")
    test_split = method_split.get("test")
    normalization = method_split.get("normalization")
    if (
        not isinstance(train_split, Mapping)
        or not isinstance(test_split, Mapping)
        or not isinstance(normalization, Mapping)
    ):
        raise ValueError("completed method split contract is malformed")
    frozen_args = argparse.Namespace(
        dataset=dataset,
        architecture_seed=ARCHITECTURE_SEED,
        seed=RUN_SEED,
        epochs=FORMAL_EPOCHS,
        batch_size=16,
        patch_size=256,
        workers=0,
        base_lr=1.0e-3,
        min_lr=1.0e-5,
        warmup_epochs=10,
        selection_begin=FORMAL_SELECTION_BEGIN,
        selection_every=FORMAL_SELECTION_EVERY,
        smoke=False,
    )
    expected_base = _formal_config(
        frozen_args,
        train_count=EXPECTED_COUNTS[dataset]["train"],
        test_count=EXPECTED_COUNTS[dataset]["test"],
        train_index_sha256=train_split["runner_index_order_sha256"],
        test_index_sha256=test_split["runner_index_order_sha256"],
        normalization=normalization,
        source_tree_sha256=authorities["training_source_manifest"]["sha256"],
        authorities=authorities,
    )
    runtime_fields = {
        "builder_metadata_sha256",
        "structurally_inactive_parameter_names",
        "state_key_count",
        "parameter_count",
    }
    if set(value) != set(expected_base) | runtime_fields:
        raise ValueError("completed training config field set differs")
    observed_base = {key: value[key] for key in expected_base}
    if _canonical_json_text(
        observed_base, label="completed training config"
    ) != _canonical_json_text(
        expected_base, label="expected completed training config"
    ):
        raise ValueError("completed frozen training config differs")
    _lower_sha256(
        value.get("builder_metadata_sha256"),
        name="training.builder_metadata_sha256",
    )
    if value.get("structurally_inactive_parameter_names") != sorted(
        EXPECTED_INACTIVE_PARAMETER_NAMES
    ):
        raise ValueError("completed inactive-parameter contract differs")
    if (
        type(value.get("state_key_count")) is not int
        or value["state_key_count"] != EXPECTED_STATE_KEY_COUNT
        or type(value.get("parameter_count")) is not int
        or value["parameter_count"] != EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("completed model size contract differs")
    return dict(value)


def _validate_completed_publication(
    summary_path: Path,
    published_paths: Mapping[str, Path],
    dataset: str,
    *,
    authorities: Mapping[str, Any],
) -> None:
    if summary_path.is_symlink() or not summary_path.is_file() or any(
        path.is_symlink() or not path.is_file() for path in published_paths.values()
    ):
        raise ValueError("completed independent publication is not regular files")
    if set(published_paths) != {"best_miou", "best_pd"}:
        raise ValueError("completed checkpoint role set differs")
    summary = contracts.load_strict_json(summary_path)
    expected_summary_fields = {
        "schema",
        "status",
        "model",
        "method",
        "dataset",
        "training",
        "candidate_count",
        "selections",
        "published_checkpoints",
        "two_distinct_physical_checkpoint_files",
        "data_role",
        "test_split_accessed",
        "test_selected",
        "selection_is_optimistic",
        "unbiased_test_claim_supported",
        "baseline_reference_kind",
        "method_config_path",
        "method_config_sha256",
        "gradient_authorization_path",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_path",
        "baseline_authority_manifest_sha256",
        "formal_launch_authorization_path",
        "formal_launch_authorization_sha256",
        "training_source_manifest_sha256",
        "selection_history",
        "formal_selection_provenance",
        "elapsed_seconds",
    }
    if not isinstance(summary, Mapping) or set(summary) != expected_summary_fields:
        raise ValueError("completed independent publication summary fields differ")
    exact_summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "model": MODEL_NAME,
        "method": METHOD_NAME,
        "dataset": dataset,
        "candidate_count": FORMAL_EPOCHS - FORMAL_SELECTION_BEGIN + 1,
        "two_distinct_physical_checkpoint_files": True,
        "data_role": "test",
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
        "baseline_reference_kind": "historical_existing",
    }
    for key, expected_value in exact_summary.items():
        observed = summary.get(key)
        if type(expected_value) is bool:
            differs = observed is not expected_value
        elif type(expected_value) is int:
            differs = type(observed) is not int or observed != expected_value
        else:
            differs = observed != expected_value
        if differs:
            raise ValueError(f"completed summary field {key!r} differs")
    elapsed = summary.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, numbers.Real)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("completed elapsed_seconds is malformed")
    training = _validate_completed_training_config(
        summary.get("training"),
        dataset=dataset,
        authorities=authorities,
    )
    summary_bindings = {
        "method_config_path": training["method_config_path"],
        "method_config_sha256": training["method_config_sha256"],
        "gradient_authorization_path": training["gradient_authorization_path"],
        "gradient_authorization_sha256": training[
            "gradient_authorization_sha256"
        ],
        "baseline_authority_manifest_path": training[
            "baseline_authority_manifest_path"
        ],
        "baseline_authority_manifest_sha256": training[
            "baseline_authority_manifest_sha256"
        ],
        "formal_launch_authorization_path": training[
            "formal_launch_authorization_path"
        ],
        "formal_launch_authorization_sha256": training[
            "formal_launch_authorization_sha256"
        ],
        "training_source_manifest_sha256": training[
            "training_source_manifest_sha256"
        ],
    }
    for key, expected_value in summary_bindings.items():
        if summary.get(key) != expected_value:
            raise ValueError(f"completed summary binding {key!r} differs")
    history = summary.get("selection_history")
    best_epochs = _validate_history(
        history,
        completed_epoch=FORMAL_EPOCHS,
        config=training,
    )
    if not isinstance(history, list):
        raise ValueError("completed selection history is malformed")
    try:
        selection_provenance = selection.select_final(
            history,
            expected_identity=training["selection_identity"],
        )
    except selection.SBSCV33TestSelectionError as exc:
        raise ValueError("completed formal selector replay failed") from exc
    if _canonical_json_text(
        summary.get("formal_selection_provenance"),
        label="completed formal selection provenance",
    ) != _canonical_json_text(
        selection_provenance,
        label="replayed formal selection provenance",
    ):
        raise ValueError("completed formal selection provenance differs")
    expected_selections: dict[str, Any] = {}
    for role in ("best_miou", "best_pd"):
        role_epoch = best_epochs[role]
        if type(role_epoch) is not int:
            raise ValueError(f"completed {role} winner is missing")
        metrics = next(
            row
            for row in history
            if type(row.get("epoch")) is int and row["epoch"] == role_epoch
        )
        expected_selections[role] = {
            "epoch": role_epoch,
            "metrics": metrics,
            "role_key": role_key_record(metrics, role),
            "model_state_sha256": metrics["model_state_sha256"],
            "test_selected": True,
            "selection_is_optimistic": True,
        }
    if _canonical_json_text(
        summary.get("selections"), label="completed selections"
    ) != _canonical_json_text(
        expected_selections, label="replayed completed selections"
    ):
        raise ValueError("completed selections differ from history replay")
    published = summary.get("published_checkpoints")
    if not isinstance(published, Mapping) or set(published) != {
        "best_miou",
        "best_pd",
    }:
        raise ValueError("completed independent publication summary differs")
    for role, path in published_paths.items():
        record = published.get(role)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "model_state_sha256"}
            or record.get("path") != path.relative_to(PROJECT_ROOT).as_posix()
            or record.get("sha256") != _sha256(path)
        ):
            raise ValueError("completed checkpoint record differs")
        metrics = expected_selections[role]["metrics"]
        if record.get("model_state_sha256") != metrics["model_state_sha256"]:
            raise ValueError("completed checkpoint state binding differs")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("completed checkpoint cannot be loaded safely") from exc
        state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError("completed checkpoint state_dict is missing")
        if _state_dict_sha256(state) != metrics["model_state_sha256"]:
            raise ValueError("completed checkpoint tensor hash differs from history")
        expected_checkpoint = _slim_checkpoint(
            state=state,
            dataset=dataset,
            role=role,
            epoch=expected_selections[role]["epoch"],
            metrics=metrics,
            config=training,
        )
        _validate_checkpoint_payload_exact(
            checkpoint,
            expected_checkpoint,
            expected_state=None,
            label=f"completed {role}",
        )
    if len({path.stat().st_ino for path in published_paths.values()}) != 2:
        raise ValueError("completed selected checkpoints are not distinct files")


def run(args: argparse.Namespace) -> Path:
    authorities = _load_formal_preflight(args.dataset)
    configure_determinism(args.seed)
    device = require_device(args.device)
    if args.dataset_root.is_symlink() or not args.dataset_root.is_dir():
        raise ValueError("dataset root must be a regular non-symlink directory")
    dataset_root = args.dataset_root.resolve(strict=True)
    canonical_dataset_root = (PROJECT_ROOT / "datasets").resolve(strict=True)
    if dataset_root != canonical_dataset_root:
        raise ValueError(
            "formal mode requires this repository's canonical datasets directory"
        )
    run_dir = PROJECT_ROOT / authorities["method_config"]["output_namespace"]
    if run_dir != args.output_root / args.dataset:
        raise RuntimeError("formal output namespace differs from the fixed CLI identity")
    current = PROJECT_ROOT
    for part in run_dir.relative_to(PROJECT_ROOT).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("formal output namespace traverses a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "last_training_state.pth.tar"
    best_recovery_paths = {
        "best_miou": run_dir / "best_miou_state.pth.tar",
        "best_pd": run_dir / "best_pd_state.pth.tar",
    }
    history_path = run_dir / "selection_history.json"
    summary_path = run_dir / "summary.json"
    published_paths = {
        "best_miou": run_dir / "best_mIoU.pth.tar",
        "best_pd": run_dir / "best_Pd.pth.tar",
    }
    protected_paths = (
        latest_path,
        *best_recovery_paths.values(),
        history_path,
        summary_path,
        *published_paths.values(),
    )
    if any(path.is_symlink() for path in protected_paths):
        raise ValueError("formal output artifacts cannot be symlinks")

    if summary_path.exists():
        if args.resume and all(path.is_file() for path in published_paths.values()):
            # Idempotent cleanup for a crash after summary commit but before
            # recovery-file removal.  The result checkpoint remains immutable.
            _validate_completed_publication(
                summary_path,
                published_paths,
                args.dataset,
                authorities=authorities,
            )
            latest_path.unlink(missing_ok=True)
            for path in best_recovery_paths.values():
                path.unlink(missing_ok=True)
            return run_dir
        raise FileExistsError("this independent experiment is already complete")
    if any(path.exists() for path in published_paths.values()) and not args.resume:
        raise FileExistsError(
            "a partial publication exists; pass --resume to validate and finalize it"
        )

    full_train = EviSIRSTTrainDataset(
        args.dataset,
        dataset_root=dataset_root,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    full_test = EviSIRSTTestDataset(
        args.dataset,
        args.dataset,
        dataset_root=dataset_root,
    )
    full_test_ids = list(full_test.sample_ids)
    test_normalization = dict(full_test.normalization)
    if len(full_train) != EXPECTED_COUNTS[args.dataset]["train"] or len(
        full_test
    ) != EXPECTED_COUNTS[args.dataset]["test"]:
        raise RuntimeError("original img_idx train/test sample count differs")
    if not set(full_train.sample_ids).isdisjoint(full_test_ids):
        raise RuntimeError("original img_idx train/test identifiers overlap")
    method_split = authorities["method_config"].get("split_contract")
    actual_split = {
        "schema": "sctransnet_sbsc_v33/img_idx_split_contract/v1",
        "dataset": args.dataset,
        "train": {
            "index_path": (
                f"datasets/{args.dataset}/img_idx/train_{args.dataset}.txt"
            ),
            "count": len(full_train.sample_ids),
            "file_sha256": contracts.sha256_file(
                dataset_root
                / args.dataset
                / "img_idx"
                / f"train_{args.dataset}.txt"
            ),
            "ordered_ids_sha256": data_protocol.ordered_ids_sha256(
                full_train.sample_ids
            ),
            "runner_index_order_sha256": _index_sha(full_train.sample_ids),
        },
        "test": {
            "index_path": (
                f"datasets/{args.dataset}/img_idx/test_{args.dataset}.txt"
            ),
            "count": len(full_test_ids),
            "file_sha256": contracts.sha256_file(
                dataset_root
                / args.dataset
                / "img_idx"
                / f"test_{args.dataset}.txt"
            ),
            "ordered_ids_sha256": data_protocol.ordered_ids_sha256(full_test_ids),
            "runner_index_order_sha256": _index_sha(full_test_ids),
        },
        "normalization": dict(full_train.normalization),
        "train_test_disjoint": True,
        "validation_split_used": False,
    }
    if (
        method_split != actual_split
        or authorities["method_config"].get("split_contract_sha256")
        != contracts.canonical_sha256(actual_split)
    ):
        raise RuntimeError("formal dataset split differs from method config")
    train_dataset: Any = full_train
    test_dataset: Any = full_test
    if args.max_train_samples is not None:
        train_dataset = Subset(full_train, range(min(args.max_train_samples, len(full_train))))
    if args.max_test_images is not None:
        test_dataset = Subset(full_test, range(min(args.max_test_images, len(full_test))))
    train_count = len(train_dataset)
    test_count = len(test_dataset)
    if not args.smoke and (
        train_count != EXPECTED_COUNTS[args.dataset]["train"]
        or test_count != EXPECTED_COUNTS[args.dataset]["test"]
    ):
        raise RuntimeError("formal independent train/test sample count differs")

    source_sha = _source_tree_sha256()
    config = _formal_config(
        args,
        train_count=train_count,
        test_count=test_count,
        train_index_sha256=_index_sha(full_train.sample_ids),
        test_index_sha256=_index_sha(full_test_ids),
        normalization=full_train.normalization,
        source_tree_sha256=source_sha,
        authorities=authorities,
    )
    if full_train.normalization != test_normalization:
        raise RuntimeError("independent train/test normalization differs")

    model, model_metadata, inactive_parameter_names = _build_model(args.dataset)
    config["builder_metadata_sha256"] = _canonical_sha256(model_metadata)
    config["structurally_inactive_parameter_names"] = list(
        inactive_parameter_names
    )
    config["state_key_count"] = EXPECTED_STATE_KEY_COUNT
    config["parameter_count"] = EXPECTED_PARAMETER_COUNT
    model.to(device)
    configure_determinism(RUN_SEED)
    expected_state = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    criterion = nn.BCELoss(reduction="mean")

    evaluation_generator = torch.Generator(device="cpu")
    evaluation_generator.manual_seed(stable_uint63(42, args.dataset, "evaluation"))
    evaluation_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=evaluation_generator,
        drop_last=False,
    )

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_epochs: dict[str, int | None] = {
        "best_miou": None,
        "best_pd": None,
    }
    elapsed_before = 0.0
    if args.resume:
        candidates: list[tuple[int, Path]] = []
        for path in (latest_path, *best_recovery_paths.values()):
            if path.is_file() and not path.is_symlink():
                candidate = _load_recovery(
                    path, config=config, expected_state=expected_state
                )
                candidates.append((int(candidate["epoch"]), path))
        if not candidates:
            raise FileNotFoundError("no matching recovery checkpoint exists")
        _, recovery_path = max(candidates, key=lambda item: item[0])
        recovery = _load_recovery(
            recovery_path, config=config, expected_state=expected_state
        )
        completed = int(recovery["epoch"])
        if completed < 0 or completed > args.epochs:
            raise ValueError("recovery epoch is outside this run")
        incompatible = model.load_state_dict(recovery["state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict SBSC-V3.3 resume state load failed")
        core.validate_sctransnet_sbsc_v33(
            model,
            method=METHOD_NAME,
            router_value_gradient_mode=ROUTER_VALUE_GRADIENT_MODE,
            require_zero_gain=False,
        )
        _validate_optimizer_state_finite(recovery["optimizer"])
        optimizer.load_state_dict(recovery["optimizer"])
        _validate_optimizer_state_finite(optimizer.state_dict())
        _restore_rng_state(recovery["rng"], device)
        history = [dict(row) for row in recovery["selection_history"]]
        best_epochs = {
            role: (
                None
                if recovery["best_epochs"][role] is None
                else int(recovery["best_epochs"][role])
            )
            for role in ("best_miou", "best_pd")
        }
        _repair_missing_winner_recoveries(
            candidate_paths=candidates,
            best_epochs=best_epochs,
            best_recovery_paths=best_recovery_paths,
            config=config,
            expected_state=expected_state,
        )
        elapsed_before = float(recovery.get("elapsed_seconds", 0.0))
        start_epoch = completed + 1
    elif latest_path.exists() or any(path.exists() for path in best_recovery_paths.values()):
        raise FileExistsError(f"recovery exists; pass --resume: {run_dir}")

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        full_train.set_epoch(epoch)
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(
            stable_uint63(
                RUN_SEED,
                SHUFFLE_STREAM,
                args.dataset,
                "shuffle",
                epoch,
            )
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=shuffle_generator,
            drop_last=False,
        )
        lr = learning_rate_for_epoch(
            epoch,
            args.epochs,
            args.base_lr,
            args.min_lr,
            args.warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        model.mode = "train"
        loss_sum = 0.0
        segmentation_loss_sum = 0.0
        router_loss_sum = 0.0
        processed = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, segmentation_loss, router_loss = _training_losses(
                model, images, masks, criterion
            )
            loss.backward()
            optimizer.step()
            core.project_sbsc_v33_constraints_(model)
            count = int(images.shape[0])
            loss_sum += float(loss.detach().item()) * count
            segmentation_loss_sum += float(segmentation_loss.detach().item()) * count
            router_loss_sum += float(router_loss.detach().item()) * count
            processed += count
        if processed != train_count:
            raise RuntimeError("processed training sample count differs")

        state = _cpu_model_state(model, expected_state)
        state_sha256 = _state_dict_sha256(state)
        improved_roles: list[str] = []
        metrics: dict[str, Any] | None = None
        if selection_due(epoch, args.selection_begin, args.selection_every):
            training_rng = _capture_rng_state(device)
            try:
                metrics = _metric_row(
                    evaluate_model(model, evaluation_loader, device),
                    epoch,
                    selection_identity=config["selection_identity"],
                    model_state_sha256=state_sha256,
                    sample_count=test_count,
                )
            finally:
                _restore_rng_state(training_rng, device)
            history.append(metrics)
            selected_now = _validate_history(
                history,
                completed_epoch=epoch,
                config=config,
            )
            for role in ("best_miou", "best_pd"):
                if selected_now[role] != best_epochs[role]:
                    if selected_now[role] != epoch:
                        raise RuntimeError(
                            "selector changed a winner to a non-current epoch"
                        )
                    best_epochs[role] = epoch
                    improved_roles.append(role)
        elapsed = elapsed_before + time.time() - started
        optimizer_state = optimizer.state_dict()
        _validate_optimizer_state_finite(optimizer_state)
        recovery = {
            "schema": RECOVERY_SCHEMA,
            "model": MODEL_NAME,
            "method": METHOD_NAME,
            "dataset": args.dataset,
            "epoch": epoch,
            "seed": args.seed,
            "state_dict": state,
            "optimizer": optimizer_state,
            "rng": _capture_rng_state(device),
            "training": config,
            "mean_train_loss": loss_sum / processed,
            "mean_segmentation_loss": segmentation_loss_sum / processed,
            "mean_router_loss": router_loss_sum / processed,
            "learning_rate": lr,
            "selection_history": history,
            "best_epochs": dict(best_epochs),
            "elapsed_seconds": elapsed,
            "data_role": "test",
            "test_split_accessed": True,
            "test_selected": True,
            "selection_is_optimistic": True,
            "unbiased_test_claim_supported": False,
            "baseline_reference_kind": "historical_existing",
        }
        for role in improved_roles:
            _atomic_torch_save(best_recovery_paths[role], recovery)
        _atomic_torch_save(latest_path, recovery)
        _write_json(
            history_path,
            {
                "schema": SUMMARY_SCHEMA + "/history",
                "dataset": args.dataset,
                "selection_history": history,
            },
        )
        suffix = ""
        if metrics is not None:
            suffix = (
                f" test_mIoU={100.0 * float(metrics['miou']):.6f}%"
                f" nIoU={100.0 * float(metrics['niou']):.6f}%"
                f" F1={100.0 * float(metrics['pixel_f1']):.6f}%"
                f" Pd={100.0 * float(metrics['pd']):.6f}%"
                f" Fa={1e6 * float(metrics['fa']):.6f}e-6"
                f" best_mIoU_epoch={best_epochs['best_miou']}"
                f" best_Pd_epoch={best_epochs['best_pd']}"
            )
        print(
            f"dataset={args.dataset} epoch={epoch}/{args.epochs} "
            f"loss={loss_sum / processed:.6f} lr={lr:.8f} samples={processed}{suffix}",
            flush=True,
        )

    if any(best_epochs[role] is None for role in ("best_miou", "best_pd")) or any(
        not path.is_file() for path in best_recovery_paths.values()
    ):
        raise RuntimeError("training completed without both selected checkpoints")
    if _source_tree_sha256() != source_sha:
        raise RuntimeError("training source tree changed during the run")
    expected_best_epochs = _validate_history(
        history,
        completed_epoch=args.epochs,
        config=config,
    )
    if expected_best_epochs != best_epochs:
        raise RuntimeError("final role epochs differ from selection history")
    final_selection_provenance: dict[str, Any] | None = None
    if not args.smoke:
        try:
            final_selection_provenance = selection.select_final(
                history,
                expected_identity=config["selection_identity"],
            )
        except selection.SBSCV33TestSelectionError as exc:
            raise RuntimeError("formal 501-record test selection is incomplete") from exc
        selector_best = {
            "best_miou": int(
                final_selection_provenance["roles"]["best_mIoU"]["selected"]["epoch"]
            ),
            "best_pd": int(
                final_selection_provenance["roles"]["best_Pd"]["selected"]["epoch"]
            ),
        }
        if selector_best != best_epochs:
            raise RuntimeError("formal final selector differs from retained winners")
    selections: dict[str, Any] = {}
    published_records: dict[str, Any] = {}
    for role in ("best_miou", "best_pd"):
        role_epoch = int(best_epochs[role])
        recovery = _load_recovery(
            best_recovery_paths[role], config=config, expected_state=expected_state
        )
        if int(recovery["epoch"]) != role_epoch:
            raise RuntimeError(f"{role} recovery epoch differs from selected epoch")
        metrics = next(row for row in history if int(row["epoch"]) == role_epoch)
        if _state_dict_sha256(recovery["state_dict"]) != metrics["model_state_sha256"]:
            raise RuntimeError(f"{role} recovery state hash differs from selector record")
        payload = _slim_checkpoint(
            state=recovery["state_dict"],
            dataset=args.dataset,
            role=role,
            epoch=role_epoch,
            metrics=metrics,
            config=config,
        )
        _validate_state(payload["state_dict"], expected_state)
        path = published_paths[role]
        if not path.exists():
            _publish_once(path, payload)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"published {role} checkpoint is not a regular file")
        try:
            reopened = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"published {role} checkpoint cannot be loaded safely"
            ) from exc
        try:
            _validate_checkpoint_payload_exact(
                reopened,
                payload,
                expected_state=expected_state,
                label=f"published {role}",
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"published {role} checkpoint exact replay failed"
            ) from exc
        checkpoint_sha = _sha256(path)
        selections[role] = {
            "epoch": role_epoch,
            "metrics": metrics,
            "role_key": role_key_record(metrics, role),
            "model_state_sha256": metrics["model_state_sha256"],
            "test_selected": True,
            "selection_is_optimistic": True,
        }
        published_records[role] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": checkpoint_sha,
            "model_state_sha256": metrics["model_state_sha256"],
        }
    if any(path.is_symlink() or not path.is_file() for path in published_paths.values()):
        raise RuntimeError("two physical selected checkpoints were not materialized")
    if len({path.stat().st_ino for path in published_paths.values()}) != 2:
        raise RuntimeError("selected checkpoints must be distinct physical files")
    contracts.write_once_json(
        summary_path,
        {
            "schema": SUMMARY_SCHEMA,
            "status": "complete",
            "model": MODEL_NAME,
            "method": METHOD_NAME,
            "dataset": args.dataset,
            "training": config,
            "candidate_count": len(history),
            "selections": selections,
            "published_checkpoints": published_records,
            "two_distinct_physical_checkpoint_files": True,
            "data_role": "test",
            "test_split_accessed": True,
            "test_selected": True,
            "selection_is_optimistic": True,
            "unbiased_test_claim_supported": False,
            "baseline_reference_kind": "historical_existing",
            "method_config_path": config["method_config_path"],
            "method_config_sha256": config["method_config_sha256"],
            "gradient_authorization_path": config[
                "gradient_authorization_path"
            ],
            "gradient_authorization_sha256": config[
                "gradient_authorization_sha256"
            ],
            "baseline_authority_manifest_path": config[
                "baseline_authority_manifest_path"
            ],
            "baseline_authority_manifest_sha256": config[
                "baseline_authority_manifest_sha256"
            ],
            "formal_launch_authorization_path": config[
                "formal_launch_authorization_path"
            ],
            "formal_launch_authorization_sha256": config[
                "formal_launch_authorization_sha256"
            ],
            "training_source_manifest_sha256": config[
                "training_source_tree_sha256"
            ],
            "selection_history": history,
            "formal_selection_provenance": final_selection_provenance,
            "elapsed_seconds": elapsed_before + time.time() - started,
        },
    )
    # Recovery checkpoints are necessary while the job is running, but the
    # user requested that only the two selected checkpoints remain after completion.
    latest_path.unlink()
    for path in best_recovery_paths.values():
        path.unlink()
    return run_dir


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()
