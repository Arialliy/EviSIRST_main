#!/usr/bin/env python3
"""Strict, write-once finalization of formal C3-SBSC V3.3 results.

This tool never evaluates a model and never chooses a checkpoint from a
partial history.  It accepts exactly one method and the three completed formal
dataset summaries, replays the frozen 501-epoch dual-role selector, verifies
both physical checkpoint publications, and writes two separate result tables:
``best_miou_rows`` and ``best_pd_rows``.  Cross-role metric splicing is not a
supported representation.

The candidate schema below is intentionally fail-closed.  A training runner
that does not persist every required identity/hash field cannot be finalized by
this tool and must be fixed at the producer rather than guessed here.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


_BOOTSTRAP_ROOT = Path(__file__).resolve(strict=True).parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import torch

from experiments import sbsc_v33_contracts as contracts
from experiments import sctransnet_sbsc_v33 as core
from experiments.four_dataset_models_seed42_v1 import state_dict_sha256


FINAL_SCHEMA = "sctransnet_sbsc_v33/formal_results_summary/v1"
METHOD_CONFIG_SCHEMA = "sctransnet_sbsc_v33/method_config/v1"
CANDIDATE_SUMMARY_SCHEMA = (
    "sctransnet_c3_sbsc_v33_img_idx_test_selected_summary/v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_c3_sbsc_v33_img_idx_test_selected_checkpoint/v1"
)
TEST_RECORD_SCHEMA = "sctransnet_sbsc_v33_test_evaluation_record/v1"
FORMAL_TRAINING_SCHEMA = "sctransnet_sbsc_v33/formal_training_config/v1"
FORMAL_PROTOCOL = (
    "sctransnet_c3_sbsc_v33_three_dataset_img_idx_test_selected/v1"
)
SOURCE_MANIFEST_SCHEMA = "sctransnet_sbsc_v33/source_manifest/v1"

MODEL_NAME = "SCTransNet-C3-SBSC-V3.3"
METHODS = ("sbsc_v33_third", "sbsc_v33_ord", "sbsc_v33_half")
ROLES = ("best_miou", "best_pd")
DATASETS = contracts.DATASETS
SELECTION_EPOCHS = tuple(range(500, 1001))
EXPECTED_COUNTS = {
    "NUAA-SIRST": {"train": 213, "test": 214},
    "NUDT-SIRST": {"train": 663, "test": 664},
    "IRSTD-1K": {"train": 800, "test": 201},
}
BALANCE_MODE = {
    "sbsc_v33_third": "one_third_two_thirds",
    "sbsc_v33_ord": "ordinary",
    "sbsc_v33_half": "half_half",
}
OUTPUT_NAMESPACE = {
    method: f"runs/{method}" for method in METHODS
}
DATASET_CONFIG_TOKEN = {
    "NUAA-SIRST": "nuaa",
    "NUDT-SIRST": "nudt",
    "IRSTD-1K": "irstd",
}
ROLE_RANKING = {
    "best_miou": (
        "miou:max",
        "pd:max",
        "fa:min",
        "niou:max",
        "tiny_pd:max",
        "test_loss:min",
        "epoch:min",
    ),
    "best_pd": (
        "pd:max",
        "fa:min",
        "tiny_pd:max",
        "miou:max",
        "niou:max",
        "test_loss:min",
        "epoch:min",
    ),
}
SERIALIZED_ROLE_RANKING = {
    role: list(order) for role, order in ROLE_RANKING.items()
}
PROBABILITY_METRICS = (
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
)
NONNEGATIVE_METRICS = ("test_loss", "fa", "false_objects_per_image")
COUNT_METRICS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
METRIC_FIELDS = (
    *PROBABILITY_METRICS,
    "tiny_pd",
    *NONNEGATIVE_METRICS,
    *COUNT_METRICS,
)
TOP_LEVEL_ALIASES = {
    "mIoU": "miou",
    "nIoU": "niou",
    "Pd": "pd",
    "Fa": "fa",
    "tinyPd": "tiny_pd",
    "loss": "test_loss",
}
HASH_BINDINGS = (
    "method_config_sha256",
    "gradient_authorization_sha256",
    "baseline_authority_manifest_sha256",
    "formal_launch_authorization_sha256",
    "training_source_manifest_sha256",
)
DISCLOSURES = {
    "data_role": "test",
    "test_split_accessed": True,
    "test_selected": True,
    "selection_is_optimistic": True,
    "unbiased_test_claim_supported": False,
}
REQUIRED_TRAINING_SOURCE_PATHS = frozenset(
    {
        "train_sctransnet_sbsc_v33_img_idx_test_selected.py",
        "train.py",
        "test.py",
        "experiments/evisirst_data.py",
        "experiments/three_dataset_v2_protocol.py",
        "experiments/sctransnet_sbsc_v33.py",
        "experiments/sbsc_v33_contracts.py",
    }
)


class SBSCV33FinalizationError(ValueError):
    """A formal candidate artifact violates the finalization contract."""


def _sha256_text(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SBSCV33FinalizationError(f"{label} must be lowercase SHA-256")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SBSCV33FinalizationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SBSCV33FinalizationError(f"{label} must be finite")
    return result


def _probability(value: Any, *, label: str) -> float:
    result = _finite_number(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise SBSCV33FinalizationError(f"{label} must be in [0,1]")
    return result


def _nonnegative(value: Any, *, label: str) -> float:
    result = _finite_number(value, label=label)
    if result < 0.0:
        raise SBSCV33FinalizationError(f"{label} must be nonnegative")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise SBSCV33FinalizationError(f"{label} must be a nonnegative int")
    return value


def _require_disclosures(value: Mapping[str, Any], *, label: str) -> None:
    for field, expected in DISCLOSURES.items():
        observed = value.get(field)
        if field == "data_role":
            differs = observed != expected
        else:
            differs = observed is not expected
        if differs:
            raise SBSCV33FinalizationError(
                f"{label}.{field} must be explicit {expected!r}"
            )


def _input_file(value: str | Path, *, label: str) -> tuple[Path, str]:
    path = Path(value)
    if path.is_absolute():
        try:
            relative = contracts.repository_relative_path(path)
        except Exception as exc:
            raise SBSCV33FinalizationError(
                f"{label} must be a repository regular file"
            ) from exc
    else:
        relative = path.as_posix()
    try:
        physical = contracts.require_repository_relative_regular_file(relative)
    except Exception as exc:
        raise SBSCV33FinalizationError(
            f"{label} must be a canonical repository-relative regular file"
        ) from exc
    return physical, relative


def _validate_source_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "files",
        "sha256",
    }:
        raise SBSCV33FinalizationError("training source manifest schema differs")
    if value.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise SBSCV33FinalizationError("training source manifest identity differs")
    records = value.get("files")
    if not isinstance(records, list) or not records:
        raise SBSCV33FinalizationError("training source records are missing")
    normalized: list[dict[str, Any]] = []
    observed_paths: list[str] = []
    for position, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise SBSCV33FinalizationError(
                f"training source record[{position}] differs"
            )
        path_text = record.get("path")
        if type(path_text) is not str:
            raise SBSCV33FinalizationError("training source path must be text")
        physical, canonical = _input_file(
            path_text, label=f"training source record[{position}]"
        )
        if canonical != path_text:
            raise SBSCV33FinalizationError("training source path is not canonical")
        expected_sha = _sha256_text(
            record.get("sha256"), label=f"source[{path_text}].sha256"
        )
        size = _nonnegative_int(
            record.get("size_bytes"), label=f"source[{path_text}].size_bytes"
        )
        if contracts.sha256_file(physical) != expected_sha:
            raise SBSCV33FinalizationError(f"training source drifted: {path_text}")
        if physical.stat().st_size != size:
            raise SBSCV33FinalizationError(
                f"training source size drifted: {path_text}"
            )
        observed_paths.append(path_text)
        normalized.append(
            {"path": path_text, "sha256": expected_sha, "size_bytes": size}
        )
    if len(observed_paths) != len(set(observed_paths)):
        raise SBSCV33FinalizationError("training source paths are duplicated")
    if not REQUIRED_TRAINING_SOURCE_PATHS.issubset(observed_paths):
        missing = sorted(REQUIRED_TRAINING_SOURCE_PATHS - set(observed_paths))
        raise SBSCV33FinalizationError(
            f"training source closure is incomplete: {missing}"
        )
    manifest_sha = _sha256_text(value.get("sha256"), label="source_manifest.sha256")
    if contracts.canonical_sha256(normalized) != manifest_sha:
        raise SBSCV33FinalizationError("training source manifest digest differs")
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "files": normalized,
        "sha256": manifest_sha,
    }


def _validate_method_split_contract(
    value: Any,
    *,
    dataset: str,
    baseline_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "dataset",
        "train",
        "test",
        "normalization",
        "train_test_disjoint",
        "validation_split_used",
    }:
        raise SBSCV33FinalizationError("method config split contract differs")
    exact = {
        "schema": "sctransnet_sbsc_v33/img_idx_split_contract/v1",
        "dataset": dataset,
        "normalization": baseline_authority["normalization"],
        "train_test_disjoint": True,
        "validation_split_used": False,
    }
    for field, expected in exact.items():
        observed = value.get(field)
        differs = (
            observed is not expected
            if type(expected) is bool
            else observed != expected
        )
        if differs:
            raise SBSCV33FinalizationError(
                f"method config split field {field!r} differs"
            )
    normalized: dict[str, Any] = {
        **dict(value),
        "normalization": dict(baseline_authority["normalization"]),
    }
    for role in ("train", "test"):
        record = value.get(role)
        if not isinstance(record, Mapping) or set(record) != {
            "index_path",
            "count",
            "file_sha256",
            "ordered_ids_sha256",
            "runner_index_order_sha256",
        }:
            raise SBSCV33FinalizationError(
                f"method config {role} split record differs"
            )
        expected_path = f"datasets/{dataset}/img_idx/{role}_{dataset}.txt"
        if record.get("index_path") != expected_path:
            raise SBSCV33FinalizationError(
                f"method config {role} index path differs"
            )
        expected_count = EXPECTED_COUNTS[dataset][role]
        if type(record.get("count")) is not int or record["count"] != expected_count:
            raise SBSCV33FinalizationError(
                f"method config {role} split count differs"
            )
        physical, canonical = _input_file(
            expected_path, label=f"method config {role} index"
        )
        if canonical != expected_path:
            raise SBSCV33FinalizationError(
                f"method config {role} index path is not canonical"
            )
        file_sha = _sha256_text(
            record.get("file_sha256"), label=f"split_contract.{role}.file_sha256"
        )
        if contracts.sha256_file(physical) != file_sha:
            raise SBSCV33FinalizationError(
                f"method config {role} index file drifted"
            )
        for field in ("ordered_ids_sha256", "runner_index_order_sha256"):
            _sha256_text(
                record.get(field), label=f"split_contract.{role}.{field}"
            )
        normalized[role] = dict(record)
    return normalized


def _validate_method_config(
    path: str | Path,
    *,
    dataset: str,
    authorization: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    physical, relative = _input_file(path, label="method config")
    config = contracts.load_strict_json(physical)
    required = {
        "schema",
        "status",
        "write_once",
        "run_kind",
        "protocol",
        "model",
        "builder",
        "method",
        "dataset",
        "balance_mode",
        "loss_schema",
        "segmentation_loss",
        "router_level_count",
        "level_reduction",
        "router_loss_weight",
        "optimizer",
        "architecture_seed",
        "run_seed",
        "router_value_gradient_mode",
        "epochs",
        "batch_size",
        "patch_size",
        "workers",
        "base_lr",
        "min_lr",
        "warmup_epochs",
        "amp",
        "selection_begin_epoch",
        "selection_end_epoch",
        "selection_every",
        "selection_roles",
        "evaluator",
        "probability_threshold",
        "probability_comparison",
        "component_match_radius",
        "component_match_comparison",
        "tiny_area_max",
        "gradient_authorization_path",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_path",
        "baseline_authority_manifest_sha256",
        "training_source_manifest",
        "training_source_manifest_sha256",
        "output_namespace",
        "dataset_root_recorded",
        "split_contract",
        "split_contract_sha256",
        "test_selected",
        "selection_is_optimistic",
        "unbiased_test_claim_supported",
    }
    if set(config) != required:
        raise SBSCV33FinalizationError("method config field set differs")
    method = config.get("method")
    if method not in METHODS:
        raise SBSCV33FinalizationError("method config method differs")
    expected_relative = (
        "experiments/sbsc_v33_methods/"
        f"{method}_{DATASET_CONFIG_TOKEN[dataset]}_formal.json"
    )
    if relative != expected_relative:
        raise SBSCV33FinalizationError("method config path differs")
    authority = baseline["authorities"][dataset]
    exact = {
        "schema": METHOD_CONFIG_SCHEMA,
        "status": "frozen",
        "write_once": True,
        "run_kind": "formal",
        "protocol": FORMAL_PROTOCOL,
        "model": MODEL_NAME,
        "builder": (
            "experiments.sctransnet_sbsc_v33."
            "build_sctransnet_sbsc_v33_method"
        ),
        "dataset": dataset,
        "balance_mode": BALANCE_MODE[method],
        "loss_schema": core.SBSC_V33_LOSS_SCHEMA,
        "segmentation_loss": "sum_of_six_BCELoss_mean_terms",
        "router_level_count": 4,
        "level_reduction": "mean",
        "router_loss_weight": 1.0,
        "optimizer": "Adam",
        "architecture_seed": 42,
        "run_seed": 42,
        "router_value_gradient_mode": authorization[
            "authorized_router_value_gradient_mode"
        ],
        "gradient_authorization_path": authorization["authorization_path"],
        "gradient_authorization_sha256": authorization["authorization_sha256"],
        "baseline_authority_manifest_path": baseline["manifest_path"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "epochs": 1000,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "amp": False,
        "selection_begin_epoch": 500,
        "selection_end_epoch": 1000,
        "selection_every": 1,
        "selection_roles": SERIALIZED_ROLE_RANKING,
        "evaluator": authority["protocol"],
        "probability_threshold": authority["threshold"],
        "probability_comparison": authority["threshold_operator"],
        "component_match_radius": authority["match_radius"],
        "component_match_comparison": authority["match_radius_operator"],
        "tiny_area_max": authority["tiny_area"],
        "output_namespace": f"{OUTPUT_NAMESPACE[method]}/formal/{dataset}",
        "dataset_root_recorded": False,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
    }
    for key, expected in exact.items():
        observed = config.get(key)
        if type(expected) is bool:
            differs = observed is not expected
        elif key in (
            "router_level_count",
            "architecture_seed",
            "run_seed",
            "epochs",
            "batch_size",
            "patch_size",
            "workers",
            "warmup_epochs",
            "selection_begin_epoch",
            "selection_end_epoch",
            "selection_every",
        ):
            differs = type(observed) is not int or observed != expected
        elif key in ("router_loss_weight", "base_lr", "min_lr"):
            differs = (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or float(observed) != expected
            )
        else:
            differs = observed != expected
        if differs:
            raise SBSCV33FinalizationError(
                f"method config field {key!r} differs"
            )
    source = _validate_source_manifest(config.get("training_source_manifest"))
    if config.get("training_source_manifest_sha256") != source["sha256"]:
        raise SBSCV33FinalizationError(
            "method config training source hash binding differs"
        )
    split = _validate_method_split_contract(
        config.get("split_contract"),
        dataset=dataset,
        baseline_authority=authority,
    )
    split_sha = _sha256_text(
        config.get("split_contract_sha256"),
        label="method_config.split_contract_sha256",
    )
    if contracts.canonical_sha256(split) != split_sha:
        raise SBSCV33FinalizationError("method config split digest differs")
    return {
        **dict(config),
        "split_contract": split,
        "split_contract_sha256": split_sha,
        "training_source_manifest": source,
        "training_source_manifest_sha256": source["sha256"],
        "path": relative,
        "sha256": contracts.sha256_file(physical),
    }


def _evaluation_contract(
    *, dataset: str, baseline_authority: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "sample_count": baseline_authority["sample_count"],
        "protocol": baseline_authority["protocol"],
        "threshold": baseline_authority["threshold"],
        "threshold_operator": baseline_authority["threshold_operator"],
        "match_radius": baseline_authority["match_radius"],
        "match_radius_operator": baseline_authority[
            "match_radius_operator"
        ],
        "tiny_area": baseline_authority["tiny_area"],
        "normalization": dict(baseline_authority["normalization"]),
    }


def _validate_formal_launch_authorization(
    *,
    path_value: Any,
    sha_value: Any,
    dataset: str,
    method_config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    baseline: Mapping[str, Any],
    training_source_manifest_sha256: str,
) -> dict[str, str]:
    if type(path_value) is not str:
        raise SBSCV33FinalizationError(
            "formal launch authorization path must be text"
        )
    physical, relative = _input_file(
        path_value, label="formal launch authorization"
    )
    if relative != path_value:
        raise SBSCV33FinalizationError(
            "formal launch authorization path is not canonical"
        )
    expected_sha = _sha256_text(
        sha_value, label="formal_launch_authorization_sha256"
    )
    if contracts.sha256_file(physical) != expected_sha:
        raise SBSCV33FinalizationError(
            "formal launch authorization physical SHA-256 differs"
        )
    launch = contracts.load_strict_json(physical)
    exact = {
        "schema": "sctransnet_sbsc_v33/formal_launch_authorization/v1",
        "status": "PASS",
        "write_once": True,
        "authorized_run": {
            "method": method_config["method"],
            "dataset": dataset,
            "run_kind": "formal",
            "output_namespace": method_config["output_namespace"],
        },
        "method_config_path": method_config["path"],
        "method_config_sha256": method_config["sha256"],
        "gradient_authorization_sha256": authorization[
            "authorization_sha256"
        ],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "training_source_manifest_sha256": training_source_manifest_sha256,
    }
    for field, expected in exact.items():
        observed = launch.get(field)
        differs = (
            observed is not expected
            if type(expected) is bool
            else observed != expected
        )
        if differs:
            raise SBSCV33FinalizationError(
                f"formal launch authorization field {field!r} differs"
            )
    return {"path": relative, "sha256": expected_sha}


def _validate_training_contract(
    value: Any,
    *,
    dataset: str,
    method_config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SBSCV33FinalizationError("candidate training contract is missing")
    authority = baseline["authorities"][dataset]
    evaluation = _evaluation_contract(
        dataset=dataset, baseline_authority=authority
    )
    evaluation_sha = contracts.canonical_sha256(evaluation)
    exact = {
        "schema": FORMAL_TRAINING_SCHEMA,
        "run_kind": "formal",
        "protocol": FORMAL_PROTOCOL,
        "model": MODEL_NAME,
        "method": method_config["method"],
        "dataset": dataset,
        "architecture_seed": 42,
        "run_seed": 42,
        "seed": 42,
        "epochs": 1000,
        "selection_begin_epoch": 500,
        "selection_end_epoch": 1000,
        "selection_every": 1,
        "train_count": EXPECTED_COUNTS[dataset]["train"],
        "test_count": EXPECTED_COUNTS[dataset]["test"],
        "evaluator": authority["protocol"],
        "probability_threshold": authority["threshold"],
        "probability_comparison": authority["threshold_operator"],
        "component_match_radius": authority["match_radius"],
        "component_match_comparison": authority["match_radius_operator"],
        "tiny_area_max": authority["tiny_area"],
        "normalization": authority["normalization"],
        "method_config_path": method_config["path"],
        "method_config_sha256": method_config["sha256"],
        "gradient_authorization_path": authorization["authorization_path"],
        "gradient_authorization_sha256": authorization["authorization_sha256"],
        "baseline_authority_manifest_path": baseline["manifest_path"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "router_value_gradient_mode": method_config[
            "router_value_gradient_mode"
        ],
        "balance_mode": method_config["balance_mode"],
        "router_level_count": 4,
        "level_reduction": "mean",
        "router_loss_weight": 1.0,
        "evaluation_contract_sha256": evaluation_sha,
        **DISCLOSURES,
    }
    for key, expected in exact.items():
        observed = value.get(key)
        if type(expected) is bool:
            differs = observed is not expected
        elif type(expected) is int:
            differs = type(observed) is not int or observed != expected
        elif isinstance(expected, float):
            differs = (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or float(observed) != expected
            )
        else:
            differs = observed != expected
        if differs:
            raise SBSCV33FinalizationError(
                f"candidate training field {key!r} differs"
            )
    if value.get("evaluation_contract") != evaluation:
        raise SBSCV33FinalizationError(
            "candidate evaluation contract payload differs"
        )
    if value.get("selection_roles") != SERIALIZED_ROLE_RANKING:
        raise SBSCV33FinalizationError("candidate selection ranking differs")
    split = value.get("split_manifest")
    if not isinstance(split, Mapping):
        raise SBSCV33FinalizationError("candidate split manifest is missing")
    split_exact = {
        "dataset": dataset,
        "train_count": EXPECTED_COUNTS[dataset]["train"],
        "test_count": EXPECTED_COUNTS[dataset]["test"],
        "normalization": authority["normalization"],
        "training_target_rule": "raw_mask_div_255",
    }
    for key, expected in split_exact.items():
        if split.get(key) != expected:
            raise SBSCV33FinalizationError(
                f"candidate split field {key!r} differs"
            )
    for key in ("train_index_order_sha256", "test_index_order_sha256"):
        _sha256_text(split.get(key), label=f"split_manifest.{key}")
    if (
        split.get("train_index_order_sha256")
        != method_config["split_contract"]["train"][
            "runner_index_order_sha256"
        ]
        or split.get("test_index_order_sha256")
        != method_config["split_contract"]["test"][
            "runner_index_order_sha256"
        ]
    ):
        raise SBSCV33FinalizationError(
            "candidate split order differs from frozen method config"
        )
    split_sha = _sha256_text(
        value.get("split_manifest_sha256"), label="split_manifest_sha256"
    )
    if contracts.canonical_sha256(dict(split)) != split_sha:
        raise SBSCV33FinalizationError("candidate split manifest digest differs")
    run_identity = _sha256_text(
        value.get("run_identity_sha256"), label="run_identity_sha256"
    )
    selection_identity = {
        "method": method_config["method"],
        "dataset": dataset,
        "architecture_seed": 42,
        "run_seed": 42,
        "split_manifest_sha256": split_sha,
        "run_identity_sha256": run_identity,
        "evaluation_contract_sha256": evaluation_sha,
        **{field: value.get(field) for field in HASH_BINDINGS},
    }
    if value.get("selection_identity") != selection_identity:
        raise SBSCV33FinalizationError(
            "candidate selection identity differs from training bindings"
        )
    source = _validate_source_manifest(value.get("training_source_manifest"))
    if value.get("training_source_manifest_sha256") != source["sha256"]:
        raise SBSCV33FinalizationError("candidate source hash binding differs")
    if (
        source != method_config["training_source_manifest"]
        or source["sha256"]
        != method_config["training_source_manifest_sha256"]
    ):
        raise SBSCV33FinalizationError(
            "candidate source closure differs from frozen method config"
        )
    evaluation_source = next(
        (record for record in source["files"] if record["path"] == "test.py"),
        None,
    )
    if (
        evaluation_source is None
        or value.get("evaluation_source_sha256")
        != evaluation_source["sha256"]
    ):
        raise SBSCV33FinalizationError(
            "candidate evaluator source hash differs from frozen source closure"
        )
    launch = _validate_formal_launch_authorization(
        path_value=value.get("formal_launch_authorization_path"),
        sha_value=value.get("formal_launch_authorization_sha256"),
        dataset=dataset,
        method_config=method_config,
        authorization=authorization,
        baseline=baseline,
        training_source_manifest_sha256=source["sha256"],
    )
    return {
        **dict(value),
        "split_manifest": dict(split),
        "split_manifest_sha256": split_sha,
        "run_identity_sha256": run_identity,
        "selection_identity": selection_identity,
        "training_source_manifest": source,
        "training_source_manifest_sha256": source["sha256"],
        "formal_launch_authorization_path": launch["path"],
        "formal_launch_authorization_sha256": launch["sha256"],
        "evaluation_contract": evaluation,
        "evaluation_contract_sha256": evaluation_sha,
        "evaluation_source_sha256": evaluation_source["sha256"],
    }


def _validate_metric_vector(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        name: _probability(value.get(name), label=f"{label}.{name}")
        for name in PROBABILITY_METRICS
    }
    tiny_value = value.get("tiny_pd")
    metrics["tiny_pd"] = (
        None
        if tiny_value is None
        else _probability(tiny_value, label=f"{label}.tiny_pd")
    )
    for name in NONNEGATIVE_METRICS:
        metrics[name] = _nonnegative(value.get(name), label=f"{label}.{name}")
    for name in COUNT_METRICS:
        metrics[name] = _nonnegative_int(
            value.get(name), label=f"{label}.{name}"
        )
    if metrics["matched_target_count"] > metrics["target_count"]:
        raise SBSCV33FinalizationError(f"{label} target counts are inconsistent")
    if metrics["matched_tiny_target_count"] > metrics["tiny_target_count"]:
        raise SBSCV33FinalizationError(f"{label} tiny counts are inconsistent")
    if (
        metrics["unmatched_predicted_object_count"]
        > metrics["predicted_object_count"]
    ):
        raise SBSCV33FinalizationError(
            f"{label} prediction counts are inconsistent"
        )
    undefined = value.get("tiny_pd_was_undefined")
    if type(undefined) is not bool or undefined is not (metrics["tiny_pd"] is None):
        raise SBSCV33FinalizationError(
            f"{label}.tiny_pd_was_undefined differs"
        )
    if (metrics["tiny_target_count"] == 0) is not (metrics["tiny_pd"] is None):
        raise SBSCV33FinalizationError(
            f"{label} tiny-Pd denominator/null contract differs"
        )
    if metrics["target_count"] <= 0:
        raise SBSCV33FinalizationError(f"{label} target_count must be positive")
    expected_pd = metrics["matched_target_count"] / metrics["target_count"]
    if not math.isclose(metrics["pd"], expected_pd, rel_tol=0.0, abs_tol=1e-12):
        raise SBSCV33FinalizationError(f"{label}.pd differs from counts")
    if metrics["tiny_pd"] is not None:
        expected_tiny = (
            metrics["matched_tiny_target_count"] / metrics["tiny_target_count"]
        )
        if not math.isclose(
            metrics["tiny_pd"], expected_tiny, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SBSCV33FinalizationError(
                f"{label}.tiny_pd differs from counts"
            )
    precision = metrics["pixel_precision"]
    recall = metrics["pixel_recall"]
    expected_f1 = 0.0 if precision + recall == 0.0 else (
        2.0 * precision * recall / (precision + recall)
    )
    if not math.isclose(
        metrics["pixel_f1"], expected_f1, rel_tol=0.0, abs_tol=1e-10
    ):
        raise SBSCV33FinalizationError(f"{label}.pixel_f1 is inconsistent")
    for alias, canonical in TOP_LEVEL_ALIASES.items():
        if value.get(alias) != metrics[canonical]:
            raise SBSCV33FinalizationError(
                f"{label}.{alias} conflicts with {canonical}"
            )
    return {
        **metrics,
        "tiny_pd_was_undefined": undefined,
        **{alias: metrics[name] for alias, name in TOP_LEVEL_ALIASES.items()},
    }


def _history_identity(
    *,
    dataset: str,
    method_config: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "method": method_config["method"],
        "dataset": dataset,
        "architecture_seed": 42,
        "run_seed": 42,
        "seed": 42,
        "split_manifest_sha256": training["split_manifest_sha256"],
        "run_identity_sha256": training["run_identity_sha256"],
        "evaluation_contract_sha256": training["evaluation_contract_sha256"],
        "evaluation_source_sha256": training["evaluation_source_sha256"],
        **{field: training[field] for field in HASH_BINDINGS},
    }


def _validate_history(
    value: Any,
    *,
    dataset: str,
    method_config: Mapping[str, Any],
    training: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(SELECTION_EPOCHS):
        raise SBSCV33FinalizationError(
            "candidate history must contain exactly 501 records"
        )
    identity = _history_identity(
        dataset=dataset, method_config=method_config, training=training
    )
    normalized: list[dict[str, Any]] = []
    observed_epochs: list[int] = []
    expected_fields = {
        "schema",
        "epoch",
        "sample_count",
        "model_state_sha256",
        *DISCLOSURES,
        *identity,
        *METRIC_FIELDS,
        "tiny_pd_was_undefined",
        *TOP_LEVEL_ALIASES,
    }
    for position, record in enumerate(value):
        label = f"selection_history[{position}]"
        if not isinstance(record, Mapping):
            raise SBSCV33FinalizationError(f"{label} must be a mapping")
        if set(record) != expected_fields:
            raise SBSCV33FinalizationError(
                f"{label} field set differs from the frozen 17-metric schema"
            )
        if record.get("schema") != TEST_RECORD_SCHEMA:
            raise SBSCV33FinalizationError(f"{label} schema differs")
        epoch = record.get("epoch")
        if type(epoch) is not int:
            raise SBSCV33FinalizationError(f"{label}.epoch must be an int")
        observed_epochs.append(epoch)
        _require_disclosures(record, label=label)
        for field, expected in identity.items():
            if record.get(field) != expected:
                raise SBSCV33FinalizationError(
                    f"{label}.{field} differs from run identity"
                )
        if record.get("sample_count") != EXPECTED_COUNTS[dataset]["test"]:
            raise SBSCV33FinalizationError(f"{label}.sample_count differs")
        state_sha = _sha256_text(
            record.get("model_state_sha256"), label=f"{label}.model_state_sha256"
        )
        metrics = _validate_metric_vector(record, label=label)
        normalized.append(
            {
                **dict(record),
                **metrics,
                "epoch": epoch,
                "model_state_sha256": state_sha,
            }
        )
    if tuple(observed_epochs) != SELECTION_EPOCHS:
        raise SBSCV33FinalizationError(
            "candidate history epochs must be exactly continuous 500..1000"
        )
    return normalized


def role_key(row: Mapping[str, Any], role: str) -> tuple[float, ...]:
    tiny = float("-inf") if row["tiny_pd"] is None else float(row["tiny_pd"])
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
    raise SBSCV33FinalizationError(f"unsupported role: {role!r}")


def role_key_record(row: Mapping[str, Any], role: str) -> list[float | None]:
    return [value if math.isfinite(value) else None for value in role_key(row, role)]


def _metric_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: row[name]
        for name in (*METRIC_FIELDS, "tiny_pd_was_undefined", *TOP_LEVEL_ALIASES)
    }


def _validate_selections(
    value: Any, history: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(ROLES):
        raise SBSCV33FinalizationError("candidate selections must contain two roles")
    replayed: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        winner = max(history, key=lambda row: role_key(row, role))
        stored = value[role]
        if not isinstance(stored, Mapping) or set(stored) != {
            "epoch",
            "model_state_sha256",
            "role_key",
            "metrics",
            "test_selected",
            "selection_is_optimistic",
        }:
            raise SBSCV33FinalizationError(f"selection {role} is malformed")
        expected_key = role_key_record(winner, role)
        if (
            stored.get("epoch") != winner["epoch"]
            or stored.get("model_state_sha256") != winner["model_state_sha256"]
            or stored.get("role_key") != expected_key
            or stored.get("metrics") != dict(winner)
            or stored.get("test_selected") is not True
            or stored.get("selection_is_optimistic") is not True
        ):
            raise SBSCV33FinalizationError(
                f"stored {role} selection differs from replayed history"
            )
        replayed[role] = {
            "epoch": int(winner["epoch"]),
            "model_state_sha256": winner["model_state_sha256"],
            "role_key": expected_key,
            "metrics": dict(winner),
        }
    return replayed


def _load_checkpoint(
    *,
    record: Any,
    role: str,
    dataset: str,
    summary_parent: str,
    selected: Mapping[str, Any],
    training: Mapping[str, Any],
    method_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "model_state_sha256",
    }:
        raise SBSCV33FinalizationError(f"published {role} record differs")
    path_text = record.get("path")
    if type(path_text) is not str:
        raise SBSCV33FinalizationError(f"published {role} path must be text")
    physical, relative = _input_file(path_text, label=f"published {role} checkpoint")
    if relative != path_text or PurePosixPath(relative).parent.as_posix() != summary_parent:
        raise SBSCV33FinalizationError(
            f"published {role} checkpoint must be beside its summary"
        )
    file_sha = _sha256_text(record.get("sha256"), label=f"{role}.sha256")
    if contracts.sha256_file(physical) != file_sha:
        raise SBSCV33FinalizationError(f"published {role} file SHA-256 differs")
    if record.get("model_state_sha256") != selected["model_state_sha256"]:
        raise SBSCV33FinalizationError(f"published {role} state binding differs")
    try:
        checkpoint = torch.load(physical, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SBSCV33FinalizationError(
            f"published {role} checkpoint cannot be loaded safely"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise SBSCV33FinalizationError(f"published {role} checkpoint is malformed")
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "model": MODEL_NAME,
        "method": method_config["method"],
        "dataset": dataset,
        "checkpoint_role": role,
        "epoch": selected["epoch"],
        "selection_epoch": selected["epoch"],
        "seed": 42,
        "model_state_sha256": selected["model_state_sha256"],
        "method_config_path": method_config["path"],
        "method_config_sha256": method_config["sha256"],
        "gradient_authorization_sha256": training[
            "gradient_authorization_sha256"
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
        "data_role": "test",
        "test_split_accessed": True,
        "test_selected": True,
        "this_checkpoint_selected_by_test": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
    }
    for key, expected_value in expected.items():
        observed = checkpoint.get(key)
        if type(expected_value) is bool:
            differs = observed is not expected_value
        elif type(expected_value) is int:
            differs = type(observed) is not int or observed != expected_value
        else:
            differs = observed != expected_value
        if differs:
            raise SBSCV33FinalizationError(
                f"published {role} checkpoint field {key!r} differs"
            )
    metric_name = "global_foreground_mIoU" if role == "best_miou" else "Pd"
    score = (
        selected["metrics"]["miou"]
        if role == "best_miou"
        else selected["metrics"]["pd"]
    )
    if (
        checkpoint.get("selection_metric") != metric_name
        or checkpoint.get("selection_score") != score
        or checkpoint.get("selection_role_key") != selected["role_key"]
        or checkpoint.get("selection_metrics") != selected["metrics"]
        or checkpoint.get("training") != dict(training)
    ):
        raise SBSCV33FinalizationError(
            f"published {role} checkpoint selection/training binding differs"
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise SBSCV33FinalizationError(f"published {role} state_dict is missing")
    try:
        core.validate_sbsc_v33_state_dict(state, method_config["method"])
        observed_state_sha = state_dict_sha256(state)
    except Exception as exc:
        raise SBSCV33FinalizationError(
            f"published {role} state_dict violates V3.3 architecture"
        ) from exc
    if observed_state_sha != selected["model_state_sha256"]:
        raise SBSCV33FinalizationError(
            f"published {role} tensor hash differs from history"
        )
    return {
        "path": relative,
        "sha256": file_sha,
        "model_state_sha256": observed_state_sha,
        "epoch": selected["epoch"],
        "physical_device": physical.stat().st_dev,
        "physical_inode": physical.stat().st_ino,
    }


def _validate_candidate_summary(
    path: str | Path,
    *,
    dataset: str,
    method_config: Mapping[str, Any],
    authorization: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    physical, relative = _input_file(path, label=f"{dataset} candidate summary")
    expected_summary_path = f"{method_config['output_namespace']}/summary.json"
    if relative != expected_summary_path:
        raise SBSCV33FinalizationError(
            f"{dataset} summary path differs from its frozen output namespace"
        )
    summary = contracts.load_strict_json(physical)
    exact = {
        "schema": CANDIDATE_SUMMARY_SCHEMA,
        "status": "complete",
        "model": MODEL_NAME,
        "method": method_config["method"],
        "dataset": dataset,
        "candidate_count": 501,
        "two_distinct_physical_checkpoint_files": True,
        "method_config_path": method_config["path"],
        "method_config_sha256": method_config["sha256"],
        "gradient_authorization_path": authorization["authorization_path"],
        "gradient_authorization_sha256": authorization["authorization_sha256"],
        "baseline_authority_manifest_path": baseline["manifest_path"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        **DISCLOSURES,
    }
    for key, expected in exact.items():
        observed = summary.get(key)
        if type(expected) is bool:
            differs = observed is not expected
        elif type(expected) is int:
            differs = type(observed) is not int or observed != expected
        else:
            differs = observed != expected
        if differs:
            raise SBSCV33FinalizationError(
                f"{dataset} summary field {key!r} differs"
            )
    training = _validate_training_contract(
        summary.get("training"),
        dataset=dataset,
        method_config=method_config,
        authorization=authorization,
        baseline=baseline,
    )
    if summary.get("training_source_manifest_sha256") != training[
        "training_source_manifest_sha256"
    ]:
        raise SBSCV33FinalizationError(
            f"{dataset} summary source hash binding differs"
        )
    if (
        summary.get("formal_launch_authorization_path")
        != training["formal_launch_authorization_path"]
        or summary.get("formal_launch_authorization_sha256")
        != training["formal_launch_authorization_sha256"]
    ):
        raise SBSCV33FinalizationError(
            f"{dataset} summary formal launch binding differs"
        )
    history = _validate_history(
        summary.get("selection_history"),
        dataset=dataset,
        method_config=method_config,
        training=training,
    )
    selections = _validate_selections(summary.get("selections"), history)
    published = summary.get("published_checkpoints")
    if not isinstance(published, Mapping) or set(published) != set(ROLES):
        raise SBSCV33FinalizationError(
            f"{dataset} published checkpoints must contain exactly two roles"
        )
    summary_parent = PurePosixPath(relative).parent.as_posix()
    checkpoints = {
        role: _load_checkpoint(
            record=published[role],
            role=role,
            dataset=dataset,
            summary_parent=summary_parent,
            selected=selections[role],
            training=training,
            method_config=method_config,
        )
        for role in ROLES
    }
    left = checkpoints["best_miou"]
    right = checkpoints["best_pd"]
    if left["path"] == right["path"] or (
        left["physical_device"], left["physical_inode"]
    ) == (right["physical_device"], right["physical_inode"]):
        raise SBSCV33FinalizationError(
            f"{dataset} best_miou and best_pd must be independent physical files"
        )
    return {
        "path": relative,
        "sha256": contracts.sha256_file(physical),
        "dataset": dataset,
        "training": training,
        "history": history,
        "selections": selections,
        "checkpoints": checkpoints,
    }


def _row_for_role(
    *,
    role: str,
    candidate: Mapping[str, Any],
    baseline_authority: Mapping[str, Any],
) -> dict[str, Any]:
    selected = candidate["selections"][role]
    candidate_metrics = _metric_record(selected["metrics"])
    baseline_metrics = dict(baseline_authority["metrics"])
    common = (
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "fa",
        "test_loss",
        "false_objects_per_image",
    )
    deltas = {
        name: float(candidate_metrics[name]) - float(baseline_metrics[name])
        for name in common
    }
    tiny_delta = None
    if candidate_metrics["tiny_pd"] is not None and baseline_metrics["tiny_pd"] is not None:
        tiny_delta = float(candidate_metrics["tiny_pd"]) - float(
            baseline_metrics["tiny_pd"]
        )
    deltas["tiny_pd"] = tiny_delta
    checkpoint = dict(candidate["checkpoints"][role])
    checkpoint.pop("physical_device", None)
    checkpoint.pop("physical_inode", None)
    return {
        "dataset": candidate["dataset"],
        "checkpoint_role": role,
        "selection_basis": ROLE_RANKING[role],
        "epoch": selected["epoch"],
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "candidate_minus_baseline": deltas,
        "candidate_checkpoint": checkpoint,
        "baseline_checkpoint": dict(baseline_authority["checkpoint"]),
        "candidate_summary_path": candidate["path"],
        "candidate_summary_sha256": candidate["sha256"],
        "metrics_are_from_one_physical_checkpoint": True,
        "cross_role_metric_splicing": False,
        **DISCLOSURES,
    }


def build_final_payload(
    *,
    method_config_paths: Mapping[str, str | Path],
    candidate_summary_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    if not isinstance(method_config_paths, Mapping) or set(
        method_config_paths
    ) != set(DATASETS):
        raise SBSCV33FinalizationError(
            "method configs must contain exactly the three datasets"
        )
    if len({str(value) for value in method_config_paths.values()}) != len(DATASETS):
        raise SBSCV33FinalizationError("method config paths must be distinct")
    if not isinstance(candidate_summary_paths, Mapping) or set(
        candidate_summary_paths
    ) != set(DATASETS):
        raise SBSCV33FinalizationError(
            "candidate summaries must contain exactly the three datasets"
        )
    if len({str(value) for value in candidate_summary_paths.values()}) != len(DATASETS):
        raise SBSCV33FinalizationError("candidate summary paths must be distinct")
    try:
        authorization = contracts.load_gradient_authorization()
        baseline = contracts.load_baseline_authority_manifest()
    except Exception as exc:
        raise SBSCV33FinalizationError(
            "frozen authorization/baseline authority validation failed"
        ) from exc
    method_configs = {
        dataset: _validate_method_config(
            method_config_paths[dataset],
            dataset=dataset,
            authorization=authorization,
            baseline=baseline,
        )
        for dataset in DATASETS
    }
    methods = {method_configs[dataset]["method"] for dataset in DATASETS}
    if len(methods) != 1:
        raise SBSCV33FinalizationError(
            "three dataset-specific configs must describe one method"
        )
    method = next(iter(methods))
    source_config_hashes = {
        method_configs[dataset]["training_source_manifest_sha256"]
        for dataset in DATASETS
    }
    if len(source_config_hashes) != 1:
        raise SBSCV33FinalizationError(
            "three method configs do not freeze one source closure"
        )
    candidates = {
        dataset: _validate_candidate_summary(
            candidate_summary_paths[dataset],
            dataset=dataset,
            method_config=method_configs[dataset],
            authorization=authorization,
            baseline=baseline,
        )
        for dataset in DATASETS
    }
    source_hashes = {
        candidates[dataset]["training"]["training_source_manifest_sha256"]
        for dataset in DATASETS
    }
    if len(source_hashes) != 1:
        raise SBSCV33FinalizationError(
            "three datasets were not trained from one frozen source closure"
        )
    best_miou_rows = [
        _row_for_role(
            role="best_miou",
            candidate=candidates[dataset],
            baseline_authority=baseline["authorities"][dataset],
        )
        for dataset in DATASETS
    ]
    best_pd_rows = [
        _row_for_role(
            role="best_pd",
            candidate=candidates[dataset],
            baseline_authority=baseline["authorities"][dataset],
        )
        for dataset in DATASETS
    ]
    finalizer_path = Path(__file__).resolve(strict=True)
    return {
        "schema": FINAL_SCHEMA,
        "status": "complete",
        "model": MODEL_NAME,
        "method": method,
        "dataset_order": list(DATASETS),
        "checkpoint_roles": list(ROLES),
        "role_rows_are_separate": True,
        "cross_role_metric_splicing": False,
        "history_epoch_range": [500, 1000],
        "history_record_count_per_dataset": 501,
        "method_configs": {
            dataset: {
                "path": method_configs[dataset]["path"],
                "sha256": method_configs[dataset]["sha256"],
            }
            for dataset in DATASETS
        },
        "gradient_authorization": {
            "path": authorization["authorization_path"],
            "sha256": authorization["authorization_sha256"],
            "authorized_router_value_gradient_mode": authorization[
                "authorized_router_value_gradient_mode"
            ],
        },
        "baseline_authority_manifest": {
            "path": baseline["manifest_path"],
            "sha256": baseline["manifest_sha256"],
        },
        "formal_launch_authorizations": {
            dataset: {
                "path": candidates[dataset]["training"][
                    "formal_launch_authorization_path"
                ],
                "sha256": candidates[dataset]["training"][
                    "formal_launch_authorization_sha256"
                ],
            }
            for dataset in DATASETS
        },
        "training_source_manifest_sha256": next(iter(source_hashes)),
        "candidate_summaries": {
            dataset: {
                "path": candidates[dataset]["path"],
                "sha256": candidates[dataset]["sha256"],
            }
            for dataset in DATASETS
        },
        "best_miou_rows": best_miou_rows,
        "best_pd_rows": best_pd_rows,
        "result_row_count": 6,
        "finalizer_source": {
            "path": contracts.repository_relative_path(finalizer_path),
            "sha256": contracts.sha256_file(finalizer_path),
        },
        **DISCLOSURES,
        "write_once": True,
    }


def finalize_results(
    *,
    method_config_paths: Mapping[str, str | Path],
    candidate_summary_paths: Mapping[str, str | Path],
    output_path: str | Path,
) -> tuple[Path, str]:
    payload = build_final_payload(
        method_config_paths=method_config_paths,
        candidate_summary_paths=candidate_summary_paths,
    )
    try:
        destination = contracts.validated_output_path(Path(output_path))
        digest = contracts.write_once_json(destination, payload)
    except Exception as exc:
        if isinstance(exc, SBSCV33FinalizationError):
            raise
        raise SBSCV33FinalizationError("write-once final publication failed") from exc
    return destination, digest


def _dataset_assignment(value: str) -> tuple[str, str]:
    dataset, separator, path = value.partition("=")
    if not separator or dataset not in DATASETS or not path:
        raise argparse.ArgumentTypeError(
            "candidate summary must be DATASET=repository/relative/summary.json"
        )
    return dataset, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-config",
        action="append",
        type=_dataset_assignment,
        required=True,
        help="repeat DATASET=path exactly once for all three datasets",
    )
    parser.add_argument(
        "--candidate-summary",
        action="append",
        type=_dataset_assignment,
        required=True,
        help="repeat exactly once for each of the three datasets",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    for option, assignments in (
        ("--method-config", args.method_config),
        ("--candidate-summary", args.candidate_summary),
    ):
        if len(assignments) != len(DATASETS) or {
            key for key, _ in assignments
        } != set(DATASETS):
            parser.error(f"{option} must name each dataset exactly once")
        if len({key for key, _ in assignments}) != len(assignments):
            parser.error(f"{option} dataset is duplicated")
    args.method_config_paths = dict(args.method_config)
    args.candidate_summary_paths = dict(args.candidate_summary)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    path, digest = finalize_results(
        method_config_paths=args.method_config_paths,
        candidate_summary_paths=args.candidate_summary_paths,
        output_path=args.output,
    )
    print(f"{path} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
