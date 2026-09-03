#!/usr/bin/env python3
"""Freeze the dataset-specific IRSTD formal config for C3-SBSC V3.3.

This tool reads only the frozen train/test index files and authority artifacts.
It never constructs a dataset, model, evaluator, or CUDA context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ENTRY = Path(__file__)
if ENTRY.is_symlink() or not ENTRY.is_file():
    raise RuntimeError("method-config freezer must be a regular file")
PROJECT_ROOT = ENTRY.resolve(strict=True).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import sbsc_v33_contracts as contracts
from experiments import three_dataset_v2_protocol as protocol
import train_sctransnet_sbsc_v33_img_idx_test_selected as runner


DATASET = "IRSTD-1K"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    return parser.parse_args(argv)


def _canonical_dataset_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("dataset root must be a regular non-symlink directory")
    resolved = path.resolve(strict=True)
    expected = (PROJECT_ROOT / "datasets").resolve(strict=True)
    if resolved != expected:
        raise ValueError("method config requires the repository dataset root")
    return resolved


def _split_contract(dataset_root: Path) -> dict[str, Any]:
    train_ids = protocol.load_index(dataset_root, DATASET, "train")
    test_ids = protocol.load_index(dataset_root, DATASET, "test")
    train_expected = protocol.EXPECTED_SPLITS[DATASET]["train"]
    test_expected = protocol.EXPECTED_SPLITS[DATASET]["test"]
    if set(train_ids) & set(test_ids):
        raise RuntimeError("frozen IRSTD train/test IDs overlap")
    return {
        "schema": "sctransnet_sbsc_v33/img_idx_split_contract/v1",
        "dataset": DATASET,
        "train": {
            "index_path": f"datasets/{DATASET}/img_idx/train_{DATASET}.txt",
            "count": len(train_ids),
            "file_sha256": train_expected["file_sha256"],
            "ordered_ids_sha256": train_expected["ordered_ids_sha256"],
            "runner_index_order_sha256": runner._index_sha(train_ids),
        },
        "test": {
            "index_path": f"datasets/{DATASET}/img_idx/test_{DATASET}.txt",
            "count": len(test_ids),
            "file_sha256": test_expected["file_sha256"],
            "ordered_ids_sha256": test_expected["ordered_ids_sha256"],
            "runner_index_order_sha256": runner._index_sha(test_ids),
        },
        "normalization": protocol.get_legacy_normalization(DATASET),
        "train_test_disjoint": True,
        "validation_split_used": False,
    }


def build_config(dataset_root: Path) -> dict[str, Any]:
    root = _canonical_dataset_root(dataset_root)
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    if gradient["authorized_router_value_gradient_mode"] != "live":
        raise RuntimeError("the frozen gradient authorization is not live")
    source_manifest = runner._training_source_manifest()
    split = _split_contract(root)
    return {
        "schema": "sctransnet_sbsc_v33/method_config/v1",
        "status": "frozen",
        "write_once": True,
        "run_kind": "formal",
        "protocol": runner.PROTOCOL_NAME,
        "model": runner.MODEL_NAME,
        "builder": (
            "experiments.sctransnet_sbsc_v33."
            "build_sctransnet_sbsc_v33_method"
        ),
        "method": runner.METHOD_NAME,
        "dataset": DATASET,
        "architecture_seed": 42,
        "run_seed": 42,
        "router_value_gradient_mode": "live",
        "balance_mode": "one_third_two_thirds",
        "loss_schema": runner.core.SBSC_V33_LOSS_SCHEMA,
        "segmentation_loss": "sum_of_six_BCELoss_mean_terms",
        "router_level_count": 4,
        "router_loss_weight": 1.0,
        "level_reduction": "mean",
        "optimizer": "Adam",
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
        "output_namespace": "runs/sbsc_v33_third/formal/IRSTD-1K",
        "dataset_root_recorded": False,
        "split_contract": split,
        "split_contract_sha256": contracts.canonical_sha256(split),
        "training_source_manifest": source_manifest,
        "training_source_manifest_sha256": source_manifest["sha256"],
        "gradient_authorization_path": gradient["authorization_path"],
        "gradient_authorization_sha256": gradient["authorization_sha256"],
        "baseline_authority_manifest_path": baseline["manifest_path"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args.dataset_root)
    digest = contracts.write_once_json(runner.METHOD_CONFIG_PATH, config)
    print(f"{runner.METHOD_CONFIG_PATH} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
