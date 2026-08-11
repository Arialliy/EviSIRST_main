#!/usr/bin/env python3
"""Reproduce SCTransNet-style SIRST3 test-mIoU checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.evisirst_data import (
    SIRST3_NORMALIZATION,
    SIRST3_SPLITS,
    SOURCE_DATASETS,
    sirst3_contract,
)
from run_sirst3_experiment import (
    PROJECT_ROOT,
    _copy_once,
    _json_bytes,
    _sha256,
    _write_bytes_atomic,
)
from train_sirst3_test_selected import (
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_SELECTION_TREE_SHA256,
    EXPECTED_TRAIN_TREE_SHA256,
    EXPECTED_TRAIN_SAMPLES,
    FORMAL_BATCH_SIZE,
    FORMAL_BEGIN_SELECT,
    FORMAL_EPOCHS,
    FORMAL_PATCH_SIZE,
    FORMAL_SELECT_EVERY,
    FORMAL_WORKERS,
    SUMMARY_SCHEMA,
    _source_tree_sha256,
)


EXPECTED_TRAINING_SOURCE_TREE_SHA256 = (
    "a48630f86ec86950aa150db4d1e4a86b9c125c5029bf3484659e1523f2bc0a53"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "sirst3_test_miou_selected_v1",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "SIRST3" / "test_mIoU_selected",
    )
    return parser.parse_args(argv)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _validate_selected_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("selected checkpoint is not a mapping")
    training = payload.get("training")
    state = payload.get("state_dict")
    epoch = payload.get("epoch")
    score = payload.get("selection_score")
    if (
        payload.get("schema") != "evisirst_clean_checkpoint/v1"
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") != "SIRST3"
        or payload.get("checkpoint_role") != "test_miou_selected"
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not FORMAL_BEGIN_SELECT <= epoch <= FORMAL_EPOCHS
        or payload.get("seed") != 42
        or payload.get("test_split_accessed") is not True
        or payload.get("run_used_test_for_selection") is not True
        or payload.get("run_selection_is_optimistic") is not True
        or payload.get("test_selected") is not True
        or payload.get("this_checkpoint_selected_by_test") is not True
        or payload.get("selection_is_optimistic") is not True
        or payload.get("selection_metric") != "global_foreground_mIoU"
        or payload.get("selection_epoch") != epoch
        or not isinstance(score, float)
        or not 0.0 <= score <= 1.0
        or not isinstance(training, Mapping)
        or training.get("epochs") != FORMAL_EPOCHS
        or training.get("batch_size") != FORMAL_BATCH_SIZE
        or training.get("patch_size") != FORMAL_PATCH_SIZE
        or training.get("workers") != FORMAL_WORKERS
        or training.get("base_lr") != 1e-3
        or training.get("min_lr") != 1e-5
        or training.get("warmup_epochs") != 10
        or training.get("optimizer") != "Adam"
        or training.get("loss") != "sum_of_six_BCELoss_mean_terms"
        or training.get("amp") is not False
        or training.get("tss_registered") is not False
        or training.get("train_sample_count") != EXPECTED_TRAIN_SAMPLES
        or training.get("train_index_order_sha256")
        != SIRST3_SPLITS["train"]["ordered_ids_sha256"]
        or training.get("selection_begin_epoch") != FORMAL_BEGIN_SELECT
        or training.get("selection_frequency_epochs") != FORMAL_SELECT_EVERY
        or training.get("checkpoint_selection_rule")
        != "strict_best_global_foreground_miou"
        or training.get("selection_split") != "SIRST3_test_concat"
        or training.get("selection_source_order") != list(SOURCE_DATASETS)
        or training.get("selection_sample_count") != 1079
        or training.get("selection_source_counts") != EXPECTED_SOURCE_COUNTS
        or training.get("selection_threshold") != 0.5
        or training.get("selection_threshold_operator") != ">"
        or training.get("selection_tie_rule")
        != "strict_greater_earliest_wins"
        or training.get("selection_normalization_dataset") != "SIRST3"
        or training.get("selection_normalization") != SIRST3_NORMALIZATION
        or training.get("data_protocol")
        != "evisirst_public_joint_training_v1"
        or training.get("shuffle_seed_algorithm")
        != "sha256_length_prefixed_str_parts_uint63(seed,dataset,shuffle,epoch)"
        or training.get("crop_seed_algorithm")
        != "historical_source_namespaced_sha256_uint64"
        or training.get("test_split_accessed") is not True
        or training.get("test_selected") is not True
        or training.get("selection_is_optimistic") is not True
        or training.get("observed_train_image_mask_tree_sha256")
        != EXPECTED_TRAIN_TREE_SHA256
        or training.get("observed_selection_image_mask_tree_sha256")
        != EXPECTED_SELECTION_TREE_SHA256
        or training.get("training_source_tree_sha256")
        != EXPECTED_TRAINING_SOURCE_TREE_SHA256
        or not isinstance(training.get("runtime"), Mapping)
        or training.get("smoke_subset") is not False
    ):
        raise ValueError("formal test-selected checkpoint identity differs")
    if not isinstance(state, Mapping) or len(state) != 564:
        raise ValueError("selected checkpoint must contain 564 tensors")
    if any(
        not isinstance(key, str)
        or not isinstance(value, torch.Tensor)
        or key.startswith("target_survival")
        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
        for key, value in state.items()
    ):
        raise ValueError("selected state violates the clean model contract")
    return dict(payload), _sha256(path)


def _valid_training_summary(path: Path, checkpoint: Mapping[str, Any]) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    selection = payload.get("selection") if isinstance(payload, Mapping) else None
    return bool(
        isinstance(payload, Mapping)
        and payload.get("schema") == SUMMARY_SCHEMA
        and payload.get("status") == "complete"
        and payload.get("model_graph") == "clean_564_key_no_tss"
        and isinstance(selection, Mapping)
        and selection.get("epoch") == checkpoint.get("epoch")
        and selection.get("miou") == checkpoint.get("selection_score")
        and selection.get("test_selected") is True
        and selection.get("selection_is_optimistic") is True
    )


def _valid_evaluation(
    path: Path,
    *,
    dataset: str,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    checkpoint = payload.get("checkpoint") if isinstance(payload, Mapping) else None
    metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
    return bool(
        isinstance(payload, Mapping)
        and payload.get("schema") == "evisirst_public_evaluation/v1"
        and payload.get("model") == "evisirst"
        and payload.get("training_dataset") == "SIRST3"
        and payload.get("evaluation_dataset") == dataset
        and payload.get("sample_count") == EXPECTED_SOURCE_COUNTS[dataset]
        and payload.get("normalization_dataset") == "SIRST3"
        and payload.get("normalization") == SIRST3_NORMALIZATION
        and payload.get("threshold") == 0.5
        and payload.get("threshold_operator") == ">"
        and isinstance(checkpoint, Mapping)
        and checkpoint.get("epoch") == checkpoint_epoch
        and checkpoint.get("role") == "test_miou_selected"
        and checkpoint.get("sha256") == checkpoint_sha256
        and isinstance(metrics, Mapping)
        and all(
            isinstance(metrics.get(key), (int, float))
            for key in ("miou", "niou", "pixel_f1", "pd", "fa")
        )
    )


def _write_tables(results_root: Path, summary: Mapping[str, Any]) -> None:
    fields = [
        "training_dataset",
        "evaluation_dataset",
        "sample_count",
        "checkpoint_role",
        "epoch",
        "checkpoint_sha256",
        "miou",
        "niou",
        "pixel_f1",
        "pd",
        "fa",
    ]
    rows: list[dict[str, Any]] = []
    markdown = [
        "# EviSIRST SIRST3 test-mIoU-selected checkpoint",
        "",
        "> This checkpoint was selected on the pooled SIRST3 test split; the results are optimistic.",
        "",
        "| Train | Test | N | Selected epoch | mIoU % | nIoU % | F1 % | Pd % | Fa ×10⁻⁶ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    checkpoint = summary["checkpoint"]
    for dataset in SOURCE_DATASETS:
        evaluation = summary["evaluations"][dataset]
        metrics = evaluation["metrics"]
        rows.append(
            {
                "training_dataset": "SIRST3",
                "evaluation_dataset": dataset,
                "sample_count": evaluation["sample_count"],
                "checkpoint_role": "test_miou_selected",
                "epoch": checkpoint["epoch"],
                "checkpoint_sha256": checkpoint["sha256"],
                **{key: metrics[key] for key in fields if key in metrics},
            }
        )
        markdown.append(
            f"| SIRST3 | {dataset} | {evaluation['sample_count']} | "
            f"{checkpoint['epoch']} | {100 * float(metrics['miou']):.4f} | "
            f"{100 * float(metrics['niou']):.4f} | "
            f"{100 * float(metrics['pixel_f1']):.4f} | "
            f"{100 * float(metrics['pd']):.4f} | "
            f"{1e6 * float(metrics['fa']):.4f} |"
        )
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _write_bytes_atomic(
        results_root / "cross_dataset_results.csv",
        csv_buffer.getvalue().encode("utf-8"),
    )
    _write_bytes_atomic(
        results_root / "cross_dataset_results.md",
        ("\n".join(markdown) + "\n").encode("utf-8"),
    )


def run(args: argparse.Namespace) -> Path:
    if _source_tree_sha256() != EXPECTED_TRAINING_SOURCE_TREE_SHA256:
        raise RuntimeError("formal EviSIRST training source tree SHA-256 differs")
    dataset_root = args.dataset_root.resolve(strict=True)
    run_root = args.run_root.resolve()
    results_root = args.results_root.resolve()
    run_dir = run_root / "SIRST3"
    selected = run_dir / "EviSIRST_SIRST3_test_mIoU_selected.pth.tar"
    training_summary = run_dir / "summary.json"
    latest = run_dir / "last_training_state.pth.tar"
    best_recovery = run_dir / "best_selection_state.pth.tar"

    if not (selected.is_file() and training_summary.is_file()):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train_sirst3_test_selected.py"),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(run_root),
            "--device",
            args.device,
        ]
        if latest.is_file() or best_recovery.is_file():
            command.append("--resume")
        _run(command)

    checkpoint, checkpoint_sha = _validate_selected_checkpoint(selected)
    if not _valid_training_summary(training_summary, checkpoint):
        raise RuntimeError("completed training summary is invalid")
    published = results_root / "EviSIRST.pth.tar"
    _copy_once(selected, published)
    epoch = int(checkpoint["epoch"])

    evaluations: dict[str, Any] = {}
    for dataset in SOURCE_DATASETS:
        output = results_root / "evaluations" / f"{dataset}.json"
        if not _valid_evaluation(
            output,
            dataset=dataset,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_epoch=epoch,
        ):
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "test.py"),
                    "--model",
                    "evisirst",
                    "--dataset",
                    dataset,
                    "--dataset-root",
                    str(dataset_root),
                    "--checkpoint",
                    str(published),
                    "--device",
                    args.device,
                    "--output-json",
                    str(output),
                ]
            )
        if not _valid_evaluation(
            output,
            dataset=dataset,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_epoch=epoch,
        ):
            raise RuntimeError(f"completed evaluation is invalid: {output}")
        evaluations[dataset] = json.loads(output.read_text(encoding="utf-8"))

    data_contract = dict(sirst3_contract())
    data_contract.update(
        {
            "target_test_used_for_training_or_selection": True,
            "test_usage": "epoch500_to_1000_checkpoint_selection_and_final_evaluation",
            "selection_is_optimistic": True,
            "observed_train_image_mask_tree_sha256": checkpoint["training"][
                "observed_train_image_mask_tree_sha256"
            ],
            "observed_selection_image_mask_tree_sha256": checkpoint["training"][
                "observed_selection_image_mask_tree_sha256"
            ],
        }
    )
    summary = {
        "schema": "evisirst_sirst3_test_miou_selected_three_source_evaluation/v1",
        "status": "complete",
        "model": "EviSIRST",
        "model_graph": "clean_564_key_no_tss",
        "seed": 42,
        "training_dataset": "SIRST3",
        "training_sample_count": EXPECTED_TRAIN_SAMPLES,
        "checkpoint_selection_rule": (
            "epoch500_to_1000_strict_best_pooled_test_global_miou"
        ),
        "test_selected": True,
        "selection_is_optimistic": True,
        "target_test_used_for_checkpoint_selection": True,
        "threshold_search_performed": False,
        "checkpoint": {
            "path": str(published),
            "epoch": epoch,
            "role": "test_miou_selected",
            "selection_miou": float(checkpoint["selection_score"]),
            "sha256": checkpoint_sha,
            "bytes": published.stat().st_size,
        },
        "normalization_dataset": "SIRST3",
        "normalization": dict(SIRST3_NORMALIZATION),
        "data_contract": data_contract,
        "evaluation_protocol": "evisirst_public_common_evaluator_v1",
        "evaluations": evaluations,
    }
    summary_path = results_root / "cross_dataset_results.json"
    _write_bytes_atomic(summary_path, _json_bytes(summary))
    _write_tables(results_root, summary)
    print(summary_path, flush=True)
    return summary_path


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
