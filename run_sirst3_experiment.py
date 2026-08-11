#!/usr/bin/env python3
"""Train clean EviSIRST on SIRST3, then test one fixed endpoint on 3 sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.evisirst_data import (
    SIRST3_NORMALIZATION,
    SIRST3_SPLITS,
    SOURCE_DATASETS,
    sirst3_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FORMAL_EPOCHS = 1000
FORMAL_BATCH_SIZE = 16
FORMAL_WORKERS = 0
FORMAL_TRAIN_INDEX_SHA256 = SIRST3_SPLITS["train"]["ordered_ids_sha256"]
EXPECTED_TEST_COUNTS = {
    "NUAA-SIRST": 214,
    "NUDT-SIRST": 664,
    "IRSTD-1K": 201,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "sirst3_clean_v1",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "SIRST3",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_once(source: Path, destination: Path) -> str:
    source_sha = _sha256(source)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError(destination)
        if _sha256(destination) != source_sha:
            raise FileExistsError(
                f"published SIRST3 checkpoint already differs: {destination}"
            )
        return source_sha
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if _sha256(temporary) != source_sha:
            raise RuntimeError("checkpoint copy SHA-256 differs")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return source_sha


def _validate_formal_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("formal checkpoint payload is not a mapping")
    training = payload.get("training")
    state = payload.get("state_dict")
    if (
        payload.get("schema") != "evisirst_clean_checkpoint/v1"
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") != "SIRST3"
        or payload.get("checkpoint_role") != "final"
        or payload.get("epoch") != FORMAL_EPOCHS
        or payload.get("seed") != 42
        or payload.get("test_split_accessed") is not False
        or not isinstance(training, Mapping)
        or training.get("epochs") != FORMAL_EPOCHS
        or training.get("batch_size") != FORMAL_BATCH_SIZE
        or training.get("workers") != FORMAL_WORKERS
        or training.get("sample_count") != 1676
        or training.get("smoke_subset") is not False
        or training.get("test_split_accessed") is not False
        or training.get("train_index_order_sha256")
        != FORMAL_TRAIN_INDEX_SHA256
    ):
        raise ValueError("formal SIRST3 checkpoint identity differs")
    if not isinstance(state, Mapping) or len(state) != 564:
        raise ValueError("formal SIRST3 checkpoint must contain 564 tensors")
    if any(
        not isinstance(key, str)
        or not isinstance(value, torch.Tensor)
        or key.startswith("target_survival")
        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
        for key, value in state.items()
    ):
        raise ValueError("formal SIRST3 state violates the clean model contract")
    return dict(payload), _sha256(path)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _valid_evaluation(path: Path, dataset: str, checkpoint_sha: str) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, Mapping)
        and payload.get("training_dataset") == "SIRST3"
        and payload.get("evaluation_dataset") == dataset
        and payload.get("sample_count") == EXPECTED_TEST_COUNTS[dataset]
        and payload.get("normalization_dataset") == "SIRST3"
        and payload.get("normalization") == SIRST3_NORMALIZATION
        and isinstance(payload.get("checkpoint"), Mapping)
        and payload["checkpoint"].get("epoch") == FORMAL_EPOCHS
        and payload["checkpoint"].get("sha256") == checkpoint_sha
        and payload.get("threshold") == 0.5
        and payload.get("threshold_operator") == ">"
    )


def _write_tables(results_root: Path, summary: Mapping[str, Any]) -> None:
    fields = [
        "training_dataset",
        "evaluation_dataset",
        "sample_count",
        "normalization_dataset",
        "checkpoint_role",
        "epoch",
        "checkpoint_sha256",
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
        "fa",
        "false_objects_per_image",
    ]
    rows: list[dict[str, Any]] = []
    markdown = [
        "# SIRST3 clean EviSIRST: source-specific held-out tests",
        "",
        "| Train | Test | N | Epoch | mIoU % | nIoU % | F1 % | Pd % | Fa ×10⁻⁶ |",
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
                "normalization_dataset": "SIRST3",
                "checkpoint_role": "fixed_final_endpoint",
                "epoch": FORMAL_EPOCHS,
                "checkpoint_sha256": checkpoint["sha256"],
                **{field: metrics.get(field) for field in fields if field in metrics},
            }
        )
        markdown.append(
            f"| SIRST3 | {dataset} | {evaluation['sample_count']} | 1000 | "
            f"{100 * float(metrics['miou']):.4f} | "
            f"{100 * float(metrics['niou']):.4f} | "
            f"{100 * float(metrics['pixel_f1']):.4f} | "
            f"{100 * float(metrics['pd']):.4f} | "
            f"{1e6 * float(metrics['fa']):.4f} |"
        )
    csv_buffer = __import__("io").StringIO(newline="")
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
    dataset_root = args.dataset_root.resolve(strict=True)
    run_root = args.run_root.resolve()
    results_root = args.results_root.resolve()
    run_dir = run_root / "SIRST3"
    train_checkpoint = run_dir / "EviSIRST.pth.tar"
    train_summary = run_dir / "summary.json"
    latest = run_dir / "last_training_state.pth.tar"

    if not (train_checkpoint.is_file() and train_summary.is_file()):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--dataset",
            "SIRST3",
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(run_root),
            "--device",
            args.device,
            "--epochs",
            str(FORMAL_EPOCHS),
            "--batch-size",
            str(FORMAL_BATCH_SIZE),
            "--workers",
            str(FORMAL_WORKERS),
        ]
        if latest.is_file():
            command.append("--resume")
        _run(command)

    _, checkpoint_sha = _validate_formal_checkpoint(train_checkpoint)
    published_checkpoint = results_root / "EviSIRST.pth.tar"
    _copy_once(train_checkpoint, published_checkpoint)

    evaluations: dict[str, Any] = {}
    for dataset in SOURCE_DATASETS:
        output = results_root / "evaluations" / f"{dataset}.json"
        if not _valid_evaluation(output, dataset, checkpoint_sha):
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
                    str(published_checkpoint),
                    "--device",
                    args.device,
                    "--output-json",
                    str(output),
                ]
            )
        if not _valid_evaluation(output, dataset, checkpoint_sha):
            raise RuntimeError(f"completed evaluation is invalid: {output}")
        evaluations[dataset] = json.loads(output.read_text(encoding="utf-8"))

    summary = {
        "schema": "evisirst_sirst3_fixed_endpoint_three_source_evaluation/v1",
        "status": "complete",
        "model": "EviSIRST",
        "model_graph": "clean_564_key_no_tss",
        "seed": 42,
        "training_dataset": "SIRST3",
        "training_sample_count": 1676,
        "checkpoint_selection_rule": "fixed_epoch_1000_endpoint",
        "test_selected": False,
        "target_test_used_for_training_or_selection": False,
        "target_test_accessed_for_final_evaluation": True,
        "threshold_search_performed": False,
        "checkpoint": {
            "path": str(published_checkpoint),
            "epoch": FORMAL_EPOCHS,
            "role": "fixed_final_endpoint",
            "sha256": checkpoint_sha,
            "bytes": published_checkpoint.stat().st_size,
        },
        "normalization_dataset": "SIRST3",
        "normalization": dict(SIRST3_NORMALIZATION),
        "data_contract": sirst3_contract(),
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
