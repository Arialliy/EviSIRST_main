#!/usr/bin/env python3
"""Publish only the selected SIRST3 EviSIRST checkpoint into ``result/``.

The long-running SIRST3 experiment keeps recovery states and three-source
evaluation reports under ``runs/``.  This finalizer waits for that evaluation
to commit, validates its selected clean checkpoint, copies the exact bytes to
``result/SIRST3/EviSIRST.pth.tar``, then removes only the known intermediate
checkpoint files.  Metric JSON/CSV/Markdown and the training summary remain in
``runs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "sirst3_test_miou_selected_v1" / "SIRST3"
DEFAULT_EVALUATION_DIR = DEFAULT_RUN_DIR / "final_evaluation"
DEFAULT_RESULT = PROJECT_ROOT / "result" / "SIRST3" / "EviSIRST.pth.tar"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not 1.0 <= args.poll_seconds <= 60.0:
        parser.error("--poll-seconds must be between 1 and 60")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("result directory must not be a symlink")
    unexpected = [item for item in destination.parent.iterdir() if item.name != destination.name]
    if unexpected:
        raise FileExistsError(f"SIRST3 result directory contains extra files: {unexpected}")
    if destination.exists():
        if destination.is_symlink() or _sha256(destination) != _sha256(source):
            raise FileExistsError("an incompatible SIRST3 result already exists")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("SIRST3 selected checkpoint bytes differ")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "evisirst_clean_checkpoint/v1"
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") != "SIRST3"
        or payload.get("checkpoint_role") != "test_miou_selected"
        or payload.get("test_selected") is not True
        or payload.get("selection_is_optimistic") is not True
        or not isinstance(payload.get("epoch"), int)
        or not 500 <= int(payload["epoch"]) <= 1000
        or not isinstance(state, Mapping)
        or len(state) != 564
        or any(str(key).startswith("target_survival") for key in state)
    ):
        raise ValueError("SIRST3 selected checkpoint identity differs")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("SIRST3 selected state is malformed")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"SIRST3 selected tensor is non-finite: {key!r}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    evaluation_dir = args.evaluation_dir.resolve()
    result = args.result.resolve()
    if result != DEFAULT_RESULT.resolve():
        raise ValueError("formal SIRST3 checkpoint must be published under EviSIRST/result")
    summary_path = evaluation_dir / "cross_dataset_results.json"
    while not summary_path.is_file():
        time.sleep(args.poll_seconds)
    if summary_path.is_symlink():
        raise ValueError("SIRST3 evaluation summary must not be a symlink")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = summary.get("checkpoint") if isinstance(summary, Mapping) else None
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema")
        != "evisirst_sirst3_test_miou_selected_three_source_evaluation/v1"
        or summary.get("status") != "complete"
        or summary.get("model") != "EviSIRST"
        or summary.get("training_dataset") != "SIRST3"
        or summary.get("test_selected") is not True
        or summary.get("selection_is_optimistic") is not True
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("role") != "test_miou_selected"
    ):
        raise ValueError("SIRST3 three-source evaluation summary differs")
    source = Path(str(checkpoint["path"])).resolve()
    expected_source = (evaluation_dir / "EviSIRST.pth.tar").resolve()
    if source != expected_source:
        raise ValueError("SIRST3 evaluation checkpoint path differs")
    checkpoint_sha = str(checkpoint.get("sha256"))
    payload = _validate_checkpoint(source, checkpoint_sha)
    if int(payload["epoch"]) != int(checkpoint["epoch"]):
        raise ValueError("SIRST3 selected epoch differs between artifacts")
    _copy_once(source, result)
    if _sha256(result) != checkpoint_sha:
        raise RuntimeError("SIRST3 result copy verification failed")

    # Preserve reports but remove every known training/publication checkpoint
    # other than the one final result explicitly requested by the user.
    for path in (
        run_dir / "last_training_state.pth.tar",
        run_dir / "best_selection_state.pth.tar",
        run_dir / "EviSIRST_SIRST3_test_mIoU_selected.pth.tar",
        run_dir / "EviSIRST_epoch1000.pth.tar",
        source,
    ):
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"refusing to remove non-regular checkpoint: {path}")
            path.unlink()
    _write_json(
        run_dir / "final_publication.json",
        {
            "schema": "evisirst_sirst3_single_checkpoint_publication/v1",
            "status": "complete",
            "model": "EviSIRST",
            "dataset": "SIRST3",
            "epoch": int(payload["epoch"]),
            "selection_miou": float(payload["selection_score"]),
            "checkpoint": str(result),
            "checkpoint_sha256": checkpoint_sha,
            "only_checkpoint_in_result_directory": True,
        },
    )
    print(result, flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
