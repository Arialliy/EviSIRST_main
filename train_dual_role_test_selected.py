#!/usr/bin/env python3
"""Train clean EviSIRST and retain best-mIoU and best-Pd checkpoints.

The formal protocol trains one of NUAA-SIRST, NUDT-SIRST, or IRSTD-1K
independently.  Epochs 1--499 are training-only.  Epochs 500--1000 run the
complete public evaluator after every epoch.  Two historical role keys select
independent best-mIoU and best-Pd checkpoints.  SIRST3 uses the strict ordered
concatenation of the three source test splits and SIRST3 normalization.

This deliberately reproduces the paper's optimistic test-selected checkpoint
role.  Recovery files and metric history live under ``runs/``.  On successful
completion the only published artifacts are
``result/<dataset>/EviSIRST_best_mIoU.pth.tar`` and
``result/<dataset>/EviSIRST_best_Pd.pth.tar``.
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

from experiments.evisirst_data import (
    EviSIRSTTestDataset,
    EviSIRSTTrainDataset,
    SOURCE_DATASETS,
    TRAINING_DATASETS,
    normalization_for,
    stable_uint63,
)
from model.EviSIRST import initialize_evisirst
from test import MATCH_RADIUS, PROBABILITY_THRESHOLD, TINY_AREA, evaluate_model
from train import (
    CHECKPOINT_SCHEMA,
    _atomic_torch_save,
    _capture_rng_state,
    _cpu_state,
    _index_sha,
    _restore_rng_state,
    _write_json,
    configure_determinism,
    deep_supervision_loss,
    learning_rate_for_epoch,
    require_device,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RECOVERY_SCHEMA = "evisirst_dual_role_test_selected_recovery/v1"
SUMMARY_SCHEMA = "evisirst_dual_role_test_selected_summary/v1"
PROTOCOL_NAME = "evisirst_four_regime_dual_role_test_selected_v1"
EXPECTED_COUNTS = {
    "NUAA-SIRST": {"train": 213, "test": 214},
    "NUDT-SIRST": {"train": 663, "test": 664},
    "IRSTD-1K": {"train": 800, "test": 201},
    "SIRST3": {"train": 1676, "test": 1079},
}
FORMAL_EPOCHS = 1000
FORMAL_SELECTION_BEGIN = 500
FORMAL_SELECTION_EVERY = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=TRAINING_DATASETS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "dual_role_test_selected_v1",
    )
    parser.add_argument(
        "--result-root", type=Path, default=PROJECT_ROOT / "result"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="permit reduced epochs/data; smoke outputs are isolated",
    )
    parser.add_argument("--selection-begin", type=int, default=FORMAL_SELECTION_BEGIN)
    parser.add_argument("--selection-every", type=int, default=FORMAL_SELECTION_EVERY)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-images", type=int)
    args = parser.parse_args(argv)

    if args.seed != 42:
        parser.error("the frozen EviSIRST initializer supports seed 42 only")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.patch_size != 256:
        parser.error("the EviSIRST protocol fixes --patch-size at 256")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not math.isfinite(args.base_lr) or not math.isfinite(args.min_lr):
        parser.error("learning rates must be finite")
    if not 0.0 < args.min_lr <= args.base_lr:
        parser.error("learning rates must satisfy 0 < min-lr <= base-lr")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if not 1 <= args.selection_begin <= args.epochs:
        parser.error("--selection-begin must be within the training run")
    if args.selection_every < 1:
        parser.error("--selection-every must be positive")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        parser.error("--max-train-samples must be positive")
    if args.max_test_images is not None and args.max_test_images < 1:
        parser.error("--max-test-images must be positive")
    if not args.smoke and (
        args.epochs != FORMAL_EPOCHS
        or args.batch_size != 16
        or args.workers != 0
        or args.base_lr != 1e-3
        or args.min_lr != 1e-5
        or args.warmup_epochs != 10
        or args.selection_begin != FORMAL_SELECTION_BEGIN
        or args.selection_every != FORMAL_SELECTION_EVERY
        or args.max_train_samples is not None
        or args.max_test_images is not None
    ):
        parser.error("formal mode fixes the complete EviSIRST training protocol")
    return args


def selection_due(epoch: int, begin: int, every: int) -> bool:
    return epoch >= begin and (epoch - begin) % every == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    paths = [
        PROJECT_ROOT / "train_dual_role_test_selected.py",
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "test.py",
        PROJECT_ROOT / "experiments" / "evisirst_data.py",
        PROJECT_ROOT / "experiments" / "three_dataset_v2_protocol.py",
        PROJECT_ROOT / "experiments" / "four_dataset_models_seed42_v1.py",
        PROJECT_ROOT / "model" / "EviSIRST.py",
        PROJECT_ROOT / "model" / "__init__.py",
        *sorted((PROJECT_ROOT / "model" / "_internal").glob("*.py")),
    ]
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise RuntimeError("independent training source tree is incomplete")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        content_digest = bytes.fromhex(_sha256(path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content_digest)
    return digest.hexdigest()


def _formal_config(
    args: argparse.Namespace,
    *,
    train_count: int,
    test_count: int,
    train_index_sha256: str,
    test_index_sha256: str,
    normalization: Mapping[str, float],
    source_tree_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "model": "EviSIRST",
        "model_components": [
            "TPD8-MPRS-DCH",
            "NER4 Tail-Aware",
            "QFG2-CROA",
        ],
        "model_graph": "clean_564_key_no_tss",
        "initialization": "scratch_seed42",
        "baseline_checkpoint_loaded": False,
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
        "loss": "sum_of_six_BCELoss_mean_terms",
        "amp": False,
        "tss_registered": False,
        "train_count": train_count,
        "test_count": test_count,
        "train_index_order_sha256": train_index_sha256,
        "test_index_order_sha256": test_index_sha256,
        "normalization": dict(normalization),
        "shuffle_seed_algorithm": (
            "sha256_length_prefixed_str_parts_uint63(seed,dataset,shuffle,epoch)"
        ),
        "crop_seed_algorithm": "historical_source_namespaced_sha256_uint64",
        "selection_begin_epoch": args.selection_begin,
        "selection_end_epoch": args.epochs,
        "selection_every": args.selection_every,
        "selection_roles": {
            "best_miou": ["miou", "pd", "-fa", "niou", "tiny_pd", "-test_loss", "-epoch"],
            "best_pd": ["pd", "-fa", "tiny_pd", "miou", "niou", "-test_loss", "-epoch"],
        },
        "selection_comparison": "strict_lexicographic_greater_than",
        "selection_tie_rule": "earliest_exact_role_key_wins",
        "selection_split": (
            "SIRST3_test_strict_source_concatenation"
            if args.dataset == "SIRST3"
            else f"{args.dataset}_test"
        ),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": "strict_greater_than",
        "component_connectivity": 8,
        "component_match_radius": MATCH_RADIUS,
        "component_match_comparison": "strict_less_than",
        "tiny_area_max": TINY_AREA,
        "evaluator": "evisirst_public_common_evaluator_v1",
        "full_metric_vector_evaluated": True,
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "training_source_tree_sha256": source_tree_sha256,
        "smoke": bool(args.smoke),
    }


def _validate_state(
    value: Any, expected: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or list(value) != list(expected):
        raise ValueError("checkpoint state key/order differs from clean EviSIRST")
    for key, tensor in value.items():
        reference = expected[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint tensor {key!r} is not a tensor")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"checkpoint tensor contract differs for {key!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint tensor {key!r} is non-finite")
    return value


def _metric_row(metrics: Mapping[str, Any], epoch: int) -> dict[str, Any]:
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
        elif isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"metric {key!r} is not numeric")
        elif not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} is non-finite")
        else:
            row[key] = (
                int(value)
                if key.endswith("_count") or key == "valid_pixel_count"
                else float(value)
            )
    miou = float(row["miou"])
    if not 0.0 <= miou <= 1.0:
        raise ValueError("selection mIoU is outside [0, 1]")
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
    history: Any, *, completed_epoch: int, begin: int, every: int
) -> dict[str, int | None]:
    if not isinstance(history, list) or any(not isinstance(row, Mapping) for row in history):
        raise ValueError("selection history is malformed")
    expected_epochs = [
        epoch
        for epoch in range(begin, completed_epoch + 1)
        if selection_due(epoch, begin, every)
    ]
    observed_epochs = [int(row.get("epoch", -1)) for row in history]
    if observed_epochs != expected_epochs:
        raise ValueError("selection history cadence differs")
    validated: list[dict[str, Any]] = []
    for raw in history:
        validated.append(
            _metric_row(
                {key: value for key, value in raw.items() if key != "epoch"},
                int(raw["epoch"]),
            )
        )
    return {
        role: (
            None
            if not validated
            else int(max(validated, key=lambda row: role_key(row, role))["epoch"])
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
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") != config["dataset"]
        or payload.get("seed") != 42
        or payload.get("training") != config
    ):
        raise ValueError("recovery identity differs")
    _validate_state(payload.get("state_dict"), expected_state)
    if not isinstance(payload.get("optimizer"), Mapping):
        raise ValueError("recovery optimizer state is malformed")
    completed = int(payload.get("epoch", -1))
    best_epochs = _validate_history(
        payload.get("selection_history"),
        completed_epoch=completed,
        begin=int(config["selection_begin_epoch"]),
        every=int(config["selection_every"]),
    )
    stored_epochs = payload.get("best_epochs")
    if not isinstance(stored_epochs, Mapping) or {
        role: (None if stored_epochs.get(role) is None else int(stored_epochs[role]))
        for role in ("best_miou", "best_pd")
    } != best_epochs:
        raise ValueError("recovery best selection differs from history")
    return dict(payload)


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
        "model": "EviSIRST",
        "dataset": dataset,
        "checkpoint_role": role,
        "epoch": epoch,
        "seed": 42,
        "state_dict": dict(state),
        "training": dict(config),
        "train_index_order_sha256": config["train_index_order_sha256"],
        "normalization": dict(config["normalization"]),
        "test_split_accessed": True,
        "run_used_test_for_selection": True,
        "run_selection_is_optimistic": True,
        "test_selected": True,
        "this_checkpoint_selected_by_test": True,
        "selection_is_optimistic": True,
        "selection_metric": metric_name,
        "selection_score": score,
        "selection_epoch": epoch,
        "selection_role_key": role_key_record(metrics, role),
        "selection_metrics": dict(metrics),
    }


def _publish_once(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"final result checkpoint already exists: {path}")
    _atomic_torch_save(path, payload)
    return _sha256(path)


def _validate_completed_publication(
    summary_path: Path, published_paths: Mapping[str, Path], dataset: str
) -> None:
    if summary_path.is_symlink() or not summary_path.is_file() or any(
        path.is_symlink() or not path.is_file() for path in published_paths.values()
    ):
        raise ValueError("completed independent publication is not regular files")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    published = summary.get("published_checkpoints") if isinstance(summary, Mapping) else None
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("dataset") != dataset
        or not isinstance(published, Mapping)
    ):
        raise ValueError("completed independent publication summary differs")
    for role, path in published_paths.items():
        record = published.get(role)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != str(path)
            or record.get("sha256") != _sha256(path)
        ):
            raise ValueError("completed checkpoint record differs")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("model") != "EviSIRST"
            or checkpoint.get("dataset") != dataset
            or checkpoint.get("checkpoint_role") != role
            or checkpoint.get("test_selected") is not True
        ):
            raise ValueError("completed independent checkpoint identity differs")


def run(args: argparse.Namespace) -> Path:
    configure_determinism(args.seed)
    device = require_device(args.device)
    dataset_root = args.dataset_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    result_root = args.result_root.resolve()
    run_dir = output_root / ("smoke" if args.smoke else "formal") / args.dataset
    if not args.smoke and result_root != (PROJECT_ROOT / "result").resolve():
        raise ValueError("formal checkpoints must be published under EviSIRST/result")
    result_dir = (
        result_root / "smoke" / args.dataset
        if args.smoke
        else result_root / args.dataset
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "last_training_state.pth.tar"
    best_recovery_paths = {
        "best_miou": run_dir / "best_miou_state.pth.tar",
        "best_pd": run_dir / "best_pd_state.pth.tar",
    }
    history_path = run_dir / "selection_history.json"
    summary_path = run_dir / "summary.json"
    published_paths = {
        "best_miou": result_dir / "EviSIRST_best_mIoU.pth.tar",
        "best_pd": result_dir / "EviSIRST_best_Pd.pth.tar",
    }

    if result_dir.exists():
        if result_dir.is_symlink() or not result_dir.is_dir():
            raise ValueError(f"result dataset path is not a regular directory: {result_dir}")
        expected_names = {path.name for path in published_paths.values()}
        unexpected = [item for item in result_dir.iterdir() if item.name not in expected_names]
        if unexpected:
            raise FileExistsError(
                f"result dataset directory contains unexpected files: {unexpected}"
            )
    if summary_path.exists():
        if args.resume and all(path.is_file() for path in published_paths.values()):
            # Idempotent cleanup for a crash after summary commit but before
            # recovery-file removal.  The result checkpoint remains immutable.
            _validate_completed_publication(summary_path, published_paths, args.dataset)
            latest_path.unlink(missing_ok=True)
            for path in best_recovery_paths.values():
                path.unlink(missing_ok=True)
            return result_dir
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
    if args.dataset == "SIRST3":
        test_parts = [
            EviSIRSTTestDataset(
                "SIRST3", source_dataset, dataset_root=dataset_root
            )
            for source_dataset in SOURCE_DATASETS
        ]
        full_test: Any = ConcatDataset(test_parts)
        full_test_ids = [
            f"{source_dataset}::{sample_id}"
            for source_dataset, part in zip(SOURCE_DATASETS, test_parts)
            for sample_id in part.sample_ids
        ]
        test_normalization = normalization_for("SIRST3")
    else:
        full_test = EviSIRSTTestDataset(
            args.dataset,
            args.dataset,
            dataset_root=dataset_root,
        )
        full_test_ids = list(full_test.sample_ids)
        test_normalization = dict(full_test.normalization)
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
    )
    if full_train.normalization != test_normalization:
        raise RuntimeError("independent train/test normalization differs")

    model, model_metadata = initialize_evisirst(
        args.dataset, seed=args.seed, training=True
    )
    if len(model.state_dict()) != 564 or hasattr(model, "target_survival"):
        raise RuntimeError("independent runner requires clean 564-key no-TSS EviSIRST")
    if model_metadata.get("target_survival_registered") is not False:
        raise RuntimeError("EviSIRST builder metadata unexpectedly registers TSS")
    pair = model_metadata.get("pair")
    if (
        model_metadata.get("method") != "final_scratch"
        or model_metadata.get("warm_start_used") is not False
        or model_metadata.get("parent_checkpoint") is not None
        or not isinstance(pair, Mapping)
        or pair.get("initialization_mode") != "true_scratch"
        or pair.get("parent_checkpoint") is not None
        or pair.get("parent_checkpoint_load_count") != 0
        or pair.get("warm_start_used") is not False
        or pair.get("optimizer_state_inherited") is not False
    ):
        raise RuntimeError("EviSIRST builder did not provide a true-scratch model")
    model.to(device)
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
        model.load_state_dict(recovery["state_dict"], strict=True)
        optimizer.load_state_dict(recovery["optimizer"])
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
        elapsed_before = float(recovery.get("elapsed_seconds", 0.0))
        start_epoch = completed + 1
    elif latest_path.exists() or any(path.exists() for path in best_recovery_paths.values()):
        raise FileExistsError(f"recovery exists; pass --resume: {run_dir}")

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        full_train.set_epoch(epoch)
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(
            stable_uint63(args.seed, args.dataset, "shuffle", epoch)
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
        processed = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = deep_supervision_loss(model(images), masks, criterion)
            loss.backward()
            optimizer.step()
            count = int(images.shape[0])
            loss_sum += float(loss.detach().item()) * count
            processed += count
        if processed != train_count:
            raise RuntimeError("processed training sample count differs")

        improved_roles: list[str] = []
        metrics: dict[str, Any] | None = None
        if selection_due(epoch, args.selection_begin, args.selection_every):
            training_rng = _capture_rng_state(device)
            try:
                metrics = _metric_row(
                    evaluate_model(model, evaluation_loader, device), epoch
                )
            finally:
                _restore_rng_state(training_rng, device)
            history.append(metrics)
            for role in ("best_miou", "best_pd"):
                previous_epoch = best_epochs[role]
                previous = (
                    None
                    if previous_epoch is None
                    else next(row for row in history[:-1] if int(row["epoch"]) == previous_epoch)
                )
                if previous is None or role_key(metrics, role) > role_key(previous, role):
                    best_epochs[role] = epoch
                    improved_roles.append(role)

        state = _cpu_state(model)
        elapsed = elapsed_before + time.time() - started
        recovery = {
            "schema": RECOVERY_SCHEMA,
            "model": "EviSIRST",
            "dataset": args.dataset,
            "epoch": epoch,
            "seed": args.seed,
            "state_dict": state,
            "optimizer": optimizer.state_dict(),
            "rng": _capture_rng_state(device),
            "training": config,
            "mean_train_loss": loss_sum / processed,
            "learning_rate": lr,
            "selection_history": history,
            "best_epochs": dict(best_epochs),
            "elapsed_seconds": elapsed,
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
        begin=args.selection_begin,
        every=args.selection_every,
    )
    if expected_best_epochs != best_epochs:
        raise RuntimeError("final role epochs differ from selection history")
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
        checkpoint_sha = _sha256(path) if path.exists() else _publish_once(path, payload)
        reopened = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(reopened, Mapping)
            or reopened.get("schema") != CHECKPOINT_SCHEMA
            or reopened.get("dataset") != args.dataset
            or reopened.get("checkpoint_role") != role
            or int(reopened.get("epoch", -1)) != role_epoch
            or list(reopened.get("selection_role_key", []))
            != list(payload["selection_role_key"])
        ):
            raise RuntimeError(f"published {role} checkpoint identity replay failed")
        _validate_state(reopened.get("state_dict"), expected_state)
        for key, expected_tensor in payload["state_dict"].items():
            if not torch.equal(reopened["state_dict"][key], expected_tensor):
                raise RuntimeError(f"published {role} tensor differs: {key!r}")
        selections[role] = {
            "epoch": role_epoch,
            "metrics": metrics,
            "role_key": role_key_record(metrics, role),
            "test_selected": True,
            "selection_is_optimistic": True,
        }
        published_records[role] = {
            "path": str(path),
            "sha256": checkpoint_sha,
        }
    if {path.name for path in result_dir.iterdir()} != {
        path.name for path in published_paths.values()
    }:
        raise RuntimeError("result directory does not contain exactly two checkpoints")
    _write_json(
        summary_path,
        {
            "schema": SUMMARY_SCHEMA,
            "status": "complete",
            "model": "EviSIRST",
            "dataset": args.dataset,
            "training": config,
            "candidate_count": len(history),
            "selections": selections,
            "published_checkpoints": published_records,
            "only_two_checkpoints_in_result_directory": True,
            "selection_history": history,
            "elapsed_seconds": elapsed_before + time.time() - started,
        },
    )
    # Recovery checkpoints are necessary while the job is running, but the
    # user requested that only the two selected checkpoints remain after completion.
    latest_path.unlink()
    for path in best_recovery_paths.values():
        path.unlink()
    return result_dir


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()
