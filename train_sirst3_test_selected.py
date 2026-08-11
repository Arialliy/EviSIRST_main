#!/usr/bin/env python3
"""Train clean EviSIRST and reproduce SCTransNet's test-mIoU selection rule.

This runner is intentionally separate from ``train.py``.  It accesses the
pooled SIRST3 test split after every epoch from 500 through 1000 and retains
the earliest checkpoint attaining each strict improvement in global foreground
mIoU.  The resulting score is therefore operational/test-selected and must not
be presented as an unbiased held-out estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from experiments.evisirst_data import (
    EviSIRSTTestDataset,
    EviSIRSTTrainDataset,
    SIRST3_NORMALIZATION,
    SOURCE_DATASETS,
    stable_uint63,
)
from experiments import evisirst_data as data_module
from model.EviSIRST import initialize_evisirst
from test import (
    PROBABILITY_THRESHOLD,
    _extract_hw,
    final_prediction,
)
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


RECOVERY_SCHEMA = "evisirst_sirst3_test_miou_selected_recovery/v1"
SUMMARY_SCHEMA = "evisirst_sirst3_test_miou_selected_training/v1"
FORMAL_EPOCHS = 1000
FORMAL_BEGIN_SELECT = 500
FORMAL_SELECT_EVERY = 1
FORMAL_BATCH_SIZE = 16
FORMAL_PATCH_SIZE = 256
FORMAL_WORKERS = 0
EXPECTED_TRAIN_SAMPLES = 1676
EXPECTED_SELECTION_SAMPLES = 1079
EXPECTED_TRAIN_TREE_SHA256 = (
    "0593df78057dbc89679ed7a1d1c14a837cd537b96371fbb4c577273dacf19f44"
)
EXPECTED_SELECTION_TREE_SHA256 = (
    "aac23b1df82514057f0b2bc573499f80defc257ee436929277f49ff3b7619d7b"
)
EXPECTED_SOURCE_COUNTS = {
    "NUAA-SIRST": 214,
    "NUDT-SIRST": 664,
    "IRSTD-1K": 201,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=FORMAL_PATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--selection-begin-epoch", type=int, default=FORMAL_BEGIN_SELECT)
    parser.add_argument("--selection-every", type=int, default=FORMAL_SELECT_EVERY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int, help="smoke-test only")
    parser.add_argument("--max-selection-images", type=int, help="smoke-test only")
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("the frozen EviSIRST initializer supports seed 42 only")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.patch_size != FORMAL_PATCH_SIZE:
        parser.error("the EviSIRST protocol fixes --patch-size at 256")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not math.isfinite(args.base_lr) or not math.isfinite(args.min_lr):
        parser.error("learning rates must be finite")
    if not 0.0 < args.min_lr <= args.base_lr:
        parser.error("learning rates must satisfy 0 < min-lr <= base-lr")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if not 1 <= args.selection_begin_epoch <= args.epochs:
        parser.error("--selection-begin-epoch must be within the training run")
    if args.selection_every < 1:
        parser.error("--selection-every must be positive")
    for name in ("max_train_samples", "max_selection_images"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    formal_identity = (
        args.epochs == FORMAL_EPOCHS
        and args.batch_size == FORMAL_BATCH_SIZE
        and args.patch_size == FORMAL_PATCH_SIZE
        and args.workers == FORMAL_WORKERS
        and args.base_lr == 1e-3
        and args.min_lr == 1e-5
        and args.warmup_epochs == 10
        and args.selection_begin_epoch == FORMAL_BEGIN_SELECT
        and args.selection_every == FORMAL_SELECT_EVERY
        and args.max_train_samples is None
        and args.max_selection_images is None
    )
    if args.smoke:
        if args.max_train_samples is None or args.max_selection_images is None:
            parser.error("--smoke requires both bounded max-sample options")
    elif not formal_identity:
        parser.error("non-formal overrides require explicit --smoke")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _file_tree_sha256(
    dataset_root: Path,
    entries: list[tuple[str, str, Path]],
) -> str:
    digest = hashlib.sha256()
    for role, sample_id, path in entries:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"data entry is not a regular file: {path}")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError(f"data entry escapes dataset root: {resolved}") from exc
        _update_length_prefixed(digest, role)
        _update_length_prefixed(digest, sample_id)
        _update_length_prefixed(digest, relative.as_posix())
        _update_length_prefixed(digest, bytes.fromhex(_sha256(resolved)))
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parent
    paths = [
        root / "train_sirst3_test_selected.py",
        root / "train.py",
        root / "test.py",
        root / "experiments" / "evisirst_data.py",
        root / "experiments" / "three_dataset_v2_protocol.py",
        root / "experiments" / "four_dataset_models_seed42_v1.py",
        root / "model" / "EviSIRST.py",
        root / "model" / "__init__.py",
        *sorted((root / "model" / "_internal").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"training source is not a regular file: {path}")
        _update_length_prefixed(digest, path.relative_to(root).as_posix())
        _update_length_prefixed(digest, bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _selection_due(epoch: int, begin: int, every: int) -> bool:
    return epoch >= begin and (epoch - begin) % every == 0


def _training_config(
    args: argparse.Namespace,
    *,
    train_count: int,
    selection_count: int,
    selection_source_counts: Mapping[str, int],
    train_index_order_sha256: str,
    observed_train_tree_sha256: str,
    observed_selection_tree_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset": "SIRST3",
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
        "train_sample_count": train_count,
        "train_index_order_sha256": train_index_order_sha256,
        "data_protocol": "evisirst_public_joint_training_v1",
        "shuffle_seed_algorithm": (
            "sha256_length_prefixed_str_parts_uint63(seed,dataset,shuffle,epoch)"
        ),
        "crop_seed_algorithm": "historical_source_namespaced_sha256_uint64",
        "checkpoint_selection_rule": "strict_best_global_foreground_miou",
        "selection_split": "SIRST3_test_concat",
        "selection_source_order": list(SOURCE_DATASETS),
        "selection_source_counts": dict(selection_source_counts),
        "selection_sample_count": selection_count,
        "selection_begin_epoch": args.selection_begin_epoch,
        "selection_frequency_epochs": args.selection_every,
        "selection_threshold": PROBABILITY_THRESHOLD,
        "selection_threshold_operator": ">",
        "selection_tie_rule": "strict_greater_earliest_wins",
        "selection_normalization_dataset": "SIRST3",
        "selection_normalization": dict(SIRST3_NORMALIZATION),
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "observed_train_image_mask_tree_sha256": observed_train_tree_sha256,
        "observed_selection_image_mask_tree_sha256": observed_selection_tree_sha256,
        "training_source_tree_sha256": source_tree_sha256,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "cuda": str(torch.version.cuda),
        },
        "smoke_subset": args.smoke,
    }


def _validate_state(
    state: Any,
    expected: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping) or list(state) != list(expected):
        raise ValueError(f"{label} state keys/order differ")
    if len(state) != 564 or any(key.startswith("target_survival") for key in state):
        raise ValueError(f"{label} state is not the clean 564-key graph")
    for key, value in state.items():
        reference = expected[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
        ):
            raise ValueError(f"{label} tensor contract differs for {key!r}")


def _load_recovery(
    path: Path,
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
        or payload.get("dataset") != "SIRST3"
        or payload.get("seed") != 42
        or payload.get("training") != dict(config)
    ):
        raise ValueError(f"recovery identity differs: {path}")
    _validate_state(payload.get("state_dict"), expected_state, label="recovery")
    history = payload.get("selection_history")
    if not isinstance(history, list):
        raise ValueError(f"recovery selection history is malformed: {path}")
    return dict(payload)


def _validate_history(
    history: list[Any],
    *,
    completed_epoch: int,
    begin: int,
    every: int,
) -> tuple[float, int | None]:
    expected_epochs = [
        epoch
        for epoch in range(begin, completed_epoch + 1)
        if _selection_due(epoch, begin, every)
    ]
    observed_epochs: list[int] = []
    best_score = 0.0
    best_epoch: int | None = None
    for row in history:
        if not isinstance(row, Mapping):
            raise ValueError("selection history row is not a mapping")
        epoch = int(row.get("epoch", -1))
        score = float(row.get("miou", float("nan")))
        intersection = row.get("intersection")
        union = row.get("union")
        sample_count = row.get("sample_count")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("selection history mIoU is invalid")
        if (
            isinstance(intersection, bool)
            or not isinstance(intersection, int)
            or isinstance(union, bool)
            or not isinstance(union, int)
            or union <= 0
            or not 0 <= intersection <= union
            or score != intersection / union
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
        ):
            raise ValueError("selection history sufficient statistics differ")
        observed_epochs.append(epoch)
        if score > best_score:
            best_score = score
            best_epoch = epoch
    if observed_epochs != expected_epochs:
        raise ValueError("selection history cadence differs")
    return best_score, best_epoch


def _selection_loader(
    dataset_root: Path,
    *,
    workers: int,
    max_images: int | None,
) -> tuple[DataLoader[Any], int, dict[str, int], str]:
    sources = []
    selection_entries: list[tuple[str, str, Path]] = []
    for dataset in SOURCE_DATASETS:
        source = EviSIRSTTestDataset(
            "SIRST3", dataset, dataset_root=dataset_root
        )
        if len(source) != EXPECTED_SOURCE_COUNTS[dataset]:
            raise RuntimeError(f"{dataset} test count differs")
        if source.normalization != SIRST3_NORMALIZATION:
            raise RuntimeError(f"{dataset} selection normalization differs")
        for sample_id in source.sample_ids:
            sample = data_module.source_protocol.resolve_sample(
                source.dataset_root,
                dataset,
                sample_id,
                split="test",
                known_ids=source._known_ids,
            )
            selection_entries.extend(
                [
                    (f"{dataset}:image", sample_id, sample.image_path),
                    (f"{dataset}:mask", sample_id, sample.mask_path),
                ]
            )
        sources.append(source)
    pooled: Any = ConcatDataset(sources)
    if len(pooled) != EXPECTED_SELECTION_SAMPLES:
        raise RuntimeError("pooled SIRST3 selection count differs")
    if max_images is not None:
        pooled = Subset(pooled, range(min(max_images, len(pooled))))
    remaining = len(pooled)
    actual_counts: dict[str, int] = {}
    for dataset in SOURCE_DATASETS:
        count = min(remaining, EXPECTED_SOURCE_COUNTS[dataset])
        actual_counts[dataset] = count
        remaining -= count
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_uint63(42, "SIRST3", "selection_loader"))
    return (
        DataLoader(
            pooled,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            generator=generator,
        ),
        len(pooled),
        actual_counts,
        _file_tree_sha256(dataset_root, selection_entries),
    )


@torch.inference_mode()
def evaluate_official_selection_miou(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float | int]:
    """Match SCTransNet ``metrics.batch_intersection_union`` exactly."""

    model.eval()
    model.mode = "test"
    total_intersection = 0
    total_union = 0
    sample_count = 0
    for images, masks, sizes, _sample_ids in loader:
        height, width = _extract_hw(sizes)
        images = images.to(device, non_blocking=True)
        raw = final_prediction(model(images))
        if not isinstance(raw, torch.Tensor) or raw.ndim != 4:
            raise RuntimeError("selection output must be a four-dimensional tensor")
        probability = raw[:, :, :height, :width]
        target = masks[:, :, :height, :width].float()
        if probability.shape != target.shape or probability.shape[:2] != (1, 1):
            raise RuntimeError("selection prediction/target shape differs")
        if not bool(torch.isfinite(probability).all()):
            raise FloatingPointError("selection prediction is non-finite")
        if bool((probability < 0).any()) or bool((probability > 1).any()):
            raise RuntimeError("selection prediction is outside [0, 1]")
        prediction = (probability > PROBABILITY_THRESHOLD).float().cpu()
        target = target.cpu()
        intersection = prediction * (prediction == target).float()
        area_intersection, _ = np.histogram(
            intersection.numpy(), bins=1, range=(1, 1)
        )
        area_prediction, _ = np.histogram(
            prediction.numpy(), bins=1, range=(1, 1)
        )
        area_target, _ = np.histogram(target.numpy(), bins=1, range=(1, 1))
        intersection_count = int(area_intersection[0])
        union_count = int(area_prediction[0] + area_target[0] - area_intersection[0])
        if intersection_count > union_count:
            raise RuntimeError("selection intersection exceeds union")
        total_intersection += intersection_count
        total_union += union_count
        sample_count += 1
    if sample_count != len(loader.dataset) or total_union <= 0:
        raise RuntimeError("selection sample count/union differs")
    return {
        "miou": total_intersection / total_union,
        "intersection": total_intersection,
        "union": total_union,
        "sample_count": sample_count,
    }


def _slim_checkpoint(
    *,
    state: Mapping[str, torch.Tensor],
    epoch: int,
    role: str,
    config: Mapping[str, Any],
    train_index_order_sha256: str,
    selection_score: float | None,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": "SIRST3",
        "checkpoint_role": role,
        "epoch": epoch,
        "seed": 42,
        "state_dict": dict(state),
        "training": dict(config),
        "train_index_order_sha256": train_index_order_sha256,
        "normalization": dict(SIRST3_NORMALIZATION),
        "test_split_accessed": True,
        "run_used_test_for_selection": True,
        "run_selection_is_optimistic": True,
        "test_selected": role == "test_miou_selected",
        "this_checkpoint_selected_by_test": role == "test_miou_selected",
        "selection_is_optimistic": role == "test_miou_selected",
        "selection_metric": (
            "global_foreground_mIoU" if role == "test_miou_selected" else None
        ),
        "selection_score": selection_score,
        "selection_epoch": epoch if role == "test_miou_selected" else None,
    }


def run(args: argparse.Namespace) -> Path:
    configure_determinism(args.seed)
    device = require_device(args.device)
    dataset_root = args.dataset_root.resolve(strict=True)
    run_dir = args.output_root.resolve() / "SIRST3"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "last_training_state.pth.tar"
    best_recovery_path = run_dir / "best_selection_state.pth.tar"
    selected_path = run_dir / "EviSIRST_SIRST3_test_mIoU_selected.pth.tar"
    endpoint_path = run_dir / f"EviSIRST_epoch{args.epochs}.pth.tar"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists: {summary_path}")

    full_train = EviSIRSTTrainDataset(
        "SIRST3",
        dataset_root=dataset_root,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    train_dataset: Any = full_train
    if args.max_train_samples is not None:
        train_dataset = Subset(
            full_train, range(min(args.max_train_samples, len(full_train)))
        )
    selection_loader, selection_count, selection_source_counts, selection_tree_sha = (
        _selection_loader(
        dataset_root,
        workers=args.workers,
        max_images=args.max_selection_images,
        )
    )
    train_count = len(train_dataset)
    if not args.smoke and (
        train_count != EXPECTED_TRAIN_SAMPLES
        or selection_count != EXPECTED_SELECTION_SAMPLES
        or selection_source_counts != EXPECTED_SOURCE_COUNTS
    ):
        raise RuntimeError("formal SIRST3 train/selection counts differ")
    train_index_sha = _index_sha(full_train.sample_ids)
    train_entries: list[tuple[str, str, Path]] = []
    for index in range(len(full_train)):
        image_path, mask_path, sample_id, source_dataset = full_train._paths(index)
        train_entries.extend(
            [
                (f"{source_dataset}:image", sample_id, image_path),
                (f"{source_dataset}:mask", sample_id, mask_path),
            ]
        )
    train_tree_sha = _file_tree_sha256(dataset_root, train_entries)
    if (
        train_tree_sha != EXPECTED_TRAIN_TREE_SHA256
        or selection_tree_sha != EXPECTED_SELECTION_TREE_SHA256
    ):
        raise RuntimeError("SIRST3 train/selection image-mask tree SHA-256 differs")
    config = _training_config(
        args,
        train_count=train_count,
        selection_count=selection_count,
        selection_source_counts=selection_source_counts,
        train_index_order_sha256=train_index_sha,
        observed_train_tree_sha256=train_tree_sha,
        observed_selection_tree_sha256=selection_tree_sha,
        source_tree_sha256=_source_tree_sha256(),
    )

    model, _metadata = initialize_evisirst("SIRST3", seed=42, training=True)
    model.to(device)
    if len(model.state_dict()) != 564 or hasattr(model, "target_survival"):
        raise RuntimeError("test-selected training requires the clean no-TSS graph")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    criterion = nn.BCELoss(reduction="mean")

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_score = 0.0
    best_epoch: int | None = None
    elapsed_before = 0.0
    if args.resume:
        candidates = []
        for path in (latest_path, best_recovery_path):
            if path.is_file() and not path.is_symlink():
                candidates.append(
                    (
                        int(
                            _load_recovery(
                                path, config, model.state_dict()
                            )["epoch"]
                        ),
                        path,
                    )
                )
        if not candidates:
            raise FileNotFoundError("no recovery checkpoint exists")
        _epoch, recovery_path = max(candidates, key=lambda item: item[0])
        recovery = _load_recovery(recovery_path, config, model.state_dict())
        completed_epoch = int(recovery["epoch"])
        if completed_epoch < 0 or completed_epoch > args.epochs:
            raise ValueError("recovery epoch is outside this run")
        history = [dict(row) for row in recovery["selection_history"]]
        best_score, best_epoch = _validate_history(
            history,
            completed_epoch=completed_epoch,
            begin=args.selection_begin_epoch,
            every=args.selection_every,
        )
        stored_best = recovery.get("best_score")
        stored_epoch = recovery.get("best_epoch")
        if (
            (best_epoch is None and (stored_best is not None or stored_epoch is not None))
            or (
                best_epoch is not None
                and (
                    float(stored_best) != best_score
                    or int(stored_epoch) != best_epoch
                )
            )
        ):
            raise ValueError("recovery best-selection metadata differs")
        model.load_state_dict(recovery["state_dict"], strict=True)
        optimizer.load_state_dict(recovery["optimizer"])
        _restore_rng_state(recovery["rng"], device)
        elapsed_before = float(recovery.get("elapsed_seconds", 0.0))
        start_epoch = completed_epoch + 1
    elif any(
        path.exists()
        for path in (
            latest_path,
            best_recovery_path,
            selected_path,
            endpoint_path,
        )
    ):
        raise FileExistsError(f"run directory already contains checkpoints: {run_dir}")

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        full_train.set_epoch(epoch)
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(stable_uint63(42, "SIRST3", "shuffle", epoch))
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

        selected_this_epoch = False
        selection_metrics: dict[str, Any] | None = None
        if _selection_due(
            epoch, args.selection_begin_epoch, args.selection_every
        ):
            selection_metrics = dict(
                evaluate_official_selection_miou(model, selection_loader, device)
            )
            score = float(selection_metrics["miou"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise FloatingPointError("selection mIoU is invalid")
            history.append(
                {
                    "epoch": epoch,
                    "miou": score,
                    "intersection": int(selection_metrics["intersection"]),
                    "union": int(selection_metrics["union"]),
                    "sample_count": int(selection_metrics["sample_count"]),
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                selected_this_epoch = True

        state = _cpu_state(model)
        elapsed = elapsed_before + (time.time() - started)
        rng = _capture_rng_state(device)
        recovery = {
            "schema": RECOVERY_SCHEMA,
            "model": "EviSIRST",
            "dataset": "SIRST3",
            "epoch": epoch,
            "seed": 42,
            "state_dict": state,
            "optimizer": optimizer.state_dict(),
            "rng": rng,
            "training": config,
            "mean_train_loss": loss_sum / processed,
            "learning_rate": lr,
            "selection_history": history,
            "best_score": None if best_epoch is None else best_score,
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed,
        }
        # An improving best checkpoint is itself a complete recovery point.  It
        # is written before rolling state, so a crash cannot lose that candidate.
        if selected_this_epoch:
            _atomic_torch_save(best_recovery_path, recovery)
        _atomic_torch_save(latest_path, recovery)
        selection_text = (
            ""
            if selection_metrics is None
            else (
                f" selection_mIoU={100 * float(selection_metrics['miou']):.6f}%"
                f" best_epoch={best_epoch} best_mIoU={100 * best_score:.6f}%"
            )
        )
        print(
            f"epoch={epoch}/{args.epochs} loss={loss_sum / processed:.6f} "
            f"lr={lr:.8f} samples={processed}{selection_text}",
            flush=True,
        )

    if best_epoch is None or not best_recovery_path.is_file():
        raise RuntimeError("training completed without a selected checkpoint")
    _unused_loader, end_selection_count, end_source_counts, end_selection_tree = (
        _selection_loader(
            dataset_root,
            workers=args.workers,
            max_images=args.max_selection_images,
        )
    )
    if (
        _source_tree_sha256() != config["training_source_tree_sha256"]
        or _file_tree_sha256(dataset_root, train_entries)
        != config["observed_train_image_mask_tree_sha256"]
        or end_selection_tree
        != config["observed_selection_image_mask_tree_sha256"]
        or end_selection_count != selection_count
        or end_source_counts != selection_source_counts
    ):
        raise RuntimeError("source/data identity changed during training")
    best_recovery = _load_recovery(
        best_recovery_path, config, model.state_dict()
    )
    if (
        int(best_recovery["epoch"]) != best_epoch
        or float(best_recovery["best_score"]) != best_score
    ):
        raise RuntimeError("best recovery does not match final selection")
    selected_payload = _slim_checkpoint(
        state=best_recovery["state_dict"],
        epoch=best_epoch,
        role="test_miou_selected",
        config=config,
        train_index_order_sha256=train_index_sha,
        selection_score=best_score,
    )
    _atomic_torch_save(selected_path, selected_payload)
    endpoint_payload = _slim_checkpoint(
        state=_cpu_state(model),
        epoch=args.epochs,
        role="final",
        config=config,
        train_index_order_sha256=train_index_sha,
        selection_score=None,
    )
    _atomic_torch_save(endpoint_path, endpoint_payload)
    _write_json(
        summary_path,
        {
            "schema": SUMMARY_SCHEMA,
            "status": "complete",
            "model": "EviSIRST",
            "model_graph": "clean_564_key_no_tss",
            "seed": 42,
            "dataset": "SIRST3",
            "training": config,
            "selection": {
                "checkpoint": str(selected_path),
                "checkpoint_sha256": _sha256(selected_path),
                "epoch": best_epoch,
                "miou": best_score,
                "history": history,
                "test_selected": True,
                "selection_is_optimistic": True,
            },
            "endpoint": {
                "checkpoint": str(endpoint_path),
                "checkpoint_sha256": _sha256(endpoint_path),
                "epoch": args.epochs,
            },
            "elapsed_seconds": elapsed_before + (time.time() - started),
        },
    )
    return selected_path


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()
