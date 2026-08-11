#!/usr/bin/env python3
"""Train the clean 564-key, no-TSS EviSIRST model from scratch.

This public runner keeps the historical segmentation recipe but deliberately
does not inspect the test split or select checkpoints on test performance.
It saves a resumable training state and one final ``EviSIRST.pth.tar``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments.evisirst_data import (
    EviSIRSTTrainDataset,
    TRAINING_DATASETS,
    stable_uint63,
)
from model.EviSIRST import initialize_evisirst


SCHEMA = "evisirst_clean_training/v1"
CHECKPOINT_SCHEMA = "evisirst_clean_checkpoint/v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=TRAINING_DATASETS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, help="smoke-test subset")
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("the frozen EviSIRST initializer supports seed 42 only")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.patch_size != 256:
        parser.error("the paper training protocol fixes --patch-size at 256")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if (
        not math.isfinite(args.base_lr)
        or not math.isfinite(args.min_lr)
        or not 0.0 < args.min_lr <= args.base_lr
    ):
        parser.error("learning rates must satisfy 0 < min-lr <= base-lr")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be between 0 and --epochs")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    return args


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def require_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {index}")
    return device


def learning_rate_for_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    decay_epochs = total_epochs - warmup_epochs
    if decay_epochs <= 0:
        return base_lr
    progress = (epoch - warmup_epochs) / decay_epochs
    return min_lr + 0.5 * (base_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def deep_supervision_loss(
    output: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if not isinstance(output, (tuple, list)) or len(output) != 6:
        raise RuntimeError("training-mode EviSIRST must return six probability maps")
    terms: list[torch.Tensor] = []
    for index, probability in enumerate(output):
        if not isinstance(probability, torch.Tensor) or probability.shape != target.shape:
            raise RuntimeError(f"deep-supervision output {index} has the wrong shape")
        if not torch.isfinite(probability).all():
            raise FloatingPointError(f"deep-supervision output {index} is non-finite")
        if bool((probability < 0).any()) or bool((probability > 1).any()):
            raise RuntimeError(
                f"deep-supervision output {index} is not a probability map"
            )
        terms.append(criterion(probability.float(), target.float()))
    return sum(terms)


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    if len(state) != 564 or any(key.startswith("target_survival") for key in state):
        raise RuntimeError("EviSIRST checkpoint is not the clean 564-key graph")
    return state


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if temporary.exists():
            temporary.unlink()


def _index_sha(sample_ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sample_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _training_config(
    args: argparse.Namespace,
    sample_count: int,
    train_index_order_sha256: str,
) -> dict[str, Any]:
    return {
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
        "test_split_accessed": False,
        "sample_count": sample_count,
        "train_index_order_sha256": train_index_order_sha256,
        "smoke_subset": args.max_samples is not None,
        "data_protocol": "evisirst_public_joint_training_v1",
        "shuffle_seed_algorithm": (
            "sha256_length_prefixed_str_parts_uint63(seed,dataset,shuffle,epoch)"
        ),
        "crop_seed_algorithm": (
            "historical_source_namespaced_sha256_uint64"
        ),
    }


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    numpy_name, numpy_keys, numpy_pos, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_name,
            "keys": torch.from_numpy(numpy_keys.copy()),
            "position": int(numpy_pos),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
        "device_type": device.type,
    }


def _restore_rng_state(payload: Any, device: torch.device) -> None:
    if not isinstance(payload, Mapping) or payload.get("device_type") != device.type:
        raise ValueError("resume RNG/device contract differs")
    python_state = payload.get("python")
    numpy_state = payload.get("numpy")
    torch_cpu = payload.get("torch_cpu")
    torch_cuda = payload.get("torch_cuda")
    if not isinstance(numpy_state, Mapping) or not isinstance(
        numpy_state.get("keys"), torch.Tensor
    ):
        raise ValueError("resume NumPy RNG state is malformed")
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8:
        raise ValueError("resume CPU RNG state is malformed")
    random.setstate(python_state)
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu.cpu())
    if device.type == "cuda":
        if not isinstance(torch_cuda, torch.Tensor) or torch_cuda.dtype != torch.uint8:
            raise ValueError("resume CUDA RNG state is malformed")
        torch.cuda.set_rng_state(torch_cuda.cpu(), device=device)
    elif torch_cuda is not None:
        raise ValueError("CPU resume unexpectedly contains CUDA RNG state")


def run(args: argparse.Namespace) -> Path:
    configure_determinism(args.seed)
    device = require_device(args.device)

    full_dataset = EviSIRSTTrainDataset(
        args.dataset,
        dataset_root=args.dataset_root,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    train_dataset: EviSIRSTTrainDataset | Subset = full_dataset
    if args.max_samples is not None:
        train_dataset = Subset(
            full_dataset,
            range(min(args.max_samples, len(full_dataset))),
        )
    sample_count = len(train_dataset)
    train_index_order_sha256 = _index_sha(full_dataset.sample_ids)
    run_dir = args.output_root.resolve() / args.dataset
    if args.max_samples is not None:
        run_dir = args.output_root.resolve() / "smoke" / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "EviSIRST.pth.tar"
    latest_path = run_dir / "last_training_state.pth.tar"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"completed run already exists: {summary_path}")

    model, model_metadata = initialize_evisirst(
        args.dataset,
        seed=args.seed,
        training=True,
    )
    model.to(device)
    if len(model.state_dict()) != 564 or hasattr(model, "target_survival"):
        raise RuntimeError("train.py requires the clean 564-key no-TSS graph")
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    config = _training_config(args, sample_count, train_index_order_sha256)
    start_epoch = 1
    if args.resume:
        if not latest_path.is_file() or latest_path.is_symlink():
            raise FileNotFoundError(latest_path)
        payload = torch.load(latest_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != SCHEMA
            or payload.get("dataset") != args.dataset
            or payload.get("seed") != args.seed
            or payload.get("training") != config
        ):
            raise ValueError("resume checkpoint identity differs")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        completed_epoch = int(payload["epoch"])
        if completed_epoch < 0 or completed_epoch > args.epochs:
            raise ValueError("resume checkpoint epoch is outside this run")
        start_epoch = completed_epoch + 1
        _restore_rng_state(payload.get("rng"), device)
    elif latest_path.exists() or final_path.exists():
        raise FileExistsError(f"run directory already contains checkpoints: {run_dir}")

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        full_dataset.set_epoch(epoch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            stable_uint63(args.seed, args.dataset, "shuffle", epoch)
        )
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=generator,
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
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = deep_supervision_loss(model(images), masks, criterion)
            loss.backward()
            optimizer.step()
            count = int(images.shape[0])
            loss_sum += float(loss.detach().item()) * count
            processed += count
        if processed != sample_count:
            raise RuntimeError("processed training sample count differs")
        state = _cpu_state(model)
        _atomic_torch_save(
            latest_path,
            {
                "schema": SCHEMA,
                "model": "EviSIRST",
                "dataset": args.dataset,
                "epoch": epoch,
                "seed": args.seed,
                "state_dict": state,
                "optimizer": optimizer.state_dict(),
                "training": config,
                "mean_train_loss": loss_sum / processed,
                "learning_rate": lr,
                "rng": _capture_rng_state(device),
            },
        )
        print(
            f"epoch={epoch}/{args.epochs} loss={loss_sum / processed:.6f} "
            f"lr={lr:.8f} samples={processed}",
            flush=True,
        )

    model.eval()
    model.mode = "test"
    final_state = _cpu_state(model)
    final_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model": "EviSIRST",
        "dataset": args.dataset,
        "checkpoint_role": "final",
        "epoch": args.epochs,
        "seed": args.seed,
        "state_dict": final_state,
        "training": config,
        "train_index_order_sha256": train_index_order_sha256,
        "normalization": full_dataset.normalization,
        "test_split_accessed": False,
    }
    _atomic_torch_save(final_path, final_payload)
    _write_json(
        summary_path,
        {
            "schema": SCHEMA,
            "status": "complete",
            "model": "EviSIRST",
            "dataset": args.dataset,
            "epoch": args.epochs,
            "seed": args.seed,
            "checkpoint": str(final_path),
            "state_key_count": len(final_state),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "target_survival_registered": False,
            "training": config,
            "elapsed_seconds": time.time() - started,
            "historical_checkpoint_reproduction_claimed": False,
        },
    )
    return final_path


def main(argv: list[str] | None = None) -> None:
    checkpoint = run(parse_args(argv))
    print(checkpoint)


if __name__ == "__main__":
    main()
