#!/usr/bin/env python3
"""Run fresh-seed, NUDT-only EviSIRST confirmatory replications.

Formal runs use the complete NUDT train/test split for 1000 epochs.  Epochs
1--499 are training-only and epochs 500--1000 evaluate the full test split
after every epoch.  The historical strict role keys retain independent
best-mIoU and best-Pd states.  This deliberately preserves the prior work's
optimistic, test-selected checkpoint convention.

Recovery state, history and summaries stay below ``runs/``.  A completed
formal seed publishes exactly two checkpoints below
``results1/NUDT-SIRST/seed_<seed>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments import evisirst_data as data_module
from experiments.evisirst_data import EviSIRSTTestDataset, stable_uint63
from experiments.evisirst_multiseed import (
    CERTIFICATION_SOURCE_LOCK_SHA256,
    COMPATIBILITY_SEED,
    CONFIRMATORY_SEED_POOL,
    CONFIRMATORY_SEEDS,
    NUDT_DATASET,
    NUDTMultiseedTrainDataset,
    REPLICATION_SEED_CONTRACT_FILE_SHA256,
    REPLICATION_SEED_CONTRACT_SCHEMA,
    initialize_multiseed_evisirst,
    require_supported_seed,
)
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
from train_dual_role_test_selected import (
    _metric_row,
    _validate_history,
    role_key,
    role_key_record,
    selection_due,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "nudt_multiseed_v1"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results1"
FORMAL_DATASET_ROOT = Path("/home/ly/SCTransNet_main/datasets")
RECOVERY_SCHEMA = "evisirst_nudt_multiseed_dual_role_recovery/v1"
SUMMARY_SCHEMA = "evisirst_nudt_multiseed_dual_role_summary/v1"
PROTOCOL_NAME = "evisirst_nudt_confirmatory_multiseed_dual_role_v1"
EXPECTED_TRAIN_COUNT = 663
EXPECTED_TEST_COUNT = 664
FORMAL_EPOCHS = 1000
FORMAL_SELECTION_BEGIN = 500
FORMAL_SELECTION_EVERY = 1
FORMAL_BATCH_SIZE = 16
FORMAL_PATCH_SIZE = 256
FORMAL_WORKERS = 0
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_CPU_THREADS = 16
EXPECTED_TRAIN_IMAGE_MASK_TREE_SHA256 = (
    "25def703a82ac17716824b201d686cd729903b035fef2f25d08ee887b7b3ba5d"
)
EXPECTED_TEST_IMAGE_MASK_TREE_SHA256 = (
    "349fb5f06a35eb75d3304b65efde9c9ef966d32222e73c08f120ee3eec4a0406"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(COMPATIBILITY_SEED, *CONFIRMATORY_SEEDS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=FORMAL_PATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument("--warmup-epochs", type=int, default=FORMAL_WARMUP_EPOCHS)
    parser.add_argument("--cpu-threads", type=int, default=FORMAL_CPU_THREADS)
    parser.add_argument("--selection-begin", type=int, default=FORMAL_SELECTION_BEGIN)
    parser.add_argument("--selection-every", type=int, default=FORMAL_SELECTION_EVERY)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-images", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="permit reduced settings and isolate all outputs below smoke/",
    )
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.patch_size != FORMAL_PATCH_SIZE:
        parser.error("the EviSIRST protocol fixes --patch-size at 256")
    if args.workers < 0 or args.cpu_threads < 1:
        parser.error("--workers must be non-negative and --cpu-threads positive")
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
    if not args.smoke:
        try:
            require_supported_seed(args.seed, formal=True)
        except ValueError as exc:
            parser.error(str(exc))
        if (
            args.epochs != FORMAL_EPOCHS
            or args.batch_size != FORMAL_BATCH_SIZE
            or args.workers != FORMAL_WORKERS
            or args.base_lr != FORMAL_BASE_LR
            or args.min_lr != FORMAL_MIN_LR
            or args.warmup_epochs != FORMAL_WARMUP_EPOCHS
            or args.cpu_threads != FORMAL_CPU_THREADS
            or args.selection_begin != FORMAL_SELECTION_BEGIN
            or args.selection_every != FORMAL_SELECTION_EVERY
            or args.max_train_samples is not None
            or args.max_test_images is not None
        ):
            parser.error("formal mode fixes the complete NUDT replication protocol")
    return args


def _configure_cpu_threads(count: int) -> None:
    torch.set_num_threads(count)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError(
                "PyTorch inter-op threads were initialized before the runner"
            ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _file_tree_sha256(
    dataset_root: Path, entries: list[tuple[str, str, Path]]
) -> str:
    """Bind ordered roles, IDs, relative paths, and exact file bytes."""

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


def _data_tree_entries(
    train_dataset: NUDTMultiseedTrainDataset,
    test_dataset: EviSIRSTTestDataset,
) -> tuple[list[tuple[str, str, Path]], list[tuple[str, str, Path]]]:
    train_entries: list[tuple[str, str, Path]] = []
    for index in range(len(train_dataset)):
        image_path, mask_path, sample_id, source_dataset = train_dataset._paths(index)
        train_entries.extend(
            [
                (f"{source_dataset}:image", sample_id, image_path),
                (f"{source_dataset}:mask", sample_id, mask_path),
            ]
        )
    test_entries: list[tuple[str, str, Path]] = []
    for sample_id in test_dataset.sample_ids:
        sample = data_module.source_protocol.resolve_sample(
            test_dataset.dataset_root,
            NUDT_DATASET,
            sample_id,
            split="test",
            known_ids=test_dataset._known_ids,
        )
        test_entries.extend(
            [
                (f"{NUDT_DATASET}:image", sample_id, sample.image_path),
                (f"{NUDT_DATASET}:mask", sample_id, sample.mask_path),
            ]
        )
    return train_entries, test_entries


def _source_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT / "train_nudt_multiseed_dual_role.py",
        PROJECT_ROOT / "train_dual_role_test_selected.py",
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "test.py",
        PROJECT_ROOT / "experiments" / "evisirst_multiseed.py",
        PROJECT_ROOT / "experiments" / "evisirst_data.py",
        PROJECT_ROOT / "experiments" / "three_dataset_v2_protocol.py",
        PROJECT_ROOT / "experiments" / "four_dataset_models_seed42_v1.py",
        PROJECT_ROOT / "model" / "EviSIRST.py",
        PROJECT_ROOT / "model" / "__init__.py",
        *sorted((PROJECT_ROOT / "model" / "_internal").glob("*.py")),
    ]
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise RuntimeError("NUDT multi-seed training source tree is incomplete")
    return paths


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in _source_paths():
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        content_digest = bytes.fromhex(_sha256(path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content_digest)
    return digest.hexdigest()


def _formal_config(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    train_count: int,
    test_count: int,
    train_index_sha256: str,
    test_index_sha256: str,
    train_tree_sha256: str,
    test_tree_sha256: str,
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
        "initialization": "true_scratch_full_run_seed",
        "baseline_checkpoint_loaded": False,
        "dataset": NUDT_DATASET,
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "replication_seed_contract_schema": REPLICATION_SEED_CONTRACT_SCHEMA,
        "replication_seed_contract_file_sha256": (
            REPLICATION_SEED_CONTRACT_FILE_SHA256
        ),
        "certification_source_lock_sha256": CERTIFICATION_SOURCE_LOCK_SHA256,
        "confirmatory_seed_pool": list(CONFIRMATORY_SEED_POOL),
        "confirmatory_seed_subset": list(CONFIRMATORY_SEEDS),
        "seed_42_counts_toward_stability": False,
        "engineering_seeds_count_toward_stability": False,
        "pythonhashseed": str(args.seed),
        "global_rng_streams": ["python", "numpy", "torch_cpu", "torch_cuda"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "workers": args.workers,
        "cpu_intraop_threads": args.cpu_threads,
        "cpu_interop_threads": 1,
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
        "data_tree_hash_algorithm": (
            "ordered_length_prefixed(role,sample_id,relative_path,file_sha256)"
        ),
        "observed_train_image_mask_tree_sha256": train_tree_sha256,
        "observed_test_image_mask_tree_sha256": test_tree_sha256,
        "expected_train_image_mask_tree_sha256": (
            EXPECTED_TRAIN_IMAGE_MASK_TREE_SHA256
        ),
        "expected_test_image_mask_tree_sha256": (
            EXPECTED_TEST_IMAGE_MASK_TREE_SHA256
        ),
        "normalization": dict(normalization),
        "shuffle_seed_algorithm": (
            "sha256_length_prefixed_str_parts_uint63(seed,dataset,shuffle,epoch)"
        ),
        "crop_seed_algorithm": (
            "historical_source_namespaced_sha256_uint64_with_full_run_seed"
        ),
        "initialization_subseed_algorithm": (
            "sha256(canonical_compact_json([run_seed,namespace]))[:8]_uint64_be"
        ),
        "selection_begin_epoch": args.selection_begin,
        "selection_end_epoch": args.epochs,
        "selection_every": args.selection_every,
        "selection_roles": {
            "best_miou": [
                "miou", "pd", "-fa", "niou", "tiny_pd", "-test_loss", "-epoch"
            ],
            "best_pd": [
                "pd", "-fa", "tiny_pd", "miou", "niou", "-test_loss", "-epoch"
            ],
        },
        "selection_comparison": "strict_lexicographic_greater_than",
        "selection_tie_rule": "earliest_exact_role_key_wins",
        "selection_split": "NUDT-SIRST_test",
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
        or payload.get("dataset") != NUDT_DATASET
        or payload.get("seed") != config["seed"]
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
        "dataset": NUDT_DATASET,
        "checkpoint_role": role,
        "epoch": epoch,
        "seed": int(config["seed"]),
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
    summary_path: Path,
    published_paths: Mapping[str, Path],
    *,
    seed: int,
) -> None:
    if summary_path.is_symlink() or not summary_path.is_file() or any(
        path.is_symlink() or not path.is_file() for path in published_paths.values()
    ):
        raise ValueError("completed NUDT publication is not regular files")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    published = summary.get("published_checkpoints") if isinstance(summary, Mapping) else None
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("dataset") != NUDT_DATASET
        or summary.get("seed") != seed
        or not isinstance(published, Mapping)
    ):
        raise ValueError("completed NUDT publication summary differs")
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
            or checkpoint.get("dataset") != NUDT_DATASET
            or checkpoint.get("seed") != seed
            or checkpoint.get("checkpoint_role") != role
            or checkpoint.get("test_selected") is not True
            or checkpoint.get("selection_is_optimistic") is not True
        ):
            raise ValueError("completed NUDT checkpoint identity differs")


def _check_python_hash_seed(args: argparse.Namespace) -> None:
    observed = os.environ.get("PYTHONHASHSEED")
    if not args.smoke and observed != str(args.seed):
        raise RuntimeError(
            f"formal mode requires PYTHONHASHSEED={args.seed}; observed {observed!r}"
        )
    if observed is not None:
        try:
            parsed = int(observed)
        except ValueError as exc:
            raise RuntimeError("PYTHONHASHSEED must be an integer") from exc
        if parsed != args.seed:
            raise RuntimeError("PYTHONHASHSEED differs from --seed")


def run(args: argparse.Namespace) -> Path:
    _check_python_hash_seed(args)
    _configure_cpu_threads(args.cpu_threads)
    configure_determinism(args.seed)
    device = require_device(args.device)
    dataset_root = args.dataset_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    result_root = args.result_root.resolve()
    if not args.smoke:
        if dataset_root != FORMAL_DATASET_ROOT.resolve(strict=True):
            raise ValueError(
                f"formal data must use canonical root {FORMAL_DATASET_ROOT}"
            )
        if output_root != DEFAULT_OUTPUT_ROOT.resolve():
            raise ValueError(f"formal recovery must stay under {DEFAULT_OUTPUT_ROOT}")
        if result_root != DEFAULT_RESULT_ROOT.resolve():
            raise ValueError(f"formal checkpoints must stay under {DEFAULT_RESULT_ROOT}")
    seed_name = f"seed_{args.seed}"
    run_dir = (
        output_root / "smoke" / seed_name if args.smoke else output_root / seed_name
    )
    result_dir = (
        result_root / "smoke" / NUDT_DATASET / seed_name
        if args.smoke
        else result_root / NUDT_DATASET / seed_name
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
            raise ValueError(f"result seed path is not a regular directory: {result_dir}")
        expected_names = {path.name for path in published_paths.values()}
        unexpected = [item for item in result_dir.iterdir() if item.name not in expected_names]
        if unexpected:
            raise FileExistsError(f"result seed directory contains unexpected files: {unexpected}")
    if summary_path.exists():
        if args.resume and all(path.is_file() for path in published_paths.values()):
            _validate_completed_publication(summary_path, published_paths, seed=args.seed)
            latest_path.unlink(missing_ok=True)
            for path in best_recovery_paths.values():
                path.unlink(missing_ok=True)
            return result_dir
        raise FileExistsError("this NUDT seed experiment is already complete")
    if any(path.exists() for path in published_paths.values()) and not args.resume:
        raise FileExistsError(
            "a partial publication exists; pass --resume to validate and finalize it"
        )

    full_train = NUDTMultiseedTrainDataset(
        dataset_root=dataset_root,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    full_test = EviSIRSTTestDataset(
        NUDT_DATASET, NUDT_DATASET, dataset_root=dataset_root
    )
    train_dataset: Any = full_train
    test_dataset: Any = full_test
    if args.max_train_samples is not None:
        train_dataset = Subset(
            full_train, range(min(args.max_train_samples, len(full_train)))
        )
    if args.max_test_images is not None:
        test_dataset = Subset(
            full_test, range(min(args.max_test_images, len(full_test)))
        )
    train_count = len(train_dataset)
    test_count = len(test_dataset)
    if not args.smoke and (
        train_count != EXPECTED_TRAIN_COUNT or test_count != EXPECTED_TEST_COUNT
    ):
        raise RuntimeError("formal NUDT train/test sample count differs")
    if full_train.normalization != full_test.normalization:
        raise RuntimeError("NUDT train/test normalization differs")

    train_entries, test_entries = _data_tree_entries(full_train, full_test)
    train_tree_sha = _file_tree_sha256(dataset_root, train_entries)
    test_tree_sha = _file_tree_sha256(dataset_root, test_entries)
    if not args.smoke and (
        train_tree_sha != EXPECTED_TRAIN_IMAGE_MASK_TREE_SHA256
        or test_tree_sha != EXPECTED_TEST_IMAGE_MASK_TREE_SHA256
    ):
        raise RuntimeError("formal NUDT train/test image-mask tree SHA-256 differs")

    source_sha = _source_tree_sha256()
    config = _formal_config(
        args,
        dataset_root=dataset_root,
        train_count=train_count,
        test_count=test_count,
        train_index_sha256=_index_sha(full_train.sample_ids),
        test_index_sha256=_index_sha(full_test.sample_ids),
        train_tree_sha256=train_tree_sha,
        test_tree_sha256=test_tree_sha,
        normalization=full_train.normalization,
        source_tree_sha256=source_sha,
    )
    model, model_metadata = initialize_multiseed_evisirst(
        NUDT_DATASET, seed=args.seed, training=True
    )
    pair = model_metadata.get("pair")
    if (
        len(model.state_dict()) != 564
        or hasattr(model, "target_survival")
        or model_metadata.get("target_survival_registered") is not False
        or model_metadata.get("method") != "final_scratch"
        or model_metadata.get("training_seed") != args.seed
        or model_metadata.get("warm_start_used") is not False
        or model_metadata.get("parent_checkpoint") is not None
        or not isinstance(pair, Mapping)
        or pair.get("initialization_mode") != "true_scratch"
        or pair.get("training_seed") != args.seed
        or pair.get("parent_checkpoint_load_count") != 0
        or pair.get("warm_start_used") is not False
        or pair.get("optimizer_state_inherited") is not False
    ):
        raise RuntimeError("multi-seed builder did not provide a true-scratch model")
    model.to(device)
    expected_state = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    criterion = nn.BCELoss(reduction="mean")

    evaluation_generator = torch.Generator(device="cpu")
    evaluation_generator.manual_seed(
        stable_uint63(args.seed, NUDT_DATASET, "evaluation")
    )
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
    best_epochs: dict[str, int | None] = {"best_miou": None, "best_pd": None}
    elapsed_before = 0.0
    if args.resume:
        candidates: list[tuple[int, Path]] = []
        for path in (latest_path, *best_recovery_paths.values()):
            if path.is_file() and not path.is_symlink():
                candidate = _load_recovery(path, config=config, expected_state=expected_state)
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
            stable_uint63(args.seed, NUDT_DATASET, "shuffle", epoch)
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
            epoch, args.epochs, args.base_lr, args.min_lr, args.warmup_epochs
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
            raise RuntimeError("processed NUDT training sample count differs")

        improved_roles: list[str] = []
        metrics: dict[str, Any] | None = None
        if selection_due(epoch, args.selection_begin, args.selection_every):
            training_rng = _capture_rng_state(device)
            try:
                metrics = _metric_row(evaluate_model(model, evaluation_loader, device), epoch)
            finally:
                _restore_rng_state(training_rng, device)
            history.append(metrics)
            for role in ("best_miou", "best_pd"):
                previous_epoch = best_epochs[role]
                previous = (
                    None
                    if previous_epoch is None
                    else next(
                        row
                        for row in history[:-1]
                        if int(row["epoch"]) == previous_epoch
                    )
                )
                if previous is None or role_key(metrics, role) > role_key(previous, role):
                    best_epochs[role] = epoch
                    improved_roles.append(role)

        state = _cpu_state(model)
        elapsed = elapsed_before + time.time() - started
        recovery = {
            "schema": RECOVERY_SCHEMA,
            "model": "EviSIRST",
            "dataset": NUDT_DATASET,
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
                "dataset": NUDT_DATASET,
                "seed": args.seed,
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
            f"dataset={NUDT_DATASET} seed={args.seed} epoch={epoch}/{args.epochs} "
            f"loss={loss_sum / processed:.6f} lr={lr:.8f} samples={processed}{suffix}",
            flush=True,
        )

    if any(best_epochs[role] is None for role in ("best_miou", "best_pd")) or any(
        not path.is_file() for path in best_recovery_paths.values()
    ):
        raise RuntimeError("training completed without both selected checkpoints")
    end_train = NUDTMultiseedTrainDataset(
        dataset_root=dataset_root,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    end_test = EviSIRSTTestDataset(
        NUDT_DATASET, NUDT_DATASET, dataset_root=dataset_root
    )
    end_train_entries, end_test_entries = _data_tree_entries(end_train, end_test)
    if (
        _source_tree_sha256() != source_sha
        or len(end_train) != len(full_train)
        or len(end_test) != len(full_test)
        or _index_sha(end_train.sample_ids) != config["train_index_order_sha256"]
        or _index_sha(end_test.sample_ids) != config["test_index_order_sha256"]
        or end_train.normalization != config["normalization"]
        or end_test.normalization != config["normalization"]
        or _file_tree_sha256(dataset_root, end_train_entries)
        != config["observed_train_image_mask_tree_sha256"]
        or _file_tree_sha256(dataset_root, end_test_entries)
        != config["observed_test_image_mask_tree_sha256"]
    ):
        raise RuntimeError("source/data identity changed during training")
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
            or reopened.get("dataset") != NUDT_DATASET
            or reopened.get("seed") != args.seed
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
        published_records[role] = {"path": str(path), "sha256": checkpoint_sha}
    if {path.name for path in result_dir.iterdir()} != {
        path.name for path in published_paths.values()
    }:
        raise RuntimeError("result seed directory does not contain exactly two checkpoints")
    _write_json(
        summary_path,
        {
            "schema": SUMMARY_SCHEMA,
            "status": "complete",
            "model": "EviSIRST",
            "dataset": NUDT_DATASET,
            "seed": args.seed,
            "training": config,
            "candidate_count": len(history),
            "selections": selections,
            "published_checkpoints": published_records,
            "only_two_checkpoints_in_result_seed_directory": True,
            "selection_history": history,
            "elapsed_seconds": elapsed_before + time.time() - started,
        },
    )
    latest_path.unlink()
    for path in best_recovery_paths.values():
        path.unlink()
    return result_dir


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()
