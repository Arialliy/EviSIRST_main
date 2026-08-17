"""Fresh-seed construction and augmentation for NUDT EviSIRST replications.

The released EviSIRST paper implementation deliberately freezes the original
development run to seed 42.  Confirmatory replications need the same graph and
initialization policy while allowing a *single run seed* to control every
random stream.  This module provides that narrow extension without modifying
the frozen seed-42 implementation.

Seed 42 remains a compatibility oracle: :func:`initialize_multiseed_evisirst`
is bitwise identical to ``model.EviSIRST.initialize_evisirst`` for that seed.
The three fresh confirmatory seeds are fixed by the replication contract and
are the only seeds accepted by the formal NUDT runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from experiments import evisirst_data as frozen_data
from experiments import four_dataset_models_seed42_v1 as frozen_builder
from experiments import three_dataset_v2_protocol as source_protocol
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
    replace_shallow_embeddings_clean_v8_mprs_dch,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    DEFAULT_TAIL_Z_THRESHOLDS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    QFG_TERMINAL_STATE_KEYS,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
)


MULTISEED_BUILDER_SCHEMA = "evisirst_confirmatory_multiseed_builder/v1"
MULTISEED_CHECKPOINT_SCHEMA = "evisirst_clean_checkpoint/v1"
REPLICATION_SEED_CONTRACT_SCHEMA = (
    "sctransnet_final_model_replication_seed_contract_v1"
)
REPLICATION_SEED_CONTRACT_FILE_SHA256 = (
    "e50923dbcbf3ab401478d8b3d784442a07434735667e9388c7055184b0f766b7"
)
CERTIFICATION_SOURCE_LOCK_SHA256 = (
    "d6334b4f863e06cd0fa744723025b6bdf1fe76a7d0664cfe84d472a19e09d13f"
)
CONFIRMATORY_SEED_POOL = (
    1446202191,
    104728269,
    262620274,
    807777981,
    1912501927,
)
CONFIRMATORY_SEEDS = (1446202191, 104728269, 262620274)
COMPATIBILITY_SEED = 42
SUPPORTED_SEEDS = (COMPATIBILITY_SEED, *CONFIRMATORY_SEEDS)
NUDT_DATASET = "NUDT-SIRST"
FINAL_STATE_KEY_COUNT = 564
FINAL_RANDOM_NAMESPACES = ("tpd", "ner", "qfg")


def require_supported_seed(seed: int, *, formal: bool = False) -> int:
    """Validate a compatibility or fresh confirmatory seed."""

    allowed = CONFIRMATORY_SEEDS if formal else SUPPORTED_SEEDS
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in allowed:
        label = "confirmatory" if formal else "supported"
        raise ValueError(f"seed must be one of the {label} seeds {allowed}, got {seed!r}")
    return seed


def stable_initialization_seed(seed: int, namespace: str) -> int:
    """Derive the exact builder substream seed used by the seed-42 oracle."""

    require_supported_seed(seed)
    if type(namespace) is not str or namespace not in FINAL_RANDOM_NAMESPACES:
        raise ValueError(
            f"initialization namespace must be one of {FINAL_RANDOM_NAMESPACES}"
        )
    payload = json.dumps(
        [seed, namespace], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _construct_original(seed: int) -> SCTransNet:
    """Reproduce the frozen Original scratch graph for an arbitrary run seed."""

    require_supported_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        with redirect_stdout(StringIO()):
            model = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
        model.apply(frozen_builder._weights_init_kaiming)
    model.train()
    return model


def _construct_raw_final(seed: int) -> nn.Module:
    """Construct the clean 564-key, no-TSS EviSIRST graph from modules only."""

    require_supported_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        with redirect_stdout(StringIO()):
            parent = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
        parent.apply(frozen_builder._weights_init_kaiming)
        replacements = replace_shallow_embeddings_clean_v8_mprs_dch(
            parent, PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT
        )
        if set(replacements) != {"embeddings_1", "embeddings_2"}:
            raise RuntimeError("Final construction replaced unexpected modules")
        for replacement in replacements.values():
            replacement.apply(frozen_builder._weights_init_kaiming)
        model = TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet(
            parent,
            variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            relay_width=DEFAULT_RELAY_WIDTH,
            relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
            dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
            tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
        )
    model.train()
    return model


def _all_zero(state: Mapping[str, torch.Tensor], keys: tuple[str, ...]) -> bool:
    return all(int(torch.count_nonzero(state[key]).item()) == 0 for key in keys)


def _build_pair(seed: int) -> tuple[SCTransNet, nn.Module, dict[str, Any]]:
    """Build and audit a paired Original/Final initialization."""

    original = _construct_original(seed)
    final = _construct_raw_final(seed)
    derived = {
        namespace: stable_initialization_seed(seed, namespace)
        for namespace in FINAL_RANDOM_NAMESPACES
    }
    frozen_builder._initialize_tpd_substream(final, derived["tpd"])
    frozen_builder._initialize_ner_substream(final, derived["ner"])
    frozen_builder._initialize_qfg_substream(final, derived["qfg"])
    shared, original_only, final_only = frozen_builder._copy_shared_state(
        original, final
    )
    frozen_builder._validate_built_pair(
        original,
        final,
        with_tss=False,
        shared=shared,
        final_only=final_only,
    )
    original_state = original.state_dict()
    final_state = final.state_dict()
    shared_hash = frozen_builder.state_dict_sha256(original_state, shared)
    if shared_hash != frozen_builder.state_dict_sha256(final_state, shared):
        raise RuntimeError("paired Original/Final shared state differs")
    if len(final_state) != FINAL_STATE_KEY_COUNT:
        raise RuntimeError("clean EviSIRST state-key count differs")
    if hasattr(final, "target_survival") or any(
        key.startswith("target_survival") for key in final_state
    ):
        raise RuntimeError("clean EviSIRST graph unexpectedly contains TSS")
    if not _all_zero(final_state, tuple(QFG_TERMINAL_STATE_KEYS)):
        raise RuntimeError("QFG identity terminals are not exactly zero")
    pair_metadata = {
        "schema": MULTISEED_BUILDER_SCHEMA,
        "dataset_name": NUDT_DATASET,
        "training_seed": seed,
        "allowed_training_seeds": list(CONFIRMATORY_SEEDS),
        "replication_seed_contract_schema": REPLICATION_SEED_CONTRACT_SCHEMA,
        "replication_seed_contract_file_sha256": (
            REPLICATION_SEED_CONTRACT_FILE_SHA256
        ),
        "certification_source_lock_sha256": CERTIFICATION_SOURCE_LOCK_SHA256,
        "confirmatory_seed_pool": list(CONFIRMATORY_SEED_POOL),
        "confirmatory_seed_subset": list(CONFIRMATORY_SEEDS),
        "compatibility_seed": COMPATIBILITY_SEED,
        "initialization_mode": "true_scratch",
        "parent_checkpoint": None,
        "parent_checkpoint_load_count": 0,
        "warm_start_used": False,
        "optimizer_state_inherited": False,
        "scheduler_state_inherited": False,
        "paired_initialization": True,
        "model_construction_preserves_caller_rng_stream": True,
        "shared_state_match_rule": "same_name_same_shape_same_dtype",
        "shared_state_key_count": len(shared),
        "shared_state_keys": list(shared),
        "shared_state_sha256": shared_hash,
        "shared_state_bitwise_equal": True,
        "original_only_state_key_count": len(original_only),
        "original_only_state_keys": list(original_only),
        "final_only_state_key_count": len(final_only),
        "final_only_state_keys": list(final_only),
        "final_only_state_sha256": frozen_builder.state_dict_sha256(
            final_state, final_only
        ),
        "derived_initialization_seed_algorithm": (
            "sha256(canonical_compact_json([run_seed,namespace]))[:8]_uint64_be"
        ),
        "derived_initialization_seeds": dict(derived),
        "derived_seeds_are_additional_training_seeds": False,
        "qfg_terminal_zero_initialized": True,
        "original_state_key_count": len(original_state),
        "original_state_sha256": frozen_builder.state_dict_sha256(original_state),
        "final_state_key_count": len(final_state),
        "final_state_sha256": frozen_builder.state_dict_sha256(final_state),
        "final_training_graph": (
            "sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa_no_tss"
        ),
        "baseline_checkpoint_loaded": False,
    }
    return original, final, pair_metadata


def initialize_multiseed_evisirst(
    dataset: str,
    *,
    seed: int,
    training: bool = True,
) -> tuple[TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet, dict[str, Any]]:
    """Create a clean EviSIRST scratch model controlled by ``seed``."""

    if dataset != NUDT_DATASET:
        raise ValueError(f"this replication module supports only {NUDT_DATASET!r}")
    require_supported_seed(seed)
    _, model, pair_metadata = _build_pair(seed)
    if not isinstance(model, TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet):
        raise TypeError("builder returned an unexpected Final architecture")
    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()
    metadata = {
        "schema": MULTISEED_BUILDER_SCHEMA,
        "method": "final_scratch",
        "training_graph_requested": bool(training),
        "dataset_name": dataset,
        "training_seed": seed,
        "pair": pair_metadata,
        "selected_model_state_sha256": frozen_builder.state_dict_sha256(
            model.state_dict()
        ),
        "selected_model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "selected_model_state_key_count": len(model.state_dict()),
        "warm_start_used": False,
        "parent_checkpoint": None,
        "public_model": "EviSIRST",
        "target_survival_registered": False,
        "state_key_count": FINAL_STATE_KEY_COUNT,
        "training_mode": bool(training),
    }
    return model, metadata


def multiseed_transform_plan(
    *,
    seed: int,
    dataset: str,
    source_dataset: str,
    sample_id: str,
    epoch: int,
    height: int,
    width: int,
    mask_any: np.ndarray,
) -> source_protocol.StatelessTransformPlan:
    """Seed-aware form of the frozen stateless crop/flip/transpose plan."""

    require_supported_seed(seed)
    if dataset != NUDT_DATASET or source_dataset != NUDT_DATASET:
        raise ValueError("multi-seed transforms are restricted to NUDT-SIRST")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    patch_size = source_protocol.PATCH_SIZE
    padded_height = max(height, patch_size)
    padded_width = max(width, patch_size)
    namespaced_id = f"{source_dataset}::{sample_id}"
    augmentation_seed = source_protocol.stable_sha256_uint64(
        seed, dataset, epoch, namespaced_id
    )
    rng = random.Random(augmentation_seed)
    attempts = 0
    while True:
        attempts += 1
        top = rng.randint(0, padded_height - patch_size)
        left = rng.randint(0, padded_width - patch_size)
        if (
            rng.random() > source_protocol.TRAIN_POSITIVE_CROP_PROBABILITY
            or bool(np.any(mask_any[top : top + patch_size, left : left + patch_size]))
        ):
            break
        if attempts >= 1_000_000:
            raise frozen_data.EviSIRSTDataError(
                "positive-biased crop exceeded safety limit"
            )
    return source_protocol.StatelessTransformPlan(
        augmentation_seed=augmentation_seed,
        crop_top=top,
        crop_left=left,
        crop_size=patch_size,
        padded_height=padded_height,
        padded_width=padded_width,
        crop_attempts=attempts,
        flip_axis0=rng.random() < 0.5,
        flip_axis1=rng.random() < 0.5,
        transpose=rng.random() < 0.5,
    )


class NUDTMultiseedTrainDataset(frozen_data.EviSIRSTTrainDataset):
    """Frozen NUDT split with the confirmatory run seed in all transforms."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        patch_size: int = source_protocol.PATCH_SIZE,
        seed: int,
        return_metadata: bool = False,
    ) -> None:
        require_supported_seed(seed)
        # The frozen constructor performs all split/index/data checks.  Seed 42
        # is supplied only to pass that constructor's legacy guard; every
        # stochastic transform below uses the requested run seed.
        super().__init__(
            NUDT_DATASET,
            dataset_root=dataset_root,
            patch_size=patch_size,
            seed=COMPATIBILITY_SEED,
            return_metadata=return_metadata,
        )
        self.seed = seed

    def __getitem__(self, index: int) -> Any:
        image_path, mask_path, sample_id, source_dataset = self._paths(index)
        image, raw_mask = frozen_data._load_pair(image_path, mask_path)
        original_height, original_width = image.shape
        image = (image - np.float32(self.normalization["mean"])) / np.float32(
            self.normalization["std"]
        )
        mask = raw_mask / np.float32(255.0)
        plan = multiseed_transform_plan(
            seed=self.seed,
            dataset=self.dataset_name,
            source_dataset=source_dataset,
            sample_id=sample_id,
            epoch=self.epoch,
            height=original_height,
            width=original_width,
            mask_any=mask > 0,
        )
        image = frozen_data._pad(image, plan.padded_height, plan.padded_width)
        mask = frozen_data._pad(mask, plan.padded_height, plan.padded_width)
        top, left, size = plan.crop_top, plan.crop_left, plan.crop_size
        image = image[top : top + size, left : left + size]
        mask = mask[top : top + size, left : left + size]
        if plan.flip_axis0:
            image, mask = image[::-1, :], mask[::-1, :]
        if plan.flip_axis1:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if plan.transpose:
            image, mask = image.transpose(1, 0), mask.transpose(1, 0)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[np.newaxis, :], dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask[np.newaxis, :], dtype=np.float32)
        )
        if not self.return_metadata:
            return image_tensor, mask_tensor
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset_name": self.dataset_name,
            "source_dataset": source_dataset,
            "sample_id": sample_id,
            "namespaced_sample_id": f"{source_dataset}::{sample_id}",
            "original_hw": (original_height, original_width),
            "epoch": self.epoch,
            "augmentation_seed": plan.augmentation_seed,
            "transform_plan": asdict(plan),
        }


def _validate_checkpoint_state(
    value: Any, expected: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or list(value) != list(expected):
        raise ValueError("checkpoint state key/order differs from clean EviSIRST")
    for key, tensor in value.items():
        reference = expected[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint state {key!r} is not a tensor")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"checkpoint state contract differs for {key!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint state {key!r} is non-finite")
    return value


def load_multiseed_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_seed: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet, dict[str, Any]]:
    """Strictly validate and load one NUDT confirmatory result checkpoint.

    This is intentionally independent of the seed-42-only public loader, so a
    result can be passed directly to ``test.evaluate_model`` for replay.
    """

    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"checkpoint is not a regular file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    seed = payload.get("seed")
    require_supported_seed(seed, formal=True)
    if expected_seed is not None and seed != require_supported_seed(
        expected_seed, formal=True
    ):
        raise ValueError("checkpoint seed differs from expected_seed")
    training = payload.get("training")
    selection_metrics = payload.get("selection_metrics")
    if (
        payload.get("schema") != MULTISEED_CHECKPOINT_SCHEMA
        or payload.get("model") != "EviSIRST"
        or payload.get("dataset") != NUDT_DATASET
        or payload.get("checkpoint_role") not in ("best_miou", "best_pd")
        or payload.get("test_selected") is not True
        or payload.get("selection_is_optimistic") is not True
        or not isinstance(training, Mapping)
        or training.get("seed") != seed
        or training.get("dataset") != NUDT_DATASET
        or not isinstance(selection_metrics, Mapping)
        or any(
            isinstance(selection_metrics.get(field), bool)
            or not isinstance(selection_metrics.get(field), (int, float))
            or not math.isfinite(float(selection_metrics[field]))
            for field in ("miou", "pd")
        )
    ):
        raise ValueError("checkpoint identity/provenance differs")
    model, initialization = initialize_multiseed_evisirst(
        NUDT_DATASET, seed=seed, training=False
    )
    state = _validate_checkpoint_state(payload.get("state_dict"), model.state_dict())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load returned incompatible keys")
    target = torch.device(device)
    model.to(target)
    model.mode = "test"
    model.eval()
    metadata = {
        "path": str(path.resolve()),
        "seed": seed,
        "dataset": NUDT_DATASET,
        "checkpoint_role": payload["checkpoint_role"],
        "epoch": int(payload["epoch"]),
        "selection_metrics": dict(selection_metrics),
        "test_selected": True,
        "selection_is_optimistic": True,
        "state_sha256": frozen_builder.state_dict_sha256(model.state_dict()),
        "initialization": initialization,
    }
    return model, metadata


__all__ = [
    "COMPATIBILITY_SEED",
    "CERTIFICATION_SOURCE_LOCK_SHA256",
    "CONFIRMATORY_SEED_POOL",
    "CONFIRMATORY_SEEDS",
    "FINAL_STATE_KEY_COUNT",
    "MULTISEED_BUILDER_SCHEMA",
    "NUDT_DATASET",
    "NUDTMultiseedTrainDataset",
    "REPLICATION_SEED_CONTRACT_FILE_SHA256",
    "REPLICATION_SEED_CONTRACT_SCHEMA",
    "SUPPORTED_SEEDS",
    "initialize_multiseed_evisirst",
    "load_multiseed_checkpoint",
    "multiseed_transform_plan",
    "require_supported_seed",
    "stable_initialization_seed",
]
