"""Experimental variable-initialization EviSIRST confirmation builder.

The released paper builder intentionally accepts only seed 42.  Confirmation
experiments need the exact same 564-key construction path under additional
initialization seeds.  This module does not copy that construction.  Instead,
it temporarily replaces only the frozen builder's seed guard inside a
non-reentrant process-local critical section, calls the original complete
``build_paper_model('final', ..., training=False)`` entry point, and restores
the exact guard object in ``finally``.

This is an experimental confirmation-only API.  It does not replace
``model.EviSIRST.initialize_evisirst`` and must not be used to reinterpret
the released seed-42 protocol.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from experiments import four_dataset_models_seed42_v1 as frozen_builder
from model.EviSIRST import EviSIRST
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    validate_formal_qfg_v2_croa_inference_model,
)


SCHEMA = "evisirst_variable_initialization_confirmation_builder/v1"
SOURCE_IDENTITY_SCHEMA = "evisirst_variable_initialization_source_identity/v1"
STATUS = "experimental_confirmation_only"
SEED_DOMAIN = "uint32_nonbool"
UINT32_MAX = (1 << 32) - 1
BASE_STATE_KEY_COUNT = 564
BASE_PARAMETER_COUNT = 10_870_130
CONFIRMATION_INIT_SEEDS = (
    1_186_821_503,
    1_664_584_613,
    984_034_075,
    518_560_408,
    432_975_590,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_REQUIRE_SEED = frozen_builder._require_seed
_PATCH_LOCK = threading.Lock()


class EviSIRSTVariableInitError(RuntimeError):
    """The variable-initialization construction contract was violated."""


def require_uint32_seed(seed: Any) -> int:
    """Accept exactly non-boolean Python integers in the uint32 domain."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= UINT32_MAX
    ):
        raise ValueError(
            f"seed must be a non-boolean uint32 integer in [0, {UINT32_MAX}]"
        )
    return seed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_entry(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise EviSIRSTVariableInitError("builder source must be a regular file")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise EviSIRSTVariableInitError(
            "builder source is outside the repository"
        ) from exc
    return {"relative_path": relative, "sha256": _sha256_file(resolved)}


def source_identity() -> dict[str, Any]:
    """Return the current wrapper/frozen-builder source identity."""

    files = {
        "variable_init_wrapper": _source_entry(Path(__file__)),
        "frozen_complete_builder": _source_entry(
            Path(frozen_builder.__file__)
        ),
    }
    return {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


@contextmanager
def _temporary_uint32_seed_guard() -> Iterator[None]:
    """Patch one seed guard under a fail-fast, exception-safe lock."""

    if not _PATCH_LOCK.acquire(blocking=False):
        raise EviSIRSTVariableInitError(
            "variable-initialization builder is already active in this process"
        )
    installed = False
    try:
        if frozen_builder._require_seed is not _ORIGINAL_REQUIRE_SEED:
            raise EviSIRSTVariableInitError(
                "frozen builder seed guard was modified before entry"
            )
        frozen_builder._require_seed = require_uint32_seed
        installed = True
        yield
    finally:
        if installed:
            frozen_builder._require_seed = _ORIGINAL_REQUIRE_SEED
        _PATCH_LOCK.release()


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _derived_substream_seeds(seed: int) -> dict[str, int]:
    derived: dict[str, int] = {}
    for namespace in ("tpd", "ner", "qfg"):
        payload = json.dumps(
            [seed, namespace],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        derived[namespace] = int.from_bytes(
            hashlib.sha256(payload).digest()[:8], "big"
        )
    return derived


def _validate_model(
    model: nn.Module,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    require_uint32_seed(seed)
    if type(model) is not EviSIRST:
        raise TypeError("variable initializer did not return exact EviSIRST")
    if hasattr(model, "target_survival"):
        raise EviSIRSTVariableInitError("variable graph unexpectedly registers TSS")
    state = model.state_dict()
    if len(state) != BASE_STATE_KEY_COUNT:
        raise EviSIRSTVariableInitError(
            "variable graph does not contain exactly 564 state tensors"
        )
    if any(key.startswith("target_survival") for key in state):
        raise EviSIRSTVariableInitError("variable state unexpectedly retains TSS")
    if _parameter_count(model) != BASE_PARAMETER_COUNT:
        raise EviSIRSTVariableInitError("variable graph parameter count differs")
    frozen = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen:
        raise EviSIRSTVariableInitError(
            "variable graph contains frozen parameters: "
            + ", ".join(frozen[:5])
        )
    validation = validate_formal_qfg_v2_croa_inference_model(
        model,
        require_identity_initialized_qfg=True,
    )
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise EviSIRSTVariableInitError("variable state contains a non-Tensor")
    return validation, dict(state)


def initialize_variable_evisirst(
    dataset: str,
    *,
    seed: int,
    training: bool = True,
) -> tuple[EviSIRST, dict[str, Any]]:
    """Build exact EviSIRST under a uint32 confirmation initialization seed."""

    seed = require_uint32_seed(seed)
    if type(dataset) is not str or dataset not in frozen_builder.SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {frozen_builder.SUPPORTED_DATASETS}"
        )
    if type(training) is not bool:
        raise TypeError("training must be bool")

    caller_rng = torch.get_rng_state().clone()
    with _temporary_uint32_seed_guard():
        # The frozen builder already isolates each random subsystem.  The
        # outer fork is a second exception-safe boundary for future internal
        # changes and guarantees this wrapper's caller-RNG contract.
        with torch.random.fork_rng(devices=[]):
            model, wrapped_metadata = frozen_builder.build_paper_model(
                "final",
                dataset,
                seed,
                training=False,
            )
    if not torch.equal(torch.get_rng_state(), caller_rng):
        raise EviSIRSTVariableInitError("model construction changed caller RNG")
    if not isinstance(wrapped_metadata, Mapping):
        raise EviSIRSTVariableInitError("wrapped builder metadata is malformed")
    if (
        wrapped_metadata.get("method") != "final_scratch"
        or wrapped_metadata.get("training_seed") != seed
        or wrapped_metadata.get("selected_model_state_key_count")
        != BASE_STATE_KEY_COUNT
        or wrapped_metadata.get("selected_model_parameter_count")
        != BASE_PARAMETER_COUNT
        or wrapped_metadata.get("warm_start_used") is not False
        or wrapped_metadata.get("parent_checkpoint") is not None
    ):
        raise EviSIRSTVariableInitError("wrapped builder result contract differs")

    # The formal graph validator requires its six-output training mode.
    model.mode = "train"
    model.train()
    validation, state = _validate_model(model, seed=seed)
    if training:
        model.mode = "train"
        model.train()
    else:
        model.mode = "test"
        model.eval()

    source = source_identity()
    state_sha256 = frozen_builder.state_dict_sha256(state)
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "public_api_replacement": False,
        "confirmation_only": True,
        "dataset": dataset,
        "architecture_initialization_seed": seed,
        "seed_domain": SEED_DOMAIN,
        "confirmation_init_seeds": list(CONFIRMATION_INIT_SEEDS),
        "construction_call": (
            "frozen_builder.build_paper_model('final',dataset,seed,training=False)"
        ),
        "temporary_patch_target": (
            "experiments.four_dataset_models_seed42_v1._require_seed"
        ),
        "temporary_patch_scope": "non_reentrant_process_local_critical_section",
        "temporary_patch_restored": (
            frozen_builder._require_seed is _ORIGINAL_REQUIRE_SEED
        ),
        "wrapped_builder_schema": wrapped_metadata.get("schema"),
        "wrapped_static_seed42_pair_metadata_is_not_reused": True,
        "initialization_mode": "true_scratch",
        "parent_checkpoint": None,
        "warm_start_used": False,
        "optimizer_state_inherited": False,
        "state_key_count": len(state),
        "parameter_count": _parameter_count(model),
        "state_sha256": state_sha256,
        "target_survival_registered": False,
        "all_parameters_trainable": True,
        "derived_initialization_seed_algorithm": (
            "sha256(canonical_compact_json([architecture_seed,namespace]))"
            "[:8]_uint64_be"
        ),
        "derived_initialization_seeds": _derived_substream_seeds(seed),
        "caller_cpu_rng_preserved": True,
        "training_mode": training,
        "formal_validation": validation,
        "source_identity": source,
    }
    metadata = {
        "schema": SCHEMA,
        "status": STATUS,
        "public_model": "EviSIRST",
        "dataset": dataset,
        "architecture_seed": seed,
        "training_mode": training,
        "state_key_count": BASE_STATE_KEY_COUNT,
        "parameter_count": BASE_PARAMETER_COUNT,
        "selected_model_state_sha256": state_sha256,
        "manifest": manifest,
        "source_identity": source,
    }
    return model, metadata


__all__ = [
    "BASE_PARAMETER_COUNT",
    "BASE_STATE_KEY_COUNT",
    "CONFIRMATION_INIT_SEEDS",
    "EviSIRSTVariableInitError",
    "SCHEMA",
    "SEED_DOMAIN",
    "STATUS",
    "UINT32_MAX",
    "initialize_variable_evisirst",
    "require_uint32_seed",
    "source_identity",
]
