#!/usr/bin/env python3
"""Run the isolated IRSTD-1K HF-Decoder V1 fixed-Seed-42 revalidation.

This is intentionally a thin contract wrapper around the frozen
``train_irstd_hf_decoder_v1`` transaction engine.  It changes only the formal
runtime seed, output namespace, protocol/source identity, and artifact schemas.
The frozen runner, architecture, selector, and historical Seed-144 artifacts
are never modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import train_irstd_hf_decoder_v1 as frozen


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "irstd_performance" / "hf_decoder_seed42_v1"
)
PROTOCOL_PATH = (
    PROJECT_ROOT / "experiments" / "IRSTD_HF_DECODER_SEED42_V1_PROTOCOL.md"
)
FROZEN_HF_PROTOCOL_PATH = (
    PROJECT_ROOT / "experiments" / "IRSTD_HF_DECODER_V1_PROTOCOL.md"
)

DATASET = frozen.DATASET
TARGET_MODE = frozen.TARGET_MODE
ARCHITECTURE_SEED = 42
FORMAL_RUN_SEED = 42
PAIRED_RUN_SEED = FORMAL_RUN_SEED  # compatibility with the frozen adapter API
SPLIT_SEED = 20260811
FORMAL_EPOCHS = frozen.FORMAL_EPOCHS
FORMAL_BATCH_SIZE = frozen.FORMAL_BATCH_SIZE
FORMAL_WORKERS = frozen.FORMAL_WORKERS
FORMAL_BASE_LR = frozen.FORMAL_BASE_LR
FORMAL_MIN_LR = frozen.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = frozen.FORMAL_WARMUP_EPOCHS
FORMAL_VAL_INTERVAL = frozen.FORMAL_VAL_INTERVAL

TRAINING_SCHEMA = "evisirst_irstd_hf_decoder_seed42_training/v1"
CANDIDATE_SCHEMA = "evisirst_irstd_hf_decoder_seed42_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_irstd_hf_decoder_seed42_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_irstd_hf_decoder_seed42_history/v1"
SELECTION_PAYLOAD_SCHEMA = (
    "evisirst_irstd_hf_decoder_seed42_selection_payload/v1"
)
SOURCE_SET_SCHEMA = "evisirst_irstd_hf_decoder_seed42_source_set/v1"
DETERMINISM_SCHEMA = "evisirst_irstd_hf_decoder_seed42_determinism/v1"
EXPERIMENT_SCHEMA = "evisirst_irstd_hf_decoder_seed42_experiment/v1"
PROMOTION_GATE_SCHEMA = "evisirst_irstd_hf_decoder_seed42_interpretation/v1"

# These hashes are the source identities embedded in the completed historical
# Seed-144 transaction.  Formal Seed-42 execution fails closed if any frozen
# dependency no longer matches that audited transaction.
FROZEN_DEPENDENCY_SHA256 = {
    "frozen_hf_runner": (
        "4e0a576d604812bc9b3369bd6065459d4c0f814010a9157665f201400a173eb5"
    ),
    "frozen_hf_architecture": (
        "e669beaeae7606e5dae96e6137d9eee9eb8fd27459432c0fa6ea607816779502"
    ),
    "frozen_zero_margin_selector": (
        "776afca34eb5a83f514d145186fed8910548e902a9690620eefa495b94ec94d8"
    ),
    "frozen_hf_protocol": (
        "680b740db3e030c588599a02994ad12d1d3408e6e663751eb632d6df9a43205b"
    ),
}

Seed42HFDecoderRunnerError = frozen.HFDecoderRunnerError
zero_selection = frozen.zero_selection
hf_decoder = frozen.hf_decoder

_CONTRACT_LOCK = threading.Lock()
_FROZEN_PARSE_ARGS = frozen.parse_args
_FROZEN_RUN = frozen.run
_FROZEN_RESOLVE_RUN_PATHS = frozen.resolve_run_paths
_FROZEN_REQUIRE_ARGS = frozen._require_variant_args
_FROZEN_DETERMINISM_IDENTITY = frozen._variant_determinism_protocol_identity
_FROZEN_ATOMIC_DUAL_FINALS = frozen._atomic_write_dual_role_finals


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


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def interpretation_gate() -> dict[str, Any]:
    """Return the frozen interpretation contract for the Seed-42 result."""

    return {
        "schema": PROMOTION_GATE_SCHEMA,
        "status": "TBD_until_seed42_terminal_epoch",
        "purpose": "fixed_single_seed_formal_revalidation",
        "formal_seed_policy": {
            "architecture_seed": ARCHITECTURE_SEED,
            "runtime_seed": FORMAL_RUN_SEED,
            "split_seed": SPLIT_SEED,
            "seed42_result_is_authoritative_regardless_of_direction": True,
            "post_hoc_choose_better_seed": False,
        },
        "historical_pilot": {
            "runtime_seed": 1446202191,
            "summary_relative_path": (
                "runs/irstd_performance/hf_decoder_v1/formal/IRSTD-1K/"
                "binary/run_seed_1446202191/summary.json"
            ),
            "role": "disclosed_historical_reference_only",
            "eligible_for_seed_selection": False,
        },
        "selection": {
            "rule_version": zero_selection.RULE_VERSION,
            "selection_margin_raw": None,
            "selection_window_applied": False,
            "roles": list(zero_selection.VALID_ROLES),
        },
        "paired_seed42_controls": {
            "included_in_this_run": False,
            "causal_hf_gain_claim_authorized": False,
        },
        "completion": {
            "required_epoch": FORMAL_EPOCHS,
            "required_state_key_count": hf_decoder.FORMAL_STATE_KEY_COUNT,
            "required_final_roles": list(zero_selection.VALID_ROLES),
            "test_split_accessed": False,
            "public_test_allowed": False,
            "result": "TBD",
        },
    }


# The frozen adapter embeds a callable named ``promotion_gate`` in artifacts.
promotion_gate = interpretation_gate


def _seed42_source_provenance() -> dict[str, Any]:
    """Bind this wrapper/protocol and every frozen implementation dependency."""

    baseline_files = frozen._BASE_R1_SOURCE_PROVENANCE.get("files")
    if not isinstance(baseline_files, Mapping):
        raise Seed42HFDecoderRunnerError("R1 source provenance is malformed")
    files = {
        f"r1/{name}": copy.deepcopy(entry)
        for name, entry in sorted(baseline_files.items())
    }
    additions = {
        "seed42_wrapper": Path(__file__),
        "seed42_protocol": PROTOCOL_PATH,
        "frozen_hf_runner": Path(frozen.__file__),
        "frozen_hf_architecture": Path(hf_decoder.__file__),
        "frozen_zero_margin_selector": Path(zero_selection.__file__),
        "frozen_hf_protocol": FROZEN_HF_PROTOCOL_PATH,
    }
    for name, path in additions.items():
        if path.is_symlink() or not path.is_file():
            raise Seed42HFDecoderRunnerError(
                f"{name} is not a regular source file"
            )
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise Seed42HFDecoderRunnerError(
                f"{name} is outside the repository"
            ) from exc
        observed_sha256 = _sha256_file(resolved)
        expected_sha256 = FROZEN_DEPENDENCY_SHA256.get(name)
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise Seed42HFDecoderRunnerError(
                f"{name} differs from the completed Seed-144 source identity"
            )
        files[name] = {
            "relative_path": relative,
            "sha256": observed_sha256,
        }
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


def _seed42_determinism_protocol_identity() -> dict[str, Any]:
    identity = _FROZEN_DETERMINISM_IDENTITY()
    source = _seed42_source_provenance()
    identity["schema"] = DETERMINISM_SCHEMA
    identity["formal_seed_contract"] = {
        "architecture_seed": ARCHITECTURE_SEED,
        "runtime_seed": FORMAL_RUN_SEED,
        "split_seed": SPLIT_SEED,
        "single_seed_only": True,
    }
    identity["source_set_schema"] = source["schema"]
    identity["source_files"] = source["files"]
    identity["source_tree_sha256"] = source["source_tree_sha256"]
    return _json_clone(identity)


_PATCHES: dict[str, Any] = {
    "__doc__": __doc__,
    "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
    "PROTOCOL_PATH": PROTOCOL_PATH,
    "PAIRED_RUN_SEED": FORMAL_RUN_SEED,
    "TRAINING_SCHEMA": TRAINING_SCHEMA,
    "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
    "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
    "HISTORY_SCHEMA": HISTORY_SCHEMA,
    "SELECTION_PAYLOAD_SCHEMA": SELECTION_PAYLOAD_SCHEMA,
    "SOURCE_SET_SCHEMA": SOURCE_SET_SCHEMA,
    "DETERMINISM_SCHEMA": DETERMINISM_SCHEMA,
    "EXPERIMENT_SCHEMA": EXPERIMENT_SCHEMA,
    "PROMOTION_GATE_SCHEMA": PROMOTION_GATE_SCHEMA,
    "promotion_gate": interpretation_gate,
    "_variant_source_provenance": _seed42_source_provenance,
    "_variant_determinism_protocol_identity": _seed42_determinism_protocol_identity,
}


@contextmanager
def _seed42_contract() -> Iterator[None]:
    """Temporarily specialize the frozen adapter, restoring it exactly."""

    if not _CONTRACT_LOCK.acquire(blocking=False):
        raise Seed42HFDecoderRunnerError("Seed-42 wrapper is already active")
    previous = {
        name: (hasattr(frozen, name), getattr(frozen, name, None))
        for name in _PATCHES
    }
    try:
        for name, value in _PATCHES.items():
            setattr(frozen, name, value)
        yield
    finally:
        for name, (existed, value) in previous.items():
            if existed:
                setattr(frozen, name, value)
            elif hasattr(frozen, name):
                delattr(frozen, name)
        _CONTRACT_LOCK.release()


def _argv_with_fixed_seed(argv: Sequence[str] | None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    has_seed = any(
        token == "--run-seed" or token.startswith("--run-seed=")
        for token in raw
    )
    if not has_seed:
        raw.extend(("--run-seed", str(FORMAL_RUN_SEED)))
    return raw


def _require_seed42_args(args: Any) -> None:
    if getattr(args, "architecture_seed", None) != ARCHITECTURE_SEED:
        raise Seed42HFDecoderRunnerError("architecture seed must be 42")
    if getattr(args, "run_seed", None) != FORMAL_RUN_SEED:
        raise Seed42HFDecoderRunnerError("runtime seed must be 42")
    if Path(getattr(args, "output_root", "")).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise Seed42HFDecoderRunnerError(
            "Seed-42 output must use the isolated hf_decoder_seed42_v1 tree"
        )
    with _seed42_contract():
        _FROZEN_REQUIRE_ARGS(args)


def parse_args(argv: Sequence[str] | None = None) -> Any:
    """Parse the frozen CLI while forcing runtime Seed 42 by default."""

    with _seed42_contract():
        args = _FROZEN_PARSE_ARGS(_argv_with_fixed_seed(argv))
    _require_seed42_args(args)
    return args


def resolve_run_paths(args: Any) -> dict[str, Path | bool]:
    _require_seed42_args(args)
    with _seed42_contract():
        return dict(_FROZEN_RESOLVE_RUN_PATHS(args))


def source_provenance() -> dict[str, Any]:
    return _seed42_source_provenance()


def determinism_protocol_identity() -> dict[str, Any]:
    with _seed42_contract():
        return _seed42_determinism_protocol_identity()


def run(args: Any) -> Path:
    """Run/resume and crash-idempotently finalize the isolated transaction."""

    _require_seed42_args(args)
    with _seed42_contract():
        return _FROZEN_RUN(args)


def _atomic_write_dual_role_finals(**kwargs: Any) -> dict[str, Any]:
    """Testable Seed-42-schema delegate to the frozen idempotent finalizer."""

    with _seed42_contract():
        return _FROZEN_ATOMIC_DUAL_FINALS(**kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    checkpoint = run(parse_args(argv))
    print(checkpoint)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "DATASET",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAL_EPOCHS",
    "FORMAL_RUN_SEED",
    "FROZEN_DEPENDENCY_SHA256",
    "HISTORY_SCHEMA",
    "PROTOCOL_PATH",
    "SELECTION_PAYLOAD_SCHEMA",
    "SPLIT_SEED",
    "TARGET_MODE",
    "TRAINING_SCHEMA",
    "determinism_protocol_identity",
    "interpretation_gate",
    "parse_args",
    "resolve_run_paths",
    "run",
    "source_provenance",
]
