#!/usr/bin/env python3
"""Train SCTransNet or C3-SBSC V3.2 with fixed-seed validation selection.

Formal runs are deliberately narrow: architecture/runtime seed 42, 1000
epochs, and one immutable-validation evaluation after every epoch from 500
through 1000.  The public test split has no constructor or evaluation path in
this entry point.  Two physical final checkpoints are emitted, one for each
pre-registered zero-margin role: ``best_mIoU`` and ``best_Pd``.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any


_ENTRY_SOURCE = Path(__file__)
if _ENTRY_SOURCE.is_symlink() or not _ENTRY_SOURCE.is_file():
    raise RuntimeError("formal runner source must be a regular non-symlink file")
_ENTRY_SOURCE = _ENTRY_SOURCE.resolve(strict=True)
PROJECT_ROOT = _ENTRY_SOURCE.parent
if PROJECT_ROOT.is_symlink() or not PROJECT_ROOT.is_dir():
    raise RuntimeError("formal runner repository root must be a regular directory")
# Resolve the repository from this verified entry source, not from cwd or a
# caller-controlled PYTHONPATH.  Remove every textual/canonical alias of that
# root and install exactly one authoritative entry before any project import.
_PROJECT_ROOT_TEXT = str(PROJECT_ROOT)
_filtered_sys_path: list[str] = []
for _entry in sys.path:
    if not isinstance(_entry, str):
        _filtered_sys_path.append(_entry)
        continue
    try:
        _canonical_entry = Path(_entry or os.curdir).expanduser().resolve(
            strict=False
        )
    except (OSError, RuntimeError):
        _canonical_entry = None
    if _entry == _PROJECT_ROOT_TEXT or _canonical_entry == PROJECT_ROOT:
        continue
    _filtered_sys_path.append(_entry)
sys.path[:] = [_PROJECT_ROOT_TEXT, *_filtered_sys_path]

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import train_validation_selected as r1
from experiments import evisirst_v2_data as v2_data
from experiments import evisirst_zero_margin_selection as zero_selection
from experiments import sbsc_v32_selection as selection
from experiments.evisirst_v2_data import DEFAULT_SPLIT_ROOT, V2SplitContract


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "sctransnet_sbsc_v32_validation"
METHODS = ("sctransnet", "sbsc_v32")
ARCHITECTURE_SEED = 42
RUN_SEED = 42
SHUFFLE_STREAM = "sctransnet_sbsc_v32_pair"
FORMAL_EPOCHS = selection.TOTAL_EPOCHS
VALIDATION_BEGIN_EPOCH = selection.VALIDATION_BEGIN_EPOCH
FORMAL_BATCH_SIZE = 16
FORMAL_WORKERS = 0
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
TRAIN_PATCH_SIZE = (
    int(v2_data.source_protocol.PATCH_SIZE),
    int(v2_data.source_protocol.PATCH_SIZE),
)
VALIDATION_GEOMETRY_POLICY = (
    f"original_hw_pad_multiple_{int(v2_data.source_protocol.PAD_MULTIPLE)}"
)
GAIN_MIN = 0.0
GAIN_MAX = 0.25
GAIN_STATE_KEYS = (
    "mtc.encoder.layer.1.channel_attn.raw_dual_risk_level_gain",
)
ROUTER_STATE_SCHEMA = {
    "mtc.encoder.layer.1.channel_attn.tri_router.value_proj.weight": (
        (8, 480, 1, 1),
        torch.float32,
    ),
    "mtc.encoder.layer.1.channel_attn.tri_router.head.weight": (
        (3, 15, 3, 3),
        torch.float32,
    ),
}
ROUTER_PARAMETER_COUNT = 4_245
SEGMENTATION_LOSS_SCHEMA = "sum_of_six_BCELoss_mean_terms"
ROUTER_AUXILIARY_LOSS_SCHEMA = (
    "unit_weight_normalized_spatial_router_ce_from_train_batch_gt_"
    "and_detached_final_prediction"
)
TOTAL_LOSS_SCHEMA = (
    "sum_of_six_BCELoss_mean_terms_plus_unit_weight_normalized_spatial_router_ce"
)
CUDA_MEDIAN_ADAPTER_SCHEMA = (
    "sctransnet_sbsc_v32/cuda_strict_median_value_adapter/v1"
)
RUNTIME_INTEGRATION_SCHEMA = "sctransnet_sbsc_v32/runtime_integration/v1"
EXPECTED_RUNTIME_INTEGRATION = {
    "schema": RUNTIME_INTEGRATION_SCHEMA,
    "cuda_median_adapter": CUDA_MEDIAN_ADAPTER_SCHEMA,
}
EXPECTED_V31_SOLVER_SOURCE_SHA256 = (
    "b2d1e3f97607b305551a0602076605968041878eafd822c6f3335840ea725a3a"
)
EXPECTED_BASELINE_STATE_KEY_COUNT = 510
EXPECTED_BASELINE_PARAMETER_COUNT = 11_325_939
EXPECTED_SBSC_V32_STATE_KEY_COUNT = 513
EXPECTED_SBSC_V32_PARAMETER_COUNT = 11_330_188
EXPECTED_INACTIVE_PARAMETER_NAMES = frozenset(
    {
        f"mtc.embeddings_{index}.position_embeddings"
        for index in range(1, 5)
    }
    | {
        f"mtc.encoder.layer.{layer}.channel_attn.q{query}_attn{attention}"
        for layer in range(4)
        for query in range(1, 5)
        for attention in range(1, 5)
    }
)

TRAINING_SCHEMA = "sctransnet_sbsc_v32_validation_training/v1"
CANDIDATE_SCHEMA = "sctransnet_sbsc_v32_validation_candidate/v1"
CHECKPOINT_SCHEMA = "sctransnet_sbsc_v32_validation_checkpoint/v1"
HISTORY_SCHEMA = "sctransnet_sbsc_v32_validation_history/v1"
SOURCE_SET_SCHEMA = "sctransnet_sbsc_v32_validation_source_set/v1"
SMOKE_SELECTION_SCHEMA = (
    "sctransnet_sbsc_v32_runner_fixture_smoke_selection/v1"
)
_CANDIDATE_RE = re.compile(r"^epoch_([0-9]+)\.pth\.tar$")
_CANDIDATE_TEMP_RE = re.compile(
    r"^\.epoch_([0-9]{4})\.pth\.tar\.([a-z0-9_]{8})\.tmp$"
)
REQUIRED_REPORTED_METRICS = frozenset(
    {
        "validation_loss",
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pd",
        "tiny_pd",
        "fa",
        "false_objects_per_image",
    }
)


class SCTransNetSBSCV32RunnerError(r1.ValidationSelectedTrainingError):
    """The requested operation violates the frozen runner contract."""


def _strict_json_clone(value: Any, *, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SCTransNetSBSCV32RunnerError(
            f"{label} is not finite strict-JSON metadata"
        ) from exc


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


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, schemas, and raw values in stable key order."""

    if not isinstance(state, Mapping):
        raise SCTransNetSBSCV32RunnerError("state hash input must be a mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise SCTransNetSBSCV32RunnerError("state hash input is malformed")
        tensor = value.detach().cpu().contiguous()
        descriptor = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = (
            b""
            if tensor.numel() == 0
            else tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _selection_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    try:
        selected = {
            "method": identity["method"],
            "dataset": identity["dataset"],
            "architecture_seed": identity["architecture_seed"],
            "run_seed": identity["run_seed"],
            "split_manifest_sha256": identity["manifest_sha256"],
            "run_identity_sha256": identity["identity_sha256"],
        }
    except KeyError as exc:
        raise SCTransNetSBSCV32RunnerError(
            "run identity cannot bind selector records"
        ) from exc
    # The formal selector is the authoritative exact-schema validator.  Use an
    # empty pre-validation prefix so no metric record is required.
    try:
        payload = selection.select_prefix(
            [], completed_epoch=0, expected_identity=selected
        )
    except selection.SBSCV32SelectionError as exc:
        raise SCTransNetSBSCV32RunnerError(str(exc)) from exc
    return dict(payload["selection_identity"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--dataset", choices=v2_data.source_protocol.DATASETS, required=True
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-mode", choices=v2_data.TARGET_MODES, required=True)
    parser.add_argument(
        "--architecture-seed", type=int, choices=(ARCHITECTURE_SEED,),
        default=ARCHITECTURE_SEED,
    )
    parser.add_argument(
        "--run-seed", type=int, choices=(RUN_SEED,), default=RUN_SEED
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument(
        "--warmup-epochs", type=int, default=FORMAL_WARMUP_EPOCHS
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="print the read-only frozen run manifest; do not open data or GPU",
    )
    parser.add_argument("--allow-sample-level-fallback", action="store_true")
    # These short-run knobs exist only for runner integration tests.  They do
    # open validation and therefore can never be reported as authority Stage C,
    # whose separate contract is strictly train-only.
    parser.add_argument("--smoke-max-train-samples", type=int)
    parser.add_argument("--smoke-max-val-samples", type=int)
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.epochs > FORMAL_EPOCHS:
        parser.error(f"--epochs must be in [1, {FORMAL_EPOCHS}]")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers non-negative")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must be in [0, epochs]")
    if (
        not math.isfinite(args.base_lr)
        or not math.isfinite(args.min_lr)
        or not 0.0 < args.min_lr <= args.base_lr
    ):
        parser.error("learning rates must satisfy 0 < min-lr <= base-lr")
    for name in ("smoke_max_train_samples", "smoke_max_val_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    smoke = r1._is_smoke(args)
    if not smoke:
        if args.target_mode != "binary":
            parser.error("formal runs require --target-mode binary")
        frozen = {
            "epochs": FORMAL_EPOCHS,
            "batch_size": FORMAL_BATCH_SIZE,
            "workers": FORMAL_WORKERS,
            "base_lr": FORMAL_BASE_LR,
            "min_lr": FORMAL_MIN_LR,
            "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        }
        for name, expected in frozen.items():
            if getattr(args, name) != expected:
                parser.error(f"formal runs require --{name.replace('_', '-')} {expected}")
    # Compatibility for the audited V2 data and path helpers.  It is not an
    # exposed scheduling knob: formal validation is frozen to epochs 500:1000.
    args.val_interval = 1
    return args


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path | bool]:
    """Resolve an isolated, repository-bound path containing the method."""

    if args.method not in METHODS:
        raise SCTransNetSBSCV32RunnerError("unsupported method")
    proxy = copy.copy(args)
    # Both components are argparse choices, so this produces exactly two safe
    # path components while reusing the audited containment/symlink checks.
    proxy.dataset = f"{args.method}/{args.dataset}"
    paths = dict(r1.resolve_run_paths(proxy))
    run_dir = Path(paths["run_dir"])
    paths.pop("final", None)
    paths.update(
        {
            "best_mIoU_final": run_dir / "best_mIoU.pth.tar",
            "best_Pd_final": run_dir / "best_Pd.pth.tar",
            "summary": run_dir / "summary.json",
            "lock": run_dir / ".runner.lock",
        }
    )
    return paths


def preflight_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Return a read-only execution manifest without importing the core."""

    paths = resolve_run_paths(args)
    smoke = bool(paths["smoke"])
    return {
        "schema": TRAINING_SCHEMA + "/preflight",
        "method": args.method,
        "dataset": args.dataset,
        "target_mode": args.target_mode,
        "execution_mode": "resume" if args.resume else "fresh",
        "run_kind": "runner_fixture_smoke" if smoke else "formal",
        "stage_c_train_only_smoke": False,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "shuffle_stream": SHUFFLE_STREAM,
        "shuffle_stream_shared_across_methods": True,
        "epochs": args.epochs,
        "validation_begin_epoch": validation_begin_epoch(args.epochs, smoke=smoke),
        "validation_end_epoch": args.epochs,
        "validation_every_epoch_inclusive": True,
        "validation_record_count": len(
            expected_validation_epochs(
                args.epochs, total_epochs=args.epochs, smoke=smoke
            )
        ),
        "selection_roles": list(zero_selection.VALID_ROLES),
        "final_physical_weights": {
            role: str(paths[f"{role}_final"])
            for role in zero_selection.VALID_ROLES
        },
        "run_dir": str(paths["run_dir"]),
        "device_requested_but_not_opened": args.device,
        "dataset_root_declared_but_not_opened": str(args.dataset_root),
        "core_imported": False,
        "test_split_accessed": False,
        "writes_performed": False,
    }


def validation_begin_epoch(total_epochs: int, *, smoke: bool) -> int:
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
        raise SCTransNetSBSCV32RunnerError("total epochs must be an integer")
    if not 1 <= total_epochs <= FORMAL_EPOCHS:
        raise SCTransNetSBSCV32RunnerError("total epochs are outside the contract")
    return min(VALIDATION_BEGIN_EPOCH, total_epochs) if smoke else VALIDATION_BEGIN_EPOCH


def training_shuffle_seed(dataset: str, epoch: int) -> int:
    """Return the method-independent paired minibatch-order seed."""

    if dataset not in v2_data.source_protocol.DATASETS:
        raise SCTransNetSBSCV32RunnerError("shuffle dataset is unsupported")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= FORMAL_EPOCHS
    ):
        raise SCTransNetSBSCV32RunnerError("shuffle epoch is outside [1, 1000]")
    return r1.stable_uint63(
        RUN_SEED,
        SHUFFLE_STREAM,
        dataset,
        "shuffle",
        epoch,
    )


def training_shuffle_generator(args: argparse.Namespace, epoch: int) -> torch.Generator:
    """Build the exact shared CPU generator used by both paired methods."""

    if args.method not in METHODS:
        raise SCTransNetSBSCV32RunnerError("shuffle method is unsupported")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(training_shuffle_seed(args.dataset, epoch))
    return generator


def expected_validation_epochs(
    completed_epoch: int, *, total_epochs: int, smoke: bool
) -> tuple[int, ...]:
    if (
        isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or not 0 <= completed_epoch <= total_epochs
    ):
        raise SCTransNetSBSCV32RunnerError("completed epoch is outside the run")
    begin = validation_begin_epoch(total_epochs, smoke=smoke)
    if completed_epoch < begin:
        return ()
    return tuple(range(begin, completed_epoch + 1))


def _select_prefix(
    history: Sequence[Mapping[str, Any]],
    *,
    completed_epoch: int,
    total_epochs: int,
    smoke: bool,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    for position, record in enumerate(history):
        if not isinstance(record, Mapping) or not isinstance(
            record.get("metrics"), Mapping
        ):
            raise SCTransNetSBSCV32RunnerError(
                f"validation record[{position}] lacks a metrics mapping"
            )
        missing = REQUIRED_REPORTED_METRICS.difference(record["metrics"])
        if missing:
            raise SCTransNetSBSCV32RunnerError(
                f"validation record[{position}] omits reported metrics: "
                + ", ".join(sorted(missing))
            )
    expected = expected_validation_epochs(
        completed_epoch, total_epochs=total_epochs, smoke=smoke
    )
    observed = tuple(
        int(record.get("epoch", -1)) if isinstance(record, Mapping) else -1
        for record in history
    )
    if observed != expected:
        raise SCTransNetSBSCV32RunnerError(
            "validation history differs from the frozen cadence"
        )
    if not smoke:
        if total_epochs != FORMAL_EPOCHS:
            raise SCTransNetSBSCV32RunnerError("formal selector requires 1000 epochs")
        if expected_identity is None:
            raise SCTransNetSBSCV32RunnerError(
                "formal selector requires the frozen run identity"
            )
        try:
            return selection.select_prefix(
                history,
                completed_epoch=completed_epoch,
                expected_identity=expected_identity,
            )
        except selection.SBSCV32SelectionError as exc:
            raise SCTransNetSBSCV32RunnerError(str(exc)) from exc
    if not history:
        return {
            "schema": SMOKE_SELECTION_SCHEMA,
            "data_role": "val",
            "completed_epoch": completed_epoch,
            "roles": {},
            "retention_frontier_epochs": [],
            "test_split_accessed": False,
        }
    try:
        provenance = zero_selection.select_checkpoints(history)
    except zero_selection.EviSIRSTZeroMarginSelectionError as exc:
        raise SCTransNetSBSCV32RunnerError(str(exc)) from exc
    roles = {
        role: dict(provenance["roles"][role])
        for role in zero_selection.VALID_ROLES
    }
    state_hashes = {
        int(record["epoch"]): str(record["model_state_sha256"])
        for record in history
    }
    for role in roles:
        selected_epoch = int(roles[role]["selected"]["epoch"])
        roles[role] = dict(roles[role])
        roles[role]["selected"] = dict(roles[role]["selected"])
        roles[role]["selected"]["model_state_sha256"] = state_hashes[
            selected_epoch
        ]
    frontier = sorted(
        {int(roles[role]["selected"]["epoch"]) for role in roles}
    )
    return {
        "schema": SMOKE_SELECTION_SCHEMA,
        "data_role": "val",
        "completed_epoch": completed_epoch,
        "roles": roles,
        "retention_frontier_epochs": frontier,
        "ranking_provenance": provenance,
        "validation_history_sha256": _canonical_sha256(list(history)),
        "model_state_sha256_by_epoch": {
            str(epoch): state_hashes[epoch] for epoch in sorted(state_hashes)
        },
        "test_split_accessed": False,
    }


def _select_final(
    history: Sequence[Mapping[str, Any]],
    *,
    total_epochs: int,
    smoke: bool,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not smoke:
        if expected_identity is None:
            raise SCTransNetSBSCV32RunnerError(
                "formal selector requires the frozen run identity"
            )
        try:
            return selection.select_final(
                history, expected_identity=expected_identity
            )
        except selection.SBSCV32SelectionError as exc:
            raise SCTransNetSBSCV32RunnerError(str(exc)) from exc
    payload = _select_prefix(
        history,
        completed_epoch=total_epochs,
        total_epochs=total_epochs,
        smoke=True,
        expected_identity=None,
    )
    if set(payload["roles"]) != set(zero_selection.VALID_ROLES):
        raise SCTransNetSBSCV32RunnerError("smoke dual-role selection is incomplete")
    return payload


def _load_core_module() -> ModuleType:
    """Import the architecture lazily and fail closed on an incomplete API."""

    try:
        module = importlib.import_module("experiments.sctransnet_sbsc_v32")
    except (ImportError, ModuleNotFoundError) as exc:
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 core builder is unavailable; training was not started"
        ) from exc
    required = (
        "build_sctransnet_sbsc_v32_method",
        "validate_sctransnet_sbsc_v32",
        "validate_sbsc_v32_state_dict",
        "project_sbsc_v32_constraints_",
        "capture_c3_v32_training_router",
        "tri_router_supervision_loss",
        "structurally_inactive_parameter_names",
        "_estimate_c3_v31_support_v32",
        "validate_sbsc_v32_runtime_integration",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        raise SCTransNetSBSCV32RunnerError("SBSC-V3.2 core API is incomplete")
    source = Path(module.__file__ or "")
    if source.is_symlink() or not source.is_file():
        raise SCTransNetSBSCV32RunnerError("SBSC-V3.2 core source is not regular")
    expected_source = (
        PROJECT_ROOT / "experiments" / "sctransnet_sbsc_v32.py"
    ).resolve(strict=True)
    try:
        resolved_source = source.resolve(strict=True)
        resolved_source.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 core source is outside the repository"
        ) from exc
    if resolved_source != expected_source:
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 core source is not the canonical repository module"
        )
    if (
        type(getattr(module, "SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA", None))
        is not str
        or module.SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA
        != CUDA_MEDIAN_ADAPTER_SCHEMA
    ):
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 CUDA median adapter schema differs"
        )
    if (
        getattr(module, "EXPECTED_V31_SOLVER_SOURCE_SHA256", None)
        != EXPECTED_V31_SOLVER_SOURCE_SHA256
        or getattr(module, "V31_SOLVER_SOURCE_SHA256", None)
        != EXPECTED_V31_SOLVER_SOURCE_SHA256
    ):
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 frozen V3.1 source contract differs"
        )
    try:
        integration = module.validate_sbsc_v32_runtime_integration()
    except Exception as exc:
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 runtime integration validation failed"
        ) from exc
    if type(integration) is not dict or integration != EXPECTED_RUNTIME_INTEGRATION:
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 runtime integration contract differs"
        )
    return module


def _declared_state_key_count(metadata: Mapping[str, Any], method: str) -> int:
    names = (
        ("baseline_state_key_count", "selected_state_key_count", "state_key_count")
        if method == "sctransnet"
        else ("candidate_state_key_count", "selected_state_key_count", "state_key_count")
    )
    observed: list[int] = []
    for name in names:
        if name not in metadata:
            continue
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SCTransNetSBSCV32RunnerError(
                f"builder metadata {name} is malformed"
            )
        observed.append(value)
    if not observed or len(set(observed)) != 1:
        raise SCTransNetSBSCV32RunnerError(
            "builder metadata has no unambiguous selected state-key count"
        )
    return observed[0]


def _gain_contract(
    method: str,
    metadata: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    actual_gain_keys = tuple(sorted(key for key in expected_state if "gain" in key))
    if method == "sctransnet":
        if actual_gain_keys:
            raise SCTransNetSBSCV32RunnerError("baseline unexpectedly contains gains")
        return {"state_keys": (), "bounds": None}
    declared_keys = metadata.get("gain_state_keys")
    declared_bounds = metadata.get("gain_bounds")
    if (
        not isinstance(declared_keys, (list, tuple))
        or tuple(sorted(declared_keys)) != tuple(sorted(GAIN_STATE_KEYS))
        or actual_gain_keys != tuple(sorted(GAIN_STATE_KEYS))
    ):
        raise SCTransNetSBSCV32RunnerError("SBSC-V3.2 gain state-key contract differs")
    if (
        not isinstance(declared_bounds, (list, tuple))
        or len(declared_bounds) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in declared_bounds)
        or tuple(float(item) for item in declared_bounds) != (GAIN_MIN, GAIN_MAX)
    ):
        raise SCTransNetSBSCV32RunnerError("SBSC-V3.2 gain bounds differ from [0, 0.25]")
    for key in GAIN_STATE_KEYS:
        tensor = expected_state[key]
        if tuple(tensor.shape) != (4,) or tensor.dtype is not torch.float32:
            raise SCTransNetSBSCV32RunnerError(
                "SBSC-V3.2 gain must be one FP32 vector [4]"
            )
    return {"state_keys": list(GAIN_STATE_KEYS), "bounds": [GAIN_MIN, GAIN_MAX]}


def _router_contract(
    method: str,
    metadata: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    actual_keys = tuple(
        sorted(key for key in expected_state if ".tri_router." in key)
    )
    if method == "sctransnet":
        if actual_keys:
            raise SCTransNetSBSCV32RunnerError(
                "baseline unexpectedly contains tri-router state"
            )
        if metadata.get("router_state_keys") not in (None, [], ()):
            raise SCTransNetSBSCV32RunnerError(
                "baseline builder declares tri-router state"
            )
        if metadata.get("router_parameter_count") not in (None, 0):
            raise SCTransNetSBSCV32RunnerError(
                "baseline builder declares tri-router parameters"
            )
        return {
            "state_keys": [],
            "parameter_count": 0,
            "auxiliary_loss_schema": None,
            "auxiliary_loss_weight": 0.0,
        }

    expected_keys = tuple(sorted(ROUTER_STATE_SCHEMA))
    declared_keys = metadata.get("router_state_keys")
    if (
        not isinstance(declared_keys, (list, tuple))
        or tuple(sorted(declared_keys)) != expected_keys
        or actual_keys != expected_keys
    ):
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 tri-router state-key contract differs"
        )
    if metadata.get("router_parameter_count") != ROUTER_PARAMETER_COUNT:
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 tri-router parameter count differs"
        )
    if metadata.get("loss_schema") != TOTAL_LOSS_SCHEMA:
        raise SCTransNetSBSCV32RunnerError(
            "SBSC-V3.2 train-only auxiliary loss schema differs"
        )
    for key, (shape, dtype) in ROUTER_STATE_SCHEMA.items():
        tensor = expected_state[key]
        if tuple(tensor.shape) != shape or tensor.dtype is not dtype:
            raise SCTransNetSBSCV32RunnerError(
                f"SBSC-V3.2 tri-router tensor contract differs: {key}"
            )
    return {
        "state_keys": list(expected_keys),
        "parameter_count": ROUTER_PARAMETER_COUNT,
        "auxiliary_loss_schema": ROUTER_AUXILIARY_LOSS_SCHEMA,
        "auxiliary_loss_weight": 1.0,
    }


def _cuda_median_adapter_contract(
    method: str, metadata: Mapping[str, Any]
) -> str | None:
    declared = metadata.get("cuda_strict_deterministic_median_adapter")
    if method == "sbsc_v32":
        if declared != CUDA_MEDIAN_ADAPTER_SCHEMA:
            raise SCTransNetSBSCV32RunnerError(
                "SBSC-V3.2 builder CUDA median adapter schema differs"
            )
        return CUDA_MEDIAN_ADAPTER_SCHEMA
    if method == "sctransnet":
        if declared is not None:
            raise SCTransNetSBSCV32RunnerError(
                "baseline builder unexpectedly declares a CUDA median adapter"
            )
        return None
    raise SCTransNetSBSCV32RunnerError("unsupported method")


def _state_contract(
    model: nn.Module, metadata: Mapping[str, Any], method: str
) -> dict[str, Any]:
    state = model.state_dict()
    declared = _declared_state_key_count(metadata, method)
    if len(state) != declared:
        raise SCTransNetSBSCV32RunnerError(
            "live state-key count differs from builder metadata"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_key_count = (
        EXPECTED_BASELINE_STATE_KEY_COUNT
        if method == "sctransnet"
        else EXPECTED_SBSC_V32_STATE_KEY_COUNT
    )
    expected_parameter_count = (
        EXPECTED_BASELINE_PARAMETER_COUNT
        if method == "sctransnet"
        else EXPECTED_SBSC_V32_PARAMETER_COUNT
    )
    if len(state) != expected_key_count:
        raise SCTransNetSBSCV32RunnerError(
            f"{method} must have exactly {expected_key_count} state keys"
        )
    if parameter_count != expected_parameter_count:
        raise SCTransNetSBSCV32RunnerError(
            f"{method} must have exactly {expected_parameter_count} parameters"
        )
    declared_adapter = _cuda_median_adapter_contract(method, metadata)
    gain = _gain_contract(method, metadata, state)
    router = _router_contract(method, metadata, state)
    tensor_schema = {
        key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for key, tensor in sorted(state.items())
    }
    return {
        "state_key_count": len(state),
        "parameter_count": parameter_count,
        "ordered_state_keys_sha256": _canonical_sha256(list(state)),
        "tensor_schema_sha256": _canonical_sha256(tensor_schema),
        "cuda_strict_deterministic_median_adapter": declared_adapter,
        "gain": gain,
        "router": router,
    }


def _freeze_structurally_inactive(
    model: nn.Module, core: ModuleType
) -> tuple[str, ...]:
    raw_names = core.structurally_inactive_parameter_names(model)
    if isinstance(raw_names, (str, bytes, Mapping)):
        raise SCTransNetSBSCV32RunnerError("inactive parameter names are malformed")
    try:
        names = tuple(raw_names)
    except TypeError as exc:
        raise SCTransNetSBSCV32RunnerError(
            "inactive parameter names are malformed"
        ) from exc
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or set(names) != EXPECTED_INACTIVE_PARAMETER_NAMES
    ):
        raise SCTransNetSBSCV32RunnerError(
            "structurally inactive SCTransNet parameter contract differs"
        )
    named = dict(model.named_parameters())
    if not set(names).issubset(named):
        raise SCTransNetSBSCV32RunnerError("inactive parameters are absent from model")
    for name in names:
        named[name].requires_grad_(False)
    if any(named[name].requires_grad for name in names):
        raise SCTransNetSBSCV32RunnerError("inactive parameter freezing failed")
    return tuple(sorted(names))


def build_model(
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], ModuleType, dict[str, Any], tuple[str, ...]]:
    core = _load_core_module()
    r1.legacy_train.configure_determinism(ARCHITECTURE_SEED)
    try:
        result = core.build_sctransnet_sbsc_v32_method(
            method=args.method,
            dataset=args.dataset,
            architecture_seed=ARCHITECTURE_SEED,
            training=True,
        )
    except Exception as exc:
        raise SCTransNetSBSCV32RunnerError("SBSC-V3.2 method builder failed") from exc
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], nn.Module)
        or not isinstance(result[1], Mapping)
    ):
        raise SCTransNetSBSCV32RunnerError("SBSC-V3.2 method builder returned malformed data")
    model, raw_metadata = result
    metadata = _strict_json_clone(dict(raw_metadata), label="builder metadata")
    if metadata.get("method", args.method) != args.method:
        raise SCTransNetSBSCV32RunnerError("builder method identity differs")
    if args.method == "sbsc_v32":
        try:
            core.validate_sctransnet_sbsc_v32(model, require_zero_gain=True)
        except Exception as exc:
            raise SCTransNetSBSCV32RunnerError(
                "fresh SBSC-V3.2 model violates its zero-gain contract"
            ) from exc
    inactive = _freeze_structurally_inactive(model, core)
    state_contract = _state_contract(model, metadata, args.method)
    return model, metadata, core, state_contract, inactive


def _validate_serialized_gains_before_load(
    value: Any,
    *,
    method: str,
    gain_contract: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise SCTransNetSBSCV32RunnerError("checkpoint state is not a mapping")
    gain_keys = tuple(gain_contract.get("state_keys", ()))
    if method == "sctransnet":
        if any("gain" in str(key) for key in value):
            raise SCTransNetSBSCV32RunnerError("baseline checkpoint contains a gain")
        return
    if set(gain_keys) != set(GAIN_STATE_KEYS):
        raise SCTransNetSBSCV32RunnerError("trusted gain contract is malformed")
    for key in gain_keys:
        tensor = value.get(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != (4,)
            or tensor.dtype is not torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise SCTransNetSBSCV32RunnerError(
                f"serialized gain is malformed: {key}"
            )
        if bool(((tensor < GAIN_MIN) | (tensor > GAIN_MAX)).any()):
            raise SCTransNetSBSCV32RunnerError(
                f"serialized gain is outside [0, 0.25]: {key}"
            )


def _validate_state_dict(
    value: Any,
    expected: Mapping[str, torch.Tensor],
    *,
    method: str,
    gain_contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    # This call intentionally precedes every live ``load_state_dict``.
    _validate_serialized_gains_before_load(
        value, method=method, gain_contract=gain_contract
    )
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise SCTransNetSBSCV32RunnerError("checkpoint state keys differ")
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        reference = expected[key]
        if (
            not isinstance(key, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
            or tensor.layout != reference.layout
        ):
            raise SCTransNetSBSCV32RunnerError(
                f"checkpoint tensor contract differs for {key!r}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise SCTransNetSBSCV32RunnerError(
                f"checkpoint tensor is non-finite: {key!r}"
            )
        normalized[key] = tensor.detach().cpu().clone()
    return normalized


def _load_model_state(
    model: nn.Module,
    value: Any,
    *,
    method: str,
    gain_contract: Mapping[str, Any],
    core: ModuleType,
) -> dict[str, torch.Tensor]:
    # The core validates exact baseline/candidate key count and raw gain domain
    # before the live model's load_state_dict is reachable.
    try:
        core.validate_sbsc_v32_state_dict(value, method)
    except Exception as exc:
        raise SCTransNetSBSCV32RunnerError(
            "serialized core state is invalid before model load"
        ) from exc
    state = _validate_state_dict(
        value, model.state_dict(), method=method, gain_contract=gain_contract
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SCTransNetSBSCV32RunnerError("strict model state load failed")
    if method == "sbsc_v32":
        try:
            core.validate_sctransnet_sbsc_v32(model, require_zero_gain=False)
        except Exception as exc:
            raise SCTransNetSBSCV32RunnerError(
                "loaded SBSC-V3.2 model violates its architecture contract"
            ) from exc
    return state


def _cpu_state(
    model: nn.Module, *, method: str, gain_contract: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }
    return _validate_state_dict(
        state, model.state_dict(), method=method, gain_contract=gain_contract
    )


def _source_provenance(core: ModuleType) -> dict[str, Any]:
    sources = {
        "runner": Path(__file__),
        "experiments_package": PROJECT_ROOT / "experiments" / "__init__.py",
        "core_builder": Path(core.__file__ or ""),
        "frozen_v31_public_solver": (
            PROJECT_ROOT / "experiments" / "sctransnet_sbsc_v31.py"
        ),
        "paired_initialization_authority": (
            PROJECT_ROOT / "experiments" / "four_dataset_models_seed42_v1.py"
        ),
        "v2_data": Path(v2_data.__file__),
        "v2_split_validator": (
            PROJECT_ROOT / "experiments" / "evisirst_v2_splits.py"
        ),
        "three_dataset_source_protocol": (
            PROJECT_ROOT / "experiments" / "three_dataset_v2_protocol.py"
        ),
        "legacy_data_dependency": (
            PROJECT_ROOT / "experiments" / "evisirst_data.py"
        ),
        "r1_imported_validation_selector": (
            PROJECT_ROOT / "experiments" / "evisirst_v2_selection.py"
        ),
        "formal_selector": Path(selection.__file__),
        "zero_margin_ranker": Path(zero_selection.__file__),
        "transaction_helpers": Path(r1.__file__),
        "training_math_rng_helpers": PROJECT_ROOT / "train.py",
        "model_package": PROJECT_ROOT / "model" / "__init__.py",
        "r1_imported_model_entry": PROJECT_ROOT / "model" / "EviSIRST.py",
        "sctransnet_config": PROJECT_ROOT / "model" / "_internal" / "Config.py",
        "sctransnet": PROJECT_ROOT / "model" / "_internal" / "SCTransNet.py",
    }
    # R1 imports the EviSIRST model entry at module load.  Bind the full local
    # model source closure so a resume cannot silently cross a changed internal
    # implementation even when the changed file is reached indirectly.
    internal_root = PROJECT_ROOT / "model" / "_internal"
    internal_paths = sorted(internal_root.glob("*.py"))
    if not internal_paths:
        raise SCTransNetSBSCV32RunnerError("model internal source tree is missing")
    for path in internal_paths:
        sources.setdefault(f"model_internal/{path.name}", path)
    files: dict[str, dict[str, str]] = {}
    root = PROJECT_ROOT.resolve(strict=True)
    for name, raw_path in sources.items():
        if raw_path.is_symlink() or not raw_path.is_file():
            raise SCTransNetSBSCV32RunnerError(
                f"protocol source is not a regular file: {name}"
            )
        path = raw_path.resolve(strict=True)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SCTransNetSBSCV32RunnerError(
                f"protocol source escapes repository: {name}"
            ) from exc
        files[name] = {"relative_path": relative, "sha256": _sha256_file(path)}
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


def _run_identity(
    args: argparse.Namespace,
    contract: V2SplitContract,
    grouping: Mapping[str, Any],
    *,
    train_count: int,
    val_count: int,
    smoke: bool,
    builder_metadata: Mapping[str, Any],
    state_contract: Mapping[str, Any],
    inactive_parameter_names: Sequence[str],
    core: ModuleType,
) -> dict[str, Any]:
    source = _source_provenance(core)
    identity: dict[str, Any] = {
        "schema": TRAINING_SCHEMA + "/run_identity",
        "model": (
            "SCTransNet"
            if args.method == "sctransnet"
            else "SCTransNet-C3-SBSC-V3.2"
        ),
        "method": args.method,
        "dataset": args.dataset,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "single_seed_only": True,
        "shuffle_stream": SHUFFLE_STREAM,
        "shuffle_stream_shared_across_methods": True,
        "shuffle_seed_namespace": [
            RUN_SEED,
            SHUFFLE_STREAM,
            args.dataset,
            "shuffle",
            "epoch",
        ],
        "target_mode": args.target_mode,
        "train_patch_size": list(TRAIN_PATCH_SIZE),
        "validation_geometry_policy": VALIDATION_GEOMETRY_POLICY,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "optimizer": "torch.optim.Adam/defaults",
        "loss": (
            SEGMENTATION_LOSS_SCHEMA
            if args.method == "sctransnet"
            else TOTAL_LOSS_SCHEMA
        ),
        "loss_schema": {
            "segmentation": SEGMENTATION_LOSS_SCHEMA,
            "router_auxiliary": (
                None
                if args.method == "sctransnet"
                else ROUTER_AUXILIARY_LOSS_SCHEMA
            ),
            "router_auxiliary_weight": (
                0.0 if args.method == "sctransnet" else 1.0
            ),
            "router_teacher_prediction_detached": (
                args.method == "sbsc_v32"
            ),
            "router_target_source": (
                None
                if args.method == "sctransnet"
                else "current_train_batch_ground_truth_only"
            ),
            "total": (
                SEGMENTATION_LOSS_SCHEMA
                if args.method == "sctransnet"
                else TOTAL_LOSS_SCHEMA
            ),
        },
        "evaluation_head": "out",
        "validation_begin_epoch": validation_begin_epoch(args.epochs, smoke=smoke),
        "validation_end_epoch": args.epochs,
        "validation_every_epoch_inclusive": True,
        "formal_schedule_schema": selection.SCHEDULE_SCHEMA,
        "selection_rule": zero_selection.RULE_VERSION,
        "selection_roles": list(zero_selection.VALID_ROLES),
        "selection_margin_raw": None,
        "builder_metadata": _strict_json_clone(
            dict(builder_metadata), label="builder metadata"
        ),
        "state_contract": _strict_json_clone(
            dict(state_contract), label="state contract"
        ),
        "structurally_inactive_parameter_names": list(inactive_parameter_names),
        "manifest_sha256": contract.manifest_sha256,
        "split_seed": contract.manifest["seeds"]["split_seed"],
        "data_tree_sha256": contract.data_tree_sha256,
        "data_tree_verified": contract.data_tree_verified,
        "grouping_policy": _strict_json_clone(dict(grouping), label="grouping"),
        "train_count": train_count,
        "val_count": val_count,
        "normalization_mode": "legacy",
        "smoke": smoke,
        "smoke_semantics": (
            "runner_fixture_with_validation_not_stage_c" if smoke else None
        ),
        "smoke_max_train_samples": args.smoke_max_train_samples,
        "smoke_max_val_samples": args.smoke_max_val_samples,
        "source_provenance": source,
        "test_split_accessed": False,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def build_validation_record(
    epoch: int,
    metrics: Mapping[str, Any],
    *,
    selection_identity: Mapping[str, Any],
    model_state_sha256: str,
) -> dict[str, Any]:
    cloned = _strict_json_clone(dict(metrics), label="validation metrics")
    missing_metrics = REQUIRED_REPORTED_METRICS.difference(cloned)
    if missing_metrics:
        raise SCTransNetSBSCV32RunnerError(
            "validation metrics omit reported fields: "
            + ", ".join(sorted(missing_metrics))
        )
    # A validation set with no <=9-pixel target has an undefined tiny-Pd.
    # Encode the neutral deterministic tie value explicitly rather than
    # allowing None to make checkpoint ranking unexecutable.
    tiny_defined = cloned.get("tiny_pd") is not None
    if not tiny_defined:
        cloned["tiny_pd"] = 0.0
    try:
        bound_identity = _strict_json_clone(
            dict(selection_identity), label="selection identity"
        )
        if set(bound_identity) != {
            "method",
            "dataset",
            "architecture_seed",
            "run_seed",
            "split_manifest_sha256",
            "run_identity_sha256",
        }:
            raise SCTransNetSBSCV32RunnerError("selection identity fields differ")
        if (
            not isinstance(model_state_sha256, str)
            or len(model_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in model_state_sha256)
        ):
            raise SCTransNetSBSCV32RunnerError("model state SHA-256 is malformed")
        record = {
            "epoch": int(epoch),
            "data_role": "val",
            "mIoU": float(cloned["miou"]),
            "nIoU": float(cloned["niou"]),
            "Pd": float(cloned["pd"]),
            "Fa": float(cloned["fa"]),
            "F1": float(cloned["pixel_f1"]),
            "Precision": float(cloned["pixel_precision"]),
            "Recall": float(cloned["pixel_recall"]),
            "tinyPd": float(cloned["tiny_pd"]),
            "false_objects_per_image": float(
                cloned["false_objects_per_image"]
            ),
            "loss": float(cloned["validation_loss"]),
            "evaluation_head": "out",
            "tiny_pd_defined": tiny_defined,
            "metrics": cloned,
            **bound_identity,
            "model_state_sha256": model_state_sha256,
            "test_split_accessed": False,
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SCTransNetSBSCV32RunnerError(
            "validation metrics are incomplete or malformed"
        ) from exc
    try:
        zero_selection.select_checkpoints([record])
    except zero_selection.EviSIRSTZeroMarginSelectionError as exc:
        raise SCTransNetSBSCV32RunnerError(str(exc)) from exc
    return record


def _candidate_path(candidate_dir: Path, epoch: int) -> Path:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise SCTransNetSBSCV32RunnerError("candidate epoch must be positive")
    return candidate_dir / f"epoch_{epoch:04d}.pth.tar"


def _regular_file(path: Path, *, run_dir: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SCTransNetSBSCV32RunnerError(f"{label} is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise SCTransNetSBSCV32RunnerError(
            f"{label} escapes the run directory"
        ) from exc
    return resolved


def _candidate_epoch(path: Path, candidate_dir: Path) -> int:
    match = _CANDIDATE_RE.fullmatch(path.name)
    if match is None:
        raise SCTransNetSBSCV32RunnerError(
            "candidate directory contains an unexpected filename"
        )
    epoch = int(match.group(1))
    if path.name != _candidate_path(candidate_dir, epoch).name:
        raise SCTransNetSBSCV32RunnerError("candidate filename is not canonical")
    return epoch


def _is_recoverable_candidate_temp(
    path: Path,
    candidate_dir: Path,
    *,
    completed_epoch: int,
    total_epochs: int,
    smoke: bool,
) -> bool:
    """Recognize only the next candidate's interrupted mkstemp artifact.

    ``r1._atomic_torch_save`` creates candidate temporaries with
    ``prefix=f".{candidate.name}."`` and ``suffix=".tmp"``.  A recoverable
    temporary must also belong to the single validation epoch immediately
    after the committed latest state.  File type, symlink, and containment
    checks remain the caller's responsibility before unlinking.
    """

    match = _CANDIDATE_TEMP_RE.fullmatch(path.name)
    if match is None or completed_epoch >= total_epochs:
        return False
    epoch = int(match.group(1))
    token = match.group(2)
    if path.name != f".{_candidate_path(candidate_dir, epoch).name}.{token}.tmp":
        return False
    next_epoch = completed_epoch + 1
    return epoch == next_epoch and epoch in expected_validation_epochs(
        next_epoch,
        total_epochs=total_epochs,
        smoke=smoke,
    )


def _save_candidate(
    *,
    path: Path,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    identity: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"candidate already exists: {path}")
    r1._atomic_torch_save(
        path,
        {
            "schema": CANDIDATE_SCHEMA,
            "model": identity["model"],
            "method": identity["method"],
            "dataset": identity["dataset"],
            "epoch": epoch,
            "run_identity": dict(identity),
            "validation_record": dict(record),
            "state_dict": dict(state),
            "test_split_accessed": False,
        },
    )


def _validate_candidate_payload(
    *,
    path: Path,
    run_dir: Path,
    candidate_dir: Path,
    identity: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    gain_contract: Mapping[str, Any],
    expected_record: Mapping[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    path = _regular_file(path, run_dir=run_dir, label="candidate")
    epoch = _candidate_epoch(path, candidate_dir)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SCTransNetSBSCV32RunnerError("candidate could not be read safely") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CANDIDATE_SCHEMA
        or payload.get("method") != identity["method"]
        or payload.get("dataset") != identity["dataset"]
        or payload.get("run_identity") != identity
        or payload.get("epoch") != epoch
        or payload.get("test_split_accessed") is not False
        or not isinstance(payload.get("validation_record"), Mapping)
    ):
        raise SCTransNetSBSCV32RunnerError("candidate identity differs")
    record = dict(payload["validation_record"])
    if record.get("epoch") != epoch:
        raise SCTransNetSBSCV32RunnerError("candidate validation epoch differs")
    try:
        zero_selection.select_checkpoints([record])
    except zero_selection.EviSIRSTZeroMarginSelectionError as exc:
        raise SCTransNetSBSCV32RunnerError(str(exc)) from exc
    if expected_record is not None and record != expected_record:
        raise SCTransNetSBSCV32RunnerError("candidate validation record differs")
    validated_state = _validate_state_dict(
        payload.get("state_dict"),
        expected_state,
        method=str(identity["method"]),
        gain_contract=gain_contract,
    )
    if record.get("model_state_sha256") != _state_dict_sha256(validated_state):
        raise SCTransNetSBSCV32RunnerError(
            "candidate state hash differs from its validation record"
        )
    expected_selection_identity = _selection_identity(identity)
    if any(record.get(key) != value for key, value in expected_selection_identity.items()):
        raise SCTransNetSBSCV32RunnerError(
            "candidate record selection identity differs"
        )
    return epoch, record


def _frontier_artifacts(
    *,
    history: Sequence[Mapping[str, Any]],
    completed_epoch: int,
    total_epochs: int,
    smoke: bool,
    candidate_dir: Path,
    identity: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    selected = _select_prefix(
        history,
        completed_epoch=completed_epoch,
        total_epochs=total_epochs,
        smoke=smoke,
        expected_identity=_selection_identity(identity) if not smoke else None,
    )
    frontier = tuple(int(epoch) for epoch in selected["retention_frontier_epochs"])
    if not frontier:
        return {}
    candidate_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[int, dict[str, Any]] = {}
    for epoch in frontier:
        path = _candidate_path(candidate_dir, epoch)
        if path.is_symlink() or not path.is_file():
            raise SCTransNetSBSCV32RunnerError("frontier candidate is missing")
        artifacts[epoch] = {
            "relative_path": f"candidates/{path.name}",
            "file_sha256": _sha256_file(path),
        }
    return artifacts


def _history_payload(
    *,
    identity: Mapping[str, Any],
    training_history: Sequence[Mapping[str, Any]],
    validation_history: Sequence[Mapping[str, Any]],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": HISTORY_SCHEMA,
        "method": identity["method"],
        "data_role": "val",
        "run_identity": dict(identity),
        "training_history": [dict(record) for record in training_history],
        "validation_history": [dict(record) for record in validation_history],
        "retention_frontier_epochs": sorted(candidate_artifacts),
        "candidate_artifacts": {
            str(epoch): dict(candidate_artifacts[epoch])
            for epoch in sorted(candidate_artifacts)
        },
        "test_split_accessed": False,
    }


def _validate_histories(
    *,
    completed_epoch: int,
    total_epochs: int,
    smoke: bool,
    training_history: Any,
    validation_history: Any,
    identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(training_history, list) or len(training_history) != completed_epoch:
        raise SCTransNetSBSCV32RunnerError("resume training history is incomplete")
    train: list[dict[str, Any]] = []
    for expected_epoch, raw in enumerate(training_history, start=1):
        if not isinstance(raw, Mapping) or raw.get("epoch") != expected_epoch:
            raise SCTransNetSBSCV32RunnerError("resume training history order differs")
        record = _strict_json_clone(dict(raw), label="training history")
        for field in (
            "mean_train_loss",
            "mean_segmentation_loss",
            "mean_router_loss",
            "router_auxiliary_weight",
            "learning_rate",
        ):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SCTransNetSBSCV32RunnerError("training history is non-finite")
        if any(
            float(record[field]) < 0.0
            for field in (
                "mean_train_loss",
                "mean_segmentation_loss",
                "mean_router_loss",
            )
        ):
            raise SCTransNetSBSCV32RunnerError("training loss must be non-negative")
        if not math.isclose(
            float(record["mean_train_loss"]),
            float(record["mean_segmentation_loss"])
            + float(record["mean_router_loss"]),
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise SCTransNetSBSCV32RunnerError(
                "training loss decomposition differs"
            )
        expected_router_weight = (
            1.0 if identity.get("method") == "sbsc_v32" else 0.0
        )
        if float(record["router_auxiliary_weight"]) != expected_router_weight:
            raise SCTransNetSBSCV32RunnerError(
                "training history router auxiliary weight differs"
            )
        expected_lr = r1.legacy_train.learning_rate_for_epoch(
            expected_epoch,
            total_epochs,
            float(identity["base_lr"]),
            float(identity["min_lr"]),
            int(identity["warmup_epochs"]),
        )
        if float(record["learning_rate"]) != expected_lr:
            raise SCTransNetSBSCV32RunnerError(
                "training history learning-rate schedule differs"
            )
        if record.get("processed_samples") != identity["train_count"]:
            raise SCTransNetSBSCV32RunnerError(
                "training history processed-sample count differs"
            )
        train.append(record)
    if not isinstance(validation_history, list):
        raise SCTransNetSBSCV32RunnerError("resume validation history is malformed")
    val = [_strict_json_clone(record, label="validation history") for record in validation_history]
    if any(not isinstance(record, dict) for record in val):
        raise SCTransNetSBSCV32RunnerError("resume validation history is malformed")
    _select_prefix(
        val,
        completed_epoch=completed_epoch,
        total_epochs=total_epochs,
        smoke=smoke,
        expected_identity=_selection_identity(identity) if not smoke else None,
    )
    return train, val


def _normalize_artifact_map(value: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise SCTransNetSBSCV32RunnerError("candidate artifact map is malformed")
    normalized: dict[int, dict[str, Any]] = {}
    for raw_epoch, raw_artifact in value.items():
        if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int):
            raise SCTransNetSBSCV32RunnerError("candidate artifact epoch is malformed")
        if not isinstance(raw_artifact, Mapping):
            raise SCTransNetSBSCV32RunnerError("candidate artifact is malformed")
        normalized[raw_epoch] = dict(raw_artifact)
    return normalized


def _validate_resume_candidates(
    *,
    run_dir: Path,
    candidate_dir: Path,
    identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    artifacts: Any,
    expected_state: Mapping[str, torch.Tensor],
    gain_contract: Mapping[str, Any],
    completed_epoch: int,
    total_epochs: int,
    smoke: bool,
) -> dict[int, dict[str, Any]]:
    selected = _select_prefix(
        history,
        completed_epoch=completed_epoch,
        total_epochs=total_epochs,
        smoke=smoke,
        expected_identity=_selection_identity(identity) if not smoke else None,
    )
    frontier = tuple(int(epoch) for epoch in selected["retention_frontier_epochs"])
    normalized = _normalize_artifact_map(artifacts)
    if tuple(sorted(normalized)) != frontier:
        raise SCTransNetSBSCV32RunnerError("resume artifacts differ from dual frontier")

    expected_files: set[Path] = set()
    by_epoch = {int(record["epoch"]): record for record in history}
    for epoch in frontier:
        path = _candidate_path(candidate_dir, epoch)
        artifact = normalized[epoch]
        if artifact.get("relative_path") != f"candidates/{path.name}":
            raise SCTransNetSBSCV32RunnerError("candidate relative path differs")
        path = _regular_file(path, run_dir=run_dir, label="frontier candidate")
        expected_files.add(path)
        if artifact.get("file_sha256") != _sha256_file(path):
            raise SCTransNetSBSCV32RunnerError("candidate hash differs")
        _validate_candidate_payload(
            path=path,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            identity=identity,
            expected_state=expected_state,
            gain_contract=gain_contract,
            expected_record=by_epoch[epoch],
        )

    actual_files: set[Path] = set()
    recoverable_temporaries: set[Path] = set()
    if candidate_dir.exists():
        if candidate_dir.is_symlink() or not candidate_dir.is_dir():
            raise SCTransNetSBSCV32RunnerError("candidate directory is not regular")
        for raw_path in candidate_dir.iterdir():
            checked = _regular_file(
                raw_path, run_dir=run_dir, label="candidate entry"
            )
            if _is_recoverable_candidate_temp(
                raw_path,
                candidate_dir,
                completed_epoch=completed_epoch,
                total_epochs=total_epochs,
                smoke=smoke,
            ):
                recoverable_temporaries.add(checked)
            else:
                actual_files.add(checked)
    if expected_files - actual_files:
        raise SCTransNetSBSCV32RunnerError("frontier candidate is missing")

    # Latest is the commit point.  Extras are removable only after their full
    # identity/state validation: either committed dominated candidates, or the
    # single next validation candidate written just before a crash.
    for path in sorted(actual_files - expected_files):
        epoch = _candidate_epoch(path, candidate_dir)
        expected_record = by_epoch.get(epoch)
        if epoch <= completed_epoch and expected_record is None:
            raise SCTransNetSBSCV32RunnerError(
                "extra candidate has no committed validation record"
            )
        if epoch > completed_epoch:
            next_epochs = expected_validation_epochs(
                completed_epoch + 1,
                total_epochs=total_epochs,
                smoke=smoke,
            )
            if epoch != completed_epoch + 1 or epoch not in next_epochs:
                raise SCTransNetSBSCV32RunnerError(
                    "extra candidate is not the next crash-recovery epoch"
                )
        _validate_candidate_payload(
            path=path,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            identity=identity,
            expected_state=expected_state,
            gain_contract=gain_contract,
            expected_record=expected_record,
        )
        path.unlink()
    # The caller reaches this routine only while holding the per-run lock.
    # Delay deletion until every other entry has passed the existing strict
    # validation so a hostile near-match cannot trigger partial cleanup.
    for path in sorted(recoverable_temporaries):
        path.unlink()
    return normalized


def _prune_after_committed_latest(
    *,
    candidate_dir: Path,
    validation_history: Sequence[Mapping[str, Any]],
    candidate_artifacts: Mapping[int, Mapping[str, Any]],
) -> None:
    """Delete only committed dominated candidates in the live locked run."""

    if not candidate_dir.exists():
        return
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise SCTransNetSBSCV32RunnerError("candidate directory is not regular")
    evaluated = {int(record["epoch"]) for record in validation_history}
    frontier = set(candidate_artifacts)
    for path in candidate_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise SCTransNetSBSCV32RunnerError("candidate entry is not regular")
        epoch = _candidate_epoch(path, candidate_dir)
        if epoch in frontier:
            continue
        if epoch not in evaluated:
            raise SCTransNetSBSCV32RunnerError(
                "live candidate has no committed validation record"
            )
        path.unlink()


def _load_resume_state(
    *,
    path: Path,
    run_dir: Path,
    candidate_dir: Path,
    identity: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    core: ModuleType,
    gain_contract: Mapping[str, Any],
    total_epochs: int,
    smoke: bool,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    path = _regular_file(path, run_dir=run_dir, label="resume state")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SCTransNetSBSCV32RunnerError("resume state could not be read safely") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TRAINING_SCHEMA
        or payload.get("run_identity") != identity
        or payload.get("method") != identity["method"]
        or payload.get("test_split_accessed") is not False
    ):
        raise SCTransNetSBSCV32RunnerError("resume run identity differs")
    completed = payload.get("epoch")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 1 <= completed <= total_epochs
    ):
        raise SCTransNetSBSCV32RunnerError("resume epoch is outside the run")
    _load_model_state(
        model,
        payload.get("state_dict"),
        method=str(identity["method"]),
        gain_contract=gain_contract,
        core=core,
    )
    try:
        r1._validate_and_load_adam_optimizer_state(
            optimizer_state=payload.get("optimizer"),
            model=model,
            optimizer=optimizer,
            identity=identity,
            completed_epoch=completed,
            total_epochs=total_epochs,
        )
    except r1.ValidationSelectedTrainingError as exc:
        raise SCTransNetSBSCV32RunnerError(str(exc)) from exc
    training_history, validation_history = _validate_histories(
        completed_epoch=completed,
        total_epochs=total_epochs,
        smoke=smoke,
        training_history=payload.get("training_history"),
        validation_history=payload.get("validation_history"),
        identity=identity,
    )
    if validation_history:
        committed_state_hash = _state_dict_sha256(model.state_dict())
        if validation_history[-1].get("model_state_sha256") != committed_state_hash:
            raise SCTransNetSBSCV32RunnerError(
                "latest model state differs from its committed validation record"
            )
    candidate_artifacts = _validate_resume_candidates(
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        identity=identity,
        history=validation_history,
        artifacts=payload.get("candidate_artifacts"),
        expected_state=model.state_dict(),
        gain_contract=gain_contract,
        completed_epoch=completed,
        total_epochs=total_epochs,
        smoke=smoke,
    )
    try:
        r1.legacy_train._restore_rng_state(payload.get("rng"), device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SCTransNetSBSCV32RunnerError("resume RNG state is malformed") from exc
    return completed + 1, training_history, validation_history, candidate_artifacts


def _load_candidate_state(
    *,
    epoch: int,
    run_dir: Path,
    candidate_dir: Path,
    identity: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    artifacts: Mapping[int, Mapping[str, Any]],
    expected_state: Mapping[str, torch.Tensor],
    gain_contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if epoch not in artifacts:
        raise SCTransNetSBSCV32RunnerError("selected epoch is absent from frontier")
    path = _regular_file(
        _candidate_path(candidate_dir, epoch),
        run_dir=run_dir,
        label="selected candidate",
    )
    if artifacts[epoch].get("file_sha256") != _sha256_file(path):
        raise SCTransNetSBSCV32RunnerError("selected candidate hash differs")
    record = next(
        (record for record in history if int(record["epoch"]) == epoch), None
    )
    if record is None:
        raise SCTransNetSBSCV32RunnerError("selected validation record is missing")
    _validate_candidate_payload(
        path=path,
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        identity=identity,
        expected_state=expected_state,
        gain_contract=gain_contract,
        expected_record=record,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _validate_state_dict(
        payload["state_dict"],
        expected_state,
        method=str(identity["method"]),
        gain_contract=gain_contract,
    )


def _final_checkpoint_payload(
    *,
    role: str,
    state: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
    normalization: Mapping[str, Any],
    normalization_provenance: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
    selected_validation_record: Mapping[str, Any],
    builder_metadata: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if role not in zero_selection.VALID_ROLES:
        raise SCTransNetSBSCV32RunnerError("invalid final checkpoint role")
    selected = selection_payload["roles"][role]["selected"]
    record = _strict_json_clone(
        dict(selected_validation_record), label="selected validation record"
    )
    if (
        record.get("epoch") != int(selected["epoch"])
        or record.get("data_role") != "val"
        or record.get("test_split_accessed") is not False
        or record.get("model_state_sha256") != _state_dict_sha256(state)
        or selected.get("model_state_sha256") != record.get("model_state_sha256")
        or not isinstance(record.get("metrics"), dict)
        or not REQUIRED_REPORTED_METRICS.issubset(record["metrics"])
    ):
        raise SCTransNetSBSCV32RunnerError(
            "selected validation record differs from role/state or lacks metrics"
        )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": identity["model"],
        "method": args.method,
        "dataset": args.dataset,
        "checkpoint_role": role,
        "selection_role": role,
        "epoch": int(selected["epoch"]),
        "seed": ARCHITECTURE_SEED,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "single_seed_only": True,
        "state_dict": dict(state),
        "target_mode": args.target_mode,
        "normalization": dict(normalization),
        "normalization_provenance": dict(normalization_provenance),
        "training": dict(identity),
        "training_identity_sha256": identity["identity_sha256"],
        "split_provenance": dict(split_provenance),
        "selection": dict(selection_payload),
        "selected": dict(selected),
        "selected_validation_record": record,
        "selected_metrics": dict(record["metrics"]),
        "selection_margin_raw": None,
        "selection_window_applied": False,
        "builder_metadata": dict(builder_metadata),
        "source_selection": "immutable_v2_validation_split",
        "selection_is_optimistic": False,
        "optimistic": False,
        "smoke": smoke,
        "test_split_accessed": False,
    }


def _require_selected_state_hash(
    selection_payload: Mapping[str, Any],
    role: str,
    state: Mapping[str, torch.Tensor],
) -> str:
    if role not in zero_selection.VALID_ROLES:
        raise SCTransNetSBSCV32RunnerError("invalid selected-state role")
    try:
        declared = selection_payload["roles"][role]["selected"][
            "model_state_sha256"
        ]
    except (KeyError, TypeError) as exc:
        raise SCTransNetSBSCV32RunnerError(
            f"{role} selection lacks a state hash"
        ) from exc
    observed = _state_dict_sha256(state)
    if declared != observed:
        raise SCTransNetSBSCV32RunnerError(
            f"{role} selected state hash differs from physical candidate"
        )
    return observed


def _atomic_write_final_pair(
    *,
    run_dir: Path,
    paths: Mapping[str, Path | bool],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_roles = set(zero_selection.VALID_ROLES)
    if set(payloads) != expected_roles:
        raise SCTransNetSBSCV32RunnerError("exactly two final role payloads are required")
    allowed_names = {
        Path(paths["best_mIoU_final"]).name,
        Path(paths["best_Pd_final"]).name,
    }
    unexpected = {
        path.name
        for path in run_dir.glob("best_*.pth.tar")
        if path.name not in allowed_names
    }
    if unexpected:
        raise SCTransNetSBSCV32RunnerError("unexpected final weight file exists")

    staged: dict[str, Path] = {}
    try:
        for role in zero_selection.VALID_ROLES:
            final_path = Path(paths[f"{role}_final"])
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{final_path.name}.", suffix=".tmp", dir=run_dir
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            torch.save(dict(payloads[role]), temporary)
            staged[role] = temporary
        for role in zero_selection.VALID_ROLES:
            os.replace(staged[role], Path(paths[f"{role}_final"]))
    finally:
        for temporary in staged.values():
            if temporary.exists():
                temporary.unlink()

    finals: dict[str, dict[str, Any]] = {}
    for role in zero_selection.VALID_ROLES:
        path = Path(paths[f"{role}_final"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("selection_role") != role
            or payload.get("test_split_accessed") is not False
        ):
            raise SCTransNetSBSCV32RunnerError("final checkpoint verification failed")
        finals[role] = {
            "epoch": int(payload["epoch"]),
            "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "file_sha256": _sha256_file(path),
        }
    actual = {path.name for path in run_dir.glob("best_*.pth.tar")}
    if actual != allowed_names:
        raise SCTransNetSBSCV32RunnerError("final physical weight set is not exact")
    return finals


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SCTransNetSBSCV32RunnerError("run lock must not be a symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SCTransNetSBSCV32RunnerError("this run is already active") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_optimizer_state_finite(value: Any) -> None:
    """Reject non-finite live optimizer tensors before an epoch is committed."""

    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all()
            ):
                raise SCTransNetSBSCV32RunnerError(
                    f"optimizer state contains a non-finite tensor at {path}"
                )
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise SCTransNetSBSCV32RunnerError(
                    f"optimizer state contains a non-finite scalar at {path}"
                )
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                raise SCTransNetSBSCV32RunnerError(
                    "optimizer state contains a container cycle"
                )
            seen.add(identity)
            for key, nested in item.items():
                visit(nested, f"{path}.{key}")
            seen.remove(identity)
            return
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                raise SCTransNetSBSCV32RunnerError(
                    "optimizer state contains a container cycle"
                )
            seen.add(identity)
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
            seen.remove(identity)

    visit(value, "optimizer")


def _commit_epoch(
    *,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    identity: Mapping[str, Any],
    training_history: Sequence[Mapping[str, Any]],
    validation_history: Sequence[Mapping[str, Any]],
    validation_record: Mapping[str, Any] | None,
    run_dir: Path,
    candidate_dir: Path,
    history_path: Path,
    latest_path: Path,
    expected_state: Mapping[str, torch.Tensor],
    gain_contract: Mapping[str, Any],
    total_epochs: int,
    smoke: bool,
) -> dict[int, dict[str, Any]]:
    """Commit candidate -> JSON mirror -> latest, then prune safe extras."""

    optimizer_state = optimizer.state_dict()
    _validate_optimizer_state_finite(optimizer_state)
    if validation_record is not None:
        if int(validation_record["epoch"]) != epoch:
            raise SCTransNetSBSCV32RunnerError("validation record epoch differs")
        _save_candidate(
            path=_candidate_path(candidate_dir, epoch),
            epoch=epoch,
            state=state,
            identity=identity,
            record=validation_record,
        )
    candidate_artifacts = _frontier_artifacts(
        history=validation_history,
        completed_epoch=epoch,
        total_epochs=total_epochs,
        smoke=smoke,
        candidate_dir=candidate_dir,
        identity=identity,
    )
    r1._write_json(
        history_path,
        _history_payload(
            identity=identity,
            training_history=training_history,
            validation_history=validation_history,
            candidate_artifacts=candidate_artifacts,
        ),
    )
    r1._atomic_torch_save(
        latest_path,
        {
            "schema": TRAINING_SCHEMA,
            "model": identity["model"],
            "method": identity["method"],
            "dataset": identity["dataset"],
            "epoch": epoch,
            "run_identity": dict(identity),
            "state_dict": dict(state),
            "optimizer": optimizer_state,
            "training_history": [dict(record) for record in training_history],
            "validation_history": [dict(record) for record in validation_history],
            "candidate_artifacts": candidate_artifacts,
            "rng": r1.legacy_train._capture_rng_state(device),
            "test_split_accessed": False,
        },
    )
    if validation_record is not None:
        # Atomic latest above is the commit point.  Pruning before it could
        # destroy the frontier needed by the previous committed state.
        _prune_after_committed_latest(
            candidate_dir=candidate_dir,
            validation_history=validation_history,
            candidate_artifacts=candidate_artifacts,
        )
    return candidate_artifacts


def _training_losses(
    *,
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    method: str,
    core: ModuleType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, six-head segmentation, and train-only router losses."""

    if method not in METHODS:
        raise SCTransNetSBSCV32RunnerError("training loss method is unsupported")
    if method == "sbsc_v32":
        try:
            with core.capture_c3_v32_training_router(model) as capture:
                outputs = model(images)
                if (
                    not isinstance(outputs, (tuple, list))
                    or len(outputs) != 6
                    or any(not isinstance(output, torch.Tensor) for output in outputs)
                ):
                    raise SCTransNetSBSCV32RunnerError(
                        "training forward must return exactly six tensors"
                    )
                router_loss = core.tri_router_supervision_loss(
                    capture,
                    outputs[-1].detach(),
                    masks,
                )
        except SCTransNetSBSCV32RunnerError:
            raise
        except Exception as exc:
            raise SCTransNetSBSCV32RunnerError(
                "SBSC-V3.2 train-only router capture/loss failed"
            ) from exc
    else:
        outputs = model(images)
        if (
            not isinstance(outputs, (tuple, list))
            or len(outputs) != 6
            or any(not isinstance(output, torch.Tensor) for output in outputs)
        ):
            raise SCTransNetSBSCV32RunnerError(
                "training forward must return exactly six tensors"
            )
        router_loss = outputs[-1].new_zeros(())

    segmentation_loss = r1.legacy_train.deep_supervision_loss(
        outputs, masks, criterion
    )
    for label, value in (
        ("segmentation", segmentation_loss),
        ("router", router_loss),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 0
            or not bool(torch.isfinite(value))
            or float(value.detach().item()) < 0.0
        ):
            raise SCTransNetSBSCV32RunnerError(
                f"{label} training loss is malformed"
            )
    total_loss = segmentation_loss + router_loss
    if total_loss.ndim != 0 or not bool(torch.isfinite(total_loss)):
        raise SCTransNetSBSCV32RunnerError("total training loss is non-finite")
    return total_loss, segmentation_loss, router_loss


def _run_locked(
    args: argparse.Namespace, paths: Mapping[str, Path | bool]
) -> dict[str, Path]:
    run_dir = Path(paths["run_dir"])
    candidate_dir = Path(paths["candidate_dir"])
    latest_path = Path(paths["latest"])
    history_path = Path(paths["history"])
    summary_path = Path(paths["summary"])
    smoke = bool(paths["smoke"])

    final_paths = {
        role: Path(paths[f"{role}_final"])
        for role in zero_selection.VALID_ROLES
    }
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError(f"completed run already exists: {summary_path}")
    mutable_artifacts = (
        latest_path,
        history_path,
        candidate_dir,
        *final_paths.values(),
    )
    if not args.resume and any(path.exists() or path.is_symlink() for path in mutable_artifacts):
        raise FileExistsError(f"fresh run directory contains artifacts: {run_dir}")
    if args.resume and (latest_path.is_symlink() or not latest_path.is_file()):
        raise FileNotFoundError(f"resume state is missing: {latest_path}")

    # These are the only dataset constructors reachable from this runner.
    full_train, full_val, grouping = r1.build_datasets(args)
    train_data: Any = full_train
    val_data: Any = full_val
    if args.smoke_max_train_samples is not None:
        train_data = Subset(
            full_train,
            range(min(args.smoke_max_train_samples, len(full_train))),
        )
    if args.smoke_max_val_samples is not None:
        val_data = Subset(
            full_val,
            range(min(args.smoke_max_val_samples, len(full_val))),
        )
    if len(train_data) < 1 or len(val_data) < 1:
        raise SCTransNetSBSCV32RunnerError("train/validation data is empty")

    model, builder_metadata, core, state_contract, inactive = build_model(args)
    identity = _run_identity(
        args,
        full_train.contract,
        grouping,
        train_count=len(train_data),
        val_count=len(val_data),
        smoke=smoke,
        builder_metadata=builder_metadata,
        state_contract=state_contract,
        inactive_parameter_names=inactive,
        core=core,
    )
    device = r1.legacy_train.require_device(args.device)
    model.to(device)
    # Architecture initialization has its own fixed stream.  Runtime training,
    # shuffling, and augmentation all start from the independent fixed stream.
    r1.legacy_train.configure_determinism(RUN_SEED)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    gain_contract = state_contract["gain"]

    start_epoch = 1
    training_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    candidate_artifacts: dict[int, dict[str, Any]] = {}
    if args.resume:
        (
            start_epoch,
            training_history,
            validation_history,
            candidate_artifacts,
        ) = _load_resume_state(
            path=latest_path,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            identity=identity,
            model=model,
            optimizer=optimizer,
            device=device,
            core=core,
            gain_contract=gain_contract,
            total_epochs=args.epochs,
            smoke=smoke,
        )
        # JSON is a mirror, never a commit source.  Repair it from latest.
        r1._write_json(
            history_path,
            _history_payload(
                identity=identity,
                training_history=training_history,
                validation_history=validation_history,
                candidate_artifacts=candidate_artifacts,
            ),
        )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    started = time.time()
    begin = validation_begin_epoch(args.epochs, smoke=smoke)
    for epoch in range(start_epoch, args.epochs + 1):
        full_train.set_epoch(epoch)
        generator = training_shuffle_generator(args, epoch)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=generator,
            drop_last=False,
        )
        learning_rate = r1.legacy_train.learning_rate_for_epoch(
            epoch,
            args.epochs,
            args.base_lr,
            args.min_lr,
            args.warmup_epochs,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        model.mode = "train"
        loss_sum = 0.0
        segmentation_loss_sum = 0.0
        router_loss_sum = 0.0
        processed = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, segmentation_loss, router_loss = _training_losses(
                model=model,
                images=images,
                masks=masks,
                criterion=criterion,
                method=args.method,
                core=core,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training loss is non-finite")
            loss.backward()
            optimizer.step()
            if args.method == "sbsc_v32":
                core.project_sbsc_v32_constraints_(model)
            count = int(images.shape[0])
            processed += count
            loss_sum += float(loss.detach().item()) * count
            segmentation_loss_sum += (
                float(segmentation_loss.detach().item()) * count
            )
            router_loss_sum += float(router_loss.detach().item()) * count
        if processed != len(train_data):
            raise RuntimeError("processed training sample count differs")
        mean_loss = loss_sum / processed
        mean_segmentation_loss = segmentation_loss_sum / processed
        mean_router_loss = router_loss_sum / processed
        training_history.append(
            {
                "epoch": epoch,
                "mean_train_loss": mean_loss,
                "mean_segmentation_loss": mean_segmentation_loss,
                "mean_router_loss": mean_router_loss,
                "router_auxiliary_weight": (
                    1.0 if args.method == "sbsc_v32" else 0.0
                ),
                "learning_rate": learning_rate,
                "processed_samples": processed,
            }
        )
        state = _cpu_state(
            model, method=args.method, gain_contract=gain_contract
        )
        validation_record: dict[str, Any] | None = None
        if epoch >= begin:
            metrics = r1.evaluate_model(model, val_loader, device)
            validation_record = build_validation_record(
                epoch,
                metrics,
                selection_identity=_selection_identity(identity),
                model_state_sha256=_state_dict_sha256(state),
            )
            validation_history.append(validation_record)
        candidate_artifacts = _commit_epoch(
            epoch=epoch,
            state=state,
            optimizer=optimizer,
            device=device,
            identity=identity,
            training_history=training_history,
            validation_history=validation_history,
            validation_record=validation_record,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            history_path=history_path,
            latest_path=latest_path,
            expected_state=model.state_dict(),
            gain_contract=gain_contract,
            total_epochs=args.epochs,
            smoke=smoke,
        )
        validation_text = ""
        if validation_record is not None:
            validation_text = (
                f" val_mIoU={validation_record['mIoU']:.6f}"
                f" val_nIoU={validation_record['nIoU']:.6f}"
                f" val_Pd={validation_record['Pd']:.6f}"
                f" val_Fa={validation_record['Fa']:.8f}"
            )
        print(
            f"method={args.method} epoch={epoch}/{args.epochs}"
            f" loss={mean_loss:.6f} seg={mean_segmentation_loss:.6f}"
            f" router={mean_router_loss:.6f}"
            f" lr={learning_rate:.8f}{validation_text}",
            flush=True,
        )

    selection_payload = _select_final(
        validation_history,
        total_epochs=args.epochs,
        smoke=smoke,
        expected_identity=_selection_identity(identity) if not smoke else None,
    )
    frontier = tuple(selection_payload["retention_frontier_epochs"])
    if tuple(sorted(candidate_artifacts)) != frontier:
        raise SCTransNetSBSCV32RunnerError("final artifacts differ from dual frontier")
    split_provenance = r1._split_provenance(full_train.contract)
    payloads: dict[str, dict[str, Any]] = {}
    for role in zero_selection.VALID_ROLES:
        epoch = int(selection_payload["roles"][role]["selected"]["epoch"])
        selected_record = next(
            (
                record
                for record in validation_history
                if int(record["epoch"]) == epoch
            ),
            None,
        )
        if selected_record is None:
            raise SCTransNetSBSCV32RunnerError(
                f"{role} selected validation record is missing"
            )
        selected_state = _load_candidate_state(
            epoch=epoch,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            identity=identity,
            history=validation_history,
            artifacts=candidate_artifacts,
            expected_state=model.state_dict(),
            gain_contract=gain_contract,
        )
        _require_selected_state_hash(selection_payload, role, selected_state)
        payloads[role] = _final_checkpoint_payload(
            role=role,
            state=selected_state,
            args=args,
            identity=identity,
            split_provenance=split_provenance,
            normalization=full_train.normalization,
            normalization_provenance=full_train.normalization_spec.as_dict(),
            selection_payload=selection_payload,
            selected_validation_record=selected_record,
            builder_metadata=builder_metadata,
            smoke=smoke,
        )
    finals = _atomic_write_final_pair(
        run_dir=run_dir, paths=paths, payloads=payloads
    )
    r1._write_json(
        summary_path,
        {
            "schema": TRAINING_SCHEMA + "/summary",
            "status": "complete",
            "model": identity["model"],
            "method": args.method,
            "dataset": args.dataset,
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": RUN_SEED,
            "single_seed_only": True,
            "target_mode": args.target_mode,
            "final_physical_weights": finals,
            "final_physical_weight_count": 2,
            "selection": selection_payload,
            "split_provenance": split_provenance,
            "normalization": dict(full_train.normalization),
            "training_history": training_history,
            "validation_history": validation_history,
            "candidate_artifacts": {
                str(epoch): dict(candidate_artifacts[epoch])
                for epoch in sorted(candidate_artifacts)
            },
            "elapsed_seconds": time.time() - started,
            "smoke": smoke,
            "test_split_accessed": False,
        },
    )
    return final_paths


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.architecture_seed != ARCHITECTURE_SEED or args.run_seed != RUN_SEED:
        raise SCTransNetSBSCV32RunnerError("architecture and runtime seed must be 42")
    if args.preflight:
        raise SCTransNetSBSCV32RunnerError(
            "preflight is read-only; call preflight_manifest instead of run"
        )
    paths = resolve_run_paths(args)
    lock_path = Path(paths["lock"])
    with _process_lock(lock_path):
        return _run_locked(args, paths)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        print(
            json.dumps(
                preflight_manifest(args),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        return
    finals = run(args)
    print(
        json.dumps(
            {role: str(path) for role, path in finals.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAL_EPOCHS",
    "GAIN_MAX",
    "GAIN_MIN",
    "GAIN_STATE_KEYS",
    "ROUTER_AUXILIARY_LOSS_SCHEMA",
    "ROUTER_PARAMETER_COUNT",
    "ROUTER_STATE_SCHEMA",
    "SEGMENTATION_LOSS_SCHEMA",
    "TOTAL_LOSS_SCHEMA",
    "EXPECTED_BASELINE_PARAMETER_COUNT",
    "EXPECTED_BASELINE_STATE_KEY_COUNT",
    "EXPECTED_SBSC_V32_PARAMETER_COUNT",
    "EXPECTED_SBSC_V32_STATE_KEY_COUNT",
    "HISTORY_SCHEMA",
    "METHODS",
    "RUN_SEED",
    "SHUFFLE_STREAM",
    "SCTransNetSBSCV32RunnerError",
    "TRAINING_SCHEMA",
    "VALIDATION_BEGIN_EPOCH",
    "build_model",
    "build_validation_record",
    "expected_validation_epochs",
    "main",
    "parse_args",
    "preflight_manifest",
    "resolve_run_paths",
    "run",
    "training_shuffle_generator",
    "training_shuffle_seed",
    "validation_begin_epoch",
]
