#!/usr/bin/env python3
"""Close the fixed Seed-42 A42/H42 IRSTD validation decision.

This command has no configurable paths, seeds, thresholds, or metrics.  It
freshly applies the frozen zero-margin selector to both complete 1000-epoch
validation histories, verifies each selected candidate against its final
checkpoint tensor by tensor, and writes one immutable validation-only ledger.
It never authorizes public-test access or multi-seed expansion.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from experiments import evisirst_zero_margin_selection as zero_selection


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET = "IRSTD-1K"
DATA_ROLE = "val"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
RUN_SEED = 42
EPOCHS = 1000

RESULT_SCHEMA = "evisirst_irstd_hf_decoder_seed42_paired_decision/v1"
GATE_SOURCE_RELATIVE_PATH = "run_irstd_hf_decoder_seed42_decision_gate.py"
SELECTOR_RELATIVE_PATH = "experiments/evisirst_zero_margin_selection.py"
SELECTOR_SHA256 = (
    "776afca34eb5a83f514d145186fed8910548e902a9690620eefa495b94ec94d8"
)

A42_RUN_RELATIVE_PATH = (
    "runs/validation_selected/formal/IRSTD-1K/binary/run_seed_42"
)
H42_RUN_RELATIVE_PATH = (
    "runs/irstd_performance/hf_decoder_seed42_v1/formal/IRSTD-1K/"
    "binary/run_seed_42"
)
A42_SUMMARY_RELATIVE_PATH = f"{A42_RUN_RELATIVE_PATH}/summary.json"
A42_FINAL_RELATIVE_PATH = f"{A42_RUN_RELATIVE_PATH}/EviSIRST.pth.tar"
H42_SUMMARY_RELATIVE_PATH = f"{H42_RUN_RELATIVE_PATH}/summary.json"
H42_FINAL_RELATIVE_PATH = (
    f"{H42_RUN_RELATIVE_PATH}/EviSIRST_best_mIoU.pth.tar"
)
OUTPUT_RELATIVE_PATH = (
    "runs/irstd_performance/hf_decoder_seed42_v1/paired_decision_gate/"
    "run_seed_42/result.json"
)

CANONICAL_MANIFEST_SHA256 = (
    "5eeacbab52d8b70b44ce4cb70dfeda94cd326767f0c379fa96ab836c4c897596"
)
CANONICAL_DATA_TREE_SHA256 = (
    "ceec8eaca922fddb327b3091432d976d92aaeb45541136d43d3f754c3a1b2c30"
)
CANONICAL_SPLIT_SEED = 20260811
CANONICAL_TRAIN_COUNT = 640
CANONICAL_VAL_COUNT = 160
A42_IDENTITY_SHA256 = (
    "7e7c2a523de8fa44add50595b71b50c2ed5ca9d1296c1f7d2f6e965f45f6c6c9"
)
H42_IDENTITY_SHA256 = (
    "24a7cd50ebdee388aeb1de6de6f039e9868d6d37fbfa683cf692b4c07cf9d147"
)
CANONICAL_SPLIT_FILES = {
    "manifest": {
        "relative_path": "splits/v2/IRSTD-1K/manifest.json",
        "sha256": CANONICAL_MANIFEST_SHA256,
    },
    "train": {
        "relative_path": "splits/v2/IRSTD-1K/train.txt",
        "sha256": "460083baae2ba23f5629e7bd346b5623f78a96856c83693b96bf917f6537ed2d",
        "ordered_ids_sha256": (
            "06c1d6055958b648aeaab9be4dc02721eeaba735d4a39fe3184a8aa4ffc6df33"
        ),
        "sample_count": CANONICAL_TRAIN_COUNT,
    },
    "val": {
        "relative_path": "splits/v2/IRSTD-1K/val.txt",
        "sha256": "05a0d0ecdb1772447c4b5b0b04a5e8cf3a748ba5fdab5cdb3f9323d9089e9576",
        "ordered_ids_sha256": (
            "1800db0d308465c9f8d906731576fdbc35fc1c25bc7790cda5232f20fb9ed408"
        ),
        "sample_count": CANONICAL_VAL_COUNT,
    },
}

_FALSE_DISCLOSURE_FIELDS = frozenset(
    {
        "test_split_accessed",
        "test_index_opened",
        "test_selected",
        "test_selection_supported",
        "public_test_allowed",
        "public_test_supported",
    }
)


class Seed42DecisionGateError(ValueError):
    """A fixed input does not prove the Seed-42 paired decision."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise Seed42DecisionGateError(
            "artifact is not strict finite JSON data"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Seed42DecisionGateError(f"{label} must be a lowercase SHA-256")
    return value


def _fixed_path(relative_path: str, *, must_exist: bool) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise Seed42DecisionGateError("internal artifact path is unsafe")
    root = PROJECT_ROOT.resolve(strict=True)
    path = root.joinpath(*pure.parts)
    current = root
    for component in pure.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise Seed42DecisionGateError(
                f"artifact path contains a symlink: {relative_path}"
            )
    if must_exist:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.resolve(strict=True) != path:
            raise Seed42DecisionGateError(
                f"artifact path was redirected: {relative_path}"
            )
    return path


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise Seed42DecisionGateError(f"{label} is not a regular file")
        content = handle.read()
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if not content or before_identity != after_identity:
        raise Seed42DecisionGateError(f"{label} is empty or changed while read")
    return content


def _strict_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            parse_float=_finite_json_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Seed42DecisionGateError(
            f"{label} is not strict finite UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise Seed42DecisionGateError(f"{label} must contain a JSON object")
    _canonical_json_bytes(value)
    return dict(value)


def _artifact_metadata(relative_path: str, content: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "sha256": _sha256_bytes(content),
        "size_bytes": len(content),
    }


def _load_json(
    relative_path: str, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    content = _read_regular_file_once(
        _fixed_path(relative_path, must_exist=True), label=label
    )
    return _strict_json_object(content, label=label), _artifact_metadata(
        relative_path, content
    )


def _load_checkpoint(
    relative_path: str, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    content = _read_regular_file_once(
        _fixed_path(relative_path, must_exist=True), label=label
    )
    try:
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise Seed42DecisionGateError(
            f"{label} is not a weights-only checkpoint"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Seed42DecisionGateError(f"{label} payload must be a mapping")
    return dict(payload), _artifact_metadata(relative_path, content)


def _require_false_disclosures(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FALSE_DISCLOSURE_FIELDS and child is not False:
                raise Seed42DecisionGateError(
                    f"{label}.{key} must be explicit false"
                )
            _require_false_disclosures(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_false_disclosures(child, label=f"{label}[{index}]")


def _require_fixed(value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for field, wanted in expected.items():
        observed = value.get(field)
        if type(observed) is not type(wanted) or observed != wanted:
            raise Seed42DecisionGateError(f"{label}.{field} differs")


def _validate_split(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Seed42DecisionGateError(f"{label} split provenance is missing")
    _require_fixed(
        value,
        {
            "schema": "evisirst_v2_train_val_split/v1",
            "manifest_relative_path": CANONICAL_SPLIT_FILES["manifest"]["relative_path"],
            "manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "split_seed": CANONICAL_SPLIT_SEED,
            "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
            "data_tree_verified": True,
            "train_count": CANONICAL_TRAIN_COUNT,
            "val_count": CANONICAL_VAL_COUNT,
            "test_index_opened": False,
        },
        label=label,
    )
    outputs = value.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"train", "val"}:
        raise Seed42DecisionGateError(f"{label} split outputs differ")
    for role in ("train", "val"):
        expected = CANONICAL_SPLIT_FILES[role]
        wanted = {
            "relative_path": expected["relative_path"],
            "file_sha256": expected["sha256"],
            "ordered_ids_sha256": expected["ordered_ids_sha256"],
            "sample_count": expected["sample_count"],
        }
        if outputs.get(role) != wanted:
            raise Seed42DecisionGateError(f"{label} canonical {role} differs")
    _require_false_disclosures(value, label=label)
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _validate_fixed_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    selector_path = _fixed_path(SELECTOR_RELATIVE_PATH, must_exist=True)
    if Path(zero_selection.__file__).resolve(strict=True) != selector_path:
        raise Seed42DecisionGateError("imported zero-margin selector path differs")
    selector_content = _read_regular_file_once(selector_path, label="selector")
    selector = _artifact_metadata(SELECTOR_RELATIVE_PATH, selector_content)
    if selector["sha256"] != SELECTOR_SHA256:
        raise Seed42DecisionGateError("zero-margin selector SHA-256 differs")
    selector.update(
        {
            "schema": zero_selection.PROVENANCE_SCHEMA,
            "rule_version": zero_selection.RULE_VERSION,
            "primary_role": zero_selection.PRIMARY_ROLE,
            "selection_margin_raw": None,
        }
    )
    split_files: dict[str, Any] = {}
    for role, expected in CANONICAL_SPLIT_FILES.items():
        path = _fixed_path(expected["relative_path"], must_exist=True)
        content = _read_regular_file_once(path, label=f"split {role}")
        metadata = _artifact_metadata(expected["relative_path"], content)
        if metadata["sha256"] != expected["sha256"]:
            raise Seed42DecisionGateError(f"canonical split {role} SHA-256 differs")
        split_files[role] = metadata
    # The manifest is JSON and must independently satisfy the strict parser.
    _strict_json_object(
        _read_regular_file_once(
            _fixed_path(CANONICAL_SPLIT_FILES["manifest"]["relative_path"], must_exist=True),
            label="split manifest",
        ),
        label="split manifest",
    )
    return selector, split_files


def _validate_identity(
    value: Any, *, arm: str, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Seed42DecisionGateError(f"{label} run identity is missing")
    _require_fixed(
        value,
        {
            "dataset": DATASET,
            "model": "EviSIRST",
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": RUN_SEED,
            "target_mode": TARGET_MODE,
            "epochs": EPOCHS,
            "batch_size": 16,
            "workers": 0,
            "base_lr": 0.001,
            "min_lr": 0.00001,
            "warmup_epochs": 10,
            "val_interval": 1,
            "normalization_mode": "legacy",
            "optimizer": "Adam",
            "loss": "sum_of_six_BCELoss_mean_terms",
            "evaluation": "evisirst_common_evaluate_model_out_head/v1",
            "manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "split_seed": CANONICAL_SPLIT_SEED,
            "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
            "train_count": CANONICAL_TRAIN_COUNT,
            "val_count": CANONICAL_VAL_COUNT,
            "smoke": False,
            "test_split_accessed": False,
        },
        label=label,
    )
    if arm == "A42":
        _require_fixed(
            value,
            {
                "schema": "evisirst_validation_selected_training/v1/run_identity",
                "selection_rule": "evisirst_v2_independent_val_lexicographic/v1",
            },
            label=label,
        )
        expected_identity_hash = A42_IDENTITY_SHA256
    elif arm == "H42":
        _require_fixed(
            value,
            {
                "schema": (
                    "evisirst_irstd_hf_decoder_seed42_training/v1/run_identity"
                ),
                "architecture_variant": "IRSTD-HF-Decoder-v1",
                "initialization": "full_model_scratch",
                "all_base_and_hf_parameters_trainable": True,
                "training_crop": "clean_R1_unchanged",
                "stacked_center_head": False,
                "stacked_complete_target_crop": False,
                "stacked_loss_change": False,
                "evaluation_head": "out",
                "selection_rule": zero_selection.RULE_VERSION,
                "selection_margin_raw": None,
                "selection_window_applied": False,
                "selection_roles": list(zero_selection.VALID_ROLES),
                "state_contract": {
                    "base_state_key_count": 564,
                    "hf_state_key_count": 8,
                    "hf_state_prefix": "decoder_hf_residual.",
                    "total_state_key_count": 572,
                },
                "public_test_supported": False,
            },
            label=label,
        )
        expected_identity_hash = H42_IDENTITY_SHA256
    else:
        raise Seed42DecisionGateError(f"unsupported paired arm: {arm}")
    identity = dict(value)
    identity_hash = _require_sha256(
        identity.get("identity_sha256"), label=f"{label}.identity_sha256"
    )
    unhashed = dict(identity)
    unhashed.pop("identity_sha256")
    if _canonical_sha256(unhashed) != identity_hash:
        raise Seed42DecisionGateError(f"{label} identity SHA-256 differs")
    if identity_hash != expected_identity_hash:
        raise Seed42DecisionGateError(
            f"{label} is not the frozen {arm} identity"
        )
    _require_false_disclosures(identity, label=label)
    return identity


def _validate_histories(
    summary: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_epochs = list(range(1, EPOCHS + 1))
    training = summary.get("training_history")
    validation = summary.get("validation_history")
    if (
        not isinstance(training, list)
        or len(training) != EPOCHS
        or [r.get("epoch") if isinstance(r, Mapping) else None for r in training]
        != expected_epochs
    ):
        raise Seed42DecisionGateError(
            f"{label} training history is not continuous 1..1000"
        )
    if (
        not isinstance(validation, list)
        or len(validation) != EPOCHS
        or [r.get("epoch") if isinstance(r, Mapping) else None for r in validation]
        != expected_epochs
    ):
        raise Seed42DecisionGateError(
            f"{label} validation history is not continuous 1..1000"
        )
    _require_false_disclosures(training, label=f"{label}.training_history")
    _require_false_disclosures(validation, label=f"{label}.validation_history")
    try:
        fresh = zero_selection.select_checkpoints(
            validation,
            primary_role=zero_selection.PRIMARY_ROLE,
            margin=None,
        )
    except (TypeError, ValueError) as exc:
        raise Seed42DecisionGateError(
            f"{label} zero-margin selection cannot be recomputed"
        ) from exc
    if (
        fresh.get("evaluated_epochs") != expected_epochs
        or fresh.get("data_role") != DATA_ROLE
        or fresh.get("test_selection_supported") is not False
        or fresh.get("test_split_accessed") is not False
    ):
        raise Seed42DecisionGateError(f"{label} fresh selection contract differs")
    selected = fresh.get("selected")
    if not isinstance(selected, Mapping):
        raise Seed42DecisionGateError(f"{label} fresh selection is missing")
    epoch = selected.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise Seed42DecisionGateError(f"{label} fresh selected epoch is malformed")
    record = validation[epoch - 1]
    metrics = record.get("metrics") if isinstance(record, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise Seed42DecisionGateError(f"{label} selected metrics are missing")
    selected_supplementary: dict[str, float] = {}
    for source_name, output_name, unit_interval in (
        ("pixel_f1", "F1", True),
        ("pixel_precision", "pixel_precision", True),
        ("pixel_recall", "pixel_recall", True),
        ("false_objects_per_image", "false_objects_per_image", False),
    ):
        raw = metrics.get(source_name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise Seed42DecisionGateError(
                f"{label} selected {source_name} must be finite"
            )
        numeric = float(raw)
        if (unit_interval and not 0.0 <= numeric <= 1.0) or (
            not unit_interval and numeric < 0.0
        ):
            raise Seed42DecisionGateError(
                f"{label} selected {source_name} is outside its metric range"
            )
        selected_supplementary[output_name] = numeric
    selected_metrics = {
        "epoch": epoch,
        "mIoU": selected["mIoU"],
        "nIoU": selected["nIoU"],
        **selected_supplementary,
        "Pd": selected["Pd"],
        "Fa": selected["Fa"],
        "tinyPd": selected["tinyPd"],
        "loss": selected["loss"],
    }
    return json.loads(_canonical_json_bytes(fresh).decode("ascii")), selected_metrics


def _validate_summary(
    summary: Mapping[str, Any], *, arm: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    is_hf = arm == "H42"
    label = f"{arm} summary"
    _require_fixed(
        summary,
        {
            "schema": (
                "evisirst_irstd_hf_decoder_seed42_training/v1/summary"
                if is_hf
                else "evisirst_validation_selected_training/v1/summary"
            ),
            "status": "complete",
            "dataset": DATASET,
            "checkpoint": H42_FINAL_RELATIVE_PATH if is_hf else A42_FINAL_RELATIVE_PATH,
            "checkpoint_role": "validation_selected",
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": RUN_SEED,
            "target_mode": TARGET_MODE,
            "source_selection": "evisirst_v2_validation_split",
            "selection_is_optimistic": False,
            "optimistic": False,
            "test_split_accessed": False,
            "smoke": False,
        },
        label=label,
    )
    if is_hf:
        _require_fixed(
            summary,
            {
                "public_test_supported": False,
                "primary_checkpoint_role": zero_selection.PRIMARY_ROLE,
                "selection_margin_raw": None,
                "selection_window_applied": False,
            },
            label=label,
        )
    _require_false_disclosures(summary, label=label)
    split = _validate_split(summary.get("split_provenance"), label=label)
    fresh, selected = _validate_histories(summary, label=label)
    selection = summary.get("selection")
    if not isinstance(selection, Mapping):
        raise Seed42DecisionGateError(f"{label} stored selection is missing")
    epoch = selected["epoch"]
    if summary.get("selected_epoch") != epoch or selection.get("selected_epoch") != epoch:
        raise Seed42DecisionGateError(
            f"{label} stored winner differs from fresh zero-margin winner"
        )
    stored_provenance = selection.get("selection_provenance")
    if not isinstance(stored_provenance, Mapping):
        raise Seed42DecisionGateError(f"{label} stored selection provenance is missing")
    if is_hf and stored_provenance != fresh:
        raise Seed42DecisionGateError(
            "H42 stored zero-margin selection differs from fresh recomputation"
        )
    if not is_hf:
        _require_fixed(
            stored_provenance,
            {
                "schema": "evisirst_v2_validation_selection/v1",
                "rule_version": "evisirst_v2_independent_val_lexicographic/v1",
                "selection_kind": "independent_checkpoint",
                "candidate_tolerance_raw": 0.001,
            },
            label="A42 historical stored selection",
        )
    stored = {
        "rule_version": stored_provenance.get("rule_version"),
        "candidate_tolerance_raw": stored_provenance.get("candidate_tolerance_raw"),
        # The historical selector predates the explicit window_applied field;
        # its frozen 0.001 candidate tolerance and decision trace prove that a
        # window was applied.  H42 records the field directly as false.
        "window_applied": (
            stored_provenance.get("window_applied") if is_hf else True
        ),
        "window_applied_source": (
            "explicit_provenance_field"
            if is_hf
            else "historical_rule_and_candidate_tolerance"
        ),
        "selected_epoch": selection.get("selected_epoch"),
    }
    return fresh, selected, {"split": split, "stored_selection": stored}


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        descriptor = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = _tensor_bytes(tensor)
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_state_dict(value: Any, *, arm: str, label: str) -> dict[str, torch.Tensor]:
    expected_count = 572 if arm == "H42" else 564
    if not isinstance(value, Mapping) or len(value) != expected_count:
        raise Seed42DecisionGateError(
            f"{label} must contain exactly {expected_count} state tensors"
        )
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise Seed42DecisionGateError(f"{label} contains malformed state")
        if tensor.layout != torch.strided:
            raise Seed42DecisionGateError(f"{label}.{key} is not strided")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise Seed42DecisionGateError(f"{label}.{key} is non-finite")
        state[key] = tensor.detach().cpu()
    hf_count = sum(key.startswith("decoder_hf_residual.") for key in state)
    if (arm == "H42" and hf_count != 8) or (arm == "A42" and hf_count != 0):
        raise Seed42DecisionGateError(f"{label} HF state-key contract differs")
    return state


def _compare_state_dicts(
    final_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> str:
    if list(sorted(final_state)) != list(sorted(candidate_state)):
        raise Seed42DecisionGateError(f"{label} state keys differ")
    for key in sorted(final_state):
        left = final_state[key]
        right = candidate_state[key]
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.layout != right.layout
            or not torch.equal(left, right)
        ):
            raise Seed42DecisionGateError(f"{label} tensor differs: {key}")
    final_hash = _state_dict_sha256(final_state)
    candidate_hash = _state_dict_sha256(candidate_state)
    if final_hash != candidate_hash:
        raise Seed42DecisionGateError(f"{label} tensor-state SHA-256 differs")
    return final_hash


def _selected_candidate_relative_path(
    summary: Mapping[str, Any], *, arm: str, epoch: int
) -> tuple[str, str]:
    selection = summary.get("selection")
    artifact = selection.get("selected_candidate") if isinstance(selection, Mapping) else None
    if not isinstance(artifact, Mapping):
        raise Seed42DecisionGateError(f"{arm} selected candidate binding is missing")
    expected_local = f"candidates/epoch_{epoch:04d}.pth.tar"
    if artifact.get("relative_path") != expected_local:
        raise Seed42DecisionGateError(f"{arm} selected candidate path differs")
    expected_sha = _require_sha256(
        artifact.get("file_sha256"), label=f"{arm} candidate SHA-256"
    )
    candidate_artifacts = summary.get("candidate_artifacts")
    if (
        not isinstance(candidate_artifacts, Mapping)
        or candidate_artifacts.get(str(epoch)) != dict(artifact)
    ):
        raise Seed42DecisionGateError(f"{arm} candidate artifact binding differs")
    run_path = H42_RUN_RELATIVE_PATH if arm == "H42" else A42_RUN_RELATIVE_PATH
    return f"{run_path}/{expected_local}", expected_sha


def _validate_checkpoint_pair(
    *,
    arm: str,
    summary: Mapping[str, Any],
    fresh: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    final_relative_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    epoch = int(selected_metrics["epoch"])
    candidate_relative_path, expected_candidate_sha = _selected_candidate_relative_path(
        summary, arm=arm, epoch=epoch
    )
    final, final_meta = _load_checkpoint(
        final_relative_path, label=f"{arm} final checkpoint"
    )
    candidate, candidate_meta = _load_checkpoint(
        candidate_relative_path, label=f"{arm} selected candidate"
    )
    if candidate_meta["sha256"] != expected_candidate_sha:
        raise Seed42DecisionGateError(f"{arm} candidate file SHA-256 differs")
    is_hf = arm == "H42"
    _require_fixed(
        final,
        {
            "schema": (
                "evisirst_irstd_hf_decoder_seed42_checkpoint/v1"
                if is_hf
                else "evisirst_clean_checkpoint/v1"
            ),
            "dataset": DATASET,
            "checkpoint_role": (
                "experimental_validation_selected_best_mIoU"
                if is_hf
                else "validation_selected"
            ),
            "epoch": epoch,
            "seed": ARCHITECTURE_SEED,
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": RUN_SEED,
            "target_mode": TARGET_MODE,
            "source_selection": "evisirst_v2_validation_split",
            "selection_is_optimistic": False,
            "optimistic": False,
            "test_split_accessed": False,
            "smoke": False,
            "split_seed": CANONICAL_SPLIT_SEED,
            "split_manifest_sha256": CANONICAL_MANIFEST_SHA256,
            "data_tree_sha256": CANONICAL_DATA_TREE_SHA256,
            "data_tree_verified": True,
        },
        label=f"{arm} final checkpoint",
    )
    _require_fixed(
        candidate,
        {
            "schema": (
                "evisirst_irstd_hf_decoder_seed42_candidate/v1"
                if is_hf
                else "evisirst_validation_candidate/v1"
            ),
            "dataset": DATASET,
            "epoch": epoch,
            "test_split_accessed": False,
        },
        label=f"{arm} selected candidate",
    )
    if is_hf:
        _require_fixed(
            final,
            {
                "public_test_supported": False,
                "selection_role": zero_selection.PRIMARY_ROLE,
                "selection_margin_raw": None,
                "selection_window_applied": False,
            },
            label="H42 final checkpoint",
        )
        _require_fixed(
            candidate,
            {
                "public_test_supported": False,
                "selection_margin_raw": None,
                "selection_window_applied": False,
            },
            label="H42 selected candidate",
        )
        role = summary.get("role_final_checkpoints", {}).get("best_mIoU")
        if (
            not isinstance(role, Mapping)
            or role.get("epoch") != epoch
            or role.get("relative_path") != "EviSIRST_best_mIoU.pth.tar"
            or role.get("sha256") != final_meta["sha256"]
        ):
            raise Seed42DecisionGateError("H42 final checkpoint hash binding differs")
    _require_false_disclosures(final, label=f"{arm} final checkpoint")
    _require_false_disclosures(candidate, label=f"{arm} selected candidate")
    final_identity = _validate_identity(
        final.get("training"), arm=arm, label=f"{arm} final"
    )
    candidate_identity = _validate_identity(
        candidate.get("run_identity"), arm=arm, label=f"{arm} candidate"
    )
    if _canonical_json_bytes(final_identity) != _canonical_json_bytes(candidate_identity):
        raise Seed42DecisionGateError(f"{arm} final/candidate run identities differ")
    if final.get("training_identity_sha256") != final_identity["identity_sha256"]:
        raise Seed42DecisionGateError(f"{arm} final training identity binding differs")
    if final.get("split_provenance") != summary.get("split_provenance"):
        raise Seed42DecisionGateError(f"{arm} final/summary split differs")
    record = summary["validation_history"][epoch - 1]
    record_sha256 = _canonical_sha256(record)
    if _canonical_json_bytes(candidate.get("validation_record")) != _canonical_json_bytes(record):
        raise Seed42DecisionGateError(f"{arm} candidate validation record differs")
    if is_hf:
        if _canonical_json_bytes(final.get("selected_validation_record")) != _canonical_json_bytes(record):
            raise Seed42DecisionGateError("H42 final selected validation record differs")
        if final.get("selected_complete_key") != fresh.get("selected"):
            raise Seed42DecisionGateError("H42 final selected complete key differs")
        if (
            summary.get("selected_validation_record_sha256") != record_sha256
            or final.get("selected_validation_record_sha256") != record_sha256
        ):
            raise Seed42DecisionGateError(
                "H42 selected validation record SHA-256 differs"
            )
    final_state = _validate_state_dict(
        final.get("state_dict"), arm=arm, label=f"{arm} final state_dict"
    )
    candidate_state = _validate_state_dict(
        candidate.get("state_dict"), arm=arm, label=f"{arm} candidate state_dict"
    )
    state_hash = _compare_state_dicts(
        final_state, candidate_state, label=f"{arm} final/candidate"
    )
    evidence = {
        "final_checkpoint": final_meta,
        "selected_candidate": candidate_meta,
        "selected_epoch": epoch,
        "selected_validation_record": json.loads(
            _canonical_json_bytes(record).decode("ascii")
        ),
        "selected_validation_record_sha256": record_sha256,
        "state_key_count": len(final_state),
        "state_dict_sha256": state_hash,
        "final_candidate_tensor_equality": True,
        "training_identity_sha256": final_identity["identity_sha256"],
    }
    return evidence, [final_meta, candidate_meta]


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Seed42DecisionGateError("decision metric must be numeric")
    result = Decimal(str(value))
    if not result.is_finite():
        raise Seed42DecisionGateError("decision metric must be finite")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def build_result_payload(
    *,
    a42_summary_artifact: Mapping[str, Any],
    h42_summary_artifact: Mapping[str, Any],
    a42_checkpoint_evidence: Mapping[str, Any],
    h42_checkpoint_evidence: Mapping[str, Any],
    a42_fresh_selection: Mapping[str, Any],
    h42_fresh_selection: Mapping[str, Any],
    a42_metrics: Mapping[str, Any],
    h42_metrics: Mapping[str, Any],
    a42_stored_selection: Mapping[str, Any],
    h42_stored_selection: Mapping[str, Any],
    selector_artifact: Mapping[str, Any],
    split_artifacts: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
    gate_source_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    metric_names = (
        "mIoU",
        "nIoU",
        "F1",
        "pixel_precision",
        "pixel_recall",
        "false_objects_per_image",
        "Pd",
        "Fa",
        "tinyPd",
        "loss",
    )
    comparison: dict[str, Any] = {}
    for metric in metric_names:
        a_value = _decimal(a42_metrics[metric])
        h_value = _decimal(h42_metrics[metric])
        delta = h_value - a_value
        comparison[metric] = {
            "A42_raw": float(a_value),
            "H42_raw": float(h_value),
            "H42_minus_A42_raw": float(delta),
            "H42_minus_A42_decimal": _decimal_text(delta),
        }
    miou_delta = _decimal(h42_metrics["mIoU"]) - _decimal(a42_metrics["mIoU"])
    result = "GO" if miou_delta > Decimal("0") else "STOP"
    formal_arm = "H42" if result == "GO" else "A42"
    formal_evidence = (
        h42_checkpoint_evidence if result == "GO" else a42_checkpoint_evidence
    )
    formal_path = H42_FINAL_RELATIVE_PATH if result == "GO" else A42_FINAL_RELATIVE_PATH
    payload = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": DATA_ROLE,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "epochs": EPOCHS,
        "inputs": {
            "A42_clean_R1": {
                "summary": dict(a42_summary_artifact),
                **dict(a42_checkpoint_evidence),
                "stored_selection": dict(a42_stored_selection),
                "fresh_zero_margin_selection_sha256": _canonical_sha256(a42_fresh_selection),
                "fresh_zero_margin_selected_metrics": dict(a42_metrics),
                "stored_and_fresh_primary_winner_equal": True,
            },
            "H42_clean_R1_plus_HF_Decoder_V1": {
                "summary": dict(h42_summary_artifact),
                **dict(h42_checkpoint_evidence),
                "stored_selection": dict(h42_stored_selection),
                "fresh_zero_margin_selection_sha256": _canonical_sha256(h42_fresh_selection),
                "fresh_zero_margin_selected_metrics": dict(h42_metrics),
                "stored_and_fresh_primary_winner_equal": True,
            },
            "zero_margin_selector": dict(selector_artifact),
            "canonical_split_files": dict(split_artifacts),
            "canonical_split_provenance": dict(split_provenance),
            "gate_source": dict(gate_source_artifact),
            "both_histories_complete_and_continuous_1_to_1000": True,
            "both_selections_freshly_recomputed": True,
            "both_final_candidate_pairs_tensor_verified": True,
        },
        "paired_comparison": {
            "definition": "H42_minus_A42",
            "metrics": comparison,
        },
        "decision": {
            "rule": "GO iff Decimal(H42_mIoU)-Decimal(A42_mIoU)>0; otherwise STOP",
            "mIoU_delta_decimal": _decimal_text(miou_delta),
            "result": result,
            "formal_model": {
                "arm": formal_arm,
                "role": "H42_HF_Decoder_V1" if result == "GO" else "A42_clean_R1",
                "relative_path": formal_path,
                "sha256": formal_evidence["final_checkpoint"]["sha256"],
                "selected_epoch": formal_evidence["selected_epoch"],
                "state_dict_sha256": formal_evidence["state_dict_sha256"],
            },
            "multi_seed_expansion_allowed": False,
            "public_test_allowed": False,
        },
        "test_split_accessed": False,
        "public_test_allowed": False,
        "multi_seed_expansion_allowed": False,
    }
    return json.loads(_canonical_json_bytes(payload).decode("ascii"))


def _assert_inputs_unchanged(artifacts: Sequence[Mapping[str, Any]]) -> None:
    for artifact in artifacts:
        relative_path = artifact.get("relative_path")
        expected = artifact.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise Seed42DecisionGateError("input artifact metadata is malformed")
        if _sha256_file(_fixed_path(relative_path, must_exist=True)) != expected:
            raise Seed42DecisionGateError(
                f"input changed during gate evaluation: {relative_path}"
            )


def evaluate_payload() -> dict[str, Any]:
    """Freshly evaluate the fixed gate without writing the formal result."""

    selector_meta, split_files = _validate_fixed_sources()
    gate_path = _fixed_path(GATE_SOURCE_RELATIVE_PATH, must_exist=True)
    gate_content = _read_regular_file_once(gate_path, label="gate source")
    gate_meta = _artifact_metadata(GATE_SOURCE_RELATIVE_PATH, gate_content)
    a_summary, a_summary_meta = _load_json(
        A42_SUMMARY_RELATIVE_PATH, label="A42 summary"
    )
    h_summary, h_summary_meta = _load_json(
        H42_SUMMARY_RELATIVE_PATH, label="H42 summary"
    )
    a_fresh, a_metrics, a_aux = _validate_summary(a_summary, arm="A42")
    h_fresh, h_metrics, h_aux = _validate_summary(h_summary, arm="H42")
    if a_aux["split"] != h_aux["split"]:
        raise Seed42DecisionGateError("A42/H42 canonical split provenance differs")
    a_checkpoint, a_checkpoint_artifacts = _validate_checkpoint_pair(
        arm="A42",
        summary=a_summary,
        fresh=a_fresh,
        selected_metrics=a_metrics,
        final_relative_path=A42_FINAL_RELATIVE_PATH,
    )
    h_checkpoint, h_checkpoint_artifacts = _validate_checkpoint_pair(
        arm="H42",
        summary=h_summary,
        fresh=h_fresh,
        selected_metrics=h_metrics,
        final_relative_path=H42_FINAL_RELATIVE_PATH,
    )
    observed = [
        selector_meta,
        *split_files.values(),
        gate_meta,
        a_summary_meta,
        h_summary_meta,
        *a_checkpoint_artifacts,
        *h_checkpoint_artifacts,
    ]
    _assert_inputs_unchanged(observed)
    # Rebind the imported selector and fixed split bytes at the end as a
    # same-evaluation race check.
    selector_final, split_final = _validate_fixed_sources()
    if selector_final != selector_meta or split_final != split_files:
        raise Seed42DecisionGateError("selector or split changed during evaluation")
    return build_result_payload(
        a42_summary_artifact=a_summary_meta,
        h42_summary_artifact=h_summary_meta,
        a42_checkpoint_evidence=a_checkpoint,
        h42_checkpoint_evidence=h_checkpoint,
        a42_fresh_selection=a_fresh,
        h42_fresh_selection=h_fresh,
        a42_metrics=a_metrics,
        h42_metrics=h_metrics,
        a42_stored_selection=a_aux["stored_selection"],
        h42_stored_selection=h_aux["stored_selection"],
        selector_artifact=selector_meta,
        split_artifacts=split_files,
        split_provenance=a_aux["split"],
        gate_source_artifact=gate_meta,
    )


def _write_fixed_json_atomic_no_replace(payload: Mapping[str, Any]) -> Path:
    path = _fixed_path(OUTPUT_RELATIVE_PATH, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise Seed42DecisionGateError("decision-gate output directory is a symlink")
    parent = path.parent.resolve(strict=True)
    runs_root = (PROJECT_ROOT.resolve(strict=True) / "runs").resolve(strict=True)
    try:
        parent.relative_to(runs_root)
    except ValueError as exc:
        raise Seed42DecisionGateError("decision-gate output escaped runs/") from exc
    content = json.dumps(
        dict(payload), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(
                    f"Seed42 decision result already exists and is immutable: {path}"
                ) from exc
            raise
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def validate_existing_result() -> dict[str, Any]:
    """Compare an existing immutable ledger with a full fresh evaluation."""

    observed, metadata = _load_json(
        OUTPUT_RELATIVE_PATH, label="existing Seed42 decision result"
    )
    expected = evaluate_payload()
    observed_bytes = _canonical_json_bytes(observed)
    expected_bytes = _canonical_json_bytes(expected)
    if observed_bytes != expected_bytes:
        raise Seed42DecisionGateError(
            "existing Seed42 decision differs from fresh canonical evaluation"
        )
    if _sha256_file(_fixed_path(OUTPUT_RELATIVE_PATH, must_exist=True)) != metadata["sha256"]:
        raise Seed42DecisionGateError("existing Seed42 decision changed during validation")
    return json.loads(expected_bytes.decode("ascii"))


def run(_args: argparse.Namespace | None = None) -> Path:
    return _write_fixed_json_atomic_no_replace(evaluate_payload())


def main(argv: Sequence[str] | None = None) -> None:
    output = run(parse_args(argv))
    print(output.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "A42_FINAL_RELATIVE_PATH",
    "A42_SUMMARY_RELATIVE_PATH",
    "H42_FINAL_RELATIVE_PATH",
    "H42_SUMMARY_RELATIVE_PATH",
    "OUTPUT_RELATIVE_PATH",
    "RESULT_SCHEMA",
    "Seed42DecisionGateError",
    "build_result_payload",
    "evaluate_payload",
    "main",
    "parse_args",
    "run",
    "validate_existing_result",
]
