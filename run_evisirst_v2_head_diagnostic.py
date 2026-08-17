#!/usr/bin/env python3
"""Run the preregistered IRSTD-1K V2 validation head diagnostic.

This entry point has one data role (``val``), one dataset (``IRSTD-1K``),
one frozen split root (``splits/v2``), and three fixed head rules: ``out``,
``d0``, and ``logit_blend(alpha=0.5)``.  Every sample uses one model forward
to expose all six frozen probability maps.  The three summaries are
descriptive only: the supplied epoch candidate has already been evaluated on
this validation split, so this command cannot select a head or checkpoint.

No public-test dataset class or test evaluator is imported by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments import evisirst_head_diagnostic_protocol as ledger_protocol
from experiments import evisirst_v2_selection as validation_selection
from experiments import three_dataset_v2_protocol as source_protocol
from experiments.evisirst_v2_data import (
    AUGMENTATION_VERSION,
    DEFAULT_SPLIT_ROOT,
    EviSIRSTV2ValDataset,
    TARGET_MODES,
)
from model.EviSIRST import initialize_evisirst
from model.evisirst_head_diagnostics import (
    FixedHeadSpec,
    extract_probability_heads,
    select_fixed_head,
)
from train_validation_selected import (
    ASSIGNMENT_ALGORITHM,
    CONNECTED_COMPONENT_CONNECTIVITY,
    CONNECTED_COMPONENT_NEIGHBORHOOD,
    EVALUATION_PROTOCOL_VERSION,
    MATCH_DISTANCE_OPERATOR,
    MATCH_RADIUS,
    PREDICTION_THRESHOLD_OPERATOR,
    PROBABILITY_THRESHOLD,
    TARGET_THRESHOLD,
    TARGET_THRESHOLD_OPERATOR,
    TINY_AREA,
    TINY_AREA_OPERATOR,
    ValidationMetrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = "IRSTD-1K"
DATA_ROLE = "val"
ARCHITECTURE_SEED = 42
MAX_RUN_SEED = (1 << 32) - 1
CANDIDATE_SCHEMA = "evisirst_validation_candidate/v1"
RUN_IDENTITY_SCHEMA = "evisirst_validation_selected_training/v1/run_identity"
DIAGNOSTIC_SCHEMA = "evisirst_v2_fixed_head_diagnostic/v1"
PREDICTION_DIGEST_SCHEMA = "evisirst_fixed_head_prediction_digest/v1"
SOURCE_SELECTION = "evisirst_v2_validation_epoch_candidate"
EXPECTED_EVALUATION = "evisirst_common_evaluate_model_out_head/v1"
EXPECTED_SELECTION_RULE = "evisirst_v2_independent_val_lexicographic/v1"

PREREGISTERED_HEAD_SPECS = (
    FixedHeadSpec("out"),
    FixedHeadSpec("d0"),
    FixedHeadSpec("logit_blend", alpha=0.5),
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_NAME_RE = re.compile(r"^epoch_([0-9]+)\.pth\.tar$")
_RUN_SEED_DIRECTORY_RE = re.compile(r"^run_seed_([0-9]+)$")
_CANDIDATE_KEYS = frozenset(
    {
        "schema",
        "model",
        "dataset",
        "epoch",
        "run_identity",
        "validation_record",
        "state_dict",
        "test_split_accessed",
    }
)
_RUN_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "model",
        "dataset",
        "architecture_seed",
        "run_seed",
        "target_mode",
        "epochs",
        "batch_size",
        "workers",
        "base_lr",
        "min_lr",
        "warmup_epochs",
        "val_interval",
        "normalization_mode",
        "optimizer",
        "loss",
        "evaluation",
        "selection_rule",
        "determinism_protocol",
        "manifest_sha256",
        "split_seed",
        "data_tree_sha256",
        "grouping_policy",
        "train_count",
        "val_count",
        "smoke",
        "smoke_max_train_samples",
        "smoke_max_val_samples",
        "test_split_accessed",
        "identity_sha256",
    }
)
_VALIDATION_RECORD_KEYS = frozenset(
    {"epoch", "data_role", "mIoU", "Fa", "Pd", "evaluation_head", "metrics"}
)
_METRIC_KEYS = frozenset(
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
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
)
_DETERMINISM_KEYS = frozenset(
    {
        "schema",
        "training_data",
        "validation_evaluation",
        "selection",
        "source_set_schema",
        "source_files",
        "source_tree_sha256",
    }
)
_SOURCE_FILE_KEYS = frozenset({"relative_path", "sha256"})
_REQUIRED_SOURCE_NAMES = frozenset(
    {"trainer", "data", "selection", "source_protocol", "model_entry"}
)


class EviSIRSTV2HeadDiagnosticError(ValueError):
    """A request violates the fixed validation diagnostic contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="explicit formal R1 validation candidate under repository runs/",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint metadata must be strict JSON without NaN/Infinity"
        ) from exc


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EviSIRSTV2HeadDiagnosticError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_exact_keys(
    value: Any, expected: frozenset[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EviSIRSTV2HeadDiagnosticError(f"{label} must be a string-keyed mapping")
    missing = sorted(expected.difference(value))
    unexpected = sorted(set(value).difference(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise EviSIRSTV2HeadDiagnosticError(
            f"{label} keys differ ({'; '.join(details)})"
        )
    return value


def _repository_relative_regular_checkpoint(
    raw_path: str | os.PathLike[str],
) -> tuple[Path, str, dict[str, int | str]]:
    lexical = Path(raw_path)
    if not lexical.is_absolute():
        lexical = PROJECT_ROOT / lexical
    lexical = Path(os.path.abspath(lexical))
    repository = PROJECT_ROOT.resolve(strict=True)
    try:
        relative = lexical.relative_to(repository)
    except ValueError as exc:
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint must be inside this repository"
        ) from exc

    current = repository
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise EviSIRSTV2HeadDiagnosticError(
                "checkpoint path must not contain symlink components"
            )
    if not lexical.is_file():
        raise FileNotFoundError(lexical)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint path must resolve without redirection"
        )

    parts = relative.parts
    if (
        len(parts) != 8
        or parts[:4]
        != ("runs", "validation_selected", "formal", DATASET)
        or parts[4] not in TARGET_MODES
        or parts[6] != "candidates"
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint must be a formal IRSTD-1K R1 validation candidate"
        )
    run_match = _RUN_SEED_DIRECTORY_RE.fullmatch(parts[5])
    epoch_match = _CANDIDATE_NAME_RE.fullmatch(parts[7])
    if run_match is None or epoch_match is None:
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint run-seed directory or candidate filename is malformed"
        )
    run_seed = int(run_match.group(1))
    epoch = int(epoch_match.group(1))
    if not 0 <= run_seed <= MAX_RUN_SEED or epoch < 1:
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint path run seed or epoch is outside the R1 contract"
        )
    return resolved, relative.as_posix(), {
        "target_mode": parts[4],
        "run_seed": run_seed,
        "epoch": epoch,
    }


def _read_regular_file_once(path: Path) -> bytes:
    """Read one immutable-by-content snapshot even if retention later unlinks it."""

    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise EviSIRSTV2HeadDiagnosticError(
                "checkpoint descriptor is not a regular file"
            )
        content = handle.read()
        after = os.fstat(handle.fileno())
    if not content or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint changed while its snapshot was being read"
        )
    return content


def _validate_run_identity(
    raw_identity: Any, *, path_identity: Mapping[str, int | str], epoch: int
) -> dict[str, Any]:
    identity = dict(
        _require_exact_keys(raw_identity, _RUN_IDENTITY_KEYS, label="run_identity")
    )
    identity_sha256 = _require_sha256(
        identity["identity_sha256"], label="run_identity.identity_sha256"
    )
    unhashed = dict(identity)
    del unhashed["identity_sha256"]
    if _sha256_bytes(_canonical_json_bytes(unhashed)) != identity_sha256:
        raise EviSIRSTV2HeadDiagnosticError(
            "run_identity identity_sha256 does not recompute"
        )

    fixed_values = {
        "schema": RUN_IDENTITY_SCHEMA,
        "model": "EviSIRST",
        "dataset": DATASET,
        "architecture_seed": ARCHITECTURE_SEED,
        "normalization_mode": "legacy",
        "optimizer": "Adam",
        "loss": "sum_of_six_BCELoss_mean_terms",
        "evaluation": EXPECTED_EVALUATION,
        "selection_rule": EXPECTED_SELECTION_RULE,
        "smoke": False,
        "smoke_max_train_samples": None,
        "smoke_max_val_samples": None,
        "test_split_accessed": False,
    }
    for field, expected in fixed_values.items():
        if type(identity.get(field)) is not type(expected) or identity.get(field) != expected:
            raise EviSIRSTV2HeadDiagnosticError(
                f"run_identity.{field} differs from the formal R1 contract"
            )
    if (
        identity["target_mode"] != path_identity["target_mode"]
        or identity["run_seed"] != path_identity["run_seed"]
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "checkpoint path and run_identity target mode/run seed differ"
        )
    run_seed = identity["run_seed"]
    if (
        isinstance(run_seed, bool)
        or not isinstance(run_seed, int)
        or not 0 <= run_seed <= MAX_RUN_SEED
    ):
        raise EviSIRSTV2HeadDiagnosticError("run_identity.run_seed is invalid")
    for field in (
        "epochs",
        "batch_size",
        "val_interval",
        "train_count",
        "val_count",
    ):
        value = identity[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EviSIRSTV2HeadDiagnosticError(
                f"run_identity.{field} must be a positive integer"
            )
    workers = identity["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise EviSIRSTV2HeadDiagnosticError(
            "run_identity.workers must be a non-negative integer"
        )
    warmup = identity["warmup_epochs"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise EviSIRSTV2HeadDiagnosticError(
            "run_identity.warmup_epochs must be a non-negative integer"
        )
    if epoch > identity["epochs"] or epoch % identity["val_interval"] != 0:
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate epoch is outside the run validation schedule"
        )
    if identity["target_mode"] not in TARGET_MODES:
        raise EviSIRSTV2HeadDiagnosticError("run_identity.target_mode is invalid")
    _require_sha256(identity["manifest_sha256"], label="run_identity.manifest_sha256")
    _require_sha256(identity["data_tree_sha256"], label="run_identity.data_tree_sha256")
    _validate_determinism_protocol(identity["determinism_protocol"])
    _canonical_json_bytes(identity["grouping_policy"])
    return identity


def _validate_determinism_protocol(raw_protocol: Any) -> None:
    protocol = _require_exact_keys(
        raw_protocol, _DETERMINISM_KEYS, label="run_identity.determinism_protocol"
    )
    expected_training_data = {
        "source_protocol_version": source_protocol.PROTOCOL_VERSION,
        "patch_size": source_protocol.PATCH_SIZE,
        "train_positive_crop_probability": (
            source_protocol.TRAIN_POSITIVE_CROP_PROBABILITY
        ),
        "augmentation_version": AUGMENTATION_VERSION,
    }
    expected_validation_evaluation = {
        "version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_head": "out",
        "prediction_threshold": PROBABILITY_THRESHOLD,
        "prediction_threshold_operator": PREDICTION_THRESHOLD_OPERATOR,
        "target_threshold": TARGET_THRESHOLD,
        "target_threshold_operator": TARGET_THRESHOLD_OPERATOR,
        "match_radius": MATCH_RADIUS,
        "match_distance_operator": MATCH_DISTANCE_OPERATOR,
        "connected_component_connectivity": CONNECTED_COMPONENT_CONNECTIVITY,
        "connected_component_neighborhood": CONNECTED_COMPONENT_NEIGHBORHOOD,
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "tiny_area": TINY_AREA,
        "tiny_area_operator": TINY_AREA_OPERATOR,
    }
    expected_selection = {
        "rule_version": validation_selection.INDEPENDENT_RULE_VERSION,
        "miou_candidate_tolerance": validation_selection.MIOU_CANDIDATE_TOLERANCE,
    }
    fixed_values = {
        "schema": "evisirst_validation_selected_determinism/v1",
        "training_data": expected_training_data,
        "validation_evaluation": expected_validation_evaluation,
        "selection": expected_selection,
        "source_set_schema": "evisirst_validation_selected_source_set/v2",
    }
    for field, expected in fixed_values.items():
        if type(protocol.get(field)) is not type(expected) or protocol.get(field) != expected:
            raise EviSIRSTV2HeadDiagnosticError(
                f"run_identity.determinism_protocol.{field} differs"
            )

    source_files = protocol["source_files"]
    if (
        not isinstance(source_files, Mapping)
        or not _REQUIRED_SOURCE_NAMES.issubset(source_files)
        or any(not isinstance(name, str) or not name for name in source_files)
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "run_identity determinism source_files is incomplete"
        )
    normalized_sources: dict[str, dict[str, str]] = {}
    for name, raw_entry in source_files.items():
        entry = _require_exact_keys(
            raw_entry,
            _SOURCE_FILE_KEYS,
            label=f"run_identity.determinism_protocol.source_files.{name}",
        )
        relative_path = entry["relative_path"]
        if not isinstance(relative_path, str):
            raise EviSIRSTV2HeadDiagnosticError(
                "determinism source relative_path must be text"
            )
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path.as_posix() != relative_path
            or relative_path in ("", ".")
        ):
            raise EviSIRSTV2HeadDiagnosticError(
                "determinism source path must be normalized repository-relative text"
            )
        normalized_sources[name] = {
            "relative_path": relative_path,
            "sha256": _require_sha256(
                entry["sha256"],
                label=f"determinism source_files.{name}.sha256",
            ),
        }
    source_tree_sha256 = _require_sha256(
        protocol["source_tree_sha256"],
        label="run_identity.determinism_protocol.source_tree_sha256",
    )
    if _sha256_bytes(_canonical_json_bytes(normalized_sources)) != source_tree_sha256:
        raise EviSIRSTV2HeadDiagnosticError(
            "run_identity determinism source-tree SHA-256 does not recompute"
        )


def _validate_validation_record(raw_record: Any, *, epoch: int) -> dict[str, Any]:
    record = dict(
        _require_exact_keys(
            raw_record, _VALIDATION_RECORD_KEYS, label="validation_record"
        )
    )
    if (
        record["epoch"] != epoch
        or record["data_role"] != DATA_ROLE
        or record["evaluation_head"] != "out"
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate validation record identity differs"
        )
    metrics = _require_exact_keys(
        record["metrics"], _METRIC_KEYS, label="validation_record.metrics"
    )
    _canonical_json_bytes(metrics)
    try:
        normalized = validation_selection.select_independent_checkpoint([record])
    except (TypeError, ValueError) as exc:
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate validation record violates the frozen selection contract"
        ) from exc
    if normalized["selected"]["epoch"] != epoch:
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate validation record epoch failed normalization"
        )
    return record


def _validate_state_dict(
    raw_state: Any, expected_state: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if (
        not isinstance(raw_state, Mapping)
        or len(raw_state) != 564
        or len(expected_state) != 564
        or set(raw_state) != set(expected_state)
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate must contain the exact clean 564-key EviSIRST state"
        )
    state: dict[str, torch.Tensor] = {}
    for key, tensor in raw_state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise EviSIRSTV2HeadDiagnosticError(
                "candidate state_dict must map strings to tensors"
            )
        reference = expected_state[key]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise EviSIRSTV2HeadDiagnosticError(
                f"candidate tensor contract differs for {key!r}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise EviSIRSTV2HeadDiagnosticError(
                f"candidate contains a non-finite tensor: {key!r}"
            )
        state[key] = tensor.detach().cpu()
    if any(key.startswith("target_survival") for key in state):
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate unexpectedly contains target-survival state"
        )
    return state


def load_candidate_model(
    checkpoint: str | os.PathLike[str], device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly load one current-schema formal R1 epoch candidate."""

    path, relative_path, path_identity = _repository_relative_regular_checkpoint(
        checkpoint
    )
    content = _read_regular_file_once(path)
    checkpoint_sha256 = _sha256_bytes(content)
    try:
        raw_payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=True
        )
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate is not a loadable weights-only checkpoint"
        ) from exc
    payload = _require_exact_keys(raw_payload, _CANDIDATE_KEYS, label="candidate")
    epoch = payload["epoch"]
    if (
        payload["schema"] != CANDIDATE_SCHEMA
        or payload["model"] != "EviSIRST"
        or payload["dataset"] != DATASET
        or payload["test_split_accessed"] is not False
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch != path_identity["epoch"]
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate top-level identity differs from the fixed diagnostic"
        )
    identity = _validate_run_identity(
        payload["run_identity"], path_identity=path_identity, epoch=epoch
    )
    validation_record = _validate_validation_record(
        payload["validation_record"], epoch=epoch
    )

    model, model_metadata = initialize_evisirst(
        DATASET, seed=ARCHITECTURE_SEED, training=False
    )
    if len(model.state_dict()) != 564 or hasattr(model, "target_survival"):
        raise EviSIRSTV2HeadDiagnosticError(
            "initializer did not return the clean 564-key EviSIRST graph"
        )
    state = _validate_state_dict(payload["state_dict"], model.state_dict())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise EviSIRSTV2HeadDiagnosticError(
            "candidate strict model load reported incompatible keys"
        )
    model.to(device)
    model.eval()
    model.mode = "test"
    return model, {
        "checkpoint_path": relative_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_schema": CANDIDATE_SCHEMA,
        "epoch": epoch,
        "run_identity": identity,
        "validation_record": validation_record,
        "model_metadata": model_metadata,
    }


def build_validation_dataset(
    *, dataset_root: str | os.PathLike[str], checkpoint_metadata: Mapping[str, Any]
) -> EviSIRSTV2ValDataset:
    identity = checkpoint_metadata["run_identity"]
    dataset = EviSIRSTV2ValDataset(
        DATASET,
        dataset_root=dataset_root,
        split_root=DEFAULT_SPLIT_ROOT,
        target_mode=identity["target_mode"],
        normalization_mode="legacy",
        verify_data_tree=True,
    )
    contract = dataset.contract
    bindings = {
        "manifest_sha256": contract.manifest_sha256,
        "data_tree_sha256": contract.data_tree_sha256,
        "split_seed": contract.manifest["seeds"]["split_seed"],
        "train_count": len(contract.train_ids),
        "val_count": len(contract.val_ids),
    }
    for field, observed in bindings.items():
        if identity[field] != observed:
            raise EviSIRSTV2HeadDiagnosticError(
                f"current frozen V2 split {field} differs from the candidate"
            )
    if (
        not contract.data_tree_verified
        or dataset.metadata.get("split") != DATA_ROLE
        or dataset.metadata.get("test_index_opened") is not False
    ):
        raise EviSIRSTV2HeadDiagnosticError(
            "V2 validation data contract is not fully verified"
        )
    return dataset


def configure_inference_determinism() -> None:
    torch.manual_seed(ARCHITECTURE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ARCHITECTURE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def require_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise EviSIRSTV2HeadDiagnosticError(
            "--device must be cpu, cuda, or cuda:N"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise EviSIRSTV2HeadDiagnosticError(
                f"CUDA device index is out of range: {index}"
            )
    return device


def _extract_hw(value: Any) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        flattened = value.reshape(-1)
        if flattened.numel() != 2:
            raise EviSIRSTV2HeadDiagnosticError("validation size tensor is malformed")
        return int(flattened[0]), int(flattened[1])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(torch.as_tensor(value[0]).reshape(-1)[0]), int(
            torch.as_tensor(value[1]).reshape(-1)[0]
        )
    raise EviSIRSTV2HeadDiagnosticError("validation size value is malformed")


def _extract_sample_id(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value or not value.isascii():
        raise EviSIRSTV2HeadDiagnosticError("validation sample ID is malformed")
    return value


def _digest_update(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _update_prediction_digest(
    digest: Any, *, sample_id: str, probability: np.ndarray
) -> None:
    canonical_probability = np.ascontiguousarray(probability, dtype="<f4")
    header = {
        "sample_id": sample_id,
        "shape": list(canonical_probability.shape),
        "dtype": "float32-little-endian",
    }
    _digest_update(digest, _canonical_json_bytes(header))
    _digest_update(digest, canonical_probability.tobytes(order="C"))


@torch.inference_mode()
def evaluate_preregistered_heads(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, dict[str, float | int | None]], dict[str, str], int]:
    """Compute all three fixed summaries from exactly one forward per sample."""

    accumulators = {
        spec.head: ValidationMetrics(PROBABILITY_THRESHOLD, MATCH_RADIUS, TINY_AREA)
        for spec in PREREGISTERED_HEAD_SPECS
    }
    digests = {}
    for spec in PREREGISTERED_HEAD_SPECS:
        digest = hashlib.sha256()
        _digest_update(digest, PREDICTION_DIGEST_SCHEMA.encode("ascii"))
        _digest_update(digest, spec.head.encode("ascii"))
        _digest_update(
            digest,
            _canonical_json_bytes(
                {
                    "alpha": spec.alpha,
                    "probability_epsilon": spec.probability_epsilon,
                }
            ),
        )
        digests[spec.head] = digest

    criterion = nn.BCELoss(reduction="mean")
    sample_count = 0
    for images, masks, sizes, raw_sample_ids in loader:
        height, width = _extract_hw(sizes)
        sample_id = _extract_sample_id(raw_sample_ids)
        images = images.to(device, non_blocking=True)
        target = masks[:, :, :height, :width].to(device, non_blocking=True)
        if target.ndim != 4 or target.shape[0:2] != (1, 1):
            raise EviSIRSTV2HeadDiagnosticError(
                "validation requires B=1, C=1 target tensors"
            )

        # This is the sole model invocation for the sample.  Head projection
        # and the fixed blend operate only on the returned probability maps.
        probability_heads = extract_probability_heads(model, images)
        for spec in PREREGISTERED_HEAD_SPECS:
            prediction = select_fixed_head(probability_heads, spec).probability[
                :, :, :height, :width
            ]
            if prediction.shape != target.shape or prediction.ndim != 4:
                raise EviSIRSTV2HeadDiagnosticError(
                    f"{spec.head} probability shape differs from validation target"
                )
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError(
                    f"{spec.head} probability contains non-finite values"
                )
            if bool(((prediction < 0.0) | (prediction > 1.0)).any()):
                raise EviSIRSTV2HeadDiagnosticError(
                    f"{spec.head} output is outside probability range [0, 1]"
                )
            loss = criterion(prediction.float(), target.float())
            probability = prediction[0, 0].float().cpu().numpy()
            target_array = target[0, 0].float().cpu().numpy()
            accumulators[spec.head].update(
                probability, target_array, float(loss.item())
            )
            _update_prediction_digest(
                digests[spec.head],
                sample_id=sample_id,
                probability=probability,
            )
        sample_count += 1

    if sample_count < 1 or sample_count != len(loader.dataset):
        raise EviSIRSTV2HeadDiagnosticError(
            "validation loader did not process the complete frozen split"
        )
    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    prediction_sha256 = {name: digest.hexdigest() for name, digest in digests.items()}
    return metrics, prediction_sha256, sample_count


def build_result_payload(
    *,
    checkpoint_metadata: Mapping[str, Any],
    dataset: EviSIRSTV2ValDataset,
    metrics: Mapping[str, Mapping[str, Any]],
    prediction_sha256: Mapping[str, str],
    sample_count: int,
) -> dict[str, Any]:
    identity = checkpoint_metadata["run_identity"]
    checkpoint_provenance = {
        "checkpoint_path": checkpoint_metadata["checkpoint_path"],
        "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "source_selection": SOURCE_SELECTION,
        # This field refers to public-test optimism.  The separate same-split
        # gate below records why these validation results remain descriptive.
        "selection_is_optimistic": False,
    }
    ledgers: dict[str, Any] = {}
    for spec in PREREGISTERED_HEAD_SPECS:
        ledger = ledger_protocol.build_diagnostic_ledger_entry(
            role=ledger_protocol.VALIDATION_ROLE,
            head_spec=spec,
            checkpoint_provenance=checkpoint_provenance,
            result={
                "dataset": DATASET,
                "metrics": dict(metrics[spec.head]),
                "prediction_artifact_sha256": prediction_sha256[spec.head],
            },
            split_manifest_sha256=dataset.contract.manifest_sha256,
            validation_selection_allowed=False,
            validation_checkpoint_selected_on_same_split=True,
        )
        if not ledger["diagnostic_only"] or ledger["selection_allowed"]:
            raise EviSIRSTV2HeadDiagnosticError(
                "same-validation-split ledger unexpectedly permits selection"
            )
        ledgers[spec.head] = ledger

    payload = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "data_role": DATA_ROLE,
        "diagnostic_only": True,
        "selection_allowed": False,
        "secondary_head_selection_prohibited": True,
        "checkpoint": {
            "relative_path": checkpoint_metadata["checkpoint_path"],
            "sha256": checkpoint_metadata["checkpoint_sha256"],
            "schema": checkpoint_metadata["checkpoint_schema"],
            "epoch": checkpoint_metadata["epoch"],
            "architecture_seed": ARCHITECTURE_SEED,
            "run_seed": identity["run_seed"],
            "target_mode": identity["target_mode"],
            "source_selection": SOURCE_SELECTION,
            "selection_is_optimistic": False,
            "selected_on_same_validation_split": True,
            "strict_564_key_load": True,
        },
        "split": {
            "manifest_relative_path": f"splits/v2/{DATASET}/manifest.json",
            "manifest_sha256": dataset.contract.manifest_sha256,
            "data_tree_sha256": dataset.contract.data_tree_sha256,
            "data_tree_verified": dataset.contract.data_tree_verified,
            "train_count": len(dataset.contract.train_ids),
            "val_count": len(dataset.contract.val_ids),
        },
        "evaluation": {
            "metrics_contract": EVALUATION_PROTOCOL_VERSION,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "probability_threshold_operator": PREDICTION_THRESHOLD_OPERATOR,
            "target_threshold": TARGET_THRESHOLD,
            "target_threshold_operator": TARGET_THRESHOLD_OPERATOR,
            "match_radius": MATCH_RADIUS,
            "match_distance_operator": MATCH_DISTANCE_OPERATOR,
            "connected_component_connectivity": CONNECTED_COMPONENT_CONNECTIVITY,
            "connected_component_neighborhood": CONNECTED_COMPONENT_NEIGHBORHOOD,
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "tiny_area": TINY_AREA,
            "tiny_area_operator": TINY_AREA_OPERATOR,
            "sample_count": sample_count,
            "one_model_forward_per_sample": True,
            "preregistered_head_order": [
                {"head": spec.head, "alpha": spec.alpha}
                for spec in PREREGISTERED_HEAD_SPECS
            ],
            "prediction_digest_schema": PREDICTION_DIGEST_SCHEMA,
            "prediction_or_target_arrays_written": False,
        },
        "head_ledgers": ledgers,
        "test_split_accessed": False,
    }
    # Round-trip once to reject non-JSON values and detach frozen ledger maps.
    return json.loads(_canonical_json_bytes(payload).decode("utf-8"))


def _resolve_output_path(raw_path: str | os.PathLike[str]) -> Path:
    lexical = Path(raw_path)
    if not lexical.is_absolute():
        lexical = PROJECT_ROOT / lexical
    lexical = Path(os.path.abspath(lexical))
    repository = PROJECT_ROOT.resolve(strict=True)
    allowed = repository / "runs"
    try:
        relative = lexical.relative_to(allowed)
    except ValueError as exc:
        raise EviSIRSTV2HeadDiagnosticError(
            "--output-json must be inside the repository runs directory"
        ) from exc
    if not relative.parts or lexical.suffix != ".json":
        raise EviSIRSTV2HeadDiagnosticError(
            "--output-json must name a .json file below runs/"
        )
    current = allowed
    if current.is_symlink():
        raise EviSIRSTV2HeadDiagnosticError("repository runs directory is a symlink")
    for component in relative.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise EviSIRSTV2HeadDiagnosticError(
                "output path must not contain symlink components"
            )
    return lexical


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = _resolve_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    try:
        parent.relative_to((PROJECT_ROOT.resolve(strict=True) / "runs"))
    except ValueError as exc:
        raise EviSIRSTV2HeadDiagnosticError(
            "output parent escaped the repository runs directory"
        ) from exc
    content = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
        allow_nan=False,
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
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> Path:
    configure_inference_determinism()
    device = require_device(args.device)
    model, checkpoint_metadata = load_candidate_model(args.checkpoint, device)
    dataset = build_validation_dataset(
        dataset_root=args.dataset_root,
        checkpoint_metadata=checkpoint_metadata,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics, prediction_sha256, sample_count = evaluate_preregistered_heads(
        model, loader, device
    )
    payload = build_result_payload(
        checkpoint_metadata=checkpoint_metadata,
        dataset=dataset,
        metrics=metrics,
        prediction_sha256=prediction_sha256,
        sample_count=sample_count,
    )
    output_path = _resolve_output_path(args.output_json)
    write_json_atomic(output_path, payload)
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    output_path = run(parse_args(argv))
    print(output_path.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_SEED",
    "CANDIDATE_SCHEMA",
    "DATASET",
    "DATA_ROLE",
    "DIAGNOSTIC_SCHEMA",
    "EviSIRSTV2HeadDiagnosticError",
    "PREREGISTERED_HEAD_SPECS",
    "build_result_payload",
    "build_validation_dataset",
    "evaluate_preregistered_heads",
    "load_candidate_model",
    "parse_args",
    "run",
    "write_json_atomic",
]
