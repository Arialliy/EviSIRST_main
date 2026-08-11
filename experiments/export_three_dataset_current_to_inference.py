#!/usr/bin/env python3
"""Freeze the three Current best-mIoU checkpoints as TSS-free bundles.

This exporter is deliberately independent from the historical official-test
evaluator.  It reads only already-persisted checkpoint, summary, protocol, and
source-code files.  It never imports an official-test loader, opens or parses
an official-test index, reads an official-test sample, runs official-test
inference, or recomputes an official-test metric.

The historical source checkpoints remain explicitly disclosed as
test-selected/optimistic operational checkpoints.  Export removes exactly the
four exact-zero training-only TSS tensors and retains the complete
TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA inference graph.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import four_dataset_models_seed42_v1 as registry  # noqa: E402
from model.tpd_forward_contract import legacy_output  # noqa: E402
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (  # noqa: E402
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (  # noqa: E402
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


SOURCE_SCHEMA = "sctransnet_three_dataset_tss_off_seed42_v1/v1"
PACKAGE_SCHEMA = "sctransnet_three_component_current_inference_package/v1"
MANIFEST_SCHEMA = "sctransnet_three_component_current_inference_manifest/v1"
COMMITTED_SCHEMA = "sctransnet_three_component_current_inference_committed/v1"
MODEL_NAME = "TPD8-MPRS-DCH + NER4 Tail-Aware + QFG2-CROA"
MODEL_PUBLIC_ID = "sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa"
IMPLEMENTATION_CLASS = "TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet"
ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256 = (
    "01836fa26f47f63c4a4362e22a3f66955a61a3fc2adba9fd76d63c6758b15b00"
)
SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256 = (
    "e9398d2b720c15cd3fa7a04c2c07ff050b8ad92a05d8f31978424654f0169cac"
)
STATE_HASH_ALGORITHM = "state_dict_sha256"
STATE_HASH_CANONICALIZATION = (
    "sorted key + key UTF-8 + dtype + shape + contiguous CPU tensor bytes"
)
OFFICIAL_BOUNDARY_SCOPE = "deployment_export_operation_only"
TRAINING_SEED = 42
TRAINING_STATE_KEY_COUNT = registry.FINAL_TRAINING_STATE_KEY_COUNT
INFERENCE_STATE_KEY_COUNT = registry.FINAL_INFERENCE_STATE_KEY_COUNT
TRAINING_PARAMETER_COUNT = registry.FINAL_TRAINING_PARAMETER_COUNT
INFERENCE_PARAMETER_COUNT = registry.FINAL_INFERENCE_PARAMETER_COUNT
TSS_PARAMETER_COUNT = TRAINING_PARAMETER_COUNT - INFERENCE_PARAMETER_COUNT
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
CHECKPOINT_ROLE = "best_miou"
CHECKPOINT_RELATIVE = Path("checkpoints/best_miou.pth.tar")
SUMMARY_RELATIVE = Path("summary.json")
PROTOCOL_RELATIVE = Path("protocol.json")
DEFAULT_SOURCE_ROOT = REPO_ROOT / "results/three_dataset_tss_off_seed42_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/current_three_component_inference_v1"
DEPLOYMENT_PROTOCOL_PATH = (
    REPO_ROOT / "experiments/CURRENT_THREE_COMPONENT_DEPLOYMENT_PROTOCOL_V1.md"
)

# These five fields describe this export operation, not the historical
# checkpoint-selection procedure.  Historical test selection is disclosed
# separately and is never relabelled as validation selection.
OFFICIAL_FALSE_FLAGS = (
    "official_evaluation_performed",
    "official_test_accessed",
    "official_test_index_opened",
    "official_test_index_parsed",
    "official_test_loader_built",
)

SOURCE_LOCKS: dict[str, dict[str, Any]] = {
    "NUAA-SIRST": {
        "epoch": 850,
        "checkpoint_bytes": 43_726_377,
        "checkpoint_file_sha256": (
            "e6958eebb4a4a5493342a9faf285b2c57a5d58804f150656a87585fec3043f0a"
        ),
        "summary_file_sha256": (
            "32a93016d28dbf7d67d7760c7e7cce1f69c2ced422c3ab23048ea97ed618ee8b"
        ),
        "protocol_file_sha256": (
            "c66ed6a767125c15dac9ef6b64fea49840d82d54a1dc44d7e9dfec9d71cc8530"
        ),
        "protocol_payload_sha256": (
            "3d36c69e342aa94d548f4f42e37514a776b08ad572694ca0fcdf6fe413d36595"
        ),
        "training_state_sha256": (
            "f8dd707c46015f826fa4cd09e65f1eedffeed49f075117ba976b388fbedf8585"
        ),
        "inference_state_sha256": (
            "8b669664a465d1f006c705434f564cbfce230170e5e27e654ddc6e593bb3bb34"
        ),
    },
    "NUDT-SIRST": {
        "epoch": 420,
        "checkpoint_bytes": 43_726_377,
        "checkpoint_file_sha256": (
            "0f5f6a5fe96fa86302807d132078d575495a3aff6690967785868a23400f3e84"
        ),
        "summary_file_sha256": (
            "ddd3013c7aefd22d15eefed1d97508fca9bedcb70710ceaa657bc4443ead6a3f"
        ),
        "protocol_file_sha256": (
            "b5fedbd78ae3845e04094c506ce6acf0781b6567e4bd41815df7ab51aa325b4f"
        ),
        "protocol_payload_sha256": (
            "d2b0fffca4bd73a0c4133058287620ded3a839e5b1d90e7089bd06c3a2f3680e"
        ),
        "training_state_sha256": (
            "c10eebf67f3f25bd252881b61b5e3f9ad0a850dc594f79c451eb53325101e2da"
        ),
        "inference_state_sha256": (
            "8949da019fc87f476b3dd29cf8f85bf194ede1cf2fdb466a6c4a6c1da5c67a05"
        ),
    },
    "IRSTD-1K": {
        "epoch": 830,
        "checkpoint_bytes": 43_726_377,
        "checkpoint_file_sha256": (
            "e8e9401500502dda0bbdc9640b830a7934fb2bc97bde706fde9adca216d965b4"
        ),
        "summary_file_sha256": (
            "f85a49e05992f25dd37cd58247088c3cd7d63fef8b8b509dd85ef1f117f54d6e"
        ),
        "protocol_file_sha256": (
            "6ebeb6f5ce1f777a15c79989808dd281d23ac49ec36b70ee2ebd1f93b6ebef48"
        ),
        "protocol_payload_sha256": (
            "bfe6af45e2740c1e44b2ce996a1f74ab726c082aa18148536dcb26f45ea2dea1"
        ),
        "training_state_sha256": (
            "d7600f61ee3d0967dae899de72a28f2e7e9e4c6381f2687189e45d84dcb3e298"
        ),
        "inference_state_sha256": (
            "c511c1c5ef82be0582fcfdd0353df515e013dc85c0e27e46d15a256f58e5434d"
        ),
    },
}

PACKAGE_FILENAMES = {
    "NUAA-SIRST": "nuaa_sirst_best_miou_epoch850_inference.pth.tar",
    "NUDT-SIRST": "nudt_sirst_best_miou_epoch420_inference.pth.tar",
    "IRSTD-1K": "irstd_1k_best_miou_epoch830_inference.pth.tar",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _official_false_payload() -> dict[str, bool]:
    return {field: False for field in OFFICIAL_FALSE_FLAGS}


def _recipe_identity() -> dict[str, Any]:
    return {
        "method": "final",
        "recipe_id": "final_tss_off",
        "requested_tss_weight": 0.0,
        "tss_lambda_token": "off",
        "tss_ratio_cap": 0.1,
        "tss_ratio_cap_applied": False,
        "tss_enabled": False,
        "tss_heads_registered": True,
        "tss_training_forward_computes_logits": True,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
    }


def _require_dataset(dataset: str) -> str:
    if type(dataset) is not str or dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    return dataset


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _regular_file_bytes(path: Path) -> bytes:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"not a regular file: {candidate}")
    return candidate.read_bytes()


def _json_regular(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    content = _regular_file_bytes(path)
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value, content


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _state_hash_contract() -> dict[str, Any]:
    return {
        "algorithm": STATE_HASH_ALGORITHM,
        "version": 1,
        "canonicalization": STATE_HASH_CANONICALIZATION,
    }


def _deployment_implementation_binding() -> dict[str, str]:
    exporter_path = Path(__file__).resolve()
    protocol_path = DEPLOYMENT_PROTOCOL_PATH.resolve()
    repository = REPO_ROOT.resolve()
    _require(exporter_path.is_relative_to(repository), "exporter escapes repository")
    _require(protocol_path.is_relative_to(repository), "deployment protocol escapes repository")
    exporter_content = _regular_file_bytes(exporter_path)
    protocol_content = _regular_file_bytes(protocol_path)
    return {
        "exporter_repo_relative_path": exporter_path.relative_to(repository).as_posix(),
        "exporter_file_sha256": _sha256_bytes(exporter_content),
        "protocol_repo_relative_path": protocol_path.relative_to(repository).as_posix(),
        "protocol_file_sha256": _sha256_bytes(protocol_content),
    }


def _architecture_contract(
    model: TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
) -> dict[str, Any]:
    _require(type(model) is TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet, "architecture class differs")
    manifest = model.architecture_manifest()
    _require(isinstance(manifest, Mapping) and bool(manifest), "architecture manifest differs")
    # Persist the exact canonical JSON body, not Python tuple spellings, so the
    # package and root JSON manifest compare identically after re-open.
    manifest = json.loads(
        json.dumps(
            dict(manifest),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    manifest_sha = _canonical_sha256(manifest)
    _require(
        manifest_sha == ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256,
        "architecture manifest SHA differs",
    )
    state_keys = sorted(model.state_dict())
    state_key_set_sha = _canonical_sha256(state_keys)
    _require(
        state_key_set_sha == SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256,
        "sorted state-key set SHA differs",
    )
    _require(len(state_keys) == INFERENCE_STATE_KEY_COUNT, "architecture state-key count differs")
    _require(
        sum(parameter.numel() for parameter in model.parameters())
        == INFERENCE_PARAMETER_COUNT,
        "architecture parameter count differs",
    )
    _require(not hasattr(model, "target_survival"), "architecture retains TSS")
    return {
        "public_name": MODEL_NAME,
        "public_id": MODEL_PUBLIC_ID,
        "implementation_class": IMPLEMENTATION_CLASS,
        "architecture_manifest": manifest,
        "architecture_manifest_canonical_json_sha256": manifest_sha,
        "sorted_state_key_set_canonical_json_sha256": state_key_set_sha,
        "state_key_count": INFERENCE_STATE_KEY_COUNT,
        "parameter_count": INFERENCE_PARAMETER_COUNT,
        "target_survival_registered": False,
        "public_output": "sigmoid(out)",
    }


def _synthetic_probe() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 32 * 32, dtype=torch.float32).reshape(
        1, 1, 32, 32
    )


def _synthetic_tensor_hashes(
    probe: torch.Tensor,
    six_outputs: Sequence[torch.Tensor],
) -> dict[str, str]:
    _require(len(six_outputs) == 6, "synthetic output count differs")
    return {
        "probe_state_dict_sha256": registry.state_dict_sha256({"probe": probe}),
        "six_outputs_state_dict_sha256": registry.state_dict_sha256(
            {f"output_{index}": value for index, value in enumerate(six_outputs)}
        ),
        "public_output_state_dict_sha256": registry.state_dict_sha256(
            {"sigmoid_out": six_outputs[-1]}
        ),
    }


def _replay_inference_synthetic_identity(
    model: TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
) -> dict[str, Any]:
    model.eval()
    probe = _synthetic_probe()
    model.mode = "train"
    with torch.inference_mode():
        six_outputs = legacy_output(model(probe))
    _require(isinstance(six_outputs, tuple) and len(six_outputs) == 6, "synthetic replay output count differs")
    model.mode = "test"
    with torch.inference_mode():
        deployed = model(probe)
    _require(isinstance(deployed, torch.Tensor), "synthetic replay deployment output differs")
    _require(torch.equal(deployed, six_outputs[-1]), "synthetic replay public output differs")
    return {
        "probe": "linspace[-1,1]_1x1x32x32_float32_cpu",
        "hash_algorithm": STATE_HASH_ALGORITHM,
        **_synthetic_tensor_hashes(probe, six_outputs),
        "six_output_count": 6,
        "deployed_single_output_equals_sixth": True,
        "deployed_output": "sigmoid(out)",
    }


def _historical_metrics_equal(
    summary_metrics: Mapping[str, Any],
    checkpoint_metrics: Mapping[str, Any],
) -> bool:
    """Compare persisted metrics across JSON and NumPy scalar spellings."""

    if set(summary_metrics) != set(checkpoint_metrics):
        return False
    for field in summary_metrics:
        summary_value = summary_metrics[field]
        checkpoint_value = checkpoint_metrics[field]
        if isinstance(summary_value, bool) or isinstance(checkpoint_value, bool):
            if summary_value is not checkpoint_value:
                return False
            continue
        try:
            summary_number = float(summary_value)
            checkpoint_number = float(checkpoint_value)
        except (TypeError, ValueError):
            if summary_value != checkpoint_value:
                return False
            continue
        if not (
            math.isfinite(summary_number)
            and math.isfinite(checkpoint_number)
            and summary_number == checkpoint_number
        ):
            return False
    return True


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _torch_bytes(value: Mapping[str, Any]) -> bytes:
    serialized = io.BytesIO()
    torch.save(dict(value), serialized)
    return serialized.getvalue()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_once(path: Path, content: bytes) -> None:
    """Atomically create one file and never replace an existing path."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise NotADirectoryError(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite: {destination}"
            ) from error
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _run_directory(source_root: Path, dataset: str) -> Path:
    return (
        Path(source_root).resolve()
        / "runs"
        / _require_dataset(dataset)
        / "final_tss_off"
        / "seed_42"
    )


def _validate_runtime_source_locks(protocol: Mapping[str, Any]) -> dict[str, str]:
    """Hash frozen source files without importing evaluator/data modules."""

    raw = protocol.get("runtime_sources")
    _require(isinstance(raw, Mapping) and bool(raw), "protocol lacks runtime_sources")
    verified: dict[str, str] = {}
    repository = REPO_ROOT.resolve()
    for name, entry in sorted(raw.items()):
        _require(isinstance(name, str) and bool(name), "runtime source name differs")
        _require(isinstance(entry, Mapping), f"runtime source {name!r} differs")
        path_text = entry.get("path")
        _require(
            isinstance(path_text, str) and bool(path_text),
            f"runtime source {name!r} path differs",
        )
        path = Path(path_text).resolve(strict=True)
        _require(
            path.is_relative_to(repository),
            f"runtime source {name!r} escapes repository",
        )
        _require(
            path.suffix in {".py", ".md"},
            f"runtime source {name!r} is not source/documentation",
        )
        digest = _sha256_bytes(_regular_file_bytes(path))
        _require(entry.get("sha256") == digest, f"runtime source {name!r} SHA differs")
        verified[name] = digest
    return verified


def _validate_model_metadata(
    metadata: Any,
    *,
    dataset: str,
) -> dict[str, Any]:
    _require(isinstance(metadata, Mapping), "checkpoint lacks model_metadata")
    ready = dict(metadata)
    expected = {
        "schema": registry.BUILDER_SCHEMA,
        "method": "final_scratch",
        "dataset_name": dataset,
        "training_seed": TRAINING_SEED,
        "training_graph_requested": True,
        "selected_model_parameter_count": TRAINING_PARAMETER_COUNT,
        "selected_model_state_key_count": TRAINING_STATE_KEY_COUNT,
        "warm_start_used": False,
        "parent_checkpoint": None,
    }
    for field, value in expected.items():
        _require(ready.get(field) == value, f"model_metadata {field!r} differs")
    _require(
        ready.get("formal_three_dataset_scope") == list(DATASETS),
        "model_metadata formal three-dataset scope differs",
    )
    objective = ready.get("formal_training_objective")
    _require(isinstance(objective, Mapping), "model_metadata lacks objective")
    objective_expected = {
        "authority": "three_dataset_tss_off_seed42_v1_run_recipe",
        "method": "final",
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "tss_heads_registered": True,
        "tss_training_forward_computes_logits": True,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
    }
    for field, value in objective_expected.items():
        _require(objective.get(field) == value, f"model objective {field!r} differs")
    _require(
        ready.get("tss_off_control") == _recipe_identity(),
        "model_metadata TSS-off control differs",
    )
    pair = ready.get("pair")
    _require(isinstance(pair, Mapping), "model_metadata lacks paired identity")
    final = pair.get("final")
    _require(isinstance(final, Mapping), "model_metadata lacks Final identity")
    final_expected = {
        "class": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival."
            "TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet"
        ),
        "training_graph": (
            "sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa_tss"
        ),
        "inference_graph": (
            "sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa"
        ),
        "parameter_count": TRAINING_PARAMETER_COUNT,
        "state_key_count": TRAINING_STATE_KEY_COUNT,
        "tss_registered": True,
        "tss_training_only": True,
    }
    for field, value in final_expected.items():
        _require(final.get(field) == value, f"Final metadata {field!r} differs")
    return ready


def require_exact_zero_tss_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Require the exact four TSS tensors and prove all 98 values are zero."""

    _require(isinstance(state_dict, Mapping), "training state must be a mapping")
    observed = {
        key for key in state_dict if isinstance(key, str) and key.startswith(SURVIVAL_STATE_PREFIX)
    }
    _require(observed == set(SURVIVAL_STATE_KEYS), "training state TSS keys differ")
    tensors: dict[str, dict[str, Any]] = {}
    total = 0
    for key in SURVIVAL_STATE_KEYS:
        value = state_dict[key]
        _require(isinstance(value, torch.Tensor), f"TSS state {key!r} is not a Tensor")
        _require(value.is_floating_point(), f"TSS state {key!r} is not floating point")
        _require(bool(torch.isfinite(value).all()), f"TSS state {key!r} is non-finite")
        nonzero = int(torch.count_nonzero(value).item())
        _require(nonzero == 0, f"TSS state {key!r} is not exact zero")
        total += int(value.numel())
        tensors[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "nonzero": nonzero,
        }
    _require(total == TSS_PARAMETER_COUNT == 98, "TSS parameter count differs")
    return {
        "state_keys": list(SURVIVAL_STATE_KEYS),
        "state_key_count": len(SURVIVAL_STATE_KEYS),
        "parameter_count": total,
        "all_exact_zero": True,
        "tensors": tensors,
    }


def _require_finite_tensor_mapping(
    state_dict: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    for key, value in state_dict.items():
        _require(type(key) is str and bool(key), f"{label} key differs")
        _require(isinstance(value, torch.Tensor), f"{label} {key!r} is not a Tensor")
        if value.is_floating_point() or value.is_complex():
            _require(bool(torch.isfinite(value).all()), f"{label} {key!r} is non-finite")


def assert_synthetic_six_output_bitwise_identity(
    state_dict: Mapping[str, torch.Tensor],
    *,
    dataset: str,
) -> dict[str, Any]:
    """Prove all six segmentation maps survive exact TSS removal bitwise."""

    dataset = _require_dataset(dataset)
    training_model, _ = registry.build_paper_model(
        "final",
        dataset,
        seed=TRAINING_SEED,
        training=True,
    )
    incompatible = training_model.load_state_dict(state_dict, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "training graph strict load returned incompatible keys",
    )
    inference_model, metadata = registry.build_final_inference_model_from_training_state_dict(
        state_dict,
        dataset_name=dataset,
        seed=TRAINING_SEED,
    )
    _require(
        type(inference_model) is TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
        "inference graph class differs",
    )
    architecture = _architecture_contract(inference_model)
    _require(not hasattr(inference_model, "target_survival"), "inference graph retains TSS")
    training_model.eval()
    inference_model.eval()
    training_model.mode = "train"
    inference_model.mode = "train"
    probe = _synthetic_probe()
    with torch.inference_mode():
        training_six = legacy_output(training_model(probe))
        inference_six = legacy_output(inference_model(probe))
    _require(
        isinstance(training_six, tuple)
        and isinstance(inference_six, tuple)
        and len(training_six) == len(inference_six) == 6,
        "synthetic identity requires exactly six segmentation outputs",
    )
    for index, (training_value, inference_value) in enumerate(
        zip(training_six, inference_six)
    ):
        _require(training_value.shape == inference_value.shape, f"output {index} shape differs")
        _require(training_value.dtype == inference_value.dtype, f"output {index} dtype differs")
        _require(torch.equal(training_value, inference_value), f"output {index} is not bitwise equal")
    inference_model.mode = "test"
    with torch.inference_mode():
        deployed = inference_model(probe)
    _require(isinstance(deployed, torch.Tensor), "mode=test did not return one Tensor")
    _require(torch.equal(deployed, inference_six[-1]), "deployed output differs from final map")
    _require(metadata.get("strict_load") is True, "inference metadata lacks strict load")
    tensor_hashes = _synthetic_tensor_hashes(probe, inference_six)
    return {
        "probe": "linspace[-1,1]_1x1x32x32_float32_cpu",
        "hash_algorithm": STATE_HASH_ALGORITHM,
        **tensor_hashes,
        "six_output_count": 6,
        "all_six_bitwise_equal": True,
        "maximum_absolute_difference": 0.0,
        "deployed_single_output_equals_sixth": True,
        "deployed_output": "sigmoid(out)",
        "architecture": architecture,
    }


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    dataset: str,
    expected_payload_sha256: str,
) -> tuple[str, dict[str, str]]:
    expected = {
        "schema": SOURCE_SCHEMA,
        "dataset": dataset,
        "method": "final",
        "training_seed": TRAINING_SEED,
        "epochs": 1000,
        "begin_test": 10,
        "eval_every": 10,
        "smoke": False,
        "test_selected": True,
        "selection_is_optimistic": True,
        "checkpoint_roles": ["best_miou", "best_pd"],
        "recipe": _recipe_identity(),
    }
    for field, value in expected.items():
        _require(protocol.get(field) == value, f"protocol {field!r} differs")
    declared = protocol.get("protocol_sha256")
    _require(_is_sha256(declared), "protocol payload SHA is malformed")
    unsigned = dict(protocol)
    del unsigned["protocol_sha256"]
    computed = _canonical_sha256(unsigned)
    _require(declared == computed == expected_payload_sha256, "protocol payload SHA differs")
    training = protocol.get("training")
    _require(isinstance(training, Mapping), "protocol lacks training contract")
    training_expected = {
        "tss_enabled": False,
        "tss_requested_weight": 0.0,
        "tss_ratio_cap_applied": False,
        "tss_survival_target_constructed": False,
        "tss_survival_logits_consumed_by_loss": False,
        "tss_training_forward_computes_logits": True,
    }
    for field, value in training_expected.items():
        _require(training.get(field) == value, f"protocol training {field!r} differs")
    return computed, _validate_runtime_source_locks(protocol)


def validate_current_source(
    source_root: Path,
    dataset: str,
) -> dict[str, Any]:
    """Validate one frozen Current source without touching official data."""

    dataset = _require_dataset(dataset)
    lock = SOURCE_LOCKS[dataset]
    run_dir = _run_directory(source_root, dataset)
    _require(not run_dir.is_symlink() and run_dir.is_dir(), f"run directory differs: {run_dir}")
    checkpoint_path = run_dir / CHECKPOINT_RELATIVE
    summary_path = run_dir / SUMMARY_RELATIVE
    protocol_path = run_dir / PROTOCOL_RELATIVE
    checkpoint_content = _regular_file_bytes(checkpoint_path)
    summary, summary_content = _json_regular(summary_path, "summary")
    protocol, protocol_content = _json_regular(protocol_path, "protocol")
    checkpoint_sha = _sha256_bytes(checkpoint_content)
    summary_sha = _sha256_bytes(summary_content)
    protocol_file_sha = _sha256_bytes(protocol_content)
    _require(len(checkpoint_content) == lock["checkpoint_bytes"], "checkpoint byte count differs")
    _require(checkpoint_sha == lock["checkpoint_file_sha256"], "checkpoint file SHA differs")
    _require(summary_sha == lock["summary_file_sha256"], "summary file SHA differs")
    _require(protocol_file_sha == lock["protocol_file_sha256"], "protocol file SHA differs")
    protocol_payload_sha, runtime_sources = _validate_protocol(
        protocol,
        dataset=dataset,
        expected_payload_sha256=lock["protocol_payload_sha256"],
    )

    summary_expected = {
        "schema": SOURCE_SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": "final",
        "seed": TRAINING_SEED,
        "epochs": 1000,
        "test_selected": True,
        "selection_is_optimistic": True,
        "recipe": _recipe_identity(),
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "protocol_sha256": protocol_payload_sha,
    }
    for field, value in summary_expected.items():
        _require(summary.get(field) == value, f"summary {field!r} differs")
    checkpoint_binding = summary.get("checkpoints", {}).get(CHECKPOINT_ROLE)
    _require(isinstance(checkpoint_binding, Mapping), "summary lacks best_miou binding")
    _require(
        checkpoint_binding.get("sha256") == checkpoint_sha
        and checkpoint_binding.get("bytes") == len(checkpoint_content)
        and Path(str(checkpoint_binding.get("path"))).resolve() == checkpoint_path.resolve(),
        "summary checkpoint binding differs",
    )
    selection = summary.get(CHECKPOINT_ROLE)
    _require(isinstance(selection, Mapping), "summary lacks best_miou selection")
    _require(
        selection.get("epoch") == lock["epoch"]
        and Path(str(selection.get("path"))).resolve() == checkpoint_path.resolve(),
        "summary best_miou selection differs",
    )

    payload = torch.load(
        io.BytesIO(checkpoint_content),
        map_location="cpu",
        weights_only=False,
    )
    _require(isinstance(payload, Mapping), "checkpoint payload must be a mapping")
    checkpoint_expected = {
        "schema": SOURCE_SCHEMA,
        "epoch": lock["epoch"],
        "dataset": dataset,
        "method": "final",
        "seed": TRAINING_SEED,
        "checkpoint_role": CHECKPOINT_ROLE,
        "selection_source": f"test_{dataset}",
        "test_selected": True,
        "selection_is_optimistic": True,
        "recipe": _recipe_identity(),
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "protocol_sha256": protocol_payload_sha,
    }
    for field, value in checkpoint_expected.items():
        _require(payload.get(field) == value, f"checkpoint {field!r} differs")
    metrics = payload.get("test_metrics")
    _require(isinstance(metrics, Mapping) and bool(metrics), "checkpoint lacks historical metrics")
    summary_metrics = selection.get("metrics")
    _require(
        isinstance(summary_metrics, Mapping)
        and _historical_metrics_equal(summary_metrics, metrics),
        "summary/checkpoint historical metrics differ",
    )
    _validate_model_metadata(payload.get("model_metadata"), dataset=dataset)
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping), "checkpoint lacks state_dict")
    _require(len(state) == TRAINING_STATE_KEY_COUNT, "training state-key count differs")
    _require_finite_tensor_mapping(state, label="training state")
    tss_audit = require_exact_zero_tss_state(state)
    qfg_keys = {key for key in state if key.startswith(QFG_STATE_PREFIX)}
    _require(qfg_keys == set(QFG_STATE_KEYS), "training QFG state keys differ")
    training_state_sha = registry.state_dict_sha256(state)
    _require(training_state_sha == lock["training_state_sha256"], "training state SHA differs")
    inference_state = registry.export_final_inference_state(state, to_cpu=True)
    _require(len(inference_state) == INFERENCE_STATE_KEY_COUNT, "inference state-key count differs")
    _require_finite_tensor_mapping(inference_state, label="inference state")
    inference_state_sha = registry.state_dict_sha256(inference_state)
    _require(inference_state_sha == lock["inference_state_sha256"], "inference state SHA differs")
    for key in QFG_STATE_KEYS:
        _require(torch.equal(inference_state[key], state[key]), f"QFG state {key!r} changed")
    identity_audit = assert_synthetic_six_output_bitwise_identity(
        state,
        dataset=dataset,
    )
    architecture = identity_audit.pop("architecture")
    return {
        "dataset": dataset,
        "training_state_dict": state,
        "inference_state_dict": inference_state,
        "binding": {
            "run_directory": str(run_dir.resolve()),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_file_sha256": checkpoint_sha,
            "checkpoint_bytes": len(checkpoint_content),
            "checkpoint_role": CHECKPOINT_ROLE,
            "epoch": lock["epoch"],
            "summary_path": str(summary_path.resolve()),
            "summary_file_sha256": summary_sha,
            "protocol_path": str(protocol_path.resolve()),
            "protocol_file_sha256": protocol_file_sha,
            "protocol_payload_sha256": protocol_payload_sha,
            "training_state_hash_algorithm": STATE_HASH_ALGORITHM,
            "training_state_sha256": training_state_sha,
            "inference_state_hash_algorithm": STATE_HASH_ALGORITHM,
            "inference_state_sha256": inference_state_sha,
            "state_hash_contract": _state_hash_contract(),
            "runtime_source_sha256": runtime_sources,
            "historical_test_metrics_canonical_sha256": _canonical_sha256(metrics),
            "historical_selection": {
                "selection_source": f"test_{dataset}",
                "test_selected": True,
                "selection_is_optimistic": True,
                "operational_checkpoint_only": True,
                "unbiased_test_claim_allowed": False,
                "metrics_recomputed_during_export": False,
            },
            "export_operation_official_access": _official_false_payload(),
            "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
            "tss_audit": tss_audit,
            "synthetic_identity": identity_audit,
            "architecture": architecture,
        },
    }


def _package_from_source(source: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _require_dataset(str(source.get("dataset")))
    binding = source.get("binding")
    state = source.get("inference_state_dict")
    _require(isinstance(binding, Mapping), "source context lacks binding")
    _require(isinstance(state, Mapping), "source context lacks inference state")
    return {
        "schema": PACKAGE_SCHEMA,
        "model_name": MODEL_NAME,
        "model_public_id": MODEL_PUBLIC_ID,
        "dataset": dataset,
        "seed": TRAINING_SEED,
        "architecture": copy.deepcopy(binding["architecture"]),
        "deployment_implementation": _deployment_implementation_binding(),
        "source": copy.deepcopy(dict(binding)),
        "state_dict": dict(state),
        "inference": {
            "graph": "sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa",
            "loader": (
                "experiments.export_three_dataset_current_to_inference."
                "load_exported_current_model"
            ),
            "state_key_count": INFERENCE_STATE_KEY_COUNT,
            "parameter_count": INFERENCE_PARAMETER_COUNT,
            "state_hash_algorithm": STATE_HASH_ALGORITHM,
            "state_hash_contract": _state_hash_contract(),
            "state_sha256": binding["inference_state_sha256"],
            "strict_load": True,
            "mode": "test",
            "output": "sigmoid(out)",
        },
        "tss": {
            "part_of_final_model": False,
            "training_only_source_state_key_count": len(SURVIVAL_STATE_KEYS),
            "training_only_source_parameter_count": TSS_PARAMETER_COUNT,
            "source_tensors_exact_zero": True,
            "removed_state_keys": list(SURVIVAL_STATE_KEYS),
            "target_survival_registered": False,
        },
        "qfg": {
            "required_for_inference": True,
            "state_key_count": len(QFG_STATE_KEYS),
            "preserved_state_keys": list(QFG_STATE_KEYS),
        },
        "historical_selection": copy.deepcopy(binding["historical_selection"]),
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
        "synthetic_identity": copy.deepcopy(binding["synthetic_identity"]),
    }


def validate_exported_current_package(
    package_path: Path,
    *,
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    """Re-open, strict-load, and validate one TSS-free package."""

    content = _regular_file_bytes(package_path)
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), "export package must be a mapping")
    _require(payload.get("schema") == PACKAGE_SCHEMA, "export package schema differs")
    dataset = _require_dataset(str(payload.get("dataset")))
    if expected_dataset is not None:
        _require(dataset == _require_dataset(expected_dataset), "export dataset differs")
    lock = SOURCE_LOCKS[dataset]
    _require(payload.get("model_name") == MODEL_NAME, "export model name differs")
    _require(payload.get("model_public_id") == MODEL_PUBLIC_ID, "export public model ID differs")
    _require(payload.get("seed") == TRAINING_SEED, "export seed differs")
    _require(
        payload.get("deployment_implementation") == _deployment_implementation_binding(),
        "deployment implementation binding differs",
    )
    _require(
        payload.get("official_boundary_scope") == OFFICIAL_BOUNDARY_SCOPE,
        "export official boundary scope differs",
    )
    source = payload.get("source")
    _require(isinstance(source, Mapping), "export package lacks source binding")
    source_expected = {
        "checkpoint_file_sha256": lock["checkpoint_file_sha256"],
        "checkpoint_bytes": lock["checkpoint_bytes"],
        "checkpoint_role": CHECKPOINT_ROLE,
        "epoch": lock["epoch"],
        "summary_file_sha256": lock["summary_file_sha256"],
        "protocol_file_sha256": lock["protocol_file_sha256"],
        "protocol_payload_sha256": lock["protocol_payload_sha256"],
        "training_state_sha256": lock["training_state_sha256"],
        "inference_state_sha256": lock["inference_state_sha256"],
    }
    for field, value in source_expected.items():
        _require(source.get(field) == value, f"export source {field!r} differs")
    _require(
        source.get("training_state_hash_algorithm") == STATE_HASH_ALGORITHM
        and source.get("inference_state_hash_algorithm") == STATE_HASH_ALGORITHM
        and source.get("state_hash_contract") == _state_hash_contract(),
        "export source state-hash contract differs",
    )
    runtime_sources = source.get("runtime_source_sha256")
    _require(
        isinstance(runtime_sources, Mapping)
        and bool(runtime_sources)
        and all(type(key) is str and _is_sha256(value) for key, value in runtime_sources.items()),
        "export source runtime-source binding differs",
    )
    _require(
        source.get("export_operation_official_access")
        == _official_false_payload(),
        "export source official-access flags differ",
    )
    _require(
        source.get("official_boundary_scope") == OFFICIAL_BOUNDARY_SCOPE,
        "export source official boundary scope differs",
    )
    historical = payload.get("historical_selection")
    _require(isinstance(historical, Mapping), "export lacks historical selection")
    _require(
        historical.get("selection_source") == f"test_{dataset}"
        and historical.get("test_selected") is True
        and historical.get("selection_is_optimistic") is True
        and historical.get("operational_checkpoint_only") is True
        and historical.get("unbiased_test_claim_allowed") is False
        and historical.get("metrics_recomputed_during_export") is False,
        "historical selection disclosure differs",
    )
    _require(source.get("historical_selection") == historical, "source/top-level historical selection differs")
    _require(
        payload.get("export_operation_official_access") == _official_false_payload(),
        "export official-access flags differ",
    )
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping), "export package lacks state_dict")
    _require(len(state) == INFERENCE_STATE_KEY_COUNT, "export state-key count differs")
    _require_finite_tensor_mapping(state, label="export state")
    _require(all(value.device.type == "cpu" for value in state.values()), "export state is not CPU")
    _require(
        not any(key.startswith(SURVIVAL_STATE_PREFIX) for key in state),
        "export state retains TSS",
    )
    _require(
        {key for key in state if key.startswith(QFG_STATE_PREFIX)} == set(QFG_STATE_KEYS),
        "export QFG state keys differ",
    )
    state_sha = registry.state_dict_sha256(state)
    _require(state_sha == lock["inference_state_sha256"], "export state SHA differs")
    model, _ = registry.build_paper_model(
        "final",
        dataset,
        seed=TRAINING_SEED,
        training=False,
    )
    incompatible = model.load_state_dict(state, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "export strict load returned incompatible keys",
    )
    _require(type(model) is TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet, "export model class differs")
    _require(not hasattr(model, "target_survival"), "export model retains TSS")
    _require(model.training is False and model.mode == "test", "export model mode differs")
    _require(
        sum(parameter.numel() for parameter in model.parameters()) == INFERENCE_PARAMETER_COUNT,
        "export parameter count differs",
    )
    architecture = _architecture_contract(model)
    _require(payload.get("architecture") == architecture, "export architecture contract differs")
    _require(source.get("architecture") == architecture, "source architecture contract differs")
    inference = payload.get("inference")
    _require(isinstance(inference, Mapping), "export lacks inference contract")
    inference_expected = {
        "graph": MODEL_PUBLIC_ID,
        "loader": (
            "experiments.export_three_dataset_current_to_inference."
            "load_exported_current_model"
        ),
        "state_key_count": INFERENCE_STATE_KEY_COUNT,
        "parameter_count": INFERENCE_PARAMETER_COUNT,
        "state_hash_algorithm": STATE_HASH_ALGORITHM,
        "state_hash_contract": _state_hash_contract(),
        "state_sha256": state_sha,
        "strict_load": True,
        "mode": "test",
        "output": "sigmoid(out)",
    }
    for field, value in inference_expected.items():
        _require(inference.get(field) == value, f"inference field {field!r} differs")
    tss = payload.get("tss")
    _require(
        isinstance(tss, Mapping)
        and tss.get("part_of_final_model") is False
        and tss.get("training_only_source_state_key_count")
        == len(SURVIVAL_STATE_KEYS)
        and tss.get("training_only_source_parameter_count")
        == TSS_PARAMETER_COUNT
        and tss.get("source_tensors_exact_zero") is True
        and tss.get("target_survival_registered") is False
        and set(tss.get("removed_state_keys", ())) == set(SURVIVAL_STATE_KEYS),
        "export TSS declaration differs",
    )
    source_tss = source.get("tss_audit")
    _require(
        isinstance(source_tss, Mapping)
        and source_tss.get("all_exact_zero") is True
        and source_tss.get("parameter_count") == TSS_PARAMETER_COUNT
        and source_tss.get("state_key_count") == len(SURVIVAL_STATE_KEYS)
        and source_tss.get("state_keys") == list(SURVIVAL_STATE_KEYS),
        "source TSS audit differs",
    )
    qfg = payload.get("qfg")
    _require(
        isinstance(qfg, Mapping)
        and qfg.get("required_for_inference") is True
        and qfg.get("state_key_count") == len(QFG_STATE_KEYS)
        and set(qfg.get("preserved_state_keys", ())) == set(QFG_STATE_KEYS),
        "export QFG declaration differs",
    )
    identity = payload.get("synthetic_identity")
    _require(
        isinstance(identity, Mapping)
        and identity.get("six_output_count") == 6
        and identity.get("all_six_bitwise_equal") is True
        and identity.get("maximum_absolute_difference") == 0.0
        and identity.get("deployed_single_output_equals_sixth") is True
        and identity.get("deployed_output") == "sigmoid(out)",
        "export synthetic identity differs",
    )
    _require(source.get("synthetic_identity") == identity, "source/top-level synthetic identity differs")
    replay = _replay_inference_synthetic_identity(model)
    for field, value in replay.items():
        _require(identity.get(field) == value, f"synthetic replay field {field!r} differs")
    return {
        "schema": PACKAGE_SCHEMA,
        "dataset": dataset,
        "epoch": lock["epoch"],
        "path": str(Path(package_path).resolve()),
        "bytes": len(content),
        "file_sha256": _sha256_bytes(content),
        "state_key_count": INFERENCE_STATE_KEY_COUNT,
        "parameter_count": INFERENCE_PARAMETER_COUNT,
        "state_sha256": state_sha,
        "state_hash_algorithm": STATE_HASH_ALGORITHM,
        "architecture": architecture,
        "deployment_implementation": _deployment_implementation_binding(),
        "strict_load": True,
        "tss_absent": True,
        "qfg_preserved": True,
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }


def load_exported_current_model(
    package_path: Path,
    *,
    expected_dataset: str | None = None,
) -> tuple[TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet, dict[str, Any]]:
    """Strictly load one already-validated 564-key deployment package."""

    binding = validate_exported_current_package(
        package_path,
        expected_dataset=expected_dataset,
    )
    content = _regular_file_bytes(package_path)
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    dataset = binding["dataset"]
    model, metadata = registry.build_paper_model(
        "final",
        dataset,
        seed=TRAINING_SEED,
        training=False,
    )
    incompatible = model.load_state_dict(state, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "deployment loader strict load returned incompatible keys",
    )
    model.eval()
    model.mode = "test"
    ready = dict(metadata)
    ready.update(
        {
            "deployment_package": binding,
            "strict_load": True,
            "target_survival_registered": False,
            "mode": "test",
            "output": "sigmoid(out)",
        }
    )
    return model, ready


def preflight_all_sources(source_root: Path = DEFAULT_SOURCE_ROOT) -> dict[str, Any]:
    contexts = {
        dataset: validate_current_source(source_root, dataset) for dataset in DATASETS
    }
    architecture = contexts[DATASETS[0]]["binding"]["architecture"]
    _require(
        all(contexts[dataset]["binding"]["architecture"] == architecture for dataset in DATASETS),
        "dataset architecture contracts differ",
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "preflight_complete",
        "model_name": MODEL_NAME,
        "datasets": list(DATASETS),
        "contexts": contexts,
        "architecture": copy.deepcopy(architecture),
        "deployment_implementation": _deployment_implementation_binding(),
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }


def _manifest_semantic_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_semantic_sha256", None)
    return _canonical_sha256(unsigned)


def _validate_uncommitted_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = Path(output_root) / "manifest.json"
    manifest, manifest_content = _json_regular(manifest_path, "deployment manifest")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema differs")
    _require(manifest.get("status") == "complete", "manifest status differs")
    _require(manifest.get("model_name") == MODEL_NAME, "manifest model name differs")
    _require(manifest.get("model_public_id") == MODEL_PUBLIC_ID, "manifest public model ID differs")
    _require(manifest.get("datasets") == list(DATASETS), "manifest dataset order differs")
    _require(
        manifest.get("export_operation_official_access") == _official_false_payload(),
        "manifest official-access flags differ",
    )
    _require(
        manifest.get("official_boundary_scope") == OFFICIAL_BOUNDARY_SCOPE,
        "manifest official boundary scope differs",
    )
    scalar_expected = {
        "training_seed": TRAINING_SEED,
        "source_schema": SOURCE_SCHEMA,
        "source_checkpoint_role": CHECKPOINT_ROLE,
        "training_state_key_count": TRAINING_STATE_KEY_COUNT,
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "training_parameter_count": TRAINING_PARAMETER_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
        "tss_part_of_final_model": False,
        "state_hash_contract": _state_hash_contract(),
        "deployment_implementation": _deployment_implementation_binding(),
    }
    for field, value in scalar_expected.items():
        _require(manifest.get(field) == value, f"manifest {field!r} differs")
    _require(
        manifest.get("tss_removed_state_keys") == list(SURVIVAL_STATE_KEYS),
        "manifest removed TSS keys differ",
    )
    _require(
        manifest.get("qfg_preserved_state_keys") == list(QFG_STATE_KEYS),
        "manifest preserved QFG keys differ",
    )
    _require(
        manifest.get("historical_selection_disclosure")
        == {
            "all_sources_test_selected": True,
            "all_sources_selection_is_optimistic": True,
            "operational_checkpoints_only": True,
            "unbiased_test_claim_allowed": False,
            "official_metrics_recomputed_during_export": False,
        },
        "manifest historical selection disclosure differs",
    )
    _require(
        manifest.get("publication_contract")
        == {
            "write_once": True,
            "completed_bundle_is_idempotently_revalidated": True,
            "partial_bundle_is_never_overwritten": True,
            "root_committed_written_last": True,
        },
        "manifest publication contract differs",
    )
    semantic = _manifest_semantic_sha256(manifest)
    _require(manifest.get("manifest_semantic_sha256") == semantic, "manifest semantic SHA differs")
    packages = manifest.get("packages")
    _require(isinstance(packages, list) and len(packages) == len(DATASETS), "manifest packages differ")
    observed: dict[str, dict[str, Any]] = {}
    for entry in packages:
        _require(isinstance(entry, Mapping), "manifest package entry differs")
        dataset = _require_dataset(str(entry.get("dataset")))
        relative = Path(str(entry.get("relative_path")))
        _require(not relative.is_absolute() and ".." not in relative.parts, "package path escapes bundle")
        package_path = (Path(output_root) / relative).resolve(strict=True)
        _require(package_path.is_relative_to(Path(output_root).resolve()), "package path escapes root")
        validated = validate_exported_current_package(package_path, expected_dataset=dataset)
        for field in ("bytes", "file_sha256", "state_sha256", "epoch"):
            _require(entry.get(field) == validated[field], f"manifest package {field!r} differs")
        _require(
            entry.get("state_hash_algorithm") == STATE_HASH_ALGORITHM
            and entry.get("state_key_count") == INFERENCE_STATE_KEY_COUNT
            and entry.get("parameter_count") == INFERENCE_PARAMETER_COUNT
            and entry.get("architecture_manifest_canonical_json_sha256")
            == ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256
            and entry.get("sorted_state_key_set_canonical_json_sha256")
            == SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256,
            "manifest package semantic contract differs",
        )
        observed[dataset] = validated
    _require(set(observed) == set(DATASETS), "manifest package dataset set differs")
    common_architecture = observed[DATASETS[0]]["architecture"]
    _require(
        all(observed[dataset]["architecture"] == common_architecture for dataset in DATASETS)
        and manifest.get("architecture") == common_architecture,
        "manifest/package architecture contracts differ",
    )
    package_dir = Path(output_root) / "packages"
    _require(not package_dir.is_symlink() and package_dir.is_dir(), "package directory differs")
    expected_names = {PACKAGE_FILENAMES[dataset] for dataset in DATASETS}
    _require({path.name for path in package_dir.iterdir()} == expected_names, "package file set differs")
    return {
        "manifest": manifest,
        "manifest_file_sha256": _sha256_bytes(manifest_content),
        "manifest_semantic_sha256": semantic,
        "packages": observed,
    }


def validate_committed_bundle(
    output_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a completed bundle without mutating it."""

    root = Path(output_root).resolve(strict=True)
    _require(not root.is_symlink() and root.is_dir(), "bundle root differs")
    committed, committed_content = _json_regular(root / "COMMITTED", "COMMITTED")
    _require(committed.get("schema") == COMMITTED_SCHEMA, "COMMITTED schema differs")
    _require(committed.get("status") == "complete", "COMMITTED status differs")
    _require(
        committed.get("export_operation_official_access") == _official_false_payload(),
        "COMMITTED official-access flags differ",
    )
    _require(
        committed.get("official_boundary_scope") == OFFICIAL_BOUNDARY_SCOPE,
        "COMMITTED official boundary scope differs",
    )
    validation = _validate_uncommitted_manifest(root)
    expected_committed = {
        "schema": COMMITTED_SCHEMA,
        "status": "complete",
        "manifest_path": "manifest.json",
        "manifest_file_sha256": validation["manifest_file_sha256"],
        "manifest_semantic_sha256": validation["manifest_semantic_sha256"],
        "package_count": len(DATASETS),
        "datasets": list(DATASETS),
        "root_committed_written_last": True,
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }
    _require(dict(committed) == expected_committed, "COMMITTED contract differs")
    expected_root_names = {"packages", "manifest.json", "COMMITTED"}
    _require({path.name for path in root.iterdir()} == expected_root_names, "bundle root file set differs")
    if source_root is not None:
        sources = preflight_all_sources(source_root)["contexts"]
        for dataset in DATASETS:
            binding = sources[dataset]["binding"]
            package = validation["packages"][dataset]
            _require(
                package["state_sha256"] == binding["inference_state_sha256"],
                f"bundle/source state differs for {dataset}",
            )
    return {
        "schema": COMMITTED_SCHEMA,
        "status": "complete",
        "output_root": str(root),
        "manifest_file_sha256": validation["manifest_file_sha256"],
        "manifest_semantic_sha256": validation["manifest_semantic_sha256"],
        "committed_file_sha256": _sha256_bytes(committed_content),
        "packages": validation["packages"],
        "idempotent_existing_bundle": True,
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }


def export_current_bundle(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Publish all three packages; a root COMMITTED marker is written last."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        if output_root.is_dir() and not output_root.is_symlink() and (output_root / "COMMITTED").is_file():
            return validate_committed_bundle(output_root, source_root=source_root)
        raise FileExistsError(f"incomplete or foreign write-once bundle exists: {output_root}")

    # Validate every source and run every synthetic identity proof before
    # reserving the output path, so source failures cannot leave a partial root.
    preflight = preflight_all_sources(source_root)
    contexts = preflight["contexts"]
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise NotADirectoryError(output_root.parent)
    output_root.mkdir(exist_ok=False)
    _fsync_directory(output_root.parent)
    packages_dir = output_root / "packages"
    packages_dir.mkdir(exist_ok=False)

    package_entries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        destination = packages_dir / PACKAGE_FILENAMES[dataset]
        package = _package_from_source(contexts[dataset])
        _write_bytes_once(destination, _torch_bytes(package))
        validated = validate_exported_current_package(
            destination,
            expected_dataset=dataset,
        )
        package_entries.append(
            {
                "dataset": dataset,
                "epoch": validated["epoch"],
                "relative_path": destination.relative_to(output_root).as_posix(),
                "bytes": validated["bytes"],
                "file_sha256": validated["file_sha256"],
                "state_sha256": validated["state_sha256"],
                "state_hash_algorithm": STATE_HASH_ALGORITHM,
                "state_key_count": validated["state_key_count"],
                "parameter_count": validated["parameter_count"],
                "architecture_manifest_canonical_json_sha256": (
                    ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256
                ),
                "sorted_state_key_set_canonical_json_sha256": (
                    SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256
                ),
            }
        )

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "model_name": MODEL_NAME,
        "model_public_id": MODEL_PUBLIC_ID,
        "datasets": list(DATASETS),
        "training_seed": TRAINING_SEED,
        "source_schema": SOURCE_SCHEMA,
        "source_checkpoint_role": CHECKPOINT_ROLE,
        "training_state_key_count": TRAINING_STATE_KEY_COUNT,
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "training_parameter_count": TRAINING_PARAMETER_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
        "state_hash_contract": _state_hash_contract(),
        "architecture": copy.deepcopy(preflight["architecture"]),
        "deployment_implementation": copy.deepcopy(
            preflight["deployment_implementation"]
        ),
        "tss_part_of_final_model": False,
        "tss_removed_state_keys": list(SURVIVAL_STATE_KEYS),
        "qfg_preserved_state_keys": list(QFG_STATE_KEYS),
        "packages": package_entries,
        "historical_selection_disclosure": {
            "all_sources_test_selected": True,
            "all_sources_selection_is_optimistic": True,
            "operational_checkpoints_only": True,
            "unbiased_test_claim_allowed": False,
            "official_metrics_recomputed_during_export": False,
        },
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
        "publication_contract": {
            "write_once": True,
            "completed_bundle_is_idempotently_revalidated": True,
            "partial_bundle_is_never_overwritten": True,
            "root_committed_written_last": True,
        },
    }
    manifest["manifest_semantic_sha256"] = _manifest_semantic_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    _write_bytes_once(manifest_path, _json_bytes(manifest))

    # Full package/manifest validation must finish before COMMITTED exists.
    uncommitted = _validate_uncommitted_manifest(output_root)
    committed = {
        "schema": COMMITTED_SCHEMA,
        "status": "complete",
        "manifest_path": "manifest.json",
        "manifest_file_sha256": uncommitted["manifest_file_sha256"],
        "manifest_semantic_sha256": uncommitted["manifest_semantic_sha256"],
        "package_count": len(DATASETS),
        "datasets": list(DATASETS),
        "root_committed_written_last": True,
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }
    # This is intentionally the final artifact write in the publication path.
    _write_bytes_once(output_root / "COMMITTED", _json_bytes(committed))
    completed = validate_committed_bundle(output_root)
    completed["idempotent_existing_bundle"] = False
    return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate all sources and synthetic identities without writing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_only:
        preflight = preflight_all_sources(args.source_root)
        public = {
            "schema": preflight["schema"],
            "status": preflight["status"],
            "model_name": preflight["model_name"],
            "datasets": preflight["datasets"],
            "architecture": preflight["architecture"],
            "deployment_implementation": preflight["deployment_implementation"],
            "sources": {
                dataset: preflight["contexts"][dataset]["binding"]
                for dataset in DATASETS
            },
            "export_operation_official_access": _official_false_payload(),
            "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
        }
        print(json.dumps(public, ensure_ascii=False, sort_keys=True), flush=True)
        return
    result = export_current_bundle(args.source_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


__all__ = [
    "CHECKPOINT_ROLE",
    "COMMITTED_SCHEMA",
    "DATASETS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256",
    "INFERENCE_PARAMETER_COUNT",
    "INFERENCE_STATE_KEY_COUNT",
    "IMPLEMENTATION_CLASS",
    "MANIFEST_SCHEMA",
    "MODEL_NAME",
    "MODEL_PUBLIC_ID",
    "OFFICIAL_BOUNDARY_SCOPE",
    "OFFICIAL_FALSE_FLAGS",
    "PACKAGE_SCHEMA",
    "SOURCE_LOCKS",
    "SOURCE_SCHEMA",
    "SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256",
    "STATE_HASH_ALGORITHM",
    "TRAINING_PARAMETER_COUNT",
    "TRAINING_STATE_KEY_COUNT",
    "assert_synthetic_six_output_bitwise_identity",
    "export_current_bundle",
    "load_exported_current_model",
    "main",
    "parse_args",
    "preflight_all_sources",
    "require_exact_zero_tss_state",
    "validate_committed_bundle",
    "validate_current_source",
    "validate_exported_current_package",
]


if __name__ == "__main__":
    main()
