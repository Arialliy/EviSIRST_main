#!/usr/bin/env python3
"""Run the frozen paired V3.2/V3.3 train-only CUDA resource benchmark.

The benchmark is deliberately not an evaluation or a speed-promotion gate.  It
uses sixteen pre-registered IRSTD-1K training batches for each fresh model:
four warm-up steps and twelve measured complete FP32 training steps.  The two
models consume the same materialized CPU tensors in the same order.  Timing
and V3.3/V3.2 ratios are warning-only diagnostics; the hard gate covers only
identity, input equality, finiteness, measurement completeness, OOM, and CUDA
capacity.

No validation/test dataset, evaluator, checkpoint, or model-selection path is
present in this tool.  Trained states are discarded at process exit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_ENTRY_SOURCE = Path(__file__)
if _ENTRY_SOURCE.is_symlink() or not _ENTRY_SOURCE.is_file():
    raise RuntimeError("resource benchmark entry point must be a regular file")
PROJECT_ROOT = _ENTRY_SOURCE.resolve(strict=True).parents[1]
if _ENTRY_SOURCE.resolve(strict=True) != PROJECT_ROOT / "tools" / _ENTRY_SOURCE.name:
    raise RuntimeError("resource benchmark entry point resolves away from tools")
_PROJECT_ROOT_TEXT = str(PROJECT_ROOT)
sys.path[:] = [
    _PROJECT_ROOT_TEXT,
    *[
        entry
        for entry in sys.path
        if entry not in ("", _PROJECT_ROOT_TEXT)
    ],
]

FROZEN_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", FROZEN_CUBLAS_WORKSPACE_CONFIG)

import torch
import torch.nn as nn

from experiments import sbsc_v33_contracts as contracts
from experiments import sctransnet_sbsc_v32 as core_v32
from experiments import sctransnet_sbsc_v33 as core_v33
from experiments import three_dataset_v2_protocol as source_protocol
from experiments.evisirst_data import EviSIRSTTrainDataset
from experiments.four_dataset_models_seed42_v1 import state_dict_sha256
from train import configure_determinism, require_device


RULES_PATH = (
    PROJECT_ROOT / "experiments" / "sbsc_v33_resource_benchmark_rules.json"
)
EXPECTED_RULES_SHA256 = (
    "110f26c67afc3b873943c62210082a285362be33d29328707c7aeacb40a3622c"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "sbsc_v33_preflight"
    / "paired_resource_benchmark.json"
)
RULES_SCHEMA = "sctransnet_sbsc_v33/resource_benchmark_rules/v1"
REPORT_SCHEMA = "sctransnet_sbsc_v33/resource_benchmark_report/v1"
MODEL_RECORD_SCHEMA = "sctransnet_sbsc_v33/resource_model_measurement/v1"
DATASET = "IRSTD-1K"
DATA_ROLE = "train"
ARCHITECTURE_SEED = 42
RUN_SEED = 42
SAMPLE_COUNT = 800
BATCH_SIZE = 16
PATCH_SIZE = 256
DATASET_EPOCH = 1
WARMUP_STEPS = 4
MEASURED_STEPS = 12
TOTAL_STEPS = WARMUP_STEPS + MEASURED_STEPS
MEASURED_SAMPLES = BATCH_SIZE * MEASURED_STEPS
MODEL_ORDER = ("sbsc_v32", "sbsc_v33_third")
GRADIENT_MODE = "live"
BALANCE_MODE = "one_third_two_thirds"
BATCH_DOMAIN = "sctransnet_sbsc_v33_resource_benchmark/v1"
EXPECTED_BATCH_PLAN_SHA256 = (
    "35120f3888b178c6df619e5eee243468ee319eef63cf0feac19c6010b0bceb7d"
)
CANARY_REPORT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "sbsc_v33_third"
    / "canary_v2"
    / "IRSTD-1K"
    / "report.json"
)
CANARY_MANIFEST_PATH = CANARY_REPORT_PATH.with_name("manifest.json")
CANARY_RULES_PATH = PROJECT_ROOT / "experiments" / "sbsc_v33_canary_rules_v2.json"
CANARY_REPORT_SCHEMA = "sctransnet_sbsc_v33/canary_report/v2"
CANARY_MANIFEST_SCHEMA = "sctransnet_sbsc_v33/canary_manifest/v2"
CANARY_RULES_SCHEMA = "sctransnet_sbsc_v33/canary_rules/v2"
EXPECTED_CANARY_RULES_SHA256 = (
    "b2fc34a711cd8a6fc8ce0e69b4015b8a300c040acaef37d1922354627ae0f2ba"
)
EXPECTED_CANARY_REPORT_SHA256 = (
    "5226e11896c241cd235c8e1689a4c5f7678b86a143823a1fd4c6bf2efc04a34c"
)
EXPECTED_CANARY_MANIFEST_SHA256 = (
    "c986f19557415b6808802cd3f7608a06de70004035b8bb6748032930e80c9810"
)
EXPECTED_CANARY_METHOD_CONFIG_SHA256 = (
    "73e86c18ba5324240513916d30bad7e50bd448b7686f175dde79d4694665f7ab"
)
EXPECTED_CANARY_SOURCE_MANIFEST_SHA256 = (
    "3bec27c79f433e8e0ce58ad7fc4f1628ab908743ac4383e552ae5d7b50418bd2"
)
HARD_GATE_ORDER = (
    "identity",
    "same_input",
    "finite",
    "complete_measurement",
    "no_oom",
    "peak_within_device",
)
ARTIFACT_BOUNDARIES = {
    "validation_dataset_constructed": False,
    "validation_loader_constructed": False,
    "test_dataset_constructed": False,
    "test_loader_constructed": False,
    "evaluation_called": False,
    "performance_metric_computed": False,
    "model_selection": None,
    "checkpoint_written": False,
    "model_state_serialized": False,
    "formal_weights_reusable": False,
}


class ResourceBenchmarkError(RuntimeError):
    """A fail-closed paired resource benchmark contract violation."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def _fail(category: str, message: str) -> None:
    raise ResourceBenchmarkError(category, message)


def _lower_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("identity", f"{name} must be a lowercase SHA-256")
    return value


def _is_lower_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_positive_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("finite", f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        _fail("finite", f"{name} must be finite and valid")
    return normalized


def _repository_file(relative: str, *, name: str) -> Path:
    try:
        return contracts.require_repository_relative_regular_file(relative)
    except Exception as exc:
        raise ResourceBenchmarkError(
            "identity", f"{name} is not a canonical repository file"
        ) from exc


def expected_batch_plan() -> list[list[int]]:
    """Return the version-independent SHA-256 pre-registered batch order."""

    def ordering_key(index: int) -> tuple[bytes, int]:
        text = f"{BATCH_DOMAIN}|{DATASET}|{RUN_SEED}|{index}"
        return hashlib.sha256(text.encode("utf-8")).digest(), index

    selected = sorted(range(SAMPLE_COUNT), key=ordering_key)[: BATCH_SIZE * TOTAL_STEPS]
    batches = [
        selected[offset : offset + BATCH_SIZE]
        for offset in range(0, len(selected), BATCH_SIZE)
    ]
    if (
        len(batches) != TOTAL_STEPS
        or any(len(batch) != BATCH_SIZE for batch in batches)
        or len({index for batch in batches for index in batch})
        != BATCH_SIZE * TOTAL_STEPS
        or contracts.canonical_sha256(batches) != EXPECTED_BATCH_PLAN_SHA256
    ):
        _fail("identity", "pre-registered batch plan differs")
    return batches


def _model_specs() -> list[dict[str, Any]]:
    return [
        {
            "order": 0,
            "role": "original_v3_2",
            "method": "sbsc_v32",
            "model": core_v32.SBSC_V32_MODEL_NAME,
            "model_schema": core_v32.SBSC_V32_SCHEMA,
            "builder": (
                "experiments.sctransnet_sbsc_v32."
                "build_sctransnet_sbsc_v32_method"
            ),
            "loss_schema": core_v32.SBSC_V32_LOSS_SCHEMA,
            "router_value_gradient_mode": None,
            "balance_mode": None,
            "state_key_count": core_v32.EXPECTED_SBSC_V32_STATE_KEY_COUNT,
            "parameter_count": core_v32.EXPECTED_SBSC_V32_PARAMETER_COUNT,
        },
        {
            "order": 1,
            "role": "candidate_v3_3",
            "method": "sbsc_v33_third",
            "model": core_v33.SBSC_V33_MODEL_NAME,
            "model_schema": core_v33.SBSC_V33_SCHEMA,
            "builder": (
                "experiments.sctransnet_sbsc_v33."
                "build_sctransnet_sbsc_v33_method"
            ),
            "loss_schema": core_v33.SBSC_V33_LOSS_SCHEMA,
            "router_value_gradient_mode": GRADIENT_MODE,
            "balance_mode": BALANCE_MODE,
            "state_key_count": core_v33.EXPECTED_SBSC_V33_STATE_KEY_COUNT,
            "parameter_count": core_v33.EXPECTED_SBSC_V33_PARAMETER_COUNT,
        },
    ]


def load_frozen_rules() -> dict[str, Any]:
    if RULES_PATH.is_symlink() or not RULES_PATH.is_file():
        _fail("identity", "resource benchmark rules are unavailable")
    if contracts.sha256_file(RULES_PATH) != EXPECTED_RULES_SHA256:
        _fail("identity", "resource benchmark rules physical SHA-256 drifted")
    rules = contracts.load_strict_json(RULES_PATH)
    if set(rules) != {
        "schema",
        "status",
        "scope",
        "write_once",
        "identity",
        "models",
        "optimizer",
        "measurement",
        "authority_bindings",
        "source_manifest",
        "hard_gate_order",
        "hard_gates",
        "warning_diagnostics",
        "artifact_boundaries",
    }:
        _fail("identity", "resource benchmark rule field set differs")
    if (
        rules.get("schema") != RULES_SCHEMA
        or rules.get("status") != "frozen"
        or rules.get("write_once") is not True
        or rules.get("scope")
        != "paired_irstd_1k_train_only_fp32_training_step_resources"
        or tuple(rules.get("hard_gate_order", ())) != HARD_GATE_ORDER
        or rules.get("models") != _model_specs()
    ):
        _fail("identity", "resource benchmark rule identity differs")
    identity = rules.get("identity")
    if identity != {
        "dataset": DATASET,
        "data_role": DATA_ROLE,
        "split": "datasets/IRSTD-1K/img_idx/train_IRSTD-1K.txt",
        "sample_count": SAMPLE_COUNT,
        "architecture_seed": ARCHITECTURE_SEED,
        "run_seed": RUN_SEED,
        "batch_size": BATCH_SIZE,
        "patch_size": PATCH_SIZE,
        "workers": 0,
        "dataset_epoch": DATASET_EPOCH,
        "precision": "FP32",
        "autocast": False,
        "gradient_scaler": False,
        "fresh_initialization": True,
        "parent_checkpoint": None,
        "resume": False,
    }:
        _fail("identity", "resource benchmark data/run identity differs")
    measurement = rules.get("measurement")
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("model_order") != list(MODEL_ORDER)
        or measurement.get("batch_order_domain_separator") != BATCH_DOMAIN
        or measurement.get("batch_plan_sha256") != EXPECTED_BATCH_PLAN_SHA256
        or measurement.get("warmup_steps_per_model") != WARMUP_STEPS
        or measurement.get("measured_steps_per_model") != MEASURED_STEPS
        or measurement.get("total_steps_per_model") != TOTAL_STEPS
        or measurement.get("measured_samples_per_model") != MEASURED_SAMPLES
        or measurement.get("same_materialized_cpu_batches_for_both_models")
        is not True
    ):
        _fail("identity", "resource benchmark measurement contract differs")
    if rules.get("artifact_boundaries") != {
        **ARTIFACT_BOUNDARIES,
        "report_path": (
            "artifacts/sbsc_v33_preflight/paired_resource_benchmark.json"
        ),
    }:
        _fail("identity", "resource benchmark artifact boundary differs")
    warnings = rules.get("warning_diagnostics")
    if (
        not isinstance(warnings, Mapping)
        or warnings.get("affects_verdict") is not False
        or warnings.get("v33_faster_than_v32_required") is not False
        or warnings.get("ratio_thresholds") is not None
    ):
        _fail("identity", "resource ratio diagnostics became a hard gate")
    expected_batch_plan()
    return rules


def _source_paths() -> tuple[str, ...]:
    fixed = (
        "tools/benchmark_sbsc_v33_resources.py",
        "experiments/sbsc_v33_resource_benchmark_rules.json",
        "experiments/sbsc_v33_canary_rules_v2.json",
        "experiments/sbsc_v33_contracts.py",
        "experiments/sctransnet_sbsc_v31.py",
        "experiments/sctransnet_sbsc_v32.py",
        "experiments/sctransnet_sbsc_v33.py",
        "experiments/four_dataset_models_seed42_v1.py",
        "experiments/evisirst_data.py",
        "experiments/three_dataset_v2_protocol.py",
        "experiments/__init__.py",
        "train.py",
    )
    model_sources = tuple(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted((PROJECT_ROOT / "model").rglob("*.py"))
        if path.is_file() and not path.is_symlink()
    )
    paths = tuple(sorted({*fixed, *model_sources}))
    if len(paths) != len(fixed) + len(model_sources):
        _fail("identity", "resource benchmark source paths are not unique")
    return paths


def build_source_manifest() -> dict[str, Any]:
    return contracts.build_source_manifest(_source_paths())


def _validate_embedded_source_manifest(value: Any, *, name: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "sctransnet_sbsc_v33/source_manifest/v1"
        or not isinstance(value.get("files"), list)
    ):
        _fail("identity", f"{name} source manifest is malformed")
    records: list[dict[str, Any]] = []
    for position, record in enumerate(value["files"]):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            _fail("identity", f"{name} source record {position} differs")
        path = _repository_file(record["path"], name=f"{name} source")
        sha = _lower_sha256(record.get("sha256"), name=f"{name} source SHA")
        size = record.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or contracts.sha256_file(path) != sha
            or path.stat().st_size != size
        ):
            _fail("identity", f"{name} source record drifted")
        records.append(dict(record))
    aggregate = _lower_sha256(value.get("sha256"), name=f"{name} source aggregate")
    if contracts.canonical_sha256(records) != aggregate:
        _fail("identity", f"{name} source aggregate differs")
    return {"schema": value["schema"], "files": records, "sha256": aggregate}


def load_bound_evidence(rules: Mapping[str, Any]) -> dict[str, Any]:
    bindings = rules["authority_bindings"]
    baseline = contracts.load_baseline_authority_manifest()
    gradient = contracts.load_gradient_authorization()
    if bindings.get("baseline") != {
        "path": baseline["manifest_path"],
        "sha256": baseline["manifest_sha256"],
    }:
        _fail("identity", "baseline authority binding differs")
    if bindings.get("gradient") != {
        "path": gradient["authorization_path"],
        "sha256": gradient["authorization_sha256"],
        "authorized_router_value_gradient_mode": GRADIENT_MODE,
    } or gradient["authorized_router_value_gradient_mode"] != GRADIENT_MODE:
        _fail("identity", "gradient authorization binding differs")

    for path, name in (
        (CANARY_REPORT_PATH, "V2 canary report"),
        (CANARY_MANIFEST_PATH, "V2 canary manifest"),
        (CANARY_RULES_PATH, "V2 canary rules"),
    ):
        if path.is_symlink() or not path.is_file():
            _fail("identity", f"{name} is unavailable")
    canary_binding = bindings.get("v2_canary")
    expected_canary_binding = {
        "report_path": contracts.repository_relative_path(CANARY_REPORT_PATH),
        "report_schema": CANARY_REPORT_SCHEMA,
        "report_sha256": EXPECTED_CANARY_REPORT_SHA256,
        "manifest_path": contracts.repository_relative_path(CANARY_MANIFEST_PATH),
        "manifest_schema": CANARY_MANIFEST_SCHEMA,
        "manifest_sha256": EXPECTED_CANARY_MANIFEST_SHA256,
        "rules_path": contracts.repository_relative_path(CANARY_RULES_PATH),
        "rules_schema": CANARY_RULES_SCHEMA,
        "rules_sha256": EXPECTED_CANARY_RULES_SHA256,
        "method_config_sha256": EXPECTED_CANARY_METHOD_CONFIG_SHA256,
        "training_source_manifest_sha256": (
            EXPECTED_CANARY_SOURCE_MANIFEST_SHA256
        ),
        "required_status": "PASS",
        "required_hard_gate_pass": True,
        "required_verdict": "GO",
    }
    if canary_binding != expected_canary_binding:
        _fail("identity", "V2 canary rule binding differs")
    canary_report = contracts.load_strict_json(CANARY_REPORT_PATH)
    canary_rules = contracts.load_strict_json(CANARY_RULES_PATH)
    canary_manifest = contracts.load_strict_json(CANARY_MANIFEST_PATH)
    report_sha = contracts.sha256_file(CANARY_REPORT_PATH)
    rules_sha = contracts.sha256_file(CANARY_RULES_PATH)
    manifest_sha = contracts.sha256_file(CANARY_MANIFEST_PATH)
    if (
        report_sha != EXPECTED_CANARY_REPORT_SHA256
        or manifest_sha != EXPECTED_CANARY_MANIFEST_SHA256
        or rules_sha != EXPECTED_CANARY_RULES_SHA256
    ):
        _fail("identity", "V2 canary physical evidence drifted")
    hard_gate = canary_report.get("hard_gate")
    if (
        canary_report.get("schema") != CANARY_REPORT_SCHEMA
        or canary_report.get("status") != "PASS"
        or canary_report.get("hard_gate_pass") is not True
        or not isinstance(hard_gate, Mapping)
        or hard_gate.get("verdict") != "GO"
        or canary_report.get("rules_path")
        != contracts.repository_relative_path(CANARY_RULES_PATH)
        or canary_report.get("rules_sha256") != rules_sha
        or canary_report.get("manifest_path")
        != contracts.repository_relative_path(CANARY_MANIFEST_PATH)
        or canary_report.get("manifest_sha256") != manifest_sha
        or canary_report.get("baseline_authority_sha256")
        != baseline["manifest_sha256"]
        or canary_report.get("gradient_authorization_sha256")
        != gradient["authorization_sha256"]
        or canary_report.get("dataset") != DATASET
        or canary_report.get("data_role") != DATA_ROLE
        or canary_report.get("method") != "sbsc_v33_third"
        or canary_report.get("architecture_seed") != ARCHITECTURE_SEED
        or canary_report.get("run_seed") != RUN_SEED
        or canary_report.get("router_value_gradient_mode") != GRADIENT_MODE
        or canary_report.get("test_dataset_constructed") is not False
        or canary_report.get("test_loader_constructed") is not False
        or canary_report.get("checkpoint_written") is not False
    ):
        _fail("identity", "V2 canary is not the bound PASS evidence")
    if (
        canary_rules.get("schema") != CANARY_RULES_SCHEMA
        or rules_sha != EXPECTED_CANARY_RULES_SHA256
        or canary_rules.get("status") != "frozen"
        or canary_rules.get("write_once") is not True
        or canary_manifest.get("schema") != CANARY_MANIFEST_SCHEMA
        or canary_manifest.get("status") != "prepared"
        or canary_manifest.get("write_once") is not True
        or canary_manifest.get("rules_path")
        != contracts.repository_relative_path(CANARY_RULES_PATH)
        or canary_manifest.get("rules_sha256") != rules_sha
        or canary_manifest.get("method_config_sha256")
        != canary_report.get("method_config_sha256")
    ):
        _fail("identity", "V2 canary rules/manifest identity differs")
    manifest_authorities = canary_manifest.get("authority_bindings")
    if (
        not isinstance(manifest_authorities, Mapping)
        or not isinstance(manifest_authorities.get("baseline"), Mapping)
        or manifest_authorities["baseline"].get("path")
        != baseline["manifest_path"]
        or manifest_authorities["baseline"].get("sha256")
        != baseline["manifest_sha256"]
        or not isinstance(manifest_authorities.get("gradient"), Mapping)
        or manifest_authorities["gradient"].get("path")
        != gradient["authorization_path"]
        or manifest_authorities["gradient"].get("sha256")
        != gradient["authorization_sha256"]
        or manifest_authorities["gradient"].get(
            "authorized_router_value_gradient_mode"
        )
        != GRADIENT_MODE
    ):
        _fail("identity", "V2 canary authority chain differs")
    canary_source = _validate_embedded_source_manifest(
        canary_manifest.get("training_source_manifest"), name="V2 canary"
    )
    if (
        canary_report.get("training_source_manifest_sha256")
        != canary_source["sha256"]
        or canary_source["sha256"]
        != EXPECTED_CANARY_SOURCE_MANIFEST_SHA256
        or canary_report.get("method_config_sha256")
        != EXPECTED_CANARY_METHOD_CONFIG_SHA256
    ):
        _fail("identity", "V2 canary source binding differs")
    return {
        "baseline": baseline,
        "gradient": gradient,
        "v2_canary": {
            "report_path": contracts.repository_relative_path(CANARY_REPORT_PATH),
            "report_sha256": report_sha,
            "report_schema": CANARY_REPORT_SCHEMA,
            "manifest_path": contracts.repository_relative_path(CANARY_MANIFEST_PATH),
            "manifest_sha256": manifest_sha,
            "rules_path": contracts.repository_relative_path(CANARY_RULES_PATH),
            "rules_sha256": rules_sha,
            "method_config_sha256": _lower_sha256(
                canary_report.get("method_config_sha256"),
                name="V2 canary method config",
            ),
            "training_source_manifest_sha256": canary_source["sha256"],
            "status": "PASS",
            "hard_gate_pass": True,
            "verdict": "GO",
        },
    }


def _tensor_sha256(value: torch.Tensor, *, name: str) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        _fail("same_input", f"{name} must be a CPU tensor")
    return state_dict_sha256({name: value.detach().contiguous()})


def materialize_training_batches(
    dataset: EviSIRSTTrainDataset,
    batch_plan: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    if len(dataset) != SAMPLE_COUNT:
        _fail("identity", "IRSTD-1K train sample count differs")
    dataset.set_epoch(DATASET_EPOCH)
    batches: list[dict[str, Any]] = []
    for position, indices in enumerate(batch_plan):
        if len(indices) != BATCH_SIZE:
            _fail("same_input", "pre-registered batch size differs")
        samples = [dataset[int(index)] for index in indices]
        if any(not isinstance(sample, (tuple, list)) or len(sample) != 2 for sample in samples):
            _fail("same_input", "training dataset sample shape differs")
        images = torch.stack([sample[0] for sample in samples]).contiguous()
        masks = torch.stack([sample[1] for sample in samples]).contiguous()
        if (
            images.dtype is not torch.float32
            or masks.dtype is not torch.float32
            or tuple(images.shape) != (BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
            or tuple(masks.shape) != tuple(images.shape)
            or not bool(torch.isfinite(images).all())
            or not bool(torch.isfinite(masks).all())
            or bool(((masks < 0.0) | (masks > 1.0)).any())
        ):
            _fail("finite", "materialized FP32 training batch differs")
        image_sha = _tensor_sha256(images, name="images")
        mask_sha = _tensor_sha256(masks, name="masks")
        batches.append(
            {
                "position": position,
                "phase": "warmup" if position < WARMUP_STEPS else "measured",
                "indices": [int(index) for index in indices],
                "images": images,
                "masks": masks,
                "image_sha256": image_sha,
                "mask_sha256": mask_sha,
                "input_pair_sha256": contracts.canonical_sha256(
                    {"images": image_sha, "masks": mask_sha}
                ),
            }
        )
    return batches


def _freeze_inactive(model: nn.Module, method: str) -> tuple[str, ...]:
    names = (
        core_v32.structurally_inactive_parameter_names(model)
        if method == "sbsc_v32"
        else core_v33.structurally_inactive_parameter_names(model)
    )
    named = dict(model.named_parameters())
    if any(name not in named for name in names):
        _fail("identity", f"{method} structurally inactive parameters differ")
    for name in names:
        named[name].requires_grad_(False)
    return tuple(sorted(names))


def _build_model(
    spec: Mapping[str, Any], device: torch.device, optimizer_rules: Mapping[str, Any]
) -> dict[str, Any]:
    configure_determinism(ARCHITECTURE_SEED)
    method = str(spec["method"])
    if method == "sbsc_v32":
        model, metadata = core_v32.build_sctransnet_sbsc_v32_method(
            method="sbsc_v32",
            dataset=DATASET,
            architecture_seed=ARCHITECTURE_SEED,
            training=True,
        )
        validation = core_v32.validate_sctransnet_sbsc_v32(
            model, require_zero_gain=True
        )
    elif method == "sbsc_v33_third":
        model, metadata = core_v33.build_sctransnet_sbsc_v33_method(
            method="sbsc_v33_third",
            dataset=DATASET,
            architecture_seed=ARCHITECTURE_SEED,
            training=True,
            router_value_gradient_mode=GRADIENT_MODE,
        )
        validation = core_v33.validate_sctransnet_sbsc_v33(
            model,
            method="sbsc_v33_third",
            router_value_gradient_mode=GRADIENT_MODE,
            require_zero_gain=True,
        )
    else:
        _fail("identity", f"unsupported benchmark method: {method}")
    inactive_names = _freeze_inactive(model, method)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        len(model.state_dict()) != spec["state_key_count"]
        or parameter_count != spec["parameter_count"]
        or metadata.get("method") != method
        or metadata.get("dataset") != DATASET
        or metadata.get("architecture_seed") != ARCHITECTURE_SEED
        or metadata.get("test_split_accessed") is not False
        or any(
            parameter.is_floating_point() and parameter.dtype is not torch.float32
            for parameter in model.parameters()
        )
    ):
        _fail("identity", f"{method} model identity differs")
    active_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    active_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not active_parameters:
        _fail("identity", f"{method} active parameter set is empty")
    initial_state_sha = state_dict_sha256(model.state_dict())
    shared_state_sha = _lower_sha256(
        metadata.get("shared_state_sha256"), name=f"{method} shared state"
    )
    model.to(device)
    model.train()
    model.mode = "train"
    optimizer = torch.optim.Adam(
        active_parameters,
        lr=float(optimizer_rules["base_lr"]),
        betas=tuple(float(value) for value in optimizer_rules["betas"]),
        eps=float(optimizer_rules["eps"]),
        weight_decay=float(optimizer_rules["weight_decay"]),
        amsgrad=bool(optimizer_rules["amsgrad"]),
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "active_names": active_names,
        "inactive_names": inactive_names,
        "initial_state_sha256": initial_state_sha,
        "shared_state_sha256": shared_state_sha,
        "builder_metadata_sha256": contracts.canonical_sha256(metadata),
        "validation": dict(validation),
    }


def _loss_v32(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with core_v32.capture_c3_v32_training_router(model) as capture:
        outputs = model(images)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        _fail("finite", "V3.2 training forward did not return six outputs")
    router_loss = core_v32.tri_router_supervision_loss(
        capture, outputs[-1].detach(), masks
    )
    segmentation_loss = sum(criterion(output, masks) for output in outputs)
    total_loss = segmentation_loss + router_loss
    return total_loss, segmentation_loss, router_loss


def _loss_v33(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total, segmentation, router, _ = core_v33.training_losses_v33(
        model,
        images,
        masks,
        criterion,
        balance_mode=BALANCE_MODE,
        router_loss_weight=1.0,
    )
    return total, segmentation, router


def _audit_final_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    active_names: Sequence[str],
    method: str,
) -> dict[str, Any]:
    named = dict(model.named_parameters())
    gain_name = (
        core_v32.SBSC_V32_GAIN_STATE_KEY
        if method == "sbsc_v32"
        else core_v33.SBSC_V33_GAIN_STATE_KEY
    )
    missing_unexpected: list[str] = []
    missing_conditionally_allowed: list[str] = []
    nonfinite_gradients: list[str] = []
    nonfinite_parameters: list[str] = []
    for name in active_names:
        parameter = named.get(name)
        if parameter is None:
            missing_unexpected.append(name)
            continue
        if not bool(torch.isfinite(parameter.detach()).all()):
            nonfinite_parameters.append(name)
        if parameter.grad is None:
            if name == gain_name:
                missing_conditionally_allowed.append(name)
            else:
                missing_unexpected.append(name)
        elif not bool(torch.isfinite(parameter.grad.detach()).all()):
            nonfinite_gradients.append(name)
    nonfinite_state = False
    for value in model.state_dict().values():
        if value.is_floating_point() and not bool(torch.isfinite(value.detach()).all()):
            nonfinite_state = True
            break
    nonfinite_optimizer_state = False
    for state in optimizer.state.values():
        for value in state.values():
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and not bool(torch.isfinite(value.detach()).all())
            ):
                nonfinite_optimizer_state = True
                break
        if nonfinite_optimizer_state:
            break
    passed = not (
        missing_unexpected
        or nonfinite_gradients
        or nonfinite_parameters
        or nonfinite_state
        or nonfinite_optimizer_state
    )
    return {
        "pass": passed,
        "active_parameter_count": len(active_names),
        "missing_unexpected": missing_unexpected,
        "missing_conditionally_allowed": missing_conditionally_allowed,
        "conditional_missing_gradient_parameter": gain_name,
        "nonfinite_gradients": nonfinite_gradients,
        "nonfinite_parameters": nonfinite_parameters,
        "model_state_finite": not nonfinite_state,
        "optimizer_state_finite": not nonfinite_optimizer_state,
    }


def _empty_model_record(
    spec: Mapping[str, Any],
    batch_plan_sha256: str,
    device_total_memory_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": MODEL_RECORD_SCHEMA,
        "method": spec["method"],
        "model": spec["model"],
        "model_schema": spec["model_schema"],
        "loss_schema": spec["loss_schema"],
        "identity_valid": False,
        "shared_state_sha256": None,
        "initial_state_sha256": None,
        "builder_metadata_sha256": None,
        "router_value_gradient_mode": spec["router_value_gradient_mode"],
        "balance_mode": spec["balance_mode"],
        "batch_plan_sha256": batch_plan_sha256,
        "input_pair_sha256_by_step": [],
        "step_records": [],
        "warmup_step_count": 0,
        "measured_step_count": 0,
        "measured_sample_count": 0,
        "measured_elapsed_seconds": None,
        "throughput_samples_per_second": None,
        "max_memory_allocated_bytes": 0,
        "max_memory_reserved_bytes": 0,
        "device_total_memory_bytes": device_total_memory_bytes,
        "finite": False,
        "training_state_audit": None,
        "oom_count": 0,
        "complete": False,
    }


def benchmark_one_model(
    spec: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    device: torch.device,
    device_total_memory_bytes: int,
    optimizer_rules: Mapping[str, Any],
) -> dict[str, Any]:
    batch_plan_sha = contracts.canonical_sha256(
        [list(batch["indices"]) for batch in batches]
    )
    record = _empty_model_record(spec, batch_plan_sha, device_total_memory_bytes)
    prepared: dict[str, Any] | None = None
    try:
        torch.cuda.empty_cache()
        prepared = _build_model(spec, device, optimizer_rules)
        model = prepared["model"]
        optimizer = prepared["optimizer"]
        criterion = nn.BCELoss(reduction="mean")
        record.update(
            {
                "identity_valid": True,
                "shared_state_sha256": prepared["shared_state_sha256"],
                "initial_state_sha256": prepared["initial_state_sha256"],
                "builder_metadata_sha256": prepared[
                    "builder_metadata_sha256"
                ],
                "state_key_count": len(model.state_dict()),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "inactive_parameter_names": list(prepared["inactive_names"]),
                "active_parameter_count": len(prepared["active_names"]),
            }
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        measured_elapsed = 0.0
        for batch in batches:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            images = batch["images"].to(device, dtype=torch.float32, non_blocking=False)
            masks = batch["masks"].to(device, dtype=torch.float32, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            if spec["method"] == "sbsc_v32":
                total, segmentation, router = _loss_v32(
                    model, images, masks, criterion
                )
            else:
                total, segmentation, router = _loss_v33(
                    model, images, masks, criterion
                )
            total.backward()
            optimizer.step()
            if spec["method"] == "sbsc_v32":
                core_v32.project_sbsc_v32_constraints_(model)
            else:
                core_v33.project_sbsc_v33_constraints_(model)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            values = {
                "total": float(total.detach().item()),
                "segmentation": float(segmentation.detach().item()),
                "router": float(router.detach().item()),
            }
            step_finite = (
                math.isfinite(elapsed)
                and elapsed > 0.0
                and all(math.isfinite(value) and value >= 0.0 for value in values.values())
            )
            position = int(batch["position"])
            phase = str(batch["phase"])
            record["step_records"].append(
                {
                    "position": position,
                    "phase": phase,
                    "batch_size": BATCH_SIZE,
                    "input_pair_sha256": batch["input_pair_sha256"],
                    "wall_elapsed_seconds": elapsed,
                    "loss": values,
                    "finite": step_finite,
                }
            )
            record["input_pair_sha256_by_step"].append(
                batch["input_pair_sha256"]
            )
            if phase == "warmup":
                record["warmup_step_count"] += 1
            else:
                record["measured_step_count"] += 1
                record["measured_sample_count"] += BATCH_SIZE
                measured_elapsed += elapsed
            del images, masks, total, segmentation, router
        record["max_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        record["max_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        record["measured_elapsed_seconds"] = measured_elapsed
        record["throughput_samples_per_second"] = (
            MEASURED_SAMPLES / measured_elapsed if measured_elapsed > 0.0 else None
        )
        training_state_audit = _audit_final_training_state(
            model, optimizer, prepared["active_names"], str(spec["method"])
        )
        record["training_state_audit"] = training_state_audit
        record["finite"] = training_state_audit["pass"] and all(
            step["finite"] for step in record["step_records"]
        )
        record["complete"] = (
            record["warmup_step_count"] == WARMUP_STEPS
            and record["measured_step_count"] == MEASURED_STEPS
            and record["measured_sample_count"] == MEASURED_SAMPLES
            and len(record["step_records"]) == TOTAL_STEPS
        )
    except torch.cuda.OutOfMemoryError:
        record["oom_count"] += 1
        record["finite"] = False
        record["complete"] = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        if prepared is not None:
            prepared.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return record


def evaluate_hard_gate(
    model_records: Mapping[str, Any],
    *,
    global_identity_valid: bool,
) -> dict[str, Any]:
    records = [model_records.get(method) for method in MODEL_ORDER]
    mappings = all(isinstance(record, Mapping) for record in records)
    identity = bool(global_identity_valid and mappings) and all(
        record.get("schema") == MODEL_RECORD_SCHEMA
        and record.get("identity_valid") is True
        and record.get("method") == method
        and record.get("model") == spec["model"]
        and record.get("model_schema") == spec["model_schema"]
        and record.get("loss_schema") == spec["loss_schema"]
        and record.get("router_value_gradient_mode")
        == spec["router_value_gradient_mode"]
        and record.get("balance_mode") == spec["balance_mode"]
        and record.get("state_key_count") == spec["state_key_count"]
        and record.get("parameter_count") == spec["parameter_count"]
        and _is_lower_sha256(record.get("shared_state_sha256"))
        and _is_lower_sha256(record.get("initial_state_sha256"))
        and _is_lower_sha256(record.get("builder_metadata_sha256"))
        and record.get("batch_plan_sha256") == EXPECTED_BATCH_PLAN_SHA256
        for record, method, spec in zip(records, MODEL_ORDER, _model_specs())
    )
    if identity:
        identity = (
            records[0].get("shared_state_sha256")
            == records[1].get("shared_state_sha256")
        )
    same_input = bool(mappings) and (
        records[0].get("input_pair_sha256_by_step")
        == records[1].get("input_pair_sha256_by_step")
        and len(records[0].get("input_pair_sha256_by_step", ())) == TOTAL_STEPS
        and all(
            _is_lower_sha256(value)
            for value in records[0].get("input_pair_sha256_by_step", ())
        )
    )
    finite = bool(mappings) and all(
        record.get("finite") is True
        and isinstance(record.get("step_records"), list)
        and isinstance(record.get("training_state_audit"), Mapping)
        and record["training_state_audit"].get("pass") is True
        and _is_positive_finite_number(record.get("measured_elapsed_seconds"))
        and _is_positive_finite_number(
            record.get("throughput_samples_per_second")
        )
        and all(
            isinstance(step, Mapping)
            and step.get("finite") is True
            and _is_positive_finite_number(step.get("wall_elapsed_seconds"))
            and _is_lower_sha256(step.get("input_pair_sha256"))
            and isinstance(step.get("loss"), Mapping)
            and set(step["loss"]) == {"total", "segmentation", "router"}
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in step["loss"].values()
            )
            for step in record["step_records"]
        )
        for record in records
    )
    complete = bool(mappings) and all(
        record.get("complete") is True
        and record.get("warmup_step_count") == WARMUP_STEPS
        and record.get("measured_step_count") == MEASURED_STEPS
        and record.get("measured_sample_count") == MEASURED_SAMPLES
        and len(record.get("step_records", ())) == TOTAL_STEPS
        and all(isinstance(step, Mapping) for step in record["step_records"])
        and [step.get("position") for step in record["step_records"]]
        == list(range(TOTAL_STEPS))
        and [step.get("phase") for step in record["step_records"]]
        == ["warmup"] * WARMUP_STEPS + ["measured"] * MEASURED_STEPS
        and all(step.get("batch_size") == BATCH_SIZE for step in record["step_records"])
        and [step.get("input_pair_sha256") for step in record["step_records"]]
        == record.get("input_pair_sha256_by_step")
        and _is_positive_finite_number(record.get("measured_elapsed_seconds"))
        and _is_positive_finite_number(
            record.get("throughput_samples_per_second")
        )
        and math.isclose(
            float(record.get("measured_elapsed_seconds", math.nan)),
            sum(
                float(step.get("wall_elapsed_seconds", math.nan))
                for step in record["step_records"][WARMUP_STEPS:]
            ),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(record.get("throughput_samples_per_second", math.nan)),
            MEASURED_SAMPLES
            / float(record.get("measured_elapsed_seconds", math.nan)),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        for record in records
    )
    no_oom = bool(mappings) and all(
        type(record.get("oom_count")) is int and record.get("oom_count") == 0
        for record in records
    )
    peak_within_device = bool(mappings) and all(
        type(record.get("device_total_memory_bytes")) is int
        and record["device_total_memory_bytes"] > 0
        and type(record.get("max_memory_allocated_bytes")) is int
        and type(record.get("max_memory_reserved_bytes")) is int
        and 0 <= record["max_memory_allocated_bytes"] <= record["device_total_memory_bytes"]
        and 0 <= record["max_memory_reserved_bytes"] <= record["device_total_memory_bytes"]
        for record in records
    )
    checks = {
        "identity": identity,
        "same_input": same_input,
        "finite": finite,
        "complete_measurement": complete,
        "no_oom": no_oom,
        "peak_within_device": peak_within_device,
    }
    reasons = [name for name in HARD_GATE_ORDER if not checks[name]]
    return {
        "schema": REPORT_SCHEMA + "/hard_gate/v1",
        "hard_gate_order": list(HARD_GATE_ORDER),
        "checks": checks,
        "verdict": "GO" if not reasons else "NO-GO",
        "reasons": reasons,
        "speed_or_ratio_used_as_gate": False,
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, (int, float))
        or not isinstance(denominator, (int, float))
    ):
        return None
    left = float(numerator)
    right = float(denominator)
    if not math.isfinite(left) or not math.isfinite(right) or right <= 0.0:
        return None
    value = left / right
    return value if math.isfinite(value) else None


def warning_diagnostics(model_records: Mapping[str, Any]) -> dict[str, Any]:
    v32 = model_records.get("sbsc_v32", {})
    v33 = model_records.get("sbsc_v33_third", {})
    ratios = {
        "v33_to_v32_elapsed": _safe_ratio(
            v33.get("measured_elapsed_seconds"), v32.get("measured_elapsed_seconds")
        ),
        "v33_to_v32_throughput": _safe_ratio(
            v33.get("throughput_samples_per_second"),
            v32.get("throughput_samples_per_second"),
        ),
        "v33_to_v32_peak_allocated": _safe_ratio(
            v33.get("max_memory_allocated_bytes"),
            v32.get("max_memory_allocated_bytes"),
        ),
        "v33_to_v32_peak_reserved": _safe_ratio(
            v33.get("max_memory_reserved_bytes"),
            v32.get("max_memory_reserved_bytes"),
        ),
    }
    warnings: list[dict[str, Any]] = []
    if ratios["v33_to_v32_throughput"] is not None and ratios[
        "v33_to_v32_throughput"
    ] < 1.0:
        warnings.append({"kind": "v33_throughput_below_v32"})
    if ratios["v33_to_v32_peak_allocated"] is not None and ratios[
        "v33_to_v32_peak_allocated"
    ] > 1.0:
        warnings.append({"kind": "v33_peak_allocated_above_v32"})
    if ratios["v33_to_v32_peak_reserved"] is not None and ratios[
        "v33_to_v32_peak_reserved"
    ] > 1.0:
        warnings.append({"kind": "v33_peak_reserved_above_v32"})
    return {
        "affects_verdict": False,
        "v33_faster_than_v32_required": False,
        "ratios": ratios,
        "warnings": warnings,
    }


def build_report(
    *,
    rules: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    batch_plan: Sequence[Sequence[int]],
    batches: Sequence[Mapping[str, Any]],
    model_records: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    elapsed_seconds: float,
    global_identity_valid: bool = True,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_plan = [list(batch) for batch in batch_plan]
    batch_input_hashes = [
        batch.get("input_pair_sha256") if isinstance(batch, Mapping) else None
        for batch in batches
    ]
    report_identity_valid = bool(
        global_identity_valid
        and set(model_records) == set(MODEL_ORDER)
        and normalized_plan == expected_batch_plan()
        and contracts.canonical_sha256(normalized_plan)
        == EXPECTED_BATCH_PLAN_SHA256
        and len(batches) == TOTAL_STEPS
        and all(
            isinstance(batch, Mapping)
            and batch.get("position") == position
            and batch.get("phase")
            == ("warmup" if position < WARMUP_STEPS else "measured")
            and list(batch.get("indices", ())) == normalized_plan[position]
            and _is_lower_sha256(batch.get("input_pair_sha256"))
            for position, batch in enumerate(batches)
        )
        and isinstance(source_manifest, Mapping)
        and source_manifest.get("schema")
        == "sctransnet_sbsc_v33/source_manifest/v1"
        and isinstance(source_manifest.get("files"), list)
        and _is_lower_sha256(source_manifest.get("sha256"))
        and isinstance(runtime_identity, Mapping)
        and runtime_identity.get("precision") == "FP32"
        and runtime_identity.get("autocast") is False
        and runtime_identity.get("gradient_scaler") is False
        and runtime_identity.get("deterministic_algorithms") is True
    )
    hard_gate = evaluate_hard_gate(
        model_records, global_identity_valid=report_identity_valid
    )
    warnings = warning_diagnostics(model_records)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if hard_gate["verdict"] == "GO" else "FAIL",
        "hard_gate_pass": hard_gate["verdict"] == "GO",
        "write_once": True,
        "scope": rules["scope"],
        "rules_path": contracts.repository_relative_path(RULES_PATH),
        "rules_sha256": EXPECTED_RULES_SHA256,
        "authority_bindings": {
            "baseline": {
                "path": evidence["baseline"]["manifest_path"],
                "sha256": evidence["baseline"]["manifest_sha256"],
            },
            "gradient": {
                "path": evidence["gradient"]["authorization_path"],
                "sha256": evidence["gradient"]["authorization_sha256"],
                "authorized_router_value_gradient_mode": evidence["gradient"][
                    "authorized_router_value_gradient_mode"
                ],
            },
            "v2_canary": dict(evidence["v2_canary"]),
        },
        "training_source_manifest": dict(source_manifest),
        "training_source_manifest_sha256": source_manifest["sha256"],
        "data_identity": {
            "dataset": DATASET,
            "data_role": DATA_ROLE,
            "sample_count": SAMPLE_COUNT,
            "index_relpath": source_protocol.EXPECTED_SPLITS[DATASET]["train"][
                "index_relpath"
            ],
            "index_file_sha256": source_protocol.EXPECTED_SPLITS[DATASET]["train"][
                "file_sha256"
            ],
            "ordered_ids_sha256": source_protocol.EXPECTED_SPLITS[DATASET]["train"][
                "ordered_ids_sha256"
            ],
            "normalization": dict(source_protocol.LEGACY_NORMALIZATION[DATASET]),
            "dataset_epoch": DATASET_EPOCH,
            "batch_size": BATCH_SIZE,
            "patch_size": PATCH_SIZE,
        },
        "batch_plan": {
            "sha256": contracts.canonical_sha256(normalized_plan),
            "indices": normalized_plan,
            "input_pair_sha256_by_step": batch_input_hashes,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "same_materialized_cpu_batches_for_both_models": True,
        },
        "runtime_identity": dict(runtime_identity),
        "model_order": list(MODEL_ORDER),
        "model_records": {method: dict(model_records[method]) for method in MODEL_ORDER},
        "elapsed_seconds": _finite_number(
            elapsed_seconds, name="benchmark elapsed", positive=True
        ),
        "hard_gate": hard_gate,
        "warning_diagnostics": warnings,
        "artifact_boundaries": dict(ARTIFACT_BOUNDARIES),
    }
    if failure is not None:
        report["failure"] = dict(failure)
    return report


def _canonical_dataset() -> EviSIRSTTrainDataset:
    dataset_root = PROJECT_ROOT / "datasets"
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        _fail("identity", "canonical dataset root is unavailable")
    dataset = EviSIRSTTrainDataset(
        DATASET,
        dataset_root=dataset_root,
        patch_size=PATCH_SIZE,
        seed=RUN_SEED,
        return_metadata=False,
    )
    if len(dataset) != SAMPLE_COUNT:
        _fail("identity", "canonical IRSTD train count differs")
    return dataset


def _require_cuda(device_value: str) -> tuple[torch.device, dict[str, Any]]:
    configure_determinism(RUN_SEED)
    device = require_device(device_value)
    if device.type != "cuda" or not torch.cuda.is_available():
        _fail("identity", "resource benchmark requires an available CUDA device")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != FROZEN_CUBLAS_WORKSPACE_CONFIG:
        _fail("identity", "CUBLAS_WORKSPACE_CONFIG conflicts")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(device)
    total_memory = int(properties.total_memory)
    if total_memory <= 0:
        _fail("peak_within_device", "CUDA device total memory is invalid")
    return device, {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_requested": device_value,
        "device": str(device),
        "device_name": str(properties.name),
        "device_total_memory_bytes": total_memory,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "precision": "FP32",
        "autocast": False,
        "gradient_scaler": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if REPORT_PATH.exists() or REPORT_PATH.is_symlink():
        raise FileExistsError("paired resource benchmark report already exists")
    rules = load_frozen_rules()
    evidence = load_bound_evidence(rules)
    source_manifest = build_source_manifest()
    batch_plan = expected_batch_plan()
    dataset = _canonical_dataset()
    batches = materialize_training_batches(dataset, batch_plan)
    device, runtime_identity = _require_cuda(args.device)
    total_memory = int(runtime_identity["device_total_memory_bytes"])
    model_records: dict[str, Any] = {}
    failure: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        specs = {spec["method"]: spec for spec in _model_specs()}
        for method in MODEL_ORDER:
            model_records[method] = benchmark_one_model(
                specs[method],
                batches,
                device,
                total_memory,
                rules["optimizer"],
            )
    except Exception as exc:
        category = (
            exc.category if isinstance(exc, ResourceBenchmarkError) else "runtime_failure"
        )
        failure = {
            "category": category,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        specs = {spec["method"]: spec for spec in _model_specs()}
        for method in MODEL_ORDER:
            model_records.setdefault(
                method,
                _empty_model_record(
                    specs[method], EXPECTED_BATCH_PLAN_SHA256, total_memory
                ),
            )
    elapsed = time.monotonic() - started
    source_unchanged = build_source_manifest() == source_manifest
    report = build_report(
        rules=rules,
        evidence=evidence,
        source_manifest=source_manifest,
        batch_plan=batch_plan,
        batches=batches,
        model_records=model_records,
        runtime_identity=runtime_identity,
        elapsed_seconds=max(elapsed, sys.float_info.min),
        global_identity_valid=source_unchanged and failure is None,
        failure=failure,
    )
    contracts.write_once_json(REPORT_PATH, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run_benchmark(parse_args(argv))
    print(contracts.canonical_json_bytes(report).decode("utf-8"), end="")
    return 0 if report["hard_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
