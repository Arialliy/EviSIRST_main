#!/usr/bin/env python3
"""Publish the three Current packages with canonical synthetic replay.

V3 is an append-only bundle wrapper around the frozen V1 package producer.
The package schema and model state remain V1.  V3 makes the synthetic hashes
portable as a *qualified* claim by binding their exact replay environment:
CPU, float32, the fixed linspace probe, the bound PyTorch version, and one
intra-op thread.  It deliberately makes no bitwise claim for arbitrary thread
counts.

The exporter reads only already-persisted source artifacts through the V1
source validator.  It never imports an official-test loader, opens or parses
an official-test index, builds an official-test loader, runs official
evaluation, or recomputes an official metric.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import export_three_dataset_current_to_inference as v1  # noqa: E402


BUNDLE_SCHEMA = "sctransnet_three_component_current_inference_bundle/v3"
MANIFEST_SCHEMA = "sctransnet_three_component_current_inference_manifest/v3"
COMMITTED_SCHEMA = "sctransnet_three_component_current_inference_committed/v3"
SYNTHETIC_EXECUTION_SCHEMA = (
    "sctransnet_canonical_synthetic_replay_execution/v1"
)
PACKAGE_SCHEMA = v1.PACKAGE_SCHEMA
MODEL_NAME = v1.MODEL_NAME
MODEL_PUBLIC_ID = v1.MODEL_PUBLIC_ID
DATASETS = v1.DATASETS
TRAINING_SEED = v1.TRAINING_SEED
SOURCE_SCHEMA = v1.SOURCE_SCHEMA
CHECKPOINT_ROLE = v1.CHECKPOINT_ROLE
TRAINING_STATE_KEY_COUNT = v1.TRAINING_STATE_KEY_COUNT
INFERENCE_STATE_KEY_COUNT = v1.INFERENCE_STATE_KEY_COUNT
TRAINING_PARAMETER_COUNT = v1.TRAINING_PARAMETER_COUNT
INFERENCE_PARAMETER_COUNT = v1.INFERENCE_PARAMETER_COUNT
TSS_PARAMETER_COUNT = v1.TSS_PARAMETER_COUNT
STATE_HASH_ALGORITHM = v1.STATE_HASH_ALGORITHM
OFFICIAL_FALSE_FLAGS = v1.OFFICIAL_FALSE_FLAGS
OFFICIAL_BOUNDARY_SCOPE = v1.OFFICIAL_BOUNDARY_SCOPE
ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256 = (
    v1.ARCHITECTURE_MANIFEST_CANONICAL_JSON_SHA256
)
SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256 = (
    v1.SORTED_STATE_KEY_SET_CANONICAL_JSON_SHA256
)
SURVIVAL_STATE_KEYS = v1.SURVIVAL_STATE_KEYS
QFG_STATE_KEYS = v1.QFG_STATE_KEYS
SOURCE_LOCKS = v1.SOURCE_LOCKS
PACKAGE_FILENAMES = dict(v1.PACKAGE_FILENAMES)
DEFAULT_SOURCE_ROOT = v1.DEFAULT_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/current_three_component_inference_v3"
DEPLOYMENT_PROTOCOL_PATH = (
    REPO_ROOT / "experiments/CURRENT_THREE_COMPONENT_DEPLOYMENT_PROTOCOL_V3.md"
)

CANONICAL_INTRAOP_THREADS = 1
CANONICAL_DEVICE = "cpu"
CANONICAL_DTYPE = "float32"
CANONICAL_PROBE = "linspace[-1,1]_1x1x32x32_float32_cpu"
CANONICAL_PROBE_CONSTRUCTOR = (
    "torch.linspace(-1.0,1.0,1024,dtype=torch.float32,device='cpu')"
    ".reshape(1,1,32,32)"
)

_PACKAGE_IDENTITY_FIELDS = (
    "probe",
    "hash_algorithm",
    "probe_state_dict_sha256",
    "six_outputs_state_dict_sha256",
    "public_output_state_dict_sha256",
    "six_output_count",
    "all_six_bitwise_equal",
    "maximum_absolute_difference",
    "deployed_single_output_equals_sixth",
    "deployed_output",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _official_false_payload() -> dict[str, bool]:
    return {field: False for field in OFFICIAL_FALSE_FLAGS}


def _state_hash_contract() -> dict[str, Any]:
    return copy.deepcopy(v1._state_hash_contract())


@contextmanager
def canonical_synthetic_replay_context() -> Iterator[None]:
    """Temporarily select every canonical replay control and restore it.

    PyTorch's inter-op thread count is intentionally neither read nor changed.
    In particular, this function never calls ``torch.set_num_interop_threads``.
    Nested use is safe: each level restores the count observed at its entry.
    """

    previous_threads = int(torch.get_num_threads())
    previous_dtype = torch.get_default_dtype()
    previous_mkldnn = bool(torch.backends.mkldnn.enabled)
    try:
        if previous_threads != CANONICAL_INTRAOP_THREADS:
            torch.set_num_threads(CANONICAL_INTRAOP_THREADS)
        if previous_dtype != torch.float32:
            torch.set_default_dtype(torch.float32)
        if not previous_mkldnn:
            torch.backends.mkldnn.enabled = True
        _require(
            int(torch.get_num_threads()) == CANONICAL_INTRAOP_THREADS,
            "canonical synthetic replay intra-op thread count differs",
        )
        _require(
            torch.get_default_dtype() == torch.float32,
            "canonical synthetic replay default dtype differs",
        )
        _require(
            torch.backends.mkldnn.enabled is True,
            "canonical synthetic replay MKLDNN setting differs",
        )
        # torch.device acts as a nestable default-device context.  Explicitly
        # disable CPU autocast so an ambient autocast region cannot change the
        # V1 model/probe replay.  Both contexts unwind before state restoration.
        with torch.device(CANONICAL_DEVICE), torch.autocast(
            device_type=CANONICAL_DEVICE,
            enabled=False,
        ):
            yield
    finally:
        if bool(torch.backends.mkldnn.enabled) != previous_mkldnn:
            torch.backends.mkldnn.enabled = previous_mkldnn
        if torch.get_default_dtype() != previous_dtype:
            torch.set_default_dtype(previous_dtype)
        if int(torch.get_num_threads()) != previous_threads:
            torch.set_num_threads(previous_threads)


def synthetic_replay_execution_contract() -> dict[str, Any]:
    """Return the exact environment to which V3 synthetic hashes are bound."""

    return {
        "schema": SYNTHETIC_EXECUTION_SCHEMA,
        "device": CANONICAL_DEVICE,
        "default_device_scoped": CANONICAL_DEVICE,
        "dtype": CANONICAL_DTYPE,
        "default_dtype": CANONICAL_DTYPE,
        "default_dtype_temporarily_scoped_and_restored": True,
        "cpu_autocast_enabled": False,
        "mkldnn_enabled": True,
        "mkldnn_temporarily_scoped_and_restored": True,
        "intraop_threads": CANONICAL_INTRAOP_THREADS,
        "intraop_threads_temporarily_scoped_and_restored": True,
        "interop_threads_bound": False,
        "interop_threads_setter_called": False,
        "probe": CANONICAL_PROBE,
        "probe_constructor": CANONICAL_PROBE_CONSTRUCTOR,
        "torch_version": str(torch.__version__),
        "torch_version_bound": True,
        "hash_algorithm": STATE_HASH_ALGORITHM,
        "hash_guarantee_scope": "canonical_replay_execution_contract_only",
        "arbitrary_intraop_thread_count_bitwise_equivalence_claimed": False,
    }


def _validate_execution_contract(value: Any) -> dict[str, Any]:
    expected = synthetic_replay_execution_contract()
    _require(
        isinstance(value, Mapping) and dict(value) == expected,
        "synthetic replay execution contract differs",
    )
    return copy.deepcopy(expected)


def _implementation_binding(
    exporter_path: Path,
    protocol_path: Path,
) -> dict[str, str]:
    repository = REPO_ROOT.resolve()
    exporter = Path(exporter_path).resolve()
    protocol = Path(protocol_path).resolve()
    _require(exporter.is_relative_to(repository), "exporter escapes repository")
    _require(
        protocol.is_relative_to(repository),
        "deployment protocol escapes repository",
    )
    return {
        "exporter_repo_relative_path": exporter.relative_to(repository).as_posix(),
        "exporter_file_sha256": _sha256_bytes(_regular_file_bytes(exporter)),
        "protocol_repo_relative_path": protocol.relative_to(repository).as_posix(),
        "protocol_file_sha256": _sha256_bytes(_regular_file_bytes(protocol)),
    }


def _bundle_implementation_binding() -> dict[str, str]:
    return _implementation_binding(Path(__file__), DEPLOYMENT_PROTOCOL_PATH)


def _package_producer_implementation_binding() -> dict[str, str]:
    binding = copy.deepcopy(v1._deployment_implementation_binding())
    _require(
        binding.get("exporter_repo_relative_path")
        == "experiments/export_three_dataset_current_to_inference.py",
        "V1 package producer exporter path differs",
    )
    _require(
        binding.get("protocol_repo_relative_path")
        == "experiments/CURRENT_THREE_COMPONENT_DEPLOYMENT_PROTOCOL_V1.md",
        "V1 package producer protocol path differs",
    )
    _require(
        _is_sha256(binding.get("exporter_file_sha256"))
        and _is_sha256(binding.get("protocol_file_sha256")),
        "V1 package producer SHA binding differs",
    )
    return binding


def _manifest_semantic_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_semantic_sha256", None)
    return _canonical_sha256(unsigned)


def _require_dataset(dataset: Any) -> str:
    if type(dataset) is not str or dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    return dataset


def _complete_synthetic_identity(identity: Any) -> dict[str, Any]:
    """Validate and copy the complete V1 identity without changing its hashes."""

    _require(isinstance(identity, Mapping), "package synthetic identity is absent")
    _require(
        set(identity) == set(_PACKAGE_IDENTITY_FIELDS),
        "package synthetic identity field set differs",
    )
    _require(identity.get("probe") == CANONICAL_PROBE, "synthetic identity probe differs")
    _require(
        identity.get("hash_algorithm") == STATE_HASH_ALGORITHM,
        "synthetic identity hash algorithm differs",
    )
    for field in (
        "probe_state_dict_sha256",
        "six_outputs_state_dict_sha256",
        "public_output_state_dict_sha256",
    ):
        _require(
            _is_sha256(identity.get(field)),
            f"synthetic identity {field!r} differs",
        )
    _require(identity.get("six_output_count") == 6, "synthetic identity output count differs")
    _require(
        identity.get("all_six_bitwise_equal") is True,
        "synthetic identity bitwise equality differs",
    )
    _require(
        identity.get("maximum_absolute_difference") == 0.0,
        "synthetic identity maximum absolute difference differs",
    )
    _require(
        identity.get("deployed_single_output_equals_sixth") is True,
        "synthetic identity deployed/sixth relation differs",
    )
    _require(
        identity.get("deployed_output") == "sigmoid(out)",
        "synthetic identity output declaration differs",
    )
    return copy.deepcopy(dict(identity))


def _load_package_payload(package_path: Path) -> tuple[dict[str, Any], bytes]:
    content = _regular_file_bytes(package_path)
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), "V1 package payload must be a mapping")
    _require(payload.get("schema") == PACKAGE_SCHEMA, "V1 package schema differs")
    return dict(payload), content


def validate_current_source_v3(
    source_root: Path,
    dataset: str,
) -> dict[str, Any]:
    """Run the complete V1 source proof under the canonical replay context."""

    dataset = _require_dataset(dataset)
    with canonical_synthetic_replay_context():
        result = v1.validate_current_source(source_root, dataset)
    binding = result.get("binding")
    _require(isinstance(binding, Mapping), "source validation binding is absent")
    _complete_synthetic_identity(binding.get("synthetic_identity"))
    return result


def _package_from_source_v3(source: Mapping[str, Any]) -> dict[str, Any]:
    """Generate the unchanged V1 package while canonical replay is selected."""

    binding = source.get("binding")
    _require(isinstance(binding, Mapping), "source context lacks binding")
    _complete_synthetic_identity(binding.get("synthetic_identity"))
    with canonical_synthetic_replay_context():
        package = v1._package_from_source(source)
    _require(isinstance(package, Mapping), "V1 package producer returned no mapping")
    return dict(package)


def validate_exported_current_package_v3(
    package_path: Path,
    *,
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    """Run V1 strict-load and replay validation at one intra-op thread."""

    if expected_dataset is not None:
        expected_dataset = _require_dataset(expected_dataset)
    with canonical_synthetic_replay_context():
        result = v1.validate_exported_current_package(
            package_path,
            expected_dataset=expected_dataset,
        )
    return dict(result)


def load_exported_current_model_v3(
    package_path: Path,
    *,
    expected_dataset: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Canonical-validate a V1 package, then return its strict-loaded model.

    The caller's intra-op thread count is restored before this function
    returns.  The returned model is not a promise that arbitrary later thread
    counts reproduce the canonical synthetic hashes bitwise.
    """

    if expected_dataset is not None:
        expected_dataset = _require_dataset(expected_dataset)
    with canonical_synthetic_replay_context():
        model, metadata = v1.load_exported_current_model(
            package_path,
            expected_dataset=expected_dataset,
        )
    ready = dict(metadata)
    ready.update(
        {
            "v3_canonical_package_validation": True,
            "synthetic_replay_execution_contract": (
                synthetic_replay_execution_contract()
            ),
        }
    )
    return model, ready


def preflight_all_sources(
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, Any]:
    contexts = {
        dataset: validate_current_source_v3(source_root, dataset)
        for dataset in DATASETS
    }
    architecture = contexts[DATASETS[0]]["binding"]["architecture"]
    _require(
        all(
            contexts[dataset]["binding"]["architecture"] == architecture
            for dataset in DATASETS
        ),
        "dataset architecture contracts differ",
    )
    return {
        "schema": BUNDLE_SCHEMA,
        "status": "preflight_complete",
        "model_name": MODEL_NAME,
        "datasets": list(DATASETS),
        "contexts": contexts,
        "architecture": copy.deepcopy(architecture),
        "synthetic_replay_execution_contract": (
            synthetic_replay_execution_contract()
        ),
        "bundle_implementation": _bundle_implementation_binding(),
        "package_producer_implementation": (
            _package_producer_implementation_binding()
        ),
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }


def _package_entry(
    package_path: Path,
    output_root: Path,
    *,
    expected_dataset: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = _require_dataset(expected_dataset)
    validated = validate_exported_current_package_v3(
        package_path,
        expected_dataset=dataset,
    )
    payload, content = _load_package_payload(package_path)
    _require(payload.get("dataset") == dataset, "V1 package dataset differs")
    _require(
        payload.get("deployment_implementation")
        == _package_producer_implementation_binding(),
        "V1 package producer implementation binding differs",
    )
    identity = _complete_synthetic_identity(payload.get("synthetic_identity"))
    source = payload.get("source")
    _require(isinstance(source, Mapping), "V1 package source binding is absent")
    _require(
        source.get("synthetic_identity") == identity,
        "V1 package source/top-level synthetic identity differs",
    )
    _require(
        validated.get("bytes") == len(content)
        and validated.get("file_sha256") == _sha256_bytes(content),
        "V1 package byte binding differs",
    )
    relative_path = Path(package_path).resolve().relative_to(
        Path(output_root).resolve()
    )
    entry = {
        "dataset": dataset,
        "epoch": validated["epoch"],
        "relative_path": relative_path.as_posix(),
        "bytes": validated["bytes"],
        "file_sha256": validated["file_sha256"],
        "package_schema": PACKAGE_SCHEMA,
        "package_producer_implementation": (
            _package_producer_implementation_binding()
        ),
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
        "strict_load": True,
        "tss_absent": True,
        "qfg_preserved": True,
        "synthetic_identity": identity,
        "synthetic_replay_execution_contract": (
            synthetic_replay_execution_contract()
        ),
    }
    return entry, dict(validated)


def _manifest_static_contract(
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "bundle_schema": BUNDLE_SCHEMA,
        "status": "complete",
        "model_name": MODEL_NAME,
        "model_public_id": MODEL_PUBLIC_ID,
        "datasets": list(DATASETS),
        "training_seed": TRAINING_SEED,
        "source_schema": SOURCE_SCHEMA,
        "source_checkpoint_role": CHECKPOINT_ROLE,
        "package_schema": PACKAGE_SCHEMA,
        "training_state_key_count": TRAINING_STATE_KEY_COUNT,
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "training_parameter_count": TRAINING_PARAMETER_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
        "state_hash_contract": _state_hash_contract(),
        "architecture": copy.deepcopy(dict(architecture)),
        "synthetic_replay_execution_contract": (
            synthetic_replay_execution_contract()
        ),
        "bundle_implementation": _bundle_implementation_binding(),
        "package_producer_implementation": (
            _package_producer_implementation_binding()
        ),
        "tss_part_of_final_model": False,
        "tss_training_only_source_state_key_count": len(SURVIVAL_STATE_KEYS),
        "tss_training_only_source_parameter_count": TSS_PARAMETER_COUNT,
        "tss_removed_state_keys": list(SURVIVAL_STATE_KEYS),
        "qfg_required_for_inference": True,
        "qfg_state_key_count": len(QFG_STATE_KEYS),
        "qfg_preserved_state_keys": list(QFG_STATE_KEYS),
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


def _validate_uncommitted_manifest(output_root: Path) -> dict[str, Any]:
    root = Path(output_root).resolve(strict=True)
    manifest, manifest_content = _json_regular(root / "manifest.json", "V3 manifest")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema differs")
    _require(
        manifest.get("bundle_schema") == BUNDLE_SCHEMA,
        "manifest bundle schema differs",
    )
    _require(manifest.get("status") == "complete", "manifest status differs")
    _validate_execution_contract(
        manifest.get("synthetic_replay_execution_contract")
    )
    packages = manifest.get("packages")
    _require(
        isinstance(packages, list) and len(packages) == len(DATASETS),
        "manifest package list differs",
    )
    package_dir = root / "packages"
    _require(
        not package_dir.is_symlink() and package_dir.is_dir(),
        "package directory differs",
    )
    expected_names = {PACKAGE_FILENAMES[dataset] for dataset in DATASETS}
    _require(
        {path.name for path in package_dir.iterdir()} == expected_names,
        "package file set differs",
    )

    observed: dict[str, dict[str, Any]] = {}
    observed_entries: list[dict[str, Any]] = []
    for dataset, supplied_entry in zip(DATASETS, packages):
        _require(isinstance(supplied_entry, Mapping), "manifest package entry differs")
        _require(
            supplied_entry.get("dataset") == dataset,
            "manifest package dataset order differs",
        )
        _validate_execution_contract(
            supplied_entry.get("synthetic_replay_execution_contract")
        )
        relative_path = Path("packages") / PACKAGE_FILENAMES[dataset]
        _require(
            supplied_entry.get("relative_path") == relative_path.as_posix(),
            "manifest package relative path differs",
        )
        package_path = (root / relative_path).resolve(strict=True)
        _require(package_path.is_relative_to(root), "package path escapes bundle")
        expected_entry, validated = _package_entry(
            package_path,
            root,
            expected_dataset=dataset,
        )
        _require(
            dict(supplied_entry) == expected_entry,
            f"manifest package entry, synthetic identity, or execution contract differs for {dataset}",
        )
        observed[dataset] = validated
        observed_entries.append(expected_entry)

    common_architecture = observed[DATASETS[0]]["architecture"]
    _require(
        all(
            observed[dataset]["architecture"] == common_architecture
            for dataset in DATASETS
        ),
        "package architecture contracts differ",
    )
    expected_manifest = _manifest_static_contract(common_architecture)
    expected_manifest["packages"] = observed_entries
    expected_manifest["manifest_semantic_sha256"] = _manifest_semantic_sha256(
        expected_manifest
    )
    _require(dict(manifest) == expected_manifest, "manifest contract differs")
    _require(
        manifest.get("manifest_semantic_sha256")
        == _manifest_semantic_sha256(manifest),
        "manifest semantic SHA differs",
    )
    return {
        "manifest": manifest,
        "manifest_file_sha256": _sha256_bytes(manifest_content),
        "manifest_semantic_sha256": manifest["manifest_semantic_sha256"],
        "packages": observed,
        "package_entries": observed_entries,
    }


def validate_committed_bundle(
    output_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a completed V3 bundle without mutating it."""

    root = Path(output_root).resolve(strict=True)
    _require(not root.is_symlink() and root.is_dir(), "bundle root differs")
    committed, committed_content = _json_regular(root / "COMMITTED", "V3 COMMITTED")
    _require(committed.get("schema") == COMMITTED_SCHEMA, "COMMITTED schema differs")
    _require(
        committed.get("bundle_schema") == BUNDLE_SCHEMA,
        "COMMITTED bundle schema differs",
    )
    _require(committed.get("status") == "complete", "COMMITTED status differs")
    _validate_execution_contract(
        committed.get("synthetic_replay_execution_contract")
    )
    _require(
        committed.get("export_operation_official_access")
        == _official_false_payload(),
        "COMMITTED official-access flags differ",
    )
    _require(
        committed.get("official_boundary_scope") == OFFICIAL_BOUNDARY_SCOPE,
        "COMMITTED official boundary scope differs",
    )
    validation = _validate_uncommitted_manifest(root)
    expected_committed = {
        "schema": COMMITTED_SCHEMA,
        "bundle_schema": BUNDLE_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "package_schema": PACKAGE_SCHEMA,
        "status": "complete",
        "manifest_path": "manifest.json",
        "manifest_file_sha256": validation["manifest_file_sha256"],
        "manifest_semantic_sha256": validation["manifest_semantic_sha256"],
        "package_count": len(DATASETS),
        "datasets": list(DATASETS),
        "synthetic_replay_execution_contract": (
            synthetic_replay_execution_contract()
        ),
        "bundle_implementation": _bundle_implementation_binding(),
        "package_producer_implementation": (
            _package_producer_implementation_binding()
        ),
        "root_committed_written_last": True,
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }
    _require(dict(committed) == expected_committed, "COMMITTED contract differs")
    _require(
        {path.name for path in root.iterdir()}
        == {"packages", "manifest.json", "COMMITTED"},
        "bundle root file set differs",
    )
    if source_root is not None:
        contexts = preflight_all_sources(source_root)["contexts"]
        for entry in validation["package_entries"]:
            dataset = entry["dataset"]
            binding = contexts[dataset]["binding"]
            _require(
                entry["state_sha256"] == binding["inference_state_sha256"],
                f"bundle/source state differs for {dataset}",
            )
            _require(
                entry["synthetic_identity"]
                == _complete_synthetic_identity(binding["synthetic_identity"]),
                f"bundle/source synthetic identity differs for {dataset}",
            )
            _require(
                entry["synthetic_replay_execution_contract"]
                == synthetic_replay_execution_contract(),
                f"bundle/source execution contract differs for {dataset}",
            )
    return {
        "schema": BUNDLE_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "committed_schema": COMMITTED_SCHEMA,
        "status": "complete",
        "output_root": str(root),
        "manifest_file_sha256": validation["manifest_file_sha256"],
        "manifest_semantic_sha256": validation["manifest_semantic_sha256"],
        "committed_file_sha256": _sha256_bytes(committed_content),
        "synthetic_replay_execution_contract": (
            synthetic_replay_execution_contract()
        ),
        "packages": validation["packages"],
        "package_entries": validation["package_entries"],
        "idempotent_existing_bundle": True,
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }


def export_current_bundle(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Publish all three V1 packages under a V3 root; write COMMITTED last."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        if (
            output_root.is_dir()
            and not output_root.is_symlink()
            and (output_root / "COMMITTED").is_file()
        ):
            return validate_committed_bundle(output_root, source_root=source_root)
        raise FileExistsError(
            f"incomplete or foreign write-once V3 bundle exists: {output_root}"
        )

    # Finish all source checks and canonical six-output proofs before reserving
    # the write-once output root.
    preflight = preflight_all_sources(source_root)
    contexts = preflight["contexts"]
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise NotADirectoryError(output_root.parent)
    output_root.mkdir(exist_ok=False)
    v1._fsync_directory(output_root.parent)
    packages_dir = output_root / "packages"
    packages_dir.mkdir(exist_ok=False)

    package_entries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        destination = packages_dir / PACKAGE_FILENAMES[dataset]
        package = _package_from_source_v3(contexts[dataset])
        v1._write_bytes_once(destination, v1._torch_bytes(package))
        entry, _ = _package_entry(
            destination,
            output_root,
            expected_dataset=dataset,
        )
        package_entries.append(entry)

    manifest = _manifest_static_contract(preflight["architecture"])
    manifest["packages"] = package_entries
    manifest["manifest_semantic_sha256"] = _manifest_semantic_sha256(manifest)
    v1._write_bytes_once(output_root / "manifest.json", _json_bytes(manifest))

    # Validate packages and manifest before publication is committed.
    uncommitted = _validate_uncommitted_manifest(output_root)
    committed = {
        "schema": COMMITTED_SCHEMA,
        "bundle_schema": BUNDLE_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "package_schema": PACKAGE_SCHEMA,
        "status": "complete",
        "manifest_path": "manifest.json",
        "manifest_file_sha256": uncommitted["manifest_file_sha256"],
        "manifest_semantic_sha256": uncommitted["manifest_semantic_sha256"],
        "package_count": len(DATASETS),
        "datasets": list(DATASETS),
        "synthetic_replay_execution_contract": (
            synthetic_replay_execution_contract()
        ),
        "bundle_implementation": _bundle_implementation_binding(),
        "package_producer_implementation": (
            _package_producer_implementation_binding()
        ),
        "root_committed_written_last": True,
        "export_operation_official_access": _official_false_payload(),
        "official_boundary_scope": OFFICIAL_BOUNDARY_SCOPE,
    }
    # This must remain the final artifact write in the publication path.
    v1._write_bytes_once(output_root / "COMMITTED", _json_bytes(committed))
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
        help="validate sources and canonical six-output identities without writing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_only:
        preflight = preflight_all_sources(args.source_root)
        public = {
            key: value
            for key, value in preflight.items()
            if key != "contexts"
        }
        public["sources"] = {
            dataset: preflight["contexts"][dataset]["binding"]
            for dataset in DATASETS
        }
        print(json.dumps(public, ensure_ascii=False, sort_keys=True), flush=True)
        return
    result = export_current_bundle(args.source_root, args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "BUNDLE_SCHEMA",
    "CANONICAL_INTRAOP_THREADS",
    "COMMITTED_SCHEMA",
    "DATASETS",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "MANIFEST_SCHEMA",
    "PACKAGE_SCHEMA",
    "canonical_synthetic_replay_context",
    "export_current_bundle",
    "load_exported_current_model_v3",
    "preflight_all_sources",
    "synthetic_replay_execution_contract",
    "validate_committed_bundle",
    "validate_current_source_v3",
    "validate_exported_current_package_v3",
]
