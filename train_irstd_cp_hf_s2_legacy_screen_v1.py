#!/usr/bin/env python3
"""Fail-closed transaction adapter for the CP-HF-S2 legacy screen.

The adapter binds the independently audited CP-HF-S2 architecture to the
existing strict transaction engine.  Formal runs retain a 1000-epoch
schedule identity but exit cleanly immediately after the atomic epoch-500
commit; smoke transactions are deliberately exempt from that pause.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import io
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "IRSTD_CP_HF_S2_LEGACY_SCREEN_V1_AMENDMENT.md"
)
RULES_PATH = (
    PROJECT_ROOT / "experiments" / "irstd_cp_hf_s2_legacy_screen_v1_rules.json"
)
ARCHITECTURE_PATH = PROJECT_ROOT / "experiments" / "irstd_cp_hf_s2_v1.py"
ARCHITECTURE_TEST_PATH = PROJECT_ROOT / "tests" / "test_irstd_cp_hf_s2_v1.py"
SELECTOR_PATH = (
    PROJECT_ROOT / "experiments" / "evisirst_zero_margin_selection.py"
)
TRANSACTION_ENGINE_PATH = PROJECT_ROOT / "train_validation_selected.py"
CANONICAL_SPLIT_ROOT = PROJECT_ROOT / "splits" / "v2"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "irstd_model_design"
    / "cp_hf_s2_v1"
    / "legacy_screen"
)

RULES_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen_rules/v1"
TRAINING_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen_training/v1"
CANDIDATE_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen_candidate/v1"
CHECKPOINT_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen_checkpoint/v1"
HISTORY_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen_history/v1"
SOURCE_SET_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen_source_set/v1"
DETERMINISM_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen_determinism/v1"
EXPERIMENT_SCHEMA = "evisirst_irstd_cp_hf_s2_legacy_screen/v1"

DATASET = "IRSTD-1K"
TARGET_MODE = "binary"
ARCHITECTURE_SEED = 42
FORMAL_RUN_SEEDS = (42, 1_446_202_191, 104_728_269)
FORMAL_EPOCHS = 1000
SCREEN_EPOCH = 500
EPOCH500_CLEAN_MARKER = "EPOCH500_CLEAN"
FORMAL_BATCH_SIZE = 16
FORMAL_WORKERS = 0
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_VAL_INTERVAL = 1
ARCHITECTURE_MODULE = "experiments.irstd_cp_hf_s2_v1"
ARCHITECTURE_VARIANT = "cp_hf_s2_v1"
ARCHITECTURE_LABEL = "EviSIRST-CP-HF-S2-v1"
ARCHITECTURE_API = (
    "build_irstd_cp_hf_s2_v1",
    "validate_irstd_cp_hf_s2_v1",
)

# The adapter below binds the landed architecture to the audited transaction
# engine; it does not duplicate or import the discarded reference package.
EXECUTION_SEALED = True


class CPHFS2LegacyScreenError(ValueError):
    """The request or artifact violates the preregistered contract."""


class CPHFS2ArchitectureNotReady(RuntimeError):
    """The independent architecture source/API has not landed yet."""


class CPHFS2ExecutionNotSealed(RuntimeError):
    """The GPU transaction adapter is intentionally unavailable."""


class CPHFS2AtomicScreenPause(RuntimeError):
    """Internal control signal raised before formal epoch 501 can begin."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CPHFS2LegacyScreenError("value is not strict canonical JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_repo_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CPHFS2LegacyScreenError(f"{label} is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise CPHFS2LegacyScreenError(f"{label} escapes the repository") from exc
    return resolved


def _strict_json_file(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_repo_file(path, label=label)

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=finite_float,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CPHFS2LegacyScreenError(f"{label} is malformed") from exc
    if not isinstance(value, dict):
        raise CPHFS2LegacyScreenError(f"{label} root must be an object")
    _canonical_bytes(value)
    return value


def load_rules() -> dict[str, Any]:
    rules = _strict_json_file(RULES_PATH, label="legacy-screen rules")
    run = rules.get("legacy_screen_run")
    comparison = rules.get("comparison")
    smoke = rules.get("smoke_contract")
    disclosure = rules.get("disclosure_contract")
    mechanism = rules.get("cp_hf_s2_mechanism_diagnostic")
    stage1 = rules.get("stage1_route_gate")
    architecture = rules.get("architecture")
    split = rules.get("split_contract")
    transaction = rules.get("transaction")
    later = rules.get("later_confirmation")
    if (
        rules.get("schema") != RULES_SCHEMA
        or rules.get("protocol_relative_path")
        != "experiments/IRSTD_CP_HF_S2_LEGACY_SCREEN_V1_AMENDMENT.md"
        or not isinstance(run, Mapping)
        or tuple(run.get("run_seeds", ())) != FORMAL_RUN_SEEDS
        or run.get("configured_total_epochs") != FORMAL_EPOCHS
        or run.get("operational_pause_after_atomic_epoch") != SCREEN_EPOCH
        or run.get("screen_records_after_epoch_500_allowed") is not False
        or run.get("runner_enforced_pause_marker") != EPOCH500_CLEAN_MARKER
        or run.get("epoch_501_entry_forbidden") is not True
        or run.get("resume_at_epoch_500_returns_clean_pause") is not True
        or run.get("dataset") != DATASET
        or run.get("target_mode") != TARGET_MODE
        or run.get("architecture_seed") != ARCHITECTURE_SEED
        or run.get("batch_size") != FORMAL_BATCH_SIZE
        or run.get("workers") != FORMAL_WORKERS
        or run.get("optimizer") != "Adam"
        or run.get("optimizer_parameter_groups") != 1
        or run.get("base_lr") != FORMAL_BASE_LR
        or run.get("min_lr") != FORMAL_MIN_LR
        or run.get("warmup_epochs") != FORMAL_WARMUP_EPOCHS
        or run.get("validation_interval") != FORMAL_VAL_INTERVAL
        or run.get("loss") != "sum_of_six_BCELoss_mean_terms"
        or run.get("initialization") != "full_model_scratch"
        or run.get("all_parameters_trainable") is not True
        or run.get("selection_rule")
        != "evisirst_zero_margin_dual_role_lexicographic/v1"
        or run.get("output_root_relative_path")
        != "runs/irstd_model_design/cp_hf_s2_v1/legacy_screen"
        or run.get("public_test_supported") is not False
        or run.get("selection_margin_raw") is not None
        or run.get("selection_window_applied") is not False
        or run.get("test_split_accessed") is not False
        or not isinstance(comparison, Mapping)
        or comparison.get("schema")
        != "evisirst_irstd_cp_hf_s2_first500_ab_gate/v1"
        or tuple(comparison.get("candidate_registry", ()))
        != ("psbfr_v1", "cp_hf_s2_v1")
        or comparison.get("candidate_decisions_are_independent") is not True
        or comparison.get("candidate_ranking_or_winner_selection") is not False
        or comparison.get("s2_route_ledger_implemented_by_this_amendment")
        is not False
        or comparison.get("s2_launch_authorized_by_this_amendment") is not False
        or comparison.get("combined_writer_sealed") is not True
        or comparison.get("sealed_helper_source_relative_path")
        != "run_irstd_cp_hf_s2_first500_ab_gate_v1.py"
        or comparison.get("sealed_helper_source_sha256")
        != _sha256_file(PROJECT_ROOT / "run_irstd_cp_hf_s2_first500_ab_gate_v1.py")
        or comparison.get("future_s2_policy")
        != "new_additive_addendum_gate_and_watcher_without_modifying_stage1_bound_sources"
        or comparison.get("both_candidates_go_policy")
        != "retain_both_without_winner_and_require_new_preregistered_confirmation_selection_amendment"
        or comparison.get("post_hoc_seed_selection") is not False
        or comparison.get("post_hoc_candidate_selection") is not False
        or comparison.get("mean_delta_mIoU_minimum") != 0.002
        or comparison.get("positive_delta_required_for_every_pair") is not True
        or comparison.get("safety_failure")
        != {"operator": "delta_Pd < -0.003 and delta_Fa > 0"}
        or comparison.get("same_seed_pair_required") is not True
        or comparison.get("screen_epoch_inclusive") != SCREEN_EPOCH
        or comparison.get("output_create")
        != "sealed_reserved_path_must_remain_absent"
        or not isinstance(smoke, Mapping)
        or smoke.get("explicit_mode_requires_both_sample_caps") is not True
        or smoke.get("promotion_eligible") is not False
        or smoke.get("epoch_500_gate_eligible") is not False
        or smoke.get("test_split_accessed") is not False
        or disclosure != _strict_false_disclosures()
        or not isinstance(mechanism, Mapping)
        or mechanism.get("schema")
        != "evisirst_irstd_cp_hf_s2_selected_mechanism_diagnostic/v1"
        or tuple(mechanism.get("run_seeds", ())) != FORMAL_RUN_SEEDS
        or mechanism.get("validation_passes_per_selected_checkpoint") != 1
        or mechanism.get("required_checks")
        != [
            "adapter_executed_for_every_validation_sample",
            "correction_nonzero",
            "correction_finite",
            "absolute_correction_bound_holds",
        ]
        or mechanism.get("device") != "cuda:0"
        or mechanism.get("source_relative_path")
        != "run_irstd_cp_hf_s2_mechanism_diagnostic_v1.py"
        or mechanism.get("source_sha256")
        != _sha256_file(PROJECT_ROOT / "run_irstd_cp_hf_s2_mechanism_diagnostic_v1.py")
        or mechanism.get("output_relative_path_template")
        != "runs/irstd_model_design/cp_hf_s2_v1/legacy_screen/comparison/mechanism/run_seed_{seed}/result.json"
        or mechanism.get("diagnostic_only") is not True
        or mechanism.get("selection_allowed") is not False
        or mechanism.get("threshold_tuning_allowed") is not False
        or mechanism.get("lockbox_accessed") is not False
        or mechanism.get("test_selected") is not False
        or mechanism.get("test_selection_supported") is not False
        or mechanism.get("public_test_supported") is not False
        or mechanism.get("public_test_allowed") is not False
        or mechanism.get("test_split_accessed") is not False
        or not isinstance(stage1, Mapping)
        or stage1.get("schema")
        != "evisirst_irstd_model_route_stage1_seed42/v1"
        or stage1.get("source_relative_path")
        != "run_irstd_cp_hf_s2_stage1_route_gate_v1.py"
        or stage1.get("source_sha256")
        != _sha256_file(PROJECT_ROOT / "run_irstd_cp_hf_s2_stage1_route_gate_v1.py")
        or tuple(stage1.get("candidate_registry", ()))
        != ("psbfr_v1", "cp_hf_s2_v1")
        or stage1.get("architecture_seed") != ARCHITECTURE_SEED
        or stage1.get("run_seed") != 42
        or stage1.get("decision_per_route")
        != "GO_if_delta_mIoU_gt_0_and_no_safety_failure_and_route_mechanism_passes_else_STOP"
        or stage1.get("candidate_decisions_are_independent") is not True
        or stage1.get("candidate_ranking_or_winner_selection") is not False
        or stage1.get("missing_route_does_not_block_other_route") is not True
        or tuple(stage1.get("go_authorizes_only_run_seeds", ()))
        != FORMAL_RUN_SEEDS[1:]
        or stage1.get("stop_authorizes_no_further_route_runs") is not True
        or stage1.get("output_relative_path_template")
        != "runs/irstd_model_design/cp_hf_s2_v1/legacy_screen/comparison/stage1/{candidate}/seed42_interim.json"
        or any(stage1.get(name) is not False for name in _strict_false_disclosures())
        or stage1.get("public_test_allowed") is not False
        or not isinstance(architecture, Mapping)
        or architecture.get("module") != ARCHITECTURE_MODULE
        or architecture.get("source_relative_path")
        != "experiments/irstd_cp_hf_s2_v1.py"
        or architecture.get("contract_tests_relative_path")
        != "tests/test_irstd_cp_hf_s2_v1.py"
        or architecture.get("builder") != ARCHITECTURE_API[0]
        or architecture.get("validator") != ARCHITECTURE_API[1]
        or architecture.get("variant_key") != ARCHITECTURE_VARIANT
        or architecture.get("label") != ARCHITECTURE_LABEL
        or architecture.get("base_state_key_count") != 564
        or architecture.get("extension_state_key_count") != 12
        or architecture.get("state_key_count") != 576
        or architecture.get("parameter_count") != 10_874_615
        or architecture.get("extension_state_prefix") != "decoder_cp_hf_s2."
        or not all(
            isinstance(architecture.get(name), str)
            and len(architecture[name]) == 64
            and all(character in "0123456789abcdef" for character in architecture[name])
            for name in ("source_sha256", "contract_tests_sha256")
        )
        or not isinstance(split, Mapping)
        or split.get("root_relative_path") != "splits/v2"
        or split.get("dataset_relative_path") != "splits/v2/IRSTD-1K"
        or split.get("split_seed") != 20260811
        or split.get("train_count") != 640
        or split.get("val_count") != 160
        or split.get("test_was_not_accessed") is not True
        or not isinstance(transaction, Mapping)
        or transaction.get("resume_filename") != "last_training_state.pth.tar"
        or transaction.get("resume_load") != "strict"
        or transaction.get("resume_identity_must_equal_fresh_identity") is not True
        or transaction.get("epoch_500_gate_requires_atomic_resume_and_candidate_commit")
        is not True
        or transaction.get("formal_entry_forbids_existing_summary_or_any_final")
        is not True
        or transaction.get("formal_entry_existing_latest_maximum_epoch")
        != SCREEN_EPOCH
        or transaction.get("formal_entry_rechecked_inside_process_lock") is not True
        or transaction.get("actual_restored_start_epoch_maximum")
        != SCREEN_EPOCH + 1
        or transaction.get("formal_completed_finalize_forbidden") is not True
        or transaction.get("output_parent_traversal")
        != "held_dirfd_O_DIRECTORY_O_NOFOLLOW"
        or transaction.get("candidate_and_checkpoint_commit")
        != "exclusive_linkat_no_replace"
        or transaction.get("every_artifact_requires_test_split_accessed_false")
        is not True
        or not isinstance(later, Mapping)
        or later.get("implemented_by_this_amendment") is not False
        or later.get("launch_allowed_before_both_candidate_screens_terminal")
        is not False
        or later.get("required_paired_initialization_count") != 5
        or later.get("configured_total_epochs") != FORMAL_EPOCHS
    ):
        raise CPHFS2LegacyScreenError("legacy-screen rules identity differs")
    return json.loads(_canonical_bytes(rules).decode("ascii"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-seed", type=int, choices=FORMAL_RUN_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-max-train-samples", type=int)
    parser.add_argument("--smoke-max-val-samples", type=int)
    args = parser.parse_args(argv)
    caps = (args.smoke_max_train_samples, args.smoke_max_val_samples)
    if (caps[0] is None) != (caps[1] is None):
        parser.error("smoke mode requires both sample caps")
    if any(value is not None and value < 1 for value in caps):
        parser.error("smoke sample caps must be positive")
    smoke = caps[0] is not None
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if smoke:
        if args.warmup_epochs is None:
            args.warmup_epochs = min(FORMAL_WARMUP_EPOCHS, args.epochs)
        if not 0 <= args.warmup_epochs <= args.epochs:
            parser.error("smoke warmup must be in [0, epochs]")
    else:
        if args.epochs != FORMAL_EPOCHS:
            parser.error(f"formal legacy screen requires --epochs {FORMAL_EPOCHS}")
        if args.warmup_epochs not in (None, FORMAL_WARMUP_EPOCHS):
            parser.error(
                f"formal legacy screen requires --warmup-epochs {FORMAL_WARMUP_EPOCHS}"
            )
        args.warmup_epochs = FORMAL_WARMUP_EPOCHS
    args.dataset = DATASET
    args.target_mode = TARGET_MODE
    args.architecture_seed = ARCHITECTURE_SEED
    args.batch_size = FORMAL_BATCH_SIZE
    args.workers = FORMAL_WORKERS
    args.base_lr = FORMAL_BASE_LR
    args.min_lr = FORMAL_MIN_LR
    args.val_interval = FORMAL_VAL_INTERVAL
    args.split_root = CANONICAL_SPLIT_ROOT
    args.output_root = DEFAULT_OUTPUT_ROOT
    args.variant = ARCHITECTURE_VARIANT
    args.allow_sample_level_fallback = True
    try:
        require_args(args)
    except CPHFS2LegacyScreenError as exc:
        parser.error(str(exc))
    return args


def require_args(args: argparse.Namespace) -> None:
    exact = {
        "dataset": DATASET,
        "target_mode": TARGET_MODE,
        "architecture_seed": ARCHITECTURE_SEED,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "val_interval": FORMAL_VAL_INTERVAL,
    }
    for name, expected in exact.items():
        if getattr(args, name, None) != expected:
            raise CPHFS2LegacyScreenError(f"legacy screen freezes {name}={expected!r}")
    if type(getattr(args, "run_seed", None)) is not int or args.run_seed not in FORMAL_RUN_SEEDS:
        raise CPHFS2LegacyScreenError("run seed is outside the frozen registry")
    if Path(args.split_root).resolve() != CANONICAL_SPLIT_ROOT.resolve():
        raise CPHFS2LegacyScreenError("split root is not repository splits/v2")
    if Path(args.output_root).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise CPHFS2LegacyScreenError("output root differs from the fixed tree")
    caps = (
        getattr(args, "smoke_max_train_samples", None),
        getattr(args, "smoke_max_val_samples", None),
    )
    if (caps[0] is None) != (caps[1] is None):
        raise CPHFS2LegacyScreenError("smoke mode requires both sample caps")
    if any(value is not None and (type(value) is not int or value < 1) for value in caps):
        raise CPHFS2LegacyScreenError("smoke sample caps must be positive integers")
    smoke = caps[0] is not None
    if smoke:
        if (
            type(args.epochs) is not int
            or args.epochs < 1
            or type(args.warmup_epochs) is not int
            or not 0 <= args.warmup_epochs <= args.epochs
        ):
            raise CPHFS2LegacyScreenError("smoke epoch/warmup contract differs")
    elif (
        args.epochs != FORMAL_EPOCHS
        or args.warmup_epochs != FORMAL_WARMUP_EPOCHS
        or args.device != "cuda:0"
    ):
        raise CPHFS2LegacyScreenError(
            "formal legacy screen requires cuda:0 and the 1000-epoch recipe"
        )
    load_rules()


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path]:
    require_args(args)
    smoke = args.smoke_max_train_samples is not None
    run_dir = (
        DEFAULT_OUTPUT_ROOT
        / ("smoke" if smoke else "formal")
        / DATASET
        / TARGET_MODE
        / f"run_seed_{args.run_seed}"
    )
    if smoke:
        smoke_identity = {
            "epochs": args.epochs,
            "val_interval": args.val_interval,
            "max_train": args.smoke_max_train_samples,
            "max_val": args.smoke_max_val_samples,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "warmup_epochs": args.warmup_epochs,
        }
        suffix = hashlib.sha256(_canonical_bytes(smoke_identity)).hexdigest()[:12]
        run_dir = run_dir / f"smoke_{suffix}"
    return {
        "run_dir": run_dir,
        "latest": run_dir / "last_training_state.pth.tar",
        "history": run_dir / "validation_history.json",
        "candidate_dir": run_dir / "candidates",
        "best_mIoU_final": run_dir / "EviSIRST_best_mIoU.pth.tar",
        "best_Pd_final": run_dir / "EviSIRST_best_Pd.pth.tar",
        "summary": run_dir / "summary.json",
    }


def split_provenance() -> dict[str, Any]:
    rules = load_rules()["split_contract"]
    directory = CANONICAL_SPLIT_ROOT / DATASET
    paths = {
        "manifest": directory / "manifest.json",
        "train": directory / "train.txt",
        "val": directory / "val.txt",
    }
    expected = {
        "manifest": rules["manifest_sha256"],
        "train": rules["train_index_sha256"],
        "val": rules["val_index_sha256"],
    }
    artifacts: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = _regular_repo_file(path, label=f"split {name}")
        observed = _sha256_file(resolved)
        if observed != expected[name]:
            raise CPHFS2LegacyScreenError(f"split {name} SHA-256 differs")
        artifacts[name] = {
            "relative_path": resolved.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": observed,
        }
    if len(paths["train"].read_text(encoding="utf-8").splitlines()) != 640:
        raise CPHFS2LegacyScreenError("train split count differs")
    if len(paths["val"].read_text(encoding="utf-8").splitlines()) != 160:
        raise CPHFS2LegacyScreenError("validation split count differs")
    manifest = _strict_json_file(paths["manifest"], label="split manifest")
    if (
        manifest.get("seeds", {}).get("split_seed") != 20260811
        or manifest.get("data_identity", {}).get("ordered_image_mask_tree_sha256")
        != rules["data_tree_sha256"]
        or manifest.get("test_access", {}).get("test_index_opened") is not False
    ):
        raise CPHFS2LegacyScreenError("split manifest contract differs")
    return {
        "schema": "evisirst_cp_hf_s2_split_seal/v1",
        "files": artifacts,
        "data_tree_sha256": rules["data_tree_sha256"],
        "split_seed": 20260811,
        "train_count": 640,
        "val_count": 160,
        "test_split_accessed": False,
    }


def architecture_api() -> dict[str, Any]:
    if not ARCHITECTURE_PATH.exists():
        raise CPHFS2ArchitectureNotReady(
            "experiments/irstd_cp_hf_s2_v1.py has not landed"
        )
    source = _regular_repo_file(ARCHITECTURE_PATH, label="CP-HF-S2 architecture")
    expected_sha256 = load_rules()["architecture"]["source_sha256"]
    if _sha256_file(source) != expected_sha256:
        raise CPHFS2ArchitectureNotReady("CP-HF-S2 architecture SHA-256 differs")
    tests = _regular_repo_file(
        ARCHITECTURE_TEST_PATH, label="CP-HF-S2 architecture contract tests"
    )
    if _sha256_file(tests) != load_rules()["architecture"]["contract_tests_sha256"]:
        raise CPHFS2ArchitectureNotReady("CP-HF-S2 contract-test SHA-256 differs")
    try:
        module = importlib.import_module(ARCHITECTURE_MODULE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise CPHFS2ArchitectureNotReady("CP-HF-S2 architecture import failed") from exc
    api: dict[str, Any] = {"module": module}
    for name in ARCHITECTURE_API:
        value = getattr(module, name, None)
        if not callable(value):
            raise CPHFS2ArchitectureNotReady(f"architecture API is missing {name}")
        api[name] = value
    return api


def source_provenance(*, require_architecture: bool = True) -> dict[str, Any]:
    paths = {
        "screen_runner": Path(__file__),
        "screen_amendment": PROTOCOL_PATH,
        "screen_rules": RULES_PATH,
        "zero_margin_selector": SELECTOR_PATH,
        "transaction_engine": TRANSACTION_ENGINE_PATH,
    }
    if require_architecture:
        architecture_api()
        paths["selected_architecture"] = ARCHITECTURE_PATH
        paths["architecture_contract_tests"] = ARCHITECTURE_TEST_PATH
    files: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = _regular_repo_file(path, label=name)
        files[name] = {
            "relative_path": resolved.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256_file(resolved),
        }
    return {
        "schema": SOURCE_SET_SCHEMA,
        "files": files,
        "source_tree_sha256": _canonical_sha256(files),
    }


_FALSE_DISCLOSURES = {
    "lockbox_accessed",
    "test_split_accessed",
    "test_selected",
    "test_selection_supported",
    "public_test_allowed",
    "public_test_supported",
    "test_index_opened",
}


def _strict_false_disclosures() -> dict[str, bool]:
    return {
        "lockbox_accessed": False,
        "test_selected": False,
        "test_selection_supported": False,
        "public_test_supported": False,
        "test_split_accessed": False,
    }


def require_test_false(value: Any, *, label: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in _FALSE_DISCLOSURES and nested is not False:
                raise CPHFS2LegacyScreenError(f"{label}.{key} must be false")
            require_test_false(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            require_test_false(nested, label=f"{label}[{index}]")


def validate_resume_envelope(
    payload: Any,
    *,
    expected_identity: Mapping[str, Any],
    expected_state_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CPHFS2LegacyScreenError("resume payload is not a mapping")
    required = set(load_rules()["transaction"]["resume_required_state"])
    if not required.issubset(payload):
        raise CPHFS2LegacyScreenError("resume payload is incomplete")
    if (
        payload.get("schema") != TRAINING_SCHEMA
        or payload.get("run_identity") != expected_identity
        or any(payload.get(name) is not False for name in _strict_false_disclosures())
    ):
        raise CPHFS2LegacyScreenError("resume identity differs")
    epoch = payload.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= FORMAL_EPOCHS:
        raise CPHFS2LegacyScreenError("resume epoch differs")
    if len(payload.get("training_history", ())) != epoch:
        raise CPHFS2LegacyScreenError("resume training history frontier differs")
    if len(payload.get("validation_history", ())) != epoch:
        raise CPHFS2LegacyScreenError("resume validation history frontier differs")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise CPHFS2LegacyScreenError("resume model state is missing")
    if expected_state_keys is not None and set(state) != expected_state_keys:
        raise CPHFS2LegacyScreenError("resume model strict key set differs")
    require_test_false(payload, label="resume")
    return payload


def validate_candidate_envelope(
    payload: Any,
    *,
    expected_identity: Mapping[str, Any],
    epoch: int,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CPHFS2LegacyScreenError("candidate payload is not a mapping")
    if (
        payload.get("schema") != CANDIDATE_SCHEMA
        or payload.get("run_identity") != expected_identity
        or payload.get("epoch") != epoch
        or any(payload.get(name) is not False for name in _strict_false_disclosures())
        or not isinstance(payload.get("state_dict"), Mapping)
        or not isinstance(payload.get("validation_record"), Mapping)
        or payload["validation_record"].get("epoch") != epoch
    ):
        raise CPHFS2LegacyScreenError("candidate envelope differs")
    require_test_false(payload, label="candidate")
    return payload


def validate_final_envelope(
    payload: Any,
    *,
    expected_identity: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    if role not in {"best_mIoU", "best_Pd"}:
        raise CPHFS2LegacyScreenError("final role differs")
    if not isinstance(payload, Mapping):
        raise CPHFS2LegacyScreenError("final payload is not a mapping")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("training") != expected_identity
        or payload.get("selection_role") != role
        or any(payload.get(name) is not False for name in _strict_false_disclosures())
        or not isinstance(payload.get("source_candidate"), Mapping)
        or not isinstance(payload.get("selected_complete_key"), Mapping)
        or payload.get("selected_complete_key_sha256")
        != _canonical_sha256(payload["selected_complete_key"])
        or not isinstance(payload.get("selection_provenance"), Mapping)
        or not isinstance(payload.get("state_dict"), Mapping)
    ):
        raise CPHFS2LegacyScreenError("final envelope differs")
    require_test_false(payload, label="final")
    return payload


def _write_bytes_no_clobber(
    path: Path, encoded: bytes, *, allow_identical: bool
) -> Path:
    """Create through held directory FDs without following mutable parents."""

    if not isinstance(encoded, bytes):
        raise TypeError("encoded output must be bytes")
    path = Path(os.path.abspath(path))
    root = PROJECT_ROOT.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CPHFS2LegacyScreenError("output escapes repository") from exc
    if not relative.parts:
        raise CPHFS2LegacyScreenError("output must be a repository child")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    file_read_flags = os.O_RDONLY
    file_write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for optional in ("O_CLOEXEC", "O_NOFOLLOW"):
        value = getattr(os, optional, 0)
        directory_flags |= value
        file_read_flags |= value
        file_write_flags |= value
    file_read_flags |= getattr(os, "O_NONBLOCK", 0)
    descriptors: list[int] = []
    temporary_name: str | None = None

    def read_existing(parent_fd: int, name: str) -> bytes:
        try:
            descriptor = os.open(name, file_read_flags, dir_fd=parent_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise FileExistsError(f"no-clobber artifact is unsafe: {path}") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise FileExistsError(f"no-clobber artifact is not regular: {path}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(descriptor)

    try:
        current_fd = os.open(root, directory_flags)
        descriptors.append(current_fd)
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                except OSError as exc:
                    raise CPHFS2LegacyScreenError(
                        "output parent is unsafe after creation"
                    ) from exc
            except OSError as exc:
                raise CPHFS2LegacyScreenError(
                    "output parent must be a real directory"
                ) from exc
            descriptors.append(next_fd)
            current_fd = next_fd

        final_name = relative.parts[-1]
        try:
            existing = read_existing(current_fd, final_name)
        except FileNotFoundError:
            existing = None
        except FileExistsError:
            raise
        if existing is not None:
            if allow_identical and existing == encoded:
                return path
            raise FileExistsError(f"no-clobber artifact differs: {path}")

        for _attempt in range(128):
            candidate = f".{final_name}.{secrets.token_hex(12)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    file_write_flags,
                    0o600,
                    dir_fd=current_fd,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        else:
            raise CPHFS2LegacyScreenError("could not allocate exclusive temp output")
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=current_fd,
                dst_dir_fd=current_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            if not allow_identical or read_existing(current_fd, final_name) != encoded:
                raise FileExistsError(f"no-clobber artifact exists: {path}") from exc
        os.unlink(temporary_name, dir_fd=current_fd)
        temporary_name = None
        os.fsync(current_fd)
    finally:
        if temporary_name is not None and descriptors:
            try:
                os.unlink(temporary_name, dir_fd=descriptors[-1])
            except FileNotFoundError:
                pass
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return path


def write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create once, or verify an existing byte-identical canonical JSON file."""

    require_test_false(payload)
    content = json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    return _write_bytes_no_clobber(
        path, content.encode("utf-8"), allow_identical=True
    )


def _validate_current_source_tree(identity: Mapping[str, Any]) -> None:
    determinism = identity.get("determinism_protocol")
    if not isinstance(determinism, Mapping):
        raise CPHFS2LegacyScreenError("training determinism identity is missing")
    files = determinism.get("source_files")
    if not isinstance(files, Mapping):
        raise CPHFS2LegacyScreenError("training source manifest is missing")
    required = {
        "cp_hf_s2_screen_wrapper",
        "architecture_contract_tests",
        "selected_architecture",
        "screen_runner",
        "transaction_adapter",
        "zero_margin_selector",
        "frozen_protocol",
        "frozen_screen_rules",
    }
    if not required.issubset(files):
        raise CPHFS2LegacyScreenError("training source manifest is incomplete")
    normalized: dict[str, Any] = {}
    for name, artifact in files.items():
        if not isinstance(name, str) or not isinstance(artifact, Mapping):
            raise CPHFS2LegacyScreenError("training source entry is malformed")
        relative = artifact.get("relative_path")
        sha256 = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(sha256, str):
            raise CPHFS2LegacyScreenError("training source entry is incomplete")
        path = _regular_repo_file(PROJECT_ROOT / relative, label=f"source {name}")
        observed = _sha256_file(path)
        if observed != sha256:
            raise CPHFS2LegacyScreenError(f"training source changed: {name}")
        normalized[name] = {"relative_path": relative, "sha256": observed}
    if (
        normalized != dict(files)
        or determinism.get("source_tree_sha256") != _canonical_sha256(normalized)
    ):
        raise CPHFS2LegacyScreenError("training source tree identity differs")
    if files["cp_hf_s2_screen_wrapper"] != {
        "relative_path": Path(__file__).resolve(strict=True).relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256_file(Path(__file__).resolve(strict=True)),
    }:
        raise CPHFS2LegacyScreenError("training wrapper source binding differs")


def _validate_atomic_epoch500_pause(args: argparse.Namespace) -> Path:
    """Re-read the committed frontier before reporting a clean pause."""

    require_args(args)
    if args.smoke_max_train_samples is not None:
        raise CPHFS2LegacyScreenError("smoke transaction cannot enter formal pause")
    paths = resolve_run_paths(args)
    run_dir = paths["run_dir"]
    latest_path = paths["latest"]
    history_path = paths["history"]
    candidate_dir = paths["candidate_dir"]
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise CPHFS2LegacyScreenError("formal run directory is malformed")
    for forbidden in (
        paths["summary"],
        paths["best_mIoU_final"],
        paths["best_Pd_final"],
        run_dir / "EviSIRST.pth.tar",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise CPHFS2LegacyScreenError(
                "epoch-500 pause unexpectedly contains a final artifact"
            )
    latest_path = _regular_repo_file(latest_path, label="epoch-500 latest")
    history_path = _regular_repo_file(history_path, label="epoch-500 history")
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise CPHFS2LegacyScreenError("candidate directory is malformed")

    torch_module = importlib.import_module("torch")
    payload = torch_module.load(latest_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise CPHFS2LegacyScreenError("epoch-500 latest is malformed")
    identity = payload.get("run_identity")
    if not isinstance(identity, Mapping):
        raise CPHFS2LegacyScreenError("epoch-500 training identity is missing")
    if (
        identity.get("schema") != TRAINING_SCHEMA + "/run_identity"
        or identity.get("architecture_variant") != ARCHITECTURE_LABEL
        or identity.get("variant_key") != ARCHITECTURE_VARIANT
        or identity.get("dataset") != DATASET
        or identity.get("target_mode") != TARGET_MODE
        or identity.get("architecture_seed") != ARCHITECTURE_SEED
        or identity.get("run_seed") != args.run_seed
        or identity.get("epochs") != FORMAL_EPOCHS
        or identity.get("formal_configured_total_epochs") != FORMAL_EPOCHS
        or identity.get("train_count") != 640
        or identity.get("val_count") != 160
        or identity.get("smoke") is not False
        or identity.get("promotion_eligible") is not True
        or identity.get("manifest_sha256")
        != load_rules()["split_contract"]["manifest_sha256"]
        or identity.get("data_tree_sha256")
        != load_rules()["split_contract"]["data_tree_sha256"]
        or any(identity.get(name) is not False for name in _strict_false_disclosures())
    ):
        raise CPHFS2LegacyScreenError("epoch-500 training identity differs")
    unhashed_identity = dict(identity)
    observed_identity_sha256 = unhashed_identity.pop("identity_sha256", None)
    if observed_identity_sha256 != _canonical_sha256(unhashed_identity):
        raise CPHFS2LegacyScreenError("epoch-500 identity SHA-256 differs")
    _validate_current_source_tree(identity)

    validate_resume_envelope(payload, expected_identity=identity)
    if payload.get("epoch") != SCREEN_EPOCH:
        raise CPHFS2LegacyScreenError("clean pause is not exactly epoch 500")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise CPHFS2LegacyScreenError("epoch-500 state is missing")
    api = architecture_api()
    authority_model, _ = api["build_irstd_cp_hf_s2_v1"](
        DATASET, seed=ARCHITECTURE_SEED, training=True
    )
    api["validate_irstd_cp_hf_s2_v1"](
        authority_model, require_identity_initialization=True
    )
    authority_state = authority_model.state_dict()
    expected_kernel = api["module"]._binomial_kernel_5()

    def validate_state(candidate_state: Any, *, label: str) -> None:
        if not isinstance(candidate_state, Mapping) or set(candidate_state) != set(
            authority_state
        ):
            raise CPHFS2LegacyScreenError(f"{label} state-key contract differs")
        for key, tensor in candidate_state.items():
            reference = authority_state[key]
            if (
                not isinstance(tensor, torch_module.Tensor)
                or tensor.shape != reference.shape
                or tensor.dtype != reference.dtype
                or (tensor.is_floating_point() and not bool(torch_module.isfinite(tensor).all()))
            ):
                raise CPHFS2LegacyScreenError(f"{label} tensor differs: {key}")
        kernel = candidate_state.get("decoder_cp_hf_s2.low_pass_kernel")
        if not isinstance(kernel, torch_module.Tensor) or not torch_module.equal(
            kernel, expected_kernel.to(dtype=kernel.dtype)
        ):
            raise CPHFS2LegacyScreenError(f"{label} fixed kernel differs")

    validate_state(state, label="epoch-500")

    training_history = payload.get("training_history")
    validation_history = payload.get("validation_history")
    if (
        not isinstance(training_history, list)
        or not isinstance(validation_history, list)
        or [record.get("epoch") for record in training_history] != list(range(1, 501))
        or [record.get("epoch") for record in validation_history] != list(range(1, 501))
    ):
        raise CPHFS2LegacyScreenError("epoch-500 histories are not exact")

    artifacts = payload.get("candidate_artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise CPHFS2LegacyScreenError("epoch-500 candidate frontier is missing")
    referenced: set[Path] = set()
    for raw_epoch, artifact in artifacts.items():
        if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int):
            raise CPHFS2LegacyScreenError("candidate frontier epoch is malformed")
        if not isinstance(artifact, Mapping):
            raise CPHFS2LegacyScreenError("candidate frontier entry is malformed")
        relative = artifact.get("relative_path")
        if relative != f"candidates/epoch_{raw_epoch:04d}.pth.tar":
            raise CPHFS2LegacyScreenError("candidate frontier path differs")
        path = _regular_repo_file(run_dir / relative, label="retained candidate")
        if _sha256_file(path) != artifact.get("file_sha256"):
            raise CPHFS2LegacyScreenError("candidate frontier SHA-256 differs")
        candidate = torch_module.load(path, map_location="cpu", weights_only=True)
        validate_candidate_envelope(candidate, expected_identity=identity, epoch=raw_epoch)
        if candidate.get("validation_record") != validation_history[raw_epoch - 1]:
            raise CPHFS2LegacyScreenError("candidate validation record differs")
        validate_state(candidate["state_dict"], label="candidate")
        referenced.add(path)
    selector = importlib.import_module(
        "experiments.evisirst_zero_margin_selection"
    )
    expected_frontier = selector.retention_frontier_epochs(validation_history)
    if tuple(sorted(artifacts)) != expected_frontier:
        raise CPHFS2LegacyScreenError("candidate retention frontier differs")
    observed_candidates = {
        _regular_repo_file(path, label="candidate directory entry")
        for path in candidate_dir.iterdir()
    }
    if observed_candidates != referenced:
        raise CPHFS2LegacyScreenError("candidate cleanup frontier differs")

    history = _strict_json_file(history_path, label="epoch-500 history")
    expected_history = {
        "schema": HISTORY_SCHEMA,
        "data_role": "val",
        "run_identity": dict(identity),
        "training_history": training_history,
        "validation_history": validation_history,
        "retention_frontier_epochs": sorted(artifacts),
        "candidate_artifacts": {
            str(epoch): dict(artifacts[epoch]) for epoch in sorted(artifacts)
        },
    }
    expected_history.update(_strict_false_disclosures())
    for name, value in expected_history.items():
        if history.get(name) != value:
            raise CPHFS2LegacyScreenError(f"epoch-500 history differs: {name}")
    require_test_false(history, label="epoch500_history")
    return latest_path


def _formal_entry_barrier(
    args: argparse.Namespace,
    *,
    paths: Mapping[str, Path] | None = None,
) -> None:
    """Reject every path that could bypass the epoch-501 interception."""

    require_args(args)
    if args.smoke_max_train_samples is not None:
        return
    active_paths = dict(resolve_run_paths(args) if paths is None else paths)
    run_dir = active_paths.get("run_dir")
    latest = active_paths.get("latest")
    if not isinstance(run_dir, Path) or not isinstance(latest, Path):
        raise CPHFS2LegacyScreenError("formal entry paths are malformed")
    forbidden = (
        active_paths.get("summary"),
        active_paths.get("best_mIoU_final"),
        active_paths.get("best_Pd_final"),
        run_dir / "EviSIRST.pth.tar",
    )
    for path in forbidden:
        if not isinstance(path, Path):
            raise CPHFS2LegacyScreenError("formal final path is malformed")
        if path.exists() or path.is_symlink():
            raise CPHFS2LegacyScreenError(
                "formal Stage-1 entry forbids summaries and final checkpoints"
            )
    if not latest.exists() and not latest.is_symlink():
        if args.resume:
            raise CPHFS2LegacyScreenError("formal resume latest is missing")
        return
    latest = _regular_repo_file(latest, label="formal resume latest")
    torch_module = importlib.import_module("torch")
    payload = torch_module.load(latest, map_location="cpu", weights_only=True)
    epoch = payload.get("epoch") if isinstance(payload, Mapping) else None
    if type(epoch) is not int or not 1 <= epoch <= SCREEN_EPOCH:
        raise CPHFS2LegacyScreenError(
            "formal Stage-1 resume epoch must be in [1, 500]"
        )


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    require_args(args)
    paths = resolve_run_paths(args)
    smoke = args.smoke_max_train_samples is not None
    report = {
        "schema": "evisirst_irstd_cp_hf_s2_legacy_screen_preflight/v1",
        "execution_sealed": EXECUTION_SEALED,
        "run_seed": args.run_seed,
        "configured_total_epochs": FORMAL_EPOCHS,
        "requested_epochs": args.epochs,
        "smoke": smoke,
        "promotion_eligible": False if smoke else True,
        "epoch_500_gate_eligible": False if smoke else True,
        "operational_pause_after_atomic_epoch": None if smoke else SCREEN_EPOCH,
        "runner_enforced_pause_marker": None if smoke else EPOCH500_CLEAN_MARKER,
        "epoch_501_entry_forbidden": False if smoke else True,
        "run_paths": {
            name: path.relative_to(PROJECT_ROOT).as_posix()
            for name, path in paths.items()
        },
        "source_provenance": source_provenance(require_architecture=True),
        "split_provenance": split_provenance(),
    }
    report.update(_strict_false_disclosures())
    return report


def _install_atomic_epoch500_pause(train_dataset: Any, *, smoke: bool) -> Any:
    """Prevent formal epoch 501 without changing the 1000-epoch identity.

    The transaction loop calls ``set_epoch`` only at the start of an epoch.
    Therefore the first possible interception point after epoch 500 is the
    request to enter epoch 501.  By then the engine has already written and
    revalidated the epoch-500 resume state and retention-frontier candidates.
    """

    if smoke:
        return train_dataset
    original_set_epoch = getattr(train_dataset, "set_epoch", None)
    if not callable(original_set_epoch):
        raise CPHFS2ExecutionNotSealed("training dataset has no set_epoch API")
    if getattr(train_dataset, "_cp_hf_s2_epoch500_pause_installed", False):
        raise CPHFS2ExecutionNotSealed("epoch-500 pause hook is already installed")

    def set_epoch_with_atomic_pause(epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise CPHFS2LegacyScreenError("dataset epoch must be an integer")
        if epoch > SCREEN_EPOCH:
            raise CPHFS2AtomicScreenPause(
                f"{EPOCH500_CLEAN_MARKER}: formal epoch {epoch} is forbidden"
            )
        original_set_epoch(epoch)

    try:
        setattr(train_dataset, "set_epoch", set_epoch_with_atomic_pause)
        setattr(train_dataset, "_cp_hf_s2_epoch500_pause_installed", True)
    except (AttributeError, TypeError) as exc:
        raise CPHFS2ExecutionNotSealed(
            "training dataset cannot host the epoch-500 pause hook"
        ) from exc
    return train_dataset


@contextmanager
def _screen_transaction_adapter(
    args: argparse.Namespace | None,
) -> Iterator[Any]:
    """Bind CP-HF-S2 to the existing audited legacy-screen transaction."""

    try:
        screen = importlib.import_module("train_irstd_model_design_screen_v1")
    except (ImportError, ModuleNotFoundError) as exc:
        raise CPHFS2ExecutionNotSealed("transaction engine import failed") from exc
    api = architecture_api()
    architecture = api["module"]
    spec = screen.VariantSpec(
        key=ARCHITECTURE_VARIANT,
        label=ARCHITECTURE_LABEL,
        allowed_run_seeds=FORMAL_RUN_SEEDS,
        architecture_source=ARCHITECTURE_PATH,
        state_key_count=architecture.FORMAL_STATE_KEY_COUNT,
        parameter_count=architecture.FORMAL_PARAMETER_COUNT,
        extension_state_key_count=architecture.EXTENSION_STATE_KEY_COUNT,
        extension_state_prefix=architecture.STATE_PREFIX,
        architecture_schema=architecture.ARCHITECTURE_SCHEMA,
        single_variable=(
            "post_d2_context_purified_bounded_high_frequency_feature_refinement"
        ),
    )
    if (
        spec.state_key_count != 576
        or spec.parameter_count != 10_874_615
        or spec.extension_state_key_count != 12
        or spec.extension_state_prefix != "decoder_cp_hf_s2."
    ):
        raise CPHFS2ExecutionNotSealed("landed architecture contract differs")

    original_source = screen._variant_source_provenance
    original_state_contract = screen._state_contract
    original_build_datasets = screen.build_datasets
    original_run_identity = screen._variant_run_identity
    original_validate_state_dict = screen._validate_state_dict
    original_final_checkpoint_payload = screen._variant_final_checkpoint_payload
    original_write_json = screen._variant_write_json
    original_atomic_torch_save = screen._variant_atomic_torch_save
    original_process_lock = screen._run_process_lock
    original_load_resume_state = screen.r1._load_resume_state
    original_evaluate_model = screen.r1.evaluate_model
    original_finalize_completed = screen.hf_transaction._finalize_completed

    def smoke_evaluate_model(*positional: Any, **keywords: Any) -> dict[str, Any]:
        metrics = dict(original_evaluate_model(*positional, **keywords))
        if metrics.get("tiny_pd") is None:
            if metrics.get("tiny_target_count") != 0:
                raise CPHFS2LegacyScreenError(
                    "tiny_pd is null despite a nonzero tiny-target count"
                )
            metrics["tiny_pd"] = 0.0
            metrics["smoke_tiny_pd_imputed_no_tiny_targets"] = True
        else:
            metrics["smoke_tiny_pd_imputed_no_tiny_targets"] = False
        return metrics

    def adapted_rules() -> dict[str, Any]:
        frozen = load_rules()
        comparison = frozen["comparison"]
        return {
            "schema": frozen["schema"],
            "protocol_relative_path": frozen["protocol_relative_path"],
            "legacy_screen": {
                "run_seeds": list(FORMAL_RUN_SEEDS),
                "mean_delta_mIoU_minimum": comparison[
                    "mean_delta_mIoU_minimum"
                ],
                "positive_delta_required_for_every_pair": comparison[
                    "positive_delta_required_for_every_pair"
                ],
                "safety_failure": dict(comparison["safety_failure"]),
            },
        }

    def adapted_source() -> dict[str, Any]:
        source = original_source()
        files = dict(source["files"])
        wrapper = _regular_repo_file(Path(__file__), label="CP-HF-S2 runner")
        files["cp_hf_s2_screen_wrapper"] = {
            "relative_path": wrapper.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256_file(wrapper),
        }
        architecture_tests = _regular_repo_file(
            ARCHITECTURE_TEST_PATH, label="CP-HF-S2 architecture tests"
        )
        files["architecture_contract_tests"] = {
            "relative_path": architecture_tests.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256_file(architecture_tests),
        }
        source["schema"] = SOURCE_SET_SCHEMA
        source["files"] = files
        source["source_tree_sha256"] = _canonical_sha256(files)
        return source

    def adapted_state_contract(active_spec: Any) -> dict[str, Any]:
        contract = original_state_contract(active_spec)
        contract.update(
            {
                "required_forward_binding": "cp_hf_s2_forward_with_relay",
                "insertion_point": (
                    "after_up_decoder2_finish_before_gt2_and_up_decoder1"
                ),
                "feature_evidence_stop_gradient": True,
                "skip_evidence_stop_gradient": True,
                "max_absolute_feature_correction": 0.25,
                "final_logit_bound_claimed": False,
            }
        )
        return contract

    def adapted_build_datasets(
        active_args: argparse.Namespace,
    ) -> tuple[Any, Any, dict[str, Any]]:
        train_dataset, val_dataset, grouping = original_build_datasets(active_args)
        _install_atomic_epoch500_pause(
            train_dataset,
            smoke=active_args.smoke_max_train_samples is not None,
        )
        return train_dataset, val_dataset, grouping

    def adapted_run_identity(*positional: Any, **keywords: Any) -> dict[str, Any]:
        identity = dict(original_run_identity(*positional, **keywords))
        identity.update(_strict_false_disclosures())
        identity.pop("identity_sha256", None)
        identity["identity_sha256"] = _canonical_sha256(identity)
        return identity

    def adapted_validate_state_dict(
        value: Any, expected: Mapping[str, Any]
    ) -> dict[str, Any]:
        state = original_validate_state_dict(value, expected)
        kernel_key = "decoder_cp_hf_s2.low_pass_kernel"
        observed_kernel = state.get(kernel_key)
        expected_kernel = expected.get(kernel_key)
        torch_module = importlib.import_module("torch")
        authority_kernel = architecture._binomial_kernel_5()
        if (
            not isinstance(observed_kernel, torch_module.Tensor)
            or not isinstance(expected_kernel, torch_module.Tensor)
            or not torch_module.equal(
                observed_kernel,
                authority_kernel.to(dtype=observed_kernel.dtype),
            )
            or not torch_module.equal(
                expected_kernel.detach().cpu(),
                authority_kernel.to(dtype=expected_kernel.dtype),
            )
        ):
            raise CPHFS2LegacyScreenError(
                "CP-HF-S2 fixed binomial kernel differs"
            )
        return state

    def adapted_final_checkpoint_payload(
        **keywords: Any,
    ) -> dict[str, Any]:
        payload = dict(original_final_checkpoint_payload(**keywords))
        payload.update(_strict_false_disclosures())
        return payload

    def adapted_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        output = dict(payload)
        output.update(_strict_false_disclosures())
        original_write_json(path, output)

    def adapted_atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
        output = dict(payload)
        output.update(_strict_false_disclosures())
        if output.get("schema") in {CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA}:
            output.update(
                {
                    "experiment_schema": EXPERIMENT_SCHEMA,
                    "architecture_variant": spec.label,
                    "variant_key": spec.key,
                    "selection_margin_raw": None,
                    "selection_window_applied": False,
                    "promotion_eligible": not screen._is_active_smoke(),
                    "public_test_supported": False,
                    "test_split_accessed": False,
                }
            )
            buffer = io.BytesIO()
            importlib.import_module("torch").save(output, buffer)
            _write_bytes_no_clobber(
                path, buffer.getvalue(), allow_identical=False
            )
            return
        original_atomic_torch_save(path, output)

    def adapted_load_resume_state(*positional: Any, **keywords: Any) -> Any:
        restored = original_load_resume_state(*positional, **keywords)
        if args is not None and args.smoke_max_train_samples is None:
            start_epoch = restored[0] if isinstance(restored, tuple) and restored else None
            if type(start_epoch) is not int or not 2 <= start_epoch <= SCREEN_EPOCH + 1:
                raise CPHFS2LegacyScreenError(
                    "formal restored frontier cannot exceed committed epoch 500"
                )
        model = keywords.get("model")
        if model is None:
            raise CPHFS2LegacyScreenError("resume model is missing")
        manifest = api["validate_irstd_cp_hf_s2_v1"](
            model, require_identity_initialization=False
        )
        if manifest.get("insertion_point") != (
            "after_up_decoder2_finish_before_gt2_and_up_decoder1"
        ):
            raise CPHFS2LegacyScreenError(
                "resumed CP-HF-S2 architecture contract differs"
            )
        return restored

    def adapted_finalize_completed(
        active_args: argparse.Namespace, active_paths: Mapping[str, Any]
    ) -> Any:
        if active_args.smoke_max_train_samples is None:
            raise CPHFS2LegacyScreenError(
                "formal Stage-1 transaction cannot finalize a completed run"
            )
        return original_finalize_completed(active_args, active_paths)

    @contextmanager
    def adapted_process_lock(active_args: argparse.Namespace) -> Iterator[Path]:
        with original_process_lock(active_args) as lock_path:
            _formal_entry_barrier(active_args)
            yield lock_path

    def adapted_initialize(
        dataset: str,
        *,
        seed: int = ARCHITECTURE_SEED,
        training: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        if dataset != DATASET or seed != ARCHITECTURE_SEED or training is not True:
            raise CPHFS2LegacyScreenError("formal constructor contract differs")
        model, metadata = api["build_irstd_cp_hf_s2_v1"](
            dataset, seed=seed, training=True
        )
        manifest = api["validate_irstd_cp_hf_s2_v1"](
            model, require_identity_initialization=True
        )
        if (
            len(model.state_dict()) != spec.state_key_count
            or sum(parameter.numel() for parameter in model.parameters())
            != spec.parameter_count
            or any(not parameter.requires_grad for parameter in model.parameters())
            or manifest.get("insertion_point")
            != "after_up_decoder2_finish_before_gt2_and_up_decoder1"
        ):
            raise CPHFS2LegacyScreenError("constructed architecture contract differs")
        ready = dict(metadata)
        ready.update(
            {
                "public_model": spec.label,
                "variant_key": spec.key,
                "architecture_manifest": manifest,
                "architecture_manifest_sha256": _canonical_sha256(manifest),
                "full_model_scratch": True,
                "all_parameters_trainable": True,
                "state_key_count": spec.state_key_count,
                "parameter_count": spec.parameter_count,
                "public_test_supported": False,
            }
        )
        return model, ready

    patches = {
        "DEFAULT_OUTPUT_ROOT": PROJECT_ROOT / "runs" / "irstd_model_design",
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "RULES_PATH": RULES_PATH,
        "VARIANTS": {ARCHITECTURE_VARIANT: spec},
        "TRAINING_SCHEMA": TRAINING_SCHEMA,
        "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "HISTORY_SCHEMA": HISTORY_SCHEMA,
        "SELECTION_PAYLOAD_SCHEMA": (
            "evisirst_irstd_cp_hf_s2_legacy_screen_selection/v1"
        ),
        "SOURCE_SET_SCHEMA": SOURCE_SET_SCHEMA,
        "DETERMINISM_SCHEMA": DETERMINISM_SCHEMA,
        "EXPERIMENT_SCHEMA": EXPERIMENT_SCHEMA,
        "PROMOTION_GATE_SCHEMA": (
            "evisirst_irstd_cp_hf_s2_legacy_screen_eligibility/v1"
        ),
        "_load_rules": adapted_rules,
        "_variant_source_provenance": adapted_source,
        "_state_contract": adapted_state_contract,
        "build_datasets": adapted_build_datasets,
        "_variant_run_identity": adapted_run_identity,
        "_validate_state_dict": adapted_validate_state_dict,
        "_variant_initialize_evisirst": adapted_initialize,
        "_variant_final_checkpoint_payload": adapted_final_checkpoint_payload,
        "_variant_write_json": adapted_write_json,
        "_variant_atomic_torch_save": adapted_atomic_torch_save,
        "_run_process_lock": adapted_process_lock,
    }
    previous = {
        name: (hasattr(screen, name), getattr(screen, name, None))
        for name in patches
    }
    try:
        for name, value in patches.items():
            setattr(screen, name, value)
        screen.r1._load_resume_state = adapted_load_resume_state
        screen.hf_transaction._finalize_completed = adapted_finalize_completed
        if args is not None and args.smoke_max_train_samples is not None:
            screen.r1.evaluate_model = smoke_evaluate_model
        yield screen
    finally:
        screen.hf_transaction._finalize_completed = original_finalize_completed
        screen.r1._load_resume_state = original_load_resume_state
        screen.r1.evaluate_model = original_evaluate_model
        for name, (existed, value) in previous.items():
            if existed:
                setattr(screen, name, value)
            elif hasattr(screen, name):
                delattr(screen, name)


def run(args: argparse.Namespace) -> Any:
    report = preflight(args)
    if args.preflight_only:
        return report
    if not EXECUTION_SEALED:
        raise CPHFS2ExecutionNotSealed(
            "training is blocked until the architecture API and transaction adapter are audited"
        )
    _formal_entry_barrier(args)
    try:
        with _screen_transaction_adapter(args) as screen:
            checkpoint = screen.run(args)
    except CPHFS2AtomicScreenPause:
        latest = _validate_atomic_epoch500_pause(args)
        relative = latest.relative_to(PROJECT_ROOT).as_posix()
        print(
            f"{EPOCH500_CLEAN_MARKER} run_seed={args.run_seed} epoch={SCREEN_EPOCH} "
            f"latest={relative} continuation_authorized=false",
            flush=True,
        )
        return latest
    return checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    if isinstance(result, Mapping):
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(result)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_API",
    "ARCHITECTURE_MODULE",
    "ARCHITECTURE_PATH",
    "ARCHITECTURE_TEST_PATH",
    "CANDIDATE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CPHFS2ArchitectureNotReady",
    "CPHFS2AtomicScreenPause",
    "CPHFS2ExecutionNotSealed",
    "CPHFS2LegacyScreenError",
    "DEFAULT_OUTPUT_ROOT",
    "EXECUTION_SEALED",
    "EPOCH500_CLEAN_MARKER",
    "FORMAL_EPOCHS",
    "FORMAL_RUN_SEEDS",
    "RULES_SCHEMA",
    "SCREEN_EPOCH",
    "TRAINING_SCHEMA",
    "architecture_api",
    "load_rules",
    "parse_args",
    "preflight",
    "require_args",
    "require_test_false",
    "resolve_run_paths",
    "run",
    "source_provenance",
    "split_provenance",
    "validate_candidate_envelope",
    "validate_final_envelope",
    "validate_resume_envelope",
    "write_json_no_clobber",
]
