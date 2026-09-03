from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from tools import finalize_sbsc_v33_results as finalizer


@dataclass
class ArtifactBundle:
    root: Path
    method_config_paths: dict[str, str]
    candidate_summary_paths: dict[str, str]
    summary_payloads: dict[str, dict[str, Any]]


def _write_json(root: Path, relative: str, payload: dict[str, Any]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(finalizer.contracts.canonical_json_bytes(payload))
    return path


def _source_manifest(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for position, relative in enumerate(
        sorted(finalizer.REQUIRED_TRAINING_SOURCE_PATHS)
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen test source {position}\n", encoding="utf-8")
        records.append(
            {
                "path": relative,
                "sha256": finalizer.contracts.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema": finalizer.SOURCE_MANIFEST_SCHEMA,
        "files": records,
        "sha256": finalizer.contracts.canonical_sha256(records),
    }


def _baseline_metrics() -> dict[str, Any]:
    return {
        "test_loss": 0.4,
        "miou": 0.65,
        "niou": 0.66,
        "pixel_precision": 0.7,
        "pixel_recall": 0.7,
        "pixel_f1": 0.7,
        "pd": 0.75,
        "tiny_pd": 0.7,
        "fa": 3.0e-5,
        "false_objects_per_image": 0.2,
        "target_count": 100,
        "matched_target_count": 75,
        "tiny_target_count": 10,
        "matched_tiny_target_count": 7,
        "predicted_object_count": 90,
        "unmatched_predicted_object_count": 15,
        "valid_pixel_count": 65_536,
    }


def _split_contract(
    root: Path,
    dataset: str,
    normalization: dict[str, float],
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for role, hash_character in (("train", "a"), ("test", "b")):
        relative = f"datasets/{dataset}/img_idx/{role}_{dataset}.txt"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{role}_{dataset}_fixture\n", encoding="utf-8")
        records[role] = {
            "index_path": relative,
            "count": finalizer.EXPECTED_COUNTS[dataset][role],
            "file_sha256": finalizer.contracts.sha256_file(path),
            "ordered_ids_sha256": hash_character * 64,
            "runner_index_order_sha256": hash_character * 64,
        }
    return {
        "schema": "sctransnet_sbsc_v33/img_idx_split_contract/v1",
        "dataset": dataset,
        "train": records["train"],
        "test": records["test"],
        "normalization": copy.deepcopy(normalization),
        "train_test_disjoint": True,
        "validation_split_used": False,
    }


def _metric_values(epoch: int) -> dict[str, Any]:
    is_miou_winner = epoch in (600, 601)
    is_pd_winner = epoch in (700, 701)
    miou = 0.92 if is_miou_winner else (0.78 if is_pd_winner else 0.7)
    pd = 0.95 if is_pd_winner else (0.85 if is_miou_winner else 0.8)
    matched_target_count = int(round(pd * 100))
    metrics = {
        "test_loss": 0.25 if is_miou_winner or is_pd_winner else 0.3,
        "miou": miou,
        "niou": 0.88 if is_miou_winner else 0.72,
        "pixel_precision": 0.8,
        "pixel_recall": 0.8,
        "pixel_f1": 0.8,
        "pd": pd,
        "tiny_pd": 0.8,
        "fa": 2.0e-5,
        "false_objects_per_image": 0.1,
        "target_count": 100,
        "matched_target_count": matched_target_count,
        "tiny_target_count": 10,
        "matched_tiny_target_count": 8,
        "predicted_object_count": 90,
        "unmatched_predicted_object_count": 10,
        "valid_pixel_count": 65_536,
    }
    return {
        **metrics,
        "tiny_pd_was_undefined": False,
        **{
            alias: metrics[canonical]
            for alias, canonical in finalizer.TOP_LEVEL_ALIASES.items()
        },
    }


def _history_record(
    *,
    epoch: int,
    dataset: str,
    method_config_sha256: str,
    training: dict[str, Any],
    model_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": finalizer.TEST_RECORD_SCHEMA,
        "epoch": epoch,
        "model": finalizer.MODEL_NAME,
        "method": "sbsc_v33_third",
        "dataset": dataset,
        "architecture_seed": 42,
        "run_seed": 42,
        "seed": 42,
        "sample_count": finalizer.EXPECTED_COUNTS[dataset]["test"],
        "split_manifest_sha256": training["split_manifest_sha256"],
        "run_identity_sha256": training["run_identity_sha256"],
        "evaluation_contract_sha256": training[
            "evaluation_contract_sha256"
        ],
        "evaluation_source_sha256": training["evaluation_source_sha256"],
        "method_config_sha256": method_config_sha256,
        "gradient_authorization_sha256": training[
            "gradient_authorization_sha256"
        ],
        "baseline_authority_manifest_sha256": training[
            "baseline_authority_manifest_sha256"
        ],
        "formal_launch_authorization_sha256": training[
            "formal_launch_authorization_sha256"
        ],
        "training_source_manifest_sha256": training[
            "training_source_manifest_sha256"
        ],
        "model_state_sha256": model_state_sha256,
        **finalizer.DISCLOSURES,
        **_metric_values(epoch),
    }


def _method_config(
    *,
    dataset: str,
    authorization: dict[str, Any],
    baseline: dict[str, Any],
    source_manifest: dict[str, Any],
    split_contract: dict[str, Any],
) -> dict[str, Any]:
    method = "sbsc_v33_third"
    return {
        "schema": finalizer.METHOD_CONFIG_SCHEMA,
        "status": "frozen",
        "write_once": True,
        "run_kind": "formal",
        "protocol": finalizer.FORMAL_PROTOCOL,
        "model": finalizer.MODEL_NAME,
        "builder": (
            "experiments.sctransnet_sbsc_v33."
            "build_sctransnet_sbsc_v33_method"
        ),
        "method": method,
        "dataset": dataset,
        "balance_mode": finalizer.BALANCE_MODE[method],
        "loss_schema": finalizer.core.SBSC_V33_LOSS_SCHEMA,
        "segmentation_loss": "sum_of_six_BCELoss_mean_terms",
        "router_level_count": 4,
        "level_reduction": "mean",
        "router_loss_weight": 1.0,
        "optimizer": "Adam",
        "architecture_seed": 42,
        "run_seed": 42,
        "router_value_gradient_mode": authorization[
            "authorized_router_value_gradient_mode"
        ],
        "epochs": 1000,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "base_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 10,
        "amp": False,
        "selection_begin_epoch": 500,
        "selection_end_epoch": 1000,
        "selection_every": 1,
        "selection_roles": copy.deepcopy(finalizer.SERIALIZED_ROLE_RANKING),
        "evaluator": baseline["authorities"][dataset]["protocol"],
        "probability_threshold": baseline["authorities"][dataset]["threshold"],
        "probability_comparison": baseline["authorities"][dataset][
            "threshold_operator"
        ],
        "component_match_radius": baseline["authorities"][dataset][
            "match_radius"
        ],
        "component_match_comparison": baseline["authorities"][dataset][
            "match_radius_operator"
        ],
        "tiny_area_max": baseline["authorities"][dataset]["tiny_area"],
        "gradient_authorization_path": authorization["authorization_path"],
        "gradient_authorization_sha256": authorization[
            "authorization_sha256"
        ],
        "baseline_authority_manifest_path": baseline["manifest_path"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "training_source_manifest": copy.deepcopy(source_manifest),
        "training_source_manifest_sha256": source_manifest["sha256"],
        "output_namespace": f"runs/{method}/formal/{dataset}",
        "dataset_root_recorded": False,
        "split_contract": copy.deepcopy(split_contract),
        "split_contract_sha256": finalizer.contracts.canonical_sha256(
            split_contract
        ),
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
    }


def _training_contract(
    *,
    dataset: str,
    method_config_path: str,
    method_config_sha256: str,
    authorization: dict[str, Any],
    baseline: dict[str, Any],
    source_manifest: dict[str, Any],
    split_contract: dict[str, Any],
    formal_launch_path: str,
    formal_launch_sha256: str,
    identity_digit: str,
) -> dict[str, Any]:
    authority = baseline["authorities"][dataset]
    evaluation = {
        "dataset": dataset,
        "sample_count": authority["sample_count"],
        "protocol": authority["protocol"],
        "threshold": authority["threshold"],
        "threshold_operator": authority["threshold_operator"],
        "match_radius": authority["match_radius"],
        "match_radius_operator": authority["match_radius_operator"],
        "tiny_area": authority["tiny_area"],
        "normalization": copy.deepcopy(authority["normalization"]),
    }
    split = {
        "dataset": dataset,
        "train_count": finalizer.EXPECTED_COUNTS[dataset]["train"],
        "test_count": finalizer.EXPECTED_COUNTS[dataset]["test"],
        "train_index_order_sha256": split_contract["train"][
            "runner_index_order_sha256"
        ],
        "test_index_order_sha256": split_contract["test"][
            "runner_index_order_sha256"
        ],
        "normalization": copy.deepcopy(authority["normalization"]),
        "training_target_rule": "raw_mask_div_255",
    }
    training = {
        "schema": finalizer.FORMAL_TRAINING_SCHEMA,
        "run_kind": "formal",
        "protocol": finalizer.FORMAL_PROTOCOL,
        "model": finalizer.MODEL_NAME,
        "method": "sbsc_v33_third",
        "dataset": dataset,
        "architecture_seed": 42,
        "run_seed": 42,
        "seed": 42,
        "epochs": 1000,
        "selection_begin_epoch": 500,
        "selection_end_epoch": 1000,
        "selection_every": 1,
        "train_count": finalizer.EXPECTED_COUNTS[dataset]["train"],
        "test_count": finalizer.EXPECTED_COUNTS[dataset]["test"],
        "evaluator": authority["protocol"],
        "probability_threshold": authority["threshold"],
        "probability_comparison": authority["threshold_operator"],
        "component_match_radius": authority["match_radius"],
        "component_match_comparison": authority["match_radius_operator"],
        "tiny_area_max": authority["tiny_area"],
        "normalization": copy.deepcopy(authority["normalization"]),
        "method_config_path": method_config_path,
        "method_config_sha256": method_config_sha256,
        "gradient_authorization_path": authorization["authorization_path"],
        "gradient_authorization_sha256": authorization[
            "authorization_sha256"
        ],
        "baseline_authority_manifest_path": baseline["manifest_path"],
        "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
        "formal_launch_authorization_path": formal_launch_path,
        "formal_launch_authorization_sha256": formal_launch_sha256,
        "training_source_manifest": copy.deepcopy(source_manifest),
        "training_source_manifest_sha256": source_manifest["sha256"],
        "router_value_gradient_mode": authorization[
            "authorized_router_value_gradient_mode"
        ],
        "balance_mode": "one_third_two_thirds",
        "router_level_count": 4,
        "level_reduction": "mean",
        "router_loss_weight": 1.0,
        "selection_roles": copy.deepcopy(finalizer.SERIALIZED_ROLE_RANKING),
        "split_manifest": split,
        "split_manifest_sha256": finalizer.contracts.canonical_sha256(split),
        "run_identity_sha256": identity_digit * 64,
        "evaluation_contract": evaluation,
        "evaluation_contract_sha256": finalizer.contracts.canonical_sha256(
            evaluation
        ),
        "evaluation_source_sha256": next(
            record["sha256"]
            for record in source_manifest["files"]
            if record["path"] == "test.py"
        ),
        **finalizer.DISCLOSURES,
    }
    training["selection_identity"] = {
        "method": training["method"],
        "dataset": training["dataset"],
        "architecture_seed": training["architecture_seed"],
        "run_seed": training["run_seed"],
        "split_manifest_sha256": training["split_manifest_sha256"],
        "run_identity_sha256": training["run_identity_sha256"],
        "evaluation_contract_sha256": training[
            "evaluation_contract_sha256"
        ],
        **{
            field: training[field]
            for field in finalizer.HASH_BINDINGS
        },
    }
    return training


def _checkpoint_payload(
    *,
    dataset: str,
    role: str,
    selected: dict[str, Any],
    state: dict[str, torch.Tensor],
    training: dict[str, Any],
    method_config_path: str,
    method_config_sha256: str,
) -> dict[str, Any]:
    metric_name = "global_foreground_mIoU" if role == "best_miou" else "Pd"
    score = selected["metrics"]["miou" if role == "best_miou" else "pd"]
    return {
        "schema": finalizer.CHECKPOINT_SCHEMA,
        "model": finalizer.MODEL_NAME,
        "method": "sbsc_v33_third",
        "dataset": dataset,
        "checkpoint_role": role,
        "epoch": selected["epoch"],
        "selection_epoch": selected["epoch"],
        "seed": 42,
        "state_dict": state,
        "model_state_sha256": selected["model_state_sha256"],
        "method_config_path": method_config_path,
        "method_config_sha256": method_config_sha256,
        "gradient_authorization_sha256": training[
            "gradient_authorization_sha256"
        ],
        "baseline_authority_manifest_sha256": training[
            "baseline_authority_manifest_sha256"
        ],
        "formal_launch_authorization_path": training[
            "formal_launch_authorization_path"
        ],
        "formal_launch_authorization_sha256": training[
            "formal_launch_authorization_sha256"
        ],
        "training_source_manifest_sha256": training[
            "training_source_manifest_sha256"
        ],
        "selection_metric": metric_name,
        "selection_score": score,
        "selection_role_key": selected["role_key"],
        "selection_metrics": selected["metrics"],
        "training": training,
        "data_role": "test",
        "test_split_accessed": True,
        "test_selected": True,
        "this_checkpoint_selected_by_test": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
    }


@pytest.fixture
def artifact_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArtifactBundle:
    root = tmp_path / "repository"
    root.mkdir()
    actual_finalizer = Path(finalizer.__file__).resolve(strict=True)

    monkeypatch.setattr(finalizer.contracts, "PROJECT_ROOT", root)

    def repository_relative_path(path: Path) -> str:
        resolved = Path(path).resolve(strict=True)
        if resolved == actual_finalizer:
            return "tools/finalize_sbsc_v33_results.py"
        return resolved.relative_to(root.resolve(strict=True)).as_posix()

    monkeypatch.setattr(
        finalizer.contracts,
        "repository_relative_path",
        repository_relative_path,
    )
    monkeypatch.setattr(
        finalizer.core,
        "validate_sbsc_v33_state_dict",
        lambda state, method: None,
    )

    authorization = {
        "authorization_path": "experiments/gradient_authorization.json",
        "authorization_sha256": "c" * 64,
        "authorized_router_value_gradient_mode": "live",
    }
    baseline = {
        "manifest_path": "experiments/baseline_authority_manifest.json",
        "manifest_sha256": "d" * 64,
        "authorities": {},
    }
    for position, dataset in enumerate(finalizer.DATASETS):
        baseline["authorities"][dataset] = {
            "sample_count": finalizer.EXPECTED_COUNTS[dataset]["test"],
            "protocol": "evisirst_public_common_evaluator_v1",
            "threshold": 0.5,
            "threshold_operator": ">",
            "match_radius": 3.0,
            "match_radius_operator": "<",
            "tiny_area": 9,
            "normalization": {
                "mean": 80.0 + position,
                "std": 30.0 + position,
            },
            "checkpoint": {
                "path": f"baseline/{dataset}.pth.tar",
                "epoch": 500 + position,
                "sha256": str(position + 1) * 64,
            },
            "metrics": _baseline_metrics(),
        }
    monkeypatch.setattr(
        finalizer.contracts,
        "load_gradient_authorization",
        lambda: copy.deepcopy(authorization),
    )
    monkeypatch.setattr(
        finalizer.contracts,
        "load_baseline_authority_manifest",
        lambda: copy.deepcopy(baseline),
    )

    source_manifest = _source_manifest(root)
    method_config_paths: dict[str, str] = {}
    candidate_summary_paths: dict[str, str] = {}
    summary_payloads: dict[str, dict[str, Any]] = {}

    for dataset_index, dataset in enumerate(finalizer.DATASETS, start=1):
        split_contract = _split_contract(
            root,
            dataset,
            baseline["authorities"][dataset]["normalization"],
        )
        config_relative = (
            "experiments/sbsc_v33_methods/"
            f"sbsc_v33_third_{finalizer.DATASET_CONFIG_TOKEN[dataset]}_formal.json"
        )
        config = _method_config(
            dataset=dataset,
            authorization=authorization,
            baseline=baseline,
            source_manifest=source_manifest,
            split_contract=split_contract,
        )
        config_path = _write_json(root, config_relative, config)
        config_sha256 = finalizer.contracts.sha256_file(config_path)
        method_config_paths[dataset] = config_relative

        launch_relative = (
            "experiments/"
            f"sbsc_v33_{finalizer.DATASET_CONFIG_TOKEN[dataset]}_"
            "formal_launch_authorization.json"
        )
        launch = {
            "schema": "sctransnet_sbsc_v33/formal_launch_authorization/v1",
            "status": "PASS",
            "write_once": True,
            "authorized_run": {
                "method": "sbsc_v33_third",
                "dataset": dataset,
                "run_kind": "formal",
                "output_namespace": f"runs/sbsc_v33_third/formal/{dataset}",
            },
            "method_config_path": config_relative,
            "method_config_sha256": config_sha256,
            "gradient_authorization_sha256": authorization[
                "authorization_sha256"
            ],
            "baseline_authority_manifest_sha256": baseline[
                "manifest_sha256"
            ],
            "training_source_manifest_sha256": source_manifest["sha256"],
        }
        launch_path = _write_json(root, launch_relative, launch)
        launch_sha256 = finalizer.contracts.sha256_file(launch_path)

        training = _training_contract(
            dataset=dataset,
            method_config_path=config_relative,
            method_config_sha256=config_sha256,
            authorization=authorization,
            baseline=baseline,
            source_manifest=source_manifest,
            split_contract=split_contract,
            formal_launch_path=launch_relative,
            formal_launch_sha256=launch_sha256,
            identity_digit=str(dataset_index),
        )
        states = {
            "best_miou": {
                "weight": torch.tensor(
                    [float(dataset_index), 1.0], dtype=torch.float32
                )
            },
            "best_pd": {
                "weight": torch.tensor(
                    [float(dataset_index), 2.0], dtype=torch.float32
                )
            },
        }
        state_hashes = {
            role: finalizer.state_dict_sha256(state)
            for role, state in states.items()
        }
        history = [
            _history_record(
                epoch=epoch,
                dataset=dataset,
                method_config_sha256=config_sha256,
                training=training,
                model_state_sha256=(
                    state_hashes["best_miou"]
                    if epoch == 600
                    else state_hashes["best_pd"]
                    if epoch == 700
                    else "0" * 64
                ),
            )
            for epoch in finalizer.SELECTION_EPOCHS
        ]
        selections: dict[str, dict[str, Any]] = {}
        for role in finalizer.ROLES:
            winner = max(history, key=lambda row, role=role: finalizer.role_key(row, role))
            selections[role] = {
                "epoch": winner["epoch"],
                "model_state_sha256": winner["model_state_sha256"],
                "role_key": finalizer.role_key_record(winner, role),
                "metrics": copy.deepcopy(winner),
                "test_selected": True,
                "selection_is_optimistic": True,
            }

        run_relative = f"runs/sbsc_v33_third/formal/{dataset}"
        published: dict[str, dict[str, Any]] = {}
        checkpoint_names = {
            "best_miou": "best_mIoU.pth.tar",
            "best_pd": "best_Pd.pth.tar",
        }
        for role in finalizer.ROLES:
            checkpoint_relative = f"{run_relative}/{checkpoint_names[role]}"
            checkpoint_path = root / checkpoint_relative
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _checkpoint_payload(
                    dataset=dataset,
                    role=role,
                    selected=selections[role],
                    state=states[role],
                    training=training,
                    method_config_path=config_relative,
                    method_config_sha256=config_sha256,
                ),
                checkpoint_path,
            )
            published[role] = {
                "path": checkpoint_relative,
                "sha256": finalizer.contracts.sha256_file(checkpoint_path),
                "model_state_sha256": state_hashes[role],
            }

        summary = {
            "schema": finalizer.CANDIDATE_SUMMARY_SCHEMA,
            "status": "complete",
            "model": finalizer.MODEL_NAME,
            "method": "sbsc_v33_third",
            "dataset": dataset,
            "candidate_count": 501,
            "training": training,
            "selection_history": history,
            "selections": selections,
            "published_checkpoints": published,
            "two_distinct_physical_checkpoint_files": True,
            "method_config_path": config_relative,
            "method_config_sha256": config_sha256,
            "gradient_authorization_path": authorization["authorization_path"],
            "gradient_authorization_sha256": authorization[
                "authorization_sha256"
            ],
            "baseline_authority_manifest_path": baseline["manifest_path"],
            "baseline_authority_manifest_sha256": baseline["manifest_sha256"],
            "formal_launch_authorization_path": launch_relative,
            "formal_launch_authorization_sha256": launch_sha256,
            "training_source_manifest_sha256": source_manifest["sha256"],
            **finalizer.DISCLOSURES,
        }
        summary_relative = f"{run_relative}/summary.json"
        _write_json(root, summary_relative, summary)
        candidate_summary_paths[dataset] = summary_relative
        summary_payloads[dataset] = summary

    return ArtifactBundle(
        root=root,
        method_config_paths=method_config_paths,
        candidate_summary_paths=candidate_summary_paths,
        summary_payloads=summary_payloads,
    )


def _rewrite_summary(
    bundle: ArtifactBundle,
    dataset: str,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    payload = copy.deepcopy(bundle.summary_payloads[dataset])
    mutation(payload)
    bundle.summary_payloads[dataset] = payload
    _write_json(bundle.root, bundle.candidate_summary_paths[dataset], payload)


def test_builds_two_separate_three_dataset_result_tables(
    artifact_bundle: ArtifactBundle,
) -> None:
    payload = finalizer.build_final_payload(
        method_config_paths=artifact_bundle.method_config_paths,
        candidate_summary_paths=artifact_bundle.candidate_summary_paths,
    )

    assert payload["schema"] == finalizer.FINAL_SCHEMA
    assert payload["dataset_order"] == list(finalizer.DATASETS)
    assert payload["role_rows_are_separate"] is True
    assert payload["cross_role_metric_splicing"] is False
    assert [row["checkpoint_role"] for row in payload["best_miou_rows"]] == [
        "best_miou"
    ] * 3
    assert [row["checkpoint_role"] for row in payload["best_pd_rows"]] == [
        "best_pd"
    ] * 3
    assert [row["epoch"] for row in payload["best_miou_rows"]] == [600] * 3
    assert [row["epoch"] for row in payload["best_pd_rows"]] == [700] * 3
    for row in payload["best_miou_rows"] + payload["best_pd_rows"]:
        assert set(finalizer.METRIC_FIELDS).issubset(row["candidate_metrics"])
        assert row["metrics_are_from_one_physical_checkpoint"] is True
        assert row["candidate_checkpoint"]["path"]
        assert len(row["candidate_checkpoint"]["sha256"]) == 64
        assert len(row["candidate_checkpoint"]["model_state_sha256"]) == 64
    assert "combined_rows" not in payload
    assert "best_miou_and_pd_rows" not in payload


def test_write_once_publication_is_idempotent_and_rejects_tampering(
    artifact_bundle: ArtifactBundle,
) -> None:
    output = "artifacts/sbsc_v33/formal_results.json"
    first_path, first_sha = finalizer.finalize_results(
        method_config_paths=artifact_bundle.method_config_paths,
        candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        output_path=output,
    )
    second_path, second_sha = finalizer.finalize_results(
        method_config_paths=artifact_bundle.method_config_paths,
        candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        output_path=output,
    )
    assert first_path == second_path == artifact_bundle.root / output
    assert first_sha == second_sha

    first_path.write_bytes(first_path.read_bytes() + b" ")
    with pytest.raises(finalizer.SBSCV33FinalizationError, match="write-once"):
        finalizer.finalize_results(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
            output_path=output,
        )


def test_rejects_missing_epoch_from_501_record_history(
    artifact_bundle: ArtifactBundle,
) -> None:
    _rewrite_summary(
        artifact_bundle,
        "NUAA-SIRST",
        lambda payload: payload["selection_history"].pop(),
    )
    with pytest.raises(finalizer.SBSCV33FinalizationError, match="exactly 501"):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_stored_selection_that_does_not_match_tie_break_replay(
    artifact_bundle: ArtifactBundle,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selections"]["best_miou"]["epoch"] = 601

    _rewrite_summary(artifact_bundle, "NUAA-SIRST", mutate)
    with pytest.raises(finalizer.SBSCV33FinalizationError, match="replayed history"):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_out_of_range_metric_even_when_alias_matches(
    artifact_bundle: ArtifactBundle,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["selection_history"][0]["miou"] = 1.01
        payload["selection_history"][0]["mIoU"] = 1.01

    _rewrite_summary(artifact_bundle, "NUAA-SIRST", mutate)
    with pytest.raises(finalizer.SBSCV33FinalizationError, match=r"\[0,1\]"):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_checkpoint_role_tamper_even_with_updated_file_sha(
    artifact_bundle: ArtifactBundle,
) -> None:
    dataset = "NUAA-SIRST"
    summary = artifact_bundle.summary_payloads[dataset]
    record = summary["published_checkpoints"]["best_miou"]
    checkpoint_path = artifact_bundle.root / record["path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["checkpoint_role"] = "best_pd"
    torch.save(checkpoint, checkpoint_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["published_checkpoints"]["best_miou"]["sha256"] = (
            finalizer.contracts.sha256_file(checkpoint_path)
        )

    _rewrite_summary(artifact_bundle, dataset, mutate)
    with pytest.raises(
        finalizer.SBSCV33FinalizationError,
        match="checkpoint field 'checkpoint_role' differs",
    ):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_one_path_for_both_checkpoint_roles(
    artifact_bundle: ArtifactBundle,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        left = payload["published_checkpoints"]["best_miou"]
        payload["published_checkpoints"]["best_pd"] = {
            **copy.deepcopy(left),
            "model_state_sha256": payload["selections"]["best_pd"][
                "model_state_sha256"
            ],
        }

    _rewrite_summary(artifact_bundle, "NUAA-SIRST", mutate)
    with pytest.raises(finalizer.SBSCV33FinalizationError):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_method_config_source_hash_without_matching_manifest(
    artifact_bundle: ArtifactBundle,
) -> None:
    dataset = "IRSTD-1K"
    relative = artifact_bundle.method_config_paths[dataset]
    config = finalizer.contracts.load_strict_json(artifact_bundle.root / relative)
    config["training_source_manifest_sha256"] = "f" * 64
    _write_json(artifact_bundle.root, relative, config)

    with pytest.raises(
        finalizer.SBSCV33FinalizationError,
        match="method config training source hash binding differs",
    ):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_method_config_missing_one_freezer_field(
    artifact_bundle: ArtifactBundle,
) -> None:
    dataset = "IRSTD-1K"
    relative = artifact_bundle.method_config_paths[dataset]
    config = finalizer.contracts.load_strict_json(artifact_bundle.root / relative)
    config.pop("segmentation_loss")
    _write_json(artifact_bundle.root, relative, config)

    with pytest.raises(
        finalizer.SBSCV33FinalizationError,
        match="method config field set differs",
    ):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_formal_launch_authorization_physical_drift(
    artifact_bundle: ArtifactBundle,
) -> None:
    dataset = "IRSTD-1K"
    launch_relative = artifact_bundle.summary_payloads[dataset]["training"][
        "formal_launch_authorization_path"
    ]
    launch_path = artifact_bundle.root / launch_relative
    launch_path.write_bytes(launch_path.read_bytes() + b" ")

    with pytest.raises(
        finalizer.SBSCV33FinalizationError,
        match="formal launch authorization physical SHA-256 differs",
    ):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_requires_exact_three_dataset_config_and_summary_sets(
    artifact_bundle: ArtifactBundle,
) -> None:
    configs = dict(artifact_bundle.method_config_paths)
    configs.pop("NUDT-SIRST")
    with pytest.raises(finalizer.SBSCV33FinalizationError, match="three datasets"):
        finalizer.build_final_payload(
            method_config_paths=configs,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )


def test_rejects_hardlinked_checkpoint_publications(
    artifact_bundle: ArtifactBundle,
) -> None:
    dataset = "NUAA-SIRST"
    summary = artifact_bundle.summary_payloads[dataset]
    left_record = summary["published_checkpoints"]["best_miou"]
    right_record = summary["published_checkpoints"]["best_pd"]
    left_path = artifact_bundle.root / left_record["path"]
    right_path = artifact_bundle.root / right_record["path"]
    right_path.unlink()
    os.link(left_path, right_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["published_checkpoints"]["best_pd"]["sha256"] = (
            finalizer.contracts.sha256_file(right_path)
        )

    _rewrite_summary(artifact_bundle, dataset, mutate)
    with pytest.raises(finalizer.SBSCV33FinalizationError):
        finalizer.build_final_payload(
            method_config_paths=artifact_bundle.method_config_paths,
            candidate_summary_paths=artifact_bundle.candidate_summary_paths,
        )
