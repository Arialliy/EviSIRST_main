from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.sctransnet_sbsc_v31 import validate_sctransnet_sbsc_v31


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_sctransnet_sbsc_v31",
    ROOT / "tools" / "diagnose_sctransnet_sbsc_v31.py",
)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_source_specs_bind_full_sha_epoch_and_validation_only_weights() -> None:
    baseline = diagnostic.SOURCE_SPECS["sctransnet"]
    v21 = diagnostic.SOURCE_SPECS["sbsc_v21"]
    assert baseline.epoch == 670
    assert v21.epoch == 543
    assert baseline.checkpoint_file_sha256 == (
        "840d596bbd6f309fffa967be05903315e663d88d8fae6f393a067a17241ca8cb"
    )
    assert v21.checkpoint_file_sha256 == (
        "25fda97b94c13f3ddf5a6fe358712daf3929bb2ff2b0ff260d4afb776def55d1"
    )
    assert diagnostic.ZERO_GAIN == (0.0, 0.0, 0.0, 0.0)
    assert diagnostic.STRESS_GAIN == (0.25, 0.25, 0.25, 0.25)
    assert len(diagnostic.DIAGNOSTIC_INTERVENTIONS) == 11


@pytest.mark.parametrize("method", ("sctransnet", "sbsc_v21"))
def test_adapter_copies_exactly_510_shared_tensors_and_is_formal_forbidden(
    method: str,
) -> None:
    adapter = diagnostic.build_c3_v31_diagnostic_adapter(method)
    evidence = adapter.evidence
    assert evidence["source_method"] == method
    assert evidence["shared_state_key_count"] == 510
    assert evidence["diagnostic_only"] is True
    assert evidence["test_split_accessed"] is False
    assert evidence["v31_candidate_validator_rejected_source_state"] is True
    gain_key = "mtc.encoder.layer.1.channel_attn.raw_dual_risk_level_gain"
    gain = adapter.diagnostic_model.state_dict()[gain_key]
    assert gain.dtype is torch.float32
    assert tuple(gain.shape) == (4,)
    assert torch.count_nonzero(gain).item() == 0
    anchor_state = adapter.anchor_model.state_dict()
    diagnostic_state = adapter.diagnostic_model.state_dict()
    assert set(diagnostic_state) - set(anchor_state) == {gain_key}
    assert all(
        torch.equal(anchor_state[key], diagnostic_state[key])
        for key in anchor_state
    )
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        validate_sctransnet_sbsc_v31(adapter.diagnostic_model)


def test_token_regions_are_dynamic_disjoint_padding_safe_and_source_specific() -> None:
    target = np.zeros((16, 16), dtype=np.float32)
    target[6:8, 6:8] = 1.0
    first_probability = np.zeros_like(target)
    first_probability[6:8, 6:8] = 1.0
    first_probability[1:3, 1:3] = 1.0
    second_probability = np.zeros_like(target)
    second_probability[6:8, 6:8] = 1.0
    second_probability[12:14, 12:14] = 1.0
    first = diagnostic.build_token_regions(
        target,
        first_probability,
        token_size=(16, 16),
        valid_hw=(10, 10),
    )
    second = diagnostic.build_token_regions(
        target,
        second_probability,
        token_size=(16, 16),
        valid_hw=(16, 16),
    )
    assert set(first) == set(diagnostic.REGION_NAMES)
    stacked = torch.stack(tuple(first.values())).to(torch.int64)
    assert int(stacked.sum(dim=0).amax().item()) <= 1
    assert not bool(stacked[:, 10:, :].any())
    assert not bool(stacked[:, :, 10:].any())
    assert bool(first["target"].any())
    assert bool(first["ring"].any())
    assert bool(first["false_object"].any())
    assert bool(first["far"].any())
    assert not torch.equal(first["false_object"], second["false_object"])


def test_component_taxonomy_uses_frozen_bins_and_strict_json_infinity() -> None:
    target = np.zeros((16, 16), dtype=np.float32)
    probability = np.zeros_like(target)
    probability[1:3, 1:3] = 1.0
    records = diagnostic.prediction_component_taxonomy(target, probability)
    assert len(records) == 1
    assert records[0]["nearest_gt_centroid_distance_px"] == "+inf"
    assert records[0]["distance_bin"] == "far"
    assert records[0]["area_bin"] == "1-4"
    assert records[0]["matched_by_frozen_hungarian"] is False
    encoded = diagnostic.strict_json_dumps(records)
    assert "Infinity" not in encoded
    assert json.loads(encoded) == records
    assert diagnostic._distance_bin(2.999) == "contact"
    assert diagnostic._distance_bin(3.0) == "near"
    assert diagnostic._distance_bin(8.0) == "mid"
    assert diagnostic._distance_bin(16.0) == "far"
    assert diagnostic._area_bin(4) == "1-4"
    assert diagnostic._area_bin(5) == "5-16"
    assert diagnostic._area_bin(17) == "17-64"
    assert diagnostic._area_bin(65) == "65-inf"


def test_support_region_stat_and_empty_region_contract() -> None:
    support = torch.full((4, 4), 1.0 / 16.0)
    region = torch.zeros((4, 4), dtype=torch.bool)
    assert diagnostic.support_region_stat(support, region) is None
    region[0, :4] = True
    observed = diagnostic.support_region_stat(support, region)
    assert observed is not None
    assert observed["mass"] == pytest.approx(0.25)
    assert observed["enrichment"] == pytest.approx(
        0.25 / (4.0 / 16.0 + 1e-6)
    )


def test_dynamic_token_geometry_is_bound_in_both_directions() -> None:
    generator = torch.Generator().manual_seed(9)
    key = F.normalize(torch.randn(2, 1, 8, 6, generator=generator), dim=-1)
    validations = tuple(torch.zeros(2, 1, 1, 6) for _ in range(4))
    support = diagnostic.v31_core.estimate_c3_v31_support_from_validations(
        key, validations
    )
    capture = {
        "levels": tuple(
            {"level_index": index, "support": level}
            for index, level in enumerate(support.levels)
        )
    }
    assert diagnostic.validate_token_geometry(capture, (32, 48)) == (2, 3)
    with pytest.raises(
        diagnostic.C3V31DiagnosticError, match="support N disagrees"
    ):
        diagnostic.validate_token_geometry(capture, (32, 32))
    with pytest.raises(
        diagnostic.C3V31DiagnosticError, match="divisible by 16"
    ):
        diagnostic.validate_token_geometry(capture, (31, 48))


def test_capture_base_contract_failure_is_an_immediate_diagnostic_error() -> None:
    levels = tuple(
        {"level_index": index, "base_contract_failure": False}
        for index in range(4)
    )
    diagnostic._reject_base_contract_failure(
        {"levels": levels}, execution="stress_fp32"
    )
    failed = list(levels)
    failed[2] = {"level_index": 2, "base_contract_failure": True}
    with pytest.raises(
        diagnostic.C3V31DiagnosticError,
        match=r"stress_bf16 base attention contract failed at levels \[2\]",
    ):
        diagnostic._reject_base_contract_failure(
            {"levels": tuple(failed)}, execution="stress_bf16"
        )


def test_paired_bootstrap_is_image_level_seed42_and_frozen_10000() -> None:
    left = [1.0, 2.0, None, 4.0]
    right = [0.0, 1.0, 5.0, 2.0]
    first = diagnostic.paired_bootstrap_lower_bound(left, right)
    second = diagnostic.paired_bootstrap_lower_bound(left, right)
    assert first == second
    assert first["pair_count"] == 3
    assert first["mean_difference"] == pytest.approx(4.0 / 3.0)
    assert first["seed"] == 42
    assert first["resamples"] == 10_000
    with pytest.raises(ValueError, match="10,000"):
        diagnostic.paired_bootstrap_lower_bound(left, right, resamples=9999)


def _region_stats() -> dict[str, dict[str, dict[str, float | int] | None]]:
    def stat(mass: float, enrichment: float) -> dict[str, float | int]:
        return {"mass": mass, "enrichment": enrichment, "token_count": 2}

    return {
        "target": {
            "C": stat(0.60, 3.0),
            "H": stat(0.10, 0.5),
            "B": stat(0.20, 1.0),
        },
        "ring": {
            "C": stat(0.20, 1.1),
            "H": stat(0.10, 0.6),
            "B": stat(0.15, 0.8),
        },
        "false_object": {
            "C": stat(0.10, 0.5),
            "H": stat(0.40, 2.0),
            "B": stat(0.20, 1.0),
        },
        "far": {
            "C": stat(0.10, 0.5),
            "H": stat(0.20, 1.0),
            "B": stat(0.50, 2.5),
        },
    }


def _semantic_image(index: int) -> dict[str, object]:
    return {
        "sample_id": f"image_{index}",
        "has_target": True,
        "empty_regions": {name: False for name in diagnostic.REGION_NAMES},
        "prediction_components": [
            {"distance_bin": "far", "area_bin": "1-4"}
        ],
        "support_cells": [
            {
                "valid": {"C": True, "H": True, "B": True},
                "regions": _region_stats(),
                "reverse_enrichment_reasons": [],
                "high_confidence_wrong_correction": False,
            }
        ],
    }


def test_semantic_summary_uses_common_valid_image_units_and_source_gates() -> None:
    images = [_semantic_image(index) for index in range(6)]
    v21 = diagnostic.summarize_semantic_evidence(
        images, source_method="sbsc_v21"
    )
    baseline = diagnostic.summarize_semantic_evidence(
        images, source_method="sctransnet"
    )
    assert v21["status"] == "GO"
    assert baseline["status"] == "GO"
    assert all(item["pair_count"] == 6 for item in v21["directions"].values())
    assert all(item["resamples"] == 10_000 for item in v21["directions"].values())
    assert all(
        item["eligible_image_count"] == 6
        and item["undefined_image_count"] == 0
        and all(unit["paired_unit_count"] == 1 for unit in item["image_units"])
        for item in v21["directions"].values()
    )
    assert v21["enrichment_medians"]["C_target"] == pytest.approx(3.0)
    assert v21["enrichment_medians"]["H_false_object"] == pytest.approx(2.0)
    assert "consistent_target_enrichment_median_ge_2" in v21["gates"]
    assert "consistent_target_enrichment_median_ge_2" not in baseline["gates"]


def test_paired_direction_never_subtracts_different_valid_masks() -> None:
    defined = _semantic_image(0)
    left_only = copy.deepcopy(defined["support_cells"][0])
    left_only["valid"]["H"] = False
    left_only["regions"]["target"]["C"]["mass"] = 100.0
    left_only["regions"]["target"]["H"]["mass"] = -100.0
    defined["support_cells"].append(left_only)
    undefined = _semantic_image(1)
    undefined["empty_regions"]["target"] = True
    for cell in undefined["support_cells"]:
        cell["regions"]["target"] = {"C": None, "H": None, "B": None}
    result = diagnostic._paired_direction(
        [defined, undefined],
        left_support="C",
        right_support="H",
        region="target",
    )
    assert result["eligible_image_count"] == 1
    assert result["undefined_image_count"] == 1
    assert result["eligible_sample_ids_in_order"] == ["image_0"]
    assert result["image_units"][0]["paired_unit_count"] == 1
    assert result["image_units"][0]["image_mean_difference"] == pytest.approx(0.5)
    assert result["image_units"][1]["defined"] is False


def _numerical_image(index: int) -> dict[str, object]:
    level = {
        "row_count": 100,
        "finite": True,
        "nonfinite_input_count": 0,
        "terminal_one_hot": True,
        "qhat_row_mass_error_max": 5e-7,
        "qout_row_mass_error_max": 5e-7,
        "qhat_min": 0.0,
        "qout_min": 0.0,
        "q_emitted_row_mass_error_max": 5e-7,
        "q_emitted_min": 0.0,
        "accepted_hard_delta_max": 5e-7,
        "accepted_background_delta_max": 5e-7,
        "accepted_hard_delta_raw_max": 5e-7,
        "accepted_background_delta_raw_max": 5e-7,
        "qout_hard_delta_max": 5e-7,
        "qout_background_delta_max": 5e-7,
        "qout_hard_delta_raw_max": 5e-7,
        "qout_background_delta_raw_max": 5e-7,
        "emitted_hard_delta_max": 5e-7,
        "emitted_background_delta_max": 5e-7,
        "emitted_hard_delta_raw_max": 5e-7,
        "emitted_background_delta_raw_max": 5e-7,
        "accepted_objective_min": 0.0,
        "qout_objective_min": 0.0,
        "emitted_objective_min": 0.0,
        "effective_kkt_max": 5e-7,
        "solver_eligible_rows": 100,
        "target_solver_eligible_rows": 100,
        "target_nontrivial_accepted_rows": 30,
        "emission_fallback_rows": 0,
        "target_nontrivial_emitted_rows": 30,
        "emission_tolerance": 1e-6,
        "qout32_certificate_failure_rows": 0,
        "emitted_certificate_failure_rows": 0,
        "emission_certificate_accounting_consistent": True,
        "emission_reason_counts": {"0": 100, "1": 0, "2": 0},
        "solver_reason_counts": {
            str(code): 100 if code == 0 else 0 for code in range(8)
        },
        "status_counts": {
            "support_identity": 0,
            "risk_identity": 0,
            "solver_fallback": 0,
            "accepted": 100,
            "hard_inactive": 0,
            "hard_defined": 100,
            "background_defined": 100,
            "hard_active": 100,
            "background_active": 100,
            "projected_newton_used": 0,
            "projected_gradient_used": 0,
        },
        "active_set_counts": {
            "-1": 0,
            "0": 100,
            "1": 0,
            "2": 0,
            "3": 0,
        },
        "failure_counts": {
            "bracket_failed_h": 0,
            "bracket_failed_b": 0,
            "correlation_blocked": 0,
            "line_search_failed": 0,
            "live_recert_failed": 0,
        },
        "candidate_counts_by_slot_empty_h_b_hb": {
            name: [100, 0, 0, 0]
            for name in (
                "candidate_valid",
                "candidate_allowed",
                "candidate_finite",
                "candidate_simplex",
                "candidate_support_preserved",
                "candidate_lambda_ok",
                "candidate_kkt_ok",
                "candidate_stationarity_ok",
                "candidate_risk_ok",
                "candidate_objective_ok",
            )
        },
        "candidate_numeric_extrema_by_slot_empty_h_b_hb": {
            name: {"min": [0.0] * 4, "max": [0.0] * 4}
            for name in (
                "candidate_objective",
                "candidate_lambda_h",
                "candidate_lambda_b",
                "candidate_kkt_max",
                "candidate_stationarity",
                "candidate_risk_max",
                "candidate_raw_risk_max",
            )
        },
        "iteration_max": {"singleton_h": 0, "singleton_b": 0, "dual": 0},
    }
    amp = {
        "eligible_rows": 100,
        "status_mismatch_rows": 0,
        "emission_status_mismatch_rows": 0,
        "correction_cosines": [0.999],
        "internal_solver_and_relations_fp32": True,
        "emitted_attention_restores_ambient_dtype": True,
    }
    return {
        "sample_id": f"image_{index}",
        "gain_zero_six_head_bitwise_equal": True,
        "all_model_outputs_finite": True,
        "full_vs_zero_output_mean_abs_change": 0.01,
        "full_vs_zero_output_mean_abs_change_by_pixel_region": {
            name: 0.01 for name in diagnostic.REGION_NAMES
        },
        "numerical_levels": [level],
        "bf16_numerical_levels": [copy.deepcopy(level)],
        "amp_levels": [amp],
    }


def test_numerical_summary_applies_all_frozen_rate_and_certificate_gates() -> None:
    images = [_numerical_image(index) for index in range(3)]
    summary = diagnostic.summarize_numerical_evidence(images)
    assert summary["status"] == "GO"
    assert summary["solver_fallback_rate"] == 0.0
    assert summary["target_nontrivial_accepted_rate"] == pytest.approx(0.30)
    assert summary["fp32_emission_fallback_rate"] == 0.0
    assert summary["bf16_emission_fallback_rate"] == 0.0
    assert summary["fp32_nontrivial_emitted_rate"] == pytest.approx(0.30)
    assert summary["bf16_nontrivial_emitted_rate"] == pytest.approx(0.30)
    assert summary["amp_correction_cosine_min"] == pytest.approx(0.999)
    images[0]["numerical_levels"][0]["status_counts"]["solver_fallback"] = 20
    failed = diagnostic.summarize_numerical_evidence(images)
    assert failed["gates"]["solver_fallback_rate_le_0_05"] is False
    assert failed["status"] == "NO-GO"


def test_three_attention_level_synthetic_boundaries_are_go() -> None:
    report = diagnostic.run_synthetic_contracts()
    assert report["fixture_geometry"] == {
        "batch": 1,
        "heads": 4,
        "positions": 256,
    }
    assert report["status"] == "GO"
    assert all(case["passed"] for case in report["cases"].values())
    assert all(
        case["emitted_output_exact_identity"]
        for case in report["cases"].values()
    )
    assert report["cases"]["consistent_hot"]["token_index"] == 42
    assert report["cases"]["shallow_only"]["peer_indices"] == [1, 2, 3]


def test_strict_json_rejects_nonfinite_evidence() -> None:
    with pytest.raises(
        diagnostic.C3V31DiagnosticError, match="finite strict JSON"
    ):
        diagnostic.strict_json_dumps({"bad": float("inf")})
    with pytest.raises(
        diagnostic.C3V31DiagnosticError, match="finite strict JSON"
    ):
        diagnostic.strict_json_dumps({"bad": float("nan")})


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.mode = "test"


class _FakeValidationDataset:
    target_mode = "binary"
    sample_ids = tuple(f"image_{index}" for index in range(160))
    metadata = {"split": "val", "test_index_opened": False}
    contract = SimpleNamespace(
        manifest_sha256="a" * 64,
        data_tree_sha256=None,
        data_tree_verified=False,
        val_ids=sample_ids,
    )

    def __len__(self) -> int:
        return 160

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


def test_validation_dataset_adapter_disables_full_tree_and_opens_val_only(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def constructor(dataset_name, **kwargs):
        observed["dataset_name"] = dataset_name
        observed.update(kwargs)
        return _FakeValidationDataset()

    monkeypatch.setattr(
        diagnostic.v2_data, "EviSIRSTV2ValDataset", constructor
    )
    dataset = diagnostic.build_validation_dataset(ROOT)
    assert isinstance(dataset, _FakeValidationDataset)
    assert observed["dataset_name"] == "IRSTD-1K"
    assert observed["return_metadata"] is True
    assert observed["verify_data_tree"] is False


def test_source_runner_rejects_non_validation_dataset_before_model_access() -> None:
    dataset = _FakeValidationDataset()
    dataset.metadata = {"split": "test", "test_index_opened": True}
    with pytest.raises(
        diagnostic.C3V31DiagnosticError, match="V2 validation only"
    ):
        diagnostic.run_source_diagnostic("sctransnet", dataset, device="cpu")


def test_source_runner_is_read_only_and_visits_exactly_160_val_items(
    monkeypatch,
) -> None:
    models = (_FakeModel(), _FakeModel(), _FakeModel())
    adapter = diagnostic.C3V31DiagnosticAdapter(
        source_model=models[0],
        anchor_model=models[1],
        diagnostic_model=models[2],
        evidence={"source_method": "sctransnet"},
    )
    visited: list[int] = []
    monkeypatch.setattr(
        diagnostic,
        "build_c3_v31_diagnostic_adapter",
        lambda method: adapter,
    )
    monkeypatch.setattr(
        diagnostic, "validation_data_tree_sha256", lambda dataset: "c" * 64
    )
    monkeypatch.setattr(
        diagnostic,
        "_sample_record",
        lambda *, adapter, item, device: visited.append(item["index"])
        or {"sample_id": str(item["index"])},
    )
    monkeypatch.setattr(
        diagnostic,
        "summarize_semantic_evidence",
        lambda images, source_method: {"status": "GO"},
    )
    monkeypatch.setattr(
        diagnostic,
        "summarize_numerical_evidence",
        lambda images: {"status": "GO"},
    )
    report = diagnostic.run_source_diagnostic(
        "sctransnet", _FakeValidationDataset(), device="cpu"
    )
    assert visited == list(range(160))
    assert report["dataset"]["split"] == "val"
    assert report["dataset"]["sample_count"] == 160
    assert report["test_split_accessed"] is False
    assert report["training_started"] is False
    assert report["gates"]["state_unchanged"] is True
    assert report["status"] == "GO"


def test_top_level_requires_two_separate_sources_and_emits_hashed_strict_json(
    monkeypatch,
) -> None:
    dataset = _FakeValidationDataset()
    monkeypatch.setattr(
        diagnostic, "build_validation_dataset", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(
        diagnostic,
        "run_synthetic_contracts",
        lambda: {"status": "GO", "cases": {}},
    )

    def source_report(method, dataset, *, device):
        return {
            "source_method": method,
            "status": "GO",
            "test_split_accessed": False,
            "training_started": False,
        }

    monkeypatch.setattr(diagnostic, "run_source_diagnostic", source_report)
    report = diagnostic.run_validation_diagnostic(
        ("sctransnet", "sbsc_v21"),
        dataset_root=ROOT,
        device="cpu",
    )
    assert report["status"] == "GO"
    assert report["decision"] == "ALLOW_1_EPOCH_SMOKE"
    assert set(report["sources"]) == {"sctransnet", "sbsc_v21"}
    assert len(report["evidence_sha256"]) == 64
    assert json.loads(diagnostic.strict_json_dumps(report)) == report

    incomplete = diagnostic.run_validation_diagnostic(
        ("sbsc_v21",), dataset_root=ROOT, device="cpu"
    )
    assert incomplete["status"] == "NO-GO"
    assert incomplete["missing_sources"] == ["sctransnet"]


def _fake_hashed_source_report(
    method: str, *, manifest_sha256: str = "d" * 64
) -> dict[str, object]:
    spec = diagnostic.SOURCE_SPECS[method]
    report = {
        "schema": diagnostic.DIAGNOSTIC_SCHEMA + "/source/v1",
        "source_method": method,
        "source_epoch": spec.epoch,
        "adapter": {
            "schema": diagnostic.DIAGNOSTIC_SCHEMA + "/adapter/v1",
            "source_method": method,
            "source_epoch": spec.epoch,
            "checkpoint_relative_path": spec.checkpoint_relative_path,
            "checkpoint_file_sha256": spec.checkpoint_file_sha256,
            "diagnostic_only": True,
            "test_split_accessed": False,
            "shared_state_key_count": 510,
            "v31_candidate_validator_rejected_source_state": True,
            "v31_gain_zero": True,
            "source_state_sha256": "1" * 64,
            "shared_state_sha256": "2" * 64,
            "anchor_state_sha256": "2" * 64,
            "diagnostic_state_sha256": "3" * 64,
        },
        "dataset": {
            "name": "IRSTD-1K",
            "split": "val",
            "sample_count": 160,
            "split_manifest_sha256": manifest_sha256,
            "validation_only_data_tree_sha256": "e" * 64,
            "full_source_data_tree_opened": False,
            "target_mode": "binary",
            "test_index_opened": False,
        },
        "gain_overrides": {
            "zero_anchor": list(diagnostic.ZERO_GAIN),
            "operator_stress": list(diagnostic.STRESS_GAIN),
            "parameter_was_not_modified": True,
        },
        "semantic": {
            "schema": diagnostic.DIAGNOSTIC_SCHEMA + "/semantic_summary/v1",
            "source_method": method,
            "image_count": 160,
            "status": "GO",
        },
        "numerical": {
            "schema": diagnostic.DIAGNOSTIC_SCHEMA + "/numerical_summary/v1",
            "image_count": 160,
            "status": "GO",
        },
        "model_state_sha256_before": {
            "source": "4" * 64,
            "anchor": "5" * 64,
            "diagnostic": "6" * 64,
        },
        "model_state_sha256_after": {
            "source": "4" * 64,
            "anchor": "5" * 64,
            "diagnostic": "6" * 64,
        },
        "model_state_unchanged": {
            "source": True,
            "anchor": True,
            "diagnostic": True,
        },
        "images": [
            {
                "sample_id": f"image_{index}",
                "data_role": "val",
                "test_split_accessed": False,
            }
            for index in range(160)
        ],
        "gates": {
            "semantic": True,
            "numerical": True,
            "all_160_validation_images": True,
            "state_unchanged": True,
        },
        "status": "GO",
        "training_started": False,
        "test_split_accessed": False,
    }
    return diagnostic._with_evidence_sha256(report)


def _fake_synthetic_report() -> dict[str, object]:
    return {
        "schema": diagnostic.DIAGNOSTIC_SCHEMA + "/synthetic/v1",
        "status": "GO",
        "cases": {},
    }


def test_single_source_status_is_independent_and_strict_merge_recomputes_go() -> None:
    synthetic = _fake_synthetic_report()
    single_reports = [
        diagnostic._build_single_source_report(
            method, _fake_hashed_source_report(method), synthetic
        )
        for method in diagnostic.SOURCE_SPECS
    ]
    assert all(report["status"] == "GO" for report in single_reports)
    assert all(
        report["decision"] == "READY_FOR_TWO_SOURCE_MERGE"
        for report in single_reports
    )
    merged = diagnostic.merge_single_source_reports(single_reports)
    assert merged["status"] == "GO"
    assert merged["decision"] == "ALLOW_1_EPOCH_SMOKE"
    assert merged["execution_mode"] == "merged_single_source_reports"
    assert set(merged["sources"]) == {"sctransnet", "sbsc_v21"}
    diagnostic._verify_evidence_sha256(merged)

    mismatched_v21 = diagnostic._build_single_source_report(
        "sbsc_v21",
        _fake_hashed_source_report("sbsc_v21", manifest_sha256="f" * 64),
        synthetic,
    )
    with pytest.raises(
        diagnostic.C3V31DiagnosticError, match="different validation data"
    ):
        diagnostic.merge_single_source_reports(
            [single_reports[0], mismatched_v21]
        )


def test_atomic_json_output_and_file_merge_are_strict(
    tmp_path: Path, capsys
) -> None:
    synthetic = _fake_synthetic_report()
    paths = []
    for method in diagnostic.SOURCE_SPECS:
        report = diagnostic._build_single_source_report(
            method, _fake_hashed_source_report(method), synthetic
        )
        path = tmp_path / f"{method}.json"
        assert diagnostic.write_json_atomic(path, report) == path
        assert diagnostic.load_strict_json_report(path) == report
        paths.append(path)
    merged = diagnostic.merge_single_source_report_files(paths)
    output = tmp_path / "merged.json"
    diagnostic.write_json_atomic(output, merged)
    assert diagnostic.load_strict_json_report(output) == merged
    assert not list(tmp_path.glob(".*.tmp"))

    cli_output = tmp_path / "merged_cli.json"
    assert diagnostic.main(
        ["--merge", str(paths[0]), str(paths[1]), "--output", str(cli_output)]
    ) == 0
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["status"] == "GO"
    assert cli_summary["evidence_sha256"] == merged["evidence_sha256"]
    assert diagnostic.load_strict_json_report(cli_output) == merged

    symlink = tmp_path / "unsafe.json"
    symlink.symlink_to(output)
    with pytest.raises(
        diagnostic.C3V31DiagnosticError, match="non-symlink"
    ):
        diagnostic.write_json_atomic(symlink, merged)


def test_cli_preflight_is_read_only_and_full_run_requires_dataset_root(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        diagnostic,
        "preflight",
        lambda methods: {
            "status": "ready",
            "dataset_opened": False,
            "gpu_opened": False,
            "official_test_accessed": False,
            "training_started": False,
        },
    )
    assert diagnostic.main(["--preflight"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["dataset_opened"] is False
    assert observed["gpu_opened"] is False
    with pytest.raises(SystemExit, match="--dataset-root is required"):
        diagnostic.main([])
    with pytest.raises(SystemExit, match="one explicit --source"):
        diagnostic.main(["--single-source"])
