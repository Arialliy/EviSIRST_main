from __future__ import annotations

import copy
import json

import pytest

from experiments.sbsc_v33_test_selection import (
    RANKING_SCHEMA,
    RECORD_SCHEMA,
    SCHEDULE_SCHEMA,
    SELECTION_SCHEMA,
    SBSCV33TestSelectionError,
    expected_test_epochs,
    retention_frontier_epochs,
    select_final,
    select_prefix,
)


_IDENTITY = {
    "method": "sbsc_v33_third",
    "dataset": "IRSTD-1K",
    "architecture_seed": 42,
    "run_seed": 42,
    "split_manifest_sha256": "1" * 64,
    "run_identity_sha256": "2" * 64,
    "evaluation_contract_sha256": "8" * 64,
    "method_config_sha256": "3" * 64,
    "gradient_authorization_sha256": "4" * 64,
    "baseline_authority_manifest_sha256": "5" * 64,
    "formal_launch_authorization_sha256": "6" * 64,
    "training_source_manifest_sha256": "7" * 64,
}

_EVALUATOR_FIELDS = {
    "test_loss",
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


def _record(
    epoch: int,
    *,
    identity: dict | None = None,
    miou: float = 0.60,
    niou: float = 0.61,
    pd: float = 0.90,
    fa: float = 2e-5,
    tiny_pd: float | None = 0.80,
    loss: float = 0.40,
    include_nested: bool = True,
) -> dict:
    metrics = {
        "test_loss": loss,
        "miou": miou,
        "niou": niou,
        "pixel_precision": 0.75,
        "pixel_recall": 0.80,
        "pixel_f1": 0.7741935483870968,
        "pd": pd,
        "tiny_pd": tiny_pd,
        "fa": fa,
        "false_objects_per_image": 0.10,
        "target_count": 100,
        "matched_target_count": int(round(pd * 100)),
        "tiny_target_count": 0 if tiny_pd is None else 100,
        "matched_tiny_target_count": (
            0 if tiny_pd is None else int(round(tiny_pd * 100))
        ),
        "predicted_object_count": 12,
        "unmatched_predicted_object_count": 3,
        "valid_pixel_count": 1024,
    }
    record = {
        "schema": RECORD_SCHEMA,
        "epoch": epoch,
        **metrics,
        "mIoU": miou,
        "nIoU": niou,
        "Pd": pd,
        "Fa": fa,
        "tinyPd": tiny_pd,
        "loss": loss,
        "tiny_pd_was_undefined": tiny_pd is None,
        **(_IDENTITY if identity is None else identity),
        "model": "SCTransNet-C3-SBSC-V3.3",
        "seed": 42,
        "sample_count": 201,
        "evaluation_source_sha256": "9" * 64,
        "model_state_sha256": f"{epoch:064x}",
        "data_role": "test",
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
    }
    if include_nested:
        record["metrics"] = dict(metrics)
    return record


def _winner(first: dict, second: dict, role: str) -> int:
    payload = select_prefix(
        [first, second],
        completed_epoch=501,
        expected_identity=_IDENTITY,
    )
    return int(payload["roles"][role]["selected"]["epoch"])


def test_v33_schemas_identity_and_disclosures_are_independent_and_explicit() -> None:
    payload = select_prefix([], completed_epoch=0, expected_identity=_IDENTITY)

    assert "v33" in RECORD_SCHEMA
    assert "v33" in SELECTION_SCHEMA
    assert "v33" in SCHEDULE_SCHEMA
    assert "v33" in RANKING_SCHEMA
    assert "v32" not in RECORD_SCHEMA + SELECTION_SCHEMA
    assert payload["schema"] == SELECTION_SCHEMA
    assert payload["schedule_schema"] == SCHEDULE_SCHEMA
    assert payload["data_role"] == "test"
    assert payload["test_split_accessed"] is True
    assert payload["test_selected"] is True
    assert payload["selection_is_optimistic"] is True
    assert payload["unbiased_test_claim_supported"] is False
    assert payload["selection_identity"] == _IDENTITY
    assert payload["roles"] == {}
    assert payload["distinct_physical_weights_required"] is True


def test_schedule_is_exactly_500_through_1000_and_final_count_is_501() -> None:
    assert expected_test_epochs(0) == ()
    assert expected_test_epochs(499) == ()
    assert expected_test_epochs(500) == (500,)
    epochs = expected_test_epochs(1000)
    assert epochs == tuple(range(500, 1001))
    assert len(epochs) == 501
    for invalid in (True, -1, 1001, 2.5):
        with pytest.raises(SBSCV33TestSelectionError):
            expected_test_epochs(invalid)


@pytest.mark.parametrize(
    "method",
    ("sbsc_v33_third", "sbsc_v33_ord", "sbsc_v33_half"),
)
def test_exact_three_v33_methods_are_allowed(method: str) -> None:
    identity = {**_IDENTITY, "method": method}
    payload = select_prefix([], completed_epoch=0, expected_identity=identity)
    assert payload["selection_identity"]["method"] == method


@pytest.mark.parametrize(
    "method",
    ("sbsc_v32", "sctransnet", "sbsc_v33", "SBSC_V33_THIRD", ""),
)
def test_every_other_method_is_rejected(method: str) -> None:
    with pytest.raises(SBSCV33TestSelectionError, match="method"):
        select_prefix(
            [],
            completed_epoch=0,
            expected_identity={**_IDENTITY, "method": method},
        )


@pytest.mark.parametrize(
    ("updates", "expected_epoch"),
    (
        ({"miou": 0.61}, 501),
        ({"pd": 0.91}, 501),
        ({"fa": 1e-5}, 501),
        ({"niou": 0.62}, 501),
        ({"tiny_pd": 0.81}, 501),
        ({"loss": 0.39}, 501),
        ({}, 500),
    ),
)
def test_best_miou_uses_strict_lexicographic_order_and_earliest_tie(
    updates: dict[str, float], expected_epoch: int
) -> None:
    values = {
        "miou": 0.60,
        "niou": 0.61,
        "pd": 0.90,
        "fa": 2e-5,
        "tiny_pd": 0.80,
        "loss": 0.40,
        **updates,
    }
    assert _winner(_record(500), _record(501, **values), "best_mIoU") == (
        expected_epoch
    )


@pytest.mark.parametrize(
    ("updates", "expected_epoch"),
    (
        ({"pd": 0.91}, 501),
        ({"fa": 1e-5}, 501),
        ({"tiny_pd": 0.81}, 501),
        ({"miou": 0.61}, 501),
        ({"niou": 0.62}, 501),
        ({"loss": 0.39}, 501),
        ({}, 500),
    ),
)
def test_best_pd_uses_strict_lexicographic_order_and_earliest_tie(
    updates: dict[str, float], expected_epoch: int
) -> None:
    values = {
        "miou": 0.60,
        "niou": 0.61,
        "pd": 0.90,
        "fa": 2e-5,
        "tiny_pd": 0.80,
        "loss": 0.40,
        **updates,
    }
    assert _winner(_record(500), _record(501, **values), "best_Pd") == (
        expected_epoch
    )


def test_roles_are_separate_and_selected_records_keep_all_metrics() -> None:
    history = [
        _record(500, miou=0.70, pd=0.90),
        _record(501, miou=0.69, pd=0.95),
    ]
    payload = select_prefix(
        history,
        completed_epoch=501,
        expected_identity=_IDENTITY,
    )

    assert payload["roles"]["best_mIoU"]["selected"]["epoch"] == 500
    assert payload["roles"]["best_Pd"]["selected"]["epoch"] == 501
    assert payload["retention_frontier_epochs"] == [500, 501]
    for role in ("best_mIoU", "best_Pd"):
        selected = payload["roles"][role]["selected"]
        assert set(selected["metrics"]) == _EVALUATOR_FIELDS
        assert selected["metrics"]["pixel_f1"] == pytest.approx(
            0.7741935483870968
        )
        assert selected["model"] == "SCTransNet-C3-SBSC-V3.3"
        assert selected["seed"] == 42
        assert selected["sample_count"] == 201
        assert selected["evaluation_source_sha256"] == "9" * 64
        assert payload["roles"][role]["separate_weight_reporting_required"] is True

    same = select_prefix(
        [_record(500)],
        completed_epoch=500,
        expected_identity=_IDENTITY,
    )
    assert set(same["roles"]) == {"best_mIoU", "best_Pd"}
    assert same["retention_frontier_epochs"] == [500]
    assert same["roles"]["best_mIoU"] is not same["roles"]["best_Pd"]


def test_ranking_provenance_preserves_full_vector_and_all_disclosures() -> None:
    payload = select_prefix(
        [_record(500)],
        completed_epoch=500,
        expected_identity=_IDENTITY,
    )
    ranking = payload["ranking_provenance"]
    assert ranking["schema"] == RANKING_SCHEMA
    assert set(ranking["evaluated_records"][0]["metrics"]) == _EVALUATOR_FIELDS
    assert ranking["evaluated_records"][0]["sample_count"] == 201
    assert ranking["evaluated_records"][0]["evaluation_source_sha256"] == "9" * 64
    for container in (payload, ranking):
        assert container["data_role"] == "test"
        assert container["test_split_accessed"] is True
        assert container["test_selected"] is True
        assert container["selection_is_optimistic"] is True
        assert container["unbiased_test_claim_supported"] is False


def test_undefined_tiny_pd_is_null_and_ranks_below_defined_zero() -> None:
    payload = select_prefix(
        [_record(500, tiny_pd=None), _record(501, tiny_pd=0.0)],
        completed_epoch=501,
        expected_identity=_IDENTITY,
    )
    for role in ("best_mIoU", "best_Pd"):
        assert payload["roles"][role]["selected"]["epoch"] == 501
        assert all(
            value is None or isinstance(value, float)
            for value in payload["roles"][role]["selected"][
                "descending_rank_key"
            ]
        )
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("data_role", "val"),
        ("test_split_accessed", False),
        ("test_selected", False),
        ("selection_is_optimistic", False),
        ("unbiased_test_claim_supported", True),
    ),
)
def test_every_test_selection_disclosure_is_mandatory(
    field: str, bad_value: object
) -> None:
    record = _record(500)
    record[field] = bad_value
    with pytest.raises(SBSCV33TestSelectionError, match=field):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)

    record = _record(500)
    record.pop(field)
    with pytest.raises(SBSCV33TestSelectionError, match=field):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


@pytest.mark.parametrize("field", sorted(_EVALUATOR_FIELDS))
def test_all_seventeen_evaluator_fields_are_required(field: str) -> None:
    record = _record(500, include_nested=False)
    record.pop(field)
    with pytest.raises(SBSCV33TestSelectionError, match=field):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


@pytest.mark.parametrize(
    "field",
    ("miou", "niou", "pixel_precision", "pixel_recall", "pixel_f1", "pd"),
)
def test_probability_metrics_reject_nonfinite_bool_and_out_of_range(
    field: str,
) -> None:
    for bad_value in (float("nan"), float("inf"), True, -0.01, 1.01):
        record = _record(500)
        record[field] = bad_value
        record["metrics"][field] = bad_value
        alias = {
            "miou": "mIoU",
            "niou": "nIoU",
            "pd": "Pd",
        }.get(field)
        if alias is not None:
            record[alias] = bad_value
        with pytest.raises(SBSCV33TestSelectionError):
            select_prefix(
                [record], completed_epoch=500, expected_identity=_IDENTITY
            )


@pytest.mark.parametrize(
    "field",
    ("test_loss", "fa", "false_objects_per_image"),
)
def test_nonnegative_metrics_reject_nonfinite_bool_and_negative(field: str) -> None:
    for bad_value in (float("nan"), float("inf"), True, -1e-9):
        record = _record(500)
        record[field] = bad_value
        record["metrics"][field] = bad_value
        alias = {"test_loss": "loss", "fa": "Fa"}.get(field)
        if alias is not None:
            record[alias] = bad_value
        with pytest.raises(SBSCV33TestSelectionError):
            select_prefix(
                [record], completed_epoch=500, expected_identity=_IDENTITY
            )


@pytest.mark.parametrize(
    "field",
    (
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    ),
)
def test_count_metrics_require_nonnegative_integers(field: str) -> None:
    for bad_value in (True, -1, 1.5):
        record = _record(500)
        record[field] = bad_value
        record["metrics"][field] = bad_value
        with pytest.raises(SBSCV33TestSelectionError, match=field):
            select_prefix(
                [record], completed_epoch=500, expected_identity=_IDENTITY
            )


@pytest.mark.parametrize(
    ("larger", "smaller"),
    (
        ("matched_target_count", "target_count"),
        ("matched_tiny_target_count", "tiny_target_count"),
        ("unmatched_predicted_object_count", "predicted_object_count"),
    ),
)
def test_evaluator_count_relations_are_enforced(larger: str, smaller: str) -> None:
    record = _record(500)
    record[larger] = record[smaller] + 1
    record["metrics"][larger] = record[larger]
    with pytest.raises(SBSCV33TestSelectionError, match="counts are inconsistent"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


def test_derived_metric_and_denominator_contracts_are_enforced() -> None:
    record = _record(500)
    record["pd"] = record["Pd"] = record["metrics"]["pd"] = 0.5
    with pytest.raises(SBSCV33TestSelectionError, match="pd differs"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)

    record = _record(500)
    record["tiny_target_count"] = record["matched_tiny_target_count"] = 0
    record["metrics"]["tiny_target_count"] = 0
    record["metrics"]["matched_tiny_target_count"] = 0
    with pytest.raises(SBSCV33TestSelectionError, match="denominator/null"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)

    record = _record(500)
    record["pixel_f1"] = record["metrics"]["pixel_f1"] = 0.1
    with pytest.raises(SBSCV33TestSelectionError, match="pixel_f1"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)

    record = _record(500)
    record["target_count"] = record["matched_target_count"] = 0
    record["metrics"]["target_count"] = 0
    record["metrics"]["matched_target_count"] = 0
    with pytest.raises(SBSCV33TestSelectionError, match="target_count"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


def test_tiny_pd_null_disclosure_nested_metrics_and_aliases_must_agree() -> None:
    record = _record(500)
    record["tiny_pd_was_undefined"] = True
    with pytest.raises(SBSCV33TestSelectionError, match="tiny_pd_was_undefined"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)

    record = _record(500)
    record["metrics"].pop("pixel_f1")
    with pytest.raises(SBSCV33TestSelectionError, match="exact evaluator"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)

    record = _record(500)
    record["metrics"]["pixel_f1"] = 0.1
    with pytest.raises(SBSCV33TestSelectionError, match="conflicting"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)

    record = _record(500)
    record["mIoU"] = 0.1
    with pytest.raises(SBSCV33TestSelectionError, match="conflicting mIoU"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("model", "SCTransNet"),
        ("seed", 7),
        ("seed", True),
        ("sample_count", 200),
        ("sample_count", True),
        ("evaluation_source_sha256", "A" * 64),
    ),
)
def test_model_seed_sample_count_and_evaluator_source_are_frozen(
    field: str,
    bad_value: object,
) -> None:
    record = _record(500)
    record[field] = bad_value
    with pytest.raises(SBSCV33TestSelectionError, match=field):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


@pytest.mark.parametrize(
    "mutation",
    ("schema", "below_range", "duplicate", "gap", "unknown_field"),
)
def test_record_schema_epoch_cadence_and_field_set_are_fail_closed(
    mutation: str,
) -> None:
    bad = copy.deepcopy([_record(500), _record(501)])
    if mutation == "schema":
        bad[1]["schema"] = "legacy"
    elif mutation == "below_range":
        bad[0]["epoch"] = 499
    elif mutation == "duplicate":
        bad[1]["epoch"] = 500
    elif mutation == "gap":
        bad[1]["epoch"] = 502
    elif mutation == "unknown_field":
        bad[1]["unregistered"] = 1
    with pytest.raises(SBSCV33TestSelectionError):
        select_prefix(bad, completed_epoch=501, expected_identity=_IDENTITY)


@pytest.mark.parametrize(
    "field",
    (
        "split_manifest_sha256",
        "run_identity_sha256",
        "evaluation_contract_sha256",
        "method_config_sha256",
        "gradient_authorization_sha256",
        "baseline_authority_manifest_sha256",
        "formal_launch_authorization_sha256",
        "training_source_manifest_sha256",
    ),
)
def test_all_identity_hashes_are_exact_and_bound_to_each_record(field: str) -> None:
    missing = dict(_IDENTITY)
    missing.pop(field)
    with pytest.raises(SBSCV33TestSelectionError, match="exact frozen"):
        select_prefix([], completed_epoch=0, expected_identity=missing)

    malformed = {**_IDENTITY, field: "A" * 64}
    with pytest.raises(SBSCV33TestSelectionError, match="lowercase SHA-256"):
        select_prefix([], completed_epoch=0, expected_identity=malformed)

    record = _record(500)
    record[field] = "0" * 64
    with pytest.raises(SBSCV33TestSelectionError, match=field):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


def test_identity_rejects_extra_fields_wrong_seed_and_changes_hash() -> None:
    with pytest.raises(SBSCV33TestSelectionError, match="exact frozen"):
        select_prefix(
            [],
            completed_epoch=0,
            expected_identity={**_IDENTITY, "extra": "forbidden"},
        )
    for field in ("architecture_seed", "run_seed"):
        with pytest.raises(SBSCV33TestSelectionError, match=field):
            select_prefix(
                [],
                completed_epoch=0,
                expected_identity={**_IDENTITY, field: 7},
            )

    first = select_prefix([], completed_epoch=0, expected_identity=_IDENTITY)
    changed = select_prefix(
        [],
        completed_epoch=0,
        expected_identity={**_IDENTITY, "method_config_sha256": "8" * 64},
    )
    assert first["selection_identity_sha256"] != changed[
        "selection_identity_sha256"
    ]


def test_final_requires_exactly_501_contiguous_records() -> None:
    history = [
        _record(epoch, miou=0.60 + (epoch - 500) * 1e-6)
        for epoch in range(500, 1001)
    ]
    payload = select_final(history, expected_identity=_IDENTITY)
    assert payload["test_record_count"] == 501
    assert payload["roles"]["best_mIoU"]["selected"]["epoch"] == 1000
    assert set(payload["roles"]) == {"best_mIoU", "best_Pd"}
    with pytest.raises(SBSCV33TestSelectionError, match="cadence differs"):
        select_final(history[:-1], expected_identity=_IDENTITY)


def test_retention_frontier_is_union_of_both_role_winners() -> None:
    assert retention_frontier_epochs(
        [_record(500, miou=0.70), _record(501, miou=0.69, pd=0.95)],
        completed_epoch=501,
        expected_identity=_IDENTITY,
    ) == (500, 501)


def test_public_aliases_cannot_weaken_frozen_final_protocol(monkeypatch) -> None:
    import experiments.sbsc_v33_test_selection as module

    history = [_record(epoch) for epoch in range(500, 1001)]
    monkeypatch.setattr(module, "TOTAL_EPOCHS", 500)
    monkeypatch.setattr(module, "TEST_BEGIN_EPOCH", 500)
    monkeypatch.setattr(module, "VALID_ROLES", ("best_mIoU",))
    payload = module.select_final(history, expected_identity=_IDENTITY)
    assert payload["test_record_count"] == 501
    assert set(payload["roles"]) == {"best_mIoU", "best_Pd"}
