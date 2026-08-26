from __future__ import annotations

import copy
import json

import pytest

from experiments.sbsc_v32_test_selection import (
    RANKING_SCHEMA,
    RECORD_SCHEMA,
    SBSCV32TestSelectionError,
    SCHEDULE_SCHEMA,
    SELECTION_SCHEMA,
    expected_test_epochs,
    select_final,
    select_prefix,
)


_IDENTITY = {
    "method": "sbsc_v32",
    "dataset": "IRSTD-1K",
    "architecture_seed": 42,
    "run_seed": 42,
    "split_manifest_sha256": "1" * 64,
    "run_identity_sha256": "2" * 64,
}


def _record(
    epoch: int,
    *,
    miou: float = 0.60,
    niou: float = 0.61,
    pd: float = 0.90,
    fa: float = 2e-5,
    tiny_pd: float | None = 0.91,
    loss: float = 0.40,
) -> dict:
    return {
        "schema": RECORD_SCHEMA,
        "epoch": epoch,
        "data_role": "test",
        "mIoU": miou,
        "nIoU": niou,
        "Pd": pd,
        "Fa": fa,
        "tinyPd": tiny_pd,
        "loss": loss,
        "metrics": {
            "miou": miou,
            "niou": niou,
            "pd": pd,
            "fa": fa,
            "tiny_pd": tiny_pd,
            "test_loss": loss,
        },
        **_IDENTITY,
        "model_state_sha256": f"{epoch:064x}",
        "test_split_accessed": True,
        "test_selected": True,
        "selection_is_optimistic": True,
        "unbiased_test_claim_supported": False,
    }


def _winner(first: dict, second: dict, role: str) -> int:
    payload = select_prefix(
        [first, second], completed_epoch=501, expected_identity=_IDENTITY
    )
    return int(payload["roles"][role]["selected"]["epoch"])


def test_schemas_and_disclosures_are_explicit_even_before_epoch_500() -> None:
    payload = select_prefix([], completed_epoch=0, expected_identity=_IDENTITY)
    assert payload["schema"] == SELECTION_SCHEMA
    assert payload["schedule_schema"] == SCHEDULE_SCHEMA
    assert payload["data_role"] == "test"
    assert payload["test_split_accessed"] is True
    assert payload["test_selected"] is True
    assert payload["selection_is_optimistic"] is True
    assert payload["unbiased_test_claim_supported"] is False
    assert payload["roles"] == {}
    assert payload["distinct_physical_weights_required"] is True
    assert payload["physical_weight_materialization_owner"] == "runner"


def test_frozen_schedule_is_exactly_500_through_1000() -> None:
    assert expected_test_epochs(0) == ()
    assert expected_test_epochs(499) == ()
    assert expected_test_epochs(500) == (500,)
    final = expected_test_epochs(1000)
    assert final[0] == 500
    assert final[-1] == 1000
    assert len(final) == 501
    with pytest.raises(SBSCV32TestSelectionError, match=r"\[0, 1000\]"):
        expected_test_epochs(1001)


@pytest.mark.parametrize(
    ("updates", "expected_epoch"),
    (
        ({"miou": 0.61}, 501),
        ({"pd": 0.91}, 501),
        ({"fa": 1e-5}, 501),
        ({"niou": 0.62}, 501),
        ({"tiny_pd": 0.92}, 501),
        ({"loss": 0.39}, 501),
        ({}, 500),
    ),
)
def test_best_miou_uses_exact_zero_margin_lexicographic_key(
    updates: dict[str, float], expected_epoch: int
) -> None:
    first = _record(500)
    values = {
        "miou": 0.60,
        "niou": 0.61,
        "pd": 0.90,
        "fa": 2e-5,
        "tiny_pd": 0.91,
        "loss": 0.40,
        **updates,
    }
    second = _record(501, **values)
    # Earlier fields are equal in each fixture until the field under test.
    assert _winner(first, second, "best_mIoU") == expected_epoch


@pytest.mark.parametrize(
    ("updates", "expected_epoch"),
    (
        ({"pd": 0.91}, 501),
        ({"fa": 1e-5}, 501),
        ({"tiny_pd": 0.92}, 501),
        ({"miou": 0.61}, 501),
        ({"niou": 0.62}, 501),
        ({"loss": 0.39}, 501),
        ({}, 500),
    ),
)
def test_best_pd_uses_exact_zero_margin_lexicographic_key(
    updates: dict[str, float], expected_epoch: int
) -> None:
    first = _record(500)
    values = {
        "miou": 0.60,
        "niou": 0.61,
        "pd": 0.90,
        "fa": 2e-5,
        "tiny_pd": 0.91,
        "loss": 0.40,
        **updates,
    }
    second = _record(501, **values)
    assert _winner(first, second, "best_Pd") == expected_epoch


def test_roles_can_differ_and_same_epoch_still_has_two_role_descriptors() -> None:
    differing = [
        _record(500, miou=0.70, pd=0.90),
        _record(501, miou=0.69, pd=0.95),
    ]
    payload = select_prefix(
        differing, completed_epoch=501, expected_identity=_IDENTITY
    )
    assert payload["roles"]["best_mIoU"]["selected"]["epoch"] == 500
    assert payload["roles"]["best_Pd"]["selected"]["epoch"] == 501
    assert payload["retention_frontier_epochs"] == [500, 501]

    same = select_prefix(
        [_record(500)], completed_epoch=500, expected_identity=_IDENTITY
    )
    assert set(same["roles"]) == {"best_mIoU", "best_Pd"}
    assert same["retention_frontier_epochs"] == [500]
    assert same["distinct_physical_weights_required"] is True
    assert same["roles"]["best_mIoU"] is not same["roles"]["best_Pd"]


def test_nonempty_payload_and_nested_ranking_provenance_never_claim_unbiased_test() -> None:
    payload = select_prefix(
        [_record(500)], completed_epoch=500, expected_identity=_IDENTITY
    )
    assert payload["ranking_provenance"]["schema"] == RANKING_SCHEMA
    for container in (payload, payload["ranking_provenance"]):
        assert container["data_role"] == "test"
        assert container["test_split_accessed"] is True
        assert container["test_selected"] is True
        assert container["selection_is_optimistic"] is True
        assert container["unbiased_test_claim_supported"] is False
    assert payload["roles"]["best_mIoU"]["selected"][
        "model_state_sha256"
    ] == f"{500:064x}"


def test_undefined_tiny_pd_is_preserved_as_json_null_and_ranks_below_defined() -> None:
    history = [
        _record(500, tiny_pd=None),
        _record(501, tiny_pd=0.0),
    ]
    payload = select_prefix(
        history, completed_epoch=501, expected_identity=_IDENTITY
    )
    for role in ("best_mIoU", "best_Pd"):
        assert payload["roles"][role]["selected"]["epoch"] == 501
        assert all(
            value is None or isinstance(value, float)
            for value in payload["roles"][role]["selected"][
                "descending_rank_key"
            ]
        )
    assert payload["ranking_provenance"]["evaluated_records"][0][
        "tinyPd"
    ] is None
    # Strict JSON serialization is the artifact contract: no Infinity/NaN.
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
    with pytest.raises(SBSCV32TestSelectionError, match=field):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)
    record.pop(field)
    with pytest.raises(SBSCV32TestSelectionError, match=field):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


@pytest.mark.parametrize(
    "mutation",
    ("schema", "below_range", "duplicate", "gap", "nan", "metric_conflict"),
)
def test_record_schema_epoch_cadence_and_metrics_are_fail_closed(
    mutation: str,
) -> None:
    history = [_record(500), _record(501)]
    bad = copy.deepcopy(history)
    if mutation == "schema":
        bad[1]["schema"] = "legacy"
    elif mutation == "below_range":
        bad[0]["epoch"] = 499
    elif mutation == "duplicate":
        bad[1]["epoch"] = 500
    elif mutation == "gap":
        bad[1]["epoch"] = 502
    elif mutation == "nan":
        bad[1]["mIoU"] = float("nan")
    elif mutation == "metric_conflict":
        bad[1]["metrics"]["pd"] = 0.1
    with pytest.raises(SBSCV32TestSelectionError):
        select_prefix(bad, completed_epoch=501, expected_identity=_IDENTITY)


def test_identity_is_exactly_sbsc_v32_seed_42_and_bound_to_every_record() -> None:
    for field, value in (
        ("method", "sctransnet"),
        ("architecture_seed", 7),
        ("run_seed", 7),
    ):
        identity = {**_IDENTITY, field: value}
        with pytest.raises(SBSCV32TestSelectionError):
            select_prefix([], completed_epoch=0, expected_identity=identity)

    record = _record(500)
    record["run_identity_sha256"] = "3" * 64
    with pytest.raises(SBSCV32TestSelectionError, match="run_identity_sha256"):
        select_prefix([record], completed_epoch=500, expected_identity=_IDENTITY)


def test_final_requires_exactly_501_records() -> None:
    history = [
        _record(epoch, miou=0.60 + (epoch - 500) * 1e-6)
        for epoch in range(500, 1001)
    ]
    payload = select_final(history, expected_identity=_IDENTITY)
    assert payload["test_record_count"] == 501
    assert payload["roles"]["best_mIoU"]["selected"]["epoch"] == 1000
    with pytest.raises(SBSCV32TestSelectionError, match="cadence differs"):
        select_final(history[:-1], expected_identity=_IDENTITY)


def test_public_schedule_aliases_cannot_weaken_final(monkeypatch) -> None:
    import experiments.sbsc_v32_test_selection as module

    history = [_record(epoch) for epoch in range(500, 1001)]
    monkeypatch.setattr(module, "TOTAL_EPOCHS", 500)
    monkeypatch.setattr(module, "TEST_BEGIN_EPOCH", 500)
    monkeypatch.setattr(module, "VALID_ROLES", ("best_mIoU",))
    payload = module.select_final(history, expected_identity=_IDENTITY)
    assert payload["test_record_count"] == 501
    assert set(payload["roles"]) == {"best_mIoU", "best_Pd"}
