from __future__ import annotations

import copy

import pytest

from experiments.sbsc_v21_selection import (
    SBSCV21SelectionError,
    expected_validation_epochs,
    select_final,
    select_prefix,
)


_IDENTITY = {
    "method": "sbsc_v21",
    "dataset": "IRSTD-1K",
    "architecture_seed": 42,
    "run_seed": 42,
    "split_manifest_sha256": "1" * 64,
    "run_identity_sha256": "2" * 64,
}


def _record(epoch: int, *, miou: float, pd: float, fa: float) -> dict:
    niou = min(1.0, miou + 0.01)
    tiny_pd = min(1.0, pd + 0.01)
    loss = 1.0 - miou
    return {
        "epoch": epoch,
        "data_role": "val",
        "mIoU": miou,
        "Pd": pd,
        "Fa": fa,
        "metrics": {
            "miou": miou,
            "niou": niou,
            "pd": pd,
            "fa": fa,
            "tiny_pd": tiny_pd,
            "validation_loss": loss,
        },
        "test_split_accessed": False,
        **_IDENTITY,
        "model_state_sha256": f"{epoch:064x}",
    }


def test_expected_schedule_covers_empty_prefix_and_501_final_records():
    assert expected_validation_epochs(0) == ()
    assert expected_validation_epochs(499) == ()
    assert expected_validation_epochs(500) == (500,)
    final = expected_validation_epochs(1000)
    assert final[0] == 500
    assert final[-1] == 1000
    assert len(final) == 501


def test_prefix_retains_union_of_two_role_winners():
    history = [
        _record(500, miou=0.61, pd=0.91, fa=2e-5),
        _record(501, miou=0.63, pd=0.90, fa=1e-5),
        _record(502, miou=0.62, pd=0.95, fa=3e-5),
    ]
    payload = select_prefix(
        history, completed_epoch=502, expected_identity=_IDENTITY
    )
    assert payload["roles"]["best_mIoU"]["selected"]["epoch"] == 501
    assert payload["roles"]["best_Pd"]["selected"]["epoch"] == 502
    assert (
        payload["roles"]["best_mIoU"]["selected"]["model_state_sha256"]
        == f"{501:064x}"
    )
    assert payload["model_state_sha256_by_epoch"]["502"] == f"{502:064x}"
    assert len(payload["validation_history_sha256"]) == 64
    assert payload["retention_frontier_epochs"] == [501, 502]


def test_same_epoch_winners_keep_one_candidate_epoch():
    history = [_record(500, miou=0.61, pd=0.91, fa=2e-5)]
    payload = select_prefix(
        history, completed_epoch=500, expected_identity=_IDENTITY
    )
    assert payload["retention_frontier_epochs"] == [500]
    assert set(payload["roles"]) == {"best_mIoU", "best_Pd"}


def test_final_requires_exact_500_through_1000_history():
    history = [
        _record(
            epoch,
            miou=0.6 + (epoch - 500) * 1e-6,
            pd=0.9,
            fa=2e-5,
        )
        for epoch in range(500, 1001)
    ]
    payload = select_final(history, expected_identity=_IDENTITY)
    assert payload["validation_record_count"] == 501
    assert payload["roles"]["best_mIoU"]["selected"]["epoch"] == 1000
    with pytest.raises(SBSCV21SelectionError, match="cadence differs"):
        select_final(history[:-1], expected_identity=_IDENTITY)


@pytest.mark.parametrize(
    "mutation",
    ("test_role", "duplicate", "nan", "out_of_range", "wrong_start"),
)
def test_history_tampering_is_rejected(mutation: str):
    history = [
        _record(500, miou=0.61, pd=0.91, fa=2e-5),
        _record(501, miou=0.62, pd=0.92, fa=1e-5),
    ]
    bad = copy.deepcopy(history)
    if mutation == "test_role":
        bad[1]["data_role"] = "test"
    elif mutation == "duplicate":
        bad[1]["epoch"] = 500
    elif mutation == "nan":
        bad[1]["mIoU"] = float("nan")
    elif mutation == "out_of_range":
        bad[1]["Pd"] = 1.1
    elif mutation == "wrong_start":
        bad[0]["epoch"] = 499
    with pytest.raises(SBSCV21SelectionError):
        select_prefix(
            bad, completed_epoch=501, expected_identity=_IDENTITY
        )


def test_pre_validation_resume_accepts_only_empty_history():
    payload = select_prefix(
        [], completed_epoch=377, expected_identity=_IDENTITY
    )
    assert payload["roles"] == {}
    with pytest.raises(SBSCV21SelectionError, match="cadence differs"):
        select_prefix(
            [_record(377, miou=0.6, pd=0.9, fa=1e-5)],
            completed_epoch=377,
            expected_identity=_IDENTITY,
        )


@pytest.mark.parametrize("value", (pytest.param("missing", id="missing"), None, True))
def test_test_access_disclosure_must_be_explicit_false(value):
    record = _record(500, miou=0.61, pd=0.91, fa=2e-5)
    if value == "missing":
        record.pop("test_split_accessed")
    else:
        record["test_split_accessed"] = value
    with pytest.raises(SBSCV21SelectionError, match="explicit false"):
        select_prefix(
            [record], completed_epoch=500, expected_identity=_IDENTITY
        )


@pytest.mark.parametrize(
    "field",
    (
        "method",
        "dataset",
        "architecture_seed",
        "run_seed",
        "split_manifest_sha256",
        "run_identity_sha256",
        "model_state_sha256",
    ),
)
def test_record_provenance_cannot_be_mixed_or_omitted(field: str):
    record = _record(500, miou=0.61, pd=0.91, fa=2e-5)
    record.pop(field)
    with pytest.raises(SBSCV21SelectionError):
        select_prefix(
            [record], completed_epoch=500, expected_identity=_IDENTITY
        )


def test_public_schedule_and_role_alias_mutation_cannot_weaken_final(monkeypatch):
    import experiments.sbsc_v21_selection as module

    history = [
        _record(epoch, miou=0.61, pd=0.91, fa=2e-5)
        for epoch in range(500, 1001)
    ]
    monkeypatch.setattr(module, "TOTAL_EPOCHS", 500)
    monkeypatch.setattr(module, "VALIDATION_BEGIN_EPOCH", 500)
    monkeypatch.setattr(module, "VALID_ROLES", ("best_mIoU",))
    payload = module.select_final(history, expected_identity=_IDENTITY)
    assert payload["validation_record_count"] == 501
    assert set(payload["roles"]) == {"best_mIoU", "best_Pd"}


def test_v2_identity_and_records_are_rejected() -> None:
    v2_identity = {**_IDENTITY, "method": "sbsc_v2"}
    v2_record = _record(500, miou=0.61, pd=0.91, fa=2e-5)
    v2_record["method"] = "sbsc_v2"
    with pytest.raises(SBSCV21SelectionError, match="method differs"):
        select_prefix(
            [v2_record], completed_epoch=500, expected_identity=v2_identity
        )

    with pytest.raises(SBSCV21SelectionError, match="method differs"):
        select_prefix(
            [v2_record], completed_epoch=500, expected_identity=_IDENTITY
        )
