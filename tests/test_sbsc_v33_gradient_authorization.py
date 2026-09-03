from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools import diagnose_sbsc_v33_gradient_conflict as diagnostic


def test_none_gradients_are_zero_filled_in_exact_parameter_order():
    first = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    second = torch.nn.Parameter(torch.tensor([5.0, 7.0, 11.0]))
    ordered = diagnostic.OrderedParameterSet(
        name="synthetic",
        names=("first", "second"),
        parameters=(first, second),
    )
    left = diagnostic.aligned_gradient_tensors(
        ordered,
        (torch.tensor([1.0, 2.0]), None),
    )
    right = diagnostic.aligned_gradient_tensors(
        ordered,
        (None, torch.tensor([3.0, 4.0, 5.0])),
    )
    assert left[0].tolist() == [1.0, 2.0]
    assert torch.count_nonzero(left[1]).item() == 0
    assert torch.count_nonzero(right[0]).item() == 0
    assert right[1].tolist() == [3.0, 4.0, 5.0]
    dot, _left_norm, _right_norm = diagnostic._gradient_dot_norms(left, right)
    assert float(dot) == 0.0


def test_zero_norm_statistics_are_invalid_not_implicit_live_evidence():
    zero = torch.zeros(3)
    nonzero = torch.ones(3)
    result = diagnostic._statistics_from_aligned((zero,), (nonzero,))
    assert result["valid"] is False
    assert result["invalid_reason"] == "zero_or_nonfinite_norm"
    assert result["cosine"] is None
    assert result["router_to_segmentation_norm"] is None


def _record(cosine: float, ratio: float) -> dict:
    groups = {}
    for name in ("value_spatial", "direct_v", "upstream_shared"):
        groups[name] = {
            "valid": True,
            "cosine": cosine,
            "router_to_segmentation_norm": ratio,
        }
    return {"groups": groups}


def test_detached_requires_every_dataset_and_every_group():
    rules = diagnostic.load_rules()
    negative = {
        dataset: [_record(-0.5, 0.2) for _ in range(16)]
        for dataset in rules["datasets"]
    }
    decision = diagnostic.decide_authorized_mode(negative, rules)
    assert decision["authorized_router_value_gradient_mode"] == "detached"
    negative["IRSTD-1K"][0]["groups"]["upstream_shared"]["cosine"] = 0.5
    # One outlier does not necessarily change a median; change nine batches.
    for index in range(9):
        negative["IRSTD-1K"][index]["groups"]["upstream_shared"][
            "cosine"
        ] = 0.5
    decision = diagnostic.decide_authorized_mode(negative, rules)
    assert decision["authorized_router_value_gradient_mode"] == "live"
    assert decision["all_dataset_group_detached_conditions_pass"] is False


def test_missing_or_invalid_batches_fail_to_live():
    rules = diagnostic.load_rules()
    records = {
        dataset: [_record(-0.5, 0.2) for _ in range(16)]
        for dataset in rules["datasets"]
    }
    records["NUAA-SIRST"][3]["groups"]["direct_v"] = {
        "valid": False,
        "cosine": None,
        "router_to_segmentation_norm": None,
    }
    decision = diagnostic.decide_authorized_mode(records, rules)
    assert decision["authorized_router_value_gradient_mode"] == "live"
    assert (
        decision["aggregates"]["NUAA-SIRST"]["direct_v"][
            "valid_batch_count"
        ]
        == 15
    )


def test_write_once_json_is_idempotent_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(diagnostic, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "artifacts" / "record.json"
    first = {"schema": "test/v1", "value": 1}
    first_sha = diagnostic._write_once_json(path, first)
    assert first_sha == diagnostic._write_once_json(path, first)
    assert json.loads(path.read_text(encoding="utf-8")) == first
    with pytest.raises(FileExistsError, match="write-once artifact differs"):
        diagnostic._write_once_json(path, {"schema": "test/v1", "value": 2})


def test_rules_are_strictly_frozen():
    rules = diagnostic.load_rules()
    assert rules["datasets"] == [
        "NUAA-SIRST",
        "NUDT-SIRST",
        "IRSTD-1K",
    ]
    assert rules["samples_per_dataset"] == 64
    assert rules["batches_per_dataset"] == 16
    assert rules["test_loader_constructed"] is False
    assert rules["fresh_state_per_batch"] is True

