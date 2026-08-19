from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from hf_decoder_v2 import (
    ContextPurifiedHFAdapter,
    assert_zero_init_identity,
    attach_hf_decoder_v2,
    load_clean_r1_warm_start,
    trainable_parameter_count,
)


def test_s2_adapter_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(42)
    adapter = ContextPurifiedHFAdapter(32, 64).eval()
    feature = torch.randn(2, 32, 64, 64)
    skip = torch.randn(2, 64, 64, 64)
    assert_zero_init_identity(adapter, feature, skip)


def test_s21_reference_parameter_budget() -> None:
    d2 = ContextPurifiedHFAdapter(32, 64)
    d1 = ContextPurifiedHFAdapter(32, 32)
    assert trainable_parameter_count(d2) == 4_549
    assert trainable_parameter_count(d1) == 2_469
    assert trainable_parameter_count(d2) + trainable_parameter_count(d1) == 7_018
    assert 7_018 / 10_870_130 < 0.001


class DummyCleanR1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


def test_baseline_warm_start_allows_only_adapter_state() -> None:
    torch.manual_seed(42)
    baseline = DummyCleanR1()
    clean_state = copy.deepcopy(baseline.state_dict())
    candidate = DummyCleanR1()
    attach_hf_decoder_v2(candidate, variant="hf_v2_s21", base_channel=32)
    report = load_clean_r1_warm_start(candidate, {"state_dict": clean_state})
    assert report["unexpected_keys"] == []
    assert report["missing_adapter_keys"]
    assert torch.equal(candidate.core.weight, baseline.core.weight)
    assert torch.equal(candidate.core.bias, baseline.core.bias)


def test_warm_start_rejects_missing_clean_weight() -> None:
    baseline = DummyCleanR1()
    incomplete_state = copy.deepcopy(baseline.state_dict())
    incomplete_state.pop("core.weight")
    candidate = DummyCleanR1()
    attach_hf_decoder_v2(candidate, variant="hf_v2_s2", base_channel=32)
    with pytest.raises(RuntimeError, match="invalid_missing"):
        load_clean_r1_warm_start(candidate, {"state_dict": incomplete_state})


def test_branch_gradients_open_after_rezero_scale_moves() -> None:
    torch.manual_seed(7)
    adapter = ContextPurifiedHFAdapter(32, 64)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    feature = torch.randn(2, 32, 16, 16)
    skip = torch.randn(2, 64, 16, 16)
    target = torch.randn_like(feature)

    optimizer.zero_grad(set_to_none=True)
    loss = (adapter(feature, skip) - target).square().mean()
    loss.backward()
    assert adapter.raw_scale.grad is not None
    assert float(adapter.raw_scale.grad.abs()) > 0.0
    first_branch_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in adapter.named_parameters()
        if name != "raw_scale" and parameter.grad is not None
    )
    assert first_branch_grad == 0.0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    loss = (adapter(feature, skip) - target).square().mean()
    loss.backward()
    second_branch_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in adapter.named_parameters()
        if name != "raw_scale" and parameter.grad is not None
    )
    assert second_branch_grad > 0.0


def test_full_v2_resume_remains_strict() -> None:
    model = DummyCleanR1()
    attach_hf_decoder_v2(model, variant="hf_v2_s2", base_channel=32)
    cloned = DummyCleanR1()
    attach_hf_decoder_v2(cloned, variant="hf_v2_s2", base_channel=32)
    incompatible = cloned.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
