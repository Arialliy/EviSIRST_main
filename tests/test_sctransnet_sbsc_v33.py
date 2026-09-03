from __future__ import annotations

import copy

import pytest
import torch

from experiments import sctransnet_sbsc_v32 as v32
from experiments import sctransnet_sbsc_v33 as v33


@pytest.fixture(autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _synthetic_capture(
    *,
    batch: int = 2,
    height: int = 4,
    width: int = 4,
) -> tuple[
    v33.C3V33TrainingRouterCollector,
    tuple[torch.Tensor, ...],
]:
    logits = tuple(
        torch.randn(
            batch,
            3,
            height,
            width,
            generator=torch.Generator().manual_seed(100 + index),
            requires_grad=True,
        )
        for index in range(4)
    )
    supports = tuple(torch.softmax(value, dim=1) for value in logits)
    availability = tuple(
        torch.ones(batch, 3, dtype=torch.bool) for _ in range(4)
    )
    mode_codes = tuple(
        torch.zeros(
            batch,
            1,
            8 + index,
            1,
            dtype=torch.int64,
        )
        for index in range(4)
    )
    capture = v33.C3V33TrainingRouterCollector(
        records=[
            {
                "schema": v33.SBSC_V33_SCHEMA + "/training_router_capture/v1",
                "module_id": 1,
                "batch_size": batch,
                "token_hw": (height, width),
                "logits": logits,
                "supports": supports,
                "availability": availability,
                "intended_mode_codes": mode_codes,
                "effective_mode_codes": mode_codes,
            }
        ]
    )
    return capture, logits


def _target_and_prediction(
    batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(batch, 1, 16, 16)
    target[0, :, 2:5, 3:6] = 1.0
    if batch > 1:
        target[1, :, 8:10, 8:11] = 1.0
    prediction = torch.rand(
        batch,
        1,
        16,
        16,
        generator=torch.Generator().manual_seed(7),
    )
    return target, prediction


def test_seed42_constructor_preserves_rng_and_router_identity():
    torch.manual_seed(903)
    before = torch.random.get_rng_state().clone()
    model, metadata = v33.build_sctransnet_sbsc_v33(
        dataset="IRSTD-1K",
        training=True,
        router_value_gradient_mode="live",
    )
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after)
    assert len(model.state_dict()) == v33.EXPECTED_SBSC_V33_STATE_KEY_COUNT
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        v33.EXPECTED_SBSC_V33_PARAMETER_COUNT
    )
    assert metadata["router_initial_state_sha256"] == (
        v33.EXPECTED_SBSC_V33_ROUTER_INIT_SHA256
    )
    module = model.mtc.encoder.layer[1].channel_attn
    assert type(module) is v33.RoleExclusiveTriEvidenceProjectionV33
    assert module.router_value_gradient_mode == "live"
    assert module.raw_dual_risk_level_gain.dtype is torch.float32
    assert torch.count_nonzero(module.raw_dual_risk_level_gain).item() == 0
    assert all(
        parameter.dtype is torch.float32
        for parameter in module.exclusive_tri_router.parameters()
    )


def test_v33_and_v32_raw_states_are_mutually_rejected():
    v33_model, _metadata = v33.build_sctransnet_sbsc_v33(
        dataset="IRSTD-1K",
        training=True,
        router_value_gradient_mode="live",
    )
    v32_model, _metadata = v32.build_sctransnet_sbsc_v32_method(
        "sbsc_v32",
        "IRSTD-1K",
        training=True,
    )
    with pytest.raises(ValueError, match="state key set differs"):
        v33.validate_sbsc_v33_state_dict(
            v32_model.state_dict(),
            "sbsc_v33_third",
        )
    with pytest.raises(ValueError, match="state key set differs"):
        v32.validate_sbsc_v32_state_dict(
            v33_model.state_dict(),
            "sbsc_v32",
        )


def test_role_axis_probability_and_mass_aware_availability():
    logits = torch.tensor(
        [
            [
                [[8.0, 8.0], [8.0, 8.0]],
                [[-8.0, -8.0], [-8.0, -8.0]],
                [[-8.0, -8.0], [-8.0, -8.0]],
            ]
        ]
    )
    probability, support, mass, integrated, existence, availability = (
        v33._mass_aware_role_statistics(
            logits,
            finite_row=torch.ones(1, dtype=torch.bool),
        )
    )
    assert torch.allclose(probability.sum(dim=1), torch.ones(1, 2, 2))
    assert torch.allclose(support.sum(dim=-1), torch.ones(1, 3))
    assert mass.shape == integrated.shape == existence.shape == (1, 3)
    assert availability.tolist() == [[True, False, False]]
    assert mass[0, 0] >= 1.0
    assert integrated[0, 0] >= 0.25


def test_build_role_targets_is_exact_simplex_and_rejects_bad_teacher():
    target, prediction = _target_and_prediction()
    targets = v33.build_role_targets_v33(target, prediction.detach(), (4, 4))
    assert targets.probability.shape == (2, 3, 4, 4)
    assert targets.probability.dtype is torch.float32
    assert torch.equal(
        targets.probability[:, 0:1],
        targets.target_region,
    )
    assert torch.equal(
        targets.target_region + targets.background_region,
        torch.ones_like(targets.target_region),
    )
    assert torch.equal(
        targets.probability.sum(dim=1),
        torch.ones(2, 4, 4),
    )
    bad = prediction.clone()
    bad[0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        v33.build_role_targets_v33(target, bad, (4, 4))
    with pytest.raises(ValueError, match="explicitly detached"):
        v33.build_role_targets_v33(
            target,
            prediction.clone().requires_grad_(),
            (4, 4),
        )


@pytest.mark.parametrize(
    "balance_mode",
    v33.SUPPORTED_BALANCE_MODES,
)
def test_router_loss_is_exact_mean_of_four_and_returns_per_role(
    balance_mode: str,
):
    capture, logits = _synthetic_capture()
    target, prediction = _target_and_prediction()
    breakdown = v33.router_loss_v33(
        capture,
        prediction.detach(),
        target,
        balance_mode=balance_mode,
    )
    assert breakdown.level_reduction == "mean"
    assert breakdown.balance_mode == balance_mode
    assert len(breakdown.per_level) == 4
    assert len(breakdown.per_level_per_image) == 4
    assert len(breakdown.per_level_per_role) == 4
    assert all(value.shape == (2,) for value in breakdown.per_level_per_image)
    assert all(value.shape == (2, 3) for value in breakdown.per_level_per_role)
    assert torch.equal(
        breakdown.total,
        torch.stack(breakdown.per_level).mean(),
    )
    for per_image, per_role in zip(
        breakdown.per_level_per_image,
        breakdown.per_level_per_role,
        strict=True,
    ):
        assert torch.allclose(per_image, per_role.sum(dim=1))
    breakdown.total.backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in logits
    )


def test_router_loss_rejects_capture_tampering():
    capture, _logits = _synthetic_capture()
    target, prediction = _target_and_prediction()
    capture.records[0]["supports"] = capture.records[0]["supports"][:3]
    with pytest.raises(RuntimeError, match="four records"):
        v33.router_loss_v33(
            capture,
            prediction.detach(),
            target,
            balance_mode="one_third_two_thirds",
        )


def test_merge_uses_combined_emission_fallback_and_records_effective_mode():
    base = torch.full((1, 1, 4, 4), 0.25)
    full = base.clone()
    hard = base.clone()
    full[:, :, 0] = torch.tensor([0.4, 0.2, 0.2, 0.2])
    hard[:, :, 1] = torch.tensor([0.1, 0.3, 0.3, 0.3])
    intended = torch.tensor(
        [[[[v33.MODE_DUAL], [v33.MODE_HARD_ONLY], [v33.MODE_BACKGROUND_ONLY], [0]]]],
        dtype=torch.int64,
    )
    shape = torch.Size((1, 1, 4, 1))
    full_diagnostics = {
        "emitted_certificate_ok": torch.ones(shape, dtype=torch.bool),
        "emission_fallback": torch.zeros(shape, dtype=torch.bool),
        "projection_accepted": torch.ones(shape, dtype=torch.bool),
    }
    hard_diagnostics = copy.deepcopy(full_diagnostics)
    # This is the exact V3.1 corner case missed by checking only emitted_ok.
    hard_diagnostics["emission_fallback"][0, 0, 1, 0] = True
    output, effective, fallback = v33.merge_mode_routed_attention_v33(
        base_attention=base,
        full_attention=full,
        hard_attention=hard,
        intended_mode=intended,
        full_diagnostics=full_diagnostics,
        hard_diagnostics=hard_diagnostics,
    )
    assert effective.flatten().tolist() == [
        v33.MODE_DUAL,
        v33.MODE_IDENTITY,
        v33.MODE_BACKGROUND_ONLY,
        v33.MODE_IDENTITY,
    ]
    assert fallback.flatten().tolist() == [False, True, False, False]
    assert torch.equal(output[0, 0, 1], base[0, 0, 1])
    assert torch.equal(output.sum(dim=-1), base.sum(dim=-1))


def test_zero_gain_is_six_output_bitwise_paired_identity():
    baseline, candidate, _metadata = v33.build_paired_sctransnet_sbsc_v33(
        "IRSTD-1K",
        method="sbsc_v33_third",
        router_value_gradient_mode="live",
    )
    baseline.train()
    candidate.train()
    baseline.mode = "train"
    candidate.mode = "train"
    inputs = torch.randn(
        1,
        1,
        32,
        32,
        generator=torch.Generator().manual_seed(133),
    )
    with torch.no_grad():
        baseline_outputs = baseline(inputs)
        candidate_outputs = candidate(inputs)
    assert len(baseline_outputs) == len(candidate_outputs) == 6
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            baseline_outputs,
            candidate_outputs,
            strict=True,
        )
    )


def test_live_and_detached_router_gradient_boundaries_are_distinct():
    inputs = torch.randn(
        1,
        1,
        32,
        32,
        generator=torch.Generator().manual_seed(211),
    )
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 12:16, 13:17] = 1.0
    observed: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    for mode in ("live", "detached"):
        model, _metadata = v33.build_sctransnet_sbsc_v33(
            dataset="IRSTD-1K",
            training=True,
            router_value_gradient_mode=mode,
        )
        with v33.capture_c3_v33_training_router(model) as capture:
            outputs = model(inputs)
        router_loss = v33.router_loss_v33(
            capture,
            outputs[-1].detach(),
            target,
            balance_mode="one_third_two_thirds",
        ).total
        module = model.mtc.encoder.layer[1].channel_attn
        observed[mode] = torch.autograd.grad(
            router_loss,
            (module.mheadv.weight, module.v.weight),
            allow_unused=True,
        )
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient).item() > 0
        for gradient in observed["live"]
    )
    assert observed["detached"] == (None, None)


def test_complete_training_loss_backward_is_finite_at_fresh_identity():
    model, _metadata = v33.build_sctransnet_sbsc_v33(
        dataset="IRSTD-1K",
        training=True,
        router_value_gradient_mode="live",
    )
    inputs = torch.randn(
        1,
        1,
        32,
        32,
        generator=torch.Generator().manual_seed(307),
    )
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 10:15, 11:16] = 1.0
    total, segmentation, router, breakdown = v33.training_losses_v33(
        model,
        inputs,
        target,
        torch.nn.BCELoss(reduction="mean"),
        balance_mode="one_third_two_thirds",
    )
    assert torch.isfinite(total)
    assert torch.isfinite(segmentation)
    assert torch.isfinite(router)
    assert torch.equal(router, torch.stack(breakdown.per_level).mean())
    total.backward()
    module = model.mtc.encoder.layer[1].channel_attn
    for parameter in (
        module.exclusive_tri_router.value_proj.weight,
        module.exclusive_tri_router.head.weight,
        module.mheadv.weight,
        module.v.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
