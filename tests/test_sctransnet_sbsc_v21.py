from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments.sctransnet_sbsc_v21 as core
from experiments.four_dataset_models_seed42_v1 import (
    ORIGINAL_PARAMETER_COUNT,
    ORIGINAL_STATE_KEY_COUNT,
    state_dict_sha256,
)
from experiments.sctransnet_sbsc_v21 import (
    EXPECTED_SBSC_V21_PARAMETER_COUNT,
    EXPECTED_SBSC_V21_STATE_KEY_COUNT,
    QueryValidatedRareSimplexTransportV21,
    SBSC_V21_GAIN_MAX,
    SBSC_V21_GAIN_STATE_KEY,
    SBSC_V21_GAIN_SUFFIX,
    build_paired_sctransnet_sbsc_v21,
    build_sctransnet_sbsc_v21_method,
    estimate_sbsc_v21_support,
    project_sbsc_v21_constraints_,
    structurally_inactive_parameter_names,
    validate_sbsc_v21_state_dict,
    validate_sctransnet_sbsc_v21,
)
from model.SCTransNet import Attention_org


@pytest.fixture(scope="module")
def paired_models():
    torch.manual_seed(7301)
    expected = torch.rand(7)
    torch.manual_seed(7301)
    baseline, candidate, metadata = build_paired_sctransnet_sbsc_v21(
        "IRSTD-1K"
    )
    observed = torch.rand(7)
    assert torch.equal(observed, expected)
    return baseline, candidate, metadata


def _module(model):
    module = model.mtc.encoder.layer[1].channel_attn
    assert type(module) is QueryValidatedRareSimplexTransportV21
    return module


def _normalized_qk(seed: int = 19, *, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    queries = tuple(
        F.normalize(
            torch.randn(2, 1, channels, 16, generator=generator).to(dtype),
            dim=-1,
        )
        for channels in (32, 64, 128, 256)
    )
    key = F.normalize(
        torch.randn(2, 1, 480, 16, generator=generator).to(dtype),
        dim=-1,
    )
    return queries, key


def _base_attentions(module, queries, key):
    return tuple(
        module.softmax(
            module.psi((query @ key.transpose(-2, -1)) / math.sqrt(480.0))
        )
        for query in queries
    )


def test_pair_contract_is_one_layer_one_parameter_and_cross_version_isolated(
    paired_models,
):
    baseline, candidate, metadata = paired_models
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    baseline_modules = tuple(
        layer.channel_attn for layer in baseline.mtc.encoder.layer
    )
    candidate_modules = tuple(
        layer.channel_attn for layer in candidate.mtc.encoder.layer
    )

    assert all(type(module) is Attention_org for module in baseline_modules)
    assert type(candidate_modules[1]) is QueryValidatedRareSimplexTransportV21
    assert all(
        type(candidate_modules[index]) is Attention_org
        for index in (0, 2, 3)
    )
    assert len(baseline_state) == ORIGINAL_STATE_KEY_COUNT == 510
    assert len(candidate_state) == EXPECTED_SBSC_V21_STATE_KEY_COUNT == 511
    assert sum(p.numel() for p in baseline.parameters()) == ORIGINAL_PARAMETER_COUNT
    assert (
        sum(p.numel() for p in candidate.parameters())
        == EXPECTED_SBSC_V21_PARAMETER_COUNT
        == 11_325_940
    )
    assert set(candidate_state) - set(baseline_state) == {
        SBSC_V21_GAIN_STATE_KEY
    }
    assert SBSC_V21_GAIN_STATE_KEY.endswith(".raw_simplex_gain")
    assert "raw_transport_gain" not in SBSC_V21_GAIN_STATE_KEY
    assert all(
        torch.equal(baseline_state[key], candidate_state[key])
        for key in baseline_state
    )
    assert metadata["method"] == "sbsc_v21"
    assert metadata["replaced_block_indices"] == [1]
    assert metadata["gain_state_keys"] == [SBSC_V21_GAIN_STATE_KEY]
    assert metadata["gain_bounds"] == [0.0, 0.5]
    assert metadata["shared_state_bitwise_equal"] is True
    assert metadata["shared_state_sha256"] == state_dict_sha256(baseline_state)
    assert metadata["parent_checkpoint"] is None
    assert metadata["warm_start_used"] is False
    assert metadata["predecessor_checkpoint_used"] is False
    assert len(structurally_inactive_parameter_names(candidate)) == 68
    validate_sctransnet_sbsc_v21(candidate, require_zero_gain=True)
    validate_sbsc_v21_state_dict(baseline_state, "sctransnet")
    validate_sbsc_v21_state_dict(candidate_state, "sbsc_v21")


@pytest.mark.parametrize("resolution", (32, 256))
def test_gain_zero_is_exact_six_output_sctransnet_identity(
    paired_models, resolution
):
    baseline, candidate, _metadata = paired_models
    baseline.eval()
    candidate.eval()
    baseline.mode = "train"
    candidate.mode = "train"
    generator = torch.Generator().manual_seed(101 + resolution)
    inputs = torch.randn(1, 1, resolution, resolution, generator=generator)
    with torch.no_grad():
        baseline_outputs = baseline(inputs)
        candidate_outputs = candidate(inputs)
    assert len(baseline_outputs) == len(candidate_outputs) == 6
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    )


def test_gain_zero_is_exact_under_cpu_amp(paired_models):
    baseline, candidate, _metadata = paired_models
    baseline.eval()
    candidate.eval()
    baseline.mode = "train"
    candidate.mode = "train"
    inputs = torch.randn(1, 1, 32, 32)
    with torch.no_grad(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        baseline_outputs = baseline(inputs)
        candidate_outputs = candidate(inputs)
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    )


def test_support_matches_frozen_k_rarity_and_query_validation_formula(
    paired_models,
):
    _baseline, candidate, _metadata = paired_models
    module = _module(candidate)
    queries, key = _normalized_qk()
    attentions = _base_attentions(module, queries, key)
    support = estimate_sbsc_v21_support(key, queries, attentions)

    work_key = key.float()
    centered_key = work_key - work_key.mean(dim=-1, keepdim=True)
    rarity = torch.sqrt(centered_key.square().mean(dim=-2))
    log_rarity = torch.log(rarity.clamp_min(1e-6))
    z_key = log_rarity - log_rarity.mean(dim=-1, keepdim=True)
    bounded = torch.tanh(z_key)
    h_key = bounded - bounded.mean(dim=-1, keepdim=True)
    positive_rarity = F.relu(h_key)
    rho = positive_rarity / (
        positive_rarity.amax(dim=-1, keepdim=True) + 1e-6
    )
    relative = bounded.abs().amax(dim=-1, keepdim=True)
    absolute = torch.tanh(
        math.sqrt(16.0) * rarity.amax(dim=-1, keepdim=True)
    )

    assert torch.equal(support.rarity, rarity)
    assert torch.equal(support.log_rarity_score, z_key)
    assert torch.equal(support.bounded_score, bounded)
    assert torch.equal(support.centered_score, h_key)
    assert torch.equal(support.normalized_positive_rarity, rho.unsqueeze(-2))
    assert torch.equal(support.relative_confidence, relative.unsqueeze(-2))
    assert torch.equal(support.absolute_confidence, absolute.unsqueeze(-2))
    assert torch.equal(
        support.key_confidence, (relative * absolute).unsqueeze(-2)
    )

    level = support.levels[0]
    query = queries[0].float()
    attention = attentions[0].float()
    moment = (query * (attention @ work_key)).mean(dim=-2, keepdim=True)
    centered = moment - moment.mean(dim=-1, keepdim=True)
    mean_square = centered.square().mean(dim=-1, keepdim=True)
    standardized = centered / (torch.sqrt(mean_square) + 1e-6)
    signed = torch.tanh(standardized)
    positive = rho.unsqueeze(-2) * F.relu(signed)
    negative = rho.unsqueeze(-2) * F.relu(-signed)
    positive_mass = positive.sum(dim=-1, keepdim=True)
    negative_mass = negative.sum(dim=-1, keepdim=True)
    expected_positive = positive / positive_mass.clamp_min(1e-6)
    expected_negative = negative / negative_mass.clamp_min(1e-6)
    balance = 2.0 * torch.minimum(positive_mass, negative_mass) / (
        positive_mass + negative_mass + 1e-6
    )
    query_confidence = torch.tanh(16.0 * torch.sqrt(mean_square))
    confidence = balance * torch.sqrt(
        ((relative * absolute).unsqueeze(-2) * query_confidence).clamp_min(0)
    )

    assert torch.equal(level.query_moment, moment)
    assert torch.equal(level.centered_query_moment, centered)
    assert torch.equal(level.standardized_query_moment, standardized)
    assert torch.equal(level.signed_query_validation, signed)
    assert torch.allclose(level.positive_support, expected_positive)
    assert torch.allclose(level.negative_support, expected_negative)
    assert torch.allclose(level.balance, balance)
    assert torch.equal(level.query_confidence, query_confidence)
    assert torch.allclose(level.confidence, confidence)
    assert torch.all((level.confidence >= 0) & (level.confidence <= 1))
    assert not any(
        tensor.requires_grad
        for tensor in (
            support.rarity,
            support.normalized_positive_rarity,
            level.query_moment,
            level.positive_support,
            level.negative_support,
            level.confidence,
        )
    )


def test_no_k_variation_uses_uniform_support_and_exact_zero_confidence(
    paired_models,
):
    _baseline, candidate, _metadata = paired_models
    module = _module(candidate)
    queries, _key = _normalized_qk(seed=31)
    key = F.normalize(torch.ones(2, 1, 480, 16), dim=-1)
    attentions = _base_attentions(module, queries, key)
    support = estimate_sbsc_v21_support(key, queries, attentions)
    uniform = torch.full((2, 1, 1, 16), 1.0 / 16.0)

    assert not torch.any(support.has_spatial_variation)
    assert torch.count_nonzero(support.normalized_positive_rarity).item() == 0
    for level in support.levels:
        assert torch.equal(level.positive_support, uniform)
        assert torch.equal(level.negative_support, uniform)
        assert torch.count_nonzero(level.confidence).item() == 0
        assert not torch.any(level.has_two_sided_mass)


def test_simplex_transport_formula_bounds_and_float64_zero_identity(paired_models):
    _baseline, candidate, _metadata = paired_models
    module = _module(candidate)
    queries, key = _normalized_qk(seed=47)
    attentions = _base_attentions(module, queries, key)
    support = estimate_sbsc_v21_support(key, queries, attentions)
    attention, diagnostics = module._simplex_tangent_transport(
        queries[0],
        key,
        attentions[0],
        support.levels[0],
        module.effective_transport_gain(0.5),
    )

    factor = diagnostics["simplex_factor"]
    assert torch.isfinite(attention).all()
    assert factor.amin().item() >= 0.0
    assert factor.amax().item() <= 2.0
    assert diagnostics["total_variation"].amin().item() >= 0.0
    assert diagnostics["total_variation"].amax().item() <= 1.0
    assert attention.amin().item() >= 0.0
    assert torch.allclose(
        attention.sum(dim=-1),
        attentions[0].sum(dim=-1),
        rtol=0.0,
        atol=2e-6,
    )
    assert (
        (attention - attentions[0]).abs().sum(dim=-1).amax().item()
        <= 0.5 + 2e-6
    )
    weighted_tangent = (
        attentions[0].float() * diagnostics["centered_direction"]
    ).sum(dim=-1)
    assert weighted_tangent.abs().amax().item() <= 2e-6

    base64 = attentions[0].double()
    zero64, zero_diagnostics = module._simplex_tangent_transport(
        queries[0].double(),
        key.double(),
        base64,
        support.levels[0],
        torch.zeros((), dtype=torch.float64),
    )
    assert torch.equal(zero64, base64)
    assert torch.count_nonzero(
        zero_diagnostics["transport_correction"]
    ).item() == 0


def test_amp_keeps_conditional_math_fp32_and_returns_ambient_attention(
    paired_models,
):
    _baseline, candidate, _metadata = paired_models
    module = _module(candidate)
    queries, key = _normalized_qk(seed=53)
    with torch.no_grad(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        attentions = _base_attentions(module, queries, key)
        support = estimate_sbsc_v21_support(key, queries, attentions)
        attention, diagnostics = module._simplex_tangent_transport(
            queries[0],
            key,
            attentions[0],
            support.levels[0],
            module.effective_transport_gain(0.5),
        )
    assert diagnostics["positive_attention"].dtype == torch.float32
    assert diagnostics["negative_attention"].dtype == torch.float32
    assert diagnostics["log_ratio_direction"].dtype == torch.float32
    assert diagnostics["simplex_factor"].dtype == torch.float32
    assert diagnostics["transport_correction_fp32"].dtype == torch.float32
    assert diagnostics["transport_correction"].dtype == attentions[0].dtype
    assert attention.dtype == attentions[0].dtype
    assert torch.isfinite(attention).all()
    assert attention.amin().item() >= 0.0
    assert torch.allclose(
        attention.float().sum(dim=-1),
        attentions[0].float().sum(dim=-1),
        rtol=0.0,
        atol=5e-3,
    )


def test_zero_init_ste_and_live_qk_have_finite_nonzero_gradients(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    module = _module(candidate)
    generator = torch.Generator().manual_seed(221)
    embeddings = tuple(
        torch.randn(2, channels, 2, 2, generator=generator)
        for channels in (32, 64, 128, 256)
    )
    emb_all = torch.cat(embeddings, dim=1)
    outputs = module(*embeddings, emb_all)[:4]
    objective = sum(output.square().mean() for output in outputs)
    requested = (
        module.raw_simplex_gain,
        module.q1.weight,
        module.k.weight,
        module.v.weight,
        module.project_out1.weight,
    )
    gradients = torch.autograd.grad(objective, requested)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().max().item() > 0.0 for gradient in gradients)


def test_constraint_projection_and_preload_validation(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    module = _module(candidate)
    with torch.no_grad():
        module.raw_simplex_gain.fill_(3.0)
    project_sbsc_v21_constraints_(candidate)
    assert module.raw_simplex_gain.item() == SBSC_V21_GAIN_MAX
    validate_sctransnet_sbsc_v21(candidate)

    bad = dict(candidate.state_dict())
    bad[SBSC_V21_GAIN_STATE_KEY] = torch.tensor(0.50001)
    with pytest.raises(ValueError, match="outside"):
        validate_sbsc_v21_state_dict(bad, "sbsc_v21")


def test_v2_and_v21_states_and_methods_are_mutually_rejected(paired_models):
    _baseline, candidate, _metadata = paired_models
    from experiments.sctransnet_sbsc_v2 import (
        build_paired_sctransnet_sbsc_v2,
        validate_sbsc_v2_state_dict,
    )

    _v2_baseline, v2_candidate, _v2_metadata = (
        build_paired_sctransnet_sbsc_v2("IRSTD-1K")
    )
    with pytest.raises(ValueError, match="method must be"):
        validate_sbsc_v21_state_dict(candidate.state_dict(), "sbsc_v2")
    with pytest.raises(ValueError, match="key set differs"):
        validate_sbsc_v21_state_dict(v2_candidate.state_dict(), "sbsc_v21")
    with pytest.raises(ValueError, match="key set differs"):
        validate_sbsc_v2_state_dict(candidate.state_dict(), "sbsc_v2")


def test_validator_rejects_wrong_site_hooks_shapes_and_public_alias_attacks(
    paired_models, monkeypatch
):
    _baseline, frozen_candidate, _metadata = paired_models

    candidate = copy.deepcopy(frozen_candidate)
    modules = tuple(layer.channel_attn for layer in candidate.mtc.encoder.layer)
    modules = (modules[1], modules[0], modules[2], modules[3])
    for layer, module in zip(candidate.mtc.encoder.layer, modules):
        layer.channel_attn = module
    with pytest.raises(TypeError, match="layer 1"):
        validate_sctransnet_sbsc_v21(candidate)

    candidate = copy.deepcopy(frozen_candidate)
    handle = _module(candidate).register_forward_hook(
        lambda _module, _args, output: output
    )
    try:
        with pytest.raises(RuntimeError, match="hooks are forbidden"):
            validate_sctransnet_sbsc_v21(candidate)
    finally:
        handle.remove()

    state = dict(frozen_candidate.state_dict())
    non_gain = next(key for key in state if key != SBSC_V21_GAIN_STATE_KEY)
    shaped = dict(state)
    shaped[non_gain] = shaped[non_gain].reshape(-1)
    with pytest.raises(ValueError, match="shape differs"):
        validate_sbsc_v21_state_dict(shaped, "sbsc_v21")
    typed = dict(state)
    typed[SBSC_V21_GAIN_STATE_KEY] = torch.zeros((), dtype=torch.float64)
    with pytest.raises(TypeError, match="dtype differs"):
        validate_sbsc_v21_state_dict(typed, "sbsc_v21")

    monkeypatch.setattr(core, "EXPECTED_SBSC_V21_STATE_KEY_COUNT", 999)
    monkeypatch.setattr(core, "SBSC_V21_GAIN_MAX", 9.0)
    forged = dict(state)
    forged[SBSC_V21_GAIN_STATE_KEY] = torch.tensor(0.75)
    with pytest.raises(ValueError, match="outside"):
        validate_sbsc_v21_state_dict(forged, "sbsc_v21")


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf")))
@pytest.mark.parametrize("method", ("sctransnet", "sbsc_v21"))
def test_preload_validator_rejects_nonfinite_shared_weights(
    paired_models, method, bad_value
):
    baseline, candidate, _metadata = paired_models
    source = baseline if method == "sctransnet" else candidate
    state = dict(source.state_dict())
    key = next(
        name
        for name, value in state.items()
        if name != SBSC_V21_GAIN_STATE_KEY
        and value.is_floating_point()
        and value.numel() > 0
    )
    attacked = state[key].clone()
    attacked.reshape(-1)[0] = bad_value
    state[key] = attacked
    with pytest.raises(ValueError, match="must be finite"):
        validate_sbsc_v21_state_dict(state, method)


def test_data_parallel_prefix_and_public_builder_arguments(paired_models):
    _baseline, candidate, _metadata = paired_models
    prefixed = {
        f"module.{key}": value for key, value in candidate.state_dict().items()
    }
    validation = validate_sbsc_v21_state_dict(prefixed, "sbsc_v21")
    assert validation["data_parallel_prefix"] is True
    assert validation["gain_state_keys"] == [
        f"module.{SBSC_V21_GAIN_STATE_KEY}"
    ]

    model, metadata = build_sctransnet_sbsc_v21_method(
        "sbsc_v21", "NUAA-SIRST", architecture_seed=42, training=False
    )
    assert model.training is False
    assert model.mode == "test"
    assert metadata["method"] == "sbsc_v21"
    assert metadata["state_key_count"] == 511
    assert metadata["test_split_accessed"] is False
    with pytest.raises(ValueError, match="architecture_seed=42"):
        build_paired_sctransnet_sbsc_v21("NUAA-SIRST", architecture_seed=41)
    with pytest.raises(ValueError, match="dataset must be"):
        build_paired_sctransnet_sbsc_v21("SIRST3")
    with pytest.raises(ValueError, match="method must be"):
        build_sctransnet_sbsc_v21_method("sbsc_v2", "NUAA-SIRST")


def test_public_gain_suffix_is_exactly_the_new_single_key():
    assert SBSC_V21_GAIN_SUFFIX == "raw_simplex_gain"
    assert SBSC_V21_GAIN_STATE_KEY == (
        "mtc.encoder.layer.1.channel_attn.raw_simplex_gain"
    )
