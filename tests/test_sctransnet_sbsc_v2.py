from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments.sctransnet_sbsc_v2 as sbsc_core
from experiments.sctransnet_sbsc_v2 import (
    EXPECTED_SBSC_V2_PARAMETER_COUNT,
    EXPECTED_SBSC_V2_STATE_KEY_COUNT,
    SBSCV2Intervention,
    SBSC_V2_GAIN_MAX,
    SBSC_V2_GAIN_SUFFIX,
    SupportBalancedConditionalCrossMomentV2,
    build_paired_sctransnet_sbsc_v2,
    build_sctransnet_sbsc_v2_method,
    estimate_sbsc_v2_support,
    forward_sctransnet_sbsc_v2_intervention,
    project_sbsc_v2_gains_,
    structurally_inactive_parameter_names,
    validate_sbsc_v2_state_dict,
    validate_sctransnet_sbsc_v2,
)
from experiments.four_dataset_models_seed42_v1 import (
    ORIGINAL_PARAMETER_COUNT,
    ORIGINAL_STATE_KEY_COUNT,
    state_dict_sha256,
)


@pytest.fixture(scope="module")
def paired_models():
    # The paired constructor is contractually transparent to the caller RNG.
    torch.manual_seed(7301)
    expected = torch.rand(7)
    torch.manual_seed(7301)
    baseline, candidate, metadata = build_paired_sctransnet_sbsc_v2(
        "IRSTD-1K"
    )
    observed = torch.rand(7)
    assert torch.equal(observed, expected)
    return baseline, candidate, metadata


def _gain_modules(model):
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, SupportBalancedConditionalCrossMomentV2)
    )


def _allclose_outputs(left, right, *, atol=0.0):
    if isinstance(left, torch.Tensor):
        return torch.allclose(left, right, rtol=0.0, atol=atol)
    return len(left) == len(right) and all(
        torch.allclose(a, b, rtol=0.0, atol=atol)
        for a, b in zip(left, right)
    )


def test_support_matches_frozen_normalized_key_jordan_formula():
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(2, 1, 7, 256, generator=generator).requires_grad_()
    key = F.normalize(raw, dim=-1)
    support = estimate_sbsc_v2_support(key)

    centered_key = key.float() - key.float().mean(dim=-1, keepdim=True)
    rarity = torch.sqrt(centered_key.square().mean(dim=-2))
    log_rarity = torch.log(rarity.clamp_min(1e-6))
    z = log_rarity - log_rarity.mean(dim=-1, keepdim=True)
    bounded = torch.tanh(z)
    h = bounded - bounded.mean(dim=-1, keepdim=True)
    positive = F.relu(h)
    negative = F.relu(-h)
    expected_positive = positive / positive.sum(dim=-1, keepdim=True)
    expected_negative = negative / negative.sum(dim=-1, keepdim=True)

    assert torch.equal(support.rarity, rarity)
    assert torch.equal(support.log_rarity_score, z)
    assert torch.equal(support.bounded_score, bounded)
    assert torch.equal(support.centered_score, h)
    assert torch.allclose(support.candidate, expected_positive)
    assert torch.allclose(support.counter_support, expected_negative)
    assert torch.allclose(
        support.candidate.sum(dim=-1),
        torch.ones_like(support.candidate.sum(dim=-1)),
    )
    assert torch.allclose(
        support.counter_support.sum(dim=-1),
        torch.ones_like(support.counter_support.sum(dim=-1)),
    )
    relative = bounded.abs().amax(dim=-1, keepdim=True)
    absolute = torch.tanh(
        (256.0 ** 0.5) * rarity.amax(dim=-1, keepdim=True)
    )
    assert torch.equal(support.relative_confidence.squeeze(-1), relative)
    assert torch.equal(support.absolute_confidence.squeeze(-1), absolute)
    assert torch.equal(support.confidence.squeeze(-1), relative * absolute)
    assert torch.all((support.confidence >= 0) & (support.confidence <= 1))
    assert torch.count_nonzero(
        support.candidate * support.counter_support
    ).item() == 0
    assert not any(
        tensor.requires_grad
        for tensor in (
            support.rarity,
            support.log_rarity_score,
            support.bounded_score,
            support.centered_score,
            support.candidate,
            support.counter_support,
            support.relative_confidence,
            support.absolute_confidence,
            support.confidence,
        )
    )


@pytest.mark.parametrize("constant", (0.0, 1.0 / 16.0))
def test_constant_support_is_exact_zero_confidence_and_uniform_jordan_fallback(
    constant,
):
    key = torch.full((2, 1, 11, 256), constant)
    support = estimate_sbsc_v2_support(key)
    uniform = torch.full_like(support.candidate, 1.0 / 256.0)
    assert torch.count_nonzero(support.log_rarity_score).item() == 0
    assert torch.count_nonzero(support.bounded_score).item() == 0
    assert torch.count_nonzero(support.centered_score).item() == 0
    assert torch.count_nonzero(support.confidence).item() == 0
    assert not torch.any(support.has_spatial_variation)
    assert torch.equal(support.candidate, uniform)
    assert torch.equal(support.counter_support, uniform)


def test_absolute_confidence_makes_near_constant_scale_continuous():
    confidences = []
    for delta in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        raw_key = torch.ones(1, 1, 16, 256)
        raw_key[..., 0] += delta
        raw_key[..., 1] -= delta
        normalized_key = F.normalize(raw_key, dim=-1)
        support = estimate_sbsc_v2_support(normalized_key)
        confidences.append(support.confidence.item())
        assert support.confidence.item() <= support.absolute_confidence.item()
    assert all(
        left >= right for left, right in zip(confidences, confidences[1:])
    )
    assert confidences[-1] == pytest.approx(0.0, abs=1e-8)
    assert confidences[0] > confidences[-1]


@pytest.mark.parametrize("target_size", (1, 2, 4, 8, 16))
def test_jordan_support_removes_target_area_mass_factor(target_size):
    positions = 256
    raw_key = torch.zeros(1, 1, 8, positions)
    raw_key[..., :target_size] = 1.0
    support = estimate_sbsc_v2_support(F.normalize(raw_key, dim=-1))
    # Fixed target/background contrast stays order-one even for m=1; it does
    # not inherit the area fraction m/N.
    assert support.confidence.item() > 0.9
    assert support.candidate[..., :target_size].sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(
        support.candidate[..., target_size:]
    ).item() == 0
    assert support.counter_support[..., target_size:].sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(
        support.counter_support[..., :target_size]
    ).item() == 0
    assert torch.allclose(
        support.candidate[..., :target_size],
        torch.full_like(
            support.candidate[..., :target_size], 1.0 / target_size
        ),
    )
    assert torch.allclose(
        support.counter_support[..., target_size:],
        torch.full_like(
            support.counter_support[..., target_size:],
            1.0 / (positions - target_size),
        ),
    )


def test_pair_contract_and_exact_shared_initialization(paired_models):
    baseline, candidate, metadata = paired_models
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    gain_keys = sorted(
        key for key in candidate_state if key.endswith(SBSC_V2_GAIN_SUFFIX)
    )

    assert len(baseline_state) == ORIGINAL_STATE_KEY_COUNT == 510
    assert len(candidate_state) == EXPECTED_SBSC_V2_STATE_KEY_COUNT == 514
    assert sum(p.numel() for p in baseline.parameters()) == ORIGINAL_PARAMETER_COUNT
    assert (
        sum(p.numel() for p in candidate.parameters())
        == EXPECTED_SBSC_V2_PARAMETER_COUNT
    )
    assert len(gain_keys) == 4
    assert set(candidate_state) - set(baseline_state) == set(gain_keys)
    assert all(
        torch.equal(baseline_state[key], candidate_state[key])
        for key in baseline_state
    )
    assert metadata["method"] == "sbsc_v2"
    assert metadata["dataset"] == "IRSTD-1K"
    assert metadata["architecture_seed"] == 42
    assert metadata["state_key_count"] == 514
    assert metadata["parameter_count"] == 11_325_943
    assert metadata["gain_state_keys"] == gain_keys
    assert metadata["gain_bounds"] == [0.0, 0.25]
    assert metadata["paired_initialization"] is True
    assert metadata["test_split_accessed"] is False
    assert metadata["shared_state_bitwise_equal"] is True
    assert metadata["shared_state_sha256"] == state_dict_sha256(baseline_state)
    assert len(structurally_inactive_parameter_names(baseline)) == 68
    assert len(structurally_inactive_parameter_names(candidate)) == 68
    validate_sctransnet_sbsc_v2(candidate, require_zero_gain=True)
    validate_sbsc_v2_state_dict(baseline_state, "sctransnet")
    validate_sbsc_v2_state_dict(candidate_state, "sbsc_v2")


@pytest.mark.parametrize("resolution", (32, 256))
def test_zero_init_is_exact_six_output_ssca_identity(
    paired_models, resolution
):
    baseline, candidate, _metadata = paired_models
    baseline.eval()
    candidate.eval()
    baseline.mode = "train"
    candidate.mode = "train"
    generator = torch.Generator().manual_seed(101)
    inputs = torch.randn(
        1, 1, resolution, resolution, generator=generator
    )
    with torch.no_grad():
        baseline_outputs = baseline(inputs)
        candidate_outputs = candidate(inputs)
    assert len(baseline_outputs) == len(candidate_outputs) == 6
    assert all(
        torch.equal(baseline_output, candidate_output)
        for baseline_output, candidate_output in zip(
            baseline_outputs, candidate_outputs
        )
    )


def test_zero_init_ste_gives_all_four_transport_gains_gradients(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    modules = _gain_modules(candidate)
    generator = torch.Generator().manual_seed(221)
    embeddings = (
        torch.randn(2, 32, 2, 2, generator=generator),
        torch.randn(2, 64, 2, 2, generator=generator),
        torch.randn(2, 128, 2, 2, generator=generator),
        torch.randn(2, 256, 2, 2, generator=generator),
    )
    emb_all = torch.cat(embeddings, dim=1)
    gradients = []
    for module in modules:
        outputs = module(*embeddings, emb_all)[:4]
        objective = sum(output.square().mean() for output in outputs)
        requested = (
            module.raw_transport_gain,
            module.q1.weight,
            module.k.weight,
            module.v.weight,
            module.project_out1.weight,
        )
        module_gradients = torch.autograd.grad(
            objective, requested, retain_graph=False
        )
        gradients.append(module_gradients[0])
        assert all(torch.isfinite(value).all() for value in module_gradients)
    assert len(gradients) == 4
    assert all(torch.isfinite(gradient) for gradient in gradients)
    assert all(gradient.abs().item() > 0.0 for gradient in gradients)


def test_post_step_projection_and_preload_gain_validation(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    modules = _gain_modules(candidate)
    raw_values = (-2.0, -0.01, 0.20, 4.0)
    with torch.no_grad():
        for module, value in zip(modules, raw_values):
            module.raw_transport_gain.fill_(value)
    project_sbsc_v2_gains_(candidate)
    projected = [module.raw_transport_gain.item() for module in modules]
    assert projected == [0.0, 0.0, pytest.approx(0.20), SBSC_V2_GAIN_MAX]
    validate_sctransnet_sbsc_v2(candidate)

    round_trip = copy.deepcopy(candidate)
    incompatible = round_trip.load_state_dict(candidate.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    bad_state = dict(candidate.state_dict())
    first_key = next(
        key for key in bad_state if key.endswith(SBSC_V2_GAIN_SUFFIX)
    )
    bad_state[first_key] = torch.tensor(0.25001)
    with pytest.raises(ValueError, match="outside"):
        validate_sbsc_v2_state_dict(bad_state, "sbsc_v2")


def test_preload_validator_rejects_key_shape_dtype_and_public_constant_attacks(
    paired_models, monkeypatch
):
    baseline, candidate, _metadata = paired_models
    baseline_state = dict(baseline.state_dict())
    candidate_state = dict(candidate.state_dict())
    validate_sbsc_v2_state_dict(
        {f"module.{key}": value for key, value in candidate_state.items()},
        "sbsc_v2",
    )

    non_gain = next(
        key for key in candidate_state if not key.endswith(SBSC_V2_GAIN_SUFFIX)
    )
    renamed = dict(candidate_state)
    renamed[non_gain + ".forged"] = renamed.pop(non_gain)
    with pytest.raises(ValueError, match="key set differs"):
        validate_sbsc_v2_state_dict(renamed, "sbsc_v2")

    shaped = dict(candidate_state)
    shaped[non_gain] = shaped[non_gain].reshape(-1)
    with pytest.raises(ValueError, match="shape differs"):
        validate_sbsc_v2_state_dict(shaped, "sbsc_v2")

    float_key = next(
        key for key, value in baseline_state.items() if value.dtype == torch.float32
    )
    typed = dict(baseline_state)
    typed[float_key] = typed[float_key].double()
    with pytest.raises(TypeError, match="dtype differs"):
        validate_sbsc_v2_state_dict(typed, "sctransnet")

    # Exported constants are documentation/API values, not validator trust
    # anchors.  Synchronizing them with an attack must not make it pass.
    monkeypatch.setattr(sbsc_core, "EXPECTED_SBSC_V2_STATE_KEY_COUNT", 515)
    monkeypatch.setattr(sbsc_core, "SBSC_V2_GAIN_MAX", 9.0)
    forged = dict(candidate_state)
    gain_key = next(key for key in forged if key.endswith(SBSC_V2_GAIN_SUFFIX))
    forged[gain_key] = torch.tensor(0.5)
    forged["forged.extra"] = torch.zeros(())
    with pytest.raises(ValueError):
        validate_sbsc_v2_state_dict(forged, "sbsc_v2")


def test_graph_validator_rejects_field_shadow_and_hook_attacks(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models

    candidate = copy.deepcopy(frozen_candidate)
    module = _gain_modules(candidate)[0]
    module.eps = 2e-6
    with pytest.raises(RuntimeError, match="eps contract"):
        validate_sctransnet_sbsc_v2(candidate)

    candidate = copy.deepcopy(frozen_candidate)
    module = _gain_modules(candidate)[0]
    module.forward = module.forward
    with pytest.raises(RuntimeError, match="instance method shadow"):
        validate_sctransnet_sbsc_v2(candidate)

    candidate = copy.deepcopy(frozen_candidate)
    module = _gain_modules(candidate)[0]
    handle = module.register_forward_hook(lambda _module, _args, output: output)
    try:
        with pytest.raises(RuntimeError, match="hooks are forbidden"):
            validate_sctransnet_sbsc_v2(candidate)
    finally:
        handle.remove()

    candidate = copy.deepcopy(frozen_candidate)
    module = _gain_modules(candidate)[0]
    module.psi = nn.InstanceNorm2d(1, affine=True)
    with pytest.raises(RuntimeError, match="InstanceNorm2d contract"):
        validate_sctransnet_sbsc_v2(candidate)

    candidate = copy.deepcopy(frozen_candidate)
    module = _gain_modules(candidate)[0]
    module.softmax = nn.Softmax(dim=2)
    with pytest.raises(TypeError, match=r"Softmax\(dim=3\)"):
        validate_sctransnet_sbsc_v2(candidate)


def test_amp_keeps_ssca_anchor_exact_and_conditional_transport_fp32(
    paired_models,
):
    baseline, candidate, _metadata = paired_models
    source = baseline.mtc.encoder.layer[0].channel_attn
    replacement = candidate.mtc.encoder.layer[0].channel_attn
    source.eval()
    replacement.eval()
    generator = torch.Generator().manual_seed(919)
    embeddings = (
        torch.randn(2, 32, 2, 2, generator=generator),
        torch.randn(2, 64, 2, 2, generator=generator),
        torch.randn(2, 128, 2, 2, generator=generator),
        torch.randn(2, 256, 2, 2, generator=generator),
    )
    emb_all = torch.cat(embeddings, dim=1)
    with torch.no_grad(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        source_outputs = source(*embeddings, emb_all)
        replacement_outputs = replacement(*embeddings, emb_all)
    assert all(
        torch.equal(left, right)
        for left, right in zip(source_outputs[:4], replacement_outputs[:4])
    )

    query = F.normalize(torch.randn(2, 1, 32, 4), dim=-1).bfloat16()
    key = F.normalize(torch.randn(2, 1, 480, 4), dim=-1).bfloat16()
    support = estimate_sbsc_v2_support(key)
    with torch.no_grad(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        base_relation = (query @ key.transpose(-2, -1)) / (480.0 ** 0.5)
        base_attention = replacement.softmax(replacement.psi(base_relation))
        attention, diagnostics = replacement._relation_transport(
            query,
            key,
            base_attention,
            support.candidate,
            support.counter_support,
            support.confidence,
            replacement.effective_transport_gain(0.25),
        )
    assert diagnostics["positive_attention"].dtype == torch.float32
    assert diagnostics["negative_attention"].dtype == torch.float32
    assert diagnostics["transport_direction"].dtype == torch.float32
    assert diagnostics["transport_correction_fp32"].dtype == torch.float32
    assert diagnostics["transport_correction"].dtype == base_attention.dtype
    assert attention.dtype == base_attention.dtype


def test_real_relation_operator_has_negative_signed_transport_witness(
    paired_models,
):
    _baseline, candidate, _metadata = paired_models
    module = _gain_modules(candidate)[0]
    generator = torch.Generator().manual_seed(0)
    query = F.normalize(
        torch.randn(1, 1, 32, 4, generator=generator), dim=-1
    )
    key = F.normalize(
        torch.randn(1, 1, 480, 4, generator=generator), dim=-1
    )
    candidate_support = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    counter_support = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    confidence = torch.ones(1, 1, 1, 1)
    base_relation = (query @ key.transpose(-2, -1)) / (480.0 ** 0.5)
    base_attention = module.softmax(module.psi(base_relation))
    attention, diagnostics = module._relation_transport(
        query,
        key,
        base_attention,
        candidate_support,
        counter_support,
        confidence,
        module.effective_transport_gain(0.25),
    )
    assert attention.amin().item() < 0.0
    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        atol=2e-6,
        rtol=0.0,
    )
    assert (
        diagnostics["transport_direction"]
        .abs()
        .sum(dim=-1)
        .amax()
        .item()
        <= 1.0 + 2e-6
    )


def test_full_graph_context_interventions_are_bounded_and_transactional(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    candidate.eval()
    candidate.mode = "test"
    with torch.no_grad():
        for module in _gain_modules(candidate):
            module.raw_transport_gain.fill_(SBSC_V2_GAIN_MAX)
    before_hash = state_dict_sha256(candidate.state_dict())
    generator = torch.Generator().manual_seed(331)
    inputs = torch.randn(2, 1, 32, 32, generator=generator)
    permutations = {
        index: torch.tensor([3, 2, 1, 0], dtype=torch.long)
        for index in range(4)
    }
    specifications = (
        SBSCV2Intervention("normal"),
        SBSCV2Intervention("zero"),
        SBSCV2Intervention("reverse"),
        SBSCV2Intervention("shuffle", spatial_permutations=permutations),
        SBSCV2Intervention(
            "cross_image", batch_permutation=torch.tensor([1, 0])
        ),
    )
    results = []
    with torch.no_grad():
        for specification in specifications:
            results.append(
                forward_sctransnet_sbsc_v2_intervention(
                    candidate, inputs, specification
                )
            )
    assert state_dict_sha256(candidate.state_dict()) == before_hash

    for (outputs, diagnostics), specification in zip(results, specifications):
        assert outputs.shape == inputs.shape
        assert torch.isfinite(outputs).all()
        assert len(diagnostics) == 4
        assert [item["layer_index"] for item in diagnostics] == list(range(4))
        assert all(item["mode"] == specification.mode for item in diagnostics)
        for item in diagnostics:
            assert torch.allclose(
                item["candidate"].sum(dim=-1),
                torch.ones_like(item["candidate"].sum(dim=-1)),
            )
            assert torch.allclose(
                item["counter_support"].sum(dim=-1),
                torch.ones_like(item["counter_support"].sum(dim=-1)),
            )
            assert torch.all((item["confidence"] >= 0) & (item["confidence"] <= 1))
            for branch in item["branches"]:
                direction = branch["transport_direction"]
                correction = branch["transport_correction"]
                attention = branch["attention"]
                assert direction.abs().sum(dim=-1).amax().item() <= 1.0 + 1e-6
                assert direction.sum(dim=-1).abs().amax().item() <= 2e-6
                assert correction.abs().sum(dim=-1).amax().item() <= 0.25 + 1e-6
                assert attention.amin().item() >= -0.25 - 1e-6
                assert attention.amax().item() <= 1.25 + 1e-6
                assert attention.abs().sum(dim=-1).amax().item() <= 1.25 + 1e-6
                assert torch.allclose(
                    attention.sum(dim=-1),
                    torch.ones_like(attention.sum(dim=-1)),
                    rtol=0.0,
                    atol=2e-6,
                )

    normal_output = results[0][0]
    zero_output = results[1][0]
    reverse_output = results[2][0]
    assert not torch.equal(normal_output, zero_output)
    assert not torch.equal(normal_output, reverse_output)
    assert all(
        item["effective_gain"].item() == 0.0
        for item in results[1][1]
    )

    # A legal signed row can contain a negative coefficient, unlike any
    # single-softmax Query gate, while preserving row sum and the L1 bound.
    witness = torch.full((480,), 1.0 / 480.0)
    witness[0] += 0.125
    witness[1] -= 0.125
    assert witness[1].item() < 0.0
    assert witness.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert witness.abs().sum().item() <= 1.25 + 1e-6


def test_intervention_failures_reset_context(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    candidate.eval()
    candidate.mode = "test"
    inputs = torch.randn(2, 1, 32, 32)
    invalid = {
        index: torch.tensor([0, 0, 1, 2], dtype=torch.long)
        for index in range(4)
    }
    with pytest.raises(ValueError, match="must be a permutation"):
        with torch.no_grad():
            forward_sctransnet_sbsc_v2_intervention(
                candidate,
                inputs,
                SBSCV2Intervention("shuffle", spatial_permutations=invalid),
            )
    identity = {
        index: torch.arange(4, dtype=torch.long) for index in range(4)
    }
    with pytest.raises(ValueError, match="no fixed points"):
        with torch.no_grad():
            forward_sctransnet_sbsc_v2_intervention(
                candidate,
                inputs,
                SBSCV2Intervention("shuffle", spatial_permutations=identity),
            )
    with torch.no_grad():
        ordinary = candidate(inputs)
        normal, diagnostics = forward_sctransnet_sbsc_v2_intervention(
            candidate, inputs, SBSCV2Intervention("normal")
        )
    assert torch.equal(ordinary, normal)
    assert len(diagnostics) == 4

    candidate.train()
    with pytest.raises(RuntimeError, match=r"require model\.eval"):
        forward_sctransnet_sbsc_v2_intervention(
            candidate, inputs, SBSCV2Intervention("normal")
        )


def test_public_method_builder_and_strict_arguments():
    model, metadata = build_sctransnet_sbsc_v2_method(
        "sbsc_v2", "NUAA-SIRST", architecture_seed=42, training=False
    )
    assert model.training is False
    assert model.mode == "test"
    assert metadata["method"] == "sbsc_v2"
    assert metadata["dataset"] == "NUAA-SIRST"
    assert metadata["state_key_count"] == 514
    assert metadata["test_split_accessed"] is False
    with pytest.raises(ValueError, match="architecture_seed=42"):
        build_paired_sctransnet_sbsc_v2("NUAA-SIRST", architecture_seed=41)
    with pytest.raises(ValueError, match="dataset must be"):
        build_paired_sctransnet_sbsc_v2("SIRST3")
    with pytest.raises(ValueError, match="method must be"):
        build_sctransnet_sbsc_v2_method("baseline", "NUAA-SIRST")
