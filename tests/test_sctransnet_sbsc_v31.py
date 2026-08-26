from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

import experiments.sctransnet_sbsc_v31 as core
from experiments.four_dataset_models_seed42_v1 import (
    ORIGINAL_PARAMETER_COUNT,
    ORIGINAL_STATE_KEY_COUNT,
    state_dict_sha256,
)
from experiments.sctransnet_sbsc_v2 import (
    SBSC_V2_GAIN_SUFFIX,
    validate_sbsc_v2_state_dict,
)
from experiments.sctransnet_sbsc_v21 import (
    SBSC_V21_GAIN_STATE_KEY,
    validate_sbsc_v21_state_dict,
)
from model.SCTransNet import Attention_org


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    """Keep the row-wise solver tests fast and restore the caller setting."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def paired_models():
    torch.manual_seed(7301)
    expected = torch.rand(7)
    torch.manual_seed(7301)
    baseline, candidate, metadata = core.build_paired_sctransnet_sbsc_v31(
        "IRSTD-1K"
    )
    observed = torch.rand(7)
    assert torch.equal(observed, expected)
    return baseline, candidate, metadata


def _module(model):
    module = model.mtc.encoder.layer[1].channel_attn
    assert type(module) is core.C3DualRiskProjectionV31
    return module


def _support_fixture():
    generator = torch.Generator().manual_seed(91)
    key = F.normalize(
        torch.randn(1, 1, 9, 7, generator=generator), dim=-1
    )
    validations = (
        torch.tensor([[[[-1.0, -0.5, 0.0, 0.5, 1.0, -0.2, 0.3]]]]),
        torch.tensor([[[[-0.8, -0.2, 0.1, 0.6, 0.9, 0.1, -0.4]]]]),
        torch.tensor([[[[-0.6, 0.4, 0.3, 0.8, -0.7, 0.2, 0.5]]]]),
        torch.tensor([[[[0.2, -0.4, 0.5, 0.7, -0.1, -0.8, 0.6]]]]),
    )
    return key, validations


def _active_set_gold_inputs():
    q0 = torch.tensor(
        [
            [
                0.44291961193084717,
                0.16163913905620575,
                0.026739977300167084,
                0.306021124124527,
                0.06268011778593063,
            ],
            [
                0.048431966453790665,
                0.07313162833452225,
                0.104494608938694,
                0.34898319840431213,
                0.42495864629745483,
            ],
            [
                0.07366841286420822,
                0.07174038887023926,
                0.1767675280570984,
                0.14716216921806335,
                0.5306615233421326,
            ],
            [
                0.0853661522269249,
                0.048407308757305145,
                0.42904889583587646,
                0.12223677337169647,
                0.31494084000587463,
            ],
        ]
    )
    benefit = torch.tensor(
        [
            [-1.360122561454773, 0.15206104516983032, 1.4271775484085083,
             0.7555384039878845, -0.6855717897415161],
            [2.240180253982544, -1.1250766515731812, -1.3166027069091797,
             -1.079889178276062, -0.3150887191295624],
            [-1.9876643419265747, 0.6612577438354492, -0.23793374001979828,
             0.14588859677314758, -1.1372586488723755],
            [0.27952393889427185, -0.9955828189849854, 0.1661229282617569,
             -0.8720988631248474, -0.529872715473175],
        ],
        requires_grad=True,
    )
    hard_risk = torch.tensor(
        [
            [0.63066166639328, 0.10262078046798706, -0.5822293162345886,
             0.14123274385929108, 0.7471047639846802],
            [0.45063406229019165, -0.3524375557899475, -1.0252059698104858,
             -1.2093125581741333, 0.334425151348114],
            [1.2675323486328125, 0.2432202845811844, 0.511140763759613,
             0.5515236854553223, 1.3228834867477417],
            [-1.6821681261062622, -2.900406837463379, -0.7191106081008911,
             0.398084431886673, -1.1138356924057007],
        ],
        requires_grad=True,
    )
    background_risk = torch.tensor(
        [
            [2.058751106262207, -1.2598752975463867, -0.37647438049316406,
             -0.8684089779853821, 1.0109305381774902],
            [-0.13003399968147278, 1.7433816194534302, 0.009779759682714939,
             0.9621071219444275, 1.364774227142334],
            [-1.748544454574585, -0.39596837759017944, 0.9449084997177124,
             -0.2654274106025696, 0.33880069851875305],
            [0.23533965647220612, 1.0629587173461914, 0.4628576934337616,
             -1.635438323020935, -0.8014009594917297],
        ],
        requires_grad=True,
    )
    reliability = torch.tensor(
        [[0.3136565387248993], [0.6140349507331848],
         [0.577394425868988], [0.8558779358863831]]
    )
    active = torch.ones(4, 1, dtype=torch.bool)
    return q0, benefit, hard_risk, background_risk, reliability, active


def test_pair_structure_state_and_authority_contract(paired_models):
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
    assert type(candidate_modules[1]) is core.C3DualRiskProjectionV31
    assert all(
        type(candidate_modules[index]) is Attention_org for index in (0, 2, 3)
    )
    assert len(baseline_state) == ORIGINAL_STATE_KEY_COUNT == 510
    assert len(candidate_state) == core.EXPECTED_SBSC_V31_STATE_KEY_COUNT == 511
    assert sum(p.numel() for p in baseline.parameters()) == ORIGINAL_PARAMETER_COUNT
    assert (
        sum(p.numel() for p in candidate.parameters())
        == core.EXPECTED_SBSC_V31_PARAMETER_COUNT
        == 11_325_943
    )
    assert set(candidate_state) - set(baseline_state) == {
        core.SBSC_V31_GAIN_STATE_KEY
    }
    assert not set(baseline_state) - set(candidate_state)
    assert all(
        torch.equal(baseline_state[key], candidate_state[key])
        for key in baseline_state
    )
    gain = candidate_state[core.SBSC_V31_GAIN_STATE_KEY]
    assert tuple(gain.shape) == (4,)
    assert gain.dtype is torch.float32
    assert torch.count_nonzero(gain).item() == 0

    source_state = baseline_modules[1].state_dict()
    replacement_state = candidate_modules[1].state_dict()
    assert set(replacement_state) - set(source_state) == {
        core.SBSC_V31_GAIN_SUFFIX
    }
    assert all(
        torch.equal(source_state[key], replacement_state[key])
        for key in source_state
    )
    assert candidate_modules[1].training is baseline_modules[1].training

    assert metadata["schema"] == core.SBSC_V31_SCHEMA
    assert metadata["method"] == "sbsc_v31"
    assert metadata["replaced_block_indices"] == [1]
    assert metadata["gain_state_keys"] == [core.SBSC_V31_GAIN_STATE_KEY]
    assert metadata["gain_bounds"] == [0.0, 0.25]
    assert metadata["independent_tri_support_cross_relation"] is True
    assert "independent_tri_support_covariance" not in metadata
    assert metadata["shared_state_bitwise_equal"] is True
    assert metadata["shared_state_sha256"] == state_dict_sha256(baseline_state)
    assert metadata["parent_checkpoint"] is None
    assert metadata["warm_start_used"] is False
    assert metadata["predecessor_checkpoint_used"] is False
    assert len(core.structurally_inactive_parameter_names(candidate)) == 68
    validation = core.validate_sctransnet_sbsc_v31(
        candidate, require_zero_gain=True
    )
    assert validation["per_row_certificate"] is True
    core.validate_sbsc_v31_state_dict(baseline_state, "sctransnet")
    core.validate_sbsc_v31_state_dict(candidate_state, "sbsc_v31")


def test_v2_v21_old_v3_and_v31_states_are_mutually_rejected(paired_models):
    baseline, candidate, _metadata = paired_models
    baseline_state = dict(baseline.state_dict())
    candidate_state = candidate.state_dict()

    v21_state = dict(baseline_state)
    v21_state[SBSC_V21_GAIN_STATE_KEY] = torch.tensor(0.0)
    validate_sbsc_v21_state_dict(v21_state, "sbsc_v21")

    v2_state = dict(baseline_state)
    for index in range(4):
        key = f"mtc.encoder.layer.{index}.channel_attn.{SBSC_V2_GAIN_SUFFIX}"
        v2_state[key] = torch.tensor(0.0)
    validate_sbsc_v2_state_dict(v2_state, "sbsc_v2")

    old_v3_state = dict(baseline_state)
    old_v3_state[
        "mtc.encoder.layer.1.channel_attn.raw_level_gain"
    ] = torch.zeros(4)

    for alien_state in (v2_state, v21_state, old_v3_state):
        with pytest.raises(ValueError, match="key set differs"):
            core.validate_sbsc_v31_state_dict(alien_state, "sbsc_v31")
    with pytest.raises(ValueError, match="key set differs"):
        validate_sbsc_v21_state_dict(candidate_state, "sbsc_v21")
    with pytest.raises(ValueError, match="key set differs"):
        validate_sbsc_v2_state_dict(candidate_state, "sbsc_v2")
    with pytest.raises(ValueError, match="method must be"):
        core.validate_sbsc_v31_state_dict(candidate_state, "sbsc_v21")


@pytest.mark.parametrize(
    ("bad_gain", "error_type"),
    (
        (torch.zeros(1), ValueError),
        (torch.zeros(4, 1), ValueError),
        (torch.zeros(4, dtype=torch.float64), TypeError),
        (torch.tensor([0.0, 0.1, float("nan"), 0.2]), ValueError),
        (torch.tensor([0.0, 0.1, 0.2, 0.25001]), ValueError),
    ),
)
def test_v31_state_rejects_wrong_gain_shape_dtype_finite_and_bounds(
    paired_models, bad_gain, error_type
):
    _baseline, candidate, _metadata = paired_models
    state = dict(candidate.state_dict())
    state[core.SBSC_V31_GAIN_STATE_KEY] = bad_gain
    with pytest.raises(error_type):
        core.validate_sbsc_v31_state_dict(state, "sbsc_v31")


def test_validator_rejects_forward_shadow_hooks_and_solver_constant_override(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models

    shadowed = copy.deepcopy(frozen_candidate)
    shadowed_module = _module(shadowed)
    shadowed_module.forward = shadowed_module.forward
    with pytest.raises(RuntimeError, match="instance method shadow"):
        core.validate_sctransnet_sbsc_v31(shadowed)

    hooked = copy.deepcopy(frozen_candidate)
    handle = _module(hooked).register_forward_hook(
        lambda _module, _inputs, output: output
    )
    try:
        with pytest.raises(RuntimeError, match="hooks are forbidden"):
            core.validate_sctransnet_sbsc_v31(hooked)
    finally:
        handle.remove()

    overridden = copy.deepcopy(frozen_candidate)
    _module(overridden).risk_tolerance = core.SBSC_V31_EPS
    with pytest.raises(RuntimeError, match="constants cannot be instance overrides"):
        core.validate_sctransnet_sbsc_v31(overridden)

    untouched_softmax = copy.deepcopy(frozen_candidate)
    untouched_softmax.mtc.encoder.layer[0].channel_attn.softmax.dim = 2
    with pytest.raises(RuntimeError, match="outside the L1 SSCA replacement"):
        core.validate_sctransnet_sbsc_v31(untouched_softmax)

    untouched_norm = copy.deepcopy(frozen_candidate)
    untouched_norm.mtc.encoder.layer[0].channel_attn.psi.eps = 0.123
    with pytest.raises(RuntimeError, match="outside the L1 SSCA replacement"):
        core.validate_sctransnet_sbsc_v31(untouched_norm)


def test_gain_zero_is_exact_six_output_identity_at_32(paired_models):
    baseline, candidate, _metadata = paired_models
    baseline.eval()
    candidate.eval()
    baseline.mode = "train"
    candidate.mode = "train"
    inputs = torch.randn(
        1, 1, 32, 32, generator=torch.Generator().manual_seed(133)
    )
    with torch.no_grad():
        baseline_outputs = baseline(inputs)
        candidate_outputs = candidate(inputs)
    assert len(baseline_outputs) == len(candidate_outputs) == 6
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP unavailable")
def test_gain_zero_is_exact_six_output_identity_under_bf16_amp(paired_models):
    frozen_baseline, frozen_candidate, _metadata = paired_models
    baseline = copy.deepcopy(frozen_baseline).cuda().eval()
    candidate = copy.deepcopy(frozen_candidate).cuda().eval()
    baseline.mode = "train"
    candidate.mode = "train"
    inputs = torch.randn(
        1,
        1,
        32,
        32,
        generator=torch.Generator().manual_seed(134),
    ).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline_outputs = baseline(inputs)
        candidate_outputs = candidate(inputs)
    assert len(baseline_outputs) == len(candidate_outputs) == 6
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("amp_bfloat16", (False, True))
def test_gain_zero_is_exact_six_output_identity_at_256_cuda(
    paired_models, amp_bfloat16
):
    frozen_baseline, frozen_candidate, _metadata = paired_models
    baseline = copy.deepcopy(frozen_baseline).cuda().eval()
    candidate = copy.deepcopy(frozen_candidate).cuda().eval()
    baseline.mode = "train"
    candidate.mode = "train"
    inputs = torch.randn(
        1,
        1,
        256,
        256,
        generator=torch.Generator().manual_seed(135),
    ).cuda()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_bfloat16)
    with torch.no_grad(), autocast:
        baseline_outputs = baseline(inputs)
        candidate_outputs = candidate(inputs)
    assert len(baseline_outputs) == len(candidate_outputs) == 6
    assert all(
        torch.equal(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    )


def test_all_diagnostic_interventions_masks_and_context_cleanup(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate).eval()
    candidate.mode = "train"
    module = _module(candidate)
    with torch.no_grad():
        module.raw_dual_risk_level_gain.fill_(0.1)
    generator = torch.Generator().manual_seed(812)
    embeddings = tuple(
        torch.randn(2, channels, 2, 2, generator=generator)
        for channels in (32, 64, 128, 256)
    )
    emb_all = torch.cat(embeddings, dim=1)
    state_before = state_dict_sha256(candidate.state_dict())
    override = torch.tensor([0.02, 0.04, 0.06, 0.08])
    disabled_raw_maxima = {"hard": [0.0, 0.0, 0.0], "background": [0.0, 0.0, 0.0]}

    with torch.no_grad():
        for intervention in core._FROZEN_DIAGNOSTIC_INTERVENTIONS:
            with core.capture_c3_v31_diagnostics(
                candidate,
                gain_override=override,
                intervention=intervention,
            ) as collector:
                outputs = module(*embeddings, emb_all)
            assert len(outputs) == 5
            assert len(collector.records) == 1
            record = collector.records[0]
            assert record["intervention"] == intervention
            assert torch.equal(record["gain_override"], override)
            assert len(record["levels"]) == 4
            for level_index, level in enumerate(record["levels"]):
                assert level["level_index"] == level_index
                assert level["intervention"] == intervention
                assert torch.equal(level["gain_override"], override)
                assert torch.equal(level["effective_gain"], override)
                assert "effective_reliability" in level
                assert isinstance(level["projection"], core.C3V31Projection)
                projection = level["projection"]
                if intervention == "disable_hard_constraint":
                    assert not bool(projection.hard_defined.any())
                    assert torch.count_nonzero(projection.hard_risk_delta).item() == 0
                    assert torch.count_nonzero(level["hard_risk_delta_qout"]).item() == 0
                    assert torch.count_nonzero(level["hard_risk_delta_emitted"]).item() == 0
                    disabled_raw_maxima["hard"][0] = max(
                        disabled_raw_maxima["hard"][0],
                        float(projection.hard_risk_delta_raw.abs().max().item()),
                    )
                    disabled_raw_maxima["hard"][1] = max(
                        disabled_raw_maxima["hard"][1],
                        float(level["hard_risk_delta_qout_raw"].abs().max().item()),
                    )
                    disabled_raw_maxima["hard"][2] = max(
                        disabled_raw_maxima["hard"][2],
                        float(level["hard_risk_delta_emitted_raw"].abs().max().item()),
                    )
                if intervention == "disable_background_constraint":
                    assert not bool(projection.background_defined.any())
                    assert (
                        torch.count_nonzero(projection.background_risk_delta).item()
                        == 0
                    )
                    assert (
                        torch.count_nonzero(level["background_risk_delta_qout"]).item()
                        == 0
                    )
                    assert (
                        torch.count_nonzero(level["background_risk_delta_emitted"]).item()
                        == 0
                    )
                    disabled_raw_maxima["background"][0] = max(
                        disabled_raw_maxima["background"][0],
                        float(
                            projection.background_risk_delta_raw.abs().max().item()
                        ),
                    )
                    disabled_raw_maxima["background"][1] = max(
                        disabled_raw_maxima["background"][1],
                        float(
                            level["background_risk_delta_qout_raw"].abs().max().item()
                        ),
                    )
                    disabled_raw_maxima["background"][2] = max(
                        disabled_raw_maxima["background"][2],
                        float(
                            level["background_risk_delta_emitted_raw"].abs().max().item()
                        ),
                    )

        assert all(value > 0.0 for value in disabled_raw_maxima["hard"])
        assert all(value > 0.0 for value in disabled_raw_maxima["background"])

        stored = module.effective_level_gain().detach()
        for mask_code in range(16):
            mask = torch.tensor(
                [(mask_code >> index) & 1 for index in range(4)],
                dtype=torch.float32,
            )
            masked_gain = stored * mask
            with core.capture_c3_v31_diagnostics(
                candidate, gain_override=masked_gain
            ) as collector:
                module(*embeddings, emb_all)
            assert torch.equal(
                collector.records[0]["gain_override"], masked_gain
            )

        with pytest.raises(RuntimeError, match="forced cleanup"):
            with core.capture_c3_v31_diagnostics(candidate):
                raise RuntimeError("forced cleanup")
        with core.capture_c3_v31_diagnostics(candidate) as recovered:
            module(*embeddings, emb_all)
        assert len(recovered.records) == 1

    assert state_dict_sha256(candidate.state_dict()) == state_before


def test_loo_uses_three_peer_median_mad_and_is_self_independent():
    key, validations = _support_fixture()
    before = core.estimate_c3_v31_support_from_validations(key, validations)
    level = before.levels[0]
    peers = torch.cat(validations[1:], dim=-2)
    expected_median = peers.median(dim=-2, keepdim=True).values
    expected_mad = (peers - expected_median).abs().median(
        dim=-2, keepdim=True
    ).values

    assert level.peer_indices == (1, 2, 3)
    assert torch.equal(level.peer_consensus, expected_median)
    assert torch.equal(level.peer_dispersion, expected_mad)
    assert not torch.equal(level.peer_consensus, peers.mean(dim=-2, keepdim=True))

    changed_validations = list(validations)
    changed_validations[0] = torch.full_like(validations[0], 42.0)
    after = core.estimate_c3_v31_support_from_validations(
        key, tuple(changed_validations)
    )
    assert not torch.equal(
        before.levels[0].self_validation, after.levels[0].self_validation
    )
    peer_only_fields = (
        "peer_consensus",
        "peer_dispersion",
        "agreement",
        "consistent_raw",
        "contradictory_raw",
        "common_raw",
        "consistent_mass",
        "contradictory_mass",
        "common_mass",
        "consistent_support",
        "contradictory_support",
        "common_support",
        "consistent_valid",
        "contradictory_valid",
        "common_valid",
        "consistent_positive_strength",
        "consistent_common_separation",
        "reliability",
    )
    for field in peer_only_fields:
        assert torch.equal(
            getattr(before.levels[0], field), getattr(after.levels[0], field)
        )


def test_consistent_contradictory_common_supports_obey_simplex_contract():
    key, validations = _support_fixture()
    support = core.estimate_c3_v31_support_from_validations(key, validations)
    assert bool(support.key_has_variation.all())

    for level in support.levels:
        for stem in ("consistent", "contradictory", "common"):
            raw = getattr(level, f"{stem}_raw")
            mass = getattr(level, f"{stem}_mass")
            valid = getattr(level, f"{stem}_valid")
            simplex = getattr(level, f"{stem}_support")
            assert bool(valid.all())
            assert bool((mass > core.SBSC_V31_EPS).all())
            assert torch.allclose(
                simplex.sum(dim=-1, keepdim=True),
                torch.ones_like(mass),
                rtol=0.0,
                atol=core.SBSC_V31_EPS,
            )
            assert torch.allclose(
                simplex,
                raw / mass,
                rtol=0.0,
                atol=core.SBSC_V31_EPS,
            )
        assert not any(
            tensor.requires_grad
            for tensor in (
                level.peer_consensus,
                level.consistent_support,
                level.contradictory_support,
                level.common_support,
                level.reliability,
            )
        )


def test_constant_key_is_uniform_zero_reliability_and_support_identity():
    _key, validations = _support_fixture()
    key = F.normalize(torch.ones(1, 1, 9, 7), dim=-1)
    support = core.estimate_c3_v31_support_from_validations(key, validations)
    uniform = torch.full((1, 1, 1, 7), 1.0 / 7.0)

    assert not bool(support.key_has_variation.any())
    assert torch.count_nonzero(support.log_rarity_score).item() == 0
    assert torch.count_nonzero(support.bounded_score).item() == 0
    assert torch.count_nonzero(support.centered_score).item() == 0
    for level in support.levels:
        assert torch.count_nonzero(level.reliability).item() == 0
        for stem in ("consistent", "contradictory", "common"):
            assert not bool(getattr(level, f"{stem}_valid").any())
            assert torch.equal(getattr(level, f"{stem}_support"), uniform)

        projection = core.solve_dual_risk_projection_v31(
            uniform,
            torch.linspace(-1.0, 1.0, 7).reshape(1, 1, 1, 7),
            torch.zeros_like(uniform),
            torch.zeros_like(uniform),
            level.reliability,
            hard_active=level.contradictory_valid,
            background_active=level.common_valid,
        )
        assert bool(projection.support_identity.all())
        assert torch.equal(projection.qhat, uniform)


def test_near_constant_key_confidence_underflow_is_exact_identity():
    key = torch.ones(1, 1, 4, 8)
    polarity = torch.tensor([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    key[:, :, 0, :] += 1e-5 * polarity
    key = F.normalize(key, dim=-1)
    validations = []
    for level_index in range(4):
        validation = torch.zeros(1, 1, 1, 8)
        if level_index in (1, 2, 3):
            validation[..., 4:] = 1.0
        validations.append(validation)

    support = core.estimate_c3_v31_support_from_validations(
        key, tuple(validations)
    )
    level = support.levels[0]
    assert bool(support.key_has_variation.all())
    assert bool((support.key_confidence <= core.SBSC_V31_EPS).all())
    assert not bool(support.key_confidence_valid.any())
    assert bool(level.consistent_valid.all())
    assert bool(level.common_valid.all())
    assert torch.count_nonzero(level.reliability).item() == 0

    uniform = torch.full((1, 1, 1, 8), 1.0 / 8.0)
    projection = core.solve_dual_risk_projection_v31(
        uniform,
        torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 1, 8),
        torch.zeros_like(uniform),
        torch.zeros_like(uniform),
        level.reliability,
        hard_active=level.contradictory_valid,
        background_active=level.common_valid,
    )
    assert bool(projection.support_identity.all())
    assert torch.equal(projection.qhat, uniform)


def test_solver_gold_rows_cover_empty_h_b_and_h_plus_b_certificates():
    q0, benefit, hard_risk, background_risk, reliability, active = (
        _active_set_gold_inputs()
    )
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard_risk,
        background_risk,
        reliability,
        hard_active=active,
        background_active=active,
    )

    assert core.C3V31Projection.__dataclass_params__.frozen is True
    assert torch.equal(
        projection.active_set_code[:, 0], torch.tensor([0, 1, 2, 3])
    )
    assert torch.equal(
        projection.candidate_valid,
        torch.eye(4, dtype=torch.bool),
    )
    assert bool(projection.accepted.all())
    assert torch.equal(projection.reason_code, torch.zeros(4, 1, dtype=torch.int64))
    assert torch.equal(projection.lambda_h[0], torch.zeros(1))
    assert torch.equal(projection.lambda_b[0], torch.zeros(1))
    assert projection.lambda_h[1].item() > 0.0
    assert projection.lambda_b[1].item() == 0.0
    assert projection.lambda_h[2].item() == 0.0
    assert projection.lambda_b[2].item() > 0.0
    assert projection.lambda_h[3].item() > 0.0
    assert projection.lambda_b[3].item() > 0.0
    # Row 1's H candidate omits B from its active set, but the omitted
    # constraint is still present in the complete risk/KKT certificate.
    assert projection.candidate_lambda_b[1, 1].item() == 0.0
    assert projection.candidate_risk_max[1, 1].item() <= 1e-6
    assert projection.candidate_kkt_max[1, 1].item() <= 1e-6
    # Conversely, row 3's B-only candidate violates the omitted H condition
    # and must not be accepted merely because its own singleton root exists.
    assert projection.candidate_lambda_h[3, 2].item() == 0.0
    assert projection.candidate_lambda_b[3, 2].item() > 0.0
    assert not projection.candidate_valid[3, 2].item()
    assert projection.candidate_risk_max[3, 2].item() > 1e-6
    assert projection.candidate_kkt_max[3, 2].item() > 1e-6
    assert bool((projection.lambda_h <= 32.0).all())
    assert bool((projection.lambda_b <= 32.0).all())
    assert torch.allclose(
        projection.qhat.sum(dim=-1), torch.ones(4), rtol=0.0, atol=1e-6
    )
    assert projection.qhat.amin().item() >= -1e-7
    assert projection.hard_risk_delta.amax().item() <= 1e-6
    assert projection.background_risk_delta.amax().item() <= 1e-6
    assert projection.kkt_max.amax().item() <= 1e-6
    assert projection.stationarity_residual.amax().item() <= 5e-6
    assert projection.objective_certificate.amin().item() >= -1e-6
    assert projection.singleton_iterations_h.amax().item() <= 8
    assert projection.singleton_iterations_b.amax().item() <= 8
    assert projection.dual_iterations.amax().item() <= 6
    constants = dict(projection.solver_constants)
    assert constants == {
        "simplex_tolerance": 1e-6,
        "risk_direction_variance_min": 1e-8,
        "risk_scale_min": 1e-4,
        "lambda_grid": (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        "lambda_max": 32.0,
        "singleton_iterations": 8.0,
        "dual_iterations": 6.0,
        "dual_alphas": (1.0, 0.5, 0.25, 0.125),
        "projected_gradient_step": 0.5,
        "eigenvalue_floor": 1e-6,
        "correlation_guard": 0.999,
        "risk_root_guard": 5e-7,
        "risk_tolerance": 1e-6,
        "projected_kkt_tolerance": 1e-6,
        "stationarity_tolerance": 5e-6,
        "lambda_on": 1e-5,
        "objective_tolerance": 1e-6,
    }
    assert projection.qhat.requires_grad
    for detached in (
        projection.lambda_h,
        projection.lambda_b,
        projection.candidate_valid,
        projection.candidate_objective,
        projection.active_set_code,
    ):
        assert not detached.requires_grad


def test_h_plus_b_slot_keeps_code_three_with_exact_zero_boundary_multiplier():
    q0 = torch.tensor([[
        1.0233670577619591e-09,
        2.6822461222764105e-05,
        1.4022510185895953e-05,
        0.9999589920043945,
        4.194736558105205e-09,
        1.4312242502256822e-09,
        1.6360675658688706e-07,
        3.7299202615415084e-10,
    ]])
    benefit = torch.tensor([[
        0.6229705214500427,
        -0.555220901966095,
        0.4247605800628662,
        0.782250165939331,
        0.8598630428314209,
        -0.23711588978767395,
        0.578286349773407,
        0.1819726973772049,
    ]])
    hard_risk = torch.tensor([[
        -0.6812615990638733,
        0.49931564927101135,
        -0.4326910376548767,
        -0.735956072807312,
        -0.8848799467086792,
        0.30799877643585205,
        -0.6055224537849426,
        -0.1612394005060196,
    ]])
    background_risk = torch.tensor([[
        -0.9554741382598877,
        -0.678473949432373,
        -0.7066000699996948,
        -0.5854964256286621,
        0.6245988011360168,
        -0.5952441692352295,
        0.9565935134887695,
        -0.5639734268188477,
    ]])
    requested = torch.ones(1, 1, dtype=torch.bool)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard_risk,
        background_risk,
        torch.tensor([[0.6337904930114746]]),
        hard_active=requested,
        background_active=requested,
    )

    assert bool(projection.accepted.item())
    assert projection.active_set_code.item() == 3
    assert projection.lambda_h.item() == 0.0
    assert projection.lambda_b.item() > 0.0
    assert torch.equal(
        projection.candidate_valid[0],
        torch.tensor([False, False, False, True]),
    )
    assert projection.kkt_max.item() <= core.SBSC_V31_EPS
    assert projection.stationarity_residual.item() <= 5e-6


def test_solver_four_terminal_states_are_exclusive_with_hard_modifier():
    q0 = torch.tensor([[0.2, 0.3, 0.5]]).repeat(4, 1)
    benefit = torch.tensor(
        [
            [0.2, -0.1, 0.4],
            [0.2, -0.1, 0.4],
            [0.2, float("nan"), 0.4],
            [0.2, -0.1, 0.4],
        ]
    )
    hard_risk = torch.tensor([[1.0, 0.0, -1.0]]).repeat(4, 1)
    background_risk = torch.tensor(
        [
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    reliability = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
    hard_requested = torch.tensor([[True], [True], [True], [False]])
    background_requested = torch.tensor([[True], [True], [True], [False]])
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard_risk,
        background_risk,
        reliability,
        hard_active=hard_requested,
        background_active=background_requested,
    )

    terminal = torch.cat(
        (
            projection.support_identity,
            projection.risk_identity,
            projection.solver_fallback,
            projection.accepted,
        ),
        dim=-1,
    )
    assert torch.equal(terminal, torch.eye(4, dtype=torch.bool))
    assert torch.equal(
        projection.reason_code[:, 0], torch.tensor([1, 2, 3, 0])
    )
    assert torch.equal(
        projection.active_set_code[:, 0], torch.tensor([-1, -1, -1, 0])
    )
    assert torch.equal(projection.qhat[:3], q0[:3])
    assert bool((projection.accepted & projection.hard_inactive)[3].item())
    assert projection.lambda_h[3].item() == 0.0
    # A requested, variable risk is defined before the row-level identity
    # decision, but no risk is effective after background degeneracy forces
    # risk_identity.  Defined and effective masks therefore are not aliases.
    assert projection.hard_defined[1].item()
    assert not projection.background_defined[1].item()
    assert not projection.hard_active[1].item()
    assert not projection.background_active[1].item()

    # Hard support can also be undefined/inactive while a background-only
    # projection remains fully eligible and accepted.
    gold = _active_set_gold_inputs()
    background_only = core.solve_dual_risk_projection_v31(
        gold[0][2:3],
        gold[1][2:3],
        gold[2][2:3],
        gold[3][2:3],
        gold[4][2:3],
        hard_active=torch.zeros(1, 1, dtype=torch.bool),
        background_active=torch.ones(1, 1, dtype=torch.bool),
    )
    assert not background_only.hard_defined.item()
    assert background_only.background_defined.item()
    assert not background_only.hard_active.item()
    assert background_only.background_active.item()
    assert background_only.hard_inactive.item()
    assert background_only.accepted.item()
    assert background_only.active_set_code.item() == 2
    assert background_only.lambda_h.item() == 0.0


def test_small_variance_near_collinear_dual_request_hits_correlation_guard():
    q0 = torch.tensor(
        [[
            0.01309138722717762,
            0.0835941806435585,
            0.28691917657852173,
            0.08786514401435852,
            0.5285300612449646,
        ]]
    )
    benefit = torch.tensor(
        [[
            -1.4415265321731567,
            -0.016975419595837593,
            -0.6816605925559998,
            -0.07357099652290344,
            0.5322969555854797,
        ]]
    )
    hard_risk = torch.tensor(
        [[
            0.00014225718041416258,
            0.0000026538525617070263,
            0.00019999999494757503,
            -0.00006928026414243504,
            -0.00010099842620547861,
        ]]
    )
    background_risk = torch.tensor(
        [[
            -0.0001420571788912639,
            -0.000002660548943822505,
            -0.0001999375963350758,
            0.00006912899698363617,
            0.000100985802419018,
        ]]
    )
    centered_h = hard_risk - (q0 * hard_risk).sum(dim=-1, keepdim=True)
    centered_b = background_risk - (
        q0 * background_risk
    ).sum(dim=-1, keepdim=True)
    variance_h = (q0 * centered_h.square()).sum(dim=-1)
    variance_b = (q0 * centered_b.square()).sum(dim=-1)
    correlation = (q0 * centered_h * centered_b).sum(dim=-1) / torch.sqrt(
        variance_h * variance_b
    )
    assert 1e-8 <= variance_h.item() < 2e-8
    assert 1e-8 <= variance_b.item() < 2e-8
    assert correlation.abs().item() > 0.999

    active = torch.ones(1, 1, dtype=torch.bool)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard_risk,
        background_risk,
        torch.ones(1, 1),
        hard_active=active,
        background_active=active,
    )
    assert projection.hard_defined.item()
    assert projection.background_defined.item()
    assert projection.hard_active.item()
    assert projection.background_active.item()
    assert projection.correlation_blocked.item()
    assert projection.solver_fallback.item()
    assert projection.reason_code.item() == 4
    assert projection.active_set_code.item() == -1
    assert not bool(projection.candidate_valid.any())
    assert torch.equal(projection.qhat, q0)


def test_gain_vector_projection_bounds_and_all_four_zero_init_gradients(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    module = _module(candidate)
    with torch.no_grad():
        module.raw_dual_risk_level_gain.copy_(
            torch.tensor([-1.0, 0.1, 0.3, 2.0])
        )
    core.project_sbsc_v31_constraints_(candidate)
    assert torch.allclose(
        module.raw_dual_risk_level_gain,
        torch.tensor([0.0, 0.1, 0.25, 0.25]),
        rtol=0.0,
        atol=0.0,
    )
    core.validate_sctransnet_sbsc_v31(candidate)

    with torch.no_grad():
        module.raw_dual_risk_level_gain.zero_()
    generator = torch.Generator().manual_seed(221)
    embeddings = tuple(
        torch.randn(1, channels, 2, 2, generator=generator)
        for channels in (32, 64, 128, 256)
    )
    emb_all = torch.cat(embeddings, dim=1)
    outputs = module(*embeddings, emb_all)[:4]
    objective = sum(output.square().mean() for output in outputs)
    gradient = torch.autograd.grad(
        objective, module.raw_dual_risk_level_gain
    )[0]
    assert tuple(gradient.shape) == (4,)
    assert bool(torch.isfinite(gradient).all())
    assert bool((gradient.abs() > 0.0).all())


def test_singleton_convergence_freezes_the_certified_root():
    q0 = torch.tensor(
        [[
            0.6076789498329163,
            0.23535163700580597,
            0.04775110259652138,
            0.10921832919120789,
        ]]
    )
    benefit = torch.tensor(
        [[
            -0.9770974516868591,
            -0.8146877288818359,
            0.5789417028427124,
            0.2884628474712372,
        ]]
    )
    risk = torch.tensor(
        [[
            -0.35333025455474854,
            0.14324653148651123,
            0.6194113492965698,
            0.972679078578949,
        ]]
    )
    centered_benefit = benefit - (q0 * benefit).sum(dim=-1, keepdim=True)
    direction = risk - (q0 * risk).sum(dim=-1, keepdim=True)
    phi = direction / direction.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    root, available, allowed, iterations = core._solve_singleton_root(
        q0,
        centered_benefit,
        phi,
        torch.ones(1, 1, dtype=torch.bool),
    )
    q_root, *_ = core._row_law(
        q0,
        centered_benefit,
        phi,
        torch.zeros_like(phi),
        root,
        torch.zeros_like(root),
    )
    residual = (q_root * phi).sum(dim=-1, keepdim=True)
    assert available.item() and allowed.item()
    assert iterations.item() < 8
    assert root.item() == pytest.approx(1.0475366, abs=2e-6)
    assert residual.item() <= 0.0
    assert abs(residual.item()) <= 1e-6


def test_projected_gradient_fallback_can_recover_a_certified_dual_row():
    q0 = torch.tensor(
        [[
            0.15556375682353973,
            0.20285366475582123,
            0.23029793798923492,
            0.3240487277507782,
            0.08723592013120651,
        ]]
    )
    benefit = torch.tensor(
        [[
            -1.095760703086853,
            -0.21956343948841095,
            0.010140337981283665,
            -0.359384149312973,
            -0.26721876859664917,
        ]]
    )
    hard = torch.tensor(
        [[
            -0.10484170913696289,
            0.3521955907344818,
            1.0365928411483765,
            -0.2796572744846344,
            1.7364007234573364,
        ]]
    )
    background = torch.tensor(
        [[
            -1.4855726957321167,
            0.0517989881336689,
            -0.9946146607398987,
            -1.9540975093841553,
            -0.5185905694961548,
        ]]
    )
    requested = torch.ones(1, 1, dtype=torch.bool)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard,
        background,
        torch.ones(1, 1),
        hard_active=requested,
        background_active=requested,
    )
    assert projection.projected_gradient_used.item()
    assert projection.projected_newton_used.item()
    assert not projection.line_search_failed.item()
    assert projection.accepted.item()
    assert projection.active_set_code.item() == 3
    assert projection.kkt_max.item() <= 1e-6
    assert projection.stationarity_residual.item() <= 5e-6
    assert projection.objective_certificate.item() >= -1e-6


def test_materialized_positive_support_underflow_invalidates_candidate():
    q0 = torch.full((1, 4), 0.25)
    centered_benefit = torch.zeros_like(q0)
    phi = torch.tensor([[-1.0, 1.0, -1.0, 1.0]])
    multiplier = torch.full((1, 1), 32.0)
    q, log_q, log_q0, _ = core._row_law(
        q0,
        centered_benefit,
        phi,
        phi,
        multiplier,
        multiplier,
    )
    active = torch.ones(1, 1, dtype=torch.bool)
    certificate = core._candidate_certificate(
        q=q,
        log_q=log_q,
        log_q0=log_q0,
        q0=q0,
        benefit=torch.zeros_like(q0),
        centered_benefit=centered_benefit,
        hard_risk=phi,
        background_risk=phi,
        phi_h=phi,
        phi_b=phi,
        lambda_h=multiplier,
        lambda_b=multiplier,
        reliability=torch.ones(1, 1),
        hard_defined=active,
        background_defined=active,
        hard_effective=active,
        background_effective=active,
        allowed=active,
    )
    assert bool((q0 > 0.0).all())
    assert bool((q == 0.0).any())
    assert not certificate["support_preserved"].item()
    assert not certificate["valid"].item()


def test_public_solver_bracket_failure_is_reason_four_exact_fallback():
    q0 = torch.tensor([[0.5, 0.5, 0.0]])
    benefit = torch.tensor([[-1.0, 1.0, 0.0]])
    hard = torch.tensor([[-1.01e-4, 1.01e-4, 1.0]])
    background = torch.zeros_like(q0)
    requested = torch.ones(1, 1, dtype=torch.bool)
    not_requested = torch.zeros_like(requested)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard,
        background,
        torch.ones(1, 1),
        hard_active=requested,
        background_active=not_requested,
    )

    assert projection.hard_defined.item()
    assert projection.hard_active.item()
    assert projection.bracket_failed_h.item()
    assert not projection.bracket_failed_b.item()
    assert projection.solver_fallback.item()
    assert projection.reason_code.item() == 4
    assert not bool(projection.candidate_valid.any())
    assert torch.equal(projection.qhat, q0)


def test_public_solver_lambda_cap_objective_failure_is_reason_seven():
    q0 = torch.tensor([[0.5, 0.5, 0.0]])
    benefit = torch.tensor([[-0.999975, 0.999975, 0.0]])
    hard = torch.tensor([[-0.03125, 0.03125, 1.0]])
    background = torch.zeros_like(q0)
    requested = torch.ones(1, 1, dtype=torch.bool)
    not_requested = torch.zeros_like(requested)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard,
        background,
        torch.ones(1, 1),
        hard_active=requested,
        background_active=not_requested,
    )

    assert not projection.bracket_failed_h.item()
    assert projection.candidate_allowed[0, 1].item()
    assert projection.candidate_lambda_h[0, 1].item() == 32.0
    assert projection.candidate_kkt_ok[0, 1].item()
    assert projection.candidate_stationarity_ok[0, 1].item()
    assert projection.candidate_risk_ok[0, 1].item()
    assert not projection.candidate_objective_ok[0, 1].item()
    assert not projection.candidate_valid[0, 1].item()
    assert projection.solver_fallback.item()
    assert projection.reason_code.item() == 7
    assert torch.equal(projection.qhat, q0)


def test_public_solver_line_search_failure_is_reason_four_exact_fallback():
    q0 = torch.tensor(
        [[0.3952520489692688, 0.21449607610702515, 0.39025190472602844]]
    )
    benefit = torch.tensor(
        [[-0.3056657314300537, 0.7083456516265869, -0.4893137216567993]]
    )
    hard = torch.tensor(
        [[-0.9645179510116577, -0.7041947841644287, -0.1864985227584839]]
    )
    background = torch.tensor(
        [[-0.14735746383666992, -0.10729765892028809, -0.27900898456573486]]
    )
    requested = torch.ones(1, 1, dtype=torch.bool)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard,
        background,
        torch.tensor([[0.8788509964942932]]),
        hard_active=requested,
        background_active=requested,
    )

    assert not projection.bracket_failed_h.item()
    assert not projection.bracket_failed_b.item()
    assert not projection.correlation_blocked.item()
    assert projection.line_search_failed.item()
    assert projection.projected_newton_used.item()
    assert not projection.projected_gradient_used.item()
    assert projection.dual_iterations.item() == 3
    assert projection.solver_fallback.item()
    assert projection.reason_code.item() == 4
    assert torch.equal(projection.qhat, q0)


def test_public_solver_j_plus_underflow_is_reason_five_exact_fallback():
    zero = torch.tensor(0.0, dtype=torch.float32)
    tiny = torch.nextafter(zero, torch.tensor(1.0, dtype=torch.float32))
    q0 = torch.stack((tiny, torch.tensor(0.5), torch.tensor(0.5)))[None]
    benefit = torch.tensor([[-1.0, 0.0, 0.0]])
    hard = torch.zeros_like(q0)
    background = torch.tensor([[0.0, -1.0, 1.0]])
    requested = torch.ones(1, 1, dtype=torch.bool)
    not_requested = torch.zeros_like(requested)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard,
        background,
        torch.ones(1, 1),
        hard_active=not_requested,
        background_active=requested,
    )

    assert q0[0, 0].item() > 0.0
    assert projection.background_defined.item()
    assert projection.background_active.item()
    assert torch.equal(
        projection.candidate_support_preserved,
        torch.zeros(1, 4, dtype=torch.bool),
    )
    assert not projection.bracket_failed_b.item()
    assert not projection.line_search_failed.item()
    assert projection.solver_fallback.item()
    assert projection.reason_code.item() == 5
    assert torch.equal(projection.qhat, q0)


def test_public_solver_raw_risk_failure_is_reason_six_exact_fallback():
    q0 = torch.tensor(
        [[0.6137796640396118, 0.22911879420280457, 0.15710148215293884]]
    )
    benefit = torch.tensor(
        [[0.20084905624389648, -0.7909368276596069, -0.002842545509338379]]
    )
    hard = torch.tensor(
        [[-0.7851356267929077, -0.6253693103790283, 0.7580462694168091]]
    )
    background = torch.tensor(
        [[0.539788007736206, -0.9104857444763184, -0.16992342472076416]]
    )
    requested = torch.ones(1, 1, dtype=torch.bool)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        hard,
        background,
        torch.tensor([[0.8013830184936523]]),
        hard_active=requested,
        background_active=requested,
    )

    dual_slot = 3
    assert projection.candidate_allowed[0, dual_slot].item()
    assert projection.candidate_kkt_ok[0, dual_slot].item()
    assert projection.candidate_stationarity_ok[0, dual_slot].item()
    assert not projection.candidate_risk_ok[0, dual_slot].item()
    assert projection.candidate_objective_ok[0, dual_slot].item()
    assert projection.candidate_risk_max[0, dual_slot].item() > 1e-6
    assert not projection.candidate_valid[0, dual_slot].item()
    assert projection.solver_fallback.item()
    assert projection.reason_code.item() == 6
    assert torch.equal(projection.qhat, q0)


def test_solver_rejects_non_fp32_probability_and_off_device_masks():
    q0 = torch.tensor([[0.4, 0.6]])
    zeros = torch.zeros_like(q0)
    requested = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(TypeError, match="must be FP32"):
        core.solve_dual_risk_projection_v31(
            q0.double(),
            zeros.double(),
            zeros.double(),
            zeros.double(),
            torch.ones(1, 1, dtype=torch.float64),
            hard_active=requested,
            background_active=requested,
        )


def test_identity_projection_bypasses_lossy_q0_reconstruction(paired_models):
    _baseline, candidate, _metadata = paired_models
    module = _module(candidate)
    module.eval()
    positions = 2
    query = F.normalize(torch.ones(1, 1, 32, positions), dim=-1)
    key = F.normalize(
        torch.ones(1, 1, module.KV_size, positions), dim=-1
    )
    validations = tuple(
        torch.zeros(1, 1, 1, positions) for _ in range(4)
    )
    support_bundle = core.estimate_c3_v31_support_from_validations(
        key, validations
    )
    support = support_bundle.levels[0]
    row = torch.zeros(module.KV_size)
    row[0] = 2.076710437028968e-20
    row[1] = 2.756201523881334e27
    base = row.reshape(1, 1, 1, -1).expand(1, 1, 32, -1).clone()
    with torch.no_grad():
        emitted, diagnostics = module._project_level(
            level_index=0,
            query=query,
            key=key,
            base_attention=base,
            support=support,
            key_confidence=support_bundle.key_confidence,
            key_confidence_valid=support_bundle.key_confidence_valid,
            gain=torch.full((4,), 0.25),
            gain_override=None,
            intervention="full",
        )
    assert diagnostics["projection"].support_identity.all()
    assert torch.equal(emitted, base)
