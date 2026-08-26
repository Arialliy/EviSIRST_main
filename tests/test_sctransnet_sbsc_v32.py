from __future__ import annotations

import copy
import os
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments.sctransnet_sbsc_v31 as v31
import experiments.sctransnet_sbsc_v32 as core
from experiments.four_dataset_models_seed42_v1 import (
    ORIGINAL_PARAMETER_COUNT,
    ORIGINAL_STATE_KEY_COUNT,
    state_dict_sha256,
)
from model.SCTransNet import Attention_org


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def paired_models():
    torch.manual_seed(7302)
    expected = torch.rand(7)
    torch.manual_seed(7302)
    baseline, candidate, metadata = core.build_paired_sctransnet_sbsc_v32(
        "IRSTD-1K", architecture_seed=42
    )
    observed = torch.rand(7)
    assert torch.equal(observed, expected)
    return baseline, candidate, metadata


def _module(model):
    module = model.mtc.encoder.layer[1].channel_attn
    assert type(module) is core.LearnedTriEvidenceProjectionV32
    return module


def _embeddings(
    *, batch: int = 1, height: int = 2, width: int = 3, seed: int = 91
):
    generator = torch.Generator().manual_seed(seed)
    levels = tuple(
        torch.randn(
            batch, channels, height, width, generator=generator
        )
        for channels in (32, 64, 128, 256)
    )
    return levels, torch.cat(levels, dim=1)


def _v31_project_level_fixture(module, *, seed: int = 1):
    """One deterministic public-support row for the inherited V3.1 wrapper."""
    generator = torch.Generator().manual_seed(seed)
    query = F.normalize(
        torch.randn(1, 1, 1, 6, generator=generator), dim=-1
    )
    key = F.normalize(
        torch.randn(1, 1, module.KV_size, 6, generator=generator), dim=-1
    )
    validations = tuple(
        torch.randn(1, 1, 1, 6, generator=generator) for _ in range(4)
    )
    support_bundle = v31.estimate_c3_v31_support_from_validations(
        key, validations
    )
    relation = (query @ key.transpose(-2, -1)) / float(
        module.KV_size
    ) ** 0.5
    base_attention = module.softmax(module.psi(relation))
    return query, key, support_bundle, base_attention


def _hook_cardinality(model):
    names = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    return tuple(
        tuple(len(getattr(module, name, {})) for name in names)
        for module in model.modules()
    )


def _assert_dataclass_bitwise(left, right):
    assert type(left) is type(right)
    for name in left.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value)
        elif (
            isinstance(left_value, tuple)
            and left_value
            and hasattr(left_value[0], "__dataclass_fields__")
        ):
            assert len(left_value) == len(right_value)
            for left_item, right_item in zip(left_value, right_value):
                _assert_dataclass_bitwise(left_item, right_item)
        else:
            assert left_value == right_value


def _cuda_support_inputs():
    generator = torch.Generator().manual_seed(667)
    key = F.normalize(
        torch.randn(1, 1, 9, 6, generator=generator), dim=-1
    ).cuda()
    queries = tuple(
        F.normalize(
            torch.randn(1, 1, channels, 6, generator=generator), dim=-1
        ).cuda()
        for channels in (2, 3, 4, 5)
    )
    attentions = tuple(
        F.softmax(
            torch.randn(1, 1, channels, 9, generator=generator), dim=-1
        ).cuda()
        for channels in (2, 3, 4, 5)
    )
    return key, queries, attentions


def test_runtime_integration_contract_and_adapter_binding_fail_closed(
    monkeypatch,
):
    expected = {
        "schema": "sctransnet_sbsc_v32/runtime_integration/v1",
        "cuda_median_adapter": (
            "sctransnet_sbsc_v32/cuda_strict_median_value_adapter/v1"
        ),
    }
    assert core.SBSC_V32_RUNTIME_INTEGRATION_SCHEMA == expected["schema"]
    assert core.validate_sbsc_v32_runtime_integration() == expected
    forward_globals = core.LearnedTriEvidenceProjectionV32._forward_impl.__globals__
    assert forward_globals["_estimate_c3_v31_support_v32"] is (
        core._FROZEN_V32_SUPPORT_ADAPTER_FUNCTION
    )
    monkeypatch.setitem(
        forward_globals,
        "_estimate_c3_v31_support_v32",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="not bound to the frozen CUDA adapter"):
        core.validate_sbsc_v32_runtime_integration()


def test_pair_structure_state_router_and_rng_contract(paired_models):
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
    assert type(candidate_modules[1]) is core.LearnedTriEvidenceProjectionV32
    assert all(
        type(candidate_modules[index]) is Attention_org for index in (0, 2, 3)
    )
    assert len(baseline_state) == ORIGINAL_STATE_KEY_COUNT == 510
    assert len(candidate_state) == core.EXPECTED_SBSC_V32_STATE_KEY_COUNT == 513
    assert sum(p.numel() for p in baseline.parameters()) == (
        ORIGINAL_PARAMETER_COUNT
    )
    assert sum(p.numel() for p in candidate.parameters()) == (
        core.EXPECTED_SBSC_V32_PARAMETER_COUNT
    ) == 11_330_188
    assert set(candidate_state) - set(baseline_state) == {
        core.SBSC_V32_GAIN_STATE_KEY,
        *core.SBSC_V32_ROUTER_STATE_KEYS,
    }
    assert not set(baseline_state) - set(candidate_state)
    assert all(
        torch.equal(baseline_state[key], candidate_state[key])
        for key in baseline_state
    )

    module = _module(candidate)
    assert type(module.tri_router.value_proj) is nn.Conv2d
    assert tuple(module.tri_router.value_proj.weight.shape) == (8, 480, 1, 1)
    assert module.tri_router.value_proj.bias is None
    assert type(module.tri_router.activation) is nn.SiLU
    assert type(module.tri_router.head) is nn.Conv2d
    assert tuple(module.tri_router.head.weight.shape) == (3, 15, 3, 3)
    assert module.tri_router.head.bias is None
    assert sum(p.numel() for p in module.tri_router.parameters()) == 4_245
    assert torch.count_nonzero(module.tri_router.value_proj.weight).item() > 0
    assert torch.count_nonzero(module.tri_router.head.weight).item() > 0
    assert torch.count_nonzero(module.raw_dual_risk_level_gain).item() == 0
    literal_router_hash = (
        "a8f549893b4819442dfe3dcb2fb3822062501fac817c3dd6b0ea949dd3a4376d"
    )
    assert core.EXPECTED_SBSC_V32_ROUTER_INIT_SHA256 == literal_router_hash
    assert state_dict_sha256(module.tri_router.state_dict()) == literal_router_hash
    assert torch.equal(
        module.tri_router.value_proj.weight.flatten()[:4],
        torch.tensor(
            [
                0.011451635509729385,
                -0.006819989066570997,
                -0.010679320432245731,
                0.03087213821709156,
            ]
        ),
    )
    assert torch.equal(
        module.tri_router.head.weight.flatten()[:4],
        torch.tensor(
            [
                0.024878138676285744,
                0.020893070846796036,
                0.038230981677770615,
                -0.06520695984363556,
            ]
        ),
    )

    assert metadata["schema"] == core.SBSC_V32_SCHEMA
    assert metadata["method"] == "sbsc_v32"
    assert metadata["router_state_keys"] == list(
        core.SBSC_V32_ROUTER_STATE_KEYS
    )
    assert metadata["router_parameter_count"] == 4_245
    assert metadata["router_initial_state_sha256"] == literal_router_hash
    assert metadata["loss_schema"] == core.SBSC_V32_LOSS_SCHEMA
    assert metadata["cuda_strict_deterministic_median_adapter"] == (
        core.SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA
    )
    assert metadata["shared_state_bitwise_equal"] is True
    assert metadata["shared_state_sha256"] == state_dict_sha256(baseline_state)
    assert metadata["parent_checkpoint"] is None
    assert metadata["warm_start_used"] is False
    assert metadata["predecessor_checkpoint_used"] is False
    assert metadata["model_construction_preserves_caller_rng_stream"] is True
    validation = core.validate_sctransnet_sbsc_v32(
        candidate, require_zero_gain=True
    )
    assert validation["live_value_evidence"] is True
    assert validation["independent_spatial_role_softmax"] is True
    assert validation["v31_solver_source_sha256"] == (
        core.V31_SOLVER_SOURCE_SHA256
    )
    assert validation["cuda_strict_deterministic_median_adapter"] == (
        core.SBSC_V32_CUDA_MEDIAN_ADAPTER_SCHEMA
    )


def test_state_contract_and_v31_v32_bidirectional_rejection(paired_models):
    baseline, candidate, _metadata = paired_models
    baseline_state = dict(baseline.state_dict())
    candidate_state = candidate.state_dict()
    v31_state = dict(baseline_state)
    v31_state[v31.SBSC_V31_GAIN_STATE_KEY] = torch.zeros(4)

    core.validate_sbsc_v32_state_dict(baseline_state, "sctransnet")
    core.validate_sbsc_v32_state_dict(candidate_state, "sbsc_v32")
    v31.validate_sbsc_v31_state_dict(v31_state, "sbsc_v31")
    with pytest.raises(ValueError, match="key set differs"):
        core.validate_sbsc_v32_state_dict(v31_state, "sbsc_v32")
    with pytest.raises(ValueError, match="key set differs"):
        v31.validate_sbsc_v31_state_dict(candidate_state, "sbsc_v31")

    malformed = dict(candidate_state)
    malformed[core.SBSC_V32_ROUTER_STATE_KEYS[0]] = torch.zeros(
        8, 480, 1, 2
    )
    with pytest.raises(ValueError, match="shape differs"):
        core.validate_sbsc_v32_state_dict(malformed, "sbsc_v32")
    malformed = dict(candidate_state)
    malformed[core.SBSC_V32_ROUTER_STATE_KEYS[1]] = malformed[
        core.SBSC_V32_ROUTER_STATE_KEYS[1]
    ].double()
    with pytest.raises(TypeError, match="dtype differs"):
        core.validate_sbsc_v32_state_dict(malformed, "sbsc_v32")
    malformed = dict(candidate_state)
    bad_router = malformed[core.SBSC_V32_ROUTER_STATE_KEYS[1]].clone()
    bad_router.flatten()[0] = float("nan")
    malformed[core.SBSC_V32_ROUTER_STATE_KEYS[1]] = bad_router
    with pytest.raises(ValueError, match="must be finite"):
        core.validate_sbsc_v32_state_dict(malformed, "sbsc_v32")
    malformed = dict(candidate_state)
    malformed[core.SBSC_V32_GAIN_STATE_KEY] = torch.tensor(
        [0.0, 0.1, 0.2, 0.25001]
    )
    with pytest.raises(ValueError, match="outside"):
        core.validate_sbsc_v32_state_dict(malformed, "sbsc_v32")


def test_bf16_source_replacement_keeps_router_fp32_rng_and_forward_identity(
    paired_models,
):
    baseline, frozen_candidate, _metadata = paired_models
    source = copy.deepcopy(
        baseline.mtc.encoder.layer[1].channel_attn
    ).to(dtype=torch.bfloat16).eval()
    torch.manual_seed(8841)
    expected_rng = torch.rand(5)
    torch.manual_seed(8841)
    replacement = core.LearnedTriEvidenceProjectionV32.from_ssca(
        source, layer_index=1
    ).eval()
    observed_rng = torch.rand(5)
    assert torch.equal(observed_rng, expected_rng)
    assert replacement.raw_dual_risk_level_gain.dtype is torch.float32
    assert replacement.tri_router.value_proj.weight.dtype is torch.float32
    assert replacement.tri_router.head.weight.dtype is torch.float32
    assert torch.equal(
        replacement.tri_router.value_proj.weight,
        _module(frozen_candidate).tri_router.value_proj.weight,
    )
    assert torch.equal(
        replacement.tri_router.head.weight,
        _module(frozen_candidate).tri_router.head.weight,
    )
    assert all(
        replacement.state_dict()[key].dtype is torch.bfloat16
        for key in source.state_dict()
    )
    levels, emb_all = _embeddings(height=2, width=3, seed=441)
    levels = tuple(value.bfloat16() for value in levels)
    emb_all = emb_all.bfloat16()
    with torch.no_grad():
        source_outputs = source(*levels, emb_all)
        replacement_outputs = replacement(*levels, emb_all)
    assert all(
        torch.equal(left, right)
        for left, right in zip(source_outputs[:4], replacement_outputs[:4])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_pair_construction_preserves_initialized_cpu_and_all_cuda_rng_states():
    original_cpu = torch.get_rng_state()
    original_cuda = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(8123)
        expected_cpu = torch.get_rng_state().clone()
        expected_cuda = tuple(
            state.clone() for state in torch.cuda.get_rng_state_all()
        )
        core.build_paired_sctransnet_sbsc_v32("NUAA-SIRST")
        assert torch.equal(torch.get_rng_state(), expected_cpu)
        observed_cuda = torch.cuda.get_rng_state_all()
        assert len(observed_cuda) == len(expected_cuda)
        assert all(
            torch.equal(observed, expected)
            for observed, expected in zip(observed_cuda, expected_cuda)
        )
    finally:
        torch.set_rng_state(original_cpu)
        torch.cuda.set_rng_state_all(original_cuda)


def test_frozen_v31_source_provenance_is_literal_and_fail_closed(
    paired_models, monkeypatch
):
    _baseline, candidate, _metadata = paired_models
    literal = "b2d1e3f97607b305551a0602076605968041878eafd822c6f3335840ea725a3a"
    assert core.EXPECTED_V31_SOLVER_SOURCE_SHA256 == literal
    assert core.V31_SOLVER_SOURCE_SHA256 == literal
    monkeypatch.setattr(core, "V31_SOLVER_SOURCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="source SHA256 differs"):
        core.validate_sctransnet_sbsc_v32(candidate)
    with pytest.raises(RuntimeError, match="source SHA256 differs"):
        core.validate_sbsc_v32_state_dict(candidate.state_dict(), "sbsc_v32")


@pytest.mark.parametrize(
    "function_name",
    ("estimate_c3_v31_support", "solve_dual_risk_projection_v31"),
)
def test_v31_runtime_function_replacement_is_rejected(
    paired_models, monkeypatch, function_name
):
    _baseline, candidate, _metadata = paired_models
    original = getattr(v31, function_name)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(v31, function_name, replacement)
    with pytest.raises(RuntimeError, match="implementation was replaced"):
        core.validate_sctransnet_sbsc_v32(candidate)


def test_validator_rejects_mutation_hooks_and_solver_override(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models

    shadowed = copy.deepcopy(frozen_candidate)
    _module(shadowed).forward = _module(shadowed).forward
    with pytest.raises(RuntimeError, match="instance method shadow"):
        core.validate_sctransnet_sbsc_v32(shadowed)

    hooked = copy.deepcopy(frozen_candidate)
    handle = _module(hooked).register_forward_hook(
        lambda _module, _inputs, output: output
    )
    try:
        with pytest.raises(RuntimeError, match="hooks are forbidden"):
            core.validate_sctransnet_sbsc_v32(hooked)
    finally:
        handle.remove()

    overridden = copy.deepcopy(frozen_candidate)
    _module(overridden).risk_tolerance = core.SBSC_V32_EPS
    with pytest.raises(RuntimeError, match="constants cannot be instance"):
        core.validate_sctransnet_sbsc_v32(overridden)

    wrong_softmax = copy.deepcopy(frozen_candidate)
    wrong_softmax.mtc.encoder.layer[0].channel_attn.softmax.dim = 2
    with pytest.raises(RuntimeError, match="outside L1 SSCA"):
        core.validate_sctransnet_sbsc_v32(wrong_softmax)

    diagnostic = copy.deepcopy(frozen_candidate)
    diagnostic.diagnostic_only = True
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        core.validate_sctransnet_sbsc_v32(diagnostic)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    (
        (("tri_router", "value_proj", "stride"), (2, 2)),
        (("tri_router", "value_proj", "padding"), (1, 1)),
        (("tri_router", "value_proj", "dilation"), (2, 2)),
        (("tri_router", "value_proj", "groups"), 2),
        (("tri_router", "value_proj", "padding_mode"), "reflect"),
        (("tri_router", "head", "stride"), (2, 2)),
        (("tri_router", "head", "padding"), (0, 0)),
        (("tri_router", "head", "dilation"), (2, 2)),
        (("tri_router", "head", "groups"), 3),
        (("tri_router", "head", "padding_mode"), "reflect"),
        (("tri_router", "activation", "inplace"), True),
    ),
)
def test_validator_freezes_every_router_operator_attribute(
    paired_models, path, bad_value
):
    _baseline, frozen_candidate, _metadata = paired_models
    mutated = copy.deepcopy(frozen_candidate)
    owner = _module(mutated)
    for name in path[:-1]:
        owner = getattr(owner, name)
    setattr(owner, path[-1], bad_value)
    with pytest.raises(RuntimeError, match="router layer contract differs"):
        core.validate_sctransnet_sbsc_v32(mutated)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    (
        (("k", "padding_mode"), "reflect"),
        (("v", "stride"), (2, 2)),
        (("q1", "groups"), 1),
        (("project_out1", "padding"), (1, 1)),
    ),
)
def test_validator_freezes_inherited_l1_ssca_conv_contract(
    paired_models, path, bad_value
):
    _baseline, frozen_candidate, _metadata = paired_models
    mutated = copy.deepcopy(frozen_candidate)
    owner = _module(mutated)
    for name in path[:-1]:
        owner = getattr(owner, name)
    setattr(owner, path[-1], bad_value)
    with pytest.raises(RuntimeError, match="inherited L1 SSCA conv"):
        core.validate_sctransnet_sbsc_v32(mutated)


@pytest.mark.parametrize("conv_name", ("k", "v", "q1", "project_out1"))
def test_validator_rejects_frozen_active_inherited_l1_weight(
    paired_models, conv_name
):
    _baseline, frozen_candidate, _metadata = paired_models
    mutated = copy.deepcopy(frozen_candidate)
    getattr(_module(mutated), conv_name).weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="inherited L1 SSCA conv"):
        core.validate_sctransnet_sbsc_v32(mutated)


@pytest.mark.parametrize(
    "parameter_path",
    (
        ("tri_router", "value_proj", "weight"),
        ("tri_router", "head", "weight"),
        ("raw_dual_risk_level_gain",),
    ),
)
def test_validator_rejects_frozen_router_or_gain_parameter(
    paired_models, parameter_path
):
    _baseline, frozen_candidate, _metadata = paired_models
    mutated = copy.deepcopy(frozen_candidate)
    parameter = _module(mutated)
    for name in parameter_path:
        parameter = getattr(parameter, name)
    parameter.requires_grad_(False)
    with pytest.raises(RuntimeError, match="must require gradients|registration"):
        core.validate_sctransnet_sbsc_v32(mutated)


def test_validator_rejects_extra_stateless_router_module(paired_models):
    _baseline, frozen_candidate, _metadata = paired_models
    mutated = copy.deepcopy(frozen_candidate)
    _module(mutated).tri_router.extra = nn.ReLU()
    with pytest.raises(RuntimeError, match="router module tree differs"):
        core.validate_sctransnet_sbsc_v32(mutated)


@pytest.mark.parametrize(
    ("owner_name", "helper_name"),
    (
        ("router", "encode_value"),
        ("router", "route"),
        ("replacement", "_route_supports"),
        ("replacement", "_project_level"),
    ),
)
def test_validator_rejects_formal_path_helper_instance_shadows(
    paired_models, owner_name, helper_name
):
    _baseline, frozen_candidate, _metadata = paired_models
    mutated = copy.deepcopy(frozen_candidate)
    module = _module(mutated)
    owner = module.tri_router if owner_name == "router" else module
    setattr(owner, helper_name, lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="helper instance shadow"):
        core.validate_sctransnet_sbsc_v32(mutated)


@pytest.mark.parametrize(
    ("owner_path", "helper_name", "error_match"),
    (
        (("k",), "_conv_forward", "inherited L1 SSCA conv"),
        (("psi",), "_apply_instance_norm", "InstanceNorm2d contract"),
        (("psi",), "_check_input_dim", "InstanceNorm2d contract"),
        (
            ("tri_router", "head"),
            "_conv_forward",
            "router layer contract differs",
        ),
        (
            ("tri_router", "value_proj"),
            "_conv_forward",
            "router layer contract differs",
        ),
    ),
)
def test_validator_rejects_actual_call_helper_shadows(
    paired_models, owner_path, helper_name, error_match
):
    _baseline, frozen_candidate, _metadata = paired_models
    mutated = copy.deepcopy(frozen_candidate)
    owner = _module(mutated)
    for name in owner_path:
        owner = getattr(owner, name)
    setattr(owner, helper_name, getattr(owner, helper_name))
    with pytest.raises(RuntimeError, match=error_match):
        core.validate_sctransnet_sbsc_v32(mutated)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_strict_median_adapter_is_repeatable_and_restores_strict_state(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    v31_sha_before = core._sha256_file(core.Path(v31.__file__).resolve())
    key, queries, attentions = _cuda_support_inputs()
    tie_query = F.normalize(
        torch.randn(
            1, 1, 3, 6, generator=torch.Generator().manual_seed(668)
        ),
        dim=-1,
    ).cuda()
    tie_attention = F.softmax(
        torch.randn(
            1, 1, 3, 9, generator=torch.Generator().manual_seed(669)
        ),
        dim=-1,
    ).cuda()
    model = copy.deepcopy(frozen_candidate).cuda().eval()
    model.mode = "train"
    inputs = torch.randn(
        1, 1, 32, 32, generator=torch.Generator().manual_seed(670)
    ).cuda()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        with pytest.raises(RuntimeError, match="deterministic implementation"):
            torch.tensor([[1.0, 1.0, 2.0]], device="cuda").median(
                dim=-1
            )

        first = core._estimate_c3_v31_support_v32(
            key, queries, attentions, eps=1e-6, detach_support=True
        )
        second = core._estimate_c3_v31_support_v32(
            key, queries, attentions, eps=1e-6, detach_support=True
        )
        _assert_dataclass_bitwise(first, second)

        tie_queries = tuple(tie_query.clone() for _ in range(4))
        tie_attentions = tuple(tie_attention.clone() for _ in range(4))
        tie_first = core._estimate_c3_v31_support_v32(
            key, tie_queries, tie_attentions, eps=1e-6, detach_support=True
        )
        tie_second = core._estimate_c3_v31_support_v32(
            key, tie_queries, tie_attentions, eps=1e-6, detach_support=True
        )
        _assert_dataclass_bitwise(tie_first, tie_second)

        with torch.no_grad():
            outputs_first = model(inputs)
            outputs_second = model(inputs)
        assert len(outputs_first) == len(outputs_second) == 6
        assert all(
            torch.equal(left, right)
            for left, right in zip(outputs_first, outputs_second)
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )
    assert core._sha256_file(core.Path(v31.__file__).resolve()) == v31_sha_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("enabled", "warn_only", "escaped_warning_count"),
    (
        (False, False, 0),
        (False, True, 0),
        (True, True, 8),
        (True, False, 0),
    ),
)
def test_cuda_median_adapter_preserves_all_determinism_flag_combinations(
    enabled, warn_only, escaped_warning_count
):
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    key, queries, attentions = _cuda_support_inputs()
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with warnings.catch_warnings(record=True) as escaped:
            warnings.simplefilter("always")
            result = core._estimate_c3_v31_support_v32(
                key, queries, attentions, eps=1e-6, detach_support=True
            )
        assert isinstance(result, v31.C3V31Support)
        assert len(escaped) == escaped_warning_count
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert (
            torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
        )
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_median_adapter_failure_paths_restore_state_and_warning_filters(
    monkeypatch,
):
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    source_sha = core._sha256_file(core.Path(v31.__file__).resolve())
    original = v31.estimate_c3_v31_support
    key, queries, attentions = _cuda_support_inputs()
    exact_message = (
        "median CUDA with indices output does not have a deterministic "
        "implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
        "You can file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this "
        "operation. (Triggered internally at "
        "/pytorch/aten/src/ATen/Context.cpp:148.)"
    )
    source_file = str(core.Path(v31.__file__).resolve())

    def call_adapter():
        return core._estimate_c3_v31_support_v32(
            key, queries, attentions, eps=1e-6, detach_support=True
        )

    def assert_strict_restored():
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        sentinel = object()

        def no_warning(*_args, **_kwargs):
            return sentinel

        monkeypatch.setattr(v31, "estimate_c3_v31_support", no_warning)
        torch.use_deterministic_algorithms(False, warn_only=True)
        assert call_adapter() is sentinel
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

        torch.use_deterministic_algorithms(True, warn_only=False)
        with pytest.raises(RuntimeError, match="exactly eight"):
            call_adapter()
        assert_strict_restored()

        def nine_expected(*_args, **_kwargs):
            for line in (547,) * 5 + (552,) * 4:
                warnings.warn_explicit(
                    exact_message, UserWarning, source_file, line
                )
            return sentinel

        monkeypatch.setattr(v31, "estimate_c3_v31_support", nine_expected)
        with pytest.raises(RuntimeError, match="exactly eight"):
            call_adapter()
        assert_strict_restored()

        def wrong_line_counts(*_args, **_kwargs):
            for line in (547,) * 5 + (552,) * 3:
                warnings.warn_explicit(
                    exact_message, UserWarning, source_file, line
                )
            return sentinel

        monkeypatch.setattr(
            v31, "estimate_c3_v31_support", wrong_line_counts
        )
        with pytest.raises(RuntimeError, match="line counts differ"):
            call_adapter()
        assert_strict_restored()

        def wrong_source_or_lineno(*_args, **_kwargs):
            for line in (547,) * 4 + (552,) * 2:
                warnings.warn_explicit(
                    exact_message, UserWarning, source_file, line
                )
            warnings.warn_explicit(
                exact_message, UserWarning, source_file, 551
            )
            warnings.warn_explicit(
                exact_message, UserWarning, __file__, 552
            )
            return sentinel

        monkeypatch.setattr(
            v31, "estimate_c3_v31_support", wrong_source_or_lineno
        )
        with pytest.raises(RuntimeError, match="unexpected warning"):
            call_adapter()
        assert_strict_restored()

        class WrongMedianCategory(UserWarning):
            pass

        def wrong_category(*_args, **_kwargs):
            for line in (547,) * 4 + (552,) * 3:
                warnings.warn_explicit(
                    exact_message, UserWarning, source_file, line
                )
            warnings.warn_explicit(
                exact_message, WrongMedianCategory, source_file, 552
            )
            return sentinel

        monkeypatch.setattr(v31, "estimate_c3_v31_support", wrong_category)
        with pytest.raises(RuntimeError, match="unexpected warning"):
            call_adapter()
        assert_strict_restored()

        def unexpected_warning(*_args, **_kwargs):
            for line in (547,) * 4 + (552,) * 3:
                warnings.warn_explicit(
                    exact_message, UserWarning, source_file, line
                )
            warnings.warn("unexpected adapter warning", UserWarning)
            return sentinel

        monkeypatch.setattr(
            v31, "estimate_c3_v31_support", unexpected_warning
        )
        with pytest.raises(UserWarning, match="unexpected adapter warning"):
            call_adapter()
        assert_strict_restored()

        def support_exception(*_args, **_kwargs):
            raise LookupError("support exploded")

        monkeypatch.setattr(v31, "estimate_c3_v31_support", support_exception)
        with pytest.raises(LookupError, match="support exploded"):
            call_adapter()
        assert_strict_restored()

        with warnings.catch_warnings(record=True) as escaped:
            warnings.simplefilter("always")
            warnings.warn("post-adapter warning", UserWarning)
        assert len(escaped) == 1
        assert str(escaped[0].message) == "post-adapter warning"
    finally:
        monkeypatch.setattr(v31, "estimate_c3_v31_support", original)
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )
    assert core._sha256_file(core.Path(v31.__file__).resolve()) == source_sha


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("side", "amp_bfloat16"), ((32, True), (256, False), (256, True))
)
def test_gain_zero_is_exact_under_cuda_size_and_bf16(
    paired_models, side, amp_bfloat16
):
    frozen_baseline, frozen_candidate, _metadata = paired_models
    baseline = copy.deepcopy(frozen_baseline).cuda().eval()
    candidate = copy.deepcopy(frozen_candidate).cuda().eval()
    baseline.mode = "train"
    candidate.mode = "train"
    inputs = torch.randn(
        1,
        1,
        side,
        side,
        generator=torch.Generator().manual_seed(134 + side),
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


def test_spatial_energy_gold_and_three_independent_support_simplexes(
    paired_models,
):
    raw = torch.tensor(
        [[[[1.0, 2.0, 4.0, 8.0], [2.0, 4.0, 8.0, 16.0]]]],
        requires_grad=True,
    )
    energy = core.LearnedTriEvidenceProjectionV32._spatial_energy(raw)
    rms = torch.sqrt(raw.detach().float().square().mean(dim=-2, keepdim=True))
    log_rms = torch.log(rms + core.SBSC_V32_EPS)
    centered = log_rms - log_rms.mean(dim=-1, keepdim=True)
    expected = torch.tanh(
        centered
        / (
            torch.sqrt(centered.square().mean(dim=-1, keepdim=True))
            + core.SBSC_V32_EPS
        )
    )
    assert energy.dtype is torch.float32
    assert not energy.requires_grad
    assert torch.equal(energy, expected)

    _baseline, frozen_candidate, _metadata = paired_models
    module = copy.deepcopy(_module(frozen_candidate)).eval()
    levels, emb_all = _embeddings(batch=2, height=2, width=3, seed=218)
    with torch.no_grad():
        _outputs, diagnostics = module.forward_with_c3_diagnostics(
            *levels, emb_all
        )
    assert len(diagnostics) == 4
    for record in diagnostics:
        supports = record["router_supports"]
        logits = record["router_logits"]
        assert tuple(supports.shape) == (2, 3, 2, 3)
        assert tuple(logits.shape) == (2, 3, 2, 3)
        assert supports.dtype is torch.float32
        assert bool(torch.isfinite(supports).all())
        assert bool((supports >= 0.0).all())
        spatial_mass = supports.flatten(2).sum(dim=-1)
        assert torch.allclose(
            spatial_mass, torch.ones_like(spatial_mass), rtol=0.0, atol=1e-6
        )
        # Independent spatial distributions have total role mass 3, whereas
        # a forbidden per-token class softmax would have total mass N=6.
        assert torch.allclose(
            supports.flatten(1).sum(dim=-1),
            torch.full((2,), 3.0),
            rtol=0.0,
            atol=1e-6,
        )


def test_constant_and_nonfinite_router_evidence_are_exact_fallbacks(
    paired_models,
):
    frozen_baseline, frozen_candidate, _metadata = paired_models
    source = copy.deepcopy(
        frozen_baseline.mtc.encoder.layer[1].channel_attn
    ).eval()
    replacement = copy.deepcopy(_module(frozen_candidate)).eval()
    with torch.no_grad():
        replacement.raw_dual_risk_level_gain.fill_(0.25)

    constant_levels = tuple(
        torch.zeros(1, channels, 2, 3)
        for channels in (32, 64, 128, 256)
    )
    constant_all = torch.cat(constant_levels, dim=1)
    with torch.no_grad():
        source_constant = source(*constant_levels, constant_all)
        replacement_constant = replacement(*constant_levels, constant_all)
    assert all(
        torch.equal(left, right)
        for left, right in zip(source_constant[:4], replacement_constant[:4])
    )

    levels, emb_all = _embeddings(height=2, width=3, seed=219)
    with torch.no_grad():
        replacement.tri_router.head.weight.flatten()[0] = float("nan")
        source_outputs = source(*levels, emb_all)
        replacement_outputs = replacement(*levels, emb_all)
    assert all(
        torch.equal(left, right)
        for left, right in zip(source_outputs[:4], replacement_outputs[:4])
    )


def test_nonfinite_base_qkv_fails_closed_instead_of_claiming_fallback(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    module = copy.deepcopy(_module(frozen_candidate)).eval()
    with torch.no_grad():
        module.v.weight.fill_(torch.finfo(torch.float32).max)
        module.raw_dual_risk_level_gain.fill_(0.25)
    levels = tuple(
        torch.ones(1, channels, 2, 3)
        for channels in (32, 64, 128, 256)
    )
    emb_all = torch.cat(levels, dim=1)
    with torch.no_grad(), pytest.raises(
        RuntimeError, match="base Q/K/V contract contains nonfinite"
    ):
        module(*levels, emb_all)
    with torch.no_grad(), pytest.raises(
        RuntimeError, match="base Q/K/V contract contains nonfinite"
    ):
        module.forward_with_c3_diagnostics(*levels, emb_all)


def test_inherited_v31_project_level_attempts_and_emits_certified_projection(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    module = _module(frozen_candidate).eval()
    assert type(module)._project_level is v31.C3DualRiskProjectionV31._project_level
    query, key, support_bundle, base_attention = _v31_project_level_fixture(
        module
    )

    with torch.no_grad():
        emitted, diagnostics = module._project_level(
            level_index=0,
            query=query,
            key=key,
            base_attention=base_attention,
            support=support_bundle.levels[0],
            key_confidence=support_bundle.key_confidence,
            key_confidence_valid=support_bundle.key_confidence_valid,
            gain=torch.full((4,), 0.25),
            gain_override=None,
            intervention="full",
        )

    projection = diagnostics["projection"]
    assert diagnostics["base_contract_failure"] is False
    assert not projection.support_identity.item()
    assert not projection.risk_identity.item()
    assert not projection.solver_fallback.item()
    assert projection.accepted.item()
    assert projection.reason_code.item() == 0
    assert projection.active_set_code.item() == 3
    assert projection.dual_iterations.item() > 0
    assert diagnostics["qout32_certificate_ok"].item()
    assert diagnostics["emitted_certificate_ok"].item()
    assert not diagnostics["emission_fallback"].item()
    assert diagnostics["emission_reason_code"].item() == 0
    assert not torch.equal(projection.qhat, projection.q0)
    assert not torch.equal(emitted, base_attention)


def test_inherited_v31_ambient_cast_failure_is_exact_a0_fallback(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    module = _module(frozen_candidate).eval()
    query, key, support_bundle, shape_reference = _v31_project_level_fixture(
        module, seed=2
    )
    # Every A0 entry is the exactly FP16-representable value 2.0, so the FP32
    # and FP16 calls begin with identical real values.  FP32 emits successfully;
    # only the FP16 correction cast makes the post-emission certificate fail.
    ambient_a0 = torch.full_like(
        shape_reference, 2.0, dtype=torch.float16
    )
    assert bool(torch.isfinite(ambient_a0).all())
    assert bool((ambient_a0 > 0.0).all())

    kwargs = {
        "level_index": 0,
        "query": query,
        "key": key,
        "support": support_bundle.levels[0],
        "key_confidence": support_bundle.key_confidence,
        "key_confidence_valid": support_bundle.key_confidence_valid,
        "gain": torch.full((4,), 0.25),
        "gain_override": None,
        "intervention": "full",
    }
    with torch.no_grad():
        fp32_emitted, fp32_diagnostics = module._project_level(
            base_attention=ambient_a0.float(), **kwargs
        )
        emitted, diagnostics = module._project_level(
            base_attention=ambient_a0, **kwargs
        )

    fp32_projection = fp32_diagnostics["projection"]
    assert fp32_projection.accepted.item()
    assert fp32_diagnostics["qout32_certificate_ok"].item()
    assert fp32_diagnostics["emitted_certificate_ok"].item()
    assert not fp32_diagnostics["emission_fallback"].item()
    assert not torch.equal(fp32_emitted, ambient_a0.float())

    projection = diagnostics["projection"]
    assert diagnostics["base_contract_failure"] is False
    assert projection.accepted.item()
    assert not projection.solver_fallback.item()
    assert diagnostics["qout32_certificate_ok"].item()
    assert not diagnostics["emitted_certificate_ok"].item()
    assert diagnostics["emission_fallback"].item()
    assert diagnostics["emission_reason_code"].item() == 2
    assert diagnostics["emission_tolerance"] == (
        4.0 * torch.finfo(torch.float16).eps
    )
    assert emitted.dtype is torch.float16
    assert torch.equal(emitted, ambient_a0)


def test_teacher_targets_cover_background_target_hard_and_empty_roles():
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 1.0
    prediction = torch.full_like(target, 0.1)
    prediction[:, :, 0:2, 0:2] = 0.99
    prediction[:, :, 2:4, 2:4] = 0.9
    result = core.build_tri_router_targets_v32(
        target, prediction.detach(), (2, 2)
    )
    pooled_target = F.adaptive_max_pool2d(target, (2, 2))
    pooled_prediction = F.adaptive_max_pool2d(prediction, (2, 2))
    outside = 1.0 - pooled_target
    expected_raw = torch.cat(
        (
            pooled_target,
            outside * -torch.log((1.0 - pooled_prediction).clamp_min(1e-6)),
            outside * (1.0 - pooled_prediction),
        ),
        dim=1,
    )
    assert torch.equal(result.raw, expected_raw)
    assert result.valid.tolist() == [[True, True, True]]
    assert torch.equal(result.raw[:, 0:1] * outside, torch.zeros_like(outside))
    assert torch.equal(
        result.raw[:, 1:2] * pooled_target,
        torch.zeros_like(pooled_target),
    )
    assert result.raw[0, 1, 1, 1] > result.raw[0, 1, 0, 1]
    assert torch.allclose(
        result.probability.flatten(2).sum(dim=-1),
        torch.ones(1, 3),
        rtol=0.0,
        atol=1e-6,
    )
    assert not result.raw.requires_grad

    background = core.build_tri_router_targets_v32(
        torch.zeros_like(target), torch.full_like(target, 0.5), (2, 2)
    )
    assert background.valid.tolist() == [[False, True, True]]
    assert torch.count_nonzero(background.probability[:, 0]).item() == 0
    no_hard = core.build_tri_router_targets_v32(
        torch.zeros_like(target), torch.zeros_like(target), (2, 2)
    )
    assert no_hard.valid.tolist() == [[False, False, True]]
    all_target = core.build_tri_router_targets_v32(
        torch.ones_like(target), torch.ones_like(target), (2, 2)
    )
    assert all_target.valid.tolist() == [[True, False, False]]


def test_router_loss_normalization_detach_and_all_empty_fail_closed():
    logits = tuple(
        torch.zeros(1, 3, 2, 2, requires_grad=True) for _ in range(4)
    )
    capture = core.C3V32TrainingRouterCollector(
        records=[{"logits": logits}]
    )
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 1.0
    prediction = torch.full_like(target, 0.5)
    loss = core.tri_router_supervision_loss(
        capture, prediction.detach(), target
    )
    assert loss.item() == pytest.approx(1.0, abs=1e-7)
    gradients = torch.autograd.grad(loss, logits)
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    with pytest.raises(ValueError, match="explicitly detached"):
        core.tri_router_supervision_loss(
            capture, prediction.requires_grad_(), target
        )

    empty_logits = tuple(
        torch.empty(0, 3, 2, 2, requires_grad=True) for _ in range(4)
    )
    empty_capture = core.C3V32TrainingRouterCollector(
        records=[{"logits": empty_logits}]
    )
    with pytest.raises(RuntimeError, match="all .* roles are empty"):
        core.tri_router_supervision_loss(
            empty_capture,
            torch.empty(0, 1, 4, 4),
            torch.empty(0, 1, 4, 4),
        )


def test_training_capture_cleanup_no_hooks_no_extra_state_and_status_flags(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    reference = copy.deepcopy(frozen_candidate).train()
    candidate = copy.deepcopy(frozen_candidate).train()
    reference.mode = "train"
    candidate.mode = "train"
    inputs = torch.randn(
        1, 1, 32, 32, generator=torch.Generator().manual_seed(411)
    )
    hooks_before = _hook_cardinality(candidate)
    keys_before = tuple(candidate.state_dict())

    torch.manual_seed(994)
    reference(inputs)
    torch.manual_seed(994)
    with core.capture_c3_v32_training_router(candidate) as capture:
        outputs = candidate(inputs)
        with pytest.raises(RuntimeError, match="nested"):
            with core.capture_c3_v32_training_router(candidate):
                pass
    assert len(outputs) == 6
    assert len(capture.records) == 1
    record = capture.records[0]
    assert record["token_hw"] == (2, 2)
    assert len(record["logits"]) == len(record["supports"]) == 4
    compact = record["compact_diagnostics"]
    assert set(compact) == {
        "projection_rows",
        "solver_attempt_rows",
        "solver_accepted_rows",
        "solver_fallback_rows",
        "emission_checked_rows",
        "emission_accepted_rows",
        "emission_fallback_rows",
    }
    assert all(type(value) is int and value >= 0 for value in compact.values())
    assert compact["projection_rows"] > 0
    assert compact["solver_attempt_rows"] <= compact["projection_rows"]
    assert compact["solver_accepted_rows"] + compact[
        "solver_fallback_rows"
    ] == compact["solver_attempt_rows"]
    assert compact["emission_checked_rows"] == compact["projection_rows"]
    assert compact["emission_accepted_rows"] + compact[
        "emission_fallback_rows"
    ] == compact["emission_checked_rows"]
    assert tuple(candidate.state_dict()) == keys_before
    assert all(
        torch.equal(reference.state_dict()[key], candidate.state_dict()[key])
        for key in reference.state_dict()
    )
    assert _hook_cardinality(candidate) == hooks_before

    with pytest.raises(RuntimeError, match="boom"):
        with core.capture_c3_v32_training_router(candidate):
            raise RuntimeError("boom")
    with core.capture_c3_v32_training_router(candidate) as recovered:
        pass
    assert recovered.records == []
    candidate.eval()
    with pytest.raises(RuntimeError, match="requires training"):
        with core.capture_c3_v32_training_router(candidate):
            pass
    candidate.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="with gradients"):
        with core.capture_c3_v32_training_router(candidate):
            pass
    with core.capture_c3_v32_training_router(candidate):
        candidate.eval()
        with pytest.raises(RuntimeError, match="gradient-enabled training"):
            candidate(inputs)


def test_gain_zero_router_and_gain_gradients_then_optimizer_projection(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate).train()
    candidate.mode = "train"
    module = _module(candidate)
    inputs = torch.randn(
        1, 1, 32, 32, generator=torch.Generator().manual_seed(512)
    )
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 4:8, 4:8] = 1.0
    with core.capture_c3_v32_training_router(candidate) as capture:
        outputs = candidate(inputs)
    router_loss = core.tri_router_supervision_loss(
        capture, outputs[-1].detach(), target
    )
    router_gradients = torch.autograd.grad(
        router_loss,
        (
            module.tri_router.value_proj.weight,
            module.tri_router.head.weight,
        ),
        retain_graph=True,
    )
    for gradient in router_gradients:
        assert bool(torch.isfinite(gradient).all())
        assert torch.count_nonzero(gradient).item() > 0

    segmentation_loss = sum(
        F.binary_cross_entropy(output, target) for output in outputs
    )
    optimizer = torch.optim.SGD(
        [
            module.tri_router.value_proj.weight,
            module.tri_router.head.weight,
            module.raw_dual_risk_level_gain,
        ],
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    (segmentation_loss + router_loss).backward()
    assert module.raw_dual_risk_level_gain.grad is not None
    assert bool(torch.isfinite(module.raw_dual_risk_level_gain.grad).all())
    assert bool(torch.isfinite(module.tri_router.value_proj.weight.grad).all())
    assert bool(torch.isfinite(module.tri_router.head.weight.grad).all())
    before = tuple(
        parameter.detach().clone()
        for parameter in module.tri_router.parameters()
    )
    optimizer.step()
    core.project_sbsc_v32_constraints_(candidate)
    after = tuple(module.tri_router.parameters())
    assert any(not torch.equal(left, right) for left, right in zip(before, after))
    assert all(bool(torch.isfinite(parameter).all()) for parameter in after)
    assert bool((module.raw_dual_risk_level_gain >= 0.0).all())
    assert bool((module.raw_dual_risk_level_gain <= 0.25).all())
    core.validate_sctransnet_sbsc_v32(candidate)


def test_gain_projection_preserves_router_and_v31_solver_is_exactly_reused(
    paired_models,
):
    _baseline, frozen_candidate, _metadata = paired_models
    candidate = copy.deepcopy(frozen_candidate)
    module = _module(candidate)
    router_before = tuple(
        parameter.detach().clone()
        for parameter in module.tri_router.parameters()
    )
    with torch.no_grad():
        module.raw_dual_risk_level_gain.copy_(
            torch.tensor([-1.0, 0.1, 0.3, 2.0])
        )
    core.project_sbsc_v32_constraints_(candidate)
    assert torch.equal(
        module.raw_dual_risk_level_gain,
        torch.tensor([0.0, 0.1, 0.25, 0.25]),
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(router_before, module.tri_router.parameters())
    )
    assert core.solve_dual_risk_projection_v31 is (
        v31.solve_dual_risk_projection_v31
    )
    assert core.LearnedTriEvidenceProjectionV32._project_level is (
        v31.C3DualRiskProjectionV31._project_level
    )
    assert core.LearnedTriEvidenceProjectionV32._conditional_attention is (
        v31.C3DualRiskProjectionV31._conditional_attention
    )
    assert core.V31_SOLVER_SOURCE_SHA256 == core._sha256_file(
        core.Path(v31.__file__).resolve()
    )

    # Public V3.1 support-identity fixture through the exact alias; the frozen
    # reason 4/5/6/7 and wrapper/emission fixtures remain in the V3.1 suite.
    q0 = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    benefit = torch.tensor([[0.2, -0.2, 0.1, -0.1]])
    risk = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    inactive = torch.zeros(1, 1, dtype=torch.bool)
    projection = core.solve_dual_risk_projection_v31(
        q0,
        benefit,
        risk,
        -risk,
        torch.zeros(1, 1),
        hard_active=inactive,
        background_active=inactive,
    )
    assert projection.support_identity.item()
    assert projection.reason_code.item() == 1
    assert torch.equal(projection.qhat, q0)
