from __future__ import annotations

import copy
import gc
import hashlib
from collections import OrderedDict
from types import MethodType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import irstd_cp_hf_s2_dcspg_v1 as dcspg
from experiments import irstd_cp_hf_s2_v1 as cp_hf
from model.EviSIRST import initialize_evisirst


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_guard_formula_boundary_gradient_bound_detach_and_safe_fallback() -> None:
    guard = dcspg.DilatedCoarseSupportPrecisionGuard()
    assert tuple(guard.state_dict()) == ("raw_strength",)
    assert _parameter_count(guard) == 1
    assert torch.equal(guard.strength.detach(), torch.zeros(()))

    out_logit = torch.linspace(-2.0, 3.0, 143).reshape(1, 1, 11, 13)
    out_logit.requires_grad_()
    d0_logit = torch.full_like(out_logit, -1.5, requires_grad=True)
    components = guard.refinement_components(out_logit, d0_logit)
    expected_support = F.max_pool2d(
        torch.sigmoid(d0_logit.detach()),
        kernel_size=dcspg.SUPPORT_KERNEL_SIZE,
        stride=1,
        padding=dcspg.SUPPORT_KERNEL_SIZE // 2,
    )
    expected_unsupported = torch.sigmoid(out_logit.detach()) * (
        1.0 - expected_support
    )
    assert torch.equal(components["dilated_d0_support"], expected_support)
    assert torch.equal(
        components["coarse_unsupported_evidence"], 1.0 - expected_support
    )
    assert torch.equal(components["unsupported_out"], expected_unsupported)
    assert not components["unsupported_out"].requires_grad

    identity = guard(out_logit, d0_logit)
    assert torch.equal(identity, out_logit)
    identity.sum().backward()
    assert guard.raw_strength.grad is not None
    expected_boundary_gradient = (
        -0.5 * dcspg.MAX_LOGIT_DELTA * expected_unsupported.sum()
    )
    assert torch.allclose(
        guard.raw_strength.grad,
        expected_boundary_gradient,
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.count_nonzero(guard.raw_strength.grad).item() == 1
    assert torch.equal(out_logit.grad, torch.ones_like(out_logit))
    assert d0_logit.grad is None

    # The actual probability-space BCE used by the training engine also
    # opens the boundary gate on its first step for unsupported background.
    guard.zero_grad(set_to_none=True)
    out_logit.grad = None
    bce_probability = torch.sigmoid(guard(out_logit, d0_logit))
    F.binary_cross_entropy(
        bce_probability, torch.zeros_like(bce_probability)
    ).backward()
    assert guard.raw_strength.grad is not None
    assert float(guard.raw_strength.grad) < 0.0
    assert torch.count_nonzero(guard.raw_strength.grad).item() == 1
    assert d0_logit.grad is None

    # A step in the negative raw-parameter direction cannot promote pixels;
    # it returns exactly to the identity and its inactive clamp is stable.
    with torch.no_grad():
        guard.raw_strength.fill_(-0.25)
    guard.zero_grad(set_to_none=True)
    out_logit.grad = None
    negative_fallback = guard(out_logit, d0_logit)
    assert torch.equal(guard.strength.detach(), torch.zeros(()))
    assert torch.equal(negative_fallback, out_logit)
    negative_fallback.sum().backward()
    assert torch.equal(guard.raw_strength.grad, torch.zeros(()))
    assert torch.equal(out_logit.grad, torch.ones_like(out_logit))
    assert d0_logit.grad is None

    with torch.no_grad():
        guard.raw_strength.fill_(20.0)
    guard.zero_grad(set_to_none=True)
    out_logit.grad = None
    saturated = guard.refinement_components(out_logit, d0_logit)
    correction = saturated["correction"]
    assert torch.equal(
        guard.strength.detach(), torch.tensor(dcspg.MAX_LOGIT_DELTA)
    )
    assert bool((correction <= 0.0).all())
    assert float(correction.detach().min()) >= -dcspg.MAX_LOGIT_DELTA
    assert float(correction.detach().max()) <= 0.0
    changed = guard(out_logit, d0_logit)
    assert bool((changed <= out_logit).all())
    assert bool((changed < out_logit).any())
    changed.sum().backward()
    assert torch.equal(out_logit.grad, torch.ones_like(out_logit))
    assert d0_logit.grad is None


def test_guard_configuration_input_and_semantic_contracts_are_strict() -> None:
    with pytest.raises(ValueError, match="support_kernel_size"):
        dcspg.DilatedCoarseSupportPrecisionGuard(support_kernel_size=7)
    with pytest.raises(ValueError, match="max_logit_delta"):
        dcspg.DilatedCoarseSupportPrecisionGuard(max_logit_delta=2.0)
    guard = dcspg.DilatedCoarseSupportPrecisionGuard()
    with pytest.raises(ValueError, match="shapes differ"):
        guard(torch.zeros(1, 1, 9, 9), torch.zeros(1, 1, 8, 9))
    with pytest.raises(ValueError, match="one channel"):
        guard(torch.zeros(1, 2, 9, 9), torch.zeros(1, 2, 9, 9))
    with pytest.raises(TypeError, match="floating point"):
        guard(
            torch.zeros(1, 1, 9, 9, dtype=torch.int64),
            torch.zeros(1, 1, 9, 9, dtype=torch.int64),
        )
    nonfinite = torch.zeros(1, 1, 9, 9)
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        guard(nonfinite, torch.zeros_like(nonfinite))
    manifest = guard.architecture_manifest()
    assert manifest["ground_truth_far_background_claimed"] is False
    assert manifest["coarse_unsupported_evidence"] == (
        "one_minus_dilated_d0_support"
    )
    assert manifest["support_kernel_size"] == 9
    assert manifest["inference_requires_raw_d0"] is True


def test_full_model_identity_head_semantics_and_deployment_path() -> None:
    clean, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
    cp_model, _ = cp_hf.build_irstd_cp_hf_s2_v1(
        "IRSTD-1K", seed=42, training=True
    )
    model, metadata = dcspg.build_irstd_cp_hf_s2_dcspg_v1(
        "IRSTD-1K", seed=42, training=True
    )
    manifest = dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )
    assert len(model.state_dict()) == dcspg.FORMAL_STATE_KEY_COUNT == 577
    assert _parameter_count(model) == dcspg.FORMAL_PARAMETER_COUNT == 10_874_616
    assert metadata["architecture_schema"] == dcspg.ARCHITECTURE_SCHEMA
    assert metadata["name"] == dcspg.EXPERIMENT_NAME
    assert metadata["architecture_seed"] == metadata["seed"] == 42
    assert metadata["baseline_checkpoint_loaded"] is False
    assert metadata["warm_start_used"] is False
    assert manifest["training_head_5"] == "raw_d0_probability"
    assert manifest["training_head_6"] == "guarded_out_probability"
    assert manifest["evaluation_head"] == "guarded_out_probability"
    assert manifest["inference_early_out_bypass_allowed"] is False
    assert manifest["inference_requires_raw_d0"] is True
    assert set(manifest["raw_d0_required_modules"]) == {
        "gt_conv2",
        "gt_conv3",
        "gt_conv4",
        "gt_conv5",
        "outconv",
    }
    for key, value in clean.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    for key, value in cp_model.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key

    for candidate in (clean, cp_model, model):
        candidate.eval()
        candidate.mode = "train"
    image = torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(99))
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    guard = getattr(model, dcspg.GUARD_MODULE_NAME)

    def capture_inputs(
        _module: nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        captured.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

    handle = guard.register_forward_pre_hook(capture_inputs)
    with torch.inference_mode():
        clean_outputs = clean(image)
        cp_outputs = cp_model(image)
        identity_outputs = model(image)
    handle.remove()
    assert len(clean_outputs) == len(cp_outputs) == len(identity_outputs) == 6
    for baseline, cp_value, variant in zip(
        clean_outputs, cp_outputs, identity_outputs
    ):
        assert torch.equal(baseline, cp_value)
        assert torch.equal(baseline, variant)
    assert len(captured) == 1
    raw_out, raw_d0 = captured[0]
    assert torch.equal(identity_outputs[4], torch.sigmoid(raw_d0))
    assert torch.equal(identity_outputs[5], torch.sigmoid(raw_out))

    with torch.no_grad():
        guard.raw_strength.fill_(0.5)
    captured.clear()
    handle = guard.register_forward_pre_hook(capture_inputs)
    with torch.inference_mode():
        guarded_training_outputs = model(image)
    handle.remove()
    assert len(guarded_training_outputs) == 6
    for index in range(5):
        assert torch.equal(guarded_training_outputs[index], cp_outputs[index])
    assert torch.equal(guarded_training_outputs[4], torch.sigmoid(captured[0][1]))
    assert bool((guarded_training_outputs[5] <= cp_outputs[5]).all())
    assert bool((guarded_training_outputs[5] < cp_outputs[5]).any())

    deployed, deploy_metadata = dcspg.build_irstd_cp_hf_s2_dcspg_v1(
        "IRSTD-1K", architecture_seed=42, training=False
    )
    deployed.load_state_dict(model.state_dict(), strict=True)
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(deployed)
    assert deployed.training is False
    assert deployed.mode == "test"
    assert deploy_metadata["training_mode"] is False
    deployment_capture: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_deployment(
        _module: nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        deployment_capture.append((inputs[0].detach(), inputs[1].detach()))

    deploy_guard = getattr(deployed, dcspg.GUARD_MODULE_NAME)
    handle = deploy_guard.register_forward_pre_hook(capture_deployment)
    with torch.inference_mode():
        deployed_output = deployed(image)
    handle.remove()
    assert len(deployment_capture) == 1
    assert deployed_output.shape == (1, 1, 32, 32)
    assert torch.isfinite(deployed_output).all().item()
    assert float(deployed_output.min()) >= 0.0
    assert float(deployed_output.max()) <= 1.0
    assert torch.equal(deployed_output, guarded_training_outputs[5])

    del clean, cp_model, model, deployed
    gc.collect()


def test_full_model_initial_outputs_and_non_guard_gradients_match_cp_hf_s2() -> None:
    cp_model, _ = cp_hf.build_irstd_cp_hf_s2_v1(seed=42, training=True)
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)
    for candidate in (cp_model, model):
        candidate.eval()
        candidate.mode = "train"
    image = torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(17))

    cp_outputs = cp_model(image)
    cp_loss = sum(probability.mean() for probability in cp_outputs)
    cp_loss.backward()
    cp_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in cp_model.named_parameters()
    }
    detached_cp_outputs = tuple(value.detach().clone() for value in cp_outputs)
    del cp_outputs, cp_loss

    outputs = model(image)
    for expected, observed in zip(detached_cp_outputs, outputs):
        assert torch.equal(expected, observed)
    sum(probability.mean() for probability in outputs).backward()
    for name, parameter in model.named_parameters():
        if name.startswith(dcspg.GUARD_STATE_PREFIX):
            continue
        expected = cp_gradients[name]
        if expected is None:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert torch.equal(parameter.grad, expected), name
    guard = getattr(model, dcspg.GUARD_MODULE_NAME)
    assert guard.raw_strength.grad is not None
    assert torch.count_nonzero(guard.raw_strength.grad).item() == 1
    cp_extension = getattr(model, dcspg.CP_MODULE_NAME)
    assert cp_extension.raw_scale.grad is not None
    assert torch.count_nonzero(cp_extension.raw_scale.grad).item() == 1

    del cp_model, model, outputs
    gc.collect()


def test_strict_state_roundtrip_and_tamper_rejection() -> None:
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)
    state = OrderedDict(
        (key, value.detach().clone()) for key, value in model.state_dict().items()
    )
    model.load_state_dict(state, strict=True)
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )
    missing = OrderedDict(state)
    missing.pop(dcspg.GUARD_STATE_KEYS[0])
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(missing, strict=True)

    guard = getattr(model, dcspg.GUARD_MODULE_NAME)
    original_guard_forward = guard.forward
    guard.forward = MethodType(  # type: ignore[method-assign]
        lambda self, out_logit, d0_logit: out_logit,
        guard,
    )
    with pytest.raises(RuntimeError, match="instance forward shadow"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    del guard.forward
    assert getattr(guard.forward, "__func__", None) is getattr(
        original_guard_forward, "__func__", None
    )

    hook = model.outconv.register_forward_hook(
        lambda _module, _inputs, output: output
    )
    with pytest.raises(RuntimeError, match="runtime hook"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    hook.remove()

    guard.support_kernel_size = 7
    with pytest.raises(RuntimeError, match="fixed configuration"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    guard.support_kernel_size = dcspg.SUPPORT_KERNEL_SIZE

    with torch.no_grad():
        guard.raw_strength.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    with torch.no_grad():
        guard.raw_strength.zero_()

    with torch.no_grad():
        guard.raw_strength.fill_(0.1)
    with pytest.raises(RuntimeError, match="identity initialized"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
            model, require_identity_initialization=True
        )
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    with torch.no_grad():
        guard.raw_strength.zero_()

    guard.raw_strength.requires_grad_(False)
    with pytest.raises(RuntimeError, match="frozen parameters"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    guard.raw_strength.requires_grad_(True)

    extension = getattr(model, dcspg.CP_MODULE_NAME)
    extension.channel_gate[2] = nn.ReLU()
    with pytest.raises(RuntimeError, match="channel gate structure differs"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    extension.channel_gate[2] = nn.SiLU()

    original_eps = model.tpd_qfg.eps
    model.tpd_qfg.eps = 2.0e-6
    with pytest.raises(ValueError, match="eps differs"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    model.tpd_qfg.eps = original_eps

    original_dc_support = model.tpd_ner._dc_support_mode
    original_manifest = copy.deepcopy(
        getattr(model, "_irstd_cp_hf_s2_dcspg_base_manifest")
    )
    original_cp_manifest = copy.deepcopy(
        getattr(model, "_irstd_cp_hf_s2_base_manifest")
    )
    model.tpd_ner._dc_support_mode = type(original_dc_support).LEGACY_GLOBAL
    synchronized_tamper = copy.deepcopy(model.architecture_manifest())
    setattr(
        model,
        "_irstd_cp_hf_s2_dcspg_base_manifest",
        copy.deepcopy(synchronized_tamper),
    )
    setattr(
        model,
        "_irstd_cp_hf_s2_base_manifest",
        copy.deepcopy(synchronized_tamper),
    )
    with pytest.raises(RuntimeError, match="complement-tail"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    model.tpd_ner._dc_support_mode = original_dc_support
    setattr(model, "_irstd_cp_hf_s2_dcspg_base_manifest", original_manifest)
    setattr(model, "_irstd_cp_hf_s2_base_manifest", original_cp_manifest)

    old_seed = getattr(model, "_irstd_cp_hf_s2_initialization_seed")
    setattr(model, "_irstd_cp_hf_s2_initialization_seed", old_seed + 1)
    with pytest.raises(RuntimeError, match="seed binding"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    setattr(model, "_irstd_cp_hf_s2_initialization_seed", old_seed)

    model.outconv.in_channels = 4
    with pytest.raises(RuntimeError, match="raw d0 fusion head"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    model.outconv.in_channels = 5

    model._forward_with_relay = MethodType(cp_hf.cp_hf_s2_forward_with_relay, model)
    with pytest.raises(RuntimeError, match="relay binding differs"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    model._forward_with_relay = MethodType(
        dcspg.cp_hf_s2_dcspg_forward_with_relay, model
    )

    model.mode = "invalid"
    with pytest.raises(RuntimeError, match="model.mode"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    model.mode = "test"
    with pytest.raises(RuntimeError, match="model.training"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    model.eval()
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )
    model.train()
    model.mode = "train"
    guard.eval()
    with pytest.raises(RuntimeError, match="child training state"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    guard.train()
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )


def test_every_qfg_level_nonstate_and_internal_contract_is_sealed() -> None:
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)
    qfg = model.tpd_qfg

    def reject_attribute(
        target: object, name: str, replacement: object
    ) -> None:
        original = getattr(target, name)
        setattr(target, name, replacement)
        try:
            with pytest.raises(RuntimeError, match="QFG"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            setattr(target, name, original)

    for index, level in enumerate(tuple(qfg.levels)):
        reject_attribute(level, "feature_channels", level.feature_channels + 1)
        reject_attribute(level, "mode", "high")
        reject_attribute(level, "hidden_channels", level.hidden_channels + 1)
        reject_attribute(level, "expected_alignment", (99, 99))
        reject_attribute(level, "detach_frequency_source", False)
        reject_attribute(level, "alpha_effective_init", 0.2)
        reject_attribute(level, "eps", 2.0e-6)
        reject_attribute(level, "validate_finite", False)

        original_level = qfg.levels[index]
        qfg.levels[index] = nn.Identity()
        try:
            with pytest.raises(RuntimeError, match="QFG level type"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            qfg.levels[index] = original_level

        original_haar = level.haar
        level.haar = nn.Identity()
        try:
            with pytest.raises(RuntimeError, match="QFG Haar"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            level.haar = original_haar
        with torch.no_grad():
            level.haar.kernels[0, 0, 0, 0].add_(1.0)
        try:
            with pytest.raises(RuntimeError, match="Haar kernels"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            with torch.no_grad():
                level.haar.kernels[0, 0, 0, 0].sub_(1.0)

        original_prior = level.prior_projection
        level.prior_projection = nn.Identity()
        try:
            with pytest.raises(RuntimeError, match="QFG internal module type"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            level.prior_projection = original_prior

        original_spatial_activation = level.spatial_projection[1]
        level.spatial_projection[1] = nn.ReLU()
        try:
            with pytest.raises(RuntimeError, match="spatial projection"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            level.spatial_projection[1] = original_spatial_activation

        original_gate_out = level.gate_out
        level.gate_out = nn.Identity()
        try:
            with pytest.raises(RuntimeError, match="QFG internal module type"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            level.gate_out = original_gate_out

    original_owner_token = qfg._prepared_owner_token
    qfg._prepared_owner_token = qfg.levels[0]._prepared_owner_token
    try:
        with pytest.raises(RuntimeError, match="owner tokens are not distinct"):
            dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    finally:
        qfg._prepared_owner_token = original_owner_token
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )


def test_all_direct_forward_helper_instance_shadows_are_rejected() -> None:
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)
    helpers: list[tuple[object, str]] = [
        (model, "explicit_embeddings"),
        (model.tpd_qfg, "prepare"),
        (model.tpd_qfg, "apply_prepared"),
        (model.tpd_qfg, "_normalize_query_sizes"),
        (model.tpd_ner, "forward_stage"),
        (model.tpd_ner, "dc_support"),
        (model.tpd_ner, "_persistent_tail_support"),
        (model.tpd_ner, "_tail_support"),
    ]
    for level in model.tpd_qfg.levels:
        helpers.extend(
            (
                (level, "prepare"),
                (level, "apply_prepared"),
                (level, "_align_prior"),
            )
        )
    for decoder_name in ("up_decoder4", "up_decoder3", "up_decoder2"):
        decoder = getattr(model, decoder_name)
        helpers.extend(((decoder, "prepare"), (decoder, "finish")))

    for target, name in helpers:
        setattr(
            target,
            name,
            MethodType(lambda self, *args, **kwargs: None, target),
        )
        try:
            with pytest.raises(RuntimeError, match="instance method shadow"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            delattr(target, name)

    original_bridge = dcspg.frequency_encoder_forward
    dcspg.frequency_encoder_forward = lambda *args, **kwargs: None
    try:
        with pytest.raises(RuntimeError, match="frequency encoder helper"):
            dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    finally:
        dcspg.frequency_encoder_forward = original_bridge
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )


def test_pytorch_global_runtime_hook_registries_are_rejected() -> None:
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)
    for index, attribute in enumerate(dcspg._GLOBAL_RUNTIME_HOOK_ATTRIBUTES):
        registry = getattr(dcspg.torch_module_hooks, attribute)
        key = -(10_000 + index)
        assert key not in registry
        registry[key] = lambda *args, **kwargs: None
        try:
            with pytest.raises(RuntimeError, match="global runtime hook"):
                dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
        finally:
            registry.pop(key, None)
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )


def test_canonical_base_state_key_digest_rejects_synchronized_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)
    batch_norm = dict(model.named_modules())["inc.0.bn1"]
    original_buffers = OrderedDict(batch_norm._buffers)
    original_new_snapshot = getattr(
        model, "_irstd_cp_hf_s2_dcspg_base_state_keys"
    )
    original_cp_snapshot = getattr(model, "_irstd_cp_hf_s2_base_state_keys")
    batch_norm._buffers = OrderedDict(
        (
            "attacked_running_mean" if name == "running_mean" else name,
            value,
        )
        for name, value in original_buffers.items()
    )
    tampered_keys = tuple(
        key
        for key in model.state_dict()
        if not key.startswith((dcspg.CP_STATE_PREFIX, dcspg.GUARD_STATE_PREFIX))
    )
    setattr(model, "_irstd_cp_hf_s2_dcspg_base_state_keys", tampered_keys)
    setattr(model, "_irstd_cp_hf_s2_base_state_keys", tampered_keys)
    monkeypatch.setattr(
        dcspg,
        "CANONICAL_BASE_STATE_KEYS_SHA256",
        dcspg._canonical_sha256(tampered_keys),
    )
    try:
        with pytest.raises(RuntimeError, match="canonical state-key sequence"):
            dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    finally:
        batch_norm._buffers = original_buffers
        setattr(
            model,
            "_irstd_cp_hf_s2_dcspg_base_state_keys",
            original_new_snapshot,
        )
        setattr(model, "_irstd_cp_hf_s2_base_state_keys", original_cp_snapshot)
    dcspg.validate_irstd_cp_hf_s2_dcspg_v1(
        model, require_identity_initialization=True
    )


def test_install_rng_isolation_reproducibility_aliases_and_rejections() -> None:
    base_a, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
    caller_rng = torch.random.get_rng_state().clone()
    model_a, metadata_a = dcspg.install_irstd_cp_hf_s2_dcspg_v1(
        base_a, architecture_seed=42
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    base_b, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
    model_b, _ = dcspg.install_irstd_cp_hf_s2_dcspg_v1(
        base_b, architecture_seed=42
    )
    for key, value in model_a.state_dict().items():
        if key.startswith((dcspg.CP_STATE_PREFIX, dcspg.GUARD_STATE_PREFIX)):
            assert torch.equal(value, model_b.state_dict()[key]), key
    assert metadata_a["source_dependency_paths"] == list(
        dcspg.SOURCE_DEPENDENCIES
    )
    assert metadata_a["source_dependencies"] == dcspg.SOURCE_DEPENDENCY_SHA256
    assert metadata_a["baseline_checkpoint_loaded"] is False
    with pytest.raises(RuntimeError, match="clean unextended"):
        dcspg.install_irstd_cp_hf_s2_dcspg_v1(model_a, architecture_seed=42)
    with pytest.raises(ValueError, match="agree"):
        dcspg.build_irstd_cp_hf_s2_dcspg_v1(
            seed=42, architecture_seed=43
        )
    with pytest.raises(ValueError, match="supports only"):
        dcspg.build_irstd_cp_hf_s2_dcspg_v1("NUDT-SIRST", seed=42)
    with pytest.raises(TypeError, match="training"):
        dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=1)  # type: ignore[arg-type]


def test_class_forward_identity_is_sealed(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)

    def replacement(
        self: dcspg.DilatedCoarseSupportPrecisionGuard,
        out_logit: torch.Tensor,
        d0_logit: torch.Tensor,
    ) -> torch.Tensor:
        return out_logit

    monkeypatch.setattr(
        dcspg.DilatedCoarseSupportPrecisionGuard,
        "forward",
        replacement,
    )
    with pytest.raises(RuntimeError, match="class method identity"):
        dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)


def test_source_dependency_hash_and_nofollow_are_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model, _ = dcspg.build_irstd_cp_hf_s2_dcspg_v1(seed=42, training=True)
    relative = dcspg.SOURCE_DEPENDENCIES[0]
    expected = dcspg.SOURCE_DEPENDENCY_SHA256[relative]
    canonical_root = dcspg.PROJECT_ROOT

    # A synchronized rewrite of every public discovery field must not redirect
    # the validator away from its definition-time canonical root/path/digest.
    attacker_root = tmp_path / "attacker"
    attacker_relative = "experiments/irstd_psbfr_v1.py"
    attacker_file = attacker_root / attacker_relative
    attacker_file.parent.mkdir(parents=True)
    attacker_file.write_text("# attacker-controlled replacement\n")
    attacker_digest = hashlib.sha256(attacker_file.read_bytes()).hexdigest()
    monkeypatch.setattr(dcspg, "PROJECT_ROOT", attacker_root)
    monkeypatch.setattr(dcspg, "SOURCE_DEPENDENCIES", (attacker_relative,))
    monkeypatch.setattr(
        dcspg,
        "SOURCE_DEPENDENCY_SHA256",
        {attacker_relative: attacker_digest},
    )
    manifest = dcspg.validate_irstd_cp_hf_s2_dcspg_v1(model)
    assert manifest["source_dependency_paths"] == [relative]
    assert manifest["source_dependencies"] == {relative: expected}

    wrong_root = tmp_path / "wrong_regular"
    wrong_file = wrong_root / relative
    wrong_file.parent.mkdir(parents=True)
    wrong_file.write_text("# wrong digest\n")
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        dcspg._validate_source_dependencies(
            _project_root=wrong_root,
            _dependency_items=((relative, expected),),
        )

    link_root = tmp_path / "link_root"
    experiments = link_root / "experiments"
    experiments.mkdir(parents=True)
    link = experiments / "irstd_cp_hf_s2_v1.py"
    link.symlink_to(canonical_root / relative)
    with pytest.raises(RuntimeError, match="symlink"):
        dcspg._validate_source_dependencies(
            _project_root=link_root,
            _dependency_items=((relative, expected),),
        )
