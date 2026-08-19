from __future__ import annotations

from types import MethodType

import pytest
import torch

from experiments import irstd_cp_hf_s2_v1 as cp_hf
from model.EviSIRST import initialize_evisirst


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _clean_and_variant():
    clean, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
    variant, _ = cp_hf.build_irstd_cp_hf_s2_v1(
        "IRSTD-1K", seed=42, training=True
    )
    return clean, variant


def test_module_identity_bound_detach_and_two_stage_gradient() -> None:
    torch.manual_seed(7)
    module = cp_hf.ContextPurifiedHighFrequencyS2(
        architecture_seed=42,
        initialization_seed=cp_hf.derive_cp_hf_s2_initialization_seed(42),
    )
    assert len(module.state_dict()) == cp_hf.EXTENSION_STATE_KEY_COUNT
    assert _parameter_count(module) == cp_hf.EXTENSION_PARAMETER_COUNT

    feature = torch.randn(2, 32, 16, 16, requires_grad=True)
    skip = torch.randn(2, 64, 16, 16, requires_grad=True)
    with pytest.raises(ValueError, match="spatial shapes differ"):
        module(feature, torch.randn(2, 64, 32, 32))
    identity = module(feature, skip)
    assert torch.equal(identity, feature)
    identity.mean().backward()
    assert module.raw_scale.grad is not None
    assert torch.count_nonzero(module.raw_scale.grad).item() == 1
    assert skip.grad is None
    expected_feature_grad = torch.full_like(feature, 1.0 / feature.numel())
    assert torch.equal(feature.grad, expected_feature_grad)
    for name, parameter in module.named_parameters():
        if name == "raw_scale":
            continue
        assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0

    with torch.no_grad():
        module.raw_scale.copy_(-1.0e-2 * module.raw_scale.grad)
    module.zero_grad(set_to_none=True)
    feature.grad = None
    output = module(feature, skip)
    components = module.refinement_components(feature, skip)
    assert not torch.equal(output, feature)
    assert (output - feature).abs().max().item() <= (
        cp_hf.MAX_FEATURE_DELTA + 1.0e-7
    )
    assert components["correction"].abs().max().item() <= (
        cp_hf.MAX_FEATURE_DELTA + 1.0e-7
    )
    output.square().mean().backward()
    assert skip.grad is None
    assert torch.equal(
        feature.grad,
        2.0 * output.detach() / output.numel(),
    )
    non_scale_nonzero = {
        name
        for name, parameter in module.named_parameters()
        if name != "raw_scale"
        and parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
    }
    for prefix in (
        "skip_projection.",
        "channel_gate.",
        "spatial_gate.",
        "residual_branch.",
    ):
        assert any(name.startswith(prefix) for name in non_scale_nonzero)


def test_full_model_initial_parity_state_and_manifest() -> None:
    clean, variant = _clean_and_variant()
    manifest = cp_hf.validate_irstd_cp_hf_s2_v1(
        variant, require_identity_initialization=True
    )
    assert len(clean.state_dict()) == cp_hf.BASE_STATE_KEY_COUNT
    assert len(variant.state_dict()) == cp_hf.FORMAL_STATE_KEY_COUNT
    assert _parameter_count(variant) == cp_hf.FORMAL_PARAMETER_COUNT
    for key, value in clean.state_dict().items():
        assert torch.equal(value, variant.state_dict()[key])
    assert manifest["affected_outputs"] == ["gt2", "d0", "out"]
    assert manifest["unchanged_outputs"] == ["gt5", "gt4", "gt3"]
    assert manifest["module_manifest"]["final_logit_bound_claimed"] is False

    generator = torch.Generator().manual_seed(123)
    image = torch.randn(1, 1, 256, 256, generator=generator)
    clean.eval()
    variant.eval()
    clean.mode = "train"
    variant.mode = "train"
    with torch.no_grad():
        frozen_outputs = clean(image)
        copied_outputs = cp_hf.baseline_reference_forward_with_relay(
            clean, image
        )
        variant_outputs = variant(image)
    assert len(frozen_outputs) == len(copied_outputs) == len(variant_outputs) == 6
    for frozen, copied, installed in zip(
        frozen_outputs, copied_outputs, variant_outputs
    ):
        assert torch.equal(frozen, copied)
        assert torch.equal(frozen, installed)


def test_full_model_first_step_base_gradient_parity_and_nonzero_effect() -> None:
    clean, variant = _clean_and_variant()
    clean.train()
    variant.train()
    clean.mode = "train"
    variant.mode = "train"
    image = torch.randn(1, 1, 256, 256, generator=torch.Generator().manual_seed(4))

    clean_outputs = clean(image)
    sum(output.mean() for output in clean_outputs).backward()
    clean_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in clean.named_parameters()
    }

    variant_outputs = variant(image)
    sum(output.mean() for output in variant_outputs).backward()
    for name, parameter in variant.named_parameters():
        if name.startswith(cp_hf.STATE_PREFIX):
            continue
        expected = clean_gradients[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, expected), name

    extension = getattr(variant, cp_hf.MODULE_NAME)
    assert extension.raw_scale.grad is not None
    assert torch.count_nonzero(extension.raw_scale.grad).item() == 1
    with torch.no_grad():
        extension.raw_scale.copy_(-1.0e-2 * extension.raw_scale.grad)
    variant.zero_grad(set_to_none=True)
    captured: dict[str, torch.Tensor] = {}

    def capture(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["before"] = inputs[0].detach()
        captured["after"] = output.detach()

    handle = extension.register_forward_hook(capture)
    with torch.no_grad():
        changed = variant(image)
    handle.remove()
    assert torch.isfinite(captured["after"]).all().item()
    assert (captured["after"] - captured["before"]).abs().max().item() <= (
        cp_hf.MAX_FEATURE_DELTA + 1.0e-7
    )
    for index in (0, 1, 2):
        assert torch.equal(changed[index], clean_outputs[index])
    for index in (3, 4, 5):
        assert not torch.equal(changed[index], clean_outputs[index])


def test_install_rng_isolation_reproducibility_and_distinct_substreams() -> None:
    base_a, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
    before = torch.random.get_rng_state().clone()
    model_a, _ = cp_hf.install_irstd_cp_hf_s2_v1(
        base_a, architecture_seed=42
    )
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)

    base_b, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
    model_b, _ = cp_hf.install_irstd_cp_hf_s2_v1(
        base_b, architecture_seed=42
    )
    extension_a = getattr(model_a, cp_hf.MODULE_NAME).state_dict()
    extension_b = getattr(model_b, cp_hf.MODULE_NAME).state_dict()
    for key in extension_a:
        assert torch.equal(extension_a[key], extension_b[key])

    base_c, _ = initialize_evisirst("IRSTD-1K", seed=42, training=True)
    model_c, _ = cp_hf.install_irstd_cp_hf_s2_v1(
        base_c, architecture_seed=43
    )
    extension_c = getattr(model_c, cp_hf.MODULE_NAME).state_dict()
    assert any(
        value.is_floating_point()
        and key != "raw_scale"
        and not torch.equal(value, extension_c[key])
        for key, value in extension_a.items()
    )
    with pytest.raises(RuntimeError, match="already installed"):
        cp_hf.install_irstd_cp_hf_s2_v1(model_a, architecture_seed=42)


def test_strict_state_roundtrip_and_binding_tamper_rejection() -> None:
    model, _ = cp_hf.build_irstd_cp_hf_s2_v1(seed=42, training=True)
    clone, _ = cp_hf.build_irstd_cp_hf_s2_v1(seed=42, training=True)
    clone.load_state_dict(model.state_dict(), strict=True)
    cp_hf.validate_irstd_cp_hf_s2_v1(clone)

    incomplete = dict(model.state_dict())
    incomplete.pop(f"{cp_hf.STATE_PREFIX}raw_scale")
    with pytest.raises(RuntimeError):
        clone.load_state_dict(incomplete, strict=True)

    extension = getattr(clone, cp_hf.MODULE_NAME)
    extension.max_feature_delta = 1.0
    with pytest.raises(RuntimeError, match="scalar/channel contract"):
        cp_hf.validate_irstd_cp_hf_s2_v1(clone)
    extension.max_feature_delta = cp_hf.MAX_FEATURE_DELTA
    with torch.no_grad():
        extension.low_pass_kernel[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="binomial kernel differs"):
        cp_hf.validate_irstd_cp_hf_s2_v1(clone)
    with torch.no_grad():
        extension.low_pass_kernel.copy_(
            torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])[:, None]
            * torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])[None, :]
            / 256.0
        )
    extension.channel_gate[2] = torch.nn.ReLU()
    with pytest.raises(RuntimeError, match="channel gate structure differs"):
        cp_hf.validate_irstd_cp_hf_s2_v1(clone)
    extension.channel_gate[2] = torch.nn.SiLU()
    extension.initialization_seed += 1
    with pytest.raises(RuntimeError, match="module seed attributes differ"):
        cp_hf.validate_irstd_cp_hf_s2_v1(clone)
    extension.initialization_seed -= 1
    extension.raw_scale.requires_grad_(False)
    with pytest.raises(RuntimeError, match="contains frozen parameters"):
        cp_hf.validate_irstd_cp_hf_s2_v1(clone)
    extension.raw_scale.requires_grad_(True)

    clone._forward_with_relay = MethodType(
        cp_hf.baseline_reference_forward_with_relay, clone
    )
    with pytest.raises(RuntimeError, match="relay binding differs"):
        cp_hf.validate_irstd_cp_hf_s2_v1(clone)


def test_eval_contract_is_one_finite_probability_map() -> None:
    model, _ = cp_hf.build_irstd_cp_hf_s2_v1(seed=42, training=False)
    clean, _ = initialize_evisirst("IRSTD-1K", seed=42, training=False)
    image = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        output = model(image)
        expected = clean(image)
    assert output.shape == image.shape
    assert torch.equal(output, expected)
    assert torch.isfinite(output).all().item()
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0
