"""Unit-quaternion / rotation-constrained weights (H9)."""

import pytest
import torch

from quaternion import UnitQuaternionLinear, normalize_quaternion, quaternion_linear


def test_normalize_produces_unit_norm():
    g = torch.Generator().manual_seed(0)
    comps = [torch.randn(4, 6, generator=g, dtype=torch.float64) for _ in range(4)]
    unit = normalize_quaternion(*comps)
    norm = torch.sqrt(sum(c * c for c in unit))
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-12)


@pytest.mark.parametrize("magnitude", ["per_output", "per_element"])
def test_reconstruct_norm_matches_magnitude(magnitude):
    layer = UnitQuaternionLinear(24, 16, magnitude=magnitude)
    wr, wi, wj, wk = layer.reconstruct_weights()
    norm = torch.sqrt(wr**2 + wi**2 + wj**2 + wk**2)
    expected = torch.exp(layer.log_magnitude).expand_as(norm)
    assert torch.allclose(norm, expected, atol=1e-6)


def test_forward_matches_reconstructed_weights():
    layer = UnitQuaternionLinear(24, 16)
    x = torch.randn(5, 24)
    wr, wi, wj, wk = layer.reconstruct_weights()
    assert torch.allclose(layer(x), quaternion_linear(x, wr, wi, wj, wk))


def test_invalid_magnitude_mode_raises():
    with pytest.raises(ValueError):
        UnitQuaternionLinear(8, 8, magnitude="bogus")


def test_gradients_flow_to_direction_and_magnitude():
    layer = UnitQuaternionLinear(24, 16)
    layer(torch.randn(3, 24)).sum().backward()
    for name in ("vr", "vi", "vj", "vk", "log_magnitude"):
        grad = getattr(layer, name).grad
        assert grad is not None
        assert torch.isfinite(grad).all()
