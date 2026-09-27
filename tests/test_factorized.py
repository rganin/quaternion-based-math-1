"""Factorized quaternion weights (H8, F1/F2)."""

import pytest
import torch

from quaternion import FactorizedQuaternionLinear, parameter_count, quaternion_linear


@pytest.mark.parametrize("mode", ["independent", "shared_input", "shared_output"])
def test_forward_shape_and_equivalence(mode):
    layer = FactorizedQuaternionLinear(24, 16, rank=3, mode=mode)
    x = torch.randn(5, 24)
    y = layer(x)
    assert y.shape == (5, 16)
    wr, wi, wj, wk = layer.reconstruct_weights()
    assert torch.allclose(y, quaternion_linear(x, wr, wi, wj, wk))


@pytest.mark.parametrize("mode", ["independent", "shared_input", "shared_output"])
def test_reconstructed_component_shapes(mode):
    layer = FactorizedQuaternionLinear(24, 16, rank=3, mode=mode)
    for w in layer.reconstruct_weights():
        assert w.shape == (4, 6)


@pytest.mark.parametrize("mode", ["independent", "shared_input", "shared_output"])
def test_parameter_count_matches_module(mode):
    layer = FactorizedQuaternionLinear(64, 32, rank=8, mode=mode)
    expected = parameter_count(64, 32, 8, mode)
    actual = sum(p.numel() for p in layer.parameters())
    assert actual == expected


def test_shared_modes_use_fewer_parameters_than_independent():
    indep = parameter_count(256, 256, 16, "independent")
    shared_in = parameter_count(256, 256, 16, "shared_input")
    shared_out = parameter_count(256, 256, 16, "shared_output")
    assert shared_in < indep
    assert shared_out < indep


def test_compression_ratio_reported():
    layer = FactorizedQuaternionLinear(256, 256, rank=8, mode="independent")
    assert layer.compression_ratio > 1.0


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        FactorizedQuaternionLinear(8, 8, rank=2, mode="bogus")


def test_backward_populates_factor_grads():
    layer = FactorizedQuaternionLinear(24, 16, rank=3)
    layer(torch.randn(4, 24)).sum().backward()
    assert layer.a_r.grad is not None
    assert layer.b_r.grad is not None
