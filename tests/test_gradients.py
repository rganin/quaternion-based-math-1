"""Phase 0: autograd and finite-difference gradient tests."""

import pytest
import torch

from quaternion import QuaternionLinear, quaternion_linear, quaternion_linear_reference


def _weights(out_groups, in_groups, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(out_groups, in_groups, generator=g, dtype=dtype)
        for _ in range(4)
    )


def test_gradcheck_float64():
    out_features, in_features = 8, 8
    wr, wi, wj, wk = _weights(out_features // 4, in_features // 4, torch.float64)
    x = torch.randn(3, in_features, dtype=torch.float64)

    for t in (wr, wi, wj, wk, x):
        t.requires_grad_(True)

    def fn(x, wr, wi, wj, wk):
        return quaternion_linear(x, wr, wi, wj, wk)

    assert torch.autograd.gradcheck(
        fn, (x, wr, wi, wj, wk), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_finite_difference_single_entry():
    out_features, in_features = 8, 8
    wr, wi, wj, wk = _weights(out_features // 4, in_features // 4, torch.float64)
    x = torch.randn(2, in_features, dtype=torch.float64)

    wr = wr.clone().requires_grad_(True)

    def loss(wr_):
        y = quaternion_linear(x, wr_, wi, wj, wk)
        return (y**2).sum()

    loss(wr).backward()
    analytic = wr.grad[1, 1].item()

    eps = 1e-6
    with torch.no_grad():
        orig = wr[1, 1].item()
        wr[1, 1] = orig + eps
        plus = loss(wr).item()
        wr[1, 1] = orig - eps
        minus = loss(wr).item()
        wr[1, 1] = orig
    numeric = (plus - minus) / (2 * eps)

    assert abs(analytic - numeric) < 1e-5


def test_module_backward_populates_all_weight_grads():
    layer = QuaternionLinear(12, 8, bias=True)
    x = torch.randn(5, 3, 12, requires_grad=True)
    out = layer(x)
    out.sum().backward()

    for name in ("wr", "wi", "wj", "wk"):
        grad = getattr(layer, name).grad
        assert grad is not None
        assert grad.shape == getattr(layer, name).shape
    assert layer.bias.grad is not None
    assert x.grad is not None


def test_efficient_and_reference_gradients_agree():
    out_features, in_features = 8, 12
    base = _weights(out_features // 4, in_features // 4, torch.float64)
    x = torch.randn(4, in_features, dtype=torch.float64)

    eff = tuple(w.clone().requires_grad_(True) for w in base)
    ref = tuple(w.clone().requires_grad_(True) for w in base)

    quaternion_linear(x, *eff).pow(2).sum().backward()
    quaternion_linear_reference(x, *ref).pow(2).sum().backward()

    for a, b in zip(eff, ref):
        assert torch.allclose(a.grad, b.grad, atol=1e-9, rtol=1e-9)
