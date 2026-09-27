"""Phase 0: packing, block-matrix and quaternion-linear equivalence tests."""

import pytest
import torch
import torch.nn.functional as F

from quaternion import (
    QuaternionLinear,
    build_block_matrix,
    pack_components,
    pack_real_to_quaternion,
    quaternion_linear,
    quaternion_linear_reference,
    split_components,
    unpack_quaternion_to_real,
)


def small_weights(out_groups, in_groups, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(out_groups, in_groups, generator=g, dtype=dtype) / in_groups**0.5
        for _ in range(4)
    ]


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (8,), (2, 12), (2, 3, 16), (1, 1, 4)])
def test_pack_roundtrip(shape):
    x = torch.randn(*shape)
    q = pack_real_to_quaternion(x)
    assert q.shape == (*shape[:-1], shape[-1] // 4, 4)
    assert torch.equal(unpack_quaternion_to_real(q), x)


def test_pack_channel_order_is_r_i_j_k():
    x = torch.arange(8, dtype=torch.float32)
    q = pack_real_to_quaternion(x)
    assert torch.equal(q[0], torch.tensor([0.0, 1, 2, 3]))
    assert torch.equal(q[1], torch.tensor([4.0, 5, 6, 7]))
    r, i, j, k = split_components(q)
    assert torch.equal(r, torch.tensor([0.0, 4]))
    assert torch.equal(i, torch.tensor([1.0, 5]))
    assert torch.equal(j, torch.tensor([2.0, 6]))
    assert torch.equal(k, torch.tensor([3.0, 7]))


def test_split_pack_components_roundtrip():
    q = torch.randn(5, 3, 4)
    r, i, j, k = split_components(q)
    assert torch.equal(pack_components(r, i, j, k), q)


@pytest.mark.parametrize("bad", [1, 2, 3, 5, 6, 7, 9])
def test_invalid_hidden_dim_raises(bad):
    with pytest.raises(ValueError):
        pack_real_to_quaternion(torch.randn(bad))


def test_invalid_linear_features_raise():
    with pytest.raises(ValueError):
        QuaternionLinear(in_features=10, out_features=8)
    with pytest.raises(ValueError):
        QuaternionLinear(in_features=8, out_features=10)


# --------------------------------------------------------------------------
# Equivalence between the three implementations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_features,in_features",
    [(4, 4), (8, 12), (12, 8), (16, 16)],
)
def test_efficient_matches_reference(out_features, in_features):
    wr, wi, wj, wk = small_weights(out_features // 4, in_features // 4)
    x = torch.randn(7, in_features)
    got = quaternion_linear(x, wr, wi, wj, wk)
    want = quaternion_linear_reference(x, wr, wi, wj, wk)
    assert got.shape == (7, out_features)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_efficient_matches_block_matrix():
    out_features, in_features = 16, 24
    wr, wi, wj, wk = small_weights(out_features // 4, in_features // 4)
    block = build_block_matrix(wr, wi, wj, wk)
    assert block.shape == (out_features, in_features)

    x = torch.randn(5, in_features)
    got = quaternion_linear(x, wr, wi, wj, wk)
    want = F.linear(x, block)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_reference_matches_block_matrix():
    out_features, in_features = 8, 12
    wr, wi, wj, wk = small_weights(out_features // 4, in_features // 4)
    block = build_block_matrix(wr, wi, wj, wk)
    x = torch.randn(4, in_features)
    got = quaternion_linear_reference(x, wr, wi, wj, wk)
    want = F.linear(x, block)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# Shapes, dtypes, module behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [(3,), (2, 5), (1, 7, 3)])
def test_batch_and_sequence_dims(extra):
    out_features, in_features = 12, 16
    wr, wi, wj, wk = small_weights(out_features // 4, in_features // 4)
    x = torch.randn(*extra, in_features)
    got = quaternion_linear(x, wr, wi, wj, wk)
    assert got.shape == (*extra, out_features)


@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 5e-2)],
)
def test_dtypes(dtype, tol):
    out_features, in_features = 16, 24
    wr, wi, wj, wk = small_weights(out_features // 4, in_features // 4)
    x = torch.randn(6, in_features)

    want64 = quaternion_linear_reference(
        x.to(torch.float64),
        *(w.to(torch.float64) for w in (wr, wi, wj, wk)),
    )
    got = quaternion_linear(
        x.to(dtype), *(w.to(dtype) for w in (wr, wi, wj, wk))
    )
    assert torch.allclose(got.to(torch.float64), want64, atol=tol, rtol=tol)


def test_module_forward_matches_functional():
    layer = QuaternionLinear(24, 16, bias=False)
    x = torch.randn(4, 3, 24)
    got = layer(x)
    want = quaternion_linear(x, layer.wr, layer.wi, layer.wj, layer.wk)
    assert torch.allclose(got, want)


def test_module_bias_is_added():
    layer = QuaternionLinear(24, 16, bias=True)
    x = torch.randn(2, 24)
    got = layer(x)
    want = quaternion_linear(x, layer.wr, layer.wi, layer.wj, layer.wk) + layer.bias
    assert torch.allclose(got, want)


def test_parameter_count_is_quarter():
    in_features, out_features = 64, 32
    layer = QuaternionLinear(in_features, out_features)
    n_params = sum(p.numel() for p in layer.parameters())
    assert n_params == out_features * in_features // 4


def test_from_real_block_matrix_roundtrip():
    out_features, in_features = 16, 24
    wr, wi, wj, wk = small_weights(out_features // 4, in_features // 4)
    block = build_block_matrix(wr, wi, wj, wk)

    layer = QuaternionLinear.from_real_block_matrix(block)
    assert torch.allclose(layer.wr, wr)
    assert torch.allclose(layer.wi, wi)
    assert torch.allclose(layer.wj, wj)
    assert torch.allclose(layer.wk, wk)

    x = torch.randn(3, in_features)
    assert torch.allclose(layer(x), F.linear(x, block), atol=1e-6, rtol=1e-6)
