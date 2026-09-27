"""Planar vs naive quaternion linear equivalence (H6, K1/K2)."""

import pytest
import torch

from quaternion import (
    quaternion_linear,
    quaternion_linear_naive,
    quaternion_linear_planar,
    quaternion_linear_reference,
)
from quaternion.planar import planar_coefficient_tensor


def weights(og, ig, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(og, ig, generator=g, dtype=torch.float64) / ig**0.5
        for _ in range(4)
    ]


def test_coefficient_tensor_shape_and_sparsity():
    c = planar_coefficient_tensor()
    assert c.shape == (4, 4, 4)
    assert (c != 0).sum().item() == 16


@pytest.mark.parametrize("shape", [(6, 8), (2, 5, 8), (3, 4, 7, 12)])
def test_all_implementations_agree(shape):
    in_features, out_features = shape[-1], shape[-1]
    w = weights(out_features // 4, in_features // 4)
    x = torch.randn(*shape, dtype=torch.float64)
    ref = quaternion_linear_reference(x, *w)
    lin = quaternion_linear(x, *w)
    naive = quaternion_linear_naive(x, *w)
    planar = quaternion_linear_planar(x, *w)
    for got in (lin, naive, planar):
        assert torch.allclose(got, ref, atol=1e-9, rtol=1e-9)


def test_planar_matches_block_matrix():
    import torch.nn.functional as F

    from quaternion import build_block_matrix

    og, ig = 3, 4
    w = weights(og, ig)
    block = build_block_matrix(*w)
    x = torch.randn(5, ig * 4, dtype=torch.float64)
    assert torch.allclose(
        quaternion_linear_planar(x, *w), F.linear(x, block), atol=1e-9, rtol=1e-9
    )


def test_planar_shapes_rectangular():
    w = weights(2, 3)  # out_features=8, in_features=12
    x = torch.randn(7, 12, dtype=torch.float64)
    assert quaternion_linear_planar(x, *w).shape == (7, 8)
    assert quaternion_linear_naive(x, *w).shape == (7, 8)


def test_feature_dim_mismatch_raises():
    w = weights(2, 3)
    x = torch.randn(4, 8)
    with pytest.raises(ValueError):
        quaternion_linear_planar(x, *w)
    with pytest.raises(ValueError):
        quaternion_linear_naive(x, *w)
