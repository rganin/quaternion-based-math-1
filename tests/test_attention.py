"""Quaternion attention variants (H7, A1-A3)."""

import pytest
import torch

from quaternion import (
    quaternion_component_attention,
    quaternion_conjugate,
    shared_score_attention,
    standard_attention,
)


def make_qkv(batch=2, heads=2, seq=4, groups=3, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    qkv = [
        torch.randn(batch, heads, seq, groups, 4, generator=g, dtype=dtype)
        for _ in range(3)
    ]
    return qkv


def test_conjugate_negates_vector_parts():
    q = torch.randn(2, 2, 4, 3, 4, dtype=torch.float64)
    qc = quaternion_conjugate(q)
    assert torch.allclose(qc[..., 0], q[..., 0])
    assert torch.allclose(qc[..., 1:], -q[..., 1:])


def test_standard_attention_shapes_and_rowsum():
    q = torch.randn(2, 2, 5, 8, dtype=torch.float64)
    k = torch.randn(2, 2, 5, 8, dtype=torch.float64)
    v = torch.randn(2, 2, 5, 8, dtype=torch.float64)
    out, weights = standard_attention(q, k, v)
    assert out.shape == (2, 2, 5, 8)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 2, 5, dtype=torch.float64))


def test_component_attention_shapes_and_rowsum():
    q, k, v = make_qkv()
    out, weights = quaternion_component_attention(q, k, v)
    assert out.shape == q.shape
    assert weights.shape == (2, 2, 4, 4, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 2, 4, 4, dtype=torch.float64))


def test_shared_score_attention_shapes_and_rowsum():
    q, k, v = make_qkv()
    out, weights = shared_score_attention(q, k, v)
    assert out.shape == q.shape
    assert weights.shape == (2, 2, 4, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 2, 4, dtype=torch.float64))


def test_shared_score_equals_real_attention_on_flattened_features():
    q, k, v = make_qkv()
    shared_out, shared_w = shared_score_attention(q, k, v)

    dim = q.shape[-2] * 4
    real_out, real_w = standard_attention(
        q.reshape(*q.shape[:-2], dim),
        k.reshape(*k.shape[:-2], dim),
        v.reshape(*v.shape[:-2], dim),
    )
    assert torch.allclose(shared_w, real_w, atol=1e-12)
    assert torch.allclose(shared_out.reshape(*q.shape[:-2], dim), real_out, atol=1e-12)


def test_causal_mask_blocks_future_positions():
    q, k, v = make_qkv(seq=4)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    _, weights = shared_score_attention(q, k, v, mask=mask)
    assert torch.all(weights[..., 0, 1:] == 0)
    assert torch.all(weights[..., 1, 2:] == 0)


def test_attention_rejects_bad_shape():
    q = torch.randn(2, 2, 4, 3)  # missing component axis
    with pytest.raises(ValueError):
        shared_score_attention(q, q, q)
