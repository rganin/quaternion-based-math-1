"""Fused Triton kernel correctness (K3).

Skipped entirely when Triton or a CUDA device is unavailable.
"""

import pytest
import torch

from quaternion import quaternion_linear, quaternion_linear_reference

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused kernel requires CUDA"
)

from kernels.triton.quaternion import (  # noqa: E402
    pack_weights,
    triton_quaternion_linear,
    triton_quaternion_linear_packed,
)


def make_weights(og, ig, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        (torch.randn(og, ig, generator=g) / ig**0.5).to(device) for _ in range(4)
    ]


@pytest.mark.parametrize(
    "dtype,ieee,num_stages,tol",
    [
        (torch.float32, True, 1, 1e-5),
        (torch.bfloat16, False, 2, 5e-2),
        (torch.float16, False, 2, 2e-2),
    ],
)
def test_matches_reference(dtype, ieee, num_stages, tol):
    og, ig = 8, 6
    w = [t.to(dtype) for t in make_weights(og, ig)]
    x = torch.randn(37, ig * 4, device="cuda", dtype=dtype)

    ref = quaternion_linear(
        x.to(torch.float32), *(t.to(torch.float32) for t in w)
    )
    got = triton_quaternion_linear(
        x, *w, ieee=ieee, num_stages=num_stages, block_m=32, block_n=32
    )
    assert got.shape == ref.shape
    assert (got.to(torch.float32) - ref).abs().max().item() < tol


def test_batched_input_shapes():
    og, ig = 4, 4
    w = make_weights(og, ig)
    x = torch.randn(2, 5, ig * 4, device="cuda")
    got = triton_quaternion_linear(x, *w, ieee=True, num_stages=1)
    ref = quaternion_linear(x, *w)
    assert got.shape == (2, 5, og * 4)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_packed_matches_unpacked_wrapper():
    og, ig = 6, 5
    w = make_weights(og, ig)
    x = torch.randn(23, ig * 4, device="cuda")
    got = triton_quaternion_linear(x, *w, ieee=True, num_stages=1)
    packed = pack_weights(*w)
    got_packed = triton_quaternion_linear_packed(x, *packed, ieee=True, num_stages=1)
    assert torch.equal(got, got_packed)


def test_matches_mathematical_reference():
    og, ig = 4, 3
    w = make_weights(og, ig)
    x = torch.randn(11, ig * 4, device="cuda")
    got = triton_quaternion_linear(x, *w, ieee=True, num_stages=1)
    ref = quaternion_linear_reference(x, *w)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_bad_feature_dim_raises():
    w = make_weights(4, 3)
    x = torch.randn(5, 10, device="cuda")  # 10 not divisible by 4
    with pytest.raises(ValueError):
        triton_quaternion_linear(x, *w)
