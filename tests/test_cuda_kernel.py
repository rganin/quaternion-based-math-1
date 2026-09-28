"""Custom fused CUDA kernel correctness (K4).

Skipped when CUDA is unavailable or the extension fails to build.
"""

import pytest
import torch

from quaternion import quaternion_linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="custom CUDA kernel requires CUDA"
)


def _get_kernel():
    from kernels.cuda.loader import quaternion_linear_cuda

    return quaternion_linear_cuda


def _weights(og, ig, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(og, ig, generator=g) / ig**0.5).to("cuda", dtype)
        for _ in range(4)
    ]


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)])
def test_matches_reference(dtype, tol):
    try:
        fn = _get_kernel()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"CUDA extension unavailable: {exc}")

    og, ig = 8, 6
    w = _weights(og, ig, dtype)
    x = torch.randn(37, ig * 4, device="cuda", dtype=dtype)
    ref = quaternion_linear(x.float(), *[t.float() for t in w])
    got = fn(x, *w)
    assert got.shape == ref.shape
    assert (got.float() - ref).abs().max().item() < tol


def test_batched_input():
    try:
        fn = _get_kernel()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"CUDA extension unavailable: {exc}")

    w = _weights(4, 4, torch.float32)
    x = torch.randn(2, 5, 16, device="cuda")
    assert fn(x, *w).shape == (2, 5, 16)
