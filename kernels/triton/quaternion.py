"""Fused Triton quaternion linear kernel (Hypothesis H6, kernel experiment K3).

A single kernel computes all four output components.  Each program owns a
``[BLOCK_M, 4 * BLOCK_N]`` output tile (all four quaternion components for a
range of output groups) and loops over input groups once, reusing each loaded
activation and weight tile across the four equations.  Nothing is materialized
except the two accumulators' inputs; the sixteen real matrix products never
appear as separate tensors.

Weights are read in ``[in_groups, out_groups]`` layout (transposed from the
module's ``[out_groups, in_groups]``) so that ``tl.dot`` sees a
``[BLOCK_K, BLOCK_N]`` operand.  Pre-transposed ("packed") weights can be passed
to keep the transpose out of the timed region.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "triton_available",
    "pack_weights",
    "triton_quaternion_linear_packed",
    "triton_quaternion_linear",
]


def triton_available() -> bool:
    return torch.cuda.is_available()


@triton.jit
def _dot(a, b, IEEE: tl.constexpr):
    if IEEE:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@triton.jit
def _quaternion_linear_kernel(
    X, WR, WI, WJ, WK, Y,
    M, K_G, N_G,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_j = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_k = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    for k0 in range(0, K_G, BLOCK_K):
        kg = k0 + offs_k
        mask_k = kg < K_G

        # Activations: X[m, 4 * kg + c]
        mask_x = mask_m[:, None] & mask_k[None, :]
        x_base = X + offs_m[:, None] * stride_xm + (kg[None, :] * 4) * stride_xk
        xr = tl.load(x_base + 0 * stride_xk, mask=mask_x, other=0.0)
        xi = tl.load(x_base + 1 * stride_xk, mask=mask_x, other=0.0)
        xj = tl.load(x_base + 2 * stride_xk, mask=mask_x, other=0.0)
        xk = tl.load(x_base + 3 * stride_xk, mask=mask_x, other=0.0)

        # Weights: W[kg, n] in [K_G, N_G] layout
        mask_w = mask_k[:, None] & (offs_n[None, :] < N_G)
        w_off = kg[:, None] * stride_wk + offs_n[None, :] * stride_wn
        wr = tl.load(WR + w_off, mask=mask_w, other=0.0)
        wi = tl.load(WI + w_off, mask=mask_w, other=0.0)
        wj = tl.load(WJ + w_off, mask=mask_w, other=0.0)
        wk = tl.load(WK + w_off, mask=mask_w, other=0.0)

        acc_r += _dot(xr, wr, IEEE) - _dot(xi, wi, IEEE) - _dot(xj, wj, IEEE) - _dot(xk, wk, IEEE)
        acc_i += _dot(xr, wi, IEEE) + _dot(xi, wr, IEEE) - _dot(xj, wk, IEEE) + _dot(xk, wj, IEEE)
        acc_j += _dot(xr, wj, IEEE) + _dot(xi, wk, IEEE) + _dot(xj, wr, IEEE) - _dot(xk, wi, IEEE)
        acc_k += _dot(xr, wk, IEEE) - _dot(xi, wj, IEEE) + _dot(xj, wi, IEEE) + _dot(xk, wr, IEEE)

    out_ty = Y.dtype.element_ty
    mask_y = mask_m[:, None] & (offs_n[None, :] < N_G)
    y_base = Y + offs_m[:, None] * stride_ym + (offs_n[None, :] * 4) * stride_yn
    tl.store(y_base + 0 * stride_yn, acc_r.to(out_ty), mask=mask_y)
    tl.store(y_base + 1 * stride_yn, acc_i.to(out_ty), mask=mask_y)
    tl.store(y_base + 2 * stride_yn, acc_j.to(out_ty), mask=mask_y)
    tl.store(y_base + 3 * stride_yn, acc_k.to(out_ty), mask=mask_y)


def pack_weights(wr, wi, wj, wk):
    """Transpose ``[out_groups, in_groups]`` weights to contiguous
    ``[in_groups, out_groups]`` for the kernel."""
    return tuple(w.t().contiguous() for w in (wr, wi, wj, wk))


def _launch(x, wr, wi, wj, wk, block_m, block_n, block_k, num_warps, num_stages, ieee):
    lead = x.shape[:-1]
    in_features = x.shape[-1]
    if in_features % 4 != 0:
        raise ValueError("in_features must be divisible by 4")
    in_groups = in_features // 4
    if wr.shape[0] != in_groups:
        raise ValueError(
            f"packed weight in_groups ({wr.shape[0]}) must match input ({in_groups})"
        )
    out_groups = wr.shape[1]
    out_features = out_groups * 4

    x2 = x.reshape(-1, in_features)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    y = torch.empty((M, out_features), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(out_groups, block_n))
    _quaternion_linear_kernel[grid](
        x2, wr, wi, wj, wk, y,
        M, in_groups, out_groups,
        x2.stride(0), x2.stride(1),
        wr.stride(0), wr.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        IEEE=ieee, num_warps=num_warps, num_stages=num_stages,
    )
    return y.reshape(*lead, out_features)


def triton_quaternion_linear_packed(
    x, wr, wi, wj, wk,
    block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2, ieee=False,
):
    """Run the kernel with pre-transposed ``[in_groups, out_groups]`` weights."""
    return _launch(
        x, wr, wi, wj, wk, block_m, block_n, block_k, num_warps, num_stages, ieee
    )


def triton_quaternion_linear(
    x, wr, wi, wj, wk,
    block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2, ieee=False,
):
    """Convenience wrapper accepting ``[out_groups, in_groups]`` weights."""
    packed = pack_weights(wr, wi, wj, wk)
    return _launch(
        x, *packed, block_m, block_n, block_k, num_warps, num_stages, ieee
    )
