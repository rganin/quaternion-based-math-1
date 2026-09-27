"""Planar quaternion linear operators (Hypothesis H6, kernel experiments K1/K2).

Two deliberately distinct implementations of the same operation:

``quaternion_linear_naive``
    Split ``X`` into its four components, compute the sixteen real matrix
    products as separate operations, combine with signs and concatenate.  This
    is the K1 baseline.

``quaternion_linear_planar``
    Form the sixteen products in a *single* batched contraction that reuses the
    packed activation and weight tensors, then combine them with a sparse
    coefficient tensor.  This is the K2 "planar" formulation: the Hamilton
    matrix is never materialized, and the input/weight tiles are shared.

Neither function is assumed to be faster than cuBLAS; the point is to have
correct, comparable references for the hardware benchmark (plan.md sections
35, 42-45).
"""

from __future__ import annotations

import torch

from .packing import (
    check_divisible,
    pack_components,
    pack_real_to_quaternion,
    split_components,
)

__all__ = [
    "quaternion_linear_naive",
    "quaternion_linear_planar",
    "planar_coefficient_tensor",
]


def quaternion_linear_naive(
    x: torch.Tensor,
    wr: torch.Tensor,
    wi: torch.Tensor,
    wj: torch.Tensor,
    wk: torch.Tensor,
) -> torch.Tensor:
    """K1 baseline: explicit split, sixteen products, concatenate."""
    in_groups = check_divisible(x.shape[-1], "in_features")
    out_groups = wr.shape[0]
    if wr.shape[1] != in_groups:
        raise ValueError("weight in_groups must match input in_groups")

    xq = pack_real_to_quaternion(x)
    xr, xi, xj, xk = split_components(xq)

    yr = xr @ wr.t() - xi @ wi.t() - xj @ wj.t() - xk @ wk.t()
    yi = xr @ wi.t() + xi @ wr.t() - xj @ wk.t() + xk @ wj.t()
    yj = xr @ wj.t() + xi @ wk.t() + xj @ wr.t() - xk @ wi.t()
    yk = xr @ wk.t() - xi @ wj.t() + xj @ wi.t() + xk @ wr.t()

    yq = pack_components(yr, yi, yj, yk)
    return yq.reshape(*yq.shape[:-2], out_groups * 4)


def planar_coefficient_tensor(
    device=None, dtype=None
) -> torch.Tensor:
    """Sparse coefficient tensor ``C[a, q, p]`` for the planar contraction.

    With ``P[n, q, p, o] = sum_i Xc[n, q, i] * Wc[p, o, i]`` the output is

        Y[n, a, o] = sum_{q, p} C[a, q, p] * P[n, q, p, o]

    where ``a`` indexes the output component, ``q`` the input component and
    ``p`` the weight component.
    """
    c = torch.zeros(4, 4, 4, device=device, dtype=dtype)
    # Yr = XrWr - XiWi - XjWj - XkWk
    c[0, 0, 0] = 1
    c[0, 1, 1] = -1
    c[0, 2, 2] = -1
    c[0, 3, 3] = -1
    # Yi = XrWi + XiWr - XjWk + XkWj
    c[1, 0, 1] = 1
    c[1, 1, 0] = 1
    c[1, 2, 3] = -1
    c[1, 3, 2] = 1
    # Yj = XrWj + XiWk + XjWr - XkWi
    c[2, 0, 2] = 1
    c[2, 1, 3] = 1
    c[2, 2, 0] = 1
    c[2, 3, 1] = -1
    # Yk = XrWk - XiWj + XjWi + XkWr
    c[3, 0, 3] = 1
    c[3, 1, 2] = -1
    c[3, 2, 1] = 1
    c[3, 3, 0] = 1
    return c


def quaternion_linear_planar(
    x: torch.Tensor,
    wr: torch.Tensor,
    wi: torch.Tensor,
    wj: torch.Tensor,
    wk: torch.Tensor,
    coefficient: torch.Tensor | None = None,
) -> torch.Tensor:
    """K2 planar formulation: one batched contraction for all 16 products."""
    in_groups = check_divisible(x.shape[-1], "in_features")
    out_groups = wr.shape[0]
    if wr.shape[1] != in_groups:
        raise ValueError("weight in_groups must match input in_groups")

    lead = x.shape[:-1]
    x_flat = x.reshape(-1, in_groups, 4).permute(0, 2, 1).contiguous()
    w_packed = pack_components(wr, wi, wj, wk)  # [out_g, in_g, 4]
    w_flat = w_packed.permute(2, 0, 1).contiguous()  # [4, out_g, in_g]

    # P[n, q, p, o] = sum_i X[n, q, i] * W[p, o, i]
    products = torch.einsum("nqi,poi->nqpo", x_flat, w_flat)

    if coefficient is None:
        coefficient = planar_coefficient_tensor(x.device, x.dtype)
    else:
        coefficient = coefficient.to(device=x.device, dtype=x.dtype)

    y = torch.einsum("aqp,nqpo->nao", coefficient, products)  # [N, 4, out_g]
    y = y.permute(0, 2, 1).reshape(*lead, out_groups * 4)
    return y
