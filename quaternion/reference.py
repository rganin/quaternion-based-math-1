"""Slow but obviously-correct reference implementations.

These functions are the correctness oracle for Phase 0.  They use the explicit
Hamilton product directly and make no attempt at efficiency.  Production code
lives in :mod:`quaternion.linear`.

Hamilton product
----------------
For ``a = (ar, ai, aj, ak)`` and ``b = (br, bi, bj, bk)``:

    r = ar*br - ai*bi - aj*bj - ak*bk
    i = ar*bi + ai*br + aj*bk - ak*bj
    j = ar*bj - ai*bk + aj*br + ak*bi
    k = ar*bk + ai*bj - aj*bi + ak*br
"""

from __future__ import annotations

import torch

from .packing import (
    check_divisible,
    pack_components,
    pack_real_to_quaternion,
    split_components,
)

__all__ = ["hamilton_product", "quaternion_linear_reference"]


def hamilton_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions with last axis ordered (r, i, j, k).

    ``a`` and ``b`` are broadcastable on all axes except the last.
    """
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("last dimension of both inputs must be 4")
    ar, ai, aj, ak = a.unbind(-1)
    br, bi, bj, bk = b.unbind(-1)

    r = ar * br - ai * bi - aj * bj - ak * bk
    i = ar * bi + ai * br + aj * bk - ak * bj
    j = ar * bj - ai * bk + aj * br + ak * bi
    k = ar * bk + ai * bj - aj * bi + ak * br
    return torch.stack((r, i, j, k), dim=-1)


def quaternion_linear_reference(
    x: torch.Tensor,
    wr: torch.Tensor,
    wi: torch.Tensor,
    wj: torch.Tensor,
    wk: torch.Tensor,
) -> torch.Tensor:
    """Reference ``y = QuaternionLinear(x)``.

    ``x`` has shape ``[..., in_features]`` and the weights have shape
    ``[out_groups, in_groups]`` where ``out_groups = out_features // 4`` and
    ``in_groups = in_features // 4``.  The output has shape
    ``[..., out_features]``.

    Implementation: broadcast the explicit Hamilton product over every
    ``(out_group, in_group)`` pair and sum, with no algebraic shortcuts.
    """
    in_features = x.shape[-1]
    in_groups = check_divisible(in_features, "in_features")
    w_out_groups, w_in_groups = wr.shape
    if (wi.shape != wr.shape) or (wj.shape != wr.shape) or (wk.shape != wr.shape):
        raise ValueError("all four weight components must share the same shape")
    if w_in_groups != in_groups:
        raise ValueError(
            f"weight in_groups ({w_in_groups}) must match input in_groups "
            f"({in_groups})"
        )
    out_features = w_out_groups * 4

    xq = pack_real_to_quaternion(x)  # [..., in_groups, 4]
    wq = pack_components(wr, wi, wj, wk)  # [out_groups, in_groups, 4]

    # Align x as [..., 1, in_groups, 4] so the Hamilton product broadcasts
    # against wq [out_groups, in_groups, 4] over the in_groups axis.
    xq = xq.unsqueeze(-3)
    products = hamilton_product(wq, xq)  # [..., out_groups, in_groups, 4]
    yq = products.sum(dim=-2)  # [..., out_groups, 4]

    return yq.reshape(*yq.shape[:-2], out_features)
