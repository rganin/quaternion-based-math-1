"""Efficient quaternion linear layer.

A conventional real linear layer ``y = x @ W.T`` with ``W`` of shape
``[out, in]`` is replaced by four real weight tensors ``Wr, Wi, Wj, Wk`` of
shape ``[out // 4, in // 4]``.  The transformation is the quaternion matrix
product

    y[o] = sum_i Wq[o, i] (x) Xq[i]

where ``(x)`` is the Hamilton product.  Writing the four real output
components, the operation is four coupled real matrix multiplications:

    Yr = Xr@Wr.T - Xi@Wi.T - Xj@Wj.T - Xk@Wk.T
    Yi = Xr@Wi.T + Xi@Wr.T - Xj@Wk.T + Xk@Wj.T
    Yj = Xr@Wj.T + Xi@Wk.T + Xj@Wr.T - Xk@Wi.T
    Yk = Xr@Wk.T - Xi@Wj.T + Xj@Wi.T + Xk@Wr.T

The parameter count is ``out * in / 4``: a 4x reduction for the same logical
dimensions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import (
    build_block_matrix,
    check_divisible,
    pack_components,
    pack_real_to_quaternion,
    split_components,
)

__all__ = ["quaternion_linear", "QuaternionLinear"]


def quaternion_linear(
    x: torch.Tensor,
    wr: torch.Tensor,
    wi: torch.Tensor,
    wj: torch.Tensor,
    wk: torch.Tensor,
) -> torch.Tensor:
    """Quaternion linear transform (no bias).

    ``x``: ``[..., in_features]``.  Weights: ``[out_groups, in_groups]``.
    Returns ``[..., out_features]``.
    """
    in_groups = check_divisible(x.shape[-1], "in_features")
    out_groups = wr.shape[0]
    if wr.shape[1] != in_groups:
        raise ValueError(
            f"weight in_groups ({wr.shape[1]}) must match input in_groups "
            f"({in_groups})"
        )
    if not (wr.shape == wi.shape == wj.shape == wk.shape):
        raise ValueError("all four weight components must share the same shape")

    xq = pack_real_to_quaternion(x)
    xr, xi, xj, xk = split_components(xq)

    yr = xr @ wr.t() - xi @ wi.t() - xj @ wj.t() - xk @ wk.t()
    yi = xr @ wi.t() + xi @ wr.t() - xj @ wk.t() + xk @ wj.t()
    yj = xr @ wj.t() + xi @ wk.t() + xj @ wr.t() - xk @ wi.t()
    yk = xr @ wk.t() - xi @ wj.t() + xj @ wi.t() + xk @ wr.t()

    yq = pack_components(yr, yi, yj, yk)  # [..., out_groups, 4]
    return yq.reshape(*yq.shape[:-2], out_groups * 4)


class QuaternionLinear(nn.Module):
    """Drop-in replacement for a real :class:`torch.nn.Linear`.

    The weight is stored as four real tensors ``Wr, Wi, Wj, Wk`` of shape
    ``[out_features // 4, in_features // 4]``.

    ``in_features`` and ``out_features`` must each be divisible by four.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        in_groups = check_divisible(in_features, "in_features")
        out_groups = check_divisible(out_features, "out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.in_groups = in_groups
        self.out_groups = out_groups

        factory_kwargs = {"device": device, "dtype": dtype}
        self.wr = nn.Parameter(torch.empty(out_groups, in_groups, **factory_kwargs))
        self.wi = nn.Parameter(torch.empty(out_groups, in_groups, **factory_kwargs))
        self.wj = nn.Parameter(torch.empty(out_groups, in_groups, **factory_kwargs))
        self.wk = nn.Parameter(torch.empty(out_groups, in_groups, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match the variance of the equivalent real block matrix of size
        # [out, in] so that a quaternion layer initializes like a real linear.
        bound = 1.0 / math.sqrt(self.in_features)
        for w in (self.wr, self.wi, self.wj, self.wk):
            nn.init.uniform_(w, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = quaternion_linear(x, self.wr, self.wi, self.wj, self.wk)
        if self.bias is not None:
            y = y + self.bias
        return y

    def real_block_matrix(self) -> torch.Tensor:
        """Return the equivalent real ``[out_features, in_features]`` matrix.

        This is the left-multiplication block matrix.  It is *not* used by
        :meth:`forward`; it exists for verification and reference.
        """
        return build_block_matrix(self.wr, self.wi, self.wj, self.wk)

    @classmethod
    def from_real_block_matrix(
        cls, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> "QuaternionLinear":
        """Extract quaternion parameters from a real block-structured matrix.

        The input is expected to already have the quaternion block structure
        (e.g. produced by :func:`real_block_matrix`), so this is an exact
        inverse, not an approximation.
        """
        if weight.dim() != 2:
            raise ValueError("weight must be 2D [out_features, in_features]")
        out_features, in_features = weight.shape
        check_divisible(out_features, "out_features")
        check_divisible(in_features, "in_features")
        layer = cls(in_features, out_features, bias=bias is not None)
        with torch.no_grad():
            layer.wr.copy_(weight[0::4, 0::4])
            layer.wi.copy_(weight[1::4, 0::4])
            layer.wj.copy_(weight[2::4, 0::4])
            layer.wk.copy_(weight[3::4, 0::4])
            if bias is not None:
                layer.bias.copy_(bias)
        return layer

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
