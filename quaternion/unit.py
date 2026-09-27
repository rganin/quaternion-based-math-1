"""Unit-quaternion / rotation-constrained weights (Hypothesis H9).

Lower-priority experiment: constrain each quaternion weight to a pure rotation
(unit norm) and carry the magnitude separately:

    Wq = scale * q_unit,    ||q_unit|| = 1

The magnitude may be shared per output group (``per_output``) or kept per
quaternion element (``per_element``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .linear import quaternion_linear
from .packing import check_divisible

__all__ = ["normalize_quaternion", "UnitQuaternionLinear"]


def normalize_quaternion(
    wr: torch.Tensor,
    wi: torch.Tensor,
    wj: torch.Tensor,
    wk: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project four real components onto the unit sphere element-wise."""
    norm = torch.sqrt(wr * wr + wi * wi + wj * wj + wk * wk).clamp_min(eps)
    return wr / norm, wi / norm, wj / norm, wk / norm


class UnitQuaternionLinear(nn.Module):
    """Quaternion linear layer with rotation-constrained unit weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        magnitude: str = "per_output",
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if magnitude not in ("per_output", "per_element"):
            raise ValueError("magnitude must be 'per_output' or 'per_element'")
        in_g = check_divisible(in_features, "in_features")
        out_g = check_divisible(out_features, "out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.in_groups = in_g
        self.out_groups = out_g
        self.magnitude = magnitude

        kw = {"device": device, "dtype": dtype}
        bound = 1.0 / math.sqrt(in_features)
        for name in ("vr", "vi", "vj", "vk"):
            setattr(self, name, nn.Parameter(torch.empty(out_g, in_g, **kw)))
            nn.init.uniform_(getattr(self, name), -bound, bound)

        if magnitude == "per_output":
            self.log_magnitude = nn.Parameter(torch.zeros(out_g, 1, **kw))
        else:
            self.log_magnitude = nn.Parameter(torch.zeros(out_g, in_g, **kw))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kw))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def reconstruct_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unit = normalize_quaternion(self.vr, self.vi, self.vj, self.vk)
        scale = torch.exp(self.log_magnitude)
        return tuple(c * scale for c in unit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wr, wi, wj, wk = self.reconstruct_weights()
        y = quaternion_linear(x, wr, wi, wj, wk)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"magnitude={self.magnitude}, bias={self.bias is not None}"
        )
