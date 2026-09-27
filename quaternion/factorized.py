"""Factorized quaternion weights (Hypothesis H8, experiments F1-F2).

The four quaternion weight components can themselves be low-rank.  Three
factorizations are supported:

``independent`` (F1)
    ``Wr = Ar Br``, ``Wi = Ai Bi``, ``Wj = Aj Bj``, ``Wk = Ak Bk``

``shared_input`` (F2)
    ``Wr = Ar B``, ``Wi = Ai B``, ``Wj = Aj B``, ``Wk = Ak B``
    (one shared right/input factor)

``shared_output`` (F2)
    ``Wr = A Br``, ``Wi = A Bi``, ``Wj = A Bj``, ``Wk = A Bk``
    (one shared left/output factor)

The forward pass reconstructs the four dense components and delegates to the
validated :func:`~quaternion.linear.quaternion_linear`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .linear import quaternion_linear
from .packing import check_divisible

__all__ = ["FactorizedQuaternionLinear", "parameter_count"]

_MODES = ("independent", "shared_input", "shared_output")


def parameter_count(
    in_features: int, out_features: int, rank: int, mode: str
) -> int:
    """Number of scalar parameters for a factorized quaternion weight matrix."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    out_g = out_features // 4
    in_g = in_features // 4
    if mode == "independent":
        return 4 * (out_g * rank + rank * in_g)
    if mode == "shared_input":
        return 4 * out_g * rank + rank * in_g
    # shared_output
    return out_g * rank + 4 * rank * in_g


def _uniform(shape, fan_in, **kwargs):
    t = torch.empty(*shape, **kwargs)
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    return nn.init.uniform_(t, -bound, bound)


class FactorizedQuaternionLinear(nn.Module):
    """Quaternion linear layer whose components are low-rank factorized."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        mode: str = "independent",
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        in_g = check_divisible(in_features, "in_features")
        out_g = check_divisible(out_features, "out_features")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.in_groups = in_g
        self.out_groups = out_g
        self.rank = rank
        self.mode = mode

        kw = {"device": device, "dtype": dtype}
        components = ("r", "i", "j", "k")
        if mode == "independent":
            for c in components:
                setattr(self, f"a_{c}", nn.Parameter(_uniform((out_g, rank), rank, **kw)))
                setattr(self, f"b_{c}", nn.Parameter(_uniform((rank, in_g), in_g, **kw)))
        elif mode == "shared_input":
            for c in components:
                setattr(self, f"a_{c}", nn.Parameter(_uniform((out_g, rank), rank, **kw)))
            self.b = nn.Parameter(_uniform((rank, in_g), in_g, **kw))
        else:  # shared_output
            self.a = nn.Parameter(_uniform((out_g, rank), rank, **kw))
            for c in components:
                setattr(self, f"b_{c}", nn.Parameter(_uniform((rank, in_g), in_g, **kw)))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kw))
            nn.init.uniform_(self.bias, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))
        else:
            self.register_parameter("bias", None)

    @property
    def compression_ratio(self) -> float:
        dense = self.out_features * self.in_features
        return dense / parameter_count(
            self.in_features, self.out_features, self.rank, self.mode
        )

    def reconstruct_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the dense quaternion components ``(wr, wi, wj, wk)``."""
        if self.mode == "independent":
            return tuple(
                getattr(self, f"a_{c}") @ getattr(self, f"b_{c}")
                for c in ("r", "i", "j", "k")
            )
        if self.mode == "shared_input":
            return tuple(getattr(self, f"a_{c}") @ self.b for c in ("r", "i", "j", "k"))
        return tuple(self.a @ getattr(self, f"b_{c}") for c in ("r", "i", "j", "k"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wr, wi, wj, wk = self.reconstruct_weights()
        y = quaternion_linear(x, wr, wi, wj, wk)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, mode={self.mode}, bias={self.bias is not None}"
        )
