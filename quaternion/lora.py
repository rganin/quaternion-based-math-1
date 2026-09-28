"""LoRA recovery on top of a frozen quaternion base (Stage 6 / Phase 6).

Architecture (plan.md section 18):

    Y = QuaternionLinear(X) + scale * ((X @ A.T) @ B.T)

with ``A: [rank, in_features]``, ``B: [out_features, rank]`` and
``scale = alpha / rank``.  The quaternion base is frozen; only ``A`` and ``B``
are trainable.  ``B`` is initialized to zero so the adapted model starts exactly
at the quaternion-only model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .linear import QuaternionLinear

__all__ = [
    "LoRAQuaternionLinear",
    "inject_lora",
    "lora_parameters",
    "mark_only_lora_trainable",
]


class LoRAQuaternionLinear(nn.Module):
    """Frozen quaternion linear layer plus a trainable low-rank correction."""

    def __init__(self, base: QuaternionLinear, rank: int = 8, alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else 2 * rank
        self.scale = self.alpha / rank

        ref = next(base.parameters())
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=ref.device, dtype=ref.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=ref.device, dtype=ref.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scale * lora

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, base=frozen"
        )


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
):
    """Wrap every :class:`QuaternionLinear` in ``model`` with LoRA.

    The model is modified in place; returns the module names that were wrapped.
    """
    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, QuaternionLinear)
    ]
    wrapped = []
    for name, mod in targets:
        wrapped_mod = LoRAQuaternionLinear(mod, rank=rank, alpha=alpha, dropout=dropout)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, wrapped_mod)
        wrapped.append(name)
    return wrapped


def lora_parameters(model: nn.Module):
    """Yield only the trainable LoRA parameters."""
    for name, p in model.named_parameters():
        if "lora_" in name:
            yield name, p


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything except LoRA parameters."""
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
