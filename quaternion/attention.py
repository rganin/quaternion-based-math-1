"""Quaternion attention variants (Hypothesis H7, experiments A1-A3).

Three score computations are provided for comparison:

``standard_attention``
    Ordinary scaled dot-product attention on real features.  Baseline.

``quaternion_component_attention`` (A1)
    Q/K/V are quaternion-valued.  Each of the four quaternion components gets
    its own score tensor and its own softmax ("four independent score
    calculations").  This is the baseline the shared-score idea improves on.

``shared_score_attention`` (A2)
    A single score ``S = Re(Q ⊗ K†)`` (equivalently the real inner product over
    the four quaternion components) is computed once, softmaxed once, and used
    to mix the quaternion-valued V.

All functions take Q/K/V in packed quaternion layout
``[batch, heads, seq, groups, 4]`` with the last axis ordered ``(r, i, j, k)``,
except :func:`standard_attention` which takes real ``[batch, heads, seq, dim]``.
"""

from __future__ import annotations

import torch

__all__ = [
    "standard_attention",
    "quaternion_component_attention",
    "shared_score_attention",
    "quaternion_conjugate",
]


def _check_quaternion(x: torch.Tensor, name: str) -> None:
    if x.dim() != 5 or x.shape[-1] != 4:
        raise ValueError(
            f"{name} must have shape [batch, heads, seq, groups, 4], got {tuple(x.shape)}"
        )


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate ``q* = qr - qi i - qj j - qk k`` for packed quaternions."""
    _check_quaternion(q, "q")
    sign = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    return q * sign


def _apply_mask(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return scores
    while mask.dim() < scores.dim():
        mask = mask.unsqueeze(0)
    return scores.masked_fill(~mask, float("-inf"))


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary scaled dot-product attention.  Q/K/V: ``[B, H, T, D]``."""
    d = q.shape[-1]
    if scale is None:
        scale = 1.0 / (d**0.5)
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = _apply_mask(scores, mask)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


def _component_scores(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    """[B,H,T,G,4] q/k -> [B,H,4,T,G?]... per-component scores [B,H,4,T,T]."""
    # component c: q[...,c] over group axis -> [B,H,T,G]
    q_c = q.permute(0, 1, 4, 2, 3)  # [B,H,4,T,G]
    k_c = k.permute(0, 1, 4, 2, 3)
    return q_c @ k_c.transpose(-1, -2) * scale  # [B,H,4,T,T]


def quaternion_component_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A1: four independent component-wise attention score tensors."""
    _check_quaternion(q, "q")
    _check_quaternion(k, "k")
    _check_quaternion(v, "v")
    groups = q.shape[-2]
    if scale is None:
        scale = 1.0 / (groups**0.5)

    scores = _component_scores(q, k, scale)  # [B,H,4,T,T]
    scores = _apply_mask(scores, mask)
    weights = torch.softmax(scores, dim=-1)  # [B,H,4,T,T]

    # v_c: [B,H,4,T,G]; weights [B,H,4,T,T] -> out [B,H,4,T,G]
    v_c = v.permute(0, 1, 4, 2, 3)
    out_c = weights @ v_c
    out = out_c.permute(0, 1, 3, 4, 2)  # [B,H,T,G,4]
    return out, weights


def shared_score_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A2: one shared real score ``S = Re(Q ⊗ K†)``, one softmax."""
    _check_quaternion(q, "q")
    _check_quaternion(k, "k")
    _check_quaternion(v, "v")
    groups = q.shape[-2]
    dim = groups * 4
    if scale is None:
        scale = 1.0 / (dim**0.5)

    q_flat = q.reshape(*q.shape[:-2], dim)
    k_flat = k.reshape(*k.shape[:-2], dim)
    scores = (q_flat @ k_flat.transpose(-1, -2)) * scale  # [B,H,T,T]
    scores = _apply_mask(scores, mask)
    weights = torch.softmax(scores, dim=-1)  # [B,H,T,T]

    v_flat = v.reshape(*v.shape[:-2], dim)
    out_flat = weights @ v_flat
    out = out_flat.reshape(*v.shape)
    return out, weights
