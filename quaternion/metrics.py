"""Reconstruction / approximation metrics (plan.md sections 4.2, 5.2).

All functions accept tensors of any shape and reduce over every element, and
compute in float64 for stable reporting.  They return plain Python floats so
results are directly JSON-serializable.
"""

from __future__ import annotations

import torch

__all__ = [
    "mse",
    "rmse",
    "relative_frobenius_error",
    "relative_l2_error",
    "cosine_similarity",
    "max_abs_error",
    "reconstruction_report",
]


def _pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return a.double(), b.double()


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _pair(a, b)
    return torch.mean((x - y) ** 2).item()


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return mse(a, b) ** 0.5


def relative_frobenius_error(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _pair(a, b)
    denom = torch.linalg.norm(y)
    if denom == 0:
        return torch.linalg.norm(x).item()
    return (torch.linalg.norm(x - y) / denom).item()


def relative_l2_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error, identical to the Frobenius error for flattened data."""
    return relative_frobenius_error(a, b)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _pair(a, b)
    return torch.nn.functional.cosine_similarity(
        x.reshape(1, -1), y.reshape(1, -1), dim=-1
    ).item()


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _pair(a, b)
    return (x - y).abs().max().item()


def reconstruction_report(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Return the standard metric bundle for ``a ≈ b``."""
    return {
        "mse": mse(a, b),
        "rmse": rmse(a, b),
        "relative_frobenius_error": relative_frobenius_error(a, b),
        "cosine_similarity": cosine_similarity(a, b),
        "max_abs_error": max_abs_error(a, b),
    }
