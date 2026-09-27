"""Fused Triton quaternion kernels (plan.md sections 14, 44)."""

from .quaternion import (
    pack_weights,
    triton_available,
    triton_quaternion_linear,
    triton_quaternion_linear_packed,
)

__all__ = [
    "triton_available",
    "pack_weights",
    "triton_quaternion_linear",
    "triton_quaternion_linear_packed",
]
