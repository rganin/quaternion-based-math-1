"""Real <-> quaternion channel packing conventions.

Packing convention (explicit and normatively tested)
----------------------------------------------------
A real vector of length ``D`` with ``D % 4 == 0`` is split into ``D // 4``
contiguous groups of four consecutive channels.  Each group is interpreted as
the quaternion

    q = q_r + q_i * i + q_j * j + q_k * k

where the four contiguous channels are ordered ``[q_r, q_i, q_j, q_k]``.

For a tensor ``x`` of shape ``[..., D]`` packing yields ``[..., D // 4, 4]``.
The final axis is *always* ordered ``(r, i, j, k)``.

The corresponding real block matrix is the left-multiplication representation

    [ Wr  -Wi  -Wj  -Wk ]
    [ Wi   Wr  -Wk   Wj ]
    [ Wj   Wk   Wr  -Wi ]
    [ Wk  -Wj   Wi   Wr ]

placed, for every output group ``o`` and input group ``i``, at rows
``4o .. 4o+3`` and columns ``4i .. 4i+3``.  With the contiguous packing above
this block matrix is exactly equivalent to the quaternion linear operator.
"""

from __future__ import annotations

import torch

__all__ = [
    "QUATERNION_GROUPS",
    "check_divisible",
    "pack_real_to_quaternion",
    "unpack_quaternion_to_real",
    "split_components",
    "pack_components",
    "build_block_matrix",
]

QUATERNION_GROUPS = 4


def check_divisible(dim: int, name: str = "dimension") -> int:
    """Validate that ``dim`` is a multiple of four and return ``dim // 4``."""
    if dim % QUATERNION_GROUPS != 0:
        raise ValueError(
            f"{name} must be divisible by {QUATERNION_GROUPS} for quaternion "
            f"packing, got {dim}"
        )
    return dim // QUATERNION_GROUPS


def pack_real_to_quaternion(x: torch.Tensor) -> torch.Tensor:
    """[..., D] -> [..., D // 4, 4] with last axis ordered (r, i, j, k)."""
    check_divisible(x.shape[-1], "last dimension")
    return x.reshape(*x.shape[:-1], x.shape[-1] // QUATERNION_GROUPS, QUATERNION_GROUPS)


def unpack_quaternion_to_real(q: torch.Tensor) -> torch.Tensor:
    """[..., G, 4] -> [..., G * 4]. Inverse of :func:`pack_real_to_quaternion`."""
    if q.shape[-1] != QUATERNION_GROUPS:
        raise ValueError(
            f"last dimension must be {QUATERNION_GROUPS}, got {q.shape[-1]}"
        )
    return q.reshape(*q.shape[:-2], q.shape[-2] * QUATERNION_GROUPS)


def split_components(q: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """[..., G, 4] -> tuple (qr, qi, qj, qk), each [..., G]."""
    if q.shape[-1] != QUATERNION_GROUPS:
        raise ValueError(
            f"last dimension must be {QUATERNION_GROUPS}, got {q.shape[-1]}"
        )
    return q.unbind(-1)


def pack_components(
    qr: torch.Tensor,
    qi: torch.Tensor,
    qj: torch.Tensor,
    qk: torch.Tensor,
) -> torch.Tensor:
    """(qr, qi, qj, qk) each [..., G] -> [..., G, 4] ordered (r, i, j, k)."""
    return torch.stack((qr, qi, qj, qk), dim=-1)


def build_block_matrix(
    wr: torch.Tensor,
    wi: torch.Tensor,
    wj: torch.Tensor,
    wk: torch.Tensor,
) -> torch.Tensor:
    """Build the real block matrix of a quaternion weight matrix.

    ``wr/wi/wj/wk`` have shape ``[out_groups, in_groups]``.  The result has
    shape ``[4 * out_groups, 4 * in_groups]`` with the per-group 4x4 block

        [ Wr  -Wi  -Wj  -Wk ]
        [ Wi   Wr  -Wk   Wj ]
        [ Wj   Wk   Wr  -Wi ]
        [ Wk  -Wj   Wi   Wr ]

    This real matrix destroys the parameter/memory advantage of the quaternion
    representation and exists only for verification and reference purposes.
    """
    for w, name in ((wr, "wr"), (wi, "wi"), (wj, "wj"), (wk, "wk")):
        if w.dim() != 2:
            raise ValueError(f"{name} must be 2D [out_groups, in_groups]")
    if not (wr.shape == wi.shape == wj.shape == wk.shape):
        raise ValueError("all four weight components must share the same shape")

    rows = [
        torch.stack((wr, -wi, -wj, -wk), dim=-1),
        torch.stack((wi, wr, -wk, wj), dim=-1),
        torch.stack((wj, wk, wr, -wi), dim=-1),
        torch.stack((wk, -wj, wi, wr), dim=-1),
    ]
    # [out_groups, in_groups, 4 (row within block), 4 (col within block)]
    block = torch.stack(rows, dim=-2)
    out_groups, in_groups = wr.shape
    # Interleave groups so flat index g*4 + component indexes each group.
    block = block.permute(0, 2, 1, 3).reshape(
        out_groups * QUATERNION_GROUPS, in_groups * QUATERNION_GROUPS
    )
    return block
