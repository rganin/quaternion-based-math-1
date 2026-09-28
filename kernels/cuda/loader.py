"""JIT loader for the custom CUDA quaternion kernel."""

from __future__ import annotations

import os

import torch

_EXT = None
_BUILD_FAILED = False


def cuda_kernel_available() -> bool:
    return torch.cuda.is_available() and not _BUILD_FAILED


def _load():
    global _EXT, _BUILD_FAILED
    if _EXT is not None:
        return _EXT
    if _BUILD_FAILED:
        raise RuntimeError("CUDA extension previously failed to build")
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "quaternion_linear.cu")
    try:
        _EXT = load(
            name="quat_cuda",
            sources=[src],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        _BUILD_FAILED = True
        raise RuntimeError(f"failed to build CUDA extension: {exc}") from exc
    return _EXT


def quaternion_linear_cuda(x, wr, wi, wj, wk):
    """Fused quaternion linear on CUDA.

    ``wr`` etc. may be in ``[out_groups, in_groups]`` (module) layout; they are
    transposed to the kernel's packed ``[in_groups, out_groups]`` layout.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    ext = _load()
    packed = [w.t().contiguous() for w in (wr, wi, wj, wk)]
    return ext.quaternion_linear(x.contiguous(), *packed)
