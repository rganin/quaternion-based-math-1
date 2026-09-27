"""PyTorch dense / naive / planar operators and a sync-timed benchmark harness.

This is the software-only member of the kernel family (plan.md sections 11,
13, 41-43).  ``dense_gemm`` is the cuBLAS baseline; ``naive`` and ``planar``
map to the K1/K2 quaternion implementations.
"""

from __future__ import annotations

import time

import torch

from quaternion.linear import quaternion_linear
from quaternion.planar import quaternion_linear_naive, quaternion_linear_planar

__all__ = ["dense_gemm", "naive", "planar", "benchmark_operator"]


def dense_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Baseline ``x @ weight.T`` (uses cuBLAS on CUDA)."""
    return x @ weight.t()


def naive(x, wr, wi, wj, wk):
    return quaternion_linear_naive(x, wr, wi, wj, wk)


def planar(x, wr, wi, wj, wk):
    return quaternion_linear_planar(x, wr, wi, wj, wk)


def _sync(device):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_operator(
    fn,
    args,
    warmup: int = 10,
    iters: int = 50,
    device: torch.device | None = None,
) -> dict:
    """Time ``fn(*args)`` with warmup and synchronized CUDA timing.

    Returns latency in milliseconds (mean and min) and an approximate
    throughput in giga-elements-per-second of output.
    """
    for _ in range(warmup):
        fn(*args)
    _sync(device)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn(*args)
        _sync(device)
        times.append((time.perf_counter() - start) * 1e3)

    t = torch.tensor(times)
    out_elems = out.numel()
    mean_ms = t.mean().item()
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_min": t.min().item(),
        "latency_ms_p50": t.median().item(),
        "output_elements": out_elems,
        "gelem_per_s": out_elems / (mean_ms * 1e6) if mean_ms > 0 else float("inf"),
        "iters": iters,
        "warmup": warmup,
    }
