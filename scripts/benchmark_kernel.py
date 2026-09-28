"""Benchmark dense vs naive vs planar vs Triton vs torch.compile (K1-K3).

Prefill and decode are benchmarked separately (plan.md sections 41-46).
Results are written as machine-readable JSON experiment records.

Usage:
    python scripts/benchmark_kernel.py --hidden 1024 --seqs 128 512 2048 --dtype bfloat16
    python scripts/benchmark_kernel.py --hidden 2048 --seqs 1 512 --device cuda --triton
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.tracking import ExperimentResult, new_experiment_id
from kernels.pytorch.ops import benchmark_operator, dense_gemm, naive, planar
from quaternion.linear import quaternion_linear


def build_args(hidden, tokens, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=g, dtype=dtype).to(device)
    in_g = hidden // 4
    w = [
        torch.randn(in_g, in_g, generator=g, dtype=dtype).to(device) / in_g**0.5
        for _ in range(4)
    ]
    block = torch.randn(hidden, hidden, generator=g, dtype=dtype).to(device) / hidden**0.5
    return x, w, block


def build_specs(x, w, block, dtype, use_triton, use_compile, use_cuda=False):
    """Return {name: (fn, args)} for every available operator."""
    specs = {
        "dense": (dense_gemm, (x, block)),
        "naive": (naive, (x, *w)),
        "planar": (planar, (x, *w)),
    }

    if use_triton and x.is_cuda:
        from kernels.triton.quaternion import (
            pack_weights,
            triton_quaternion_linear_packed,
        )

        ieee = dtype == torch.float32
        stages = 1 if ieee else 2
        packed = pack_weights(*w)

        def triton_fn(x, _packed=packed, _ieee=ieee, _stages=stages):
            return triton_quaternion_linear_packed(
                x, *_packed, ieee=_ieee, num_stages=_stages
            )

        specs["triton"] = (triton_fn, (x,))

    if use_compile and x.is_cuda:
        compiled = torch.compile(quaternion_linear)
        specs["compile"] = (compiled, (x, *w))

    if use_cuda and x.is_cuda:
        try:
            from kernels.cuda.loader import quaternion_linear_cuda

            specs["cuda"] = (quaternion_linear_cuda, (x, *w))
        except Exception as exc:  # pragma: no cover
            print(f"[benchmark] CUDA kernel unavailable: {exc}")

    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--seqs", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--triton", action="store_true", help="include fused Triton kernel")
    parser.add_argument("--compile", action="store_true", help="include torch.compile")
    parser.add_argument("--cuda", action="store_true", help="include custom fused CUDA kernel")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print(f"device={device} dtype={args.dtype} hidden={args.hidden}")
    names = ["dense", "naive", "planar"]
    if args.triton:
        names.append("triton")
    if args.compile:
        names.append("compile")
    if args.cuda:
        names.append("cuda")
    header = f"{'regime':>8} {'tokens':>8} " + " ".join(
        f"{n + '_ms':>11}" for n in names
    ) + f"{'tri/dense':>11}"
    print(header)
    print("-" * len(header))

    for seq in args.seqs:
        tokens = args.batch * seq
        x, w, block = build_args(args.hidden, tokens, dtype, device)
        regimes = {"prefill": x}
        if args.batch == 1:
            regimes["decode"] = x[:1]

        for regime, xr in regimes.items():
            specs = build_specs(
                xr, w, block, dtype, args.triton, args.compile, args.cuda
            )
            results = {
                name: benchmark_operator(
                    fn, fn_args, warmup=args.warmup, iters=args.iters, device=device
                )
                for name, (fn, fn_args) in specs.items()
            }
            dense_ms = results["dense"]["latency_ms_mean"]
            row = f"{regime:>8} {xr.shape[0]:>8} "
            row += " ".join(
                f"{results[n]['latency_ms_mean']:>11.4f}" if n in results else f"{'n/a':>11}"
                for n in names
            )
            tri_ms = results.get("triton", results["dense"])["latency_ms_mean"]
            row += f"{tri_ms / dense_ms:>11.2f}"
            print(row)

            if not args.no_save:
                ExperimentResult(
                    experiment_id=new_experiment_id(
                        f"kernel_{regime}", version=1, suffix=f"h{args.hidden}_{args.dtype}"
                    ),
                    method="benchmark",
                    metrics=results,
                    config={
                        "hidden": args.hidden,
                        "tokens": int(xr.shape[0]),
                        "regime": regime,
                        "dtype": args.dtype,
                        "device": str(device),
                        "operators": names,
                    },
                ).save()


if __name__ == "__main__":
    main()
