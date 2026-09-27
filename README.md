# Quaternion Transformer Compression & Hardware Efficiency

Research codebase for investigating whether pretrained Transformer linear layers
can be compressed through quaternion-valued parameterization and recovered with
LoRA, and whether the structured computation can be mapped efficiently to
Blackwell GPU hardware.

See [`plan.md`](plan.md) for the full research plan and stage ordering.

> **Guiding rule:** math -> matrix reconstruction -> activation reconstruction
> -> block reconstruction -> 1B model -> quality eval -> kernel optimization ->
> LoRA -> 7-9B -> 27-31B. Do not start with a 31B model.

## Representation

A real matrix `W ∈ R^(out × in)` is replaced by quaternion weights
`Wq = Wr + Wi i + Wj j + Wk k`, stored as four real tensors of shape
`[out/4, in/4]`. This is a 4x reduction in scalar parameters for the same
logical dimensions.

The equivalent real operator is the block matrix

```
[ Wr  -Wi  -Wj  -Wk ]
[ Wi   Wr  -Wk   Wj ]
[ Wj   Wk   Wr  -Wi ]
[ Wk  -Wj   Wi   Wr ]
```

which is used only for verification; kernels must exploit the quaternion
structure directly.

## Channel packing convention

A length-`D` feature vector (`D % 4 == 0`) is split into `D/4` contiguous groups
of four. Each group is the quaternion `[q_r, q_i, q_j, q_k]`. Packing `[..., D]`
yields `[..., D/4, 4]` with the last axis always ordered `(r, i, j, k)`. This is
documented in `quaternion/packing.py` and normatively tested.

## Hamilton product

```
(a,b,c,d) * (x,y,z,w) =
    r = ax - by - cz - dw
    i = ay + bx + cw - dz
    j = az - bw + cx + dy
    k = aw + bz - cy + dx
```

## Extended research directions (plan.md sections 33-57)

Beyond basic 4x compression, the codebase exposes the building blocks for the
extended hypotheses:

| Hypothesis | Component | Module |
|---|---|---|
| H6 planar quaternion GEMM | naive (K1) + planar (K2) operators | `quaternion/planar.py` |
| H7 shared-score attention | standard / component (A1) / shared-score (A2) | `quaternion/attention.py` |
| H8 factorized quaternion weights | independent / shared-input / shared-output (F1/F2) | `quaternion/factorized.py` |
| H9 unit-quaternion constraints | rotation-constrained weights | `quaternion/unit.py` |

## Status

| Stage | Component | Status |
|---|---|---|
| 0 | Quaternion arithmetic + equivalence tests | **done** |
| 1 | Matrix reconstruction (closed-form + AdamW) | **done** |
| 2 | Activation-aware reconstruction | **done** (AdamW) |
| -- | Planar GEMM operators (H6) | **done** (PyTorch) |
| -- | Shared-score attention (H7) | **done** (PyTorch) |
| -- | Factorized quaternion weights (H8) | **done** |
| -- | Unit-quaternion weights (H9) | **done** |
| -- | Reconstruction metrics + experiment tracking | **done** |
| 3 | Block reconstruction | not started |
| 4/8 | Full 1B conversion | not started |
| 5/11 | Kernel benchmarks | PyTorch + fused Triton done; CUTLASS/CuTe pending |
| 6/18 | LoRA recovery | not started |

## Reconstruction

`conversion/fit_linear.py` supports two targets. Because both objectives are
linear in the quaternion parameters, the matrix-reconstruction optimum has a
closed form (`project_onto_quaternion`), which is the default; the plan's AdamW
path is retained for comparison and for the activation-aware objective.

## Experiment tracking

`experiments/tracking.py` writes one machine-readable JSON file per run to
`experiments/results/<experiment_id>.json`, including metrics, config, git
commit and timestamp (plan.md section 26).

## Kernel benchmark

```bash
python scripts/benchmark_kernel.py --hidden 1024 --seqs 128 512 2048 \
    --dtype bfloat16 --device cuda --triton --compile
```

Operators compared: `dense` (cuBLAS), `naive` (K1), `planar` (K2),
`triton` (K3 fused kernel), and `torch.compile`, separately for prefill and
decode. It does not assume the quaternion path is faster; it measures it.

### Measured on DGX Spark (GB10, sm_121), hidden=1024, bf16

| regime | dense | naive | planar | Triton (fused) | tri/dense |
|---|---:|---:|---:|---:|---:|
| prefill 128 | 0.0185 ms | 0.272 ms | 0.219 ms | 0.068 ms | 3.7x |
| decode 1 | 0.0124 ms | 0.218 ms | 0.175 ms | 0.059 ms | 4.8x |
| prefill 2048 | 0.0642 ms | 0.410 ms | 0.500 ms | 0.171 ms | 2.7x |

The fused Triton kernel is ~3-4x faster than the naive/planar PyTorch
implementations, confirming that fusion matters (H6/K3), but it still trails
cuBLAS. Closing that gap is the CUTLASS/CuTe experiment (K4).

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

Phase 0 tests cover Hamilton-product algebra, packing round-trips, equivalence
between the efficient four-equation implementation, the explicit reference
oracle and the real block matrix, FP32/FP16/BF16 behaviour, and autograd
gradients (gradcheck + finite differences). Additional tests cover the planar
and naive operators, factorized components, the three attention variants,
unit-quaternion weights, reconstruction metrics, experiment tracking and both
reconstruction targets.
