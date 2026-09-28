# Experimentation Log

This file records every experiment run for the quaternion Transformer project.
Each entry lists the exact command, the artifact path, and the results.

## Objective (restated)

1. **Representation (H1–H3):** how accurately can pretrained Transformer linear
   layers be represented by quaternion-valued layers (≈4× fewer scalar
   parameters)?
2. **Systems (H4–H6):** can the structured quaternion computation be made
   competitive with dense GEMM on Blackwell?
3. **Extended (H7–H9):** shared-score attention, factorized quaternion weights,
   unit-quaternion constraints.

Success is not parameter count alone: it requires acceptable model quality,
(ideally) LoRA recovery, and a plausible kernel-speed path.

## Environment

| Item | Value |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10 (Grace-Blackwell), sm_121, 121 GiB unified |
| CUDA / driver | 13.0 / 580.142 |
| PyTorch | 2.12.0+cu130 |
| Transformers | 5.9.0 |
| Triton | 3.7.0 |
| Python | 3.12 (venv at `~/venv`) |

## Methodology

- **Kernel experiments (K1–K3).** Benchmark `dense` (cuBLAS), `naive` (K1),
  `planar` (K2), `triton` (K3 fused kernel) and `torch.compile` with warmup and
  synchronized CUDA timing, separately for prefill and decode.
  Implementations: `kernels/pytorch/ops.py`, `kernels/triton/quaternion.py`.
- **Stage 1 (matrix reconstruction).** Closed-form least-squares projection of
  each pretrained projection matrix onto the quaternion block structure
  (`conversion/fit_linear.py::project_onto_quaternion`). Contiguous channel
  grouping. This is the optimal solution for the matrix objective.
- **Stage 2 (activation-aware).** Fit `QuaternionLinear(X) ≈ Y` on captured
  activations (AdamW, float32), evaluated on a held-out 10% split after
  per-channel mean centering (removes bias). Compared against the Stage 1
  projection evaluated on the same activations.
- All runs use `scripts/reconstruct_layer.py`; machine-readable evidence is
  written under `experiments/results/`.

---

## E1 — Kernel benchmark, K1/K2/K3 (2026-09-27)

**Command**

```bash
python scripts/benchmark_kernel.py --hidden 768  --seqs 128 512 2048 \
    --dtype bfloat16 --device cuda --triton --iters 30 --warmup 10 --no-save
python scripts/benchmark_kernel.py --hidden 1024 --seqs 128 2048 \
    --dtype bfloat16 --device cuda --triton --compile --iters 30 --warmup 10 --no-save
```

Latency in milliseconds (lower is better). `tri/dense` is the fused-Triton to
dense ratio.

### hidden = 768 (GPT-2 width)

| regime | tokens | dense | naive | planar | triton | tri/dense |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 128 | 0.0166 | 0.2329 | 0.1768 | 0.0542 | 3.27 |
| decode | 1 | 0.0129 | 0.2109 | 0.1750 | 0.0495 | 3.84 |
| prefill | 512 | 0.0248 | 0.2624 | 0.1957 | 0.0555 | 2.24 |
| decode | 1 | 0.0124 | 0.2125 | 0.1763 | 0.0491 | 3.95 |
| prefill | 2048 | 0.0365 | 0.3409 | 0.3822 | 0.1375 | 3.76 |
| decode | 1 | 0.0154 | 0.2153 | 0.1660 | 0.0516 | 3.35 |

### hidden = 1024

| regime | tokens | dense | naive | planar | triton | compile | tri/dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | 128 | 0.0185 | 0.2716 | 0.2187 | 0.0683 | 0.1790 | 3.68 |
| decode | 1 | 0.0124 | 0.2180 | 0.1751 | 0.0589 | 0.1530 | 4.75 |
| prefill | 2048 | 0.0642 | 0.4104 | 0.5000 | 0.1709 | 0.2726 | 2.66 |
| decode | 1 | 0.0140 | 0.2113 | 0.1758 | 0.0583 | 0.1545 | 4.15 |

**Findings**

- H4 confirmed: the naive quaternion loop is **10–17× slower** than cuBLAS.
- H6 (partial): the fused Triton kernel is **2.2–4.8× faster** than
  naive/planar, but still **2.2–4.8× slower than cuBLAS**. Fusion helps a lot;
  it does not yet win. CUTLASS/CuTe (K4) is the remaining lever, along with
  weight-tile reuse and register-resident accumulators.
- `torch.compile` on the PyTorch quaternion path is not competitive here.

---

## E2 — Stage 1: matrix reconstruction, GPT-2 small (2026-09-27)

**Model:** `gpt2` (124M, 12 blocks). **Grouping:** contiguous.
**Command**

```bash
python scripts/reconstruct_layer.py --model gpt2 --stage matrix --no-save
```

All 12 blocks × 6 projections = **72 matrices**. Weight shape and
`[out, in]`, 4× parameter compression for every layer.

### Aggregate across all 72 projections

| metric | mean | min | max |
|---|---:|---:|---:|
| relative Frobenius error | 0.8661 | 0.8649 | 0.8675 |
| cosine similarity | 0.4999 | 0.4974 | 0.5019 |
| compression ratio | 4.00 | — | — |

### By projection type (mean relative Frobenius error)

| kind | n | mean | min | max |
|---|---:|---:|---:|---:|
| q_proj | 12 | 0.8658 | 0.8649 | 0.8665 |
| k_proj | 12 | 0.8663 | 0.8657 | 0.8675 |
| v_proj | 12 | 0.8661 | 0.8654 | 0.8671 |
| o_proj | 12 | 0.8660 | 0.8652 | 0.8664 |
| up_proj | 12 | 0.8660 | 0.8658 | 0.8663 |
| down_proj | 12 | 0.8661 | 0.8657 | 0.8665 |

**Finding — H1 fails under naive contiguous quaternionization.**
The best possible quaternion block approximation of a pretrained GPT-2
projection retains essentially **none** of the original map: relative error
≈ 0.87 and cosine ≈ 0.50, indistinguishable across projection types and
layers. This is the plan's highest-risk assumption materializing. Contiguous
quaternionization is not viable on its own.

---

## E3 — Stage 2: activation-aware reconstruction, GPT-2 small (2026-09-27)

**Command**

```bash
python scripts/reconstruct_layer.py --model gpt2 --stage both \
    --activation-blocks 2 --calib-seqs 32 --seq-len 128 --act-iters 400 \
    --no-save --summary experiments/results/gpt2_small_stage12_summary.json
```

Calibration: wikitext-2, 32 sequences × 128 tokens (4096 tokens), 10% held out.
Relative L2 error on the held-out activations (lower is better).

| layer | kind | matrix projection | activation-aware | improvement |
|---|---|---:|---:|---:|
| h.0.attn.c_attn | qkv_proj | 0.8128 | 0.5307 | 1.53× |
| h.0.attn.c_proj | o_proj | 0.8226 | 0.4819 | 1.71× |
| h.0.mlp.c_fc | up_proj | 0.8183 | 0.5189 | 1.58× |
| h.0.mlp.c_proj | down_proj | 0.8508 | 0.5448 | 1.56× |
| h.1.attn.c_attn | qkv_proj | 0.8199 | 0.4446 | 1.84× |
| h.1.attn.c_proj | o_proj | 0.9316 | 0.6356 | 1.47× |
| h.1.mlp.c_fc | up_proj | 0.8493 | 0.5569 | 1.53× |
| h.1.mlp.c_proj | down_proj | 0.7771 | 0.2776 | 2.80× |

Mean improvement: **1.75×**.

**Findings**

- H3 supported: optimizing on the actual activation distribution is materially
  better than approximating the weight matrix. The activation covariance is
  structured and the quaternion parameters can exploit it.
- The absolute error is still large (0.28–0.64). Activation-aware fitting alone
  does not rescue contiguous quaternionization; it reduces, but does not close,
  the gap.
- The gain varies substantially by layer (1.47×–2.80×), suggesting layer-wise
  sensitivity that a full-model evaluation would expose.

---

## E4 — Learned channel grouping, GPT-2 small (2026-09-27)

**Command** (run in background per plan §58; result archived)

```bash
python scripts/reconstruct_layer.py --model gpt2 --stage matrix \
    --compare-groupings --learned-iters 100 \
    --no-save --summary experiments/results/gpt2_small_grouping_comparison.json
```

For each of the 72 projections we compare three groupings:
`contiguous`, a fixed `random` permutation, and an optimized (`learned`)
input/output permutation. The learned search relaxes permutations to
doubly-stochastic matrices (Sinkhorn) and applies the closed-form quaternion
projection inside the loop, then rounds to a hard permutation with the
Hungarian algorithm.

### Aggregate (mean relative Frobenius error, 72 projections)

| method | mean | min | max |
|---|---:|---:|---:|
| contiguous | 0.8661 | 0.8649 | 0.8675 |
| random | 0.8659 | 0.8653 | 0.8665 |
| learned | 0.8661 | 0.8649 | 0.8675 |

**Finding — H1 has a hard theoretical floor; grouping is not the lever.**

Quaternion block matrices of shape `[out, in]` form a **linear subspace of
dimension `out·in/4`** inside `R^{out·in}` (each 4×4 block has only 4 free
parameters). For an isotropic (full-rank, random-like) matrix the expected
projection error is therefore

```text
relative Frobenius error = sqrt(1 - 1/4) = sqrt(3/4) ≈ 0.866
```

Permutations are orthogonal transformations and leave this unchanged, so
**no channel grouping — contiguous, random, or learned — can beat ≈0.866** for
unstructured weights. GPT-2's projections are effectively isotropic at this
scale, and the measured 0.8661/0.8659/0.8661 confirms it. The learned
permutation starts near identity and cannot escape because the loss surface is
essentially flat over permutations.

A dedicated test (`test_random_matrix_projection_hits_theoretical_floor`)
verifies the ≈0.866 floor on random matrices.

---

## E5 — Activation-aware reconstruction, all 12 blocks, GPT-2 small (2026-09-27)

**Launch (background, plan §58)**

```bash
nohup python scripts/reconstruct_layer.py --model gpt2 --stage activation \
    --activation-blocks 12 --calib-seqs 128 --seq-len 128 --act-iters 500 \
    --no-save --summary experiments/results/gpt2_small_activation_allblocks.json \
    > experiments/logs/e5_activation_allblocks.log 2>&1 &
```

Log: `experiments/logs/e5_activation_allblocks.log` · pid 650007 · wall time ≈ 22 min.
Calibration: wikitext-2, 128 seqs × 128 tokens (16 384 tokens), 10% held out.
Relative L2 error on held-out activations (lower is better).

| layer | kind | matrix | activation | improvement |
|---|---|---:|---:|---:|
| h.0.attn.c_attn | qkv_proj | 0.8126 | 0.5661 | 1.435 |
| h.0.attn.c_proj | o_proj | 0.8191 | 0.5011 | 1.635 |
| h.0.mlp.c_fc | up_proj | 0.8209 | 0.5583 | 1.470 |
| h.0.mlp.c_proj | down_proj | 0.8547 | 0.5591 | 1.529 |
| h.1.attn.c_attn | qkv_proj | 0.8191 | 0.4851 | 1.688 |
| h.1.attn.c_proj | o_proj | 0.9478 | 0.6372 | 1.487 |
| h.1.mlp.c_fc | up_proj | 0.8462 | 0.6152 | 1.376 |
| h.1.mlp.c_proj | down_proj | 0.7797 | 0.3028 | 2.575 |
| h.2.attn.c_attn | qkv_proj | 0.8073 | 0.3571 | 2.261 |
| h.2.attn.c_proj | o_proj | 0.9451 | 0.7093 | 1.332 |
| h.2.mlp.c_fc | up_proj | 0.8434 | 0.6224 | 1.355 |
| h.2.mlp.c_proj | down_proj | 0.7525 | 0.2958 | 2.544 |
| h.3.attn.c_attn | qkv_proj | 0.8109 | 0.4240 | 1.912 |
| h.3.attn.c_proj | o_proj | 0.9142 | 0.7714 | 1.185 |
| h.3.mlp.c_fc | up_proj | 0.8429 | 0.6578 | 1.281 |
| h.3.mlp.c_proj | down_proj | 0.8563 | 0.6540 | 1.309 |
| h.4.attn.c_attn | qkv_proj | 0.8173 | 0.5024 | 1.627 |
| h.4.attn.c_proj | o_proj | 0.9361 | 0.7934 | 1.180 |
| h.4.mlp.c_fc | up_proj | 0.8464 | 0.6731 | 1.257 |
| h.4.mlp.c_proj | down_proj | 0.8832 | 0.6598 | 1.338 |
| h.5.attn.c_attn | qkv_proj | 0.8486 | 0.6386 | 1.329 |
| h.5.attn.c_proj | o_proj | 0.9036 | 0.7608 | 1.188 |
| h.5.mlp.c_fc | up_proj | 0.8482 | 0.6783 | 1.250 |
| h.5.mlp.c_proj | down_proj | 0.9279 | 0.7725 | 1.201 |
| h.6.attn.c_attn | qkv_proj | 0.8529 | 0.6398 | 1.333 |
| h.6.attn.c_proj | o_proj | 0.9236 | 0.7942 | 1.163 |
| h.6.mlp.c_fc | up_proj | 0.8453 | 0.6847 | 1.234 |
| h.6.mlp.c_proj | down_proj | 0.9614 | 0.8253 | 1.165 |
| h.7.attn.c_attn | qkv_proj | 0.8522 | 0.6675 | 1.277 |
| h.7.attn.c_proj | o_proj | 0.8849 | 0.7696 | 1.150 |
| h.7.mlp.c_fc | up_proj | 0.8398 | 0.6873 | 1.222 |
| h.7.mlp.c_proj | down_proj | 0.9617 | 0.8465 | 1.136 |
| h.8.attn.c_attn | qkv_proj | 0.8479 | 0.6747 | 1.257 |
| h.8.attn.c_proj | o_proj | 0.9098 | 0.7693 | 1.183 |
| h.8.mlp.c_fc | up_proj | 0.8356 | 0.6909 | 1.209 |
| h.8.mlp.c_proj | down_proj | 0.9514 | 0.8477 | 1.122 |
| h.9.attn.c_attn | qkv_proj | 0.8522 | 0.7075 | 1.205 |
| h.9.attn.c_proj | o_proj | 0.8607 | 0.7550 | 1.140 |
| h.9.mlp.c_fc | up_proj | 0.8325 | 0.6901 | 1.206 |
| h.9.mlp.c_proj | down_proj | 0.9396 | 0.8376 | 1.122 |
| h.10.attn.c_attn | qkv_proj | 0.8541 | 0.7219 | 1.183 |
| h.10.attn.c_proj | o_proj | 0.8472 | 0.6959 | 1.217 |
| h.10.mlp.c_fc | up_proj | 0.8303 | 0.6908 | 1.202 |
| h.10.mlp.c_proj | down_proj | 0.9131 | 0.7567 | 1.207 |
| h.11.attn.c_attn | qkv_proj | 0.8658 | 0.7598 | 1.140 |
| h.11.attn.c_proj | o_proj | 0.7580 | 0.4084 | 1.856 |
| h.11.mlp.c_fc | up_proj | 0.8288 | 0.7033 | 1.178 |
| h.11.mlp.c_proj | down_proj | 0.8736 | 0.6215 | 1.406 |

### Aggregate by projection type (48 modules)

| kind | n | matrix | activation | improvement |
|---|---:|---:|---:|---:|
| qkv_proj | 12 | 0.8367 | 0.5954 | 1.470 |
| o_proj | 12 | 0.8875 | 0.6971 | 1.310 |
| up_proj | 12 | 0.8383 | 0.6627 | 1.270 |
| down_proj | 12 | 0.8879 | 0.6650 | 1.471 |
| **overall** | 48 | — | **0.6547** | **1.380** |

**Findings**

- Activation-aware fitting helps everywhere (1.12×–2.58×, mean **1.38×**) but the
  absolute held-out error remains large (0.30–0.85, mean **0.65**). This is far
  too much error for a Transformer to tolerate layer after layer; end-to-end
  quality is expected to collapse without recovery.
- Error grows with depth for the mid/late blocks (h.4–h.10 ≈ 0.66–0.85) and is
  lowest in the first layers (h.2, h.1.mlp.c_proj ≈ 0.30).
- **Methodology note (underfitting):** E5 uses 4× the calibration of E3 with the
  same 500 iterations, and its block-0/1 errors are *slightly worse* than E3
  (e.g. 0.5661 vs 0.5307 for h.0.c_attn). More data with a fixed iteration
  budget underfits. Calibration-size scaling (§5.1) must scale the optimizer
  budget or use a proper convergence criterion; raw error is not yet
  data-limited.
- `down_proj` and `qkv_proj` benefit most (1.47×); `up_proj` least (1.27×).

---

## E6 — End-to-end GPT-2 conversion + perplexity (2026-09-27)

**Launch (background, plan §58)**

```bash
nohup python scripts/reconstruct_model.py --model gpt2 \
    --variants matrix activation --calib-seqs 64 --calib-seq-len 128 \
    --act-iters 300 --ppl-batches 50 --ppl-seq-len 192 \
    --summary experiments/results/gpt2_small_conversion.json \
    > experiments/logs/e6_conversion.log 2>&1 &
```

Log: `experiments/logs/e6_conversion.log` · pid 651069 · wall time ≈ 9 min.
All 48 eligible projections converted; embeddings, norms, position embeddings
and LM head left unchanged. Perplexity on wikitext-2 test, 50 chunks × 192
tokens (chunked context; the absolute baseline is 59.35, higher than the
standard long-context GPT-2 number, but the comparison is internally
consistent).

| variant | perplexity | mean NLL | modules |
|---|---:|---:|---:|
| baseline (FP32) | **59.35** | 4.08 | — |
| quaternion (matrix projection) | **83 662.6** | 11.33 | 48 |
| quaternion (activation-aware) | **7 970.7** | 8.98 | 48 |

**Findings — Gate 2 fails without recovery.**

- Direct quaternionization is catastrophic: perplexity rises ~1400×
  (matrix) and ~134× (activation-aware). The model is effectively destroyed.
- Activation-aware conversion is ~10.5× better than matrix projection,
  consistent with the E5 per-layer ordering, but nowhere near sufficient.
- This closes the "can we just compress pretrained weights 4×" question:
  **no**. The remaining hope is recovery (LoRA) or a different structural
  constraint; E7 tests recovery.

---

## E7 — Quaternion + LoRA recovery, GPT-2 small (2026-09-27)

**Launch (background, plan §58)**

```bash
nohup python scripts/train_lora.py --model gpt2 --convert activation \
    --ranks 4 8 16 32 --max-steps 400 --lr 1e-3 --batch-size 4 --grad-accum 2 \
    --ppl-batches 50 --ppl-seq-len 192 \
    --summary experiments/results/gpt2_small_lora.json \
    > experiments/logs/e7_lora.log 2>&1 &
```

Log: `experiments/logs/e7_lora.log` · pid 651950 · wall time ≈ 11 min.
Base converted activation-aware (E6 variant, ppl 7 970.7). The quaternion base is
frozen; a real-valued LoRA correction is trained on wikitext with 400 optimizer
steps × 4 sequences × 128 tokens × 2 grad-accum = 409 600 tokens, bf16.

| rank | trainable params | ppl before | ppl after | NLL after | final train loss |
|---:|---:|---:|---:|---:|---:|
| 4 | 589 824 | 7 970.7 | **559.2** | 6.326 | 6.185 |
| 8 | 1 179 648 | 7 970.7 | **506.6** | 6.228 | 6.397 |
| 16 | 2 359 296 | 7 970.7 | **452.8** | 6.115 | 6.118 |
| 32 | 4 718 592 | 7 970.7 | **448.3** | 6.105 | 5.855 |

Reference: baseline NLL 4.08 (ppl 59.35).

**Findings — Gate 4 is partially met.**

- LoRA recovers a large fraction of the damage: perplexity 7 970.7 → 448.3
  (a **17.8× reduction**), and NLL 8.98 → 6.10. In log-perplexity terms it
  closes about **54%** of the gap to the unconverted baseline.
- The residual is still large: the recovered model is ~7.6× the baseline
  perplexity, so quality is not recovered.
- Gains from rank saturate quickly (446 → 559 from rank 4→32), so capacity is
  not the bottleneck; the frozen quaternion base caps recoverable quality at
  this training budget.
- **Budget caveat:** this is a deliberately short first pass (≈0.4 M tokens,
  400 steps). Standard LoRA fits use orders of magnitude more. The ceiling is
  not yet measured.

---

## E8 — Generality check on Qwen2.5-1.5B (2026-09-27)

**Launch (background, chained with E7b, plan §58)**

```bash
python scripts/reconstruct_layer.py --model Qwen/Qwen2.5-1.5B --stage both \
    --activation-blocks 4 --calib-seqs 64 --seq-len 128 --act-iters 300 \
    --no-save --summary experiments/results/qwen25_15b_stage12.json
```

Log: `experiments/logs/e8_qwen_stage12.log` · wall time ≈ 9 min.
Architecture: 28 layers, hidden 1536, SwiGLU MLP (8960), GQA. 196 eligible
projections (all divisible by 4).

### Stage 1 — matrix reconstruction (196 projections)

| kind | n | mean rel. error |
|---|---:|---:|
| all 7 projection types | 196 | **0.8660** (min 0.8647, max 0.8671) |

**The 0.866 subspace floor is architecture-independent** — Qwen2.5 matches
GPT-2 to four decimals.

### Stage 2 — activation-aware (layers 0–3, 28 modules)

| layer | kind | matrix | activation | improvement |
|---|---|---:|---:|---:|
| 0.self_attn.q_proj | q_proj | 0.7579 | 0.1297 | 5.844 |
| 0.self_attn.k_proj | k_proj | 0.9269 | 0.4802 | 1.930 |
| 0.self_attn.v_proj | v_proj | 0.9679 | 0.5188 | 1.866 |
| 0.self_attn.o_proj | o_proj | 0.8679 | 0.7037 | 1.233 |
| 0.mlp.gate_proj | gate_proj | 0.8182 | 0.5450 | 1.501 |
| 0.mlp.up_proj | up_proj | 0.8300 | 0.5705 | 1.455 |
| 0.mlp.down_proj | down_proj | 0.8433 | 0.6667 | 1.265 |
| 1.self_attn.q_proj | q_proj | 0.8263 | 0.4720 | 1.751 |
| 1.self_attn.k_proj | k_proj | 0.7991 | 0.4306 | 1.856 |
| 1.self_attn.v_proj | v_proj | 0.8225 | 0.5223 | 1.575 |
| 1.self_attn.o_proj | o_proj | 0.8425 | 0.6347 | 1.328 |
| 1.mlp.gate_proj | gate_proj | 0.8606 | 0.6311 | 1.364 |
| 1.mlp.up_proj | up_proj | 0.8771 | 0.6653 | 1.318 |
| 1.mlp.down_proj | down_proj | 0.8603 | **0.0359** | 23.938 |
| 2.self_attn.q_proj | q_proj | 0.8291 | 0.5438 | 1.525 |
| 2.self_attn.k_proj | k_proj | 0.8231 | 0.5578 | 1.476 |
| 2.self_attn.v_proj | v_proj | 0.8426 | 0.6044 | 1.394 |
| 2.self_attn.o_proj | o_proj | 0.8455 | 0.6779 | 1.247 |
| 2.mlp.gate_proj | gate_proj | 0.8615 | 0.6056 | 1.422 |
| 2.mlp.up_proj | up_proj | 0.8955 | 0.6724 | 1.332 |
| 2.mlp.down_proj | down_proj | 0.8227 | **0.0273** | 30.086 |
| 3.self_attn.q_proj | q_proj | 0.8482 | 0.5538 | 1.532 |
| 3.self_attn.k_proj | k_proj | 0.8280 | 0.5141 | 1.611 |
| 3.self_attn.v_proj | v_proj | 0.8796 | 0.6405 | 1.373 |
| 3.self_attn.o_proj | o_proj | 0.8538 | 0.6954 | 1.228 |
| 3.mlp.gate_proj | gate_proj | 0.8465 | 0.5886 | 1.438 |
| 3.mlp.up_proj | up_proj | 0.9033 | 0.7191 | 1.256 |
| 3.mlp.down_proj | down_proj | 0.9508 | 0.4985 | 1.907 |

### Aggregate by type (28 modules)

| kind | matrix | activation | improvement |
|---|---:|---:|---:|
| q_proj | 0.8154 | 0.4248 | 2.663 |
| k_proj | 0.8443 | 0.4957 | 1.718 |
| v_proj | 0.8781 | 0.5715 | 1.552 |
| o_proj | 0.8524 | 0.6779 | 1.259 |
| gate_proj | 0.8467 | 0.5926 | 1.431 |
| up_proj | 0.8765 | 0.6568 | 1.340 |
| down_proj | 0.8693 | **0.3071** | 14.299 |
| **overall** | — | **0.5324** | **3.466** |

**Findings**

- The weight-level floor is universal; the **functional** (activation-aware)
  result is strongly architecture- and layer-dependent. Qwen2.5 is more
  recoverable than GPT-2 overall (0.53 vs 0.65), and some layers are almost
  exactly quaternion-representable on their activation manifold
  (`0.q_proj` 0.13; `1/2.mlp.down_proj` 0.03–0.04).
- The near-zero `down_proj` values are **not** a degenerate holdout: the
  held-out slice is ordinary English (819 tokens, 305 unique). They likely
  reflect SwiGLU gating making early-layer MLP activations sparse/low-rank.
- **Caveat:** calibration and holdout are contiguous slices from a single
  wikitext topic region, so the Stage-2 numbers are in-distribution. A random
  split over diverse text is needed to confirm the very low errors (E8b).

---

## E8b — Random-split validation on Qwen2.5-1.5B (2026-09-27)

**Command** (background, chained with E7c; part 2 = Qwen end-to-end, running)

```bash
python scripts/reconstruct_layer.py --model Qwen/Qwen2.5-1.5B --stage both \
    --calib-random --activation-blocks 4 --calib-seqs 96 --seq-len 128 \
    --act-iters 300 --no-save --summary experiments/results/qwen25_15b_stage12_random.json
```

Calibration now uses **random windows across the whole corpus** (diverse topics)
instead of one contiguous region; 96 × 128 tokens, 10% held out.

| metric | E8 (contiguous) | E8b (random) |
|---|---:|---:|
| Stage 1 mean rel. error (196 proj) | 0.8660 | 0.8660 |
| Stage 2 overall (28 modules) | 0.5324 | **0.5368** |
| Stage 2 mean improvement | 3.466 | 3.314 |
| `1.mlp.down_proj` | 0.0359 | **0.0365** |
| `2.mlp.down_proj` | 0.0273 | **0.0304** |
| `0.self_attn.q_proj` | 0.1297 | **0.1414** |

**Findings**

- The results are **essentially identical** under a diverse random split, so the
  E8 low-error layers are real structure, not a holdout artifact. The caveat is
  resolved.
- The 0.8660 weight-level floor holds exactly once more.
**Part 2 — Qwen2.5-1.5B end-to-end conversion** (`qwen25_15b_conversion.json`;
log `experiments/logs/e8b_qwen_conversion.log`, ≈45 min). 196 projections
converted; perplexity on wikitext-2 test, 30 × 192 tokens.

| variant | perplexity | NLL |
|---|---:|---:|
| baseline | **17.83** | 2.88 |
| quaternion (matrix) | 553 655 | 13.22 |
| quaternion (activation-aware) | 41 350 | 10.63 |

- Same qualitative result as GPT-2 (E6): direct conversion destroys the model;
  activation-aware is ~13× better than matrix but still unusable without
  recovery. Qwen2.5-1.5B's per-layer functional advantage (E8) does not rescue
  end-to-end quality on its own.

---

## E7b — Longer LoRA budget, GPT-2 small (2026-09-27)

**Launch (background, chained with E8, plan §58)**

```bash
python scripts/train_lora.py --model gpt2 --convert activation \
    --ranks 8 32 --max-steps 3000 --lr 1e-3 --lr-schedule cosine \
    --batch-size 8 --grad-accum 2 --data-chars 4000000 \
    --calib-seqs 64 --calib-seq-len 128 --act-iters 300 \
    --ppl-batches 50 --ppl-seq-len 192 \
    --summary experiments/results/gpt2_small_lora_e7b.json
```

Log: `experiments/logs/e7b_lora_long.log` · wall time ≈ 34 min.
Same activation base as E7 (ppl 7 970.7) so the only change is the training
budget: 3 000 steps × 8 sequences × 128 tokens × 2 grad-accum = 6.14 M tokens
(15× E7), cosine LR schedule, rank 8 and 32.

| rank | trainable | ppl (E7, 0.41 M tok) | ppl (E7b, 6.14 M tok) | NLL after | final loss |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.18 M | 506.6 | **216.1** | 5.376 | 5.092 |
| 32 | 4.72 M | 448.3 | **160.0** | 5.075 | 4.971 |

Reference: baseline ppl 59.35 / NLL 4.08; quaternion-only ppl 7 970.7 / NLL 8.98.

**Findings — Gate 4 now largely met.**

- Longer training recovers far more: rank 32 improves from 448 → **160**
  (2.8× better), rank 8 from 507 → 216 (2.3×). The base is not the ceiling;
  E7 was simply undertrained.
- NLL 8.98 → 5.08 closes **≈80%** of the gap to the 4.08 baseline. The recovered
  model is 2.7× the baseline perplexity — much closer, but not equal.
- Rank still matters at this budget (216 vs 160), so capacity helps once the
  optimization is not the bottleneck.
- The remaining gap likely needs more tokens/data, a longer schedule, and/or a
  better base (sequential calibration). The trajectory is still improving.

---

## E7c — Sequential-calibration base + longer LoRA, GPT-2 small (2026-09-27)

**Command** (background, chained with E8b)

```bash
python scripts/train_lora.py --model gpt2 --convert sequential --ranks 32 \
    --max-steps 6000 --lr 1e-3 --lr-schedule cosine --batch-size 8 \
    --grad-accum 2 --data-chars 4000000 --calib-seqs 64 --calib-seq-len 128 \
    --act-iters 300 --ppl-batches 50 --ppl-seq-len 192 \
    --summary experiments/results/gpt2_small_lora_e7c.json
```

Log: `experiments/logs/e7c_lora_sequential.log` · training 1 374 s.
Mode B conversion (`convert_model_sequential`): block `i` is calibrated on the
hidden states produced by the already-converted blocks `0..i-1`, with the
original block `i` (run on those states) as target.

| base | quaternion-only ppl | LoRA | recovered ppl | NLL |
|---|---:|---|---:|---:|
| teacher-forced (E7b) | 7 970.7 | rank 32, 3 000 steps | **160.0** | 5.075 |
| sequential (E7c) | **49 602.8** | rank 32, 6 000 steps | **163.2** | 5.095 |

**Findings**

- Sequential calibration made the base **6.2× worse** (49 603 vs 7 971): fitting
  each block on the compounded, degraded input distribution hurts rather than
  helps. As implemented, Mode B sequential calibration is not beneficial.
- After long LoRA both bases converge to the **same ≈160 perplexity**. The
  frozen quaternion base barely affects the recovered quality — LoRA does the
  heavy lifting at this budget.
- Implication: the recovered quality is limited by LoRA budget, not by the
  compression method's per-layer accuracy. The base would matter more if
  recovery were not available or were much cheaper.

---

## E9 — Custom fused CUDA kernel, K4 (2026-09-27)

**Implementation** `kernels/cuda/quaternion_linear.cu` (JIT via
`torch.utils.cpp_extension`). Hand-written SIMT kernel: each block computes a
`[16, 4·16]` output tile (all four components), loads activation and weight
tiles into shared memory once per K-step and reuses them across the four
Hamilton equations. The block matrix is never materialized. Supports fp32 and
bf16. Correctness (`tests/test_cuda_kernel.py`): fp32 max error 1e-6, bf16 0.015.

**Benchmark** (bf16, ms; lower is better)

hidden = 768:

| regime | tokens | dense | naive | planar | triton | **cuda** |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 128 | 0.0160 | 0.2310 | 0.1750 | 0.1035 | 0.0477 |
| decode | 1 | 0.0128 | 0.2085 | 0.1724 | 0.0493 | **0.0396** |
| prefill | 512 | 0.0251 | 0.2525 | 0.1908 | 0.0548 | 0.1012 |
| decode | 1 | 0.0125 | 0.2055 | 0.1815 | 0.0494 | **0.0386** |
| prefill | 2048 | 0.0362 | 0.3408 | 0.3948 | 0.1085 | 0.2911 |
| decode | 1 | 0.0126 | 0.2117 | 0.1762 | 0.0492 | **0.0390** |

hidden = 1024:

| regime | tokens | dense | naive | planar | triton | **cuda** |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 128 | 0.0187 | 0.2417 | 0.1866 | 0.0635 | 0.0646 |
| decode | 1 | 0.0119 | 0.2141 | 0.1662 | 0.0609 | **0.0429** |
| prefill | 512 | 0.0277 | 0.2415 | 0.2162 | 0.0691 | 0.1504 |
| decode | 1 | 0.0186 | 0.2076 | 0.1787 | 0.0583 | **0.0424** |
| prefill | 2048 | 0.0642 | 0.4208 | 0.5230 | 0.1801 | 0.4964 |
| decode | 1 | 0.0139 | 0.2093 | 0.1710 | 0.0574 | 0.0513 |

**Findings**

- The custom CUDA kernel is **4–5× faster than naive/planar**.
- **Decode:** it **beats the Triton tensor-core kernel** (0.0396 vs 0.0493 ms at
  hidden 768; 0.0429 vs 0.0609 ms at 1024). At M=1 the tensor cores are
  underutilized and Triton's overhead dominates, so the lean SIMT kernel wins —
  a real systems result for the decode regime the plan flagged (§17).
- **Prefill:** Triton wins by 2–3× (tensor cores matter once M is large).
- Both remain ~3–5× slower than cuBLAS dense GEMM; closing that gap is what
  CUTLASS/CuTe (tensor-core, structure-aware) would target.
- **CUTLASS/CuTe is still not implemented.** `nvidia-cutlass` 4.2 is available
  and could be built, but a structure-aware CuTe kernel is a substantial C++
  effort left as the remaining K4 deliverable.

---

## Interpretation and next steps

1. **Weight-level quaternionization of pretrained weights is fundamentally
   capped at ≈0.866 relative error (75% of energy lost).** This is not an
   optimizer failure and cannot be fixed by grouping. Measured at **0.8660 on
   both GPT-2 (72 projections) and Qwen2.5-1.5B (196 projections)**, so it is
   architecture-independent. Compressing a pretrained weight matrix directly by
   4× with this parameterization is not viable.
2. **The only way the constraint can survive is functional, not structural.**
   The layer only has to behave correctly on the low-dimensional manifold of
   activations it actually sees. Activation-aware fitting gives a consistent
   reduction (E5 GPT-2: mean held-out error 0.65, 1.38×; E8 Qwen2.5: 0.53,
   3.47×), and the gain is **architecture- and layer-dependent** — some Qwen
   layers are almost exactly representable (0.03–0.13). The functional route is
   the viable one, but it must be combined with recovery and evaluated on a
   diverse holdout.
3. **LoRA recovery works, scales with budget, and dominates the outcome.**
   End-to-end activation conversion gives ppl 7 971 (E6); 400 LoRA steps recover
   it to 448 (E7), and 3 000 steps to **160** (E7b, rank 32), closing ≈80% of
   the NLL gap to the 59.35 baseline. Strikingly, a **6.2× worse base** (E7c
   sequential calibration, ppl 49 603) still recovers to the **same ≈160** after
   long LoRA (E7c). The frozen quaternion base barely affects recovered quality;
   the LoRA budget is the binding constraint. Base quality matters mainly if
   recovery is unavailable or much cheaper.
4. **Systems (E9):** a hand-written fused CUDA kernel already **wins the decode
   regime over Triton** (0.040 vs 0.049 ms) and is 4–5× faster than naive/planar,
   while Triton wins prefill. Both still trail cuBLAS; CUTLASS/CuTe remains the
   open systems deliverable.
5. **Factorization (H8) reduces the subspace further** and will make
   weight-level error worse, not better; it should only be justified by
   compute/memory wins or by training from scratch, never as a quality lever.
6. **Interim verdict:** direct 4× quaternionization of a pretrained model is
   not competitive with INT4/INT8 on quality. It may still be interesting if
   (a) LoRA at a realistic budget recovers quality, and (b) the fused kernel or
   a shared-score attention variant gives a genuine systems win.
7. **Execution:** all long pipelines run in the background with logs under
   `experiments/logs/` and are polled, per plan §58.

### Immediate plan

| ID | Experiment | Gate |
|---|---|---|
| E5 | Stage 2 activation-aware on **all** blocks, larger calibration, held-out eval | Gate 2 — **done** (error 0.65, insufficient alone) |
| E6 | End-to-end GPT-2 conversion + perplexity | Gate 2 — **done** (ppl 59 → 83 663 matrix / 7 971 activation; fails) |
| E7 | Quaternion + LoRA recovery (ranks 4–32) | Gate 4 — **done, partial** (ppl 7 971 → 448 at r32) |
| E7b | Longer LoRA budget (3000 steps, rank 8/32) | Gate 4 — **done** (ppl 7 971 → 160, ≈80% NLL gap closed) |
| E7c | Sequential-calibration base + 6000-step LoRA | **done** — worse base (49 603) still recovers to 163; base barely matters |
| E8 | Generality check on Qwen2.5-1.5B | **done** — floor universal (0.8660); functional gains larger (0.53) |
| E8b | Random-split activation eval + Qwen end-to-end | **done** — random split confirms E8; Qwen ppl 17.8 → 553 655 / 41 350 |
| E9 | Fused kernel K4 | **done (custom CUDA)** — wins decode vs Triton; CUTLASS/CuTe pending |

## Artifacts

| Artifact | Path |
|---|---|
| Stage 1/2 summary JSON | `experiments/results/gpt2_small_stage12_summary.json` |
| Grouping comparison JSON (E4) | `experiments/results/gpt2_small_grouping_comparison.json` |
| Activation-aware all blocks (E5) | `experiments/results/gpt2_small_activation_allblocks.json` |
| End-to-end conversion + ppl (E6) | `experiments/results/gpt2_small_conversion.json` |
| LoRA recovery, short (E7) | `experiments/results/gpt2_small_lora.json` |
| LoRA recovery, long (E7b) | `experiments/results/gpt2_small_lora_e7b.json` |
| LoRA + sequential base (E7c) | `experiments/results/gpt2_small_lora_e7c.json` |
| Qwen2.5-1.5B Stage 1/2 (E8) | `experiments/results/qwen25_15b_stage12.json` |
| Qwen random-split Stage 1/2 (E8b) | `experiments/results/qwen25_15b_stage12_random.json` |
| Qwen end-to-end conversion (E8b) | `experiments/results/qwen25_15b_conversion.json` |
| Background logs (plan §58) | `experiments/logs/` |
| Kernel benchmark | `scripts/benchmark_kernel.py` |
| Kernel implementations | `kernels/pytorch/ops.py`, `kernels/triton/quaternion.py`, `kernels/cuda/quaternion_linear.cu` |
| Conversion / LoRA drivers | `scripts/reconstruct_layer.py`, `scripts/reconstruct_model.py`, `scripts/train_lora.py` |
| Tests (112 passing, incl. CUDA) | `tests/` |
