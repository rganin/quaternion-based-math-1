# Quaternion Transformer Compression & Hardware Efficiency Research Plan

## 0. Executive Summary

This project investigates whether pretrained Transformer models can be compressed through quaternion-valued linear layers and subsequently recovered with LoRA, while also determining whether quaternion arithmetic can become computationally competitive when implemented as a hardware-aware fused kernel.

The primary recommendation is:

> **Do not start with Gemma-4-31B. Start with a small model, prove the mathematics and reconstruction method, then optimize the kernel, and only then scale to 7–9B and finally 27–31B.**

The project should be treated as two coupled research problems:

1. **Representation problem:** How accurately can pretrained real-valued Transformer linear layers be projected onto a quaternion-constrained parameterization?
2. **Systems problem:** Can the quaternion computation be mapped efficiently to GPU matrix hardware so that the parameter reduction does not result in slower inference?

The initial target should be a ~1B Transformer such as Gemma 3 1B, followed by a 7–9B model and only then a 27–31B model.

The DGX Spark / Blackwell platform is a good target. The first implementation should use PyTorch for correctness, followed by Triton and/or CUTLASS/CuTe for optimized kernels.

---

# 1. Research Hypotheses

## H1 — Quaternion parameterization can approximate pretrained Transformer layers

A conventional real-valued matrix:

```text
W ∈ R^(out × in)
```

is replaced by quaternion parameters:

```text
Wq = Wr + Wi*i + Wj*j + Wk*k
```

with approximately 4× fewer scalar parameters for the same logical dimensions.

The corresponding real-valued transformation has a structured block form:

```text
[ Wr  -Wi  -Wj  -Wk ]
[ Wi   Wr  -Wk   Wj ]
[ Wj   Wk   Wr  -Wi ]
[ Wk  -Wj   Wi   Wr ]
```

The first research question is how much reconstruction error this structural constraint introduces.

---

## H2 — Layer-wise reconstruction is not sufficient

Independent optimization of `q_proj`, `k_proj`, `v_proj`, etc. may produce locally good approximations while producing poor attention/block behavior.

Therefore compare:

1. Individual linear reconstruction.
2. Attention sub-block reconstruction.
3. MLP sub-block reconstruction.
4. Full Transformer block reconstruction.

The primary optimization target should eventually be the **block output**, not only the output of each individual linear layer.

---

## H3 — Intermediate-state correction is essential

If a reconstructed block produces:

```text
Yq = Y + error
```

then feeding the original activation into every later layer hides accumulated reconstruction error.

The full conversion experiment should therefore support:

```text
Original:
X0 → Block0 → X1 → Block1 → X2 → ...

Quaternion:
X0 → QBlock0 → X1q → QBlock1 → X2q → ...
```

Calibration should progressively use reconstructed states.

---

## H4 — Naive quaternion arithmetic will probably be slower

A naive implementation may expand into:

- many elementwise multiplications;
- many additions;
- reshaping/transposition;
- multiple CUDA kernel launches;
- additional memory traffic.

This can be substantially slower than a conventional GEMM even though the quaternion representation has fewer parameters.

This does **not** prove quaternion computation is intrinsically slower.

The real systems question is whether the structured operation can be transformed into an efficient sequence/fusion of GPU matrix operations.

---

## H5 — Hardware-aware implementation may reverse the performance result

The quaternion transformation can be written as four coupled real-valued equations:

```text
Yr = Wr Xr - Wi Xi - Wj Xj - Wk Xk

Yi = Wi Xr + Wr Xi - Wk Xj + Wj Xk

Yj = Wj Xr + Wk Xi + Wr Xj - Wi Xk

Yk = Wk Xr - Wj Xi + Wi Xj + Wr Xk
```

These contain matrix multiplications.

The systems research question becomes:

> Can the structured computation be scheduled/fused so that the GPU's matrix-multiply hardware is utilized efficiently while reducing memory movement and kernel-launch overhead?

---

# 2. Overall Research Ladder

Do not jump directly to 31B.

```text
Stage 0
Mathematical/unit-test validation
        ↓
Stage 1
Single Linear layer reconstruction
        ↓
Stage 2
Single Transformer block
        ↓
Stage 3
Gemma 3 1B full-model reconstruction
        ↓
Stage 4
Kernel optimization
        ↓
Stage 5
7–9B model
        ↓
Stage 6
27–31B model
        ↓
Stage 7
Quaternion + LoRA recovery
```

The ordering is deliberate.

The initial goal is **fast iteration**, not maximum model size.

---

# 3. Stage 0 — Mathematical Validation

## Goal

Prove that the quaternion implementation is mathematically correct before involving a real Transformer.

## 3.1 Implement `QuaternionLinear`

Represent:

```python
Wr
Wi
Wj
Wk
```

as four real-valued tensors.

Do not initially construct the full real block matrix.

Implement:

```python
quaternion_linear(x, Wr, Wi, Wj, Wk)
```

where the last dimension of `x` is interpreted as groups of four real channels.

---

## 3.2 Define the channel mapping explicitly

For hidden dimension `D`, require:

```text
D % 4 == 0
```

Pack:

```text
[x0, x1, x2, x3]
```

into:

```text
x0 + x1*i + x2*j + x3*k
```

Repeat for all groups.

Do not leave the packing convention implicit.

The exact convention must be documented and tested.

---

## 3.3 Reference implementation

Create a slow but obviously correct implementation using explicit quaternion multiplication.

Example conceptual operation:

```text
(a,b,c,d) * (x,y,z,w)
```

produces:

```text
r = ax - by - cz - dw
i = ay + bx + cw - dz
j = az - bw + cx + dy
k = aw + bz - cy + dx
```

Use this as the correctness oracle.

---

## 3.4 Real block-matrix implementation

Construct the equivalent real block matrix:

```text
[ Wr  -Wi  -Wj  -Wk ]
[ Wi   Wr  -Wk   Wj ]
[ Wj   Wk   Wr  -Wi ]
[ Wk  -Wj   Wi   Wr ]
```

and compare:

```text
Quaternion implementation
        vs
ordinary matrix multiplication
```

Tests should verify:

```text
max_abs_error
mean_abs_error
relative_error
```

at FP32 first.

Then test FP16/BF16.

---

## 3.5 Unit tests

Required tests:

- random input equivalence;
- random weight equivalence;
- batch dimensions;
- sequence dimensions;
- FP32;
- BF16;
- FP16;
- odd batch/sequence dimensions;
- invalid hidden dimensions;
- gradient correctness;
- autograd gradient comparison against finite differences or a reference implementation where practical.

Do not proceed until these pass.

---

# 4. Stage 1 — Pure Matrix Approximation

Before touching an LLM, answer:

> How well can an arbitrary real matrix be approximated by a quaternion-structured matrix?

## 4.1 Generate matrices

Test:

```text
256 × 256
512 × 512
1024 × 1024
4096 × 4096
```

and rectangular matrices where practical.

Use:

1. random Gaussian matrices;
2. pretrained Transformer matrices;
3. matrices from different projection types.

---

## 4.2 Optimization

Given:

```text
W_original
```

optimize:

```text
Wr, Wi, Wj, Wk
```

so that:

```text
Q(Wq) ≈ W_original
```

Use AdamW initially.

Record:

```text
MSE
relative Frobenius error
cosine similarity
parameter count
compression ratio
optimization time
```

---

## 4.3 Critical experiment: learned channel grouping

Do not assume that arbitrary groups of four pretrained channels are optimal.

Compare:

### A. Contiguous grouping

```text
0,1,2,3
4,5,6,7
...
```

### B. Random fixed permutation

```text
P(X)
```

### C. Learned permutation or channel transform

Investigate whether a learned orthogonal/channel permutation before quaternionization substantially reduces error.

This could become one of the most important research directions.

---

# 5. Stage 2 — Activation-Aware Linear Reconstruction

Matrix approximation alone is insufficient.

A Transformer only cares about how the weight behaves on its actual activation distribution.

For a pretrained linear layer:

```text
Y = X W^T
```

collect real activation samples `X`.

Then optimize:

```text
Yq = QuaternionLinear(X)
```

against:

```text
Y
```

---

## 5.1 Calibration dataset

Start with:

```text
256 sequences
512 sequences
1024 sequences
```

Do not immediately assume 2048 is necessary.

Test calibration-size scaling.

Use real text tokenized with the target model's tokenizer.

Keep a held-out calibration set.

---

## 5.2 Metrics

For every layer report:

```text
MSE
RMSE
relative L2 error
cosine similarity
maximum absolute error
```

Also report:

```text
error by token position
error distribution
error by activation magnitude
```

---

# 6. Stage 3 — Transformer Block Reconstruction

This is where the project becomes meaningful.

Given a pretrained block:

```text
X → OriginalBlock(X) → Y
```

construct:

```text
X → QuaternionBlock(X) → Yq
```

and optimize all selected quaternion parameters jointly.

---

## 6.1 Start with attention only

Test:

```text
q_proj
k_proj
v_proj
o_proj
```

Keep everything else identical.

Optimize against the original attention output.

---

## 6.2 Then MLP

Test:

```text
gate_proj
up_proj
down_proj
```

Keep norms and activations unchanged.

---

## 6.3 Then complete block

Optimize all eligible projections jointly.

Objective:

```text
L = MSE(Yq, Y)
    + λ * (1 - cosine(Yq, Y))
```

Tune `λ`.

Test pure MSE versus combined objective.

---

# 7. Progressive Intermediate-State Correction

Implement two modes.

## Mode A — Teacher-forced calibration

Every block receives original hidden states.

```text
X_original → QBlock → compare to original output
```

This measures local reconstruction quality.

## Mode B — Sequential calibration

After converting block N:

```text
X0
 ↓
QBlock0
 ↓
X1q
 ↓
QBlock1
 ↓
X2q
```

The next block is calibrated using `X1q`.

This measures and addresses error compounding.

---

# 8. Stage 4 — Full Gemma 3 1B Conversion

Only after the previous stages work.

Recommended first target:

**Gemma 3 1B** or another approximately 1B Transformer with a compatible architecture.

---

## 8.1 Do not quaternionize everything

Initially keep these unchanged:

```text
embedding
RMSNorm
RoPE
LM head
```

Initially quaternionize only:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Then perform ablations.

---

## 8.2 Required ablations

Compare:

```text
A: q/k/v
B: q/k/v/o
C: MLP only
D: attention + MLP
E: attention + MLP + LM head
```

This determines where quaternionization causes the most damage.

---

# 9. Full-Model Evaluation

Do not judge success using reconstruction MSE alone.

Measure:

## Language modeling

```text
Perplexity
validation loss
```

## Logits

```text
logit cosine similarity
top-1 agreement
top-5 agreement
KL divergence
```

## Generation

Use a fixed prompt suite.

Measure:

```text
exact/semantic agreement where appropriate
qualitative output
repetition
hallucination changes
```

## Downstream tasks

Use a small representative benchmark set.

---

# 10. Baselines

Always compare against:

```text
FP16/BF16 baseline
INT8
INT4
Quaternion
Quaternion + LoRA
```

The comparison should include:

```text
parameter count
disk size
RAM/VRAM
latency
tokens/sec
perplexity
quality
```

This is critical because a 4× parameter reduction is not automatically better than a well-optimized INT4 implementation.

---

# 11. Stage 5 — Hardware Benchmark

This should be treated as an independent project.

Benchmark the same mathematical operation using:

```text
1. PyTorch reference
2. optimized PyTorch implementation
3. torch.compile where applicable
4. Triton kernel
5. CUTLASS/CuTe kernel
```

Target platform:

```text
DGX Spark / Blackwell
```

---

# 12. Benchmark Methodology

Use fixed matrix sizes representative of Transformer projections.

Example:

```text
hidden sizes:
1024
2048
4096
8192

batch × sequence combinations:
1 × 128
1 × 512
1 × 2048
4 × 512
```

Measure:

```text
kernel latency
end-to-end layer latency
throughput
GPU utilization
memory bandwidth
kernel launches
peak memory
```

Use warmup iterations and synchronized CUDA timing.

Never benchmark a single execution.

---

# 13. Naive Quaternion Baseline

Implement the easiest correct version first.

For example:

```text
split X
calculate Yr
calculate Yi
calculate Yj
calculate Yk
concatenate
```

Do not optimize it initially.

This establishes the baseline that demonstrates why naive quaternion arithmetic can be slow.

---

# 14. Triton Version

Implement a fused kernel.

Goals:

- reduce intermediate tensors;
- fuse elementwise operations;
- minimize global-memory traffic;
- reduce kernel launches;
- exploit tiled matrix operations where appropriate.

Do not assume Triton will automatically beat cuBLAS.

Measure it.

---

# 15. CUTLASS/CuTe Version

This is the most important hardware experiment.

Investigate whether the quaternion equations can be expressed as structured matrix operations that use Blackwell Tensor Core paths.

Potential design:

```text
Xr Xi Xj Xk
 │  │  │  │
 └──┴──┴──┴──── fused quaternion kernel
               │
               ▼
             Y
```

Avoid constructing the full block matrix explicitly.

The full real matrix destroys much of the parameter/memory advantage.

---

# 16. Kernel Optimization Questions

Test these independently:

### Q1
Can multiple quaternion products be fused?

### Q2
Can the four output components remain in registers/shared memory?

### Q3
Can weight tiles be reused across the four equations?

### Q4
Can input tiles be reused across multiple quaternion components?

### Q5
Can Tensor Core MMA instructions be used directly?

### Q6
What is the arithmetic intensity?

### Q7
Is the operation compute-bound or memory-bound?

### Q8
Does the optimal implementation differ between prefill and decode?

---

# 17. Very Important: Prefill vs Decode

Do not report a single "tokens/sec" number.

Transformer inference has two very different regimes.

## Prefill

Large matrices:

```text
sequence length = hundreds/thousands
```

Quaternion matrix multiplication may potentially utilize GPU compute efficiently.

## Decode

Typically:

```text
batch × 1 token
```

and weight memory traffic becomes extremely important.

A representation with 4× fewer weights could potentially have a much larger advantage during decode if the implementation is memory-bandwidth efficient.

Therefore benchmark separately:

```text
prefill latency
decode latency
tokens/sec
```

---

# 18. Stage 6 — LoRA Recovery

After establishing the quaternion-only baseline, add standard real-valued LoRA.

Architecture:

```text
                 ┌── Quaternion base ──┐
X ───────────────┤                     ├── Y
                 └── LoRA correction ──┘
```

Use:

```text
A: [rank, in_features]
B: [out_features, rank]

scale = alpha / rank
```

Initial experiments:

```text
rank = 4
rank = 8
rank = 16
rank = 32

alpha = 2 × rank
```

Freeze quaternion parameters for the primary recovery experiment.

---

# 19. LoRA Experiments

Compare:

```text
FP16 baseline
Quaternion
Quaternion + LoRA rank 4
Quaternion + LoRA rank 8
Quaternion + LoRA rank 16
Quaternion + LoRA rank 32
```

Then optionally test:

```text
Quaternion + partially trainable quaternion parameters + LoRA
```

This determines whether LoRA is sufficient to recover quaternionization loss.

---

# 20. Hugging Face Training

Use standard Hugging Face Trainer/PEFT infrastructure where possible.

The training configuration should:

- freeze quaternion parameters;
- expose only LoRA parameters;
- use BF16 where supported;
- use gradient checkpointing if necessary;
- use small micro-batches;
- use gradient accumulation;
- avoid optimizer states for frozen parameters.

Do not use AdamW state for the frozen quaternion base.

---

# 21. Memory Accounting

Report memory separately for:

```text
FP16 model
INT8 model
INT4 model
Quaternion weights
Quaternion + LoRA
training model
inference model
```

Also distinguish:

```text
parameter storage
activation memory
KV cache
optimizer states
temporary workspace
```

Do not call the model "4× smaller" solely because the quaternion parameters are 4× fewer.

---

# 22. Scaling to 7–9B

Only start this stage if the 1B results show:

1. acceptable perplexity degradation;
2. reproducible reconstruction;
3. stable sequential calibration;
4. useful LoRA recovery;
5. a plausible kernel performance path.

Repeat the exact same benchmark.

Do not change the methodology merely to make the larger model work.

---

# 23. Scaling to 27–31B

At this point the original Gemma-4-31B experiment becomes appropriate.

The 31B implementation should reuse the validated framework.

Important requirements:

- checkpoint after every block;
- resume conversion;
- shard calibration data;
- avoid holding unnecessary original and quaternion tensors simultaneously;
- process one block at a time;
- optionally offload unused blocks;
- log every reconstruction metric;
- save FP/BF16 reference checkpoints separately.

---

# 24. Production Conversion Pipeline

The final converter should look conceptually like:

```text
HF checkpoint
      │
      ▼
load model
      │
      ▼
collect calibration activations
      │
      ▼
for each Transformer block:
      │
      ├── capture reference output
      ├── construct quaternion block
      ├── optimize quaternion parameters
      ├── evaluate reconstruction
      ├── save block checkpoint
      └── feed quaternion output forward
      │
      ▼
evaluate complete model
      │
      ▼
optional LoRA recovery
      │
      ▼
save deployment checkpoint
```

---

# 25. Checkpoint Format

Store:

```text
model_config
quaternionization_config
channel permutation
Wr
Wi
Wj
Wk
LoRA parameters
calibration statistics
conversion version
git commit
dtype
```

Also record:

```text
original layer name
input dimension
output dimension
grouping convention
relative reconstruction error
```

This makes experiments reproducible.

---

# 26. Experiment Tracking

Every run should have a unique experiment ID.

Example:

```text
gemma1b_q_v001
gemma1b_q_v002_learned_perm
gemma1b_q_block_v001
gemma1b_q_lora_r16
gemma1b_q_triton_v001
gemma1b_q_cutlass_v001
```

Save machine-readable results:

```json
{
  "model": "...",
  "layer": "...",
  "method": "...",
  "calibration_samples": 1024,
  "mse": 0.0,
  "cosine": 0.0,
  "ppl": 0.0,
  "latency_ms": 0.0,
  "tokens_per_sec": 0.0
}
```

---

# 27. Recommended Repository Structure

```text
quaternion-transformer/
│
├── README.md
├── pyproject.toml
│
├── quaternion/
│   ├── reference.py
│   ├── linear.py
│   ├── packing.py
│   ├── block.py
│   └── lora.py
│
├── conversion/
│   ├── collect_calibration.py
│   ├── fit_linear.py
│   ├── fit_block.py
│   ├── convert_model.py
│   └── checkpoint.py
│
├── kernels/
│   ├── pytorch/
│   ├── triton/
│   └── cutlass/
│
├── evaluation/
│   ├── perplexity.py
│   ├── logits.py
│   ├── generation.py
│   └── benchmarks.py
│
├── tests/
│   ├── test_hamilton.py
│   ├── test_matrix_equivalence.py
│   ├── test_gradients.py
│   └── test_kernels.py
│
├── experiments/
│   ├── configs/
│   └── results/
│
└── scripts/
    ├── benchmark_kernel.py
    ├── reconstruct_layer.py
    ├── reconstruct_block.py
    └── reconstruct_model.py
```

---

# 28. Initial Milestone Checklist

Do not start with model conversion.

First achieve:

```text
[ ] Quaternion multiplication mathematically verified
[ ] Real block-matrix equivalence verified
[ ] Autograd verified
[ ] Matrix reconstruction implemented
[ ] Matrix reconstruction benchmarked
[ ] Activation-aware reconstruction implemented
[ ] Calibration train/validation split implemented
[ ] Single Transformer block reconstructed
[ ] Sequential state propagation implemented
[ ] Metrics logged automatically
```

Then:

```text
[ ] Gemma 3 1B conversion
[ ] Full-model perplexity evaluation
[ ] Quaternion-only baseline
[ ] Quaternion + LoRA
```

Then:

```text
[ ] Naive GPU benchmark
[ ] Triton benchmark
[ ] CUTLASS/CuTe benchmark
[ ] Prefill benchmark
[ ] Decode benchmark
```

Only then:

```text
[ ] 7–9B
[ ] 27–31B
```

---

# 29. Success Criteria

Define success before running the experiments.

## Mathematical success

Quaternion reconstruction should achieve high cosine similarity and acceptably low relative error on real Transformer activations.

## Model-quality success

The converted model should retain useful language-model quality without LoRA.

LoRA should substantially recover the degradation.

## Compression success

Measure actual serialized parameter storage rather than theoretical parameter count.

## Systems success

A successful kernel should demonstrate that the quaternion implementation is not inherently tied to the poor performance of the naive implementation.

The strongest result would be:

```text
Quaternion:
~4× fewer parameters
+
acceptable perplexity
+
LoRA recovery
+
competitive or superior inference performance
```

---

# 30. Most Important Experimental Matrix

The first serious result table should look approximately like this:

| Model | Representation | Params | Size | PPL | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| 1B | BF16 | baseline | baseline | baseline | baseline | baseline |
| 1B | INT8 | | | | | |
| 1B | INT4 | | | | | |
| 1B | Quaternion | | | | | |
| 1B | Quaternion + LoRA | | | | | |
| 1B | Quaternion + optimized kernel | | | | | |

Then repeat at 7–9B and 27–31B.

---

# 31. Recommended OpenCode Development Prompt

Use the following as the initial prompt rather than asking OpenCode to build the entire 31B converter immediately.

```text
Act as a research engineer specializing in Transformer compression,
quaternion neural networks, CUDA/Triton optimization, and PyTorch.

Build a research-grade repository for investigating quaternion
parameterization of pretrained Transformer linear layers.

IMPORTANT:
Do NOT start with Gemma-4-31B.
The first target is a ~1B Transformer such as Gemma 3 1B.

The project has two goals:

1. Determine how accurately pretrained Transformer layers can be
   represented by quaternion-valued linear layers.

2. Determine whether quaternion computation can be implemented
   efficiently enough on NVIDIA Blackwell to become competitive with
   conventional dense GEMM.

PHASE 0:
Implement mathematically correct quaternion arithmetic.

Requirements:
- Four real tensors Wr, Wi, Wj, Wk.
- Explicit real-to-quaternion channel packing.
- Explicit Hamilton product.
- Reference implementation.
- Equivalent real block-matrix implementation.
- Unit tests proving numerical equivalence.
- FP32, BF16 and FP16 tests.
- Gradient tests.

PHASE 1:
Implement arbitrary real matrix → quaternion matrix reconstruction.

Given W_original, optimize Wr/Wi/Wj/Wk.

Report:
- MSE
- relative Frobenius error
- cosine similarity
- parameter count
- compression ratio
- optimization time

Test random matrices and pretrained Transformer matrices.

PHASE 2:
Implement activation-aware reconstruction.

Collect real activations X from a calibration corpus.

Optimize:

QuaternionLinear(X) ≈ OriginalLinear(X)

Support:
- 256
- 512
- 1024
- 2048 calibration sequences

Maintain a held-out calibration set.

Report:
- MSE
- relative L2 error
- cosine similarity
- max error

PHASE 3:
Implement Transformer block reconstruction.

Start with:
q_proj/k_proj/v_proj/o_proj

Then:
gate_proj/up_proj/down_proj

Then all eligible projections jointly.

Optimize block output:

QuaternionBlock(X) ≈ OriginalBlock(X)

Use a combined MSE + cosine objective.

PHASE 4:
Implement sequential calibration.

After converting block N, feed its quaternion output into
block N+1 calibration.

Support both:
- teacher-forced calibration
- sequential calibration

Report error accumulation across depth.

PHASE 5:
Convert the full ~1B model.

Initially do NOT quaternionize:
- embeddings
- norms
- RoPE
- LM head

Evaluate:
- perplexity
- logit cosine similarity
- top-k agreement
- generation quality

PHASE 6:
Implement LoRA recovery.

Quaternion parameters are frozen.

LoRA remains standard real-valued:

Y = QuaternionLinear(X) + scale * ((X @ A.T) @ B.T)

Test ranks:
4, 8, 16, 32.

Compare quaternion-only versus quaternion + LoRA.

PHASE 7:
Build GPU benchmark implementations.

Implement:
1. naive PyTorch
2. torch.compile where useful
3. Triton
4. CUTLASS/CuTe where practical

Do NOT construct the full real block matrix for production kernels.

Exploit quaternion structure directly.

Benchmark:
- latency
- throughput
- memory
- kernel launches
- GPU utilization
- prefill
- decode

Use NVIDIA Blackwell/DGX Spark as the primary hardware target.

PHASE 8:
Only after the 1B experiments are successful, support 7–9B.
Only after that, support 27–31B.

The repository must:
- checkpoint after each layer/block
- resume failed conversions
- log all metrics
- save machine-readable JSON results
- preserve conversion configuration
- have deterministic evaluation scripts
- contain unit tests for all core mathematical operations

Do not optimize prematurely.
First establish a correct reference implementation,
then optimize the kernel.

The final README must explain:
- quaternion representation
- channel packing
- Hamilton product
- real block-matrix equivalence
- reconstruction procedure
- sequential calibration
- LoRA recovery
- GPU kernel strategy
- benchmark methodology
- limitations
```

---

# 32. Final Recommendation

The most important change from the original plan is the order:

```text
NOT:

31B → quaternionize → LoRA → hope it works

BUT:

math
 ↓
matrix reconstruction
 ↓
activation reconstruction
 ↓
block reconstruction
 ↓
1B model
 ↓
quality evaluation
 ↓
kernel optimization
 ↓
LoRA recovery
 ↓
7–9B
 ↓
27–31B
```

This minimizes wasted compute while preserving the possibility of a much more interesting result.

The **highest-risk assumption** is not whether quaternion weights can be trained. It is whether the quaternion constraint preserves enough of a pretrained Transformer's useful linear transformations.

The **second highest-risk assumption** is whether the resulting structured operation can be mapped efficiently to Blackwell hardware.

Those two questions should therefore be answered independently before committing to a full 31B conversion.


---

# 33. Extended Research Direction — Quantum-Inspired Quaternion Architecture

The project should be expanded beyond simple quaternion compression.

The goal is now to investigate whether the quaternion representation can reduce **both parameter storage and computation**, rather than merely replacing real weights with four quaternion components.

Three additional research directions should be tested:

1. **Structured/planar quaternion GEMM**
2. **Quaternion shared-score attention**
3. **Factorized quaternion weights**

A fourth, lower-priority direction is:

4. **Unit-quaternion / rotation-constrained transformations**

Actual quantum-computing inference should remain outside the main project because the classical input/output encoding and measurement overhead make it unlikely to provide a practical speedup for ordinary Transformer inference.

---

# 34. Extended Architecture Hypothesis

The long-term target architecture is:

```text
                 Quaternion Transformer
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Quaternion       Shared-score    Factorized
      Linear          Attention       Quaternion
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    LoRA recovery
```

The three components attack different costs.

## A. Quaternion Linear

Reduces parameter count and potentially weight-memory bandwidth.

## B. Shared-score quaternion attention

Attempts to avoid redundant attention-score calculations across quaternion components.

## C. Factorized quaternion weights

Attempts to reduce the quaternion parameters further through shared low-rank factors.

## D. LoRA

Provides task/domain-specific recovery after aggressive compression.

---

# 35. New Research Hypothesis H6 — Planar Quaternion GEMM

Do not construct the full real Hamilton matrix.

Instead retain:

```text
Wr
Wi
Wj
Wk
```

as separate matrices and compute:

```text
Yr = Wr Xr - Wi Xi - Wj Xj - Wk Xk

Yi = Wi Xr + Wr Xi - Wk Xj + Wj Xk

Yj = Wj Xr + Wk Xi + Wr Xj - Wi Xk

Yk = Wk Xr - Wj Xi + Wi Xj + Wr Xk
```

The implementation should attempt to reuse input and weight tiles and fuse the operations.

### Hypothesis

A fused planar quaternion GEMM can substantially outperform a naive quaternion implementation and may become competitive with conventional FP16/BF16 GEMM for some Transformer shapes.

The key metrics are:

```text
latency
effective TFLOPS
memory bandwidth
kernel launches
GPU utilization
```

---

# 36. New Research Hypothesis H7 — Quaternion Shared-Score Attention

Investigate whether attention scores can be represented using a quaternion inner product whose real/scalar component becomes a shared real-valued attention score.

Conceptually:

```text
Q = Qr + Qi*i + Qj*j + Qk*k
K = Kr + Ki*i + Kj*j + Kk*k
```

Compute a scalar similarity from the quaternion product:

```text
S = Re(Q ⊗ K†)
```

and use:

```text
softmax(S)
```

once for all quaternion components.

Instead of:

```text
four independent attention-score calculations
four score tensors
four softmax operations
```

attempt:

```text
one shared score
one softmax
quaternion-valued V
```

### Hypothesis

Shared-score attention can reduce attention computation and memory traffic while preserving enough representational capacity.

This must be tested independently of quaternion weight compression.

---

# 37. New Research Hypothesis H8 — Factorized Quaternion Weights

Investigate whether quaternion components themselves have exploitable low-rank/shared structure.

Candidate forms:

```text
Wr = Ar B
Wi = Ai B
Wj = Aj B
Wk = Ak B
```

or:

```text
Wr = A Br
Wi = A Bi
Wj = A Bj
Wk = A Bk
```

or a more general shared-factor decomposition.

The goal is to determine whether the quaternion representation can be compressed beyond the basic ~4× parameter reduction.

Possible targets:

```text
4×
6×
8×
12×
16×
```

while monitoring perplexity.

Do not assume higher compression is useful. Measure quality and runtime.

---

# 38. New Research Hypothesis H9 — Unit Quaternion / Rotation Constraints

As a lower-priority experiment, investigate constraining quaternion weights to represent normalized rotations:

```text
q = r + i*i + j*j + k*k

||q|| ≈ 1
```

with a separate scalar magnitude where appropriate.

Conceptually:

```text
W ≈ scale × quaternion_rotation
```

This could provide stronger parameter sharing and potentially more efficient computation.

However, this is primarily a compression/representation hypothesis, not an expected immediate speed optimization.

Only investigate it after the basic quaternion and factorized models work.

---

# 39. Small-Model Experimental Ladder

The new architecture should be tested in increasingly complex experiments.

Do not begin with full-model training.

## Experiment Group A — Pure Quaternion Mathematics

Target:

```text
synthetic matrices
```

Goal:

- prove correctness;
- measure approximation error;
- establish arithmetic cost.

---

## Experiment Group B — Single Linear Layer

Target:

```text
small Transformer
```

Replace one linear layer.

Test:

```text
baseline Linear
Quaternion Linear
Quaternion Linear + learned channel permutation
```

Measure:

```text
output MSE
cosine similarity
latency
memory
```

---

## Experiment Group C — Single Attention Block

Replace:

```text
q_proj
k_proj
v_proj
o_proj
```

Keep the rest unchanged.

Compare:

```text
Real attention
Quaternion attention
Quaternion shared-score attention
```

This isolates the attention hypothesis.

---

## Experiment Group D — Single MLP Block

Replace:

```text
gate_proj
up_proj
down_proj
```

Compare:

```text
real
quaternion
factorized quaternion
```

This isolates weight reconstruction.

---

## Experiment Group E — Full Small Model

Use a model in the approximately:

```text
100M–500M
```

range first if convenient.

Create:

```text
FP16/BF16 baseline
Quaternion
Quaternion + LoRA
Quaternion shared-score
Factorized quaternion
Factorized quaternion + LoRA
```

Measure quality and speed.

---

# 40. Recommended Small Models

Use models with straightforward Transformer architectures.

Good experimental candidates include:

```text
GPT-2 small / medium
TinyLlama
Qwen2.5 0.5B
SmolLM / SmolLM2 small variants
Gemma 3 270M
Gemma 3 1B
```

The exact model is less important than keeping the architecture well understood.

Use **one primary model** for reproducibility and one secondary model to test generality.

Recommended:

```text
Primary:
Gemma 3 270M or 1B

Secondary:
Qwen2.5 0.5B
```

---

# 41. Experiment Group F — Kernel Benchmark Before Full Model

This is critical.

Do not wait until the complete Transformer exists to benchmark the hardware implementation.

Construct synthetic matrices matching the target model.

Example:

```text
hidden = 512
hidden = 768
hidden = 1024
hidden = 2048
```

Test:

```text
batch × sequence:
1 × 128
1 × 512
1 × 2048
4 × 512
8 × 512
```

Implement:

```text
1. PyTorch dense GEMM
2. naive PyTorch quaternion
3. optimized planar PyTorch
4. Triton quaternion
5. CUTLASS/CuTe quaternion
```

---

# 42. Kernel Experiment K1 — Naive Quaternion

Implementation:

```text
split X into Xr/Xi/Xj/Xk
calculate four output components
concatenate
```

Purpose:

Establish the deliberately simple baseline.

Record:

```text
latency
memory
kernel count
```

---

# 43. Kernel Experiment K2 — Planar Quaternion

Keep:

```text
Wr Wi Wj Wk
```

separate.

Do not materialize the Hamilton matrix.

Attempt to reuse tiles across:

```text
Wr Xr
Wr Xi
Wr Xj
Wr Xk
```

and corresponding terms.

Compare directly against K1.

---

# 44. Kernel Experiment K3 — Triton

Build a fused Triton kernel.

Goals:

```text
reduce intermediate tensors
reduce global-memory traffic
reduce kernel launches
reuse weight tiles
reuse activation tiles
```

Compare against K2.

Do not assume Triton will outperform cuBLAS.

---

# 45. Kernel Experiment K4 — CUTLASS/CuTe

Investigate a Tensor-Core-aware implementation.

Primary question:

> Can the quaternion structure be mapped onto Blackwell matrix-multiply hardware without expanding the full Hamilton matrix?

Measure:

```text
effective FLOPS
memory bandwidth
occupancy
latency
```

This is the highest-priority performance experiment after correctness.

---

# 46. Kernel Experiment K5 — Decode vs Prefill

Every kernel must be tested separately for:

## Prefill

```text
sequence = 128
512
2048
```

## Decode

```text
sequence contribution = 1 token
```

Decode is especially important because the reduced quaternion weight representation may reduce memory bandwidth.

Report:

```text
prefill tok/s
decode tok/s
latency/token
```

---

# 47. Attention Experiment A1 — Standard Quaternion Attention

Start with normal attention computation using quaternion Q/K/V.

Do not change the attention score formula yet.

Measure:

```text
PPL
attention output MSE
latency
memory
```

This establishes the baseline.

---

# 48. Attention Experiment A2 — Shared-Score Attention

Replace multiple component-specific score calculations with a scalar quaternion inner-product-derived score.

Compare:

```text
Standard real attention
Quaternion attention
Quaternion shared-score attention
```

Keep Q/K/V projection weights identical across the comparison where possible.

This isolates the benefit of the new score computation.

---

# 49. Attention Experiment A3 — Quality/Speed Tradeoff

Sweep:

```text
attention heads
head dimension
sequence length
```

Measure whether shared-score attention becomes more beneficial as sequence length increases.

The expected advantage should be evaluated separately for:

```text
short context
long context
```

because attention's quadratic score calculation becomes increasingly important at longer context lengths.

---

# 50. Factorization Experiment F1 — Independent Factors

For each quaternion component:

```text
Wr = Ar Br
Wi = Ai Bi
Wj = Aj Bj
Wk = Ak Bk
```

Sweep rank:

```text
r = 1%
2%
5%
10%
20%
of original matrix dimension
```

Compare reconstruction error.

---

# 51. Factorization Experiment F2 — Shared Factors

Test:

```text
Wr = Ar B
Wi = Ai B
Wj = Aj B
Wk = Ak B
```

and:

```text
Wr = A Br
Wi = A Bi
Wj = A Bj
Wk = A Bk
```

Compare parameter counts and runtime.

The shared-factor versions are particularly interesting because they may create opportunities for computation reuse.

---

# 52. Factorization Experiment F3 — Quaternion + Factorization

Compare:

```text
Real
Quaternion
Quaternion + independent low rank
Quaternion + shared input factor
Quaternion + shared output factor
```

Measure:

```text
compression
PPL
latency
```

The goal is not maximum compression in isolation.

The goal is:

> maximum quality per unit of computation and memory.

---

# 53. LoRA Recovery Experiments

For every major compressed architecture test:

```text
base compressed model
+
LoRA rank 4
+
LoRA rank 8
+
LoRA rank 16
+
LoRA rank 32
```

Compare:

```text
Quaternion + LoRA
Shared-score Quaternion + LoRA
Factorized Quaternion + LoRA
Factorized Shared-score Quaternion + LoRA
```

This determines whether LoRA can compensate for increasingly aggressive structural constraints.

---

# 54. Experiment Matrix for the Small Model

The first complete experiment should look like:

| ID | Architecture | Compression | LoRA | Shared Score | Factorized | Kernel |
|---|---|---:|---:|---:|---:|---|
| B0 | BF16 baseline | 1× | No | No | No | cuBLAS |
| Q1 | Quaternion | ~4× | No | No | No | PyTorch |
| Q2 | Quaternion | ~4× | No | No | No | Triton |
| Q3 | Quaternion | ~4× | No | No | No | CUTLASS |
| A1 | Quaternion attention | ~4× | No | No | No | optimized |
| A2 | Q shared-score | ~4× | No | Yes | No | optimized |
| F1 | Q factorized | >4× | No | No | Yes | optimized |
| L1 | Q + LoRA | ~4× | Yes | No | No | optimized |
| L2 | Q shared-score + LoRA | ~4× | Yes | Yes | No | optimized |
| L3 | Q factorized + LoRA | >4× | Yes | No | Yes | optimized |

---

# 55. Decision Gates

Do not proceed to larger models unless the small-model experiments pass the following gates.

## Gate 1 — Mathematical

```text
[ ] Quaternion implementation exactly matches reference
[ ] Gradients are correct
```

## Gate 2 — Reconstruction

```text
[ ] Quaternion linear reconstruction is stable
[ ] Sequential block reconstruction is stable
[ ] Quality degradation is measurable and understood
```

## Gate 3 — Compression

```text
[ ] Real serialized model size is significantly reduced
```

## Gate 4 — LoRA

```text
[ ] LoRA recovers a meaningful portion of quality loss
```

## Gate 5 — Hardware

```text
[ ] Optimized quaternion kernel substantially beats naive implementation
```

## Gate 6 — Architectural innovation

At least one of:

```text
[ ] shared-score attention improves speed
[ ] factorization improves quality/compression tradeoff
[ ] optimized quaternion GEMM becomes competitive with baseline
```

Only then move to 7–9B.

---

# 56. Priority Ranking of New Ideas

The recommended order is:

## Priority 1 — Fused/planar quaternion GEMM

Highest probability of producing a practical inference improvement.

```text
★★★★★
```

## Priority 2 — Shared-score quaternion attention

Potentially significant reduction in attention-score computation.

```text
★★★★★
```

## Priority 3 — Factorized quaternion weights

Potentially much greater compression, but quality/runtime tradeoffs are uncertain.

```text
★★★★☆
```

## Priority 4 — Learned channel permutation

Very cheap experiment with potentially significant reconstruction benefit.

```text
★★★★☆
```

## Priority 5 — Unit quaternion / rotation constraints

Interesting mathematically, but less directly connected to immediate inference speed.

```text
★★★☆☆
```

## Priority 6 — Actual quantum-computing inference

Keep as a separate exploratory research project.

```text
★☆☆☆☆
```

---

# 57. Updated Long-Term Target

The ambitious endpoint is no longer simply:

```text
Gemma
 ↓
4× Quaternion compression
 ↓
LoRA
```

Instead investigate:

```text
                 pretrained Transformer
                          │
                          ▼
                 quaternion projection
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       learned grouping          factorization
             │                         │
             └────────────┬────────────┘
                          ▼
                 quaternion weights
                          │
                          ▼
              shared-score attention
                          │
                          ▼
               fused Blackwell kernel
                          │
                          ▼
                     LoRA recovery
```

The ideal result would be a model that is simultaneously:

```text
~4× or more parameter-efficient
+
memory-bandwidth efficient
+
competitive in perplexity
+
LoRA recoverable
+
fast at prefill
+
especially efficient during decode
```

This is a much stronger research objective than compression alone.
