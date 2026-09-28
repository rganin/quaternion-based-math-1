"""Fit quaternion parameters to a real linear operator (Stages 1-2).

Two targets are supported:

* **matrix reconstruction** (Stage 1): approximate a real matrix ``W`` by the
  quaternion block matrix ``Block(Wq)``;
* **activation-aware reconstruction** (Stage 2): approximate the outputs of a
  real layer on real activations, ``QuaternionLinear(X) ≈ X W^T``.

Because both objectives are linear in the quaternion parameters, the
matrix-reconstruction optimum has a closed form (:func:`project_onto_quaternion`).
It is provided as the default; the AdamW path (plan.md section 4.2) is retained
for comparison and for the activation-aware objective.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from quaternion.linear import quaternion_linear
from quaternion.metrics import reconstruction_report
from quaternion.packing import build_block_matrix, check_divisible

__all__ = [
    "project_onto_quaternion",
    "ReconstructionResult",
    "reconstruct_matrix",
    "reconstruct_from_activations",
]


def _project_raw(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable projection in the input's own dtype/device."""
    wr = (w[0::4, 0::4] + w[1::4, 1::4] + w[2::4, 2::4] + w[3::4, 3::4]) / 4
    wi = (-w[0::4, 1::4] + w[1::4, 0::4] - w[2::4, 3::4] + w[3::4, 2::4]) / 4
    wj = (-w[0::4, 2::4] + w[1::4, 3::4] + w[2::4, 0::4] - w[3::4, 1::4]) / 4
    wk = (-w[0::4, 3::4] - w[1::4, 2::4] + w[2::4, 1::4] + w[3::4, 0::4]) / 4
    return wr, wi, wj, wk


def project_onto_quaternion(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form least-squares projection of a real matrix onto the
    quaternion block structure.

    The optimal ``(Wr, Wi, Wj, Wk)`` for ``||Block(Wq) - W||_F`` is the signed
    average of the four block positions associated with each component.
    """
    if weight.dim() != 2:
        raise ValueError("weight must be 2D [out, in]")
    out_features, in_features = weight.shape
    check_divisible(out_features, "out_features")
    check_divisible(in_features, "in_features")
    return _project_raw(weight.double())


@dataclass
class ReconstructionResult:
    wr: torch.Tensor
    wi: torch.Tensor
    wj: torch.Tensor
    wk: torch.Tensor
    metrics: dict
    method: str
    seconds: float
    history: list[float] = field(default_factory=list)
    parameters: int = 0
    dense_parameters: int = 0
    compression_ratio: float = 0.0
    in_permutation: torch.Tensor | None = None
    out_permutation: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)


def _make_grouping(
    in_features: int,
    out_features: int,
    grouping: str,
    generator: torch.Generator,
):
    if grouping == "contiguous":
        return None, None
    if grouping == "random":
        return (
            torch.randperm(in_features, generator=generator),
            torch.randperm(out_features, generator=generator),
        )
    raise ValueError("grouping must be 'contiguous' or 'random'")


def _permute_rows_cols(weight, in_perm, out_perm):
    if in_perm is not None:
        weight = weight[:, in_perm]
    if out_perm is not None:
        weight = weight[out_perm, :]
    return weight


def _sinkhorn(logits: torch.Tensor, n_iters: int, tau: float) -> torch.Tensor:
    """Doubly-stochastic relaxation of a permutation (log-domain Sinkhorn)."""
    x = logits / tau
    for _ in range(n_iters):
        x = x - torch.logsumexp(x, dim=1, keepdim=True)
        x = x - torch.logsumexp(x, dim=0, keepdim=True)
    return x.exp()


def _round_perm(p: torch.Tensor) -> torch.Tensor:
    """Round a near-permutation matrix to an index permutation via Hungarian."""
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-p.detach().cpu().numpy())
        return torch.as_tensor(cols, dtype=torch.long)
    except ImportError:  # pragma: no cover - scipy normally available
        return p.detach().argmax(dim=1)


def _learn_permutation(
    weight: torch.Tensor,
    iters: int,
    lr: float,
    tau_start: float,
    tau_end: float,
    sinkhorn_iters: int,
    seed: int,
    device,
):
    """Learn input/output channel permutations that make ``weight`` better
    approximated by a quaternion block matrix.

    Permutations are relaxed to doubly-stochastic matrices (Sinkhorn) and the
    quaternion projection is applied in closed form inside the loop, so the
    loss is exactly the residual after optimal quaternionization.  The best
    iterate is rounded to a hard permutation with the Hungarian algorithm.
    """
    out_features, in_features = weight.shape
    dev = torch.device(device) if device is not None else weight.device
    w = weight.detach().to(device=dev, dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def init_logits(n):
        noise = 0.01 * torch.randn(n, n, generator=gen)
        return (torch.eye(n) * 5.0 + noise).to(dev).requires_grad_(True)

    logit_in = init_logits(in_features)
    logit_out = init_logits(out_features)
    opt = torch.optim.Adam([logit_in, logit_out], lr=lr)

    best = None
    history: list[float] = []
    for it in range(iters):
        frac = it / max(1, iters - 1)
        tau = tau_start * (tau_end / tau_start) ** frac
        p_in = _sinkhorn(logit_in, sinkhorn_iters, tau)
        p_out = _sinkhorn(logit_out, sinkhorn_iters, tau)
        w_perm = p_out @ w @ p_in.t()
        wq = _project_raw(w_perm)
        block = build_block_matrix(*wq)
        loss = torch.nn.functional.mse_loss(block, w_perm)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if best is None or loss.item() < best[0]:
            best = (loss.item(), p_in.detach().clone(), p_out.detach().clone())

    _, p_in, p_out = best
    return {
        "in_perm": _round_perm(p_in),
        "out_perm": _round_perm(p_out),
        "history": history,
    }


def _result(
    wr, wi, wj, wk, target, method, seconds, history, in_perm, out_perm, extras=None
) -> ReconstructionResult:
    block = build_block_matrix(wr, wi, wj, wk)
    metrics = reconstruction_report(block, target)
    dense = wr.shape[0] * 4 * wr.shape[1] * 4
    params = wr.numel() * 4
    return ReconstructionResult(
        wr=wr.detach(),
        wi=wi.detach(),
        wj=wj.detach(),
        wk=wk.detach(),
        metrics=metrics,
        method=method,
        seconds=seconds,
        history=history,
        parameters=params,
        dense_parameters=dense,
        compression_ratio=dense / params,
        in_permutation=in_perm,
        out_permutation=out_perm,
        extras=extras or {},
    )


def reconstruct_matrix(
    weight: torch.Tensor,
    method: str = "projection",
    iters: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    grouping: str = "contiguous",
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    learn_iters: int = 150,
    learn_lr: float = 0.05,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
    sinkhorn_iters: int = 20,
    device=None,
) -> ReconstructionResult:
    """Approximate ``weight`` with a quaternion block matrix.

    ``method='projection'`` uses the closed-form optimum.  ``method='adamw'``
    runs the plan's AdamW optimization.  ``method='learned'`` additionally
    learns input/output channel permutations (plan.md section 4.3) to improve
    the grouping before the closed-form projection.
    """
    if method not in ("projection", "adamw", "learned"):
        raise ValueError("method must be 'projection', 'adamw' or 'learned'")
    if weight.dim() != 2:
        raise ValueError("weight must be 2D [out, in]")
    out_features, in_features = weight.shape
    check_divisible(out_features, "out_features")
    check_divisible(in_features, "in_features")

    start = time.perf_counter()
    if method == "learned":
        learned = _learn_permutation(
            weight.detach(),
            iters=learn_iters,
            lr=learn_lr,
            tau_start=tau_start,
            tau_end=tau_end,
            sinkhorn_iters=sinkhorn_iters,
            seed=seed,
            device=device,
        )
        in_perm, out_perm = learned["in_perm"], learned["out_perm"]
        target = _permute_rows_cols(weight.detach().double(), in_perm, out_perm)
        wr, wi, wj, wk = project_onto_quaternion(target)
        history = learned["history"]
        extras = {
            "grouping": "learned",
            "learn_iters": learn_iters,
            "learn_lr": learn_lr,
            "tau_start": tau_start,
            "tau_end": tau_end,
        }
    else:
        generator = torch.Generator().manual_seed(seed)
        in_perm, out_perm = _make_grouping(
            in_features, out_features, grouping, generator
        )
        target = _permute_rows_cols(weight.detach().double(), in_perm, out_perm)
        if method == "projection":
            wr, wi, wj, wk = project_onto_quaternion(target)
            history = []
        else:
            wr, wi, wj, wk, history = _optimize_block(
                target, iters=iters, lr=lr, weight_decay=weight_decay, seed=seed
            )
        extras = {"grouping": grouping, "iters": iters, "lr": lr}
    seconds = time.perf_counter() - start

    return _result(
        wr,
        wi,
        wj,
        wk,
        target,
        method,
        seconds,
        history,
        in_perm,
        out_perm,
        extras=extras,
    )


def reconstruct_from_activations(
    x: torch.Tensor,
    y: torch.Tensor,
    iters: int = 1000,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ReconstructionResult:
    """Activation-aware reconstruction: fit ``QuaternionLinear(X) ≈ Y``.

    ``dtype`` defaults to float64 for stable reference fits; pass
    ``torch.float32`` and a CUDA ``device`` for fast large-scale runs.
    """
    if x.dim() < 2 or y.dim() < 2:
        raise ValueError("x and y must have at least 2 dimensions")
    in_features = x.shape[-1]
    out_features = y.shape[-1]
    check_divisible(in_features, "in_features")
    check_divisible(out_features, "out_features")
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError("x and y must share leading dimensions")

    xd = x.reshape(-1, in_features).to(device=device, dtype=dtype)
    yd = y.reshape(-1, out_features).to(device=device, dtype=dtype)

    start = time.perf_counter()
    wr, wi, wj, wk, history = _optimize_linear(
        xd, yd, iters=iters, lr=lr, weight_decay=weight_decay, seed=seed
    )
    seconds = time.perf_counter() - start

    pred = quaternion_linear(xd, wr, wi, wj, wk)
    metrics = reconstruction_report(pred, yd)
    dense = out_features * in_features
    params = wr.numel() * 4
    return ReconstructionResult(
        wr=wr.detach(),
        wi=wi.detach(),
        wj=wj.detach(),
        wk=wk.detach(),
        metrics=metrics,
        method="activation_adamw",
        seconds=seconds,
        history=history,
        parameters=params,
        dense_parameters=dense,
        compression_ratio=dense / params,
        extras={"iters": iters, "lr": lr},
    )


def _init_components(out_g, in_g, dtype, seed, device=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 / (in_g) ** 0.5
    comps = []
    for _ in range(4):
        t = (
            torch.randn(out_g, in_g, generator=g, dtype=torch.float64)
            .to(dtype=dtype, device=device)
            * scale
        )
        t.requires_grad_(True)
        comps.append(t)
    return comps


def _optimize_block(target, iters, lr, weight_decay, seed):
    out_g = target.shape[0] // 4
    in_g = target.shape[1] // 4
    params = _init_components(out_g, in_g, target.dtype, seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    for _ in range(iters):
        opt.zero_grad()
        block = build_block_matrix(*params)
        loss = torch.nn.functional.mse_loss(block, target)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return (*params, history)


def _optimize_linear(x, y, iters, lr, weight_decay, seed):
    out_g = y.shape[1] // 4
    in_g = x.shape[1] // 4
    params = _init_components(out_g, in_g, y.dtype, seed, device=y.device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    for _ in range(iters):
        opt.zero_grad()
        pred = quaternion_linear(x, *params)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return (*params, history)
