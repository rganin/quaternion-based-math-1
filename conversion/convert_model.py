"""Full-model conversion to quaternion projections (Stages 4-5 / Phases 4-5).

Two conversion methods are supported:

``matrix``
    Replace each eligible projection with the closed-form quaternion block
    approximation of its weight matrix (no calibration data needed).

``activation``
    Teacher-forced activation-aware fitting: run the original model over a
    calibration corpus, capture each projection's input/output, fit a quaternion
    layer on the activation distribution, and account for the bias exactly.

Embeddings, norms, position embeddings and the LM head are left unchanged
(plan.md section 8.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from conversion.fit_linear import project_onto_quaternion, reconstruct_from_activations
from conversion.model_utils import iter_projections, module_weight_out_in
from quaternion.linear import QuaternionLinear, quaternion_linear

__all__ = [
    "ConversionRecord",
    "eligible_modules",
    "convert_model",
    "convert_model_sequential",
    "capture_activations",
]


@dataclass
class ConversionRecord:
    name: str
    in_features: int
    out_features: int
    kind: str
    method: str
    metrics: dict = field(default_factory=dict)


def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf, new)


def eligible_modules(model: nn.Module):
    """Unique eligible projection modules (fused qkv kept whole)."""
    seen: set[int] = set()
    result = []
    for proj in iter_projections(model, split_fused=False):
        w = module_weight_out_in(proj.module)
        if w.shape[0] % 4 or w.shape[1] % 4:
            continue
        if id(proj.module) in seen:
            continue
        seen.add(id(proj.module))
        result.append(proj)
    return result


def _build_layer(module: nn.Module, wr, wi, wj, wk, bias):
    w = module_weight_out_in(module)
    out_features, in_features = w.shape
    layer = QuaternionLinear(in_features, out_features, bias=bias is not None)
    with torch.no_grad():
        layer.wr.copy_(wr.to(layer.wr.dtype))
        layer.wi.copy_(wi.to(layer.wi.dtype))
        layer.wj.copy_(wj.to(layer.wj.dtype))
        layer.wk.copy_(wk.to(layer.wk.dtype))
        if bias is not None:
            layer.bias.copy_(bias)
    return layer.to(w.device, w.dtype)


def _find_blocks(model: nn.Module):
    """Return ``(container_name, ModuleList)`` for the Transformer block list."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            child_names = [n for n, _ in mod[0].named_children()]
            if any("attn" in n for n in child_names):
                return name, mod
    raise RuntimeError("could not locate Transformer block list")


@torch.no_grad()
def _capture_block_input(student, block, token_ids, device, batch_size=8):
    """Input hidden states fed to ``block`` by the (partially converted) student."""
    store = []

    def hook(_m, inputs):
        store.append(inputs[0].detach().to("cpu", torch.float32))

    handle = block.register_forward_pre_hook(hook)
    try:
        for i in range(0, token_ids.shape[0], batch_size):
            student(token_ids[i : i + batch_size])
    finally:
        handle.remove()
    return torch.cat(store, dim=0)


@torch.no_grad()
def _capture_block_projections(original_block, projections, hidden, device, batch_size=8):
    """Run the original block on ``hidden`` and capture its projection I/O."""
    store: dict[int, dict[str, list]] = {id(p.module): {"x": [], "y": []} for p in projections}
    handles = []

    def make_hook(mod):
        def fn(_m, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            store[id(mod)]["x"].append(inputs[0].detach().to("cpu", torch.float32))
            store[id(mod)]["y"].append(out.detach().to("cpu", torch.float32))
        return fn

    for p in projections:
        handles.append(p.module.register_forward_hook(make_hook(p.module)))
    try:
        for i in range(0, hidden.shape[0], batch_size):
            original_block(hidden[i : i + batch_size].to(device))
    finally:
        for h in handles:
            h.remove()
    return {
        mid: (
            torch.cat(rec["x"], 0).reshape(-1, rec["x"][0].shape[-1]),
            torch.cat(rec["y"], 0).reshape(-1, rec["y"][0].shape[-1]),
        )
        for mid, rec in store.items()
    }


def convert_model_sequential(
    model,
    calibration_tokens,
    iters: int = 300,
    lr: float = 1e-2,
    device=None,
    verbose: bool = True,
):
    """Sequential (Mode B) calibration: feed converted outputs forward.

    Blocks are converted in order; block ``i`` is calibrated on the hidden
    states produced by the already-converted blocks ``0..i-1``, with the
    *original* block ``i`` (run on those same states) providing the targets.
    This measures and addresses error compounding (plan.md section 7).
    """
    import copy

    device = device or next(model.parameters()).device
    original = copy.deepcopy(model)
    original.eval()
    student = model
    student.eval()

    block_name, blocks = _find_blocks(student)
    records = []
    for i, block in enumerate(blocks):
        hidden = _capture_block_input(student, block, calibration_tokens, device)
        original_block = original.get_submodule(f"{block_name}.{i}")
        rel_projections = [
            p
            for p in iter_projections(original_block, split_fused=False)
            if p.weight.shape[0] % 4 == 0 and p.weight.shape[1] % 4 == 0
        ]
        caps = _capture_block_projections(
            original_block, rel_projections, hidden, device
        )

        for proj in rel_projections:
            module = proj.module
            w = module_weight_out_in(module)
            out_features, in_features = w.shape
            x, y = caps[id(module)]
            x = x.to(device)
            y = y.to(device)
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            res = reconstruct_from_activations(
                x - mean_x, y - mean_y, iters=iters, lr=lr,
                dtype=torch.float32, device=device,
            )
            wr, wi, wj, wk = res.wr, res.wi, res.wj, res.wk
            pred_mean = quaternion_linear(
                mean_x,
                wr.to(torch.float32), wi.to(torch.float32),
                wj.to(torch.float32), wk.to(torch.float32),
            )
            bias = (mean_y - pred_mean).reshape(-1).detach()
            new_layer = _build_layer(module, wr, wi, wj, wk, bias)
            full_name = f"{block_name}.{i}.{proj.name}"
            _set_submodule(student, full_name, new_layer)
            records.append(
                ConversionRecord(full_name, in_features, out_features, proj.kind, "sequential", res.metrics)
            )
        if verbose:
            n = len(rel_projections)
            print(f"[sequential] block {i}/{len(blocks)-1}: converted {n} projections", flush=True)
    return student, records


@torch.no_grad()
def capture_activations(model, modules, token_ids, batch_size=8):
    """Return ``{module_id: (x, y)}`` on CPU, float32, teacher-forced."""
    store: dict[int, dict[str, list]] = {id(m): {"x": [], "y": []} for m in modules}
    handles = []

    def hook(mod):
        def fn(_m, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            store[id(mod)]["x"].append(inputs[0].detach().to("cpu", torch.float32))
            store[id(mod)]["y"].append(out.detach().to("cpu", torch.float32))
        return fn

    for m in modules:
        handles.append(m.register_forward_hook(hook(m)))
    try:
        for i in range(0, token_ids.shape[0], batch_size):
            model(token_ids[i : i + batch_size])
    finally:
        for h in handles:
            h.remove()

    result = {}
    for mid, rec in store.items():
        x = torch.cat(rec["x"], 0).reshape(-1, rec["x"][0].shape[-1])
        y = torch.cat(rec["y"], 0).reshape(-1, rec["y"][0].shape[-1])
        result[mid] = (x, y)
    return result


def convert_model(
    model,
    method: str = "matrix",
    calibration_tokens=None,
    iters: int = 300,
    lr: float = 1e-2,
    device=None,
    verbose: bool = True,
):
    """Replace eligible projections with quaternion layers in-place.

    Returns ``(model, records)``.  For ``method='activation'`` a calibration
    token tensor ``[n_seqs, seq_len]`` on the model device is required.
    """
    if method not in ("matrix", "activation"):
        raise ValueError("method must be 'matrix' or 'activation'")
    projections = eligible_modules(model)
    device = device or next(model.parameters()).device

    captured = None
    if method == "activation":
        if calibration_tokens is None:
            raise ValueError("activation conversion requires calibration_tokens")
        if verbose:
            print(f"[convert] capturing activations for {len(projections)} modules", flush=True)
        captured = capture_activations(
            model, [p.module for p in projections], calibration_tokens
        )

    records = []
    for idx, proj in enumerate(projections, 1):
        module = proj.module
        w = module_weight_out_in(module)
        out_features, in_features = w.shape
        bias = module.bias.detach().to(device) if getattr(module, "bias", None) is not None else None

        if method == "matrix":
            wr, wi, wj, wk = project_onto_quaternion(w)
            metrics = {}
        else:
            x, y = captured[id(module)]
            x = x.to(device)
            y = y.to(device)
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            res = reconstruct_from_activations(
                x - mean_x, y - mean_y, iters=iters, lr=lr,
                dtype=torch.float32, device=device,
            )
            wr, wi, wj, wk = res.wr, res.wi, res.wj, res.wk
            # exact bias so the full mapping (including mean) is reproduced
            pred_mean = quaternion_linear(
                mean_x,
                wr.to(torch.float32), wi.to(torch.float32),
                wj.to(torch.float32), wk.to(torch.float32),
            )
            bias = (mean_y - pred_mean).reshape(-1).detach()
            metrics = res.metrics

        new_layer = _build_layer(module, wr, wi, wj, wk, bias)
        _set_submodule(model, proj.name, new_layer)
        records.append(
            ConversionRecord(proj.name, in_features, out_features, proj.kind, method, metrics)
        )
        if verbose:
            err = metrics.get("relative_frobenius_error")
            suffix = f" train_rel={err:.4f}" if err is not None else ""
            print(f"[convert] ({idx}/{len(projections)}) {proj.name}{suffix}", flush=True)

    return model, records
