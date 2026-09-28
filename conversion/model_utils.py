"""Utilities for discovering and extracting Transformer projections.

Supports both modern ``nn.Linear`` projections (q/k/v/o/gate/up/down) and
GPT-2-style fused ``Conv1D`` layers (``c_attn`` split into q/k/v, ``c_fc``,
``c_proj``).  Weights are always returned as ``[out_features, in_features]``
real tensors, regardless of the underlying module layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["Projection", "iter_projections", "module_weight_out_in"]

_PROJ_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass
class Projection:
    name: str
    kind: str
    weight: torch.Tensor  # [out_features, in_features]
    module: nn.Module | None = None


def _is_linear_like(mod: nn.Module) -> bool:
    if isinstance(mod, nn.Linear):
        return True
    name = type(mod).__name__
    return name == "Conv1D" and hasattr(mod, "weight") and mod.weight.dim() == 2


def module_weight_out_in(mod: nn.Module) -> torch.Tensor:
    """Return a ``[out, in]`` weight for a Linear or GPT-2 Conv1D module."""
    w = mod.weight
    if isinstance(mod, nn.Linear):
        return w.detach()
    # Conv1D stores [in_features, out_features].
    return w.detach().t()


def iter_projections(model: nn.Module, split_fused: bool = True):
    """Yield :class:`Projection` records for all eligible linear layers."""
    for name, mod in model.named_modules():
        if not _is_linear_like(mod):
            continue
        if name.endswith("lm_head") or ".lm_head" in name:
            continue
        leaf = name.split(".")[-1]

        if leaf in _PROJ_SUFFIXES:
            yield Projection(name, leaf, module_weight_out_in(mod).clone(), mod)
        elif leaf == "c_attn":
            w = module_weight_out_in(mod)
            if split_fused:
                chunk = w.shape[0] // 3
                for idx, kind in enumerate(("q_proj", "k_proj", "v_proj")):
                    yield Projection(
                        f"{name}.{kind}", kind, w[idx * chunk : (idx + 1) * chunk].clone(), mod
                    )
            else:
                yield Projection(name, "qkv_proj", w.clone(), mod)
        elif leaf in ("c_fc", "c_proj"):
            if leaf == "c_fc":
                kind = "up_proj"
            else:
                kind = "o_proj" if ".attn." in name else "down_proj"
            yield Projection(name, kind, module_weight_out_in(mod).clone(), mod)
