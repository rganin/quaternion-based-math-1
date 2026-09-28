"""Perplexity / validation-loss evaluation (plan.md section 9)."""

from __future__ import annotations

import math

import torch

__all__ = ["load_text", "compute_perplexity"]


def load_text(dataset: str = "wikitext", split: str = "test", limit_chars: int | None = None) -> str:
    from datasets import load_dataset

    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join(t for t in ds["text"] if t.strip())
    else:
        ds = load_dataset(dataset, split=split)
        text = "\n".join(ds["text"])
    if limit_chars:
        text = text[:limit_chars]
    return text


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    text: str,
    seq_len: int = 192,
    max_batches: int | None = 50,
    batch_size: int = 1,
    device: str = "cuda",
) -> dict:
    """Token-level perplexity over a text corpus.

    Returns ``{"perplexity", "nll", "tokens", "batches"}``.  The mean negative
    log-likelihood is accumulated in float64 to keep large values finite.
    """
    model.eval()
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.numel() // seq_len
    n_chunks = n_chunks - (n_chunks % batch_size)
    if n_chunks == 0:
        raise ValueError("text too short for one sequence")

    total_nll = 0.0
    total_tokens = 0
    batches = 0
    for start in range(0, n_chunks, batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        chunk = ids[start : start + batch_size]
        batch = torch.stack(
            [ids[i * seq_len : (i + 1) * seq_len] for i in range(start, start + batch_size)]
        ).to(device)
        out = model(batch, labels=batch)
        n_tokens = batch.numel()
        total_nll += out.loss.double().item() * n_tokens
        total_tokens += n_tokens
        batches += 1

    mean_nll = total_nll / total_tokens if total_tokens else float("nan")
    return {
        "perplexity": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
        "nll": mean_nll,
        "tokens": total_tokens,
        "batches": batches,
    }
