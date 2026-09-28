"""End-to-end model conversion + perplexity (Stage 4 / E6).

Converts every eligible projection to a quaternion layer and measures perplexity
for the baseline and each conversion method.

Usage (background, plan §58):
    nohup python scripts/reconstruct_model.py --model gpt2 \
        --variants matrix activation \
        --summary experiments/results/gpt2_small_conversion.json \
        > experiments/logs/e6_conversion.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from conversion.convert_model import convert_model
from evaluation.perplexity import compute_perplexity, load_text


def load_model(name):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)


def make_calibration(
    tokenizer, n_seqs, seq_len, dataset="wikitext", split="train", random_windows=False, seed=0
):
    text = load_text(dataset, split, limit_chars=2_000_000)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if random_windows:
        gen = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, ids.numel() - seq_len, (n_seqs,), generator=gen)
        return torch.stack([ids[s : s + seq_len] for s in starts])
    need = n_seqs * seq_len
    if ids.numel() < need:
        raise ValueError("calibration corpus too short")
    return ids[:need].view(n_seqs, seq_len)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--variants", nargs="+", default=["matrix", "activation"])
    parser.add_argument("--calib-seqs", type=int, default=64)
    parser.add_argument("--calib-seq-len", type=int, default=128)
    parser.add_argument("--calib-random", action="store_true")
    parser.add_argument("--act-iters", type=int, default=300)
    parser.add_argument("--ppl-batches", type=int, default=50)
    parser.add_argument("--ppl-seq-len", type=int, default=192)
    parser.add_argument("--summary", default="experiments/results/gpt2_small_conversion.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = load_text("wikitext", "test")

    results = {"model": args.model, "config": vars(args), "variants": {}}

    # Baseline
    print("[e6] evaluating baseline", flush=True)
    model = load_model(args.model).to(device).eval()
    t0 = time.perf_counter()
    base = compute_perplexity(
        model, tokenizer, text, seq_len=args.ppl_seq_len,
        max_batches=args.ppl_batches, device=device,
    )
    base["seconds"] = time.perf_counter() - t0
    print(f"[e6] baseline ppl={base['perplexity']:.4f}", flush=True)
    results["variants"]["baseline"] = base
    del model
    torch.cuda.empty_cache()

    calib = None
    for variant in args.variants:
        print(f"[e6] converting variant={variant}", flush=True)
        model = load_model(args.model).to(device).eval()
        if variant == "activation":
            if calib is None:
                calib = make_calibration(
                    tokenizer, args.calib_seqs, args.calib_seq_len,
                    random_windows=args.calib_random,
                ).to(device)
            _, records = convert_model(
                model, method="activation", calibration_tokens=calib,
                iters=args.act_iters, device=device,
            )
        else:
            _, records = convert_model(model, method="matrix", device=device)

        t0 = time.perf_counter()
        ppl = compute_perplexity(
            model, tokenizer, text, seq_len=args.ppl_seq_len,
            max_batches=args.ppl_batches, device=device,
        )
        ppl["seconds"] = time.perf_counter() - t0
        ppl["n_converted"] = len(records)
        ppl["mean_train_rel"] = (
            sum(r.metrics.get("relative_frobenius_error", 0.0) for r in records)
            / len(records)
            if records and records[0].method == "activation"
            else None
        )
        print(
            f"[e6] variant={variant} ppl={ppl['perplexity']:.4f} "
            f"nll={ppl['nll']:.4f} n={len(records)}",
            flush=True,
        )
        results["variants"][variant] = ppl
        del model
        torch.cuda.empty_cache()

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(results, indent=2))
    print(f"[e6] saved -> {args.summary}", flush=True)


if __name__ == "__main__":
    main()
