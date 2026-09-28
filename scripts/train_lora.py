"""LoRA recovery of a quaternion-converted model (Stage 6 / E7).

For each LoRA rank: convert the model to quaternion projections, freeze the
quaternion base, inject a real-valued LoRA correction, train it on wikitext, and
report perplexity before and after recovery.

Usage (background, plan §58):
    nohup python scripts/train_lora.py --model gpt2 --convert activation \
        --ranks 4 8 16 32 --max-steps 400 \
        --summary experiments/results/gpt2_small_lora.json \
        > experiments/logs/e7_lora.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from conversion.convert_model import convert_model, convert_model_sequential
from evaluation.perplexity import compute_perplexity, load_text
from quaternion.lora import (
    count_trainable,
    inject_lora,
    lora_parameters,
    mark_only_lora_trainable,
)


def load_model(name):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)


class TokenBatches:
    def __init__(self, tokenizer, seq_len, batch_size, device, dataset="wikitext", limit=1_000_000):
        text = load_text(dataset, "train", limit_chars=limit)
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        n = ids.numel() // seq_len
        self.data = ids[: n * seq_len].view(n, seq_len).to(device)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.perm = torch.randperm(n)

    def steps_per_epoch(self):
        return len(self.perm) // self.batch_size

    def batch(self, step):
        n = len(self.perm)
        idx = [self.perm[(step * self.batch_size + i) % n] for i in range(self.batch_size)]
        idx = torch.tensor(idx)
        return self.data[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--convert", choices=["matrix", "activation", "sequential"], default="activation"
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--alpha-mult", type=float, default=2.0, help="alpha = mult * rank")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--data-chars", type=int, default=2_000_000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--calib-seqs", type=int, default=64)
    parser.add_argument("--calib-seq-len", type=int, default=128)
    parser.add_argument("--act-iters", type=int, default=300)
    parser.add_argument("--ppl-batches", type=int, default=50)
    parser.add_argument("--ppl-seq-len", type=int, default=192)
    parser.add_argument("--summary", default="experiments/results/gpt2_small_lora.json")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    test_text = load_text("wikitext", "test")

    def calibrate(model):
        if args.convert == "matrix":
            return convert_model(model, method="matrix", device=device)
        text = load_text("wikitext", "train", limit_chars=2_000_000)
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        need = args.calib_seqs * args.calib_seq_len
        calib = ids[:need].view(args.calib_seqs, args.calib_seq_len).to(device)
        if args.convert == "sequential":
            return convert_model_sequential(
                model, calib, iters=args.act_iters, device=device
            )
        return convert_model(
            model, method="activation", calibration_tokens=calib,
            iters=args.act_iters, device=device,
        )

    results = {"model": args.model, "config": vars(args), "runs": []}

    import copy

    print(f"[e7] converting model once ({args.convert})", flush=True)
    base_model = load_model(args.model).to(device).eval()
    base_model, _ = calibrate(base_model)
    pre = compute_perplexity(
        base_model, tokenizer, test_text, seq_len=args.ppl_seq_len,
        max_batches=args.ppl_batches, device=device,
    )
    print(f"[e7] quaternion-only ppl={pre['perplexity']:.3f}", flush=True)
    results["ppl_quaternion_only"] = pre

    for rank in args.ranks:
        model = copy.deepcopy(base_model)

        wrapped = inject_lora(model, rank=rank, alpha=args.alpha_mult * rank)
        mark_only_lora_trainable(model)
        trainable = count_trainable(model)
        print(f"[e7] rank={rank}: injected {len(wrapped)} LoRA layers, {trainable} trainable params", flush=True)

        opt = torch.optim.AdamW(
            [p for _, p in lora_parameters(model)], lr=args.lr, weight_decay=0.0
        )
        scheduler = None
        if args.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.max_steps, eta_min=args.lr * 0.1
            )
        data = TokenBatches(
            tokenizer, args.seq_len, args.batch_size, device, limit=args.data_chars
        )
        model.train()
        t0 = time.perf_counter()
        losses = []
        for step in range(args.max_steps):
            opt.zero_grad()
            loss_val = 0.0
            for _ in range(args.grad_accum):
                batch = data.batch(step)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    out = model(batch, labels=batch)
                    loss = out.loss / args.grad_accum
                loss.backward()
                loss_val += loss.item()
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in lora_parameters(model)], 1.0
            )
            opt.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss_val)
            if step % 25 == 0 or step == args.max_steps - 1:
                print(f"[e7] rank={rank} step {step}/{args.max_steps} loss={loss_val:.4f}", flush=True)
        train_seconds = time.perf_counter() - t0

        model.eval()
        post = compute_perplexity(
            model, tokenizer, test_text, seq_len=args.ppl_seq_len,
            max_batches=args.ppl_batches, device=device,
        )
        print(
            f"[e7] rank={rank}: recovered ppl={post['perplexity']:.3f} "
            f"(was {pre['perplexity']:.3f}), {train_seconds:.0f}s",
            flush=True,
        )
        results["runs"].append(
            {
                "rank": rank,
                "alpha": args.alpha_mult * rank,
                "trainable_params": trainable,
                "ppl_before": pre,
                "ppl_after": post,
                "train_seconds": train_seconds,
                "final_loss": losses[-1] if losses else None,
            }
        )
        del model
        torch.cuda.empty_cache()

    if not args.no_save:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(results, indent=2))
        print(f"[e7] saved -> {args.summary}", flush=True)


if __name__ == "__main__":
    main()
