"""Stage 1/2 experiments on a pretrained model's projections.

Stage 1 (``--stage matrix``): closed-form quaternion projection of every
projection matrix, reporting reconstruction metrics.

Stage 2 (``--stage activation``): activation-aware reconstruction on captured
model activations, compared against the matrix-only projection on the same
activation distribution.

Results are printed as a markdown table and stored as JSON experiment records.

Example:
    python scripts/reconstruct_layer.py --model gpt2 --stage matrix
    python scripts/reconstruct_layer.py --model gpt2 --stage activation \
        --activation-blocks 2 --calib-seqs 32 --seq-len 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from conversion.fit_linear import (
    reconstruct_from_activations,
    reconstruct_matrix,
)
from conversion.model_utils import iter_projections
from experiments.tracking import ExperimentResult, new_experiment_id


def print_table(rows, columns):
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[c]) for c in columns) + " |")


def load_calibration(tokenizer, dataset, n_seqs, seq_len, seed=0, random_windows=False):
    from datasets import load_dataset

    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in ds["text"] if t.strip())
    else:
        ds = load_dataset(dataset, split="train")
        text = "\n".join(ds[: n_seqs * 4]["text"])

    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    need = n_seqs * seq_len
    if ids.numel() < need:
        raise ValueError("calibration corpus too small; increase seqs or lower seq_len")
    if random_windows:
        gen = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, ids.numel() - seq_len, (n_seqs,), generator=gen)
        return torch.stack([ids[s : s + seq_len] for s in starts])
    return ids[:need].view(n_seqs, seq_len)


@torch.no_grad()
def collect_activations(model, modules, token_ids, batch_size=8):
    """Return {module_id: (x, y)} with inputs/outputs on CPU, float32."""
    captured: dict[int, dict[str, list]] = {}
    handles = []

    def make_hook(mod):
        def hook(_m, inputs, output):
            rec = captured[id(mod)]
            rec["x"].append(inputs[0].detach().to("cpu", torch.float32))
            out = output[0] if isinstance(output, tuple) else output
            rec["y"].append(out.detach().to("cpu", torch.float32))
        return hook

    for mod in modules:
        captured[id(mod)] = {"x": [], "y": []}
        handles.append(mod.register_forward_hook(make_hook(mod)))

    try:
        for i in range(0, token_ids.shape[0], batch_size):
            model(token_ids[i : i + batch_size])
    finally:
        for h in handles:
            h.remove()

    return {
        mid: (
            torch.cat(rec["x"], dim=0).reshape(-1, rec["x"][0].shape[-1]),
            torch.cat(rec["y"], dim=0).reshape(-1, rec["y"][0].shape[-1]),
        )
        for mid, rec in captured.items()
    }


def centered(x, y):
    return x - x.mean(dim=0, keepdim=True), y - y.mean(dim=0, keepdim=True)


def run_matrix_stage(model, args, exp_prefix):
    projections = list(iter_projections(model, split_fused=True))
    if args.max_blocks is not None:
        projections = [p for p in projections if block_index(p.name) < args.max_blocks]

    device = next(model.parameters()).device
    methods = ["contiguous"]
    if args.compare_groupings:
        methods += ["random", "learned"]

    rows = []
    for proj in projections:
        w = proj.weight.float()
        if w.shape[0] % 4 or w.shape[1] % 4:
            continue
        row = {"name": proj.name, "kind": proj.kind, "out": w.shape[0], "in": w.shape[1]}
        for meth in methods:
            if meth == "contiguous":
                res = reconstruct_matrix(w, method="projection", grouping="contiguous")
            elif meth == "random":
                res = reconstruct_matrix(w, method="projection", grouping="random", seed=args.grouping_seed)
            else:
                res = reconstruct_matrix(
                    w, method="learned", seed=args.grouping_seed,
                    learn_iters=args.learned_iters, device=device,
                )
            row[f"{meth}_rel"] = res.metrics["relative_frobenius_error"]
            row[f"{meth}_cos"] = res.metrics["cosine_similarity"]
        row["compression"] = 4.0
        rows.append(row)

    return rows


def run_activation_stage(model, tokenizer, args, exp_prefix, matrix_rows):
    blocks = args.activation_blocks
    modules = []
    seen = set()
    for proj in iter_projections(model, split_fused=False):
        if block_index(proj.name) >= blocks:
            continue
        if id(proj.module) in seen:
            continue
        if proj.weight.shape[0] % 4 or proj.weight.shape[1] % 4:
            continue
        seen.add(id(proj.module))
        modules.append(proj)

    print(
        f"[activation] collecting activations for {len(modules)} modules "
        f"({args.calib_seqs}x{args.seq_len} tokens)",
        flush=True,
    )
    token_ids = load_calibration(
        tokenizer, args.calib_dataset, args.calib_seqs, args.seq_len,
        seed=args.calib_seed, random_windows=args.calib_random,
    )
    device = next(model.parameters()).device
    captured = collect_activations(model, [p.module for p in modules], token_ids.to(device))
    print("[activation] collection done", flush=True)

    # Matrix-only errors on the same activation distribution, per kind, keyed by name.
    matrix_by_name = {r["name"]: r for r in matrix_rows}

    rows = []
    total = len(modules)
    for idx, proj in enumerate(modules, 1):
        x, y = captured[id(proj.module)]
        if proj.kind not in ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"):
            continue
        print(f"[activation] ({idx}/{total}) fitting {proj.name}", flush=True)
        xc, yc = centered(x, y)
        holdout = max(1, xc.shape[0] // 10)
        x_tr, y_tr = xc[:-holdout], yc[:-holdout]
        x_va, y_va = xc[-holdout:], yc[-holdout:]

        wq_direct = reconstruct_matrix(proj.weight.float(), method="projection")
        act = reconstruct_from_activations(
            x_tr, y_tr, iters=args.act_iters, lr=args.act_lr,
            dtype=torch.float32, device=device,
        )

        from quaternion import quaternion_linear

        dev = act.wr.device
        x_val = x_va.to(dev)
        y_val = y_va.to(dev)
        pred_direct = quaternion_linear(
            x_val,
            wq_direct.wr.to(dev, torch.float32),
            wq_direct.wi.to(dev, torch.float32),
            wq_direct.wj.to(dev, torch.float32),
            wq_direct.wk.to(dev, torch.float32),
        )
        pred_act = quaternion_linear(x_val, act.wr, act.wi, act.wj, act.wk)
        denom = y_val.norm()
        direct_err = ((pred_direct - y_val).norm() / denom).item()
        act_err = ((pred_act - y_val).norm() / denom).item()

        rows.append(
            {
                "name": proj.name,
                "kind": proj.kind,
                "out": proj.weight.shape[0],
                "in": proj.weight.shape[1],
                "matrix_val_rel": direct_err,
                "activation_val_rel": act_err,
                "improvement": direct_err / act_err if act_err > 0 else float("inf"),
            }
        )
        ExperimentResult(
            experiment_id=new_experiment_id(exp_prefix + "_act", 1, sanitize(proj.name)),
            model=args.model,
            layer=proj.name,
            method="activation_adamw",
            calibration_samples=int(x_tr.shape[0]),
            metrics={"matrix_val_rel": direct_err, "activation_val_rel": act_err},
            config={"stage": "activation", "kind": proj.kind, "iters": args.act_iters},
        ).save() if not args.no_save else None

    return rows


def block_index(name: str) -> int:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return 0


def sanitize(name: str) -> str:
    return name.replace(".", "-").replace("/", "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--stage", choices=["matrix", "activation", "both"], default="matrix")
    parser.add_argument("--max-blocks", type=int, default=None, help="limit Stage 1 blocks")
    parser.add_argument("--activation-blocks", type=int, default=2)
    parser.add_argument("--calib-dataset", default="wikitext")
    parser.add_argument("--calib-seqs", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument(
        "--calib-random", action="store_true",
        help="sample random calibration windows across the corpus (E8b)",
    )
    parser.add_argument("--act-iters", type=int, default=500)
    parser.add_argument("--act-lr", type=float, default=1e-2)
    parser.add_argument(
        "--compare-groupings",
        action="store_true",
        help="Stage 1: also evaluate random and learned channel groupings",
    )
    parser.add_argument("--grouping-seed", type=int, default=0)
    parser.add_argument("--learned-iters", type=int, default=150)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--summary", default=None, help="write combined JSON summary here")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# model={args.model} device={device} dtype=float32")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.eval()

    matrix_rows = []
    act_rows = []
    if args.stage in ("matrix", "both"):
        matrix_rows = run_matrix_stage(model, args, f"{args.model}_layer")
        methods = [
            m
            for m in ("contiguous", "random", "learned")
            if matrix_rows and f"{m}_rel" in matrix_rows[0]
        ]
        print("\n## Stage 1 — matrix reconstruction\n")
        cols = ["kind", "out", "in"] + [f"{m}_rel" for m in methods]
        print_table(
            [
                {
                    "kind": r["kind"],
                    "out": r["out"],
                    "in": r["in"],
                    **{f"{m}_rel": f"{r[f'{m}_rel']:.4f}" for m in methods},
                }
                for r in matrix_rows
            ],
            cols,
        )
        print("\n### Stage 1 aggregate (mean relative Frobenius error)\n")
        agg_cols = ["kind", "n"] + [f"{m}_mean" for m in methods]
        kinds = sorted({r["kind"] for r in matrix_rows})
        print_table(
            [
                {
                    "kind": k,
                    "n": sum(1 for r in matrix_rows if r["kind"] == k),
                    **{
                        f"{m}_mean": f"{sum(r[f'{m}_rel'] for r in matrix_rows if r['kind'] == k) / sum(1 for r in matrix_rows if r['kind'] == k):.4f}"
                        for m in methods
                    },
                }
                for k in kinds
            ],
            agg_cols,
        )
        print("\n### Stage 1 overall\n")
        print_table(
            [
                {
                    "method": m,
                    "mean": f"{sum(r[f'{m}_rel'] for r in matrix_rows) / len(matrix_rows):.4f}",
                    "min": f"{min(r[f'{m}_rel'] for r in matrix_rows):.4f}",
                    "max": f"{max(r[f'{m}_rel'] for r in matrix_rows):.4f}",
                }
                for m in methods
            ],
            ["method", "mean", "min", "max"],
        )

    if args.stage in ("activation", "both"):
        act_rows = run_activation_stage(
            model, tokenizer, args, f"{args.model}_layer", matrix_rows
        )
        print("\n## Stage 2 — activation-aware reconstruction\n")
        print_table(
            [
                {
                    "name": r["name"].split("transformer.")[-1],
                    "kind": r["kind"],
                    "matrix_val_rel": f"{r['matrix_val_rel']:.4f}",
                    "activation_val_rel": f"{r['activation_val_rel']:.4f}",
                    "improvement": f"{r['improvement']:.3f}",
                }
                for r in act_rows
            ],
            ["name", "kind", "matrix_val_rel", "activation_val_rel", "improvement"],
        )

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "config": {
                        "stage": args.stage,
                        "max_blocks": args.max_blocks,
                        "activation_blocks": args.activation_blocks,
                        "calib_seqs": args.calib_seqs,
                        "seq_len": args.seq_len,
                        "act_iters": args.act_iters,
                    },
                    "stage1_matrix": matrix_rows,
                    "stage2_activation": act_rows,
                },
                indent=2,
            )
        )
        print(f"\nsaved summary -> {args.summary}")


if __name__ == "__main__":
    main()
