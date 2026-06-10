#!/usr/bin/env python3
"""exp31 runner — Phase L: blind-preference generation for WritingPrompts.

Mirrors exp28's per-persona pipeline but generates 150-tok continuations
under C_lora and C_rag (instead of scoring LL). Outputs go to
runs/31_strict_judge/<pid>/gens.json for downstream judging.

Adapters were not saved in exp28; this reuses the exact same training
hyperparameters (epochs=3 lr=2e-4 r=64 alpha=128 batch=2, q/k/v/o).

Usage:
    .venv/bin/python experiments/31_runner.py
    .venv/bin/python experiments/31_runner.py --limit 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PERSONAE_PATH = ROOT / "runs" / "28_writingprompts" / "dataset.json"
OUT_DIR = ROOT / "runs" / "31_strict_judge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_KW = dict(max_new_tokens=150, do_sample=True, temperature=0.7,
              top_p=0.95)
GEN_SEED = 42

# Reuse exp28 pipeline (load_base, retrieval, training, render_prefix)
sys.path.insert(0, str(ROOT / "experiments"))
spec28 = importlib.util.spec_from_file_location(
    "exp28", ROOT / "experiments" / "28_runner.py"
)
exp28 = importlib.util.module_from_spec(spec28)
spec28.loader.exec_module(exp28)


@torch.no_grad()
def generate_text(model, tok, prefix_text: str, max_prefix_tokens: int = 6000) -> str:
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    if len(prefix_ids) > max_prefix_tokens:
        prefix_ids = prefix_ids[-max_prefix_tokens:]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
    attn = torch.ones_like(ids)
    torch.manual_seed(GEN_SEED)
    out = model.generate(input_ids=ids, attention_mask=attn,
                         pad_token_id=tok.eos_token_id, **GEN_KW)
    new_ids = out[0, ids.shape[1]:].tolist()
    return tok.decode(new_ids, skip_special_tokens=True)


def run_persona(persona, tok, base_qwen):
    pid = persona["id"]
    user_dir = OUT_DIR / pid
    user_dir.mkdir(parents=True, exist_ok=True)
    out_path = user_dir / "gens.json"
    if out_path.exists():
        print(f"[31] {pid}: cached", flush=True)
        return

    t0 = time.time()
    print(f"[31] {pid}: training γ-LoRA on {len(persona['train_stories'])} stories",
          flush=True)
    lora_model, losses = exp28.train_lora_lm(
        base_qwen, tok, persona["train_stories"],
        epochs=3, lr=2e-4, r=64, alpha=128, batch=2, qid=pid,
    )
    train_t = time.time() - t0

    retrieved_per_eval = [
        exp28.rag_topk(ep["prefix"], persona["train_stories"], k=3)
        for ep in persona["eval_prompts"]
    ]

    records = []
    for ei, ep in enumerate(persona["eval_prompts"]):
        # C_rag: base model (LoRA disabled) with retrieved-context prefix
        rag_prefix = exp28.render_prefix("C_rag", ep, persona,
                                         retrieved_per_eval[ei])
        with lora_model.disable_adapter():
            rag_text = generate_text(lora_model, tok, rag_prefix)
        # C_lora: adapter active, no extra context
        lora_prefix = exp28.render_prefix("C_lora", ep, persona)
        lora_text = generate_text(lora_model, tok, lora_prefix)
        records.append({
            "persona_id": pid,
            "eval_idx": ei,
            "prefix_tail": ep["prefix"][-300:],
            "gold": ep["gold_continuation"],
            "lora_text": lora_text,
            "rag_text": rag_text,
        })
        print(f"[31] {pid} eval[{ei}] lora_chars={len(lora_text)} "
              f"rag_chars={len(rag_text)}", flush=True)

    out_path.write_text(json.dumps({
        "persona_id": pid,
        "train_seconds": train_t,
        "train_losses": losses,
        "n_records": len(records),
        "records": records,
    }))
    del lora_model
    torch.cuda.empty_cache()
    print(f"[31] {pid} done ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu", file=sys.stderr)
        sys.exit(2)

    p = json.loads(PERSONAE_PATH.read_text())
    held_out = [x for x in p["personae"] if x.get("split") == "held_out"]
    print(f"[31] loaded {len(held_out)} personae", flush=True)
    if args.start:
        held_out = held_out[args.start:]
    if args.limit:
        held_out = held_out[:args.limit]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(exp28.exp19.QWEN_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_qwen = exp28.exp19.load_base()

    t0 = time.time()
    for i, persona in enumerate(held_out):
        try:
            run_persona(persona, tok, base_qwen)
        except Exception as e:
            print(f"[31] {persona['id']} failed: {e}", flush=True)
            import traceback; traceback.print_exc()
        elapsed = time.time() - t0
        print(f"[31] {i+1}/{len(held_out)} elapsed={elapsed:.0f}s "
              f"({elapsed/(i+1):.0f}s/persona)", flush=True)
        del base_qwen
        torch.cuda.empty_cache()
        base_qwen = exp28.exp19.load_base()

    print(f"[31] DONE total elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
