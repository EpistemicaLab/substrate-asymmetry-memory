#!/usr/bin/env python3
"""
V3g_cache_teacher.py
====================

Pre-compute teacher activations for all (persona, qa_idx) pairs in the train
split, save to disk. Training then loads from cache instead of running the
teacher forward each step (which would be ~3× slower).

Per persona p and Q/A index q, we compute:
  base.model([backstory_p, prompt_q, answer_q]) → hidden_states
  extract activations at layers [8,17,26,35] over Q+A positions only
  save (n_personae * n_qa, len(MATCH_LAYERS), max_qa_len, hidden_dim) bf16 tensor
  + qa_lens (n_personae * n_qa,) int

Cache file: runs/V3g_distill/teacher_acts_train.pt
Format:
  {
    "match_layers": [8,17,26,35],
    "personae_ids": [str, ...],       # length P
    "qa_per_persona": int (=6),
    "qa_lens": int tensor (P*Q,),
    "max_qa_len": int,
    "hidden_dim": int,
    "acts": bf16 tensor (P*Q, len(M), max_qa_len, H),
    "row_to_persona_idx": int tensor (P*Q,),
    "row_to_qa_idx": int tensor (P*Q,),
  }

Index: row r ↔ persona_idx = r // Q, qa_idx = r % Q.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from V1_oracle_prefix import PROMPT_TEMPLATE, load_model_and_tokenizer
from V3a_train import load_personae, encode_persona_inputs, build_qa_batch
from V3g_train import MATCH_LAYER_INDICES, ActivationCollector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONAE = ROOT / "runs" / "V3_personae" / "personae.json"
DEFAULT_OUT = ROOT / "runs" / "V3g_distill" / "teacher_acts_train.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--personae", default=str(DEFAULT_PERSONAE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--max-bs-len", type=int, default=1024)
    p.add_argument("--smoke", action="store_true",
                   help="Cache only first 5 train personae")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[V3g.cache] loading base + tokenizer ...", flush=True)
    base, tok = load_model_and_tokenizer()
    base.eval()

    print(f"[V3g.cache] loading personae from {args.personae}", flush=True)
    train_p, _ = load_personae(Path(args.personae))
    if args.smoke:
        train_p = train_p[:5]
    print(f"[V3g.cache] caching activations for {len(train_p)} train personae", flush=True)

    train_packed = encode_persona_inputs(train_p, tok, max_bs_len=args.max_bs_len)
    P = len(train_packed)
    Q = len(train_packed[0]["main_qa"])
    print(f"[V3g.cache] {P} personae × {Q} Q/A = {P*Q} rows", flush=True)

    collector = ActivationCollector()
    matched_modules = [base.model.layers[i] for i in MATCH_LAYER_INDICES]
    collector.attach(matched_modules)

    H = base.config.hidden_size
    L = len(MATCH_LAYER_INDICES)

    # First pass: discover max_qa_len
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    qa_lens = []
    for p_idx, p in enumerate(train_packed):
        for q_idx, qa in enumerate(p["main_qa"]):
            prompt_str = PROMPT_TEMPLATE.format(q=qa["q"])
            p_ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=64).input_ids.squeeze(0)
            a_str = " " + qa["a"].strip()
            a_ids = tok(a_str, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=31).input_ids.squeeze(0)
            qa_len = p_ids.size(0) + a_ids.size(0) + 1  # +1 for eos
            qa_lens.append(qa_len)

    max_qa_len = max(qa_lens)
    print(f"[V3g.cache] max_qa_len = {max_qa_len}", flush=True)

    # Allocate cache tensor on CPU (bf16)
    n_rows = P * Q
    acts_cache = torch.zeros(n_rows, L, max_qa_len, H, dtype=torch.bfloat16)
    qa_lens_t = torch.tensor(qa_lens, dtype=torch.long)

    print(f"[V3g.cache] allocated cache: {acts_cache.numel() * 2 / 1e9:.2f} GB", flush=True)

    t0 = time.time()
    # Process one persona at a time (its 6 Q/As share the same backstory)
    for p_idx, p in enumerate(train_packed):
        # Build a "batch" of size 1 persona × 6 Q/A
        single_persona_batch = [p]
        batch = build_qa_batch(single_persona_batch, tok)

        # Build teacher input_ids: [bs, q, a] per row
        bs_ids = batch["backstory_ids"][0].cuda()
        N = len(batch["prompt_ids"])  # = Q
        rows_ids = []
        starts = []
        lens = []
        for n in range(N):
            p_ids = batch["prompt_ids"][n].cuda()
            a_ids = batch["answer_ids"][n].cuda()
            full = torch.cat([bs_ids, p_ids, a_ids])
            rows_ids.append(full)
            starts.append(bs_ids.size(0))
            lens.append(p_ids.size(0) + a_ids.size(0))

        max_T = max(r.size(0) for r in rows_ids)
        input_ids = torch.full((N, max_T), pad_id, dtype=torch.long, device="cuda")
        attn = torch.zeros(N, max_T, dtype=torch.long, device="cuda")
        for n, r in enumerate(rows_ids):
            input_ids[n, : r.size(0)] = r
            attn[n, : r.size(0)] = 1

        collector.reset()
        with torch.no_grad():
            _ = base.model(input_ids=input_ids, attention_mask=attn)

        # Extract Q+A slices and write to cache
        for n in range(N):
            row_idx = p_idx * Q + n
            s = starts[n]
            L_qa = lens[n]
            L_clamped = min(L_qa, max_qa_len)
            for li, layer_act in enumerate(collector.acts):
                slice_ = layer_act[n, s : s + L_clamped]  # (L_qa, H)
                acts_cache[row_idx, li, :L_clamped] = slice_.to(torch.bfloat16).cpu()

        if (p_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[V3g.cache] {p_idx+1}/{P} personae  elapsed={elapsed:.0f}s", flush=True)

    collector.remove()

    # Save
    persona_ids = [p["id"] for p in train_packed]
    blob = {
        "match_layers": MATCH_LAYER_INDICES,
        "personae_ids": persona_ids,
        "qa_per_persona": Q,
        "qa_lens": qa_lens_t,
        "max_qa_len": max_qa_len,
        "hidden_dim": H,
        "acts": acts_cache,
    }
    print(f"[V3g.cache] saving to {out_path} ...", flush=True)
    torch.save(blob, out_path)
    print(f"[V3g.cache] done in {time.time()-t0:.0f}s. cache size on disk:", flush=True)
    print(f"  {out_path.stat().st_size / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
