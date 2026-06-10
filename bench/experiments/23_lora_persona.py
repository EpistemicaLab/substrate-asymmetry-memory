#!/usr/bin/env python3
"""
23_lora_persona.py
==================

Per-persona γ-LoRA on the v0.3 held-out grid.

For each held-out persona (n=19):
  1. Extract synthetic Q/A from the persona's backstory using
     Claude Sonnet 4.6 (two-pass: factual + relational).
  2. Train r=128 LoRA on Qwen3-4B for 20 epochs (assistant-only loss).
  3. Sanity-check on a held-back slice of the synthetic Q/A.
  4. Evaluate on the persona's 6 main_qa + 6 probe2_qa.
  5. Judge predictions with Claude Sonnet 4.6.
  6. Reload base for next persona (avoid adapter stacking).

Reuses exp19's train_lora / judge / gen_chat / load_base / extract_all_parallel
via importlib (per research-experiment-discipline skill).

Outputs:
  runs/23_persona_lora/<persona_id>/synthqa.jsonl
  runs/23_persona_lora/<persona_id>/eval.json
  runs/23_persona_lora/summary.jsonl       (one line per (persona, q, eval_kind))
  runs/23_persona_lora/aggregate.json      (main_acc, probe2_acc, sanity)

Usage (smoke, 2 personae):
    export AWS_PROFILE=$YOUR_PROFILE
    .venv/bin/python experiments/23_lora_persona.py --limit 2

Usage (full):
    export AWS_PROFILE=$YOUR_PROFILE
    .venv/bin/python experiments/23_lora_persona.py
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PERSONAE_PATH = ROOT / "runs" / "V3_personae" / "personae.json"
OUT_DIR = ROOT / "runs" / "23_persona_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT_DIR / "summary.jsonl"

# Reuse exp19 helpers via importlib so we get the EXACT same training loop
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py"
)
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)


def chunk_backstory(backstory: str, max_chars: int = 4000, stride_chars: int = 3000):
    """Backstory is short (~3K chars); usually 1 chunk, occasionally 2.
    Format as a synthetic chat turn so exp19's extraction prompts work."""
    bs = backstory.strip()
    if len(bs) <= max_chars:
        return [bs]
    chunks = []
    i = 0
    while i < len(bs):
        chunks.append(bs[i:i + max_chars])
        if i + max_chars >= len(bs):
            break
        i += stride_chars
    return chunks


def render_persona_chunk(text_chunk: str) -> str:
    """Wrap raw backstory text as if it were a user turn so exp19's
    extractor (which expects '[date] role: content' format) can handle it."""
    return f"[2024-01-01] user: {text_chunk}"


def extract_persona_pairs(backstory: str, persona_id: str):
    """Two-pass synthetic Q/A extraction over backstory chunks."""
    chunks = chunk_backstory(backstory)
    rendered = [render_persona_chunk(ch) for ch in chunks]
    client = exp19.make_client()
    pairs1 = []
    for i, ch_text in enumerate(rendered):
        try:
            ps = exp19._extract(client, exp19.EXTRACT_SYSTEM_PASS1, ch_text)
            pairs1.extend(ps)
        except Exception as e:
            print(f"  [{persona_id}] pass1 chunk {i}: SKIP {e}", flush=True)
    pairs2 = []
    for i, ch_text in enumerate(rendered):
        try:
            ps = exp19._extract(client, exp19.EXTRACT_SYSTEM_PASS2, ch_text)
            pairs2.extend(ps)
        except Exception as e:
            print(f"  [{persona_id}] pass2 chunk {i}: SKIP {e}", flush=True)
    # dedup by (q, a)
    seen = set()
    merged = []
    for q, a in pairs1 + pairs2:
        key = (q.lower().strip(), a.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        merged.append({"q": q, "a": a})
    return merged, len(pairs1), len(pairs2)


def run_persona(persona, tok, judge_client):
    pid = persona["id"]
    pdir = OUT_DIR / pid
    pdir.mkdir(parents=True, exist_ok=True)

    # 1. Resume guard
    eval_path = pdir / "eval.json"
    if eval_path.exists():
        print(f"[{pid}] cached, skipping", flush=True)
        return json.loads(eval_path.read_text())

    t0 = time.time()
    print(f"\n=== {pid} ===", flush=True)

    # 2. Extract synthetic Q/A
    synth_path = pdir / "synthqa.jsonl"
    if synth_path.exists():
        pairs = [json.loads(l) for l in synth_path.read_text().splitlines() if l.strip()]
        n1, n2 = -1, -1
        print(f"  [{pid}] synth cached: {len(pairs)} pairs", flush=True)
    else:
        pairs, n1, n2 = extract_persona_pairs(persona["backstory"], pid)
        with synth_path.open("w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"  [{pid}] extracted {len(pairs)} pairs (pass1={n1}, pass2={n2})", flush=True)
    if len(pairs) < 10:
        print(f"  [{pid}] WARN: only {len(pairs)} pairs extracted", flush=True)
    t_extract = time.time() - t0

    # 3. Hold back 5 pairs for sanity check, train on the rest
    # (deterministic: take last 5 by post-dedup order)
    sanity_pairs = pairs[-5:] if len(pairs) >= 10 else []
    train_pairs = pairs[:-5] if len(pairs) >= 10 else pairs

    # 4. Load base (fresh per persona)
    print(f"  [{pid}] loading base...", flush=True)
    base = exp19.load_base()
    t_train_start = time.time()

    model, losses = exp19.train_lora(base, tok, train_pairs, qid=pid)
    t_train = time.time() - t_train_start

    # 5. Sanity check
    sanity_results = []
    for p in sanity_pairs:
        sp_text = exp19.gen_chat(tok, model, [{"role": "user", "content": p["q"]}])
        ok = exp19.judge(judge_client, p["q"], p["a"], sp_text)
        sanity_results.append({"q": p["q"], "a_gold": p["a"], "a_pred": sp_text, "ok": ok})
    sanity_acc = (sum(s["ok"] for s in sanity_results) / max(1, len(sanity_results))) if sanity_results else None
    print(f"  [{pid}] sanity_acc={sanity_acc} (n={len(sanity_results)})", flush=True)

    # 6. Eval on main_qa + probe2_qa
    eval_records = []
    for kind in ("main_qa", "probe2_qa"):
        for q in persona[kind]:
            pred = exp19.gen_chat(tok, model, [{"role": "user", "content": q["q"]}])
            correct = exp19.judge(judge_client, q["q"], q["a"], pred)
            rec = {
                "persona_id": pid,
                "eval_kind": "main" if kind == "main_qa" else "probe2",
                "q": q["q"],
                "gold": q["a"],
                "pred": pred,
                "correct": correct,
            }
            eval_records.append(rec)
            with SUMMARY.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    # 7. Save per-persona record
    out = {
        "persona_id": pid,
        "n_pairs_total": len(pairs),
        "n_train_pairs": len(train_pairs),
        "final_loss": losses[-1] if losses else None,
        "sanity_acc": sanity_acc,
        "sanity_examples": sanity_results,
        "main_correct": sum(1 for r in eval_records if r["eval_kind"] == "main" and r["correct"]),
        "main_n": sum(1 for r in eval_records if r["eval_kind"] == "main"),
        "probe2_correct": sum(1 for r in eval_records if r["eval_kind"] == "probe2" and r["correct"]),
        "probe2_n": sum(1 for r in eval_records if r["eval_kind"] == "probe2"),
        "t_extract_s": t_extract,
        "t_train_s": t_train,
        "records": eval_records,
    }
    eval_path.write_text(json.dumps(out, indent=2))

    main_acc = out["main_correct"] / max(1, out["main_n"])
    probe2_acc = out["probe2_correct"] / max(1, out["probe2_n"])
    print(f"  [{pid}] main {out['main_correct']}/{out['main_n']}={main_acc:.2%} "
          f"probe2 {out['probe2_correct']}/{out['probe2_n']}={probe2_acc:.2%}",
          flush=True)

    # 8. Cleanup before next persona
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    return out


def aggregate():
    """Compute main/probe2 aggregates across all personae with finished evals."""
    rows = []
    for pdir in sorted(OUT_DIR.iterdir()):
        if not pdir.is_dir():
            continue
        ep = pdir / "eval.json"
        if not ep.exists():
            continue
        rows.append(json.loads(ep.read_text()))
    if not rows:
        return None
    main_n = sum(r["main_n"] for r in rows)
    main_c = sum(r["main_correct"] for r in rows)
    p2_n = sum(r["probe2_n"] for r in rows)
    p2_c = sum(r["probe2_correct"] for r in rows)
    sanity_vals = [r["sanity_acc"] for r in rows if r.get("sanity_acc") is not None]
    agg = {
        "n_personae": len(rows),
        "main": {"n": main_n, "correct": main_c, "acc": main_c / max(1, main_n)},
        "probe2": {"n": p2_n, "correct": p2_c, "acc": p2_c / max(1, p2_n)},
        "sanity_mean": sum(sanity_vals) / len(sanity_vals) if sanity_vals else None,
        "sanity_n_personae": len(sanity_vals),
    }
    (OUT_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N held-out personae (0 = all)")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip the first K held-out personae")
    args = ap.parse_args()

    # Fail-fast: refuse to run as root (Pitfall 21)
    import os
    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu (currently root). Wrap with: sudo -u ubuntu", file=sys.stderr)
        sys.exit(2)

    # Bedrock thread preflight (Pitfall 20)
    from concurrent.futures import ThreadPoolExecutor
    import boto3
    def _t(_):
        return boto3.client("bedrock-runtime", region_name="us-east-1").meta.region_name
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(_t, range(2)))
    except Exception as e:
        print(f"ERROR: Bedrock thread preflight failed: {e}", file=sys.stderr)
        print("Likely cause: AWS_PROFILE not exported. See Pitfall 20.", file=sys.stderr)
        sys.exit(3)

    # Load personae
    p = json.loads(PERSONAE_PATH.read_text())
    held_out = [x for x in p["personae"] if x.get("split") == "held_out"]
    print(f"loaded {len(held_out)} held-out personae", flush=True)
    if args.start:
        held_out = held_out[args.start:]
    if args.limit:
        held_out = held_out[:args.limit]

    # Tokenizer (load once, base reloaded per persona)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(exp19.QWEN_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    judge_client = exp19.make_client()

    for i, persona in enumerate(held_out):
        try:
            run_persona(persona, tok, judge_client)
        except Exception as e:
            import traceback
            print(f"!! [{persona['id']}] FAILED: {e}", flush=True)
            traceback.print_exc()
            # Don't crash the whole run on one persona's failure
            continue
        agg = aggregate()
        if agg:
            print(f"  [running] main {agg['main']['correct']}/{agg['main']['n']}="
                  f"{agg['main']['acc']:.2%}  "
                  f"probe2 {agg['probe2']['correct']}/{agg['probe2']['n']}="
                  f"{agg['probe2']['acc']:.2%}  "
                  f"(personae done: {agg['n_personae']})", flush=True)

    print("\n=== FINAL ===", flush=True)
    agg = aggregate()
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
