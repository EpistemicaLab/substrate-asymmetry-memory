#!/usr/bin/env python3
"""Phase F runner: per-user γ-LoRA on LaMP-3 (Personalized Product Rating).

Mirrors exp25 verbatim — only PERSONAE_PATH and OUT_DIR change. The
`personae` here are 50 LaMP-3 users built by experiments/F_data_prep.py
into a V3-personae-compatible JSON shape, so exp23's `run_persona`
(synth-extract → train r=128 LoRA → sanity → eval main+probe2 → judge)
applies unmodified.

Outputs:
  runs/F_lamp3/gamma_lora/<persona_id>/{synthqa.jsonl,eval.json}
  runs/F_lamp3/gamma_lora/{summary.jsonl,aggregate.json}

Usage:
    .venv/bin/python experiments/F_lamp3_runner.py
    .venv/bin/python experiments/F_lamp3_runner.py --limit 2  # smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONAE_PATH = ROOT / "runs" / "F_lamp3" / "dataset.json"
OUT_DIR = ROOT / "runs" / "F_lamp3" / "gamma_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load exp23 module (which itself loads exp19) and rebind its OUT_DIR
sys.path.insert(0, str(ROOT / "experiments"))
spec23 = importlib.util.spec_from_file_location(
    "exp23", ROOT / "experiments" / "23_lora_persona.py"
)
exp23 = importlib.util.module_from_spec(spec23)
spec23.loader.exec_module(exp23)
exp23.OUT_DIR = OUT_DIR
exp23.SUMMARY = OUT_DIR / "summary.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu", file=sys.stderr)
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
        sys.exit(3)

    p = json.loads(PERSONAE_PATH.read_text())
    held_out = [x for x in p["personae"] if x.get("split") == "held_out"]
    print(f"[F_lamp3] loaded {len(held_out)} held-out personae from {PERSONAE_PATH.name}", flush=True)
    if args.start:
        held_out = held_out[args.start:]
    if args.limit:
        held_out = held_out[:args.limit]

    from transformers import AutoTokenizer
    exp19 = exp23.exp19
    tok = AutoTokenizer.from_pretrained(exp19.QWEN_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    judge_client = exp19.make_client()

    t0 = time.time()
    for i, persona in enumerate(held_out):
        try:
            exp23.run_persona(persona, tok, judge_client)
        except Exception as e:
            print(f"[F_lamp3] persona {persona['id']} failed: {e}", flush=True)
            import traceback; traceback.print_exc()
        elapsed = time.time() - t0
        print(f"[F_lamp3] {i+1}/{len(held_out)} done, elapsed={elapsed:.0f}s "
              f"({elapsed/(i+1):.0f}s/persona)", flush=True)

    # Aggregate
    exp23.aggregate()
    print(f"[F_lamp3] DONE total elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
