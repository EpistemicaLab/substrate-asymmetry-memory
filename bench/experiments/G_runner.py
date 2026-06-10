#!/usr/bin/env python3
"""
G_runner.py — retrain γ-LoRA on a 5-persona subset, save adapters to disk.

Reuses exp23.train_lora via importlib. Difference vs exp23: we call
peft_model.save_pretrained(out_dir / "lora") so adapters survive
for the Frobenius decomposition.

Usage:
    export AWS_PROFILE=$YOUR_PROFILE
    .venv/bin/python experiments/G_runner.py
"""
from __future__ import annotations
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PERSONAE_PATH = ROOT / "runs" / "V3_personae" / "personae.json"
OUT_DIR = ROOT / "runs" / "G_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 representative held-out personae (see G_design.md §protocol)
PICKS = ["V3_P_101", "V3_P_103", "V3_P_105", "V3_P_125", "V3_P_145"]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def main():
    if not PERSONAE_PATH.exists():
        print(f"ERROR: {PERSONAE_PATH} missing", file=sys.stderr)
        sys.exit(2)

    exp19 = _load_module("exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py")
    exp23 = _load_module("exp23", ROOT / "experiments" / "23_lora_persona.py")

    personae_data = json.load(open(PERSONAE_PATH))
    personae = personae_data["personae"] if isinstance(personae_data, dict) else personae_data
    by_id = {p.get("id") or p.get("persona_id"): p for p in personae}

    print(f"[G_runner] loading base model...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(exp19.QWEN_PATH)
    base = exp19.load_base()

    for pid in PICKS:
        if pid not in by_id:
            print(f"[G_runner] WARN: {pid} not in personae.json, skipping", flush=True)
            continue

        out_dir = OUT_DIR / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = out_dir / "lora"
        if (adapter_dir / "adapter_model.safetensors").exists():
            print(f"[G_runner] {pid}: adapter cached, skip", flush=True)
            continue

        # Reuse exp23 synthqa cache
        cache_paths = [
            ROOT / "runs" / "23_persona_lora" / pid / "synthqa.jsonl",
            ROOT / "runs" / "25_persona_lora_v2" / pid / "synthqa.jsonl",
        ]
        synthqa_path = next((p for p in cache_paths if p.exists()), None)
        if synthqa_path is None:
            print(f"[G_runner] {pid}: no synthqa cache found, ABORT", file=sys.stderr)
            sys.exit(3)

        with open(synthqa_path) as f:
            train_pairs = [json.loads(line) for line in f if line.strip()]
        # Hold out 20% sanity slice (same as exp23); use first 80% for training
        n_train = int(0.8 * len(train_pairs))
        train_only = train_pairs[:n_train]
        print(f"[G_runner] {pid}: training on {len(train_only)} pairs (cache @ {synthqa_path.name})", flush=True)

        t0 = time.time()
        model, losses = exp19.train_lora(base, tok, train_only, qid=pid)
        dt = time.time() - t0
        final_loss = losses[-1] if losses else float("nan")
        print(f"[G_runner] {pid}: train done in {dt:.1f}s, final_loss={final_loss:.4f}", flush=True)

        # Save the adapter
        model.save_pretrained(str(adapter_dir))
        with open(out_dir / "meta.json", "w") as f:
            json.dump({
                "pid": pid,
                "n_train_pairs": len(train_only),
                "synthqa_source": str(synthqa_path.relative_to(ROOT)),
                "final_loss": final_loss,
                "train_seconds": dt,
            }, f, indent=2)

        # Reload base for next persona to avoid adapter stacking
        del model
        gc.collect()
        torch.cuda.empty_cache()
        base = exp19.load_base()

    print(f"[G_runner] done; adapters saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
