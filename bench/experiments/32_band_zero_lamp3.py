#!/usr/bin/env python3
"""32_band_zero_lamp3.py — LaMP-3-flavored band-zero intervention (S032 Mistral).

Mistral robustness check for the §5.6 band-zero causal claim. Where the
original Qwen-era 32_band_zero_intervention.py evaluated on V3_P_*
absence probes + probe2 factrecall (artifacts that don't exist in the
Mistral pipeline), this runner evaluates on the LaMP-3 main_qa /
probe2_qa eval set — the same eval scaffolding that REPLICATE_MISTRAL_D
already used.

Design (per ICLR2027_PUSH_PLAN.md S032 v2):
  Two configs per persona:
    C_lora_full : adapter loaded as trained (CONTROL — must reproduce
                  REPLICATE_MISTRAL_D/<pid>/eval.json numbers; reused
                  from cache, NOT re-run, to save ~9h GPU)
    C_lora_zero : same adapter with L21-35 q_proj lora_A zeroed
                  (INTERVENTION — predicted: main_acc/probe2_acc both
                  drop, robustness check on the same band identified
                  in Qwen §5.6)

Inputs:
  runs/F_lamp3/dataset.json                  50 personae (held_out split) × 4 main + 4 probe2
  runs/40_lamp3_mit/REPLICATE_MISTRAL_D/<pid>/lora/   saved γ-LoRA adapters
  runs/40_lamp3_mit/REPLICATE_MISTRAL_D/<pid>/eval.json  cached C_lora_full

Outputs:
  runs/iclr2027_push/MISTRAL_band_zero/<pid>/eval_C_lora_zero.json
  runs/iclr2027_push/MISTRAL_band_zero/aggregate.json   {n_personae, full, zero, deltas}

Acceptance (S032 v2):
  - aggregate.json exists with n_personae=50 (both configs)
  - reports delta_main_acc, delta_probe2_acc with 95% bootstrap CI
  - direction-test passes if both deltas are negative (zero → fewer
    correct LaMP-3 predictions); magnitude is reported but is NOT a
    pre-registered falsifier (per Pitfall 4a)

Wallclock: ~1.5h on g6e.xlarge (50 personae × ~110s/persona for
C_lora_zero only; C_lora_full reused from cache).

Run on EC2:
  bash scripts/<launcher>.sh  # see scripts/ for runnable equivalents
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "runs" / "F_lamp3" / "dataset.json"
REPLICATE_DIR = ROOT / "runs" / "40_lamp3_mit" / "REPLICATE_MISTRAL_D"
OUT_DIR = ROOT / "runs" / "iclr2027_push" / "MISTRAL_band_zero"

# Module-level globals; CLI can override
BAND_LAYERS = list(range(21, 36))  # L21..L35 inclusive (matches Qwen §5.6)
BAND_PROJ = "q_proj"

# importlib glue — same pattern as 32_band_zero_intervention.py
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
spec19 = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py")
exp19 = importlib.util.module_from_spec(spec19)
spec19.loader.exec_module(exp19)


# ── adapter loading + zero-band intervention ────────────────────────────────

def load_persona_with_adapter(pid: str, zero_band: bool):
    """Load Mistral base + this persona's REPLICATE_MISTRAL_D adapter; optionally zero band."""
    from peft import PeftModel
    adapter_dir = REPLICATE_DIR / pid / "lora"
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"  [{pid}] adapter missing at {adapter_dir}", flush=True)
        return None
    base = exp19.load_base()
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if zero_band:
        n_zeroed = 0
        for name, param in model.named_parameters():
            if ".lora_A." not in name:
                continue
            parts = name.split(".")
            try:
                li = int(parts[parts.index("layers") + 1])
                proj_idx = parts.index("lora_A") - 1
                proj = parts[proj_idx]
            except (ValueError, IndexError):
                continue
            if li in BAND_LAYERS and proj == BAND_PROJ:
                with torch.no_grad():
                    param.zero_()
                n_zeroed += 1
        print(f"  [{pid}] zeroed {n_zeroed} lora_A tensors in band L{BAND_LAYERS[0]}-{BAND_LAYERS[-1]} {BAND_PROJ}", flush=True)
        if n_zeroed == 0:
            print(f"  [{pid}] WARN: zero_band=True but n_zeroed=0 — adapter likely doesn't target {BAND_PROJ}", flush=True)
    model.eval()
    return model


# ── eval (LaMP-3 main + probe2, same recipe as exp40 free-decoding arms) ────

def eval_persona(model, tok, judge_client, persona: dict) -> dict:
    """Mirror exp40.run_persona's eval-only block, free decoding."""
    pid = persona["id"]
    eval_records = []
    for kind in ("main_qa", "probe2_qa"):
        for q in persona.get(kind, []):
            qq = q.get("q") or q.get("question", "")
            gold = q.get("a") or q.get("answer", "")
            pred = exp19.gen_chat(tok, model, [{"role": "user", "content": qq}])
            correct = exp19.judge(judge_client, qq, gold, pred)
            eval_records.append({
                "persona_id": pid,
                "eval_kind": "main" if kind == "main_qa" else "probe2",
                "q": qq, "gold": gold, "pred": pred, "correct": correct,
            })
    main_n = sum(1 for r in eval_records if r["eval_kind"] == "main")
    probe2_n = sum(1 for r in eval_records if r["eval_kind"] == "probe2")
    main_acc = sum(r["correct"] for r in eval_records if r["eval_kind"] == "main") / max(1, main_n)
    probe2_acc = sum(r["correct"] for r in eval_records if r["eval_kind"] == "probe2") / max(1, probe2_n)
    return {
        "persona_id": pid,
        "main_acc": main_acc, "probe2_acc": probe2_acc,
        "n_main": main_n, "n_probe2": probe2_n,
        "eval_records": eval_records,
    }


def run_persona_zero(persona: dict, tok, judge_client) -> Optional[dict]:
    """Run the C_lora_zero condition only. C_lora_full is reused from cache."""
    pid = persona["id"]
    out_path = OUT_DIR / pid / "eval_C_lora_zero.json"
    if out_path.exists():
        print(f"[{pid}] cached C_lora_zero, skipping", flush=True)
        return json.loads(out_path.read_text())

    t0 = time.time()
    print(f"\n=== S032 band-zero | {pid} ===", flush=True)
    model = load_persona_with_adapter(pid, zero_band=True)
    if model is None:
        return None
    rec = eval_persona(model, tok, judge_client, persona)
    rec["wallclock_s"] = round(time.time() - t0, 1)
    rec["config"] = "C_lora_zero"
    rec["band_layers"] = [BAND_LAYERS[0], BAND_LAYERS[-1]]
    rec["band_proj"] = BAND_PROJ
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2))
    print(f"  [{pid}] zero main={rec['main_acc']:.3f} probe2={rec['probe2_acc']:.3f} t={rec['wallclock_s']:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def load_cached_full(pid: str) -> Optional[dict]:
    """Read REPLICATE_MISTRAL_D/<pid>/eval.json as the C_lora_full control."""
    p = REPLICATE_DIR / pid / "eval.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {
        "persona_id": pid,
        "main_acc": d["main_acc"],
        "probe2_acc": d["probe2_acc"],
        "n_main": d.get("n_main", 0),
        "n_probe2": d.get("n_probe2", 0),
    }


# ── aggregate + bootstrap CI ────────────────────────────────────────────────

def _bootstrap_ci(deltas: list[float], n_iter: int = 2000, seed: int = 0):
    import random
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return (None, None)
    means = []
    for _ in range(n_iter):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_iter)]
    hi = means[int(0.975 * n_iter)]
    return (round(lo, 4), round(hi, 4))


def aggregate():
    full_rows = []
    zero_rows = []
    for pdir in sorted(REPLICATE_DIR.iterdir()):
        if not pdir.is_dir():
            continue
        pid = pdir.name
        f = load_cached_full(pid)
        z_path = OUT_DIR / pid / "eval_C_lora_zero.json"
        if not z_path.exists() or f is None:
            continue
        z = json.loads(z_path.read_text())
        full_rows.append(f)
        zero_rows.append({
            "persona_id": pid,
            "main_acc": z["main_acc"], "probe2_acc": z["probe2_acc"],
            "n_main": z["n_main"], "n_probe2": z["n_probe2"],
        })
    n = len(zero_rows)
    if n == 0:
        print("[agg] no C_lora_zero results", flush=True)
        return

    def _mean(xs): return sum(xs) / len(xs) if xs else None

    full_main = [r["main_acc"] for r in full_rows]
    full_p2 = [r["probe2_acc"] for r in full_rows]
    zero_main = [r["main_acc"] for r in zero_rows]
    zero_p2 = [r["probe2_acc"] for r in zero_rows]

    # paired deltas (zero - full) per persona
    full_by_pid = {r["persona_id"]: r for r in full_rows}
    delta_main = []
    delta_p2 = []
    for r in zero_rows:
        f = full_by_pid.get(r["persona_id"])
        if f is None:
            continue
        delta_main.append(r["main_acc"] - f["main_acc"])
        delta_p2.append(r["probe2_acc"] - f["probe2_acc"])

    agg = {
        "version": "S032_v2_lamp3",
        "n_personae": n,
        "band_layers": [BAND_LAYERS[0], BAND_LAYERS[-1]],
        "band_proj": BAND_PROJ,
        "control_source": "runs/40_lamp3_mit/REPLICATE_MISTRAL_D/<pid>/eval.json (cached)",
        "intervention_source": "runs/iclr2027_push/MISTRAL_band_zero/<pid>/eval_C_lora_zero.json",
        "C_lora_full": {
            "n": len(full_rows),
            "main_acc_mean": round(_mean(full_main), 4),
            "probe2_acc_mean": round(_mean(full_p2), 4),
        },
        "C_lora_zero": {
            "n": n,
            "main_acc_mean": round(_mean(zero_main), 4),
            "probe2_acc_mean": round(_mean(zero_p2), 4),
        },
        "effect_zero_minus_full": {
            "n_paired": len(delta_main),
            "delta_main_acc_mean": round(_mean(delta_main), 4) if delta_main else None,
            "delta_main_acc_ci95": _bootstrap_ci(delta_main),
            "delta_probe2_acc_mean": round(_mean(delta_p2), 4) if delta_p2 else None,
            "delta_probe2_acc_ci95": _bootstrap_ci(delta_p2),
            "n_personae_main_drop": sum(1 for d in delta_main if d < 0),
            "n_personae_probe2_drop": sum(1 for d in delta_p2 if d < 0),
            "predicted_signs": {
                "delta_main_acc": "− (zeroing band → fewer correct main predictions)",
                "delta_probe2_acc": "− (zeroing band → fewer correct probe2 recalls)",
            },
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n=== Aggregate (S032 LaMP-3 band-zero) ===")
    print(json.dumps(agg, indent=2))


# ── entry ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all held-out personae")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--layers", default="21-35", help="L<lo>-<hi> inclusive")
    ap.add_argument("--proj", default="q_proj",
                    choices=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip GPU work; just rebuild aggregate.json from existing per-persona files")
    args = ap.parse_args()

    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu", file=sys.stderr); sys.exit(2)

    global BAND_LAYERS, BAND_PROJ
    lo, hi = [int(x) for x in args.layers.split("-")]
    BAND_LAYERS = list(range(lo, hi + 1))
    BAND_PROJ = args.proj
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[config] BAND_LAYERS=L{lo}..L{hi} BAND_PROJ={BAND_PROJ} OUT_DIR={OUT_DIR}", flush=True)

    if args.aggregate_only:
        aggregate()
        return

    if not DATASET_PATH.exists():
        raise SystemExit(f"missing {DATASET_PATH}")
    p = json.loads(DATASET_PATH.read_text())
    held = [x for x in p["personae"] if x.get("split") == "held_out"]
    if args.start: held = held[args.start:]
    if args.limit: held = held[:args.limit]
    print(f"[S032] n_personae={len(held)}", flush=True)

    # Pre-flight: every persona must have a REPLICATE_MISTRAL_D adapter
    missing = [h["id"] for h in held if not (REPLICATE_DIR / h["id"] / "lora" / "adapter_config.json").exists()]
    if missing:
        print(f"FAIL pre-flight: {len(missing)} personae missing adapters: {missing[:5]}...", file=sys.stderr)
        sys.exit(2)

    # Bedrock thread preflight (mirror exp40)
    from concurrent.futures import ThreadPoolExecutor
    import boto3
    def _t(_):
        return boto3.client("bedrock-runtime", region_name="us-east-1").meta.region_name
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(_t, range(2)))
    except Exception as e:
        print(f"ERROR: Bedrock thread preflight failed: {e}", file=sys.stderr); sys.exit(3)

    from transformers import AutoTokenizer
    from experiments import _base_model as _bm
    tok = AutoTokenizer.from_pretrained(_bm.active().path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    judge_client = exp19.make_client()

    t0 = time.time()
    for i, persona in enumerate(held):
        try:
            run_persona_zero(persona, tok, judge_client)
        except Exception as e:
            print(f"[{persona['id']}] FAIL: {e}", flush=True)
            import traceback; traceback.print_exc()
        elapsed = time.time() - t0
        print(f"[S032] {i+1}/{len(held)} elapsed={elapsed:.0f}s", flush=True)

    aggregate()
    print(f"[S032] DONE total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
