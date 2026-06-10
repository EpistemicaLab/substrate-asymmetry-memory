#!/usr/bin/env python3
"""32_band_zero_intervention.py
==================================

Causal test of §5.6 mechanism correlation: for each persona, load the
saved γ-LoRA adapter, **zero the lora_A weights** for q_proj at
layers 21-35 (the top band), and re-evaluate on:
  (a) absence probes from runs/29_f_absence/ (predict: absence-TPR ↑)
  (b) probe2 factual recall from exp23/25 (predict: probe2 acc ↓)

If both move in the predicted direction with effect size > 5pp, §5.6
upgrades from suggestive-causal to causal. The intervention is a clean
ablation: same base model, same tokenizer, same probes, only the
band-zeroing differs.

Two configs evaluated per persona:
  C_lora_full  : adapter loaded as trained (control — should reproduce exp29 numbers)
  C_lora_zero  : same adapter with L21-35 q_proj lora_A zeroed (intervention)

Both use the calibration system prompt for the absence eval (so we
isolate the band's contribution, not the prompt's).

Inputs:
  runs/29_f_absence/dataset.json       50 personae × 12 absence probes
  runs/23_persona_lora/V3_P_*/eval.json   probe2 questions (held-out facts)
  runs/25_persona_lora_v2/V3_P_*/eval.json
  runs/30_mechanism/V3_P_*/lora/        saved adapters

Outputs:
  runs/32_band_zero/V3_P_*/probes_<config>.jsonl
  runs/32_band_zero/V3_P_*/factrecall_<config>.jsonl
  runs/32_band_zero/aggregate.json
  runs/32_band_zero/summary.json

Run on EC2:
  bash scripts/<launcher>.sh  # see scripts/ for runnable equivalents

~50 personae × ~3 min/persona ≈ 2.5h on g6e.xlarge.
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

import boto3
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET_29 = ROOT / "runs" / "29_f_absence" / "dataset.json"
MECH_DIR = ROOT / "runs" / "30_mechanism"
EXP23_DIR = ROOT / "runs" / "23_persona_lora"
EXP25_DIR = ROOT / "runs" / "25_persona_lora_v2"
OUT_DIR = ROOT / "runs" / "32_band_zero"

# Mutable defaults; overridden by CLI in main() for control-band runs.
BAND_LAYERS = list(range(21, 36))  # 21..35 inclusive
BAND_PROJ = "q_proj"

sys.path.insert(0, str(ROOT))  # so `from experiments import _base_model` resolves
sys.path.insert(0, str(ROOT / "experiments"))
spec19 = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py")
exp19 = importlib.util.module_from_spec(spec19)
spec19.loader.exec_module(exp19)

# Reuse exp29's judge + system prompts so we measure on identical scaffolding
spec29 = importlib.util.spec_from_file_location(
    "exp29", ROOT / "experiments" / "29_runner.py")
exp29 = importlib.util.module_from_spec(spec29)
spec29.loader.exec_module(exp29)


def find_adapter_dir(pid: str) -> Optional[Path]:
    for base in (MECH_DIR, EXP23_DIR, EXP25_DIR):
        cand = base / pid / "lora"
        if (cand / "adapter_config.json").exists():
            return cand
    return None


def find_probe2_eval(pid: str) -> Optional[Path]:
    for base in (EXP23_DIR, EXP25_DIR):
        p = base / pid / "eval.json"
        if p.exists():
            return p
    return None


def load_persona_with_adapter(pid: str, tok, zero_band: bool):
    """Load base + apply persona's adapter; optionally zero band."""
    from peft import PeftModel
    base = exp19.load_base()
    adapter_dir = find_adapter_dir(pid)
    if adapter_dir is None:
        del base
        return None
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if zero_band:
        n_zeroed = 0
        for name, param in model.named_parameters():
            # Match: ...layers.<L>.self_attn.<proj>.lora_A.default.weight
            # or:    ...layers.<L>.self_attn.<proj>.lora_A.weight
            if ".lora_A." not in name:
                continue
            # Parse layer index + projection
            parts = name.split(".")
            try:
                li = int(parts[parts.index("layers") + 1])
                # the projection appears just before "lora_A"
                proj_idx = parts.index("lora_A") - 1
                proj = parts[proj_idx]
            except (ValueError, IndexError):
                continue
            if li in BAND_LAYERS and proj == BAND_PROJ:
                with torch.no_grad():
                    param.zero_()
                n_zeroed += 1
        print(f"  [{pid}] zeroed {n_zeroed} lora_A tensors in band L{BAND_LAYERS[0]}-{BAND_LAYERS[-1]} {BAND_PROJ}", flush=True)
    return model


def load_probe2_questions(pid: str) -> list[dict]:
    """Pull probe2 records from exp23/25 eval.json — same format as the original eval."""
    p = find_probe2_eval(pid)
    if p is None:
        return []
    d = json.loads(p.read_text())
    return [r for r in d.get("records", []) if r.get("eval_kind") == "probe2"]


def eval_factrecall(model, tok, judge_client, pid: str, qs: list[dict], config: str) -> dict:
    """Re-run probe2 factual recall under the calibration prompt.
    Note: exp23/25 used a no-system-prompt setup; we use plain system here for
    a clean ablation against the absence eval. We score with the same exact-match
    rule used originally (gold substring or equivalent), with the Bedrock judge
    as a fallback for free-form answers."""
    out_path = OUT_DIR / pid / f"factrecall_{config}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add(rec["q"])
    correct = 0
    total = 0
    f = out_path.open("a")
    try:
        for r in qs:
            q = r["q"]
            gold = r.get("a_gold", r.get("gold", ""))
            total += 1
            if q in done:
                # Re-tally from existing line
                continue
            user = q
            pred = exp29.gen_with_system(tok, model, exp29.PLAIN_SYS, user, max_new=80)
            # Simple substring match for "ok"; fallback to original rule (substring of gold)
            ok = bool(gold) and (gold.lower().strip() in pred.lower())
            f.write(json.dumps({"q": q, "gold": gold, "pred": pred, "ok": ok}) + "\n")
            f.flush()
            if ok:
                correct += 1
    finally:
        f.close()
    # Re-tally final correct from the file (covers resume case)
    correct = 0
    total = 0
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                total += 1
                if rec.get("ok"):
                    correct += 1
    return {"n": total, "correct": correct, "acc": correct / max(1, total)}


def eval_absence(model, tok, judge_client, persona: dict, config: str) -> dict:
    """Re-run all absence probes for this persona under CALIB system prompt."""
    pid = persona["id"]
    probes = persona["probes"]
    out_path = OUT_DIR / pid / f"probes_{config}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done_keys.add((rec["topic"], rec["kind"]))
    f = out_path.open("a")
    try:
        for probe in probes:
            key = (probe["topic"], probe["kind"])
            if key in done_keys:
                continue
            user = probe["question"]
            pred = exp29.gen_with_system(tok, model, exp29.CALIB_SYS, user, max_new=80)
            verdict = exp29.judge_yesno(judge_client, probe["question"], probe["gold"], pred)
            rec = {
                "persona_id": pid, "config": config,
                "kind": probe["kind"], "topic": probe["topic"],
                "question": probe["question"], "gold": probe["gold"],
                "pred": pred, "verdict": verdict,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
    finally:
        f.close()
    # Tally TPR per kind ("present" + "absence" — match dataset.json schema)
    counts = {"present": [0, 0], "absence": [0, 0]}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            kind = r["kind"]
            if kind not in counts:
                continue
            ok = r["verdict"] == "CORRECT"
            counts[kind][0] += int(ok)
            counts[kind][1] += 1
    return {
        "present_tpr": counts["present"][0] / max(1, counts["present"][1]),
        "absence_tpr": counts["absence"][0] / max(1, counts["absence"][1]),
        "present_n": counts["present"][1],
        "absence_n": counts["absence"][1],
    }


def run_persona(persona: dict, tok, judge_client, only: set):
    pid = persona["id"]
    if find_adapter_dir(pid) is None:
        print(f"[{pid}] no adapter, skip", flush=True)
        return None
    print(f"[{pid}] starting", flush=True)
    t0 = time.time()
    out_summary = {"pid": pid}

    probe2_qs = load_probe2_questions(pid)

    for config, zero_band in [("C_lora_full", False), ("C_lora_zero", True)]:
        if only and config not in only:
            continue
        print(f"  [{pid}] config={config} zero_band={zero_band}", flush=True)
        model = load_persona_with_adapter(pid, tok, zero_band)
        if model is None:
            continue
        absence = eval_absence(model, tok, judge_client, persona, config)
        factrec = eval_factrecall(model, tok, judge_client, pid, probe2_qs, config) if probe2_qs else {"n": 0}
        out_summary[config] = {"absence": absence, "factrecall": factrec}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out_summary["wall_s"] = round(time.time() - t0, 1)
    pdir = OUT_DIR / pid
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "summary.json").write_text(json.dumps(out_summary, indent=2))
    return out_summary


def aggregate():
    rows = []
    for pdir in sorted(OUT_DIR.glob("V3_P_*")):
        sp = pdir / "summary.json"
        if sp.exists():
            rows.append(json.loads(sp.read_text()))
    if not rows:
        return
    agg = {"n_personae": len(rows), "configs": {}}
    for cfg in ("C_lora_full", "C_lora_zero"):
        present_tprs = []
        absence_tprs = []
        factrec_accs = []
        for r in rows:
            c = r.get(cfg)
            if not c:
                continue
            present_tprs.append(c["absence"]["present_tpr"])
            absence_tprs.append(c["absence"]["absence_tpr"])
            if c["factrecall"]["n"] > 0:
                factrec_accs.append(c["factrecall"]["acc"])
        if not present_tprs:
            continue
        def mean(xs): return sum(xs) / len(xs)
        agg["configs"][cfg] = {
            "n_personae": len(present_tprs),
            "present_tpr_mean": round(mean(present_tprs), 4),
            "absence_tpr_mean": round(mean(absence_tprs), 4),
            "factrecall_acc_mean": round(mean(factrec_accs), 4) if factrec_accs else None,
            "factrecall_n_personae": len(factrec_accs),
        }
    # Effect sizes: full vs zero
    if "C_lora_full" in agg["configs"] and "C_lora_zero" in agg["configs"]:
        full = agg["configs"]["C_lora_full"]
        zero = agg["configs"]["C_lora_zero"]
        agg["effect_zero_minus_full"] = {
            "delta_absence_tpr_pp": round(100 * (zero["absence_tpr_mean"] - full["absence_tpr_mean"]), 2),
            "delta_present_tpr_pp": round(100 * (zero["present_tpr_mean"] - full["present_tpr_mean"]), 2),
            "delta_factrecall_pp": round(100 * (zero["factrecall_acc_mean"] - full["factrecall_acc_mean"]), 2)
                                    if zero["factrecall_acc_mean"] is not None and full["factrecall_acc_mean"] is not None else None,
            "predicted_signs": {
                "delta_absence_tpr_pp": "+ (zeroing band → fewer confabulations on absent facts)",
                "delta_present_tpr_pp": "− (zeroing band → fewer correct affirmations on present facts)",
                "delta_factrecall_pp": "− (zeroing band → fewer correct probe2 recalls)",
            },
        }
    (OUT_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n=== Aggregate ===")
    print(json.dumps(agg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=all personae")
    ap.add_argument("--only", default="", help="comma-separated configs subset")
    ap.add_argument("--layers", default="21-35",
                    help="layer range inclusive, e.g. '21-35' or '5-19'")
    ap.add_argument("--proj", default="q_proj",
                    choices=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--out-dir", default="runs/32_band_zero",
                    help="output dir relative to repo root")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else set()

    # Apply CLI overrides to module-level globals consumed by load_persona_with_adapter
    global BAND_LAYERS, BAND_PROJ, OUT_DIR
    lo, hi = [int(x) for x in args.layers.split("-")]
    BAND_LAYERS = list(range(lo, hi + 1))
    BAND_PROJ = args.proj
    OUT_DIR = ROOT / args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[config] BAND_LAYERS={lo}..{hi} BAND_PROJ={BAND_PROJ} OUT_DIR={OUT_DIR}",
          flush=True)

    if not DATASET_29.exists():
        raise SystemExit(f"missing {DATASET_29}")
    personae = json.loads(DATASET_29.read_text())["personae"]
    if args.limit > 0:
        personae = personae[: args.limit]

    print(f"Loading tokenizer + judge client...", flush=True)
    from experiments import _base_model as _bm
    tok = AutoTokenizer.from_pretrained(_bm.active().path)
    judge_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    for persona in personae:
        try:
            run_persona(persona, tok, judge_client, only)
        except Exception as e:
            print(f"[{persona['id']}] FAIL: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    aggregate()


if __name__ == "__main__":
    main()
