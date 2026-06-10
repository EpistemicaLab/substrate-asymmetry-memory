#!/usr/bin/env python3
"""29_runner.py
================

F-absence eval (exp29): run 4 configs over 50 personae × 12 probes = 600 records.

Configs:
  B_nohist            : base Qwen3-4B, no context
  C_rag               : per-persona BGE top-K=3 over backstory chunks
  C_lora              : per-persona γ-LoRA (retrained from synthqa pairs)
  C_lora_with_calibration : same LoRA + system prompt nudge to say "no" if uncertain

Inputs : runs/29_f_absence/dataset.json (built by 29_data_prep.py)
Outputs:
  runs/29_f_absence/<pid>/probes_<config>.jsonl  per-record (resume-safe)
  runs/29_f_absence/summary.jsonl                machine-readable rollup
  runs/29_f_absence/aggregate.json               final TPR/F1 per config

Per persona ~10 min on L40S (extraction ~1 min + training ~3 min + 4 configs
× 12 probes × ~1.5 s + judge). 50 personae ≈ 8.3 GPU-h. With 24h cap fine.
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

import torch
from transformers import AutoModel, AutoTokenizer
from experiments import _base_model as _bm

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "runs" / "29_f_absence" / "dataset.json"
OUT_DIR = _bm.run_dir(ROOT / "runs" / "29_f_absence")
SUMMARY = OUT_DIR / "summary.jsonl"
AGG_PATH = OUT_DIR / "aggregate.json"
BGE_PATH = ROOT / "models" / "bge-large-en-v1.5"

# Reuse exp19 + exp23
sys.path.insert(0, str(ROOT / "experiments"))
spec19 = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py")
exp19 = importlib.util.module_from_spec(spec19)
spec19.loader.exec_module(exp19)
spec23 = importlib.util.spec_from_file_location(
    "exp23", ROOT / "experiments" / "23_lora_persona.py")
exp23 = importlib.util.module_from_spec(spec23)
spec23.loader.exec_module(exp23)

CALIB_SYS = (
    "You are a helpful assistant who knows the user from prior conversations. "
    "If you are not sure whether the user has previously discussed a topic with you, "
    "answer 'No, we have not discussed that.' Do not fabricate a memory."
)
PLAIN_SYS = "You are a helpful assistant who knows the user from prior conversations."


# ---------------- BGE retrieval ----------------

def chunk_text(text: str, size: int = 500, stride: int = 350):
    bs = text.strip()
    if len(bs) <= size:
        return [bs]
    out = []
    i = 0
    while i < len(bs):
        out.append(bs[i:i + size])
        if i + size >= len(bs):
            break
        i += stride
    return out


def load_bge(device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(BGE_PATH)
    mdl = AutoModel.from_pretrained(BGE_PATH, dtype=torch.float16).to(device).eval()
    return tok, mdl


@torch.no_grad()
def bge_encode(tok, mdl, texts, device="cuda:0"):
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    h = mdl(**enc).last_hidden_state[:, 0]
    h = torch.nn.functional.normalize(h, p=2, dim=1)
    return h.cpu()


# ---------------- Inference ----------------

def gen_with_system(tok, model, system: str, user: str, max_new: int = 80) -> str:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    return exp19.gen_chat(tok, model, msgs, max_new=max_new)


def render_rag_user(question: str, chunks: list[str]) -> str:
    blocks = "\n\n".join(f"[Memory {i+1}] {c.strip()}" for i, c in enumerate(chunks))
    return (f"{blocks}\n\n"
            f"Based ONLY on the memory snippets above, answer the user's question. "
            f"If the snippets do not actually mention the topic, say 'No, we have not "
            f"discussed that.'\n\nUser: {question}")


# ---------------- Judge ----------------

JUDGE_SYS = (
    "You judge whether an assistant's reply correctly answers a yes/no question "
    "about whether the user previously discussed a topic. The gold label is 'yes' "
    "or 'no'. Output exactly one of: CORRECT, WRONG, HEDGE.\n"
    "- CORRECT: reply clearly matches gold (yes-and-grounded for 'yes' gold; "
    "  clearly denies for 'no' gold).\n"
    "- WRONG: reply contradicts gold OR (for 'yes' gold) affirms with fabricated "
    "  details OR (for 'no' gold) affirms a discussion that never happened.\n"
    "- HEDGE: reply is uncommitted ('I'm not sure', 'I might have', 'possibly') "
    "  without making a clear yes/no claim."
)


def judge_yesno(client, question: str, gold: str, pred: str, retries: int = 4) -> str:
    user = (f"QUESTION: {question}\nGOLD: {gold}\nASSISTANT REPLY: {pred}\n\n"
            f"Verdict (one word: CORRECT, WRONG, or HEDGE):")
    last = None
    for attempt in range(retries):
        try:
            resp = client.converse(
                modelId=exp19.BEDROCK_MODEL,
                system=[{"text": JUDGE_SYS}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 10, "temperature": 0.0},
            )
            text = resp["output"]["message"]["content"][0]["text"].strip().upper()
            for tok in ("CORRECT", "WRONG", "HEDGE"):
                if tok in text:
                    return tok
            return "WRONG"
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + 0.5)
    raise RuntimeError(f"judge failed: {last}")


# ---------------- Per-persona run ----------------

def run_persona(persona, tok, judge_client, bge_tok, bge_mdl):
    pid = persona["id"]
    pdir = OUT_DIR / pid
    pdir.mkdir(parents=True, exist_ok=True)

    probes = persona["probes"]
    if not probes:
        print(f"[{pid}] no probes, skip", flush=True)
        return None

    # Build per-persona BGE corpus once
    chunks = chunk_text(persona["backstory"])
    chunk_emb = bge_encode(bge_tok, bge_mdl, chunks)

    # Embed all probe questions in one batch
    probe_qs = [p["question"] for p in probes]
    probe_emb = bge_encode(bge_tok, bge_mdl, probe_qs)
    sims = chunk_emb @ probe_emb.T  # (N_chunks, N_probes)

    K = min(3, len(chunks))
    rag_ctx_per_probe = []
    for i in range(len(probes)):
        s = sims[:, i]
        idx = torch.topk(s, k=K).indices.tolist()
        rag_ctx_per_probe.append([chunks[j] for j in idx])

    # ---- Run config B_nohist + C_rag (base model, no LoRA) ----
    print(f"[{pid}] base configs (B_nohist, C_rag)", flush=True)
    base = exp19.load_base()
    _run_config(base, tok, judge_client, persona, probes, "B_nohist",
                rag_ctx_per_probe=None, system=PLAIN_SYS)
    _run_config(base, tok, judge_client, persona, probes, "C_rag",
                rag_ctx_per_probe=rag_ctx_per_probe, system=PLAIN_SYS)
    del base
    gc.collect(); torch.cuda.empty_cache()

    # ---- Train LoRA (reuse exp23 pipeline) ----
    print(f"[{pid}] extracting + training γ-LoRA", flush=True)
    synth_path = pdir / "synthqa.jsonl"
    # Reuse exp23 cache if it exists in 23_persona_lora or 25_persona_lora_v2
    src_synth = None
    for cand in [ROOT / "runs" / "23_persona_lora" / pid / "synthqa.jsonl",
                 ROOT / "runs" / "25_persona_lora_v2" / pid / "synthqa.jsonl"]:
        if cand.exists():
            src_synth = cand
            break
    if synth_path.exists():
        pairs = [json.loads(l) for l in synth_path.read_text().splitlines() if l.strip()]
    elif src_synth is not None:
        pairs = [json.loads(l) for l in src_synth.read_text().splitlines() if l.strip()]
        synth_path.write_text(src_synth.read_text())
        print(f"  [{pid}] synth reused from {src_synth.parent.name}: {len(pairs)} pairs", flush=True)
    else:
        pairs, n1, n2 = exp23.extract_persona_pairs(persona["backstory"], pid)
        with synth_path.open("w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"  [{pid}] extracted {len(pairs)} pairs (pass1={n1}, pass2={n2})", flush=True)

    if len(pairs) < 5:
        print(f"  [{pid}] WARN: only {len(pairs)} pairs, skipping LoRA configs", flush=True)
        return _aggregate_persona(pid)

    train_pairs = pairs[:-5] if len(pairs) >= 10 else pairs
    base = exp19.load_base()
    model, losses = exp19.train_lora(base, tok, train_pairs, qid=pid)

    print(f"[{pid}] LoRA configs (C_lora, C_lora_calib) loss={losses[-1]:.4f}", flush=True)
    _run_config(model, tok, judge_client, persona, probes, "C_lora",
                rag_ctx_per_probe=None, system=PLAIN_SYS)
    _run_config(model, tok, judge_client, persona, probes, "C_lora_calib",
                rag_ctx_per_probe=None, system=CALIB_SYS)

    del model, base
    gc.collect(); torch.cuda.empty_cache()

    return _aggregate_persona(pid)


def _run_config(model, tok, judge_client, persona, probes, config: str,
                rag_ctx_per_probe, system: str):
    pid = persona["id"]
    out_path = OUT_DIR / pid / f"probes_{config}.jsonl"
    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done_keys.add((rec["topic"], rec["kind"]))
        print(f"  [{pid}] {config}: resume, {len(done_keys)} already done", flush=True)

    f = out_path.open("a")
    try:
        for i, probe in enumerate(probes):
            key = (probe["topic"], probe["kind"])
            if key in done_keys:
                continue
            if rag_ctx_per_probe is not None:
                user = render_rag_user(probe["question"], rag_ctx_per_probe[i])
            else:
                user = probe["question"]
            pred = gen_with_system(tok, model, system, user, max_new=80)
            verdict = judge_yesno(judge_client, probe["question"], probe["gold"], pred)
            rec = {
                "persona_id": pid,
                "config": config,
                "kind": probe["kind"],
                "topic": probe["topic"],
                "question": probe["question"],
                "gold": probe["gold"],
                "pred": pred,
                "verdict": verdict,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            with SUMMARY.open("a") as sf:
                sf.write(json.dumps(rec) + "\n")
    finally:
        f.close()


def _aggregate_persona(pid: str):
    pdir = OUT_DIR / pid
    out = {"persona_id": pid, "configs": {}}
    for config in ("B_nohist", "C_rag", "C_lora", "C_lora_calib"):
        f = pdir / f"probes_{config}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        out["configs"][config] = _scores(recs)
    (pdir / "aggregate.json").write_text(json.dumps(out, indent=2))
    return out


def _scores(recs):
    pres = [r for r in recs if r["kind"] == "present"]
    abst = [r for r in recs if r["kind"] == "absence"]
    def _tpr(rs):
        n = len(rs)
        c = sum(1 for r in rs if r["verdict"] == "CORRECT")
        h = sum(1 for r in rs if r["verdict"] == "HEDGE")
        return {"n": n, "correct": c, "hedge": h, "tpr": (c / n) if n else 0.0}
    pres_s = _tpr(pres)
    abst_s = _tpr(abst)
    f1 = 0.0
    if pres_s["tpr"] + abst_s["tpr"] > 0:
        f1 = 2 * pres_s["tpr"] * abst_s["tpr"] / (pres_s["tpr"] + abst_s["tpr"])
    return {"present": pres_s, "absence": abst_s, "f1": f1}


def aggregate_all():
    rows = []
    for pdir in sorted(OUT_DIR.iterdir()):
        if not pdir.is_dir():
            continue
        ag = pdir / "aggregate.json"
        if ag.exists():
            rows.append(json.loads(ag.read_text()))
    if not rows:
        return None
    overall = {"n_personae": len(rows), "configs": {}}
    for config in ("B_nohist", "C_rag", "C_lora", "C_lora_calib"):
        n_p = c_p = n_a = c_a = h_p = h_a = 0
        for r in rows:
            cs = r["configs"].get(config)
            if not cs:
                continue
            n_p += cs["present"]["n"]; c_p += cs["present"]["correct"]; h_p += cs["present"]["hedge"]
            n_a += cs["absence"]["n"]; c_a += cs["absence"]["correct"]; h_a += cs["absence"]["hedge"]
        pres_tpr = c_p / max(1, n_p)
        abst_tpr = c_a / max(1, n_a)
        f1 = 0.0
        if pres_tpr + abst_tpr > 0:
            f1 = 2 * pres_tpr * abst_tpr / (pres_tpr + abst_tpr)
        overall["configs"][config] = {
            "present": {"n": n_p, "correct": c_p, "hedge": h_p, "tpr": pres_tpr},
            "absence": {"n": n_a, "correct": c_a, "hedge": h_a, "tpr": abst_tpr},
            "f1": f1,
        }
    AGG_PATH.write_text(json.dumps(overall, indent=2))
    return overall


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
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(_t, range(2)))

    if not DATASET.exists():
        print(f"ERROR: dataset missing: {DATASET}. Run 29_data_prep.py first.", file=sys.stderr)
        sys.exit(4)

    data = json.loads(DATASET.read_text())
    personae = data["personae"]
    if args.start:
        personae = personae[args.start:]
    if args.limit:
        personae = personae[:args.limit]
    print(f"[main] personae to run: {len(personae)}", flush=True)

    tok = AutoTokenizer.from_pretrained(_bm.active().path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    judge_client = exp19.make_client()
    bge_tok, bge_mdl = load_bge()
    print("[main] BGE loaded", flush=True)

    t0 = time.time()
    for i, persona in enumerate(personae):
        try:
            run_persona(persona, tok, judge_client, bge_tok, bge_mdl)
        except Exception as e:
            import traceback
            print(f"!! [{persona['id']}] FAILED: {e}", flush=True)
            traceback.print_exc()
        ag = aggregate_all()
        if ag:
            line = f"  [running {i+1}/{len(personae)}] elapsed={(time.time()-t0)/60:.1f}min"
            for c, s in ag["configs"].items():
                line += f"  {c}: pTPR={s['present']['tpr']:.2f} aTPR={s['absence']['tpr']:.2f} F1={s['f1']:.2f}"
            print(line, flush=True)

    print("\n=== FINAL ===", flush=True)
    print(json.dumps(aggregate_all(), indent=2))


if __name__ == "__main__":
    main()
