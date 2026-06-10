#!/usr/bin/env python3
"""
37_p_true_baseline.py — P(True) baseline (Kadavath et al., 2022).

For each (persona, probe), reuse the cached γ-LoRA and RAG predictions from
exp35 (no need to regenerate answers). For each (substrate, prediction),
prompt the SAME model that produced it with a calibration question:

    Question: {q}
    Proposed answer: {pred}
    Is the proposed answer correct?
    A) True
    B) False
    The proposed answer is:

Read the next-token logits, take softmax over the "A" and "B" token ids,
and define P(True) := P(A | A,B). Route by argmax(p_true_lora, p_true_rag);
ties → use_rag (matches absence-prior).

Outputs: runs/37_p_true_baseline/{features.jsonl, aggregate.json}
(suffixed by ENGRAM_BASE_MODEL via _base_model.run_dir).

Resume-safe: per-persona features.jsonl rows; aggregate at end.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._base_model import run_dir as _run_dir  # noqa: E402

EXP29 = _run_dir(ROOT / "runs" / "29_f_absence")
EXP30 = _run_dir(ROOT / "runs" / "30_mechanism")
EXP35 = _run_dir(ROOT / "runs" / "35_logit_features")
OUT = _run_dir(ROOT / "runs" / "37_p_true_baseline")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py"
)
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)

PROMPT_TEMPLATE = (
    "Question: {q}\n"
    "Proposed answer: {a}\n"
    "Is the proposed answer correct?\n"
    "A) True\n"
    "B) False\n"
    "The proposed answer is:"
)


def _ab_token_ids(tok):
    """Get the token ids for ' A' and ' B' (with leading space, the form
    they appear in after 'is:'). Falls back to bare A/B if needed."""
    candidates_a = [" A", "A"]
    candidates_b = [" B", "B"]
    a_id, b_id = None, None
    for c in candidates_a:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            a_id = ids[0]; break
    for c in candidates_b:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            b_id = ids[0]; break
    if a_id is None or b_id is None:
        # Fallback: take first id of each
        a_id = tok.encode(" A", add_special_tokens=False)[0]
        b_id = tok.encode(" B", add_special_tokens=False)[0]
    return a_id, b_id


@torch.no_grad()
def p_true(tok, mdl, question: str, answer: str, a_id: int, b_id: int) -> float:
    user = PROMPT_TEMPLATE.format(q=question, a=answer)
    msgs = [{"role": "system", "content": "You are the user's helpful assistant."},
            {"role": "user", "content": user}]
    cfg = exp19._bm.active()
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   **cfg.chat_kwargs)
    enc = tok(text, return_tensors="pt").to("cuda")
    out = mdl(**enc)
    logits = out.logits[0, -1].float()
    pa, pb = logits[a_id], logits[b_id]
    return float(F.softmax(torch.stack([pa, pb]), dim=-1)[0])


def load_lora_for(pid: str, base):
    adapter_dir = EXP30 / pid / "lora"
    if not adapter_dir.exists():
        return None
    from peft import PeftModel
    m = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).cuda()
    m.eval()
    return m


def stage_score():
    """Score every cached probe with P(True) for both substrates."""
    from transformers import AutoTokenizer
    cfg = exp19._bm.active()
    tok = AutoTokenizer.from_pretrained(cfg.path)
    a_id, b_id = _ab_token_ids(tok)
    print(f"[score] tok ids: A={a_id} B={b_id}", flush=True)

    persona_dirs = sorted(EXP35.glob("V3_P_*"))
    print(f"[score] {len(persona_dirs)} personae from {EXP35}", flush=True)

    for pdir in persona_dirs:
        pid = pdir.name
        feat_path = pdir / "features.jsonl"
        if not feat_path.exists():
            continue
        out_pdir = OUT / pid
        out_pdir.mkdir(parents=True, exist_ok=True)
        out_path = out_pdir / "features.jsonl"
        rows = [json.loads(l) for l in feat_path.read_text().splitlines() if l.strip()]
        if out_path.exists():
            existing = sum(1 for _ in out_path.read_text().splitlines() if _.strip())
            if existing >= len(rows):
                print(f"[{pid}] cached ({existing}), skip", flush=True)
                continue

        # Adapter to score lora answers; disable_adapter to score rag answers.
        base = exp19.load_base()
        lora = load_lora_for(pid, base)
        if lora is None:
            print(f"[{pid}] WARN no adapter, skip", flush=True)
            del base; gc.collect(); torch.cuda.empty_cache()
            continue

        f = out_path.open("w")
        try:
            for r in rows:
                q = r["question"]
                lora_pred = r.get("lora_pred", "")
                rag_pred = r.get("rag_pred", "")
                pt_lora = p_true(tok, lora, q, lora_pred, a_id, b_id)
                with lora.disable_adapter():
                    pt_rag = p_true(tok, lora, q, rag_pred, a_id, b_id)
                row = {
                    "persona_id": pid, "kind": r["kind"], "topic": r["topic"],
                    "question": q, "gold": r["gold"],
                    "lora_pred": lora_pred, "rag_pred": rag_pred,
                    "p_true_lora": pt_lora, "p_true_rag": pt_rag,
                }
                f.write(json.dumps(row) + "\n"); f.flush()
        finally:
            f.close()
        del lora, base
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{pid}] done ({len(rows)} probes)", flush=True)


def stage_aggregate():
    """Join with exp29 verdicts, compute routing F1 + per-kind TPR."""
    rows = []
    for pdir in sorted(OUT.glob("V3_P_*")):
        pid = pdir.name
        fp = pdir / "features.jsonl"
        if not fp.exists():
            continue
        lora_path = EXP29 / pid / "probes_C_lora.jsonl"
        rag_path = EXP29 / pid / "probes_C_rag.jsonl"
        if not lora_path.exists() or not rag_path.exists():
            continue
        L = {(json.loads(l)["kind"], json.loads(l)["topic"]): json.loads(l)
             for l in lora_path.read_text().splitlines() if l.strip()}
        R = {(json.loads(l)["kind"], json.loads(l)["topic"]): json.loads(l)
             for l in rag_path.read_text().splitlines() if l.strip()}
        for r in (json.loads(l) for l in fp.read_text().splitlines() if l.strip()):
            key = (r["kind"], r["topic"])
            if key not in L or key not in R: continue
            r["lora_verdict"] = L[key]["verdict"]
            r["rag_verdict"] = R[key]["verdict"]
            rows.append(r)

    def route(r):
        # Tie or near-tie → use_rag (absence-prior).
        if r["p_true_lora"] > r["p_true_rag"] + 1e-6:
            return "lora"
        return "rag"

    def correct(r):
        choice = route(r)
        return (r["lora_verdict"] if choice == "lora" else r["rag_verdict"]) == "CORRECT"

    persona_ids = sorted({r["persona_id"] for r in rows})
    present = [r for r in rows if r["kind"] == "present"]
    absence = [r for r in rows if r["kind"] == "absence"]
    p_tpr = sum(1 for r in present if correct(r)) / max(len(present), 1)
    a_tpr = sum(1 for r in absence if correct(r)) / max(len(absence), 1)
    f1 = 2 * p_tpr * a_tpr / max(p_tpr + a_tpr, 1e-9)

    routes = [route(r) for r in rows]
    out = {
        "n_personae": len(persona_ids),
        "n_rows": len(rows),
        "present_TPR": p_tpr,
        "absence_TPR": a_tpr,
        "F1": f1,
        "route_lora_pct": routes.count("lora") / max(len(routes), 1),
        "route_rag_pct": routes.count("rag") / max(len(routes), 1),
    }

    # Compare to pure substrates from exp29 + hybrid_logits from exp36 if present.
    try:
        agg29 = json.loads((EXP29 / "aggregate.json").read_text())
        pures = agg29["configs"]
        out["baselines"] = {
            "lora_alone_F1": pures["C_lora"]["f1"],
            "rag_alone_F1": pures["C_rag"]["f1"],
            "lora_calib_F1": pures["C_lora_calib"]["f1"],
        }
    except Exception as e:
        out["baselines_err"] = str(e)
    try:
        agg36 = json.loads((_run_dir(ROOT / "runs" / "36_calib_head_logits") /
                            "aggregate.json").read_text())
        out["hybrid_logits_F1"] = agg36.get("eval_F1")
    except Exception:
        pass

    (OUT / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "aggregate", "all"], default="all")
    args = ap.parse_args()
    if args.stage in {"score", "all"}: stage_score()
    if args.stage in {"aggregate", "all"}: stage_aggregate()


if __name__ == "__main__":
    main()
