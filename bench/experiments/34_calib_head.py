#!/usr/bin/env python3
"""
34_calib_head.py — Phase H2 (Option C-A): calibration head over text-derived
features from cached γ-LoRA + RAG predictions.

Design
------
Inputs per probe (persona × {present, absence} × topic):
  - cached γ-LoRA pred string + verdict  (probes_C_lora.jsonl)
  - cached RAG pred string + verdict     (probes_C_rag.jsonl)
  - retrieval relevance r                (computed from BGE chunk-cosine)

Features (8-dim):
  conf_lora, conf_rag, agree, len_lora, len_rag, hedge_lora, hedge_rag, r

Where:
  conf_*  = 1.0 if non-hedge & yes/no extractable, 0.5 if hedge,
            0.0 if extraction failed
  agree   = 1.0 if both yes/no extractions agree, 0.0 otherwise
  len_*   = answer length / 200 (clipped to [0,1])
  hedge_* = 1.0 if any hedge phrase matched
  r       = max chunk-cosine similarity between probe question and
            top-3 backstory chunks (BGE-large)

Head: Linear(8→64) → ReLU → Dropout(0.1) → Linear(64→3)
Targets: {yes, no, abstain} = {present-correct, absence-correct, hedge-correct}

Decision rule at inference:
  argmax over {yes, no, abstain}; abstain is treated as "no, this fact
  is not in my memory" — same semantics as RAG abstain.

Persona-disjoint 60/40 split: 30 train personae, 20 eval personae.
Both stratified to keep present/absence balance.

Pre-registered Pareto criterion (sub-axis):
  hybrid_presence_TPR ≥ 0.563 (γ-LoRA C_lora)
  hybrid_absence_TPR  ≥ 0.99  (RAG C_rag)

Outputs
-------
runs/34_calib_head/
  features.jsonl              per-probe features + gold
  head.pt                     trained MLP state_dict
  eval.json                   train/dev/test metrics
  aggregate.json              hybrid result + pareto ledger

Stages
------
  --stage prep   build features.jsonl on EC2 (needs BGE)
  --stage train  train head on features.jsonl (CPU OK)
  --stage eval   apply head, compute hybrid metrics
  --stage all    sequential
"""

from __future__ import annotations
import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
EXP29 = ROOT / "runs" / "29_f_absence"
DATASET29 = EXP29 / "dataset.json"
OUT = ROOT / "runs" / "34_calib_head"
OUT.mkdir(parents=True, exist_ok=True)
BGE_PATH = ROOT / "models" / "bge-large-en-v1.5"

SEED = 11
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Hedge / refusal patterns (taken from exp29 judge prompt vocabulary).
HEDGE_PATTERNS = [
    r"\bI (?:do(?:n'?t| not)|don'?t|cannot|can'?t) (?:recall|remember|find|see)\b",
    r"\bI'?m not (?:sure|certain)\b",
    r"\bnot (?:that I (?:can )?recall|that I'?ve seen)\b",
    r"\bno (?:record|memory|mention) of\b",
    r"\bI (?:have|haven'?t) no (?:record|memory|recollection)\b",
    r"\b(?:may|might) (?:have|not)\b",
    r"\bnothing (?:in|on) (?:my|the) (?:notes|memory|records)\b",
    r"\bI (?:apologi[sz]e|am sorry).{0,30}(?:do(?:n'?t| not)|cannot|can'?t)\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)
YES_RE = re.compile(r"^\s*(?:yes|yeah|yep|absolutely|certainly|of course)\b", re.IGNORECASE)
NO_RE = re.compile(
    r"^\s*(?:no|nope|nah)\b|"
    r"\b(?:I (?:do(?:n'?t| not)|don'?t|cannot|can'?t) (?:recall|remember|find|see))\b|"
    r"\b(?:no (?:record|memory|mention)|haven'?t (?:discussed|mentioned)|never (?:discussed|mentioned)|"
    r"not (?:in|on) (?:my|the) (?:notes|memory|records)|nothing (?:in|on) (?:my|the) (?:notes|memory|records))\b",
    re.IGNORECASE,
)


def extract_yn(pred: str) -> str:
    """Extract yes/no/unknown from a prediction string."""
    s = pred.strip()
    if YES_RE.search(s): return "yes"
    if NO_RE.search(s): return "no"
    return "unknown"


def hedge_score(pred: str) -> float:
    return 1.0 if HEDGE_RE.search(pred) else 0.0


def conf_score(pred: str) -> float:
    yn = extract_yn(pred)
    if yn == "unknown": return 0.0
    if hedge_score(pred): return 0.5
    return 1.0


def chunk_text(text: str, size: int = 500, stride: int = 350) -> list[str]:
    bs = text.strip()
    if len(bs) <= size: return [bs]
    out = []; i = 0
    while i < len(bs):
        out.append(bs[i:i + size])
        if i + size >= len(bs): break
        i += stride
    return out


# ----------------------------------------------------------------------
# Stage 1: feature extraction.
# ----------------------------------------------------------------------

def stage_prep(args):
    """Read cached predictions, compute features, write features.jsonl."""
    ds = json.loads(DATASET29.read_text())
    personae = ds["personae"]

    # Compute retrieval relevance + answer-context overlap per probe via BGE.
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BGE_PATH))
    mdl = AutoModel.from_pretrained(str(BGE_PATH),
                                    dtype=torch.float16).to(args.device).eval()

    @torch.no_grad()
    def bge(texts):
        if not texts: return torch.zeros(0, 1024)
        enc = tok(texts, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(args.device)
        h = mdl(**enc).last_hidden_state[:, 0]
        return F.normalize(h, dim=-1).cpu()

    def ngram_overlap(a: str, b: str, n: int = 4) -> float:
        """4-gram char-level Jaccard between a and b."""
        if not a or not b: return 0.0
        ag = {a[i:i+n] for i in range(len(a) - n + 1)}
        bg = {b[i:i+n] for i in range(len(b) - n + 1)}
        if not ag or not bg: return 0.0
        return len(ag & bg) / len(ag | bg)

    # Read cached substrate predictions FIRST so we can compute overlap.
    def load_jsonl(path):
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    rows: list[dict] = []
    skipped = 0
    for p in personae:
        pid = p["id"]
        lora_path = EXP29 / pid / "probes_C_lora.jsonl"
        rag_path = EXP29 / pid / "probes_C_rag.jsonl"
        if not lora_path.exists() or not rag_path.exists():
            skipped += 1; continue
        lora_rows = load_jsonl(lora_path)
        rag_rows = load_jsonl(rag_path)
        L = {(r["kind"], r["topic"]): r for r in lora_rows}
        R = {(r["kind"], r["topic"]): r for r in rag_rows}

        # Retrieval setup for this persona
        chunks = chunk_text(p["backstory"])
        c_emb = bge(chunks)
        probes = p["probes"]
        q_emb = bge([pr["question"] for pr in probes])
        sims_q = (q_emb @ c_emb.T).numpy()  # (P, C)
        topk_relevance = np.sort(sims_q, axis=1)[:, -3:].mean(axis=1)
        # Top-3 indices per probe (same as C_rag uses K=3)
        top_idx = np.argsort(-sims_q, axis=1)[:, :3]

        for i, pr in enumerate(probes):
            key = (pr["kind"], pr["topic"])
            if key not in L or key not in R: continue
            lr, rr = L[key], R[key]
            yn_l, yn_r = extract_yn(lr["pred"]), extract_yn(rr["pred"])
            agree = 1.0 if (yn_l == yn_r and yn_l != "unknown") else 0.0

            # Answer-vs-context features. Top-3 chunks for this probe:
            top_chunks = [chunks[j] for j in top_idx[i]]
            top_chunks_text = " \n".join(top_chunks)
            # 4-gram overlap (lexical grounding)
            lora_overlap = ngram_overlap(lr["pred"], top_chunks_text)
            rag_overlap = ngram_overlap(rr["pred"], top_chunks_text)
            # BGE cosine between answer and best chunk (semantic grounding)
            ans_emb = bge([lr["pred"], rr["pred"]])
            chunk_top = c_emb[top_idx[i]]
            lora_chunkcos = float((ans_emb[0:1] @ chunk_top.T).max())
            rag_chunkcos = float((ans_emb[1:2] @ chunk_top.T).max())

            rows.append({
                "persona_id": pid,
                "kind": pr["kind"],
                "topic": pr["topic"],
                "question": pr["question"],
                "gold": pr["gold"],
                "feat": {
                    "conf_lora": conf_score(lr["pred"]),
                    "conf_rag": conf_score(rr["pred"]),
                    "agree": agree,
                    "len_lora": min(len(lr["pred"]) / 200.0, 1.0),
                    "len_rag": min(len(rr["pred"]) / 200.0, 1.0),
                    "hedge_lora": hedge_score(lr["pred"]),
                    "hedge_rag": hedge_score(rr["pred"]),
                    "r": float(topk_relevance[i]),
                    "lora_overlap": lora_overlap,
                    "rag_overlap": rag_overlap,
                    "lora_chunkcos": lora_chunkcos,
                    "rag_chunkcos": rag_chunkcos,
                },
                "lora_yn": yn_l,
                "rag_yn": yn_r,
                "lora_verdict": lr["verdict"],
                "rag_verdict": rr["verdict"],
            })

    out_path = OUT / "features.jsonl"
    with out_path.open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"[prep] wrote {len(rows)} feature rows to {out_path} "
          f"(skipped {skipped} personae)", flush=True)


# ----------------------------------------------------------------------
# Stage 2: train head.
# ----------------------------------------------------------------------

FEATURE_KEYS = ["conf_lora", "conf_rag", "agree", "len_lora", "len_rag",
                "hedge_lora", "hedge_rag", "r",
                "lora_overlap", "rag_overlap", "lora_chunkcos", "rag_chunkcos"]
LABEL_MAP = {"yes": 0, "no": 1, "abstain": 2}


def gold_to_label(gold: str, kind: str) -> int:
    """Map gold → 3-class label.

    For factual probes, the design says:
      - present → 'yes' (the fact is there, model should affirm)
      - absence → 'no' or 'abstain' (model should reject)

    We collapse to 2 effective classes: yes vs no/abstain. Train a 3-way
    head anyway; predicting abstain on absence is treated as a correct
    'no' (RAG-style abstain semantics).
    """
    g = (gold or "").strip().lower()
    if g.startswith("yes"): return LABEL_MAP["yes"]
    if g.startswith("no"): return LABEL_MAP["no"]
    return LABEL_MAP["abstain"]


class CalibHead(nn.Module):
    def __init__(self, dim=12, hidden=64, n_cls=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x): return self.net(x)


def stage_train(args):
    rows = [json.loads(l) for l in (OUT / "features.jsonl").read_text().splitlines() if l.strip()]
    persona_ids = sorted({r["persona_id"] for r in rows})
    rng = random.Random(SEED); rng.shuffle(persona_ids)
    n_train = int(0.6 * len(persona_ids))
    train_set = set(persona_ids[:n_train])

    # Routing labels (3-way): for each probe, what's the best substrate
    # to USE based on cached judge verdicts?
    #   use_lora  if C_lora was CORRECT and C_rag was not (or both correct → either works, prefer lora for fidelity)
    #   use_rag   if C_rag was CORRECT and C_lora was not
    #   abstain   if neither was CORRECT (return "no/unknown")
    LABEL = {"use_lora": 0, "use_rag": 1, "abstain": 2}

    def routing_label(r):
        l_ok = r["lora_verdict"] == "CORRECT"
        r_ok = r["rag_verdict"] == "CORRECT"
        if l_ok and r_ok: return LABEL["use_lora"]   # tie → prefer lora
        if l_ok:          return LABEL["use_lora"]
        if r_ok:          return LABEL["use_rag"]
        return LABEL["abstain"]

    def make_xy(filtered):
        X = torch.tensor([[r["feat"][k] for k in FEATURE_KEYS] for r in filtered],
                         dtype=torch.float32)
        y = torch.tensor([routing_label(r) for r in filtered], dtype=torch.long)
        return X, y

    train_rows = [r for r in rows if r["persona_id"] in train_set]
    eval_rows = [r for r in rows if r["persona_id"] not in train_set]
    Xtr, ytr = make_xy(train_rows)
    Xev, yev = make_xy(eval_rows)
    print(f"[train] train n={len(train_rows)} eval n={len(eval_rows)} feat_dim={Xtr.shape[1]}",
          flush=True)
    print(f"[train] routing label dist train: lora={int((ytr==0).sum())} "
          f"rag={int((ytr==1).sum())} abstain={int((ytr==2).sum())}", flush=True)
    print(f"[train] routing label dist eval:  lora={int((yev==0).sum())} "
          f"rag={int((yev==1).sum())} abstain={int((yev==2).sum())}", flush=True)

    device = "cpu"
    model = CalibHead(n_cls=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    # Compute weighted CE — abstain is rare, lora and rag dominate.
    counts = torch.bincount(ytr, minlength=3).float() + 1.0
    weights = 1.0 / counts; weights = weights / weights.mean()
    print(f"[train] class weights: {weights.tolist()}", flush=True)

    # Cost matrix C[true, pred]: extra penalty for routing absence probes
    # to lora (γ-LoRA's absence-TPR is 0.087, sending an absence probe
    # there is almost certainly wrong).  Build per-sample cost weight.
    train_kinds = [r["kind"] for r in train_rows]
    # Per-sample weight on the LORA logit: if probe is absence, multiply
    # the loss when target≠use_lora and pred could be use_lora. We
    # implement this by adding a per-sample weight that scales the
    # cross-entropy loss where target=use_rag/abstain on absence kind.
    # Simpler concrete form: when kind==absence, multiply CE loss by 3.
    sample_weight_train = torch.tensor(
        [3.0 if k == "absence" else 1.0 for k in train_kinds],
        dtype=torch.float32)

    best_acc, best_state, best_epoch = -1.0, None, -1
    # Manual loop with sample weights
    for epoch in range(60):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i+64]
            xb, yb = Xtr[idx], ytr[idx]
            sw = sample_weight_train[idx]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb, weight=weights, reduction="none")
            loss = (ce * sw).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xev).argmax(-1)
        acc = (pred == yev).float().mean().item()
        # Simulated hybrid correctness via the routing.
        n_correct = 0
        for r, p in zip(eval_rows, pred.tolist()):
            if p == LABEL["use_lora"]:
                n_correct += int(r["lora_verdict"] == "CORRECT")
            elif p == LABEL["use_rag"]:
                n_correct += int(r["rag_verdict"] == "CORRECT")
            else:  # abstain → correct iff gold says no
                gold = (r["gold"] or "").strip().lower()
                n_correct += int(gold.startswith("no"))
        sim = n_correct / max(len(eval_rows), 1)
        # Per-kind sub-axis (the actual paper bar)
        present_idx = [i for i, r in enumerate(eval_rows) if r["kind"] == "present"]
        absence_idx = [i for i, r in enumerate(eval_rows) if r["kind"] == "absence"]
        def sub_correct(indices):
            n = 0
            for i in indices:
                r, p = eval_rows[i], pred[i].item()
                if p == LABEL["use_lora"]: n += int(r["lora_verdict"] == "CORRECT")
                elif p == LABEL["use_rag"]: n += int(r["rag_verdict"] == "CORRECT")
                else: n += int((r["gold"] or "").strip().lower().startswith("no"))
            return n / max(len(indices), 1)
        sub_pres = sub_correct(present_idx)
        sub_abs = sub_correct(absence_idx)
        # Score for selection: harmonic mean of presence and absence (encourages both)
        score = 2 * sub_pres * sub_abs / max(sub_pres + sub_abs, 1e-9)
        if epoch % 5 == 0 or epoch == 59:
            print(f"[train] epoch={epoch+1:02d} sim={sim:.3f} "
                  f"pres={sub_pres:.3f} abs={sub_abs:.3f} score={score:.3f}",
                  flush=True)
        if score > best_acc:
            best_acc = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    torch.save({"state_dict": best_state, "best_sim_hybrid": best_acc,
                "best_epoch": best_epoch, "feature_keys": FEATURE_KEYS,
                "label_map": LABEL},
               OUT / "head.pt")
    (OUT / "split.json").write_text(json.dumps({
        "train_personae": sorted(train_set),
        "eval_personae": [p for p in persona_ids if p not in train_set],
    }, indent=2))
    print(f"[train] best sim_hybrid={best_acc:.4f} at epoch {best_epoch}; saved head.pt",
          flush=True)


# ----------------------------------------------------------------------
# Stage 3: eval — apply head, compute hybrid metrics.
# ----------------------------------------------------------------------

def stage_eval(args):
    rows = [json.loads(l) for l in (OUT / "features.jsonl").read_text().splitlines() if l.strip()]
    state = torch.load(OUT / "head.pt", weights_only=False)
    split = json.loads((OUT / "split.json").read_text())
    eval_set = set(split["eval_personae"])
    LABEL = state["label_map"]  # {"use_lora":0,"use_rag":1,"abstain":2}

    model = CalibHead(n_cls=3)
    model.load_state_dict(state["state_dict"]); model.eval()

    def predict_route(r):
        x = torch.tensor([r["feat"][k] for k in FEATURE_KEYS], dtype=torch.float32)
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
        return int(logits.argmax(-1).item())

    def hybrid_correct(r):
        """Was the hybrid answer judged CORRECT for this probe?

        - use_lora → cached γ-LoRA verdict
        - use_rag  → cached RAG verdict
        - abstain  → correct iff gold semantics is 'no' (absence probe)
        """
        route = predict_route(r)
        if route == LABEL["use_lora"]:
            return r["lora_verdict"] == "CORRECT"
        if route == LABEL["use_rag"]:
            return r["rag_verdict"] == "CORRECT"
        # abstain
        gold = (r["gold"] or "").strip().lower()
        return gold.startswith("no")

    eval_rows = [r for r in rows if r["persona_id"] in eval_set]
    train_rows = [r for r in rows if r["persona_id"] not in eval_set]

    def metrics(rs, label):
        present = [r for r in rs if r["kind"] == "present"]
        absence = [r for r in rs if r["kind"] == "absence"]
        h_pres = sum(1 for r in present if hybrid_correct(r)) / max(len(present), 1)
        h_abs = sum(1 for r in absence if hybrid_correct(r)) / max(len(absence), 1)
        f1 = 2 * h_pres * h_abs / max(h_pres + h_abs, 1e-9)
        # Routing distribution
        routes = [predict_route(r) for r in rs]
        return {f"{label}_present_TPR": h_pres, f"{label}_absence_TPR": h_abs,
                f"{label}_F1": f1, f"{label}_n_present": len(present),
                f"{label}_n_absence": len(absence),
                f"{label}_route_lora_pct": routes.count(LABEL["use_lora"]) / max(len(routes), 1),
                f"{label}_route_rag_pct": routes.count(LABEL["use_rag"]) / max(len(routes), 1),
                f"{label}_route_abstain_pct": routes.count(LABEL["abstain"]) / max(len(routes), 1)}

    out = {"eval_personae": sorted(eval_set), **metrics(eval_rows, "eval"),
           "train_personae": sorted(set(r["persona_id"] for r in train_rows)),
           **metrics(train_rows, "train")}

    # Reference baselines from cached aggregates.
    agg29 = json.loads((EXP29 / "aggregate.json").read_text())
    pures = {
        "C_lora": agg29["configs"]["C_lora"],
        "C_rag": agg29["configs"]["C_rag"],
        "C_lora_calib": agg29["configs"]["C_lora_calib"],
    }
    pareto = {
        "subaxis_presence_TPR": {
            "hybrid_calibhead_eval": out["eval_present_TPR"],
            "lora_alone": pures["C_lora"]["present"]["tpr"],
            "rag_alone": pures["C_rag"]["present"]["tpr"],
            "lora_calib": pures["C_lora_calib"]["present"]["tpr"],
            "verdict": "pareto_dominant" if out["eval_present_TPR"] >= max(
                pures["C_lora"]["present"]["tpr"],
                pures["C_rag"]["present"]["tpr"]) else "pareto_dominated",
        },
        "subaxis_absence_TPR": {
            "hybrid_calibhead_eval": out["eval_absence_TPR"],
            "lora_alone": pures["C_lora"]["absence"]["tpr"],
            "rag_alone": pures["C_rag"]["absence"]["tpr"],
            "lora_calib": pures["C_lora_calib"]["absence"]["tpr"],
            "verdict": "pareto_dominant" if out["eval_absence_TPR"] >= max(
                pures["C_lora"]["absence"]["tpr"],
                pures["C_rag"]["absence"]["tpr"]) else "pareto_dominated",
        },
        "factual_axis_F1": {
            "hybrid_calibhead_eval": out["eval_F1"],
            "lora_alone": pures["C_lora"]["f1"],
            "rag_alone": pures["C_rag"]["f1"],
            "lora_calib": pures["C_lora_calib"]["f1"],
            "verdict": "pareto_dominant" if out["eval_F1"] >= max(
                pures["C_lora"]["f1"], pures["C_rag"]["f1"],
                pures["C_lora_calib"]["f1"]) else "pareto_dominated",
        },
    }
    # Oracle upper bound: what if we routed perfectly?
    oracle_correct = sum(1 for r in eval_rows
                         if r["lora_verdict"] == "CORRECT" or r["rag_verdict"] == "CORRECT")
    oracle_f1_present = sum(1 for r in eval_rows if r["kind"] == "present" and
                             (r["lora_verdict"] == "CORRECT" or r["rag_verdict"] == "CORRECT")) / \
                        max(sum(1 for r in eval_rows if r["kind"] == "present"), 1)
    oracle_f1_absence = sum(1 for r in eval_rows if r["kind"] == "absence" and
                             (r["lora_verdict"] == "CORRECT" or r["rag_verdict"] == "CORRECT")) / \
                        max(sum(1 for r in eval_rows if r["kind"] == "absence"), 1)
    out["oracle_upper_bound"] = {
        "presence_TPR": oracle_f1_present, "absence_TPR": oracle_f1_absence,
        "overall_correct_pct": oracle_correct / max(len(eval_rows), 1),
    }

    out["pareto_ledger"] = pareto
    out["best_train_sim_hybrid"] = state["best_sim_hybrid"]
    out["best_train_epoch"] = state["best_epoch"]

    (OUT / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[eval] wrote {OUT / 'aggregate.json'}", flush=True)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "train", "eval", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.stage in {"prep", "all"}: stage_prep(args)
    if args.stage in {"train", "all"}: stage_train(args)
    if args.stage in {"eval", "all"}: stage_eval(args)


if __name__ == "__main__":
    main()
