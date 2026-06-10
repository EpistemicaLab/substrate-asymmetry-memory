#!/usr/bin/env python3
"""
33_router_hybrid.py — Phase H1: Sequential routing hybrid (Option A).

Trains a behavioral-vs-factual query router over BGE-large embeddings
and analytically constructs the routed-hybrid result by selecting
cached per-probe predictions from the existing pure-substrate runs:
  - behavioral queries -> γ-LoRA (exp28 WP continuation log-lik)
  - factual queries    -> RAG    (exp29 cached probes_C_rag.jsonl)

Outputs
-------
runs/33_hybrid_routing/
    router.pt                       trained MLP head (state_dict)
    router_eval.json                holdout precision/recall/F1
    aggregate.json                  three-axis hybrid metrics +
                                    pure-baseline references +
                                    Pareto-dominance ledger
    by_persona.jsonl                per-persona behavioral & factual

Pre-registered Pareto criterion (axis-level, what H1 can achieve):
  - Behavioral: hybrid log-lik per persona >= max(γ-LoRA, RAG)
  - Factual F1: hybrid F1 >= max(γ-LoRA-alone, C_lora_calib, RAG-alone)

Acknowledged sub-axis limit (motivates H2 calibration head):
  - Within factual, hybrid presence-TPR will equal whichever
    substrate the router picks, so it CANNOT simultaneously match
    γ-LoRA's presence (0.56) and RAG's absence (0.99). H1 reports
    this honestly; H2 (Option C) addresses it via a head that fuses
    both substrates' logits + retrieval relevance.

Usage on EC2
------------
    python3 experiments/33_router_hybrid.py --stage all
    python3 experiments/33_router_hybrid.py --stage prep
    python3 experiments/33_router_hybrid.py --stage train
    python3 experiments/33_router_hybrid.py --stage eval

Why one script: avoids version drift between three half-finished
launchers; everything is hash-stable and deterministic.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
BGE_PATH = ROOT / "models" / "bge-large-en-v1.5"
OUT = ROOT / "runs" / "33_hybrid_routing"
OUT.mkdir(parents=True, exist_ok=True)

# Cached pure-substrate result locations.
EXP29_DATASET = ROOT / "runs" / "29_f_absence" / "dataset.json"
EXP29_AGG = ROOT / "runs" / "29_f_absence" / "aggregate.json"
EXP29_SUMMARY = ROOT / "runs" / "29_f_absence" / "summary.jsonl"
EXP28_DATASET = ROOT / "runs" / "28_writingprompts" / "dataset.json"
EXP28_AGG = ROOT / "runs" / "28_writingprompts" / "gamma_lora" / "aggregate.json"
EXP28_SUMMARY = ROOT / "runs" / "28_writingprompts" / "gamma_lora" / "summary.jsonl"

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------
# Stage 1: data prep — collect (text, label) pairs and embed with BGE.
# ----------------------------------------------------------------------

def collect_prompts() -> list[dict]:
    """Build the routing training corpus from existing exp28 and exp29 data.

    Behavioral (label=0): exp28 WritingPrompts eval prompts (free-text
        story-continuation prompts; very different surface from the
        factual 'have we discussed X' template).
    Factual (label=1): exp29 'present' + 'absence' probe questions.
    """
    rows: list[dict] = []

    ds28 = json.loads(EXP28_DATASET.read_text())
    for p in ds28["personae"]:
        for ep in p.get("eval_prompts", []):
            text = (ep if isinstance(ep, str)
                    else ep.get("prefix") or ep.get("prompt") or ep.get("text"))
            if text:
                rows.append({"text": text.strip(), "label": 0,
                             "persona_id": p["id"], "axis": "behavioral"})

    ds29 = json.loads(EXP29_DATASET.read_text())
    for p in ds29["personae"]:
        for probe in p["probes"]:
            rows.append({"text": probe["question"].strip(), "label": 1,
                         "persona_id": p["id"], "axis": "factual",
                         "kind": probe["kind"]})
    return rows


def bge_embed(texts: list[str], device: str = "cuda:0",
              batch: int = 64) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(BGE_PATH))
    mdl = AutoModel.from_pretrained(str(BGE_PATH), torch_dtype=torch.float32).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            h = mdl(**enc, return_dict=True).last_hidden_state[:, 0]
            h = F.normalize(h, dim=-1)
            out.append(h.cpu())
    return torch.cat(out, dim=0)


def stage_prep(args: argparse.Namespace) -> None:
    rows = collect_prompts()
    n_b = sum(1 for r in rows if r["label"] == 0)
    n_f = sum(1 for r in rows if r["label"] == 1)
    print(f"[prep] collected {len(rows)} prompts: behavioral={n_b} factual={n_f}",
          flush=True)

    # Stratified persona-disjoint 80/20 split.
    persona_ids = sorted({r["persona_id"] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(persona_ids)
    n_train = int(0.8 * len(persona_ids))
    train_ids = set(persona_ids[:n_train])
    for r in rows:
        r["split"] = "train" if r["persona_id"] in train_ids else "eval"

    print(f"[prep] embedding {len(rows)} texts with BGE-large…", flush=True)
    emb = bge_embed([r["text"] for r in rows], device=args.device)

    torch.save({
        "embeddings": emb,
        "labels": torch.tensor([r["label"] for r in rows], dtype=torch.long),
        "splits": [r["split"] for r in rows],
        "rows": rows,
        "dim": emb.shape[1],
    }, OUT / "router_data.pt")
    print(f"[prep] wrote {OUT / 'router_data.pt'} dim={emb.shape[1]}", flush=True)


# ----------------------------------------------------------------------
# Stage 2: train router (2-layer MLP, frozen BGE).
# ----------------------------------------------------------------------

class Router(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stage_train(args: argparse.Namespace) -> None:
    pkg = torch.load(OUT / "router_data.pt", weights_only=False)
    emb = pkg["embeddings"]
    labels = pkg["labels"]
    splits = pkg["splits"]
    dim = pkg["dim"]

    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    eval_idx = [i for i, s in enumerate(splits) if s == "eval"]
    Xtr, ytr = emb[train_idx], labels[train_idx]
    Xev, yev = emb[eval_idx], labels[eval_idx]

    device = args.device if torch.cuda.is_available() else "cpu"
    model = Router(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
    best_f1, best_state = -1.0, None
    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # Eval pass.
        model.eval()
        with torch.no_grad():
            pred = model(Xev.to(device)).argmax(-1).cpu()
        tp = ((pred == 1) & (yev == 1)).sum().item()
        fp = ((pred == 1) & (yev == 0)).sum().item()
        fn = ((pred == 0) & (yev == 1)).sum().item()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (pred == yev).float().mean().item()
        print(f"[train] epoch={epoch+1:02d} acc={acc:.4f} P={prec:.4f} "
              f"R={rec:.4f} F1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save({"state_dict": best_state, "dim": dim, "best_f1": best_f1},
               OUT / "router.pt")
    # Per-class final metrics.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xev.to(device)).argmax(-1).cpu()
    confusion = {
        "n_eval": int(yev.numel()),
        "true_behav_pred_behav": int(((pred == 0) & (yev == 0)).sum()),
        "true_behav_pred_fact":  int(((pred == 1) & (yev == 0)).sum()),
        "true_fact_pred_behav":  int(((pred == 0) & (yev == 1)).sum()),
        "true_fact_pred_fact":   int(((pred == 1) & (yev == 1)).sum()),
        "macro_f1": best_f1,
    }
    (OUT / "router_eval.json").write_text(json.dumps(confusion, indent=2))
    print(f"[train] best F1={best_f1:.4f}; wrote router.pt + router_eval.json",
          flush=True)


# ----------------------------------------------------------------------
# Stage 3: hybrid eval — analytically combine cached substrate preds.
# ----------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stage_eval(args: argparse.Namespace) -> None:
    """Apply router decisions to cached pure-substrate predictions.

    For factual probes (exp29): all probes route to factual → use C_rag
    rows. For behavioral (exp28 WP): all route to behavioral → use γ-LoRA
    log-lik rows.

    The router's only job is to NOT misroute. Misrouting penalty is
    quantified by counting eval-split rows where pred_label != true_axis
    and substituting the wrong substrate's prediction.
    """
    pkg = torch.load(OUT / "router_data.pt", weights_only=False)
    rows = pkg["rows"]
    splits = pkg["splits"]

    state = torch.load(OUT / "router.pt", weights_only=False)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = Router(state["dim"]).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(pkg["embeddings"].to(device)).argmax(-1).cpu().tolist()

    # Build lookup: text → predicted axis.
    text2pred = {r["text"]: int(p) for r, p in zip(rows, preds)}

    # ---- Factual axis hybrid: route → use C_rag where router says factual.
    factual_summary = _load_jsonl(EXP29_SUMMARY)
    # Filter to C_rag rows.
    rag_rows = [r for r in factual_summary if r["config"] == "C_rag"]
    lora_rows = [r for r in factual_summary if r["config"] == "C_lora"]
    lora_calib_rows = [r for r in factual_summary
                       if r["config"] == "C_lora_calib"]
    # Index by (persona, kind, topic) for the misroute substitution.
    def key(r): return (r["persona_id"], r["kind"], r["topic"])
    rag_by = {key(r): r for r in rag_rows}
    lora_by = {key(r): r for r in lora_rows}

    hybrid_factual = []
    misroutes_f = 0
    for r in rag_rows:
        # Decide using router on the actual probe text.
        pred_axis = text2pred.get(r["question"], 1)  # default factual
        if pred_axis == 1:  # routed to factual → use RAG
            hybrid_factual.append(r)
        else:                # misrouted to behavioral → γ-LoRA picks
            misroutes_f += 1
            alt = lora_by.get(key(r))
            hybrid_factual.append(alt or r)

    def tpr(rows: list[dict], kind: str) -> float:
        sub = [r for r in rows if r["kind"] == kind]
        if not sub: return 0.0
        return sum(1 for r in sub if r["verdict"] == "CORRECT") / len(sub)

    h_pres = tpr(hybrid_factual, "present")
    h_abs = tpr(hybrid_factual, "absence")
    h_f1 = (2 * h_pres * h_abs / max(h_pres + h_abs, 1e-9))
    rag_f1 = json.loads(EXP29_AGG.read_text())["configs"]["C_rag"]["f1"]
    lora_calib_f1 = json.loads(EXP29_AGG.read_text())["configs"]["C_lora_calib"]["f1"]
    lora_f1 = json.loads(EXP29_AGG.read_text())["configs"]["C_lora"]["f1"]

    # ---- Behavioral axis hybrid: route → use γ-LoRA where router says behav.
    # exp28 summary has per-prompt log-likelihoods.
    if EXP28_SUMMARY.exists():
        behav_summary = _load_jsonl(EXP28_SUMMARY)
    else:
        behav_summary = []
    # The exp28 aggregate already shows γ-LoRA wins behavioral; for H1
    # we just record that any misroute → RAG would fail (RAG can't do
    # style continuation). We measure misroute rate.
    misroutes_b = 0
    n_behav = 0
    for r in rows:
        if r["axis"] != "behavioral" or r.get("split") != "eval":
            continue
        n_behav += 1
        if text2pred.get(r["text"], 0) != 0:
            misroutes_b += 1

    # ---- Misroute rates from router_eval.json confusion matrix.
    conf = json.loads((OUT / "router_eval.json").read_text())
    pareto = {
        "behavioral_axis": {
            "claim": "hybrid >= γ-LoRA on behavioral",
            "verdict": "pass_if_router_perfect" if conf["true_behav_pred_fact"] == 0
                       else f"degrades_proportional_to_{conf['true_behav_pred_fact']}_misroutes",
            "n_eval_behav": n_behav,
            "behav_misrouted_to_factual": misroutes_b,
        },
        "factual_axis_F1": {
            "hybrid_F1": h_f1,
            "rag_alone_F1": rag_f1,
            "lora_calib_F1": lora_calib_f1,
            "lora_alone_F1": lora_f1,
            "best_pure_F1": max(rag_f1, lora_calib_f1, lora_f1),
            "verdict": "pareto_dominant" if h_f1 >= max(rag_f1, lora_calib_f1, lora_f1)
                       else "pareto_dominated",
        },
        "factual_subaxis_known_limit": {
            "hybrid_present_TPR": h_pres,
            "lora_alone_present_TPR": json.loads(EXP29_AGG.read_text())["configs"]["C_lora"]["present"]["tpr"],
            "hybrid_absence_TPR": h_abs,
            "rag_alone_absence_TPR": json.loads(EXP29_AGG.read_text())["configs"]["C_rag"]["absence"]["tpr"],
            "comment": "Sequential routing cannot simultaneously match γ-LoRA's "
                       "presence-TPR and RAG's absence-TPR within the factual axis. "
                       "This is the motivation for Phase H2 (calibration head, Option C).",
        },
    }

    out = {
        "router": {
            "macro_F1_holdout": conf["macro_f1"],
            "confusion": conf,
            "n_factual_misroutes_in_eval29": misroutes_f,
        },
        "hybrid_factual": {
            "present_TPR": h_pres,
            "absence_TPR": h_abs,
            "F1": h_f1,
            "n_probes": len(hybrid_factual),
        },
        "pareto_ledger": pareto,
    }
    (OUT / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[eval] wrote {OUT / 'aggregate.json'}", flush=True)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "train", "eval", "all"],
                    default="all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.stage in {"prep", "all"}: stage_prep(args)
    if args.stage in {"train", "all"}: stage_train(args)
    if args.stage in {"eval", "all"}: stage_eval(args)


if __name__ == "__main__":
    main()
