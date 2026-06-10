#!/usr/bin/env python3
"""
36_calib_head_logits.py — Phase H2 C-B trainer.

Same head architecture as exp34, but features now include logit-grade
confidence + entropy + margin from exp35 instead of text-derived proxies.

Inputs:
  runs/35_logit_features/V3_P_*/features.jsonl  (lora & rag preds + logits)
  runs/29_f_absence/V3_P_*/probes_C_lora.jsonl  (judge verdicts to use as labels)
  runs/29_f_absence/V3_P_*/probes_C_rag.jsonl

Output:
  runs/36_calib_head_logits/{features.jsonl, head.pt, split.json, aggregate.json}
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from experiments._base_model import run_dir as _run_dir  # noqa: E402

EXP29 = _run_dir(ROOT / "runs" / "29_f_absence")
EXP35 = _run_dir(ROOT / "runs" / "35_logit_features")
OUT = _run_dir(ROOT / "runs" / "36_calib_head_logits")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 11
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Logit features extracted in exp35.
LOGIT_KEYS = ["top1_prob", "entropy", "margin", "logprob_sum",
              "mean_entropy", "n_gen_tokens"]
# 12 features total: 6 lora-side + 6 rag-side + r (relevance) + agree.
FEATURE_KEYS = ([f"lora_{k}" for k in LOGIT_KEYS] +
                [f"rag_{k}" for k in LOGIT_KEYS] +
                ["r", "agree"])


def stage_prep():
    """Join exp35 logits + exp29 verdicts → features.jsonl."""
    rows: list[dict] = []
    for pdir in sorted(EXP35.glob("V3_P_*")):
        pid = pdir.name
        feat_path = pdir / "features.jsonl"
        lora_path = EXP29 / pid / "probes_C_lora.jsonl"
        rag_path = EXP29 / pid / "probes_C_rag.jsonl"
        if not feat_path.exists() or not lora_path.exists() or not rag_path.exists():
            continue
        feats = [json.loads(l) for l in feat_path.read_text().splitlines() if l.strip()]
        L = {(json.loads(l)["kind"], json.loads(l)["topic"]): json.loads(l)
             for l in lora_path.read_text().splitlines() if l.strip()}
        R = {(json.loads(l)["kind"], json.loads(l)["topic"]): json.loads(l)
             for l in rag_path.read_text().splitlines() if l.strip()}
        for f in feats:
            key = (f["kind"], f["topic"])
            if key not in L or key not in R: continue
            lr = L[key]; rr = R[key]
            # n_gen_tokens scaling (clip to [0,1] via /80)
            def norm_feat(x):
                out = dict(x)
                out["n_gen_tokens"] = min(out.get("n_gen_tokens", 0) / 80.0, 1.0)
                return out
            l_feat = norm_feat(f["lora_logit_feat"])
            r_feat = norm_feat(f["rag_logit_feat"])
            # Agreement: do lora and rag answers extract the same yes/no?
            def yn(s):
                s = s.lower().strip()
                if s.startswith(("yes", "yeah", "yep")): return "yes"
                if s.startswith(("no", "nope", "nah")): return "no"
                if "don't recall" in s or "no record" in s or "haven't" in s:
                    return "no"
                return "unknown"
            agree = 1.0 if yn(f["lora_pred"]) == yn(f["rag_pred"]) and \
                          yn(f["lora_pred"]) != "unknown" else 0.0
            row = {
                "persona_id": pid, "kind": f["kind"], "topic": f["topic"],
                "question": f["question"], "gold": f["gold"],
                "feat": {**{f"lora_{k}": l_feat[k] for k in LOGIT_KEYS},
                         **{f"rag_{k}": r_feat[k] for k in LOGIT_KEYS},
                         "r": f["r"], "agree": agree},
                "lora_verdict": lr["verdict"], "rag_verdict": rr["verdict"],
            }
            rows.append(row)
    out_path = OUT / "features.jsonl"
    with out_path.open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"[prep] wrote {len(rows)} rows to {out_path}")


class CalibHead(nn.Module):
    def __init__(self, dim, hidden=64, n_cls=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x): return self.net(x)


def stage_train():
    rows = [json.loads(l) for l in (OUT / "features.jsonl").read_text().splitlines() if l.strip()]
    persona_ids = sorted({r["persona_id"] for r in rows})
    rng = random.Random(SEED); rng.shuffle(persona_ids)
    n_train = int(0.6 * len(persona_ids))
    train_set = set(persona_ids[:n_train])

    LABEL = {"use_lora": 0, "use_rag": 1, "abstain": 2}

    def routing_label(r):
        l_ok = r["lora_verdict"] == "CORRECT"
        rg_ok = r["rag_verdict"] == "CORRECT"
        if l_ok and rg_ok: return LABEL["use_lora"]
        if l_ok: return LABEL["use_lora"]
        if rg_ok: return LABEL["use_rag"]
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
    dim = Xtr.shape[1]
    print(f"[train] train n={len(train_rows)} eval n={len(eval_rows)} feat_dim={dim}")
    print(f"[train] train labels: lora={int((ytr==0).sum())} rag={int((ytr==1).sum())} "
          f"abstain={int((ytr==2).sum())}")
    print(f"[train] eval labels:  lora={int((yev==0).sum())} rag={int((yev==1).sum())} "
          f"abstain={int((yev==2).sum())}")

    counts = torch.bincount(ytr, minlength=3).float() + 1.0
    weights = 1.0 / counts; weights = weights / weights.mean()

    train_kinds = [r["kind"] for r in train_rows]
    sample_weight_train = torch.tensor(
        [3.0 if k == "absence" else 1.0 for k in train_kinds], dtype=torch.float32)

    model = CalibHead(dim=dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_score, best_state, best_epoch = -1.0, None, -1
    for epoch in range(80):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i+64]
            xb, yb, sw = Xtr[idx], ytr[idx], sample_weight_train[idx]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb, weight=weights, reduction="none")
            ((ce * sw).mean()).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xev).argmax(-1)
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
        score = 2 * sub_pres * sub_abs / max(sub_pres + sub_abs, 1e-9)
        if epoch % 5 == 0 or epoch == 79:
            print(f"[train] epoch={epoch+1:02d} pres={sub_pres:.3f} "
                  f"abs={sub_abs:.3f} score={score:.3f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    torch.save({"state_dict": best_state, "best_score": best_score,
                "best_epoch": best_epoch, "feature_keys": FEATURE_KEYS,
                "label_map": LABEL, "dim": dim},
               OUT / "head.pt")
    (OUT / "split.json").write_text(json.dumps({
        "train_personae": sorted(train_set),
        "eval_personae": [p for p in persona_ids if p not in train_set],
    }, indent=2))
    print(f"[train] best score={best_score:.4f} at epoch {best_epoch}")


def stage_eval():
    rows = [json.loads(l) for l in (OUT / "features.jsonl").read_text().splitlines() if l.strip()]
    state = torch.load(OUT / "head.pt", weights_only=False)
    split = json.loads((OUT / "split.json").read_text())
    eval_set = set(split["eval_personae"])
    LABEL = state["label_map"]

    model = CalibHead(dim=state["dim"])
    model.load_state_dict(state["state_dict"]); model.eval()

    def predict(r):
        x = torch.tensor([r["feat"][k] for k in FEATURE_KEYS], dtype=torch.float32)
        with torch.no_grad():
            return int(model(x.unsqueeze(0)).argmax(-1).item())

    def hybrid_correct(r):
        p = predict(r)
        if p == LABEL["use_lora"]: return r["lora_verdict"] == "CORRECT"
        if p == LABEL["use_rag"]: return r["rag_verdict"] == "CORRECT"
        return (r["gold"] or "").strip().lower().startswith("no")

    eval_rows = [r for r in rows if r["persona_id"] in eval_set]

    def metrics(rs, label):
        present = [r for r in rs if r["kind"] == "present"]
        absence = [r for r in rs if r["kind"] == "absence"]
        h_pres = sum(1 for r in present if hybrid_correct(r)) / max(len(present), 1)
        h_abs = sum(1 for r in absence if hybrid_correct(r)) / max(len(absence), 1)
        f1 = 2 * h_pres * h_abs / max(h_pres + h_abs, 1e-9)
        routes = [predict(r) for r in rs]
        return {f"{label}_present_TPR": h_pres, f"{label}_absence_TPR": h_abs,
                f"{label}_F1": f1,
                f"{label}_route_lora_pct": routes.count(LABEL["use_lora"]) / max(len(routes), 1),
                f"{label}_route_rag_pct": routes.count(LABEL["use_rag"]) / max(len(routes), 1),
                f"{label}_route_abstain_pct": routes.count(LABEL["abstain"]) / max(len(routes), 1)}

    out = {**metrics(eval_rows, "eval"),
           "n_eval_personae": len(eval_set), "n_eval_rows": len(eval_rows)}

    agg29 = json.loads((EXP29 / "aggregate.json").read_text())
    pures = agg29["configs"]
    pareto = {
        "presence_TPR": {
            "hybrid_logits": out["eval_present_TPR"],
            "lora_alone": pures["C_lora"]["present"]["tpr"],
            "rag_alone": pures["C_rag"]["present"]["tpr"],
            "verdict": "pareto_dominant" if out["eval_present_TPR"] >= max(
                pures["C_lora"]["present"]["tpr"],
                pures["C_rag"]["present"]["tpr"]) else "pareto_dominated",
        },
        "absence_TPR": {
            "hybrid_logits": out["eval_absence_TPR"],
            "lora_alone": pures["C_lora"]["absence"]["tpr"],
            "rag_alone": pures["C_rag"]["absence"]["tpr"],
            "verdict": "pareto_dominant" if out["eval_absence_TPR"] >= max(
                pures["C_lora"]["absence"]["tpr"],
                pures["C_rag"]["absence"]["tpr"]) else "pareto_dominated",
        },
        "F1": {
            "hybrid_logits": out["eval_F1"],
            "lora_alone": pures["C_lora"]["f1"],
            "rag_alone": pures["C_rag"]["f1"],
            "lora_calib": pures["C_lora_calib"]["f1"],
            "verdict": "pareto_dominant" if out["eval_F1"] >= max(
                pures["C_lora"]["f1"], pures["C_rag"]["f1"],
                pures["C_lora_calib"]["f1"]) else "pareto_dominated",
        },
    }
    # Oracle ceiling
    oracle_p = sum(1 for r in eval_rows if r["kind"] == "present" and
                    (r["lora_verdict"] == "CORRECT" or r["rag_verdict"] == "CORRECT")) / \
                max(sum(1 for r in eval_rows if r["kind"] == "present"), 1)
    oracle_a = sum(1 for r in eval_rows if r["kind"] == "absence" and
                    (r["lora_verdict"] == "CORRECT" or r["rag_verdict"] == "CORRECT")) / \
                max(sum(1 for r in eval_rows if r["kind"] == "absence"), 1)
    out["oracle_upper_bound"] = {"presence_TPR": oracle_p, "absence_TPR": oracle_a}
    out["pareto_ledger"] = pareto
    (OUT / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "train", "eval", "all"], default="all")
    args = ap.parse_args()
    if args.stage in {"prep", "all"}: stage_prep()
    if args.stage in {"train", "all"}: stage_train()
    if args.stage in {"eval", "all"}: stage_eval()


if __name__ == "__main__":
    main()
