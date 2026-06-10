#!/usr/bin/env python3
"""30_analyze.py — Frobenius decomposition of n=50 saved γ-LoRA adapters.

Same logic as G_analyze.py but on runs/30_mechanism with n up to 50.
Adds simple stat tests (Pearson p-value via t-distribution) for the
behavior–mechanism correlation rows.
"""
from __future__ import annotations
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))
from experiments._base_model import run_dir as _run_dir  # noqa: E402

G_DIR = _run_dir(ROOT / "runs" / "30_mechanism")
EXP29_DIR = _run_dir(ROOT / "runs" / "29_f_absence")
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]


def parse_lora_key(key):
    m = re.search(r"layers\.(\d+)\.[^.]+\.([a-z_]+_proj)\.lora_(A|B)\.", key)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def frob_decompose(adapter_path: Path):
    sd = load_file(str(adapter_path))
    pairs = defaultdict(dict)
    for k, v in sd.items():
        parsed = parse_lora_key(k)
        if parsed is None:
            continue
        layer, proj, ab = parsed
        pairs[(layer, proj)][ab] = v
    out = {}
    for (layer, proj), d in pairs.items():
        if "A" in d and "B" in d:
            A = d["A"].float()
            B = d["B"].float()
            dW = B @ A
            out[(layer, proj)] = float(torch.linalg.norm(dW).item())
    return out


def load_exp29_outcome(pid):
    p = EXP29_DIR / pid / "aggregate.json"
    if not p.exists():
        return {"present_tpr": None, "absence_tpr": None}
    d = json.load(open(p))
    cl = d.get("C_lora", {})
    return {
        "present_tpr": cl.get("present_tpr"),
        "absence_tpr": cl.get("absence_tpr"),
    }


def pearson_with_p(x, y):
    mask = ~(torch.isnan(x) | torch.isnan(y))
    n = int(mask.sum())
    if n < 4:
        return None, None, n
    x = x[mask]; y = y[mask]
    x = x - x.mean(); y = y - y.mean()
    denom = (x.norm() * y.norm()).item()
    if denom == 0:
        return None, None, n
    r = float((x * y).sum().item() / denom)
    if abs(r) >= 1.0:
        return r, 0.0, n
    # two-sided p via t = r * sqrt((n-2)/(1-r^2))
    t = r * math.sqrt(max(n - 2, 1) / max(1 - r * r, 1e-12))
    # rough two-sided p approx with n-2 dof — use survival from normal at large n
    # for n<=50 use a small-n approx via incomplete-beta would be cleaner;
    # we just report t and let the doc step decide.
    # Quick approx: p ≈ 2 * (1 - Φ(|t|))  good enough for n>=20.
    z = abs(t)
    p_approx = math.erfc(z / math.sqrt(2))
    return r, p_approx, n


def main():
    persona_dirs = sorted([p for p in G_DIR.iterdir()
                           if p.is_dir() and (p / "lora" / "adapter_model.safetensors").exists()])
    if not persona_dirs:
        print(f"No adapters found in {G_DIR}")
        return

    n_layers = 36  # Qwen3-4B
    P = len(persona_dirs)
    L = n_layers
    PJ = len(PROJECTIONS)
    proj_idx = {p: i for i, p in enumerate(PROJECTIONS)}
    tensor = torch.full((P, L, PJ), float("nan"))
    pids = []
    outcomes = []

    for pi, pdir in enumerate(persona_dirs):
        pid = pdir.name
        pids.append(pid)
        outcomes.append(load_exp29_outcome(pid))
        norms = frob_decompose(pdir / "lora" / "adapter_model.safetensors")
        for (layer, proj), val in norms.items():
            if proj in proj_idx and 0 <= layer < L:
                tensor[pi, layer, proj_idx[proj]] = val

    torch.save(tensor, G_DIR / "frobenius_tensor.pt")

    mean_lp = torch.nanmean(tensor, dim=0)
    std_lp = torch.zeros((L, PJ))
    for li in range(L):
        for pj in range(PJ):
            col = tensor[:, li, pj]
            col = col[~torch.isnan(col)]
            if len(col) > 1:
                std_lp[li, pj] = col.std().item()

    # Top-K cells
    flat = mean_lp.flatten()
    valid_mask = ~torch.isnan(flat)
    nz_idx = torch.where(valid_mask)[0]
    if len(nz_idx) == 0:
        print("ERROR: no valid cells")
        return
    k = min(10, int(valid_mask.sum()))
    topk = torch.topk(flat[valid_mask], k=k)
    top_cells = []
    for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
        gi = nz_idx[i].item()
        layer = gi // PJ
        proj = PROJECTIONS[gi % PJ]
        std_v = std_lp[layer, proj_idx[proj]].item()
        top_cells.append({
            "layer": layer, "proj": proj,
            "mean_frob": round(v, 4),
            "std_frob": round(std_v, 4),
            "cv": round(std_v / v, 3) if v > 0 else None,
        })

    # Per-projection means for attn-vs-FFN claim
    proj_mean = {}
    for proj, pj in proj_idx.items():
        col = mean_lp[:, pj]
        col = col[~torch.isnan(col)]
        proj_mean[proj] = round(float(col.mean()), 4) if len(col) else None

    # Layer-sum × outcome correlations
    layer_sums = torch.nansum(tensor, dim=2)  # (P, L)
    pres = torch.tensor([o["present_tpr"] if o["present_tpr"] is not None else float("nan")
                         for o in outcomes])
    absc = torch.tensor([o["absence_tpr"] if o["absence_tpr"] is not None else float("nan")
                         for o in outcomes])

    corr_present = []
    corr_absence = []
    for li in range(L):
        rp, pp, np_ = pearson_with_p(layer_sums[:, li], pres)
        ra, pa, na = pearson_with_p(layer_sums[:, li], absc)
        corr_present.append({"layer": li, "r": rp, "p": pp, "n": np_})
        corr_absence.append({"layer": li, "r": ra, "p": pa, "n": na})

    def topk_corr(corrs, key="r", descending=True, k=5):
        scored = [c for c in corrs if c[key] is not None]
        scored.sort(key=lambda c: c[key], reverse=descending)
        return scored[:k]

    top_pres_pos = topk_corr(corr_present, descending=True)
    top_pres_neg = topk_corr(corr_present, descending=False)
    top_abs_pos = topk_corr(corr_absence, descending=True)
    top_abs_neg = topk_corr(corr_absence, descending=False)

    # Decision-matrix-friendly summary numbers
    top1 = top_cells[0]
    ffn_down_mean = proj_mean.get("down_proj") or 0.0
    attn_max_mean = max(proj_mean.get(p) or 0 for p in ["q_proj", "k_proj", "v_proj", "o_proj"])
    ffn_max_mean = max(proj_mean.get(p) or 0 for p in ["gate_proj", "up_proj", "down_proj"])
    background = float(torch.nanmean(mean_lp).item())
    top1_over_background = top1["mean_frob"] / background if background > 0 else None

    summary = {
        "n_personae": P,
        "personae": pids,
        "top10_cells_by_mean_frob": top_cells,
        "proj_mean": proj_mean,
        "attn_vs_ffn": {
            "attn_max_mean": round(attn_max_mean, 4),
            "ffn_max_mean": round(ffn_max_mean, 4),
            "ratio_attn_over_ffn": round(attn_max_mean / ffn_max_mean, 3) if ffn_max_mean > 0 else None,
        },
        "top1_cell_over_background": round(top1_over_background, 3) if top1_over_background else None,
        "ffn_down_mean": ffn_down_mean,
        "background_mean_all_cells": round(background, 4),
        "phase_g_top3_recap": [
            {"layer": 35, "proj": "q_proj"},
            {"layer": 22, "proj": "q_proj"},
            {"layer": 30, "proj": "o_proj"},
        ],
        "phase_g_top3_present_at_n50": [
            any(c["layer"] == 35 and c["proj"] == "q_proj" for c in top_cells),
            any(c["layer"] == 22 and c["proj"] == "q_proj" for c in top_cells),
            any(c["layer"] == 30 and c["proj"] == "o_proj" for c in top_cells),
        ],
        "behavior_correlation": {
            "n_outcomes_available": int((~torch.isnan(pres)).sum()),
            "top5_layers_pos_corr_present_tpr": top_pres_pos,
            "top5_layers_neg_corr_present_tpr": top_pres_neg,
            "top5_layers_pos_corr_absence_tpr": top_abs_pos,
            "top5_layers_neg_corr_absence_tpr": top_abs_neg,
        },
    }
    json.dump(summary, open(G_DIR / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 10))
        im = ax.imshow(mean_lp.numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(PJ)); ax.set_xticklabels(PROJECTIONS, rotation=45)
        ax.set_yticks(range(0, L, 4)); ax.set_yticklabels(range(0, L, 4))
        ax.set_xlabel("projection"); ax.set_ylabel("layer")
        ax.set_title(f"mean ||ΔW||_F across {P} personae")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(G_DIR / "heatmap_layer_x_proj.png", dpi=110)
        print(f"saved heatmap → {G_DIR / 'heatmap_layer_x_proj.png'}")
    except Exception as e:
        print(f"heatmap skipped: {e}")


if __name__ == "__main__":
    main()
