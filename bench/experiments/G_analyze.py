#!/usr/bin/env python3
"""
G_analyze.py — Frobenius decomposition of saved γ-LoRA adapters.

For each persona's saved adapter:
  - Load adapter_model.safetensors
  - For each (layer_idx, projection): compute ||B @ A||_F
  - Stack into (persona × layer × projection) tensor

Then produce summary.json with:
  - mean and std heatmap (layer × projection)
  - top-K cells by mean magnitude
  - correlation of per-layer ||ΔW||_F with each persona's exp29
    present-TPR and absence-TPR

And a heatmap PNG.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
G_DIR = ROOT / "runs" / "G_mechanism"
EXP29_DIR = ROOT / "runs" / "29_f_absence"

PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]


def parse_lora_key(key):
    """e.g. 'base_model.model.model.layers.17.self_attn.q_proj.lora_A.weight'
       → (17, 'q_proj', 'A')"""
    m = re.search(r"layers\.(\d+)\.[^.]+\.([a-z_]+_proj)\.lora_(A|B)\.", key)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def frob_decompose(adapter_path: Path):
    """Return dict[(layer, proj)] -> ||B@A||_F (float)."""
    sd = load_file(str(adapter_path))
    pairs = defaultdict(dict)  # (layer, proj) -> {'A': tensor, 'B': tensor}
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
            dW = B @ A  # standard PEFT convention
            out[(layer, proj)] = float(torch.linalg.norm(dW).item())
    return out


def load_exp29_outcome(pid):
    p = EXP29_DIR / pid / "aggregate.json"
    if not p.exists():
        return {"present_tpr": None, "absence_tpr": None}
    d = json.load(open(p))
    # exp29 stores per-config TPRs; we want C_lora's
    cl = d.get("C_lora", {})
    return {
        "present_tpr": cl.get("present_tpr"),
        "absence_tpr": cl.get("absence_tpr"),
    }


def main():
    # Discover saved adapters
    persona_dirs = sorted([p for p in G_DIR.iterdir()
                           if p.is_dir() and (p / "lora" / "adapter_model.safetensors").exists()])
    if not persona_dirs:
        print(f"No adapters found in {G_DIR}")
        return

    # Build (persona × layer × proj) tensor
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
        print(f"[{pid}] {len(norms)} (layer,proj) cells decoded")
        for (layer, proj), val in norms.items():
            if proj in proj_idx and 0 <= layer < L:
                tensor[pi, layer, proj_idx[proj]] = val

    torch.save(tensor, G_DIR / "frobenius_tensor.pt")

    # Per-cell mean/std
    valid = ~torch.isnan(tensor)
    mean_lp = torch.nanmean(tensor, dim=0)  # (L, PJ)
    # std with NaN-skip
    std_lp = torch.zeros((L, PJ))
    for li in range(L):
        for pj in range(PJ):
            col = tensor[:, li, pj]
            col = col[~torch.isnan(col)]
            if len(col) > 1:
                std_lp[li, pj] = col.std().item()

    # Top-3 cells by mean magnitude
    flat = mean_lp.flatten()
    topk = torch.topk(flat[~torch.isnan(flat)], k=min(3, int((~torch.isnan(flat)).sum())))
    nz_idx = torch.where(~torch.isnan(flat))[0]
    top_cells = []
    for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
        gi = nz_idx[i].item()
        layer = gi // PJ
        proj = PROJECTIONS[gi % PJ]
        top_cells.append({"layer": layer, "proj": proj, "mean_frob": v})

    # Correlation per-layer (sum across projections) with present_tpr & absence_tpr
    layer_sums = torch.nansum(tensor, dim=2)  # (P, L)
    pres = torch.tensor([o["present_tpr"] if o["present_tpr"] is not None else float("nan")
                         for o in outcomes])
    absc = torch.tensor([o["absence_tpr"] if o["absence_tpr"] is not None else float("nan")
                         for o in outcomes])

    def pearson(x, y):
        mask = ~(torch.isnan(x) | torch.isnan(y))
        if mask.sum() < 3:
            return None
        x = x[mask]; y = y[mask]
        x = x - x.mean(); y = y - y.mean()
        denom = (x.norm() * y.norm()).item()
        if denom == 0:
            return None
        return float((x * y).sum().item() / denom)

    corr_present = []
    corr_absence = []
    for li in range(L):
        cp = pearson(layer_sums[:, li], pres)
        ca = pearson(layer_sums[:, li], absc)
        corr_present.append(cp)
        corr_absence.append(ca)

    def topk_layers(corrs, descending=True, k=3):
        scored = [(li, c) for li, c in enumerate(corrs) if c is not None]
        scored.sort(key=lambda x: x[1], reverse=descending)
        return [{"layer": li, "corr": round(c, 3)} for li, c in scored[:k]]

    summary = {
        "n_personae": P,
        "personae": pids,
        "exp29_outcomes": outcomes,
        "top3_cells_by_mean_frob": top_cells,
        "top3_layers_pos_corr_present_tpr": topk_layers(corr_present, descending=True),
        "top3_layers_neg_corr_absence_tpr": topk_layers(corr_absence, descending=False),
        "mean_layer_x_proj_shape": list(mean_lp.shape),
    }
    json.dump(summary, open(G_DIR / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # Heatmap (matplotlib optional)
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
