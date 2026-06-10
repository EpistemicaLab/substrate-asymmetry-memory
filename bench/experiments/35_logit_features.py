#!/usr/bin/env python3
"""
35_logit_features.py — Phase H2 C-B: extract logit-grade features for the
calibration head.

For each persona × probe, run γ-LoRA and RAG forward, capturing:
  - top-1 token probability at the FIRST generation step
  - entropy at the first generation step
  - margin (top1 - top2 prob)
  - log-prob of the actual generated answer (sum over generated tokens)
  - mean per-token entropy across generation

These replace the noisy `conf_lora` / `conf_rag` text-derived proxies
from C-A. Adapter weights live at runs/30_mechanism/V3_P_*/lora/.

Output: runs/35_logit_features/features.jsonl (12 logit features +
keeps the C-A grounding features for ablation).

Caveats:
- This script is per-probe expensive (~3 sec per probe × 600 probes ×
  2 configs ≈ 1 GPU-hour). It's resume-safe via per-persona output
  files.
- We use deterministic decoding (do_sample=False) to match exp29.
"""

from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._base_model import run_dir as _run_dir  # noqa: E402

EXP29 = _run_dir(ROOT / "runs" / "29_f_absence")
EXP30 = _run_dir(ROOT / "runs" / "30_mechanism")  # adapter cache
DATASET29 = EXP29 / "dataset.json"
OUT = _run_dir(ROOT / "runs" / "35_logit_features")
OUT.mkdir(parents=True, exist_ok=True)
BGE_PATH = ROOT / "models" / "bge-large-en-v1.5"

# Reuse exp19 helpers
sys.path.insert(0, str(ROOT))  # so `from experiments import _base_model` works
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py"
)
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)


def chunk_text(text: str, size: int = 500, stride: int = 350) -> list[str]:
    bs = text.strip()
    if len(bs) <= size: return [bs]
    out = []; i = 0
    while i < len(bs):
        out.append(bs[i:i + size])
        if i + size >= len(bs): break
        i += stride
    return out


def render_rag_user(question: str, chunks: list[str]) -> str:
    blocks = "\n\n".join(f"[Memory {i+1}] {c.strip()}" for i, c in enumerate(chunks))
    return (f"{blocks}\n\n"
            f"Based ONLY on the memory snippets above, answer the user's question. "
            f"If the snippets do not actually mention the topic, say 'No, we have not "
            f"discussed that.'\n\nUser: {question}")


PLAIN_SYS = "You are the user's helpful assistant."


@torch.no_grad()
def gen_with_logits(tok, mdl, system: str, user: str, max_new: int = 80):
    """Generate, returning text + first-step + sequence-level confidence."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    cfg = exp19._bm.active()
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   **cfg.chat_kwargs)
    enc = tok(text, return_tensors="pt").to("cuda")
    out = mdl.generate(
        **enc, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True, output_scores=True,
    )
    seq = out.sequences[0]
    new_tokens = seq[enc["input_ids"].shape[1]:]
    pred = exp19.strip_think(tok.decode(new_tokens, skip_special_tokens=True))

    # First-step features (most diagnostic for "do I know this?").
    if not out.scores:
        return pred, {"top1_prob": 0.0, "entropy": 0.0, "margin": 0.0,
                      "logprob_sum": 0.0, "mean_entropy": 0.0,
                      "n_gen_tokens": 0}
    first = F.softmax(out.scores[0][0].float(), dim=-1)
    top2 = torch.topk(first, k=2)
    top1_prob = float(top2.values[0])
    top2_prob = float(top2.values[1])
    margin = top1_prob - top2_prob
    entropy = float(-(first * (first.clamp_min(1e-12)).log()).sum())

    # Sequence-level features.
    logprob_sum = 0.0
    entropies = []
    for i, score in enumerate(out.scores):
        probs = F.softmax(score[0].float(), dim=-1)
        tok_id = new_tokens[i].item() if i < len(new_tokens) else None
        if tok_id is not None and tok_id < probs.shape[-1]:
            logprob_sum += float(probs[tok_id].clamp_min(1e-12).log())
        entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    return pred, {
        "top1_prob": top1_prob, "entropy": entropy, "margin": margin,
        "logprob_sum": logprob_sum, "mean_entropy": mean_entropy,
        "n_gen_tokens": len(new_tokens),
    }


def load_lora_for(pid: str, base):
    """Attach the cached γ-LoRA adapter for this persona."""
    adapter_dir = EXP30 / pid / "lora"
    if not adapter_dir.exists():
        return None
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(adapter_dir),
                                      is_trainable=False).cuda()
    model.eval()
    return model


def process_persona(pid: str, persona, tok, bge_tok, bge_mdl, base):
    out_path = OUT / pid / "features.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = sum(1 for _ in out_path.read_text().splitlines() if _.strip())
        if existing >= len(persona["probes"]):
            print(f"[{pid}] cached ({existing} rows), skip", flush=True)
            return

    probes = persona["probes"]
    # Pre-compute RAG chunks + retrieval relevance.
    chunks = chunk_text(persona["backstory"])

    @torch.no_grad()
    def bge(texts):
        if not texts: return torch.zeros(0, 1024)
        enc = bge_tok(texts, padding=True, truncation=True, max_length=512,
                     return_tensors="pt").to("cuda")
        h = bge_mdl(**enc).last_hidden_state[:, 0]
        return F.normalize(h, dim=-1).cpu()

    c_emb = bge(chunks)
    q_emb = bge([p["question"] for p in probes])
    sims_q = (q_emb @ c_emb.T).numpy()
    top_idx = np.argsort(-sims_q, axis=1)[:, :3]
    topk_relevance = np.sort(sims_q, axis=1)[:, -3:].mean(axis=1)

    # Adapter for this persona.
    lora = load_lora_for(pid, base)
    if lora is None:
        print(f"[{pid}] WARN no adapter, skip", flush=True)
        return

    # Iterate probes: γ-LoRA forward (no RAG context), then base RAG forward.
    f = out_path.open("w")
    try:
        for i, pr in enumerate(probes):
            top_chunks = [chunks[j] for j in top_idx[i]]
            # γ-LoRA: question only, plain system.
            pred_l, feat_l = gen_with_logits(tok, lora, PLAIN_SYS, pr["question"])
            f.flush()
            row = {
                "persona_id": pid, "kind": pr["kind"], "topic": pr["topic"],
                "question": pr["question"], "gold": pr["gold"],
                "lora_pred": pred_l, "lora_logit_feat": feat_l,
                "r": float(topk_relevance[i]),
            }
            # RAG: base (no LoRA), question with retrieved chunks.
            # We unwrap adapter by deactivating it for this forward pass.
            with lora.disable_adapter():
                user_rag = render_rag_user(pr["question"], top_chunks)
                pred_r, feat_r = gen_with_logits(tok, lora, PLAIN_SYS, user_rag)
            row["rag_pred"] = pred_r
            row["rag_logit_feat"] = feat_r
            f.write(json.dumps(row) + "\n"); f.flush()
            if i == 0:
                print(f"[{pid}] probe[0] lora_top1={feat_l['top1_prob']:.3f} "
                      f"rag_top1={feat_r['top1_prob']:.3f}", flush=True)
    finally:
        f.close()
    del lora
    gc.collect(); torch.cuda.empty_cache()
    print(f"[{pid}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=50)
    args = ap.parse_args()

    ds = json.loads(DATASET29.read_text())
    personae = ds["personae"][args.start:args.end]
    print(f"[main] processing {len(personae)} personae [{args.start}:{args.end}]",
          flush=True)

    from transformers import AutoTokenizer, AutoModel
    cfg = exp19._bm.active()
    tok = AutoTokenizer.from_pretrained(cfg.path)

    bge_tok = AutoTokenizer.from_pretrained(str(BGE_PATH))
    bge_mdl = AutoModel.from_pretrained(str(BGE_PATH),
                                        dtype=torch.float16).cuda().eval()

    for p in personae:
        try:
            # Reload base per persona so previous adapter state is gone.
            base = exp19.load_base()
            process_persona(p["id"], p, tok, bge_tok, bge_mdl, base)
            del base
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"[{p['id']}] FAIL: {e}", flush=True)
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
