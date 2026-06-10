#!/usr/bin/env python3
"""exp28 runner — Phase I: behavioral-memory γ-LoRA on WritingPrompts.

Differs from exp23/F_lamp3 (which use Q/A synthetic-extract pipeline):
behavioral memory is a plain language-modeling task. Per-user γ-LoRA is
trained as standard causal-LM on the user's train_stories (no synth Q/A
extract step). Eval is log-likelihood of gold continuation under each
of 4 configs:

  B_nohist : base Qwen3-4B, prompt only
  B_full   : base + full backstory (prompt + train stories) prepended
  C_rag    : base + top-K=3 BGE-retrieved train stories prepended
  C_lora   : base + per-user LoRA, no extra context

Saves per-user eval.json, plus rolled-up summary.jsonl + aggregate.json.

Output:
  runs/28_writingprompts/gamma_lora/<user_id>/eval.json
  runs/28_writingprompts/gamma_lora/<user_id>/lora/  (adapters, optional)
  runs/28_writingprompts/gamma_lora/summary.jsonl
  runs/28_writingprompts/gamma_lora/aggregate.json

Usage:
    .venv/bin/python experiments/28_runner.py
    .venv/bin/python experiments/28_runner.py --limit 2  # smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PERSONAE_PATH = ROOT / "runs" / "28_writingprompts" / "dataset.json"
from experiments import _base_model as _bm
OUT_DIR = _bm.run_dir(ROOT / "runs" / "28_writingprompts") / "gamma_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT_DIR / "summary.jsonl"

# Reuse exp19's load_base / train_lora are Q/A-shaped. We need a plain-LM
# variant. Import exp19 to get the model loader path.
sys.path.insert(0, str(ROOT / "experiments"))
spec19 = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py"
)
exp19 = importlib.util.module_from_spec(spec19)
spec19.loader.exec_module(exp19)


# ---------- training: plain causal LM on stories ----------

def collate_lm(batch_ids, pad_id):
    maxlen = max(len(x) for x in batch_ids)
    ids, lbl, attn = [], [], []
    for x in batch_ids:
        pad = maxlen - len(x)
        ids.append(x + [pad_id] * pad)
        lbl.append(x + [-100] * pad)
        attn.append([1] * len(x) + [0] * pad)
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(lbl, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long))


def train_lora_lm(qwen, tok, stories, *, epochs=3, lr=2e-4, r=64, alpha=128,
                  batch=2, max_chunk_tokens=1024, qid=""):
    from peft import get_peft_model, LoraConfig, TaskType
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type=TaskType.CAUSAL_LM)
    qwen.gradient_checkpointing_enable()
    qwen.enable_input_require_grads()
    model = get_peft_model(qwen, cfg)
    model.train()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Tokenize each story; chunk long ones to <= max_chunk_tokens
    chunks = []
    for st in stories:
        ids = tok(st, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_chunk_tokens:
            chunks.append(ids)
        else:
            for i in range(0, len(ids), max_chunk_tokens):
                seg = ids[i:i + max_chunk_tokens]
                if len(seg) >= 64:
                    chunks.append(seg)
    if not chunks:
        raise RuntimeError(f"[{qid}] no training chunks built from {len(stories)} stories")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_steps = epochs * ((len(chunks) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_steps))
    losses = []
    for ep in range(epochs):
        order = torch.randperm(len(chunks)).tolist()
        ep_loss, n = 0.0, 0
        for bi in range(0, len(order), batch):
            b = [chunks[i] for i in order[bi:bi + batch]]
            ids, lbl, attn = collate_lm(b, pad_id)
            ids, lbl, attn = ids.cuda(), lbl.cuda(), attn.cuda()
            out = model(input_ids=ids, attention_mask=attn, labels=lbl)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ep_loss += out.loss.item(); n += 1
        avg = ep_loss / max(1, n)
        losses.append(avg)
        print(f"    [{qid}] LM epoch {ep+1}/{epochs} n_chunks={len(chunks)} loss={avg:.4f}",
              flush=True)
    model.eval()
    return model, losses


# ---------- LL scoring ----------

@torch.no_grad()
def gold_logprob(model, tok, prefix_text: str, gold_text: str,
                 max_prefix_tokens: int = 6000) -> dict:
    """Compute sum log P(gold | prefix) under model.

    Returns {logprob_sum, n_tokens, mean_nat_per_tok, prefix_truncated}.
    """
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    truncated = False
    if len(prefix_ids) > max_prefix_tokens:
        prefix_ids = prefix_ids[-max_prefix_tokens:]
        truncated = True
    gold_ids = tok(gold_text, add_special_tokens=False)["input_ids"]
    if not gold_ids:
        return {"logprob_sum": 0.0, "n_tokens": 0, "mean_nat_per_tok": 0.0,
                "prefix_truncated": truncated}

    full = prefix_ids + gold_ids
    ids = torch.tensor([full], dtype=torch.long, device="cuda")
    out = model(input_ids=ids)
    logits = out.logits[0]  # [T, V]
    # P(token at position t) given context [:t-1] is logits[t-1]
    # We want logprob of gold tokens, which start at index len(prefix_ids)
    start = len(prefix_ids)
    logp = torch.log_softmax(logits[start - 1: start - 1 + len(gold_ids)].float(), dim=-1)
    gold_t = torch.tensor(gold_ids, device="cuda")
    chosen = logp.gather(-1, gold_t.unsqueeze(-1)).squeeze(-1)  # [n_gold]
    s = float(chosen.sum().item())
    return {"logprob_sum": s, "n_tokens": len(gold_ids),
            "mean_nat_per_tok": s / len(gold_ids), "prefix_truncated": truncated}


# ---------- BGE retrieval (lightweight) ----------

_bge_cache = {"model": None, "tok": None}


def load_bge():
    if _bge_cache["model"] is not None:
        return _bge_cache["model"], _bge_cache["tok"]
    from transformers import AutoModel, AutoTokenizer
    bge_path = "BAAI/bge-base-en-v1.5"
    # If not cached, fall back to remote (cron has internet)
    try:
        tok = AutoTokenizer.from_pretrained(bge_path)
        mdl = AutoModel.from_pretrained(bge_path).cuda().eval()
    except Exception:
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        mdl = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").cuda().eval()
    _bge_cache.update({"model": mdl, "tok": tok})
    return mdl, tok


@torch.no_grad()
def embed(texts, batch=8):
    mdl, tok = load_bge()
    out = []
    for i in range(0, len(texts), batch):
        b = texts[i:i + batch]
        enc = tok(b, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to("cuda")
        h = mdl(**enc).last_hidden_state[:, 0]
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        out.append(h.float().cpu())
    return torch.cat(out, dim=0)


def rag_topk(prefix: str, train_stories: list, k: int = 3) -> list:
    if not train_stories:
        return []
    q_emb = embed([prefix[-2000:]])
    s_emb = embed(train_stories)
    sims = (q_emb @ s_emb.T).squeeze(0)
    top = torch.topk(sims, min(k, len(train_stories))).indices.tolist()
    return [train_stories[i] for i in top]


# ---------- per-user pipeline ----------

def render_prefix(config: str, eval_prompt: dict, persona: dict, retrieved: list = None) -> str:
    prompt = persona.get("prompt", "")
    if config == "B_nohist":
        return f"Continue this story in a fitting style.\n\nWriting prompt: {prompt}\n\nStory so far:\n{eval_prompt['prefix']}"
    if config == "B_full":
        bs = persona["backstory"]
        return f"Continue this story in a fitting style.\n\n{bs}\n\nNow continue this new story:\n\nStory so far:\n{eval_prompt['prefix']}"
    if config == "C_rag":
        retr = retrieved or []
        ctx = "\n\n".join(f"--- Example story by this author ---\n{s}" for s in retr)
        return f"Continue this story in a fitting style.\n\nWriting prompt: {prompt}\n\n{ctx}\n\nNow continue this new story:\n\nStory so far:\n{eval_prompt['prefix']}"
    if config == "C_lora":
        return f"Continue this story in a fitting style.\n\nWriting prompt: {prompt}\n\nStory so far:\n{eval_prompt['prefix']}"
    raise ValueError(config)


def run_persona(persona, tok, base_qwen):
    """Train per-user LoRA + score 4 configs over eval prompts."""
    pid = persona["id"]
    user_dir = OUT_DIR / pid
    user_dir.mkdir(parents=True, exist_ok=True)
    eval_path = user_dir / "eval.json"
    if eval_path.exists():
        print(f"[28] {pid}: cached", flush=True)
        try:
            return json.loads(eval_path.read_text())
        except Exception:
            pass

    t0 = time.time()
    print(f"[28] {pid}: training γ-LoRA on {len(persona['train_stories'])} stories", flush=True)
    lora_model, losses = train_lora_lm(
        base_qwen, tok, persona["train_stories"],
        epochs=3, lr=2e-4, r=64, alpha=128, batch=2, qid=pid,
    )
    train_t = time.time() - t0

    # Build retrieved stories per eval prompt (RAG)
    retrieved_per_eval = [
        rag_topk(ep["prefix"], persona["train_stories"], k=3)
        for ep in persona["eval_prompts"]
    ]

    # Score base configs (B_nohist, B_full, C_rag) under base model.
    # We must use the *same underlying weights* for B_*/C_rag — that's
    # base_qwen; but base_qwen is currently wrapped in PEFT. Use the
    # underlying base via the PEFT disable_adapter context.
    records = []
    for ei, ep in enumerate(persona["eval_prompts"]):
        rec = {"persona_id": pid, "eval_idx": ei,
               "n_gold_chars": len(ep["gold_continuation"])}
        for cfg in ["B_nohist", "B_full", "C_rag"]:
            prefix = render_prefix(cfg, ep, persona,
                                   retrieved_per_eval[ei] if cfg == "C_rag" else None)
            with lora_model.disable_adapter():
                lp = gold_logprob(lora_model, tok, prefix, ep["gold_continuation"])
            rec[f"LL_{cfg}"] = lp
        # C_lora: with adapter active
        prefix = render_prefix("C_lora", ep, persona)
        lp = gold_logprob(lora_model, tok, prefix, ep["gold_continuation"])
        rec["LL_C_lora"] = lp
        records.append(rec)
        # Brief progress
        ll_l = rec["LL_C_lora"]["mean_nat_per_tok"]
        ll_n = rec["LL_B_nohist"]["mean_nat_per_tok"]
        ll_r = rec["LL_C_rag"]["mean_nat_per_tok"]
        print(f"[28] {pid} eval[{ei}] mean_nat/tok: nohist={ll_n:.3f} "
              f"rag={ll_r:.3f} lora={ll_l:.3f}", flush=True)

    out = {
        "persona_id": pid,
        "train_seconds": train_t,
        "train_losses": losses,
        "n_eval": len(records),
        "records": records,
    }
    eval_path.write_text(json.dumps(out))
    # Append per-record summary
    with SUMMARY.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # Free LoRA-specific memory
    del lora_model
    torch.cuda.empty_cache()
    print(f"[28] {pid} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def aggregate():
    if not SUMMARY.exists():
        print("[28] no summary.jsonl; skip aggregate", flush=True)
        return
    rows = [json.loads(l) for l in SUMMARY.read_text().splitlines() if l.strip()]
    if not rows:
        return
    configs = ["B_nohist", "B_full", "C_rag", "C_lora"]

    def mean_LL(cfg):
        vals = [r[f"LL_{cfg}"]["mean_nat_per_tok"] for r in rows]
        return sum(vals) / len(vals), vals

    means = {}
    for cfg in configs:
        m, _ = mean_LL(cfg)
        means[cfg] = m

    # Δ vs B_nohist
    deltas = {cfg: means[cfg] - means["B_nohist"] for cfg in configs}

    # Pre-registered primary test
    delta_lora_vs_rag = means["C_lora"] - means["C_rag"]

    # Per-record Δ stats (paired bootstrap-able)
    paired = {cfg: [r[f"LL_{cfg}"]["mean_nat_per_tok"]
                    - r["LL_B_nohist"]["mean_nat_per_tok"] for r in rows]
              for cfg in configs}
    paired_lora_minus_rag = [r["LL_C_lora"]["mean_nat_per_tok"]
                             - r["LL_C_rag"]["mean_nat_per_tok"] for r in rows]
    n = len(paired_lora_minus_rag)
    mean_l_r = sum(paired_lora_minus_rag) / n
    var = sum((x - mean_l_r) ** 2 for x in paired_lora_minus_rag) / max(1, n - 1)
    se = math.sqrt(var / n)
    ci_lo = mean_l_r - 1.96 * se
    ci_hi = mean_l_r + 1.96 * se

    agg = {
        "n_records": len(rows),
        "n_personae": len(set(r["persona_id"] for r in rows)),
        "mean_LL_per_token": means,
        "delta_vs_B_nohist": deltas,
        "primary_test_delta_lora_minus_rag_nat_per_tok": mean_l_r,
        "primary_test_se": se,
        "primary_test_ci95": [ci_lo, ci_hi],
        "primary_test_passes_falsifier": mean_l_r >= 0.5,
        "lora_meaningful_vs_nohist": deltas["C_lora"] >= 0.3,
        "sanity_full_helps": deltas["B_full"] > 0,
    }
    out_path = OUT_DIR / "aggregate.json"
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"[28] aggregate written: {agg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu", file=sys.stderr)
        sys.exit(2)

    p = json.loads(PERSONAE_PATH.read_text())
    held_out = [x for x in p["personae"] if x.get("split") == "held_out"]
    print(f"[28] loaded {len(held_out)} personae", flush=True)
    if args.start:
        held_out = held_out[args.start:]
    if args.limit:
        held_out = held_out[:args.limit]

    from transformers import AutoTokenizer
    from experiments import _base_model as _bm
    tok = AutoTokenizer.from_pretrained(_bm.active().path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_qwen = exp19.load_base()

    t0 = time.time()
    for i, persona in enumerate(held_out):
        try:
            run_persona(persona, tok, base_qwen)
        except Exception as e:
            print(f"[28] {persona['id']} failed: {e}", flush=True)
            import traceback; traceback.print_exc()
        # Note: train_lora_lm wraps base_qwen with PEFT in-place via
        # get_peft_model. To run a fresh persona we need to unwrap or
        # reload. PEFT 0.x stores adapters on the base module — easiest
        # is to reload base each persona at modest cost.
        elapsed = time.time() - t0
        print(f"[28] {i+1}/{len(held_out)} elapsed={elapsed:.0f}s "
              f"({elapsed/(i+1):.0f}s/persona)", flush=True)
        # Reload base to clear LoRA wrappers between personae
        del base_qwen
        torch.cuda.empty_cache()
        base_qwen = exp19.load_base()

    aggregate()
    print(f"[28] DONE total elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
