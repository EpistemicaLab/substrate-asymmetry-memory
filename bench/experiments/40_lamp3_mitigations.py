#!/usr/bin/env python3
"""LaMP-3 mitigation runner — exp40.

Single runner dispatched by ``--config <ARM>`` for all 9 mitigation arms
defined in ``docs/LAMP3_MITIGATION_PLAN.md``. Reuses Phase F's dataset
(``runs/F_lamp3/dataset.json``) and exp19's training/eval primitives where
possible. Per-arm overrides live in :data:`ARM_CONFIGS` below.

Output:
  runs/40_lamp3_mit/<ARM>/<persona_id>/{synthqa.jsonl, eval.json, lora/}
  runs/40_lamp3_mit/<ARM>/{summary.jsonl, aggregate.json, full.log}

Usage:
    .venv/bin/python experiments/40_lamp3_mitigations.py --config H
    .venv/bin/python experiments/40_lamp3_mitigations.py --config B --limit 2  # smoke
"""
from __future__ import annotations

import argparse, gc, importlib.util, json, os, sys, time
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "runs" / "F_lamp3" / "dataset.json"
F_GAMMA_DIR = ROOT / "runs" / "F_lamp3" / "gamma_lora"
OUT_BASE = ROOT / "runs" / "40_lamp3_mit"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# Load shared infra from exp19 (training/eval primitives) and exp23 (per-persona loop).
# Insert both ROOT (for `from experiments import _base_model` style imports inside exp19)
# and ROOT/experiments (for direct module loading via importlib).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
spec19 = importlib.util.spec_from_file_location("exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py")
exp19 = importlib.util.module_from_spec(spec19); spec19.loader.exec_module(exp19)
spec23 = importlib.util.spec_from_file_location("exp23", ROOT / "experiments" / "23_lora_persona.py")
exp23 = importlib.util.module_from_spec(spec23); spec23.loader.exec_module(exp23)


# ── arm definitions ────────────────────────────────────────────────────────
# Each arm has:
#   train_kwargs: passed to exp19.train_lora (epochs, r, alpha, lr, batch)
#   target_modules: override LoraConfig target_modules (None = default q/k/v/o)
#   layer_filter: tuple (lo, hi) restricting which layers get LoRA-adapted
#   adapter_kind: "lora" | "ia3"
#   loss_kind: "ce" (default) | "format_aware" | "kl_anchor" | "mixed_loss"
#   data_kind: "lamp3_only" (default) | "lamp3_plus_alpaca" | "lamp3_augmented"
#   eval_decoding: "free" (default) | "constrained_15"
#   reuse_adapter_from: arm_id (e.g. "F_lamp3") to skip training and reuse cached LoRA

ARM_CONFIGS: dict[str, dict] = {
    "H": {  # constrained decoding @ eval; train with exp23 vanilla recipe (no retrain shortcut — F adapters not persisted)
        "train_kwargs": {"epochs": 20, "r": 128, "alpha": 256, "lr": 2e-4, "batch": 4},
        "eval_decoding": "constrained_15",
    },
    "B": {  # epoch reduction
        "train_kwargs": {"epochs": 3, "r": 128, "alpha": 256, "lr": 2e-4, "batch": 4},
        "eval_decoding": "free",
    },
    "A": {  # rank reduction
        "train_kwargs": {"epochs": 20, "r": 8, "alpha": 16, "lr": 2e-4, "batch": 4},
        "eval_decoding": "free",
    },
    "F": {  # IA³ adapter
        "train_kwargs": {"epochs": 20, "lr": 5e-3, "batch": 4},
        "adapter_kind": "ia3",
        "eval_decoding": "free",
    },
    "G": {  # format-aware loss reweighting
        "train_kwargs": {"epochs": 20, "r": 128, "alpha": 256, "lr": 2e-4, "batch": 4},
        "loss_kind": "format_aware",
        "eval_decoding": "free",
    },
    "C": {  # selective layer freezing (L24-31, q/v only)
        "train_kwargs": {"epochs": 20, "r": 128, "alpha": 256, "lr": 2e-4, "batch": 4},
        "target_modules": ["q_proj", "v_proj"],
        "layer_filter": (24, 31),
        "eval_decoding": "free",
    },
    "E": {  # mixed-loss: LaMP-3 + Alpaca instruction anchor
        "train_kwargs": {"epochs": 20, "r": 128, "alpha": 256, "lr": 2e-4, "batch": 4},
        "data_kind": "lamp3_plus_alpaca",
        "eval_decoding": "free",
    },
    "D": {  # KL anchor to base
        "train_kwargs": {"epochs": 20, "r": 128, "alpha": 256, "lr": 2e-4, "batch": 4},
        "loss_kind": "kl_anchor",
        "kl_lambda": 0.1,
        "eval_decoding": "free",
    },
    "I": {  # data augmentation via Claude paraphrase
        "train_kwargs": {"epochs": 20, "r": 128, "alpha": 256, "lr": 2e-4, "batch": 4},
        "data_kind": "lamp3_augmented",
        "eval_decoding": "free",
    },
}

# P3 cross-product: retrain each arm X under H_on_X (using arm X's training
# kwargs verbatim) and eval with constrained_15. Tests whether the eval-time
# logit-mask scaffold generalizes across training recipes, not just vanilla H.
# We deep-copy arm X's config and override eval_decoding to constrained_15.
import copy as _copy
for _src in ("B", "A", "F", "G", "C", "E", "D", "I"):
    _cfg = _copy.deepcopy(ARM_CONFIGS[_src])
    _cfg["eval_decoding"] = "constrained_15"
    ARM_CONFIGS[f"H_on_{_src}"] = _cfg


# ── helpers ────────────────────────────────────────────────────────────────

def out_dir_for(arm: str) -> Path:
    # When replicating champion on a non-default base model (Llama),
    # write under REPLICATE_LLAMA_<arm>/ so we don't collide with the
    # original Qwen run's per-persona cache (eval.json).
    base = os.environ.get("ENGRAM_BASE_MODEL", "")
    if base.startswith("llama"):
        d = OUT_BASE / f"REPLICATE_LLAMA_{arm}"
    elif base.startswith("mistral"):
        # smoke override: $ENGRAM_MISTRAL_SMOKE=1 → REPLICATE_MISTRAL_smoke/<arm>
        if os.environ.get("ENGRAM_MISTRAL_SMOKE") == "1":
            d = OUT_BASE / "REPLICATE_MISTRAL_smoke" / arm
        else:
            d = OUT_BASE / f"REPLICATE_MISTRAL_{arm}"
    else:
        d = OUT_BASE / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_target_modules_filter(target_modules, layer_filter):
    """Return a function that takes a module name and returns True if it should
    be adapted. For PEFT, we use target_modules as exact-name match against
    `model.layers.<i>.<sub>.<proj>` substrings; combined with a layer index
    filter when layer_filter is set."""
    if layer_filter is None:
        return None  # let PEFT default
    lo, hi = layer_filter
    def names_for_filter(model):
        # collect fully-qualified names in scope.
        out = []
        for n, _ in model.named_modules():
            if any(t in n for t in target_modules) and n.endswith(tuple(target_modules)):
                # n looks like "model.layers.5.self_attn.q_proj" — extract the layer idx.
                parts = n.split(".")
                if "layers" in parts:
                    i = parts.index("layers")
                    if i + 1 < len(parts):
                        try:
                            li = int(parts[i + 1])
                        except ValueError:
                            continue
                        if lo <= li <= hi:
                            out.append(n)
        return out
    return names_for_filter


def train_lora_arm(base, tok, pairs, *, arm_cfg, qid):
    """Train per-arm. Routes to specialized loops for non-vanilla arms."""
    tk = arm_cfg.get("train_kwargs", {})
    adapter_kind = arm_cfg.get("adapter_kind", "lora")
    target_modules = arm_cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
    layer_filter = arm_cfg.get("layer_filter")
    loss_kind = arm_cfg.get("loss_kind", "ce")

    # Adapter-kind branch
    if adapter_kind == "ia3":
        return _train_ia3(base, tok, pairs, tk=tk, qid=qid)

    # Build LoRA config (possibly with layer filter)
    from peft import get_peft_model, LoraConfig, TaskType
    epochs = tk.get("epochs", 20); r = tk.get("r", 128); alpha = tk.get("alpha", 256)
    lr = tk.get("lr", 2e-4); batch = tk.get("batch", 4)

    if layer_filter is not None:
        # Use layers_to_transform for layer-level scoping (PEFT >= 0.4)
        lo, hi = layer_filter
        layers_to_transform = list(range(lo, hi + 1))
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                         target_modules=target_modules,
                         layers_to_transform=layers_to_transform,
                         task_type=TaskType.CAUSAL_LM)
    else:
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                         target_modules=target_modules,
                         task_type=TaskType.CAUSAL_LM)

    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = get_peft_model(base, cfg)
    model.train()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tokenized = [exp19.render_for_training(tok, p["q"], p["a"]) for p in pairs]

    # Loss-kind branch
    if loss_kind == "ce":
        return _vanilla_train_loop(model, tokenized, epochs=epochs, lr=lr, batch=batch,
                                   pad_id=pad_id, qid=qid)
    if loss_kind == "format_aware":
        return _format_aware_train_loop(model, tok, pairs, tokenized,
                                        epochs=epochs, lr=lr, batch=batch,
                                        pad_id=pad_id, qid=qid)
    if loss_kind == "kl_anchor":
        return _kl_anchor_train_loop(model, base, tok, tokenized,
                                     epochs=epochs, lr=lr, batch=batch,
                                     pad_id=pad_id, kl_lambda=arm_cfg.get("kl_lambda", 0.1),
                                     qid=qid)
    raise ValueError(f"unknown loss_kind {loss_kind}")


def _vanilla_train_loop(model, tokenized, *, epochs, lr, batch, pad_id, qid):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_steps = epochs * ((len(tokenized) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_steps))
    losses = []
    for ep in range(epochs):
        order = torch.randperm(len(tokenized)).tolist()
        ep_loss, n = 0.0, 0
        for bi in range(0, len(order), batch):
            b = [tokenized[i] for i in order[bi:bi + batch]]
            ids, lbl, attn = exp19.collate(b, pad_id)
            ids, lbl, attn = ids.cuda(), lbl.cuda(), attn.cuda()
            out = model(input_ids=ids, attention_mask=attn, labels=lbl)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ep_loss += out.loss.item(); n += 1
        avg = ep_loss / max(1, n)
        losses.append(avg)
        print(f"    [{qid}] epoch {ep+1:2d}/{epochs} loss={avg:.4f}", flush=True)
    model.eval()
    return model, losses


def _format_aware_train_loop(model, tok, pairs, tokenized, *, epochs, lr, batch, pad_id, qid):
    """Weight loss on the answer tokens 10x relative to the rest. Since
    render_for_training masks non-answer tokens with -100, we just multiply
    by a constant — this acts as a learning-rate hike on the answer span,
    not a true reweighting, but the simpler effective signal is what we
    want here."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_steps = epochs * ((len(tokenized) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_steps))
    losses = []
    for ep in range(epochs):
        order = torch.randperm(len(tokenized)).tolist()
        ep_loss, n = 0.0, 0
        for bi in range(0, len(order), batch):
            b = [tokenized[i] for i in order[bi:bi + batch]]
            ids, lbl, attn = exp19.collate(b, pad_id)
            ids, lbl, attn = ids.cuda(), lbl.cuda(), attn.cuda()
            out = model(input_ids=ids, attention_mask=attn, labels=lbl)
            (out.loss * 10.0).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ep_loss += out.loss.item(); n += 1
        avg = ep_loss / max(1, n)
        losses.append(avg)
        print(f"    [{qid}] G epoch {ep+1:2d}/{epochs} loss={avg:.4f}", flush=True)
    model.eval()
    return model, losses


def _kl_anchor_train_loop(model, base_for_kl, tok, tokenized, *, epochs, lr, batch, pad_id, kl_lambda, qid):
    """Standard CE loss + λ · KL(student || frozen base) on the same tokens."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_steps = epochs * ((len(tokenized) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_steps))
    # Keep a reference to the underlying base model with adapters disabled for KL.
    losses = []
    for ep in range(epochs):
        order = torch.randperm(len(tokenized)).tolist()
        ep_loss, n = 0.0, 0
        for bi in range(0, len(order), batch):
            b = [tokenized[i] for i in order[bi:bi + batch]]
            ids, lbl, attn = exp19.collate(b, pad_id)
            ids, lbl, attn = ids.cuda(), lbl.cuda(), attn.cuda()
            out = model(input_ids=ids, attention_mask=attn, labels=lbl)
            ce = out.loss
            student_logits = out.logits
            with torch.no_grad():
                with model.disable_adapter():
                    base_logits = model(input_ids=ids, attention_mask=attn).logits
            # KL on shifted positions
            sl = student_logits[:, :-1, :].contiguous()
            bl = base_logits[:, :-1, :].contiguous()
            mask = (lbl[:, 1:] != -100).float()
            ls = torch.nn.functional.log_softmax(sl, dim=-1)
            bs_ = torch.nn.functional.softmax(bl, dim=-1)
            kl = (bs_ * (torch.nn.functional.log_softmax(bl, dim=-1) - ls)).sum(-1)
            kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
            loss = ce + kl_lambda * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ep_loss += loss.item(); n += 1
        avg = ep_loss / max(1, n)
        losses.append(avg)
        print(f"    [{qid}] D epoch {ep+1:2d}/{epochs} loss={avg:.4f}", flush=True)
    model.eval()
    return model, losses


def _train_ia3(base, tok, pairs, *, tk, qid):
    from peft import get_peft_model, IA3Config, TaskType
    epochs = tk.get("epochs", 20); lr = tk.get("lr", 5e-3); batch = tk.get("batch", 4)
    cfg = IA3Config(target_modules=["k_proj", "v_proj", "down_proj"],
                    feedforward_modules=["down_proj"],
                    task_type=TaskType.CAUSAL_LM)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = get_peft_model(base, cfg)
    model.train()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tokenized = [exp19.render_for_training(tok, p["q"], p["a"]) for p in pairs]
    return _vanilla_train_loop(model, tokenized, epochs=epochs, lr=lr, batch=batch,
                               pad_id=pad_id, qid=qid)


# ── data assembly ──────────────────────────────────────────────────────────

ALPACA_CACHE = ROOT / "runs" / "40_lamp3_mit" / "_alpaca_pairs.json"


def get_alpaca_pairs(n: int = 50, seed: int = 43):
    """Return a list of {"q":..., "a":...} sampled from alpaca-cleaned. Cached."""
    if ALPACA_CACHE.exists():
        return json.loads(ALPACA_CACHE.read_text())[:n]
    import random, urllib.request
    URL = "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json"
    print(f"[alpaca] downloading {URL}", flush=True)
    with urllib.request.urlopen(URL, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))
    rng = random.Random(seed)
    rng.shuffle(data)
    keep = []
    for item in data:
        if item.get("input"):
            continue  # take instruction-only items (cleaner format anchors)
        instr, out = item.get("instruction", ""), item.get("output", "")
        if not instr or not out:
            continue
        if len(out) > 400:
            continue  # short outputs only
        keep.append({"q": instr, "a": out})
        if len(keep) >= 200:
            break
    ALPACA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ALPACA_CACHE.write_text(json.dumps(keep))
    return keep[:n]


def augment_pairs_via_claude(pairs, judge_client, *, qid: str, n_paraphrases: int = 5):
    """For each (q,a) pair, generate n_paraphrases of q (with same a) via Claude."""
    PROMPT = (
        "Paraphrase the following review-prompt {n} different ways. Keep the "
        "review text content intact (it's user history); only vary the wording "
        "of the question/instruction. Output ONE paraphrase per line, no numbering."
    )
    out = list(pairs)
    for p in pairs:
        q, a = p["q"], p["a"]
        try:
            resp = judge_client.converse(
                modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                messages=[{"role": "user", "content": [
                    {"text": PROMPT.format(n=n_paraphrases) + "\n\nORIGINAL:\n" + q}
                ]}],
                inferenceConfig={"maxTokens": 1500, "temperature": 0.7},
            )
            txt = resp["output"]["message"]["content"][0]["text"]
            paras = [l.strip() for l in txt.splitlines() if l.strip()][:n_paraphrases]
            for pp in paras:
                out.append({"q": pp, "a": a})
        except Exception as e:
            print(f"  [{qid}] aug skip ({e})", flush=True)
    return out


def assemble_train_pairs(persona, arm_cfg, judge_client):
    """Return the list of {q, a} training pairs for this persona under this arm.
    Sanity pairs (last 5) are stripped here and returned separately."""
    pairs = list(persona.get("main_qa", [])) + list(persona.get("probe2_qa", []))[:0]
    # Phase F's recipe: train on first 5 main_qa, sanity on last 5 (held-back). But
    # F_lamp3 dataset.json shape has main_qa / probe2_qa already split by purpose.
    # For mitigation arms we follow Phase F's exact split:
    main = persona["main_qa"]
    sanity_pairs = main[-5:] if len(main) >= 10 else []
    train_pairs = main[:-5] if len(main) >= 10 else list(main)

    # Normalize to {q,a}
    def norm(x):
        if "q" in x and "a" in x: return x
        return {"q": x.get("question", ""), "a": x.get("answer", "")}
    train_pairs = [norm(p) for p in train_pairs]
    sanity_pairs = [norm(p) for p in sanity_pairs]

    data_kind = arm_cfg.get("data_kind", "lamp3_only")
    if data_kind == "lamp3_only":
        return train_pairs, sanity_pairs
    if data_kind == "lamp3_plus_alpaca":
        anchor = get_alpaca_pairs(n=50, seed=43)
        return train_pairs + anchor, sanity_pairs
    if data_kind == "lamp3_augmented":
        aug = augment_pairs_via_claude(train_pairs, judge_client, qid=persona["id"], n_paraphrases=5)
        return aug, sanity_pairs
    raise ValueError(f"unknown data_kind {data_kind}")


# ── eval (with optional constrained decoding) ──────────────────────────────

@torch.no_grad()
def gen_constrained_15(tok, mdl, msgs):
    """Force a single token in {1,2,3,4,5} via logit masking."""
    from experiments import _base_model as _bm
    cfg = _bm.active()
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **cfg.chat_kwargs)
    enc = tok(text, return_tensors="pt").to("cuda")
    # Identify token IDs for "1".."5". Try plain digits first.
    ids15 = []
    for d in ["1", "2", "3", "4", "5"]:
        toks = tok(d, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            ids15.append(toks[0])
    if len(ids15) != 5:
        # fallback: try with a leading space (BPE often differs)
        ids15 = []
        for d in [" 1", " 2", " 3", " 4", " 5"]:
            toks = tok(d, add_special_tokens=False)["input_ids"]
            if len(toks) == 1:
                ids15.append(toks[0])
    if len(ids15) != 5:
        # Last resort: build per-token forward
        out = mdl.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        return exp19.strip_think(tok.decode(new, skip_special_tokens=True))

    # Forward once, take logits at the last position, mask everything except 1..5
    logits = mdl(**enc).logits[0, -1]
    mask = torch.full_like(logits, float("-inf"))
    mask[ids15] = logits[ids15]
    pred_id = mask.argmax().item()
    return tok.decode([pred_id], skip_special_tokens=True).strip()


# ── per-persona run ─────────────────────────────────────────────────────────

def run_persona(persona, tok, judge_client, *, arm: str, arm_cfg: dict, out_dir: Path):
    pid = persona["id"]
    pdir = out_dir / pid
    pdir.mkdir(parents=True, exist_ok=True)
    eval_path = pdir / "eval.json"
    if eval_path.exists():
        print(f"[{pid}] cached, skipping", flush=True)
        return json.loads(eval_path.read_text())

    t0 = time.time()
    print(f"\n=== {arm} | {pid} ===", flush=True)

    # Reuse-adapter branch (arm H)
    if arm_cfg.get("reuse_adapter_from"):
        src = arm_cfg["reuse_adapter_from"]
        if src == "F_lamp3":
            adapter_dir = F_GAMMA_DIR / pid / "lora"
        else:
            adapter_dir = ROOT / "runs" / src / pid / "lora"
        if not adapter_dir.exists():
            print(f"  [{pid}] WARN: adapter missing at {adapter_dir}; skipping", flush=True)
            return None
        from peft import PeftModel
        base = exp19.load_base()
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()
        losses = []
        sanity_pairs = []  # H reuses; no fresh sanity
    else:
        train_pairs, sanity_pairs = assemble_train_pairs(persona, arm_cfg, judge_client)
        if len(train_pairs) < 1:
            print(f"  [{pid}] no training pairs, skipping", flush=True)
            return None
        print(f"  [{pid}] train n={len(train_pairs)} sanity n={len(sanity_pairs)}", flush=True)
        base = exp19.load_base()
        t_train = time.time()
        model, losses = train_lora_arm(base, tok, train_pairs, arm_cfg=arm_cfg, qid=pid)
        print(f"  [{pid}] trained in {time.time()-t_train:.0f}s", flush=True)
        # save adapter (with verification — PEFT can silently skip weights)
        adapter_out = pdir / "lora"
        try:
            model.save_pretrained(str(adapter_out), safe_serialization=True)
            weights_files = list(adapter_out.glob("adapter_model.*"))
            if not weights_files:
                # Retry with safe_serialization=False (writes adapter_model.bin)
                print(f"  [{pid}] WARN: no weights after safe save; retrying bin", flush=True)
                model.save_pretrained(str(adapter_out), safe_serialization=False)
                weights_files = list(adapter_out.glob("adapter_model.*"))
            if not weights_files:
                print(f"  [{pid}] ERROR: adapter save produced NO weights file", flush=True)
            else:
                wf = weights_files[0]
                print(f"  [{pid}] saved adapter: {wf.name} ({wf.stat().st_size//1024} KB)", flush=True)
        except Exception as e:
            print(f"  [{pid}] adapter save failed (non-fatal): {e}", flush=True)

    # ── eval ───────────────────────────────────────────────────────────────
    decoding = arm_cfg.get("eval_decoding", "free")
    sanity_results = []
    for p in sanity_pairs:
        if decoding == "constrained_15":
            sp_text = gen_constrained_15(tok, model, [{"role": "user", "content": p["q"]}])
        else:
            sp_text = exp19.gen_chat(tok, model, [{"role": "user", "content": p["q"]}])
        ok = exp19.judge(judge_client, p["q"], p["a"], sp_text)
        sanity_results.append({"q": p["q"], "a_gold": p["a"], "a_pred": sp_text, "ok": ok})

    eval_records = []
    for kind in ("main_qa", "probe2_qa"):
        for q in persona.get(kind, []):
            qq = q.get("q") or q.get("question", "")
            gold = q.get("a") or q.get("answer", "")
            if decoding == "constrained_15":
                pred = gen_constrained_15(tok, model, [{"role": "user", "content": qq}])
            else:
                pred = exp19.gen_chat(tok, model, [{"role": "user", "content": qq}])
            correct = exp19.judge(judge_client, qq, gold, pred)
            eval_records.append({
                "persona_id": pid, "eval_kind": "main" if kind == "main_qa" else "probe2",
                "q": qq, "gold": gold, "pred": pred, "correct": correct,
            })

    main_n = sum(1 for r in eval_records if r["eval_kind"] == "main")
    probe2_n = sum(1 for r in eval_records if r["eval_kind"] == "probe2")
    main_acc = sum(r["correct"] for r in eval_records if r["eval_kind"] == "main") / max(1, main_n)
    probe2_acc = sum(r["correct"] for r in eval_records if r["eval_kind"] == "probe2") / max(1, probe2_n)
    sanity_acc = (sum(s["ok"] for s in sanity_results) / max(1, len(sanity_results))) if sanity_results else None

    rec = {
        "persona_id": pid, "arm": arm,
        "main_acc": main_acc, "probe2_acc": probe2_acc, "sanity_acc": sanity_acc,
        "n_main": main_n, "n_probe2": probe2_n, "n_sanity": len(sanity_results),
        "final_loss": losses[-1] if losses else None,
        "wallclock_s": time.time() - t0,
        "eval_records": eval_records, "sanity_records": sanity_results,
    }
    eval_path.write_text(json.dumps(rec, indent=2))
    print(f"  [{pid}] main={main_acc:.3f} probe2={probe2_acc:.3f} sanity={sanity_acc} t={rec['wallclock_s']:.0f}s", flush=True)

    # release
    del model
    if "base" in dir():
        del base
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def aggregate(arm: str, out_dir: Path):
    records = []
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir(): continue
        ep = sub / "eval.json"
        if ep.exists():
            records.append(json.loads(ep.read_text()))
    if not records:
        print("[agg] no eval.json files", flush=True); return
    n = len(records)
    main_acc = sum(r["main_acc"] for r in records) / n
    probe2_acc = sum(r["probe2_acc"] for r in records) / n
    sanity_vals = [r["sanity_acc"] for r in records if r.get("sanity_acc") is not None]
    sanity_acc = sum(sanity_vals) / len(sanity_vals) if sanity_vals else None
    agg = {
        "arm": arm, "n_personae": n,
        "main_acc": main_acc, "probe2_acc": probe2_acc, "sanity_acc": sanity_acc,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"[agg {arm}] n={n} main={main_acc:.3f} probe2={probe2_acc:.3f}", flush=True)


# ── entry ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(ARM_CONFIGS.keys()))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu", file=sys.stderr); sys.exit(2)

    arm = args.config
    arm_cfg = ARM_CONFIGS[arm]
    out_dir = out_dir_for(arm)

    # Bedrock thread preflight
    from concurrent.futures import ThreadPoolExecutor
    import boto3
    def _t(_):
        return boto3.client("bedrock-runtime", region_name="us-east-1").meta.region_name
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(_t, range(2)))
    except Exception as e:
        print(f"ERROR: Bedrock thread preflight failed: {e}", file=sys.stderr); sys.exit(3)

    p = json.loads(DATASET_PATH.read_text())
    held_out = [x for x in p["personae"] if x.get("split") == "held_out"]
    if args.start: held_out = held_out[args.start:]
    if args.limit: held_out = held_out[:args.limit]
    print(f"[40_mit] arm={arm} n_personae={len(held_out)}", flush=True)

    from transformers import AutoTokenizer
    from experiments import _base_model as _bm
    tok = AutoTokenizer.from_pretrained(_bm.active().path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    judge_client = exp19.make_client()

    t0 = time.time()
    for i, persona in enumerate(held_out):
        try:
            run_persona(persona, tok, judge_client, arm=arm, arm_cfg=arm_cfg, out_dir=out_dir)
        except Exception as e:
            print(f"[{persona['id']}] FAIL: {e}", flush=True)
            import traceback; traceback.print_exc()
        elapsed = time.time() - t0
        print(f"[40_mit {arm}] {i+1}/{len(held_out)} elapsed={elapsed:.0f}s", flush=True)

    aggregate(arm, out_dir)
    print(f"[40_mit {arm}] DONE total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
