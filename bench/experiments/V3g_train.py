#!/usr/bin/env python3
"""
V3g_train.py
============

V3-γ training loop: V3-α encoder + activation-distillation loss.

Per V3g_context.md:
  - Student forward: same as V3-α (encoder → soft prompt → frozen base).
  - Teacher forward: same frozen base, but reading [backstory_tokens, q, a]
    with no_grad. NO encoder.
  - Loss: α·L_qa + β·L_act
    L_qa = answer-only CE (same as V3-α)
    L_act = mean over layers M={8,17,26,35} of MSE between student and teacher
            activations at Q+A positions, after layer-norm normalization.
  - Forward hooks on base.model.layers[8,17,26,35] capture activations.
  - Per-row indexing extracts Q+A slices to common (N, max_qa_len, H).

Usage:
    python experiments/V3g_train.py --steps 10000 --out-dir runs/V3g_distill
    python experiments/V3g_train.py --smoke --steps 200 --out-dir runs/V3g_distill_smoke
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from V1_oracle_prefix import PROMPT_TEMPLATE, load_model_and_tokenizer
from V3a_encoder import SoftPromptEncoder, count_params
from V3a_train import (
    DEFAULT_PERSONAE,
    load_personae,
    encode_persona_inputs,
    build_qa_batch,
    assemble_inputs_embeds,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "V3g_distill"

# Match layers — block outputs at depths 9, 18, 27, 36 (1-indexed); 0-indexed
# Python access on `base.model.layers[*]`.
MATCH_LAYER_INDICES = [8, 17, 26, 35]


class ActivationCollector:
    """Forward-hook based collector. Captures outputs of registered modules.

    Each forward pass appends one tensor per registered hook (in registration
    order). Call .reset() between teacher and student forwards.
    """

    def __init__(self):
        self.acts: List[torch.Tensor] = []
        self.handles: list = []

    def reset(self):
        self.acts = []

    def attach(self, modules):
        """modules: list of nn.Module. Hooks fire in registration order."""
        for m in modules:
            def make_hook():
                def hook(_module, _inputs, output):
                    # Qwen2DecoderLayer.forward returns a tuple (hidden_states, ...)
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    self.acts.append(h)
                return hook
            self.handles.append(m.register_forward_hook(make_hook()))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_teacher_inputs(batch, base, tok, device: str):
    """Assemble teacher input_ids: per row, [backstory_ids_i, prompt_ids_i, answer_ids_i].

    Returns:
      input_ids: (N, T_total_max)
      attention_mask: (N, T_total_max)
      qa_start: (N,) int — index where Q+A starts in each row
      qa_len:   (N,) int — length of Q+A in each row
    """
    bs_ids_list = batch["backstory_ids"]  # one per persona, list of (T_bs,)
    persona_idx = batch["persona_idx"]    # (N,) — which persona each row belongs to
    prompt_ids_list = batch["prompt_ids"]  # one per row, list of (T_q,)
    answer_ids_list = batch["answer_ids"]  # one per row, list of (T_a,)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    N = len(prompt_ids_list)
    rows_ids = []
    qa_start_list = []
    qa_len_list = []
    for n in range(N):
        p_i = persona_idx[n].item() if isinstance(persona_idx, torch.Tensor) else persona_idx[n]
        bs_ids = bs_ids_list[p_i].to(device)
        p_ids = prompt_ids_list[n].to(device)
        a_ids = answer_ids_list[n].to(device)
        full = torch.cat([bs_ids, p_ids, a_ids])
        rows_ids.append(full)
        qa_start_list.append(bs_ids.size(0))
        qa_len_list.append(p_ids.size(0) + a_ids.size(0))

    max_len = max(r.size(0) for r in rows_ids)
    input_ids = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(N, max_len, dtype=torch.long, device=device)
    for i, r in enumerate(rows_ids):
        input_ids[i, : r.size(0)] = r
        attn_mask[i, : r.size(0)] = 1

    qa_start = torch.tensor(qa_start_list, dtype=torch.long, device=device)
    qa_len = torch.tensor(qa_len_list, dtype=torch.long, device=device)
    return input_ids, attn_mask, qa_start, qa_len


def gather_qa_slices(
    activations: torch.Tensor,
    qa_start: torch.Tensor,
    qa_len: torch.Tensor,
    max_qa_len: int,
):
    """Extract per-row Q+A activation slices into (N, max_qa_len, H), with mask.

    activations: (N, T_total, H)
    qa_start:    (N,)
    qa_len:      (N,)
    Returns:
      out:    (N, max_qa_len, H), with rows zero-padded beyond qa_len_i
      mask:   (N, max_qa_len), 1 where valid
    """
    N, T, H = activations.shape
    device = activations.device
    out = torch.zeros(N, max_qa_len, H, dtype=activations.dtype, device=device)
    mask = torch.zeros(N, max_qa_len, dtype=torch.long, device=device)
    for n in range(N):
        s = qa_start[n].item()
        L = qa_len[n].item()
        L_clamped = min(L, max_qa_len)
        out[n, :L_clamped] = activations[n, s : s + L_clamped]
        mask[n, :L_clamped] = 1
    return out, mask


def masked_mse(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor):
    """Layer-normed MSE between student and teacher, masked.

    student/teacher: (N, T, H), mask: (N, T)
    Returns scalar = mean over valid positions, after F.layer_norm on hidden dim.
    """
    H = student.size(-1)
    s_n = F.layer_norm(student.to(torch.float32), (H,))
    t_n = F.layer_norm(teacher.to(torch.float32), (H,))
    diff_sq = (s_n - t_n).pow(2).mean(dim=-1)  # (N, T)
    valid = mask.to(torch.float32)
    denom = valid.sum().clamp(min=1.0)
    return (diff_sq * valid).sum() / denom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--personae", default=str(DEFAULT_PERSONAE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--teacher-cache", default=str(ROOT / "runs" / "V3g_distill" / "teacher_acts_train.pt"),
                   help="Pre-computed teacher activations (from V3g_cache_teacher.py)")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-personae", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-bs-len", type=int, default=1024)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--encoder-d-model", type=int, default=384)
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0, help="weight on L_qa")
    p.add_argument("--beta", type=float, default=1.0, help="weight on L_act")
    args = p.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 200)
        args.save_every = max(50, args.steps // 4)
        args.log_every = 5

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    print(f"[V3g.train] loading base + tokenizer ...", flush=True)
    base, tok = load_model_and_tokenizer()
    base.eval()
    base.enable_input_require_grads()

    # Verify match layers exist
    n_layers = len(base.model.layers)
    print(f"[V3g.train] base has {n_layers} transformer layers", flush=True)
    bad = [i for i in MATCH_LAYER_INDICES if i >= n_layers]
    if bad:
        raise RuntimeError(f"Match layers {bad} exceed n_layers={n_layers}. Edit MATCH_LAYER_INDICES.")
    print(f"[V3g.train] matching at base.model.layers{MATCH_LAYER_INDICES}", flush=True)

    print(f"[V3g.train] loading personae from {args.personae}", flush=True)
    train_p, held_p = load_personae(Path(args.personae))
    print(f"[V3g.train] {len(train_p)} train, {len(held_p)} held-out", flush=True)
    if len(train_p) < args.batch_personae:
        raise RuntimeError(f"Not enough train personae ({len(train_p)})")

    print(f"[V3g.train] tokenizing backstories ...", flush=True)
    train_packed = encode_persona_inputs(train_p, tok, max_bs_len=args.max_bs_len)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encoder_vocab = max(len(tok), tok.vocab_size, (pad_id or 0) + 1)

    print(f"[V3g.train] building encoder (k={args.k}, d_model={args.encoder_d_model}) ...", flush=True)
    encoder = SoftPromptEncoder(
        vocab_size=encoder_vocab,
        d_model=args.encoder_d_model,
        n_layers=args.encoder_layers,
        d_base=base.config.hidden_size,
        k=args.k,
        max_seq_len=args.max_bs_len,
        pad_token_id=pad_id if pad_id is not None else 0,
    ).cuda().to(torch.float32)
    n_params = count_params(encoder)
    print(f"[V3g.train] encoder trainable params: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=0.01)

    # Register hooks on the matched layers (only for student forward now)
    collector = ActivationCollector()
    matched_modules = [base.model.layers[i] for i in MATCH_LAYER_INDICES]
    collector.attach(matched_modules)

    # ============ Load pre-computed teacher activations ============
    cache_path = Path(args.teacher_cache)
    if not cache_path.exists():
        raise RuntimeError(
            f"Teacher cache not found at {cache_path}. Run V3g_cache_teacher.py first."
        )
    print(f"[V3g.train] loading teacher cache from {cache_path} ...", flush=True)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_match_layers = cache["match_layers"]
    if cache_match_layers != MATCH_LAYER_INDICES:
        raise RuntimeError(
            f"Cache match_layers={cache_match_layers} != current MATCH_LAYER_INDICES={MATCH_LAYER_INDICES}"
        )
    cache_persona_to_idx = {pid: i for i, pid in enumerate(cache["personae_ids"])}
    cache_acts = cache["acts"]      # (P*Q, L, max_qa_len, H), bf16, on CPU
    cache_qa_lens = cache["qa_lens"]  # (P*Q,)
    cache_max_qa_len = cache["max_qa_len"]
    cache_Q = cache["qa_per_persona"]
    print(f"[V3g.train] cache: {cache_acts.shape}, max_qa_len={cache_max_qa_len}, Q/persona={cache_Q}", flush=True)
    # Move to GPU once if it fits (~1 GB for 100×6×4×80×2560×2 bytes)
    cache_acts = cache_acts.cuda()
    cache_qa_lens = cache_qa_lens.cuda()
    print(f"[V3g.train] cache moved to GPU", flush=True)

    log_path = out_dir / "v3g_train.log"
    print(f"[V3g.train] log → {log_path}", flush=True)
    log_f = open(log_path, "a", buffering=1)
    log_f.write(f"# V3g train started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"# args: {vars(args)}\n")
    log_f.write(f"# match_layers: {MATCH_LAYER_INDICES}\n")
    log_f.write(f"# teacher_cache: {cache_path}\n")

    t0 = time.time()
    encoder.train()
    accum_qa = 0.0
    accum_act = 0.0
    accum_total = 0.0
    accum_count = 0

    for step in range(args.steps):
        sample = rng.sample(train_packed, args.batch_personae)
        batch = build_qa_batch(sample, tok)

        # ============ Look up cached teacher activations ============
        # Each row in `batch` is (persona_idx_in_sample, qa_idx). Map to cache row.
        persona_idx_local = batch["persona_idx"]  # (N,) int — index into `sample`
        N = persona_idx_local.size(0)
        # qa_idx is the position within each persona's main_qa list. build_qa_batch
        # iterates main_qa in order, so for persona n in sample, its rows are
        # at indices [n*Q, n*Q+Q) in batch. Recover qa_idx as row_within_persona.
        # Easier: re-build the (persona_id, qa_idx) for each row by tracking.
        cache_rows = []
        for n in range(N):
            local_p = persona_idx_local[n].item()
            persona_id = sample[local_p]["id"]
            global_p = cache_persona_to_idx[persona_id]
            # qa_idx within persona: count rows already seen for this local_p
            qa_idx_within = sum(1 for j in range(n) if persona_idx_local[j].item() == local_p)
            cache_rows.append(global_p * cache_Q + qa_idx_within)
        cache_rows_t = torch.tensor(cache_rows, dtype=torch.long, device="cuda")

        # Gather: (N, L, max_qa_len, H), (N,) qa_lens
        teacher_qa = cache_acts[cache_rows_t]   # (N, L, max_qa_len, H)
        teacher_qa_lens = cache_qa_lens[cache_rows_t]  # (N,)

        # ============ STUDENT forward (with grad, hooks fire) ============
        collector.reset()
        inputs_embeds, attn_mask, labels = assemble_inputs_embeds(
            encoder, base, batch, k=args.k, device="cuda",
        )
        inputs_embeds_bf = inputs_embeds.to(base.dtype)
        out = base(
            inputs_embeds=inputs_embeds_bf,
            attention_mask=attn_mask,
            labels=labels,
        )
        student_acts = list(collector.acts)  # L tensors, each (N, T_student, H)
        if len(student_acts) != len(MATCH_LAYER_INDICES):
            raise RuntimeError(f"Expected {len(MATCH_LAYER_INDICES)} student hooks, got {len(student_acts)}")

        loss_qa = out.loss

        # ============ Q+A activation matching ============
        # Student Q+A starts at index k for all rows, length = teacher_qa_lens[i]
        qa_len = teacher_qa_lens  # (N,)
        local_max_qa_len = qa_len.max().item()
        qa_start_s = torch.full_like(qa_len, args.k)

        loss_act = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        for li, s_act in enumerate(student_acts):
            s_qa, mask_s = gather_qa_slices(s_act, qa_start_s, qa_len, local_max_qa_len)
            # Teacher cache is padded to cache_max_qa_len; trim to local_max_qa_len
            t_qa = teacher_qa[:, li, :local_max_qa_len, :]  # (N, local_max_qa_len, H)
            loss_act = loss_act + masked_mse(s_qa, t_qa, mask_s)
        loss_act = loss_act / len(MATCH_LAYER_INDICES)

        loss = args.alpha * loss_qa + args.beta * loss_act

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        accum_qa += loss_qa.item()
        accum_act += loss_act.item()
        accum_total += loss.item()
        accum_count += 1

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            avg_qa = accum_qa / accum_count
            avg_act = accum_act / accum_count
            avg_tot = accum_total / accum_count
            msg = (f"step={step+1:>6d}  L_total={avg_tot:.4f}  L_qa={avg_qa:.4f}  "
                   f"L_act={avg_act:.4f}  elapsed={elapsed:.0f}s  rate={(step+1)/elapsed:.2f} st/s")
            print(msg, flush=True)
            log_f.write(msg + "\n")
            accum_qa = accum_act = accum_total = 0.0
            accum_count = 0

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            ckpt_path = out_dir / f"ckpt_step_{step+1:06d}.pt"
            torch.save({
                "step": step + 1,
                "encoder_state": encoder.state_dict(),
                "args": vars(args),
                "vocab_size": encoder_vocab,
                "d_base": base.config.hidden_size,
                "match_layers": MATCH_LAYER_INDICES,
            }, ckpt_path)
            print(f"[V3g.train] saved {ckpt_path}", flush=True)
            log_f.write(f"# saved {ckpt_path}\n")

    collector.remove()
    log_f.write(f"# done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.close()
    print(f"[V3g.train] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
