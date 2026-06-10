#!/usr/bin/env python3
"""
19_lora_synthqa_eval_v3.py
==========================

Production v2-extraction run on 25 SS-user questions. Combines:
  - exp17's per-question loop (base model reload, 30Q-style sweep)
  - exp18's two-pass extraction (overlapping chunks + relational pass)

Usage:
    AWS_PROFILE=$YOUR_PROFILE python experiments/19_lora_synthqa_eval_v3.py --n 25
"""
from __future__ import annotations
import argparse, gc, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "longmemeval" / "longmemeval_s"
# Active base model is selected via ENGRAM_BASE_MODEL env var
# (default: qwen3-4b → byte-identical to v1). Llama-3.1-8B is the
# v2 cross-model arm; see docs/LLAMA_REPLICATION_PLAN.md.
from experiments import _base_model as _bm
QWEN_PATH = _bm.QWEN3_4B.path  # legacy alias kept for downstream imports
OUT_DIR = ROOT / "runs" / "19_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT_DIR / "summary.jsonl"

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

EXTRACT_SYSTEM_PASS1 = (
    "You read a chunk of a user's chat history with an assistant and extract "
    "QA pairs that test factual claims THE USER made about themselves. "
    "Generate questions that (a) ask about the user (\"What ... did I ...?\"), "
    "(b) have a one-short-phrase answer drawn from the chunk, "
    "(c) are grounded in the chunk -- do not invent facts. "
    "Output JSON only: a list of objects {\"q\": ..., \"a\": ...}. "
    "Aim for 6-10 pairs per chunk. Vary phrasing; include both direct ('What is my X?') "
    "and indirect ('Which Y did I mention?') styles. Keep answers terse (a few words)."
)

EXTRACT_SYSTEM_PASS2 = (
    "You read a chunk of a user's chat history. Your job is to extract "
    "QA pairs that BRIDGE FACTS ACROSS DIFFERENT TURNS in this chunk: "
    "questions whose answer requires combining information from two or more "
    "user turns within the chunk. Examples of the *style* to produce:\n"
    "  - 'Where did I do <activity X>?' when one turn mentions doing X and "
    "    another turn (or earlier in the same conversation) names a place.\n"
    "  - 'Which <brand/store/app/location> did I use for <activity Z>?'\n"
    "  - 'When did I do <event Y>?' when activity and date are in different turns.\n"
    "  - 'What <object> did I buy from <place>?' linking purchases to vendors.\n"
    "  - 'How <quantity> of <thing> at <location>?' linking numbers to entities.\n"
    "If the user mentions a habit/preference in one turn (e.g. \"I shop at "
    "Target every other week\") and an event in another (e.g. \"redeemed a "
    "coupon last Sunday\"), produce the bridging QA: 'Where did I redeem the "
    "coupon? -> Target'. Only produce a pair if BOTH halves are explicitly "
    "in this chunk. Do not invent facts. If the chunk has no clear "
    "cross-turn bridges, output []. "
    "Output JSON only: a list of {\"q\": ..., \"a\": ...} objects, terse answers."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. You will be given a question, a reference "
    "(gold) answer, and a model's predicted answer. Decide if the prediction "
    "is correct. A prediction is CORRECT if it conveys the same essential "
    "information as the gold answer (even if phrased differently). It is "
    "WRONG otherwise. \"I don't know\" is always WRONG when the gold has "
    "specific information. Respond with exactly one word: CORRECT or WRONG."
)


def flatten_haystack(hs, hd):
    flat = []
    for s_idx, (sess, date) in enumerate(zip(hs, hd)):
        for t_idx, t in enumerate(sess):
            role = t.get("role", "user") if isinstance(t, dict) else "user"
            content = t.get("content", "") if isinstance(t, dict) else str(t)
            flat.append({"role": role, "content": content, "date": date,
                         "session_idx": s_idx, "turn_idx": t_idx})
    return flat


def render_chunk(turns):
    return "\n\n".join(f"[{t['date']}] {t['role']}: {t['content']}" for t in turns)


def make_chunks_overlap(flat, max_chars=8000, stride_chars=6000):
    sizes = [len(f"[{t['date']}] {t['role']}: {t['content']}") + 2 for t in flat]
    n = len(flat)
    chunks = []
    i = 0
    while i < n:
        j = i
        cur = 0
        while j < n and (cur + sizes[j] <= max_chars or j == i):
            cur += sizes[j]
            j += 1
        chunks.append(flat[i:j])
        if j >= n:
            break
        drop = 0
        new_i = i
        while new_i < j - 1 and drop < stride_chars:
            drop += sizes[new_i]
            new_i += 1
        if new_i == i:
            new_i = i + 1
        i = new_i
    return chunks


def make_client(profile=None):
    profile = profile or os.environ["AWS_PROFILE"]
    return boto3.Session(profile_name=profile).client("bedrock-runtime", region_name="us-east-1")


def _extract(client, system, chunk_text, retries=4, max_tokens=2048):
    user_text = (
        f"CHAT HISTORY CHUNK:\n{chunk_text}\n\n"
        "Output a JSON array of {\"q\": ..., \"a\": ...} pairs. JSON only."
    )
    last = None
    for attempt in range(retries):
        try:
            resp = client.converse(
                modelId=BEDROCK_MODEL,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user_text}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
            )
            text = resp["output"]["message"]["content"][0]["text"]
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                if "[]" in text:
                    return []
                raise ValueError("no JSON array")
            pairs = json.loads(m.group(0))
            return [(p["q"].strip(), str(p["a"]).strip()) for p in pairs
                    if isinstance(p, dict) and "q" in p and "a" in p]
        except Exception as e:
            last = e; time.sleep(2 ** attempt + 0.2)
    raise RuntimeError(f"extract failed: {last}")


def extract_all_parallel(chunks, system, workers=12, label=""):
    client = make_client()
    pairs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract, client, system, render_chunk(ch)): i for i, ch in enumerate(chunks)}
        for fu in as_completed(futs):
            try:
                ps = fu.result()
                pairs.extend(ps)
            except Exception as e:
                print(f"  [{label}] chunk {futs[fu]}: SKIP {e}")
    return pairs


def judge(client, question, gold, pred, retries=4):
    user = f"QUESTION: {question}\nGOLD: {gold}\nPREDICTION: {pred}\n\nVerdict:"
    last = None
    for attempt in range(retries):
        try:
            resp = client.converse(
                modelId=BEDROCK_MODEL,
                system=[{"text": JUDGE_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 8, "temperature": 0},
            )
            t = resp["output"]["message"]["content"][0]["text"].upper()
            return ("CORRECT" in t) and ("WRONG" not in t)
        except Exception as e:
            last = e; time.sleep(2 ** attempt + 0.2)
    raise RuntimeError(f"judge failed: {last}")


def render_for_training(tok, q, a):
    cfg = _bm.active()
    msgs = [{"role": "user", "content": q}]
    prompt_text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, **cfg.chat_kwargs)
    full_text = prompt_text + a + cfg.stop_token
    prompt_ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    full_ids = tok(full_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    labels = full_ids.clone()
    labels[: prompt_ids.shape[0]] = -100
    return full_ids, labels


def collate(batch, pad_id):
    maxlen = max(b[0].shape[0] for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    lbl = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :a.shape[0]] = a
        lbl[i, :b.shape[0]] = b
        attn[i, :a.shape[0]] = 1
    return ids, lbl, attn


def train_lora(qwen, tok, pairs, *, epochs=20, lr=2e-4, r=128, alpha=256, batch=4, qid=""):
    from peft import get_peft_model, LoraConfig, TaskType
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type=TaskType.CAUSAL_LM)
    qwen.gradient_checkpointing_enable()
    qwen.enable_input_require_grads()
    model = get_peft_model(qwen, cfg)
    model.train()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tokenized = [render_for_training(tok, p["q"], p["a"]) for p in pairs]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_steps = epochs * ((len(tokenized) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_steps))
    losses = []
    for ep in range(epochs):
        order = torch.randperm(len(tokenized)).tolist()
        ep_loss, n = 0.0, 0
        for bi in range(0, len(order), batch):
            b = [tokenized[i] for i in order[bi:bi + batch]]
            ids, lbl, attn = collate(b, pad_id)
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


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()


@torch.no_grad()
def gen_chat(tok, mdl, msgs, max_new=64):
    cfg = _bm.active()
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **cfg.chat_kwargs)
    enc = tok(text, return_tensors="pt").to("cuda")
    out = mdl.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new = out[0, enc["input_ids"].shape[1]:]
    return strip_think(tok.decode(new, skip_special_tokens=True))


def load_base():
    from transformers import AutoModelForCausalLM
    cfg = _bm.active()
    qwen = AutoModelForCausalLM.from_pretrained(cfg.path, dtype=torch.bfloat16).cuda()
    qwen.config.use_cache = False
    return qwen


def run_one(q, tok, judge_client):
    qid = q["question_id"]
    out_path = OUT_DIR / f"{qid}.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        print(f"  [{qid}] cached: correct={prev['correct']} pred={prev['pred'][:80]!r}")
        return prev

    flat = flatten_haystack(q["haystack_sessions"], q["haystack_dates"])
    chunks = make_chunks_overlap(flat, max_chars=8000, stride_chars=6000)

    t0 = time.time()
    print(f"  [{qid}] PASS1: extracting QA from {len(chunks)} overlapping chunks ...", flush=True)
    raw1 = extract_all_parallel(chunks, EXTRACT_SYSTEM_PASS1, workers=12, label="p1")
    t_p1 = time.time() - t0
    print(f"  [{qid}] pass1: {len(raw1)} pairs in {t_p1:.0f}s", flush=True)

    t0b = time.time()
    print(f"  [{qid}] PASS2: relational/cross-turn extraction ...", flush=True)
    raw2 = extract_all_parallel(chunks, EXTRACT_SYSTEM_PASS2, workers=12, label="p2")
    t_p2 = time.time() - t0b
    print(f"  [{qid}] pass2: {len(raw2)} relational pairs in {t_p2:.0f}s", flush=True)

    seen = set(); merged = []
    for q_, a_ in raw1 + raw2:
        k = (q_.lower().strip(), a_.lower().strip())
        if k in seen: continue
        seen.add(k); merged.append({"q": q_, "a": a_})
    print(f"  [{qid}] merged unique pairs: {len(merged)}", flush=True)

    if not merged:
        rec = {"qid": qid, "correct": False, "pred": "(no QA pairs)", "skipped": True,
               "question": q["question"], "gold": q["answer"], "n_pairs": 0}
        out_path.write_text(json.dumps(rec, indent=2))
        with SUMMARY.open("a") as f: f.write(json.dumps(rec) + "\n")
        return rec

    print(f"  [{qid}] loading fresh base model ...", flush=True)
    base_qwen = load_base()

    t1 = time.time()
    print(f"  [{qid}] training LoRA on {len(merged)} pairs ...", flush=True)
    model, losses = train_lora(base_qwen, tok, merged, qid=qid)
    t_train = time.time() - t1
    print(f"  [{qid}] train done in {t_train:.0f}s, final loss={losses[-1]:.4f}", flush=True)

    msgs = [{"role": "user", "content": q["question"]}]
    pred = gen_chat(tok, model, msgs, max_new=64)
    correct = judge(judge_client, q["question"], q["answer"], pred)

    sanity = []
    for p in merged[:4]:
        sp = gen_chat(tok, model, [{"role": "user", "content": p["q"]}], max_new=64)
        ok = p["a"].lower().strip().rstrip(".") in sp.lower()
        sanity.append({"q": p["q"], "a_gold": p["a"], "a_pred": sp, "ok": ok})
    sanity_acc = sum(s["ok"] for s in sanity) / max(1, len(sanity))

    rec = {
        "qid": qid,
        "question": q["question"],
        "gold": q["answer"],
        "pred": pred,
        "correct": correct,
        "n_pairs_pass1": len(raw1),
        "n_pairs_pass2": len(raw2),
        "n_pairs_merged": len(merged),
        "n_chunks": len(chunks),
        "final_loss": losses[-1],
        "epoch_losses": losses,
        "t_extract_pass1_s": round(t_p1, 1),
        "t_extract_pass2_s": round(t_p2, 1),
        "t_train_s": round(t_train, 1),
        "sanity_acc": sanity_acc,
        "sanity_examples": sanity,
    }
    out_path.write_text(json.dumps(rec, indent=2))
    with SUMMARY.open("a") as f:
        f.write(json.dumps({k: rec[k] for k in (
            "qid","correct","pred","gold","n_pairs_pass1","n_pairs_pass2","n_pairs_merged",
            "final_loss","sanity_acc","t_extract_pass1_s","t_extract_pass2_s","t_train_s")}) + "\n")

    del model
    del base_qwen
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    print("Loading tokenizer ...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_bm.active().path)

    judge_client = make_client()

    data = json.load(open(DATA_PATH))
    ss = [q for q in data if q["question_type"] == "single-session-user"]
    targets = ss[args.start: args.start + args.n]
    print(f"Running on {len(targets)} SS-user questions (start={args.start}) with v2 extraction", flush=True)

    results = []
    correct = 0
    t_all = time.time()
    for i, q in enumerate(targets):
        print(f"\n=== {i+1}/{len(targets)}  qid={q['question_id']}  Q: {q['question'][:80]} ===", flush=True)
        rec = run_one(q, tok, judge_client)
        if rec.get("correct"):
            correct += 1
        results.append(rec)
        n_done = i + 1
        print(f"  running acc: {correct}/{n_done} = {correct/n_done:.1%}  (elapsed {time.time()-t_all:.0f}s)", flush=True)

    print("\n=== DONE ===", flush=True)
    print(f"final acc: {correct}/{len(results)} = {correct/max(1,len(results)):.1%}", flush=True)
    for r in results:
        print(f"  {'✓' if r.get('correct') else '✗'}  [{r['qid']}] gold={r.get('gold','')!r}  pred={r.get('pred','')[:80]!r}")


if __name__ == "__main__":
    main()
