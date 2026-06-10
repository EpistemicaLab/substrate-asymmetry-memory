#!/usr/bin/env python3
"""S040 — Run 2 additional LLM judges over the existing 750-pair §4.5 corpus.

The original 31_strict_judge run used claude-sonnet-4-6. We add:
  J1 = anthropic.claude-opus-4-8       (different scale, same family)
  J2 = us.amazon.nova-premier-v1:0     (different family entirely)

Output: runs/iclr2027_push/three_judge/{opus.jsonl, nova.jsonl}
        runs/iclr2027_push/three_judge/three_judge_kappa.json

Reads inputs from runs/31_strict_judge/{judge_calls.jsonl, <persona>/gens.json}.

Auth: AWS_PROFILE=$YOUR_PROFILE .

This is an additive run — does NOT mutate the original 31_strict_judge results.

Re-runnable: each call's output keyed on (persona_id, eval_idx, template, judge).
Reuses cached verdicts on resume.
"""
from __future__ import annotations
import json
import os
import sys
import time
import argparse
import random
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "runs" / "31_strict_judge"
OUT_DIR = REPO_ROOT / "runs" / "iclr2027_push" / "three_judge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JUDGES = {
    # judge_key -> (bedrock model id, max_tokens, temperature)
    "opus": ("us.anthropic.claude-opus-4-8", 32, 0.0),
    "nova": ("us.amazon.nova-premier-v1:0", 32, 0.0),
}

# Load templates from the original judge module (so we are bit-identical)
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "judge31", REPO_ROOT / "experiments" / "31_judge.py")
judge31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge31)

TEMPLATES = judge31.TEMPLATES
parse_vote = judge31.parse_vote


def make_client():
    return boto3.Session(profile_name=os.environ["AWS_PROFILE"]).client(
        "bedrock-runtime", region_name="us-east-1")


def build_user_prompt(rec: dict) -> str:
    """Reconstruct the same A/B prompt the original judge saw, using A_is to
    determine which side LoRA was on (so we preserve the original randomization)."""
    a_is = rec["A_is"]  # 'lora' or 'rag'
    persona_id = rec["persona_id"]
    eval_idx = rec["eval_idx"]
    gens = json.loads((SRC_DIR / persona_id / "gens.json").read_text())
    record = next(r for r in gens["records"] if r["eval_idx"] == eval_idx)
    if a_is == "lora":
        a_text, b_text = record["lora_text"], record["rag_text"]
    else:
        a_text, b_text = record["rag_text"], record["lora_text"]
    return (
        f"PREFIX:\n{record['prefix_tail']}\n\n"
        f"GOLD continuation:\n{record['gold']}\n\n"
        f"A:\n{a_text}\n\n"
        f"B:\n{b_text}\n\n"
        "Which is closer in style/voice to GOLD? Answer A, B, or TIE."
    )


def call_judge(client, model_id: str, system: str, user: str, max_tokens=32,
               temperature=0.0, retries: int = 4) -> str:
    last = ""
    # Reasoning models (Opus 4.x) reject temperature param entirely
    is_reasoning = "opus-4-8" in model_id or "opus-4-7" in model_id
    inference_cfg = {"maxTokens": max_tokens}
    if not is_reasoning:
        inference_cfg["temperature"] = temperature
    for attempt in range(retries):
        try:
            resp = client.converse(
                modelId=model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig=inference_cfg,
            )
            # Defensive: safety filters / refusals can return empty content list
            content = resp.get("output", {}).get("message", {}).get("content", []) or []
            if not content:
                stop_reason = resp.get("stopReason", "unknown")
                return f"__EMPTY_RESPONSE__:{stop_reason}"
            last = content[0].get("text", "")
            return last
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ThrottlingException", "ServiceUnavailableException"):
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue
            raise
    return last


def run_judge(judge_key: str, model_id: str, max_tokens: int, temperature: float,
              limit: int | None = None) -> Path:
    out_path = OUT_DIR / f"{judge_key}.jsonl"
    cache: dict[tuple, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["persona_id"], r["eval_idx"], r["template"])] = r

    client = make_client()
    src_records = [json.loads(l) for l in (SRC_DIR / "judge_calls.jsonl").read_text().splitlines()]
    if limit:
        src_records = src_records[:limit]

    print(f"[{judge_key}] {len(src_records)} src records, {len(cache)} cached")
    written = 0
    with out_path.open("a") as fout:
        for i, rec in enumerate(src_records):
            key = (rec["persona_id"], rec["eval_idx"], rec["template"])
            if key in cache:
                continue
            system = TEMPLATES[rec["template"]]
            user = build_user_prompt(rec)
            raw = call_judge(client, model_id, system, user, max_tokens, temperature)
            vote = parse_vote(raw)
            if vote is None:
                # one retry with stricter note
                raw2 = call_judge(client, model_id, system + judge31.STRICTER_RETRY_NOTE, user,
                                  max_tokens, temperature)
                vote = parse_vote(raw2) or "FAIL"
            winner = None
            if vote == "A":
                winner = "lora" if rec["A_is"] == "lora" else "rag"
            elif vote == "B":
                winner = "rag" if rec["A_is"] == "lora" else "lora"
            elif vote == "TIE":
                winner = "tie"
            else:
                winner = "fail"
            row = {
                "persona_id": rec["persona_id"], "eval_idx": rec["eval_idx"],
                "template": rec["template"], "A_is": rec["A_is"],
                "judge": judge_key, "model_id": model_id,
                "raw": raw, "vote": vote, "winner": winner,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            written += 1
            if (i + 1) % 50 == 0:
                print(f"[{judge_key}] {i+1}/{len(src_records)} new={written}")
    print(f"[{judge_key}] DONE — wrote {written} new, total {len(cache)+written}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", choices=list(JUDGES) + ["all"], default="all")
    p.add_argument("--limit", type=int, default=None,
                   help="cap records per judge (smoke test)")
    args = p.parse_args()
    targets = list(JUDGES.items()) if args.judge == "all" else [(args.judge, JUDGES[args.judge])]
    for k, (mid, mt, t) in targets:
        run_judge(k, mid, mt, t, limit=args.limit)
    print("ALL JUDGES DONE")


if __name__ == "__main__":
    main()
