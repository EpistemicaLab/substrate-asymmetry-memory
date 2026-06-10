#!/usr/bin/env python3
"""exp31 judge — Phase L: blind preference between C_lora and C_rag.

Reads runs/31_strict_judge/<pid>/gens.json files, runs Bedrock Sonnet 4.6
across 3 prompt templates per record (semantic / lexical / combined),
shuffling A/B order per record. Appends to judge_calls.jsonl and writes
aggregate.json with per-template win rate + bootstrap CI.

Usage:
    AWS_PROFILE set by launcher
    .venv/bin/python experiments/31_judge.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "31_strict_judge"
CALLS_PATH = RUN_DIR / "judge_calls.jsonl"
AGG_PATH = RUN_DIR / "aggregate.json"

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"
BEDROCK_REGION = "us-east-1"
BEDROCK_PROFILE = os.environ["AWS_PROFILE"]

TEMPLATES = {
    "semantic": (
        "You are a literary-style judge. You will be shown a GOLD continuation "
        "of a story written by a particular author, and two candidate "
        "continuations A and B of the SAME story prefix produced by different "
        "systems.\n\n"
        "Your job: pick the continuation that better matches the STYLE AND "
        "VOICE of the gold continuation (mood, narrator stance, tone). Do "
        "NOT reward continuation A or B for matching the gold's literal "
        "content; reward stylistic similarity.\n\n"
        "Output exactly one of: A, B, TIE. Then a newline. Nothing else. "
        "Do not hedge. Do not explain."
    ),
    "lexical": (
        "You are a literary-style judge. You will be shown a GOLD continuation "
        "of a story by a particular author, and two candidate continuations A "
        "and B of the same prefix.\n\n"
        "Your job: pick the candidate that better matches the GOLD in "
        "VOCABULARY, sentence structure, and rhythm — the lexical surface of "
        "the writing.\n\n"
        "Output exactly one of: A, B, TIE. Then a newline. Nothing else. "
        "Do not hedge. Do not explain."
    ),
    "combined": (
        "You are a literary-style judge. You will be shown a GOLD continuation "
        "of a story by a particular author, and two candidate continuations A "
        "and B of the same prefix.\n\n"
        "Holistically, which candidate reads more like a continuation written "
        "by the SAME AUTHOR who wrote the gold? Consider style, voice, "
        "vocabulary, rhythm together.\n\n"
        "Output exactly one of: A, B, TIE. Then a newline. Nothing else. "
        "Do not hedge. Do not explain."
    ),
}

STRICTER_RETRY_NOTE = (
    "\n\nIMPORTANT: Your previous answer did not strictly match A, B, or TIE. "
    "Output ONLY one of those three tokens, no other text. No explanation."
)


def make_client():
    return boto3.Session(profile_name=BEDROCK_PROFILE).client(
        "bedrock-runtime", region_name=BEDROCK_REGION)


def parse_vote(raw: str) -> str | None:
    """Return 'A', 'B', 'TIE', or None if hedged/unparseable."""
    if not raw:
        return None
    line = raw.strip().splitlines()[0].strip().upper()
    line = line.rstrip(".").rstrip(":").strip()
    if line in ("A", "B", "TIE"):
        return line
    if line.startswith("A ") or line.startswith("A,") or line == "A":
        return "A"
    if line.startswith("B ") or line.startswith("B,") or line == "B":
        return "B"
    return None


def judge_call(client, system: str, user: str, retries: int = 4) -> str:
    last = None
    for attempt in range(retries):
        try:
            resp = client.converse(
                modelId=BEDROCK_MODEL,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 8, "temperature": 0},
            )
            return resp["output"]["message"]["content"][0]["text"]
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + 0.2)
    raise RuntimeError(f"judge_call failed: {last}")


def judge_one(client, persona_id: str, eval_idx: int, template: str,
              gold: str, lora_text: str, rag_text: str) -> dict:
    # Per-record deterministic A/B shuffle
    seed = int(hashlib.sha1(f"{persona_id}|{eval_idx}|{template}".encode()).hexdigest(), 16)
    rng = random.Random(seed)
    A_is = rng.choice(["lora", "rag"])
    if A_is == "lora":
        A_text, B_text = lora_text, rag_text
    else:
        A_text, B_text = rag_text, lora_text

    user = (
        f"GOLD continuation:\n---\n{gold}\n---\n\n"
        f"Candidate A:\n---\n{A_text}\n---\n\n"
        f"Candidate B:\n---\n{B_text}\n---\n\n"
        f"Pick A, B, or TIE."
    )
    system = TEMPLATES[template]
    raw = judge_call(client, system, user)
    vote = parse_vote(raw)
    if vote is None:
        # one stricter retry
        raw2 = judge_call(client, system + STRICTER_RETRY_NOTE, user)
        vote2 = parse_vote(raw2)
        vote = vote2 if vote2 is not None else "TIE"
        raw = f"{raw}|||RETRY|||{raw2}"
    # Map A/B back to lora/rag
    if vote == "A":
        winner = A_is
    elif vote == "B":
        winner = "rag" if A_is == "lora" else "lora"
    else:
        winner = "tie"
    return {
        "persona_id": persona_id,
        "eval_idx": eval_idx,
        "template": template,
        "A_is": A_is,
        "raw": raw,
        "vote": vote,
        "winner": winner,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def gather_records():
    out = []
    for gens_path in sorted(RUN_DIR.glob("*/gens.json")):
        d = json.loads(gens_path.read_text())
        for rec in d["records"]:
            out.append(rec)
    return out


def already_done():
    done = set()
    if CALLS_PATH.exists():
        for line in CALLS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["persona_id"], r["eval_idx"], r["template"]))
            except Exception:
                continue
    return done


def bootstrap_ci(votes, n_resamples=1000, seed=7):
    """votes: list of 1.0 (lora win) / 0.0 (rag win) / 0.5 (tie)."""
    if not votes:
        return (None, None, None)
    rng = random.Random(seed)
    n = len(votes)
    samples = []
    for _ in range(n_resamples):
        s = sum(votes[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    return (samples[int(0.025 * n_resamples)], samples[int(0.975 * n_resamples)],
            sum(votes) / n)


def aggregate():
    if not CALLS_PATH.exists():
        return
    rows = [json.loads(l) for l in CALLS_PATH.read_text().splitlines() if l.strip()]
    out = {"n_calls_total": len(rows)}
    by_template = {}
    for tpl in list(TEMPLATES.keys()) + ["__combined_macro__"]:
        if tpl == "__combined_macro__":
            sub = rows
        else:
            sub = [r for r in rows if r["template"] == tpl]
        votes = []
        ties = 0
        for r in sub:
            if r["winner"] == "lora":
                votes.append(1.0)
            elif r["winner"] == "rag":
                votes.append(0.0)
            else:
                votes.append(0.5); ties += 1
        ci_low, ci_high, mean = bootstrap_ci(votes)
        by_template[tpl] = {
            "n": len(sub),
            "lora_win_rate": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ties": ties,
            "lora_strict_wins": sum(1 for r in sub if r["winner"] == "lora"),
            "rag_strict_wins": sum(1 for r in sub if r["winner"] == "rag"),
        }
    out["by_template"] = by_template
    AGG_PATH.write_text(json.dumps(out, indent=2))
    print(f"[31j] aggregate written: {AGG_PATH}", flush=True)
    print(json.dumps(by_template, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="cap calls (debug)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    recs = gather_records()
    print(f"[31j] {len(recs)} generation records found", flush=True)
    if not recs:
        print("[31j] no gens.json files; abort", flush=True)
        return

    done = already_done()
    print(f"[31j] {len(done)} calls already on disk", flush=True)

    todo = []
    for r in recs:
        for tpl in TEMPLATES:
            key = (r["persona_id"], r["eval_idx"], tpl)
            if key not in done:
                todo.append((r, tpl))
    if args.limit:
        todo = todo[:args.limit]
    print(f"[31j] dispatching {len(todo)} new calls "
          f"(workers={args.workers})", flush=True)

    client = make_client()
    # Preflight: ensure boto3 works in worker threads (Pitfall 20)
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(lambda _: make_client().meta.region_name, range(2)))

    written = 0
    t0 = time.time()
    with CALLS_PATH.open("a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(judge_one, make_client(), r["persona_id"],
                          r["eval_idx"], tpl, r["gold"], r["lora_text"],
                          r["rag_text"]): (r["persona_id"], r["eval_idx"], tpl)
                for (r, tpl) in todo
            }
            for fut in as_completed(futs):
                pid, ei, tpl = futs[fut]
                try:
                    res = fut.result()
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
                    written += 1
                    if written % 25 == 0:
                        rate = written / (time.time() - t0)
                        print(f"[31j] {written}/{len(todo)} done "
                              f"({rate:.1f}/s)", flush=True)
                except Exception as e:
                    print(f"[31j] FAIL {pid}/{ei}/{tpl}: {e}", flush=True)

    print(f"[31j] all calls done; writing aggregate", flush=True)
    aggregate()


if __name__ == "__main__":
    main()
