#!/usr/bin/env python3
"""LLM judge over the 30 human-eval rows for §4.1.6 Cohen's κ.

Reads:
  runs/human_eval/human_eval_sheet_done.csv  (human picks)
  runs/human_eval/human_eval_key.json        (A/B → lora/rag mapping)

Calls Bedrock Sonnet 4.6 once per row with the same task as the human
(judge style match against gold). Output is mapped to a pick (A / B /
TIE), and Cohen's κ is computed against the human picks.

Writes:
  runs/human_eval/human_eval_sheet_judge2.csv  — judge picks side-by-side
  runs/human_eval/kappa.json                   — κ + agreement stats

Usage:
  AWS_PROFILE=$YOUR_PROFILE python3 scripts/llm_judge_human_eval.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
HE_DIR = ROOT / "runs" / "human_eval"
SHEET = HE_DIR / "human_eval_sheet_done.csv"
KEY = HE_DIR / "human_eval_key.json"
OUT_CSV = HE_DIR / "human_eval_sheet_judge2.csv"
KAPPA = HE_DIR / "kappa.json"

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"
BEDROCK_REGION = "us-east-1"

PROMPT = (
    "You are a literary-style judge. You will be shown a GOLD continuation "
    "of a story written by a particular author, and two candidate "
    "continuations A and B of the SAME story prefix produced by different "
    "systems.\n\n"
    "Your job: pick the continuation that better matches the STYLE AND "
    "VOICE of the gold continuation (mood, narrator stance, tone, rhythm). "
    "Do NOT reward A or B for matching the gold's literal content; reward "
    "stylistic similarity.\n\n"
    "Output exactly one of: A, B, TIE. Then a newline. Nothing else. "
    "Do not hedge. Do not explain."
)


def call_judge(client, prefix: str, gold: str, a: str, b: str, attempt: int = 0) -> str:
    user = (
        f"PREFIX (last 300 chars of the story so far):\n{prefix}\n\n"
        f"GOLD continuation (target style):\n{gold}\n\n"
        f"Candidate A:\n{a}\n\n"
        f"Candidate B:\n{b}\n\n"
        "Which candidate better matches the gold's STYLE? Output A, B, or TIE."
    )
    try:
        resp = client.converse(
            modelId=BEDROCK_MODEL,
            messages=[{"role": "user", "content": [{"text": user}]}],
            system=[{"text": PROMPT}],
            inferenceConfig={"maxTokens": 8, "temperature": 0.0},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip().upper()
        m = re.match(r"^(A|B|TIE)", text)
        return m.group(1) if m else "TIE"
    except Exception as e:
        if attempt < 3:
            time.sleep(2 ** attempt)
            return call_judge(client, prefix, gold, a, b, attempt + 1)
        raise


def cohens_kappa(a: list[str], b: list[str], labels: list[str]) -> dict:
    """Cohen's κ for two raters over the same items."""
    n = len(a)
    assert len(b) == n
    # observed agreement
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # expected by chance
    pe = 0.0
    for lab in labels:
        pa = a.count(lab) / n
        pb = b.count(lab) / n
        pe += pa * pb
    kappa = (po - pe) / (1 - pe) if pe != 1.0 else 1.0
    return {
        "n": n,
        "po": po,
        "pe": pe,
        "kappa": kappa,
        "interpretation": (
            "almost perfect" if kappa >= 0.81 else
            "substantial" if kappa >= 0.61 else
            "moderate" if kappa >= 0.41 else
            "fair" if kappa >= 0.21 else
            "slight" if kappa >= 0.0 else
            "worse than chance"
        ),
    }


def main() -> int:
    if not SHEET.exists():
        print(f"missing {SHEET}", file=sys.stderr)
        return 1
    with open(SHEET) as f:
        rows = list(csv.DictReader(f))
    print(f"loaded {len(rows)} human-eval rows")

    session = boto3.Session(
        profile_name=os.environ["AWS_PROFILE"],
        region_name=BEDROCK_REGION,
    )
    client = session.client("bedrock-runtime")

    out_rows = []
    judge_picks = []
    human_picks = []
    for r in rows:
        rid = r["row_id"]
        prefix = r["prefix_tail_300chars"]
        gold = r["gold_continuation"]
        a = r["A"]
        b = r["B"]
        human = r["your_pick_A_B_or_TIE"].upper()
        if human == "TIE" or human == "tie":
            human = "TIE"

        t0 = time.time()
        judge = call_judge(client, prefix, gold, a, b)
        dt = time.time() - t0
        print(f"  row {rid}: human={human:3} judge={judge:3} ({dt:.1f}s)")

        out_rows.append({
            "row_id": rid,
            "persona_id": r["persona_id"],
            "human_pick": human,
            "judge_pick": judge,
            "agree": "yes" if human == judge else "no",
            "human_confidence": r["your_confidence_1_2_or_3"],
        })
        judge_picks.append(judge)
        human_picks.append(human)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT_CSV}")

    # Cohen's κ over {A, B, TIE}
    kappa_3way = cohens_kappa(human_picks, judge_picks, ["A", "B", "TIE"])

    # Two-class κ (treat TIE as drop) over decisive rows only
    pairs = [(h, j) for h, j in zip(human_picks, judge_picks)
             if h != "TIE" and j != "TIE"]
    h2 = [p[0] for p in pairs]
    j2 = [p[1] for p in pairs]
    kappa_2way = cohens_kappa(h2, j2, ["A", "B"]) if pairs else None

    # Per-confidence breakdown
    by_conf = {}
    for r in out_rows:
        c = r["human_confidence"]
        by_conf.setdefault(c, {"n": 0, "agree": 0})
        by_conf[c]["n"] += 1
        if r["agree"] == "yes":
            by_conf[c]["agree"] += 1
    for c, stats in by_conf.items():
        stats["agreement"] = stats["agree"] / stats["n"]

    out = {
        "n_rows": len(rows),
        "kappa_3way_ABTIE": kappa_3way,
        "kappa_2way_decisive_only": kappa_2way,
        "by_human_confidence": by_conf,
        "judge_model": BEDROCK_MODEL,
        "prompt_template": "semantic-style-only",
    }
    with open(KAPPA, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {KAPPA}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
