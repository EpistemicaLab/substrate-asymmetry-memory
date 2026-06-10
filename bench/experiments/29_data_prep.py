#!/usr/bin/env python3
"""29_data_prep.py
==================

Build the F-absence probe dataset for exp29.

Inputs
------
- runs/V3_personae/personae.json     (19 held_out)
- runs/V3_personae_v2/personae_extra.json  (31 held_out)

Pipeline (per persona, 50 total):
  1. Present probes (n=6): paraphrase main_qa[i].q into a "Have we ever
     discussed X?" template. Gold = "yes" + brief grounded detail.
  2. Absence probes (n=6): Claude generates 6 topically-plausible-but-
     NOT-in-backstory probes, each with a verification pass that checks
     the topic does not appear in the backstory.

Output
------
- runs/29_f_absence/dataset.json
    { "personae": [
        { "id": ..., "backstory": ..., "main_qa": ..., "probe2_qa": ...,
          "probes": [
            {"kind":"present","topic":...,"question":...,"gold":"yes",...},
            {"kind":"absence","topic":...,"question":...,"gold":"no",...},
            ...
          ]
        }, ...
      ]
    }

Cost: 50 personae × ~$0.04 (paraphrase + absence-gen+verify). Total ~$2.
Budget 6 min wall on a single thread; ~2 min with ThreadPoolExecutor.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P_V1 = ROOT / "runs" / "V3_personae" / "personae.json"
P_V2 = ROOT / "runs" / "V3_personae_v2" / "personae_extra.json"
OUT_DIR = ROOT / "runs" / "29_f_absence"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET = OUT_DIR / "dataset.json"

# Reuse exp19 helpers for Bedrock client
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py"
)
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)

PARAPHRASE_PROMPT = """You will receive an existing factual question about a user.
Rewrite it in the form: "Have we ever discussed X?" — where X is a short
topical noun phrase (3-10 words) capturing the subject of the original
question, NOT the answer. Do not include the answer in X. Output a single
line of valid JSON: {"topic": "...", "question": "Have we ever discussed ...?"}

Original question: %s
Original answer: %s
"""

ABSENCE_GEN_PROMPT = """You will be given a user's backstory. Generate exactly
6 absence-probe topics: short noun phrases (3-10 words) that:
  (a) sound topically plausible given the user's domains/interests,
  (b) are DEFINITELY NOT mentioned in the backstory (no entity name,
      no event, no fact from these topics appears in the text),
  (c) are diverse — different categories of topic (hobby, place, person,
      event, object, opinion).

For each topic, also produce the question: "Have we ever discussed <topic>?"

Output a JSON array of 6 objects:
[{"topic":"...", "question":"Have we ever discussed ...?"}, ...]

BACKSTORY:
%s
"""

VERIFY_PROMPT = """You will receive a user's backstory and a candidate
"absence-probe" topic. Decide: does the backstory mention this topic in
any form (direct, paraphrase, or via specific entities/events that
clearly fall under it)?

Reply with ONE WORD: "PRESENT" if the backstory mentions it, "ABSENT"
if it does not.

BACKSTORY:
%s

CANDIDATE TOPIC: %s
"""


def claude(client, system: str, user: str, max_tokens: int = 600,
           temperature: float = 0.5, retries: int = 4) -> str:
    """Single Claude call with exponential backoff."""
    last = None
    for attempt in range(retries):
        try:
            resp = client.converse(
                modelId=exp19.BEDROCK_MODEL,
                system=[{"text": system}] if system else [{"text": "You are a careful assistant."}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            return resp["output"]["message"]["content"][0]["text"]
        except Exception as e:
            last = e
            time.sleep(2 ** attempt + 0.5)
    raise RuntimeError(f"claude call failed after {retries}: {last}")


def _strip_codefence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def paraphrase_to_present(client, q: str, a: str):
    raw = claude(client, "You output only valid JSON, no commentary.",
                 PARAPHRASE_PROMPT % (q, a), max_tokens=200, temperature=0.2)
    raw = _strip_codefence(raw)
    obj = json.loads(raw)
    return {"kind": "present", "topic": obj["topic"], "question": obj["question"],
            "gold": "yes", "source_q": q, "source_a": a}


def gen_absence_topics(client, backstory: str):
    raw = claude(client, "You output only valid JSON, no commentary.",
                 ABSENCE_GEN_PROMPT % backstory, max_tokens=900, temperature=0.7)
    raw = _strip_codefence(raw)
    arr = json.loads(raw)
    if not isinstance(arr, list) or len(arr) < 6:
        raise ValueError(f"absence-gen returned {type(arr).__name__} len={len(arr) if isinstance(arr,list) else '?'}")
    return arr[:6]


def verify_absent(client, backstory: str, topic: str) -> bool:
    raw = claude(client, "You answer in exactly one word: PRESENT or ABSENT.",
                 VERIFY_PROMPT % (backstory, topic), max_tokens=20, temperature=0.0)
    raw = raw.strip().upper()
    return raw.startswith("ABSENT")


def build_persona_probes(client, persona, max_attempts: int = 3):
    """Generate 6 present + 6 absence probes for one persona, with verification."""
    pid = persona["id"]
    backstory = persona["backstory"]
    main_qa = persona["main_qa"][:6]

    present = []
    for qa in main_qa:
        try:
            p = paraphrase_to_present(client, qa["q"], qa["a"])
            present.append(p)
        except Exception as e:
            print(f"  [{pid}] present paraphrase fail: {e}", flush=True)

    absence = []
    for attempt in range(max_attempts):
        if len(absence) >= 6:
            break
        try:
            cands = gen_absence_topics(client, backstory)
        except Exception as e:
            print(f"  [{pid}] absence-gen attempt {attempt+1} fail: {e}", flush=True)
            continue
        for c in cands:
            if len(absence) >= 6:
                break
            topic = c.get("topic", "").strip()
            question = c.get("question", "").strip()
            if not topic or not question:
                continue
            if any(a["topic"].lower() == topic.lower() for a in absence):
                continue
            try:
                if verify_absent(client, backstory, topic):
                    absence.append({"kind": "absence", "topic": topic,
                                    "question": question, "gold": "no"})
            except Exception as e:
                print(f"  [{pid}] verify fail topic={topic!r}: {e}", flush=True)

    if len(absence) < 6:
        print(f"  [{pid}] WARN: only {len(absence)} absence probes (target 6)", flush=True)

    return {
        "id": pid,
        "backstory": backstory,
        "main_qa": persona["main_qa"],
        "probe2_qa": persona.get("probe2_qa", []),
        "probes": present + absence,
        "n_present": len(present),
        "n_absence": len(absence),
    }


def load_personae():
    p1 = json.loads(P_V1.read_text())["personae"]
    p2 = json.loads(P_V2.read_text())["personae"]
    held = [x for x in p1 if x.get("split") == "held_out"]
    held += [x for x in p2 if x.get("split") == "held_out"]
    return held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Smoke: limit personae")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # Resume guard
    if DATASET.exists():
        existing = json.loads(DATASET.read_text())
        existing_ids = {p["id"] for p in existing.get("personae", [])}
        print(f"[dataset] resume: {len(existing_ids)} already done", flush=True)
    else:
        existing = {"personae": []}
        existing_ids = set()

    held = load_personae()
    if args.limit:
        held = held[:args.limit]
    todo = [p for p in held if p["id"] not in existing_ids]
    print(f"[main] total held={len(held)} todo={len(todo)}", flush=True)

    client = exp19.make_client()
    t0 = time.time()
    out = list(existing["personae"])

    def _worker(p):
        return build_persona_probes(client, p)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, p): p["id"] for p in todo}
        for i, fut in enumerate(as_completed(futs)):
            pid = futs[fut]
            try:
                rec = fut.result()
                out.append(rec)
                # Persist incrementally so a crash doesn't lose work
                DATASET.write_text(json.dumps({"personae": out}, indent=2))
                print(f"  [{i+1}/{len(todo)}] {pid} present={rec['n_present']} absence={rec['n_absence']} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
            except Exception as e:
                print(f"  [{i+1}/{len(todo)}] {pid} FAIL: {e}", flush=True)

    # Final summary
    out_sorted = sorted(out, key=lambda r: r["id"])
    DATASET.write_text(json.dumps({"personae": out_sorted}, indent=2))
    n_present = sum(p["n_present"] for p in out_sorted)
    n_absence = sum(p["n_absence"] for p in out_sorted)
    print(f"\n=== DATASET BUILT === n_personae={len(out_sorted)} "
          f"present={n_present} absence={n_absence} elapsed={(time.time()-t0)/60:.1f}min",
          flush=True)


if __name__ == "__main__":
    main()
