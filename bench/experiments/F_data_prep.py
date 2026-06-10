#!/usr/bin/env python3
"""F_data_prep.py — Build LaMP-3 subset for Phase F.

Downloads LaMP-3 (Personalized Product Rating) train+dev questions/outputs
from the public LaMP benchmark mirror, filters users with >=8 history
reviews, samples 50 users with seed 43, and writes a personae-shaped JSON
matching the V3 schema so the existing runners can consume it.

Output:
  runs/F_lamp3/dataset.json   — { "personae": [ {id, backstory, main_qa[],
                                  probe2_qa[], split:"held_out"}, ... ] }
  runs/F_lamp3/raw/          — cached raw downloads

Usage:
    .venv/bin/python experiments/F_data_prep.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "F_lamp3"
RAW_DIR = OUT_DIR / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Public LaMP-3 download URLs (user-based split: dev = held-out users)
# Schema: questions.json = list[{id, input, profile:[{text, score, ...}]}]
#         outputs.json   = {"task":"...", "golds":[{id, output}]}
LAMP3_URLS = {
    "train_q": "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/train/train_questions.json",
    "train_o": "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/train/train_outputs.json",
    "dev_q":   "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/dev/dev_questions.json",
    "dev_o":   "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/dev/dev_outputs.json",
}

SEED = 43
N_USERS = 50
MIN_HISTORY = 8       # require >=8 prior reviews to form a backstory
N_MAIN = 4            # per-user main_qa (held-in: from training fold)
N_PROBE2 = 4          # per-user probe2_qa (held-out: from this user's other reviews)
HISTORY_REVIEWS = 8   # backstory: first 8 history reviews


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"[F_data_prep] downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def load_lamp3():
    files = {}
    for k, url in LAMP3_URLS.items():
        files[k] = fetch(url, RAW_DIR / (k + ".json"))
    train_q = json.loads(files["train_q"].read_text())
    train_o = json.loads(files["train_o"].read_text())
    dev_q = json.loads(files["dev_q"].read_text())
    dev_o = json.loads(files["dev_o"].read_text())
    # outputs may be {"task":..., "golds":[...]} or list — normalize to dict[id] -> output
    def to_map(o):
        if isinstance(o, dict) and "golds" in o:
            return {str(g["id"]): str(g["output"]) for g in o["golds"]}
        if isinstance(o, list):
            return {str(g["id"]): str(g["output"]) for g in o}
        raise ValueError(f"unknown outputs shape: {type(o)}")
    return train_q, to_map(train_o), dev_q, to_map(dev_o)


def shorten(s: str, n: int = 600) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def build_persona(user_q: dict, gold_map: dict, rng: random.Random):
    """user_q has {'id': str, 'input': str, 'profile': [...]}; profile is the
    user's full history of past (text, score). 'input' is one held-in test
    review whose gold rating is in gold_map.

    For Phase F we ignore the single LaMP test 'input' and instead carve up
    'profile' ourselves so each user yields a balanced backstory + main_qa
    + probe2_qa, mirroring the V3 personae shape exactly.
    """
    profile = user_q.get("profile", [])
    if len(profile) < MIN_HISTORY + N_MAIN + N_PROBE2:
        return None  # not enough data for this user

    # Deterministic shuffle with seed = user_id-based for reproducibility
    items = list(profile)
    rng.shuffle(items)
    history = items[:HISTORY_REVIEWS]
    main_pool = items[HISTORY_REVIEWS:HISTORY_REVIEWS + N_MAIN]
    probe2_pool = items[HISTORY_REVIEWS + N_MAIN:HISTORY_REVIEWS + N_MAIN + N_PROBE2]

    # Backstory: concatenated past reviews with their ratings
    bs_parts = []
    for r in history:
        text = shorten(str(r.get("text", "")), 400)
        score = r.get("score", "?")
        bs_parts.append(f"- (rated {score}/5) {text}")
    backstory = (
        "Past product reviews and ratings written by this user:\n"
        + "\n".join(bs_parts)
    )

    def to_qa(r):
        review_text = shorten(str(r.get("text", "")), 500)
        score = str(r.get("score", "")).strip()
        return {
            "q": (
                "Given this user's past reviewing style and rating "
                "tendencies, what rating (an integer from 1 to 5) "
                "would they give to a product with the following review "
                f"text?\n\nReview: \"{review_text}\"\n\n"
                "Answer with a single integer 1, 2, 3, 4, or 5."
            ),
            "a": score,
            "raw_text": review_text,
            "raw_score": score,
        }

    return {
        "id": f"LAMP3_U_{user_q['id']}",
        "split": "held_out",
        "backstory": backstory,
        "main_qa": [to_qa(r) for r in main_pool],
        "probe2_qa": [to_qa(r) for r in probe2_pool],
        "_source": "LaMP-3",
        "_n_history": len(history),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N_USERS,
                    help="number of users (default 50)")
    args = ap.parse_args()

    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu", file=sys.stderr)
        sys.exit(2)

    train_q, train_g, dev_q, dev_g = load_lamp3()
    print(f"[F_data_prep] LaMP-3 loaded: train_users={len(train_q)} dev_users={len(dev_q)}", flush=True)

    # Pool both train+dev users; we build held-out personae from scratch
    all_users = list(train_q) + list(dev_q)
    print(f"[F_data_prep] total users: {len(all_users)}", flush=True)

    # Filter
    filtered = [u for u in all_users if len(u.get("profile", [])) >= MIN_HISTORY + N_MAIN + N_PROBE2]
    print(f"[F_data_prep] users with >= {MIN_HISTORY + N_MAIN + N_PROBE2} reviews: {len(filtered)}", flush=True)
    if len(filtered) < args.limit:
        print(f"[F_data_prep] WARNING: only {len(filtered)} users meet filter, requested {args.limit}", flush=True)
        sample = filtered
    else:
        rng = random.Random(SEED)
        sample = rng.sample(filtered, args.limit)

    # Build personae
    personae = []
    for u in sample:
        # Per-user RNG so each user's split is deterministic and independent
        urng = random.Random(SEED * 1000 + int(str(u["id"]).split("_")[-1]) if str(u["id"]).split("_")[-1].isdigit() else hash(u["id"]) & 0xFFFFFFFF)
        p = build_persona(u, {**train_g, **dev_g}, urng)
        if p is not None:
            personae.append(p)

    print(f"[F_data_prep] built {len(personae)} personae", flush=True)

    out = {
        "schema": "v3_personae_compatible",
        "source": "LaMP-3 (Personalized Product Rating)",
        "seed": SEED,
        "n_users_target": args.limit,
        "n_users_built": len(personae),
        "personae": personae,
    }
    out_path = OUT_DIR / "dataset.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[F_data_prep] wrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)

    # Print a sample for sanity
    if personae:
        s = personae[0]
        print(f"[F_data_prep] sample persona: id={s['id']} bs_chars={len(s['backstory'])} "
              f"main={len(s['main_qa'])} probe2={len(s['probe2_qa'])}", flush=True)
        print(f"[F_data_prep] sample main[0].q[:200]: {s['main_qa'][0]['q'][:200]}", flush=True)
        print(f"[F_data_prep] sample main[0].a: {s['main_qa'][0]['a']}", flush=True)


if __name__ == "__main__":
    main()
