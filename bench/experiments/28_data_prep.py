#!/usr/bin/env python3
"""exp28 data prep — build 50 prompt-cluster "users" from WritingPrompts.

Each "user" is a (prompt) cluster with K stories under it. We hold out
the last 5 stories as continuation eval prompts (last 100 chars of each
held-out story is the gold continuation; the preceding text is the prefix).

Output:
  runs/28_writingprompts/dataset.json — V3-personae-compatible:
    { "personae": [
        { "id": "WP_USER_<prompt_idx>",
          "split": "held_out",
          "backstory": <prompt + concatenated train stories>,
          "train_stories": [<full text>, ...],          # K-5 stories
          "eval_prompts": [
            {"prefix": <story[:-100]>, "gold_continuation": <story[-100:]>,
             "full_story": <story>},
            ... 5 records
          ],
        }, ... 50 personae ]
    }

Usage:
    .venv/bin/python experiments/28_data_prep.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "28_writingprompts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 43
N_USERS = 50
MIN_STORIES_PER_PROMPT = 8   # need 3 train + 5 eval at minimum
N_EVAL = 5
GOLD_TAIL_CHARS = 100
MIN_STORY_CHARS = 600        # ensure prefix has enough material


def load_writingprompts():
    """Load via HuggingFace datasets. Requires `datasets` installed in .venv."""
    from datasets import load_dataset
    print("[28_data_prep] loading euclaise/writingprompts (train split)...", flush=True)
    ds = load_dataset("euclaise/writingprompts", split="train")
    print(f"[28_data_prep] loaded {len(ds)} (prompt, story) pairs", flush=True)
    return ds


def cluster_by_prompt(ds):
    """Group stories by (normalized) prompt text. Return dict[prompt] -> [stories]."""
    clusters = {}
    for ex in ds:
        prompt = (ex.get("prompt") or "").strip()
        story = (ex.get("story") or "").strip()
        if not prompt or len(story) < MIN_STORY_CHARS:
            continue
        clusters.setdefault(prompt, []).append(story)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N_USERS)
    args = ap.parse_args()

    if os.geteuid() == 0:
        print("ERROR: must run as ubuntu", file=sys.stderr)
        sys.exit(2)

    ds = load_writingprompts()
    clusters = cluster_by_prompt(ds)
    eligible = {p: s for p, s in clusters.items() if len(s) >= MIN_STORIES_PER_PROMPT}
    print(f"[28_data_prep] {len(clusters)} unique prompts, "
          f"{len(eligible)} have >={MIN_STORIES_PER_PROMPT} stories", flush=True)

    if len(eligible) < args.limit:
        print(f"[28_data_prep] WARNING: only {len(eligible)} eligible prompts, "
              f"requested {args.limit}", flush=True)

    rng = random.Random(SEED)
    chosen_prompts = sorted(eligible.keys())
    rng.shuffle(chosen_prompts)
    chosen_prompts = chosen_prompts[:args.limit]

    personae = []
    for idx, prompt in enumerate(chosen_prompts):
        stories = sorted(eligible[prompt], key=lambda s: (len(s), s))[:20]  # cap
        urng = random.Random(SEED * 1000 + idx)
        urng.shuffle(stories)
        eval_stories = stories[:N_EVAL]
        train_stories = stories[N_EVAL:]
        if len(train_stories) < 3:
            continue
        eval_prompts = []
        for st in eval_stories:
            if len(st) <= GOLD_TAIL_CHARS + 200:
                continue
            prefix = st[:-GOLD_TAIL_CHARS]
            gold = st[-GOLD_TAIL_CHARS:]
            eval_prompts.append({
                "prefix": prefix,
                "gold_continuation": gold,
                "full_story": st,
            })
        if len(eval_prompts) < N_EVAL:
            continue
        # Backstory = prompt + train stories concatenated (used for B_full / synthetic Q/A)
        backstory_parts = [f"Writing prompt: {prompt}", "", "Past stories by this author under this prompt:"]
        for j, st in enumerate(train_stories):
            backstory_parts.append(f"\n--- Story {j+1} ---\n{st}")
        backstory = "\n".join(backstory_parts)
        personae.append({
            "id": f"WP_USER_{idx:03d}",
            "split": "held_out",
            "backstory": backstory,
            "prompt": prompt,
            "train_stories": train_stories,
            "eval_prompts": eval_prompts,
            "_source": "WritingPrompts",
            "_n_train": len(train_stories),
        })

    print(f"[28_data_prep] built {len(personae)} personae "
          f"(target {args.limit})", flush=True)

    out = {
        "schema": "v3_personae_compatible_behavioral",
        "source": "WritingPrompts (euclaise/writingprompts)",
        "seed": SEED,
        "n_users_target": args.limit,
        "n_users_built": len(personae),
        "n_eval_per_user": N_EVAL,
        "gold_tail_chars": GOLD_TAIL_CHARS,
        "personae": personae,
    }
    out_path = OUT_DIR / "dataset.json"
    out_path.write_text(json.dumps(out))
    print(f"[28_data_prep] wrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)

    if personae:
        s = personae[0]
        print(f"[28_data_prep] sample: id={s['id']} "
              f"n_train_stories={len(s['train_stories'])} "
              f"n_eval={len(s['eval_prompts'])} "
              f"backstory_chars={len(s['backstory'])}", flush=True)
        ep = s['eval_prompts'][0]
        print(f"[28_data_prep]   eval[0]: prefix_chars={len(ep['prefix'])} "
              f"gold_chars={len(ep['gold_continuation'])}", flush=True)


if __name__ == "__main__":
    main()
