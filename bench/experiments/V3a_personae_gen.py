#!/usr/bin/env python3
"""
V3a_personae_gen.py
===================

Generate 120 synthetic personae for V3-α (PERSOMA-style soft-prompt encoder).

Per V3_context.md § 6:
  - 100 train + 20 held-out personae (split frozen at generation time)
  - Each persona: backstory (400-600 words, ~10-12 facts) + 6 main_qa
    + 6 probe2_qa, with no answer overlap between main and probe2
  - Diverse seeds: stratified across age × profession × locale to avoid
    the "Bedrock collapses on creative-professional-abroad mode" failure
    described in V3_context.md § 9.12

Also includes a post-generation diversity check (TF-IDF cosine
similarity on backstories) — flags if any pair > 0.5 cosine.

Reuses V1_personae_gen's PERSONA_SYSTEM prompt and parse helpers.

Output: runs/V3_personae/personae.json   (list of 120 dicts, with
        "split": "train" or "held_out" set per entry)

Usage:
    AWS_PROFILE=$YOUR_PROFILE python experiments/V3a_personae_gen.py
    # smoke (5 personae only):
    AWS_PROFILE=$YOUR_PROFILE python experiments/V3a_personae_gen.py --smoke
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "V3_personae"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "personae.json"

BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

# Diversity stratification axes
AGES = ["22", "27", "31", "34", "39", "42", "47", "52", "58", "65", "71"]
PROFESSIONS = [
    "freelance illustrator", "retired nuclear engineer", "graduate student in oceanography",
    "executive chef", "air traffic controller", "neonatal ICU nurse",
    "tax accountant", "high-school music teacher", "marine biologist",
    "industrial blacksmith", "mortgage loan officer", "park ranger",
    "small-press book editor", "wedding photographer", "geriatric social worker",
    "wildlife rehabilitator", "stand-up comedian", "underwater welder",
    "lighthouse keeper", "competitive sheepdog handler", "synagogue cantor",
    "calligrapher", "sommelier", "subway conductor",
    "third-generation watchmaker", "competitive figure skater turned coach",
    "auctioneer at a livestock yard", "blacksmith specializing in damascus knives",
    "sound engineer for indie podcasts", "architectural conservator",
]
LOCALES = [
    "Lisbon, Portugal", "Saskatoon, Saskatchewan", "Santa Cruz, California",
    "Brooklyn, New York", "Auckland, New Zealand", "Marseille, France",
    "Reykjavik, Iceland", "Kyoto, Japan", "Hobart, Tasmania",
    "Asheville, North Carolina", "Galway, Ireland", "Cape Town, South Africa",
    "Halifax, Nova Scotia", "Trondheim, Norway", "Mendoza, Argentina",
    "Edinburgh, Scotland", "Chiang Mai, Thailand", "Tallinn, Estonia",
    "Vancouver Island, BC", "Marrakech, Morocco", "Wellington, New Zealand",
    "Tbilisi, Georgia", "Antwerp, Belgium", "Salt Lake City, Utah",
    "Porto, Portugal", "Bergen, Norway",
]
HOBBIES = [
    "amateur radio", "dragon boat racing", "ultramarathons", "vintage Triumph motorcycles",
    "vinyl jazz collecting", "rock climbing", "competitive cribbage", "falconry",
    "kombucha brewing", "amateur astronomy", "lock picking sport", "swing dancing",
    "letterpress printing", "competitive whistling", "underwater hockey",
    "amateur taxidermy", "competitive yo-yo", "highland bagpipes",
    "disc golf", "volunteer search-and-rescue", "kintsugi pottery repair",
    "amateur radio direction-finding", "open-water marathon swimming",
    "amateur paleontology", "jujitsu", "competitive crossword constructing",
    "shibori dyeing",
]


def make_seed(rng: random.Random, idx: int) -> dict:
    """Stratified random seed. Ensures broad coverage across the four axes."""
    age = rng.choice(AGES)
    prof = rng.choice(PROFESSIONS)
    loc = rng.choice(LOCALES)
    h1 = rng.choice(HOBBIES)
    h2 = rng.choice([h for h in HOBBIES if h != h1])
    # Add a single distinctive trait
    extras = [
        "left-handed", "vegan since 2016", "gluten-free for medical reasons",
        "twin sibling", "adopted at age 3", "speaks 4 languages",
        "type 1 diabetes", "celiac disease", "color-blind",
        "first-generation immigrant", "raised on a farm", "former pro athlete",
        "lifelong stutter", "perfect pitch", "amateur radio call sign",
    ]
    extra = rng.choice(extras)
    hint = (
        f"a {age}-year-old {prof} living in {loc}, "
        f"hobbies include {h1} and {h2}, {extra}"
    )
    return {"id": f"V3_P_{idx:03d}", "hint": hint}


PERSONA_SYSTEM = (
    "You are generating a synthetic user persona for a controlled NLP evaluation. "
    "Given the seed description, write:\n"
    "  - 'backstory': 400-600 words of first-person narrative with ~10-12 specific concrete facts "
    "    (places, names, dates, numbers, preferences). Stay grounded — every fact must be specific "
    "    and recoverable from the text. Write naturally, not as a list. Make the persona feel "
    "    like a real distinct individual; avoid generic 'creative professional in cosmopolitan city' tropes.\n"
    "  - 'main_qa': 6 Q/A pairs whose answer is a concrete short phrase from the backstory. "
    "    Mix easy direct lookups and 1-hop combinations.\n"
    "  - 'probe2_qa': 6 ADDITIONAL Q/A pairs about *different* facts in the same backstory — "
    "    no overlap with main_qa answers. Same Q/A style.\n"
    "Output JSON only:\n"
    "{\n"
    "  \"backstory\": \"...\",\n"
    "  \"main_qa\":   [{\"q\": \"...\", \"a\": \"...\"}, ...],\n"
    "  \"probe2_qa\": [{\"q\": \"...\", \"a\": \"...\"}, ...]\n"
    "}"
)


def call_bedrock(client, system: str, user: str, max_tokens: int = 4096) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    for attempt in range(5):
        try:
            resp = client.invoke_model(modelId=BEDROCK_MODEL, body=json.dumps(body))
            payload = json.loads(resp["body"].read())
            return payload["content"][0]["text"].strip()
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def parse_persona_json(raw: str):
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def gen_one(client, seed: dict) -> dict:
    raw = call_bedrock(client, PERSONA_SYSTEM, f"Seed: {seed['hint']}")
    obj = parse_persona_json(raw)
    assert "backstory" in obj and len(obj["backstory"]) > 200, "backstory too short"
    assert len(obj.get("main_qa", [])) == 6, f"main_qa len={len(obj.get('main_qa', []))}"
    assert len(obj.get("probe2_qa", [])) == 6, f"probe2_qa len={len(obj.get('probe2_qa', []))}"
    m_ans = {qa["a"].strip().lower() for qa in obj["main_qa"]}
    p_ans = {qa["a"].strip().lower() for qa in obj["probe2_qa"]}
    overlap = m_ans & p_ans
    obj["id"] = seed["id"]
    obj["seed_hint"] = seed["hint"]
    obj["_overlap_warning"] = sorted(overlap) if overlap else []
    return obj


def diversity_check(personae) -> dict:
    """Cheap TF-IDF cosine similarity on backstories. Flags pairs > 0.5."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return {"skipped": "sklearn not available"}
    texts = [p["backstory"] for p in personae]
    vec = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    n = len(personae)
    flagged = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > 0.5:
                flagged.append({
                    "i": personae[i]["id"], "j": personae[j]["id"],
                    "cos": float(sim[i, j]),
                })
    return {
        "n_pairs": n * (n - 1) // 2,
        "max_offdiag": float(sim[~(sim == sim.diagonal()[:, None])].max()) if n > 1 else 0.0,
        "n_flagged_gt_0.5": len(flagged),
        "flagged": flagged[:10],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Generate only 5 personae for testing.")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-held-out", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel Bedrock calls.")
    p.add_argument("--out", default=str(OUT_FILE))
    p.add_argument("--idx-start", type=int, default=0,
                   help="Starting index for V3_P_{idx:03d} IDs. "
                        "Use to extend an existing batch without ID collision "
                        "(e.g. --idx-start 120 to add to the V3_P_001..119 set).")
    p.add_argument("--all-held-out", action="store_true",
                   help="Mark every generated persona as 'held_out' (used when "
                        "extending an existing held-out split).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    if args.smoke:
        n_total = 5
        n_train = 4
        n_held_out = 1
    else:
        n_train = args.n_train
        n_held_out = args.n_held_out
        n_total = n_train + n_held_out

    seeds = [make_seed(rng, args.idx_start + i) for i in range(n_total)]
    print(f"[V3.gen] generating {n_total} personae "
          f"({n_train} train + {n_held_out} held-out), workers={args.workers}")

    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    out = [None] * n_total
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, client, seed): i for i, seed in enumerate(seeds)}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            try:
                obj = f.result()
                out[i] = obj
            except Exception as e:
                print(f"  [{seeds[i]['id']}] FAILED: {e}")
                out[i] = None
            done += 1
            if done % 10 == 0 or done == n_total:
                print(f"  progress: {done}/{n_total}  ({time.time()-t0:.0f}s)")

    out = [o for o in out if o is not None]
    print(f"[V3.gen] {len(out)}/{n_total} succeeded")

    # Assign train/held-out splits deterministically (first n_train = train)
    for i, o in enumerate(out):
        if args.all_held_out:
            o["split"] = "held_out"
        else:
            o["split"] = "train" if i < n_train else "held_out"

    # Diversity check
    div = diversity_check(out)
    print(f"[V3.gen] diversity: {json.dumps(div, indent=2)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"personae": out, "diversity_report": div}, f, indent=2)
    print(f"[V3.gen] wrote {out_path}")

    # Print one example
    if out:
        ex = out[0]
        print(f"\n[V3.gen] example {ex['id']} ({ex['split']}):")
        print(f"  backstory: {len(ex['backstory'])} chars")
        print(f"  main:    {[qa['q'][:50] for qa in ex['main_qa']]}")
        print(f"  probe2:  {[qa['q'][:50] for qa in ex['probe2_qa']]}")
        if ex.get("_overlap_warning"):
            print(f"  overlap warning: {ex['_overlap_warning']}")


if __name__ == "__main__":
    main()
