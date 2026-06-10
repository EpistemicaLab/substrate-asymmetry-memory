#!/usr/bin/env python3
"""
38_self_consistency.py — Self-consistency baseline (Wang et al., 2023).

For each (persona, probe), sample k=5 generations at temperature 0.7
from BOTH substrates (γ-LoRA and RAG, same models exp35 used).
Compute self-consistency = max majority-vote fraction among the k
samples (after light normalization). Route by argmax of
(sc_lora, sc_rag); ties → use_rag (absence-prior).

Verdict for the chosen substrate is taken from exp29's deterministic
verdicts (CORRECT/INCORRECT) — we don't re-judge the sampled outputs.
This matches P(True): the baseline is a *router*, evaluated by
whether its routing decision sends us to the correct substrate.

Outputs: runs/38_self_consistency/{V3_P_*/features.jsonl,
aggregate.json} (suffixed by ENGRAM_BASE_MODEL).

Resume-safe per persona.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._base_model import run_dir as _run_dir  # noqa: E402

EXP29 = _run_dir(ROOT / "runs" / "29_f_absence")
EXP30 = _run_dir(ROOT / "runs" / "30_mechanism")
EXP35 = _run_dir(ROOT / "runs" / "35_logit_features")
OUT = _run_dir(ROOT / "runs" / "38_self_consistency")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location(
    "exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py"
)
exp19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp19)

spec35 = importlib.util.spec_from_file_location(
    "exp35", ROOT / "experiments" / "35_logit_features.py"
)
exp35 = importlib.util.module_from_spec(spec35)
spec35.loader.exec_module(exp35)

PLAIN_SYS = "You are the user's helpful assistant."

K = 5
TEMP = 0.7


_norm_re = re.compile(r"[^a-z0-9 ]+")


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = _norm_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Collapse common absence phrasings.
    if "have not discussed" in s or "haven't discussed" in s or \
       "no we have not" in s or "not discussed" in s or \
       "don't think we" in s or "do not think we" in s:
        return "<<ABSENCE>>"
    # Truncate to first ~10 words to absorb tail variation.
    return " ".join(s.split()[:10])


@torch.no_grad()
def sample_k(tok, mdl, system: str, user: str, k: int, temp: float,
             max_new: int = 64) -> list[str]:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    cfg = exp19._bm.active()
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   **cfg.chat_kwargs)
    enc = tok(text, return_tensors="pt").to("cuda")
    out = mdl.generate(
        **enc, max_new_tokens=max_new,
        do_sample=True, temperature=temp, top_p=0.95,
        num_return_sequences=k,
        pad_token_id=tok.eos_token_id,
    )
    in_len = enc["input_ids"].shape[1]
    preds = []
    for i in range(out.shape[0]):
        new = out[i, in_len:]
        preds.append(exp19.strip_think(tok.decode(new, skip_special_tokens=True)))
    return preds


def consistency(preds: list[str]) -> tuple[float, str]:
    if not preds: return 0.0, ""
    norm = [normalize(p) for p in preds]
    cnt = Counter(norm)
    top, n = cnt.most_common(1)[0]
    return n / len(preds), top


def load_lora_for(pid: str, base):
    adapter_dir = EXP30 / pid / "lora"
    if not adapter_dir.exists():
        return None
    from peft import PeftModel
    m = PeftModel.from_pretrained(base, str(adapter_dir),
                                  is_trainable=False).cuda()
    m.eval()
    return m


def stage_score():
    from transformers import AutoTokenizer
    cfg = exp19._bm.active()
    tok = AutoTokenizer.from_pretrained(cfg.path)

    persona_dirs = sorted(EXP35.glob("V3_P_*"))
    print(f"[score] {len(persona_dirs)} personae from {EXP35}", flush=True)

    # Load dataset for backstory chunks (RAG context).
    ds = json.loads((EXP29 / "dataset.json").read_text())
    persona_by_id = {p["id"]: p for p in ds["personae"]}

    import numpy as np
    import torch.nn.functional as F
    from transformers import AutoModel
    BGE_PATH = ROOT / "models" / "bge-large-en-v1.5"
    bge_tok = AutoTokenizer.from_pretrained(str(BGE_PATH))
    bge_mdl = AutoModel.from_pretrained(str(BGE_PATH),
                                        dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def bge(texts):
        if not texts: return torch.zeros(0, 1024)
        enc = bge_tok(texts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
        h = bge_mdl(**enc).last_hidden_state[:, 0]
        return F.normalize(h, dim=-1).cpu()

    for pdir in persona_dirs:
        pid = pdir.name
        feat_path = pdir / "features.jsonl"
        if not feat_path.exists():
            continue
        out_pdir = OUT / pid
        out_pdir.mkdir(parents=True, exist_ok=True)
        out_path = out_pdir / "features.jsonl"
        rows = [json.loads(l) for l in feat_path.read_text().splitlines()
                if l.strip()]
        if out_path.exists():
            existing = sum(1 for _ in out_path.read_text().splitlines() if _.strip())
            if existing >= len(rows):
                print(f"[{pid}] cached ({existing}), skip", flush=True)
                continue

        persona = persona_by_id.get(pid)
        if persona is None:
            print(f"[{pid}] WARN no persona in dataset, skip", flush=True)
            continue

        chunks = exp35.chunk_text(persona["backstory"])
        c_emb = bge(chunks)
        q_emb = bge([r["question"] for r in rows])
        sims_q = (q_emb @ c_emb.T).numpy()
        top_idx = np.argsort(-sims_q, axis=1)[:, :3]

        base = exp19.load_base()
        lora = load_lora_for(pid, base)
        if lora is None:
            print(f"[{pid}] WARN no adapter, skip", flush=True)
            del base; gc.collect(); torch.cuda.empty_cache()
            continue

        f = out_path.open("w")
        try:
            for i, r in enumerate(rows):
                q = r["question"]
                top_chunks = [chunks[j] for j in top_idx[i]]
                # γ-LoRA: question only.
                lora_samples = sample_k(tok, lora, PLAIN_SYS, q, K, TEMP)
                sc_lora, lora_maj = consistency(lora_samples)
                # RAG: disable adapter, RAG-style prompt.
                user_rag = exp35.render_rag_user(q, top_chunks)
                with lora.disable_adapter():
                    rag_samples = sample_k(tok, lora, PLAIN_SYS, user_rag, K, TEMP)
                sc_rag, rag_maj = consistency(rag_samples)
                row = {
                    "persona_id": pid, "kind": r["kind"], "topic": r["topic"],
                    "question": q, "gold": r["gold"],
                    "lora_samples": lora_samples, "rag_samples": rag_samples,
                    "lora_majority": lora_maj, "rag_majority": rag_maj,
                    "sc_lora": sc_lora, "sc_rag": sc_rag,
                }
                f.write(json.dumps(row) + "\n"); f.flush()
        finally:
            f.close()
        del lora, base
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{pid}] done ({len(rows)} probes)", flush=True)


def stage_aggregate():
    rows = []
    for pdir in sorted(OUT.glob("V3_P_*")):
        pid = pdir.name
        fp = pdir / "features.jsonl"
        if not fp.exists():
            continue
        lora_path = EXP29 / pid / "probes_C_lora.jsonl"
        rag_path = EXP29 / pid / "probes_C_rag.jsonl"
        if not lora_path.exists() or not rag_path.exists():
            continue
        L = {(json.loads(l)["kind"], json.loads(l)["topic"]): json.loads(l)
             for l in lora_path.read_text().splitlines() if l.strip()}
        R = {(json.loads(l)["kind"], json.loads(l)["topic"]): json.loads(l)
             for l in rag_path.read_text().splitlines() if l.strip()}
        for r in (json.loads(l) for l in fp.read_text().splitlines() if l.strip()):
            key = (r["kind"], r["topic"])
            if key not in L or key not in R:
                continue
            r["lora_verdict"] = L[key]["verdict"]
            r["rag_verdict"] = R[key]["verdict"]
            rows.append(r)

    def route(r):
        if r["sc_lora"] > r["sc_rag"] + 1e-6:
            return "lora"
        return "rag"

    def correct(r):
        choice = route(r)
        return (r["lora_verdict"] if choice == "lora" else r["rag_verdict"]) == "CORRECT"

    persona_ids = sorted({r["persona_id"] for r in rows})
    present = [r for r in rows if r["kind"] == "present"]
    absence = [r for r in rows if r["kind"] == "absence"]
    p_tpr = sum(1 for r in present if correct(r)) / max(len(present), 1)
    a_tpr = sum(1 for r in absence if correct(r)) / max(len(absence), 1)
    f1 = 2 * p_tpr * a_tpr / max(p_tpr + a_tpr, 1e-9)
    routes = [route(r) for r in rows]
    out = {
        "k": K, "temperature": TEMP,
        "n_personae": len(persona_ids),
        "n_rows": len(rows),
        "present_TPR": p_tpr, "absence_TPR": a_tpr,
        "F1": f1,
        "route_lora_pct": routes.count("lora") / max(len(routes), 1),
        "route_rag_pct": routes.count("rag") / max(len(routes), 1),
    }

    try:
        agg29 = json.loads((EXP29 / "aggregate.json").read_text())
        pures = agg29["configs"]
        out["baselines"] = {
            "lora_alone_F1": pures["C_lora"]["f1"],
            "rag_alone_F1": pures["C_rag"]["f1"],
            "lora_calib_F1": pures["C_lora_calib"]["f1"],
        }
    except Exception as e:
        out["baselines_err"] = str(e)
    try:
        agg36 = json.loads((_run_dir(ROOT / "runs" / "36_calib_head_logits") /
                            "aggregate.json").read_text())
        out["hybrid_logits_F1"] = agg36.get("eval_F1")
    except Exception:
        pass

    (OUT / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "aggregate", "all"], default="all")
    args = ap.parse_args()
    if args.stage in {"score", "all"}: stage_score()
    if args.stage in {"aggregate", "all"}: stage_aggregate()


if __name__ == "__main__":
    main()
