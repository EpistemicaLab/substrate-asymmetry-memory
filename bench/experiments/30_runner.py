#!/usr/bin/env python3
"""30_runner.py — Phase K mechanism re-run at n=50.

Trains γ-LoRA on the full 50-persona held-out pool using cached
synth-QA pairs from exp23 and exp25, saves adapters to disk for the
Frobenius decomposition step (30_analyze.py).

Design: experiments/30_design.md
Plan:   docs/PAPER_PUSH_PLAN.md, Phase K
"""
from __future__ import annotations
import gc
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PERSONAE_V1 = ROOT / "runs" / "V3_personae" / "personae.json"
PERSONAE_V2 = ROOT / "runs" / "V3_personae_v2" / "personae_extra.json"
SYNTHQA_DIRS = [
    ROOT / "runs" / "23_persona_lora",
    ROOT / "runs" / "25_persona_lora_v2",
]
G_DIR = ROOT / "runs" / "G_mechanism"  # reuse 3 saved adapters from Phase G

# Suffix output dir by base model so Llama runs don't collide with Qwen runs.
sys.path.insert(0, str(ROOT))
from experiments._base_model import run_dir as _run_dir  # noqa: E402
OUT_DIR = _run_dir(ROOT / "runs" / "30_mechanism")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def _load_personae():
    by_id = {}
    for path in (PERSONAE_V1, PERSONAE_V2):
        if not path.exists():
            print(f"[30_runner] WARN: {path} missing", flush=True)
            continue
        data = json.load(open(path))
        ps = data["personae"] if isinstance(data, dict) else data
        for p in ps:
            pid = p.get("id") or p.get("persona_id")
            if pid:
                by_id.setdefault(pid, p)
    return by_id


def _find_synthqa(pid: str):
    for d in SYNTHQA_DIRS:
        p = d / pid / "synthqa.jsonl"
        if p.exists():
            return p
    return None


def _reuse_phase_g(pid: str, out_dir: Path) -> bool:
    """If Phase G already trained an adapter for pid, copy it over.

    Phase G adapters are Qwen3-4B-shape. Only reuse them when the active base
    model is Qwen3-4B; otherwise we'd copy Qwen-shape weights into a Llama dir
    and exp35 would fail to load them with "size mismatch".
    """
    # Lazy import to avoid circular import at module load
    from experiments._base_model import active as _active_cfg
    if _active_cfg().name != "qwen3-4b":
        return False
    src = G_DIR / pid / "lora"
    if (src / "adapter_model.safetensors").exists():
        adapter_dir = out_dir / "lora"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, adapter_dir / f.name)
        # Note source so the audit can tell apart copies vs trains
        meta = {"pid": pid, "source": "phase_G_copy"}
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return True
    return False


def main():
    by_id = _load_personae()
    pids = sorted(by_id.keys())
    held = [pid for pid in pids if by_id[pid].get("split") == "held_out" or pid >= "V3_P_120"]
    # V3_P_120..150 from personae_extra are already the new held-out batch (no
    # split field on those records); 101..119 are explicitly held_out.
    held = sorted(set(held))
    print(f"[30_runner] {len(held)} held-out personae to process", flush=True)

    exp19 = _load_module("exp19", ROOT / "experiments" / "19_lora_synthqa_eval_v3.py")

    print(f"[30_runner] loading base model + tokenizer...", flush=True)
    from transformers import AutoTokenizer
    from experiments._base_model import active as _active_cfg
    _cfg = _active_cfg()
    print(f"[30_runner] base model: {_cfg.name} ({_cfg.path})", flush=True)
    tok = AutoTokenizer.from_pretrained(str(_cfg.path))
    base = exp19.load_base()

    n_skipped_cached = 0
    n_skipped_phase_g = 0
    n_trained = 0
    n_no_synthqa = 0

    for pid in held:
        out_dir = OUT_DIR / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = out_dir / "lora"
        if (adapter_dir / "adapter_model.safetensors").exists():
            n_skipped_cached += 1
            continue

        # Reuse Phase G adapter if available (V3_P_101/103/105)
        if _reuse_phase_g(pid, out_dir):
            print(f"[30_runner] {pid}: copied from Phase G", flush=True)
            n_skipped_phase_g += 1
            continue

        synthqa_path = _find_synthqa(pid)
        if synthqa_path is None:
            print(f"[30_runner] {pid}: NO synthqa cache, skip", file=sys.stderr, flush=True)
            n_no_synthqa += 1
            continue

        with open(synthqa_path) as f:
            train_pairs = [json.loads(line) for line in f if line.strip()]
        n_train = int(0.8 * len(train_pairs))
        train_only = train_pairs[:n_train]
        print(f"[30_runner] {pid}: training on {len(train_only)} pairs (cache @ {synthqa_path.relative_to(ROOT)})", flush=True)

        t0 = time.time()
        model, losses = exp19.train_lora(base, tok, train_only, qid=pid)
        dt = time.time() - t0
        final_loss = losses[-1] if losses else float("nan")
        print(f"[30_runner] {pid}: train done in {dt:.1f}s, final_loss={final_loss:.4f}", flush=True)

        model.save_pretrained(str(adapter_dir))
        with open(out_dir / "meta.json", "w") as f:
            json.dump({
                "pid": pid,
                "source": "trained",
                "n_train_pairs": len(train_only),
                "synthqa_source": str(synthqa_path.relative_to(ROOT)),
                "final_loss": final_loss,
                "train_seconds": dt,
            }, f, indent=2)
        n_trained += 1

        del model
        gc.collect()
        torch.cuda.empty_cache()
        base = exp19.load_base()

    print(f"[30_runner] done: trained={n_trained} reused_G={n_skipped_phase_g} cached={n_skipped_cached} missing_synthqa={n_no_synthqa}", flush=True)


if __name__ == "__main__":
    main()
