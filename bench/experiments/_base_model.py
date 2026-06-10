"""Model-agnostic base-model registry for cross-model replication.

Switches between Qwen3-4B (the v1 paper substrate) and
Llama-3.1-8B-Instruct (the v2 cross-family replication) without
duplicating any of the experiment code. Default is Qwen so all
existing runs stay byte-identical when this module is imported but
not engaged.

Usage in a runner:

    from experiments import _base_model as bm
    cfg = bm.active()
    tok = AutoTokenizer.from_pretrained(cfg.path)
    model = AutoModelForCausalLM.from_pretrained(cfg.path, dtype=torch.bfloat16)
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   **cfg.chat_kwargs)
    full = prompt_text + answer + cfg.stop_token

Env var:

    ENGRAM_BASE_MODEL=qwen3-4b   (default — v1 numbers, byte-identical)
    ENGRAM_BASE_MODEL=llama3.1-8b (v2 cross-model arm)

Add a new model by appending to ``_REGISTRY``. Output run dirs
should suffix the model tag (cfg.run_suffix) so Qwen and Llama
results don't collide.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelConfig:
    name: str                # short tag for paths/logs
    path: Path               # local HF snapshot dir
    stop_token: str          # appended to training-render after the answer
    chat_kwargs: dict[str, Any] = field(default_factory=dict)
    # Empty string = legacy/default suffix (Qwen). Non-empty for any
    # alt base model so dirs like runs/28_writingprompts__llama3.1-8b/
    # don't collide with the existing Qwen runs.
    run_suffix: str = ""


QWEN3_4B = ModelConfig(
    name="qwen3-4b",
    path=ROOT / "models" / "Qwen3-4B",
    stop_token="<|im_end|>",
    chat_kwargs={"enable_thinking": False},
    run_suffix="",  # default — preserves byte-identical run paths for v1
)


LLAMA31_8B = ModelConfig(
    name="llama3.1-8b",
    path=ROOT / "models" / "Llama-3.1-8B-Instruct",
    # Llama-3.1 chat template uses <|eot_id|> as turn-end. The
    # tokenizer renders it; we append it explicitly in
    # render_for_training so the assistant-answer span is correctly
    # bounded for the loss mask.
    stop_token="<|eot_id|>",
    chat_kwargs={},  # Llama tokenizer doesn't accept enable_thinking
    run_suffix="__llama3.1-8b",
)


MISTRAL7B_INSTRUCT_V03 = ModelConfig(
    name="mistral7b-instruct-v0.3",
    path=ROOT / "models" / "mistral-7b-instruct-v0.3",
    # Mistral-Instruct chat template emits `</s>` after the assistant
    # answer; we append it explicitly in render_for_training so the
    # answer span is correctly bounded for the loss mask.
    stop_token="</s>",
    chat_kwargs={},  # Mistral tokenizer doesn't accept enable_thinking
    run_suffix="__mistral7b-v0.3",
)


_REGISTRY: dict[str, ModelConfig] = {
    QWEN3_4B.name: QWEN3_4B,
    LLAMA31_8B.name: LLAMA31_8B,
    MISTRAL7B_INSTRUCT_V03.name: MISTRAL7B_INSTRUCT_V03,
}


def active() -> ModelConfig:
    """Return the model config selected by ``ENGRAM_BASE_MODEL``.

    Defaults to Qwen3-4B so existing runs without the env var are
    byte-identical to the v1 paper.
    """
    name = os.environ.get("ENGRAM_BASE_MODEL", QWEN3_4B.name)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown ENGRAM_BASE_MODEL={name!r}. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def run_dir(base: Path | str) -> Path:
    """Suffix a run directory with the active model's tag.

    ``run_dir('runs/28_writingprompts')`` returns
    ``runs/28_writingprompts`` for Qwen (legacy) or
    ``runs/28_writingprompts__llama3.1-8b`` for Llama.
    """
    p = Path(base)
    cfg = active()
    if not cfg.run_suffix:
        return p
    return p.with_name(p.name + cfg.run_suffix)
