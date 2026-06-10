# bench/ — load-bearing reproduction code

Each script below reproduces a specific headline number from the paper. Run order matters within a section (data-prep → train → eval → analyze) but sections are independent.

## §4.1 Behavioral memory — γ-LoRA writes style

| Script | Reproduces |
|---|---|
| `experiments/V3a_personae_gen.py` | Regenerate the 50-persona synthetic corpus from scratch via Bedrock. The shipped `corpus/V3_personae_v2/` is the canonical n=50 set used in the paper. |
| `experiments/28_data_prep.py` | WritingPrompts split: 50 single-author held-out continuations. |
| `experiments/V3g_cache_teacher.py` | Pre-cache teacher logits for the per-user γ-LoRA distillation objective. |
| `experiments/V3g_train.py` | Train per-user γ-LoRA (r=128, assistant-only loss masking). |
| `experiments/23_lora_persona.py` | Persona QA-pair extraction from backstory chunks (relational + entity passes). |
| `experiments/19_lora_synthqa_eval_v3.py` | Shared eval primitives (`load_base`, `train_lora`, `gen_chat`, judge call). Imported by 29_runner. |
| `experiments/28_runner.py` | Held-out log-likelihood lift: γ-LoRA = +0.473 nat/tok over no-history baseline; BGE retrieval = +0.060 (no-op). |
| `experiments/31_runner.py` | Generate blind A/B preference pairs (n=750 LoRA-vs-RAG continuations). |
| `experiments/31_judge.py` | Strict 3-template blind-preference judge → 59.8% LoRA preference, CI [56.5, 63.1]. |
| `scripts/run_three_judge.py` | 3-judge replication across vendors (Sonnet 4.6, Opus 4.8, Nova Premier) → 60.3% majority. |
| `scripts/llm_judge_human_eval.py` | Cohen's κ between Sonnet judge and human picks on the n=30 human A/B subset. |

## §4.2 Calibration asymmetry — RAG abstains, γ-LoRA confabulates

| Script | Reproduces |
|---|---|
| `experiments/29_data_prep.py` | 12 probes per persona × 4 configs (B_nohist / C_rag / C_lora / C_lora_calib). |
| `experiments/29_runner.py` | The headline 4.2 table: Presence TPR / Absence TPR / F1 across configs. RAG vs γ-LoRA + abstain at matched prompt: **+45.7pp absence-TPR**. |

## §4.3 Mechanism — shared QKᵥ subspace, opposite-sign load-bearing

| Script | Reproduces |
|---|---|
| `experiments/30_runner.py` | Mechanism re-run at n=50 (synthetic, Qwen3-4B). |
| `experiments/30_analyze.py` | Layer/projection sensitivity analysis, opposite-sign deltas. |
| `experiments/32_band_zero_intervention.py` | Zero LoRA weights on QKᵥ projections in layers 21–35: absence-probe TPR +33pp, presence-probe TPR −20pp. |
| `experiments/32_band_zero_lamp3.py` | Same intervention transferred to LaMP-3 to check whether the mechanism is task-general. |
| `scripts/print_band_zero_deltas.py` | Pretty-print the per-layer / per-projection delta tables. |

## §4.4 Real-data probe (LaMP-3) + alignment-tax replication

| Script | Reproduces |
|---|---|
| `experiments/F_data_prep.py` | LaMP-3 (Personalized Product Rating) split: per-user 80/20 with majority/minority class control. |
| `experiments/F_lamp3_runner.py` | Per-user γ-LoRA on LaMP-3 → underperforms majority baseline by 28pp on Qwen, replicates on Llama. |
| `experiments/40_lamp3_mitigations.py` | The 9 × 2 mitigation cross-product: 9 training recipes × 2 eval-time prompt variants. The `{1..5}` logit mask drives main-rating accuracy to ≥0.995 on every recipe → instruction-following collapse, not substrate failure. |
| `experiments/G_runner.py` + `G_analyze.py` | Probe-2 ceiling characterization (0.605–0.660), task-structural not recipe-tunable. |

## §4.5 Routing-as-classification

| Script | Reproduces |
|---|---|
| `experiments/35_logit_features.py` | Per-token logit features for entropy / margin / max-prob baselines. |
| `experiments/37_p_true_baseline.py` | p-true (Kadavath et al.) confidence baseline. |
| `experiments/38_self_consistency.py` | Self-consistency router (sample-N + majority). |
| `experiments/34_calib_head.py` + `36_calib_head_logits.py` | Calibration heads on top of logit features. |
| `experiments/33_router_hybrid.py` | DistilBERT-110M on question text → beats every logit-based router. The headline §4.5 result. |

## Run-dir conventions

All runners write to `runs/<exp_name>/<base_model_tag>/...`. The base-model tag is taken from `experiments/_base_model.py` (`run_suffix` field), so the Qwen and Llama arms produce non-colliding directories like `runs/28_runner/qwen3-4b/` and `runs/28_runner/llama3.1-8b/`.

## Switching base models

```bash
ENGRAM_BASE_MODEL=qwen3-4b      python bench/experiments/<script>.py   # default
ENGRAM_BASE_MODEL=llama3.1-8b   python bench/experiments/<script>.py   # cross-family arm
```

To add a third base model, append an entry to `_REGISTRY` in `experiments/_base_model.py`.

## Bedrock judge & extractor

The synthetic-corpus generator, persona QA-pair extractor, and judge ensemble all use AWS Bedrock. Set `AWS_PROFILE` to a profile that has access to the three model IDs used:

- `us.anthropic.claude-sonnet-4-6` (primary judge + extractor)
- `us.anthropic.claude-opus-4-8` (judge replication, vendor-1)
- `us.amazon.nova-premier-v1:0` (judge replication, vendor-2)

Reasoning models (Opus 4.8) reject `temperature` in `inferenceConfig`; the judge code passes only `maxTokens` for those model IDs.

## Disk space

Per-persona LoRA adapters at r=128 are ~360 MB each. A 50-user × 8-arm sweep produces ~144 GB of adapter weights. Either (a) checkpoint to a separate volume, or (b) drop `save_adapter=True` in `V3g_train.py` and rely on the cached metric outputs alone.
