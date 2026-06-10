# Substrate Asymmetry in User-Side Memory

A diagnostic framework for evaluating user-side memory in LLM agents along three orthogonal axes — **behavioral consistency**, **factual presence**, and **factual absence** — and the empirical finding that **no single substrate wins all three**.

This repository is the **benchmark reproducer** for:

> **Substrate Asymmetry in User-Side Memory: A Diagnostic Framework**
> Youwang Deng, *EpistemicaLab — Independent Research*, 2026.
> arXiv preprint: link added after submission.

## Headline findings

On a controlled 50-user synthetic corpus (Qwen3-4B, replicated on Llama-3.1-8B-Instruct):

| Axis | Winner | Margin |
|---|---|---|
| Behavioral style (WritingPrompts continuation, log-likelihood + 3-judge blind preference, n=750) | per-user γ-LoRA | +0.473 nat/tok; 60.3% pref |
| Factual absence (abstain-when-fact-missing, F1) | BGE-large RAG | +45.7pp absence-TPR vs γ-LoRA at matched abstain prompt |
| Mechanism | shared QKᵥ subspace (layers 21–35) | zeroing those LoRA weights raises absence-TPR +33pp **and** drops presence-TPR −20pp |

On the more heavily RLHF-tuned Llama-3.1-8B-Instruct the asymmetry **strengthens, not heals**: the parametric behavioral advantage collapses while RAG's absence-calibration lead widens — an *alignment tax on parametric user-memory*.

The synthetic-to-real transfer fails: on LaMP-3, γ-LoRA underperforms a one-line majority predictor by 28pp. A 9 × 2 mitigation cross-product diagnoses this as **instruction-following collapse, not substrate failure**: the eval-time `{1..5}` logit mask drives main-rating accuracy to ≥0.995 on every training recipe.

Substrate-selection routing turns out to be **question-classification, not calibration**: a 110M DistilBERT on the question text alone beats every logit-based router (entropy, p-true, self-consistency) tested.

## What's inside

```
bench/
  experiments/      Load-bearing experiment scripts (numbered by paper appendix).
  scripts/          Judge ensemble + analysis helpers.
  corpus/V3_personae_v2/
                    n=50 synthetic personae (backstories + per-persona main_qa
                    + presence/absence probes). 168 KB. The full persona
                    generation prompt + Bedrock pipeline is in
                    experiments/V3a_personae_gen.py for regen.
  paper/            Pointer to the arXiv preprint (added post-submission).
```

## Reproducing the headline numbers

The full re-run requires a single L40S-class GPU (48 GB VRAM is enough; per-user γ-LoRA at r=128 fits comfortably) plus AWS Bedrock access for the LLM-judge ensemble (`anthropic.claude-sonnet-4-6`, `anthropic.claude-opus-4-8`, `amazon.nova-premier-v1:0`). Total ≈ 40 GPU-hours + a few hundred USD of closed-API inference for the full headline corpus.

```bash
git clone https://github.com/EpistemicaLab/substrate-asymmetry-memory.git
cd substrate-asymmetry-memory
python -m venv .venv && source .venv/bin/activate
pip install -e .

export AWS_PROFILE=<your-bedrock-profile>      # for judge / extractor
export ENGRAM_BASE_MODEL=qwen3-4b              # or llama3.1-8b for the cross-model arm

# §4.1 Behavioral memory — train per-user γ-LoRA + measure log-likelihood lift
python bench/experiments/28_data_prep.py
python bench/experiments/V3g_train.py
python bench/experiments/28_runner.py
python bench/experiments/31_runner.py     # blind-preference pair generation
python bench/experiments/31_judge.py      # 3-judge majority (Sonnet / Opus / Nova)

# §4.2 Calibration asymmetry — presence/absence probes, 4 configs × 50 personae × 12 probes
python bench/experiments/29_data_prep.py
python bench/experiments/29_runner.py

# §4.3 Mechanism — band-zero intervention on QKᵥ layers 21–35
python bench/experiments/30_runner.py
python bench/experiments/32_band_zero_intervention.py
python bench/experiments/30_analyze.py

# §4.4 Real-data probe (LaMP-3) + 9 × 2 mitigation cross-product
python bench/experiments/F_data_prep.py
python bench/experiments/F_lamp3_runner.py
python bench/experiments/40_lamp3_mitigations.py
python bench/experiments/G_runner.py && python bench/experiments/G_analyze.py

# §4.5 Routing-as-classification — DistilBERT vs entropy / p-true / self-consistency
python bench/experiments/35_logit_features.py
python bench/experiments/37_p_true_baseline.py
python bench/experiments/38_self_consistency.py
python bench/experiments/33_router_hybrid.py
```

See [`bench/README.md`](bench/README.md) for which script writes which paper number, the run-dir conventions (`runs/<exp>/<base_model>/`), and the cross-model replication switch.

## Cross-model replication

Every runner reads the active base via `experiments/_base_model.py`:

- `ENGRAM_BASE_MODEL=qwen3-4b` → Qwen3-4B (default; v1 numbers).
- `ENGRAM_BASE_MODEL=llama3.1-8b` → Llama-3.1-8B-Instruct (the alignment-tax replication arm).

Run-dir suffixes prevent the two arms from colliding.

## Code license

[Apache License 2.0](LICENSE).

## Contact

`dengyouwang@gmail.com` &middot; [epistemicalab.github.io](https://epistemicalab.github.io/)
