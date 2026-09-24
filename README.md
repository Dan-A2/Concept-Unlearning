# TRIAGE — LLM Unlearning Evaluation

**TRIAGE** is a research framework for evaluating **machine unlearning methods** in large language models. The pipeline covers fine-tuning, unlearning, structural evaluation (CoFi / CHess / perplexity with confidence intervals), behavioural evaluation, robustness probes (relearning attack), ablations, and result aggregation — across multiple benchmarks and model families.

---

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **WMDP** | Weapons of Mass Destruction Proxy — bio & cyber MCQ accuracy (+ MMLU utility control) |
| **TOFU** | Fictional-persona unlearning (`forget01`, `forget05`, `forget10` splits) |
| **MUSE** | Memorization unlearning on `books` and `news` corpora |

## Unlearning Methods

| Script | Method | Notes |
|--------|--------|-------|
| `rmu.py` | RMU — Representation Misdirection Unlearning | Supports `--nu` noise regularization |
| `adaptive-rmu.py` | Adaptive RMU | Layer-adaptive variant of RMU |
| `ga.py` | Gradient Ascent | |
| `gd.py` | Gradient Difference | |
| `npo.py` | Negative Preference Optimization | |
| `dpo.py` | Direct Preference Optimization | |
| `simnpo.py` | SimNPO | |
| `rsv.py` | Representation Shift via Vectors | |
| `loku.py` | LoKu — Low-rank Knowledge Unlearning | Two-phase: importance → unlearn (ICLR 2025) |
| `atu.py` | ATU — Align-Then-Unlearn | No `--nu`; two-phase alignment + unlearning |
| `spul.py` | SPUL — Soft Prompt Unlearning | No `--nu`; LoRA merge + P-tuning |
| `obliviate.py` | Obliviate — Vocabulary-Masking Unlearning | No `--nu` |

---

## Repository Structure

```
Concept-Unlearning/
│
├── baselines/
│   ├── benchmarks.py       # Benchmark configs and CLI arg helpers
│   ├── losses.py           # GA, GD, NPO, KL, IHL, and other loss functions
│   ├── eval_metrics.py     # In-house TOFU and MUSE metrics (no external harness)
│   └── utils.py            # Model loading, data loading, shared utilities
│
├── data/
│   ├── idontknow.json      # "I don't know" response templates
│   └── keywords.json       # Sensitive keyword lists for vocabulary masking
│
├── rmu.py                  # Unlearning scripts (one per method)
├── adaptive-rmu.py
├── ga.py
├── gd.py
├── npo.py
├── dpo.py
├── simnpo.py
├── rsv.py
├── loku.py
├── atu.py
├── spul.py
├── obliviate.py
│
├── tofu_finetune.py        # Fine-tune on TOFU dataset
├── muse_finetune.py        # Fine-tune on MUSE corpus
│
├── test.py                 # Structural evaluation (CoFi / CHess / perplexity, per subset)
├── behavioral_eval.py      # Behavioural evaluation (WMDP MCQ, TOFU, MUSE metrics)
├── relearn_attack.py       # Relearning (fine-tuning) attack on unlearned checkpoints
│
├── analyze_metrics.py      # Summary tables + behaviour-group heatmaps per benchmark
├── localization_scatter.py # Forget-vs-retain scatter with 95% CI ellipses
├── behavioral_subjects.py  # Tripartite behavioural table (forget / adjacent / general)
├── aggregate_results.py    # Aggregate behavioural, adjacent-probe and relearn JSONs
│
├── chess_probe_ablation.py # CoFi/CHess robustness ablation (sample size, probes, params)
├── cofi_relative_report.py # Re-report the ablation as Relative Drop (%)
├── e2_correlations.py      # Structural vs behavioural correlations (CPU only)
├── e3_adjacency.py         # Adjacency diagnostic (Fisher overlap + embeddings)
└── exp_common.py           # Shared helpers for the validity experiments
```

> **Not tracked in this repo:** `checkpoints/`, `Results/`, `OUT/`, `importances/`, `cofi_cache/`, `data/bio-forget-corpus/`, `data/cyber-forget-corpus/`, and all `*.slurm` job scripts — generated at runtime, machine-specific, or too large for version control.

---

## Setup

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install torch transformers peft datasets accelerate
```

The evaluation stages additionally need `lm-eval` (WMDP / MMLU), `sentence-transformers` (adjacency diagnostic), and `pandas`, `seaborn`, `matplotlib`, `scipy` for the analysis scripts.

> On Sulis HPC the required modules are pre-loaded by Slurm — see the [HPC section](#running-on-hpc-slurm) below.

### 2. Data

The forget corpora are downloaded automatically by HuggingFace Datasets on first run. To cache them locally under `data/`:

```bash
python - <<'EOF'
from datasets import load_dataset
load_dataset("cais/wmdp-corpora", "bio-forget-corpus", split="train").save_to_disk("data/bio-forget-corpus")
load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train").save_to_disk("data/cyber-forget-corpus")
EOF
```

---

## Workflow

The pipeline runs in stages 1–5, with optional robustness and validity experiments on top. All commands assume you are in the project root.

---

### Stage 1 — Fine-Tuning

Fine-tune a base model before unlearning. **Skip this stage for WMDP** — unlearning runs directly on the pretrained model.

**TOFU**
```bash
python tofu_finetune.py \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --epochs 5
```

**MUSE**
```bash
python muse_finetune.py \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --muse_corpus books \
    --epochs 10 \
    --lr 1e-5
```

Checkpoints are saved to:
```
checkpoints/tofu_finetune/<model_name>/
checkpoints/muse_finetune/<corpus>/<model_name>/
```

---

### Stage 2 — Unlearning

Run any unlearning method against a checkpoint. Most methods share a common CLI interface.

**TOFU example:**
```bash
python rmu.py \
    --model_path checkpoints/tofu_finetune/Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --benchmark tofu \
    --tofu_split forget10 \
    --nu 0.0
```

**MUSE example:**
```bash
python ga.py \
    --model_path checkpoints/muse_finetune/books/Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --benchmark muse \
    --muse_corpus books \
    --nu 0.001
```

**WMDP example** (no fine-tuned checkpoint needed):
```bash
python rmu.py \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --benchmark wmdp \
    --nu 0.0
```

**LoKu** requires two sequential phases:
```bash
# Phase 1 — compute Fisher importances (once per model + benchmark)
python loku.py --mode importance \
    --model_path checkpoints/tofu_finetune/Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --benchmark tofu --tofu_split forget10

# Phase 2 — unlearn with FILA-initialized LoRA + IHL
python loku.py --mode unlearn \
    --model_path checkpoints/tofu_finetune/Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --benchmark tofu --tofu_split forget10
```

**Methods without `--nu`** (ATU, SPUL, Obliviate):
```bash
python atu.py \
    --model_path checkpoints/tofu_finetune/Llama-3.1-8B \
    --model_name Llama-3.1-8B \
    --benchmark tofu --tofu_split forget10
```

Unlearned checkpoints are saved to:
```
checkpoints/<method>/<bench_label>/<model_name>/
checkpoints/<method>/<bench_label>/<model_name>_nu/   # when --nu > 0
```

---

### Stage 3 — Structural Evaluation

`test.py` computes the TRIAGE structural metrics — CoFi (diagonal Fisher), CHess (Hutchinson diagonal Hessian), and perplexity — for the base model and each unlearned checkpoint, on every forget / retain / general corpus of the benchmark.

```bash
python test.py \
    --base-model checkpoints/tofu_finetune/Llama-3.1-8B \
    --benchmark tofu --tofu-split forget10 \
    --checkpoint checkpoints/ga/tofu-forget10/Llama-3.1-8B  "ga/Llama-3.1-8B" \
    --checkpoint checkpoints/rmu/tofu-forget10/Llama-3.1-8B "rmu/Llama-3.1-8B" \
    --subset-ids 0 1 2 \
    --n-samples-per-subset 200 \
    --ppl-max-tokens 50000 \
    --n-hutchinson 2 \
    --chess-batch-size 8 \
    --chess-max-length 512
```

Each `--checkpoint` takes a path followed by a display label.

**Confidence intervals.** Every metric is computed independently on `--subset-ids` document subsets (default `0 1 2`) of `--n-samples-per-subset` documents each (default 200). The subsets are drawn from a deterministic, model-independent seed and recorded under `cofi_cache/_subsets/<benchmark>/`, so every model and method is scored on identical documents and the spread across subsets gives a 95% CI. Results are cached per subset:

```
cofi_cache/<method>/<benchmark>/<model>/<corpus>__<metric>__sub<N>.json   # scalar
cofi_cache/<method>/<benchmark>/<model>/<corpus>__<metric>__sub<N>.pt     # diagonal
```

Re-running skips any (corpus, subset) pair already cached, so evaluation can be resumed after a wall-time limit. Use `--base-only` to precompute the base-model cache and `--require-base-cache` in the per-checkpoint jobs.

> **Note on MUSE-Books:** its forget / retain splits contain only 4–13 (very long) documents, which is fewer than one subset, so all subsets contain the same text and no meaningful CI can be derived. MUSE-Books is therefore reported as point estimates; all other benchmarks carry CIs.

---

### Stage 4 — Behavioural Evaluation

`behavioral_eval.py` dispatches on the benchmark:

* **WMDP** — `wmdp_bio`, `wmdp_cyber` and MMLU accuracy via the EleutherAI lm-evaluation-harness. Every per-subject MMLU accuracy is persisted for the tripartite table.
* **TOFU** — Forget Q_A Prob / ROUGE, forget Truth Ratio, Model Utility, and Forget Quality when a retain reference is supplied.
* **MUSE** — VerbMem and KnowMem ROUGE (forget and retain).

```bash
python behavioral_eval.py --checkpoint checkpoints/rmu/wmdp/Llama-3.1-8B --benchmark wmdp
python behavioral_eval.py --checkpoint <ckpt> --benchmark tofu --tofu-split forget10
python behavioral_eval.py --checkpoint <ckpt> --benchmark muse --muse-corpus news
```

Results are saved as `<ckpt>/behavioral_eval_<bench_label>.json`.

---

### Stage 5 — Relearning Attack

`relearn_attack.py` probes whether unlearning removed the knowledge or merely suppressed it: it merges the unlearned adapter, fine-tunes a fresh LoRA adapter on a small held-out slice of the forget corpus, and re-runs the behavioural evaluation.

```bash
python relearn_attack.py \
    --checkpoint checkpoints/rmu/wmdp/Llama-3.1-8B \
    --base-model meta-llama/Meta-Llama-3.1-8B \
    --benchmark wmdp \
    --relearn-samples 50 --relearn-epochs 5 --relearn-lr 1e-5 \
    --eval-before
```

The attack is LoRA-based for every model so the 3B/8B models and Qwen3-32B are attacked on equal footing. Each run writes before/after accuracies, the recovery delta and the full hyperparameter record to `Results/relearn/<benchmark>/`.

---

### Stage 6 — Analysis & Aggregation

**Structural tables and heatmaps:**
```bash
python analyze_metrics.py --benchmark muse-books
```
Prints the combined CoFi / CHess / PPL table (mean ± 95% CI) and writes one heatmap per model to `Results/<benchmark>_<model>_heatmap.{png,pdf}`. Methods are stacked in behaviour groups — partially-localized, collateral-dominant, globally destructive, no-op — configured per (benchmark, model) in `BEHAVIOUR_GROUPS` at the top of the script.

**Localization scatter** (forget-shift vs retain-shift, with 95% CI ellipses and the `y = x` diagonal separating partially-localized from collateral-dominant):
```bash
python localization_scatter.py --benchmark wmdp
```

**Behavioural tripartite table** (WMDP questions grouped into forget / adjacent-retain / general-info):
```bash
python behavioral_subjects.py
python behavioral_subjects.py --per-subject
python behavioral_subjects.py --adjacent virology college_biology computer_security
```

**Aggregate everything to tables and reports:**
```bash
python aggregate_results.py --out-dir Results/aggregate

# Scope to specific benchmarks / model families:
python aggregate_results.py \
    --out-dir Results/aggregate \
    --benchmark wmdp --benchmark muse-books \
    --model Llama-3.1-8B
```
One run produces the per-(benchmark, model) behavioural tables, the WMDP tripartite, the focused adjacent per-question probe (`Results/behavioral_adjacent`), and the relearning-attack comparison (`Results/relearn`). Sections can be disabled with `--no-tripartite`, `--no-adjacent-probe`, `--no-relearn`.

---

## Robustness Ablations

`chess_probe_ablation.py` measures how sensitive the CoFi / CHess conclusions are to the estimator setup, along three axes: sample size *n* ∈ {50, 100, 200}, Hutchinson probe count *K* ∈ {2, 4, 8} (CHess only), and parameter subset (attention / FFN / both). It reuses the exact estimators from `test.py`, and every break-point is checkpointed to disk so a preempted job resumes without recomputation.

```bash
python chess_probe_ablation.py \
    --base-model meta-llama/Meta-Llama-3.1-8B \
    --metric both --sizes 50 100 200 --probe-counts 2 4 8

# Re-report the same numbers as Relative Drop (%) — CPU only, nothing re-run
python cofi_relative_report.py --metric both
```

Output goes to `Results/chess_ablation/`.

---

## Validity Experiments

| Script | Question |
|--------|----------|
| `e2_correlations.py` | Do the structural metrics track the behavioural ones? Spearman correlations with BCa bootstrap CIs between CoFi shift, behavioural drop, adjacency gap and relearn recovery. CPU only. |
| `e3_adjacency.py` | Is a candidate adjacent-retain set valid? Fisher top-*k* overlap (per model) and sentence-embedding similarity (per benchmark). |

```bash
python e2_correlations.py
python e3_adjacency.py --benchmarks wmdp muse-books
```

`e2_correlations.py` and `e3_adjacency.py` write to `Results/rebuttal/`; Shared machinery for both lives in `exp_common.py`.

---

## Models

The framework is tested on:

- [`meta-llama/Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) (4-bit base checkpoint, dequantized for CoFi/CHess)

Any HuggingFace-compatible causal LM can be plugged in via `--model_path`.
