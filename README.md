# TRIAGE — LLM Unlearning Evaluation

**TRIAGE** is a research framework for evaluating **machine unlearning methods** in large language models. The pipeline covers fine-tuning, unlearning, evaluation, behavioral testing, and result aggregation — across multiple benchmarks and model families.

---

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **WMDP** | Weapons of Mass Destruction Proxy — bio & cyber MCQ accuracy |
| **TOFU** | Fictional-persona unlearning (`forget01`, `forget05`, `forget10` splits) |
| **MUSE** | Memorization unlearning on `books` and `news` corpora |
| **BLUR** | Broad language unlearning (`rwku`, `whp` tasks) |

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
├── test.py                 # Main evaluation script (TRIAGE metrics)
├── behavioral_eval.py      # MCQ accuracy
│
├── analyze_metrics.py      # Plot unlearning metric curves per benchmark
├── localization_scatter.py # Scatter plots for layer localization analysis
└── aggregate_results.py    # Aggregate evaluation + behavioral JSONs to CSVs
```

> **Not tracked in this repo:** `checkpoints/`, `Results/`, `importances/`, `cofi_cache/`, `data/bio-forget-corpus/`, `data/cyber-forget-corpus/` — generated at runtime or too large for version control.

---

## Setup

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install torch transformers peft datasets accelerate
```

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

The full pipeline runs in five stages. All commands assume you are in the project root.

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

### Stage 3 — Evaluation

`test.py` evaluates one or more unlearned checkpoints using the TRIAGE metrics (perplexity, Hutchinson trace, CHess).

```bash
python test.py \
    --base-model checkpoints/tofu_finetune/Llama-3.1-8B \
    --benchmark tofu --tofu-split forget10 \
    --checkpoint checkpoints/ga/tofu-forget10/Llama-3.1-8B  "ga/Llama-3.1-8B" \
    --checkpoint checkpoints/rmu/tofu-forget10/Llama-3.1-8B "rmu/Llama-3.1-8B" \
    --ppl-max-tokens 50000 \
    --n-hutchinson 2 \
    --chess-batch-size 8 \
    --chess-max-length 512
```

Each `--checkpoint` takes a path followed by a display label. Results are saved as JSON files alongside each checkpoint.

---

### Stage 4 — Behavioral Evaluation

Runs MCQ accuracy (WMDP / MMLU):

```bash
python behavioral_eval.py \
    --benchmark wmdp \
    --checkpoint checkpoints/rmu/wmdp/Llama-3.1-8B
```

Results are saved as `<ckpt>/behavioral_eval_<benchmark>.json`.

---

### Stage 5 — Analysis & Aggregation

**Plot metric curves per benchmark:**
```bash
python analyze_metrics.py --benchmark muse-books
python localization_scatter.py --benchmark muse-news
```

**Aggregate all results to CSV tables:**
```bash
python aggregate_results.py --csv-dir Results/tables

# Scope to specific benchmarks / model families:
python aggregate_results.py \
    --csv-dir Results/tables \
    --benchmark wmdp --benchmark muse-books \
    --model Llama-3.1-8B
```

Output plots go to `Results/` and CSV tables to `Results/tables/`.

---

## Running on HPC (Slurm)

All stages have corresponding Slurm scripts. Adjust the `#SBATCH` header for your account and partition, then submit:

| Stage | Slurm Script |
|-------|--------------|
| TOFU fine-tuning | `sbatch tofu_finetune.slurm` |
| MUSE fine-tuning | `sbatch muse_finetune.slurm` |
| Full unlearn sweep — TOFU | `sbatch run_all_unlearn_tofu.slurm` |
| Full unlearn sweep — MUSE | `sbatch run_all_unlearn_muse.slurm` |
| Evaluation | `sbatch run_analysis.slurm` |
| Aggregation | `sbatch aggregate_results.slurm` |

> GPU jobs request 2× A100 (80 GB) with up to 100 GB RAM. Aggregation and analysis run on CPU-only partitions.

---

## Models

The framework is tested on:

- [`meta-llama/Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

Any HuggingFace-compatible causal LM can be plugged in via `--model_path`.
