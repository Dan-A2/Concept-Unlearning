"""Self-contained TOFU and MUSE evaluation metrics.

We implement the canonical metrics directly rather than depending on an
external framework, reusing the generic primitives in ``baselines.utils``
(ROUGE-L recall, length-normalised answer probability, batched greedy
generation).

TOFU  (locuslab/TOFU)
  * forget_q_a_prob   — length-normalised P(answer | question) on the forget set
  * forget_q_a_rouge  — ROUGE-L recall of greedy answer vs ground truth (forget)
  * forget_truth_ratio_mean — mean truth ratio on the forget set
  * forget_quality    — KS-test p-value between this model's forget truth-ratio
                        distribution and a RETAIN reference model's
                        (requires --retain-ref; skipped otherwise)
  * model_utility     — harmonic mean of {Prob, ROUGE, 1-TruthRatio} over the
                        retain / real-authors / world-facts probe sets

  Truth ratio for an example with a correct (paraphrased) answer a* and a set
  of perturbed wrong answers {a_i}:
        R = mean_i P_norm(a_i) / P_norm(a*),   P_norm = exp(mean token logp).
  Higher R on the forget set -> the model treats wrong answers as plausible as
  the right one -> better forgetting. For utility we use max(0, 1-R) so that
  higher = better, matching the TOFU paper's "higher is better" aggregation.

MUSE  (muse-bench/MUSE-{News,Books})
  * verbmem_forget_rouge — verbatim memorisation: prompt the model with the
                           first k tokens of each forget document, greedily
                           continue, ROUGE-L recall vs the true continuation
  * knowmem_forget_rouge — knowledge memorisation on the forget QA set
                           (few-shot ICL), ROUGE-L recall vs the gold answer
  * knowmem_retain_rouge — same on the retain QA set (utility; higher better)

  Lower verbmem/knowmem-forget -> better unlearning; higher knowmem-retain ->
  less collateral damage.

All dataset-dependent pieces degrade gracefully: if an optional HF config is
unavailable, the dependent metric is skipped and noted under "_skipped".
"""

import math
import random
from typing import List, Optional

import numpy as np
from datasets import load_dataset

from baselines.utils import rougeL_recall, answer_logprob, batch_generate

# TOFU was fine-tuned with this exact template (see tofu_finetune.py), so we
# must reuse it for probabilities / generation to stay in-distribution.
TOFU_CTX = "Question: {q}\nAnswer: "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _harmonic_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def _norm_prob(model, tokenizer, context: str, answer: str, max_length=512) -> float:
    """Length-normalised P(answer | context) = exp(mean token log-prob)."""
    lp, n = answer_logprob(model, tokenizer, context, answer, max_length=max_length)
    return math.exp(lp) if n > 0 and np.isfinite(lp) else 0.0


def _truth_ratio(model, tokenizer, question: str, correct: str,
                 perturbed: List[str], max_length=512) -> Optional[float]:
    """R = mean_i P_norm(perturbed_i) / P_norm(correct)."""
    if not perturbed or not correct:
        return None
    ctx = TOFU_CTX.format(q=question)
    p_correct = _norm_prob(model, tokenizer, ctx, correct, max_length)
    if p_correct <= 0:
        return None
    p_perts = [_norm_prob(model, tokenizer, ctx, p, max_length) for p in perturbed]
    return float(np.mean(p_perts) / p_correct)


def _try_load(path, config, split):
    try:
        return load_dataset(path, config, split=split)
    except Exception as e:  # noqa: BLE001 — missing/renamed config -> skip metric
        print(f"    [eval_metrics] could not load {path}:{config}:{split} ({e})")
        return None


# ---------------------------------------------------------------------------
# Held-out forget split (for relearning attacks)
# ---------------------------------------------------------------------------
# Relearning attacks fine-tune on part of the forget set and must measure
# recovery on a DISJOINT held-out part, otherwise the metric conflates genuine
# knowledge recovery with re-memorisation of the exact examples just trained on.
# The split is deterministic (seed) and shared between the relearn-data builders
# below and the eval functions, so the two halves never overlap. holdout_frac=0
# disables the split (full-set eval / no relearn contamination — the default).

def _split_indices(n, holdout_frac, seed=1234):
    """Deterministic (relearn_indices, heldout_indices), both sorted lists."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = int(round(holdout_frac * n))
    return sorted(idx[k:]), sorted(idx[:k])


def _select_heldout(ds, holdout_frac, seed):
    """Return the held-out subset of a HF dataset (or the whole ds if no split)."""
    if not holdout_frac or holdout_frac <= 0:
        return ds
    _, heldout = _split_indices(len(ds), holdout_frac, seed)
    return ds.select(heldout)


def tofu_relearn_texts(tofu_split, holdout_frac, seed, n_samples=0):
    """TOFU relearn-half training texts ('Question: ..\\nAnswer: ..' format).

    Complement of the held-out eval split, so training and evaluation never
    share examples. Returns a flat list of strings.
    """
    ds = load_dataset("locuslab/TOFU", tofu_split, split="train")
    relearn, _ = _split_indices(len(ds), holdout_frac, seed)
    texts = [f"Question: {ds[i]['question']}\nAnswer: {ds[i]['answer']}" for i in relearn]
    return texts[:n_samples] if n_samples else texts


def muse_relearn_texts(corpus, holdout_frac, seed, n_samples=0):
    """MUSE relearn-half raw forget documents (complement of held-out verbmem)."""
    hub = f"muse-bench/MUSE-{corpus.capitalize()}"
    ds = load_dataset(hub, "raw", split="forget")
    texts = [r["text"] for r in ds
             if isinstance(r.get("text"), str) and r["text"].strip()]
    relearn, _ = _split_indices(len(texts), holdout_frac, seed)
    out = [texts[i] for i in relearn]
    return out[:n_samples] if n_samples else out


# ---------------------------------------------------------------------------
# TOFU
# ---------------------------------------------------------------------------

def _perturbed_list(example):
    """TOFU stores perturbed answers as a list (or single string)."""
    p = example.get("perturbed_answer", [])
    if isinstance(p, str):
        return [p]
    return list(p) if p else []


def compute_forget_truth_ratios(model, tokenizer, tofu_split="forget10",
                                n_samples=0, max_length=512,
                                holdout_frac=0.0, holdout_seed=1234) -> List[float]:
    """Per-example truth ratios on the forget set (for forget_quality KS test).

    When holdout_frac>0, only the held-out half is scored (forget10 and
    forget10_perturbed are index-aligned, so the same split applies).
    """
    cfg = f"{tofu_split}_perturbed"
    ds = _try_load("locuslab/TOFU", cfg, "train")
    if ds is None:
        return []
    ds = _select_heldout(ds, holdout_frac, holdout_seed)
    if n_samples and len(ds) > n_samples:
        ds = ds.select(range(n_samples))
    ratios = []
    for ex in ds:
        correct = ex.get("paraphrased_answer") or ex.get("answer")
        r = _truth_ratio(model, tokenizer, ex["question"], correct,
                         _perturbed_list(ex), max_length)
        if r is not None and np.isfinite(r):
            ratios.append(r)
    return ratios


def _utility_block(model, tokenizer, cfg, n_samples, gen_batch_size,
                   gen_new_tokens, max_length):
    """Collect {Prob, ROUGE, 1-TruthRatio} on one TOFU utility probe set."""
    ds = _try_load("locuslab/TOFU", cfg, "train")
    if ds is None:
        return None
    if n_samples and len(ds) > n_samples:
        ds = ds.select(range(n_samples))

    probs, tr_utils = [], []
    prompts, refs = [], []
    for ex in ds:
        q, ans = ex["question"], ex["answer"]
        ctx = TOFU_CTX.format(q=q)
        probs.append(_norm_prob(model, tokenizer, ctx, ans, max_length))
        correct = ex.get("paraphrased_answer") or ans
        r = _truth_ratio(model, tokenizer, q, correct, _perturbed_list(ex), max_length)
        if r is not None:
            tr_utils.append(max(0.0, 1.0 - r))
        prompts.append(ctx)
        refs.append(ans)

    gens = batch_generate(model, tokenizer, prompts,
                          max_new_tokens=gen_new_tokens, batch_size=gen_batch_size,
                          max_prompt_length=max_length)
    rouges = [rougeL_recall(g, r) for g, r in zip(gens, refs)]

    return {
        "prob": float(np.mean(probs)) if probs else 0.0,
        "rouge": float(np.mean(rouges)) if rouges else 0.0,
        "truth": float(np.mean(tr_utils)) if tr_utils else 0.0,
    }


def eval_tofu(model, tokenizer, tofu_split="forget10", n_samples=0,
              gen_batch_size=8, gen_new_tokens=64, max_length=512,
              retain_ref_truth_ratios: Optional[List[float]] = None,
              holdout_frac=0.0, holdout_seed=1234) -> dict:
    """Full TOFU behavioural eval. Returns a flat dict of scalar metrics.

    holdout_frac>0 restricts the FORGET metrics (forget Q_A prob/ROUGE and the
    forget truth ratios) to the held-out half — used by relearning attacks so
    recovery is measured on examples the attacker never fine-tuned on. Model
    Utility (retain / real-authors / world-facts) is unaffected: those probe
    sets are never part of the relearn data.
    """
    results = {}
    skipped = []

    # --- Forget set: Prob, ROUGE, truth ratios ---
    forget = _try_load("locuslab/TOFU", tofu_split, "train")
    if forget is not None:
        forget = _select_heldout(forget, holdout_frac, holdout_seed)
        if n_samples and len(forget) > n_samples:
            forget = forget.select(range(n_samples))
        f_probs, prompts, refs = [], [], []
        for ex in forget:
            ctx = TOFU_CTX.format(q=ex["question"])
            f_probs.append(_norm_prob(model, tokenizer, ctx, ex["answer"], max_length))
            prompts.append(ctx)
            refs.append(ex["answer"])
        gens = batch_generate(model, tokenizer, prompts,
                              max_new_tokens=gen_new_tokens, batch_size=gen_batch_size,
                              max_prompt_length=max_length)
        results["forget_q_a_prob"] = float(np.mean(f_probs)) if f_probs else float("nan")
        results["forget_q_a_rouge"] = float(np.mean(
            [rougeL_recall(g, r) for g, r in zip(gens, refs)])) if gens else float("nan")
    else:
        skipped.append(tofu_split)
        results["forget_q_a_prob"] = float("nan")
        results["forget_q_a_rouge"] = float("nan")

    # --- Truth ratios on forget (for forget_quality) ---
    forget_trs = compute_forget_truth_ratios(
        model, tokenizer, tofu_split, n_samples, max_length,
        holdout_frac=holdout_frac, holdout_seed=holdout_seed)
    results["forget_truth_ratio_mean"] = (
        float(np.mean(forget_trs)) if forget_trs else float("nan"))
    results["_forget_truth_ratios"] = forget_trs  # kept for building references

    if retain_ref_truth_ratios:
        from scipy.stats import ks_2samp
        if forget_trs:
            results["forget_quality"] = float(
                ks_2samp(forget_trs, retain_ref_truth_ratios).pvalue)
        else:
            results["forget_quality"] = float("nan")
    else:
        results["forget_quality"] = None  # no retain reference provided

    # --- Model Utility over the three probe sets ---
    util_vals = []
    for cfg in ("retain_perturbed", "real_authors_perturbed", "world_facts_perturbed"):
        block = _utility_block(model, tokenizer, cfg, n_samples, gen_batch_size,
                               gen_new_tokens, max_length)
        if block is None:
            skipped.append(cfg)
            continue
        results[f"util_{cfg}_prob"] = block["prob"]
        results[f"util_{cfg}_rouge"] = block["rouge"]
        results[f"util_{cfg}_truth"] = block["truth"]
        util_vals.extend([block["prob"], block["rouge"], block["truth"]])
    results["model_utility"] = _harmonic_mean(util_vals) if util_vals else float("nan")

    if holdout_frac and holdout_frac > 0:
        results["_holdout_frac"] = holdout_frac   # forget metrics use held-out split
    if skipped:
        results["_skipped"] = skipped
    return results


# ---------------------------------------------------------------------------
# MUSE
# ---------------------------------------------------------------------------

def eval_muse(model, tokenizer, corpus="news", n_verbmem=100, n_knowmem=100,
              verbmem_prompt_tokens=32, verbmem_new_tokens=128,
              gen_batch_size=8, max_length=2048,
              holdout_frac=0.0, holdout_seed=1234) -> dict:
    """MUSE VerbMem + KnowMem ROUGE. Returns a flat dict of scalar metrics.

    holdout_frac>0 restricts VerbMem to the held-out forget documents (used by
    relearning attacks, so verbatim recovery is measured on docs the attacker
    never fine-tuned on). KnowMem is a separate QA config whose surface form is
    disjoint from the raw-text relearn data (like WMDP corpus->MCQ), so it is
    left on the full set.
    """
    hub = f"muse-bench/MUSE-{corpus.capitalize()}"
    results = {}
    skipped = []

    # --- VerbMem: continue forget documents from a short prefix ---
    raw = _try_load(hub, "raw", "forget")
    if raw is not None:
        texts = [r["text"] for r in raw if isinstance(r.get("text"), str) and r["text"].strip()]
        if holdout_frac and holdout_frac > 0:
            _, heldout = _split_indices(len(texts), holdout_frac, holdout_seed)
            texts = [texts[i] for i in heldout]
        if n_verbmem and len(texts) > n_verbmem:
            texts = texts[:n_verbmem]
        prompts, refs = [], []
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False).input_ids
            if len(ids) <= verbmem_prompt_tokens + 8:
                continue
            prompt_ids = ids[:verbmem_prompt_tokens]
            cont_ids = ids[verbmem_prompt_tokens:verbmem_prompt_tokens + verbmem_new_tokens]
            prompts.append(tokenizer.decode(prompt_ids, skip_special_tokens=True))
            refs.append(tokenizer.decode(cont_ids, skip_special_tokens=True))
        gens = batch_generate(model, tokenizer, prompts,
                              max_new_tokens=verbmem_new_tokens,
                              batch_size=gen_batch_size,
                              max_prompt_length=max_length)
        results["verbmem_forget_rouge"] = float(np.mean(
            [rougeL_recall(g, r) for g, r in zip(gens, refs)])) if gens else float("nan")
    else:
        skipped.append("raw:forget")
        results["verbmem_forget_rouge"] = float("nan")

    # --- KnowMem: QA on forget + retain, few-shot ICL primed ---
    def _knowmem(split_qa, split_icl, key):
        qa = _try_load(hub, "knowmem", split_qa)
        if qa is None:
            skipped.append(f"knowmem:{split_qa}")
            results[key] = float("nan")
            return
        icl = _try_load(hub, "knowmem", split_icl)
        icl_prefix = ""
        if icl is not None:
            for ex in list(icl)[:4]:
                icl_prefix += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        rows = list(qa)
        if n_knowmem and len(rows) > n_knowmem:
            rows = rows[:n_knowmem]
        prompts = [icl_prefix + f"Question: {ex['question']}\nAnswer: " for ex in rows]
        refs = [ex["answer"] for ex in rows]
        gens = batch_generate(model, tokenizer, prompts, max_new_tokens=64,
                              batch_size=gen_batch_size, max_prompt_length=max_length)
        results[key] = float(np.mean(
            [rougeL_recall(g, r) for g, r in zip(gens, refs)])) if gens else float("nan")

    _knowmem("forget_qa", "forget_qa_icl", "knowmem_forget_rouge")
    _knowmem("retain_qa", "retain_qa_icl", "knowmem_retain_rouge")

    if holdout_frac and holdout_frac > 0:
        results["_holdout_frac"] = holdout_frac   # verbmem uses held-out split
    if skipped:
        results["_skipped"] = skipped
    return results
