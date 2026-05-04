"""Behavioral evaluations for unlearned Llama checkpoints.

Two modes (driven by --benchmark):

    wmdp : MCQ accuracy on wmdp_bio, wmdp_cyber and MMLU (utility control),
           computed with EleutherAI lm-eval-harness.
    muse : Memorization Accuracy and 10-gram Extraction Likelihood on the
           MUSE forget split (and retain split as a control).

The script prints a compact summary to stdout and dumps a JSON next to the
checkpoint at  <ckpt>/behavioral_eval_<benchmark>.json  so a later analysis
script can aggregate results without re-running anything.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Model loading (mirrors test.py: merges LoRA adapter when present)
# --------------------------------------------------------------------------

def _dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_model_and_tokenizer(model_path: str):
    print(f"  loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        base = cfg["base_model_name_or_path"]
        print(f"  LoRA adapter detected — base={base}")
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=_dtype(), trust_remote_code=True, device_map="auto",
        )
        model = PeftModel.from_pretrained(model, model_path)
        if cfg.get("peft_type", "LORA").upper() == "LORA":
            model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=_dtype(), trust_remote_code=True, device_map="auto",
        )
    model.eval()
    return model, tokenizer


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# WMDP / MMLU via lm-evaluation-harness
# --------------------------------------------------------------------------

def run_wmdp_mmlu(model_path: str, batch_size: int, mmlu_limit: int | None):
    """Use lm-eval-harness to score wmdp_bio, wmdp_cyber and MMLU.

    For LoRA checkpoints we load the base model and pass `peft=` so the harness
    handles adapter loading. For a full HF model we just point `pretrained=` at
    the path.
    """
    try:
        from lm_eval import simple_evaluate
    except ImportError as e:
        raise SystemExit(
            "lm-eval is not installed. Install with: pip install lm-eval==0.4.5"
        ) from e

    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        base = cfg["base_model_name_or_path"]
        model_args = (
            f"pretrained={base},peft={model_path},dtype=bfloat16,trust_remote_code=True"
        )
    else:
        model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"

    tasks = ["wmdp_bio", "wmdp_cyber", "mmlu"]
    print(f"  lm-eval tasks: {tasks}")
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        limit=mmlu_limit,  # applied per-subtask; safe to leave None
    )

    out = {}
    for task in tasks:
        if task == "mmlu":
            # Aggregate score lives at "mmlu"; each subject also reported
            res = results["results"].get("mmlu", {})
            out["mmlu"] = {
                "acc": res.get("acc,none", res.get("acc", float("nan"))),
                "acc_stderr": res.get("acc_stderr,none", res.get("acc_stderr", float("nan"))),
            }
        else:
            res = results["results"][task]
            out[task] = {
                "acc": res.get("acc,none", res.get("acc", float("nan"))),
                "acc_stderr": res.get("acc_stderr,none", res.get("acc_stderr", float("nan"))),
            }
    return out


# --------------------------------------------------------------------------
# MUSE: Memorization Accuracy + n-gram Extraction Likelihood
# --------------------------------------------------------------------------

def _load_muse_texts(corpus: str, split: str):
    hub_id = f"muse-bench/MUSE-{corpus.capitalize()}"
    ds = load_dataset(hub_id, "raw", split=split)
    return [r["text"] for r in ds if isinstance(r["text"], str) and r["text"].strip()]


@torch.no_grad()
def memorization_accuracy(
    model, tokenizer, texts, prefix_tokens=200, gen_tokens=100, max_docs=100,
):
    """Greedy-decode `gen_tokens` tokens from a `prefix_tokens`-token prefix
    and report token-level exact-match accuracy against the ground truth.

    Mean over documents of (matches / gen_tokens).
    """
    device = next(model.parameters()).device
    accs = []
    used = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < prefix_tokens + gen_tokens:
            continue
        used += 1
        if used > max_docs:
            break
        prefix = torch.tensor(ids[:prefix_tokens], device=device).unsqueeze(0)
        gold = torch.tensor(ids[prefix_tokens:prefix_tokens + gen_tokens], device=device)

        gen = model.generate(
            prefix,
            max_new_tokens=gen_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        pred = gen[0, prefix_tokens:prefix_tokens + gen_tokens]
        # If generation hit EOS early, pad with -1 so they count as wrong.
        if pred.shape[0] < gen_tokens:
            pad = torch.full((gen_tokens - pred.shape[0],), -1, device=device, dtype=pred.dtype)
            pred = torch.cat([pred, pad])
        accs.append((pred == gold).float().mean().item())
        if (used % 10) == 0:
            print(f"    memacc  {used:>4d}/{max_docs}  running={sum(accs)/len(accs):.4f}")

    if not accs:
        return float("nan"), 0
    return sum(accs) / len(accs), len(accs)


@torch.no_grad()
def ngram_extraction_likelihood(
    model, tokenizer, texts, n=10, prefix_tokens=200, max_docs=100, max_windows_per_doc=5,
):
    """Average log-probability the model assigns to a verbatim n-gram from the
    forget text given the preceding `prefix_tokens`-token context.

    For each document we sample up to `max_windows_per_doc` non-overlapping
    windows, return mean log-prob (per-token) and likelihood (geometric mean
    probability).
    """
    device = next(model.parameters()).device
    log_probs_per_token = []
    used = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        max_start = len(ids) - prefix_tokens - n
        if max_start <= 0:
            continue
        used += 1
        if used > max_docs:
            break
        # Evenly spaced window starts
        starts = list(range(0, max_start, max(1, max_start // max_windows_per_doc)))[:max_windows_per_doc]
        for s in starts:
            ctx = torch.tensor(ids[s:s + prefix_tokens + n], device=device).unsqueeze(0)
            out = model(ctx)
            logits = out.logits[0, :-1, :]              # predicts ctx[1:]
            targets = ctx[0, 1:]                         # ground-truth shifted
            # Take only the last n positions (the n-gram we want extracted)
            ng_logits = logits[-n:, :].float()
            ng_targets = targets[-n:]
            log_p = torch.log_softmax(ng_logits, dim=-1)
            tok_lp = log_p.gather(1, ng_targets.unsqueeze(1)).squeeze(1)  # (n,)
            log_probs_per_token.extend(tok_lp.tolist())
        if (used % 10) == 0:
            mean_lp = sum(log_probs_per_token) / len(log_probs_per_token)
            print(f"    ngram   {used:>4d}/{max_docs}  mean_logp/tok={mean_lp:.4f}")

    if not log_probs_per_token:
        return float("nan"), float("nan"), 0
    mean_lp = sum(log_probs_per_token) / len(log_probs_per_token)
    return mean_lp, math.exp(mean_lp), len(log_probs_per_token) // n


def run_muse(model, tokenizer, corpus: str, max_docs: int):
    """Run MemAcc + 10-gram extraction on forget and retain1 splits."""
    out = {}
    for split in ("forget", "retain1"):
        print(f"  MUSE/{corpus}/{split}: loading texts ...")
        texts = _load_muse_texts(corpus, split)
        print(f"    {len(texts)} docs")

        print(f"  MUSE/{corpus}/{split}: memorization accuracy ...")
        memacc, n_used = memorization_accuracy(model, tokenizer, texts, max_docs=max_docs)

        print(f"  MUSE/{corpus}/{split}: 10-gram extraction likelihood ...")
        mean_lp, lik, n_grams = ngram_extraction_likelihood(
            model, tokenizer, texts, n=10, max_docs=max_docs,
        )
        out[split] = {
            "mem_acc": memacc,
            "mem_acc_n_docs": n_used,
            "ngram10_logp_per_tok": mean_lp,
            "ngram10_extraction_likelihood": lik,
            "ngram10_n_grams": n_grams,
        }
        print(f"    memacc={memacc:.4f}  ngram10 logp/tok={mean_lp:.4f}  lik={lik:.4e}")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["wmdp", "muse"], required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a checkpoint (LoRA adapter dir or full HF model).")
    p.add_argument("--label", default=None,
                   help="Short label for printing/output filename. "
                        "Defaults to <method>__<model_dir>.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write the result JSON. Default: alongside the checkpoint.")
    # WMDP options
    p.add_argument("--lm-eval-batch-size", default="auto",
                   help="lm-eval batch size (int or 'auto', default 'auto').")
    p.add_argument("--mmlu-limit", type=int, default=None,
                   help="Optional per-subject sample cap for MMLU (default: full eval).")
    # MUSE options
    p.add_argument("--muse-corpus", choices=["news", "books"], default=None)
    p.add_argument("--muse-max-docs", type=int, default=100,
                   help="Number of documents used per split for MUSE metrics.")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    label = args.label or f"{ckpt_path.parent.parent.name}__{ckpt_path.name}"
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"behavioral_eval_{args.benchmark}.json"

    print("=" * 72)
    print(f"label     : {label}")
    print(f"checkpoint: {ckpt_path}")
    print(f"benchmark : {args.benchmark}")
    print(f"output    : {out_path}")
    print("=" * 72)

    t0 = time.time()
    payload = {"label": label, "checkpoint": str(ckpt_path), "benchmark": args.benchmark}

    if args.benchmark == "wmdp":
        # lm-eval handles model loading; we don't pre-load.
        bs = args.lm_eval_batch_size
        try:
            bs_arg = int(bs)
        except ValueError:
            bs_arg = bs
        results = run_wmdp_mmlu(str(ckpt_path), bs_arg, args.mmlu_limit)
        payload["results"] = results

    else:  # muse
        if args.muse_corpus is None:
            sys.exit("--muse-corpus required when --benchmark muse")
        model, tokenizer = load_model_and_tokenizer(str(ckpt_path))
        results = run_muse(model, tokenizer, args.muse_corpus, args.muse_max_docs)
        payload["muse_corpus"] = args.muse_corpus
        payload["results"] = results
        del model, tokenizer
        _free()

    payload["elapsed_sec"] = round(time.time() - t0, 1)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}  ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
