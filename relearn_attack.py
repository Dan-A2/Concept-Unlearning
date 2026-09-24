"""Relearning (fine-tuning) attack on unlearned checkpoints.

Threat model
------------
An adversary who has the unlearned model and a *small* amount of forget-domain
data fine-tunes the model and checks whether the supposedly-removed knowledge
comes back. This is the standard robustness probe for LLM unlearning
("relearning" / "fine-tuning" attack). If a method only suppresses behaviour
without removing the underlying knowledge, a few gradient steps recover it.

We run a parameter-efficient (LoRA) relearning attack uniformly across all
models so that the 8B/3B models and Qwen3-32B are attacked on equal footing
and the largest model stays tractable.

Pipeline (per checkpoint)
-------------------------
  1. Load the base model and merge the unlearned LoRA adapter -> unlearned model.
  2. (optional) behavioural eval BEFORE relearning  -> baseline accuracy.
  3. Attach a fresh LoRA adapter and fine-tune on a small slice of the
     *forget* corpus (causal LM objective).
  4. Merge the relearned adapter -> relearned model.
  5. behavioural eval AFTER relearning -> recovered accuracy.
  6. Write JSON with before/after accuracies, the recovery delta, and the
     full hyperparameter record.

Default relearning hyperparameters
----------------------------------
Chosen to match the conventions used across the recent robust-unlearning
literature (RMU/WMDP robustness, MUSE, RWKU, Lynch et al. 2024 "Eight Methods
to Evaluate Robust Unlearning", Łucki et al. 2024 "An Adversarial Perspective
on Machine Unlearning", Deeb & Roger 2024 "Do Unlearning Methods Remove
Information from LLM Weights?"):

  * AdamW, lr = 1e-5  (small LR: these works recover forget performance with
    1e-5..5e-5; 1e-5 is the conservative, widely-reported setting),
  * 5 epochs over a small forget slice (a few hundred docs) — relearning
    attacks deliberately use *limited* data and *few* steps,
  * effective batch size 4, linear warmup (3%) then linear decay,
  * LoRA r=16, alpha=32, dropout=0.05 on the attention+MLP projections,
    matching the adapters used during unlearning,
  * a global --max-steps cap so wall-time is bounded.

Tune lr / epochs / samples to sweep attack strength; all are CLI flags.
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from baselines.utils import get_data, clear_cuda_cache, load_merged_model
from baselines.benchmarks import get_benchmark_config
from baselines.losses import make_labels, compute_loss_from_logits
from baselines.eval_metrics import (
    tofu_relearn_texts, muse_relearn_texts, compute_forget_truth_ratios,
)
from behavioral_eval import (
    evaluate_benchmark, write_eval_json, bench_label_of, resolve_tasks,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def attach_relearn_lora(model, args):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()
    return model


def relearn(model, tokenizer, forget_batches, max_lengths, args):
    """Fine-tune `model` (with LoRA attached) on the forget corpus."""
    model.train()
    device = next(model.parameters()).device

    # Flatten the per-topic batched corpus into a flat list of text batches,
    # capped at --relearn-samples documents per topic.
    flat = []
    for topic_idx, topic_batches in enumerate(forget_batches):
        max_len = max_lengths[topic_idx % len(max_lengths)]
        docs_used = 0
        for batch in topic_batches:
            if args.relearn_samples and docs_used >= args.relearn_samples:
                break
            flat.append((batch, max_len))
            docs_used += len(batch)
    if not flat:
        raise RuntimeError("No relearning data after sampling — check corpus.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.relearn_lr)

    steps_per_epoch = len(flat)
    total_steps = steps_per_epoch * args.relearn_epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(args.warmup_ratio * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    print(f"  [relearn] {len(flat)} batches/epoch × {args.relearn_epochs} epochs "
          f"→ {total_steps} steps (warmup {warmup}, lr {args.relearn_lr})")

    step = 0
    for epoch in range(args.relearn_epochs):
        random.shuffle(flat)
        for batch, max_len in flat:
            if step >= total_steps:
                break
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            ).to(device)
            labels = make_labels(enc.input_ids, enc.attention_mask)
            outputs = model(**enc)
            loss = compute_loss_from_logits(outputs.logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 20 == 0 or step == total_steps:
                print(f"    step {step}/{total_steps}  loss={loss.item():.4f}")
        if step >= total_steps:
            break

    model.eval()
    return model


def _flatten_scalars(results: dict) -> dict:
    """Flatten an eval-results dict into {name: float} scalar metrics.

    WMDP results are {task: {"acc": ..}}; TOFU/MUSE results are already flat
    {metric: value}. Non-numeric / bookkeeping keys (lists, None, '_skipped')
    are dropped.
    """
    out = {}
    for k, v in results.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):                 # WMDP per-task block
            if "acc" in v and v["acc"] is not None:
                out[f"{k}_acc"] = float(v["acc"])
        elif isinstance(v, (int, float)) and v is not None:
            out[k] = float(v)
    return out


def _eval(model, tokenizer, args, retain_ref=None):
    return evaluate_benchmark(
        model, tokenizer, args.benchmark,
        tasks=args.tasks, tofu_split=args.tofu_split, muse_corpus=args.muse_corpus,
        n_samples=args.eval_samples, gen_batch_size=args.gen_batch_size,
        lm_eval_batch_size=args.lm_eval_batch_size, mmlu_limit=args.mmlu_limit,
        holdout_frac=args.holdout_frac, holdout_seed=args.holdout_seed,
        retain_ref_truth_ratios=retain_ref,
    )


# Original (pristine) base for each model in the pool — needed to fine-tune a
# retain reference from scratch (the saved 'full' model has its _name_or_path
# stripped, so it can't be recovered from disk).
_ORIG_BASE = {
    "Llama-3.1-8B":   "meta-llama/Meta-Llama-3.1-8B",
    "Llama-3.2-3B":   "meta-llama/Llama-3.2-3B",
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
    "Qwen3-32B":      "Qwen/Qwen3-32B",
}


def _first_checkpoint_base(args):
    """Base-model dir recorded in the first available checkpoint's adapter cfg."""
    for path, _ in args.checkpoint:
        cfg = Path(path).resolve() / "adapter_config.json"
        if cfg.exists():
            with open(cfg) as f:
                base = json.load(f).get("base_model_name_or_path")
            if base:
                return Path(base)
    return None


def _ensure_retain_model(args):
    """Locate the retain reference model at the conventional path, fine-tuning
    and saving it if absent. Returns the Path (or None if it can't be produced).

    Convention: <checkpoints>/tofu_retain/<tofu_split>/<model_short>, where
    model_short is the basename of the unlearned checkpoints' base dir
    (e.g. tofu_finetune/Llama-3.1-8B -> 'Llama-3.1-8B').
    """
    base = _first_checkpoint_base(args)
    if base is None:
        print("  [retain-ref] could not determine base model; skipping.")
        return None
    model_short = base.name
    ckpt_root = base.parent.parent            # .../checkpoints
    retain_path = ckpt_root / "tofu_retain" / args.tofu_split / model_short
    if retain_path.exists():
        return retain_path

    hf_base = _ORIG_BASE.get(model_short)
    if hf_base is None:
        print(f"  [retain-ref] retain model missing and no known original base "
              f"for '{model_short}'; train it via tofu_finetune.py --retain. Skipping.")
        return None

    print(f"  [{time.strftime('%H:%M:%S')}] [retain-ref] retain model not found — "
          f"fine-tuning {model_short} on the retain split ({args.tofu_split}); "
          f"this is a one-time cost, then reused.")
    import subprocess
    cmd = [sys.executable, str(SCRIPT_DIR / "tofu_finetune.py"),
           "--model_path", hf_base, "--model_name", model_short,
           "--retain", "--tofu_split", args.tofu_split]
    subprocess.run(cmd, check=True)
    return retain_path if retain_path.exists() else None


def compute_retain_reference(args):
    """Return the retain model's forget truth-ratio list for TOFU Forget Quality
    (or None). With --retain-ref, locates (and fine-tunes if absent) the retain
    reference, then computes its truth ratios on the SAME held-out forget split
    as the eval, caching them next to the model so later runs reuse them.
    """
    if args.benchmark != "tofu" or not args.retain_ref:
        return None
    rp = _ensure_retain_model(args)
    if rp is None:
        return None

    cache = rp / (f"forget_tr_{args.tofu_split}_h{args.holdout_frac}"
                  f"_s{args.holdout_seed}_n{args.eval_samples}.json")
    if cache.exists():
        with open(cache) as f:
            trs = json.load(f)
        print(f"  [retain-ref] loaded cached reference truth ratios ({len(trs)}).")
        return trs

    print(f"  [{time.strftime('%H:%M:%S')}] [retain-ref] computing reference "
          f"truth ratios from {rp} ...")
    model, tokenizer, _ = load_merged_model(str(rp), None, args.load_in_4bit)
    tokenizer.padding_side = "right"
    trs = compute_forget_truth_ratios(
        model, tokenizer, args.tofu_split, args.eval_samples,
        holdout_frac=args.holdout_frac, holdout_seed=args.holdout_seed,
    )
    del model, tokenizer
    clear_cuda_cache()
    try:
        with open(cache, "w") as f:
            json.dump(trs, f)
        print(f"  [retain-ref] cached -> {cache}")
    except OSError:
        pass
    return trs


def _build_relearn_batches(texts, batch_size):
    """Pack a flat list of texts into a single-topic forget_batches structure."""
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    return [batches]   # one topic


def run_one(adapter_path, label, args, retain_ref=None):
    """Full attack + eval for a single checkpoint. Returns the payload dict."""
    print("\n" + "=" * 78)
    print(f"[{time.strftime('%H:%M:%S')}] RELEARN ATTACK: {label}")
    print(f"  checkpoint: {adapter_path}")
    print("=" * 78)

    payload = {
        "label": label,
        "checkpoint": str(adapter_path),
        "benchmark": args.benchmark,
        "relearn": {
            "lr": args.relearn_lr,
            "epochs": args.relearn_epochs,
            "samples_per_topic": args.relearn_samples,
            "max_steps": args.max_steps,
            "batch_size": args.relearn_batch_size,
            "warmup_ratio": args.warmup_ratio,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "load_in_4bit": args.load_in_4bit,
            "seed": args.seed,
            "holdout_frac": args.holdout_frac if args.benchmark in ("tofu", "muse") else 0.0,
            "holdout_seed": args.holdout_seed,
        },
    }

    # --- 1. Load unlearned model (merge adapter into base) -----------------
    model, tokenizer, base_name = load_merged_model(
        adapter_path, args.base_model, args.load_in_4bit
    )
    tokenizer.padding_side = "right"
    payload["base_model"] = base_name

    # --- 2. (optional) eval BEFORE relearning ------------------------------
    # This "before" eval IS the behavioural eval of the unlearned checkpoint,
    # so we also write it next to the checkpoint in the exact format
    # behavioral_eval.py produces — running relearn_attack thus subsumes a
    # separate behavioral_eval run for the unlearned model.
    if args.eval_before:
        print(f"  [{time.strftime('%H:%M:%S')}] eval BEFORE relearning ...")
        t_before = time.time()
        payload["before"] = _eval(model, tokenizer, args, retain_ref=retain_ref)
        blabel = bench_label_of(args.benchmark, args.tofu_split, args.muse_corpus)
        tasks = resolve_tasks("wmdp", args.tasks) if args.benchmark == "wmdp" else None
        beh_path = write_eval_json(
            adapter_path, label, blabel, payload["before"],
            tasks=tasks, base_model=base_name,
            elapsed_sec=round(time.time() - t_before, 1),
        )
        print(f"  [{time.strftime('%H:%M:%S')}] wrote behavioral eval -> {beh_path}")

    # --- 3. Load forget data + relearn -------------------------------------
    # For TOFU/MUSE, train on the RELEARN half of the forget set (disjoint from
    # the held-out half the recovery is measured on). WMDP needs no split: it
    # relearns on the forget corpus and is evaluated on separate MCQs.
    cfg = get_benchmark_config(
        args.benchmark, args.tofu_split, args.muse_corpus
    )
    if args.benchmark == "tofu":
        texts = tofu_relearn_texts(args.tofu_split, args.holdout_frac,
                                   args.holdout_seed)
        forget_batches = _build_relearn_batches(texts, args.relearn_batch_size)
    elif args.benchmark == "muse":
        texts = muse_relearn_texts(args.muse_corpus, args.holdout_frac,
                                   args.holdout_seed)
        forget_batches = _build_relearn_batches(texts, args.relearn_batch_size)
    else:  # wmdp (and any corpus-based benchmark) — relearn on the full corpus
        forget_batches, _ = get_data(
            forget_corpora=cfg["forget_corpora"],
            retain_corpora=cfg["retain_corpora"],
            batch_size=args.relearn_batch_size,
            tokenizer=tokenizer,
            chunk_sizes=cfg["max_lengths"],
        )

    model = attach_relearn_lora(model, args)
    t0 = time.time()
    model = relearn(model, tokenizer, forget_batches, cfg["max_lengths"], args)
    payload["relearn"]["elapsed_sec"] = round(time.time() - t0, 1)

    # --- 4. Merge relearned adapter (4-bit can't merge cleanly -> in place) -
    eval_model = model if args.load_in_4bit else model.merge_and_unload()
    tokenizer.padding_side = "right"

    # --- 5. eval AFTER relearning ------------------------------------------
    print(f"  [{time.strftime('%H:%M:%S')}] eval AFTER relearning ...")
    payload["after"] = _eval(eval_model, tokenizer, args, retain_ref=retain_ref)

    # --- 6. Recovery deltas over common scalar metrics ---------------------
    if "before" in payload:
        before = _flatten_scalars(payload["before"])
        after = _flatten_scalars(payload["after"])
        payload["recovery"] = {
            m: {"before": before[m], "after": after[m],
                "delta": after[m] - before[m]}
            for m in before if m in after
        }

    del model, eval_model, tokenizer
    clear_cuda_cache()
    return payload


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Relearning (fine-tuning) attack + behavioural eval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # What to attack
    p.add_argument("--checkpoint", nargs=2, metavar=("PATH", "LABEL"),
                   action="append", default=[],
                   help="Adapter dir + label. Repeat for several checkpoints.")
    p.add_argument("--base-model", default=None,
                   help="Override base model (default: read from adapter_config).")
    p.add_argument("--retain-ref", action="store_true",
                   help="TOFU only: enable Forget Quality. Looks for the retain "
                        "reference at checkpoints/tofu_retain/<split>/<model>; if "
                        "absent, fine-tunes and saves it (one-time), then uses it. "
                        "The reference truth-ratios are cached next to the model.")
    # Benchmark / tasks
    p.add_argument("--benchmark", default="wmdp",
                   choices=["wmdp", "tofu", "muse"])
    p.add_argument("--tofu-split", default=None,
                   choices=["forget01", "forget05", "forget10"])
    p.add_argument("--muse-corpus", default=None, choices=["news", "books"])
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Override lm-eval task list (WMDP only).")
    # Relearn hyperparameters (literature-grounded defaults)
    p.add_argument("--relearn-lr", type=float, default=1e-5)
    p.add_argument("--relearn-epochs", type=int, default=5)
    p.add_argument("--relearn-samples", type=int, default=50,
                   help="Max forget docs per topic used for relearning "
                        "(0 = use all available). Models the limited-data attacker.")
    p.add_argument("--relearn-batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=500,
                   help="Global cap on relearning optimizer steps (0 = no cap).")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load base in 4-bit NF4 (for Qwen3-32B).")
    # Eval
    p.add_argument("--eval-before", action="store_true",
                   help="Also score the model before relearning (for recovery delta).")
    p.add_argument("--holdout-frac", type=float, default=0.5,
                   help="TOFU/MUSE only: fraction of the forget set held out for "
                        "recovery eval; relearning trains on the disjoint "
                        "complement. 0 disables the split. WMDP ignores this "
                        "(its MCQ eval never overlaps the relearn corpus).")
    p.add_argument("--holdout-seed", type=int, default=1234,
                   help="Seed for the deterministic forget relearn/held-out split.")
    p.add_argument("--eval-samples", type=int, default=0,
                   help="Cap eval examples per split for TOFU/MUSE (0 = all).")
    p.add_argument("--gen-batch-size", type=int, default=8,
                   help="Generation batch size for TOFU/MUSE ROUGE metrics.")
    p.add_argument("--lm-eval-batch-size", default="auto")
    p.add_argument("--mmlu-limit", type=int, default=None,
                   help="Per-subject sample cap for the WMDP eval (default: full).")
    # IO
    p.add_argument("--out-dir", default=str(SCRIPT_DIR / "Results" / "relearn"),
                   help="Directory for the per-checkpoint result JSONs.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    _set_seed(args.seed)

    if not args.checkpoint:
        raise SystemExit("Provide at least one --checkpoint PATH LABEL.")

    bench_cfg = get_benchmark_config(
        args.benchmark, args.tofu_split, args.muse_corpus
    )
    bench_label = bench_cfg["bench_label"]

    out_dir = Path(args.out_dir) / bench_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Retain reference for TOFU Forget Quality — computed once (all checkpoints
    # in this call share the same base model / retain reference).
    retain_ref = compute_retain_reference(args)

    for path, label in args.checkpoint:
        adapter_path = Path(path).resolve()
        if not adapter_path.exists():
            print(f"[skip] checkpoint not found: {adapter_path}")
            continue

        # Skip prompt-based (e.g. SPUL / P-tuning) checkpoints: their base
        # weights are frozen during unlearning, so a weight-fine-tuning relearn
        # attack isn't a clean, comparable measurement (the soft prompt can't be
        # carried through LoRA relearning). See discussion in the repo notes.
        cfg_file = adapter_path / "adapter_config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                peft_type = str(json.load(f).get("peft_type", "LORA")).upper()
            if peft_type not in ("LORA", "LOHA", "LOKR", "ADALORA", "IA3", "VERA", "BONE"):
                print(f"[skip] {label}: prompt-based adapter ({peft_type}) not "
                      f"supported for relearning attack.")
                continue

        safe = label.replace("/", "__")
        out_path = out_dir / f"relearn__{safe}.json"
        if out_path.exists():
            print(f"[skip] already done: {out_path}")
            continue

        payload = run_one(adapter_path, label, args, retain_ref=retain_ref)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {out_path}")

        # Compact stdout summary over scalar metrics.
        print(f"\n  --- {label} ({bench_label}) ---")
        if "recovery" in payload:
            for m, rec in payload["recovery"].items():
                print(f"    {m:26s}: {rec['before']:.4f} -> {rec['after']:.4f}  "
                      f"(Δ {rec['delta']:+.4f})")
        else:
            for m, v in _flatten_scalars(payload["after"]).items():
                print(f"    {m:26s}: {v:.4f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] All relearning attacks done.")


if __name__ == "__main__":
    main()
