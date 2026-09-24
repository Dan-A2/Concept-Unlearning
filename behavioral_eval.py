"""Behavioural evaluation for unlearned / relearned checkpoints, all benchmarks.

Dispatch by benchmark:
  * wmdp — MCQ accuracy via EleutherAI lm-evaluation-harness
           (wmdp_bio, wmdp_cyber, + MMLU utility control).
  * tofu — Forget Q_A Prob / ROUGE, forget Truth Ratio, Model Utility, and
           (optionally) Forget Quality, via baselines.eval_metrics.eval_tofu.
  * muse — VerbMem + KnowMem ROUGE, via baselines.eval_metrics.eval_muse.

TOFU / MUSE metrics are implemented in-house (see baselines/eval_metrics.py)
so we don't depend on an external framework. They need the model in memory, so
the standalone CLI merges LoRA adapters before scoring.

Two entry points:
  * run_lm_eval(...)        — WMDP via lm-eval (path or in-memory model).
  * evaluate_benchmark(...) — unified dispatch on an *in-memory* model object,
                              used by relearn_attack.py.

Standalone:
    python behavioral_eval.py --checkpoint <ckpt> --benchmark wmdp
    python behavioral_eval.py --checkpoint <ckpt> --benchmark muse --muse-corpus news
    python behavioral_eval.py --checkpoint <ckpt> --benchmark tofu --tofu-split forget10
writes <ckpt>/behavioral_eval_<bench_label>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

BENCHMARK_TASKS = {
    "wmdp": ["wmdp_bio", "wmdp_cyber", "mmlu"],
}


def resolve_tasks(benchmark: str, override=None):
    if override:
        return list(override)
    tasks = BENCHMARK_TASKS.get(benchmark)
    if not tasks:
        raise SystemExit(
            f"No default lm-eval tasks for benchmark '{benchmark}'. "
            f"Pass --tasks explicitly (known: {sorted(BENCHMARK_TASKS)})."
        )
    return tasks


def _extract_acc(results: dict, tasks) -> dict:
    out = {}
    res = results.get("results", {})
    for task in tasks:
        r = res.get(task, {}) or {}
        out[task] = {
            "acc": r.get("acc,none", r.get("acc", float("nan"))),
            "acc_stderr": r.get("acc_stderr,none", r.get("acc_stderr", float("nan"))),
        }
    return out


def run_lm_eval(model, tokenizer=None, tasks=None, batch_size="auto",
                limit=None, load_in_4bit=False, log_samples=False):
    """Score `model` on lm-eval `tasks`. `model` is a path or model object.

    `limit` (per-subject sample cap) is applied ONLY to MMLU tasks — the
    utility control that dominates runtime. WMDP tasks (the forgetting signal)
    are always run in full, so subsampling MMLU never silently truncates them.
    The model is instantiated once and reused across both task groups.

    With `log_samples=True`, per-question records are collected and returned
    under the "_samples" key of the result dict ({task: [per-doc records]}),
    so you can report exactly which questions each model gets right/wrong.
    """
    if tasks is None:
        raise ValueError("tasks must be provided")
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise SystemExit("lm-eval not installed: pip install lm-eval==0.4.5") from e

    # Build the LM once (base + adapter, or an in-memory model object).
    if isinstance(model, str):
        adapter_cfg = Path(model) / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg) as f:
                base = json.load(f)["base_model_name_or_path"]
            lm = HFLM(pretrained=base, peft=model, dtype="bfloat16",
                      batch_size=batch_size, load_in_4bit=load_in_4bit)
            print(f"  lm-eval tasks: {tasks}  (base={base}, peft={model})")
        else:
            lm = HFLM(pretrained=model, dtype="bfloat16",
                      batch_size=batch_size, load_in_4bit=load_in_4bit)
            print(f"  lm-eval tasks: {tasks}  (model={model})")
    else:
        bs = batch_size
        # lm-eval's batch_size='auto' runs a synthetic max-length forward to
        # size batches; that probe crashes on prompt-tuning models wrapped for
        # virtual-token stripping (e.g. SPUL) with a CUDA launch failure. Force
        # a fixed batch size for those so real (short) MCQ scoring proceeds.
        from baselines.utils import _StripVtWrapper
        if isinstance(model, _StripVtWrapper) and (bs == "auto" or bs is None):
            bs = 8
            print("  [note] prompt-tuning model: using fixed lm-eval "
                  "batch_size=8 (auto-detection is unstable for wrapped "
                  "P-tuning models).")
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)
        print(f"  lm-eval tasks: {tasks}  (in-memory model)")

    # Split so `limit` caps only MMLU; everything else runs in full.
    mmlu_tasks = [t for t in tasks if "mmlu" in t]
    full_tasks = [t for t in tasks if "mmlu" not in t]

    merged = {"results": {}, "samples": {}}
    if full_tasks:
        r = simple_evaluate(model=lm, tasks=full_tasks, verbosity="ERROR",
                            log_samples=log_samples)
        merged["results"].update(r.get("results", {}))
        if log_samples:
            merged["samples"].update(r.get("samples", {}))
    if mmlu_tasks:
        r = simple_evaluate(model=lm, tasks=mmlu_tasks, limit=limit,
                            verbosity="ERROR", log_samples=log_samples)
        merged["results"].update(r.get("results", {}))
        if log_samples:
            merged["samples"].update(r.get("samples", {}))

    out = _extract_acc(merged, tasks)

    # Persist every per-subject MMLU accuracy (all 57), so the behavioural
    # per-subject table can be re-aggregated later at zero GPU cost. lm-eval
    # scored each subject under keys 'mmlu_<subject>'. No grouping here.
    subjects = {}
    for k, v in merged.get("results", {}).items():
        if k.startswith("mmlu_") and isinstance(v, dict):
            acc = v.get("acc,none", v.get("acc"))
            if acc is not None:
                subjects[k[len("mmlu_"):]] = float(acc)
    if subjects:
        out["_mmlu_subjects"] = subjects

    if log_samples:
        out["_samples"] = {t: _distill_samples(recs)
                           for t, recs in merged.get("samples", {}).items()}
    return out


def _distill_samples(records):
    """Reduce lm-eval per-doc sample records to compact per-question rows:
    doc_id, correctness (acc / acc_norm), the gold target, and the raw doc
    (question + choices + answer) so specific questions can be cited later."""
    out = []
    for i, rec in enumerate(records):
        item = {"doc_id": rec.get("doc_id", i)}
        for k in ("acc", "acc_norm"):
            if k in rec and rec[k] is not None:
                item[k] = float(rec[k])
        if "target" in rec:
            item["target"] = rec["target"]
        if "doc" in rec:
            item["doc"] = rec["doc"]
        out.append(item)
    return out


def evaluate_benchmark(model, tokenizer, benchmark, *, tasks=None,
                       tofu_split=None, muse_corpus=None,
                       n_samples=0, gen_batch_size=8, lm_eval_batch_size="auto",
                       mmlu_limit=None, holdout_frac=0.0, holdout_seed=1234,
                       retain_ref_truth_ratios=None):
    """Unified eval dispatch on an in-memory model. Returns a flat metrics dict.

    Used by relearn_attack.py (before/after) and the standalone CLI for
    TOFU/MUSE. WMDP routes through lm-eval; TOFU/MUSE through eval_metrics.

    holdout_frac>0 (TOFU/MUSE only) restricts the overlapping forget metrics to
    a held-out split disjoint from the relearn data — see eval_metrics. WMDP is
    unaffected (its MCQ eval never overlaps the relearn corpus).
    """
    if benchmark == "wmdp":
        tasks = resolve_tasks("wmdp", tasks)
        return run_lm_eval(model, tokenizer=tokenizer, tasks=tasks,
                           batch_size=lm_eval_batch_size, limit=mmlu_limit)
    if benchmark == "tofu":
        from baselines.eval_metrics import eval_tofu
        return eval_tofu(model, tokenizer, tofu_split=tofu_split,
                         n_samples=n_samples, gen_batch_size=gen_batch_size,
                         holdout_frac=holdout_frac, holdout_seed=holdout_seed,
                         retain_ref_truth_ratios=retain_ref_truth_ratios)
    if benchmark == "muse":
        from baselines.eval_metrics import eval_muse
        return eval_muse(model, tokenizer, corpus=muse_corpus,
                         n_verbmem=n_samples or 100, n_knowmem=n_samples or 100,
                         gen_batch_size=gen_batch_size,
                         holdout_frac=holdout_frac, holdout_seed=holdout_seed)
    raise ValueError(f"Unsupported benchmark for behavioural eval: {benchmark}")


def bench_label_of(benchmark, tofu_split=None, muse_corpus=None, blur_task=None):
    """Canonical benchmark label used in output filenames / checkpoint dirs."""
    if benchmark == "tofu":
        return f"tofu-{tofu_split}"
    if benchmark == "muse":
        return f"muse-{muse_corpus}"
    if benchmark == "blur":
        return f"blur-{blur_task}"
    return benchmark


def _bench_label(args):
    return bench_label_of(args.benchmark, getattr(args, "tofu_split", None),
                          getattr(args, "muse_corpus", None),
                          getattr(args, "blur_task", None))


def write_eval_json(ckpt_path, label, benchmark_label, results, *,
                    tasks=None, base_model=None, elapsed_sec=None, out_dir=None):
    """Write a checkpoint's behavioural-eval result to
    ``<out_dir or ckpt>/behavioral_eval_<benchmark_label>.json`` — the same
    file (and schema) that behavioral_eval.py's CLI produces, so relearn_attack
    can emit it for the unlearned model without a separate behavioral_eval run.
    Returns the output path.
    """
    out_dir = Path(out_dir) if out_dir else Path(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"behavioral_eval_{benchmark_label}.json"
    payload = {"label": label, "checkpoint": str(ckpt_path),
               "benchmark": benchmark_label, "results": results}
    if tasks is not None:
        payload["tasks"] = tasks
    if base_model is not None:
        payload["base_model"] = base_model
    if elapsed_sec is not None:
        payload["elapsed_sec"] = elapsed_sec
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="LoRA adapter dir or full HF model.")
    p.add_argument("--benchmark", default="wmdp", choices=["wmdp", "tofu", "muse"])
    p.add_argument("--tofu-split", default=None,
                   choices=["forget01", "forget05", "forget10"])
    p.add_argument("--muse-corpus", default=None, choices=["news", "books"])
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Override lm-eval tasks (wmdp only).")
    p.add_argument("--base-model", default=None,
                   help="Override base model (default: read from adapter_config).")
    p.add_argument("--label", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-samples", type=int, default=0,
                   help="Cap eval examples per split for TOFU/MUSE (0 = all).")
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--lm-eval-batch-size", default="auto")
    p.add_argument("--mmlu-limit", type=int, default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--log-samples", action="store_true",
                   help="WMDP: also save per-question records (right/wrong + the "
                        "question) to behavioral_samples_<bench>.json. Routes "
                        "through the in-memory loader so P-tuning (SPUL) is scored "
                        "correctly (virtual-token positions stripped).")
    return p.parse_args()


def main():
    args = parse_args()
    _local = Path(args.checkpoint)
    if _local.exists():
        ckpt_path = _local.resolve()
    else:
        # Not a local path -> treat as a HuggingFace hub id (e.g. the original
        # WMDP base model). Keep the raw string (do NOT resolve, which would turn
        # it into a bogus absolute path) and require an explicit --out-dir since
        # there's no checkpoint dir to write next to.
        if not args.out_dir:
            sys.exit(f"'{args.checkpoint}' is not a local path; if it's a HF hub "
                     f"id, pass --out-dir for the output.")
        ckpt_path = Path(args.checkpoint)

    bench_label = _bench_label(args)
    label = args.label or f"{ckpt_path.parent.parent.name}__{ckpt_path.name}"
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"behavioral_eval_{bench_label}.json"

    print("=" * 72)
    print(f"label     : {label}")
    print(f"checkpoint: {ckpt_path}")
    print(f"benchmark : {bench_label}")
    print(f"output    : {out_path}")
    print("=" * 72)

    bs = args.lm_eval_batch_size
    try:
        bs = int(bs)
    except ValueError:
        pass

    t0 = time.time()
    payload = {"label": label, "checkpoint": str(ckpt_path), "benchmark": bench_label}

    if args.benchmark == "wmdp":
        tasks = resolve_tasks("wmdp", args.tasks)
        payload["tasks"] = tasks
        if args.log_samples:
            # In-memory load so per-question logging works AND P-tuning (SPUL)
            # is scored correctly (lm-eval's peft= path would misalign it).
            from baselines.utils import load_merged_model, clear_cuda_cache
            model, tokenizer, base = load_merged_model(
                str(ckpt_path), args.base_model, args.load_in_4bit)
            payload["base_model"] = base
            payload["results"] = run_lm_eval(
                model, tokenizer=tokenizer, tasks=tasks, batch_size=bs,
                limit=args.mmlu_limit, log_samples=True,
            )
            samples = payload["results"].pop("_samples", {})
            spath = out_dir / f"behavioral_samples_{bench_label}.json"
            with open(spath, "w") as f:
                json.dump({"label": label, "checkpoint": str(ckpt_path),
                           "benchmark": bench_label, "samples": samples}, f, indent=2)
            n_q = sum(len(v) for v in samples.values())
            print(f"  wrote per-question samples -> {spath}  ({n_q} questions)")
            del model, tokenizer
            clear_cuda_cache()
        else:
            # lm-eval handles the (peft) checkpoint by path — no manual merge.
            payload["results"] = run_lm_eval(
                str(ckpt_path), tasks=tasks, batch_size=bs,
                limit=args.mmlu_limit, load_in_4bit=args.load_in_4bit,
            )
    else:
        # TOFU / MUSE need the model in memory -> merge adapter first.
        from baselines.utils import load_merged_model, clear_cuda_cache
        model, tokenizer, base = load_merged_model(
            str(ckpt_path), args.base_model, args.load_in_4bit)
        payload["base_model"] = base
        payload["results"] = evaluate_benchmark(
            model, tokenizer, args.benchmark,
            tofu_split=args.tofu_split, muse_corpus=args.muse_corpus,
            n_samples=args.n_samples, gen_batch_size=args.gen_batch_size,
        )
        del model, tokenizer
        clear_cuda_cache()

    payload["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}  ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
