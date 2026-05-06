"""Behavioral evaluation for unlearned Llama checkpoints.

Computes MCQ accuracy on wmdp_bio, wmdp_cyber and MMLU (utility control)
using EleutherAI's lm-eval-harness.

The script prints a compact summary to stdout and dumps a JSON next to the
checkpoint at  <ckpt>/behavioral_eval_wmdp.json  so a later analysis script
can aggregate results without re-running anything.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# WMDP / MMLU via lm-evaluation-harness
# --------------------------------------------------------------------------

def run_wmdp_mmlu(model_path: str, batch_size, mmlu_limit):
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
        limit=mmlu_limit,
    )

    out = {}
    for task in tasks:
        if task == "mmlu":
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
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a checkpoint (LoRA adapter dir or full HF model).")
    p.add_argument("--label", default=None,
                   help="Short label for printing/output filename. "
                        "Defaults to <method>__<model_dir>.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write the result JSON. Default: alongside the checkpoint.")
    p.add_argument("--lm-eval-batch-size", default="auto",
                   help="lm-eval batch size (int or 'auto', default 'auto').")
    p.add_argument("--mmlu-limit", type=int, default=None,
                   help="Optional per-subject sample cap for MMLU (default: full eval).")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    label = args.label or f"{ckpt_path.parent.parent.name}__{ckpt_path.name}"
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "behavioral_eval_wmdp.json"

    print("=" * 72)
    print(f"label     : {label}")
    print(f"checkpoint: {ckpt_path}")
    print(f"benchmark : wmdp")
    print(f"output    : {out_path}")
    print("=" * 72)

    t0 = time.time()
    payload = {"label": label, "checkpoint": str(ckpt_path), "benchmark": "wmdp"}

    bs = args.lm_eval_batch_size
    try:
        bs_arg = int(bs)
    except ValueError:
        bs_arg = bs
    payload["results"] = run_wmdp_mmlu(str(ckpt_path), bs_arg, args.mmlu_limit)

    payload["elapsed_sec"] = round(time.time() - t0, 1)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}  ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
