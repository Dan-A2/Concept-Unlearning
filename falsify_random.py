"""RANDOM falsification: norm-matched Gaussian noise injected into the base.

Procedure (per source unlearned checkpoint, e.g. RMU):
  1. Load the LoRA adapter and the base model.
  2. For every parameter touched by the adapter (e.g. q/k/v/o_proj of the
     transformed layers), compute  delta_p = theta_unlearn_p - theta_base_p.
  3. sigma = sqrt(sum_p ||delta_p||_F^2)   — total Frobenius norm of the
     unlearning displacement, restricted to the LoRA-targeted weights.
  4. Sample Gaussian noise of identical shape on those same parameters and
     rescale so the joint Frobenius norm equals sigma.
  5. theta_random_p = theta_base_p + noise_p, save as a *full* HF model under
     checkpoints/random_<method>/<bench_label>/<model_name>/ so test.py
     auto-discovers it (no adapter_config.json => loaded as a full model).

Usage:
  python falsify_random.py \
      --base-model meta-llama/Meta-Llama-3.1-8B \
      --source-checkpoint checkpoints/rmu/wmdp/Llama-3.1-8B \
      --method-tag rmu \
      --bench-label wmdp \
      --model-name Llama-3.1-8B
"""

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent


def _dtype():
    # Always float32 here — bf16 has only ~7 mantissa bits, so when the noise
    # has joint Frobenius norm sigma but is spread over hundreds of matrices,
    # individual additions can fall below the bf16 quantum at the local weight
    # magnitude and silently round to zero. The result is a "random" checkpoint
    # that is bit-identical to base. float32 keeps the additions honest.
    return torch.float32


def _which_params_were_adapted(adapter_cfg_path: Path):
    """Return the set of substrings (target_modules) and the layer indices
    (layers_to_transform, possibly None) that the adapter touched."""
    with open(adapter_cfg_path) as f:
        cfg = json.load(f)
    target_modules = cfg.get("target_modules") or []
    layers = cfg.get("layers_to_transform")  # may be None or a list
    base = cfg["base_model_name_or_path"]
    return base, list(target_modules), layers


def _is_targeted(name: str, target_modules, layers):
    """Decide whether a parameter name was adapted by the LoRA config."""
    if not any(tm in name for tm in target_modules):
        return False
    if layers is None:
        return True
    # Match e.g. 'model.layers.5.self_attn.q_proj.weight' -> '5'
    parts = name.split(".")
    try:
        i = parts.index("layers")
        layer_id = int(parts[i + 1])
    except (ValueError, IndexError):
        return False
    return layer_id in layers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True,
                   help="Hub id or local path of the base model.")
    p.add_argument("--source-checkpoint", required=True,
                   help="LoRA adapter dir (e.g. checkpoints/rmu/wmdp/Llama-3.1-8B).")
    p.add_argument("--method-tag", required=True,
                   help="Short method tag used in the output dir name (e.g. rmu).")
    p.add_argument("--bench-label", required=True,
                   help="Benchmark label, e.g. 'wmdp', 'muse-books', 'tofu-forget10'.")
    p.add_argument("--model-name", required=True,
                   help="Output checkpoint dir name (e.g. Llama-3.1-8B).")
    p.add_argument("--out-root", default=str(PROJECT_ROOT / "checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    src = Path(args.source_checkpoint).resolve()
    adapter_cfg = src / "adapter_config.json"
    if not adapter_cfg.exists():
        raise SystemExit(f"No adapter_config.json in {src} — expected a LoRA checkpoint.")

    base_id, target_modules, layers = _which_params_were_adapted(adapter_cfg)
    if args.base_model and base_id != args.base_model:
        print(f"[note] adapter base = {base_id}, --base-model = {args.base_model} "
              f"(using --base-model)")
        base_id = args.base_model

    if layers is not None:
        print(f"[note] adapter layers_to_transform = {layers}; ignoring and "
              f"applying noise to all layers matching target_modules.")
        layers = None

    print(f"base model        : {base_id}")
    print(f"source checkpoint : {src}")
    print(f"target modules    : {target_modules}")
    print(f"layers            : {layers}  (all layers)")

    dtype = _dtype()

    # 1. Load base model (this becomes the canonical theta_base).
    print("loading base model ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, trust_remote_code=True, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_state = {k: v.detach().clone() for k, v in base.named_parameters()}

    # 2. Build the merged unlearned model on a *separate* base copy so the
    #    in-memory `base` is untouched.
    print("loading merged unlearned model ...")
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, trust_remote_code=True, device_map="auto",
    )
    peft = PeftModel.from_pretrained(base_for_merge, str(src))
    merged = peft.merge_and_unload()
    merged_state = dict(merged.named_parameters())

    # 3. Compute per-parameter delta and total Frobenius norm.
    targeted = [n for n in base_state if _is_targeted(n, target_modules, layers)
                and n in merged_state]
    if not targeted:
        raise RuntimeError("No targeted parameters matched — check target_modules/layers.")
    print(f"targeted params   : {len(targeted)}")

    sum_sq = 0.0
    for n in targeted:
        d = (merged_state[n].float() - base_state[n].float())
        sum_sq += d.pow(2).sum().item()
    sigma = math.sqrt(sum_sq)
    print(f"sigma = ||theta_unlearn - theta_base||_F (LoRA-targeted) = {sigma:.6f}")

    # 4. Free the merged copy now that we have what we need.
    del merged, peft, base_for_merge, merged_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Generate norm-matched Gaussian noise and add to base in place.
    print("sampling norm-matched Gaussian noise ...")
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise_sq = 0.0
    noises = {}
    for n in targeted:
        z = torch.randn(base_state[n].shape, generator=g, dtype=torch.float32)
        noises[n] = z
        noise_sq += z.pow(2).sum().item()
    raw_norm = math.sqrt(noise_sq)
    scale = sigma / raw_norm
    print(f"noise raw norm    : {raw_norm:.6f}   scale = sigma / raw = {scale:.6e}")

    # In-place update of the base model's parameters.
    for n, p in base.named_parameters():
        if n not in noises:
            continue
        p.data.add_(noises[n].to(p.device, dtype=p.dtype) * scale)

    # 6. Save as a full HF model (no adapter_config.json) so test.py loads it
    #    directly as a checkpoint.
    out_dir = Path(args.out_root) / f"random_{args.method_tag}" / args.bench_label / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving full HF model to: {out_dir}")
    base.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Sanity: in-memory Frobenius distance from base should equal sigma.
    new_state = dict(base.named_parameters())
    post_sq = sum(
        (new_state[n].float() - base_state[n].float()).pow(2).sum().item()
        for n in targeted
    )
    print(f"verify (in-mem) ||theta_random - theta_base||_F = {math.sqrt(post_sq):.6f}  (target {sigma:.6f})")

    # Now drop the in-memory copies and re-load from disk, so we measure what
    # was actually persisted (not what was momentarily in RAM).
    del base, new_state
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("re-loading saved checkpoint to verify on-disk values ...")
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(out_dir), dtype=torch.float32, trust_remote_code=True, device_map="auto",
    )
    reloaded_state = dict(reloaded.named_parameters())
    disk_sq = sum(
        (reloaded_state[n].float() - base_state[n].float()).pow(2).sum().item()
        for n in targeted
    )
    disk_dist = math.sqrt(disk_sq)
    rel_err = abs(disk_dist - sigma) / sigma if sigma > 0 else float("inf")
    print(f"verify (on-disk) ||theta_random - theta_base||_F = {disk_dist:.6f}  "
          f"(target {sigma:.6f}, rel.err {rel_err:.2%})")
    if rel_err > 0.01:
        print(f"WARNING: on-disk distance differs from sigma by {rel_err:.2%}. "
              f"Probably a precision issue at save time.")
    del reloaded, reloaded_state

    # Drop a small JSON next to the checkpoint with the falsification metadata.
    with open(out_dir / "random_falsify_info.json", "w") as f:
        json.dump({
            "source_checkpoint": str(src),
            "method_tag": args.method_tag,
            "base_model": base_id,
            "target_modules": target_modules,
            "layers_to_transform": layers,
            "n_targeted_params": len(targeted),
            "sigma_unlearn": sigma,
            "post_save_distance_inmem": math.sqrt(post_sq),
            "post_save_distance_disk": disk_dist,
            "seed": args.seed,
        }, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
