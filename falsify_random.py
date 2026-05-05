"""RANDOM falsification: norm-matched noisy *LoRA adapter*, saved as an adapter.

Why this is structured as a noisy LoRA adapter and not as element-wise noise on
the merged weight:

  Earlier versions sampled full-rank Gaussian noise on every LoRA-targeted
  parameter and rescaled to the joint Frobenius norm of (theta_unlearn -
  theta_base). That sounded fair on paper but produced a checkpoint that was
  behaviourally identical to base. Two reasons:
    1. bf16 has ~7 mantissa bits; once full-rank noise is spread over hundreds
       of matrices its per-entry magnitude drops below the local quantum and
       silently rounds to zero on save.
    2. More fundamentally, LoRA only updates a rank-r subspace. The unlearning
       displacement is concentrated in that subspace, so per-direction it is
       O(sqrt(d_out * d_in / r)) larger than full-rank noise of the same total
       Frobenius norm. Spreading the noise full-rank makes each direction
       negligible — the model's behaviour is untouched.

The fix: build a *random* LoRA adapter with the same shape as the unlearning
adapter (same target_modules, same rank, same alpha, same per-layer scaling),
draw lora_A and lora_B as Gaussians, and rescale per layer so that
  ||scaling * B_rand @ A_rand||_F == ||scaling * B @ A||_F
for every (layer, adapter) pair. We then save *just the adapter* (the same
on-disk shape as the source unlearning checkpoint, ~hundreds of MB instead of
~16 GB). test.py auto-detects adapter_config.json and merges at load time.

Usage:
  python falsify_random.py \
      --base-model meta-llama/Meta-Llama-3.1-8B \
      --source-checkpoint checkpoints/rmu/wmdp/Llama-3.1-8B \
      --method-tag rmu \
      --bench-label wmdp \
      --model-name Llama-3.1-8B
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent


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


def _is_lora_layer(module):
    """Heuristic for a peft LoraLayer: has lora_A and lora_B ModuleDicts whose
    entries are nn.Linear-like (with .weight)."""
    if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
        return False
    if not isinstance(module.lora_A, torch.nn.ModuleDict):
        return False
    if not isinstance(module.lora_B, torch.nn.ModuleDict):
        return False
    return True


def _load_adapter_weights_from_disk(adapter_dir: Path):
    """Read the saved adapter tensors back, regardless of safetensors vs .bin."""
    st = adapter_dir / "adapter_model.safetensors"
    bn = adapter_dir / "adapter_model.bin"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    if bn.exists():
        return torch.load(str(bn), map_location="cpu")
    raise FileNotFoundError(
        f"No adapter_model.safetensors or adapter_model.bin in {adapter_dir}"
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    src = Path(args.source_checkpoint).resolve()
    adapter_cfg_path = src / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise SystemExit(f"No adapter_config.json in {src} — expected a LoRA checkpoint.")

    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    base_id = args.base_model or adapter_cfg["base_model_name_or_path"]

    target_modules = adapter_cfg.get("target_modules") or []
    layers = adapter_cfg.get("layers_to_transform")  # may be None or list

    print(f"base model        : {base_id}")
    print(f"source checkpoint : {src}")
    print(f"target modules    : {target_modules}")
    print(f"layers_to_transform: {layers}")

    # Compute everything in float32 — the per-layer rescaling has to be exact,
    # and bf16 gives only ~7 mantissa bits which is not enough to compute
    # ||B_rand @ A_rand||_F reliably across hundreds of layers.
    work_dtype = torch.float32

    # 1. Load base model.
    print("loading base model ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=work_dtype, trust_remote_code=True, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Wrap base with the *unlearning* LoRA adapter so we can read A, B and
    #    scaling for each layer; we will then overwrite them with random ones.
    print("loading LoRA adapter ...")
    peft_model = PeftModel.from_pretrained(base, str(src))
    peft_model.eval()

    # 3. For each LoRA layer (per adapter), draw fresh A_rand, B_rand of the
    #    same shape, rescale so that ||B_rand @ A_rand||_F == ||B @ A||_F
    #    (this also matches the scaled delta norm since the per-adapter scaling
    #    factor cancels). Replace the adapter weights in place.
    print("randomizing LoRA adapter weights (per-layer Frobenius matched) ...")
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    per_layer_info = []
    sum_orig_delta_sq = 0.0
    sum_rand_delta_sq = 0.0

    n_layers_seen = 0
    with torch.no_grad():
        for module_name, module in peft_model.named_modules():
            if not _is_lora_layer(module):
                continue
            for adapter_name in list(module.lora_A.keys()):
                A_layer = module.lora_A[adapter_name]
                B_layer = module.lora_B[adapter_name]
                if not (hasattr(A_layer, "weight") and hasattr(B_layer, "weight")):
                    continue
                A = A_layer.weight  # (r, in_features)
                B = B_layer.weight  # (out_features, r)
                scaling = float(module.scaling[adapter_name])

                A_dev, A_dtype = A.device, A.dtype
                B_dev, B_dtype = B.device, B.dtype

                # Original per-layer effective delta norm.
                orig_BA = (B.detach().to("cpu", dtype=torch.float32)
                           @ A.detach().to("cpu", dtype=torch.float32))
                orig_BA_norm = torch.linalg.norm(orig_BA).item()
                orig_delta_norm = scaling * orig_BA_norm

                # Sample random A, B (CPU fp32 for deterministic generator).
                A_rand = torch.randn(tuple(A.shape), generator=g, dtype=torch.float32)
                B_rand = torch.randn(tuple(B.shape), generator=g, dtype=torch.float32)

                rand_BA = B_rand @ A_rand
                rand_BA_norm = torch.linalg.norm(rand_BA).item()

                if rand_BA_norm > 0 and orig_BA_norm > 0:
                    # Rescale B_rand alone so ||B_rand @ A_rand||_F = ||B @ A||_F.
                    # Keeping A_rand at unit-variance scale leaves the merged
                    # delta numerically well-behaved when test.py merges later.
                    factor = orig_BA_norm / rand_BA_norm
                    B_rand.mul_(factor)
                    final_BA_norm = torch.linalg.norm(B_rand @ A_rand).item()
                else:
                    # Degenerate (orig adapter is ~0 here); leave noise at 0.
                    A_rand.zero_(); B_rand.zero_()
                    final_BA_norm = 0.0

                final_delta_norm = scaling * final_BA_norm

                # Write back to the LoRA layers (preserve their dtype/device).
                A_layer.weight.data.copy_(A_rand.to(device=A_dev, dtype=A_dtype))
                B_layer.weight.data.copy_(B_rand.to(device=B_dev, dtype=B_dtype))

                sum_orig_delta_sq += orig_delta_norm ** 2
                sum_rand_delta_sq += final_delta_norm ** 2
                per_layer_info.append({
                    "module": module_name,
                    "adapter": adapter_name,
                    "shape_A": list(A.shape),
                    "shape_B": list(B.shape),
                    "scaling": scaling,
                    "orig_delta_norm": orig_delta_norm,
                    "rand_delta_norm": final_delta_norm,
                })
                n_layers_seen += 1

    if n_layers_seen == 0:
        raise RuntimeError("No LoRA layers were found on the loaded PEFT model.")
    total_orig = math.sqrt(sum_orig_delta_sq)
    total_rand = math.sqrt(sum_rand_delta_sq)
    print(f"randomized {n_layers_seen} LoRA (layer, adapter) pairs")
    print(f"  total ||delta_unlearn||_F = {total_orig:.6f}")
    print(f"  total ||delta_random ||_F = {total_rand:.6f}")

    # 4. Save the *adapter only* — same on-disk shape as the source checkpoint,
    #    so test.py picks it up via adapter_config.json and merges at load.
    out_dir = Path(args.out_root) / f"random_{args.method_tag}" / args.bench_label / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving noisy LoRA adapter to: {out_dir}")
    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 5. Verify what was actually persisted: read the adapter weights back from
    #    disk and recompute per-layer scaling * ||B_disk @ A_disk||_F. Cheaper
    #    than re-instantiating the base model just to confirm a few hundred MB
    #    of adapter weights round-tripped correctly.
    print("re-reading saved adapter weights to verify on-disk values ...")
    disk_state = _load_adapter_weights_from_disk(out_dir)

    # peft saves keys like 'base_model.model.<module_name>.lora_A.<adapter>.weight'.
    # Group A/B pairs by their (module_name, adapter_name) key.
    pairs = {}
    for k, t in disk_state.items():
        if ".lora_A." in k:
            kind = "A"
        elif ".lora_B." in k:
            kind = "B"
        else:
            continue
        # Strip off optional 'base_model.model.' prefix and '.weight' suffix to
        # get a stable identity for the (module, adapter).
        stem = k
        for pref in ("base_model.model.", "base_model."):
            if stem.startswith(pref):
                stem = stem[len(pref):]; break
        if stem.endswith(".weight"):
            stem = stem[: -len(".weight")]
        parts = stem.split(".")
        # parts: [..., 'lora_A'|'lora_B', adapter_name]
        adapter_name = parts[-1]
        module_name = ".".join(parts[:-2])
        pairs.setdefault((module_name, adapter_name), {})[kind] = t

    # Map (module_name, adapter_name) -> scaling captured during randomization.
    scaling_lookup = {(info["module"], info["adapter"]): info["scaling"]
                      for info in per_layer_info}
    target_lookup = {(info["module"], info["adapter"]): info["rand_delta_norm"]
                     for info in per_layer_info}

    disk_sq = 0.0
    n_checked = 0
    max_per_layer_rel_err = 0.0
    for (module_name, adapter_name), ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].to(torch.float32)
        B = ab["B"].to(torch.float32)
        scaling = scaling_lookup.get((module_name, adapter_name))
        if scaling is None:
            # Try one more lookup form (module_name without 'base_model.model'
            # already stripped above, so usually a hit). If still missing, skip.
            continue
        BA_norm = torch.linalg.norm(B @ A).item()
        delta_norm = scaling * BA_norm
        disk_sq += delta_norm ** 2
        target = target_lookup[(module_name, adapter_name)]
        if target > 0:
            rel = abs(delta_norm - target) / target
            max_per_layer_rel_err = max(max_per_layer_rel_err, rel)
        n_checked += 1

    disk_total = math.sqrt(disk_sq)
    rel_err = abs(disk_total - total_rand) / total_rand if total_rand > 0 else float("inf")
    print(f"verify (on-disk) total ||delta_random||_F = {disk_total:.6f}  "
          f"(target {total_rand:.6f}, rel.err {rel_err:.2%}, layers checked {n_checked})")
    print(f"  max per-layer rel.err = {max_per_layer_rel_err:.2%}")
    if rel_err > 0.02 or max_per_layer_rel_err > 0.05:
        print("WARNING: on-disk adapter norms drift more than expected — check "
              "the dtype peft used for the save.")

    with open(out_dir / "random_falsify_info.json", "w") as f:
        json.dump({
            "approach": "noisy LoRA adapter (low-rank, per-layer Frobenius matched); saved as adapter only",
            "source_checkpoint": str(src),
            "method_tag": args.method_tag,
            "base_model": base_id,
            "target_modules": target_modules,
            "layers_to_transform": layers,
            "n_lora_pairs": n_layers_seen,
            "total_orig_delta_norm": total_orig,
            "total_random_delta_norm": total_rand,
            "post_save_distance_disk": disk_total,
            "max_per_layer_rel_err_disk": max_per_layer_rel_err,
            "seed": args.seed,
            "per_layer": per_layer_info,
        }, f, indent=2)

    # Free GPU before the slurm script's downstream test.py invocation.
    del peft_model, base, disk_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("done.")


if __name__ == "__main__":
    main()
