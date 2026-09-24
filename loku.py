"""LoKu — Low-rank Knowledge Unlearning.

Implements the algorithm from:
  "Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs"
  (Choi et al., ICLR 2025)
  https://github.com/csm9493/efficient-llm-unlearning

LoKu combines two components:
  1. IHL (Inverted Hinge Loss) — bounded unlearning loss
  2. FILA (Fisher Information-weighted Low-rank Approximation) — smart LoRA init

Usage:
  # Step 1: Compute Fisher importances (once per model + benchmark)
  python loku.py --mode importance --model_path meta-llama/Meta-Llama-3.1-8B

  # Step 2: Unlearn with FILA-initialized LoRA + IHL
  python loku.py --mode unlearn --model_path meta-llama/Meta-Llama-3.1-8B

  # Or skip FILA (use standard LoRA init + IHL only):
  python loku.py --mode unlearn --model_path meta-llama/Meta-Llama-3.1-8B --skip_fila
"""

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from pathlib import Path

from baselines.utils import load_model, get_data, clear_cuda_cache, save_topic_loss_plot
from baselines.benchmarks import add_benchmark_args, apply_benchmark_config
from baselines.losses import ihl, compute_loss_from_logits, make_labels
from peft import LoraConfig, get_peft_model, TaskType

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    # Model
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "zephyr-7b-beta"

    # Data
    retain_corpora = ["wikitext", "wikitext"]
    forget_corpora = ["bio-forget-corpus", "cyber-forget-corpus"]

    # LoKu hyperparameters
    alpha = [1.0, 1.0]        # retain loss weight per topic
    nu = 0.0                   # noise scale (unused in LoKu, kept for interface compat)
    lr = 1e-4

    batch_size = 4
    max_num_batches = 500

    target_layers = [7]

    # PEFT (LoRA) config
    use_peft = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # FILA config
    fila_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]

    seed = 42


def _get_model_device(model):
    return next(model.parameters()).device


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoKu unlearning.")
    parser.add_argument("--mode", type=str, default="unlearn",
                        choices=["importance", "unlearn"],
                        help="'importance' to compute Fisher importances, 'unlearn' to train.")
    parser.add_argument("--nu", type=float, default=0.0,
                        help="Noise scale (kept for interface compat; not used in LoKu).")
    parser.add_argument("--model_path", type=str, default=Args.model_path,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name for checkpoint dirs.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target module names.")
    parser.add_argument("--skip_fila", action="store_true",
                        help="Skip FILA init; use standard LoRA zero-init instead.")
    parser.add_argument("--importance_batches", type=int, default=200,
                        help="Number of batches for Fisher computation. "
                             "Defaults to 200 (original LoKu repo setting).")
    parser.add_argument("--importance_batch_size", type=int, default=2,
                        help="Per-step batch size used during Fisher computation. "
                             "Defaults to 2 (original LoKu repo setting).")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit NF4 (QLoRA) for memory-constrained models such as Qwen3-32B.")
    add_benchmark_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: Fisher importance computation
# ---------------------------------------------------------------------------

def compute_fisher_importances(model, tokenizer, forget_data_list, retain_data_list, args):
    """Compute empirical Fisher information on forget and retain data.

    Returns a dict with keys: f_cnt, r_cnt, importance_f, importance_r.
    """
    model.eval()
    device = _get_model_device(model)

    target_names = args.fila_target_modules

    # Freeze non-target params so backward doesn't compute their grads.
    # Quantized (e.g. 4-bit) weights are stored as non-float dtypes and can't
    # carry autograd; skip those silently and collect them so the caller can
    # decide what to do (typically: skip FILA init).
    original_requires_grad = {}
    tracked_params = []
    skipped_non_float = 0
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        is_target = any(t in name for t in target_names) and "weight" in name
        if is_target and not param.is_floating_point():
            skipped_non_float += 1
            param.requires_grad_(False)
            continue
        param.requires_grad_(is_target)
        if is_target:
            tracked_params.append((name, param))

    if skipped_non_float > 0:
        print(f"Skipped {skipped_non_float} non-float (likely 4-bit) target weights "
              "for Fisher computation.")

    if not tracked_params:
        # Restore flags and bail out — caller will fall back to skip_fila.
        for name, param in model.named_parameters():
            param.requires_grad_(original_requires_grad.get(name, True))
        return None

    # GPU-resident fp32 accumulators; one .cpu() at the end.
    importance_f = {
        name: torch.zeros_like(param, dtype=torch.float32, device=param.device)
        for name, param in tracked_params
    }
    importance_r = {
        name: torch.zeros_like(param, dtype=torch.float32, device=param.device)
        for name, param in tracked_params
    }

    print(f"Tracking Fisher importance for {len(tracked_params)} weight matrices.")

    truncation_side = tokenizer.truncation_side
    padding_side = tokenizer.padding_side
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    n_topics = len(forget_data_list)
    max_batches_per_topic = min(len(d) for d in forget_data_list)
    if args.importance_batches is not None:
        max_batches_per_topic = min(max_batches_per_topic, args.importance_batches)

    f_cnt = 0
    r_cnt = 0

    for batch_idx in range(max_batches_per_topic):
        for topic_idx in range(n_topics):
            max_length = args.max_lengths[topic_idx]

            # --- Forget ---
            forget_batch = forget_data_list[topic_idx][batch_idx]
            forget_inputs = tokenizer(
                forget_batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)

            labels = make_labels(forget_inputs.input_ids, forget_inputs.attention_mask)
            model.zero_grad(set_to_none=True)
            outputs = model(**forget_inputs)
            loss = compute_loss_from_logits(outputs.logits, labels)
            loss.backward()

            cnt = (labels != -100).sum().item()
            for name, param in tracked_params:
                if param.grad is not None:
                    importance_f[name].add_(param.grad.float().pow_(2), alpha=cnt)
                    param.grad = None
            f_cnt += cnt

            # --- Retain ---
            retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]
            retain_inputs = tokenizer(
                retain_batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)

            labels = make_labels(retain_inputs.input_ids, retain_inputs.attention_mask)
            model.zero_grad(set_to_none=True)
            outputs = model(**retain_inputs)
            loss = compute_loss_from_logits(outputs.logits, labels)
            loss.backward()

            cnt = (labels != -100).sum().item()
            for name, param in tracked_params:
                if param.grad is not None:
                    importance_r[name].add_(param.grad.float().pow_(2), alpha=cnt)
                    param.grad = None
            r_cnt += cnt

    model.zero_grad(set_to_none=True)
    tokenizer.truncation_side = truncation_side
    tokenizer.padding_side = padding_side

    # Move accumulators to CPU once at the end.
    importance_f = {n: t.detach().cpu() for n, t in importance_f.items()}
    importance_r = {n: t.detach().cpu() for n, t in importance_r.items()}

    # Restore original requires_grad flags.
    for name, param in model.named_parameters():
        param.requires_grad_(original_requires_grad.get(name, True))

    return {
        "f_cnt": f_cnt,
        "r_cnt": r_cnt,
        "importance_f": importance_f,
        "importance_r": importance_r,
    }


# ---------------------------------------------------------------------------
# Phase 2: FILA initialization
# ---------------------------------------------------------------------------

def _get_module_by_name(model, name):
    """Navigate dotted name to get a module attribute."""
    parts = name.split(".")
    obj = model
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def apply_fila_init(model, imp, args):
    """Re-initialize LoRA A/B weights using Fisher-weighted SVD (FILA).

    model must already have LoRA adapters applied via PEFT.
    ``imp`` is the dict returned by :func:`compute_fisher_importances`.
    """
    f_cnt = imp["f_cnt"]
    r_cnt = imp["r_cnt"]
    importance_f = imp["importance_f"]
    importance_r = imp["importance_r"]

    # Relative importance: forget / retain
    importances = {
        n: torch.div(importance_f[n] / f_cnt, 1e-5 + importance_r[n] / r_cnt)
        for n in importance_f
    }

    lora_targets = args.fila_target_modules
    r = args.lora_r
    initialized = 0

    for orig_name, importance in importances.items():
        if not any(t in orig_name for t in lora_targets):
            continue

        # Map original parameter name to PEFT module paths.
        # Original: "model.layers.7.self_attn.q_proj.weight"
        # PEFT:     "base_model.model.model.layers.7.self_attn.q_proj.lora_A.default"
        name = orig_name.replace("module.", "")
        base_name = name.replace(".weight", "")
        peft_prefix = "base_model.model." + base_name

        try:
            lora_A = _get_module_by_name(model, peft_prefix + ".lora_A.default")
            lora_B = _get_module_by_name(model, peft_prefix + ".lora_B.default")
            base_layer = _get_module_by_name(model, peft_prefix + ".base_layer")
            scaling_dict = _get_module_by_name(model, peft_prefix + ".scaling")
        except (AttributeError, IndexError):
            continue

        scaling = scaling_dict["default"]
        W = base_layer.weight.data.clone()
        dtype = W.dtype
        W = W.float()

        # Row-wise weighted low-rank approximation
        row_importance = importance.sum(dim=1).sqrt().to(W.device)
        U, S, V = torch.svd_lowrank(row_importance[:, None] * W, q=r)

        S = S / scaling

        new_lora_A = (V * torch.sqrt(S)).t()
        new_lora_B = (1.0 / (row_importance + 1e-5))[:, None] * (U * torch.sqrt(S))
        new_residual = base_layer.weight.data - scaling * (new_lora_B @ new_lora_A)

        lora_A.weight.data = new_lora_A.contiguous().to(dtype)
        lora_B.weight.data = new_lora_B.contiguous().to(dtype)
        base_layer.weight.data = new_residual.contiguous().to(dtype)

        initialized += 1

    print(f"FILA initialized {initialized} LoRA adapter pairs.")
    if initialized == 0:
        print("WARNING: No LoRA adapters were matched for FILA initialization. "
              "Check that --lora_target_modules aligns with the importance file.")


# ---------------------------------------------------------------------------
# Phase 3: Unlearning training loop
# ---------------------------------------------------------------------------

def _apply_lora(updated_model, args):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    updated_model = get_peft_model(updated_model, lora_config)
    updated_model.enable_input_require_grads()
    updated_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    updated_model.print_trainable_parameters()
    return updated_model


def run_loku(
    updated_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
    importances=None,
):
    updated_model = _apply_lora(updated_model, args)

    # FILA initialization (optional)
    if not args.skip_fila:
        if importances is None:
            raise ValueError(
                "FILA init requested but no importances provided. "
                "Pass an importances dict or use --skip_fila."
            )
        apply_fila_init(updated_model, importances, args)

    updated_model = updated_model.train()

    params = [p for p in updated_model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("No trainable parameters found.")

    model_device = _get_model_device(updated_model)
    optimizer = AdamW(params, lr=args.lr)

    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    if args.max_num_batches > max_possible:
        args.max_num_batches = max_possible

    loss_history = {i: {"total": [], "unlearn": [], "retain": []} for i in range(len(forget_data_list))}

    truncation_side = tokenizer.truncation_side
    padding_side = tokenizer.padding_side
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    for idx in range(args.max_num_batches):
        topic_idx = idx % len(forget_data_list)
        batch_idx = idx // len(forget_data_list)

        unlearn_batch = forget_data_list[topic_idx][batch_idx]
        retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]

        max_length = args.max_lengths[topic_idx]

        # Forget loss: IHL
        unlearn_inputs = tokenizer(
            unlearn_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        unlearn_loss = ihl(updated_model=updated_model, unlearn_inputs=unlearn_inputs)

        # Retain loss: standard cross-entropy
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        retain_labels = make_labels(retain_inputs.input_ids, retain_inputs.attention_mask)
        retain_outputs = updated_model(**retain_inputs)
        retain_loss = compute_loss_from_logits(retain_outputs.logits, retain_labels)
        retain_loss *= args.alpha[topic_idx]

        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[topic_idx]["total"].append(loss.item())
        loss_history[topic_idx]["unlearn"].append(unlearn_loss.item())
        loss_history[topic_idx]["retain"].append(retain_loss.item())

        # pbar.update(1)

    tokenizer.truncation_side = truncation_side
    tokenizer.padding_side = padding_side

    fila_tag = "fila" if not args.skip_fila else "nofila"
    path = str(SCRIPT_DIR / f"checkpoints/loku/{args.bench_label}/{fila_tag}-{args.model_name}")

    save_topic_loss_plot(loss_history, path, filename="loss_by_topic.png", topic_names=args.topic_names)
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = Args()
    cli_args = parse_args()
    args.nu = cli_args.nu
    args.model_path = cli_args.model_path
    args.model_name = cli_args.model_name if cli_args.model_name is not None else cli_args.model_path.split("/")[-1]
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules
    args.skip_fila = cli_args.skip_fila
    args.importance_batches = cli_args.importance_batches
    args.importance_batch_size = cli_args.importance_batch_size
    args.load_in_4bit = cli_args.load_in_4bit
    apply_benchmark_config(args, cli_args.benchmark, cli_args.tofu_split, cli_args.muse_corpus)

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if cli_args.mode == "importance":
        # Fisher importances are now computed on-the-fly during unlearn mode
        # to avoid the storage cost of saving them. This mode is kept as a
        # no-op so existing wrapper scripts that call --mode importance first
        # still work.
        print("Note: --mode importance is a no-op; Fisher importances are now "
              "computed in-memory during --mode unlearn (no save).")
        import sys; sys.exit(0)

    elif cli_args.mode == "unlearn":
        # --- FILA init (in-memory Fisher) + IHL training ---
        fila_tag = "fila" if not args.skip_fila else "nofila"
        save_path = SCRIPT_DIR / f"checkpoints/loku/{args.bench_label}/{fila_tag}-{args.model_name}"
        if save_path.exists():
            print(f"Unlearned model already saved at {save_path}; skipping.")
            import sys
            sys.exit(0)

        updated_model, tokenizer = load_model(args.model_path, load_in_4bit=args.load_in_4bit)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Compute Fisher importances in-memory (before LoRA is applied).
        importances = None
        if not args.skip_fila:
            imp_forget_data, imp_retain_data = get_data(
                forget_corpora=args.forget_corpora,
                retain_corpora=args.retain_corpora,
                batch_size=args.importance_batch_size,
                tokenizer=tokenizer,
                chunk_sizes=args.max_lengths,
            )
            importances = compute_fisher_importances(
                updated_model, tokenizer, imp_forget_data, imp_retain_data, args,
            )
            del imp_forget_data, imp_retain_data
            clear_cuda_cache()

            if importances is None:
                print("No float-dtype target weights available for Fisher "
                      "importance (base model is fully quantized). "
                      "Falling back to standard LoRA zero-init (skip_fila).")
                args.skip_fila = True

        forget_data_list, retain_data_list = get_data(
            forget_corpora=args.forget_corpora,
            retain_corpora=args.retain_corpora,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            chunk_sizes=args.max_lengths,
        )

        run_loku(
            updated_model=updated_model,
            tokenizer=tokenizer,
            forget_data_list=forget_data_list,
            retain_data_list=retain_data_list,
            args=args,
            importances=importances,
        )

    clear_cuda_cache()
