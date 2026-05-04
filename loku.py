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

from baselines.utils import load_model, get_params, get_data, clear_cuda_cache, save_topic_loss_plot
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
    layer_ids = [5, 6, 7]
    param_ids = [6]

    # PEFT (LoRA) config
    use_peft = True
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

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
    parser.add_argument("--importance_batches", type=int, default=None,
                        help="Number of batches for Fisher computation. "
                             "Defaults to all available batches.")
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

    importance_f = {}
    importance_r = {}
    for name, param in model.named_parameters():
        if any(t in name for t in target_names) and "weight" in name:
            importance_f[name] = torch.zeros_like(param, dtype=torch.float32, device="cpu")
            importance_r[name] = torch.zeros_like(param, dtype=torch.float32, device="cpu")

    print(f"Tracking Fisher importance for {len(importance_f)} weight matrices.")

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

    total = max_batches_per_topic * n_topics
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
            for name, param in model.named_parameters():
                if name in importance_f and param.grad is not None:
                    importance_f[name] += (param.grad.pow(2).float() * cnt).detach().cpu()
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
            for name, param in model.named_parameters():
                if name in importance_r and param.grad is not None:
                    importance_r[name] += (param.grad.pow(2).float() * cnt).detach().cpu()
                    param.grad = None
            r_cnt += cnt

            # pbar.update(1)

    model.zero_grad(set_to_none=True)
    tokenizer.truncation_side = truncation_side
    tokenizer.padding_side = padding_side

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


def apply_fila_init(model, importance_file, args):
    """Re-initialize LoRA A/B weights using Fisher-weighted SVD (FILA).

    model must already have LoRA adapters applied via PEFT.
    """
    imp = torch.load(importance_file, map_location="cpu", weights_only=False)
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
        layers_to_transform=args.layer_ids,
    )
    updated_model = get_peft_model(updated_model, lora_config)
    updated_model.enable_input_require_grads()
    updated_model.print_trainable_parameters()
    return updated_model


def run_loku(
    updated_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    updated_model = _apply_lora(updated_model, args)

    # FILA initialization (optional)
    if not args.skip_fila:
        imp_path = SCRIPT_DIR / "importances" / f"{args.model_name}_{args.bench_label}.pt"
        if not imp_path.exists():
            raise FileNotFoundError(
                f"Importance file not found: {imp_path}\n"
                f"Run with --mode importance first, or use --skip_fila."
            )
        apply_fila_init(updated_model, str(imp_path), args)

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
    apply_benchmark_config(args, cli_args.benchmark, cli_args.tofu_split, cli_args.muse_corpus, cli_args.blur_task)

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if cli_args.mode == "importance":
        # --- Phase 1: Compute Fisher importances ---
        save_dir = SCRIPT_DIR / "importances"
        save_path = save_dir / f"{args.model_name}_{args.bench_label}.pt"

        if save_path.exists():
            print(f"Fisher importances already exist at {save_path}; loading instead of recomputing.")
        else:
            model, tokenizer = load_model(args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            forget_data_list, retain_data_list = get_data(
                forget_corpora=args.forget_corpora,
                retain_corpora=args.retain_corpora,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                chunk_sizes=args.max_lengths,
            )

            importances = compute_fisher_importances(
                model, tokenizer, forget_data_list, retain_data_list, args,
            )

            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(importances, save_path)
            print(f"Saved Fisher importances to {save_path}")

    elif cli_args.mode == "unlearn":
        # --- Phase 2+3: FILA init + IHL training ---
        updated_model, tokenizer = load_model(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        )

    clear_cuda_cache()
