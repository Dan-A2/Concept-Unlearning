"""Obliviate — Vocabulary-Masking Unlearning.

Implements the algorithm from:
  Choi et al., "Obliviate: Neutralizing Task-agnostic Backdoors within
  the Activation Space" / WMDP vocabulary-masking variant.

Three loss components:
  1. Vocabulary-masking KL (forget data) — zero out sensitive-token logits,
     KL between the full and masked distributions.
  2. MSE distillation (retain data) — match student logits to frozen
     teacher logits.
  3. CE retain (retain data) — cross-entropy against teacher argmax targets.

Total loss = kl_loss + Lambda_1 * distill_loss + Lambda_2 * retain_loss

Usage:
  python obliviate.py --model_path HuggingFaceH4/zephyr-7b-beta
"""

import numpy as np
import torch
import argparse
from torch.optim import AdamW
from pathlib import Path

from baselines.utils import (
    load_model, get_params, get_data, clear_cuda_cache,
    save_topic_loss_plot, get_sensitive_token_ids,
)
from baselines.benchmarks import add_benchmark_args, apply_benchmark_config
from baselines.losses import obliviate_vocab_kl, obliviate_distill_mse, obliviate_retain_ce
from peft import LoraConfig, get_peft_model, TaskType

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    # Model
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "zephyr-7b-beta"

    # Data
    retain_corpora = ["wikitext", "wikitext"]
    forget_corpora = ["bio-forget-corpus", "cyber-forget-corpus"]

    # Obliviate hyperparameters
    Lambda_1 = 0.2         # distillation loss weight
    Lambda_2 = 0.7         # retain CE loss weight
    lr = 1e-5

    batch_size = 4
    max_num_batches = 500

    layer_ids = [5, 6, 7]
    param_ids = [6]

    # PEFT (LoRA) config
    use_peft = True
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    seed = 42


def _get_model_device(model):
    return next(model.parameters()).device


def _apply_lora_if_enabled(updated_model, args):
    if not args.use_peft:
        return updated_model

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Obliviate vocabulary-masking unlearning.")
    parser.add_argument("--model_path", type=str, default=Args.model_path,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name for checkpoint dirs.")
    parser.add_argument("--Lambda_1", type=float, default=Args.Lambda_1,
                        help="Distillation MSE loss weight.")
    parser.add_argument("--Lambda_2", type=float, default=Args.Lambda_2,
                        help="Retain CE loss weight.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target module names.")
    add_benchmark_args(parser)
    return parser.parse_args()


def run_obliviate(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    sensitive_token_ids,
    args,
):
    updated_model = _apply_lora_if_enabled(updated_model, args)
    updated_model.train()

    if args.use_peft:
        params = [p for p in updated_model.parameters() if p.requires_grad]
    else:
        params = get_params(updated_model, args.layer_ids, args.param_ids)

    if len(params) == 0:
        raise ValueError("No trainable parameters found.")

    model_device = _get_model_device(updated_model)
    optimizer = AdamW(params, lr=args.lr)

    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    if args.max_num_batches > max_possible:
        args.max_num_batches = max_possible
    loss_history = {i: {"total": [], "unlearn": [], "retain": []}
                    for i in range(len(forget_data_list))}

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for idx in range(args.max_num_batches):
        topic_idx = idx % len(forget_data_list)
        batch_idx = idx // len(forget_data_list)

        unlearn_batch = forget_data_list[topic_idx][batch_idx]
        retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]
        max_length = args.max_lengths[topic_idx]

        # --- Forget loss: vocabulary-masking KL ---
        unlearn_inputs = tokenizer(
            unlearn_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        kl_loss = obliviate_vocab_kl(
            updated_model=updated_model,
            unlearn_inputs=unlearn_inputs,
            sensitive_token_ids=sensitive_token_ids,
        )

        # --- Retain losses: MSE distillation + CE to teacher argmax ---
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        distill_loss = obliviate_distill_mse(
            updated_model=updated_model,
            frozen_model=frozen_model,
            retain_inputs=retain_inputs,
        )

        retain_loss = obliviate_retain_ce(
            updated_model=updated_model,
            frozen_model=frozen_model,
            retain_inputs=retain_inputs,
            pad_token_id=pad_token_id,
        )

        loss = kl_loss + args.Lambda_1 * distill_loss + args.Lambda_2 * retain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        unlearn_component = kl_loss.item()
        retain_component = (args.Lambda_1 * distill_loss + args.Lambda_2 * retain_loss).item()
        loss_history[topic_idx]["total"].append(loss.item())
        loss_history[topic_idx]["unlearn"].append(unlearn_component)
        loss_history[topic_idx]["retain"].append(retain_component)
        # pbar.update(1)

    tokenizer.truncation_side = truncation_side

    path = str(SCRIPT_DIR / f"checkpoints/obliviate/{args.bench_label}/{args.model_name}")
    save_topic_loss_plot(loss_history, path, filename="loss_by_topic.png",
                         topic_names=args.topic_names)
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    args = Args()
    cli_args = parse_args()
    args.model_path = cli_args.model_path
    args.model_name = (cli_args.model_name if cli_args.model_name is not None
                       else cli_args.model_path.split("/")[-1])
    args.Lambda_1 = cli_args.Lambda_1
    args.Lambda_2 = cli_args.Lambda_2
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules
    apply_benchmark_config(args, cli_args.benchmark, cli_args.tofu_split,
                           cli_args.muse_corpus, cli_args.blur_task)

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    frozen_model, tokenizer = load_model(args.model_path)
    updated_model, _ = load_model(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build sensitive token ID list from keywords.json for active topics
    topics = list(args.topic_names.values())
    sensitive_token_ids = get_sensitive_token_ids(tokenizer, topics)
    print(f"Loaded {len(sensitive_token_ids)} sensitive token IDs for topics: {topics}")

    forget_data_list, retain_data_list = get_data(
        forget_corpora=args.forget_corpora,
        retain_corpora=args.retain_corpora,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        chunk_sizes=args.max_lengths,
    )

    run_obliviate(
        updated_model=updated_model,
        frozen_model=frozen_model,
        tokenizer=tokenizer,
        forget_data_list=forget_data_list,
        retain_data_list=retain_data_list,
        sensitive_token_ids=sensitive_token_ids,
        args=args,
    )
    clear_cuda_cache()
