"""SPUL — Soft Prompt Unlearning.

Implements the algorithm from:
  "SPUL: Soft Prompt Unlearning for Large Language Models"
  (Bhaila et al., 2024)
  https://github.com/karuna-bhaila/llm_unlearning

Two-phase approach:
  Phase 1 — LoRA fine-tuning on combined forget + retain data, so the
            model internalises the knowledge to be unlearned.  LoRA
            weights are merged back into the base model afterwards.
  Phase 2 — P-tuning (soft prompt) unlearning.  The LLM weights stay
            frozen; only a prompt encoder is trained with three losses:
              1. Forget GA   — gradient ascent (negated CE) on forget data.
              2. Retain CE   — standard next-token-prediction on retain data.
              3. Retain KL   — KL between prompted and unprompted (frozen)
                              model on retain data.
            total = forget_loss + alpha * retain_ce + beta * retain_kl

Usage:
  python spul.py --model_path HuggingFaceH4/zephyr-7b-beta
"""

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from copy import deepcopy
from torch.optim import AdamW
from pathlib import Path

from baselines.utils import load_model, get_data, clear_cuda_cache, save_topic_loss_plot
from baselines.benchmarks import add_benchmark_args, apply_benchmark_config
from baselines.losses import make_labels
from peft import LoraConfig, PromptEncoderConfig, get_peft_model, TaskType

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    # Model
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "zephyr-7b-beta"

    # Data
    retain_corpora = ["wikitext", "wikitext"]
    forget_corpora = ["bio-forget-corpus", "cyber-forget-corpus"]

    # Phase 1: LoRA fine-tuning
    finetune_lr = 1e-4
    finetune_steps = 300

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Phase 2: P-tuning unlearning
    alpha = [1.0, 1.0]          # retain CE loss weight per topic
    beta = 1.0                  # retain KL loss weight
    num_prompt_tokens = 20      # number of virtual tokens
    encoder_hidden_size = 128   # hidden size of the prompt encoder MLP
    lr = 1e-3

    batch_size = 4
    max_num_batches = 500

    seed = 42


def _get_model_device(model):
    return next(model.parameters()).device


def parse_args():
    parser = argparse.ArgumentParser(description="Run SPUL (Soft Prompt Unlearning).")
    parser.add_argument("--model_path", type=str, default=Args.model_path,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name for checkpoint dirs.")
    # Phase 1
    parser.add_argument("--finetune_lr", type=float, default=Args.finetune_lr,
                        help="Learning rate for Phase 1 LoRA fine-tuning.")
    parser.add_argument("--finetune_steps", type=int, default=Args.finetune_steps,
                        help="Number of fine-tuning steps in Phase 1.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target modules for Phase 1 fine-tuning.")
    # Phase 2
    parser.add_argument("--num_prompt_tokens", type=int, default=Args.num_prompt_tokens,
                        help="Number of virtual tokens for the prompt encoder.")
    parser.add_argument("--encoder_hidden_size", type=int, default=Args.encoder_hidden_size,
                        help="Hidden size of the prompt encoder MLP.")
    parser.add_argument("--alpha", type=float, nargs="+", default=None,
                        help="Retain CE loss weight per topic.")
    parser.add_argument("--beta", type=float, default=Args.beta,
                        help="Retain KL loss weight.")
    parser.add_argument("--lr", type=float, default=Args.lr,
                        help="Learning rate for Phase 2 prompt encoder parameters.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit NF4 (QLoRA) for memory-constrained models such as Qwen3-32B.")
    add_benchmark_args(parser)
    return parser.parse_args()


# ================================================================
# Phase 1: LoRA fine-tuning
# ================================================================

def _run_finetune(model, tokenizer, forget_data_list, retain_data_list, args):
    """Fine-tune the base model on combined forget + retain data with LoRA,
    then merge LoRA weights back into the base model."""
    if args.finetune_steps <= 0:
        print("Skipping Phase 1 (finetune_steps <= 0)")
        return model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.finetune_lr)
    model_device = _get_model_device(model)

    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    finetune_steps = min(args.finetune_steps, max_possible)

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    print(f"SPUL Phase 1: LoRA fine-tuning ({finetune_steps} steps)")
    for idx in range(finetune_steps):
        topic_idx = idx % len(forget_data_list)
        batch_idx = idx // len(forget_data_list)
        max_length = args.max_lengths[topic_idx]

        # CE on forget data
        forget_batch = forget_data_list[topic_idx][batch_idx]
        forget_inputs = tokenizer(
            forget_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)
        forget_labels = make_labels(forget_inputs.input_ids, forget_inputs.attention_mask)
        forget_loss = model(**forget_inputs, labels=forget_labels).loss

        # CE on retain data
        retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)
        retain_labels = make_labels(retain_inputs.input_ids, retain_inputs.attention_mask)
        retain_loss = model(**retain_inputs, labels=retain_labels).loss

        loss = forget_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # pbar.update(1)

    tokenizer.truncation_side = truncation_side

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()
    print("Phase 1 complete: LoRA merged into base model.")
    return model


# ================================================================
# Phase 2: P-tuning unlearning
# ================================================================

def run_spul(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    """P-tuning unlearning on the fine-tuned model."""
    pt_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_prompt_tokens,
        encoder_hidden_size=args.encoder_hidden_size,
    )
    updated_model = get_peft_model(updated_model, pt_config)
    updated_model.print_trainable_parameters()
    updated_model.train()

    params = [p for p in updated_model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("No trainable parameters found.")

    model_device = _get_model_device(updated_model)
    optimizer = AdamW(params, lr=args.lr)

    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    if args.max_num_batches > max_possible:
        args.max_num_batches = max_possible
    loss_history = {i: {"total": [], "unlearn": [], "retain": []}
                    for i in range(len(forget_data_list))}

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    num_vt = args.num_prompt_tokens

    print(f"SPUL Phase 2: P-tuning unlearning ({args.max_num_batches} steps)")
    for idx in range(args.max_num_batches):
        topic_idx = idx % len(forget_data_list)
        batch_idx = idx // len(forget_data_list)

        unlearn_batch = forget_data_list[topic_idx][batch_idx]
        retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]
        max_length = args.max_lengths[topic_idx]

        # --- Forget loss: gradient ascent (negated CE) ---
        unlearn_inputs = tokenizer(
            unlearn_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        unlearn_labels = make_labels(
            unlearn_inputs.input_ids, unlearn_inputs.attention_mask,
        )
        unlearn_outputs = updated_model(
            input_ids=unlearn_inputs.input_ids,
            attention_mask=unlearn_inputs.attention_mask,
            labels=unlearn_labels,
        )
        forget_loss = -unlearn_outputs.loss

        # --- Retain CE loss ---
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        retain_labels = make_labels(
            retain_inputs.input_ids, retain_inputs.attention_mask,
        )
        retain_outputs = updated_model(
            input_ids=retain_inputs.input_ids,
            attention_mask=retain_inputs.attention_mask,
            labels=retain_labels,
        )
        retain_ce_loss = retain_outputs.loss

        # --- Retain KL loss (prompted model vs. frozen unprompted model) ---
        # Slice off virtual-token positions so shapes match the frozen model
        prompted_logits = retain_outputs.logits[:, num_vt:, :]

        with torch.no_grad():
            frozen_outputs = frozen_model(**retain_inputs)
        frozen_logits = frozen_outputs.logits.to(prompted_logits.device)

        probs = F.log_softmax(prompted_logits.float(), dim=-1).view(
            -1, prompted_logits.shape[-1],
        )
        ref_probs = F.log_softmax(frozen_logits.float(), dim=-1).view(
            -1, frozen_logits.shape[-1],
        )
        retain_kl_loss = F.kl_div(
            probs, ref_probs, reduction="batchmean", log_target=True,
        )

        # --- Total loss ---
        retain_loss = args.alpha[topic_idx] * retain_ce_loss + args.beta * retain_kl_loss
        loss = forget_loss + retain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[topic_idx]["total"].append(loss.item())
        loss_history[topic_idx]["unlearn"].append(forget_loss.item())
        loss_history[topic_idx]["retain"].append(retain_loss.item())

    tokenizer.truncation_side = truncation_side

    path = str(SCRIPT_DIR / f"checkpoints/spul/{args.bench_label}/{args.model_name}")
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
    args.finetune_lr = cli_args.finetune_lr
    args.finetune_steps = cli_args.finetune_steps
    args.num_prompt_tokens = cli_args.num_prompt_tokens
    args.encoder_hidden_size = cli_args.encoder_hidden_size
    if cli_args.alpha is not None:
        args.alpha = cli_args.alpha
    args.beta = cli_args.beta
    args.lr = cli_args.lr
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules
    args.load_in_4bit = cli_args.load_in_4bit
    apply_benchmark_config(args, cli_args.benchmark, cli_args.tofu_split, cli_args.muse_corpus)

    save_path = SCRIPT_DIR / f"checkpoints/spul/{args.bench_label}/{args.model_name}"
    if save_path.exists():
        print(f"Unlearned model already saved at {save_path}; skipping.")
        import sys; sys.exit(0)

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model once
    model, tokenizer = load_model(args.model_path, load_in_4bit=args.load_in_4bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_data_list, retain_data_list = get_data(
        forget_corpora=args.forget_corpora,
        retain_corpora=args.retain_corpora,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        chunk_sizes=args.max_lengths,
    )

    # Phase 1: LoRA fine-tune on forget + retain, then merge
    model = _run_finetune(model, tokenizer, forget_data_list, retain_data_list, args)

    # Create frozen reference from fine-tuned model
    frozen_model = deepcopy(model)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    # Phase 2: P-tuning unlearning
    run_spul(
        updated_model=model,
        frozen_model=frozen_model,
        tokenizer=tokenizer,
        forget_data_list=forget_data_list,
        retain_data_list=retain_data_list,
        args=args,
    )
    clear_cuda_cache()
