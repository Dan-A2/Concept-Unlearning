"""Fine-tune a base model on the full TOFU dataset.

This is a prerequisite for TOFU unlearning experiments.  The fine-tuned
checkpoint is then passed as --model_path to the unlearning scripts with
--benchmark tofu --tofu_split <split>.

Pre-trained alternatives (skip this script):
  locuslab/tofu_ft_llama2-7b   — Llama-2-7B fine-tuned on full TOFU
  locuslab/tofu_ft_phi-1.5     — Phi-1.5 fine-tuned on full TOFU

Usage:
  python tofu_finetune.py --model_path meta-llama/Llama-2-7b-hf
  python tofu_finetune.py --model_path meta-llama/Meta-Llama-3.1-8B --epochs 5
"""

import numpy as np
import torch
import argparse
from torch.optim import AdamW
import tqdm as tqdm
from pathlib import Path
from datasets import load_dataset

from baselines.utils import load_model, clear_cuda_cache
from baselines.losses import compute_loss_from_logits, make_labels
from peft import LoraConfig, get_peft_model, TaskType

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    model_path = "meta-llama/Meta-Llama-3.1-8B"
    model_name = "Llama-3.1-8B"

    lr = 2e-5
    batch_size = 4
    epochs = 5
    max_length = 512

    # LoRA config
    use_peft = True
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    seed = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on full TOFU dataset.")
    parser.add_argument("--model_path", type=str, default=Args.model_path,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name for checkpoint dir. Defaults to last segment of --model_path.")
    parser.add_argument("--lr", type=float, default=Args.lr)
    parser.add_argument("--batch_size", type=int, default=Args.batch_size)
    parser.add_argument("--epochs", type=int, default=Args.epochs)
    parser.add_argument("--max_length", type=int, default=Args.max_length)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target module names.")
    return parser.parse_args()


def main():
    args = Args()
    cli_args = parse_args()
    args.model_path = cli_args.model_path
    args.model_name = cli_args.model_name or cli_args.model_path.split("/")[-1]
    args.lr = cli_args.lr
    args.batch_size = cli_args.batch_size
    args.epochs = cli_args.epochs
    args.max_length = cli_args.max_length
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, tokenizer = load_model(args.model_path)

    if args.use_peft:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    model.train()
    device = next(model.parameters()).device

    # Load full TOFU dataset (4000 QA pairs)
    dataset = load_dataset("locuslab/TOFU", "full", split="train")
    texts = [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in dataset]
    print(f"Loaded {len(texts)} TOFU QA pairs for fine-tuning.")

    batches = [texts[i:i + args.batch_size] for i in range(0, len(texts), args.batch_size)]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr)

    orig_trunc = tokenizer.truncation_side
    orig_pad = tokenizer.padding_side
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        with tqdm.tqdm(total=len(batches), desc=f"Epoch {epoch + 1}/{args.epochs}") as pbar:
            for batch in batches:
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_length,
                ).to(device)

                labels = make_labels(inputs.input_ids, inputs.attention_mask)
                outputs = model(**inputs)
                loss = compute_loss_from_logits(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        avg = epoch_loss / len(batches)
        print(f"Epoch {epoch + 1}: avg loss = {avg:.4f}")

    tokenizer.truncation_side = orig_trunc
    tokenizer.padding_side = orig_pad

    # Merge LoRA weights so the checkpoint can be loaded as a normal model
    if args.use_peft:
        model = model.merge_and_unload()

    path = str(SCRIPT_DIR / f"checkpoints/tofu_finetune/{args.model_name}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved fine-tuned model to {path}")


if __name__ == "__main__":
    main()
    clear_cuda_cache()
