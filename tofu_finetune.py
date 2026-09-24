"""Fine-tune a base model on the full TOFU dataset.

This is a prerequisite for TOFU unlearning experiments.  The fine-tuned
checkpoint is then passed as --model_path to the unlearning scripts with
--benchmark tofu --tofu_split <split>.

Hyperparameters follow the TOFU paper (Maini et al., 2024,
https://arxiv.org/abs/2401.06121; github.com/locuslab/tofu) and the
open-unlearning benchmark (github.com/locuslab/open-unlearning):

  learning rate         : 1e-5
  epochs                : 5
  effective batch size  : 32    (via gradient accumulation)
  weight decay          : 0.01
  LR schedule           : linear warmup over the first epoch of optimizer
                          steps, then linear decay to 0
  loss                  : answer tokens only ("Question: {q}\n" is masked,
                          matching locuslab/tofu label masking)
  fine-tuning           : full-parameter (both benchmarks); pass
                          --use_peft to fall back to LoRA + merge

Usage:
  python tofu_finetune.py --model_path meta-llama/Llama-2-7b-hf
  python tofu_finetune.py --model_path meta-llama/Meta-Llama-3.1-8B --epochs 5
"""

import math
import numpy as np
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import tqdm as tqdm
from pathlib import Path
from datasets import load_dataset

from baselines.utils import load_model, clear_cuda_cache
from baselines.losses import compute_loss_from_logits, make_labels
from baselines.benchmarks import TOFU_SPLITS
from peft import LoraConfig, get_peft_model, TaskType

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    model_path = "meta-llama/Meta-Llama-3.1-8B"
    model_name = "Llama-3.1-8B"

    # TOFU paper / open-unlearning defaults
    lr = 1e-5
    epochs = 5
    max_length = 512
    weight_decay = 0.01
    warmup_epochs = 1.0

    # Per-device batch size and gradient accumulation to reach an
    # effective batch size of 32 (TOFU paper / open-unlearning).
    batch_size = 4
    grad_accum_steps = 8

    # LoRA config (only used with --use_peft; the benchmarks fine-tune
    # all parameters)
    use_peft = False
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    seed = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on full TOFU dataset.")
    parser.add_argument("--model_path", type=str, default=Args.model_path,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name for checkpoint dir. Defaults to last segment of --model_path.")
    parser.add_argument("--lr", type=float, default=Args.lr)
    parser.add_argument("--batch_size", type=int, default=Args.batch_size,
                        help="Per-device batch size.")
    parser.add_argument("--grad_accum_steps", type=int, default=Args.grad_accum_steps,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).")
    parser.add_argument("--epochs", type=int, default=Args.epochs)
    parser.add_argument("--max_length", type=int, default=Args.max_length)
    parser.add_argument("--weight_decay", type=float, default=Args.weight_decay)
    parser.add_argument("--warmup_epochs", type=float, default=Args.warmup_epochs,
                        help="Epochs of linear LR warmup before linear decay "
                             "(TOFU/open-unlearning use 1.0).")
    parser.add_argument("--use_peft", action="store_true",
                        help="Use LoRA instead of full fine-tuning (deviates from the benchmarks).")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target module names.")
    parser.add_argument("--retain", action="store_true",
                        help="Train a RETAIN reference model: fine-tune on the "
                             "retain split (complement of --tofu_split) instead of "
                             "'full', and save under checkpoints/tofu_retain/<split>/. "
                             "This is the reference model TOFU Forget Quality needs.")
    parser.add_argument("--tofu_split", type=str, default="forget10",
                        choices=["forget01", "forget05", "forget10"],
                        help="Forget split whose RETAIN complement to train on "
                             "(only used with --retain).")
    return parser.parse_args()


def _linear_warmup_decay(optimizer, num_warmup, num_total):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = float(step - num_warmup) / float(max(1, num_total - num_warmup))
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


def main():
    args = Args()
    cli_args = parse_args()
    args.model_path = cli_args.model_path
    args.model_name = cli_args.model_name or cli_args.model_path.split("/")[-1]
    args.lr = cli_args.lr
    args.batch_size = cli_args.batch_size
    args.grad_accum_steps = cli_args.grad_accum_steps
    args.epochs = cli_args.epochs
    args.max_length = cli_args.max_length
    args.weight_decay = cli_args.weight_decay
    args.warmup_epochs = cli_args.warmup_epochs
    args.use_peft = cli_args.use_peft
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules

    # Retain-reference mode: train on the retain complement of the forget split
    # (e.g. forget10 -> retain90) and save under a dedicated directory.
    args.retain = cli_args.retain
    args.tofu_split = cli_args.tofu_split
    if args.retain:
        args.tofu_config = TOFU_SPLITS[args.tofu_split]   # e.g. "retain90"
    else:
        args.tofu_config = "full"

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

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    model.train()
    device = next(model.parameters()).device

    # Load the TOFU training split. Default 'full' (4000 QA); with --retain,
    # the retain complement (e.g. retain90, 3600 QA). Loss is on the answer
    # tokens only: the "Question: {q}\n" prefix is masked out, matching the
    # TOFU reference implementation.
    dataset = load_dataset("locuslab/TOFU", args.tofu_config, split="train")
    prompts = [f"Question: {x['question']}\n" for x in dataset]
    texts = [f"{p}Answer: {x['answer']}" for p, x in zip(prompts, dataset)]
    print(f"Loaded {len(texts)} TOFU QA pairs from '{args.tofu_config}' "
          f"for fine-tuning{' (RETAIN reference)' if args.retain else ''}.")

    n_batches = math.ceil(len(texts) / args.batch_size)
    steps_per_epoch = math.ceil(n_batches / args.grad_accum_steps)
    total_optim_steps = steps_per_epoch * args.epochs
    num_warmup = max(1, int(args.warmup_epochs * steps_per_epoch))

    # AdamW with no weight decay on biases / LayerNorm params
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optim_groups = [
        {"params": [p for n, p in trainable if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in trainable if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=args.lr)
    scheduler = _linear_warmup_decay(optimizer, num_warmup, total_optim_steps)

    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}  "
          f"(per-device {args.batch_size} x {args.grad_accum_steps} grad accum)")
    print(f"LR schedule: linear warmup {num_warmup} steps, linear decay over "
          f"{total_optim_steps} total optimizer steps (peak lr {args.lr})")

    orig_trunc = tokenizer.truncation_side
    orig_pad = tokenizer.padding_side
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        # Reshuffle examples every epoch (the HF Trainer used by the
        # benchmarks does the same).
        order = np.random.permutation(len(texts))
        batches = [order[i:i + args.batch_size]
                   for i in range(0, len(order), args.batch_size)]

        with tqdm.tqdm(total=len(batches), desc=f"Epoch {epoch + 1}/{args.epochs}") as pbar:
            for micro_idx, idx in enumerate(batches):
                batch_texts = [texts[j] for j in idx]
                batch_prompts = [prompts[j] for j in idx]

                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_length,
                ).to(device)

                labels = make_labels(inputs.input_ids, inputs.attention_mask)
                # Mask the question tokens so loss is on the answer only.
                prompt_ids = tokenizer(batch_prompts)["input_ids"]
                for row, p_ids in enumerate(prompt_ids):
                    n = min(len(p_ids), labels.size(1))
                    labels[row, :n] = -100

                outputs = model(**inputs)
                loss = compute_loss_from_logits(outputs.logits, labels)
                (loss / args.grad_accum_steps).backward()

                epoch_loss += loss.item()

                do_step = ((micro_idx + 1) % args.grad_accum_steps == 0
                           or (micro_idx + 1) == len(batches))
                if do_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")
                pbar.update(1)

        avg = epoch_loss / len(batches)
        print(f"Epoch {epoch + 1}: avg loss = {avg:.4f}")

    tokenizer.truncation_side = orig_trunc
    tokenizer.padding_side = orig_pad

    # Merge LoRA weights so the checkpoint can be loaded as a normal model
    if args.use_peft:
        model = model.merge_and_unload()
    model.config.use_cache = True

    if args.retain:
        path = str(SCRIPT_DIR / f"checkpoints/tofu_retain/{args.tofu_split}/{args.model_name}")
    else:
        path = str(SCRIPT_DIR / f"checkpoints/tofu_finetune/{args.model_name}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved fine-tuned model to {path}")


if __name__ == "__main__":
    main()
    clear_cuda_cache()
