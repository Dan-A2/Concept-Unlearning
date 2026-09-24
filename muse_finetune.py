"""Fine-tune a base model on the MUSE 'train' subset (Books or News).

The 'train' subset of muse-bench/MUSE-{Books,News} contains the data the
target model is meant to be pre-trained on (no sub-splits — only the
'raw' subset has forget/retain1/retain2/holdout splits).

Texts (especially for Books — whole books per example) are far longer
than max_length, so we follow the standard causal-LM pre-training
pattern: concatenate everything, tokenize, then chunk into fixed-size
blocks of `max_length` tokens.

The fine-tuned checkpoint is then passed as --model_path to the unlearning
scripts with --benchmark muse --muse_corpus {books,news}. Unlearning and
evaluation use the 'raw' subset (forget / retain1 / retain2 / holdout).

Hyperparameters follow the MUSE paper (Shi et al., 2024,
https://arxiv.org/abs/2407.06460) and its reference implementation
(github.com/swj0419/muse_bench).  The paper fine-tunes the target
models "for 5 epochs with a constant learning rate of 1e-5 and a
batch size of 32":

  learning rate         : 1e-5
  epochs                : 5
  max sequence length   : 2048
  weight decay          : 0.0   (HF default, as in muse_bench)
  LR schedule           : constant (paper; muse_bench's finetune.py
                          default is cosine — see --lr_scheduler)
  effective batch size  : 32    (via gradient accumulation)
  fine-tuning           : full-parameter (pass --use_peft for LoRA)

Usage:
  python muse_finetune.py --model_path meta-llama/Meta-Llama-3.1-8B \
                          --muse_corpus books
"""

import math
import numpy as np
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import tqdm as tqdm
from pathlib import Path
from datasets import load_dataset, Dataset

from baselines.utils import load_model, clear_cuda_cache
from baselines.losses import compute_loss_from_logits
from peft import LoraConfig, get_peft_model, TaskType

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    model_path = "meta-llama/Meta-Llama-3.1-8B"
    model_name = "Llama-3.1-8B"
    muse_corpus = "books"   # "books" or "news"

    # MUSE paper defaults
    lr = 1e-5
    epochs = 5
    max_length = 2048
    weight_decay = 0.0
    warmup_ratio = 0.0
    lr_scheduler = "constant"   # "constant" (paper), "cosine" (muse_bench), "linear"

    # Per-device batch size and gradient accumulation to reach an
    # effective batch size of 32 (MUSE paper).
    batch_size = 1
    grad_accum_steps = 32

    # LoRA config (only used with --use_peft; MUSE target models are
    # fully fine-tuned)
    use_peft = False
    load_in_4bit = False   # QLoRA: 4-bit base + LoRA (needed for Qwen3-32B)
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    seed = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a base model on MUSE train subset.")
    parser.add_argument("--model_path", type=str, default=Args.model_path,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name for checkpoint dir. Defaults to last segment of --model_path.")
    parser.add_argument("--muse_corpus", type=str, default=Args.muse_corpus,
                        choices=["books", "news"],
                        help="Which MUSE corpus to fine-tune on.")
    parser.add_argument("--lr", type=float, default=Args.lr)
    parser.add_argument("--batch_size", type=int, default=Args.batch_size,
                        help="Per-device batch size (in chunks of max_length tokens).")
    parser.add_argument("--grad_accum_steps", type=int, default=Args.grad_accum_steps,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).")
    parser.add_argument("--epochs", type=int, default=Args.epochs)
    parser.add_argument("--max_length", type=int, default=Args.max_length)
    parser.add_argument("--weight_decay", type=float, default=Args.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=Args.warmup_ratio,
                        help="Fraction of optimizer steps used for linear warmup "
                             "(applies to all schedules; paper uses none).")
    parser.add_argument("--lr_scheduler", type=str, default=Args.lr_scheduler,
                        choices=["constant", "cosine", "linear"],
                        help="LR schedule: 'constant' matches the MUSE paper, "
                             "'cosine' matches muse_bench's finetune.py default.")
    parser.add_argument("--use_peft", action="store_true",
                        help="Use LoRA instead of full fine-tuning (deviates from MUSE).")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load base in 4-bit NF4 (QLoRA); use with --use_peft for Qwen3-32B.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target module names.")
    return parser.parse_args()


def _make_scheduler(optimizer, kind, num_warmup, num_total):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        if kind == "constant":
            return 1.0
        progress = float(step - num_warmup) / float(max(1, num_total - num_warmup))
        progress = min(1.0, progress)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)  # linear
    return LambdaLR(optimizer, lr_lambda)


def _load_train_subset(hub_id: str):
    """Load the MUSE 'train' subset, which has no sub-splits.

    Since HF's `datasets` library always wraps single-split configs in a
    DatasetDict with a default key, we load without specifying `split=`
    and then pick the sole Dataset that comes back.
    """
    ds = load_dataset(hub_id, "train")
    if isinstance(ds, Dataset):
        return ds
    # DatasetDict (or dict-like): grab the single split, whatever it's called.
    keys = list(ds.keys())
    if len(keys) == 0:
        raise RuntimeError(f"'train' subset of {hub_id} is empty.")
    if len(keys) > 1:
        print(f"[warn] 'train' subset has {len(keys)} splits {keys}; "
              f"concatenating all of them.")
        from datasets import concatenate_datasets
        return concatenate_datasets([ds[k] for k in keys])
    return ds[keys[0]]


def load_and_chunk(hub_id: str, tokenizer, max_length: int):
    """Load the train subset, tokenize, and chunk into fixed-length blocks.

    Returns a list of 1-D LongTensors, each of length `max_length`.
    """
    ds = _load_train_subset(hub_id)
    # Identify the text column (MUSE uses "text", but be defensive).
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    all_ids: list[int] = []
    n_docs = 0
    for x in ds:
        text = x.get(text_col)
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        if tokenizer.eos_token_id is not None:
            all_ids.append(tokenizer.eos_token_id)
        n_docs += 1

    print(f"Loaded {n_docs} documents from {hub_id} (train subset).")
    print(f"Total tokens: {len(all_ids):,}")

    num_chunks = len(all_ids) // max_length
    if num_chunks == 0:
        raise RuntimeError(
            f"Not enough tokens ({len(all_ids)}) to form a single chunk of "
            f"{max_length}. Reduce --max_length."
        )
    chunks = [
        torch.tensor(all_ids[i * max_length:(i + 1) * max_length], dtype=torch.long)
        for i in range(num_chunks)
    ]
    print(f"Chunked into {num_chunks} blocks of {max_length} tokens "
          f"(dropped {len(all_ids) - num_chunks * max_length} trailing tokens).")
    return chunks


def main():
    args = Args()
    cli_args = parse_args()
    args.model_path = cli_args.model_path
    args.model_name = cli_args.model_name or cli_args.model_path.split("/")[-1]
    args.muse_corpus = cli_args.muse_corpus
    args.lr = cli_args.lr
    args.batch_size = cli_args.batch_size
    args.grad_accum_steps = cli_args.grad_accum_steps
    args.epochs = cli_args.epochs
    args.max_length = cli_args.max_length
    args.weight_decay = cli_args.weight_decay
    args.warmup_ratio = cli_args.warmup_ratio
    args.lr_scheduler = cli_args.lr_scheduler
    args.use_peft = cli_args.use_peft
    args.load_in_4bit = cli_args.load_in_4bit
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, tokenizer = load_model(args.model_path, load_in_4bit=args.load_in_4bit)

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

    hub_id = f"muse-bench/MUSE-{args.muse_corpus.capitalize()}"
    chunks = load_and_chunk(hub_id, tokenizer, args.max_length)

    # Group chunks into micro-batches
    batches = [chunks[i:i + args.batch_size]
               for i in range(0, len(chunks), args.batch_size)]

    steps_per_epoch = math.ceil(len(batches) / args.grad_accum_steps)
    total_optim_steps = steps_per_epoch * args.epochs
    num_warmup = int(args.warmup_ratio * total_optim_steps)

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
    scheduler = _make_scheduler(optimizer, args.lr_scheduler, num_warmup, total_optim_steps)

    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}  "
          f"(per-device {args.batch_size} x {args.grad_accum_steps} grad accum)")
    print(f"Micro-batches per epoch: {len(batches)}  "
          f"Optimizer steps per epoch: {steps_per_epoch}")
    print(f"Total optimizer steps: {total_optim_steps}  "
          f"(schedule {args.lr_scheduler}, warmup {num_warmup}, "
          f"max_len {args.max_length}, lr {args.lr})")

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        order = np.random.permutation(len(batches))

        with tqdm.tqdm(total=len(batches), desc=f"Epoch {epoch + 1}/{args.epochs}") as pbar:
            for micro_idx, b_idx in enumerate(order):
                batch = batches[b_idx]
                input_ids = torch.stack(batch).to(device)
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = compute_loss_from_logits(outputs.logits, labels)
                (loss / args.grad_accum_steps).backward()

                epoch_loss += loss.item()

                do_step = ((micro_idx + 1) % args.grad_accum_steps == 0
                           or (micro_idx + 1) == len(batches))
                if do_step:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")
                pbar.update(1)

        avg = epoch_loss / max(1, len(batches))
        print(f"Epoch {epoch + 1}: avg loss = {avg:.4f}")

    # Merge LoRA weights so the checkpoint can be loaded as a normal model
    if args.use_peft:
        model = model.merge_and_unload()
    model.config.use_cache = True

    path = str(SCRIPT_DIR / f"checkpoints/muse_finetune/{args.muse_corpus}/{args.model_name}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved fine-tuned model to {path}")


if __name__ == "__main__":
    main()
    clear_cuda_cache()
