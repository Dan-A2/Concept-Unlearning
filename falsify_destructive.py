"""Destructive falsification: GA with the retain regularizer disabled.

Identical to ga.py but forces alpha=[0, ..., 0] and a small max_num_batches so
the run becomes a pure gradient-ascent destroyer of the model. Saved to
checkpoints/destructive/{bench_label}/{model_name}/ so test.py auto-discover
picks it up.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW

from baselines.benchmarks import add_benchmark_args, apply_benchmark_config
from baselines.losses import ga, kl, make_labels, mse
from baselines.utils import (RandomizedModel, clear_cuda_cache, get_data,
                             get_params, load_model, save_topic_loss_plot)

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "zephyr-7b-beta"

    retain_corpora = ["wikitext", "wikitext"]
    forget_corpora = ["bio-forget-corpus", "cyber-forget-corpus"]

    # Retain weights forced to zero — destroyer mode.
    alpha = [0.0, 0.0]
    retain_loss_fn = "kl"
    nu = 0.0
    lr = 1e-5

    batch_size = 4
    max_num_batches = 100  # short on purpose

    target_layers = [7]
    layer_ids = [5, 6, 7]
    param_ids = [6]

    use_peft = True
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    seed = 42


def _device(model):
    return next(model.parameters()).device


def _apply_lora(model, args):
    if not args.use_peft:
        return model
    # Note: layers_to_transform intentionally NOT passed -> LoRA covers every
    # layer that matches `target_modules`. The original ga.py restricts to
    # layers [5,6,7]; for the destroyer we want broad reach so the destruction
    # signal is unambiguous on every model size.
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Destructive (alpha=0) GA falsification.")
    parser.add_argument("--model_path", default=Args.model_path)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_num_batches", type=int, default=Args.max_num_batches)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)
    add_benchmark_args(parser)
    return parser.parse_args()


def run(updated_model, ref_model, tokenizer, forget_data_list, retain_data_list, args):
    updated_model = _apply_lora(updated_model, args).train()
    params = [p for p in updated_model.parameters() if p.requires_grad] \
        if args.use_peft else get_params(updated_model, args.layer_ids, args.param_ids)
    if not params:
        raise ValueError("No trainable parameters.")

    optimizer = AdamW(params, lr=args.lr)
    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    args.max_num_batches = min(args.max_num_batches, max_possible)

    loss_history = {i: {"total": [], "unlearn": [], "retain": []}
                    for i in range(len(forget_data_list))}

    trunc, pad = tokenizer.truncation_side, tokenizer.padding_side
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"

    dev = _device(updated_model)
    for idx in range(args.max_num_batches):
        topic_idx = idx % len(forget_data_list)
        batch_idx = idx // len(forget_data_list)

        unlearn_batch = forget_data_list[topic_idx][batch_idx]
        retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]
        max_length = args.max_lengths[topic_idx]

        unlearn_inputs = tokenizer(unlearn_batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=max_length).to(dev)
        _ = make_labels(unlearn_inputs.input_ids, unlearn_inputs.attention_mask)
        unlearn_loss = ga(updated_model=updated_model, unlearn_inputs=unlearn_inputs)

        # Retain loss is computed but multiplied by 0 — keeps the loop shape.
        retain_inputs = tokenizer(retain_batch, return_tensors="pt", padding=True,
                                  truncation=True, max_length=max_length).to(dev)
        if args.retain_loss_fn == "kl":
            retain_loss = kl(updated_model=updated_model, ref_model=ref_model,
                             retain_inputs=retain_inputs, nu=args.nu)
        else:
            retain_loss = mse(updated_model=updated_model, ref_model=ref_model,
                              retain_inputs=retain_inputs, nu=args.nu)
        retain_loss = retain_loss * args.alpha[topic_idx]   # 0.0 — no retain pull

        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[topic_idx]["total"].append(loss.item())
        loss_history[topic_idx]["unlearn"].append(unlearn_loss.item())
        loss_history[topic_idx]["retain"].append(retain_loss.item())

    tokenizer.truncation_side, tokenizer.padding_side = trunc, pad

    out = SCRIPT_DIR / f"checkpoints/destructive/{args.bench_label}/{args.model_name}"
    out.mkdir(parents=True, exist_ok=True)
    save_topic_loss_plot(loss_history, str(out), filename="loss_by_topic.png",
                         topic_names=args.topic_names)
    updated_model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved -> {out}")


if __name__ == "__main__":
    args = Args()
    cli = parse_args()
    args.model_path = cli.model_path
    args.model_name = cli.model_name or cli.model_path.split("/")[-1]
    args.max_num_batches = cli.max_num_batches
    if cli.lora_target_modules is not None:
        args.lora_target_modules = cli.lora_target_modules
    apply_benchmark_config(args, cli.benchmark, cli.tofu_split, cli.muse_corpus, cli.blur_task)
    # Force destroyer config no matter what apply_benchmark_config did.
    args.alpha = [0.0] * len(args.forget_corpora)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model, tokenizer = load_model(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = RandomizedModel(model_name_or_path=args.model_path,
                                target_layers=args.target_layers)

    forget_data_list, retain_data_list = get_data(
        forget_corpora=args.forget_corpora,
        retain_corpora=args.retain_corpora,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        chunk_sizes=args.max_lengths,
    )

    run(model, ref_model, tokenizer, forget_data_list, retain_data_list, args)
    clear_cuda_cache()
