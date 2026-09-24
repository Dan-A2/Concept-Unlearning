import numpy as np
import torch
import argparse
from torch.optim import AdamW
import sys
from pathlib import Path

from baselines.utils import load_model, get_data, RandomizedModel, clear_cuda_cache, save_topic_loss_plot
from baselines.benchmarks import add_benchmark_args, apply_benchmark_config
from baselines.losses import ga, kl, mse, make_labels
from peft import LoraConfig, get_peft_model, TaskType

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    # Model
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "zephyr-7b-beta"

    # Data
    retain_corpora = ["wikitext", "wikitext"]
    forget_corpora = ["bio-forget-corpus", "cyber-forget-corpus"]

    # Hyperparameters (Jang et al., 2022; Li et al., 2024 WMDP)
    alpha = [1.0, 1.0]         # retain loss weight per topic
    retain_loss_fn = "kl"      # "kl" or "mse"
    nu = 0.0                   # noise scale for retain loss (0 = no noise)
    lr = 1e-5

    batch_size = 4
    max_num_batches = 500

    target_layers = [7]

    # PEFT (LoRA) config
    use_peft = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
    )

    updated_model = get_peft_model(updated_model, lora_config)
    updated_model.enable_input_require_grads()
    updated_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    updated_model.print_trainable_parameters()
    return updated_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run GA unlearning.")
    parser.add_argument("--nu", type=float, default=0.0, help="Noise scale for retain loss.")
    parser.add_argument("--model_path", type=str, default=Args.model_path, help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None, help="Short name used for checkpoint dirs. Defaults to the last path segment of --model_path.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="LoRA target module names (e.g. q_proj v_proj). Defaults to class default.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit NF4 (QLoRA) for memory-constrained models such as Qwen3-32B.")
    add_benchmark_args(parser)
    return parser.parse_args()


def run_ga(
    updated_model,
    ref_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    updated_model = _apply_lora_if_enabled(updated_model, args)
    updated_model = updated_model.train()

    params = [p for p in updated_model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise ValueError("No trainable parameters found. Check PEFT config or selected params.")

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

        # Forget loss: gradient ascent (negate CE)
        unlearn_inputs = tokenizer(
            unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model_device)

        labels = make_labels(unlearn_inputs.input_ids, unlearn_inputs.attention_mask)
        
        unlearn_loss = ga(updated_model=updated_model, unlearn_inputs=unlearn_inputs)

        # Retain loss: KL or MSE to reference model
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model_device)

        if args.retain_loss_fn == "kl":
            retain_loss = kl(updated_model=updated_model, ref_model=ref_model, retain_inputs=retain_inputs, nu=args.nu)
        elif args.retain_loss_fn == "mse":
            retain_loss = mse(updated_model=updated_model, ref_model=ref_model, retain_inputs=retain_inputs, nu=args.nu)
        else:
            raise ValueError(f"Unsupported retain loss: {args.retain_loss_fn}")
        retain_loss *= args.alpha[topic_idx]

        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[topic_idx]["total"].append(loss.item())
        loss_history[topic_idx]["unlearn"].append(unlearn_loss.item())
        loss_history[topic_idx]["retain"].append(retain_loss.item())

        # param_change = params[0].grad.abs().mean().item() if params[0].grad is not None else 0.0
        # print(f"loss topic {topic_idx}: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {param_change:.4g}")
        # pbar.update(1)

    tokenizer.truncation_side = truncation_side
    tokenizer.padding_side = padding_side

    if args.nu > 0.0:
        path = str(SCRIPT_DIR / f"checkpoints/ga/{args.bench_label}/{args.retain_loss_fn}-{args.model_name}_nu")
    else:
        path = str(SCRIPT_DIR / f"checkpoints/ga/{args.bench_label}/{args.retain_loss_fn}-{args.model_name}")

    save_topic_loss_plot(loss_history, path, filename="loss_by_topic.png", topic_names=args.topic_names)
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    # print(f"Saved model to {path}")


if __name__ == "__main__":
    args = Args()
    cli_args = parse_args()
    args.nu = cli_args.nu
    args.model_path = cli_args.model_path
    args.model_name = cli_args.model_name if cli_args.model_name is not None else cli_args.model_path.split("/")[-1]
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules
    args.load_in_4bit = cli_args.load_in_4bit
    apply_benchmark_config(args, cli_args.benchmark, cli_args.tofu_split, cli_args.muse_corpus)

    if args.nu > 0.0:
        save_path = SCRIPT_DIR / f"checkpoints/ga/{args.bench_label}/{args.retain_loss_fn}-{args.model_name}_nu"
    else:
        save_path = SCRIPT_DIR / f"checkpoints/ga/{args.bench_label}/{args.retain_loss_fn}-{args.model_name}"
    if save_path.exists():
        print(f"Unlearned model already saved at {save_path}; skipping.")
        import sys; sys.exit(0)

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    updated_model, tokenizer = load_model(args.model_path, load_in_4bit=args.load_in_4bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = RandomizedModel(model_name_or_path=args.model_path, target_layers=args.target_layers, load_in_4bit=args.load_in_4bit)

    forget_data_list, retain_data_list = get_data(
        forget_corpora=args.forget_corpora,
        retain_corpora=args.retain_corpora,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        chunk_sizes=args.max_lengths,
    )

    run_ga(
        updated_model=updated_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        forget_data_list=forget_data_list,
        retain_data_list=retain_data_list,
        args=args,
    )
    clear_cuda_cache()
