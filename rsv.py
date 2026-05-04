import numpy as np
import torch
import argparse
from torch.optim import AdamW
import json
from datasets import load_dataset
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from baselines.utils import get_random_vector, get_params, forward_with_cache, load_model, get_data as _get_data_base, clear_cuda_cache, save_topic_loss_plot
from baselines.benchmarks import add_benchmark_args, apply_benchmark_config
from peft import LoraConfig, get_peft_model, TaskType

class Args:
    # Model
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "zephyr-7b-beta"
    module_str = "{model_name}.model.layers[{layer_id}]"

    # Data
    retain_corpora = ["wikitext", "wikitext"]
    forget_corpora = ["bio-forget-corpus", "cyber-forget-corpus"]

    # RSV hyperparameters
    alpha = 5000.0
    steering_coeff_list = [10, 10]
    nu = 0.0
    lr = 5e-5

    batch_size = 4
    max_num_batches = 500

    layer_id = 7
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


def _get_layers_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers
        if hasattr(base, "layers"):
            return base.layers

    raise ValueError("Could not locate transformer layers in model.")


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
    parser = argparse.ArgumentParser(description="Run RSV unlearning.")
    parser.add_argument("--nu", type=float, default=0.0, help="Noise scale for retain loss.")
    parser.add_argument("--model_path", type=str, default=Args.model_path, help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None, help="Short name used for checkpoint dirs. Defaults to the last path segment of --model_path.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="LoRA target module names (e.g. q_proj v_proj). Defaults to class default.")
    add_benchmark_args(parser)
    return parser.parse_args()


def get_data(forget_corpora, retain_corpora, min_len=0, max_len=2000, batch_size=4,
             tokenizer=None, chunk_sizes=None):
    CORPUS_TO_KEYWORD_KEY = {
        "bio-forget-corpus": "bio",
        "cyber-forget-corpus": "cyber",
    }

    def corpus_to_keywords(corpus):
        if corpus in CORPUS_TO_KEYWORD_KEY:
            return CORPUS_TO_KEYWORD_KEY[corpus]
        if corpus.startswith("tofu-"):
            return "tofu"
        if corpus.startswith("muse-books"):
            return "muse-books"
        if corpus.startswith("muse-news"):
            return "muse-news"
        raise NotImplementedError(f"Add your keywords for {corpus} to the keywords.json file.")

    with open(SCRIPT_DIR / "data" / "keywords.json", "r") as f:
        keywords = json.load(f)

    forget_data_list, retain_data_list = _get_data_base(
        forget_corpora, retain_corpora, batch_size=batch_size, min_len=min_len,
        tokenizer=tokenizer, chunk_sizes=chunk_sizes,
    )

    return (
        [keywords[corpus_to_keywords(c)] for c in forget_corpora],
        forget_data_list,
        retain_data_list,
    )

def run_rsv(
    updated_model,
    frozen_model,
    tokenizer,
    keywords_list,
    forget_data_list,
    retain_data_list,
    args,
):
    assert len(keywords_list) == len(forget_data_list) == len(retain_data_list)

    updated_model = _apply_lora_if_enabled(updated_model, args)
    updated_model = updated_model.train()

    if args.use_peft:
        params = [p for p in updated_model.parameters() if p.requires_grad]
    else:
        params = get_params(updated_model, args.layer_ids, args.param_ids)

    if len(params) == 0:
        raise ValueError("No trainable parameters found. Check PEFT config or selected params.")

    optimizer = AdamW(params, lr=args.lr)

    frozen_module = _get_layers_container(frozen_model)[args.layer_id]
    updated_module = _get_layers_container(updated_model)[args.layer_id]

    model_device = _get_model_device(updated_model)

    # Get steering vectors
    steering_vectors_list = [[] for _ in range(len(keywords_list))]
    for i in range(len(steering_vectors_list)):
        steering_vectors_list[i].append(
            get_random_vector(
                model=frozen_model,
                tokenizer=tokenizer,
                module=frozen_module
            )
        )

    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    if args.max_num_batches > max_possible:
        args.max_num_batches = max_possible
    num_batches = args.max_num_batches
    loss_history = {i: {"total": [], "unlearn": [], "retain": []} for i in range(len(forget_data_list))}
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    for idx in range(num_batches):
        topic_idx = idx % len(keywords_list)
        batch_idx = idx // len(keywords_list)

        steering_vecs = steering_vectors_list[topic_idx]
        steering_vec_idx = np.random.choice(len(steering_vecs))
        steering_vec = steering_vecs[steering_vec_idx]

        unlearn_batch = forget_data_list[topic_idx][batch_idx]
        retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]

        max_length = args.max_lengths[topic_idx]
        # Unlearning loss
        unlearn_inputs = tokenizer(unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model_device)
        updated_activations = forward_with_cache(
            updated_model, unlearn_inputs, module=updated_module, no_grad=False
        ).to(model_device)

        frozen_activations = forward_with_cache(
            frozen_model, unlearn_inputs, module=frozen_module, no_grad=True
        ).to(model_device)

        frozen_activations += args.steering_coeff_list[topic_idx] * steering_vec

        frozen_activations.to(model_device)

        unlearn_loss = torch.nn.functional.mse_loss(
            updated_activations, frozen_activations
        )

        # Retain loss
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model_device)
        
        updated_retain_activations = forward_with_cache(
            updated_model, retain_inputs, module=updated_module, no_grad=False
        ).to(model_device)
        
        frozen_retain_activations = forward_with_cache(
            frozen_model, retain_inputs, module=frozen_module, no_grad=True
        ).to(model_device)

        # add random noise
        noise = torch.randn_like(frozen_retain_activations) * args.nu
        noise_frozen_retain_activations = frozen_retain_activations + noise

        retain_loss = torch.nn.functional.mse_loss(
            updated_retain_activations, noise_frozen_retain_activations
        )

        retain_loss *= args.alpha

        # Update model
        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[topic_idx]["total"].append(loss.item())
        loss_history[topic_idx]["unlearn"].append(unlearn_loss.item())
        loss_history[topic_idx]["retain"].append(retain_loss.item())

        # param_change = params[0].grad.abs().mean().item() if params[0].grad is not None else 0.0
        # print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {param_change:.4g}")
        # pbar.update(1)
        
    tokenizer.truncation_side = truncation_side
    
    if args.nu > 0.0:
        path = str(SCRIPT_DIR / f"checkpoints/rsv/{args.bench_label}/{args.model_name}_nu")
    else:
        path = str(SCRIPT_DIR / f"checkpoints/rsv/{args.bench_label}/{args.model_name}")

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
    apply_benchmark_config(args, cli_args.benchmark, cli_args.tofu_split, cli_args.muse_corpus, cli_args.blur_task)
    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    frozen_model, tokenizer = load_model(args.model_path)
    updated_model, _ = load_model(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    keywords_list, forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        chunk_sizes=args.max_lengths,
    )
    run_rsv(
        updated_model,
        frozen_model,
        tokenizer,
        keywords_list,
        forget_data_list,
        retain_data_list,
        args,
    )
    clear_cuda_cache()
