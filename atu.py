"""ATU — Align-Then-Unlearn.

Implements the algorithm from:
  "Align-then-Unlearn: Exploiting Semantic Prediction for Conceptual
   Unlearning in LLMs"
  (Spohn et al., 2025)
  https://github.com/ExplainableML/align-then-unlearn

Two-phase approach:
  1. Alignment — train an embedding predictor to map LLM hidden states
     to a text encoder's semantic space using negative cosine similarity.
  2. Unlearning — fine-tune the LLM (with LoRA) so that the predicted
     embeddings for forget data move away from the target concept
     embedding, while preserving retain-data performance via KL.

Usage:
  python atu.py --model_path HuggingFaceH4/zephyr-7b-beta
"""

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from pathlib import Path

from baselines.utils import (
    load_model, forward_with_cache, get_data,
    clear_cuda_cache, save_topic_loss_plot,
    EmbeddingPredictor, mean_pooling,
)
from baselines.benchmarks import add_benchmark_args, apply_benchmark_config
from baselines.losses import kl_frozen
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModel, AutoTokenizer as EncTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


class Args:
    # Model
    model_path = "HuggingFaceH4/zephyr-7b-beta"
    model_name = "zephyr-7b-beta"

    # Data
    retain_corpora = ["wikitext", "wikitext"]
    forget_corpora = ["bio-forget-corpus", "cyber-forget-corpus"]

    # ATU hyperparameters
    alpha = [1.0, 1.0]          # retain loss weight per topic
    text_encoder_name = "sentence-transformers/all-mpnet-base-v2"
    hook_layer = 7               # LLM layer to extract hidden states from
    alignment_lr = 1e-4          # learning rate for embedding predictor
    threshold = 0.5              # cosine-similarity threshold for unlearning
    lr = 5e-5                    # LoRA learning rate for LLM unlearning

    alignment_steps = 500
    batch_size = 4
    max_num_batches = 500

    # PEFT (LoRA) config
    use_peft = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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
    )

    updated_model = get_peft_model(updated_model, lora_config)
    updated_model.enable_input_require_grads()
    updated_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    updated_model.print_trainable_parameters()
    return updated_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run ATU (Align-Then-Unlearn) unlearning.")
    parser.add_argument("--model_path", type=str, default=Args.model_path,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short name for checkpoint dirs.")
    parser.add_argument("--text_encoder_name", type=str, default=Args.text_encoder_name,
                        help="HuggingFace text encoder for semantic embeddings.")
    parser.add_argument("--hook_layer", type=int, default=Args.hook_layer,
                        help="LLM layer to extract hidden states from.")
    parser.add_argument("--alignment_lr", type=float, default=Args.alignment_lr,
                        help="Learning rate for the alignment (predictor) phase.")
    parser.add_argument("--alignment_steps", type=int, default=Args.alignment_steps,
                        help="Number of alignment training steps.")
    parser.add_argument("--threshold", type=float, default=Args.threshold,
                        help="Cosine-similarity threshold for unlearning loss.")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="LoRA target module names.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit NF4 (QLoRA) for memory-constrained models such as Qwen3-32B.")
    add_benchmark_args(parser)
    return parser.parse_args()


def run_atu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    model_device = _get_model_device(updated_model)

    # --- Load text encoder ---
    text_encoder = AutoModel.from_pretrained(args.text_encoder_name).to(model_device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    enc_tokenizer = EncTokenizer.from_pretrained(args.text_encoder_name)

    # --- Create embedding predictor ---
    llm_hidden_dim = updated_model.config.hidden_size
    enc_hidden_dim = text_encoder.config.hidden_size
    predictor = EmbeddingPredictor(llm_hidden_dim, enc_hidden_dim).to(model_device)

    # --- Compute target concept embeddings per topic ---
    concept_embeddings = {}
    for topic_idx, topic_name in args.topic_names.items():
        enc_inputs = enc_tokenizer(
            topic_name, return_tensors="pt", padding=True, truncation=True,
        ).to(model_device)
        with torch.no_grad():
            enc_out = text_encoder(**enc_inputs)
        concept_embeddings[topic_idx] = mean_pooling(
            enc_out, enc_inputs.attention_mask
        ).squeeze(0)

    frozen_module = _get_layers_container(frozen_model)[args.hook_layer]

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    # ==================================================================
    # Phase 1: Alignment — train embedding predictor (LLM frozen)
    # ==================================================================
    predictor.train()
    updated_model.eval()
    pred_optimizer = AdamW(predictor.parameters(), lr=args.alignment_lr)

    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    align_steps = min(args.alignment_steps, max_possible)

    print(f"ATU Phase 1: Alignment ({align_steps} steps)")
    for idx in range(align_steps):
        topic_idx = idx % len(forget_data_list)
        batch_idx = idx // len(forget_data_list)
        max_length = args.max_lengths[topic_idx]

        # Alternate between forget and retain data per topic
        if batch_idx % 2 == 0:
            batch = forget_data_list[topic_idx][batch_idx // 2]
        else:
            batch = retain_data_list[topic_idx][(batch_idx // 2) % len(retain_data_list[topic_idx])]

        # LLM hidden states (frozen, no grad)
        llm_inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        with torch.no_grad():
            hidden_states = forward_with_cache(
                frozen_model, llm_inputs, frozen_module, no_grad=True,
            ).to(model_device)

        # Mean-pool hidden states
        attn_mask = llm_inputs.attention_mask.unsqueeze(-1).float()
        pooled_hidden = (hidden_states * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)

        # Text encoder target embeddings
        enc_inputs = enc_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(model_device)
        with torch.no_grad():
            enc_out = text_encoder(**enc_inputs)
        target_emb = mean_pooling(enc_out, enc_inputs.attention_mask)

        # Alignment loss: negative cosine similarity
        predicted = predictor(pooled_hidden)
        align_loss = -F.cosine_similarity(predicted, target_emb, dim=-1).mean()

        pred_optimizer.zero_grad()
        align_loss.backward()
        pred_optimizer.step()
        # pbar.update(1)

    # ==================================================================
    # Phase 2: Unlearning — fine-tune LLM (predictor frozen)
    # ==================================================================
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    updated_model = _apply_lora_if_enabled(updated_model, args)
    updated_model.train()

    params = [p for p in updated_model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise ValueError("No trainable parameters found.")

    optimizer = AdamW(params, lr=args.lr)
    updated_module = _get_layers_container(updated_model)[args.hook_layer]

    max_possible = min(len(d) for d in forget_data_list) * len(forget_data_list)
    if args.max_num_batches > max_possible:
        args.max_num_batches = max_possible

    loss_history = {i: {"total": [], "unlearn": [], "retain": []}
                    for i in range(len(forget_data_list))}

    print(f"ATU Phase 2: Unlearning ({args.max_num_batches} steps, threshold={args.threshold})")
    for idx in range(args.max_num_batches):
        topic_idx = idx % len(forget_data_list)
        batch_idx = idx // len(forget_data_list)
        max_length = args.max_lengths[topic_idx]

        # --- Unlearn loss ---
        unlearn_batch = forget_data_list[topic_idx][batch_idx]
        unlearn_inputs = tokenizer(
            unlearn_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)

        # Get hidden states with gradient flow through LoRA
        hidden_states = forward_with_cache(
            updated_model, unlearn_inputs, updated_module, no_grad=False,
        ).to(model_device)

        # Mean-pool
        attn_mask = unlearn_inputs.attention_mask.unsqueeze(-1).float()
        pooled_hidden = (hidden_states * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)

        # Predicted embedding vs target concept embedding
        predicted = predictor(pooled_hidden)
        target = concept_embeddings[topic_idx].unsqueeze(0).expand_as(predicted)

        # ReLU-gated cosine similarity: push below threshold
        cos_sim = F.cosine_similarity(predicted, target, dim=-1)
        unlearn_loss = F.relu(cos_sim - args.threshold).mean()

        # --- Retain loss (KL to frozen model) ---
        retain_batch = retain_data_list[topic_idx][batch_idx % len(retain_data_list[topic_idx])]
        retain_inputs = tokenizer(
            retain_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model_device)
        retain_loss = kl_frozen(
            updated_model=updated_model,
            frozen_model=frozen_model,
            retain_inputs=retain_inputs,
        )
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

    path = str(SCRIPT_DIR / f"checkpoints/atu/{args.bench_label}/{args.model_name}")
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
    args.text_encoder_name = cli_args.text_encoder_name
    args.hook_layer = cli_args.hook_layer
    args.alignment_lr = cli_args.alignment_lr
    args.alignment_steps = cli_args.alignment_steps
    args.threshold = cli_args.threshold
    if cli_args.lora_target_modules is not None:
        args.lora_target_modules = cli_args.lora_target_modules
    args.load_in_4bit = cli_args.load_in_4bit
    apply_benchmark_config(args, cli_args.benchmark, cli_args.tofu_split, cli_args.muse_corpus)

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    frozen_model, tokenizer = load_model(args.model_path, load_in_4bit=args.load_in_4bit)
    updated_model, _ = load_model(args.model_path, load_in_4bit=args.load_in_4bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_data_list, retain_data_list = get_data(
        args.forget_corpora, args.retain_corpora, args.batch_size,
        tokenizer=tokenizer, chunk_sizes=args.max_lengths,
    )

    run_atu(
        updated_model=updated_model,
        frozen_model=frozen_model,
        tokenizer=tokenizer,
        forget_data_list=forget_data_list,
        retain_data_list=retain_data_list,
        args=args,
    )
    clear_cuda_cache()
