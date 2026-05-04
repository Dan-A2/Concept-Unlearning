import os
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, load_dataset
import random
random.seed(0)

# Root directory of the project (one level above baselines/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    
#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True, 
        use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_random_vector(model, tokenizer, module, keyword="helper", dtype=torch.bfloat16):
    inputs = tokenizer(keyword, return_tensors="pt", padding=True).to(model.device)
    activations = forward_with_cache(model, inputs, module)
    # return activations
    gaussian_noise = torch.randn([1, 1, activations.shape[-1]])
    gaussian_noise = gaussian_noise.to(device=model.device, dtype=dtype)
    gaussian_noise = gaussian_noise / gaussian_noise.norm(dim=-1, keepdim=True)
    return gaussian_noise

def get_steering_vec(model, tokenizer, keyword, module, dtype=torch.bfloat16):

    # p_novice = f"You are a novice in {keyword} who often makes mistakes."
    # p_expert = f"You are a world-class expert in {keyword}."
    inputs = tokenizer(keyword, return_tensors="pt", padding=True).to(model.device)
    activations = forward_with_cache(model, inputs, module)

    # novice - expert
    # direction = activations[0:1, token_pos:, :] - activations[1:, token_pos:, :]
    direction = activations.mean(dim=1, keepdim=True)
    direction = direction.to(device=model.device, dtype=dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    return direction

def _chunk_long_docs(texts, tokenizer, chunk_size):
    """Split documents whose token length exceeds chunk_size into
    non-overlapping windows of chunk_size tokens. Shorter documents are
    kept verbatim.

    Standard token-window splitting (cf. MUSE, RMU). Document boundaries
    are preserved (no concatenation across docs).
    """
    if tokenizer is None or chunk_size is None:
        return texts
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) <= chunk_size:
            out.append(t)
            continue
        for i in range(0, len(ids), chunk_size):
            piece_ids = ids[i:i + chunk_size]
            out.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
    return out


def get_data(forget_corpora, retain_corpora, batch_size=4, min_len=0,
             tokenizer=None, chunk_sizes=None):
    """Load and batch text data for unlearning.

    When ``tokenizer`` and ``chunk_sizes`` are both provided, each long
    document (more than ``chunk_sizes[i]`` tokens) is split into
    non-overlapping token windows before batching. ``chunk_sizes`` is a
    list with one entry per topic and is applied to both forget_corpora[i]
    and retain_corpora[i]. When either is None, no chunking is performed
    and behavior is identical to the legacy version.
    """
    n_topics = len(forget_corpora)
    if chunk_sizes is None:
        chunk_sizes = [None] * n_topics

    def get_dataset(name, chunk_size):
        data = []

        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                text = x["text"]
                if len(text) > min_len:
                    data.append(text)

        elif name.startswith("tofu-"):
            split_name = name[len("tofu-"):]  # e.g. "forget01", "retain99"
            raw_data = load_dataset("locuslab/TOFU", split_name, split="train")
            for x in raw_data:
                text = f"Question: {x['question']}\nAnswer: {x['answer']}"
                if len(text) > min_len:
                    data.append(text)

        elif name.startswith("muse-"):
            # e.g. "muse-news-forget"   → hub="muse-bench/MUSE-News",  split="forget"
            # e.g. "muse-books-retain1" → hub="muse-bench/MUSE-Books", split="retain1"
            # e.g. "muse-books-retain2" → hub="muse-bench/MUSE-Books", split="retain2"
            # e.g. "muse-books-holdout" → hub="muse-bench/MUSE-Books", split="holdout"
            parts = name.split("-", 2)  # ["muse", "news"|"books", "<split>"]
            corpus = parts[1]
            split = parts[2]
            hub_id = f"muse-bench/MUSE-{corpus.capitalize()}"
            raw_data = load_dataset(hub_id, "raw", split=split)
            for x in raw_data:
                if len(x["text"]) > min_len:
                    data.append(x["text"])

        elif name.startswith("blur-"):
            # e.g. "blur-rwku-forget" → config="rwku_forget"
            # e.g. "blur-whp-retain"  → config="whp_retain"
            parts = name.split("-")  # ["blur", "rwku"|"whp", "forget"|"retain"]
            config = f"{parts[1]}_{parts[2]}"
            raw_data = load_dataset("forgelab/BLUR", config, split="train")
            for x in raw_data:
                if len(x["text"]) > min_len:
                    data.append(x["text"])

        else:
            # Load HF dataset from disk
            dataset = load_from_disk(os.path.join(PROJECT_ROOT, "data", name))

            # Pick split (usually "train")
            if isinstance(dataset, dict):
                dataset = dataset["train"]

            # Infer text column
            if "text" in dataset.column_names:
                text_key = "text"
            else:
                raise ValueError(
                    f"Cannot find text column in dataset {name}. "
                    f"Columns: {dataset.column_names}"
                )

            for x in dataset:
                text = x[text_key]
                if len(text) > min_len:
                    data.append(text)

        data = _chunk_long_docs(data, tokenizer, chunk_size)

        # Batch the data
        data = [
            data[i:i + batch_size]
            for i in range(0, len(data), batch_size)
        ]

        return data

    return (
        [get_dataset(c, chunk_sizes[i]) for i, c in enumerate(forget_corpora)],
        [get_dataset(c, chunk_sizes[i]) for i, c in enumerate(retain_corpora)],
    )


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("Cleared CUDA cache.")


def save_topic_loss_plot(loss_history, save_dir, filename="loss_by_topic.png", topic_names=None):
    os.makedirs(save_dir, exist_ok=True)

    if topic_names is None:
        topic_names = {0: "bio", 1: "cyber"}

    topics = sorted(loss_history.keys())
    if len(topics) == 0:
        return

    fig, axes = plt.subplots(len(topics), 1, figsize=(10, 4 * len(topics)), sharex=False)
    if len(topics) == 1:
        axes = [axes]

    for axis, topic_idx in zip(axes, topics):
        series = loss_history[topic_idx]
        total_vals = series.get("total", [])
        unlearn_vals = series.get("unlearn", [])
        retain_vals = series.get("retain", [])

        if len(total_vals) > 0:
            axis.plot(total_vals, label="total")
        if len(unlearn_vals) > 0:
            axis.plot(unlearn_vals, label="unlearn")
        if len(retain_vals) > 0:
            axis.plot(retain_vals, label="retain")

        topic_name = topic_names.get(topic_idx, f"topic-{topic_idx}")
        axis.set_title(f"{topic_name} loss")
        axis.set_xlabel("step")
        axis.set_ylabel("loss")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close(fig)

class RandomizedModel:
    def __init__(self, model_name_or_path: str, target_layers: List[int]):
        self.model, self.tokenizer = load_model(model_name_or_path=model_name_or_path)
        self.target_layers = target_layers
    
    
    def forward_with_injected_noise(self, input_ids, attention_mask, nu, no_grad=True):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                noise = torch.randn_like(hidden_states) * nu
                return (hidden_states + noise, *output[1:])
            else:
                noise = torch.randn_like(output) * nu
                return output + noise

        hooks = []
        for layer_id in self.target_layers:
            layer = self.model.model.layers[layer_id]
            hooks.append(layer.register_forward_hook(hook_fn))

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        for h in hooks:
            h.remove()

        return outputs


    def generate(self, prompt, max_length=15, nu=0.01, device="cuda"):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generated_ids = input_ids.clone()

        for _ in range(max_length):
            context_length = min(len(generated_ids[0]), self.model.config.max_position_embeddings)
            input_context = generated_ids[:, -context_length:]

            with torch.no_grad():
                outputs = self.forward_with_injected_noise(input_context, nu=nu)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if next_token_id[0, 0].item() == self.tokenizer.eos_token_id:
                break
        generated_text = self.tokenizer.decode(generated_ids[0])
        return generated_text[len(prompt):]


######################################
##### ATU / Obliviate utilities  #####
######################################

class EmbeddingPredictor(nn.Module):
    """MLP that projects LLM hidden states to text encoder embedding space (ATU)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = (input_dim + output_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings weighted by attention mask."""
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


def get_sensitive_token_ids(tokenizer, topics, keywords_path=None):
    """Build a deduplicated list of token IDs for sensitive keywords (Obliviate).

    Reads keyword phrases from ``data/keywords.json``, tokenises every phrase
    for the requested *topics*, and returns a sorted list of unique token IDs.
    """
    import json as _json

    if keywords_path is None:
        keywords_path = PROJECT_ROOT / "data" / "keywords.json"

    with open(keywords_path, "r") as f:
        all_keywords = _json.load(f)

    token_ids = set()
    for topic in topics:
        for phrase in all_keywords.get(topic, []):
            ids = tokenizer.encode(phrase, add_special_tokens=False)
            token_ids.update(ids)

    return sorted(token_ids)


def create_unmemorize_mask(attention_mask, start, stride, span):
    """Create binary mask selecting token positions for Obliviate unmemorization.

    Selects *span* consecutive tokens every *stride + span* positions,
    beginning *start* tokens after the first valid position in each sequence.
    """
    batch_size, seq_len = attention_mask.shape
    mask = torch.zeros_like(attention_mask)
    for b in range(batch_size):
        nonzero = (attention_mask[b] != 0).nonzero(as_tuple=True)[0]
        if len(nonzero) == 0:
            continue
        first = nonzero[0].item() + 1   # skip BOS / leading token
        idx = first + start
        while idx < seq_len and attention_mask[b, idx] != 0:
            for s in range(span):
                pos = idx + s
                if pos < seq_len and attention_mask[b, pos] != 0:
                    mask[b, pos] = 1
            idx += span + stride
    return mask