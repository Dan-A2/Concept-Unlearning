import gc
import hashlib
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json
import re
import sys
import time
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cofi_cache"
_BENCHMARK = None   # Set by main() before any cache operations
_CHESS_SEED = None  # Optional, set by main(); makes CHess Rademacher draws deterministic

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class _StripVtWrapper(torch.nn.Module):
    """Wraps a P-tuning PeftModel and strips virtual-token logit positions.

    P-tuning prepends `num_virtual_tokens` learned tokens to every input, so
    the model outputs `T + num_virtual_tokens` logit positions for a T-token
    input.  All metric functions in this file assume logits and input_ids have
    the same sequence length, so we strip the leading virtual positions here
    once rather than patching every call site.
    """

    def __init__(self, peft_model: torch.nn.Module, num_virtual_tokens: int):
        super().__init__()
        self._peft = peft_model
        self._nvt = num_virtual_tokens

    def forward(self, *args, **kwargs):
        out = self._peft(*args, **kwargs)
        out.logits = out.logits[:, self._nvt:, :]
        return out

    def named_parameters(self, *args, **kwargs):
        # Delegate to the raw base model so parameter names match the cached
        # base-model dictionaries (e.g. 'model.layers.0...') rather than the
        # PEFT-prefixed names ('_peft.base_model.model.layers.0...').
        # The prompt-encoder parameters are excluded intentionally — they are
        # never part of target_modules and must not appear in CoFi/CHess dicts.
        return self._peft.base_model.named_parameters(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._peft, name)


def _ts():
    """Compact timestamp for log lines."""
    return time.strftime("%H:%M:%S")


def _mem_report():
    """Return a short string describing current memory usage."""
    parts = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            resv = torch.cuda.memory_reserved(i) / (1024 ** 3)
            parts.append(f"GPU{i}: {alloc:.1f}/{resv:.1f}GB")
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # GB on Linux
    parts.append(f"CPU-RSS: {rss:.1f}GB")
    return "  ".join(parts)


def _flush_memory():
    """Aggressively free GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _chunk_long_texts(texts: List[str], tokenizer, max_length: int) -> List[str]:
    """Split documents whose token length exceeds max_length into
    non-overlapping windows of max_length tokens. Shorter documents are
    kept verbatim. Used so per-batch curvature metrics (CoFi/CHess) cover
    the full corpus on long-document benchmarks (e.g., MUSE-Books) instead
    of the first max_length tokens of each document.
    """
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) <= max_length:
            out.append(t)
            continue
        for i in range(0, len(ids), max_length):
            out.append(tokenizer.decode(ids[i:i + max_length], skip_special_tokens=True))
    return out


def load_texts_from_local_corpus(corpus_path: Path) -> List[str]:
    dataset = load_from_disk(str(corpus_path))
    if isinstance(dataset, dict):
        dataset = dataset.get("train", next(iter(dataset.values())))
    if "text" not in dataset.column_names:
        raise ValueError(f"{corpus_path.name} has no 'text' column.")
    return [row["text"] for row in dataset if isinstance(row["text"], str) and row["text"].strip()]


def load_hf_texts(hf_dataset_name: str, config: Optional[str], split: str) -> List[str]:
    dataset = (
        load_dataset(hf_dataset_name, config, split=split)
        if config else load_dataset(hf_dataset_name, split=split)
    )
    return [
        row.get("text", row.get("question", ""))
        for row in dataset
        if isinstance(row.get("text", row.get("question", "")), str)
        and row.get("text", row.get("question", "")).strip()
    ]


def load_tofu_texts(split_name: str) -> List[str]:
    raw_data = load_dataset("locuslab/TOFU", split_name, split="train")
    return [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in raw_data]


def get_benchmark_corpora(
    benchmark: str,
    tofu_split: Optional[str] = None,
    muse_corpus: Optional[str] = None,
    blur_task: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """Return (corpora_dict, forget_names, retain_names) for a given benchmark."""
    if benchmark == "wmdp":
        corpora = {
            "bio-forget":   load_texts_from_local_corpus(DATA_DIR / "bio-forget-corpus"),
            "cyber-forget": load_texts_from_local_corpus(DATA_DIR / "cyber-forget-corpus"),
            "bio-retain":   load_hf_texts("cais/wmdp-corpora", "bio-retain-corpus", "train"),
            "cyber-retain": load_hf_texts("cais/wmdp-corpora", "cyber-retain-corpus", "train"),
            "retain2":      load_hf_texts("wikitext", "wikitext-2-raw-v1", "train"),
        }
        forget_names = ["bio-forget", "cyber-forget"]
        retain_names = ["bio-retain", "cyber-retain", "retain2"]

    elif benchmark == "tofu":
        from baselines.benchmarks import TOFU_SPLITS
        if tofu_split not in TOFU_SPLITS:
            raise ValueError(f"--tofu_split must be one of {list(TOFU_SPLITS)}, got {tofu_split}")
        retain_split = TOFU_SPLITS[tofu_split]
        corpora = {
            f"tofu-{tofu_split}":   load_tofu_texts(tofu_split),
            f"tofu-{retain_split}": load_tofu_texts(retain_split),
            "wikitext":             load_hf_texts("wikitext", "wikitext-2-raw-v1", "train"),
        }
        forget_names = [f"tofu-{tofu_split}"]
        retain_names = [f"tofu-{retain_split}", "wikitext"]

    elif benchmark == "muse":
        if muse_corpus not in ("news", "books"):
            raise ValueError(f"--muse-corpus must be 'news' or 'books', got {muse_corpus}")
        hub_id = f"muse-bench/MUSE-{muse_corpus.capitalize()}"
        corpora = {
            f"{muse_corpus}-forget":  load_hf_texts(hub_id, "raw", "forget"),
            f"{muse_corpus}-retain1": load_hf_texts(hub_id, "raw", "retain1"),
            f"{muse_corpus}-retain2": load_hf_texts(hub_id, "raw", "retain2"),
            "wikitext":               load_hf_texts("wikitext", "wikitext-2-raw-v1", "train"),
        }
        forget_names = [f"{muse_corpus}-forget"]
        retain_names = [f"{muse_corpus}-retain1", f"{muse_corpus}-retain2", "wikitext"]

    elif benchmark == "blur":
        if blur_task not in ("rwku", "whp"):
            raise ValueError(f"--blur-task must be 'rwku' or 'whp', got {blur_task}")
        corpora = {
            f"{blur_task}-forget": load_hf_texts("forgelab/BLUR", f"{blur_task}_forget", "train"),
            f"{blur_task}-retain": load_hf_texts("forgelab/BLUR", f"{blur_task}_retain", "train"),
            "wikitext":            load_hf_texts("wikitext", "wikitext-2-raw-v1", "train"),
        }
        forget_names = [f"{blur_task}-forget"]
        retain_names = [f"{blur_task}-retain", "wikitext"]

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return corpora, forget_names, retain_names


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_input_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model_and_tokenizer(
    model_path: str,
    torch_dtype: torch.dtype,
    device_map: Union[str, torch.device, None] = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"  [{_ts()}] Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        import json
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        base_model_name = cfg.get("base_model_name_or_path", "HuggingFaceH4/zephyr-7b-beta")
        print(f"  [{_ts()}] Adapter detected — loading base model {base_model_name} ...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch_dtype, trust_remote_code=True,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base, model_path)
        peft_type = cfg.get("peft_type", "LORA").upper()
        MERGEABLE = {"LORA", "LOHA", "LOKR", "ADALORA", "IA3", "VERA", "BONE"}
        if peft_type in MERGEABLE:
            print(f"  [{_ts()}] Merging adapter ({peft_type}) ...")
            model = model.merge_and_unload()
        else:
            num_vt = getattr(model.active_peft_config, "num_virtual_tokens", 0)
            print(
                f"  [{_ts()}] Adapter type {peft_type} does not support merging — "
                f"keeping as PEFT model (num_virtual_tokens={num_vt})."
            )
            if num_vt > 0:
                model = _StripVtWrapper(model, num_vt)
    else:
        print(f"  [{_ts()}] Loading model weights ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch_dtype, trust_remote_code=True,
            device_map=device_map,
        )

    print(f"  [{_ts()}] Model loaded. {_mem_report()}")
    return model, tokenizer


def get_target_params(model, target_modules):
    return [
        (name, param)
        for name, param in model.named_parameters()
        if any(m in name for m in target_modules)
    ]


# ---------------------------------------------------------------------------
# Perplexity  (concatenate + sliding-window, matching test.py logic)
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    stride: int,
    device: torch.device,
    max_tokens: Optional[int] = None,
) -> Tuple[float, int, int]:
    """
    Concatenate all texts into a single corpus, tokenize once, then evaluate
    perplexity with a strided sliding window.

    Returns (perplexity, total_scored_tokens, num_input_texts).
    """
    model.eval()

    corpus = "\n\n".join(texts)
    encodings = tokenizer(corpus, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]

    vocab_size = model.config.vocab_size
    input_ids = input_ids.clamp(0, vocab_size - 1)

    seq_len = input_ids.size(0)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_toks = 0

    prev_end = 0
    start_positions = list(range(0, seq_len, stride))

    with torch.no_grad():
        for i, begin in enumerate(start_positions):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end

            chunk_ids = input_ids[begin:end].unsqueeze(0).to(device)

            target_ids = chunk_ids.clone()
            target_ids[:, :-target_len] = -100

            outputs = model(chunk_ids)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous().float().view(-1, logits.size(-1))
            shift_labels = target_ids[:, 1:].contiguous().view(-1).to(shift_logits.device)

            shift_labels[(shift_labels != -100) & (shift_labels >= shift_logits.size(-1))] = -100
            shift_labels[(shift_labels != -100) & (shift_labels < 0)] = -100

            nll = loss_fn(shift_logits, shift_labels)
            n = (shift_labels != -100).sum().item()

            total_nll += nll.item()
            total_toks += n
            prev_end = end

            # Free intermediate tensors explicitly
            del chunk_ids, target_ids, outputs, logits, shift_logits, shift_labels, nll

            if (i + 1) % 20 == 0 or end == seq_len:
                cur_ppl = math.exp(total_nll / total_toks) if total_toks > 0 else float("nan")
                print(
                    f"    [{i+1:>4}/{len(start_positions)}]  "
                    f"tokens={total_toks:>8,}  running PPL={cur_ppl:.2f}",
                )

    if total_toks == 0:
        return float("nan"), 0, len(texts)

    ppl = math.exp(total_nll / total_toks)
    return ppl, total_toks, len(texts)


def _autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------------
# CoFi — Log-Space Diagonal Fisher
# ---------------------------------------------------------------------------

def compute_cofi(model, tokenizer, texts, max_length, batch_size, target_params):
    model.eval()
    for _, param in target_params:
        param.requires_grad_(True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    input_device = _get_input_device(model)
    amp_dtype = _autocast_dtype()

    # Accumulate directly on CPU to save GPU memory
    accum = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu")
             for name, p in target_params}
    n_steps = 0

    scale_c = 1e6
    lambda_val = 1e-5

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
        enc = tokenizer(
            texts[i:i + batch_size], return_tensors="pt",
            padding=True, truncation=True, max_length=max_length
        )

        input_ids = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_device)
        if input_ids.numel() == 0:
            continue
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            shift_labels = shift_labels.masked_fill(shift_labels >= shift_logits.size(-1), -100)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        loss.backward()

        for name, param in target_params:
            if param.grad is not None:
                g = param.grad.detach().float()
                g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
                g.div_(math.sqrt(scale_c))
                # Move to CPU before accumulating
                accum[name] += g.pow_(2).cpu()
                param.grad = None
        n_steps += 1

        del input_ids, attention_mask, labels, enc, loss, logits, shift_logits, shift_labels

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            print(f"      [{_ts()}] CoFi batch {batch_idx+1}/{total_batches}")

    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()

    if n_steps > 0:
        for name in accum:
            accum[name].div_(n_steps)
            accum[name].add_(lambda_val / scale_c)
            accum[name].log_().add_(math.log(scale_c))

    return accum


# ---------------------------------------------------------------------------
# CHess — Log-Space Diagonal Hessian via Hutchinson Finite Difference
# ---------------------------------------------------------------------------

def _chess_probe_seed(base_seed, corpus_name, batch_idx, h_idx):
    """Stable 32-bit seed from (base, corpus, batch, hutchinson_index)."""
    s = f"{base_seed}|{corpus_name}|{batch_idx}|{h_idx}"
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def compute_chess(model, tokenizer, texts, max_length, batch_size, target_params, n_hutchinson,
                  use_checkpointing: bool = False, base_seed=None, corpus_name=None):
    """Hutchinson finite-difference Hessian diagonal.

    Per Hutchinson sample we do 2 forward+backward passes (for g(θ+εz) and g(θ-εz)),
    so the total cost is ~2*n_hutchinson*batches forward+backward passes.

    Setting use_checkpointing=False trades GPU memory for ~1.3-1.5x speedup by
    avoiding the recompute pass during backward.
    """
    model.eval()
    for _, param in target_params:
        param.requires_grad_(True)
    if use_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        # Make sure it's off in case it was left enabled by a previous call.
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    input_device = _get_input_device(model)
    amp_dtype = _autocast_dtype()

    eps = 1e-3

    # Accumulate directly on CPU to save GPU memory
    accum = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu")
             for name, p in target_params}
    n_steps = 0

    scale_c = 1e6
    lambda_val = 1e-5

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
        enc = tokenizer(
            texts[i:i + batch_size], return_tensors="pt",
            padding=True, truncation=True, max_length=max_length
        )
        input_ids = enc["input_ids"].to(input_device)
        attention_mask = enc["attention_mask"].to(input_device)
        if input_ids.numel() == 0:
            continue
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        for h_idx in range(n_hutchinson):
            # Generate z_dict on CPU to avoid holding large Rademacher vectors on GPU.
            # Perturbations are applied one parameter at a time via a temporary .to(device) call.
            if base_seed is not None:
                gen = torch.Generator(device="cpu").manual_seed(
                    _chess_probe_seed(base_seed, corpus_name, batch_idx, h_idx)
                )
                z_dict = {
                    name: (torch.randint(0, 2, p.shape, generator=gen) * 2 - 1).to(p.dtype)
                    for name, p in target_params
                }
            else:
                z_dict = {
                    name: (torch.randint(0, 2, p.shape, dtype=p.dtype) * 2 - 1)
                    for name, p in target_params
                }

            for name, p in target_params:
                p.data.add_(z_dict[name].to(p.device), alpha=eps)

            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
                shift_labels = shift_labels.masked_fill(shift_labels >= shift_logits.size(-1), -100)
                loss_plus = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
                )
            loss_plus.backward()

            # Move g_plus to CPU immediately to free GPU memory before the second backward.
            g_plus = {}
            for name, p in target_params:
                if p.grad is not None:
                    g_plus[name] = p.grad.detach().float().cpu()
                    p.grad = None

            for name, p in target_params:
                p.data.sub_(z_dict[name].to(p.device), alpha=2*eps)

            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
                shift_labels = shift_labels.masked_fill(shift_labels >= shift_logits.size(-1), -100)
                loss_minus = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
                )
            loss_minus.backward()

            # HVP computation done on CPU since g_plus, z_dict, and accum are all on CPU.
            for name, p in target_params:
                if p.grad is not None and name in g_plus:
                    hvp = g_plus.pop(name)  # float32 CPU tensor
                    hvp.sub_(p.grad.float().cpu()).div_(2 * eps)
                    hvp.mul_(z_dict[name].float())
                    hvp.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
                    hvp.div_(scale_c)
                    accum[name].add_(hvp)

                p.data.add_(z_dict[name].to(p.device), alpha=eps)
                p.grad = None

            del loss_plus, loss_minus, g_plus, outputs, shift_logits, shift_labels, z_dict

        n_steps += 1
        del input_ids, attention_mask, labels, enc

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            print(f"      [{_ts()}] CHess batch {batch_idx+1}/{total_batches}")

    model.zero_grad(set_to_none=True)
    if use_checkpointing:
        model.gradient_checkpointing_disable()

    if n_steps > 0:
        for name in accum:
            accum[name].div_(n_steps * n_hutchinson)
            accum[name].abs_()
            accum[name].add_(lambda_val / scale_c)
            accum[name].log_().add_(math.log(scale_c))

    return accum


# ---------------------------------------------------------------------------
# Frobenius norm utilities (Normalized by 1 / sqrt(N))
# ---------------------------------------------------------------------------

def frob_norm(d):
    n_params = sum(v.numel() for v in d.values())
    if n_params == 0:
        return 0.0
    raw_norm = sum(v.double().pow(2).sum().item() for v in d.values()) ** 0.5
    return raw_norm / (n_params ** 0.5)


def frob_norm_drop(d_orig, d_unlearned):
    """Compute norm drop by streaming key-by-key to avoid holding both full dicts."""
    n_params = sum(v.numel() for v in d_orig.values())
    if n_params == 0:
        return 0.0
    sum_sq = 0.0
    for k in d_orig:
        diff = d_orig[k].double() - d_unlearned[k].double()
        sum_sq += diff.pow(2).sum().item()
        del diff
    return sum_sq ** 0.5 / (n_params ** 0.5)


def _cache_path(label: str, corpus: str, metric: str, ext: str = ".pt") -> Path:
    """Return a nested cache path: CACHE_DIR/method/benchmark/model/corpus__metric.ext

    Mirrors the checkpoints directory structure (method/benchmark/model).
    _BENCHMARK must be set (by main()) before calling this function.
    """
    if _BENCHMARK is None:
        raise RuntimeError("_BENCHMARK not set — call main() first")

    if label.startswith("base__"):
        method_dir = "base"
        model_dir = label[len("base__"):]
    elif "/" in label:
        method_dir, model_dir = label.split("/", 1)
    else:
        method_dir = "other"
        model_dir = label

    safe_model = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_dir)
    suffix = f"_seed{_CHESS_SEED}" if (metric == "chess" and _CHESS_SEED is not None) else ""
    return CACHE_DIR / method_dir / _BENCHMARK / safe_model / f"{corpus}__{metric}{suffix}{ext}"


def _json_cache_path(label: str, corpus: str, metric: str) -> Path:
    return _cache_path(label, corpus, metric, ext=".json")


def _pt_cache_path(label: str, corpus: str, metric: str) -> Path:
    return _cache_path(label, corpus, metric, ext=".pt")


def _save_pt(d: dict, path: Path):
    """Save a tensor dict as half-precision .pt (used for base model only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() for k, v in d.items()}, path)


def _load_pt(path: Path) -> dict:
    """Load a tensor dict from .pt, converting back to float32."""
    return {k: v.float() for k, v in
            torch.load(path, map_location="cpu", weights_only=True).items()}


def _save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Output Logger
# ---------------------------------------------------------------------------
class OutputLogger:
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # auto-flush after every write

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--checkpoints-dir", default=str(PROJECT_ROOT / "checkpoints"))
    p.add_argument("--checkpoint", nargs=2, metavar=("PATH", "LABEL"), action="append", default=[])
    p.add_argument("--auto-discover", action="store_true")
    p.add_argument("--orig-max-samples", type=int, default=200)
    p.add_argument("--unlearned-max-samples", type=int, default=200,
                   help="Number of samples for CoFi/CHess on unlearned models (default: 200).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for CoFi/CHess gradient computation.")
    p.add_argument("--batch-size-grad", type=int, default=4)
    p.add_argument("--ppl-max-length", type=int, default=2048,
                   help="Sliding window size in tokens for perplexity (default: 2048).")
    p.add_argument("--ppl-stride", type=int, default=512,
                   help="Stride between windows for perplexity (default: 512).")
    p.add_argument("--ppl-max-tokens", type=int, default=None,
                   help="Truncate each corpus to this many tokens for perplexity.")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Max sequence length for CoFi/CHess tokenization (default: 1024).")
    p.add_argument("--n-hutchinson", type=int, default=2,
                   help="Number of Hutchinson probes per sample for CHess. "
                        "Each probe costs 2 forward+backward passes, so this is the "
                        "biggest knob for CHess speed (default: 2).")
    p.add_argument("--chess-batch-size", type=int, default=None,
                   help="Override batch size used during CHess gradient passes "
                        "(default: same as --batch-size-grad). Bigger batches are "
                        "much faster but need more GPU memory.")
    p.add_argument("--chess-max-length", type=int, default=None,
                   help="Override max sequence length used for CHess tokenization "
                        "(default: same as --max-length). Smaller is much faster.")
    p.add_argument("--chess-max-samples", type=int, default=None,
                   help="Override number of samples used for CHess on both base "
                        "and unlearned models (default: same as --unlearned-max-samples).")
    p.add_argument("--chess-seed", type=int, default=None,
                   help="If set, use deterministic Rademacher probes for CHess "
                        "derived from (seed, corpus, batch, h_idx). Two runs over "
                        "the same data with the same seed produce identical Hessian "
                        "estimates → NULL CHess drops to ~0. Cached separately under "
                        "..._seed<N>.pt to avoid colliding with unseeded runs.")
    p.add_argument("--chess-grad-checkpoint", action="store_true",
                   help="Re-enable gradient checkpointing inside CHess. By default "
                        "checkpointing is OFF for CHess (~1.3-1.5x faster) since CHess "
                        "uses small batches; turn it back on if you OOM.")
    p.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    p.add_argument("--skip-chess", action="store_true")
    p.add_argument("--skip-perplexity", action="store_true")
    p.add_argument("--benchmark", type=str, default="wmdp",
                   choices=["wmdp", "tofu", "muse", "blur"],
                   help="Evaluation benchmark (default: wmdp).")
    p.add_argument("--tofu-split", type=str, default=None,
                   choices=["forget01", "forget05", "forget10"],
                   help="TOFU forget split (required when --benchmark tofu).")
    p.add_argument("--muse-corpus", type=str, default=None,
                   choices=["news", "books"],
                   help="MUSE corpus (required when --benchmark muse).")
    p.add_argument("--blur-task", type=str, default=None,
                   choices=["rwku", "whp"],
                   help="BLUR task (required when --benchmark blur).")
    p.add_argument("--model-filter", type=str, default=None,
                   help="Only discover checkpoints whose name contains this string.")
    return p.parse_args()


def discover_checkpoints(checkpoints_dir, benchmark_label=None, model_filter=None):
    """Discover checkpoints in the 3-level structure: {method}/{benchmark}/{model_variant}."""
    result = []
    for method_dir in sorted(Path(checkpoints_dir).iterdir()):
        if not method_dir.is_dir():
            continue
        for bench_dir in sorted(method_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            if benchmark_label and bench_dir.name != benchmark_label:
                continue
            for ckpt_dir in sorted(bench_dir.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                if model_filter and model_filter not in ckpt_dir.name:
                    continue
                label = f"{method_dir.name}/{ckpt_dir.name}"
                result.append((label, str(ckpt_dir)))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    # 1. Clean the base model name
    model_name = args.base_model.split("/")[-1]
    base_label = f"base__{model_name}"

    # 2. Extract unique method names from the checkpoint labels
    methods = []
    for _, label in args.checkpoint:
        method = label.split("/")[0]
        if method not in methods:
            methods.append(method)

    method_str = "_".join(methods) if methods else "base_only"
    if len(method_str) > 50:
        method_str = "multiple_methods"

    # 3. Create the nested directory structure
    # log_filename = f"{method_str}.txt"
    # log_dir = PROJECT_ROOT / "test" / model_name
    # log_dir.mkdir(parents=True, exist_ok=True)

    # 4. Initialize the logger
    # sys.stdout = OutputLogger(log_dir / log_filename)

    torch_dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )

    # Determine the benchmark label used in checkpoint paths
    bench_label_map = {
        "wmdp": "wmdp",
        "tofu": f"tofu-{args.tofu_split}" if args.tofu_split else None,
        "muse": f"muse-{args.muse_corpus}" if args.muse_corpus else None,
        "blur": f"blur-{args.blur_task}" if args.blur_task else None,
    }
    bench_label = bench_label_map.get(args.benchmark)

    global _BENCHMARK, _CHESS_SEED
    _BENCHMARK = bench_label
    _CHESS_SEED = args.chess_seed

    unlearned: List[Tuple[str, str]] = []
    if args.auto_discover:
        unlearned += discover_checkpoints(
            args.checkpoints_dir,
            benchmark_label=bench_label,
            model_filter=args.model_filter,
        )
    unlearned += [(label, path) for path, label in args.checkpoint]

    # Resolve CHess-specific overrides (fall back to the shared values).
    chess_batch_size = args.chess_batch_size if args.chess_batch_size is not None else args.batch_size_grad
    chess_max_length = args.chess_max_length if args.chess_max_length is not None else args.max_length
    chess_max_samples = args.chess_max_samples if args.chess_max_samples is not None else args.orig_max_samples
    chess_use_checkpointing = bool(args.chess_grad_checkpoint)

    if not args.skip_chess:
        print(f"[{_ts()}] CHess settings: n_hutchinson={args.n_hutchinson}  "
              f"batch={chess_batch_size}  max_len={chess_max_length}  "
              f"max_samples={chess_max_samples}  grad_ckpt={chess_use_checkpointing}")

    print(f"[{_ts()}] Loading datasets (benchmark={args.benchmark})...")
    corpora, forget_names, retain_names = get_benchmark_corpora(
        args.benchmark, args.tofu_split, args.muse_corpus, args.blur_task
    )

    for name in list(corpora.keys()):
        if len(corpora[name]) > args.orig_max_samples:
            corpora[name] = random.sample(corpora[name], args.orig_max_samples)

    for name, texts in corpora.items():
        print(f"  {name:15s}: {len(texts):6d} samples")

    print(f"[{_ts()}] Datasets loaded. {_mem_report()}")

    all_corpora_names = list(corpora.keys())

    # ====================================================================
    # STEP 1 — Original (base) model
    # ====================================================================
    print("\n" + "=" * 70)
    print(f"[{_ts()}] ORIGINAL MODEL")
    print("=" * 70)

    # Base tensor dicts (.pt) are saved to disk so we never have to
    # recompute them.  Only the base model uses .pt files; unlearned
    # models only store scalar JSON results.
    missing_cofi_pt = [
        c for c in all_corpora_names
        if not _pt_cache_path(base_label, c, "cofi").exists()
    ]
    missing_chess_pt = (
        [] if args.skip_chess else
        [c for c in all_corpora_names
         if not _pt_cache_path(base_label, c, "chess").exists()]
    )
    missing_cofi_json = [
        c for c in all_corpora_names
        if not _json_cache_path(base_label, c, "cofi").exists()
    ]
    missing_chess_json = (
        [] if args.skip_chess else
        [c for c in all_corpora_names
         if not _json_cache_path(base_label, c, "chess").exists()]
    )
    missing_ppl = (
        [] if args.skip_perplexity else
        [c for c in all_corpora_names
         if not _json_cache_path(base_label, c, "ppl").exists()]
    )

    need_model = bool(missing_cofi_pt or missing_chess_pt or missing_ppl)

    n = len(all_corpora_names)
    print(f"[{_ts()}] Base cache status:")
    print(f"    CoFi .pt : {n - len(missing_cofi_pt)}/{n} cached"
          + (f"  (missing: {missing_cofi_pt})" if missing_cofi_pt else ""))
    if not args.skip_chess:
        print(f"    CHess.pt : {n - len(missing_chess_pt)}/{n} cached"
              + (f"  (missing: {missing_chess_pt})" if missing_chess_pt else ""))
    if not args.skip_perplexity:
        print(f"    PPL      : {n - len(missing_ppl)}/{n} cached"
              + (f"  (missing: {missing_ppl})" if missing_ppl else ""))

    if not need_model:
        print(f"[{_ts()}] All base metrics cached — not loading base model.")
    else:
        print(f"[{_ts()}] Loading base model: {args.base_model}")
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, torch_dtype, device_map="auto"
        )
        model.eval()
        target_params = get_target_params(model, args.target_modules)
        input_device = _get_input_device(model)

        if missing_ppl:
            print(f"\n  [{_ts()}] Perplexity (concatenated sliding-window):")
            for corpus_name, texts in corpora.items():
                ppl_json = _json_cache_path(base_label, corpus_name, "ppl")
                if ppl_json.exists():
                    cached = _load_json(ppl_json)
                    print(f"    {corpus_name:15s}: PPL = {cached['ppl']:.4f}  [cached]")
                    continue
                print(f"    [{_ts()}] Starting PPL for '{corpus_name}' ...")
                ppl, n_toks, n_docs = compute_perplexity(
                    model, tokenizer, texts[:100],
                    max_length=args.ppl_max_length,
                    stride=args.ppl_stride,
                    device=input_device,
                    max_tokens=args.ppl_max_tokens,
                )
                _save_json({"ppl": ppl, "n_tokens": n_toks, "n_docs": n_docs}, ppl_json)
                print(f"    {corpus_name:15s}: PPL = {ppl:.4f}  ({n_toks:,} tokens, {n_docs} docs)")
            _flush_memory()

        if missing_cofi_pt:
            print(f"\n  [{_ts()}] Computing CoFi for base model "
                  f"({len(missing_cofi_pt)} corpora):")
            for corpus_name in missing_cofi_pt:
                texts = corpora[corpus_name]
                grad_texts = _chunk_long_texts(texts, tokenizer, args.max_length)[:args.orig_max_samples]
                print(f"    [{_ts()}] CoFi on '{corpus_name}' ({len(grad_texts)} samples) ...")
                result = compute_cofi(
                    model, tokenizer, grad_texts,
                    args.max_length, args.batch_size_grad, target_params
                )
                _save_pt(result, _pt_cache_path(base_label, corpus_name, "cofi"))
                norm_val = frob_norm(result)
                _save_json({"frob_norm": norm_val}, _json_cache_path(base_label, corpus_name, "cofi"))
                del result
                _flush_memory()
                print(f"    {corpus_name:15s}: ||F||_F = {norm_val:.6f}  {_mem_report()}")

        if missing_chess_pt:
            print(f"\n  [{_ts()}] Computing CHess for base model "
                  f"({len(missing_chess_pt)} corpora):")
            for corpus_name in missing_chess_pt:
                texts = corpora[corpus_name]
                grad_texts = _chunk_long_texts(texts, tokenizer, chess_max_length)[:args.orig_max_samples]
                print(f"    [{_ts()}] CHess on '{corpus_name}' ({len(grad_texts)} samples) ...")
                result = compute_chess(
                    model, tokenizer, grad_texts,
                    chess_max_length, chess_batch_size, target_params, args.n_hutchinson,
                    use_checkpointing=chess_use_checkpointing,
                    base_seed=args.chess_seed, corpus_name=corpus_name,
                )
                _save_pt(result, _pt_cache_path(base_label, corpus_name, "chess"))
                norm_val = frob_norm(result)
                _save_json({"frob_norm": norm_val}, _json_cache_path(base_label, corpus_name, "chess"))
                del result
                _flush_memory()
                print(f"    {corpus_name:15s}: ||H||_F = {norm_val:.6f}  {_mem_report()}")

        # Also write any missing base frob_norm JSONs from existing .pt files
        for corpus_name in (set(missing_cofi_json) - set(missing_cofi_pt)):
            d = _load_pt(_pt_cache_path(base_label, corpus_name, "cofi"))
            _save_json({"frob_norm": frob_norm(d)}, _json_cache_path(base_label, corpus_name, "cofi"))
            del d
        for corpus_name in (set(missing_chess_json) - set(missing_chess_pt)):
            d = _load_pt(_pt_cache_path(base_label, corpus_name, "chess"))
            _save_json({"frob_norm": frob_norm(d)}, _json_cache_path(base_label, corpus_name, "chess"))
            del d

        print(f"\n  [{_ts()}] Unloading base model ...")
        del model, tokenizer, target_params
        _flush_memory()
        print(f"  [{_ts()}] Base model unloaded. {_mem_report()}")

    # ====================================================================
    # STEP 2 — Each unlearned model
    # ====================================================================
    ppl_table:  Dict[str, Dict[str, float]] = {}
    cofi_drops: Dict[str, Dict[str, float]] = {}
    chess_drops: Dict[str, Dict[str, float]] = {}

    total_models = len(unlearned)
    for model_idx, (label, model_path) in enumerate(unlearned, 1):
        print("\n" + "=" * 70)
        print(f"[{_ts()}] UNLEARNED MODEL {model_idx}/{total_models}: {label}")
        print(f"Path: {model_path}")
        print("=" * 70)

        unl_missing_cofi = [
            c for c in all_corpora_names
            if not _json_cache_path(label, c, "cofi").exists()
        ]
        unl_missing_chess = (
            [] if args.skip_chess else
            [c for c in all_corpora_names
             if not _json_cache_path(label, c, "chess").exists()]
        )
        unl_missing_ppl = (
            [] if args.skip_perplexity else
            [c for c in all_corpora_names
             if not _json_cache_path(label, c, "ppl").exists()]
        )
        all_cached = not (unl_missing_cofi or unl_missing_chess or unl_missing_ppl)

        n_tot = len(all_corpora_names)
        print(f"  [{_ts()}] Cache status:")
        print(f"    CoFi  : {n_tot - len(unl_missing_cofi)}/{n_tot} cached"
              + (f"  (missing: {unl_missing_cofi})" if unl_missing_cofi else ""))
        if not args.skip_chess:
            print(f"    CHess : {n_tot - len(unl_missing_chess)}/{n_tot} cached"
                  + (f"  (missing: {unl_missing_chess})" if unl_missing_chess else ""))
        if not args.skip_perplexity:
            print(f"    PPL   : {n_tot - len(unl_missing_ppl)}/{n_tot} cached"
                  + (f"  (missing: {unl_missing_ppl})" if unl_missing_ppl else ""))

        need_model = not all_cached

        model = tokenizer = target_params = None
        if need_model:
            model, tokenizer = load_model_and_tokenizer(
                model_path, torch_dtype, device_map="auto"
            )
            model.eval()
            target_params = get_target_params(model, args.target_modules)
        else:
            print(f"  [{_ts()}] All metrics cached — skipping model load.")

        # --- Perplexity (cached as JSON) ---
        if not args.skip_perplexity:
            ppl_table[label] = {}
            if unl_missing_ppl:
                input_device = _get_input_device(model)
                print(f"\n  [{_ts()}] Perplexity (concatenated sliding-window):")
                for corpus_name, texts in corpora.items():
                    ppl_json = _json_cache_path(label, corpus_name, "ppl")
                    if ppl_json.exists():
                        cached = _load_json(ppl_json)
                        ppl_table[label][corpus_name] = cached["ppl"]
                        print(f"    {corpus_name:15s}: PPL = {cached['ppl']:.4f}  [cached]")
                        continue
                    print(f"    [{_ts()}] Starting PPL for '{corpus_name}' ...")
                    ppl, n_toks, n_docs = compute_perplexity(
                        model, tokenizer, texts[:100],
                        max_length=args.ppl_max_length,
                        stride=args.ppl_stride,
                        device=input_device,
                        max_tokens=args.ppl_max_tokens,
                    )
                    _save_json({"ppl": ppl, "n_tokens": n_toks, "n_docs": n_docs}, ppl_json)
                    ppl_table[label][corpus_name] = ppl
                    print(f"    {corpus_name:15s}: PPL = {ppl:.4f}  ({n_toks:,} tokens, {n_docs} docs)")
                _flush_memory()
            else:
                print(f"\n  [{_ts()}] Perplexity: all cached.")
                for corpus_name in all_corpora_names:
                    cached = _load_json(_json_cache_path(label, corpus_name, "ppl"))
                    ppl_table[label][corpus_name] = cached["ppl"]

        # --- CoFi norm drops (load base .pt on demand, save scalar JSON) ---
        cofi_drops[label] = {}
        print(f"\n  [{_ts()}] CoFi norm drops (vs. base):")
        for corpus_name, texts in corpora.items():
            cofi_json = _json_cache_path(label, corpus_name, "cofi")

            if cofi_json.exists():
                drop = _load_json(cofi_json)["norm_drop"]
                print(f"    {corpus_name:15s}: [cached]  ||Δ F||_F = {drop:.6f}")
            else:
                grad_texts = _chunk_long_texts(texts, tokenizer, args.max_length)[:args.unlearned_max_samples]
                print(f"    [{_ts()}] CoFi on '{corpus_name}' ({len(grad_texts)} samples) ...")
                result = compute_cofi(
                    model, tokenizer, grad_texts,
                    args.max_length, args.batch_size_grad, target_params
                )
                base_dict = _load_pt(_pt_cache_path(base_label, corpus_name, "cofi"))
                drop = frob_norm_drop(base_dict, result)
                del base_dict, result
                _flush_memory()
                _save_json({"norm_drop": drop}, cofi_json)
                print(f"    {corpus_name:15s}: ||Δ F||_F = {drop:.6f}")
            cofi_drops[label][corpus_name] = drop

        if not args.skip_chess:
            chess_drops[label] = {}
            print(f"\n  [{_ts()}] CHess norm drops (vs. base):")
            for corpus_name, texts in corpora.items():
                chess_json = _json_cache_path(label, corpus_name, "chess")

                if chess_json.exists():
                    drop = _load_json(chess_json)["norm_drop"]
                    print(f"    {corpus_name:15s}: [cached]  ||Δ H||_F = {drop:.6f}")
                else:
                    grad_texts = _chunk_long_texts(texts, tokenizer, chess_max_length)[:args.unlearned_max_samples]
                    print(f"    [{_ts()}] CHess on '{corpus_name}' ({len(grad_texts)} samples) ...")
                    result = compute_chess(
                        model, tokenizer, grad_texts,
                        chess_max_length, chess_batch_size, target_params, args.n_hutchinson,
                        use_checkpointing=chess_use_checkpointing,
                        base_seed=args.chess_seed, corpus_name=corpus_name,
                    )
                    base_dict = _load_pt(_pt_cache_path(base_label, corpus_name, "chess"))
                    drop = frob_norm_drop(base_dict, result)
                    del base_dict, result
                    _flush_memory()
                    _save_json({"norm_drop": drop}, chess_json)
                    print(f"    {corpus_name:15s}: ||Δ H||_F = {drop:.6f}")
                chess_drops[label][corpus_name] = drop

        if model is not None:
            print(f"  [{_ts()}] Unloading model '{label}' ...")
            del model, tokenizer, target_params
            _flush_memory()
            print(f"  [{_ts()}] Model unloaded. {_mem_report()}")

    # ====================================================================
    # STEP 3 — Summary tables
    # ====================================================================
    if not unlearned:
        return

    col_w = max(len(lbl) for lbl, _ in unlearned)
    all_corpora = forget_names + retain_names

    def print_table(drops, title):
        print("\n" + "=" * 70)
        print(title)
        print("(↑ forget = better unlearning   ↓ retain = less collateral damage)")
        print("=" * 70)
        header = f"{'Model':{col_w}s} | " + " | ".join(f"{c:>15s}" for c in all_corpora)
        print(header)
        print("-" * len(header))
        for label, _ in unlearned:
            if label not in drops:
                continue
            row = f"{label:{col_w}s} | "
            row += " | ".join(f"{drops[label].get(c, float('nan')):>15.6f}" for c in all_corpora)
            print(row)

    print_table(cofi_drops,  "Normalized CoFi Frobenius Norm Drops  (1/√N)||F^o − F^u||_F")
    if not args.skip_chess:
        print_table(chess_drops, "Normalized CHess Frobenius Norm Drops  (1/√N)||H^o − H^u||_F")

    if not args.skip_perplexity and ppl_table:
        print("\n" + "=" * 70)
        print("Perplexity  (concatenated sliding-window)")
        print("=" * 70)
        header = f"{'Model':{col_w}s} | " + " | ".join(f"{c:>15s}" for c in all_corpora)
        print(header)
        print("-" * len(header))
        for label, _ in unlearned:
            if label not in ppl_table:
                continue
            row = f"{label:{col_w}s} | "
            row += " | ".join(f"{ppl_table[label].get(c, float('nan')):>15.4f}" for c in all_corpora)
            print(row)

    print(f"\n[{_ts()}] All done.")


if __name__ == "__main__":
    main()