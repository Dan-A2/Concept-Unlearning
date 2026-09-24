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
import zipfile
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
    muse_corpus: Optional[str] = None
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

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return corpora, forget_names, retain_names


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_input_device(model) -> torch.device:
    return next(model.parameters()).device


def _is_quantized_checkpoint(model_path) -> bool:
    """True if the checkpoint at `model_path` was saved with a bnb quant config
    (e.g. the muse-books Qwen3-32B QLoRA finetune)."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        return getattr(cfg, "quantization_config", None) is not None
    except Exception:
        return False


def _bf16_balanced_device_map(model_path, torch_dtype):
    """Plan a device_map that balances the *bf16* model across the GPUs.

    A 4-bit checkpoint loaded with device_map='auto' is planned for its ~18GB
    4-bit footprint (packed onto GPU0 first). Dequantizing then inflates every
    layer ~3.5x to bf16 *in place*, so the bf16 weights pile up lopsided and
    OOM. Planning the split from an empty bf16 skeleton (exactly what
    device_map='auto' does for a bf16 load -- the balanced layout WMDP used)
    and loading the 4-bit weights onto it keeps the post-dequant model balanced,
    leaving each GPU the head-room the CoFi/CHess grad-streaming hooks rely on.
    """
    from transformers import AutoConfig
    from accelerate import init_empty_weights, infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    # Rebuild the config WITHOUT the quantization_config key entirely; setting
    # it to None leaves an attribute that the quantizer path calls .to_dict() on.
    cfg_dict = cfg.to_dict()
    cfg_dict.pop("quantization_config", None)
    clean_cfg = type(cfg).from_dict(cfg_dict)
    with init_empty_weights():
        skel = AutoModelForCausalLM.from_config(clean_cfg, trust_remote_code=True)
    no_split = getattr(skel, "_no_split_modules", None)
    max_mem = get_balanced_memory(skel, dtype=torch_dtype,
                                  no_split_module_classes=no_split)
    dmap = infer_auto_device_map(skel, max_memory=max_mem, dtype=torch_dtype,
                                 no_split_module_classes=no_split)
    del skel
    return dmap


def _load_maybe_quantized(model_path, torch_dtype, device_map):
    """Load a checkpoint, dequantizing bnb-4bit ones to bf16 on a balanced map.

    Non-quantized checkpoints load exactly as before. Quantized ones (only the
    muse-books Qwen finetune) are loaded onto a bf16-balanced device_map and
    dequantized -- so CoFi's `requires_grad_` works and the layout matches WMDP.
    """
    if not _is_quantized_checkpoint(model_path):
        return AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch_dtype, trust_remote_code=True,
            device_map=device_map,
        )
    try:
        dmap = _bf16_balanced_device_map(model_path, torch_dtype)
    except Exception as e:
        print(f"  [{_ts()}] [warn] bf16-balanced device_map failed ({e}); "
              f"falling back to device_map={device_map!r}")
        dmap = device_map
    print(f"  [{_ts()}] Quantized (4-bit) checkpoint -- loading on a "
          f"bf16-balanced device_map, then dequantizing for CoFi/CHess ...")
    # Pass dtype=torch_dtype so model.dtype is bf16: dequantize() targets
    # model.dtype (transformers integrations/bitsandbytes.dequantize_and_replace),
    # and the fp32 default would need ~128GB (32B x 4B) and OOM. bf16 -> ~64GB.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch_dtype, trust_remote_code=True, device_map=dmap,
    )
    return model.dequantize()


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
        # 4-bit base (e.g. muse-books Qwen finetune) is dequantized to bf16 on a
        # balanced device_map so the merged LoRA has float weights CoFi can
        # differentiate, without OOMing on an unbalanced dequant.
        base = _load_maybe_quantized(base_model_name, torch_dtype, device_map)
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
        model = _load_maybe_quantized(model_path, torch_dtype, device_map)

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
    # Freeze every non-target param so backward only allocates .grad for the
    # weights we actually care about. HF models load with requires_grad=True
    # everywhere, which on large models wastes GPU memory on grad buffers for
    # embedding / lm_head / layer norms.
    target_ids = {id(p) for _, p in target_params}
    original_rg = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(id(p) in target_ids)
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
    sqrt_scale_c = math.sqrt(scale_c)

    # Stream each grad to CPU and clear it the instant autograd finishes
    # accumulating it. Without this, every target param's .grad buffer is
    # live on its GPU simultaneously during backward — on a 32B model that
    # roughly doubles the per-GPU memory footprint and pushes A100-40GB
    # over the wall. With this, peak GPU grad memory is ~one layer's worth.
    def _make_cofi_hook(name):
        target = accum[name]
        def _hook(p):
            if p.grad is None:
                return
            g = p.grad.detach().float()
            g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
            g.div_(sqrt_scale_c)
            target.add_(g.pow_(2).cpu())
            p.grad = None
        return _hook

    handles = [p.register_post_accumulate_grad_hook(_make_cofi_hook(name))
               for name, p in target_params]

    total_batches = (len(texts) + batch_size - 1) // batch_size

    try:
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
            loss.backward()  # hooks fire here, streaming grads off-GPU
            n_steps += 1

            del input_ids, attention_mask, labels, enc, loss, logits, shift_logits, shift_labels

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                print(f"      [{_ts()}] CoFi batch {batch_idx+1}/{total_batches}")
    finally:
        for h in handles:
            h.remove()

    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    for p, rg in original_rg:
        p.requires_grad_(rg)

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
    # Freeze every non-target param (see note in compute_cofi).
    target_ids = {id(p) for _, p in target_params}
    original_rg = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(id(p) in target_ids)
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

    # State shared between phase-aware hooks (see comment in compute_cofi for
    # the rationale — streams grads off-GPU during backward to bound peak GPU
    # memory at ~one layer's worth of grad buffers).
    g_plus_cpu: Dict[str, torch.Tensor] = {}
    z_dict_ref: Dict[str, torch.Tensor] = {}
    phase = ["plus"]   # mutable: "plus" or "minus"

    def _make_chess_hook(name):
        target = accum[name]
        def _hook(p):
            if p.grad is None:
                return
            g = p.grad.detach().float().cpu()
            p.grad = None
            if phase[0] == "plus":
                g_plus_cpu[name] = g
            else:
                gp = g_plus_cpu.pop(name, None)
                if gp is None:
                    return
                z = z_dict_ref[name].float()
                hvp = gp.sub_(g).div_(2 * eps).mul_(z)
                hvp.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
                hvp.div_(scale_c)
                target.add_(hvp)
        return _hook

    handles = [p.register_post_accumulate_grad_hook(_make_chess_hook(name))
               for name, p in target_params]

    total_batches = (len(texts) + batch_size - 1) // batch_size

    try:
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
                # Build the Rademacher probe (CPU, in param dtype).
                if base_seed is not None:
                    gen = torch.Generator(device="cpu").manual_seed(
                        _chess_probe_seed(base_seed, corpus_name, batch_idx, h_idx)
                    )
                    z_dict_local = {
                        name: (torch.randint(0, 2, p.shape, generator=gen) * 2 - 1).to(p.dtype)
                        for name, p in target_params
                    }
                else:
                    z_dict_local = {
                        name: (torch.randint(0, 2, p.shape, dtype=p.dtype) * 2 - 1)
                        for name, p in target_params
                    }
                z_dict_ref.clear()
                z_dict_ref.update(z_dict_local)

                for name, p in target_params:
                    p.data.add_(z_dict_local[name].to(p.device), alpha=eps)

                # --- plus pass ---
                phase[0] = "plus"
                g_plus_cpu.clear()
                model.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
                    shift_labels = shift_labels.masked_fill(shift_labels >= shift_logits.size(-1), -100)
                    loss_plus = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
                    )
                loss_plus.backward()   # hooks populate g_plus_cpu

                for name, p in target_params:
                    p.data.sub_(z_dict_local[name].to(p.device), alpha=2*eps)

                # --- minus pass ---
                phase[0] = "minus"
                model.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
                    shift_labels = shift_labels.masked_fill(shift_labels >= shift_logits.size(-1), -100)
                    loss_minus = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
                    )
                loss_minus.backward()   # hooks compute HVP and add to accum

                # Restore params and drop the probe.
                for name, p in target_params:
                    p.data.add_(z_dict_local[name].to(p.device), alpha=eps)
                z_dict_ref.clear()
                del z_dict_local, loss_plus, loss_minus, outputs, shift_logits, shift_labels

            n_steps += 1
            del input_ids, attention_mask, labels, enc

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                print(f"      [{_ts()}] CHess batch {batch_idx+1}/{total_batches}")
    finally:
        for h in handles:
            h.remove()

    model.zero_grad(set_to_none=True)
    if use_checkpointing:
        model.gradient_checkpointing_disable()
    for p, rg in original_rg:
        p.requires_grad_(rg)

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


def _cache_path(label: str, corpus: str, metric: str, subset_id: Optional[int] = None,
                ext: str = ".pt") -> Path:
    """Return a nested cache path: CACHE_DIR/method/benchmark/model/corpus__metric[__subN].ext

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
    sub_suffix = f"__sub{subset_id}" if subset_id is not None else ""
    seed_suffix = f"_seed{_CHESS_SEED}" if (metric == "chess" and _CHESS_SEED is not None) else ""
    return (CACHE_DIR / method_dir / _BENCHMARK / safe_model /
            f"{corpus}__{metric}{sub_suffix}{seed_suffix}{ext}")


def _json_cache_path(label: str, corpus: str, metric: str, subset_id: Optional[int] = None) -> Path:
    return _cache_path(label, corpus, metric, subset_id=subset_id, ext=".json")


def _pt_cache_path(label: str, corpus: str, metric: str, subset_id: Optional[int] = None) -> Path:
    return _cache_path(label, corpus, metric, subset_id=subset_id, ext=".pt")


# ---------------------------------------------------------------------------
# Deterministic subset sampling for confidence intervals
# ---------------------------------------------------------------------------

def _subset_index_path(corpus_name: str, subset_id: int) -> Path:
    return CACHE_DIR / "_subsets" / _BENCHMARK / f"{corpus_name}__sub{subset_id}.json"


def _subset_indices(corpus_name: str, n_total: int, subset_id: int, n_samples: int) -> List[int]:
    """Return the index list for (corpus, subset_id).

    The saved JSON is the source of truth: if it exists, indices are loaded
    from there and returned verbatim (n_samples is ignored on hit). On first
    call, indices are generated from a deterministic seed
    md5(f"{corpus}|sub{subset_id}")[:8] and written to disk for all future
    callers (other models, other methods, re-runs after wall-time hits).
    """
    path = _subset_index_path(corpus_name, subset_id)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        idx = data["indices"]
        out_of_range = [i for i in idx if i >= n_total]
        if out_of_range:
            raise RuntimeError(
                f"Saved subset {path} has {len(out_of_range)} index/indices "
                f">= current corpus size {n_total} (e.g. {out_of_range[:3]}). "
                "The corpus has changed since this subset was recorded. "
                "Delete the file to regenerate, but note that any cached "
                "metrics under this (corpus, subset_id) will then refer to "
                "different documents and should also be removed."
            )
        return idx

    seed = int(hashlib.md5(f"{corpus_name}|sub{subset_id}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    if n_total <= n_samples:
        idx = list(range(n_total))
    else:
        idx = sorted(rng.sample(range(n_total), n_samples))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"corpus": corpus_name, "subset_id": subset_id,
                   "n_total": n_total, "n_samples": len(idx), "indices": idx}, f)
    return idx


def _subset_texts(texts: List[str], corpus_name: str, subset_id: int,
                  n_samples: int) -> List[str]:
    idx = _subset_indices(corpus_name, len(texts), subset_id, n_samples)
    return [texts[i] for i in idx]


def _chess_subsample(texts: List[str], corpus_name: str, subset_id: int,
                     n_chess: int) -> List[str]:
    """Take a deterministic random subset of `n_chess` docs from `texts`.

    Used to make CHess cheaper on large models without touching the on-disk
    subset record (CoFi/PPL still see the full subset). The selection is
    seeded from (corpus, subset_id) so base and unlearned models pick the
    *same* docs and their CHess Frobenius drop remains comparable. Indices
    are not persisted — this is a runtime-only convenience.
    """
    if n_chess >= len(texts):
        return texts
    seed = int(hashlib.md5(f"{corpus_name}|sub{subset_id}|chess".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return [texts[i] for i in sorted(rng.sample(range(len(texts)), n_chess))]


def _save_pt(d: dict, path: Path):
    """Save a tensor dict as half-precision .pt (used for base model only).

    Convert each tensor in place so we don't briefly hold both the fp32 and
    fp16 versions of the whole dict in memory at once. On Qwen3-32B that
    doubling crashed cgroup-limited jobs even at --mem=400G.

    Written atomically (temp file + rename) so an interrupted or disk-full
    write never leaves a truncated .pt behind that later crashes torch.load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for k in list(d.keys()):
        d[k] = d[k].half()
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(d, tmp)
        os.replace(tmp, path)          # atomic on the same filesystem
    finally:
        if tmp.exists():
            tmp.unlink()               # clean up a failed partial write


def _pt_valid(path: Path) -> bool:
    """Cheap integrity check for a torch .pt (zip) file.

    torch's .pt is a zip whose central directory is written last, so a
    truncated/failed write fails to open as a zip. Reading the central
    directory is O(1)-ish (seeks to the end), far cheaper than loading the
    62 GB of tensors. Returns False for missing, non-zip, or truncated files.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        # Opening the zip parses the end-of-central-directory record (written
        # last by torch.save); a truncated file raises here. namelist() is a
        # cheap central-directory read — we do NOT testzip() (that would CRC
        # the full 62 GB).
        with zipfile.ZipFile(path) as zf:
            return len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def _load_pt(path: Path) -> dict:
    """Load a tensor dict from .pt, converting back to float32."""
    if not _pt_valid(path):
        raise OSError(
            f"Corrupt/truncated cache file: {path}. Delete it (or rerun the "
            f"base eval) to regenerate."
        )
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
# CLI
# ---------------------------------------------------------------------------

class _QuietArgParser(argparse.ArgumentParser):
    """Argparse subclass that omits the multi-line usage dump on errors."""
    def error(self, message):
        self.exit(2, f"test.py: error: {message}\n")


def parse_args():
    p = _QuietArgParser()
    p.add_argument("--base-model", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--checkpoints-dir", default=str(PROJECT_ROOT / "checkpoints"))
    p.add_argument("--checkpoint", nargs=2, metavar=("PATH", "LABEL"), action="append", default=[])
    p.add_argument("--auto-discover", action="store_true")
    p.add_argument("--orig-max-samples", type=int, default=200,
                   help="(Legacy alias; see --n-samples-per-subset.) Samples per subset.")
    p.add_argument("--unlearned-max-samples", type=int, default=200,
                   help="(Legacy alias; see --n-samples-per-subset.) Samples per subset.")
    p.add_argument("--subset-ids", type=int, nargs="+", default=[0, 1, 2],
                   help="Subset IDs to evaluate. Each ID picks a deterministic, "
                        "model-independent sample of size --n-samples-per-subset from "
                        "each corpus, so confidence intervals can be derived across "
                        "subsets. Default: 0 1 2.")
    p.add_argument("--n-samples-per-subset", type=int, default=200,
                   help="Number of documents sampled per subset (default: 200).")
    p.add_argument("--base-only", action="store_true",
                   help="Compute base-model metrics for all subsets and exit. "
                        "Unlearned --checkpoint args are ignored.")
    p.add_argument("--require-base-cache", action="store_true",
                   help="Refuse to (re)compute base metrics; require .pt cache to exist. "
                        "Use this on the per-checkpoint slurms once base eval has run.")
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
    p.add_argument("--skip-cofi", action="store_true",
                   help="Skip CoFi computation. Useful on large models where the "
                        "fp32 CoFi accumulator (~param-count bytes × 4) exceeds "
                        "available CPU RAM.")
    p.add_argument("--skip-perplexity", action="store_true")
    p.add_argument("--benchmark", type=str, default="wmdp",
                   choices=["wmdp", "tofu", "muse"],
                   help="Evaluation benchmark (default: wmdp).")
    p.add_argument("--tofu-split", type=str, default=None,
                   choices=["forget01", "forget05", "forget10"],
                   help="TOFU forget split (required when --benchmark tofu).")
    p.add_argument("--muse-corpus", type=str, default=None,
                   choices=["news", "books"],
                   help="MUSE corpus (required when --benchmark muse).")
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

    torch_dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )

    # Determine the benchmark label used in checkpoint paths
    bench_label_map = {
        "wmdp": "wmdp",
        "tofu": f"tofu-{args.tofu_split}" if args.tofu_split else None,
        "muse": f"muse-{args.muse_corpus}" if args.muse_corpus else None
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
        args.benchmark, args.tofu_split, args.muse_corpus
    )

    # Subsampling is now done per (corpus, subset_id) — see _subset_texts.
    # The full corpora are kept in memory so subsets can be drawn deterministically.
    n_per_sub = args.n_samples_per_subset
    subset_ids = list(args.subset_ids)
    for name, texts in corpora.items():
        print(f"  {name:15s}: {len(texts):6d} docs total  "
              f"→ {min(len(texts), n_per_sub)} per subset × {len(subset_ids)} subsets")

    # Materialise the subset index lists upfront so they exist before any
    # eval step needs them; from here on _subset_indices() will always read
    # from JSON rather than re-deriving from the seed.
    for name, texts in corpora.items():
        for sid in subset_ids:
            _subset_indices(name, len(texts), sid, n_per_sub)

    print(f"[{_ts()}] Datasets loaded. {_mem_report()}")

    all_corpora_names = list(corpora.keys())

    # ====================================================================
    # STEP 1 — Original (base) model
    # ====================================================================
    print("\n" + "=" * 70)
    print(f"[{_ts()}] ORIGINAL MODEL")
    print("=" * 70)

    # Per-subset cache misses. Base needs .pt for CoFi/CHess and JSON for PPL,
    # one per (corpus, subset_id). A truncated/corrupt .pt counts as missing so
    # it gets recomputed rather than crashing a later drop computation.
    missing_cofi_pt = ([] if args.skip_cofi else
                       [(c, sid) for c in all_corpora_names for sid in subset_ids
                        if not _pt_valid(_pt_cache_path(base_label, c, "cofi", sid))])
    missing_chess_pt = ([] if args.skip_chess else
                        [(c, sid) for c in all_corpora_names for sid in subset_ids
                         if not _pt_valid(_pt_cache_path(base_label, c, "chess", sid))])
    missing_ppl = ([] if args.skip_perplexity else
                   [(c, sid) for c in all_corpora_names for sid in subset_ids
                    if not _json_cache_path(base_label, c, "ppl", sid).exists()])

    need_model = bool(missing_cofi_pt or missing_chess_pt or missing_ppl)

    n_pairs = len(all_corpora_names) * len(subset_ids)
    print(f"[{_ts()}] Base cache status ({n_pairs} corpus×subset pairs):")
    if not args.skip_cofi:
        print(f"    CoFi .pt : {n_pairs - len(missing_cofi_pt)}/{n_pairs} cached")
    if not args.skip_chess:
        print(f"    CHess.pt : {n_pairs - len(missing_chess_pt)}/{n_pairs} cached")
    if not args.skip_perplexity:
        print(f"    PPL      : {n_pairs - len(missing_ppl)}/{n_pairs} cached")

    if need_model and args.require_base_cache:
        raise SystemExit(
            f"--require-base-cache is set but {len(missing_cofi_pt)} CoFi / "
            f"{len(missing_chess_pt)} CHess / {len(missing_ppl)} PPL base entries "
            f"are missing. Run the base-eval slurm first."
        )

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
            print(f"\n  [{_ts()}] Perplexity (per subset, full subset):")
            for corpus_name, sid in missing_ppl:
                ppl_json = _json_cache_path(base_label, corpus_name, "ppl", sid)
                ppl_texts = _subset_texts(corpora[corpus_name], corpus_name, sid, n_per_sub)
                print(f"    [{_ts()}] PPL '{corpus_name}' sub{sid} ({len(ppl_texts)} docs) ...")
                ppl, n_toks, n_docs = compute_perplexity(
                    model, tokenizer, ppl_texts,
                    max_length=args.ppl_max_length, stride=args.ppl_stride,
                    device=input_device, max_tokens=args.ppl_max_tokens,
                )
                _save_json({"ppl": ppl, "n_tokens": n_toks, "n_docs": n_docs,
                            "subset_id": sid}, ppl_json)
                print(f"    {corpus_name:15s} sub{sid}: PPL = {ppl:.4f}  ({n_toks:,} tokens)")
            _flush_memory()

        if missing_cofi_pt:
            print(f"\n  [{_ts()}] Computing CoFi for base model "
                  f"({len(missing_cofi_pt)} corpus×subset pairs):")
            for corpus_name, sid in missing_cofi_pt:
                sub_texts = _subset_texts(corpora[corpus_name], corpus_name, sid, n_per_sub)
                grad_texts = _chunk_long_texts(sub_texts, tokenizer, args.max_length)[:n_per_sub]
                print(f"    [{_ts()}] CoFi '{corpus_name}' sub{sid} ({len(grad_texts)}) ...")
                result = compute_cofi(
                    model, tokenizer, grad_texts,
                    args.max_length, args.batch_size_grad, target_params,
                )
                # Frobenius norm first (on fp32 tensors), then save — _save_pt
                # converts to fp16 in place to keep CPU memory low on large
                # models, so the fp32 reading must happen beforehand.
                norm_val = frob_norm(result)
                _save_json({"frob_norm": norm_val, "subset_id": sid},
                           _json_cache_path(base_label, corpus_name, "cofi", sid))
                _save_pt(result, _pt_cache_path(base_label, corpus_name, "cofi", sid))
                del result
                _flush_memory()
                print(f"    {corpus_name:15s} sub{sid}: ||F||_F = {norm_val:.6f}  {_mem_report()}")

        if missing_chess_pt:
            print(f"\n  [{_ts()}] Computing CHess for base model "
                  f"({len(missing_chess_pt)} corpus×subset pairs):")
            for corpus_name, sid in missing_chess_pt:
                sub_texts = _subset_texts(corpora[corpus_name], corpus_name, sid, n_per_sub)
                sub_texts = _chess_subsample(sub_texts, corpus_name, sid, chess_max_samples)
                grad_texts = _chunk_long_texts(sub_texts, tokenizer, chess_max_length)[:chess_max_samples]
                print(f"    [{_ts()}] CHess '{corpus_name}' sub{sid} ({len(grad_texts)}) ...")
                result = compute_chess(
                    model, tokenizer, grad_texts,
                    chess_max_length, chess_batch_size, target_params, args.n_hutchinson,
                    use_checkpointing=chess_use_checkpointing,
                    base_seed=args.chess_seed,
                    corpus_name=f"{corpus_name}__sub{sid}",
                )
                # Frobenius norm first (fp32) — _save_pt mutates to fp16.
                norm_val = frob_norm(result)
                _save_json({"frob_norm": norm_val, "subset_id": sid},
                           _json_cache_path(base_label, corpus_name, "chess", sid))
                _save_pt(result, _pt_cache_path(base_label, corpus_name, "chess", sid))
                del result
                _flush_memory()
                print(f"    {corpus_name:15s} sub{sid}: ||H||_F = {norm_val:.6f}  {_mem_report()}")

        print(f"\n  [{_ts()}] Unloading base model ...")
        del model, tokenizer, target_params
        _flush_memory()
        print(f"  [{_ts()}] Base model unloaded. {_mem_report()}")

    if args.base_only:
        print(f"\n[{_ts()}] --base-only: skipping unlearned evaluation.")
        return

    # ====================================================================
    # STEP 2 — Each unlearned model
    # ====================================================================
    # Per-subset measurements: {label: {corpus: {subset_id: value}}}
    ppl_per_sub:   Dict[str, Dict[str, Dict[int, float]]] = {}
    cofi_per_sub:  Dict[str, Dict[str, Dict[int, float]]] = {}
    chess_per_sub: Dict[str, Dict[str, Dict[int, float]]] = {}

    total_models = len(unlearned)
    for model_idx, (label, model_path) in enumerate(unlearned, 1):
        print("\n" + "=" * 70)
        print(f"[{_ts()}] UNLEARNED MODEL {model_idx}/{total_models}: {label}")
        print(f"Path: {model_path}")
        print("=" * 70)

        unl_missing_cofi = ([] if args.skip_cofi else
                            [(c, sid) for c in all_corpora_names for sid in subset_ids
                             if not _json_cache_path(label, c, "cofi", sid).exists()])
        unl_missing_chess = ([] if args.skip_chess else
                             [(c, sid) for c in all_corpora_names for sid in subset_ids
                              if not _json_cache_path(label, c, "chess", sid).exists()])
        unl_missing_ppl = ([] if args.skip_perplexity else
                           [(c, sid) for c in all_corpora_names for sid in subset_ids
                            if not _json_cache_path(label, c, "ppl", sid).exists()])
        all_cached = not (unl_missing_cofi or unl_missing_chess or unl_missing_ppl)

        n_tot = len(all_corpora_names) * len(subset_ids)
        print(f"  [{_ts()}] Cache status ({n_tot} corpus×subset pairs):")
        if not args.skip_cofi:
            print(f"    CoFi  : {n_tot - len(unl_missing_cofi)}/{n_tot} cached")
        if not args.skip_chess:
            print(f"    CHess : {n_tot - len(unl_missing_chess)}/{n_tot} cached")
        if not args.skip_perplexity:
            print(f"    PPL   : {n_tot - len(unl_missing_ppl)}/{n_tot} cached")

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

        ppl_per_sub.setdefault(label, {c: {} for c in all_corpora_names})
        cofi_per_sub.setdefault(label, {c: {} for c in all_corpora_names})
        chess_per_sub.setdefault(label, {c: {} for c in all_corpora_names})

        # --- Perplexity (per subset) ---
        if not args.skip_perplexity:
            input_device = _get_input_device(model) if model is not None else None
            for corpus_name in all_corpora_names:
                for sid in subset_ids:
                    ppl_json = _json_cache_path(label, corpus_name, "ppl", sid)
                    if ppl_json.exists():
                        ppl_per_sub[label][corpus_name][sid] = _load_json(ppl_json)["ppl"]
                        continue
                    ppl_texts = _subset_texts(corpora[corpus_name], corpus_name, sid, n_per_sub)
                    print(f"    [{_ts()}] PPL '{corpus_name}' sub{sid} ({len(ppl_texts)} docs) ...")
                    ppl, n_toks, n_docs = compute_perplexity(
                        model, tokenizer, ppl_texts,
                        max_length=args.ppl_max_length, stride=args.ppl_stride,
                        device=input_device, max_tokens=args.ppl_max_tokens,
                    )
                    _save_json({"ppl": ppl, "n_tokens": n_toks, "n_docs": n_docs,
                                "subset_id": sid}, ppl_json)
                    ppl_per_sub[label][corpus_name][sid] = ppl
                    print(f"    {corpus_name:15s} sub{sid}: PPL = {ppl:.4f}")
            _flush_memory()

        # --- CoFi norm drops (per subset, vs. base[subset]) ---
        if not args.skip_cofi:
            print(f"\n  [{_ts()}] CoFi norm drops (vs. base, per subset):")
            for corpus_name in all_corpora_names:
                for sid in subset_ids:
                    cofi_json = _json_cache_path(label, corpus_name, "cofi", sid)
                    if cofi_json.exists():
                        drop = _load_json(cofi_json)["norm_drop"]
                        print(f"    {corpus_name:15s} sub{sid}: [cached]  ||Δ F||_F = {drop:.6f}")
                    else:
                        sub_texts = _subset_texts(corpora[corpus_name], corpus_name, sid, n_per_sub)
                        grad_texts = _chunk_long_texts(sub_texts, tokenizer, args.max_length)[:n_per_sub]
                        print(f"    [{_ts()}] CoFi '{corpus_name}' sub{sid} ({len(grad_texts)}) ...")
                        result = compute_cofi(
                            model, tokenizer, grad_texts,
                            args.max_length, args.batch_size_grad, target_params,
                        )
                        base_pt = _pt_cache_path(base_label, corpus_name, "cofi", sid)
                        if not base_pt.exists():
                            raise FileNotFoundError(
                                f"Base CoFi cache missing: {base_pt}. Run base eval first."
                            )
                        base_dict = _load_pt(base_pt)
                        drop = frob_norm_drop(base_dict, result)
                        del base_dict, result
                        _flush_memory()
                        _save_json({"norm_drop": drop, "subset_id": sid}, cofi_json)
                        print(f"    {corpus_name:15s} sub{sid}: ||Δ F||_F = {drop:.6f}")
                    cofi_per_sub[label][corpus_name][sid] = drop

        if not args.skip_chess:
            print(f"\n  [{_ts()}] CHess norm drops (vs. base, per subset):")
            for corpus_name in all_corpora_names:
                for sid in subset_ids:
                    chess_json = _json_cache_path(label, corpus_name, "chess", sid)
                    if chess_json.exists():
                        drop = _load_json(chess_json)["norm_drop"]
                        print(f"    {corpus_name:15s} sub{sid}: [cached]  ||Δ H||_F = {drop:.6f}")
                    else:
                        sub_texts = _subset_texts(corpora[corpus_name], corpus_name, sid, n_per_sub)
                        sub_texts = _chess_subsample(sub_texts, corpus_name, sid, chess_max_samples)
                        grad_texts = _chunk_long_texts(sub_texts, tokenizer, chess_max_length)[:chess_max_samples]
                        print(f"    [{_ts()}] CHess '{corpus_name}' sub{sid} ({len(grad_texts)}) ...")
                        result = compute_chess(
                            model, tokenizer, grad_texts,
                            chess_max_length, chess_batch_size, target_params, args.n_hutchinson,
                            use_checkpointing=chess_use_checkpointing,
                            base_seed=args.chess_seed,
                            corpus_name=f"{corpus_name}__sub{sid}",
                        )
                        base_pt = _pt_cache_path(base_label, corpus_name, "chess", sid)
                        if not base_pt.exists():
                            raise FileNotFoundError(
                                f"Base CHess cache missing: {base_pt}. Run base eval first."
                            )
                        base_dict = _load_pt(base_pt)
                        drop = frob_norm_drop(base_dict, result)
                        del base_dict, result
                        _flush_memory()
                        _save_json({"norm_drop": drop, "subset_id": sid}, chess_json)
                        print(f"    {corpus_name:15s} sub{sid}: ||Δ H||_F = {drop:.6f}")
                    chess_per_sub[label][corpus_name][sid] = drop

        if model is not None:
            print(f"  [{_ts()}] Unloading model '{label}' ...")
            del model, tokenizer, target_params
            _flush_memory()
            print(f"  [{_ts()}] Model unloaded. {_mem_report()}")

    # ====================================================================
    # STEP 3 — Summary tables (mean ± std across subsets, with 95% CI half-width)
    # ====================================================================
    if not unlearned:
        return

    col_w = max(len(lbl) for lbl, _ in unlearned)
    all_corpora = forget_names + retain_names

    def _agg(values):
        """Return (mean, std, ci95_half) where ci95_half is the t-distribution
        half-width for a 95% CI (or nan if <2 samples)."""
        vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return float("nan"), float("nan"), float("nan")
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0, float("nan")
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
        # t-critical for 95% with df=n-1. Tabulated for n=2..5.
        t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(len(vals) - 1, 1.96)
        ci_half = t_crit * std / math.sqrt(len(vals))
        return mean, std, ci_half

    def print_table(per_sub, title, fmt="{:.6f}"):
        print("\n" + "=" * 70)
        print(title)
        if "Drop" in title:
            print("(↑ forget = better unlearning   ↓ retain = less collateral damage)")
        print(f"Subsets: {subset_ids}   format: mean ± 95% CI (std)")
        print("=" * 70)
        cell_w = 26  # room for "mean ± ci (std=...)" style
        header = f"{'Model':{col_w}s} | " + " | ".join(f"{c:>{cell_w}s}" for c in all_corpora)
        print(header)
        print("-" * len(header))
        for label, _ in unlearned:
            if label not in per_sub:
                continue
            cells = []
            for c in all_corpora:
                vals = list(per_sub[label].get(c, {}).values())
                mean, std, ci = _agg(vals)
                if math.isnan(mean):
                    cells.append("nan".rjust(cell_w))
                elif math.isnan(ci):
                    cells.append(fmt.format(mean).rjust(cell_w))
                else:
                    cells.append(f"{fmt.format(mean)} ± {fmt.format(ci)} (σ={fmt.format(std)})".rjust(cell_w))
            print(f"{label:{col_w}s} | " + " | ".join(cells))

    if not args.skip_cofi:
        print_table(cofi_per_sub,  "Normalized CoFi Frobenius Norm Drops  (1/√N)||F^o − F^u||_F")
    if not args.skip_chess:
        print_table(chess_per_sub, "Normalized CHess Frobenius Norm Drops  (1/√N)||H^o − H^u||_F")
    if not args.skip_perplexity:
        print_table(ppl_per_sub, "Perplexity  (concatenated sliding-window)", fmt="{:.4f}")

    print(f"\n[{_ts()}] All done.")


if __name__ == "__main__":
    main()