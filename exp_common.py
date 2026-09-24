"""Shared helpers for the reviewer-rebuttal experiments (E0-E3).

Everything reuses test.py's CoFi/CHess/frob machinery UNCHANGED so the numbers
are identical to production. The only new concept is the relative shift from
Eq. (2) of the paper:

    shift(X, Y) = 100 * ||X - Y||_F / ||X||_F

computed over the log-space-stabilised diagonal Fisher/Hessian vectors. Both
||.||_F are test.frob_norm (normalised by 1/sqrt(N)); the normalisation cancels,
so shift == 100 * frob_norm_drop(X,Y) / frob_norm(X).

Production settings (spec): CoFi 200 docs / max_len 1024 / batch 4; CHess 200
docs / max_len 512 / batch 8 / K=2 / eps=1e-3; targets {q,k,v,o,gate,up,down}_proj.
"""

import csv
import hashlib
import os
import random
from pathlib import Path

import torch

import test as T   # CoFi/CHess/frob/loaders — reused unchanged

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "rebuttal"

# WMDP base models (original HF checkpoints).
MODELS = {
    "llama-3.1-8b":   "meta-llama/Meta-Llama-3.1-8B",
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
}

# Production CoFi/CHess knobs.
COFI_MAX_LEN, COFI_BATCH = 1024, 4
CHESS_MAX_LEN, CHESS_BATCH = 512, 8
CHESS_K, CHESS_EPS = 2, 1e-3   # eps is hard-coded to 1e-3 inside test.compute_chess


# ---------------------------------------------------------------------------
# Determinism (required for E0-L0/L3; harmless elsewhere)
# ---------------------------------------------------------------------------

def setup_determinism():
    """Make CoFi/CHess reproducible bit-for-bit where the ops allow it, and
    return a dict of the precision knobs to log into every output row."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    dtype = "bfloat16" if (torch.cuda.is_available()
                           and torch.cuda.is_bf16_supported()) else "float16"
    return {"dtype": dtype, "tf32": False}


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model / corpus / target helpers (thin wrappers over test.py)
# ---------------------------------------------------------------------------

def torch_dtype():
    return (torch.bfloat16 if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported() else torch.float16)


def load_model(path):
    model, tok = T.load_model_and_tokenizer(path, torch_dtype())
    model.eval()
    return model, tok


def targets(model):
    return T.get_target_params(model, T.DEFAULT_TARGET_MODULES)


def wmdp_corpora():
    """Return (corpora_dict, forget_names, retain_names) for WMDP."""
    return T.get_benchmark_corpora("wmdp")


# ---------------------------------------------------------------------------
# CoFi / CHess diagonals (production knobs) + the shift metric
# ---------------------------------------------------------------------------

def cofi_diag(model, tok, texts, target_params):
    return T.compute_cofi(model, tok, texts, COFI_MAX_LEN, COFI_BATCH, target_params)


def chess_diag(model, tok, texts, target_params, n_hutchinson=CHESS_K,
               base_seed=0, corpus_name="c"):
    return T.compute_chess(model, tok, texts, CHESS_MAX_LEN, CHESS_BATCH,
                           target_params, n_hutchinson, use_checkpointing=False,
                           base_seed=base_seed, corpus_name=corpus_name)


def shift(X, Y):
    """100 * ||X - Y||_F / ||X||_F over the diagonal dicts (Eq. 2)."""
    denom = T.frob_norm(X)
    if denom == 0:
        return float("nan")
    return 100.0 * T.frob_norm_drop(X, Y) / denom


# ---------------------------------------------------------------------------
# Deterministic disjoint sampling (E0 needs TWO disjoint samples per corpus)
# ---------------------------------------------------------------------------

def _seed_int(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def two_disjoint_samples(texts, n, seed):
    """Two disjoint index-based samples of size n from `texts` (largest feasible
    if 2n > len). Returns (sample_a, sample_b, actual_n)."""
    total = len(texts)
    actual_n = min(n, total // 2)
    rng = random.Random(_seed_int("disjoint", seed, total, n))
    idx = list(range(total))
    rng.shuffle(idx)
    a = sorted(idx[:actual_n])
    b = sorted(idx[actual_n:2 * actual_n])
    return [texts[i] for i in a], [texts[i] for i in b], actual_n


def one_sample(texts, n, seed):
    """A single deterministic sample of size n (for L0/L3 same-sample tests)."""
    total = len(texts)
    m = min(n, total)
    rng = random.Random(_seed_int("one", seed, total, n))
    idx = sorted(rng.sample(range(total), m))
    return [texts[i] for i in idx]


# ---------------------------------------------------------------------------
# Long-format CSV writer (append one row per measurement; never aggregate here)
# ---------------------------------------------------------------------------

class CSVWriter:
    def __init__(self, path, columns, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = columns
        append = resume and self.path.exists()
        self._f = open(self.path, "a" if append else "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=columns)
        if not append:
            self._w.writeheader()
        self._f.flush()

    def row(self, **kw):
        self._w.writerow({c: kw.get(c, "") for c in self.columns})
        self._f.flush()

    def close(self):
        self._f.close()


def load_done_keys(path, key_columns):
    """Already-written (key_columns) tuples from an existing long-format CSV,
    so a resumable script can skip measurements it already has on disk."""
    path = Path(path)
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {tuple(row.get(c, "") for c in key_columns) for row in csv.DictReader(f)}


def free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Answer-only-masked CoFi / CHess.
#
# The whole-sequence primary reuses test.compute_cofi / test.compute_chess
# UNCHANGED. Answer-token gating is the specific hypothesis E1 tests, so the
# secondary needs a loss over answer tokens only. These are faithful copies of
# the production estimators (identical scale_c / lambda / eps / log-space
# finalize / grad-streaming) with ONE change: labels for the prompt prefix are
# set to -100. `pairs` is a list of (full_text, prompt_prefix); everything after
# the prefix is the answer span that carries the loss.
# ---------------------------------------------------------------------------

import math as _math

_SCALE_C = 1e6
_LAMBDA = 1e-5
_EPS = 1e-3


def _masked_labels(tok, batch_pairs, input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i, (_full, prompt) in enumerate(batch_pairs):
        plen = len(tok(prompt, add_special_tokens=True).input_ids)
        plen = min(plen, labels.size(1))
        labels[i, :plen] = -100          # mask the prompt; keep only answer tokens
    return labels


def cofi_diag_answer_only(model, tok, pairs, target_params,
                          max_length=COFI_MAX_LEN, batch_size=COFI_BATCH):
    tok.padding_side = "right"           # so real tokens start at 0 -> mask [:plen]
    target_ids = {id(p) for _, p in target_params}
    orig = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(id(p) in target_ids)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dev = T._get_input_device(model)
    amp = T._autocast_dtype()
    accum = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu") for n, p in target_params}
    sqrt_c = _math.sqrt(_SCALE_C)

    def mk(name):
        tgt = accum[name]
        def h(p):
            if p.grad is None:
                return
            g = torch.nan_to_num(p.grad.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)
            g.div_(sqrt_c)
            tgt.add_(g.pow_(2).cpu())
            p.grad = None
        return h
    handles = [p.register_post_accumulate_grad_hook(mk(n)) for n, p in target_params]
    n_steps = 0
    try:
        for i in range(0, len(pairs), batch_size):
            bp = pairs[i:i + batch_size]
            enc = tok([f for f, _ in bp], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            ii = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev)
            if ii.numel() == 0:
                continue
            labels = _masked_labels(tok, bp, enc["input_ids"], enc["attention_mask"]).to(dev)
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp):
                logits = model(input_ids=ii, attention_mask=am).logits
                sl = logits[..., :-1, :].contiguous()
                st = labels[..., 1:].contiguous()
                st = st.masked_fill(st >= sl.size(-1), -100)
                loss = torch.nn.functional.cross_entropy(
                    sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            if torch.isfinite(loss):
                loss.backward()
                n_steps += 1
            del enc, ii, am, labels, logits, sl, st, loss
    finally:
        for h in handles:
            h.remove()
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    for p, rg in orig:
        p.requires_grad_(rg)
    if n_steps > 0:
        for n in accum:
            accum[n].div_(n_steps).add_(_LAMBDA / _SCALE_C).log_().add_(_math.log(_SCALE_C))
    return accum


def chess_diag_answer_only(model, tok, pairs, target_params, n_hutchinson=CHESS_K,
                           base_seed=0, corpus_name="c",
                           max_length=CHESS_MAX_LEN, batch_size=CHESS_BATCH):
    tok.padding_side = "right"
    target_ids = {id(p) for _, p in target_params}
    orig = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(id(p) in target_ids)
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    dev = T._get_input_device(model)
    amp = T._autocast_dtype()
    accum = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu") for n, p in target_params}
    g_plus, z_ref, phase = {}, {}, ["plus"]

    def mk(name):
        tgt = accum[name]
        def h(p):
            if p.grad is None:
                return
            g = p.grad.detach().float().cpu()
            p.grad = None
            if phase[0] == "plus":
                g_plus[name] = g
            else:
                gp = g_plus.pop(name, None)
                if gp is None:
                    return
                hvp = gp.sub_(g).div_(2 * _EPS).mul_(z_ref[name].float())
                hvp.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4).div_(_SCALE_C)
                tgt.add_(hvp)
        return h
    handles = [p.register_post_accumulate_grad_hook(mk(n)) for n, p in target_params]

    def _ce(ii, am, labels):
        with torch.amp.autocast("cuda", dtype=amp):
            sl = model(input_ids=ii, attention_mask=am).logits[..., :-1, :].contiguous()
            st = labels[..., 1:].contiguous()
            st = st.masked_fill(st >= sl.size(-1), -100)
            return torch.nn.functional.cross_entropy(
                sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)

    n_steps = 0
    try:
        for bidx, i in enumerate(range(0, len(pairs), batch_size)):
            bp = pairs[i:i + batch_size]
            enc = tok([f for f, _ in bp], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            ii = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev)
            if ii.numel() == 0:
                continue
            labels = _masked_labels(tok, bp, enc["input_ids"], enc["attention_mask"]).to(dev)
            n_steps += 1
            for hidx in range(n_hutchinson):
                gen = torch.Generator(device="cpu").manual_seed(
                    T._chess_probe_seed(base_seed, corpus_name, bidx, hidx))
                z = {n: (torch.randint(0, 2, p.shape, generator=gen) * 2 - 1).to(p.dtype)
                     for n, p in target_params}
                z_ref.clear(); z_ref.update(z)
                for n, p in target_params:                       # theta + eps z
                    p.data.add_(z[n].to(p.device), alpha=_EPS)
                phase[0] = "plus"; g_plus.clear(); model.zero_grad(set_to_none=True)
                _ce(ii, am, labels).backward()                   # hooks store g_plus
                for n, p in target_params:                       # theta - eps z
                    p.data.sub_(z[n].to(p.device), alpha=2 * _EPS)
                phase[0] = "minus"; model.zero_grad(set_to_none=True)
                _ce(ii, am, labels).backward()                   # hooks accumulate hvp
                for n, p in target_params:                       # restore theta
                    p.data.add_(z[n].to(p.device), alpha=_EPS)
                z_ref.clear()
            del enc, ii, am, labels
    finally:
        for h in handles:
            h.remove()
    model.zero_grad(set_to_none=True)
    for p, rg in orig:
        p.requires_grad_(rg)
    if n_steps > 0:
        for n in accum:
            a = accum[n].div_(n_steps * n_hutchinson).abs_()
            a.add_(_LAMBDA / _SCALE_C).log_().add_(_math.log(_SCALE_C))
    return accum
