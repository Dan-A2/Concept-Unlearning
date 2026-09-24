"""CoFi/CHess robustness ablation for the rebuttal.

Answers "how sensitive are CoFi/CHess conclusions to the diagonal
approximation setup (sample size / probe count / parameter subset)?" on the 8B
model, WMDP, sub0 only. Three axes, both metrics:

  A. Sample size n in {50,100,200} docs          -- CoFi and CHess
  B. Hutchinson probe count K in {2,4,8}         -- CHess only (CoFi has no probes)
  C. Parameter subset attn / FFN / both          -- CoFi and CHess (free re-agg)

For each (model, corpus) we run the *exact* estimators from test.py
(compute_cofi / compute_chess -- same scale_c / lambda / eps / probe seeds /
log-space finalize) but structured so a single pass snapshots the running
diagonal at each break-point:

  * CoFi  size sweep : docs-outer, snapshot at n break-points.
  * CHess size sweep : docs-outer, K fixed (=--size-k), probes-inner, snapshot at n.
  * CHess probe sweep: probes-outer, docs fixed (=--probe-n), snapshot at K.

Each snapshot is a per-parameter diagonal; the drop (base vs unlearned Frobenius
distance) is then computed streaming, restricted to attn (q/k/v/o), FFN
(gate/up/down), or both -- axis C, at zero extra GPU cost. We also report the
CHess probe-convergence ||d_K - d_{K/2}|| / ||d_K|| (~K^-1/2).


CRASH / PREEMPTION SAFETY  (the point of this file's checkpoint layer)
----------------------------------------------------------------------
Every breakpoint of every sweep, for the base model *and* for each method, is
persisted to disk and NEVER deleted:

    <work-dir>/ck__<corpus>__<sweep>__<role>.diag<point>.pt   finalised diagonal
    <work-dir>/ck__<corpus>__<sweep>__<role>.accum.pt         raw resume state
    <work-dir>/ck__<corpus>__<sweep>__<role>.meta.json        progress mirror

So if the CHess probe sweep for `base` dies while working towards K=8, the next
run re-yields K=2 and K=4 straight off disk, reloads the raw accumulator saved
at K=4, and probes onward from there -- it does not redo K=2/K=4. The same
holds per method (atu/npo/rmu), per corpus, and for the CoFi and CHess
sample-size sweeps. Diagonals of earlier breakpoints stay on disk after later
ones are written (K=2 survives K=4 being saved), and nothing is removed when a
role or sweep finishes.

Consistency rule: the raw accumulator file embeds its own progress metadata, so
a kill mid-write can never leave "accumulator ahead of bookkeeping" (which would
double-count batches). All writes are atomic (tmp file + rename). meta.json is
a human-readable mirror plus the "this role is complete" marker, which lets a
finished role be skipped without even loading the 8B weights.

Extra safety valves:
  * --ckpt-every-probes / --ckpt-every-batches write intermediate resume state
    *between* breakpoints (the long K=4 -> K=8 stretch), at the cost of extra
    ~28GB writes.
  * SIGUSR1/SIGTERM (see the slurm script's --signal=B:USR1@600) trigger a
    clean checkpoint-and-exit at the next safe point instead of losing the
    partial accumulator.
  * Per-breakpoint drop numbers are committed to the results JSON as soon as
    they are computed, not only when a method finishes.

Disk cost is real: the fp32 accumulator is ~28GB and each fp16 diagonal ~14GB
for Llama-3.1-8B over q/k/v/o/gate/up/down. Budget accordingly, or pass
--cleanup-accums to drop *only* the raw resume accumulators (never diagonals)
once a role has finished.
"""

import argparse
import json
import math
import re
import signal
from pathlib import Path

import torch

import test as T   # reuse estimators, loaders, corpora, subset indices

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "Results" / "chess_ablation"

# Estimator constants -- MUST match test.compute_cofi / compute_chess.
_EPS = 1e-3
_SCALE_C = 1e6
_LAMBDA = 1e-5
_SQRT_SCALE_C = math.sqrt(_SCALE_C)

_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_FFN = ("gate_proj", "up_proj", "down_proj")
_SUBSETS = ("both", "attn", "ffn")


# ---------------------------------------------------------------------------
# Cooperative preemption: checkpoint, then exit -- instead of dying mid-probe.
# ---------------------------------------------------------------------------

class Preempted(Exception):
    """Raised at a safe point after state has been checkpointed."""


_STOP = {"flag": False}


def _install_signal_handlers():
    def _handler(signum, _frame):
        if not _STOP["flag"]:
            print(f"\n[{T._ts()}] [signal] caught {signum}: will checkpoint and exit "
                  f"at the next safe point (rerun to resume)", flush=True)
        _STOP["flag"] = True
    for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


def _safe(s):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", str(s))


def _subset_keys(diag_keys, subset):
    if subset == "both":
        return list(diag_keys)
    tags = _ATTN if subset == "attn" else _FFN
    return [k for k in diag_keys if any(t in k for t in tags)]


def _rel_frob_dist(a, b, keys):
    """||a - b|| / ||a||  over `keys`."""
    num = den = 0.0
    for k in keys:
        av = a[k].double()
        num += (av - b[k].double()).pow(2).sum().item()
        den += av.pow(2).sum().item()
    return (num ** 0.5) / (den ** 0.5) if den else float("nan")


def _frob_drop(base, meth, keys):
    """Normalised Frobenius drop over `keys`, matching test.frob_norm_drop."""
    n = sum(base[k].numel() for k in keys)
    if n == 0:
        return float("nan")
    ss = 0.0
    for k in keys:
        ss += (base[k].double() - meth[k].double()).pow(2).sum().item()
    return ss ** 0.5 / n ** 0.5


def _cofi_finalize(accum, denom):
    """CoFi log-space finalize -- a pure function of (accumulator, #steps), which
    is what makes a break-point diagonal rebuildable from a checkpoint."""
    out = {}
    for name, a in accum.items():
        x = a / denom
        x.add_(_LAMBDA / _SCALE_C).log_().add_(math.log(_SCALE_C))
        out[name] = x.half()
    return out


def _set_target_grads(model, target_params):
    target_ids = {id(p) for _, p in target_params}
    original = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(id(p) in target_ids)
    return original


# ---------------------------------------------------------------------------
# Checkpointing
#
# One `Ckpt` per (corpus, sweep, role). Files are additive and permanent:
# every breakpoint's diagonal is kept forever, and the raw accumulator is
# overwritten in place (atomically) as progress advances.
#
# The accumulator file stores {"meta": ..., "accum": ...} in ONE atomic write,
# so the resume bookkeeping can never disagree with the tensors it describes.
# meta.json duplicates that bookkeeping for eyeballing / fast "is this role
# finished?" checks without touching the 28GB accumulator.
# ---------------------------------------------------------------------------

def _atomic_torch_save(obj, path):
    tmp = path.parent / (path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_json_dump(obj, path):
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


class Ckpt:
    def __init__(self, prefix):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)

    def _p(self, suffix):
        return self.prefix.parent / (self.prefix.name + suffix)

    @property
    def accum_path(self):
        return self._p(".accum.pt")

    @property
    def meta_path(self):
        return self._p(".meta.json")

    def diag_path(self, point):
        return self._p(f".diag{point}.pt")

    # -- reads ------------------------------------------------------------
    def load_meta(self):
        if not self.meta_path.exists():
            return None
        try:
            with open(self.meta_path) as f:
                return json.load(f)
        except Exception:
            return None

    def load_state(self):
        """-> (accum, meta) or None. Tolerates the legacy raw-accum format."""
        if not self.accum_path.exists():
            return None
        try:
            obj = torch.load(self.accum_path, weights_only=True)
        except Exception as e:
            print(f"[{T._ts()}] [warn] unreadable accumulator {self.accum_path} ({e}); "
                  f"this role restarts from scratch")
            return None
        if isinstance(obj, dict) and "accum" in obj and "meta" in obj:
            return obj["accum"], dict(obj["meta"])
        meta = self.load_meta()          # legacy layout: raw accum + meta.json
        if meta is None:
            return None
        return obj, dict(meta)

    def load_diag(self, point):
        return torch.load(self.diag_path(point), weights_only=True)

    def have_diags(self, points):
        return all(self.diag_path(p).exists() for p in points)

    # -- writes -----------------------------------------------------------
    def save_diag(self, point, diag):
        _atomic_torch_save(diag, self.diag_path(point))

    def save_state(self, accum, meta):
        meta = dict(meta)
        meta["complete"] = False
        _atomic_torch_save({"meta": meta, "accum": accum}, self.accum_path)
        _atomic_json_dump(meta, self.meta_path)

    def mark_complete(self, meta):
        meta = dict(meta)
        meta["complete"] = True
        _atomic_json_dump(meta, self.meta_path)

    def drop_accum(self):
        """Opt-in (--cleanup-accums). Diagonals are never touched."""
        try:
            self.accum_path.unlink(missing_ok=True)
        except OSError:
            pass


def _ckpt_bootstrap(ckpt, finalize=None):
    """-> ("complete", None, meta) | ("resume", accum, meta) | ("fresh", None, None)

    "complete": meta.json says the role finished and every diagonal it lists is
    present, so the caller can re-yield from disk and stop.
    "resume":   a consistent accumulator exists; its embedded bookkeeping wins.
    "fresh":    nothing usable (or something is missing/corrupt) -> start over.

    Self-repair: a break-point is written as (1) accumulator+bookkeeping, then
    (2) the diagonal itself. A kill landing between the two leaves a recorded
    break-point with no diagonal on disk -- but the accumulator is then sitting
    at *exactly* that break-point, so the diagonal is re-derivable from it
    (it is a pure function of accumulator and denominator, both checkpointed).
    We rebuild it instead of throwing the whole role away.
    """
    if ckpt is None:
        return ("fresh", None, None)
    meta = ckpt.load_meta()
    if meta is not None and meta.get("complete"):
        pts = list(meta.get("points", []))
        if pts and ckpt.have_diags(pts):
            return ("complete", None, meta)
    state = ckpt.load_state()
    if state is None:
        return ("fresh", None, None)
    accum, smeta = state
    pts = list(smeta.get("points", []))
    missing = [p for p in pts if not ckpt.diag_path(p).exists()]
    if missing:
        pstate = smeta.get("point_state", {})
        last = pts[-1]
        info = pstate.get(str(last), {})
        repairable = (missing == [last] and finalize is not None
                      and info.get("denom") and info.get("progress") == smeta.get("progress"))
        if not repairable:
            print(f"[{T._ts()}] [warn] {ckpt.prefix.name}: diagonal(s) {missing} missing and "
                  f"not re-derivable; restarting this role from scratch")
            return ("fresh", None, None)
        print(f"[{T._ts()}] [repair] {ckpt.prefix.name}: rebuilding break-point {last} "
              f"from the checkpointed accumulator")
        ckpt.save_diag(last, finalize(accum, info["denom"]))
    return ("resume", accum, smeta)


def _maybe_stop(ckpt, accum, meta, where):
    """Honour a pending SIGUSR1/SIGTERM: checkpoint, then unwind cleanly."""
    if not _STOP["flag"]:
        return
    if ckpt is not None and accum is not None:
        print(f"[{T._ts()}] [signal] checkpointing at {where} before exit ...", flush=True)
        ckpt.save_state(accum, meta)
        print(f"[{T._ts()}] [signal] checkpoint written to {ckpt.accum_path}", flush=True)
    raise Preempted(where)


# ---------------------------------------------------------------------------
# CoFi -- sample-size sweep (docs-outer, snapshot at n break-points)
# ---------------------------------------------------------------------------

def cofi_size_snapshots(model, tokenizer, texts, max_length, batch_size,
                        target_params, size_breaks, role="?", corpus="?",
                        ckpt=None, ckpt_every=0):
    """Yield (actual_n, diag_fp16) as more docs are folded into the Fisher.
    Faithful to test.compute_cofi restricted to the doc prefix.

    Resumable via `ckpt`: the accumulator (+ doc/step counters) and every
    breakpoint diagonal are written to disk, so a killed run re-yields the
    finished break-points from disk and folds only the remaining documents.

    `role` is "base" or the unlearned method's label, `corpus` the corpus
    name -- both only used to make the progress prints unambiguous."""
    breaks = sorted(b for b in size_breaks if b <= len(texts))
    n_breaks = len(breaks)

    status, accum, meta = _ckpt_bootstrap(ckpt, _cofi_finalize)
    if status == "complete":
        for pt in meta["points"]:
            print(f"      [{T._ts()}] [CoFi][{corpus}][{role}] n={pt} loaded from checkpoint")
            yield (pt, ckpt.load_diag(pt))
        return

    batch_done = n_steps = seen = 0
    points, point_state = [], {}
    if status == "resume":
        batch_done = meta.get("progress", 0)
        n_steps = meta.get("n_steps", 0)
        seen = meta.get("seen", 0)
        points = list(meta.get("points", []))
        point_state = dict(meta.get("point_state", {}))
        print(f"      [{T._ts()}] [CoFi][{corpus}][{role}] resuming after {batch_done} "
              f"batches ({seen} docs); {len(points)}/{n_breaks} break-points already on disk")
        for pt in points:
            print(f"      [{T._ts()}] [CoFi][{corpus}][{role}] n={pt} loaded from checkpoint")
            yield (pt, ckpt.load_diag(pt))
        if len(points) >= n_breaks:
            if ckpt is not None:
                ckpt.mark_complete({"points": points, "point_state": point_state,
                                    "progress": batch_done, "n_steps": n_steps,
                                    "seen": seen})
            return

    model.eval()
    original = _set_target_grads(model, target_params)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    input_device = T._get_input_device(model)
    amp_dtype = T._autocast_dtype()

    if accum is None:
        accum = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu")
                 for name, p in target_params}

    def _make_hook(name):
        target = accum[name]

        def _hook(p):
            if p.grad is None:
                return
            g = p.grad.detach().float()
            g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
            g.div_(_SQRT_SCALE_C)
            target.add_(g.pow_(2).cpu())
            p.grad = None
        return _hook

    handles = [p.register_post_accumulate_grad_hook(_make_hook(name))
               for name, p in target_params]

    def _meta():
        return {"progress": batch_done, "n_steps": n_steps, "seen": seen,
                "points": points, "point_state": point_state}

    bi = len(points)
    try:
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            if batch_idx < batch_done:
                continue                       # already folded into the loaded accum
            enc = tokenizer(texts[i:i + batch_size], return_tensors="pt",
                            padding=True, truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(input_device)
            attention_mask = enc["attention_mask"].to(input_device)
            if input_ids.numel() == 0:
                batch_done = batch_idx + 1
                continue
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                sl = logits[..., :-1, :].contiguous()
                st = labels[..., 1:].contiguous().to(sl.device)
                st = st.masked_fill(st >= sl.size(-1), -100)
                loss = torch.nn.functional.cross_entropy(
                    sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
            loss.backward()
            n_steps += 1
            seen += input_ids.size(0)
            batch_done = batch_idx + 1
            del input_ids, attention_mask, labels, enc, loss, logits, sl, st
            while bi < n_breaks and seen >= breaks[bi]:
                diag = _cofi_finalize(accum, n_steps)
                points.append(seen)
                point_state[str(seen)] = {"denom": n_steps, "progress": batch_done}
                if ckpt is not None:
                    ckpt.save_state(accum, _meta())     # bookkeeping first ...
                    ckpt.save_diag(seen, diag)          # ... then the diagonal
                    print(f"      [{T._ts()}] [CoFi][{corpus}][{role}] snapshot n={seen} "
                          f"saved (resumable)")
                else:
                    print(f"      [{T._ts()}] [CoFi][{corpus}][{role}] snapshot n={seen}")
                yield (seen, diag)
                bi += 1
            if ckpt is not None and ckpt_every and batch_done % ckpt_every == 0:
                ckpt.save_state(accum, _meta())
            _maybe_stop(ckpt, accum, _meta(), f"CoFi {corpus}/{role} batch {batch_done}")
        if ckpt is not None:
            ckpt.mark_complete(_meta())
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing_disable()
        for p, rg in original:
            p.requires_grad_(rg)


# ---------------------------------------------------------------------------
# CHess -- shared probe machinery + two sweep structures
# ---------------------------------------------------------------------------

def _chess_setup(model, target_params):
    model.eval()
    original = _set_target_grads(model, target_params)
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    return original


def _chess_hooks(accum, g_plus_cpu, z_dict_ref, phase, target_params):
    def _make(name):
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
                hvp = gp.sub_(g).div_(2 * _EPS).mul_(z)
                hvp.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
                hvp.div_(_SCALE_C)
                target.add_(hvp)
        return _hook
    return [p.register_post_accumulate_grad_hook(_make(name)) for name, p in target_params]


def _chess_finalize(accum, denom):
    out = {}
    for name, a in accum.items():
        x = a / denom
        x.abs_().add_(_LAMBDA / _SCALE_C).log_().add_(math.log(_SCALE_C))
        out[name] = x.half()
    return out


def _chess_one_probe(model, tokenizer, text_batch, max_length, target_params,
                     z_local, g_plus_cpu, z_dict_ref, phase, input_device, amp_dtype):
    """+eps / -eps finite-difference for one Rademacher probe on one batch;
    hooks accumulate the HVP. Returns True if the batch was non-empty."""
    enc = tokenizer(text_batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(input_device)
    attention_mask = enc["attention_mask"].to(input_device)
    if input_ids.numel() == 0:
        return False
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    z_dict_ref.clear()
    z_dict_ref.update(z_local)
    for name, p in target_params:
        p.data.add_(z_local[name].to(p.device), alpha=_EPS)

    phase[0] = "plus"
    g_plus_cpu.clear()
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        sl = out.logits[..., :-1, :].contiguous()
        st = labels[..., 1:].contiguous().to(sl.device)
        st = st.masked_fill(st >= sl.size(-1), -100)
        loss = torch.nn.functional.cross_entropy(
            sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
    loss.backward()

    for name, p in target_params:
        p.data.sub_(z_local[name].to(p.device), alpha=2 * _EPS)

    phase[0] = "minus"
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        sl = out.logits[..., :-1, :].contiguous()
        st = labels[..., 1:].contiguous().to(sl.device)
        st = st.masked_fill(st >= sl.size(-1), -100)
        loss = torch.nn.functional.cross_entropy(
            sl.view(-1, sl.size(-1)), st.view(-1), ignore_index=-100)
    loss.backward()

    for name, p in target_params:
        p.data.add_(z_local[name].to(p.device), alpha=_EPS)
    z_dict_ref.clear()
    del enc, input_ids, attention_mask, labels, out, sl, st, loss
    return True


def _rademacher(target_params, base_seed, corpus, batch_idx, k):
    seed = T._chess_probe_seed(base_seed, corpus, batch_idx, k)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return {name: (torch.randint(0, 2, p.shape, generator=gen) * 2 - 1).to(p.dtype)
            for name, p in target_params}


def chess_probe_snapshots(model, tokenizer, texts, max_length, batch_size,
                          target_params, probe_breaks, base_seed, corpus, ckpt=None,
                          role="?", ckpt_every=0):
    """probes-outer / docs-inner (all `texts`); yield (K, diag) at each K.

    Resumable when `ckpt` is given. Concretely, for K in {2,4,8}: once K=2 has
    been computed its diagonal AND the raw accumulator are on disk, so a job
    killed at K=8 re-yields K=2 and K=4 from disk on the next run, reloads the
    accumulator as of K=4, and probes 5..8 only. Earlier diagonals are never
    overwritten or removed. `ckpt_every` (>0) additionally checkpoints the
    accumulator every N probes, bounding the loss inside a K interval.

    `role` is "base" or the unlearned method's label, used only to make the
    progress prints unambiguous.
    """
    KS = sorted(set(probe_breaks))
    M = KS[-1]
    n_proc_all = len(range(0, len(texts), batch_size))

    status, accum, meta = _ckpt_bootstrap(ckpt, _chess_finalize)
    if status == "complete":
        for pt in meta["points"]:
            print(f"      [{T._ts()}] [CHess][{corpus}][{role}] probe K={pt} loaded from checkpoint")
            yield (pt, ckpt.load_diag(pt))
        return

    k_done, n_proc, points, point_state = 0, n_proc_all, [], {}
    if status == "resume":
        k_done = meta.get("progress", 0)
        n_proc = meta.get("n_proc", n_proc_all)
        points = list(meta.get("points", []))
        point_state = dict(meta.get("point_state", {}))
        print(f"      [{T._ts()}] [CHess][{corpus}][{role}] resuming probe sweep after "
              f"{k_done}/{M} probes; {len(points)} break-point(s) already on disk")
        for pt in points:
            print(f"      [{T._ts()}] [CHess][{corpus}][{role}] probe K={pt} loaded from checkpoint")
            yield (pt, ckpt.load_diag(pt))
        if k_done >= M:
            if ckpt is not None:
                ckpt.mark_complete({"progress": k_done, "n_proc": n_proc,
                                    "points": points, "point_state": point_state})
            return

    original = _chess_setup(model, target_params)
    input_device = T._get_input_device(model)
    amp_dtype = T._autocast_dtype()
    if accum is None:
        accum = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu")
                 for name, p in target_params}
    g_plus_cpu, z_dict_ref, phase = {}, {}, ["plus"]
    handles = _chess_hooks(accum, g_plus_cpu, z_dict_ref, phase, target_params)

    def _meta(k):
        return {"progress": k, "n_proc": n_proc, "points": points,
                "point_state": point_state}

    try:
        for k in range(k_done, M):
            for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
                z = _rademacher(target_params, base_seed, corpus, batch_idx, k)
                _chess_one_probe(model, tokenizer, texts[i:i + batch_size], max_length,
                                 target_params, z, g_plus_cpu, z_dict_ref, phase,
                                 input_device, amp_dtype)
            if (k + 1) in KS:
                denom = n_proc * (k + 1)
                diag = _chess_finalize(accum, denom)
                points.append(k + 1)
                point_state[str(k + 1)] = {"denom": denom, "progress": k + 1}
                if ckpt is not None:
                    ckpt.save_state(accum, _meta(k + 1))   # bookkeeping first ...
                    ckpt.save_diag(k + 1, diag)            # ... then the diagonal
                    print(f"      [{T._ts()}] [CHess][{corpus}][{role}] probe K={k+1} "
                          f"saved (resumable)")
                else:
                    print(f"      [{T._ts()}] [CHess][{corpus}][{role}] probe snapshot K={k+1}")
                yield (k + 1, diag)
                del diag
            elif ckpt is not None and ckpt_every and (k + 1) % ckpt_every == 0:
                ckpt.save_state(accum, _meta(k + 1))
                print(f"      [{T._ts()}] [CHess][{corpus}][{role}] intermediate state "
                      f"saved after probe {k+1}")
            _maybe_stop(ckpt, accum, _meta(k + 1),
                        f"CHess-probe {corpus}/{role} probe {k+1}")
        if ckpt is not None:
            ckpt.mark_complete(_meta(M))
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        for p, rg in original:
            p.requires_grad_(rg)


def chess_size_snapshots(model, tokenizer, texts, max_length, batch_size,
                         target_params, k_fixed, size_breaks, base_seed, corpus, ckpt=None,
                         role="?", ckpt_every=0):
    """docs-outer / probes-inner (K=k_fixed); yield (actual_n, diag) at n breaks.
    Resumable exactly like chess_probe_snapshots, checkpointed per n break-point
    (and every `ckpt_every` batches if set). Nothing is ever deleted.

    `role` is "base" or the unlearned method's label, used only to make the
    progress prints unambiguous."""
    breaks = sorted(b for b in size_breaks if b <= len(texts))
    n_breaks = len(breaks)

    status, accum, meta = _ckpt_bootstrap(ckpt, _chess_finalize)
    if status == "complete":
        for pt in meta["points"]:
            print(f"      [{T._ts()}] [CHess][{corpus}][{role}] size n={pt} loaded from checkpoint")
            yield (pt, ckpt.load_diag(pt))
        return

    batch_done, n_proc, seen, points, point_state = 0, 0, 0, [], {}
    if status == "resume":
        batch_done = meta.get("progress", 0)
        n_proc = meta.get("n_proc", 0)
        seen = meta.get("seen", 0)
        points = list(meta.get("points", []))
        point_state = dict(meta.get("point_state", {}))
        print(f"      [{T._ts()}] [CHess][{corpus}][{role}] resuming size sweep after "
              f"{batch_done} batches ({seen} docs); {len(points)}/{n_breaks} break-points on disk")
        for pt in points:
            print(f"      [{T._ts()}] [CHess][{corpus}][{role}] size n={pt} loaded from checkpoint")
            yield (pt, ckpt.load_diag(pt))
        if len(points) >= n_breaks:
            if ckpt is not None:
                ckpt.mark_complete({"progress": batch_done, "n_proc": n_proc,
                                    "seen": seen, "points": points,
                                    "point_state": point_state})
            return

    original = _chess_setup(model, target_params)
    input_device = T._get_input_device(model)
    amp_dtype = T._autocast_dtype()
    if accum is None:
        accum = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu")
                 for name, p in target_params}
    g_plus_cpu, z_dict_ref, phase = {}, {}, ["plus"]
    handles = _chess_hooks(accum, g_plus_cpu, z_dict_ref, phase, target_params)

    def _meta():
        return {"progress": batch_done, "n_proc": n_proc, "seen": seen,
                "points": points, "point_state": point_state}

    bi = len(points)
    try:
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            if batch_idx < batch_done:
                continue                                 # already folded into loaded accum
            batch = texts[i:i + batch_size]
            processed = False
            for k in range(k_fixed):
                ok = _chess_one_probe(model, tokenizer, batch, max_length, target_params,
                                      _rademacher(target_params, base_seed, corpus, batch_idx, k),
                                      g_plus_cpu, z_dict_ref, phase, input_device, amp_dtype)
                processed = processed or ok
            if processed:
                n_proc += 1
                seen += len(batch)
            batch_done = batch_idx + 1
            while bi < n_breaks and seen >= breaks[bi]:
                denom = n_proc * k_fixed
                diag = _chess_finalize(accum, denom)
                points.append(seen)
                point_state[str(seen)] = {"denom": denom, "progress": batch_done}
                if ckpt is not None:
                    ckpt.save_state(accum, _meta())     # bookkeeping first ...
                    ckpt.save_diag(seen, diag)          # ... then the diagonal
                    print(f"      [{T._ts()}] [CHess][{corpus}][{role}] size n={seen} "
                          f"(K={k_fixed}) saved (resumable)")
                else:
                    print(f"      [{T._ts()}] [CHess][{corpus}][{role}] size snapshot "
                          f"n={seen} (K={k_fixed})")
                yield (seen, diag)
                del diag
                bi += 1
            if ckpt is not None and ckpt_every and batch_done % ckpt_every == 0:
                ckpt.save_state(accum, _meta())
            _maybe_stop(ckpt, accum, _meta(),
                        f"CHess-size {corpus}/{role} batch {batch_done}")
        if ckpt is not None:
            ckpt.mark_complete(_meta())
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        for p, rg in original:
            p.requires_grad_(rg)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _free():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bucket_done(bucket):
    """Is this method's sweep finished? Back-compatible with pre-`_done` JSONs."""
    if not bucket:
        return False
    if "_done" in bucket:
        return bool(bucket["_done"])
    return bool(bucket.get("both"))          # legacy results: assume complete


def _new_bucket(existing=None):
    b = dict(existing) if existing else {}
    for s in _SUBSETS:
        b.setdefault(s, {})
    b["_done"] = False
    return b


def _run_sweep(gen_factory, base_model_path, method_specs, corpus, work,
               torch_dtype, want_convergence, results_bucket, convergence_bucket,
               save_cb, sweep_tag, cleanup_accums=False):
    """Compute base snapshots, then each method's drops (+ optional convergence)
    streaming against the base snapshots.

    Resumable at three levels:
      * methods already finished in `results_bucket` (from a prior run's JSON)
        are skipped entirely;
      * per break-point drops are committed to the JSON as soon as they are
        computed, so a method killed at K=8 keeps its K=2/K=4 numbers;
      * the base and each in-progress method checkpoint their raw accumulator +
        every break-point diagonal to disk under a role-specific prefix, so a
        job killed mid-accumulation resumes from the last saved break-point
        (e.g. K=4 -> continue to K=8) rather than recomputing it.

    Nothing on disk is deleted; `cleanup_accums` only drops the raw resume
    accumulators (never the diagonals) once a role has finished.
    """
    missing = [(l, p) for (l, p) in method_specs
               if not _bucket_done(results_bucket.get(l))]
    if not missing:
        print(f"[{T._ts()}]   [{sweep_tag}][{corpus}] [skip] all {len(method_specs)} methods cached")
        return

    def _ckpt(role):
        return Ckpt(work / f"ck__{_safe(corpus)}__{sweep_tag}__{role}")

    base_ckpt = _ckpt("base")
    base_paths, base_conv = {}, convergence_bucket.get("base", {})

    # Base already finished in a previous run? Then we don't even need to load
    # the 8B weights -- the diagonals we compare against are already on disk.
    bmeta = base_ckpt.load_meta()
    base_cached = (bmeta is not None and bmeta.get("complete")
                   and bmeta.get("points") and base_ckpt.have_diags(bmeta["points"])
                   and (not want_convergence or base_conv))
    if base_cached:
        base_paths = {p: base_ckpt.diag_path(p) for p in bmeta["points"]}
        print(f"[{T._ts()}]   [{sweep_tag}][{corpus}] base [cached] "
              f"points={bmeta['points']}  ({len(missing)}/{len(method_specs)} methods to do)")
    else:
        print(f"[{T._ts()}]   [{sweep_tag}][{corpus}] base ... "
              f"({len(missing)}/{len(method_specs)} methods to do)")
        base_model, tok = T.load_model_and_tokenizer(base_model_path, torch_dtype)
        tp = T.get_target_params(base_model, T.DEFAULT_TARGET_MODULES)
        prev = None
        if want_convergence:
            convergence_bucket["base"] = base_conv
        for point, diag in gen_factory(base_model, tok, tp, corpus, base_ckpt, role="base"):
            base_paths[point] = base_ckpt.diag_path(point)   # generator saved it there
            if want_convergence and prev is not None and str(point) not in base_conv:
                base_conv[str(point)] = _rel_frob_dist(diag, prev, list(diag.keys()))
                save_cb(report=False)
            prev = diag
        del base_model, tok, tp, prev
        _free()
        if cleanup_accums:
            base_ckpt.drop_accum()
        if want_convergence:
            save_cb(report=False)

    for label, path in missing:
        print(f"[{T._ts()}]   [{sweep_tag}][{corpus}] method {label} ...")
        m_ckpt = _ckpt(_safe(label))
        m, tok = T.load_model_and_tokenizer(path, torch_dtype)
        tp = T.get_target_params(m, T.DEFAULT_TARGET_MODULES)
        bucket = _new_bucket(results_bucket.get(label))
        results_bucket[label] = bucket            # partial results are kept, not discarded
        mconv = convergence_bucket.get(label, {})
        if want_convergence:
            convergence_bucket[label] = mconv
        prev = None
        for point, diag in gen_factory(m, tok, tp, corpus, m_ckpt, role=label):
            need_drop = any(str(point) not in bucket[s] for s in _SUBSETS)
            if need_drop:
                base_diag = torch.load(base_paths[point], weights_only=True)
                for s in _SUBSETS:
                    keys = _subset_keys(diag.keys(), s)
                    bucket[s][str(point)] = _frob_drop(base_diag, diag, keys)
                del base_diag
            if want_convergence and prev is not None and str(point) not in mconv:
                mconv[str(point)] = _rel_frob_dist(diag, prev, list(diag.keys()))
            prev = diag
            save_cb(report=False)                 # commit this break-point immediately
            _free()
        bucket["_done"] = True
        del m, tok, tp, prev
        _free()
        if cleanup_accums:
            m_ckpt.drop_accum()
        save_cb()                                 # persist + re-report after each method


def run_compute(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    T._BENCHMARK = "wmdp"
    T._CHESS_SEED = args.base_seed

    corpora, forget_names, retain_names = T.get_benchmark_corpora("wmdp")
    all_corpora = forget_names + retain_names
    selected = args.corpora if args.corpora else all_corpora
    KS = sorted(set(args.probe_counts))
    SIZES = sorted(set(args.sizes))
    methods = [(m.split("::", 1)[0], m.split("::", 1)[1]) for m in args.methods]

    torch_dtype = torch.bfloat16
    # CoFi and CHess are independent metrics with wildly different cost (see
    # --metric), so they get separate output files -- each resumable on its
    # own, so a 'cofi'-only run can never touch (let alone lose) CHess
    # progress, and vice versa.
    cofi_json = _cofi_json(args.out_tag)
    chess_json = _chess_json(args.out_tag)

    def _meta():
        return {
            "benchmark": "wmdp", "subset": 0,
            "probe_counts": KS, "sizes": SIZES,
            "size_k": args.size_k, "probe_n": args.probe_n,
            "base_model": args.base_model,
            "methods": {lbl: path for lbl, path in methods},
            "forget_corpora": [c for c in selected if "forget" in c],
            "retain_corpora": [c for c in selected if "forget" not in c],
        }

    def _load(path):
        if not path.exists() or args.fresh:
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[{T._ts()}] [warn] could not read {path} ({e}); starting fresh")
            return {}

    cofi_prev = _load(cofi_json)
    cofi_results = cofi_prev.get("results", {})
    cofi_results.setdefault("cofi", {}).setdefault("size", {})
    if cofi_prev:
        print(f"[{T._ts()}] resuming CoFi from {cofi_json} "
              f"(already-done methods will be skipped)")

    chess_prev = _load(chess_json)
    chess_results = chess_prev.get("results", {})
    chess_results.setdefault("chess", {}).setdefault("probe", {})
    chess_results["chess"].setdefault("size", {})
    convergence = chess_prev.get("convergence", {})   # convergence[corpus][label][str(K)]
    if chess_prev:
        print(f"[{T._ts()}] resuming CHess from {chess_json} "
              f"(already-done methods will be skipped)")

    def _atomic_write(path, payload):
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(path)   # atomic: a kill mid-write can't corrupt the file

    def _save_cofi(report=True):
        # Fires after every break-point commits and after every algorithm's
        # CoFi sweep (see _run_sweep), so the CoFi-only report/plot stays
        # live-updated as atu/npo/rmu each finish -- and also fires once here
        # even if everything was already cached, so re-running on saved
        # checkpoints still (re)produces it.
        payload = {**_meta(), "results": cofi_results}
        _atomic_write(cofi_json, payload)
        if report:
            analyze_cofi(payload)
            if not args.no_plots:
                make_plots_cofi(payload)

    def _save_chess(report=True):
        # Same idea for CHess (per break-point write; report per algorithm).
        payload = {**_meta(), "results": chess_results, "convergence": convergence}
        _atomic_write(chess_json, payload)
        if report:
            analyze_chess(payload)
            if not args.no_plots:
                make_plots_chess(payload)

    for corpus in selected:
        print(f"\n{'='*70}\n[{T._ts()}] CORPUS: {corpus}\n{'='*70}")
        texts = T._subset_texts(corpora[corpus], corpus, 0, args.n_subset)
        texts = texts[:max(SIZES + [args.probe_n])]
        print(f"  {len(texts)} docs available (sub0)")
        too_big = [s for s in SIZES if s > len(texts)]
        if too_big:
            print(f"  [warn] --sizes {too_big} exceed available docs ({len(texts)}) "
                  f"and will be silently skipped -- raise --n-subset (currently "
                  f"{args.n_subset}) to at least {max(too_big)} to reach them.")

        # setdefault, so loaded results for this corpus are NOT clobbered.
        cofi_results["cofi"]["size"].setdefault(corpus, {})
        chess_results["chess"]["probe"].setdefault(corpus, {})
        chess_results["chess"]["size"].setdefault(corpus, {})
        convergence.setdefault(corpus, {})

        # ---- CoFi size sweep (cheap, but checkpointed all the same) ----
        if args.metric in ("both", "cofi"):
            print(f"[{T._ts()}] [cofi_size][{corpus}] CoFi size sweep {SIZES}")

            def cofi_gen(model, tok, tp, corp, ckpt=None, role="?"):
                return cofi_size_snapshots(model, tok, texts, args.max_length,
                                           args.batch_size, tp, SIZES, role=role,
                                           corpus=corp, ckpt=ckpt,
                                           ckpt_every=args.ckpt_every_batches)
            _run_sweep(cofi_gen, args.base_model, methods, corpus, work, torch_dtype,
                       False, cofi_results["cofi"]["size"][corpus], {}, _save_cofi,
                       sweep_tag="cofi_size", cleanup_accums=args.cleanup_accums)
        else:
            print(f"[{T._ts()}] [{corpus}] CoFi size sweep skipped (--metric {args.metric})")

        # ---- CHess probe sweep (fixed n = probe_n) -- checkpointed per K ----
        # ---- CHess size sweep (fixed K = size_k) -- checkpointed per n ----
        if args.metric in ("both", "chess"):
            print(f"[{T._ts()}] [chess_probe][{corpus}] CHess probe sweep {KS}  (n={args.probe_n})")

            def chess_probe_gen(model, tok, tp, corp, ckpt=None, role="?"):
                return chess_probe_snapshots(model, tok, texts[:args.probe_n], args.max_length,
                                             args.batch_size, tp, KS, args.base_seed, corp,
                                             ckpt=ckpt, role=role,
                                             ckpt_every=args.ckpt_every_probes)
            _run_sweep(chess_probe_gen, args.base_model, methods, corpus, work, torch_dtype,
                       True, chess_results["chess"]["probe"][corpus], convergence[corpus],
                       _save_chess, sweep_tag="chess_probe",
                       cleanup_accums=args.cleanup_accums)

            print(f"[{T._ts()}] [chess_size][{corpus}] CHess size sweep {SIZES}  (K={args.size_k})")

            def chess_size_gen(model, tok, tp, corp, ckpt=None, role="?"):
                return chess_size_snapshots(model, tok, texts, args.max_length,
                                            args.batch_size, tp, args.size_k, SIZES,
                                            args.base_seed, corp, ckpt=ckpt, role=role,
                                            ckpt_every=args.ckpt_every_batches)
            _run_sweep(chess_size_gen, args.base_model, methods, corpus, work, torch_dtype,
                       False, chess_results["chess"]["size"][corpus], {}, _save_chess,
                       sweep_tag="chess_size", cleanup_accums=args.cleanup_accums)
        else:
            print(f"[{T._ts()}] [{corpus}] CHess sweeps skipped (--metric {args.metric})")

    if args.metric in ("both", "cofi"):
        _save_cofi()
        print(f"\n[{T._ts()}] wrote {cofi_json}")
    if args.metric in ("both", "chess"):
        _save_chess()
        print(f"[{T._ts()}] wrote {chess_json}")


# ---------------------------------------------------------------------------
# Reporting  (reads JSON; no GPU)
# ---------------------------------------------------------------------------

def _fmt(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "   n/a"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def _get(d, k):
    return d.get(str(k), d.get(k))


def _has_data(sweep_results):
    return any(sweep_results.get(corpus) for corpus in sweep_results)


def _points_of(sweep_results, subset="both"):
    """Union of break-points seen so far (a partially-finished method may have
    fewer than the base, which is fine -- missing cells print as n/a)."""
    pts = set()
    for corpus in sweep_results:
        for label in sweep_results[corpus]:
            pts.update(int(p) for p in sweep_results[corpus][label].get(subset, {}))
    return sorted(pts)


def _drop_table(title, sweep_results, forget, labels, xname):
    pts = _points_of(sweep_results)
    print(f"\n{title}")
    for label in labels:
        print(f"\n  method: {label}")
        print("  " + "corpus".ljust(14) + "".join(f"{xname}={p}".rjust(12) for p in pts))
        for corpus in sweep_results:
            if label not in sweep_results[corpus]:
                continue
            row = sweep_results[corpus][label]["both"]
            tag = "F" if corpus in forget else "R"
            print(f"  [{tag}] {corpus.ljust(10)}"
                  + "".join(_fmt(_get(row, p)).rjust(12) for p in pts))


def _margin_table(title, sweep_results, forget, labels, xname):
    pts = _points_of(sweep_results)
    print(f"\n{title}")
    print("    margin = mean(forget drop) - mean(retain drop); stable & large => robust class")
    print("  " + "method".ljust(18) + "".join(f"{xname}={p}".rjust(12) for p in pts))
    for label in labels:
        cells = []
        for p in pts:
            fv, rv = [], []
            for corpus in sweep_results:
                if label not in sweep_results[corpus]:
                    continue
                v = _get(sweep_results[corpus][label]["both"], p)
                if v is None or not math.isfinite(v):
                    continue
                (fv if corpus in forget else rv).append(v)
            cells.append(sum(fv)/len(fv) - sum(rv)/len(rv) if fv and rv else float("nan"))
        print("  " + label.ljust(18) + "".join(_fmt(c).rjust(12) for c in cells))


def _subset_table(title, sweep_results, forget, labels):
    pts = _points_of(sweep_results)
    pmax = pts[-1] if pts else None
    print(f"\n{title}  (at largest point = {pmax})")
    print("  " + "method".ljust(16) + "corpus".ljust(14)
          + "attn".rjust(12) + "ffn".rjust(12) + "both".rjust(12))
    for label in labels:
        for corpus in sweep_results:
            if label not in sweep_results[corpus]:
                continue
            r = sweep_results[corpus][label]
            tag = "F" if corpus in forget else "R"
            print(f"  {label.ljust(14)}[{tag}] {corpus.ljust(10)}"
                  + _fmt(_get(r["attn"], pmax)).rjust(12)
                  + _fmt(_get(r["ffn"], pmax)).rjust(12)
                  + _fmt(_get(r["both"], pmax)).rjust(12))


def _pending_note(sweep_results, labels):
    """Flag methods whose sweep is only partially computed (resume will finish)."""
    partial = sorted({f"{corpus}/{label}"
                      for corpus in sweep_results
                      for label in sweep_results[corpus]
                      if label in labels and not _bucket_done(sweep_results[corpus][label])})
    if partial:
        print(f"\n  [partial] still in progress (rerun to resume): {', '.join(partial)}")


def analyze_cofi(payload):
    """CoFi-only report: axis A (sample size) + axis C (parameter subset)."""
    results = payload["results"]
    if "cofi" not in results or not _has_data(results["cofi"]["size"]):
        print("\n[skip] no CoFi results found (run with --metric cofi or both)")
        return
    forget = set(payload["forget_corpora"])
    labels = list(payload["methods"].keys())

    print("\n" + "#" * 78)
    print("# CoFi ablation  (WMDP, sub0, Llama-3.1-8B)")
    print(f"#   sizes={payload['sizes']}")
    print("#" * 78)

    _drop_table("[A1] CoFi drop vs sample size n  (subset=both)",
                results["cofi"]["size"], forget, labels, "n")
    _margin_table("[A2] CoFi class margin vs n", results["cofi"]["size"], forget, labels, "n")
    _subset_table("[C1] CoFi parameter-subset sensitivity", results["cofi"]["size"], forget, labels)
    _pending_note(results["cofi"]["size"], labels)


def analyze_chess(payload):
    """CHess-only report: axis A (sample size) + axis B (probe count) + axis C
    (parameter subset) + probe-count convergence."""
    results = payload["results"]
    if "chess" not in results or not (_has_data(results["chess"]["probe"])
                                      or _has_data(results["chess"]["size"])):
        print("\n[skip] no CHess results found (run with --metric chess or both)")
        return
    forget = set(payload["forget_corpora"])
    labels = list(payload["methods"].keys())

    print("\n" + "#" * 78)
    print("# CHess ablation  (WMDP, sub0, Llama-3.1-8B)")
    print(f"#   probes={payload['probe_counts']}  sizes={payload['sizes']}  "
          f"size-sweep K={payload['size_k']}  probe-sweep n={payload['probe_n']}")
    print("#" * 78)

    _drop_table("[A3] CHess drop vs sample size n  (subset=both)",
                results["chess"]["size"], forget, labels, "n")
    _margin_table("[A4] CHess class margin vs n", results["chess"]["size"], forget, labels, "n")
    _pending_note(results["chess"]["size"], labels)

    _drop_table("[B1] CHess drop vs probe count K  (subset=both)",
                results["chess"]["probe"], forget, labels, "K")
    _margin_table("[B2] CHess class margin vs K", results["chess"]["probe"], forget, labels, "K")

    _subset_table("[C2] CHess parameter-subset sensitivity", results["chess"]["probe"], forget, labels)
    _pending_note(results["chess"]["probe"], labels)

    conv = payload.get("convergence", {})
    KS = payload["probe_counts"]
    conv_Ks = [k for k in KS if k != KS[0]]
    print("\n[B3] CHess estimator convergence ||d_K - d_{K/2}|| / ||d_K||  (mean over corpora; ~K^-1/2)")
    print("  " + "model".ljust(18) + "".join(f"K={k}".rjust(12) for k in conv_Ks))
    for label in ["base"] + labels:
        vals = []
        for k in conv_Ks:
            per = [_get(conv[c][label], k) for c in conv if label in conv[c]]
            per = [v for v in per if v is not None and math.isfinite(v)]
            vals.append(sum(per)/len(per) if per else float("nan"))
        print("  " + label.ljust(18) + "".join(_fmt(v).rjust(12) for v in vals))
    if conv_Ks:
        ref = [(conv_Ks[0]/k) ** 0.5 for k in conv_Ks]
        print("    K^-1/2 ref (norm. to first): "
              + "  ".join(f"K={k}:{v:.3f}" for k, v in zip(conv_Ks, ref)))


def _drop_vs_x_fig(sweep_results, forget, labels, xname, title, out_stem, logx_base=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e})")
        return
    pts = _points_of(sweep_results)
    if not pts:
        return
    fig, axes = plt.subplots(1, len(labels), figsize=(max(4, 3*len(labels)), 3.4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        for corpus in sweep_results:
            if label not in sweep_results[corpus]:
                continue
            row = sweep_results[corpus][label]["both"]
            xy = [(p, _get(row, p)) for p in pts]
            xy = [(x, y) for x, y in xy if y is not None and math.isfinite(y)]
            if not xy:
                continue
            ax.plot([x for x, _ in xy], [y for _, y in xy], marker="o",
                    ls="-" if corpus in forget else "--", lw=1.6, ms=4, label=corpus)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(xname)
        if logx_base:
            ax.set_xscale("log", base=logx_base)
            ax.set_xticks(pts)
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("drop")
    fig.suptitle(title + "  (solid=forget, dashed=retain)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT_DIR/out_stem}.png")


def _convergence_fig(payload):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e})")
        return
    conv = payload["convergence"]
    KS = payload["probe_counts"]
    labels = ["base"] + list(payload["methods"].keys())
    conv_Ks = [k for k in KS if k != KS[0]]
    if not conv_Ks:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    anchor = None
    for label in labels:
        ys = []
        for k in conv_Ks:
            per = [_get(conv[c][label], k) for c in conv if label in conv[c]]
            per = [v for v in per if v is not None and math.isfinite(v)]
            ys.append(sum(per)/len(per) if per else float("nan"))
        ax.plot(conv_Ks, ys, marker="o", lw=1.6, ms=4, label=label)
        if anchor is None and ys and math.isfinite(ys[0]):
            anchor = ys[0]
    if anchor:
        ax.plot(conv_Ks, [anchor*(conv_Ks[0]/k) ** 0.5 for k in conv_Ks],
                ls=":", color="black", label=r"$K^{-1/2}$ guide")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(conv_Ks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Hutchinson probes K")
    ax.set_ylabel(r"$\|d_K-d_{K/2}\| / \|d_K\|$")
    ax.set_title("CHess diagonal convergence", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"chess_convergence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT_DIR/'chess_convergence.png'}")


def make_plots_cofi(payload):
    results = payload["results"]
    if "cofi" not in results or not _has_data(results["cofi"]["size"]):
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forget = set(payload["forget_corpora"])
    labels = list(payload["methods"].keys())
    _drop_vs_x_fig(results["cofi"]["size"], forget, labels, "sample size n",
                   "CoFi drop vs sample size", "cofi_drop_vs_n")


def make_plots_chess(payload):
    results = payload["results"]
    if "chess" not in results or not (_has_data(results["chess"]["probe"])
                                      or _has_data(results["chess"]["size"])):
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forget = set(payload["forget_corpora"])
    labels = list(payload["methods"].keys())
    _drop_vs_x_fig(results["chess"]["size"], forget, labels, "sample size n",
                   "CHess drop vs sample size", "chess_drop_vs_n")
    _drop_vs_x_fig(results["chess"]["probe"], forget, labels, "Hutchinson probes K",
                   "CHess drop vs probe count", "chess_drop_vs_K", logx_base=2)
    _convergence_fig(payload)


def show_checkpoints(args):
    """Print what is already on disk (per corpus / sweep / role) and stop."""
    work = Path(args.work_dir)
    metas = sorted(work.glob("ck__*.meta.json"))
    if not metas:
        print(f"No checkpoints under {work}")
        return
    print(f"Checkpoints under {work}:\n")
    print("  " + "corpus".ljust(14) + "sweep".ljust(14) + "role".ljust(10)
          + "state".ljust(12) + "break-points")
    total = 0
    for mp in metas:
        stem = mp.name[:-len(".meta.json")]
        parts = stem.split("__")
        corpus, sweep, role = (parts + ["?", "?", "?"])[1:4]
        try:
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            continue
        pts = meta.get("points", [])
        state = "complete" if meta.get("complete") else f"partial({meta.get('progress', 0)})"
        print("  " + corpus.ljust(14) + sweep.ljust(14) + role.ljust(10)
              + state.ljust(12) + str(pts))
        for f_ in work.glob(stem + ".*"):
            try:
                total += f_.stat().st_size
            except OSError:
                pass
    print(f"\n  total checkpoint size: {total / 2**30:.1f} GiB")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B")
    p.add_argument("--methods", nargs="+", default=[],
                   help="LABEL::CKPT_PATH entries.")
    p.add_argument("--metric", choices=["both", "cofi", "chess"], default="both",
                   help="Restrict compute to one metric this run: 'cofi' (axis A "
                        "only, cheap) or 'chess' (axes A/B/C, expensive). CoFi and "
                        "CHess are written to separate result files (cofi_ablation_*  "
                        "/ chess_ablation_*), each independently resumable, so you can "
                        "finish 'cofi' first and run 'chess' separately later without "
                        "the two ever touching each other's progress (default: both).")
    p.add_argument("--probe-counts", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--size-k", type=int, default=16,
                   help="Fixed Hutchinson K for the CHess sample-size sweep "
                        "(default: most converged K).")
    p.add_argument("--probe-n", type=int, default=200,
                   help="Fixed #docs for the CHess probe sweep.")
    p.add_argument("--corpora", nargs="*", default=None)
    p.add_argument("--n-subset", type=int, default=200)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument("--work-dir", default=str(OUT_DIR / "_snapshots"),
                   help="Where the resumable checkpoints live. Must be the SAME "
                        "path across runs for resuming to work, and must survive "
                        "between jobs (i.e. not node-local scratch).")
    p.add_argument("--ckpt-every-probes", type=int, default=0,
                   help="CHess probe sweep: also write the raw accumulator every N "
                        "probes, not just at the K break-points -- bounds the work "
                        "lost inside a long K=4 -> K=8 stretch. Each write is ~28GB "
                        "for the 8B model, so 0 (break-points only) is the default.")
    p.add_argument("--ckpt-every-batches", type=int, default=0,
                   help="Same idea for the document-outer sweeps (CoFi size, CHess "
                        "size): checkpoint every N batches on top of the n "
                        "break-points. 0 = break-points only (default).")
    p.add_argument("--cleanup-accums", action="store_true",
                   help="After a role finishes, delete only its raw resume "
                        "accumulator (~28GB). Break-point diagonals are ALWAYS kept. "
                        "Off by default: nothing is ever deleted.")
    p.add_argument("--show-checkpoints", action="store_true",
                   help="List what is already checkpointed under --work-dir and exit.")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing results JSON and recompute from scratch. "
                        "(On-disk checkpoints under --work-dir are still reused; move "
                        "or rename --work-dir too for a truly clean run.)")
    p.add_argument("--out-tag", default="",
                   help="Suffix for the results JSONs, e.g. a corpus name. Lets "
                        "per-corpus jobs run in parallel writing distinct files "
                        "(cofi_ablation__<tag>.json / chess_ablation__<tag>.json); "
                        "analysis merges them.")
    return p.parse_args()


def _cofi_json(tag=""):
    name = f"cofi_ablation__{tag}.json" if tag else "cofi_ablation.json"
    return OUT_DIR / name


def _chess_json(tag=""):
    name = f"chess_ablation__{tag}.json" if tag else "chess_ablation.json"
    return OUT_DIR / name


def _merge_payloads(paths):
    """Merge per-corpus result JSONs into one payload (union over corpora)."""
    merged = {"results": {"cofi": {"size": {}}, "chess": {"probe": {}, "size": {}}},
              "convergence": {}}
    meta_keys = ("benchmark", "subset", "probe_counts", "sizes", "size_k",
                 "probe_n", "base_model", "methods")
    for path in paths:
        try:
            with open(path) as f:
                p = json.load(f)
        except Exception as e:
            print(f"[merge] skip {path} ({e})")
            continue
        for mk in meta_keys:
            if mk in p and mk not in merged:
                merged[mk] = p[mk]
        for metric, axes in p.get("results", {}).items():
            for axis, per_corpus in axes.items():
                merged["results"].setdefault(metric, {}).setdefault(axis, {}).update(per_corpus)
        merged["convergence"].update(p.get("convergence", {}))
    # Derive forget/retain corpus lists from whatever corpora ended up present.
    corpora = set()
    for metric in merged["results"].values():
        for per_corpus in metric.values():
            corpora.update(per_corpus.keys())
    merged["forget_corpora"] = sorted(c for c in corpora if "forget" in c)
    merged["retain_corpora"] = sorted(c for c in corpora if "forget" not in c)
    return merged


def _merge_all():
    """Merge every per-tag CoFi and CHess results JSON in OUT_DIR into one payload."""
    paths = sorted(OUT_DIR.glob("cofi_ablation__*.json")) + \
            (sorted([OUT_DIR / "cofi_ablation.json"]) if (OUT_DIR / "cofi_ablation.json").exists() else []) + \
            sorted(OUT_DIR.glob("chess_ablation__*.json")) + \
            (sorted([OUT_DIR / "chess_ablation.json"]) if (OUT_DIR / "chess_ablation.json").exists() else [])
    if not paths:
        raise SystemExit(f"No results JSONs found under {OUT_DIR}")
    print(f"[merge] combining {len(paths)} file(s): {[p.name for p in paths]}")
    payload = _merge_payloads(paths)
    dst = OUT_DIR / "ablation__merged.json"
    tmp = dst.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(dst)
    return payload


def main():
    args = parse_args()
    if args.show_checkpoints:
        show_checkpoints(args)
        return
    if not args.analyze_only:
        if not args.methods:
            raise SystemExit("--methods is required unless --analyze-only")
        _install_signal_handlers()
        # run_compute() already reports live: _save_cofi/_save_chess re-run
        # analyze_cofi/analyze_chess (+ plots) after every algorithm's sweep
        # commits, and once more unconditionally at the end of each metric's
        # block -- so re-running on fully-cached checkpoints still (re)prints
        # the report even though no new compute happened.
        try:
            run_compute(args)
        except Preempted as e:
            print(f"\n[{T._ts()}] preempted at {e}: state checkpointed under "
                  f"{args.work_dir}. Resubmit the same command to resume.")
            raise SystemExit(0)
        return
    # --analyze-only: merge every per-corpus JSON on disk and print both
    # reports (whichever has data; the other prints its own [skip] line).
    payload = _merge_all()
    analyze_cofi(payload)
    analyze_chess(payload)
    if not args.no_plots:
        make_plots_cofi(payload)
        make_plots_chess(payload)


if __name__ == "__main__":
    main()