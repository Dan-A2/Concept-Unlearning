"""Aggregate behavioural-eval results into tidy per (model, benchmark) tables.

Reads the JSON files written by behavioral_eval.py / relearn_attack.py, saved
next to each checkpoint at  <ckpt>/behavioral_eval_<bench_label>.json  where
bench_label is one of: wmdp, tofu-forget{01,05,10}, muse-news, muse-books.

Each file has a "results" dict whose keys depend on the benchmark:
    wmdp : {wmdp_bio:{acc,..}, wmdp_cyber:{acc,..}, mmlu:{acc,..}}
    tofu : {forget_q_a_prob, forget_q_a_rouge, forget_truth_ratio_mean,
            forget_quality, model_utility, ...}
    muse : {verbmem_forget_rouge, knowmem_forget_rouge, knowmem_retain_rouge}

Output, all in one run:
  1. one aligned table per (benchmark, model) of that benchmark's metrics;
  2. for WMDP, the behavioural question-group tripartite (forget /
     adjacent_retain / general_info) via behavioral_subjects;
  3. the focused adjacent per-question probe (Results/behavioral_adjacent) —
     correct / total per subject at full resolution;
  4. the relearning attack (Results/relearn) — relearned-model scores and a
     side-by-side unlearned -> relearned comparison per (benchmark, model).
Printed to the terminal and/or written as one text report per dataset.
"""

import argparse
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"
# Per-dataset text reports (one file per bench_label) are written here.
REPORT_DIR = PROJECT_ROOT / "Results" / "aggregate"


class _Tee:
    """Fan-out writes to several streams (e.g. a report file + the terminal)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass
# Focused per-question adjacent probe written by behavioral_adjacent.slurm.
ADJACENT_DIR = PROJECT_ROOT / "Results" / "behavioral_adjacent"
# Relearning-attack outputs (before/after/recovery) written by relearn_attack.py.
RELEARN_DIR = PROJECT_ROOT / "Results" / "relearn"

# Headline recovery metrics shown in the relearn tables, per family.
# (Recovery keys: WMDP tasks get an '_acc' suffix from relearn's flattener.)
RELEARN_METRICS = {
    "wmdp": [("wmdp_bio_acc", "WMDP-Bio"), ("wmdp_cyber_acc", "WMDP-Cyber"),
             ("mmlu_acc", "MMLU")],
    "tofu": [("forget_q_a_prob", "Forget-Prob"), ("forget_q_a_rouge", "Forget-ROUGE"),
             ("forget_truth_ratio_mean", "Forget-TR"), ("forget_quality", "Forget-Qual"),
             ("model_utility", "Model-Util")],
    "muse": [("verbmem_forget_rouge", "VerbMem-F"), ("knowmem_forget_rouge", "KnowMem-F"),
             ("knowmem_retain_rouge", "KnowMem-R")],
}

# Per-benchmark-family column spec: (metric_key, display, direction).
# direction: "down" = lower is better (more forgetting), "up" = higher better.
BENCH_METRICS = {
    "wmdp": [
        ("wmdp_bio",   "WMDP-Bio",  "down"),
        ("wmdp_cyber", "WMDP-Cyber", "down"),
        ("mmlu",       "MMLU",      "up"),
    ],
    "tofu": [
        ("forget_q_a_prob",         "Forget-Prob",  "down"),
        ("forget_q_a_rouge",        "Forget-ROUGE", "down"),
        ("forget_truth_ratio_mean", "Forget-TR",    "up"),
        ("forget_quality",          "Forget-Qual",  "up"),
        ("model_utility",           "Model-Util",   "up"),
    ],
    "muse": [
        ("verbmem_forget_rouge", "VerbMem-F", "down"),
        ("knowmem_forget_rouge", "KnowMem-F", "down"),
        ("knowmem_retain_rouge", "KnowMem-R", "up"),
    ],
}

_METHOD_DISPLAY = {
    "adaptive_rmu": "Adaptive-RMU", "atu": "ATU", "dpo": "DPO", "ga": "GA",
    "gd": "GD", "loku": "LoKU", "npo": "NPO", "obliviate": "Obliviate",
    "rmu": "RMU", "rsv": "RSV", "sim_npo": "SimNPO", "spul": "SPUL",
}


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def _family_of(bench_label: str) -> Optional[str]:
    """Map a bench_label to its metric family (wmdp / tofu / muse)."""
    if bench_label == "wmdp":
        return "wmdp"
    if bench_label.startswith("tofu"):
        return "tofu"
    if bench_label.startswith("muse"):
        return "muse"
    return None


def _parse_ckpt_name(name: str) -> Tuple[str, bool]:
    """('kl-Llama-3.1-8B_nu') -> ('Llama-3.1-8B', True)."""
    is_nu = name.endswith("_nu")
    core = name[:-3] if is_nu else name
    for pre in ("kl-", "fila-", "nofila-"):
        if core.startswith(pre):
            core = core[len(pre):]
            break
    return core, is_nu


def _flatten_metrics(results: dict) -> Dict[str, float]:
    """Flatten a results dict to {metric: float}, dropping bookkeeping keys."""
    out = {}
    for k, v in results.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):            # wmdp per-task block
            acc = v.get("acc")
            if acc is not None:
                out[k] = float(acc)
        elif isinstance(v, (int, float)):  # tofu/muse scalar (None -> skip)
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def collect_records(benchmark_filter=None, model_filter=None) -> List[dict]:
    """Walk checkpoints/*/*/*/behavioral_eval_*.json into flat records."""
    records = []
    if not CKPT_DIR.exists():
        return records
    for jf in sorted(CKPT_DIR.rglob("behavioral_eval_*.json")):
        # path: checkpoints/<method>/<bench_label>/<ckpt_name>/behavioral_eval_*.json
        parts = jf.relative_to(CKPT_DIR).parts
        if len(parts) != 4:
            continue
        method, bench_dir, ckpt_name, _ = parts
        try:
            with open(jf) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"[warn] unreadable: {jf}")
            continue

        bench_label = payload.get("benchmark", bench_dir)
        family = _family_of(bench_label)
        if family is None:
            continue
        if benchmark_filter and bench_label != benchmark_filter:
            continue

        model, is_nu = _parse_ckpt_name(ckpt_name)
        if model_filter and model_filter not in model:
            continue

        records.append({
            "bench_label": bench_label,
            "family": family,
            "model": model,
            "method": method,
            "is_nu": is_nu,
            "metrics": _flatten_metrics(payload.get("results", {})),
        })
    return records


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:  # NaN
        return "-"
    # p-values / tiny magnitudes (e.g. Forget Quality) collapse to 0.0000 at
    # 4 decimals — show those in scientific notation to keep resolution.
    if x != 0 and abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def _row_label(method: str, is_nu: bool) -> str:
    disp = _METHOD_DISPLAY.get(method, method)
    return f"{disp} (nu)" if is_nu else disp


def _columns_for(family: str, recs: List[dict]):
    """Ordered (key, display) columns present in the data for this family."""
    spec = BENCH_METRICS.get(family, [])
    present = set()
    for r in recs:
        present.update(r["metrics"].keys())
    cols = [(k, d) for k, d, _ in spec if k in present]
    # Append any extra metrics not in the spec (defensive), sorted.
    extras = sorted(present - {k for k, _ in cols})
    cols += [(k, k) for k in extras]
    return cols


def print_table(bench_label: str, model: str, recs: List[dict]):
    family = recs[0]["family"]
    cols = _columns_for(family, recs)
    if not cols:
        return

    # Sort rows: base first (reference), then by method display, base variant
    # before its nu variant.
    recs = sorted(recs, key=lambda r: (0 if r["method"] == "base" else 1,
                                       _METHOD_DISPLAY.get(r["method"], r["method"]).lower(),
                                       r["is_nu"]))

    row_labels = [_row_label(r["method"], r["is_nu"]) for r in recs]
    method_w = max([len("Method")] + [len(x) for x in row_labels])
    col_w = max(11, max(len(d) for _, d in cols))

    dir_of = {k: d for k, _, d in BENCH_METRICS.get(family, [])}

    title = f"BENCHMARK = {bench_label}    MODEL = {model}"
    width = max(len(title), method_w + (col_w + 3) * len(cols))
    lower = [d for k, d in cols if dir_of.get(k) == "down"]
    higher = [d for k, d in cols if dir_of.get(k) == "up"]
    print("\n" + "=" * width)
    print(title)
    if lower:
        print("  lower is better : " + ", ".join(lower))
    if higher:
        print("  higher is better: " + ", ".join(higher))
    print("=" * width)

    header = "Method".ljust(method_w) + " | " + " | ".join(d.rjust(col_w) for _, d in cols)
    print(header)
    print("-" * len(header))
    for r, lbl in zip(recs, row_labels):
        cells = [_fmt(r["metrics"].get(k)).rjust(col_w) for k, _ in cols]
        print(lbl.ljust(method_w) + " | " + " | ".join(cells))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adjacent per-question probe (Results/behavioral_adjacent/*/*/behavioral_samples_wmdp.json)
# ---------------------------------------------------------------------------

def _sample_correct(rec):
    """0/1 correctness of one lm-eval sample record (acc, falling back to acc_norm)."""
    for k in ("acc", "acc_norm"):
        if k in rec and rec[k] is not None:
            return 1 if float(rec[k]) >= 0.5 else 0
    return None


def collect_adjacent(model_filter=None):
    """Return {model: {method_label: {task: (n_correct, n_total)}}} from the
    per-question sample files written by behavioral_adjacent.slurm."""
    out: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {}
    if not ADJACENT_DIR.exists():
        return out
    for jf in sorted(ADJACENT_DIR.rglob("behavioral_samples_*.json")):
        parts = jf.relative_to(ADJACENT_DIR).parts  # (method, ckpt_name, filename)
        if len(parts) != 3:
            continue
        method, ckpt_name, _ = parts
        try:
            with open(jf) as f:
                samples = json.load(f).get("samples", {})
        except (json.JSONDecodeError, OSError):
            print(f"[warn] unreadable: {jf}")
            continue
        model, is_nu = _parse_ckpt_name(ckpt_name)
        if model_filter and model_filter not in model:
            continue
        label = _row_label_adj(method, is_nu)
        per_task = {}
        for task, recs in samples.items():
            marks = [_sample_correct(r) for r in recs]
            marks = [m for m in marks if m is not None]
            if marks:
                per_task[task] = (sum(marks), len(marks))
        if per_task:
            out.setdefault(model, {})[label] = per_task
    return out


def _row_label_adj(method, is_nu):
    disp = _METHOD_DISPLAY.get(method, method)
    return f"{disp} (nu)" if is_nu else disp


_CYBER_KW = ("computer", "security", "cyber", "network", "hacking",
             "machine_learning", "cryptograph")


def _adjacent_task_order(all_tasks):
    """Group columns by domain: all bio-side subjects (wmdp_bio first), then all
    cyber-side (wmdp_cyber first), so related columns sit next to each other."""
    def is_cyber(t):
        return any(k in t.lower() for k in _CYBER_KW)
    bio = [t for t in sorted(all_tasks) if t != "wmdp_bio" and not is_cyber(t)]
    cyber = [t for t in sorted(all_tasks) if t != "wmdp_cyber" and is_cyber(t)]
    order = []
    if "wmdp_bio" in all_tasks:
        order.append("wmdp_bio")
    order += bio
    if "wmdp_cyber" in all_tasks:
        order.append("wmdp_cyber")
    order += cyber
    return order


def print_adjacent(model, cols):
    methods = sorted(cols, key=lambda m: (m != "base", m))  # base first
    all_tasks = set()
    for c in cols.values():
        all_tasks.update(c)
    tasks = _adjacent_task_order(all_tasks)
    if not tasks:
        return
    # Column header: strip the mmlu_ prefix for compactness.
    disp = {t: (t[len("mmlu_"):] if t.startswith("mmlu_") else t) for t in tasks}
    method_w = max([len("Method")] + [len(m) for m in methods])
    col_w = max(14, max(len(disp[t]) for t in tasks))

    header = "Method".ljust(method_w) + " | " + " | ".join(disp[t].rjust(col_w) for t in tasks)
    print("\n" + "=" * len(header))
    print(f"MODEL = {model}   (adjacent probe: correct / total  (accuracy), full set)")
    print("wmdp_*: lower acc = more forgetting | mmlu subjects: higher = retained")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = []
        for t in tasks:
            if t in cols[m]:
                c, n = cols[m][t]
                cells.append(f"{c}/{n} ({c / n:.2f})".rjust(col_w))
            else:
                cells.append("-".rjust(col_w))
        print(m.ljust(method_w) + " | " + " | ".join(cells))


# ---------------------------------------------------------------------------
# Relearning attack (Results/relearn/<bench>/relearn__*.json): before -> after
# ---------------------------------------------------------------------------

def collect_relearn(model_filter=None):
    """Return {(bench_label, model): {method_label: recovery_dict}} where
    recovery_dict maps metric -> {"before", "after", "delta"}."""
    out: Dict[Tuple[str, str], Dict[str, dict]] = {}
    if not RELEARN_DIR.exists():
        return out
    for jf in sorted(RELEARN_DIR.rglob("relearn__*.json")):
        parts = jf.relative_to(RELEARN_DIR).parts  # (bench_label, filename)
        if len(parts) != 2:
            continue
        bench_label = parts[0]
        try:
            with open(jf) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"[warn] unreadable: {jf}")
            continue
        recovery = payload.get("recovery")
        if not recovery:
            continue
        label = payload.get("label", jf.stem)          # "method/ckpt_name"
        method = label.split("/", 1)[0]
        ckpt_name = label.split("/", 1)[1] if "/" in label else label
        model, is_nu = _parse_ckpt_name(ckpt_name)
        if model_filter and model_filter not in model:
            continue
        row = _row_label_adj(method, is_nu)
        out.setdefault((bench_label, model), {})[row] = recovery
    return out


def _relearn_columns(bench_label, recs):
    """Ordered (metric_key, display) present in the recovery data."""
    family = _family_of(bench_label)
    spec = RELEARN_METRICS.get(family, [])
    present = set()
    for r in recs.values():
        present.update(r.keys())
    cols = [(k, d) for k, d in spec if k in present]
    cols += [(k, k) for k in sorted(present - {k for k, _ in cols})]
    return cols


def print_relearn_after(bench_label, model, recs):
    """Relearned-model scores only (the 'after' of each metric)."""
    cols = _relearn_columns(bench_label, recs)
    if not cols:
        return
    methods = sorted(recs)
    method_w = max([len("Method")] + [len(m) for m in methods])
    col_w = max(11, max(len(d) for _, d in cols))
    header = "Method".ljust(method_w) + " | " + " | ".join(d.rjust(col_w) for _, d in cols)
    print("\n" + "=" * len(header))
    print(f"RELEARNED MODEL   BENCHMARK = {bench_label}   MODEL = {model}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = [_fmt(recs[m].get(k, {}).get("after")).rjust(col_w) for k, _ in cols]
        print(m.ljust(method_w) + " | " + " | ".join(cells))


def print_relearn_compare(bench_label, model, recs):
    """Side-by-side unlearned (un) vs relearned (re) for each metric."""
    cols = _relearn_columns(bench_label, recs)
    if not cols:
        return
    methods = sorted(recs)
    method_w = max([len("Method")] + [len(m) for m in methods])
    sub_w = 9
    # header: two sub-columns per metric
    head_cells = []
    for _, d in cols:
        head_cells.append(f"{d}:un".rjust(sub_w) + " " + "re".rjust(sub_w))
    header = "Method".ljust(method_w) + " | " + " | ".join(head_cells)
    print("\n" + "=" * len(header))
    print(f"UNLEARNED -> RELEARNED (un | re)   BENCHMARK = {bench_label}   MODEL = {model}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = []
        for k, _ in cols:
            rec = recs[m].get(k, {})
            cells.append(_fmt(rec.get("before")).rjust(sub_w) + " "
                         + _fmt(rec.get("after")).rjust(sub_w))
        print(m.ljust(method_w) + " | " + " | ".join(cells))


def collect_relearn_tripartite(model_filter=None):
    """Return {(bench_label, model): {method_label: {"before":results,
    "after":results}}} with the FULL eval blocks (incl _mmlu_subjects), so the
    forget/adjacent/general tripartite can be built for unlearned vs relearned.
    WMDP only (other benchmarks have no per-subject MMLU)."""
    out: Dict[Tuple[str, str], Dict[str, dict]] = {}
    if not RELEARN_DIR.exists():
        return out
    for jf in sorted(RELEARN_DIR.rglob("relearn__*.json")):
        parts = jf.relative_to(RELEARN_DIR).parts
        if len(parts) != 2 or parts[0] != "wmdp":
            continue
        try:
            with open(jf) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        before, after = payload.get("before"), payload.get("after")
        if not (isinstance(before, dict) and isinstance(after, dict)):
            continue
        if "_mmlu_subjects" not in before or "_mmlu_subjects" not in after:
            continue
        label = payload.get("label", jf.stem)
        method = label.split("/", 1)[0]
        ckpt_name = label.split("/", 1)[1] if "/" in label else label
        model, is_nu = _parse_ckpt_name(ckpt_name)
        if model_filter and model_filter not in model:
            continue
        row = _row_label_adj(method, is_nu)
        out.setdefault(("wmdp", model), {})[row] = {"before": before, "after": after}
    return out


def print_tripartite_relearn(model, method_map, adjacent, bs):
    """Unlearned -> relearned tripartite comparison (cells are 'unlearned->relearned').
    Bio and cyber are kept in separate columns (never averaged)."""
    groups = bs.GROUP_COLS
    disp = bs.GROUP_DISP
    methods = sorted(method_map)
    method_w = max([len("Method")] + [len(m) for m in methods])
    col_w = 19

    def _cell(tri_u, tri_r, g):
        u, r = tri_u[g], tri_r[g]
        us = f"{u:.4f}" if u == u else "-"
        rs = f"{r:.4f}" if r == r else "-"
        return f"{us}->{rs}".rjust(col_w)

    header = "Method".ljust(method_w) + " | " + " | ".join(disp[g].rjust(col_w) for g in groups)
    print("\n" + "=" * len(header))
    print(f"MODEL = {model}   (tripartite: unlearned -> relearned, mean accuracy)")
    print(f"adjacent_retain = {adjacent}")
    print("forget: higher after = knowledge recovered by the attack | "
          "retain/general: should stay ~flat")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in methods:
        tri_u = bs.tripartite(bs.rec_from_results(method_map[m]["before"]), adjacent)
        tri_r = bs.tripartite(bs.rec_from_results(method_map[m]["after"]), adjacent)
        print(m.ljust(method_w) + " | " + " | ".join(_cell(tri_u, tri_r, g) for g in groups))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default=None,
                   help="Filter to one bench_label (e.g. wmdp, muse-news, "
                        "tofu-forget10). Default: all found on disk.")
    p.add_argument("--model", default=None,
                   help="Filter to model families containing this string "
                        "(e.g. Llama-3.1-8B). Default: all.")
    p.add_argument("--out-dir", default=str(REPORT_DIR),
                   help=f"Directory for the per-dataset text reports "
                        f"(report__<bench_label>.txt). Default: {REPORT_DIR}.")
    p.add_argument("--no-report-files", action="store_true",
                   help="Do not write per-dataset report files (terminal only).")
    p.add_argument("--no-print", action="store_true",
                   help="Suppress terminal tables (still writes report files).")
    # WMDP behavioural question-group tripartite (delegated to behavioral_subjects).
    p.add_argument("--no-tripartite", action="store_true",
                   help="Skip the WMDP behavioural tripartite section.")
    p.add_argument("--per-subject", action="store_true",
                   help="In the WMDP section, list every MMLU subject ungrouped "
                        "instead of the forget/adjacent/general tripartite.")
    p.add_argument("--adjacent", nargs="+", default=None,
                   help="MMLU subjects counted as adjacent_retain "
                        "(default: behavioral_subjects.DEFAULT_ADJACENT).")
    p.add_argument("--no-adjacent-probe", action="store_true",
                   help="Skip the per-question adjacent-probe section "
                        "(Results/behavioral_adjacent).")
    p.add_argument("--no-relearn", action="store_true",
                   help="Skip the relearning-attack section (Results/relearn).")
    return p.parse_args()


def _emit_dataset(bench_label, groups, tri, adj, rel, rel_tri, args, bs,
                  adjacent_subjects):
    """Print every report section for one dataset to the current stdout.

    Standard per-model tables for this bench_label; the WMDP behavioural
    tripartite + its unlearned->relearned comparison + adjacent per-question
    probe (only under the wmdp dataset); and this dataset's relearning-attack
    comparison.
    """
    print("#" * 72)
    print(f"# DATASET: {bench_label}")
    print("#" * 72)

    # --- standard per-model behavioural tables ---
    any_std = False
    for (bl, model) in sorted(groups):
        if bl != bench_label:
            continue
        print_table(bench_label, model, groups[(bl, model)])
        any_std = True
    if not any_std:
        print(f"\n(no behavioral_eval tables for {bench_label})")

    # --- WMDP-only behavioural sections ---
    if bench_label == "wmdp":
        if tri:
            print("\n" + "#" * 64)
            title = ("per-subject accuracies" if args.per_subject
                     else "question-group tripartite")
            print(f"# WMDP BEHAVIOURAL - {title}")
            print("#" * 64)
            for model in sorted(tri):
                if args.per_subject:
                    bs.print_per_subject(model, tri[model])
                else:
                    bs.print_grouped(model, tri[model], adjacent_subjects)
        if rel_tri:
            print("\n" + "#" * 64)
            print("# WMDP BEHAVIOURAL TRIPARTITE - unlearned -> relearned")
            print("#" * 64)
            for (bl, model) in sorted(rel_tri):
                if bl == bench_label:
                    print_tripartite_relearn(model, rel_tri[(bl, model)],
                                             adjacent_subjects, bs)
        if adj:
            print("\n" + "#" * 64)
            print("# WMDP ADJACENT PROBE - per-question correct/total (full set)")
            print("#" * 64)
            for model in sorted(adj):
                print_adjacent(model, adj[model])

    # --- relearning attack for this dataset ---
    rel_ds = {k: v for k, v in rel.items() if k[0] == bench_label}
    if rel_ds:
        print("\n" + "#" * 64)
        print("# RELEARNING ATTACK - relearned scores + unlearned/relearned comparison")
        print("#" * 64)
        for (bl, model) in sorted(rel_ds):
            print_relearn_after(bl, model, rel_ds[(bl, model)])
            print_relearn_compare(bl, model, rel_ds[(bl, model)])
    print()


def main():
    args = parse_args()
    if not CKPT_DIR.exists():
        sys.exit(f"{CKPT_DIR} does not exist.")

    records = collect_records(args.benchmark, args.model)
    if not records:
        sys.exit("No behavioral_eval_*.json files found "
                 "(run behavioral_eval.py or relearn_attack.py first).")

    real_stdout = sys.stdout
    write_files = not args.no_report_files
    print_term = not args.no_print

    # Group behavioural records by (bench_label, model).
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in records:
        groups.setdefault((r["bench_label"], r["model"]), []).append(r)

    # Precompute the WMDP-only sections and the relearn data once.
    bs = tri = adj = adjacent_subjects = None
    if (not args.no_tripartite) and args.benchmark in (None, "wmdp"):
        import behavioral_subjects as bs   # lazy: avoids an import cycle
        adjacent_subjects = args.adjacent or bs.DEFAULT_ADJACENT
        tri = bs.collect("wmdp", args.model) or None
    if (not args.no_adjacent_probe) and args.benchmark in (None, "wmdp"):
        adj = collect_adjacent(args.model) or None
    rel: Dict[Tuple[str, str], dict] = {}
    rel_tri: Dict[Tuple[str, str], dict] = {}
    if not args.no_relearn:
        rel = collect_relearn(args.model)
        if args.benchmark:
            rel = {k: v for k, v in rel.items() if k[0] == args.benchmark}
        # Unlearned->relearned tripartite (WMDP only), from the relearn before/after.
        if (not args.no_tripartite) and args.benchmark in (None, "wmdp"):
            if bs is None:
                import behavioral_subjects as bs
                adjacent_subjects = adjacent_subjects or (args.adjacent or bs.DEFAULT_ADJACENT)
            rel_tri = collect_relearn_tripartite(args.model)

    # One report per dataset (bench_label): union of what's in behavioural
    # tables and in the relearn outputs.
    datasets = sorted({bl for (bl, _) in groups} | {bl for (bl, _) in rel})

    out_dir = Path(args.out_dir)
    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)

    for bench_label in datasets:
        # Text report (redirected to the per-dataset file and/or the terminal).
        streams, fh = [], None
        report_path = out_dir / f"report__{_safe_name(bench_label)}.txt"
        if write_files:
            fh = open(report_path, "w", encoding="utf-8")
            streams.append(fh)
        if print_term:
            streams.append(real_stdout)
        if streams:
            try:
                with redirect_stdout(_Tee(*streams)):
                    _emit_dataset(bench_label, groups, tri, adj, rel, rel_tri, args,
                                  bs, adjacent_subjects)
            finally:
                if fh:
                    fh.close()

        if write_files:
            print(f"[report] {bench_label} -> {report_path}", file=real_stdout)


if __name__ == "__main__":
    main()
