"""Aggregate falsification + behavioral-eval results into per (model, benchmark) tables.

Two sources of truth:

1. Falsification metrics — JSON files written by test.py under cofi_cache/.
   For null/destructive/random_* checkpoints we report the *relative* drops
   (matching analyze_metrics.py):
       CoFi%   = (norm_drop / base_frob_norm) * 100
       CHess%  = (norm_drop / base_frob_norm) * 100
       PPLr    = unlearned_ppl / base_ppl

2. Behavioral eval — JSON files written by behavioral_eval.py, saved alongside
   each checkpoint at  <ckpt>/behavioral_eval_<bench>.json.
       wmdp -> wmdp_bio acc, wmdp_cyber acc, mmlu acc
       muse -> mem_acc & 10-gram extraction likelihood for forget / retain1

Output: one table per (model, benchmark) for each section. Either pretty-printed
to the terminal or written as CSVs (one file per table) into --csv-dir.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cofi_cache"
CKPT_DIR = PROJECT_ROOT / "checkpoints"

FALSIFY_METHODS_PREFIXES = ("null", "destructive", "random_")

# Benchmark label (directory name on disk) -> behavioral_eval_<tag>.json suffix
BENCH_TO_BEHAV_TAG = {
    "wmdp": "wmdp",
    "muse-books": "muse",
    "muse-news": "muse",
    "tofu-forget01": "tofu",
    "tofu-forget05": "tofu",
    "tofu-forget10": "tofu",
}


# ---------------------------------------------------------------------------
# Model-name matching helpers (mirrors analyze_metrics.py)
# ---------------------------------------------------------------------------

def _core_model_name(base_label: str) -> str:
    """'base__Meta-Llama-3_1-8B' -> 'Llama-3_1-8B'"""
    raw = base_label.replace("base__", "")
    return re.sub(r"^[A-Z][a-z]*-", "", raw)


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def _model_family(model_dir: str) -> str:
    """Map any unlearned-checkpoint directory name to a canonical model family.

    'kl-Llama-3.1-8B'    -> 'Llama-3.1-8B'
    'Llama-3.1-8B_nu'    -> 'Llama-3.1-8B'
    'zephyr-7b-beta'     -> 'zephyr-7b-beta'
    """
    s = model_dir
    for prefix in ("kl-", "fila-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("_nu"):
        s = s[:-3]
    return s


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_benchmarks() -> List[str]:
    if not CACHE_DIR.exists():
        return []
    out = set()
    for method_dir in CACHE_DIR.iterdir():
        if not method_dir.is_dir():
            continue
        for bench_dir in method_dir.iterdir():
            if bench_dir.is_dir():
                out.add(bench_dir.name)
    return sorted(out)


def discover_model_families(bench: str) -> List[str]:
    """All model families (canonical) that appear under any method for `bench`."""
    fams = set()
    if CACHE_DIR.exists():
        for method_dir in CACHE_DIR.iterdir():
            if method_dir.name == "base":
                continue
            bd = method_dir / bench
            if not bd.is_dir():
                continue
            for model_dir in bd.iterdir():
                if not model_dir.is_dir():
                    continue
                fams.add(_model_family(model_dir.name))
    if CKPT_DIR.exists():
        for method_dir in CKPT_DIR.iterdir():
            if method_dir.name == "base":
                continue
            bd = method_dir / bench
            if not bd.is_dir():
                continue
            for model_dir in bd.iterdir():
                if not model_dir.is_dir():
                    continue
                fams.add(_model_family(model_dir.name))
    # Dedupe variants that differ only in punctuation (e.g. 'Llama-3.1-8B'
    # from checkpoints/ vs 'Llama-3_1-8B' from the sanitized cofi_cache form).
    # Prefer the form containing '.' since that matches the on-disk checkpoint
    # directory names that the behavioral lookup compares against literally.
    by_key: Dict[str, str] = {}
    for f in fams:
        key = _safe_name(f)
        if key not in by_key or ("." in f and "." not in by_key[key]):
            by_key[key] = f
    return sorted(by_key.values())


# ---------------------------------------------------------------------------
# Base-model loading (frob_norms + ppl per corpus)
# ---------------------------------------------------------------------------

def load_base_for_family(bench: str, family: str) -> Tuple[Optional[str], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return (base_label, cofi_norms, chess_norms, ppl) for the base model
    matching `family` under benchmark `bench`. Empty dicts if not found.
    """
    base_dir = CACHE_DIR / "base" / bench
    if not base_dir.is_dir():
        return None, {}, {}, {}

    # Find a base model whose core name matches the family (case-insensitive).
    fam_safe = _safe_name(family).lower()
    chosen = None
    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue
        core = _core_model_name(f"base__{sub.name}").lower()
        if fam_safe in core or core in fam_safe:
            chosen = sub
            break
    if chosen is None:
        return None, {}, {}, {}

    cofi, chess, ppl = {}, {}, {}
    for jf in chosen.glob("*.json"):
        stem = jf.stem
        if "__" not in stem:
            continue
        corpus, metric = stem.split("__", 1)
        with open(jf) as f:
            data = json.load(f)
        if metric == "ppl":
            ppl[corpus] = data.get("ppl")
        elif metric == "cofi":
            cofi[corpus] = data.get("frob_norm")
        elif metric == "chess":
            chess[corpus] = data.get("frob_norm")
    return f"base__{chosen.name}", cofi, chess, ppl


# ---------------------------------------------------------------------------
# Falsification table
# ---------------------------------------------------------------------------

def collect_falsify_rows(bench: str, family: str) -> Tuple[List[str], List[Dict]]:
    """Return (corpora_in_order, rows) for falsification methods.

    Each row is a dict with key 'method' plus '<corpus>__cofi%', '<corpus>__chess%',
    '<corpus>__pplr' columns.
    """
    _, base_cofi, base_chess, base_ppl = load_base_for_family(bench, family)

    fam_safe = _safe_name(family)
    rows = []
    corpora_seen: List[str] = []
    seen_set = set()

    for method_dir in sorted(CACHE_DIR.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        if not any(method.startswith(p) for p in FALSIFY_METHODS_PREFIXES):
            continue
        bd = method_dir / bench
        if not bd.is_dir():
            continue

        # Match the model variant for this family
        match_dir = None
        for sub in bd.iterdir():
            if not sub.is_dir():
                continue
            if _model_family(sub.name) == family or _safe_name(_model_family(sub.name)) == fam_safe:
                match_dir = sub
                break
        if match_dir is None:
            continue

        per_corpus: Dict[str, Dict[str, float]] = defaultdict(dict)
        for jf in match_dir.glob("*.json"):
            stem = jf.stem
            if "__" not in stem:
                continue
            corpus, metric = stem.split("__", 1)
            if corpus not in seen_set:
                seen_set.add(corpus)
                corpora_seen.append(corpus)
            with open(jf) as f:
                data = json.load(f)
            if metric == "cofi":
                drop = data.get("norm_drop")
                base = base_cofi.get(corpus)
                if drop is not None and base and base > 0:
                    per_corpus[corpus]["cofi%"] = (drop / base) * 100
            elif metric == "chess":
                drop = data.get("norm_drop")
                base = base_chess.get(corpus)
                if drop is not None and base and base > 0:
                    per_corpus[corpus]["chess%"] = (drop / base) * 100
            elif metric == "ppl":
                ppl = data.get("ppl")
                base = base_ppl.get(corpus)
                if ppl is not None and base and base > 0:
                    per_corpus[corpus]["pplr"] = ppl / base

        row = {"method": method}
        for corpus in corpora_seen:
            for m in ("cofi%", "chess%", "pplr"):
                row[f"{corpus}__{m}"] = per_corpus.get(corpus, {}).get(m)
        rows.append(row)

    # Order corpora: forget first, then retain, wikitext last
    def _corpus_sort(c: str) -> Tuple[int, str]:
        cl = c.lower()
        if cl in ("wikitext", "retain2"):
            return (2, c)
        if "forget" in cl:
            return (0, c)
        return (1, c)

    corpora_seen = sorted(corpora_seen, key=_corpus_sort)
    # Re-order row keys to follow corpora_seen order
    ordered_rows = []
    for r in rows:
        new = {"method": r["method"]}
        for c in corpora_seen:
            for m in ("cofi%", "chess%", "pplr"):
                new[f"{c}__{m}"] = r.get(f"{c}__{m}")
        ordered_rows.append(new)
    return corpora_seen, ordered_rows


# ---------------------------------------------------------------------------
# Behavioral eval table
# ---------------------------------------------------------------------------

def collect_behavioral_rows(bench: str, family: str) -> Tuple[List[str], List[Dict]]:
    """Walk checkpoints/*/{bench}/*/behavioral_eval_<tag>.json for the family.

    Returns (column_names, rows). Columns depend on the benchmark kind.
    """
    behav_tag = BENCH_TO_BEHAV_TAG.get(bench)
    if behav_tag is None:
        return [], []

    fname = f"behavioral_eval_{behav_tag}.json"
    rows = []
    for method_dir in sorted(CKPT_DIR.iterdir()):
        if not method_dir.is_dir():
            continue
        bd = method_dir / bench
        if not bd.is_dir():
            continue
        for sub in sorted(bd.iterdir()):
            if not sub.is_dir():
                continue
            if _model_family(sub.name) != family:
                continue
            jf = sub / fname
            if not jf.exists():
                continue
            with open(jf) as f:
                payload = json.load(f)
            results = payload.get("results", {})
            row = {"method": f"{method_dir.name}/{sub.name}"}
            if behav_tag == "wmdp":
                for task in ("wmdp_bio", "wmdp_cyber", "mmlu"):
                    r = results.get(task, {}) or {}
                    row[f"{task}__acc"] = r.get("acc")
                    row[f"{task}__stderr"] = r.get("acc_stderr")
            elif behav_tag == "muse":
                for split in ("forget", "retain1"):
                    r = results.get(split, {}) or {}
                    row[f"{split}__mem_acc"] = r.get("mem_acc")
                    row[f"{split}__ngram10_logp"] = r.get("ngram10_logp_per_tok")
                    row[f"{split}__ngram10_lik"] = r.get("ngram10_extraction_likelihood")
            rows.append(row)

    if not rows:
        return [], []

    cols = [k for k in rows[0].keys() if k != "method"]
    return cols, rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:  # NaN
        return "—"
    if kind == "pct":
        return f"{x:7.2f}"
    if kind == "ratio":
        return f"{x:9.3e}"
    if kind == "acc":
        return f"{x:6.4f}"
    if kind == "logp":
        return f"{x:8.3f}"
    if kind == "lik":
        return f"{x:9.3e}"
    return f"{x:.4f}"


def _kind_for(col: str) -> str:
    if col.endswith("__cofi%") or col.endswith("__chess%"):
        return "pct"
    if col.endswith("__pplr"):
        return "ratio"
    if col.endswith("__acc"):
        return "acc"
    if col.endswith("__stderr"):
        return "acc"
    if col.endswith("__mem_acc"):
        return "acc"
    if col.endswith("__ngram10_logp"):
        return "logp"
    if col.endswith("__ngram10_lik"):
        return "lik"
    return "raw"


def print_table(title: str, cols: List[str], rows: List[Dict]):
    if not rows:
        print(f"\n[{title}] no rows.")
        return
    method_w = max(len(r["method"]) for r in rows + [{"method": "method"}])
    headers = ["method"] + cols
    widths = [method_w] + [max(len(c), 9) for c in cols]

    def line(parts):
        return " | ".join(p.rjust(w) for p, w in zip(parts, widths))

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(line(headers))
    print("-" * (sum(widths) + 3 * len(widths)))
    for r in rows:
        cells = [r["method"]] + [_fmt(r.get(c), _kind_for(c)) for c in cols]
        print(line(cells))


def write_csv(path: Path, cols: List[str], rows: List[Dict]):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols)
        for r in rows:
            w.writerow([r["method"]] + [r.get(c) for c in cols])
    try:
        shown = path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = path
    print(f"  wrote {shown}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", action="append", default=None,
                   help="Benchmark dir name (e.g. wmdp, muse-books). Repeat to select several. "
                        "Default: all benchmarks discovered on disk.")
    p.add_argument("--model", action="append", default=None,
                   help="Model family (e.g. Llama-3.1-8B, zephyr-7b-beta). Repeat to select several. "
                        "Default: all model families found per benchmark.")
    p.add_argument("--csv-dir", default=None,
                   help="Write CSVs into this directory instead of (or in addition to) printing.")
    p.add_argument("--no-print", action="store_true",
                   help="Suppress terminal output (use with --csv-dir).")
    p.add_argument("--skip-falsify", action="store_true")
    p.add_argument("--skip-behavioral", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not CACHE_DIR.exists() and not CKPT_DIR.exists():
        sys.exit(f"Neither {CACHE_DIR} nor {CKPT_DIR} exists.")

    benches = args.benchmark or discover_benchmarks()
    if not benches:
        sys.exit("No benchmarks found under cofi_cache/.")

    csv_dir = Path(args.csv_dir) if args.csv_dir else None

    for bench in benches:
        families = args.model or discover_model_families(bench)
        if not families:
            print(f"[{bench}] no models found.")
            continue

        for family in families:
            header = f"BENCHMARK = {bench}    MODEL = {family}"
            if not args.no_print:
                print("\n" + "#" * (len(header) + 4))
                print(f"# {header} #")
                print("#" * (len(header) + 4))

            if not args.skip_falsify:
                cols, rows = collect_falsify_rows(bench, family)
                if rows:
                    fcols = []
                    for c in cols:
                        for m in ("cofi%", "chess%", "pplr"):
                            fcols.append(f"{c}__{m}")
                    if not args.no_print:
                        print_table(
                            f"Falsification (relative drops)  —  {family} / {bench}",
                            fcols, rows,
                        )
                    if csv_dir:
                        write_csv(
                            csv_dir / f"falsify__{bench}__{_safe_name(family)}.csv",
                            fcols, rows,
                        )
                elif not args.no_print:
                    print(f"\n[falsification] no rows for {family} / {bench}.")

            if not args.skip_behavioral:
                cols, rows = collect_behavioral_rows(bench, family)
                if rows:
                    if not args.no_print:
                        print_table(
                            f"Behavioral eval  —  {family} / {bench}",
                            cols, rows,
                        )
                    if csv_dir:
                        write_csv(
                            csv_dir / f"behavioral__{bench}__{_safe_name(family)}.csv",
                            cols, rows,
                        )
                elif not args.no_print:
                    print(f"\n[behavioral] no rows for {family} / {bench}.")


if __name__ == "__main__":
    main()
