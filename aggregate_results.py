"""Aggregate behavioral-eval results into per (model, benchmark) tables.

Reads JSON files written by behavioral_eval.py, saved alongside each
checkpoint at  <ckpt>/behavioral_eval_<bench>.json.
    wmdp -> wmdp_bio acc, wmdp_cyber acc, mmlu acc
    tofu -> whatever behavioral_eval writes for tofu

Output: one table per (model, benchmark). Either pretty-printed to the terminal
or written as CSVs (one file per table) into --csv-dir.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"

# Benchmark label (directory name on disk) -> behavioral_eval_<tag>.json suffix
BENCH_TO_BEHAV_TAG = {
    "wmdp": "wmdp",
    "tofu-forget01": "tofu",
    "tofu-forget05": "tofu",
    "tofu-forget10": "tofu",
}


# ---------------------------------------------------------------------------
# Model-name matching helpers
# ---------------------------------------------------------------------------

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
    if not CKPT_DIR.exists():
        return []
    out = set()
    for method_dir in CKPT_DIR.iterdir():
        if not method_dir.is_dir():
            continue
        for bench_dir in method_dir.iterdir():
            if bench_dir.is_dir() and bench_dir.name in BENCH_TO_BEHAV_TAG:
                out.add(bench_dir.name)
    return sorted(out)


def discover_model_families(bench: str) -> List[str]:
    """All model families (canonical) that appear under any method for `bench`."""
    fams = set()
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
    # Dedupe variants that differ only in punctuation. Prefer the form with '.'
    # since that matches the on-disk checkpoint directory names.
    by_key: Dict[str, str] = {}
    for f in fams:
        key = _safe_name(f)
        if key not in by_key or ("." in f and "." not in by_key[key]):
            by_key[key] = f
    return sorted(by_key.values())


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
    if kind == "acc":
        return f"{x:6.4f}"
    return f"{x:.4f}"


def _kind_for(col: str) -> str:
    if col.endswith("__acc") or col.endswith("__stderr"):
        return "acc"
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
                   help="Benchmark dir name (e.g. wmdp). Repeat to select several. "
                        "Default: all benchmarks discovered on disk.")
    p.add_argument("--model", action="append", default=None,
                   help="Model family (e.g. Llama-3.1-8B, zephyr-7b-beta). Repeat to select several. "
                        "Default: all model families found per benchmark.")
    p.add_argument("--csv-dir", default=None,
                   help="Write CSVs into this directory instead of (or in addition to) printing.")
    p.add_argument("--no-print", action="store_true",
                   help="Suppress terminal output (use with --csv-dir).")
    return p.parse_args()


def main():
    args = parse_args()

    if not CKPT_DIR.exists():
        sys.exit(f"{CKPT_DIR} does not exist.")

    benches = args.benchmark or discover_benchmarks()
    if not benches:
        sys.exit(f"No supported benchmarks found under {CKPT_DIR}.")

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
