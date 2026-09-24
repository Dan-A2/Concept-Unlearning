"""Behavioural question-group table for the WMDP behavioural eval.

Groups the behavioural-eval questions (NOT the algorithms) into three sets and
reports mean accuracy per method:

  * forget          — WMDP-Bio + WMDP-Cyber MCQs (should drop after unlearning)
  * adjacent_retain — MMLU subjects topically adjacent to bio/cyber
                      (default: virology, college_biology, computer_security)
  * general_info    — all remaining MMLU subjects (broad-utility probe)

This is the "behavioural tripartite" baseline. It does NOT touch the
algorithm-impact grouping (no-op / localized / destructive) — those labels are
yours to assign and may change.

Reads <ckpt>/behavioral_eval_wmdp.json (behavioral_eval.py now persists every
per-subject MMLU accuracy under results["_mmlu_subjects"]). Re-run the WMDP
behavioural eval once — ideally at full MMLU — to populate them.

Usage:
    python behavioral_subjects.py                       # grouped table per model
    python behavioral_subjects.py --per-subject         # every subject, ungrouped
    python behavioral_subjects.py --adjacent virology college_biology computer_security
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from aggregate_results import _parse_ckpt_name, _METHOD_DISPLAY

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"

FORGET_METRICS = ["wmdp_bio", "wmdp_cyber"]
DEFAULT_ADJACENT = ["virology", "college_biology", "computer_security"]
# MMLU category rollup keys occasionally present in results; never a real subject.
_MMLU_GROUP_KEYS = {"stem", "humanities", "social_sciences", "other"}

# bio vs cyber: bio and cyber accuracies are very different and must never be
# averaged together, so the tripartite reports them in separate columns. An
# adjacent MMLU subject is cyber-side if its name carries a cyber keyword,
# otherwise bio-side.
_CYBER_KW = ("computer", "security", "cyber", "network", "hacking",
             "machine_learning", "cryptograph")


def _domain(subject: str) -> str:
    return "cyber" if any(k in subject.lower() for k in _CYBER_KW) else "bio"


def _row_label(method: str, is_nu: bool) -> str:
    disp = _METHOD_DISPLAY.get(method, method)
    return f"{disp} (nu)" if is_nu else disp


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def rec_from_results(results: dict) -> dict:
    """Build a tripartite `rec` ({"forget":{metric:acc}, "subjects":{...}}) from a
    raw behavioural-eval results dict. Works for behavioral_eval JSONs and for the
    relearn attack's before/after blocks (same shape)."""
    forget = {}
    for m in FORGET_METRICS:
        v = results.get(m)
        if isinstance(v, dict) and v.get("acc") is not None:
            forget[m] = float(v["acc"])
    subjects = {k: float(v) for k, v in results.get("_mmlu_subjects", {}).items()}
    return {"forget": forget, "subjects": subjects}


def tripartite(rec, adjacent) -> Dict[str, float]:
    """Public wrapper around the forget/adjacent_retain/general_info rollup."""
    return _tripartite(rec, adjacent)


def collect(benchmark="wmdp", model_filter=None):
    """Return {model: {method_label: {"forget":.., subjects:{...}}}}."""
    out: Dict[str, Dict[str, dict]] = {}
    for jf in sorted(CKPT_DIR.rglob(f"behavioral_eval_{benchmark}.json")):
        parts = jf.relative_to(CKPT_DIR).parts
        if len(parts) != 4:
            continue
        method, _bench, ckpt_name, _ = parts
        try:
            with open(jf) as f:
                results = json.load(f).get("results", {})
        except (json.JSONDecodeError, OSError):
            print(f"[warn] unreadable: {jf}")
            continue

        model, is_nu = _parse_ckpt_name(ckpt_name)
        if model_filter and model_filter not in model:
            continue

        out.setdefault(model, {})[_row_label(method, is_nu)] = rec_from_results(results)
    return out


# ---------------------------------------------------------------------------
# Grouped (tripartite) view
# ---------------------------------------------------------------------------

# Tripartite columns: bio and cyber kept separate for forget and adjacent-retain
# (never averaged together); ordered so all bio columns sit together, then all
# cyber, then the domain-agnostic general_info (MMLU).
GROUP_COLS = ["forget_bio", "adjacent_bio",
              "forget_cyber", "adjacent_cyber", "general_info"]
GROUP_DISP = {
    "forget_bio":     "Forget-Bio",
    "adjacent_bio":   "Retain-Bio",
    "forget_cyber":   "Forget-Cyber",
    "adjacent_cyber": "Retain-Cyber",
    "general_info":   "General-Info",
}


def _tripartite(rec, adjacent) -> Dict[str, float]:
    subj = rec["subjects"]
    fg = rec["forget"]
    generic = [subj[s] for s in subj
               if s not in adjacent and s not in _MMLU_GROUP_KEYS]
    return {
        "forget_bio":     fg.get("wmdp_bio", float("nan")),
        "forget_cyber":   fg.get("wmdp_cyber", float("nan")),
        "adjacent_bio":   _mean([subj[s] for s in adjacent
                                 if s in subj and _domain(s) == "bio"]),
        "adjacent_cyber": _mean([subj[s] for s in adjacent
                                 if s in subj and _domain(s) == "cyber"]),
        "general_info":   _mean(generic),
    }


def print_grouped(model, cols, adjacent):
    methods = sorted(cols, key=lambda m: (m != "base", m))  # base first
    method_w = max([len("Method")] + [len(m) for m in methods])
    col_w = 15

    header = "Method".ljust(method_w) + " | " + " | ".join(
        GROUP_DISP[g].rjust(col_w) for g in GROUP_COLS)
    print("\n" + "=" * len(header))
    print(f"MODEL = {model}   (behavioural tripartite, mean accuracy; bio/cyber split)")
    print(f"adjacent_retain = {adjacent}")
    print("forget: lower = more forgetting | retain/general: higher = less damage")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in methods:
        tri = _tripartite(cols[m], adjacent)
        cells = [(f"{tri[g]:.4f}" if tri[g] == tri[g] else "-").rjust(col_w)
                 for g in GROUP_COLS]
        print(m.ljust(method_w) + " | " + " | ".join(cells))


# ---------------------------------------------------------------------------
# Ungrouped (per-subject) view
# ---------------------------------------------------------------------------

def _all_metric_rows(cols):
    subjects = set()
    for rec in cols.values():
        subjects.update(rec["subjects"])
    subjects -= _MMLU_GROUP_KEYS
    return FORGET_METRICS + sorted(subjects)


def _cell(rec, metric):
    if metric in FORGET_METRICS:
        return rec["forget"].get(metric)
    return rec["subjects"].get(metric)


def print_per_subject(model, cols):
    methods = sorted(cols, key=lambda m: (m != "base", m))  # base first
    rows = _all_metric_rows(cols)
    row_w = max([len("subject")] + [len(r) for r in rows])
    col_w = max(8, max(len(m) for m in methods))
    header = "subject".ljust(row_w) + " | " + " | ".join(m.rjust(col_w) for m in methods)
    print("\n" + "=" * len(header))
    print(f"MODEL = {model}   (per-subject accuracy, ungrouped)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for m in methods:
            v = _cell(cols[m], r)
            cells.append((f"{v:.4f}" if v is not None else "-").rjust(col_w))
        marker = "  <-- forget" if r in FORGET_METRICS else ""
        print(r.ljust(row_w) + " | " + " | ".join(cells) + marker)




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="wmdp",
                   help="Benchmark label (default: wmdp; only WMDP has MMLU subjects).")
    p.add_argument("--adjacent", nargs="+", default=DEFAULT_ADJACENT,
                   help="MMLU subjects to count as adjacent_retain "
                        f"(default: {' '.join(DEFAULT_ADJACENT)}).")
    p.add_argument("--per-subject", action="store_true",
                   help="Print every MMLU subject ungrouped instead of the tripartite.")
    p.add_argument("--model", default=None, help="Filter to models containing this string.")
    return p.parse_args()


def main():
    args = parse_args()
    if not CKPT_DIR.exists():
        sys.exit(f"{CKPT_DIR} does not exist.")

    data = collect(args.benchmark, args.model)
    if not data:
        sys.exit(f"No behavioral_eval_{args.benchmark}.json found.")

    has_subjects = any(rec["subjects"] for cols in data.values() for rec in cols.values())
    if not has_subjects:
        print("[note] No _mmlu_subjects in the JSONs — only WMDP forget accuracies "
              "are available. Re-run behavioral_eval.py (updated) to capture "
              "per-subject MMLU.\n")

    for model in sorted(data):
        cols = data[model]
        if args.per_subject:
            print_per_subject(model, cols)
        else:
            print_grouped(model, cols, args.adjacent)
    print()


if __name__ == "__main__":
    main()
