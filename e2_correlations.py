"""E2 - Correlation analysis (Reviewer uNM1 Q2). No GPU, no model runs.

Joins the already-produced tables on (method, model) and reports four Spearman
correlations with BCa bootstrap 95% CIs (descriptive, not significance claims;
with n~12 the CI is the honest summary, not p).

  1  dCoFi(C_F)     vs delta_F                       (structural forget vs behavioural forget)
  2  adjgap_cofi    vs adjgap_beh                    (structural gap vs behavioural gap)
  3  dCoFi(C_F)     vs recovery                      (structural forget-shift vs relearn recovery)
  4  GLR            vs recovery                      (general-leakage ratio vs relearn recovery)

Derived (true base-model accuracies, NOT ATU as proxy):
  delta_F = (acc_base_F - acc_F)/acc_base_F ; delta_A likewise
  adjgap_beh  = delta_F - delta_A
  adjgap_cofi = dCoFi(C_F) - dCoFi(C_A)
  GLR         = dCoFi(C_G) / dCoFi(C_F)
  recovery    = acc_F_after_relearn - acc_F_before_relearn

Correlation 1 is also run on the subset excluding methods floored at chance
(forget acc <= 0.27): if rho improves, the MCQ floor is what weakens it, which
supports that structural metrics carry information the behavioural probe cannot.

Inputs (must already exist):
  structural  -> analyze_metrics.load_data(benchmark_filter='wmdp')
  behavioural -> checkpoints/{method}/wmdp/{model}/behavioral_eval_wmdp.json
                 + base row checkpoints/base/wmdp/{model}/...
  relearn     -> Results/relearn/wmdp/relearn__*.json

Outputs: Results/rebuttal/correlations.csv , Results/rebuttal/correlation_grid.pdf
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_metrics import load_data
from aggregate_results import _parse_ckpt_name
import behavioral_subjects as bs

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RELEARN_DIR = PROJECT_ROOT / "Results" / "relearn"
OUT_DIR = PROJECT_ROOT / "Results" / "rebuttal"

# Dataset names as load_data emits them (pretty_dataset only maps retain2->wikitext;
# everything else stays the raw lowercase corpus name).
C_F_DS = {"bio-forget", "cyber-forget"}
C_A_DS = {"bio-retain", "cyber-retain"}
C_G_DS = {"wikitext"}
ADJACENT = bs.DEFAULT_ADJACENT


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _canon(core):
    # cofi_cache dirs use '_' (Llama-3_1-8B); checkpoints/relearn use '.'
    # (Llama-3.1-8B). Canonicalise to the dotted form so the tables join.
    return core.replace("_", ".")


# ---------------------------------------------------------------------------
# Structural: dCoFi(C_F/C_A/C_G), PPL_ratio(C_A) per (method, model)
# ---------------------------------------------------------------------------

def structural_table(model_filter):
    df = load_data(benchmark_filter="wmdp")
    if df.empty:
        return {}
    out = {}
    for _, r in df.iterrows():
        label = r["Algorithm_Model"]
        method, model_dir = (label.split("/", 1) + [""])[:2]
        core, is_nu = _parse_ckpt_name(model_dir)
        core = _canon(core)
        if is_nu:
            continue
        if model_filter and model_filter not in core:
            continue
        key = (method, core)
        rec = out.setdefault(key, {"cofi": {}})
        if r["Metric"] == "COFI":
            rec["cofi"][r["Dataset"]] = r["Value"]
    res = {}
    for key, rec in out.items():
        cf = _mean([rec["cofi"].get(d) for d in C_F_DS])
        ca = _mean([rec["cofi"].get(d) for d in C_A_DS])
        cg = _mean([rec["cofi"].get(d) for d in C_G_DS])
        res[key] = {"dCoFi_F": cf, "dCoFi_A": ca, "dCoFi_G": cg}
    return res


# ---------------------------------------------------------------------------
# Behavioural: acc_F/A/G for each checkpoint + the base model
# ---------------------------------------------------------------------------

def _acc_triplet(results):
    tri = bs.tripartite(bs.rec_from_results(results), ADJACENT)
    return (_mean([tri["forget_bio"], tri["forget_cyber"]]),
            _mean([tri["adjacent_bio"], tri["adjacent_cyber"]]),
            tri["general_info"])


def behavioural_table(model_filter):
    unl, base = {}, {}
    for jf in CKPT_DIR.rglob("behavioral_eval_wmdp.json"):
        parts = jf.relative_to(CKPT_DIR).parts
        if len(parts) != 4:
            continue
        method, _bench, model_dir, _ = parts
        core, is_nu = _parse_ckpt_name(model_dir)
        core = _canon(core)
        if is_nu or (model_filter and model_filter not in core):
            continue
        try:
            results = json.load(open(jf)).get("results", {})
        except Exception:
            continue
        aF, aA, aG = _acc_triplet(results)
        if method == "base":
            base[core] = (aF, aA, aG)
        else:
            unl[(method, core)] = (aF, aA, aG)
    return unl, base


# ---------------------------------------------------------------------------
# Relearn: forget acc before/after per (method, model)
# ---------------------------------------------------------------------------

def relearn_table(model_filter):
    out = {}
    if not RELEARN_DIR.exists():
        return out
    for jf in (RELEARN_DIR / "wmdp").rglob("relearn__*.json") if (RELEARN_DIR / "wmdp").exists() else []:
        try:
            p = json.load(open(jf))
        except Exception:
            continue
        label = p.get("label", jf.stem)
        method = label.split("/", 1)[0]
        ckpt = label.split("/", 1)[1] if "/" in label else label
        core, is_nu = _parse_ckpt_name(ckpt)
        core = _canon(core)
        if is_nu or (model_filter and model_filter not in core):
            continue
        before, after = p.get("before", {}), p.get("after", {})

        def acc_f(block):
            vals = []
            for t in ("wmdp_bio", "wmdp_cyber"):
                v = block.get(t)
                if isinstance(v, dict) and v.get("acc") is not None:
                    vals.append(float(v["acc"]))
            return _mean(vals)
        out[(method, core)] = (acc_f(before), acc_f(after))
    return out


# ---------------------------------------------------------------------------
# Assemble per-(method, model) rows and the derived quantities
# ---------------------------------------------------------------------------

def build_rows(model_filter):
    struct = structural_table(model_filter)
    unl, base = behavioural_table(model_filter)
    rel = relearn_table(model_filter)
    rows = []
    for key in sorted(set(struct) | set(unl)):
        method, core = key
        s = struct.get(key, {})
        b = base.get(core)
        u = unl.get(key)
        row = {"method": method, "model": core}
        row.update({k: s.get(k, float("nan")) for k in ("dCoFi_F", "dCoFi_A", "dCoFi_G")})
        if u and b and b[0] and b[1]:
            aF, aA, aG = u
            bF, bA, bG = b
            row["acc_F"] = aF
            row["delta_F"] = (bF - aF) / bF if bF else float("nan")
            row["delta_A"] = (bA - aA) / bA if bA else float("nan")
        else:
            row["acc_F"] = row["delta_F"] = row["delta_A"] = float("nan")
        row["adjgap_beh"] = row["delta_F"] - row["delta_A"]
        row["adjgap_cofi"] = row["dCoFi_F"] - row["dCoFi_A"]
        row["GLR"] = (row["dCoFi_G"] / row["dCoFi_F"]
                      if row["dCoFi_F"] not in (0, float("nan")) and np.isfinite(row["dCoFi_F"]) and row["dCoFi_F"] != 0
                      else float("nan"))
        if key in rel:
            bef, aft = rel[key]
            row["recovery"] = aft - bef
        else:
            row["recovery"] = float("nan")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Spearman + BCa bootstrap
# ---------------------------------------------------------------------------

def spearman_ci(x, y, n_boot=10000, seed=0):
    from scipy.stats import spearmanr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n, float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    lo, hi = float("nan"), float("nan")
    try:
        from scipy.stats import bootstrap
        res = bootstrap((x, y), lambda a, b: spearmanr(a, b).statistic,
                        method="BCa", n_resamples=n_boot, paired=True,
                        random_state=seed)
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
                continue
            boots.append(spearmanr(x[idx], y[idx]).correlation)
        if boots:
            lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rho), float(p), n, lo, hi


PAIRS = [
    (1, "dCoFi_F", "delta_F", "all"),
    (2, "adjgap_cofi", "adjgap_beh", "all"),
    (3, "dCoFi_F", "recovery", "all"),
    (4, "GLR", "recovery", "all"),
]


def _xy(rows, xvar, yvar, subset):
    def val(r, v):
        return r.get(v, float("nan"))
    rr = rows
    if subset == "restricted":
        rr = [r for r in rows if np.isfinite(r.get("acc_F", float("nan"))) and r["acc_F"] > 0.27]
    xs = [val(r, xvar) for r in rr]
    ys = [val(r, yvar) for r in rr]
    labels = [r["method"] for r in rr]
    return xs, ys, labels


def _safe(name):
    return name.replace(".", "_").replace("/", "_")


def _run_one(model, no_plot):
    rows = build_rows(model)
    if not rows:
        print(f"[{model}] no joined rows — skipping (structural/behavioural tables present?)")
        return
    tag = _safe(model)
    import csv
    out_csv = OUT_DIR / f"correlations__{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pair_id", "x_var", "y_var", "subset", "n",
                    "rho", "p", "ci_low", "ci_high"])
        for pid, xv, yv, subset in PAIRS:
            xs, ys, _ = _xy(rows, xv, yv, subset)
            rho, p, n, lo, hi = spearman_ci(xs, ys)
            w.writerow([model, pid, xv, yv, subset, n, f"{rho:.4f}", f"{p:.4g}",
                        f"{lo:.4f}", f"{hi:.4f}"])
        # correlation 1 restricted (drop chance-floored methods)
        xs, ys, _ = _xy(rows, "dCoFi_F", "delta_F", "restricted")
        rho, p, n, lo, hi = spearman_ci(xs, ys)
        w.writerow([model, 1, "dCoFi_F", "delta_F", "restricted", n, f"{rho:.4f}",
                    f"{p:.4g}", f"{lo:.4f}", f"{hi:.4f}"])
    print(f"wrote {out_csv}  ({len(rows)} methods)")
    if not no_plot:
        _grid(rows, model, tag)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        print(f"\n===== {model} =====")
        _run_one(model, args.no_plot)


def _grid(rows, model, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (pid, xv, yv, subset) in zip(axes.flat, PAIRS):
        xs, ys, labels = _xy(rows, xv, yv, subset)
        rho, p, n, lo, hi = spearman_ci(xs, ys)
        ax.scatter(xs, ys, s=40)
        for x, y, lb in zip(xs, ys, labels):
            if np.isfinite(x) and np.isfinite(y):
                ax.annotate(lb, (x, y), fontsize=6, alpha=0.8)
        ax.set_xlabel(xv)
        ax.set_ylabel(yv)
        ax.set_title(f"({pid}) {yv} vs {xv}   rho={rho:.2f}, n={n}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"WMDP correlations - {model}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"correlation_grid__{tag}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT_DIR / f'correlation_grid__{tag}.pdf'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+",
                   default=["Llama-3.1-8B", "zephyr-7b-beta", "Llama-3.2-3B"],
                   help="One correlations file per model (WMDP). Default: 8B, 7B, "
                        "3B (Qwen3-32B intentionally excluded).")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
