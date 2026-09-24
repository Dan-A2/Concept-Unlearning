import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cofi_cache"
RESULTS_DIR = PROJECT_ROOT / "Results"

# Display name overrides for datasets
DATASET_DISPLAY_NAMES = {
    "retain2": "wikitext",
}


def pretty_dataset(name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(name, name)


def _core_model_name(base_label: str) -> str:
    """Extract the core model family identifier from a base label.

    e.g. 'base__Meta-Llama-3_1-8B' -> 'Llama-3_1-8B'
         'base__Llama-3_2-3B'      -> 'Llama-3_2-3B'
         'base__Qwen3-32B'         -> 'Qwen3-32B'
         'base__zephyr-7b-beta'    -> 'zephyr-7b-beta'

    Only strips a known organisation prefix (e.g. 'Meta-'); we must NOT use a
    generic '^[A-Z][a-z]*-' rule because it would mangle 'Llama-3_2-3B' into
    '3_2-3B'.
    """
    raw = base_label.replace("base__", "")
    for org in ("Meta-", "Google-", "Microsoft-", "Mistral-", "EleutherAI-"):
        if raw.startswith(org):
            return raw[len(org):]
    return raw


def _t_crit(n: int) -> float:
    """t-distribution critical value for a 95% CI with df=n-1 (n=2..5),
    falling back to the normal 1.96 for larger n. Mirrors test.py's _agg."""
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 1.96)


def _aggregate(values):
    """(mean, std, ci95_halfwidth, n) over the finite values."""
    arr = [v for v in values if v is not None and np.isfinite(v)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, np.nan, n
    std = float(np.std(arr, ddof=1))
    ci = _t_crit(n) * std / math.sqrt(n)
    return mean, std, ci, n


def _parse_metric_file(file: Path):
    """Parse a metric-cache filename stem into (corpus, metric, subset_id).

    New layout : '{corpus}__{metric}__sub{N}'  (chess may carry a trailing
                 '_seed{S}', e.g. '{corpus}__chess__sub0_seed42').
    Legacy     : '{corpus}__{metric}'           (no subset)  -> subset_id None.

    Returns None if the stem does not match either form.
    """
    parts = file.stem.split("__")
    if len(parts) == 2:
        corpus, metric = parts
        return corpus, metric, None
    if len(parts) == 3:
        corpus, metric, sub_tok = parts
        m = re.match(r"sub(\d+)", sub_tok)
        subset_id = int(m.group(1)) if m else None
        return corpus, metric, subset_id
    return None


def load_data(benchmark_filter=None):
    """Load cached metrics from the nested cofi_cache structure and aggregate
    across subsets into mean ± 95% CI.

    Directory layout:
        cofi_cache/{method}/{benchmark}/{model}/{corpus}__{metric}__sub{N}.json

    Relative drops / PPL ratios are computed *per subset* against the matching
    base subset, then aggregated across subsets — this matches how the raw
    per-subset numbers were produced in test.py, so the confidence interval
    reflects subset-to-subset variability.

    Returns a DataFrame with columns:
        Algorithm_Model, Dataset, Metric, Value (mean), Std, CI95, N
    """
    # Per-subset base values.
    base_frob = {}    # (base_label, corpus, metric, subset_id) -> frob_norm
    base_ppl = {}     # (base_label, corpus, subset_id) -> ppl
    base_labels = set()

    # 1. First pass: base models.
    base_dir = CACHE_DIR / "base"
    if base_dir.is_dir():
        for file in sorted(base_dir.rglob("*.json")):
            rel = file.relative_to(CACHE_DIR)
            parts = rel.parts  # ("base", benchmark, model, filename)
            if len(parts) != 4:
                continue
            _, benchmark, model, _ = parts
            if benchmark_filter and benchmark != benchmark_filter:
                continue

            parsed = _parse_metric_file(file)
            if parsed is None:
                continue
            corpus, metric, subset_id = parsed

            label = f"base__{model}"
            base_labels.add(label)

            with open(file) as f:
                data = json.load(f)

            if metric == "ppl":
                base_ppl[(label, corpus, subset_id)] = data.get("ppl")
            else:
                base_frob[(label, corpus, metric, subset_id)] = data.get("frob_norm", 1.0)

    # core_name -> base_label, for matching unlearned checkpoints to their base.
    core_to_base = {}
    for bl in base_labels:
        core_to_base[_core_model_name(bl)] = bl

    # 2. Second pass: unlearned models — collect per-subset relative values.
    #    raw[(label, dataset, METRIC)] -> list of per-subset values
    raw = defaultdict(list)

    for file in sorted(CACHE_DIR.rglob("*.json")):
        rel = file.relative_to(CACHE_DIR)
        parts = rel.parts  # (method, benchmark, model, filename)
        if len(parts) != 4:
            continue
        method, benchmark, model, _ = parts
        if method == "base":
            continue
        if benchmark_filter and benchmark != benchmark_filter:
            continue

        parsed = _parse_metric_file(file)
        if parsed is None:
            continue
        corpus, metric, subset_id = parsed

        label = f"{method}/{model}"

        with open(file) as f:
            data = json.load(f)

        # Match this checkpoint to its base model.
        matched_base = None
        for core, bl in core_to_base.items():
            if core in label:
                matched_base = bl
                break

        dataset = pretty_dataset(corpus)

        if metric == "ppl":
            ppl_val = data.get("ppl")
            ratio = np.nan
            if matched_base is not None and ppl_val is not None:
                base_val = base_ppl.get((matched_base, corpus, subset_id))
                if base_val and base_val > 0:
                    ratio = ppl_val / base_val
            raw[(label, dataset, "PPL")].append(ratio)
        else:
            abs_drop = data.get("norm_drop", np.nan)
            rel_drop = np.nan
            if matched_base is not None and abs_drop is not None and np.isfinite(abs_drop):
                base_norm = base_frob.get((matched_base, corpus, metric, subset_id))
                if base_norm and base_norm > 0:
                    rel_drop = (abs_drop / base_norm) * 100
            raw[(label, dataset, metric.upper())].append(rel_drop)

    # 3. Aggregate across subsets.
    rows = []
    for (label, dataset, metric), values in raw.items():
        mean, std, ci, n = _aggregate(values)
        rows.append({
            "Algorithm_Model": label,
            "Dataset": dataset,
            "Metric": metric,
            "Value": mean,
            "Std": std,
            "CI95": ci,
            "N": n,
        })

    return pd.DataFrame(rows)


def _ordered_columns(columns, forget_first=True):
    """Sort dataset columns: forget first, other retain next, WikiText last."""
    wiki = [c for c in columns if c.lower() in ("wikitext", "retain2")]
    forget = [c for c in columns if "forget" in c.lower()]
    retain = [c for c in columns if c not in forget and c not in wiki]
    return forget + retain + wiki if forget_first else retain + forget + wiki


_METHOD_DISPLAY = {
    "adaptive_rmu": "Adaptive-RMU",
    "atu":          "ATU",
    "dpo":          "DPO",
    "ga":           "GA",
    "gd":           "GD",
    "loku":         "LoKU",
    "npo":          "NPO",
    "obliviate":    "Obliviate",
    "rmu":          "RMU",
    "rsv":          "RSV",
    "sim_npo":      "SimNPO",
    "spul":         "SPUL",
}

# (substring matched against the unlearned label, display title). Keys must be
# specific substrings of the cache dir names (which use '_' for '.'), distinct
# enough not to collide (Llama-3_1-8B vs Llama-3_2-3B).
MODELS = [
    ("Llama-3_1-8B",   "Llama-3.1-8B"),
    ("Llama-3_2-3B",   "Llama-3.2-3B"),
    ("zephyr-7b-beta", r"Zephyr-7B-$\beta$"),
    ("Qwen3-32B",      "Qwen3-32B"),
]

# Behaviour categories per (benchmark, model): {group_name: [method_keys] | _REST}.
# _REST means "every method not explicitly listed in another group of this
# model". Benchmarks/models NOT listed here fall back to a single "all" group
# (every method plotted together) — e.g. TOFU / Qwen3-32B, which has no runs.
_REST = "__rest__"
BEHAVIOUR_GROUPS = {
    "wmdp": {
        "Llama-3_2-3B":   {"no-op": ["atu", "obliviate", "rsv"], "partially-localized": ["rmu"], "collateral-dominant": ["adaptive_rmu", "npo", "sim_npo", "dpo"], "globally-destructive": _REST},
        "zephyr-7b-beta": {"no-op": ["atu"], "partially-localized": ["rsv", "adaptive_rmu"], "collateral-dominant": _REST, "globally-destructive": ["loku", "gd", "ga", "spul"]},
        "Llama-3_1-8B":   {"no-op": ["atu", "obliviate", "rsv"], "partially-localized": ["rmu"], "collateral-dominant": ["adaptive_rmu", "npo", "sim_npo", "dpo"], "globally-destructive": _REST},
        "Qwen3-32B":      {"no-op": ["atu", "rmu", "adaptive_rmu", "rsv", "obliviate"], "partially-localized": [], "collateral-dominant": _REST, "globally-destructive": ["spul"]},
    },
    "tofu-forget10": {
        "Llama-3_1-8B":   {"no-op": _REST, "partially-localized": ["ga", "gd"], "collateral-dominant": [],       "globally-destructive": ["loku"]},
        "Llama-3_2-3B":   {"no-op": _REST, "partially-localized": [],           "collateral-dominant": [],       "globally-destructive": ["loku"]},
        "zephyr-7b-beta": {"no-op": _REST, "partially-localized": ["ga"],       "collateral-dominant": ["gd"],   "globally-destructive": ["loku"]},
    },
    "muse-news": {
        "Llama-3_1-8B":   {"no-op": _REST, "partially-localized": [], "collateral-dominant": [], "globally-destructive": ["loku", "ga", "gd"]},
        "Llama-3_2-3B":   {"no-op": _REST, "partially-localized": [], "collateral-dominant": [], "globally-destructive": ["loku", "ga", "gd"]},
        "zephyr-7b-beta": {"no-op": _REST, "partially-localized": [], "collateral-dominant": [], "globally-destructive": ["loku", "ga", "gd"]},
    },
    "muse-books": {
        "Llama-3_1-8B":   {"no-op": ["atu", "obliviate", "rsv"], "partially-localized": _REST,                  "collateral-dominant": [],       "globally-destructive": ["loku", "ga", "gd", "spul"]},
        "Llama-3_2-3B":   {"no-op": ["atu", "obliviate", "rsv"], "partially-localized": _REST,                  "collateral-dominant": [],       "globally-destructive": ["loku", "ga", "gd", "spul"]},
        "zephyr-7b-beta": {"no-op": ["atu", "adaptive_rmu"],     "partially-localized": _REST,                  "collateral-dominant": ["loku"], "globally-destructive": ["ga", "gd", "spul"]},
        "Qwen3-32B":      {"no-op": _REST,                       "partially-localized": ["npo", "sim_npo", "dpo"], "collateral-dominant": [],    "globally-destructive": ["ga", "gd", "spul"]},
    },
}

# Vertical stacking order of behaviour groups in the combined heatmap
# (partially-localized on top, then collateral-dominant, globally destructive,
# and no-op at the bottom) and their display labels. Multi-word labels are
# wrapped so they fit beside single-row groups.
_GROUP_ORDER = ["partially-localized", "collateral-dominant", "globally-destructive", "no-op"]
_GROUP_DISPLAY = {
    "partially-localized":  "Partially-\nlocalized",
    "collateral-dominant":  "Collateral-\ndominant",
    "globally-destructive": "Globally\ndestructive",
    "no-op":                "No-op",
    "all":         "",
}

_DATASET_DISPLAY = {
    "bio-forget":    "Bio-Forget",
    "cyber-forget":  "Cyber-Forget",
    "bio-retain":    "Bio-Retain",
    "cyber-retain":  "Cyber-Retain",
    "retain2":       "WikiText",
    "wikitext":      "WikiText",
}


def _row_label(algorithm_model: str) -> str:
    """'npo/kl-Llama-3_1-8B_nu'  →  'NPO (ν)' """
    method = algorithm_model.split("/")[0]
    model_dir = algorithm_model.split("/", 1)[1] if "/" in algorithm_model else ""
    label = _METHOD_DISPLAY.get(method, method)
    if "_nu" in model_dir:
        label += r" ($\nu$)"
    return label


def _present_methods(df, model_key):
    """Set of method names present in the data for a given model."""
    mask = df["Algorithm_Model"].str.contains(model_key, case=False, regex=False)
    return set(df.loc[mask, "Algorithm_Model"].str.split("/").str[0])


def _resolve_groups(benchmark, model_key, present_methods):
    """Return [(group_name, [method_keys])] for a model.

    Uses BEHAVIOUR_GROUPS when configured; otherwise a single 'all' group with
    every present method (TOFU / MUSE-News / not-yet-grouped MUSE-Books).
    """
    cfg = BEHAVIOUR_GROUPS.get(benchmark, {}).get(model_key)
    if not cfg:
        return [("all", sorted(present_methods))]
    explicit = set()
    for methods in cfg.values():
        if isinstance(methods, list):
            explicit.update(methods)
    out = []
    for group_name, methods in cfg.items():
        if methods == _REST:
            resolved = sorted(present_methods - explicit)
        else:
            resolved = [m for m in methods if m in present_methods]
        out.append((group_name, resolved))
    return out


def _group_pivot(df, metric, model_key, methods, log_scale):
    """Pivot (methods × datasets) of `metric` for one model+method group.

    Returns (display, raw) or None if there's no data. Rows are ordered by the
    given `methods` order (base variant before its nu variant).
    """
    s = df[(df["Metric"] == metric)
           & df["Algorithm_Model"].str.contains(model_key, case=False, regex=False)].copy()
    if s.empty:
        return None
    s["_method"] = s["Algorithm_Model"].str.split("/").str[0]
    s = s[s["_method"].isin(methods)]
    if s.empty:
        return None

    s["_row"] = s["Algorithm_Model"].map(_row_label)
    s["_col"] = s["Dataset"].map(lambda d: _DATASET_DISPLAY.get(d, d))
    pos = {m: i for i, m in enumerate(methods)}
    s["_nu"] = s["Algorithm_Model"].str.contains("_nu", regex=False)

    order = {}
    for _, r in s.iterrows():
        order.setdefault(r["_row"], (pos.get(r["_method"], 99), bool(r["_nu"])))
    row_order = sorted(order, key=lambda rl: order[rl])

    raw = s.pivot_table(index="_row", columns="_col", values="Value", aggfunc="first")
    raw = raw.reindex(row_order)
    raw = raw[_ordered_columns(raw.columns)]
    display = np.log10(raw.clip(lower=1e-300)) if log_scale else raw
    return display, raw


# Panels shown side by side in each figure, in order: CoFi (left), CHess
# (middle), PPL (right).
_PANEL_SPECS = [
    ("COFI",  "CoFi",  False),
    ("CHESS", "CHess", False),
    ("PPL",   "PPL",   True),
]


def generate_combined_figure(df, bench_tag, model_key, model_title, groups):
    """One figure per model: behaviour groups stacked vertically (partially-
    localized on top, collateral-dominant, globally destructive, then no-op), metrics (CoFi | CHess | PPL) as columns.

    Every heatmap cell is the same physical size across all figures (fixed
    `cell` inches), so a group with fewer methods simply yields a shorter block
    instead of padding the figure with empty space. Within a model, all three
    group blocks of a given metric share one colour scale (and one colourbar),
    so scores are directly comparable top-to-bottom.
    """
    metrics = [(metric, name, log) for metric, name, log in _PANEL_SPECS]

    # pivots[(gi, mi)] -> (display, raw) or None ; also collect per-metric column
    # union and shared colour range across groups.
    pivots = {}
    metric_cols = {}          # mi -> ordered list of dataset columns
    metric_vmin_vmax = {}     # mi -> (vmin, vmax)
    group_nrows = []          # rows per group (max across metrics)

    for gi, (group_name, methods) in enumerate(groups):
        nrows_g = 0
        for mi, (metric, _name, log) in enumerate(metrics):
            res = _group_pivot(df, metric, model_key, methods, log) if methods else None
            pivots[(gi, mi)] = res
            if res is not None:
                nrows_g = max(nrows_g, res[0].shape[0])
        group_nrows.append(nrows_g)

    for mi, (metric, _name, log) in enumerate(metrics):
        cols, finite_vals = [], []
        for gi in range(len(groups)):
            res = pivots[(gi, mi)]
            if res is None:
                continue
            disp, _raw = res
            for c in disp.columns:
                if c not in cols:
                    cols.append(c)
            v = disp.values[np.isfinite(disp.values)]
            if v.size:
                finite_vals.append(v)
        metric_cols[mi] = _ordered_columns(cols) if cols else []
        if finite_vals:
            allv = np.concatenate(finite_vals)
            metric_vmin_vmax[mi] = (float(allv.min()), float(allv.max()))
        else:
            metric_vmin_vmax[mi] = None

    # Drop groups / metrics that carry no data at all.
    keep_g = [gi for gi in range(len(groups)) if group_nrows[gi] > 0]
    keep_m = [mi for mi in range(len(metrics)) if metric_vmin_vmax[mi] is not None]
    if not keep_g or not keep_m:
        print(f"  [skip] {model_title}: no data")
        return

    try:
        cmap = plt.get_cmap("mako").copy()      # registered by seaborn on import
    except (ValueError, KeyError):
        cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")                        # NaN cells render white

    cell = 0.42          # inches per heatmap cell -> constant block size
    cbar_w = 0.16
    cbar_pad = 0.06
    col_spacer = 0.60
    row_gap = 0.28       # inches between stacked group blocks

    # ---- column tracks: [hm, pad, cbar, (spacer)] per kept metric ----
    gs_widths, hm_col, cbar_col = [], {}, {}
    for j, mi in enumerate(keep_m):
        ncol = max(1, len(metric_cols[mi]))
        hm_col[mi] = len(gs_widths); gs_widths.append(cell * ncol)
        gs_widths.append(cbar_pad)
        cbar_col[mi] = len(gs_widths); gs_widths.append(cbar_w)
        if j < len(keep_m) - 1:
            gs_widths.append(col_spacer)

    # ---- row tracks: [group_block, (gap)] per kept group ----
    gs_heights, grp_row = [], {}
    for k, gi in enumerate(keep_g):
        grp_row[gi] = len(gs_heights); gs_heights.append(cell * group_nrows[gi])
        if k < len(keep_g) - 1:
            gs_heights.append(row_gap)

    left_in, right_in, top_in, bottom_in = 1.9, 0.15, 0.85, 1.05
    fig_w = left_in + sum(gs_widths) + right_in
    fig_h = top_in + sum(gs_heights) + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        len(gs_heights), len(gs_widths),
        width_ratios=gs_widths, height_ratios=gs_heights,
        wspace=0.0, hspace=0.0,
        left=left_in / fig_w, right=1 - right_in / fig_w,
        top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
    )

    row_span = (grp_row[keep_g[0]], grp_row[keep_g[-1]])  # for full-height cbars
    for j, mi in enumerate(keep_m):
        _metric, mname, _log = metrics[mi]
        vmin, vmax = metric_vmin_vmax[mi]
        cax = fig.add_subplot(gs[row_span[0]:row_span[1] + 1, cbar_col[mi]])
        cbar_done = False
        for k, gi in enumerate(keep_g):
            group_name = groups[gi][0]
            ax = fig.add_subplot(gs[grp_row[gi], hm_col[mi]])
            res = pivots[(gi, mi)]
            if res is None:
                ax.axis("off")
                continue
            disp = res[0].reindex(columns=metric_cols[mi])
            draw_cbar = not cbar_done
            sns.heatmap(disp, ax=ax, annot=False, cmap=cmap, vmin=vmin, vmax=vmax,
                        mask=disp.isna(), linewidths=0.3, linecolor="#ededed",
                        cbar=draw_cbar, cbar_ax=(cax if draw_cbar else None))
            if draw_cbar:
                cax.tick_params(labelsize=6)
                cbar_done = True

            if k == 0:
                ax.set_title(mname, fontsize=11, pad=4)
            ax.set_xlabel("")
            if gi == keep_g[-1]:
                ax.tick_params(axis="x", rotation=45, labelsize=7)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", rotation=0, labelsize=7)
            if j == 0:
                ax.set_ylabel(_GROUP_DISPLAY.get(group_name, group_name),
                              fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
        if not cbar_done:
            cax.axis("off")

    fig.suptitle(model_title, fontsize=13, y=1 - 0.28 / fig_h)

    out = RESULTS_DIR / f"{bench_tag}_{model_key}_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {out.relative_to(PROJECT_ROOT)}  "
          f"({len(keep_g)} groups, {sum(group_nrows[gi] for gi in keep_g)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cached CoFi/CHess/PPL metrics and generate summary tables + heatmaps."
    )
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Filter results to a specific benchmark "
                             "(e.g., wmdp, tofu-forget10). Default: all benchmarks.")
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        print(f"Error: Directory '{CACHE_DIR}' not found.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(benchmark_filter=args.benchmark)
    if df.empty:
        print("No unlearned model JSONs found in the cache.")
        return

    # ----------------------------------------------------------------
    # Combined table  (CoFi %, CHess %, PPL ratio — all in one pivot).
    # Each cell shows "mean ± 95% CI" across the evaluated subsets.
    # ----------------------------------------------------------------
    def _cell_str(row):
        mean, ci, metric = row["Value"], row["CI95"], row["Metric"]
        if mean is None or not np.isfinite(mean):
            return "—"
        if metric == "PPL":
            body = f"{mean:.2e}"
            return body if not np.isfinite(ci) else f"{body} ± {ci:.1e}"
        body = f"{mean:.2f}"
        return body if not np.isfinite(ci) else f"{body} ± {ci:.2f}"

    df = df.copy()
    df["_cell"] = df.apply(_cell_str, axis=1)

    combined = df.pivot_table(
        index="Algorithm_Model",
        columns=["Metric", "Dataset"],
        values="_cell",
        aggfunc="first",
    )

    # Sort the column levels: metric order, then forget-before-retain
    if isinstance(combined.columns, pd.MultiIndex):
        metric_order = {"COFI": 0, "CHESS": 1, "PPL": 2}
        sorted_cols = sorted(
            combined.columns,
            key=lambda c: (
                metric_order.get(c[0], 99),
                0 if "forget" in c[1].lower() else 1,
                c[1],
            ),
        )
        combined = combined[sorted_cols]

    n_subsets = int(df["N"].max()) if "N" in df and not df["N"].empty else 0
    print("\n" + "=" * 100)
    print("COMBINED RESULTS  (mean ± 95% CI across "
          f"{n_subsets} subset{'s' if n_subsets != 1 else ''})")
    print("  CoFi / CHess : Relative Drop (%)  — ↑ forget = better unlearning,  ↓ retain = less damage")
    print("  PPL          : Ratio (unlearned / original) — ↑ forget = better,  ≈1 retain = better")
    print("=" * 100)

    print(combined.fillna("—").to_string())
    print("=" * 100)

    # ----------------------------------------------------------------
    # Heatmaps: one figure per model, behaviour groups stacked vertically
    # (partially-localized, collateral-dominant, globally destructive, no-op) with CoFi | CHess | PPL as columns.
    # Constant cell size + shared per-metric colour scale across the groups.
    # ----------------------------------------------------------------
    print("\nGenerating combined heatmaps ...")
    bench_tag = (args.benchmark or "all").replace("-", "_")
    for model_key, model_title in MODELS:
        present = _present_methods(df, model_key)
        if not present:
            continue
        groups = _resolve_groups(args.benchmark, model_key, present)
        # Stack in the requested order; keep any extra groups after the known ones.
        groups = [g for g in groups if g[1]]  # drop empty groups
        groups.sort(key=lambda g: _GROUP_ORDER.index(g[0])
                    if g[0] in _GROUP_ORDER else len(_GROUP_ORDER))
        if not groups:
            print(f"  [skip] {model_title}: no non-empty groups")
            continue
        generate_combined_figure(df, bench_tag, model_key, model_title, groups)

    print("\nDone.")


if __name__ == "__main__":
    main()
