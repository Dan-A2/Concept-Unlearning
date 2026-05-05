import argparse
import json
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
         'base__zephyr-7b-beta'    -> 'zephyr-7b-beta'

    Strategy: strip the 'base__' prefix, then drop any leading
    organisation-style prefix that ends before the model family name.
    We look for common patterns like 'Meta-', 'Microsoft-', etc.
    """
    raw = base_label.replace("base__", "")
    # Strip a leading Org- prefix (e.g. Meta-, Google-, Microsoft-)
    # Pattern: one capitalised word followed by a hyphen at the start
    cleaned = re.sub(r"^[A-Z][a-z]*-", "", raw)
    return cleaned


def load_data(benchmark_filter=None):
    """Load cached metrics from the nested cofi_cache structure.

    Directory layout: cofi_cache/{method}/{benchmark}/{model}/{corpus}__{metric}.json
    """
    base_frob = {}    # (base_label, corpus, metric) -> frob_norm for CoFi/CHess
    base_ppl = {}     # (base_label, corpus) -> ppl
    base_labels = set()

    # 1. First pass: Load all base models
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

            stem_parts = file.stem.split("__")
            if len(stem_parts) != 2:
                continue
            corpus, metric = stem_parts

            label = f"base__{model}"
            base_labels.add(label)

            with open(file) as f:
                data = json.load(f)

            if metric == "ppl":
                base_ppl[(label, corpus)] = data.get("ppl")
            else:
                base_frob[(label, corpus, metric)] = data.get("frob_norm", 1.0)

    # Build a lookup: core_name -> base_label for matching
    core_to_base = {}
    for bl in base_labels:
        core = _core_model_name(bl)
        core_to_base[core] = bl

    # 2. Second pass: Load unlearned models
    unlearned_data = []

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

        stem_parts = file.stem.split("__")
        if len(stem_parts) != 2:
            continue
        corpus, metric = stem_parts

        label = f"{method}/{model}"

        with open(file) as f:
            data = json.load(f)

        # Find the matching base model by checking if the core name
        # appears as a substring of the unlearned label
        matched_base = None
        for core, bl in core_to_base.items():
            if core in label:
                matched_base = bl
                break

        if metric == "ppl":
            ppl_val = data.get("ppl")
            ppl_ratio = np.nan
            if matched_base:
                base_ppl_val = base_ppl.get((matched_base, corpus))
                if base_ppl_val and base_ppl_val > 0 and ppl_val is not None:
                    ppl_ratio = ppl_val / base_ppl_val

            unlearned_data.append({
                "Algorithm_Model": label,
                "Dataset": pretty_dataset(corpus),
                "Metric": "PPL",
                "Value": ppl_ratio,
            })
        else:
            abs_drop = data.get("norm_drop", np.nan)
            rel_drop = np.nan

            if matched_base and not np.isnan(abs_drop):
                base_norm = base_frob.get((matched_base, corpus, metric), None)
                if base_norm and base_norm > 0:
                    rel_drop = (abs_drop / base_norm) * 100

            unlearned_data.append({
                "Algorithm_Model": label,
                "Dataset": pretty_dataset(corpus),
                "Metric": metric.upper(),
                "Value": rel_drop,
            })

    return pd.DataFrame(unlearned_data)


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

_MODEL_PANELS = [
    ("Llama",   "Llama-3.1-8B"),
    ("zephyr",  r"Zephyr-7B-$\beta$"),
]

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


def _build_pivot(subset, model_kw, log_scale):
    """Filter subset to one model panel and return (pivot_display, pivot_raw)."""
    mask = subset["Algorithm_Model"].str.contains(model_kw, case=False)
    s = subset[mask].copy()
    if s.empty:
        return None, None

    s["_row"] = s["Algorithm_Model"].map(_row_label)
    s["_col"] = s["Dataset"].map(lambda d: _DATASET_DISPLAY.get(d, d))

    raw = s.pivot_table(index="_row", columns="_col", values="Value", aggfunc="first")
    raw = raw[_ordered_columns(raw.columns)]

    if log_scale:
        display = np.log10(raw.clip(lower=1e-300))
    else:
        display = raw

    return display, raw


def generate_heatmap(df, metric, fmt=".1f", cbar_label=None, title=None,  # noqa: ARG001 (title kept for call-site compat)
                     exclude_rows=None, col_width=1.5, log_scale=False,
                     benchmark=None):
    """Side-by-side heatmap: Llama (left) and Zephyr (right) panels."""

    subset = df[df["Metric"] == metric].copy()
    if exclude_rows:
        pattern = "|".join(exclude_rows)
        subset = subset[~subset["Algorithm_Model"].str.contains(pattern, case=False)]
    if subset.empty:
        print(f"  [skip] No data for {metric}")
        return

    if cbar_label is None:
        cbar_label = "Relative Drop (%)" if metric != "PPL" else "PPL ratio (log₁₀)"

    cmap = "mako"

    # Build both panels first so we can share vmin/vmax.
    panels = {}
    for kw, label in _MODEL_PANELS:
        disp, raw = _build_pivot(subset, kw, log_scale)
        if disp is not None:
            panels[label] = (disp, raw)

    if not panels:
        print(f"  [skip] No panel data for {metric}")
        return

    # Each panel gets its own vmin/vmax and its own colorbar.
    n_panels = len(panels)
    panel_items = list(panels.items())
    max_rows = max(len(p[0]) for _, p in panel_items)
    fig_height = max(5.0, 0.80 * max_rows + 2.5)

    cbar_width = 0.28
    spacer_width = 0.9  # explicit blank column between the two [heatmap+cbar] groups
    panel_widths = [col_width * len(p[0].columns) for _, p in panel_items]

    # Column layout: heatmap0, cbar0, spacer, heatmap1, cbar1, ...
    # Each panel occupies indices [idx*3, idx*3+1]; spacer sits at idx*3+2 (except last).
    gs_widths = []
    for i, pw in enumerate(panel_widths):
        gs_widths.append(pw)
        gs_widths.append(cbar_width)
        if i < n_panels - 1:
            gs_widths.append(spacer_width)
    n_gs_cols = len(gs_widths)
    fig_width = sum(gs_widths) + 1.0

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    gs = fig.add_gridspec(
        1, n_gs_cols,
        width_ratios=gs_widths,
        wspace=0.06,
        left=0.13, right=0.98,
        top=0.93, bottom=0.20,
    )

    for idx, (panel_label, (disp, raw)) in enumerate(panel_items):
        col_offset = idx * 3  # heatmap at 0, cbar at 1, spacer at 2 (per group)
        ax      = fig.add_subplot(gs[0, col_offset])
        cbar_ax = fig.add_subplot(gs[0, col_offset + 1])

        vals = disp.values.ravel()
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

        if log_scale:
            annot = np.where(
                np.isnan(raw.values), "",
                np.vectorize(lambda x: f"{x:.2e}")(raw.values),
            )
            sns.heatmap(
                disp, ax=ax, annot=annot, fmt="", cmap=cmap,
                vmin=vmin, vmax=vmax,
                linewidths=0.4, linecolor="#e0e0e0",
                cbar=True, cbar_ax=cbar_ax,
                cbar_kws={"label": cbar_label},
                annot_kws={"size": 20},
            )
            step = max(1, int(np.ceil((vmax - vmin) / 6)))
            ticks = np.arange(int(np.floor(vmin)), int(np.ceil(vmax)) + 1, step)
            cbar_ax.set_yticks(ticks)
            cbar_ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in ticks], fontsize=20)
            cbar_ax.set_ylabel(cbar_label, fontsize=21)
        else:
            sns.heatmap(
                disp, ax=ax, annot=True, fmt=fmt, cmap=cmap,
                vmin=vmin, vmax=vmax,
                linewidths=0.4, linecolor="#e0e0e0",
                cbar=True, cbar_ax=cbar_ax,
                cbar_kws={"label": cbar_label},
                annot_kws={"size": 20},
            )
            cbar_ax.tick_params(labelsize=20)
            cbar_ax.set_ylabel(cbar_label, fontsize=21)

        ax.set_title(panel_label, fontsize=25, fontweight="semibold", pad=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=20)
        ax.tick_params(axis="y", rotation=0,  labelsize=20)

        if idx == 0:
            ax.set_ylabel("Unlearning Method", fontsize=23)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        for j, col in enumerate(disp.columns):
            if "Forget" in col:
                ax.axvline(j,     color="#888888", lw=0.5, ls="--", zorder=3)
                ax.axvline(j + 1, color="#888888", lw=0.5, ls="--", zorder=3)

    # Shared x-axis label centred across all panels.
    fig.text(0.5, 0.10, "Evaluation Corpus", ha="center", fontsize=23)

    # Derive benchmark tag from data when not supplied explicitly.
    if benchmark is None:
        # Algorithm_Model looks like "method/bench_label/model" — grab the middle part.
        sample = df["Algorithm_Model"].dropna().iloc[0] if not df.empty else ""
        parts = sample.split("/")
        benchmark = parts[1] if len(parts) >= 3 else "unknown"

    bench_tag = benchmark.replace("-", "_")
    out = RESULTS_DIR / f"{metric.lower()}_{bench_tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {out.relative_to(PROJECT_ROOT)}")
    print(f"  -> Saved {out.with_suffix('.pdf').relative_to(PROJECT_ROOT)}")


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
    # Combined table  (CoFi %, CHess %, PPL ratio — all in one pivot)
    # ----------------------------------------------------------------
    combined = df.pivot_table(
        index="Algorithm_Model",
        columns=["Metric", "Dataset"],
        values="Value",
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

    print("\n" + "=" * 100)
    print("COMBINED RESULTS")
    print("  CoFi / CHess : Relative Drop (%)  — ↑ forget = better unlearning,  ↓ retain = less damage")
    print("  PPL          : Ratio (unlearned / original) — ↑ forget = better,  ≈1 retain = better")
    print("=" * 100)

    def _fmt(x):
        if np.isnan(x):
            return "    —"
        return f"{x:8.2f}"

    def _fmt_ppl(x):
        if np.isnan(x):
            return "    —"
        return f"{x:.3e}"

    col_formatter = {
        col: (_fmt_ppl if (isinstance(col, tuple) and col[0] == "PPL") else _fmt)
        for col in combined.columns
    }
    print(combined.to_string(formatters=col_formatter))
    print("=" * 100)

    # ----------------------------------------------------------------
    # Heatmaps  (one per metric)
    # ----------------------------------------------------------------
    print("\nGenerating heatmaps ...")
    generate_heatmap(df, "COFI", benchmark=args.benchmark)
    generate_heatmap(df, "CHESS", benchmark=args.benchmark)
    generate_heatmap(
        df, "PPL", fmt=".3e",
        cbar_label="PPL ratio — log₁₀ scale",
        title="PPL Ratio  (unlearned / original)",
        col_width=2.0,
        log_scale=True,
        benchmark=args.benchmark,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
