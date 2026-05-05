"""Localization scatter for concept-unlearning benchmarks.

Generates one plot per model (CoFi only), saved as PNG and PDF.

Three zones, decided geometrically:

  - No-op corner       : both axes <= NOOP_BOX  -> solid gray box
  - Collateral-dominant: above y = x diagonal   -> salmon/orange
  - Partially-localized: below y = x diagonal   -> green

A narrow strip around y = x is drawn as a HATCHED band on top of the
underlying zone color (so the green/salmon shows through), marking
points whose forget-vs-adjacent ordering is too close to call.

Cluster of every point is decided strictly by the diagonal.  Points
inside the hatched band get a dashed outer ring -- visual flag only,
their cluster assignment doesn't change.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze_metrics import load_data

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results"

# All methods we evaluate.  Cluster of each point is decided by its zone,
# not by a hand-picked dictionary.
METHODS = [
    "atu", "obliviate", "rmu", "rsv", "loku",
    "adaptive_rmu", "ga", "gd", "npo", "sim_npo", "dpo", "spul",
]

METHOD_DISPLAY = {
    "adaptive_rmu": "Adaptive-RMU",
    "sim_npo":      "SimNPO",
    "atu":          "ATU",
    "obliviate":    "Obliviate",
    "rmu":          "RMU",
    "rsv":          "RSV",
    "loku":         "LoKU",
    "ga":           "GA",
    "gd":           "GD",
    "npo":          "NPO",
    "dpo":          "DPO",
    "spul":         "SPUL",
}

MODEL_TAGS = {
    "Llama-3_1-8B":   "Llama-3.1-8B",
    "zephyr-7b-beta": "Zephyr-7B-$\\beta$",
}

# Per-benchmark forget / retain corpora (values of the Dataset column in the
# DataFrame produced by load_data, i.e. after pretty_dataset() mapping).
BENCHMARK_CORPORA: dict[str, tuple[list[str], list[str]]] = {
    "wmdp":          (["bio-forget", "cyber-forget"],
                      ["bio-retain", "cyber-retain"]),
    "muse-books":    (["books-forget"],
                      ["books-retain1", "books-retain2"]),
    "muse-news":     (["news-forget"],
                      ["news-retain1", "news-retain2"]),
    "tofu-forget10": (["tofu-forget10"],
                      ["tofu-retain90"]),
}

# Geometry constants ---------------------------------------------------------

NOOP_BOX = 1.0           # corner box: both forget and retain <= this -> no-op
DIAG_BAND_FRAC = 0.15    # ambiguous band around y = x: |y - x| / max(x,y) < this

# Color palette --------------------------------------------------------------

ZONE_COLORS = {
    "partial":     dict(face="#A8D5BA", alpha=0.55),  # green-ish
    "collateral":  dict(face="#E8A4A4", alpha=0.45),  # muted red
    "noop":        dict(face="#95A5A6", alpha=0.55),  # gray, opaque
}

# Hatched ambiguous band style.  Tighter cross-hatching for a cleaner,
# more professional look against either zone color.
HATCH_STYLE = dict(facecolor="none", edgecolor="#2c2c2c",
                   hatch="xxx", linewidth=0.0, alpha=0.45)

# Point styling
POINT_STYLE = {
    "partial":     dict(face="#2E8B57", edge="#1B5E3A", marker="o",
                        label="Partially-localized"),
    "collateral":  dict(face="#B0413E", edge="#6E1F1D", marker="D",
                        label="Collateral-dominant"),
    "noop":        dict(face="#95A5A6", edge="#566970", marker="s",
                        label="No-op"),
}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _strip_method_prefix(model_dir: str) -> str:
    for tag in MODEL_TAGS:
        if tag in model_dir:
            return tag
    return model_dir


def aggregate(df: pd.DataFrame, metric: str,
              forget_corpora: list[str], retain_corpora: list[str]) -> pd.DataFrame:
    """metric is 'COFI'."""
    sel = df[df["Metric"] == metric].copy()
    if sel.empty:
        raise SystemExit(f"No {metric} rows in cache.")
    sel = sel[~sel["Algorithm_Model"].str.endswith("_nu")]

    rows = []
    for label, sub in sel.groupby("Algorithm_Model"):
        method, model_dir = label.split("/", 1)
        if method not in METHODS:
            continue
        model_tag = _strip_method_prefix(model_dir)
        if model_tag not in MODEL_TAGS:
            continue
        f_vals = sub.loc[sub["Dataset"].isin(forget_corpora), "Value"].dropna()
        r_vals = sub.loc[sub["Dataset"].isin(retain_corpora), "Value"].dropna()
        if f_vals.empty or r_vals.empty:
            continue
        rows.append({
            "method": method,
            "display": METHOD_DISPLAY.get(method, method),
            "model": model_tag,
            "forget": float(f_vals.mean()),
            "retain": float(r_vals.mean()),
        })
    return pd.DataFrame(rows)


def classify(forget, retain, noop_box=NOOP_BOX):
    """Return (cluster, in_diag_band)."""
    if forget <= noop_box and retain <= noop_box:
        return "noop", False

    larger = max(forget, retain)
    in_diag_band = (abs(retain - forget) / max(larger, 1e-9)) < DIAG_BAND_FRAC \
                   if larger > noop_box else False

    if retain > forget:
        return "collateral", in_diag_band
    return "partial", in_diag_band


# --------------------------------------------------------------------------
# Label placement (greedy radial nudging)
# --------------------------------------------------------------------------

def _place_labels(ax, points, fontsize=9):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    placed_bboxes = []
    candidates = [
        (8, 6), (8, -6), (-10, 6), (-10, -6),
        (8, 14), (-10, 14), (8, -14), (-10, -14),
        (14, 0), (-16, 0), (0, 12), (0, -12),
    ]

    for x, y, text in points:
        best = None
        best_txt = None
        for dx, dy in candidates:
            txt = ax.annotate(
                text, (x, y),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, color="#1a1a1a",
                ha=("left" if dx >= 0 else "right"),
                va=("bottom" if dy >= 0 else "top"),
                zorder=6,
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2.0, foreground="white"),
                path_effects.Normal(),
            ])
            bbox = txt.get_window_extent(renderer=renderer)
            overlaps = sum(int(bbox.overlaps(b)) for b in placed_bboxes)
            if best is None or overlaps < best[0]:
                if best_txt is not None:
                    best_txt.remove()
                best = (overlaps, bbox, (dx, dy))
                best_txt = txt
            else:
                txt.remove()
            if overlaps == 0:
                break
        if best_txt is not None:
            placed_bboxes.append(best[1])


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

def _style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 9.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        # Hatch line width controls the thickness of the diagonal hatch lines.
        "hatch.linewidth": 0.6,
    })


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _plot_axis(ax, sub, title, metric_label):
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888888")
        ax.set_title(title)
        return

    hi = float(np.nanmax([sub["forget"].max(), sub["retain"].max()]))
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    pad = 0.12 * hi
    lo = -0.03 * hi
    top = hi + pad

    x_grid = np.linspace(lo, top, 400)

    # ---- Background zones ----------------------------------------------
    # Partially-localized: y < x
    ax.fill_between(
        x_grid,
        np.full_like(x_grid, lo),
        x_grid,
        where=(x_grid > lo),
        facecolor=ZONE_COLORS["partial"]["face"],
        alpha=ZONE_COLORS["partial"]["alpha"],
        zorder=0, linewidth=0,
    )

    # Collateral-dominant: y > x
    ax.fill_between(
        x_grid,
        x_grid,
        np.full_like(x_grid, top),
        where=(x_grid < top),
        facecolor=ZONE_COLORS["collateral"]["face"],
        alpha=ZONE_COLORS["collateral"]["alpha"],
        zorder=0, linewidth=0,
    )

    # No-op corner (solid gray, overrides zone color in the corner)
    ax.add_patch(plt.Rectangle(
        (lo, lo), NOOP_BOX - lo, NOOP_BOX - lo,
        facecolor=ZONE_COLORS["noop"]["face"],
        alpha=ZONE_COLORS["noop"]["alpha"],
        edgecolor="none", zorder=0.6,
    ))

    # ---- Hatched ambiguous band around y = x ---------------------------
    # |y - x| / max(x, y) < DIAG_BAND_FRAC, only outside the no-op corner.
    band_low  = x_grid * (1 - DIAG_BAND_FRAC)
    band_high = x_grid * (1 + DIAG_BAND_FRAC)
    ax.fill_between(
        x_grid, band_low, band_high,
        where=(x_grid > NOOP_BOX),
        facecolor="none",
        edgecolor=HATCH_STYLE["edgecolor"],
        hatch=HATCH_STYLE["hatch"],
        alpha=HATCH_STYLE["alpha"],
        linewidth=0.0,
        zorder=1,
    )

    # ---- Reference line ------------------------------------------------
    ax.plot([lo, top], [lo, top],
            color="#444444", linestyle=(0, (5, 4)), linewidth=1.0,
            zorder=2, label="_nolegend_")
    ax.text(top * 0.98, top * 0.98, "y = x",
            ha="right", va="top", fontsize=8.5,
            color="#666666", style="italic", zorder=3)

    # ---- Points --------------------------------------------------------
    classifications = []
    for _, row in sub.iterrows():
        cluster, in_diag = classify(row["forget"], row["retain"])
        classifications.append((cluster, in_diag))

    label_pts = []
    for cluster_key in ["partial", "collateral", "noop"]:
        idxs = [i for i, c in enumerate(classifications) if c[0] == cluster_key]
        if not idxs:
            continue
        style = POINT_STYLE[cluster_key]
        xs = [sub.iloc[i]["forget"] for i in idxs]
        ys = [sub.iloc[i]["retain"] for i in idxs]
        ambiguous = [classifications[i][1] for i in idxs]

        # Filled marker
        ax.scatter(
            xs, ys, s=130,
            facecolor=style["face"],
            edgecolor="white",
            linewidth=1.4,
            marker=style["marker"],
            zorder=4,
        )
        # Dark ring for definition
        ax.scatter(
            xs, ys, s=130,
            facecolor="none",
            edgecolor=style["edge"],
            linewidth=0.6,
            marker=style["marker"],
            zorder=4.5,
        )
        # Dashed outer ring on points inside the ambiguous band
        amb_xs = [x for x, a in zip(xs, ambiguous) if a]
        amb_ys = [y for y, a in zip(ys, ambiguous) if a]
        if amb_xs:
            ax.scatter(
                amb_xs, amb_ys, s=260,
                facecolor="none",
                edgecolor="#3a3a3a",
                linewidth=1.1,
                linestyle=(0, (2, 2)),
                marker="o",
                zorder=4.6,
            )
        for i in idxs:
            label_pts.append((sub.iloc[i]["forget"],
                              sub.iloc[i]["retain"],
                              sub.iloc[i]["display"]))

    _place_labels(ax, label_pts, fontsize=9)

    ax.set_xlim(lo, top)
    ax.set_ylim(lo, top)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(rf"$\Delta\mathrm{{{metric_label}}}(\mathcal{{C}}_F)$  —  forget corpora (%)")
    ax.set_ylabel(rf"$\Delta\mathrm{{{metric_label}}}(\mathcal{{C}}_A)$  —  retain corpora (%)")
    ax.set_title(title)
    ax.grid(True, which="major", color="#E5E5E5", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _build_legend(fig):
    # Marker handles for the three clusters
    cluster_handles = [
        Line2D([0], [0], marker=POINT_STYLE[k]["marker"], color="white",
               markerfacecolor=POINT_STYLE[k]["face"],
               markeredgecolor=POINT_STYLE[k]["edge"],
               markeredgewidth=1.0, markersize=10,
               label=POINT_STYLE[k]["label"])
        for k in ["partial", "collateral", "noop"]
    ]

    # Hatched-band handle.  Patch with the same hatch style as the band
    # itself, so the legend shows what the hatched region in the plot is.
    ambiguous_band_handle = Patch(
        facecolor="none",
        edgecolor=HATCH_STYLE["edgecolor"],
        hatch=HATCH_STYLE["hatch"],
        alpha=HATCH_STYLE["alpha"],
        linewidth=0.0,
        label="Ambiguous band ($y \\approx x$)",
    )

    diag_handle = Line2D(
        [0], [0], color="#444444", linestyle=(0, (5, 4)),
        linewidth=1.0, label="$y = x$",
    )

    fig.legend(
        handles=cluster_handles + [ambiguous_band_handle, diag_handle],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        edgecolor="#dddddd",
        fontsize=9.5,
    )


def plot_single(agg: pd.DataFrame, model_tag: str, model_label: str,
                metric: str, metric_label: str, out_path: Path):
    _style()
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.8))
    _plot_axis(
        ax,
        agg[agg["model"] == model_tag],
        f"{model_label} — {metric_label}",
        metric_label,
    )
    _build_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}  and  {out_path.with_suffix('.pdf')}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="wmdp",
                   help=(
                       "Benchmark to plot (e.g. wmdp, muse-books, muse-news, "
                       "tofu-forget10). Must be a key in BENCHMARK_CORPORA or "
                       "you can supply --forget-corpora / --retain-corpora manually."
                   ))
    p.add_argument("--forget-corpora", nargs="+", default=None,
                   help="Override forget corpus names (Dataset column values).")
    p.add_argument("--retain-corpora", nargs="+", default=None,
                   help="Override retain corpus names (Dataset column values).")
    p.add_argument("--out-dir", default=str(RESULTS_DIR),
                   help="Directory to write outputs into.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.forget_corpora or args.retain_corpora:
        if not (args.forget_corpora and args.retain_corpora):
            raise SystemExit("Provide both --forget-corpora and --retain-corpora together.")
        forget_corpora = args.forget_corpora
        retain_corpora = args.retain_corpora
    elif args.benchmark in BENCHMARK_CORPORA:
        forget_corpora, retain_corpora = BENCHMARK_CORPORA[args.benchmark]
    else:
        known = ", ".join(sorted(BENCHMARK_CORPORA))
        raise SystemExit(
            f"Unknown benchmark '{args.benchmark}'. Known: {known}. "
            "Use --forget-corpora / --retain-corpora to specify corpora manually."
        )

    df = load_data(benchmark_filter=args.benchmark)
    if df.empty:
        raise SystemExit(
            f"No cached metrics found for benchmark '{args.benchmark}'. "
            f"Run test.py with --benchmark {args.benchmark} first."
        )

    bench_safe = args.benchmark.replace("-", "_")

    metric, metric_label, label_lower = "COFI", "CoFi", "cofi"
    agg = aggregate(df, metric, forget_corpora, retain_corpora)
    if agg.empty:
        print(f"[skip] no rows for {metric}")
        return

    for model_tag, model_label in MODEL_TAGS.items():
        sub = agg[agg["model"] == model_tag]
        if sub.empty:
            print(f"[skip] {model_tag} {metric}: empty")
            continue
        tag_safe = model_tag.replace("-", "_").replace(".", "_")
        out_path = out_dir / f"{bench_safe}_{label_lower}_{tag_safe}.png"
        plot_single(agg, model_tag, model_label, metric, metric_label, out_path)


if __name__ == "__main__":
    main()