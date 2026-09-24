"""Re-report the CoFi and/or CHess ablation using the "Relative Drop (%)"
convention that analyze_metrics.py / the main results tables use, instead of
the raw absolute Frobenius distance chess_probe_ablation.py reports natively.

    relative_drop_pct = 100 * absolute_drop / base_frob_norm

- `absolute_drop` comes from the already-computed cofi_ablation__*.json /
  chess_ablation__*.json files under Results/chess_ablation
  (chess_probe_ablation.py's own output). Nothing is re-run;
  chess_probe_ablation.py is not touched or imported for compute, only for
  its reporting functions (analyze_cofi/make_plots_cofi for CoFi,
  analyze_chess/make_plots_chess for CHess), so the printed tables look
  exactly like chess_probe_ablation.py's own output.
- `base_frob_norm` (the base model's own CoFi- or CHess-diagonal norm) comes
  from cofi_cache/base/wmdp/<model>/<corpus>__<metric>__sub0.json -- the same
  cache analyze_metrics.py itself reads, produced by test.py's original run.
  Verified for both metrics: same "sub0" document subset used by the
  ablation (same corpus name, same subset_id=0, same deterministic index
  cache), so dividing by it is a correct like-for-like normalization, not an
  approximation.

CAVEAT: base_frob_norm is a single value (from whatever n/K test.py's own run
used for its full sub0 pass) applied uniformly across every breakpoint in the
ablation sweep (n=50/100/200/... for CoFi's size axis and CHess's size axis;
K=2/4/8 for CHess's probe axis). It is NOT re-derived at each breakpoint's own
n/K -- that would require re-running CoFi/CHess on the base model at each one
(GPU work), which this script deliberately does not do. Everything here is a
CPU-only post-processing pass over numbers already on disk.

--metric {cofi,chess,both} selects which ablation to re-report (default: both).

No GPU needed, no model loaded -- pure JSON/CSV arithmetic.
"""

import argparse
import json

import chess_probe_ablation as CPA

BASE_MODEL_CORE = "Meta-Llama-3_1-8B"   # matches cofi_cache's base dir naming


def _base_norm(corpus, metric):
    p = (CPA.PROJECT_ROOT / "cofi_cache" / "base" / "wmdp" / BASE_MODEL_CORE /
         f"{corpus}__{metric}__sub0.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("frob_norm")


def _rescale_size_dict(size, metric):
    """Rescale a {corpus: {method: {subset: {point: value}}}} dict from
    absolute drop to 100*drop/base_norm (%). Shared by CoFi's "size" axis and
    CHess's "probe"/"size" axes -- same shape, same rescaling rule, just a
    different `metric` tag for the base_norm lookup.

    Non-dict entries (e.g. an in-progress method's "_done": False bookkeeping
    flag) are passed through untouched -- only actual {point: value} dicts
    get rescaled."""
    out = {}
    missing = []
    for corpus, methods in size.items():
        bn = _base_norm(corpus, metric)
        if not bn:
            missing.append(corpus)
            out[corpus] = methods   # leave un-rescaled (still absolute) if no norm found
            continue
        out[corpus] = {
            method: {subset: ({pt: (100.0 * v / bn) for pt, v in points.items()}
                              if isinstance(points, dict) else points)
                     for subset, points in subsets.items()}
            for method, subsets in methods.items()
        }
    if missing:
        print(f"[warn] no cached base frob_norm ({metric}) in cofi_cache for corpora {missing}; "
              f"those columns are left as absolute drop, NOT a percentage.")
    return out


def rescale_cofi(payload):
    size = payload["results"].get("cofi", {}).get("size", {})
    new_payload = dict(payload)
    new_payload["results"] = {**payload["results"],
                              "cofi": {**payload["results"].get("cofi", {}),
                                       "size": _rescale_size_dict(size, "cofi")}}
    return new_payload


def rescale_chess(payload):
    chess = payload["results"].get("chess", {})
    new_payload = dict(payload)
    new_payload["results"] = {**payload["results"],
                              "chess": {**chess,
                                        "probe": _rescale_size_dict(chess.get("probe", {}), "chess"),
                                        "size": _rescale_size_dict(chess.get("size", {}), "chess")}}
    return new_payload


def _fix_methods(payload, *size_dicts):
    """chess_probe_ablation.py's _meta() overwrites (not merges) "methods" on
    every save with whatever --methods the LAST job used -- so if you scale
    back to fewer methods for one run, earlier-computed methods' data is
    still sitting in `results` untouched, but the report's label list goes
    stale and silently drops them. Rebuild "methods" as the union of every
    label actually present in the data, keeping real paths where known and a
    placeholder for anything discovered only in the data (paths are never
    used for reporting/plots, only compute, so this is safe)."""
    known = dict(payload.get("methods", {}))
    for d in size_dicts:
        for methods in d.values():
            for label in methods:
                known.setdefault(label, "(path unknown -- discovered from saved results)")
    new_payload = dict(payload)
    new_payload["methods"] = known
    return new_payload


def _header(metric):
    print("\n" + "#" * 78)
    print(f"# {metric.upper() if metric == 'cofi' else 'CHess'} ablation -- "
          f"RELATIVE DROP (%) convention (matches analyze_metrics.py)")
    print(f"# value = 100 * absolute_drop / base_frob_norm(corpus)   [base_frob_norm from cofi_cache]")
    print("#" * 78)


def report_cofi():
    paths = sorted(CPA.OUT_DIR.glob("cofi_ablation__*.json"))
    single = CPA.OUT_DIR / "cofi_ablation.json"
    if single.exists():
        paths.append(single)
    if not paths:
        print(f"[skip] no cofi_ablation_*.json found under {CPA.OUT_DIR}")
        return
    print(f"[merge] combining {len(paths)} CoFi file(s): {[p.name for p in paths]}")
    merged = CPA._merge_payloads(paths)
    payload = _fix_methods(rescale_cofi(merged), merged["results"].get("cofi", {}).get("size", {}))
    _header("cofi")
    CPA.analyze_cofi(payload)
    CPA.make_plots_cofi(payload)


def report_chess():
    paths = sorted(CPA.OUT_DIR.glob("chess_ablation__*.json"))
    single = CPA.OUT_DIR / "chess_ablation.json"
    if single.exists():
        paths.append(single)
    if not paths:
        print(f"[skip] no chess_ablation_*.json found under {CPA.OUT_DIR}")
        return
    print(f"[merge] combining {len(paths)} CHess file(s): {[p.name for p in paths]}")
    merged = CPA._merge_payloads(paths)
    chess = merged["results"].get("chess", {})
    payload = _fix_methods(rescale_chess(merged), chess.get("probe", {}), chess.get("size", {}))
    _header("chess")
    CPA.analyze_chess(payload)
    CPA.make_plots_chess(payload)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", choices=["cofi", "chess", "both"], default="both",
                   help="Which already-computed ablation to re-report (default: both).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.metric in ("both", "cofi"):
        report_cofi()
    if args.metric in ("both", "chess"):
        report_chess()


if __name__ == "__main__":
    main()
