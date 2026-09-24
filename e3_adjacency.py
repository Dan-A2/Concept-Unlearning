"""E3 - Adjacency diagnostic: CoFi Fisher-overlap AND sentence-embedding, both.

Operational test of whether a candidate Adjacent-Retain set C_A is valid,
via two independent mechanisms:

  CoFi (per model)   - do C_F and C_A share top Fisher-mass parameters far
                        more than C_F and C_G? Reuses test.compute_cofi (the
                        log-space transform is monotonic, so top-k-by-value
                        == top-k-by-Fisher-mass); overlap(X,Y) =
                        |top_k(X) & top_k(Y)| / k for k in {1e3,1e4,1e5}.
                        Model-dependent (base model's own parameter geometry).

  Embedding (once per benchmark) - is C_A's content semantically closer to
                        C_F than to C_G? sim = mean pairwise cosine
                        similarity (sentence-transformers/all-MiniLM-L6-v2,
                        the single most-downloaded model on Hugging Face and
                        the de facto default sentence embedder). Model-
                        independent, so computed once per benchmark and
                        repeated across every model's rows.
                            justified_adjacent_embed  <=>  sim(F,A) > sim(A,G)

Expected for a valid adjacent set: overlap(C_F,C_A) >> overlap(C_F,C_G) (CoFi)
AND sim(C_F,C_A) > sim(C_A,C_G) (embedding) -- two independent lines of
evidence for the same claim, one structural (model-internal), one semantic
(content-only). WMDP high vs MUSE-Books low on the CoFi side is the expected
cross-benchmark contrast.

Output: Results/rebuttal/adjacency_diagnostic.csv
"""

import argparse
from pathlib import Path

import torch

import exp_common as X
import test as T

COLUMNS = ["model", "benchmark", "pair", "k", "overlap_cofi",
           "sim_embed", "justified_adjacent_embed"]

_CORE_DIR = {"llama-3.1-8b": "Llama-3.1-8B", "zephyr-7b-beta": "zephyr-7b-beta"}

PAIRS = {"C_F-C_A": ("C_F", "C_A"), "C_F-C_G": ("C_F", "C_G"), "C_A-C_G": ("C_A", "C_G")}


def _triple(benchmark):
    """Return (corpora_dict, {'C_F':.., 'C_A':.., 'C_G':..}) for a benchmark."""
    if benchmark == "wmdp":
        corpora, f, r = T.get_benchmark_corpora("wmdp")
    elif benchmark.startswith("muse-"):
        corpora, f, r = T.get_benchmark_corpora("muse", muse_corpus=benchmark.split("-", 1)[1])
    elif benchmark.startswith("tofu-"):
        corpora, f, r = T.get_benchmark_corpora("tofu", tofu_split=benchmark.split("-", 1)[1])
    else:
        raise ValueError(benchmark)
    return corpora, {"C_F": f[0], "C_A": r[0], "C_G": r[-1]}


def _base_path(mkey, benchmark):
    """The 'base' (pre-unlearning) model: HF id for WMDP, finetuned checkpoint
    for muse/tofu."""
    if benchmark == "wmdp":
        return X.MODELS[mkey]
    core = _CORE_DIR[mkey]
    if benchmark.startswith("muse-"):
        corpus = benchmark.split("-", 1)[1]
        return str(X.PROJECT_ROOT / "checkpoints" / "muse_finetune" / corpus / core)
    if benchmark.startswith("tofu-"):
        return str(X.PROJECT_ROOT / "checkpoints" / "tofu_finetune" / core)
    raise ValueError(benchmark)


# ---------------------------------------------------------------------------
# CoFi Fisher top-k overlap (per model)
# ---------------------------------------------------------------------------

def _topk_sets(model, tok, texts, tp, ks):
    """CoFi diagonal -> {k: set(top-k flat indices)} over the concatenated
    target-parameter vector (fixed sorted key order)."""
    d = X.cofi_diag(model, tok, texts, tp)
    keys = sorted(d.keys())
    flat = torch.cat([d[k].reshape(-1).float() for k in keys])
    out = {}
    for k in ks:
        kk = min(int(k), flat.numel())
        out[k] = set(torch.topk(flat, kk).indices.tolist())
    del d, flat
    X.free()
    return out


# ---------------------------------------------------------------------------
# Sentence-embedding similarity (model-independent)
# ---------------------------------------------------------------------------

_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        print(f"[{T._ts()}] loading sentence-transformers/all-MiniLM-L6-v2 ...")
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ENCODER


def mean_cosine_sim(a_texts, b_texts):
    """Mean pairwise cosine similarity between every doc in a_texts and every
    doc in b_texts (embeddings are normalized, so dot product == cosine sim)."""
    enc = _encoder()
    ea = enc.encode(a_texts, normalize_embeddings=True)
    eb = enc.encode(b_texts, normalize_embeddings=True)
    return float((ea @ eb.T).mean())


def run(args):
    X.setup_determinism()
    ks = args.ks
    out = X.CSVWriter(X.RESULTS_DIR / "adjacency_diagnostic.csv", COLUMNS)

    for bench in args.benchmarks:
        corpora, tri = _triple(bench)
        print(f"\n[{T._ts()}] benchmark {bench}  triple={tri}")
        sub = {name: X.one_sample(corpora[c], args.n, seed=0) for name, c in tri.items()}

        # Embedding similarity is model-independent -- compute once per
        # benchmark, reuse across every model below.
        sim = {pair: mean_cosine_sim(sub[p], sub[q]) for pair, (p, q) in PAIRS.items()}
        justified = sim["C_F-C_A"] > sim["C_A-C_G"]
        print(f"  [embed] sim(F,A)={sim['C_F-C_A']:.4f}  sim(A,G)={sim['C_A-C_G']:.4f}  "
              f"sim(F,G)={sim['C_F-C_G']:.4f}  justified_adjacent={justified}")

        for mkey in args.models:
            bp = _base_path(mkey, bench)
            if bench != "wmdp" and not Path(bp).exists():
                print(f"[{T._ts()}]  skip {mkey}/{bench}: base checkpoint missing ({bp})")
                continue
            print(f"[{T._ts()}]  [cofi] model {mkey}  base={bp}")
            model, tok = X.load_model(bp)
            tp = X.targets(model)
            top = {name: _topk_sets(model, tok, sub[name], tp, ks) for name in tri}
            del model, tok, tp
            X.free()

            for pair, (p, q) in PAIRS.items():
                for k in ks:
                    denom = min(int(k), len(top[p][k]) or 1)
                    ov = len(top[p][k] & top[q][k]) / denom
                    out.row(model=mkey, benchmark=bench, pair=pair, k=int(k),
                            overlap_cofi=f"{ov:.6f}",
                            sim_embed=f"{sim[pair]:.6f}",
                            justified_adjacent_embed=(justified if pair == "C_F-C_A" else ""))

    out.close()
    print(f"\n[{T._ts()}] wrote {X.RESULTS_DIR / 'adjacency_diagnostic.csv'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["llama-3.1-8b", "zephyr-7b-beta"])
    p.add_argument("--benchmarks", nargs="+",
                   default=["wmdp", "tofu-forget10", "muse-books", "muse-news"],
                   help="WMDP + MUSE (and TOFU) for the cross-benchmark contrast.")
    p.add_argument("--ks", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument("--n", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
