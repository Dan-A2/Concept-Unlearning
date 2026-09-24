"""Benchmark configurations for unlearning experiments.

Supported benchmarks:
  wmdp  — WMDP bio/cyber forget corpora (2 topics, default)
  tofu  — TOFU fictitious authors (1 topic, requires prior fine-tuning)
  muse  — MUSE news/books corpora (1 topic, --muse_corpus required)
"""

TOFU_SPLITS = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}

MUSE_CORPORA = {"news", "books"}


def get_benchmark_config(benchmark, tofu_split=None, muse_corpus=None):
    """Return a dict with forget_corpora, retain_corpora, topic_names,
    max_lengths, and bench_label for the chosen benchmark."""

    if benchmark == "wmdp":
        return dict(
            forget_corpora=["bio-forget-corpus", "cyber-forget-corpus"],
            retain_corpora=["wikitext", "wikitext"],
            topic_names={0: "bio", 1: "cyber"},
            max_lengths=[512, 768],
            bench_label="wmdp",
        )

    elif benchmark == "tofu":
        if tofu_split not in TOFU_SPLITS:
            raise ValueError(
                f"--tofu_split must be one of {list(TOFU_SPLITS)}, got {tofu_split}"
            )
        retain_split = TOFU_SPLITS[tofu_split]
        return dict(
            forget_corpora=[f"tofu-{tofu_split}"],
            retain_corpora=[f"tofu-{retain_split}"],
            topic_names={0: f"tofu-{tofu_split}"},
            max_lengths=[512],
            bench_label=f"tofu-{tofu_split}",
        )

    elif benchmark == "muse":
        if muse_corpus not in MUSE_CORPORA:
            raise ValueError(
                f"--muse_corpus must be one of {sorted(MUSE_CORPORA)}, "
                f"got {muse_corpus}"
            )
        # MUSE "raw" subset has splits: forget, retain1, retain2, holdout.
        # retain1 is the calibrator used during unlearning;
        # retain2 is held out for evaluation (see test.py).
        return dict(
            forget_corpora=[f"muse-{muse_corpus}-forget"],
            retain_corpora=[f"muse-{muse_corpus}-retain1"],
            topic_names={0: f"muse-{muse_corpus}"},
            max_lengths=[512],
            bench_label=f"muse-{muse_corpus}",
        )

    elif benchmark == "inject":
        # Known-fact injection oracle experiment: unlearn F from theta_inj while
        # retaining A. Corpora are saved to data/inject-{F,A} by
        # known_fact_injection.py --stages save_inj.
        return dict(
            forget_corpora=["inject-F"],
            retain_corpora=["inject-A"],
            topic_names={0: "inject"},
            max_lengths=[128],
            bench_label="inject",
        )

    raise ValueError(f"Unknown benchmark: {benchmark}")


def apply_benchmark_config(args, benchmark, tofu_split=None, muse_corpus=None):
    """Modify an Args object in-place based on the chosen benchmark."""
    config = get_benchmark_config(benchmark, tofu_split, muse_corpus)

    args.forget_corpora = config["forget_corpora"]
    args.retain_corpora = config["retain_corpora"]
    args.topic_names = config["topic_names"]
    args.max_lengths = config["max_lengths"]
    args.bench_label = config["bench_label"]

    n_topics = len(args.forget_corpora)

    # Resize per-topic lists to match the number of topics
    for attr in ("alpha", "steering_coeff_list", "scale"):
        val = getattr(args, attr, None)
        if isinstance(val, list) and len(val) != n_topics:
            setattr(args, attr, [val[0]] * n_topics)

    return args


def add_benchmark_args(parser):
    """Add --benchmark and related arguments to an argparse parser."""
    parser.add_argument(
        "--benchmark", type=str, default="wmdp",
        choices=["wmdp", "tofu", "muse", "inject"],
        help="Unlearning benchmark (default: wmdp).",
    )
    parser.add_argument(
        "--tofu_split", type=str, default=None,
        choices=["forget01", "forget05", "forget10"],
        help="TOFU forget split (required when --benchmark tofu).",
    )
    parser.add_argument(
        "--muse_corpus", type=str, default=None,
        choices=["news", "books"],
        help="MUSE corpus (required when --benchmark muse).",
    )
