"""Benchmark configurations for unlearning experiments.

Supported benchmarks:
  wmdp  — WMDP bio/cyber forget corpora (2 topics, default)
  tofu  — TOFU fictitious authors (1 topic, requires prior fine-tuning)
  muse  — MUSE news/books corpora (1 topic, --muse_corpus required)
  blur  — BLUR forget/retain evaluation (1 topic, --blur_task required)
"""

TOFU_SPLITS = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}

MUSE_CORPORA = {"news", "books"}

BLUR_TASKS = {"rwku", "whp"}


def get_benchmark_config(benchmark, tofu_split=None, muse_corpus=None,
                         blur_task=None):
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

    if benchmark == "tofu":
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

    if benchmark == "muse":
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

    if benchmark == "blur":
        if blur_task not in BLUR_TASKS:
            raise ValueError(
                f"--blur_task must be one of {sorted(BLUR_TASKS)}, "
                f"got {blur_task}"
            )
        return dict(
            forget_corpora=[f"blur-{blur_task}-forget"],
            retain_corpora=[f"blur-{blur_task}-retain"],
            topic_names={0: f"blur-{blur_task}"},
            max_lengths=[512],
            bench_label=f"blur-{blur_task}",
        )

    raise ValueError(f"Unknown benchmark: {benchmark}")


def apply_benchmark_config(args, benchmark, tofu_split=None,
                           muse_corpus=None, blur_task=None):
    """Modify an Args object in-place based on the chosen benchmark."""
    config = get_benchmark_config(benchmark, tofu_split, muse_corpus,
                                  blur_task)

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
        choices=["wmdp", "tofu", "muse", "blur"],
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
    parser.add_argument(
        "--blur_task", type=str, default=None,
        choices=["rwku", "whp"],
        help="BLUR task (required when --benchmark blur).",
    )
