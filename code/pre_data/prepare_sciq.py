"""Download SciQ (allenai/sciq) and reshape it into the input format the KDA metric uses.

Every emitted sample carries exactly the fields KDA consumes:
(passage, question, options, answer_idx).

Two modes are available:

  full   (default)  Export every split in its entirety:
                      datasets/sciq/{sciq_train_full,sciq_val_full,sciq_test_full}.json
                      and sciq_all_combined.json (all splits merged, each sample tagged
                      with a "split" attribute).

  sample            Draw a fixed-size random subset from a single split, e.g. the
                    50-question set used by the KDA_tiny experiment (sciq_50.json).

Summary statistics are written to stdout and to results/ex1_reproduce_KDA_pipeline/prep_sciq.log.
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset

# --------------------------------------------------------------------------------------
# Cross-stage imports. This script lives in `code/<stage>/`, so `code/` itself is put on
# `sys.path`; `utils.paths` and `ex1_reproduce_KDA.kda_tiny` then resolve no matter which
# directory the script is launched from. The *project root* is deliberately NOT added --
# it contains a `datasets/` folder that would shadow the HuggingFace `datasets` package.
# --------------------------------------------------------------------------------------
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent, resolve

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DATASET_NAME = "allenai/sciq"
ALL_SPLITS = ("train", "validation", "test")

# Requested output names; "validation" is abbreviated to "val" in the filename.
SPLIT_FILENAMES = {
    "train": "sciq_train_full.json",
    "validation": "sciq_val_full.json",
    "test": "sciq_test_full.json",
}
COMBINED_FILENAME = "sciq_all_combined.json"

NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub", "fsspec", "datasets")


def setup_logging(log_path: str, append: bool) -> logging.Logger:
    """Mirror every record to `log_path` and to stdout."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):  # avoid duplicate handlers on re-entry
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("sciq_prep")


def has_support(row: dict) -> bool:
    """True when the row carries a usable target fact.

    KDA contrasts a with-fact pass against a without-fact pass, so a missing or blank
    'support' makes both passes identical and the resulting score meaningless.
    """
    support = row.get("support")
    return bool(support and support.strip())


def build_sample(row: dict, rng: random.Random) -> dict:
    """Combine the correct answer with the three distractors and shuffle them.

    `answer_idx` is recomputed after shuffling so it always points at the correct option.
    """
    correct = row["correct_answer"].strip()
    options = [
        correct,
        row["distractor1"].strip(),
        row["distractor2"].strip(),
        row["distractor3"].strip(),
    ]
    rng.shuffle(options)
    return {
        "passage": row["support"].strip() if row.get("support") else "",
        "question": row["question"].strip(),
        "options": options,
        "answer_idx": options.index(correct),
        "correct_answer": correct,
    }


def validate(samples: Sequence[dict]) -> None:
    """Sanity check: options[answer_idx] must always be the correct answer."""
    for sample in samples:
        assert len(sample["options"]) == 4, f"expected 4 options, got {len(sample['options'])}"
        assert sample["options"][sample["answer_idx"]] == sample["correct_answer"], (
            f"answer_idx does not point at the correct answer for question "
            f"{sample['question'][:60]!r}"
        )


def answer_distribution(samples: Sequence[dict]) -> Dict[int, int]:
    return {i: sum(1 for s in samples if s["answer_idx"] == i) for i in range(4)}


def write_json(payload, path: str, logger: logging.Logger) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError:
        logger.exception("Failed to write '%s'", path)
        raise


def process_split(
    split: str,
    seed: int,
    allow_empty_support: bool,
    logger: logging.Logger,
) -> Tuple[List[dict], Dict]:
    """Load one split, drop unusable rows, and format every remaining sample."""
    logger.info("-" * 70)
    logger.info("Processing split '%s'", split)

    try:
        dataset = load_dataset(DATASET_NAME, split=split)
    except Exception:
        logger.exception("Failed to load split '%s' of %s", split, DATASET_NAME)
        raise

    total_raw = len(dataset)
    logger.info("  Raw samples                : %d", total_raw)

    indices = list(range(total_raw))
    if allow_empty_support:
        removed = 0
        logger.info("  Empty-support filter       : DISABLED (--allow-empty-support)")
    else:
        kept = [i for i in indices if has_support(dataset[i])]
        removed = total_raw - len(kept)
        indices = kept
        logger.info(
            "  Removed (empty 'support')  : %d (%.2f%% of raw)",
            removed,
            100.0 * removed / total_raw if total_raw else 0.0,
        )

    # One RNG per split, advanced in iteration order, so the shuffling is reproducible
    # for a given (seed, split, filter setting).
    rng = random.Random(seed)
    samples = []
    for rank, idx in enumerate(indices):
        sample = build_sample(dataset[idx], rng)
        sample["id"] = rank
        sample["sciq_index"] = idx
        samples.append(sample)

    validate(samples)
    distribution = answer_distribution(samples)

    logger.info("  Final processed samples    : %d", len(samples))
    logger.info(
        "  answer_idx distribution    : %s",
        {k: f"{v} ({100.0 * v / len(samples):.1f}%)" for k, v in distribution.items()}
        if samples
        else distribution,
    )

    stats = {
        "split": split,
        "total_raw": total_raw,
        "removed_empty_support": removed,
        "final_count": len(samples),
        "answer_idx_distribution": distribution,
    }
    return samples, stats


def export_full(args, logger: logging.Logger) -> None:
    """Export every requested split in full, plus the optional combined file."""
    logger.info("Mode                         : full export")
    logger.info("Dataset                      : %s", DATASET_NAME)
    logger.info("Splits                       : %s", ", ".join(args.splits))
    logger.info("Seed                         : %d", args.seed)
    logger.info("Keep empty 'support'         : %s", args.allow_empty_support)
    logger.info("Output directory             : %s", args.out_dir)

    all_stats = []
    combined: List[dict] = []

    for split in args.splits:
        samples, stats = process_split(split, args.seed, args.allow_empty_support, logger)

        path = f"{args.out_dir.rstrip('/')}/{SPLIT_FILENAMES[split]}"
        write_json(samples, path, logger)
        stats["output_file"] = path
        logger.info("  Wrote                      : %s", path)

        if args.combined:
            combined.extend({**sample, "split": split} for sample in samples)

        all_stats.append(stats)

    if args.combined and combined:
        path = f"{args.out_dir.rstrip('/')}/{COMBINED_FILENAME}"
        write_json(combined, path, logger)
        logger.info("-" * 70)
        logger.info("Combined file                : %s (%d samples)", path, len(combined))

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "  %-12s %10s %12s %10s  %s",
        "split", "raw", "removed", "final", "answer_idx distribution",
    )
    for stats in all_stats:
        logger.info(
            "  %-12s %10d %12d %10d  %s",
            stats["split"],
            stats["total_raw"],
            stats["removed_empty_support"],
            stats["final_count"],
            stats["answer_idx_distribution"],
        )
    logger.info(
        "  %-12s %10d %12d %10d",
        "TOTAL",
        sum(s["total_raw"] for s in all_stats),
        sum(s["removed_empty_support"] for s in all_stats),
        sum(s["final_count"] for s in all_stats),
    )
    if args.combined and combined:
        logger.info(
            "  combined answer_idx distribution: %s", answer_distribution(combined)
        )


def export_sample(args, logger: logging.Logger) -> None:
    """Draw a fixed-size random subset from a single split.

    The RNG call order here is deliberately identical to the original single-split
    script, so re-running it regenerates a byte-identical sciq_50.json.
    """
    logger.info("Mode                         : fixed-size sample")
    logger.info("Dataset / split              : %s / %s", DATASET_NAME, args.split)
    logger.info("Sample size                  : %d", args.n)
    logger.info("Seed                         : %d", args.seed)
    logger.info("Keep empty 'support'         : %s", args.allow_empty_support)

    dataset = load_dataset(DATASET_NAME, split=args.split)
    total_raw = len(dataset)
    logger.info("  Raw samples                : %d", total_raw)

    indices = list(range(total_raw))
    removed = 0
    if not args.allow_empty_support:
        kept = [i for i in indices if has_support(dataset[i])]
        removed = total_raw - len(kept)
        indices = kept
        logger.info("  Removed (empty 'support')  : %d", removed)
        logger.info("  Eligible for sampling      : %d", len(indices))

    if args.n > len(indices):
        logger.error(
            "Requested %d samples but only %d are eligible in split '%s'",
            args.n, len(indices), args.split,
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    chosen = rng.sample(indices, args.n)

    samples = []
    for rank, idx in enumerate(chosen):
        sample = build_sample(dataset[idx], rng)
        sample["id"] = rank
        sample["sciq_index"] = idx
        samples.append(sample)

    validate(samples)
    write_json(samples, args.out, logger)

    logger.info("  Final processed samples    : %d", len(samples))
    logger.info("  answer_idx distribution    : %s", answer_distribution(samples))
    logger.info("  Wrote                      : %s", args.out)
    logger.debug("First sample:\n%s", json.dumps(samples[0], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and format the SciQ dataset for the KDA metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python code/pre_data/prepare_sciq.py\n"
            "      Export all three splits in full plus the combined file.\n"
            "  python code/pre_data/prepare_sciq.py --splits test --no-combined\n"
            "      Export only sciq_test_full.json.\n"
            "  python code/pre_data/prepare_sciq.py --mode sample --n 50 --split test --out datasets/sciq/sciq_50.json\n"
            "      Regenerate the 50-question subset used by the KDA_tiny experiment.\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("full", "sample"),
        default="full",
        help="'full' exports whole splits (default); 'sample' draws a fixed-size subset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=list(ALL_SPLITS),
        help="Splits to export in full mode (default: all three).",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/sciq",
        help="Directory for full-mode output files (relative to the project root).",
    )
    parser.add_argument(
        "--combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Also write {COMBINED_FILENAME} with a 'split' attribute per sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for option shuffling.")
    parser.add_argument(
        "--allow-empty-support",
        action="store_true",
        help=(
            "Keep samples whose 'support' field is empty or missing. They are dropped by "
            "default: KDA compares a with-fact against a without-fact pass, so an empty "
            "passage makes both passes identical and the score meaningless."
        ),
    )
    parser.add_argument(
        "--log-file", default="results/ex1_reproduce_KDA_pipeline/prep_sciq.log", help="Execution log path."
    )
    parser.add_argument("--append-log", action="store_true", help="Append to the log file.")

    # sample-mode only
    parser.add_argument("--split", default="test", help="[sample mode] Split to sample from.")
    parser.add_argument("--n", type=int, default=50, help="[sample mode] Number of samples.")
    parser.add_argument(
        "--out",
        default="datasets/sciq/sciq_50.json",
        help="[sample mode] Output file.",
    )

    args = parser.parse_args()

    # Every path on the command line is interpreted relative to the project root, so
    # the script behaves the same whatever the current working directory is.
    args.out_dir = resolve(args.out_dir)
    args.out = ensure_parent(resolve(args.out))
    args.log_file = ensure_parent(resolve(args.log_file))
    os.makedirs(args.out_dir, exist_ok=True)

    logger = setup_logging(args.log_file, args.append_log)
    logger.info("=" * 70)
    logger.info("SciQ dataset preparation")
    logger.info("=" * 70)

    try:
        if args.mode == "full":
            export_full(args, logger)
        else:
            export_sample(args, logger)
    except Exception:
        logger.exception("Dataset preparation failed")
        raise

    logger.info("Execution log                : %s", args.log_file)
    logger.info("Done.")


if __name__ == "__main__":
    main()
