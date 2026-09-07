"""Download OpenBookQA (allenai/openbookqa) and reshape it into the KDA input format.

Every emitted sample carries exactly the fields the KDA metric consumes
(passage, question, options, answer_idx), so the same experiment runner works on
OpenBookQA and on SciQ without changes.

Field mapping (OpenBookQA -> KDA format):

    question       <- question_stem
    passage        <- fact1            (the core science fact; the KDA "target fact")
    options        <- choices['text']  (4 texts, kept in their original A/B/C/D order)
    correct_answer <- the option whose choices['label'] equals answerKey
    answer_idx     <- position of that option in `options` (0-3)
    id             <- 0-based rank within the exported file (integer)
    obqa_index     <- 0-based row number in the upstream split (integer)
    obqa_id        <- the upstream string id (e.g. "8-343"), kept for traceability

Note on configs: `fact1` is published only in the `additional` config, while `main`
is the canonical benchmark. This script loads `main` and joins `fact1` from
`additional` on the shared `id`, verifying that question, options and answer key
agree across the two configs before accepting the join.

Output files (mirroring the SciQ layout produced by prepare_sciq.py):

    obqa_train_full.json      full `train` split
    obqa_val_full.json        full `validation` split
    obqa_test_full.json       full `test` split (the 500-question benchmark)
    obqa_all_combined.json    all splits merged, each sample tagged with `split`
    obqa_50.json              the first 50 samples of the test split, for smoke tests

Summary statistics are written to stdout and to results/ex1_reproduce_KDA_pipeline/prep_openbookqa.log.
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

DATASET_NAME = "allenai/openbookqa"
MAIN_CONFIG = "main"
FACT_CONFIG = "additional"  # the only config that ships fact1
ALL_SPLITS = ("train", "validation", "test")

# "validation" is abbreviated to "val" in the filename, as in the SciQ export.
SPLIT_FILENAMES = {
    "train": "obqa_train_full.json",
    "validation": "obqa_val_full.json",
    "test": "obqa_test_full.json",
}
COMBINED_FILENAME = "obqa_all_combined.json"
DEBUG_FILENAME = "obqa_50.json"

# The small debug set is the head of this split, so it stays a strict prefix of the
# full file and every id / obqa_index lines up with the full export.
DEBUG_SOURCE_SPLIT = "test"
DEBUG_SIZE = 50

N_OPTIONS = 4

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

    return logging.getLogger("obqa_prep")


def load_split(split: str, logger: logging.Logger):
    """Load the `main` split and the matching `additional` split that carries fact1."""
    try:
        main = load_dataset(DATASET_NAME, MAIN_CONFIG, split=split)
    except Exception:
        logger.exception("Failed to load %s/%s split '%s'", DATASET_NAME, MAIN_CONFIG, split)
        raise
    try:
        extra = load_dataset(DATASET_NAME, FACT_CONFIG, split=split)
    except Exception:
        logger.exception(
            "Failed to load %s/%s split '%s' (needed for the fact1 target facts)",
            DATASET_NAME, FACT_CONFIG, split,
        )
        raise
    return main, extra


def build_fact_index(extra, logger: logging.Logger) -> Dict[str, dict]:
    """Map upstream id -> row of the `additional` config."""
    index: Dict[str, dict] = {}
    for row in extra:
        if row["id"] in index:
            logger.warning("Duplicate id '%s' in the '%s' config; keeping the first row",
                           row["id"], FACT_CONFIG)
            continue
        index[row["id"]] = row
    return index


def rows_agree(main_row: dict, fact_row: dict) -> bool:
    """True when the two configs describe the same question, options and answer."""
    return (
        main_row["question_stem"] == fact_row["question_stem"]
        and list(main_row["choices"]["text"]) == list(fact_row["choices"]["text"])
        and main_row["answerKey"] == fact_row["answerKey"]
    )


def build_sample(main_row: dict, fact: str, rng: Optional[random.Random]) -> dict:
    """Convert one OpenBookQA row into the KDA input format.

    Options keep their published A/B/C/D order by default: unlike SciQ (where the
    correct answer is a separate field that would always land at index 0), OpenBookQA
    already distributes the answer key across the four positions. `rng` is only passed
    when --shuffle-options is requested; `answer_idx` is recomputed either way.
    """
    labels = [label.strip() for label in main_row["choices"]["label"]]
    options = [text.strip() for text in main_row["choices"]["text"]]
    answer_key = main_row["answerKey"].strip()

    correct = options[labels.index(answer_key)]
    if rng is not None:
        rng.shuffle(options)

    return {
        "passage": fact.strip(),
        "question": main_row["question_stem"].strip(),
        "options": options,
        "answer_idx": options.index(correct),
        "correct_answer": correct,
        "answer_key": answer_key,
    }


def validate(samples: Sequence[dict]) -> None:
    """Sanity check: options[answer_idx] must always be the correct answer."""
    for sample in samples:
        assert len(sample["options"]) == N_OPTIONS, (
            f"expected {N_OPTIONS} options, got {len(sample['options'])} for question "
            f"{sample['question'][:60]!r}"
        )
        assert sample["options"][sample["answer_idx"]] == sample["correct_answer"], (
            f"answer_idx does not point at the correct answer for question "
            f"{sample['question'][:60]!r}"
        )
        assert sample["passage"], (
            f"empty target fact for question {sample['question'][:60]!r}"
        )


def answer_distribution(samples: Sequence[dict]) -> Dict[int, int]:
    return {i: sum(1 for s in samples if s["answer_idx"] == i) for i in range(N_OPTIONS)}


def write_json(payload, path: str, logger: logging.Logger) -> None:
    try:
        ensure_parent(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError:
        logger.exception("Failed to write '%s'", path)
        raise


def process_split(
    split: str,
    seed: int,
    shuffle_options: bool,
    logger: logging.Logger,
) -> Tuple[List[dict], Dict]:
    """Load one split, join the target facts, and format every usable row."""
    logger.info("-" * 70)
    logger.info("Processing split '%s'", split)

    main, extra = load_split(split, logger)
    total_raw = len(main)
    logger.info("  Raw samples (%-10s)   : %d", MAIN_CONFIG, total_raw)
    logger.info("  Raw samples (%-10s)   : %d", FACT_CONFIG, len(extra))

    facts = build_fact_index(extra, logger)

    # One RNG per split, advanced in iteration order, so a shuffled export is
    # reproducible for a given (seed, split).
    rng = random.Random(seed) if shuffle_options else None

    samples: List[dict] = []
    missing_fact = 0
    mismatched = 0
    malformed = 0

    for idx, row in enumerate(main):
        fact_row = facts.get(row["id"])
        if fact_row is None or not (fact_row.get("fact1") or "").strip():
            # KDA contrasts a with-fact pass against a without-fact pass, so a row with
            # no target fact would make both passes identical and its score meaningless.
            missing_fact += 1
            logger.debug("  Row %d (id=%s): no fact1 available; dropped", idx, row["id"])
            continue
        if not rows_agree(row, fact_row):
            mismatched += 1
            logger.warning(
                "  Row %d (id=%s): '%s' and '%s' disagree on the question/options/answer; dropped",
                idx, row["id"], MAIN_CONFIG, FACT_CONFIG,
            )
            continue

        try:
            sample = build_sample(row, fact_row["fact1"], rng)
        except (KeyError, ValueError, IndexError, AttributeError):
            malformed += 1
            logger.exception("  Row %d (id=%s): malformed record; dropped", idx, row["id"])
            continue

        sample["id"] = len(samples)
        sample["obqa_index"] = idx
        sample["obqa_id"] = row["id"]
        samples.append(sample)

    validate(samples)
    distribution = answer_distribution(samples)
    dropped = missing_fact + mismatched + malformed

    logger.info("  Dropped (no fact1)         : %d", missing_fact)
    logger.info("  Dropped (config mismatch)  : %d", mismatched)
    logger.info("  Dropped (malformed row)    : %d", malformed)
    logger.info("  Final processed samples    : %d", len(samples))
    logger.info(
        "  answer_idx distribution    : %s",
        {k: f"{v} ({100.0 * v / len(samples):.1f}%)" for k, v in distribution.items()}
        if samples
        else distribution,
    )
    if samples:
        logger.info(
            "  Mean fact length (chars)   : %.1f",
            sum(len(s["passage"]) for s in samples) / len(samples),
        )

    stats = {
        "split": split,
        "total_raw": total_raw,
        "dropped_missing_fact": missing_fact,
        "dropped_config_mismatch": mismatched,
        "dropped_malformed": malformed,
        "dropped_total": dropped,
        "final_count": len(samples),
        "answer_idx_distribution": distribution,
    }
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download OpenBookQA and export it in the KDA input format "
            "(passage / question / options / answer_idx)."
        ),
        epilog=(
            "Examples:\n"
            "  python code/pre_data/prepare_openbookqa.py\n"
            "      Export all three splits plus the combined and 50-question files.\n"
            "  python code/pre_data/prepare_openbookqa.py --splits test --no-combined --no-debug-set\n"
            "      Export only datasets/openbookqa/obqa_test_full.json.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=list(ALL_SPLITS),
        help="Splits to export (default: all three).",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/openbookqa",
        help="Directory for the exported files (relative to the project root).",
    )
    parser.add_argument(
        "--shuffle-options",
        action="store_true",
        help=(
            "Shuffle the four options instead of keeping the published A/B/C/D order. "
            "Off by default: OpenBookQA already spreads the answer key over all four "
            "positions, so the original order carries no positional bias."
        ),
    )
    parser.add_argument(
        "--combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Also write {COMBINED_FILENAME} with a 'split' attribute per sample.",
    )
    parser.add_argument(
        "--debug-set",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            f"Also write {DEBUG_FILENAME}: the first {DEBUG_SIZE} samples of the "
            f"'{DEBUG_SOURCE_SPLIT}' split, for quick smoke tests."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for --shuffle-options.")
    parser.add_argument(
        "--log-file", default="results/ex1_reproduce_KDA_pipeline/prep_openbookqa.log", help="Execution log path."
    )
    parser.add_argument("--append-log", action="store_true", help="Append to the log file.")
    args = parser.parse_args()

    # Every path on the command line is interpreted relative to the project root, so
    # the script behaves the same whatever the current working directory is.
    args.out_dir = resolve(args.out_dir)
    args.log_file = ensure_parent(resolve(args.log_file))
    os.makedirs(args.out_dir, exist_ok=True)

    logger = setup_logging(args.log_file, args.append_log)

    logger.info("=" * 70)
    logger.info("OpenBookQA -> KDA dataset preparation")
    logger.info("=" * 70)
    logger.info("Dataset                      : %s", DATASET_NAME)
    logger.info("Configs                      : %s (+ %s for fact1)", MAIN_CONFIG, FACT_CONFIG)
    logger.info("Splits                       : %s", ", ".join(args.splits))
    logger.info("Shuffle options              : %s (seed %d)", args.shuffle_options, args.seed)
    logger.info("Output directory             : %s", args.out_dir)
    logger.info("Execution log                : %s", args.log_file)

    all_stats = []
    combined: List[dict] = []
    by_split: Dict[str, List[dict]] = {}

    for split in args.splits:
        samples, stats = process_split(split, args.seed, args.shuffle_options, logger)
        if not samples:
            logger.error("Split '%s' produced no usable sample; aborting.", split)
            sys.exit(1)

        path = os.path.join(args.out_dir, SPLIT_FILENAMES[split])
        write_json(samples, path, logger)
        stats["output_file"] = path
        logger.info("  Wrote                      : %s", path)
        logger.debug("First sample:\n%s", json.dumps(samples[0], ensure_ascii=False, indent=2))

        by_split[split] = samples
        if args.combined:
            # Same convention as sciq_all_combined.json: tag each sample with its split
            # so the merged file can be filtered back apart.
            combined.extend({**sample, "split": split} for sample in samples)
        all_stats.append(stats)

    # ---------------------------------------------------------------------------------
    # Derived files. Both are written from the in-memory samples that were just exported,
    # so they can never disagree with the per-split files.
    # ---------------------------------------------------------------------------------
    combined_path = None
    if args.combined and combined:
        combined_path = os.path.join(args.out_dir, COMBINED_FILENAME)
        write_json(combined, combined_path, logger)
        logger.info("-" * 70)
        logger.info("Combined file                : %s (%d samples)", combined_path, len(combined))
        logger.info(
            "  answer_idx distribution    : %s", answer_distribution(combined)
        )
        logger.info("  Samples per split          : %s",
                    {split: len(rows) for split, rows in by_split.items()})

    debug_path = None
    if args.debug_set:
        source = by_split.get(DEBUG_SOURCE_SPLIT)
        if source is None:
            logger.warning(
                "--debug-set skipped: the '%s' split was not exported this run "
                "(requested splits: %s).",
                DEBUG_SOURCE_SPLIT, ", ".join(args.splits),
            )
        elif len(source) < DEBUG_SIZE:
            logger.warning(
                "--debug-set skipped: the '%s' split holds only %d usable samples, "
                "fewer than the %d requested.",
                DEBUG_SOURCE_SPLIT, len(source), DEBUG_SIZE,
            )
        else:
            debug_samples = source[:DEBUG_SIZE]
            validate(debug_samples)
            debug_path = os.path.join(args.out_dir, DEBUG_FILENAME)
            write_json(debug_samples, debug_path, logger)
            logger.info("-" * 70)
            logger.info(
                "Debug file                   : %s (first %d samples of '%s')",
                debug_path, len(debug_samples), DEBUG_SOURCE_SPLIT,
            )
            logger.info(
                "  answer_idx distribution    : %s", answer_distribution(debug_samples)
            )

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  %-12s %8s %9s %8s  %s", "split", "raw", "dropped", "final",
                "answer_idx distribution")
    for stats in all_stats:
        logger.info(
            "  %-12s %8d %9d %8d  %s",
            stats["split"], stats["total_raw"], stats["dropped_total"],
            stats["final_count"], stats["answer_idx_distribution"],
        )
    logger.info(
        "  %-12s %8d %9d %8d",
        "TOTAL",
        sum(s["total_raw"] for s in all_stats),
        sum(s["dropped_total"] for s in all_stats),
        sum(s["final_count"] for s in all_stats),
    )
    if args.combined and combined:
        logger.info(
            "  %-12s %8s %9s %8d  %s",
            "COMBINED", "-", "-", len(combined), answer_distribution(combined),
        )

    logger.info("")
    logger.info("Files written:")
    for stats in all_stats:
        logger.info("  %-12s %6d samples -> %s",
                    stats["split"], stats["final_count"], stats["output_file"])
    if combined_path:
        logger.info("  %-12s %6d samples -> %s", "combined", len(combined), combined_path)
    if debug_path:
        logger.info("  %-12s %6d samples -> %s", "debug (50)", DEBUG_SIZE, debug_path)

    # Integrity gate: a silently dropped row would break the SciQ/OpenBookQA comparison,
    # so make it loud rather than leaving it in the per-split lines above.
    total_dropped = sum(stats["dropped_total"] for stats in all_stats)
    if total_dropped:
        logger.warning(
            "%d row(s) were dropped across the exported splits; see the per-split "
            "breakdown above for the reason.", total_dropped,
        )
    else:
        logger.info("")
        logger.info(
            "Integrity: 0 rows dropped across %d split(s) -- every upstream row joined "
            "cleanly between the '%s' and '%s' configs.",
            len(all_stats), MAIN_CONFIG, FACT_CONFIG,
        )


if __name__ == "__main__":
    main()
