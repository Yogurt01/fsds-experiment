"""Categorise the baseline KDA experiment output into the four before/after buckets.

Every evaluated sample is placed in exactly one bucket per model, using the pair
(is_correct_without_fact, is_correct_with_fact) that the baseline run already logged
for each ensemble member:

    both_correct     (Case 2)  correct WITHOUT the fact AND correct WITH the fact
    wrong_to_correct (Case 3)  wrong WITHOUT the fact,   correct WITH the fact
    both_wrong       (Case 1)  wrong WITHOUT the fact,   wrong  WITH the fact
    correct_to_wrong (Case 4)  correct WITHOUT the fact, wrong  WITH the fact

`both_correct` is the bucket Pillar 3 targets: the fact cannot be shown to matter
there, because the model was already right without it. `run_counterfactual_experiment.py`
consumes these buckets to separate genuine context dependence from prior reliance.

This script is read-only with respect to the raw data: it never rewrites the input
results file or any dataset file, it only emits new files under --out-dir.

Usage:
    python code/ex1_reproduce_KDA/categorize_kda_results.py                  # auto-detect the results file
    python code/ex1_reproduce_KDA/categorize_kda_results.py --results results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json \
        --out-dir results/ex1_category_questions/basic_category
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------------------
# Cross-stage imports. This script lives in `code/<stage>/`, so `code/` itself is put on
# `sys.path`; `utils.paths` and `ex1_reproduce_KDA.kda_tiny` then resolve no matter which
# directory the script is launched from. The *project root* is deliberately NOT added --
# it contains a `datasets/` folder that would shadow the HuggingFace `datasets` package.
# --------------------------------------------------------------------------------------
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import PROJECT_ROOT, ensure_parent, resolve

# --------------------------------------------------------------------------------------
# Paths are resolved relative to the project root (not to this file's directory), so the
# script runs identically from any working directory.
# --------------------------------------------------------------------------------------
WORKSPACE = PROJECT_ROOT

# Preference order used when --results is not given: richest / most recent run first.
RESULTS_CANDIDATES = (
    "results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json",
    "results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_full.json",
    "results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json",
)

# Bucket names in the order they are reported everywhere (files, console, summary).
CATEGORIES = ("both_correct", "wrong_to_correct", "both_wrong", "correct_to_wrong")

CATEGORY_CASE = {
    "both_correct": "Case 2",
    "wrong_to_correct": "Case 3",
    "both_wrong": "Case 1",
    "correct_to_wrong": "Case 4",
}

CATEGORY_DESCRIPTION = {
    "both_correct": "correct without fact AND with fact (fact not demonstrably needed)",
    "wrong_to_correct": "wrong without fact, correct with fact (fact-driven gain)",
    "both_wrong": "wrong without fact AND with fact (fact did not help)",
    "correct_to_wrong": "correct without fact, wrong with fact (fact hurt / distracted)",
}

# Label used for the pooled-probability ensemble, which is treated as one more "model".
ENSEMBLE_KEY = "ENSEMBLE_pooled_probability_vote"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("categorize_kda")


def setup_logging(log_path: str, verbose: bool) -> None:
    """DEBUG to the log file, INFO (or DEBUG with --verbose) to stdout."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):  # avoid duplicate handlers on re-entry
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)


def slugify_model(model_name: str) -> str:
    """'Riiid/kda-mpnet-base-race' -> 'kda_mpnet_base_race' (safe as a filename)."""
    tail = model_name.split("/")[-1]
    return "".join(char if char.isalnum() else "_" for char in tail).strip("_").lower()


def resolve_results_path(explicit: Optional[str]) -> str:
    """Return the results file to read, honouring --results then the candidate list."""
    if explicit:
        path = resolve(explicit, WORKSPACE)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        return path

    for candidate in RESULTS_CANDIDATES:
        path = resolve(candidate, WORKSPACE)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No results file found in "
        f"{WORKSPACE}; looked for {', '.join(RESULTS_CANDIDATES)}. Pass --results."
    )


def classify(is_correct_without_fact: bool, is_correct_with_fact: bool) -> str:
    """Map one (without-fact, with-fact) correctness pair onto its bucket name."""
    if is_correct_without_fact:
        return "both_correct" if is_correct_with_fact else "correct_to_wrong"
    return "wrong_to_correct" if is_correct_with_fact else "both_wrong"


def pooled_prediction(record: Dict, model_names: Sequence[str], key: str) -> int:
    """argmax of the probability vectors averaged over every ensemble member.

    This is the same 'pooled probability vote' the baseline summary reports, recomputed
    here so the ensemble can be bucketed exactly like an individual model.
    """
    n_models = len(model_names)
    n_options = len(record["options"])
    averaged = [
        sum(record["per_model"][name][key][i] for name in model_names) / n_models
        for i in range(n_options)
    ]
    return max(range(n_options), key=averaged.__getitem__)


def build_entry(
    record: Dict,
    predicted_idx_without_fact: int,
    predicted_idx_with_fact: int,
    category: str,
) -> Dict:
    """Assemble the per-sample entry written into the categorised JSON files."""
    options = record["options"]
    answer_idx = record["answer_idx"]
    return {
        "id": record["id"],
        "sciq_index": record["sciq_index"],
        "question": record["question"],
        "passage": record["passage"],
        "options": options,
        "answer_idx": answer_idx,
        "correct_answer": record["correct_answer"],
        # Predictions are reported as the option text, with the index kept alongside so
        # the entry can be joined back onto the raw results without re-matching strings.
        "pred_without_fact": options[predicted_idx_without_fact],
        "pred_with_fact": options[predicted_idx_with_fact],
        "pred_idx_without_fact": predicted_idx_without_fact,
        "pred_idx_with_fact": predicted_idx_with_fact,
        "is_correct_without_fact": predicted_idx_without_fact == answer_idx,
        "is_correct_with_fact": predicted_idx_with_fact == answer_idx,
        "category": category,
        "case": CATEGORY_CASE[category],
        "kda_score": record.get("kda_score"),
    }


def categorise_model(
    records: Sequence[Dict],
    model_name: str,
    model_names: Sequence[str],
) -> Dict[str, List[Dict]]:
    """Bucket every record for one model (or for the pooled ensemble)."""
    buckets: Dict[str, List[Dict]] = {name: [] for name in CATEGORIES}

    for record in records:
        if model_name == ENSEMBLE_KEY:
            predicted_without = pooled_prediction(
                record, model_names, "probabilities_without_fact"
            )
            predicted_with = pooled_prediction(
                record, model_names, "probabilities_with_fact"
            )
        else:
            stats = record["per_model"][model_name]
            predicted_without = stats["predicted_idx_without_fact"]
            predicted_with = stats["predicted_idx_with_fact"]

        answer_idx = record["answer_idx"]
        category = classify(predicted_without == answer_idx, predicted_with == answer_idx)
        buckets[category].append(
            build_entry(record, predicted_without, predicted_with, category)
        )

    return buckets


def summarise(buckets: Dict[str, List[Dict]], n_total: int) -> Dict:
    """Counts, percentages and the id list for each bucket of one model."""
    per_category = {}
    for category in CATEGORIES:
        entries = buckets[category]
        per_category[category] = {
            "case": CATEGORY_CASE[category],
            "description": CATEGORY_DESCRIPTION[category],
            "count": len(entries),
            "percentage": round(100.0 * len(entries) / n_total, 4) if n_total else 0.0,
            "ids": [entry["id"] for entry in entries],
            "sciq_indices": [entry["sciq_index"] for entry in entries],
        }

    n_correct_without = len(buckets["both_correct"]) + len(buckets["correct_to_wrong"])
    n_correct_with = len(buckets["both_correct"]) + len(buckets["wrong_to_correct"])
    return {
        "n_samples": n_total,
        "categories": per_category,
        "accuracy_without_fact": n_correct_without / n_total if n_total else 0.0,
        "accuracy_with_fact": n_correct_with / n_total if n_total else 0.0,
        "accuracy_gain": (n_correct_with - n_correct_without) / n_total if n_total else 0.0,
        # Share of the with-fact correct answers that the model could already produce
        # without the fact: the ceiling on how much of Acc_wf may be prior-driven.
        "prior_suspect_share_of_correct_with_fact": (
            len(buckets["both_correct"]) / n_correct_with if n_correct_with else 0.0
        ),
    }


def log_distribution_table(summaries: Dict[str, Dict]) -> None:
    """Console/log table: one row per model, one column per bucket."""
    logger.info("")
    logger.info("=" * 108)
    logger.info("CATEGORY DISTRIBUTION (count and %% of all evaluated samples)")
    logger.info("=" * 108)
    header = (
        f"  {'model':<40}"
        f"{'both_correct':>16}{'wrong_to_correct':>18}"
        f"{'both_wrong':>14}{'correct_to_wrong':>18}"
    )
    logger.info(header)
    logger.info("  " + "-" * 104)
    for model_name, summary in summaries.items():
        cells = "".join(
            f"{summary['categories'][category]['count']:>7} "
            f"({summary['categories'][category]['percentage']:5.1f}%)".rjust(width)
            for category, width in zip(CATEGORIES, (16, 18, 14, 18))
        )
        logger.info("  %-40s%s", model_name[:40], cells)
    logger.info("  " + "-" * 104)
    logger.info("")
    logger.info("Accuracy view (derived from the same buckets):")
    logger.info(
        "  %-40s%12s%12s%12s%14s",
        "model", "Acc_wof", "Acc_wf", "delta", "prior_share",
    )
    for model_name, summary in summaries.items():
        logger.info(
            "  %-40s%11.2f%%%11.2f%%%+11.2f%%%13.2f%%",
            model_name[:40],
            summary["accuracy_without_fact"] * 100,
            summary["accuracy_with_fact"] * 100,
            summary["accuracy_gain"] * 100,
            summary["prior_suspect_share_of_correct_with_fact"] * 100,
        )
    logger.info("")
    logger.info(
        "prior_share = both_correct / (both_correct + wrong_to_correct): the fraction of "
        "with-fact correct answers that Pillar 3 has to test for prior reliance."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Categorise baseline KDA results into both_correct / wrong_to_correct / "
            "both_wrong / correct_to_wrong buckets per model."
        )
    )
    parser.add_argument(
        "--results",
        default=None,
        help=(
            "Baseline results JSON. Defaults to the first of "
            f"{', '.join(RESULTS_CANDIDATES)} that exists in the workspace."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="results/ex1_category_questions/basic_category",
        help="Output directory for the categorised JSON files (created if missing).",
    )
    parser.add_argument(
        "--primary-model",
        default="Riiid/kda-mpnet-base-race",
        help="Model written to categorized_mpnet_base.json (the Pillar 3 primary model).",
    )
    parser.add_argument(
        "--log-file",
        default="results/ex1_category_questions/basic_category/categorize_kda_results.log",
        help="Execution log path.",
    )
    parser.add_argument("--verbose", action="store_true", help="Stream DEBUG to console.")
    args = parser.parse_args()

    log_path = ensure_parent(resolve(args.log_file, WORKSPACE))
    setup_logging(log_path, args.verbose)

    logger.info("=" * 108)
    logger.info("KDA result categorisation")
    logger.info("=" * 108)
    logger.info("Workspace   : %s", WORKSPACE)

    results_path = resolve_results_path(args.results)
    logger.info("Results file: %s", results_path)

    with open(results_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    # Both layouts are supported: {"summary": ..., "results": [...]} and a bare list.
    if isinstance(payload, dict):
        records = payload.get("results", [])
        source_summary = payload.get("summary", {})
    else:
        records = payload
        source_summary = {}
    if not records:
        logger.error("The results file contains no scored samples; nothing to categorise.")
        sys.exit(1)

    model_names: List[str] = list(source_summary.get("models") or records[0]["per_model"].keys())
    n_total = len(records)
    logger.info("Samples     : %d", n_total)
    logger.info("Models      : %d", len(model_names))
    for name in model_names:
        logger.info("  - %s", name)

    primary_model = args.primary_model
    if primary_model not in model_names:
        # The user-facing spelling 'Riid/...' and other near-misses are matched loosely
        # against the checkpoints actually present in the results file.
        matches = [n for n in model_names if slugify_model(n) == slugify_model(primary_model)]
        if matches:
            logger.warning(
                "Primary model '%s' not found verbatim; using '%s'", primary_model, matches[0]
            )
            primary_model = matches[0]
        else:
            logger.warning(
                "Primary model '%s' is absent from the results; falling back to '%s'",
                primary_model,
                model_names[0],
            )
            primary_model = model_names[0]
    logger.info("Primary     : %s", primary_model)

    out_dir = resolve(args.out_dir, WORKSPACE)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output dir  : %s", out_dir)

    # ---------------------------------------------------------------------------------
    # Bucket every model, then the pooled-probability ensemble.
    # ---------------------------------------------------------------------------------
    targets = list(model_names) + [ENSEMBLE_KEY]
    all_buckets: Dict[str, Dict[str, List[Dict]]] = {}
    summaries: Dict[str, Dict] = {}
    written_files: Dict[str, str] = {}

    for target in targets:
        logger.info("Categorising %s ...", target)
        buckets = categorise_model(records, target, model_names)
        all_buckets[target] = buckets
        summaries[target] = summarise(buckets, n_total)

        for category in CATEGORIES:
            logger.debug(
                "  %-16s %-8s n=%4d (%.2f%%)",
                category,
                CATEGORY_CASE[category],
                len(buckets[category]),
                summaries[target]["categories"][category]["percentage"],
            )

        filename = (
            "categorized_ensemble_pooled.json"
            if target == ENSEMBLE_KEY
            else f"categorized_{slugify_model(target)}.json"
        )
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": target,
                    "source_results_file": os.path.basename(results_path),
                    "n_samples": n_total,
                    "summary": summaries[target],
                    "categories": buckets,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        written_files[target] = out_path
        logger.info("  wrote %s", out_path)

    # The primary model gets a second copy under the name the study expects.
    primary_path = os.path.join(out_dir, "categorized_mpnet_base.json")
    with open(primary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": primary_model,
                "source_results_file": os.path.basename(results_path),
                "n_samples": n_total,
                "summary": summaries[primary_model],
                "categories": all_buckets[primary_model],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("  wrote %s (primary model: %s)", primary_path, primary_model)

    # ---------------------------------------------------------------------------------
    # Cross-model summary file.
    # ---------------------------------------------------------------------------------
    summary_path = os.path.join(out_dir, "categorized_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_results_file": os.path.basename(results_path),
                "dataset": source_summary.get("dataset", "allenai/sciq"),
                "split": source_summary.get("split", "test"),
                "n_samples": n_total,
                "models": model_names,
                "primary_model": primary_model,
                "category_definitions": {
                    category: {
                        "case": CATEGORY_CASE[category],
                        "description": CATEGORY_DESCRIPTION[category],
                    }
                    for category in CATEGORIES
                },
                "per_model": summaries,
                "output_files": {
                    target: os.path.basename(path) for target, path in written_files.items()
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("  wrote %s", summary_path)

    log_distribution_table(summaries)

    primary_summary = summaries[primary_model]
    logger.info("")
    logger.info("=" * 108)
    logger.info("PILLAR 3 TARGET SET (primary model %s)", primary_model)
    logger.info("=" * 108)
    both_correct = primary_summary["categories"]["both_correct"]
    logger.info(
        "both_correct (Case 2): %d samples (%.2f%%) need counterfactual testing to tell "
        "context dependence from prior reliance.",
        both_correct["count"],
        both_correct["percentage"],
    )
    logger.info(
        "First 25 ids: %s%s",
        both_correct["ids"][:25],
        " ..." if both_correct["count"] > 25 else "",
    )
    logger.info("")
    logger.info("Log file: %s", log_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
