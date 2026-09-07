"""Pillar 3: counterfactual context perturbation and counterfactual-adjusted KDA.

The baseline KDA run scores each question in two settings (no passage / gold passage).
That cannot distinguish a model that *reads* the target fact from one that already knew
the answer: both land in the `both_correct` bucket. Pillar 3 adds a third setting in
which the passage asserts a *distractor* instead of the gold answer, and asks what the
model does when its prior and the context disagree.

    Setting A  no passage            question + options
    Setting B  original passage      question + options + gold support passage
    Setting C  counterfactual passage question + options + perturbed support passage

For every `both_correct` sample (correct in A and in B) the Setting C prediction is
classified as:

    context_dependent  -> the prediction flips to the counterfactual target
                          (the model follows the context, its B-correctness is earned)
    prior_dependent    -> the prediction stays on the gold answer
                          (the model overrides the context with a parametric prior or an
                          option-surface shortcut; its B-correctness is NOT evidence of
                          knowledge dependence)
    unstable_other     -> the prediction moves to a third option (neither gold nor the
                          counterfactual target): the perturbation broke the model
                          without steering it

Counterfactual-adjusted KDA then discounts, per model, the with-fact probability mass of
every question the model answers from prior rather than from context. See
`compute_adjusted_kda` for the exact formulas.

Design notes
------------
* Models are swept one at a time (`sequential`): the KDA_small suite does not fit in 4GB
  of VRAM together. Each model runs all three settings over the whole dataset, then its
  weights are released.
* Scoring reuses `kda_tiny.make_student`, so Setting A/B numbers are produced by exactly
  the same code path as the baseline run and are directly comparable.
* Nothing in this script writes to an existing raw data or results file; all output goes
  to `--out` (default results/ex2_counterfactual/results_counterfactual_sciq_test_full.json) and `--log-file`.

Usage:
    python code/ex2_counterfactual/run_counterfactual_experiment.py                       # full KDA_small suite
    python code/ex2_counterfactual/run_counterfactual_experiment.py --limit 20 --models Riiid/kda-mpnet-base-race \
        Riiid/kda-scibert-uncased-race                            # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------------------
# Cross-stage imports. This script lives in `code/<stage>/`, so `code/` itself is put on
# `sys.path`; `utils.paths` and `ex1_reproduce_KDA.kda_tiny` then resolve no matter which
# directory the script is launched from. The *project root* is deliberately NOT added --
# it contains a `datasets/` folder that would shadow the HuggingFace `datasets` package.
# --------------------------------------------------------------------------------------
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex1_reproduce_KDA.kda_tiny import DEFAULT_MODELS, KDA_SMALL, MODEL_PRESETS, make_student
from ex2_counterfactual.counterfactual_passage import TIERS, build_counterfactual, tier_at_least
from utils.paths import PROJECT_ROOT, ensure_parent
from utils.paths import resolve as resolve_path

# The scripts live in code/<stage>/, but every data and result path is relative to the
# project root two levels up.
WORKSPACE = PROJECT_ROOT

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries log every HTTP request at INFO, which would bury the run's own
# records in the console stream and the log file.
NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "filelock",
    "huggingface_hub", "transformers", "datasets", "fsspec",
)

DENOMINATOR_EPSILON = 1e-12

# Baseline buckets, recomputed here from settings A and B (same definitions as
# categorize_kda_results.py, kept in sync deliberately rather than imported so this
# script can run standalone).
CATEGORIES = ("both_correct", "wrong_to_correct", "both_wrong", "correct_to_wrong")
CF_CLASSES = ("context_dependent", "prior_dependent", "unstable_other")

ENSEMBLE_KEY = "ENSEMBLE_pooled_probability_vote"

logger = logging.getLogger("counterfactual_experiment")


# ======================================================================================
# Logging / small helpers
# ======================================================================================
def setup_logging(log_path: str, verbose: bool, append: bool) -> None:
    """DEBUG to the log file, INFO (or DEBUG with --verbose) to stdout."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def resolve(path: str) -> str:
    """Make a user-supplied path absolute relative to the project root."""
    return resolve_path(path, WORKSPACE)


def shorten(text: str, limit: int = 150) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def classify_baseline(correct_a: bool, correct_b: bool) -> str:
    """Bucket one (Setting A, Setting B) correctness pair."""
    if correct_a:
        return "both_correct" if correct_b else "correct_to_wrong"
    return "wrong_to_correct" if correct_b else "both_wrong"


def classify_counterfactual(
    predicted_idx_c: int, answer_idx: int, counterfactual_target_idx: int
) -> str:
    """Label a Setting C prediction as context- or prior-driven."""
    if predicted_idx_c == counterfactual_target_idx:
        return "context_dependent"
    if predicted_idx_c == answer_idx:
        return "prior_dependent"
    return "unstable_other"


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


# ======================================================================================
# Scoring: three settings, one model resident at a time
# ======================================================================================
def score_three_settings(
    samples: Sequence[Dict],
    model_names: Sequence[str],
    device: str,
    progress_every: int,
) -> Dict[str, List[Dict[str, List[float]]]]:
    """Run every model over every sample in Settings A, B and C.

    Returns {model_name: [{"A": probs, "B": probs, "C": probs}, ...]} aligned with
    `samples`. Only one model is resident in memory at any time.
    """
    cache: Dict[str, List[Dict[str, List[float]]]] = {}
    started = time.time()
    n_models = len(model_names)

    for model_position, model_name in enumerate(model_names, start=1):
        logger.info("=" * 78)
        logger.info("Sequential pass %d/%d: %s", model_position, n_models, model_name)
        student = make_student(model_name, device, logging.getLogger("kda_tiny"))

        pass_started = time.time()
        per_sample: List[Dict[str, List[float]]] = []
        for position, sample in enumerate(samples, start=1):
            question = sample["question"]
            options = sample["options"]

            # Each passage variant is truncated independently: the substitution can
            # change the token count, so reusing one budget would misalign the settings.
            original = student.truncate_passage(sample["passage"], question, options)
            counterfactual = student.truncate_passage(
                sample["counterfactual_passage"], question, options
            )
            prefix_b = (original.strip() + " ") if original.strip() else ""
            prefix_c = (counterfactual.strip() + " ") if counterfactual.strip() else ""

            per_sample.append(
                {
                    "A": student.choice_probabilities("", question, options),
                    "B": student.choice_probabilities(prefix_b, question, options),
                    "C": student.choice_probabilities(prefix_c, question, options),
                }
            )

            if position % progress_every == 0 or position == len(samples):
                fraction = ((model_position - 1) * len(samples) + position) / (
                    n_models * len(samples)
                )
                elapsed = time.time() - started
                eta = elapsed / fraction - elapsed if fraction > 0 else 0.0
                filled = int(round(fraction * 24))
                logger.info(
                    "[%s] %5.1f%%  model %d/%d (%s)  %d/%d samples  elapsed=%.1fs  ETA=%.1fs",
                    "#" * filled + "-" * (24 - filled),
                    fraction * 100,
                    model_position,
                    n_models,
                    model_name.split("/")[-1],
                    position,
                    len(samples),
                    elapsed,
                    eta,
                )

        elapsed = time.time() - pass_started
        logger.info(
            "Pass %d/%d done: %d samples x 3 settings in %.2fs (%.4fs per sample)",
            model_position, n_models, len(samples), elapsed, elapsed / max(1, len(samples)),
        )
        cache[model_name] = per_sample

        student.unload()
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded '%s' and cleared the CUDA cache", model_name)

    return cache


# ======================================================================================
# Per-sample aggregation
# ======================================================================================
def build_record(
    sample: Dict,
    probabilities: Dict[str, Dict[str, List[float]]],
    model_names: Sequence[str],
) -> Dict:
    """Assemble one sample's full three-setting record, per model and pooled."""
    answer_idx = sample["answer_idx"]
    options = sample["options"]
    cf_idx = sample["counterfactual_target_idx"]
    cf_valid = sample["counterfactual_valid"]

    per_model: Dict[str, Dict] = {}
    numerator = 0.0
    denominator = 0.0
    numerator_hard = 0.0
    numerator_soft = 0.0

    for model_name in model_names:
        probs = probabilities[model_name]
        probs_a, probs_b, probs_c = probs["A"], probs["B"], probs["C"]

        p_correct_a = probs_a[answer_idx]   # P_m(R^q = 1)
        p_correct_b = probs_b[answer_idx]   # P_m(R^{q+f} = 1)
        p_correct_c = probs_c[answer_idx]
        weight = 1.0 - p_correct_a          # P_m(R^q = 0)

        predicted_a, predicted_b, predicted_c = argmax(probs_a), argmax(probs_b), argmax(probs_c)
        correct_a = predicted_a == answer_idx
        correct_b = predicted_b == answer_idx
        baseline_category = classify_baseline(correct_a, correct_b)

        # Probability the model assigns to the counterfactual target under Setting C.
        # This is the soft measure of "did the model follow the perturbed context".
        p_cf_target_c = probs_c[cf_idx] if cf_valid else None
        cf_class = (
            classify_counterfactual(predicted_c, answer_idx, cf_idx) if cf_valid else None
        )

        # Gates used by the adjusted KDA (0 when the counterfactual is unusable, so the
        # sample contributes nothing but its weight -- see compute_adjusted_kda).
        gate_hard = 1.0 if (cf_valid and predicted_c == cf_idx) else 0.0
        gate_soft = float(p_cf_target_c) if cf_valid else 0.0

        numerator += weight * p_correct_b
        numerator_hard += weight * p_correct_b * gate_hard
        numerator_soft += weight * p_correct_b * gate_soft
        denominator += weight

        per_model[model_name] = {
            "p_correct_no_passage": p_correct_a,
            "p_correct_original_passage": p_correct_b,
            "p_correct_counterfactual_passage": p_correct_c,
            "p_counterfactual_target_counterfactual_passage": p_cf_target_c,
            "weight": weight,
            "predicted_idx_no_passage": predicted_a,
            "predicted_idx_original_passage": predicted_b,
            "predicted_idx_counterfactual_passage": predicted_c,
            "pred_no_passage": options[predicted_a],
            "pred_original_passage": options[predicted_b],
            "pred_counterfactual_passage": options[predicted_c],
            "is_correct_no_passage": correct_a,
            "is_correct_original_passage": correct_b,
            "is_correct_counterfactual_passage": predicted_c == answer_idx,
            "follows_counterfactual": bool(gate_hard),
            "baseline_category": baseline_category,
            "counterfactual_class": cf_class,
            "probabilities_no_passage": probs_a,
            "probabilities_original_passage": probs_b,
            "probabilities_counterfactual_passage": probs_c,
        }

    # Pooled-probability ensemble: average the distributions, then take the argmax.
    n_models = len(model_names)
    n_options = len(options)
    pooled = {
        setting: [
            sum(probabilities[name][setting][i] for name in model_names) / n_models
            for i in range(n_options)
        ]
        for setting in ("A", "B", "C")
    }
    pooled_a, pooled_b, pooled_c = argmax(pooled["A"]), argmax(pooled["B"]), argmax(pooled["C"])
    ensemble = {
        "predicted_idx_no_passage": pooled_a,
        "predicted_idx_original_passage": pooled_b,
        "predicted_idx_counterfactual_passage": pooled_c,
        "is_correct_no_passage": pooled_a == answer_idx,
        "is_correct_original_passage": pooled_b == answer_idx,
        "is_correct_counterfactual_passage": pooled_c == answer_idx,
        "follows_counterfactual": bool(cf_valid and pooled_c == cf_idx),
        "baseline_category": classify_baseline(pooled_a == answer_idx, pooled_b == answer_idx),
        "counterfactual_class": (
            classify_counterfactual(pooled_c, answer_idx, cf_idx) if cf_valid else None
        ),
        "p_correct_no_passage": pooled["A"][answer_idx],
        "p_correct_original_passage": pooled["B"][answer_idx],
        "p_correct_counterfactual_passage": pooled["C"][answer_idx],
        "p_counterfactual_target_counterfactual_passage": pooled["C"][cf_idx] if cf_valid else None,
    }

    degenerate = denominator <= DENOMINATOR_EPSILON
    if degenerate:
        logger.warning(
            "Denominator sum(1 - P_m(R^q=1)) is ~0 for id=%s; KDA forced to 0.",
            sample["id"],
        )

    def ratio(value: float) -> float:
        return 0.0 if degenerate else value / denominator

    return {
        "id": sample["id"],
        "sciq_index": sample.get("sciq_index"),
        "obqa_index": sample.get("obqa_index"),
        "source_index": sample.get(
            "sciq_index", sample.get("obqa_index", sample["id"])
        ),
        "question": sample["question"],
        "passage": sample["passage"],
        "counterfactual_passage": sample["counterfactual_passage"],
        "options": options,
        "answer_idx": answer_idx,
        "correct_answer": sample["correct_answer"],
        "counterfactual_target": sample["counterfactual_target"],
        "counterfactual_target_idx": cf_idx,
        "counterfactual_valid": cf_valid,
        "substitution_tier": sample["substitution_tier"],
        "substitution_matched_text": sample["substitution_matched_text"],
        "n_substitutions": sample["n_substitutions"],
        "residual_answer_mentions": sample["residual_answer_mentions"],
        "residual_glued_mentions": sample["residual_glued_mentions"],
        "kda_original": ratio(numerator),
        "kda_adjusted_hard": ratio(numerator_hard),
        "kda_adjusted_soft": ratio(numerator_soft),
        "kda_numerator": numerator,
        "kda_denominator": denominator,
        "zero_denominator": degenerate,
        "per_model": per_model,
        "ensemble": ensemble,
    }


# ======================================================================================
# Metrics
# ======================================================================================
def _view(record: Dict, target: str) -> Dict:
    """Per-model or pooled-ensemble view of one record."""
    return record["ensemble"] if target == ENSEMBLE_KEY else record["per_model"][target]


def compute_model_metrics(
    records: Sequence[Dict], target: str, eligible: Sequence[Dict]
) -> Dict:
    """Accuracy, bucket distribution and counterfactual behaviour for one model.

    `records`  : every scored sample (accuracy in A/B is reported over all of them).
    `eligible` : the subset with a usable counterfactual, over which Setting C and every
                 counterfactual-derived rate is computed.
    """
    n = len(records)
    n_eligible = len(eligible)

    accuracy = {
        "no_passage": sum(_view(r, target)["is_correct_no_passage"] for r in records) / n,
        "original_passage": sum(
            _view(r, target)["is_correct_original_passage"] for r in records
        ) / n,
        # Accuracy against the GOLD answer under the perturbed passage: a *low* value is
        # the desirable outcome here, it means the model followed the counterfactual.
        "counterfactual_passage_vs_gold": (
            sum(_view(r, target)["is_correct_counterfactual_passage"] for r in eligible)
            / n_eligible if n_eligible else 0.0
        ),
        # Accuracy against the counterfactual target: the context-following rate.
        "counterfactual_passage_vs_cf_target": (
            sum(_view(r, target)["follows_counterfactual"] for r in eligible)
            / n_eligible if n_eligible else 0.0
        ),
    }
    accuracy["gain_original_over_no_passage"] = (
        accuracy["original_passage"] - accuracy["no_passage"]
    )

    buckets = {category: [] for category in CATEGORIES}
    for record in records:
        buckets[_view(record, target)["baseline_category"]].append(record["id"])

    # Counterfactual classification, restricted to the Case 2 (both_correct) samples the
    # study targets, and also reported over every eligible sample for context.
    case2 = [r for r in eligible if _view(r, target)["baseline_category"] == "both_correct"]
    case2_classes = {name: [] for name in CF_CLASSES}
    for record in case2:
        case2_classes[_view(record, target)["counterfactual_class"]].append(record["id"])

    all_classes = {name: 0 for name in CF_CLASSES}
    for record in eligible:
        all_classes[_view(record, target)["counterfactual_class"]] += 1

    n_case2 = len(case2)
    n_context = len(case2_classes["context_dependent"])
    n_prior = len(case2_classes["prior_dependent"])

    # wrong_to_correct samples that are also eligible: the fact demonstrably moved the
    # prediction, so they count as knowledge-dependent without needing Setting C.
    n_w2c_eligible = sum(
        1 for r in eligible if _view(r, target)["baseline_category"] == "wrong_to_correct"
    )
    n_correct_with_fact_eligible = n_case2 + n_w2c_eligible

    return {
        "n_samples": n,
        "n_counterfactual_eligible": n_eligible,
        "accuracy": accuracy,
        "baseline_categories": {
            category: {
                "count": len(ids),
                "percentage": round(100.0 * len(ids) / n, 4),
                "ids": ids,
            }
            for category, ids in buckets.items()
        },
        "counterfactual_classes_all_eligible": {
            name: {
                "count": count,
                "percentage": round(100.0 * count / n_eligible, 4) if n_eligible else 0.0,
            }
            for name, count in all_classes.items()
        },
        "both_correct_analysis": {
            "n_both_correct_eligible": n_case2,
            "context_dependent": {
                "count": n_context,
                "percentage": round(100.0 * n_context / n_case2, 4) if n_case2 else 0.0,
                "ids": case2_classes["context_dependent"],
            },
            "prior_dependent": {
                "count": n_prior,
                "percentage": round(100.0 * n_prior / n_case2, 4) if n_case2 else 0.0,
                "ids": case2_classes["prior_dependent"],
            },
            "unstable_other": {
                "count": len(case2_classes["unstable_other"]),
                "percentage": (
                    round(100.0 * len(case2_classes["unstable_other"]) / n_case2, 4)
                    if n_case2 else 0.0
                ),
                "ids": case2_classes["unstable_other"],
            },
        },
        # Accuracy_with_fact stripped of the correct answers that Setting C shows to be
        # prior-driven: (wrong_to_correct + context-verified both_correct) / n_eligible.
        "context_verified_accuracy_with_fact": (
            (n_w2c_eligible + n_context) / n_eligible if n_eligible else 0.0
        ),
        "raw_accuracy_with_fact_on_eligible": (
            n_correct_with_fact_eligible / n_eligible if n_eligible else 0.0
        ),
        "prior_inflation_of_accuracy_with_fact": (
            n_prior / n_eligible if n_eligible else 0.0
        ),
    }


def compute_adjusted_kda(records: Sequence[Dict], eligible: Sequence[Dict]) -> Dict:
    """Counterfactual-adjusted KDA, in three complementary formulations.

    Baseline (unchanged, recomputed from Settings A and B):

        KDA_cont(q) = sum_m w_m(q) * P_m^B(correct) / sum_m w_m(q),
        w_m(q)      = 1 - P_m^A(correct)

    (1) hard-gated -- per model, the with-fact mass only counts when that model actually
        follows the perturbed context on that question:

        KDA_adj_hard(q) = sum_m w_m * P_m^B(correct) * 1[argmax P_m^C == cf_target]
                          / sum_m w_m

    (2) soft-gated -- the indicator is replaced by the probability mass the model moves
        onto the counterfactual target, which avoids a hard threshold:

        KDA_adj_soft(q) = sum_m w_m * P_m^B(correct) * P_m^C(cf_target) / sum_m w_m

    (3) sample-level exclusion -- the formulation stated in the study plan: questions the
        ensemble answers from prior (both_correct AND Setting C keeps the gold answer)
        are dropped from the knowledge-dependency count entirely:

        KDA_adj_excl = mean over eligible q of KDA_cont(q) * 1[q is not prior-dependent]

    All three are reported over the counterfactual-eligible subset; the unadjusted mean is
    reported over both the eligible subset and the full set so the two are comparable.
    """
    all_scores = [r["kda_original"] for r in records]
    eligible_scores = [r["kda_original"] for r in eligible]
    hard = [r["kda_adjusted_hard"] for r in eligible]
    soft = [r["kda_adjusted_soft"] for r in eligible]

    kept, dropped = [], []
    for record in eligible:
        ensemble = record["ensemble"]
        is_prior = (
            ensemble["baseline_category"] == "both_correct"
            and ensemble["counterfactual_class"] == "prior_dependent"
        )
        (dropped if is_prior else kept).append(record)

    # Dropped questions contribute 0, so the mean stays over the same denominator and
    # the adjusted score is directly comparable with the unadjusted one.
    kept_ids = {r["id"] for r in kept}
    excluded_mean = statistics.fmean(
        [r["kda_original"] if r["id"] in kept_ids else 0.0 for r in eligible]
    ) if eligible else 0.0

    def stats(values: Sequence[float]) -> Dict:
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    mean_original = statistics.fmean(eligible_scores) if eligible_scores else 0.0
    return {
        "kda_original_all_samples": stats(all_scores),
        "kda_original_eligible": stats(eligible_scores),
        "kda_adjusted_hard": stats(hard),
        "kda_adjusted_soft": stats(soft),
        "kda_adjusted_sample_exclusion": {
            "mean": excluded_mean,
            "n_eligible": len(eligible),
            "n_kept": len(kept),
            "n_dropped_prior_dependent": len(dropped),
            "dropped_ids": [r["id"] for r in dropped],
        },
        "retention_ratio_hard": (
            statistics.fmean(hard) / mean_original if mean_original else 0.0
        ),
        "retention_ratio_soft": (
            statistics.fmean(soft) / mean_original if mean_original else 0.0
        ),
        "retention_ratio_sample_exclusion": (
            excluded_mean / mean_original if mean_original else 0.0
        ),
    }


# ======================================================================================
# Console / log reporting
# ======================================================================================
def log_tables(
    per_target_metrics: Dict[str, Dict],
    adjusted: Dict,
    generation_stats: Dict,
    primary_model: str,
) -> None:
    """Emit the tabular summary to the console and the log file."""
    logger.info("")
    logger.info("=" * 100)
    logger.info("COUNTERFACTUAL PASSAGE GENERATION")
    logger.info("=" * 100)
    logger.info(
        "  %-24s %8s %10s", "substitution tier", "count", "share",
    )
    for tier in list(TIERS) + ["none"]:
        count = generation_stats["tiers"].get(tier, 0)
        logger.info(
            "  %-24s %8d %9.2f%%",
            tier, count, 100.0 * count / generation_stats["n_samples"],
        )
    logger.info(
        "  %-24s %8d %9.2f%%",
        "ELIGIBLE (usable CF)",
        generation_stats["n_eligible"],
        100.0 * generation_stats["n_eligible"] / generation_stats["n_samples"],
    )
    logger.info(
        "  mean substitutions per eligible passage : %.2f",
        generation_stats["mean_substitutions"],
    )
    logger.info(
        "  passages with the answer still fused inside a longer word : %d "
        "(left intact on purpose)",
        generation_stats["n_with_glued_residual"],
    )

    logger.info("")
    logger.info("=" * 100)
    logger.info("ACCURACY UNDER THE THREE SETTINGS")
    logger.info("=" * 100)
    logger.info(
        "  %-42s %9s %9s %9s %9s",
        "model", "A:none", "B:orig", "C:gold", "C:cf-tgt",
    )
    logger.info("  " + "-" * 96)
    for target, metrics in per_target_metrics.items():
        accuracy = metrics["accuracy"]
        logger.info(
            "  %-42s %8.2f%% %8.2f%% %8.2f%% %8.2f%%",
            target[:42],
            accuracy["no_passage"] * 100,
            accuracy["original_passage"] * 100,
            accuracy["counterfactual_passage_vs_gold"] * 100,
            accuracy["counterfactual_passage_vs_cf_target"] * 100,
        )
    logger.info("  " + "-" * 96)
    logger.info(
        "  A/B over all samples; C over the %d counterfactual-eligible samples.",
        next(iter(per_target_metrics.values()))["n_counterfactual_eligible"],
    )
    logger.info(
        "  'C:gold' = still answers the gold option despite the perturbed context "
        "(lower is better); 'C:cf-tgt' = follows the perturbed context (higher is better)."
    )

    logger.info("")
    logger.info("=" * 100)
    logger.info("CONTEXT DEPENDENCE vs PRIOR RELIANCE on the both_correct (Case 2) bucket")
    logger.info("=" * 100)
    logger.info(
        "  %-42s %8s %12s %12s %12s",
        "model", "n_case2", "context-dep", "prior-dep", "unstable",
    )
    logger.info("  " + "-" * 96)
    for target, metrics in per_target_metrics.items():
        analysis = metrics["both_correct_analysis"]
        logger.info(
            "  %-42s %8d %5d(%5.1f%%) %5d(%5.1f%%) %5d(%5.1f%%)",
            target[:42],
            analysis["n_both_correct_eligible"],
            analysis["context_dependent"]["count"],
            analysis["context_dependent"]["percentage"],
            analysis["prior_dependent"]["count"],
            analysis["prior_dependent"]["percentage"],
            analysis["unstable_other"]["count"],
            analysis["unstable_other"]["percentage"],
        )
    logger.info("  " + "-" * 96)

    logger.info("")
    logger.info("=" * 100)
    logger.info("ACCURACY_WITH_FACT BEFORE AND AFTER REMOVING PRIOR-DRIVEN CORRECT ANSWERS")
    logger.info("=" * 100)
    logger.info(
        "  %-42s %12s %12s %12s",
        "model", "Acc_wf raw", "Acc_wf ctx", "inflation",
    )
    logger.info("  " + "-" * 96)
    for target, metrics in per_target_metrics.items():
        logger.info(
            "  %-42s %11.2f%% %11.2f%% %11.2f%%",
            target[:42],
            metrics["raw_accuracy_with_fact_on_eligible"] * 100,
            metrics["context_verified_accuracy_with_fact"] * 100,
            metrics["prior_inflation_of_accuracy_with_fact"] * 100,
        )
    logger.info("  " + "-" * 96)
    logger.info(
        "  Acc_wf ctx keeps a with-fact correct answer only when the fact demonstrably "
        "drove it (wrong->correct, or both_correct that flips under the counterfactual)."
    )

    logger.info("")
    logger.info("=" * 100)
    logger.info("COUNTERFACTUAL-ADJUSTED KDA (ensemble)")
    logger.info("=" * 100)
    logger.info(
        "  %-38s %9s %9s %9s %9s %7s",
        "metric", "mean", "median", "std", "min/max", "n",
    )
    logger.info("  " + "-" * 96)
    rows = (
        ("KDA_cont (all samples)", adjusted["kda_original_all_samples"]),
        ("KDA_cont (CF-eligible)", adjusted["kda_original_eligible"]),
        ("KDA_adj hard-gated", adjusted["kda_adjusted_hard"]),
        ("KDA_adj soft-gated", adjusted["kda_adjusted_soft"]),
    )
    for label, values in rows:
        logger.info(
            "  %-38s %9.4f %9.4f %9.4f %4.2f/%4.2f %7d",
            label,
            values["mean"], values["median"], values["std"],
            values["min"], values["max"], values["n"],
        )
    exclusion = adjusted["kda_adjusted_sample_exclusion"]
    logger.info(
        "  %-38s %9.4f %9s %9s %9s %7d",
        "KDA_adj sample-exclusion",
        exclusion["mean"], "-", "-", "-", exclusion["n_eligible"],
    )
    logger.info("  " + "-" * 96)
    logger.info(
        "  retention vs unadjusted : hard %.1f%% | soft %.1f%% | sample-exclusion %.1f%%",
        adjusted["retention_ratio_hard"] * 100,
        adjusted["retention_ratio_soft"] * 100,
        adjusted["retention_ratio_sample_exclusion"] * 100,
    )
    logger.info(
        "  %d of %d eligible questions dropped as ensemble prior-dependent.",
        exclusion["n_dropped_prior_dependent"], exclusion["n_eligible"],
    )

    primary = per_target_metrics.get(primary_model)
    if primary:
        analysis = primary["both_correct_analysis"]
        logger.info("")
        logger.info("=" * 100)
        logger.info("HEADLINE (primary model: %s)", primary_model)
        logger.info("=" * 100)
        logger.info(
            "  Acc: %.2f%% (no passage) -> %.2f%% (original passage) -> %.2f%% (counterfactual, vs gold)",
            primary["accuracy"]["no_passage"] * 100,
            primary["accuracy"]["original_passage"] * 100,
            primary["accuracy"]["counterfactual_passage_vs_gold"] * 100,
        )
        logger.info(
            "  Of %d both_correct samples: %d (%.1f%%) are context-dependent, "
            "%d (%.1f%%) are prior-dependent, %d (%.1f%%) unstable.",
            analysis["n_both_correct_eligible"],
            analysis["context_dependent"]["count"],
            analysis["context_dependent"]["percentage"],
            analysis["prior_dependent"]["count"],
            analysis["prior_dependent"]["percentage"],
            analysis["unstable_other"]["count"],
            analysis["unstable_other"]["percentage"],
        )
        logger.info(
            "  Accuracy_with_fact drops from %.2f%% to %.2f%% once prior-driven correct "
            "answers are removed.",
            primary["raw_accuracy_with_fact_on_eligible"] * 100,
            primary["context_verified_accuracy_with_fact"] * 100,
        )


def log_examples(records: Sequence[Dict], primary_model: str, n_each: int = 3) -> None:
    """Show a few concrete context-dependent and prior-dependent cases."""
    for wanted in ("prior_dependent", "context_dependent"):
        chosen = [
            r for r in records
            if r["counterfactual_valid"]
            and r["per_model"].get(primary_model, {}).get("baseline_category") == "both_correct"
            and r["per_model"][primary_model]["counterfactual_class"] == wanted
        ][:n_each]
        if not chosen:
            continue
        logger.info("")
        logger.info("-" * 100)
        logger.info("EXAMPLES: %s (%s)", wanted.upper(), primary_model)
        logger.info("-" * 100)
        for record in chosen:
            stats = record["per_model"][primary_model]
            logger.info("  id=%s (source #%s)", record["id"], record["source_index"])
            logger.info("    Q         : %s", shorten(record["question"]))
            logger.info(
                "    options   : %s | gold='%s' | cf target='%s'",
                record["options"], record["correct_answer"], record["counterfactual_target"],
            )
            logger.info("    passage   : %s", shorten(record["passage"]))
            logger.info("    cf passage: %s", shorten(record["counterfactual_passage"]))
            logger.info(
                "    preds     : A='%s' B='%s' C='%s' | P_C(gold)=%.3f P_C(cf)=%.3f",
                stats["pred_no_passage"],
                stats["pred_original_passage"],
                stats["pred_counterfactual_passage"],
                stats["p_correct_counterfactual_passage"],
                stats["p_counterfactual_target_counterfactual_passage"],
            )


# ======================================================================================
# Entry point
# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pillar 3: counterfactual context perturbation and counterfactual-adjusted "
            "KDA on SciQ."
        )
    )
    parser.add_argument(
        "--data",
        default="datasets/sciq/sciq_test_full.json",
        help="Formatted SciQ samples (relative to the project root).",
    )
    parser.add_argument(
        "--out",
        default="results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
        help="Structured output path (never an existing raw data file).",
    )
    parser.add_argument(
        "--log-file", default="results/ex2_counterfactual/counterfactual_sciq_test_full.log", help="Execution log path."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["KDA_SMALL"],
        help=(
            "Ensemble members, or a preset name (KDA_SMALL / DEFAULT). At least two are "
            "required for a non-degenerate ensemble KDA_cont."
        ),
    )
    parser.add_argument(
        "--primary-model",
        default="Riiid/kda-mpnet-base-race",
        help="Model used for the headline numbers and the worked examples.",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected if unset).")
    parser.add_argument(
        "--limit", type=int, default=0, help="Score only the first N samples (0 = all)."
    )
    parser.add_argument(
        "--min-substitution-tier",
        choices=TIERS,
        default="partial",
        help=(
            "Strictest tier accepted as counterfactual-eligible. 'exact' keeps only "
            "verbatim substitutions; the default 'partial' keeps every tier that matched."
        ),
    )
    parser.add_argument(
        "--progress-every", type=int, default=0,
        help="Progress-bar cadence in samples. 0 selects 5%% of the dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate and report the counterfactual passages; load no models.",
    )
    parser.add_argument("--verbose", action="store_true", help="Stream DEBUG to console.")
    parser.add_argument("--append-log", action="store_true", help="Append to the log file.")
    args = parser.parse_args()

    # Expand preset names (e.g. --models KDA_SMALL) into explicit checkpoint lists.
    model_names: List[str] = []
    for entry in args.models:
        preset = MODEL_PRESETS.get(entry.upper())
        model_names.extend(preset if preset else [entry])

    log_path = ensure_parent(resolve(args.log_file))
    setup_logging(log_path, args.verbose, args.append_log)
    run_started = time.time()

    logger.info("=" * 100)
    logger.info("PILLAR 3 -- COUNTERFACTUAL CONTEXT PERTURBATION ON SciQ")
    logger.info("=" * 100)
    logger.info("Workspace     : %s", WORKSPACE)
    logger.info("Python        : %s", platform.python_version())
    logger.info(
        "torch         : %s (CUDA available: %s)", torch.__version__, torch.cuda.is_available()
    )
    logger.info("Ensemble |M|  : %d", len(model_names))
    for name in model_names:
        logger.info("  - %s", name)
    logger.info("Primary model : %s", args.primary_model)

    data_path = resolve(args.data)
    out_path = ensure_parent(resolve(args.out))
    if os.path.abspath(data_path) == os.path.abspath(out_path):
        logger.error("Refusing to run: --out would overwrite the input dataset %s", data_path)
        sys.exit(1)
    logger.info("Data file     : %s", data_path)
    logger.info("Results file  : %s", out_path)
    logger.info("Log file      : %s", log_path)

    with open(data_path, encoding="utf-8") as handle:
        raw_samples = json.load(handle)
    if args.limit:
        raw_samples = raw_samples[: args.limit]
        logger.info("--limit active: scoring the first %d samples only", len(raw_samples))
    logger.info("Loaded %d questions", len(raw_samples))

    # ---------------------------------------------------------------------------------
    # Step 1: counterfactual passage generation (pure text, no model needed).
    # ---------------------------------------------------------------------------------
    logger.info("")
    logger.info("Generating counterfactual passages ...")
    samples: List[Dict] = []
    tier_counts: Dict[str, int] = {tier: 0 for tier in list(TIERS) + ["none"]}
    for raw in raw_samples:
        counterfactual = build_counterfactual(raw)
        # A tier that matched but is looser than --min-substitution-tier is downgraded to
        # ineligible: the perturbation still exists, it just does not enter the metrics.
        if counterfactual["counterfactual_valid"] and not tier_at_least(
            counterfactual["substitution_tier"], args.min_substitution_tier
        ):
            counterfactual["counterfactual_valid"] = False
        tier_counts[counterfactual["substitution_tier"]] += 1
        merged = {**raw, **counterfactual}
        # A sample without a target still needs an index for the probability lookups;
        # it is never counted because counterfactual_valid is False.
        if merged["counterfactual_target_idx"] is None:
            merged["counterfactual_target_idx"] = (
                (merged["answer_idx"] + 1) % len(merged["options"])
            )
        samples.append(merged)
        logger.debug(
            "id=%s tier=%s matched=%r gold='%s' -> cf='%s' (n=%d)",
            merged["id"], merged["substitution_tier"], merged["substitution_matched_text"],
            merged["correct_answer"], merged["counterfactual_target"], merged["n_substitutions"],
        )

    eligible_samples = [s for s in samples if s["counterfactual_valid"]]
    generation_stats = {
        "n_samples": len(samples),
        "n_eligible": len(eligible_samples),
        "tiers": tier_counts,
        "min_substitution_tier": args.min_substitution_tier,
        "mean_substitutions": (
            statistics.fmean([s["n_substitutions"] for s in eligible_samples])
            if eligible_samples else 0.0
        ),
        "n_with_glued_residual": sum(
            1 for s in eligible_samples if s["residual_glued_mentions"]
        ),
        "ineligible_ids": [s["id"] for s in samples if not s["counterfactual_valid"]],
    }
    logger.info(
        "Generated %d usable counterfactual passages out of %d samples (tiers: %s)",
        generation_stats["n_eligible"], generation_stats["n_samples"], tier_counts,
    )
    if not eligible_samples:
        logger.error("No usable counterfactual passage was produced; aborting.")
        sys.exit(1)

    if args.dry_run:
        preview_stem = os.path.splitext(os.path.basename(resolve(args.out)))[0]
        preview_path = ensure_parent(
            os.path.join(
                os.path.dirname(resolve(args.out)),
                f"counterfactual_passages_preview_{preview_stem}.json",
            )
        )
        with open(preview_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"generation": generation_stats, "samples": samples},
                handle, ensure_ascii=False, indent=2,
            )
        logger.info("--dry-run: wrote %s and loaded no models.", preview_path)
        return

    # ---------------------------------------------------------------------------------
    # Step 2: score all three settings.
    # ---------------------------------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    progress_every = args.progress_every or max(1, len(samples) // 20)
    logger.info("Device        : %s", device)
    logger.info("")
    logger.info("Scoring %d samples x 3 settings x %d models ...", len(samples), len(model_names))

    scoring_started = time.time()
    cache = score_three_settings(samples, model_names, device, progress_every)
    scoring_seconds = time.time() - scoring_started

    # ---------------------------------------------------------------------------------
    # Step 3: aggregate and score the metrics.
    # ---------------------------------------------------------------------------------
    logger.info("")
    logger.info("Aggregating %d records ...", len(samples))
    records = [
        build_record(sample, {name: cache[name][index] for name in model_names}, model_names)
        for index, sample in enumerate(samples)
    ]
    eligible_records = [r for r in records if r["counterfactual_valid"]]

    targets = list(model_names) + [ENSEMBLE_KEY]
    per_target_metrics = {
        target: compute_model_metrics(records, target, eligible_records) for target in targets
    }
    adjusted = compute_adjusted_kda(records, eligible_records)

    primary_model = args.primary_model
    if primary_model not in model_names:
        logger.warning(
            "Primary model '%s' is not in the ensemble; using '%s' for the headline.",
            primary_model, model_names[0],
        )
        primary_model = model_names[0]

    base = os.path.basename(data_path).lower()
    dataset_name = (
        "allenai/openbookqa" if "obqa" in base or "openbookqa" in base else "allenai/sciq"
    )
    summary = {
        "experiment": "pillar_3_counterfactual_context_perturbation",
        "dataset": dataset_name,
        "split": "test",
        "data_file": os.path.basename(data_path),
        "n_samples": len(records),
        "models": model_names,
        "primary_model": primary_model,
        "device": device,
        "settings": {
            "A": "no passage (question + options)",
            "B": "original passage (question + options + gold support passage)",
            "C": "counterfactual passage (gold answer rewritten as a distractor)",
        },
        "counterfactual_generation": generation_stats,
        "per_model_metrics": per_target_metrics,
        "adjusted_kda": adjusted,
        "formulas": {
            "kda_cont": (
                "KDA_cont(q) = sum_m (1 - P_m^A(correct)) * P_m^B(correct) "
                "/ sum_m (1 - P_m^A(correct))"
            ),
            "kda_adjusted_hard": (
                "sum_m (1 - P_m^A) * P_m^B * 1[argmax P_m^C == cf_target] "
                "/ sum_m (1 - P_m^A)"
            ),
            "kda_adjusted_soft": (
                "sum_m (1 - P_m^A) * P_m^B * P_m^C(cf_target) / sum_m (1 - P_m^A)"
            ),
            "kda_adjusted_sample_exclusion": (
                "mean over CF-eligible q of KDA_cont(q) * "
                "1[q is not ensemble prior-dependent]"
            ),
            "context_verified_accuracy_with_fact": (
                "(wrong_to_correct + context-dependent both_correct) / n_eligible"
            ),
        },
        "scoring_seconds": round(scoring_seconds, 2),
        "total_runtime_seconds": round(time.time() - run_started, 2),
    }

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": records}, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", out_path)

    log_tables(per_target_metrics, adjusted, generation_stats, primary_model)
    log_examples(records, primary_model)

    logger.info("")
    logger.info("Scoring time  : %.2fs (%.3fs per question)", scoring_seconds, scoring_seconds / len(records))
    logger.info("Total runtime : %.2fs", summary["total_runtime_seconds"])
    logger.info("Results       : %s", out_path)
    logger.info("Log           : %s", log_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
