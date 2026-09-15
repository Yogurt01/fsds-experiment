"""E1: Setting C under a saturated solver.

The question
------------
Every Setting-C number in this repository comes from 355M-parameter encoders. README section
6.5 names the open question directly: does the counterfactual intervention still separate
prior from context when the solver is strong enough to be saturated? Experiment 1b showed
`Qwen3-4B-Instruct-2507` answering 95.4% of SciQ with no passage at all, which is the case
the intervention was designed for and has never been tested on.

This script scores Settings A / B / C with that model and classifies every `both_correct`
item exactly as the encoder runs do.

Pre-registered outcomes (all four are informative; none is a null result)
------------------------------------------------------------------------
The encoder suite showed prior-dependence rising monotonically with a solver's parametric
knowledge -- 8.2% for kda-mpnet-base-race up to 63.9% for the closed-book t5-small-ssm-nq
(README section 6.3). Qwen3-4B has far more parametric knowledge than any of them.

    H1  prior_dependent >= ~64%      the trend extrapolates; the gate still discriminates
                                     and most of SciQ's apparent knowledge dependency at
                                     LLM scale is prior.
    H2  prior_dependent 15-64%       the gate discriminates cleanly on the ~819 items
                                     baseline KDA cannot see. Best case for the method.
                                     (The H1/H2 cut is 63.9%, t5-small-ssm-nq's rate; a
                                     result whose CI spans it is reported as a boundary
                                     case rather than forced into one band.)
    H3  context_dependent >= ~90%    context sycophancy -- the gate saturates the OTHER
                                     way and passes almost everything. A negative result,
                                     and the one ClashEval's confidence-dependent blending
                                     actually predicts for a strong instruct model.
    H4  unstable_other >= ~20%       the SciQ version of OBQA's degeneration: the
                                     perturbation breaks the solver without steering it.

H1 and H3 are opposite predictions drawn from the two papers this project builds on, which
is what makes this a test rather than a description.

What this deliberately does NOT report
--------------------------------------
`KDA_cont` and the three adjusted estimators. With a single solver |M| = 1 and the weight
(1 - P(R^q = 1)) cancels between numerator and denominator, so `KDA_cont` degenerates to
P(R^{q+f} = 1) -- `kda_tiny.py` refuses this configuration for exactly that reason. Reporting
an "LLM-scale adjusted KDA" from one solver would be the degenerate quantity wearing a
better name. The classification split, context-verified accuracy and prior inflation need no
ensemble and are reported instead. The estimator layer arrives when a second LLM solver
(Qwen2.5-7B, queued for Kaggle) makes |M| = 2 possible.

Correctness gates (both run before any headline is written)
-----------------------------------------------------------
1. The Setting A/B prompts are re-implemented here rather than imported, because
   `kda_qwen_eval.build_prompt` is boolean (with_fact / without) and ex1b's published
   results must stay reproducible. `--verify-ab` therefore re-scores the first N samples and
   asserts the Setting A/B predictions match `kda_qwen3_4b_results.json` EXACTLY. If they do
   not, the prompt construction has drifted and every downstream number is incomparable.
2. The counterfactual passages are regenerated with the unmodified ex2 generator and
   asserted equal to the committed dry-run preview, so Setting C is byte-identical to the
   perturbation the encoder runs saw.

Usage:
    # gate 1 only, ~1 min: does the re-implemented prompt reproduce ex1b?
    python code/ex2_counterfactual/run_counterfactual_qwen.py --verify-ab 50 --verify-only

    # full SciQ run (~10 min on a 4GB card)
    python code/ex2_counterfactual/run_counterfactual_qwen.py

    # OBQA as well (secondary; 124 both_correct items, but L1 applies)
    python code/ex2_counterfactual/run_counterfactual_qwen.py --datasets sciq obqa
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# `render_chat` carries kda_qwen_eval's SYSTEM_PROMPT, and `score_prompt` its letter-logit
# protocol, so Settings A and B here go through exactly the code path ex1b used.
from ex1_reproduce_KDA.kda_qwen_eval import (
    DATASET_SPECS,
    LETTERS,
    letter_token_ids,
    load_model,
    render_chat,
    score_prompt,
)
from ex2_counterfactual.counterfactual_passage import build_counterfactual, tier_at_least
from ex2_counterfactual.run_counterfactual_experiment import (
    CF_CLASSES,
    TIERS,
    argmax,
    classify_baseline,
    classify_counterfactual,
)
from utils.paths import MODELS_DIR, PROJECT_ROOT, ensure_parent
from utils.paths import resolve as resolve_path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "filelock",
    "huggingface_hub", "transformers", "datasets", "fsspec",
)

# ex1b's published run, used as the Setting A/B reference for correctness gate 1.
EX1B_RESULTS = "results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json"

# The committed dry-run previews, used as the Setting C reference for correctness gate 2.
CF_PREVIEWS = {
    "sciq": "results/ex2_counterfactual/counterfactual_passages_preview_sciq.json",
    "obqa": "results/ex2_counterfactual/counterfactual_passages_preview_obqa.json",
}

# The encoder-ensemble Setting-C runs, joined for the cross-solver agreement analysis.
ENCODER_RUNS = {
    "sciq": "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
    "obqa": "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json",
}

logger = logging.getLogger("counterfactual_qwen")


def setup_logging(log_path: str, verbose: bool, append: bool) -> None:
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


# ======================================================================================
# Prompting -- three settings
# ======================================================================================
def build_prompt(sample: Dict, passage: Optional[str]) -> str:
    """Render the user turn for one sample with an arbitrary passage (or none).

    Byte-for-byte identical to `kda_qwen_eval.build_prompt` when `passage` is None
    (Setting A) or the sample's own `passage` (Setting B); correctness gate 1 asserts
    exactly that. Setting C passes the counterfactual passage instead. The gold and
    counterfactual prompts therefore differ ONLY in the text after "Fact: ".
    """
    lines: List[str] = []
    if passage is not None:
        lines.append(f"Fact: {passage.strip()}")
        lines.append("")
    lines.append(f"Question: {sample['question'].strip()}")
    for letter, option in zip(LETTERS, sample["options"]):
        lines.append(f"{letter}. {str(option).strip()}")
    lines.append("")
    lines.append("Answer with a single letter (A, B, C, or D).")
    return "\n".join(lines)


# ======================================================================================
# Correctness gates
# ======================================================================================
def verify_setting_ab(
    model, tokenizer, letter_ids, device: str, dataset_key: str, samples: Sequence[Dict],
    n_check: int,
) -> Dict:
    """Gate 1: the re-implemented prompt must reproduce ex1b's A/B predictions exactly."""
    reference_path = resolve_path(EX1B_RESULTS)
    with open(reference_path, encoding="utf-8") as handle:
        reference = {r["id"]: r for r in json.load(handle)["results"][dataset_key]}

    checked = 0
    mismatches: List[str] = []
    max_prob_delta = 0.0
    for sample in samples[:n_check]:
        ref = reference.get(sample["id"])
        if ref is None:
            continue
        probs_a, _ = score_prompt(
            model, tokenizer, render_chat(tokenizer, build_prompt(sample, None)),
            letter_ids, device,
        )
        probs_b, _ = score_prompt(
            model, tokenizer,
            render_chat(tokenizer, build_prompt(sample, sample["passage"])),
            letter_ids, device,
        )
        if argmax(probs_a) != ref["predicted_idx_without_fact"]:
            mismatches.append(
                f"id={sample['id']} Setting A: {argmax(probs_a)} != {ref['predicted_idx_without_fact']}"
            )
        if argmax(probs_b) != ref["predicted_idx_with_fact"]:
            mismatches.append(
                f"id={sample['id']} Setting B: {argmax(probs_b)} != {ref['predicted_idx_with_fact']}"
            )
        max_prob_delta = max(
            max_prob_delta,
            abs(probs_a[sample["answer_idx"]] - ref["p_correct_without_fact"]),
            abs(probs_b[sample["answer_idx"]] - ref["p_correct_with_fact"]),
        )
        checked += 1

    return {
        "reference": EX1B_RESULTS,
        "n_checked": checked,
        "n_prediction_mismatches": len(mismatches),
        "max_abs_probability_delta": max_prob_delta,
        "mismatches": mismatches[:20],
        "passed": not mismatches,
    }


def verify_counterfactual_generation(dataset_key: str, generated: Sequence[Dict]) -> Dict:
    """Gate 2: regenerated passages must equal the committed dry-run preview."""
    preview_path = resolve_path(CF_PREVIEWS[dataset_key])
    with open(preview_path, encoding="utf-8") as handle:
        committed = {s["id"]: s for s in json.load(handle)["samples"]}

    mismatches: List[str] = []
    for sample in generated:
        reference = committed.get(sample["id"])
        if reference is None:
            mismatches.append(f"id={sample['id']} absent from the committed preview")
            continue
        for field in (
            "counterfactual_passage", "counterfactual_target_idx",
            "counterfactual_valid", "substitution_tier",
        ):
            if sample[field] != reference[field]:
                mismatches.append(f"id={sample['id']} {field} differs")
    return {
        "reference": CF_PREVIEWS[dataset_key],
        "n_compared": len(generated),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "passed": not mismatches,
    }


# ======================================================================================
# Scoring
# ======================================================================================
def score_dataset(
    model, tokenizer, letter_ids, device: str, dataset_key: str, samples: Sequence[Dict],
    progress_every: int,
) -> List[Dict]:
    """Settings A, B and C for every sample. One solver, three forward passes each."""
    records: List[Dict] = []
    started = time.time()
    progress_every = progress_every or max(1, len(samples) // 20)

    for index, sample in enumerate(samples):
        answer_idx = sample["answer_idx"]
        cf_idx = sample["counterfactual_target_idx"]
        cf_valid = sample["counterfactual_valid"]

        probs_a, _ = score_prompt(
            model, tokenizer, render_chat(tokenizer, build_prompt(sample, None)),
            letter_ids, device,
        )
        probs_b, _ = score_prompt(
            model, tokenizer,
            render_chat(tokenizer, build_prompt(sample, sample["passage"])),
            letter_ids, device,
        )
        probs_c, _ = score_prompt(
            model, tokenizer,
            render_chat(tokenizer, build_prompt(sample, sample["counterfactual_passage"])),
            letter_ids, device,
        )

        pred_a, pred_b, pred_c = argmax(probs_a), argmax(probs_b), argmax(probs_c)
        records.append({
            "id": sample["id"],
            "question": sample["question"],
            "options": sample["options"],
            "answer_idx": answer_idx,
            "correct_answer": sample["correct_answer"],
            "passage": sample["passage"],
            "counterfactual_passage": sample["counterfactual_passage"],
            "counterfactual_target": sample["counterfactual_target"],
            "counterfactual_target_idx": cf_idx,
            "counterfactual_valid": cf_valid,
            "substitution_tier": sample["substitution_tier"],
            "residual_answer_mentions": sample["residual_answer_mentions"],
            "residual_glued_mentions": sample["residual_glued_mentions"],
            "probabilities_no_passage": probs_a,
            "probabilities_original_passage": probs_b,
            "probabilities_counterfactual_passage": probs_c,
            "predicted_idx_no_passage": pred_a,
            "predicted_idx_original_passage": pred_b,
            "predicted_idx_counterfactual_passage": pred_c,
            "p_correct_no_passage": probs_a[answer_idx],
            "p_correct_original_passage": probs_b[answer_idx],
            "p_correct_counterfactual_passage": probs_c[answer_idx],
            "p_counterfactual_target_counterfactual_passage": (
                probs_c[cf_idx] if cf_valid else None
            ),
            "is_correct_no_passage": pred_a == answer_idx,
            "is_correct_original_passage": pred_b == answer_idx,
            "is_correct_counterfactual_passage": pred_c == answer_idx,
            "follows_counterfactual": bool(cf_valid and pred_c == cf_idx),
            "baseline_category": classify_baseline(pred_a == answer_idx, pred_b == answer_idx),
            "counterfactual_class": (
                classify_counterfactual(pred_c, answer_idx, cf_idx) if cf_valid else None
            ),
        })

        if (index + 1) % progress_every == 0 or index + 1 == len(samples):
            elapsed = time.time() - started
            logger.info(
                "  [%s] %d/%d (%.1f%%) | %.2f s/question | elapsed %.1f s",
                dataset_key, index + 1, len(samples), 100.0 * (index + 1) / len(samples),
                elapsed / (index + 1), elapsed,
            )
    return records


# ======================================================================================
# Analysis
# ======================================================================================
def wilson_interval(k: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """Wilson score interval for a binomial proportion (well-behaved near 0 and 1)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (centre - half, centre + half)


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str], classes: Sequence[str]) -> Optional[float]:
    """Unweighted Cohen's kappa between two label sequences over a fixed class set."""
    n = len(labels_a)
    if n == 0:
        return None
    observed = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in classes
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def analyse(records: Sequence[Dict], dataset_key: str) -> Dict:
    """The headline classification split, plus the analyses that cost nothing extra."""
    eligible = [r for r in records if r["counterfactual_valid"]]
    n, n_elig = len(records), len(eligible)

    buckets: Dict[str, int] = {}
    for record in records:
        buckets[record["baseline_category"]] = buckets.get(record["baseline_category"], 0) + 1

    both_correct = [r for r in eligible if r["baseline_category"] == "both_correct"]
    split = {c: sum(1 for r in both_correct if r["counterfactual_class"] == c) for c in CF_CLASSES}
    n_bc = len(both_correct)

    # The pre-registered outcome test.
    def share(cls: str) -> float:
        return split[cls] / n_bc if n_bc else 0.0

    # H1/H2 cut = 0.639, the closed-book t5-small-ssm-nq rate the extrapolation is against.
    h1_cut = 0.639
    prior_lo, prior_hi = wilson_interval(split["prior_dependent"], n_bc)
    if share("context_dependent") >= 0.90:
        outcome = "H3 (context_dependent >= 90%): context sycophancy -- the gate saturates"
    elif share("unstable_other") >= 0.20:
        outcome = "H4 (unstable_other >= 20%): the perturbation breaks the solver"
    elif prior_lo <= h1_cut <= prior_hi:
        outcome = (
            f"H1/H2 BOUNDARY: prior_dependent {share('prior_dependent'):.1%} "
            f"[95% CI {prior_lo:.1%}-{prior_hi:.1%}] contains the H1 cut of {h1_cut:.1%}, "
            "so the monotone extrapolation is not distinguishable from H2 at this n"
        )
    elif share("prior_dependent") > h1_cut:
        outcome = "H1 (prior_dependent > 63.9%): the monotone trend extrapolates"
    elif share("prior_dependent") >= 0.15:
        outcome = "H2 (prior_dependent 15-63.9%): the gate discriminates cleanly"
    else:
        outcome = (
            "none of H1-H4 as pre-registered "
            f"(prior {share('prior_dependent'):.1%}, context {share('context_dependent'):.1%}, "
            f"unstable {share('unstable_other'):.1%})"
        )

    n_w2c_eligible = sum(1 for r in eligible if r["baseline_category"] == "wrong_to_correct")
    n_context = split["context_dependent"]
    # Strict convention (docs/ex2_counterfactual/unstable_other_convention.md): only context_dependent counts.
    n_failed_check = split["prior_dependent"] + split["unstable_other"]

    analysis = {
        "n_samples": n,
        "n_counterfactual_eligible": n_elig,
        "baseline_buckets": buckets,
        "both_correct_eligible": {
            "n": n_bc,
            **{
                c: {
                    "count": split[c],
                    "share": share(c),
                    "ci95": wilson_interval(split[c], n_bc),
                }
                for c in CF_CLASSES
            },
        },
        "pre_registered_outcome": outcome,
        "accuracy": {
            "no_passage": sum(r["is_correct_no_passage"] for r in records) / n,
            "original_passage": sum(r["is_correct_original_passage"] for r in records) / n,
            "counterfactual_passage_vs_gold": (
                sum(r["is_correct_counterfactual_passage"] for r in eligible) / n_elig
                if n_elig else 0.0
            ),
            "counterfactual_passage_vs_cf_target": (
                sum(r["follows_counterfactual"] for r in eligible) / n_elig if n_elig else 0.0
            ),
        },
        "context_verified_accuracy_with_fact": (
            (n_w2c_eligible + n_context) / n_elig if n_elig else 0.0
        ),
        "raw_accuracy_with_fact_on_eligible": (
            sum(r["is_correct_original_passage"] for r in eligible) / n_elig if n_elig else 0.0
        ),
        "prior_inflation_of_accuracy_with_fact": (
            n_failed_check / n_elig if n_elig else 0.0
        ),
        "unstable_other_convention": "strict",
    }

    # --- secondary: is Setting-C class a property of the QUESTION or of the SOLVER? -----
    encoder_path = ENCODER_RUNS.get(dataset_key)
    if encoder_path and os.path.exists(resolve_path(encoder_path)):
        with open(resolve_path(encoder_path), encoding="utf-8") as handle:
            encoder = {r["id"]: r for r in json.load(handle)["results"]}
        pairs = [
            (r["counterfactual_class"], encoder[r["id"]]["ensemble"]["counterfactual_class"])
            for r in eligible
            if r["id"] in encoder and encoder[r["id"]]["counterfactual_valid"]
        ]
        if pairs:
            qwen_labels = [p[0] for p in pairs]
            enc_labels = [p[1] for p in pairs]
            agree = sum(1 for a, b in pairs if a == b)
            analysis["cross_solver_agreement"] = {
                "compared_against": encoder_path,
                "n_items": len(pairs),
                "raw_agreement": agree / len(pairs),
                "cohens_kappa": cohen_kappa(qwen_labels, enc_labels, CF_CLASSES),
                "confusion": {
                    f"qwen={a}|encoder={b}": sum(
                        1 for x, y in pairs if x == a and y == b
                    )
                    for a in CF_CLASSES for b in CF_CLASSES
                },
                "note": (
                    "Low agreement means the Setting-C class is a property of the SOLVER, "
                    "not of the question -- which would block using these labels to filter "
                    "a question bank, the premise of RQ3's disentangled-filtering arm."
                ),
            }

    # --- secondary: does the KDA weight predict a failed context check at this scale? ---
    saturated = sum(1 for r in records if r["p_correct_no_passage"] >= 0.999)
    analysis["saturation"] = {
        "n_p_correct_no_passage_ge_0.999": saturated,
        "share": saturated / n if n else 0.0,
        "mean_kda_weight": (
            sum(1.0 - r["p_correct_no_passage"] for r in records) / n if n else 0.0
        ),
        "note": (
            "The KDA_cont weight is (1 - P(R^q=1)). |M|=1 makes the metric itself "
            "degenerate, so this is reported as a diagnostic only."
        ),
    }
    return analysis


def log_analysis(dataset_key: str, analysis: Dict) -> None:
    logger.info("")
    logger.info("=" * 100)
    logger.info("RESULTS -- %s", dataset_key.upper())
    logger.info("=" * 100)
    accuracy = analysis["accuracy"]
    logger.info(
        "  Acc: %.2f%% (A, no passage) -> %.2f%% (B, gold) -> %.2f%% (C, vs gold)",
        accuracy["no_passage"] * 100, accuracy["original_passage"] * 100,
        accuracy["counterfactual_passage_vs_gold"] * 100,
    )
    logger.info(
        "  Setting C follows the counterfactual target: %.2f%%",
        accuracy["counterfactual_passage_vs_cf_target"] * 100,
    )
    logger.info("  Baseline buckets: %s", analysis["baseline_buckets"])
    block = analysis["both_correct_eligible"]
    logger.info("")
    logger.info("  both_correct AND CF-eligible -- the Setting-C target set: n = %d", block["n"])
    for cf_class in CF_CLASSES:
        lo, hi = block[cf_class]["ci95"]
        logger.info(
            "    %-20s %5d  %6.2f%%  95%% CI [%.2f%%, %.2f%%]",
            cf_class, block[cf_class]["count"], block[cf_class]["share"] * 100,
            lo * 100, hi * 100,
        )
    logger.info("")
    logger.info("  PRE-REGISTERED OUTCOME: %s", analysis["pre_registered_outcome"])
    logger.info("")
    logger.info(
        "  Acc_wf on eligible %.4f -> context-verified %.4f (prior inflation %.4f)",
        analysis["raw_accuracy_with_fact_on_eligible"],
        analysis["context_verified_accuracy_with_fact"],
        analysis["prior_inflation_of_accuracy_with_fact"],
    )
    saturation = analysis["saturation"]
    logger.info(
        "  Saturation: %d/%d (%.1f%%) at P(R^q=1) >= 0.999 | mean KDA weight %.4f",
        saturation["n_p_correct_no_passage_ge_0.999"], analysis["n_samples"],
        saturation["share"] * 100, saturation["mean_kda_weight"],
    )
    agreement = analysis.get("cross_solver_agreement")
    if agreement:
        logger.info("")
        logger.info(
            "  Cross-solver agreement on the Setting-C class (Qwen3-4B vs the encoder "
            "ensemble), n = %d: raw %.3f, Cohen's kappa %s",
            agreement["n_items"], agreement["raw_agreement"],
            "n/a" if agreement["cohens_kappa"] is None else f"{agreement['cohens_kappa']:.3f}",
        )
        for key, count in agreement["confusion"].items():
            if count:
                logger.info("    %-52s %5d", key, count)


# ======================================================================================
# Main
# ======================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="E1: Setting C with a saturated LLM solver (Qwen3-4B, 4-bit NF4)."
    )
    parser.add_argument(
        "--model-path", default=os.path.join(MODELS_DIR, "Qwen3-4B-Instruct-2507")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["sciq"], choices=sorted(DATASET_SPECS),
        help="Default sciq only: OBQA is secondary and carries limitation L1.",
    )
    parser.add_argument(
        "--out", default="results/ex2_counterfactual/results_counterfactual_qwen3_4b.json"
    )
    parser.add_argument(
        "--log-file", default="results/ex2_counterfactual/counterfactual_qwen3_4b.log"
    )
    parser.add_argument(
        "--passages-out",
        default="results/ex2_counterfactual/counterfactual_passages_for_cloud.json",
        help=(
            "Standalone counterfactual passages, so the self-contained Kaggle script can "
            "run Setting C without importing this repository."
        ),
    )
    parser.add_argument("--min-substitution-tier", choices=TIERS, default="partial")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--verify-ab", type=int, default=50,
        help="Correctness gate 1: re-score N samples against ex1b (0 disables).",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Run the gates and stop."
    )
    parser.add_argument("--double-quant", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    args = parser.parse_args()

    log_path = ensure_parent(resolve_path(args.log_file))
    setup_logging(log_path, args.verbose, args.append_log)
    run_started = time.time()

    logger.info("=" * 100)
    logger.info("E1 -- SETTING C UNDER A SATURATED SOLVER")
    logger.info("=" * 100)
    logger.info("Workspace : %s", PROJECT_ROOT)
    logger.info("Python    : %s", platform.python_version())
    logger.info("torch     : %s (CUDA %s)", torch.__version__, torch.cuda.is_available())

    model_path = resolve_path(args.model_path)
    if not os.path.isdir(model_path):
        logger.error("Model directory not found: %s", model_path)
        return 1

    # --- counterfactual generation (no model needed) ----------------------------------
    per_dataset_samples: Dict[str, List[Dict]] = {}
    generation_gates: Dict[str, Dict] = {}
    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        with open(resolve_path(spec["path"]), encoding="utf-8") as handle:
            raw = json.load(handle)
        if args.limit:
            raw = raw[: args.limit]

        samples: List[Dict] = []
        tier_counts = {tier: 0 for tier in list(TIERS) + ["none"]}
        for entry in raw:
            counterfactual = build_counterfactual(entry)
            if counterfactual["counterfactual_valid"] and not tier_at_least(
                counterfactual["substitution_tier"], args.min_substitution_tier
            ):
                counterfactual["counterfactual_valid"] = False
            tier_counts[counterfactual["substitution_tier"]] += 1
            merged = {**entry, **counterfactual}
            if merged["counterfactual_target_idx"] is None:
                merged["counterfactual_target_idx"] = (
                    (merged["answer_idx"] + 1) % len(merged["options"])
                )
            samples.append(merged)
        per_dataset_samples[dataset_key] = samples
        n_eligible = sum(1 for s in samples if s["counterfactual_valid"])
        logger.info(
            "[%s] generated %d usable counterfactual passages of %d (tiers: %s)",
            dataset_key, n_eligible, len(samples), tier_counts,
        )

        # Correctness gate 2 -- only meaningful on a full, untruncated run.
        if not args.limit and args.min_substitution_tier == "partial":
            gate = verify_counterfactual_generation(dataset_key, samples)
            generation_gates[dataset_key] = gate
            logger.info(
                "[%s] GATE 2 counterfactual generation vs committed preview: %s (%d compared)",
                dataset_key, "PASS" if gate["passed"] else "FAIL", gate["n_compared"],
            )
            if not gate["passed"]:
                for mismatch in gate["mismatches"]:
                    logger.error("    %s", mismatch)
                return 1

    # --- the passages file the cloud script will consume ------------------------------
    passages_path = ensure_parent(resolve_path(args.passages_out))
    with open(passages_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "_provenance": {
                    "generator": "code/ex2_counterfactual/run_counterfactual_qwen.py",
                    "what": (
                        "Counterfactual passages produced by the unmodified ex2 generator, "
                        "exported standalone so a self-contained cloud script (e.g. "
                        "run_qwen7b_eval.py on Kaggle) can run Setting C with no repo import."
                    ),
                    "min_substitution_tier": args.min_substitution_tier,
                },
                "datasets": {
                    key: [
                        {
                            "id": s["id"],
                            "question": s["question"],
                            "options": s["options"],
                            "answer_idx": s["answer_idx"],
                            "passage": s["passage"],
                            "counterfactual_passage": s["counterfactual_passage"],
                            "counterfactual_target_idx": s["counterfactual_target_idx"],
                            "counterfactual_valid": s["counterfactual_valid"],
                            "substitution_tier": s["substitution_tier"],
                        }
                        for s in samples
                    ]
                    for key, samples in per_dataset_samples.items()
                },
            },
            handle, ensure_ascii=False, indent=2,
        )
    logger.info("Wrote %s", passages_path)

    # --- model ------------------------------------------------------------------------
    model, tokenizer, memory = load_model(model_path, args, logger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    letter_ids = letter_token_ids(tokenizer, logger)

    # --- correctness gate 1 -----------------------------------------------------------
    ab_gates: Dict[str, Dict] = {}
    if args.verify_ab:
        for dataset_key in args.datasets:
            gate = verify_setting_ab(
                model, tokenizer, letter_ids, device, dataset_key,
                per_dataset_samples[dataset_key], args.verify_ab,
            )
            ab_gates[dataset_key] = gate
            logger.info(
                "[%s] GATE 1 Setting A/B vs ex1b: %s (%d checked, %d mismatches, "
                "max |dP| = %.2e)",
                dataset_key, "PASS" if gate["passed"] else "FAIL",
                gate["n_checked"], gate["n_prediction_mismatches"],
                gate["max_abs_probability_delta"],
            )
            if not gate["passed"]:
                for mismatch in gate["mismatches"]:
                    logger.error("    %s", mismatch)
                logger.error(
                    "The re-implemented prompt does not reproduce ex1b. Refusing to "
                    "produce numbers that cannot be compared with it."
                )
                return 1

    if args.verify_only:
        logger.info("--verify-only: gates passed, stopping before the full run.")
        return 0

    # --- scoring ----------------------------------------------------------------------
    payload: Dict[str, Dict] = {}
    for dataset_key in args.datasets:
        samples = per_dataset_samples[dataset_key]
        logger.info("")
        logger.info("Scoring %s (%d samples, 3 settings each) ...", dataset_key, len(samples))
        records = score_dataset(
            model, tokenizer, letter_ids, device, dataset_key, samples, args.progress_every
        )
        analysis = analyse(records, dataset_key)
        log_analysis(dataset_key, analysis)
        payload[dataset_key] = {"analysis": analysis, "records": records}

    out_path = ensure_parent(resolve_path(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": {
                    "experiment": (
                        "E1 -- Setting C (counterfactual context perturbation) with a "
                        "saturated single LLM solver"
                    ),
                    "model": os.path.basename(model_path),
                    "quantization": "4-bit NF4",
                    "ensemble_size": 1,
                    "not_reported": (
                        "KDA_cont and the three adjusted estimators. With |M| = 1 the "
                        "weight (1 - P(R^q=1)) cancels and KDA_cont degenerates to "
                        "P(R^{q+f}=1); see kda_tiny.MIN_ENSEMBLE_SIZE."
                    ),
                    "correctness_gates": {
                        "setting_ab_vs_ex1b": ab_gates,
                        "counterfactual_generation": generation_gates,
                    },
                    "min_substitution_tier": args.min_substitution_tier,
                    "memory": memory,
                    "per_dataset": {k: v["analysis"] for k, v in payload.items()},
                    "total_runtime_seconds": round(time.time() - run_started, 2),
                },
                "results": {k: v["records"] for k, v in payload.items()},
            },
            handle, ensure_ascii=False, indent=2,
        )
    logger.info("")
    logger.info("Wrote %s", out_path)
    logger.info("Total runtime: %.1f s", time.time() - run_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
