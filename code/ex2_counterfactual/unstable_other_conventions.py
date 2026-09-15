"""How much does the treatment of `unstable_other` move the adjusted-KDA headlines?

The problem
-----------
Setting C sorts each (model, question) pair into three classes. Two of them have an obvious
reading: `context_dependent` (the model followed the perturbed passage) earns credit,
`prior_dependent` (it kept the gold answer) does not. The third, `unstable_other` -- the
prediction moved to a option that is neither gold nor the counterfactual target -- has no
obvious reading, and the estimators as originally written *disagreed* about it:

    hard / soft         gate on `argmax P^C == cf_target`, so unstable scores ZERO
                        -- identical treatment to prior_dependent.
    sample exclusion    drops only `prior_dependent`, so unstable is RETAINED at its full
                        KDA_cont -- identical treatment to context_dependent.

The same 76 SciQ items (8.8% of eligible) and 23 OBQA items (15.9%) are therefore scored
one way by two estimators and the opposite way by the third. This script quantifies the
disagreement and supports the choice of a single convention.

The three conventions
---------------------
strict    Credit requires positive evidence of context-following. `unstable_other` earns
          nothing. (What hard/soft already did.)
lenient   Credit anything that is not prior-driven -- the diagnostic question is "did the
          answer come from the prior?", and scattering says it did not. (What sample
          exclusion already did.) Soft's analogue is to gate on 1 - P^C(gold) rather than
          on P^C(cf_target).
abstain   The perturbation returned no verdict for that pair, so drop it from BOTH the
          numerator and the denominator rather than scoring it.

Every number here is derived from the per-sample records of a committed run. No model is
loaded and no input file is modified.

Usage:
    python code/ex2_counterfactual/unstable_other_conventions.py
    python code/ex2_counterfactual/unstable_other_conventions.py --results <file> ...
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

DEFAULT_RESULTS = (
    "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
    "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json",
    "results/ex2_counterfactual/results_counterfactual_obqa_test_exact_tier.json",
)
DEFAULT_OUT = "results/ex2_counterfactual/unstable_other_conventions.json"

CONVENTIONS = ("strict", "lenient", "abstain")
DENOMINATOR_EPSILON = 1e-12


# --------------------------------------------------------------------------------------
# Per-question estimators under one convention
# --------------------------------------------------------------------------------------
def question_scores(record: Dict, model_names: Sequence[str], convention: str) -> Dict:
    """KDA_cont, hard and soft for one question under one unstable_other convention.

    Returns None-valued scores when the convention abstains on every model, which is the
    case that must not be silently dropped -- see `coverage` in the caller.
    """
    numerator_hard = 0.0
    numerator_soft = 0.0
    numerator_base = 0.0
    denominator = 0.0
    n_abstained = 0

    for name in model_names:
        view = record["per_model"][name]
        weight = view["weight"]
        p_b = view["p_correct_original_passage"]
        cf_class = view["counterfactual_class"]
        p_cf_target = view["p_counterfactual_target_counterfactual_passage"] or 0.0
        p_gold_c = view["p_correct_counterfactual_passage"]

        if convention == "abstain" and cf_class == "unstable_other":
            n_abstained += 1
            continue

        if convention == "lenient":
            gate_hard = 1.0 if cf_class != "prior_dependent" else 0.0
            # Soft analogue of "anything but the prior": mass that left the gold option.
            gate_soft = 1.0 - p_gold_c
        else:  # strict and abstain both require positive context-following
            gate_hard = 1.0 if cf_class == "context_dependent" else 0.0
            gate_soft = p_cf_target

        numerator_base += weight * p_b
        numerator_hard += weight * p_b * gate_hard
        numerator_soft += weight * p_b * gate_soft
        denominator += weight

    if denominator <= DENOMINATOR_EPSILON:
        return {"kda_cont": None, "hard": None, "soft": None, "n_abstained": n_abstained}
    return {
        "kda_cont": numerator_base / denominator,
        "hard": numerator_hard / denominator,
        "soft": numerator_soft / denominator,
        "n_abstained": n_abstained,
    }


def exclusion_under(eligible: Sequence[Dict], convention: str) -> Dict:
    """Sample-exclusion estimator (both readings) under one convention.

    The ensemble's `both_correct` conditioning is unchanged; only the treatment of
    `unstable_other` varies.

        strict    drop both_correct & (prior_dependent OR unstable_other)
        lenient   drop both_correct & prior_dependent            (the original)
        abstain   drop both_correct & prior_dependent, and remove both_correct &
                  unstable_other from the population entirely
    """
    kept, dropped, removed = [], [], []
    for record in eligible:
        ensemble = record["ensemble"]
        is_both_correct = ensemble["baseline_category"] == "both_correct"
        cf_class = ensemble["counterfactual_class"]
        if is_both_correct and cf_class == "prior_dependent":
            dropped.append(record)
        elif is_both_correct and cf_class == "unstable_other":
            if convention == "strict":
                dropped.append(record)
            elif convention == "abstain":
                removed.append(record)
            else:
                kept.append(record)
        else:
            kept.append(record)

    # `abstain` shrinks the population; the other two keep it whole.
    population = kept + dropped
    kept_scores = [r["kda_original"] for r in kept]
    retained_mean = statistics.fmean(kept_scores) if kept_scores else 0.0
    zero_filled = (
        statistics.fmean(
            [r["kda_original"] for r in kept] + [0.0] * len(dropped)
        )
        if population
        else 0.0
    )
    return {
        "mean_over_retained": retained_mean,
        "mean_over_eligible_zero_filled": zero_filled,
        "n_population": len(population),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "n_removed_no_verdict": len(removed),
    }


def verified_accuracy_under(eligible: Sequence[Dict], convention: str) -> Dict:
    """Context-verified accuracy under one convention, on the pooled ensemble."""
    n_w2c = 0
    n_context = 0
    n_unstable_bc = 0
    for record in eligible:
        ensemble = record["ensemble"]
        category = ensemble["baseline_category"]
        cf_class = ensemble["counterfactual_class"]
        if category == "wrong_to_correct":
            n_w2c += 1
        elif category == "both_correct":
            if cf_class == "context_dependent":
                n_context += 1
            elif cf_class == "unstable_other":
                n_unstable_bc += 1

    numerator = n_w2c + n_context
    denominator = len(eligible)
    if convention == "lenient":
        numerator += n_unstable_bc
    elif convention == "abstain":
        denominator -= n_unstable_bc
    return {
        "context_verified_accuracy": numerator / denominator if denominator else 0.0,
        "numerator": numerator,
        "denominator": denominator,
        "n_unstable_both_correct": n_unstable_bc,
    }


# --------------------------------------------------------------------------------------
# Per-run analysis
# --------------------------------------------------------------------------------------
def analyse(path: str) -> Dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload["summary"]
    records = payload["results"]
    eligible = [r for r in records if r["counterfactual_valid"]]
    model_names = summary["models"]

    # How much of the data is actually at stake.
    n_pairs = len(eligible) * len(model_names)
    n_unstable_pairs = sum(
        1
        for r in eligible
        for m in model_names
        if r["per_model"][m]["counterfactual_class"] == "unstable_other"
    )
    n_unstable_ensemble = sum(
        1 for r in eligible if r["ensemble"]["counterfactual_class"] == "unstable_other"
    )

    per_convention: Dict[str, Dict] = {}
    for convention in CONVENTIONS:
        scored = [question_scores(r, model_names, convention) for r in eligible]
        usable = [s for s in scored if s["hard"] is not None]
        no_verdict = len(scored) - len(usable)
        baseline = statistics.fmean(s["kda_cont"] for s in usable) if usable else 0.0
        hard_mean = statistics.fmean(s["hard"] for s in usable) if usable else 0.0
        soft_mean = statistics.fmean(s["soft"] for s in usable) if usable else 0.0
        exclusion = exclusion_under(eligible, convention)
        verified = verified_accuracy_under(eligible, convention)
        per_convention[convention] = {
            "n_questions_scored": len(usable),
            "n_questions_no_verdict": no_verdict,
            "kda_cont_mean": baseline,
            "hard_mean": hard_mean,
            "soft_mean": soft_mean,
            "hard_retention": hard_mean / baseline if baseline else 0.0,
            "soft_retention": soft_mean / baseline if baseline else 0.0,
            "sample_exclusion": exclusion,
            "context_verified_accuracy": verified,
        }

    return {
        "results_file": os.path.relpath(path, resolve_path(".")),
        "dataset": summary["dataset"],
        "min_substitution_tier": summary["counterfactual_generation"]["min_substitution_tier"],
        "models": model_names,
        "exposure": {
            "n_eligible_questions": len(eligible),
            "n_model_question_pairs": n_pairs,
            "n_unstable_pairs": n_unstable_pairs,
            "unstable_pair_rate": n_unstable_pairs / n_pairs if n_pairs else 0.0,
            "n_unstable_ensemble": n_unstable_ensemble,
            "unstable_ensemble_rate": (
                n_unstable_ensemble / len(eligible) if eligible else 0.0
            ),
        },
        "conventions": per_convention,
    }


def print_report(blocks: Sequence[Dict]) -> None:
    print("\n" + "=" * 100)
    print("EXPOSURE -- how much of each run is `unstable_other`")
    print("=" * 100)
    print(f"{'run':<34} {'pairs':>18} {'ensemble items':>20}")
    for block in blocks:
        exposure = block["exposure"]
        label = f"{block['dataset']} ({block['min_substitution_tier']})"
        print(
            f"{label:<34} "
            f"{exposure['n_unstable_pairs']:>7}/{exposure['n_model_question_pairs']:<6} "
            f"{exposure['unstable_pair_rate'] * 100:>4.1f}% "
            f"{exposure['n_unstable_ensemble']:>8}/{exposure['n_eligible_questions']:<5} "
            f"{exposure['unstable_ensemble_rate'] * 100:>4.1f}%"
        )

    for block in blocks:
        print("\n" + "=" * 100)
        print(f"{block['dataset']} ({block['min_substitution_tier']}) "
              f"-- headline movement by convention")
        print("=" * 100)
        print(
            f"  {'convention':<10} {'hard':>8} {'ret':>7} {'soft':>8} {'ret':>7} "
            f"{'excl-ret':>9} {'excl-zf':>8} {'Acc_ver':>8} {'n scored':>9}"
        )
        for convention in CONVENTIONS:
            data = block["conventions"][convention]
            exclusion = data["sample_exclusion"]
            print(
                f"  {convention:<10} "
                f"{data['hard_mean']:>8.4f} {data['hard_retention'] * 100:>6.1f}% "
                f"{data['soft_mean']:>8.4f} {data['soft_retention'] * 100:>6.1f}% "
                f"{exclusion['mean_over_retained']:>9.4f} "
                f"{exclusion['mean_over_eligible_zero_filled']:>8.4f} "
                f"{data['context_verified_accuracy']['context_verified_accuracy']:>8.4f} "
                f"{exclusion['n_population']:>9}"
            )
        abstain = block["conventions"]["abstain"]["sample_exclusion"]
        if abstain["n_removed_no_verdict"]:
            print(
                f"  note: `abstain` removes {abstain['n_removed_no_verdict']} question(s) "
                f"from the exclusion denominator entirely."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify how the treatment of `unstable_other` moves each adjusted-KDA "
            "headline. Reads committed runs; never writes to them."
        )
    )
    parser.add_argument("--results", nargs="+", default=list(DEFAULT_RESULTS))
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    blocks: List[Dict] = []
    for entry in args.results:
        path = resolve_path(entry)
        print(f"Reading {path}")
        blocks.append(analyse(path))

    print_report(blocks)

    out_path = ensure_parent(resolve_path(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "_provenance": {
                    "generator": "code/ex2_counterfactual/unstable_other_conventions.py",
                    "what": (
                        "Adjusted-KDA headlines recomputed under three conventions for "
                        "`unstable_other`, from the per-sample records of the committed "
                        "Setting-C runs. No model was run; no input was modified."
                    ),
                    "conventions": {
                        "strict": "unstable earns no credit (positive evidence required)",
                        "lenient": "credit anything that is not prior_dependent",
                        "abstain": "no verdict; drop from numerator and denominator",
                    },
                },
                "runs": blocks,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
