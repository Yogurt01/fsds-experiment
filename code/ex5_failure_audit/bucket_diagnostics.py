"""Diagnostics for the two KDA buckets no report analyses: `both_wrong` and `correct_to_wrong`.

Both analyses here are **pure joins over files already on disk**. No model is loaded and no
inference is run; every number is recomputed from committed results files, so the whole module is
re-runnable in seconds and its findings are reproducible rather than quoted.

Analysis 1 -- confidence gate on `correct_to_wrong`
---------------------------------------------------
An item that is correct without the fact and wrong with it is one of:

    H4  genuinely adversarial context   the material misleads a model that had it right
    H5  distractor activation           the material makes a distractor more attractive
    H6  lucky guess undone              the zero-context "correct" answer was a coin flip

H6 is separable from H4/H5 without any new inference: it implies the pre-flip confidence
`P(R^q=1)` on the gold option was near the chance floor (0.25 for four options), whereas H4/H5
imply it was high. Items are gated at `< 0.40` (near-chance) and `>= 0.70` (confident, the same
threshold criterion C2 uses in the RQ1 pool).

Analysis 2 -- Setting-C ineligibility join on `both_wrong`
----------------------------------------------------------
An item that is wrong in both settings is one of:

    H1  genuinely hard      needs reasoning the student cannot do even with the fact
    H2  mis-keyed gold      the labelled answer is wrong or defensibly contested
    H3  dataset noise       malformed options, unanswerable stem, ill-posed question

The counterfactual generator already records, per item, whether the gold answer can be located in
the reference passage at all (`substitution_tier == "none"` means it cannot). An item that is
`both_wrong` **and** Setting-C-ineligible is one where the passage never lexically states the
answer -- the strongest available prior for H3, and it costs nothing to compute.

This is deliberately *not* the ensemble-agreement test, which was measured and ruled out (0
unanimous-convergent failures on SciQ, 1 on OBQA); unanimity is reported below only as a
cross-tabulation column.

Usage:
    uv run --active python code/ex5_failure_audit/bucket_diagnostics.py
    uv run --active python code/ex5_failure_audit/bucket_diagnostics.py --out results/ex5_failure_audit/bucket_diagnostics.json
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

from utils.paths import ensure_parent, resolve

PRIMARY_MODEL = "Riiid/kda-mpnet-base-race"
NEAR_CHANCE = 0.40
CONFIDENT = 0.70

DATASET_SPECS = {
    "sciq": {
        "kda": "results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json",
        "counterfactual": "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
        "qwen_key": "sciq",
    },
    "obqa": {
        "kda": "results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json",
        "counterfactual": "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json",
        "qwen_key": "obqa",
    },
}

QWEN_RESULTS = "results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json"


def classify(correct_without_fact: bool, correct_with_fact: bool) -> str:
    """Identical to categorize_kda_results.classify (kept local so this module stands alone)."""
    if correct_without_fact:
        return "both_correct" if correct_with_fact else "correct_to_wrong"
    return "wrong_to_correct" if correct_with_fact else "both_wrong"


def pooled_prediction(record: Dict, models: Sequence[str], key: str) -> int:
    n_options = len(record["options"])
    return max(
        range(n_options),
        key=lambda i: sum(record["per_model"][m][key][i] for m in models),
    )


def load_kda(dataset_key: str):
    with open(resolve(DATASET_SPECS[dataset_key]["kda"]), encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["results"]
    models = list(payload.get("summary", {}).get("models") or records[0]["per_model"].keys())
    return records, models


# --------------------------------------------------------------------------------------
# Analysis 1: confidence gate on correct_to_wrong
# --------------------------------------------------------------------------------------
def confidence_gate_kda_small(dataset_key: str) -> Dict:
    records, models = load_kda(dataset_key)
    flips: List[Dict] = []
    for record in records:
        gold = record["answer_idx"]
        pred_a = pooled_prediction(record, models, "probabilities_without_fact")
        pred_b = pooled_prediction(record, models, "probabilities_with_fact")
        if classify(pred_a == gold, pred_b == gold) != "correct_to_wrong":
            continue
        p_gold_a = sum(
            record["per_model"][m]["probabilities_without_fact"][gold] for m in models
        ) / len(models)
        flips.append(
            {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record["correct_answer"],
                "p_rq_gold_ensemble": p_gold_a,
                "predicted_with_fact": record["options"][pred_b],
            }
        )
    return summarise_gate(flips, "p_rq_gold_ensemble", f"KDA_small ensemble / {dataset_key}")


def confidence_gate_qwen(dataset_key: str) -> Optional[Dict]:
    path = resolve(QWEN_RESULTS)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["results"][DATASET_SPECS[dataset_key]["qwen_key"]]
    flips = [
        {
            "id": r["id"],
            "question": r["question"],
            "gold_answer": r["correct_answer"],
            "p_rq_gold": r["p_correct_without_fact"],
            "p_rqf_gold": r["p_correct_with_fact"],
            "predicted_with_fact": r["options"][r["predicted_idx_with_fact"]],
            "fact": r.get("passage", ""),
        }
        for r in records
        if r["r_q"] == 1 and r["r_q_plus_f"] == 0
    ]
    return summarise_gate(flips, "p_rq_gold", f"Qwen3-4B / {dataset_key}")


def summarise_gate(flips: List[Dict], field: str, label: str) -> Dict:
    values = [f[field] for f in flips]
    near = [f for f in flips if f[field] < NEAR_CHANCE]
    confident = [f for f in flips if f[field] >= CONFIDENT]
    return {
        "label": label,
        "n_correct_to_wrong": len(flips),
        "mean_p_rq_gold": statistics.fmean(values) if values else None,
        "median_p_rq_gold": statistics.median(values) if values else None,
        "n_near_chance_lt_0_40": len(near),
        "n_confident_gte_0_70": len(confident),
        "verdict": (
            "no data" if not flips
            else "predominantly H6 (lucky guesses undone)" if not confident
            else "contains H4/H5 (genuine adversarial context / distractor activation)"
        ),
        "confident_flips": sorted(confident, key=lambda f: -f[field]),
        "all_flips": flips,
    }


# --------------------------------------------------------------------------------------
# Analysis 2: Setting-C ineligibility join on both_wrong
# --------------------------------------------------------------------------------------
def ineligibility_join(dataset_key: str) -> Dict:
    records, models = load_kda(dataset_key)

    cf_path = resolve(DATASET_SPECS[dataset_key]["counterfactual"])
    tiers: Dict[int, str] = {}
    if os.path.isfile(cf_path):
        with open(cf_path, encoding="utf-8") as handle:
            for record in json.load(handle)["results"]:
                tiers[record["id"]] = record.get("substitution_tier", "none")

    ineligible: List[Dict] = []
    eligible: List[Dict] = []
    unanimous_ineligible = 0
    unanimous_eligible = 0

    for record in records:
        gold = record["answer_idx"]
        primary = record["per_model"][PRIMARY_MODEL]
        if classify(
            primary["predicted_idx_without_fact"] == gold,
            primary["predicted_idx_with_fact"] == gold,
        ) != "both_wrong":
            continue

        all_models_wrong = all(
            record["per_model"][m]["predicted_idx_with_fact"] != gold
            and record["per_model"][m]["predicted_idx_without_fact"] != gold
            for m in models
        )
        entry = {
            "id": record["id"],
            "question": record["question"],
            "gold_answer": record["correct_answer"],
            "options": record["options"],
            "passage": record["passage"],
            "substitution_tier": tiers.get(record["id"], "unknown"),
            "all_four_models_both_wrong": all_models_wrong,
            "primary_pred_with_fact": record["options"][primary["predicted_idx_with_fact"]],
        }
        if entry["substitution_tier"] == "none":
            ineligible.append(entry)
            unanimous_ineligible += all_models_wrong
        else:
            eligible.append(entry)
            unanimous_eligible += all_models_wrong

    total = len(ineligible) + len(eligible)

    # Base rate: ineligibility is only evidence about `both_wrong` if it is ENRICHED there
    # relative to the split as a whole. On OBQA 71% of every item is ineligible, so a high
    # in-bucket rate means nothing without this comparison.
    split_ineligible = sum(1 for t in tiers.values() if t == "none")
    split_total = len(tiers)
    base_rate = (split_ineligible / split_total) if split_total else None
    bucket_rate = (len(ineligible) / total) if total else None
    risk_ratio = (bucket_rate / base_rate) if (base_rate and bucket_rate is not None) else None

    return {
        "dataset": dataset_key,
        "counterfactual_results_available": bool(tiers),
        "n_both_wrong_primary": total,
        "split_ineligibility_base_rate": {
            "n_ineligible": split_ineligible,
            "n_split": split_total,
            "rate": base_rate,
        },
        "enrichment": {
            "bucket_rate": bucket_rate,
            "base_rate": base_rate,
            "risk_ratio": risk_ratio,
            "interpretation": (
                None if risk_ratio is None
                else "ineligibility is strongly enriched in both_wrong -> usable H3 screen"
                if risk_ratio >= 2.0
                else "ineligibility is at the split base rate -> NOT informative for this dataset"
            ),
        },
        "setting_c_ineligible": {
            "count": len(ineligible),
            "percentage_of_both_wrong": round(100.0 * len(ineligible) / total, 2) if total else 0.0,
            "also_failed_by_all_four_models": unanimous_ineligible,
            "interpretation": "passage never lexically states the gold answer -> H3 (broken/ill-posed) prior",
            "ids": [e["id"] for e in ineligible],
        },
        "setting_c_eligible": {
            "count": len(eligible),
            "percentage_of_both_wrong": round(100.0 * len(eligible) / total, 2) if total else 0.0,
            "also_failed_by_all_four_models": unanimous_eligible,
            "interpretation": "passage does state the answer, yet the model still failed -> H1 (hard) or H2 (mis-key)",
            "ids": [e["id"] for e in eligible],
        },
        "examples_ineligible": ineligible[:5],
        "examples_eligible": eligible[:5],
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def print_report(payload: Dict) -> None:
    print("=" * 100)
    print("ANALYSIS 1 -- confidence gate on `correct_to_wrong`")
    print("=" * 100)
    print(f"  gate: near-chance < {NEAR_CHANCE}, confident >= {CONFIDENT}\n")
    print(f"  {'population':28} {'n':>4} {'mean P(R^q=1)':>15} {'<0.40':>7} {'>=0.70':>7}  verdict")
    print("  " + "-" * 96)
    for gate in payload["confidence_gate"]:
        mean = f"{gate['mean_p_rq_gold']:.3f}" if gate["mean_p_rq_gold"] is not None else "n/a"
        print(
            f"  {gate['label']:28} {gate['n_correct_to_wrong']:4} {mean:>15} "
            f"{gate['n_near_chance_lt_0_40']:7} {gate['n_confident_gte_0_70']:7}  {gate['verdict']}"
        )

    for gate in payload["confidence_gate"]:
        if gate["confident_flips"]:
            print(f"\n  Confident flips in {gate['label']}:")
            for flip in gate["confident_flips"]:
                key = "p_rq_gold" if "p_rq_gold" in flip else "p_rq_gold_ensemble"
                print(f"    id={flip['id']:4} P(gold) {flip[key]:.3f} -> chose {flip['predicted_with_fact']!r}")
                print(f"          Q: {flip['question'][:88]}")
                print(f"          gold: {flip['gold_answer']!r}" + (f"  | fact: {flip.get('fact','')[:60]!r}" if flip.get("fact") else ""))

    print()
    print("=" * 100)
    print("ANALYSIS 2 -- Setting-C ineligibility join on `both_wrong` (primary model)")
    print("=" * 100)
    print(f"  {'dataset':8} {'both_wrong':>11} {'C-ineligible':>14} {'C-eligible':>12}   (all-4-models-wrong in each)")
    print("  " + "-" * 96)
    for join in payload["ineligibility_join"]:
        a, b = join["setting_c_ineligible"], join["setting_c_eligible"]
        print(
            f"  {join['dataset']:8} {join['n_both_wrong_primary']:11} "
            f"{a['count']:6} ({a['percentage_of_both_wrong']:5.1f}%) "
            f"{b['count']:5} ({b['percentage_of_both_wrong']:5.1f}%)"
            f"   ({a['also_failed_by_all_four_models']} / {b['also_failed_by_all_four_models']})"
        )
    print()
    print("  Enrichment vs the split base rate (the number that decides whether this is a signal):")
    print(f"  {'dataset':8} {'in-bucket':>10} {'split base':>11} {'risk ratio':>11}   verdict")
    print("  " + "-" * 96)
    for join in payload["ineligibility_join"]:
        e = join["enrichment"]
        if e["risk_ratio"] is None:
            continue
        print(
            f"  {join['dataset']:8} {e['bucket_rate']:9.3f} {e['base_rate']:11.3f} "
            f"{e['risk_ratio']:11.2f}   {e['interpretation']}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bucket diagnostics (pure joins, no inference).")
    parser.add_argument("--datasets", nargs="+", default=["sciq", "obqa"], choices=sorted(DATASET_SPECS))
    parser.add_argument("--out", default="results/ex5_failure_audit/bucket_diagnostics.json")
    args = parser.parse_args(argv)

    gates: List[Dict] = []
    joins: List[Dict] = []
    for key in args.datasets:
        gates.append(confidence_gate_kda_small(key))
        qwen = confidence_gate_qwen(key)
        if qwen:
            gates.append(qwen)
        joins.append(ineligibility_join(key))

    payload = {
        "thresholds": {"near_chance": NEAR_CHANCE, "confident": CONFIDENT},
        "primary_model": PRIMARY_MODEL,
        "confidence_gate": gates,
        "ineligibility_join": joins,
    }
    print_report(payload)

    out_path = ensure_parent(resolve(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
