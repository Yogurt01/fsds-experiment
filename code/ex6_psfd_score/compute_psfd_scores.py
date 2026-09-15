"""Compute P/S/F/D from a committed Setting-C results file. No model inference.

Every quantity this script emits is a function of probability vectors already stored in
`results/ex2_counterfactual/results_counterfactual_*.json`, so it runs on CPU in seconds
and can be re-run against the immutable execution records at any time.

It never writes to its input. Committed result files are immutable execution records
(PROJECT_READING_GUIDE section 5.5); output goes to `results/ex6_psfd_score/`.

Two named variants are emitted side by side on every split and every solver:

    D  = (1 - P) * S * max(F, 0)    the formulation as specified
    D' =           S * max(F, 0)    the same score without the prior gate

and every ensemble figure is emitted twice, over all four KDA_small members and with the
near-uniform `kda-albert-xlarge-v2-race` excluded.

Faithfulness check. P, S, p(gold|C) and p(cf|C) are recomputed out of the raw probability
vectors and compared against the scalar fields the run already stored for them. If any
pair disagrees the file is not laid out the way this script assumes and it exits non-zero
rather than writing numbers nobody can trust.

Usage:
    python code/ex6_psfd_score/compute_psfd_scores.py                 # all four runs
    python code/ex6_psfd_score/compute_psfd_scores.py --results <file> [<file> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex6_psfd_score.psfd import ALBERT, aggregate, compute_psfd, solver_views
from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

# tag -> (results file, flat-layout solver name or None, dataset key inside `results`)
DEFAULT_RUNS: Tuple[Tuple[str, str, Optional[str], Optional[str]], ...] = (
    (
        "sciq_test_full",
        "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
        None,
        None,
    ),
    (
        "obqa_test_full",
        "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json",
        None,
        None,
    ),
    (
        "obqa_test_exact_tier",
        "results/ex2_counterfactual/results_counterfactual_obqa_test_exact_tier.json",
        None,
        None,
    ),
    (
        "qwen3_4b_sciq",
        "results/ex2_counterfactual/results_counterfactual_qwen3_4b.json",
        "Qwen3-4B-Instruct-2507",
        "sciq",
    ),
)

DEFAULT_OUT_DIR = "results/ex6_psfd_score"

# The stored scalars and the stored vectors come from the same forward pass, so they
# should agree to float noise. Anything looser means the layout is not what we assume.
TOLERANCE = 1e-9

# A Setting-B distribution flatter than this is treated as a non-answer: the solver is
# near-uniform over the four options and its S/F carry no usable signal.
NEAR_UNIFORM_SPREAD = 0.05
# |p(gold|C) - p(cf|C)| below this makes the *hard* Setting-C label a coin flip. F stays
# well defined; the count exists to show how much the continuous form buys.
TIE_EPSILON = 1e-3


def _records_and_eligibility(payload: Dict, dataset_key: Optional[str]) -> Tuple[List[Dict], int, int]:
    """Records plus the run's own (n_samples, n_eligible), for the faithfulness check."""
    results = payload["results"]
    if isinstance(results, dict):
        key = dataset_key or next(iter(results))
        records = results[key]
        stats = payload["summary"]["per_dataset"][key]
        return records, stats["n_samples"], stats["n_counterfactual_eligible"]
    generation = payload["summary"]["counterfactual_generation"]
    return results, generation["n_samples"], generation["n_eligible"]


def _check_against_stored(view: Dict, scores: Dict[str, float], tag: str, item_id) -> None:
    """The vectors we read must reproduce the scalars the run already recorded."""
    for computed, stored_key in (
        ("P", "p_correct_no_passage"),
        ("S", "p_correct_original_passage"),
        ("p_gold_C", "p_correct_counterfactual_passage"),
        ("p_cf_C", "p_counterfactual_target_counterfactual_passage"),
    ):
        stored = view.get(stored_key)
        if stored is None:
            continue
        if abs(scores[computed] - stored) > TOLERANCE:
            raise SystemExit(
                f"[{tag}] id={item_id}: recomputed {computed}={scores[computed]!r} "
                f"disagrees with stored {stored_key}={stored!r}. The results file is not "
                "laid out the way this script assumes; refusing to write."
            )


def _summarise(values: Sequence[float]) -> Dict[str, Optional[float]]:
    values = list(values)
    if not values:
        return {"mean": None, "median": None, "sd": None, "min": None, "max": None,
                "n": 0, "n_zero": 0, "frac_zero": None}
    n_zero = sum(1 for v in values if v == 0.0)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "n_zero": n_zero,
        "frac_zero": n_zero / len(values),
    }


def score_run(tag: str, path: str, flat_solver: Optional[str], dataset_key: Optional[str]) -> Dict:
    """Per-item P/S/F/D for one committed run, plus the census blocks."""
    resolved = resolve_path(path)
    with open(resolved, encoding="utf-8") as handle:
        payload = json.load(handle)
    records, n_samples_claimed, n_eligible_claimed = _records_and_eligibility(
        payload, dataset_key
    )

    eligible = [r for r in records if r.get("counterfactual_valid")]
    if len(records) != n_samples_claimed or len(eligible) != n_eligible_claimed:
        raise SystemExit(
            f"[{tag}] eligibility mismatch: found {len(records)} records / "
            f"{len(eligible)} eligible, file claims {n_samples_claimed} / "
            f"{n_eligible_claimed}. Refusing to write."
        )

    solver_names = sorted(solver_views(eligible[0], flat_solver).keys())
    ex_albert_names = [n for n in solver_names if n != ALBERT]

    scored: List[Dict] = []
    # (solver, question)-pair census, keyed by solver.
    census = {
        name: {
            "near_uniform_B": 0,
            "P_gt_0.99": 0,
            "one_minus_P_lt_0.01": 0,
            "gold_cf_tie_in_C": 0,
            "F_raw_positive": 0,
            "F_raw_negative": 0,
            "term1_only_positive": 0,
        }
        for name in solver_names
    }
    # Q3: term1-only-positive F, split by the pair's own Setting-C class.
    divergence: Dict[str, Dict[str, int]] = {}

    for record in records:
        if not record.get("counterfactual_valid"):
            continue
        gold_idx = record["answer_idx"]
        cf_idx = record["counterfactual_target_idx"]
        views = solver_views(record, flat_solver)

        per_solver: Dict[str, Dict] = {}
        for name in solver_names:
            view = views[name]
            scores = compute_psfd(view, gold_idx, cf_idx)
            _check_against_stored(view, scores, tag, record["id"])
            scores["baseline_category"] = view.get("baseline_category")
            scores["counterfactual_class"] = view.get("counterfactual_class")
            per_solver[name] = scores

            cell = census[name]
            p_b = view["probabilities_original_passage"]
            if max(p_b) - min(p_b) < NEAR_UNIFORM_SPREAD:
                cell["near_uniform_B"] += 1
            if scores["P"] > 0.99:
                cell["P_gt_0.99"] += 1
            if 1.0 - scores["P"] < 0.01:
                cell["one_minus_P_lt_0.01"] += 1
            if abs(scores["p_gold_C"] - scores["p_cf_C"]) < TIE_EPSILON:
                cell["gold_cf_tie_in_C"] += 1
            if scores["F_raw"] > 0:
                cell["F_raw_positive"] += 1
            elif scores["F_raw"] < 0:
                cell["F_raw_negative"] += 1
            if scores["term1_only_positive"]:
                cell["term1_only_positive"] += 1

            cf_class = scores["counterfactual_class"] or "unclassified"
            bucket = divergence.setdefault(
                cf_class, {"n_pairs": 0, "n_F_raw_positive": 0, "n_term1_only_positive": 0}
            )
            bucket["n_pairs"] += 1
            if scores["F_raw"] > 0:
                bucket["n_F_raw_positive"] += 1
            if scores["term1_only_positive"]:
                bucket["n_term1_only_positive"] += 1

        ensemble: Dict[str, Dict] = {}
        for label, names in (("all", solver_names), ("ex_albert", ex_albert_names)):
            if not names:
                continue
            ensemble[label] = {
                "solvers": list(names),
                "D": aggregate([per_solver[n]["D"] for n in names]),
                "D_prime": aggregate([per_solver[n]["D_prime"] for n in names]),
                "P_mean": statistics.fmean([per_solver[n]["P"] for n in names]),
                "S_mean": statistics.fmean([per_solver[n]["S"] for n in names]),
                "F_mean": statistics.fmean([per_solver[n]["F"] for n in names]),
                "F_raw_mean": statistics.fmean([per_solver[n]["F_raw"] for n in names]),
            }

        scored.append(
            {
                "id": record["id"],
                "substitution_tier": record.get("substitution_tier"),
                "kda_cont": record.get("kda_original"),
                "kda_adjusted_hard": record.get("kda_adjusted_hard"),
                "kda_adjusted_soft": record.get("kda_adjusted_soft"),
                "ensemble_baseline_category": (
                    record.get("ensemble", {}).get("baseline_category")
                    if "ensemble" in record else record.get("baseline_category")
                ),
                "ensemble_counterfactual_class": (
                    record.get("ensemble", {}).get("counterfactual_class")
                    if "ensemble" in record else record.get("counterfactual_class")
                ),
                "per_solver": per_solver,
                "ensemble": ensemble,
            }
        )

    for bucket in divergence.values():
        bucket["rate_term1_only_of_pairs"] = (
            bucket["n_term1_only_positive"] / bucket["n_pairs"] if bucket["n_pairs"] else None
        )
        bucket["rate_term1_only_of_F_positive"] = (
            bucket["n_term1_only_positive"] / bucket["n_F_raw_positive"]
            if bucket["n_F_raw_positive"] else None
        )

    distributions: Dict[str, Dict] = {"per_solver": {}, "ensemble": {}}
    for name in solver_names:
        distributions["per_solver"][name] = {
            key: _summarise([item["per_solver"][name][key] for item in scored])
            for key in ("P", "S", "F", "F_raw", "D", "D_prime")
        }
    for label in ("all", "ex_albert"):
        if not scored or label not in scored[0]["ensemble"]:
            continue
        distributions["ensemble"][label] = {
            "D": _summarise([item["ensemble"][label]["D"]["mean"] for item in scored]),
            "D_prime": _summarise(
                [item["ensemble"][label]["D_prime"]["mean"] for item in scored]
            ),
            "P_mean": _summarise([item["ensemble"][label]["P_mean"] for item in scored]),
            "S_mean": _summarise([item["ensemble"][label]["S_mean"] for item in scored]),
            "F_mean": _summarise([item["ensemble"][label]["F_mean"] for item in scored]),
            "mean_solver_sd_of_D": statistics.fmean(
                [item["ensemble"][label]["D"]["sd"] for item in scored]
            ) if len(solver_names) > 1 else None,
        }

    kda_values = [item["kda_cont"] for item in scored if item["kda_cont"] is not None]
    kda_summary = _summarise(kda_values) if kda_values else None

    return {
        "tag": tag,
        "source_results_file": path,
        "dataset": payload["summary"].get("dataset", payload["summary"].get("model")),
        "n_samples": len(records),
        "n_eligible": len(eligible),
        "eligibility_rate": len(eligible) / len(records),
        "min_substitution_tier": payload["summary"].get("min_substitution_tier")
        or payload["summary"].get("counterfactual_generation", {}).get("min_substitution_tier"),
        "solvers": solver_names,
        "solvers_ex_albert": ex_albert_names,
        "distributions": distributions,
        "kda_cont_on_eligible": kda_summary,
        "pair_census": census,
        "term1_only_divergence_by_cf_class": divergence,
        "items": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results", nargs="+", default=None,
        help="Setting-C results file(s). Defaults to the four committed runs.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.results:
        runs = tuple(
            (os.path.splitext(os.path.basename(p))[0].replace("results_counterfactual_", ""),
             p, None, None)
            for p in args.results
        )
    else:
        runs = DEFAULT_RUNS

    for tag, path, flat_solver, dataset_key in runs:
        payload = score_run(tag, path, flat_solver, dataset_key)
        out_path = ensure_parent(
            resolve_path(os.path.join(args.out_dir, f"psfd_scores_{tag}.json"))
        )
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        dist = payload["distributions"]["ensemble"].get("all") or next(
            iter(payload["distributions"]["per_solver"].values())
        )
        print(
            f"{tag:22s} n_eligible={payload['n_eligible']:4d}"
            f"  mean D={dist['D']['mean']:.4f}"
            f"  mean D'={dist['D_prime']['mean']:.4f}"
            f"  zero-D={dist['D']['frac_zero']:.1%}"
            f"  -> {os.path.relpath(out_path, resolve_path('.'))}"
        )


if __name__ == "__main__":
    main()
