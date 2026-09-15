"""Evaluate P/S/F/D against the ground truth this project already has. No inference.

Three tiers, in descending evidential strength. The tiering is the point: the strongest
diagnostic available is also partly circular, and the least circular one rests on labels
with no measured inter-annotator agreement. Both facts are reported with the numbers.

  Tier 1  Discrimination of the Setting-C classes (context_dependent vs prior_dependent).
          Fully automated. PARTLY DEFINITIONAL: F and the Setting-C label are both
          functions of Setting C, so a high AUC here is a calibration result -- "does the
          continuous score track the hard label" -- not independent validation.

  Tier 2  Structure visible from Settings A and B alone: the four contingency buckets,
          the within-`both_correct` split KDA_cont cannot see at all, and rank agreement
          with KDA_cont. Independent of how F is defined.

  Tier 3  The manual five-category failure taxonomy. CORROBORATION ONLY. Single
          annotator, single pass, no rubric written in advance, no kappa, one forced
          label per item (docs/ex5_failure_audit/failure_taxonomy_methodology.md section 0).
          It is never a pass/fail gate, and the caveat is emitted with every Tier-3 block.
          There is also no clean class: all flagged items are failure cases, so the only
          available control is the *unflagged* eligible remainder, and "not flagged" does
          not mean "verified sound".

Five pass/fail criteria were pre-registered in the RQ2 planning document before any of
these numbers existed. The one free parameter left unset there -- the margin by which D
must beat its best single component under criterion 5 -- is fixed at 0.02 AUC in
`CRITERION_5_MARGIN` below, written before the script was first run.

Usage:
    python code/ex6_psfd_score/evaluate_psfd.py            # all four scored runs
    python code/ex6_psfd_score/evaluate_psfd.py --scores <file> [<file> ...]
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

from ex6_psfd_score.psfd import ALBERT, auc, spearman
from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

DEFAULT_SCORE_FILES = (
    "results/ex6_psfd_score/psfd_scores_sciq_test_full.json",
    "results/ex6_psfd_score/psfd_scores_obqa_test_full.json",
    "results/ex6_psfd_score/psfd_scores_obqa_test_exact_tier.json",
    "results/ex6_psfd_score/psfd_scores_qwen3_4b_sciq.json",
)

# Which run each verdict is adjudicated on. SciQ full is the pre-registered primary: it is
# the only split where the counterfactual is clean at scale (97.3% eligible, 92.0% exact).
PRIMARY_TAG = "sciq_test_full"

TAXONOMY_FILES = {
    "sciq": ("results/ex5_failure_analysis/rq1_flagged_questions.json", "sciq"),
    "obqa": ("results/ex5_failure_analysis/rq1_flagged_questions_rebuilt_obqa.json", None),
}

TAXONOMY_CAVEAT = (
    "CORROBORATION ONLY -- not a validation gate. The five failure-category labels are "
    "single-annotator, single-pass, with no rubric written in advance and no measured "
    "inter-annotator agreement (no kappa); each item carries exactly one forced label. "
    "See docs/ex5_failure_audit/failure_taxonomy_methodology.md section 0. The flagged pool "
    "contains only failure cases, so the 'unflagged_eligible' row is a remainder, not a "
    "verified-clean control."
)

CRITERION_1_AUC = 0.75
CRITERION_2_AUC = 0.70
CRITERION_3_RHO = 0.90
CRITERION_4_MIN_CATEGORIES = 3
# Fixed at implementation time, before the script was run: how much AUC D must add over
# its best single component for the product form to be worth keeping.
CRITERION_5_MARGIN = 0.02

CATEGORIES = (
    "common_knowledge",
    "parametric_knowledge",
    "reasoning_shortcut",
    "option_leakage",
    "material_not_necessary",
)


# ---------------------------------------------------------------------------------------
# Score accessors
# ---------------------------------------------------------------------------------------


def _ens(item: Dict, label: str, key: str) -> Optional[float]:
    block = item["ensemble"].get(label)
    return None if block is None else block[key]["mean"]


def _signal(item: Dict, name: str, label: str = "all") -> Optional[float]:
    """One comparable per-question score, by name."""
    block = item["ensemble"].get(label)
    if name == "D":
        return _ens(item, label, "D")
    if name == "D_prime":
        return _ens(item, label, "D_prime")
    if name == "KDA_cont":
        return item["kda_cont"]
    if block is None:
        return None
    if name == "S":
        return block["S_mean"]
    if name == "F":
        return block["F_mean"]
    if name == "one_minus_P":
        return 1.0 - block["P_mean"]
    raise KeyError(name)


SIGNALS = ("D", "D_prime", "S", "F", "one_minus_P", "KDA_cont")


def _auc_by_class(
    items: Sequence[Dict], signal: str, label: str = "all",
    positive: str = "context_dependent", negative: str = "prior_dependent",
) -> Dict[str, Optional[float]]:
    """AUC separating two Setting-C classes, on the items where the signal is defined."""
    pos, neg = [], []
    for item in items:
        value = _signal(item, signal, label)
        if value is None:
            continue
        cls = item["ensemble_counterfactual_class"]
        if cls == positive:
            pos.append(value)
        elif cls == negative:
            neg.append(value)
    return {"auc": auc(pos, neg), "n_positive": len(pos), "n_negative": len(neg)}


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


# ---------------------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------------------


def tier1(payload: Dict) -> Dict:
    items = payload["items"]
    out: Dict = {
        "what": "AUC separating context_dependent (positive) from prior_dependent "
                "(negative), ensemble-level Setting-C class.",
        "circularity_note": "F and the Setting-C class are both functions of Setting C. "
                            "This is a calibration check, not independent validation.",
        "ensemble_all": {s: _auc_by_class(items, s, "all") for s in SIGNALS},
    }
    if "ex_albert" in (items[0]["ensemble"] if items else {}):
        out["ensemble_ex_albert"] = {
            s: _auc_by_class(items, s, "ex_albert") for s in SIGNALS
        }
    per_solver = {}
    for name in payload["solvers"]:
        pos, neg = [], []
        for item in items:
            cell = item["per_solver"][name]
            if cell["counterfactual_class"] == "context_dependent":
                pos.append(cell["D"])
            elif cell["counterfactual_class"] == "prior_dependent":
                neg.append(cell["D"])
        per_solver[name] = {"auc": auc(pos, neg), "n_positive": len(pos), "n_negative": len(neg)}
    out["per_solver_D"] = per_solver
    solver_aucs = [v["auc"] for v in per_solver.values() if v["auc"] is not None]
    out["per_solver_D_auc_spread"] = (
        max(solver_aucs) - min(solver_aucs) if len(solver_aucs) > 1 else None
    )
    return out


def tier2(payload: Dict) -> Dict:
    items = payload["items"]
    by_bucket: Dict[str, List[Dict]] = {}
    for item in items:
        by_bucket.setdefault(item["ensemble_baseline_category"] or "none", []).append(item)

    both_correct = by_bucket.get("both_correct", [])
    out: Dict = {
        "what": "Structure visible from Settings A and B alone; independent of how F is "
                "defined.",
        "bucket_means": {
            bucket: {
                "n": len(rows),
                "D": _mean_or_none([_ens(r, "all", "D") for r in rows]),
                "D_prime": _mean_or_none([_ens(r, "all", "D_prime") for r in rows]),
                "KDA_cont": _mean_or_none([r["kda_cont"] for r in rows]),
            }
            for bucket, rows in sorted(by_bucket.items())
        },
        "within_both_correct": {
            "n": len(both_correct),
            "note": "The bucket baseline KDA cannot see: correct in A and in B, so the "
                    "ignorance weight is small and the item is credited anyway.",
            **{
                s: _auc_by_class(both_correct, s, "all") for s in SIGNALS
            },
        },
    }

    w2c = by_bucket.get("wrong_to_correct", [])
    if w2c and both_correct:
        out["wrong_to_correct_vs_both_correct"] = {
            s: {
                "auc": auc(
                    [v for v in (_signal(r, s, "all") for r in w2c) if v is not None],
                    [v for v in (_signal(r, s, "all") for r in both_correct) if v is not None],
                ),
                "n_positive": len(w2c),
                "n_negative": len(both_correct),
            }
            for s in SIGNALS
        }

    paired = [(item, item["kda_cont"]) for item in items if item["kda_cont"] is not None]
    if paired:
        kda = [k for _, k in paired]
        out["rank_agreement_with_kda_cont"] = {
            "n": len(paired),
            "spearman_D": spearman([_ens(i, "all", "D") for i, _ in paired], kda),
            "spearman_D_prime": spearman([_ens(i, "all", "D_prime") for i, _ in paired], kda),
            "spearman_S": spearman([i["ensemble"]["all"]["S_mean"] for i, _ in paired], kda),
            "spearman_F": spearman([i["ensemble"]["all"]["F_mean"] for i, _ in paired], kda),
        }
        mean_kda = statistics.fmean(kda)
        out["retention_vs_kda_cont"] = {
            "mean_kda_cont": mean_kda,
            "mean_D": _mean_or_none([_ens(i, "all", "D") for i, _ in paired]),
            "mean_D_prime": _mean_or_none([_ens(i, "all", "D_prime") for i, _ in paired]),
            "mean_kda_adjusted_hard": _mean_or_none(
                [i["kda_adjusted_hard"] for i, _ in paired]
            ),
            "mean_kda_adjusted_soft": _mean_or_none(
                [i["kda_adjusted_soft"] for i, _ in paired]
            ),
            "note": "D and KDA_cont are not on a common scale; the ratio is descriptive "
                    "only, and is NOT a retention figure comparable to hard/soft.",
        }
    return out


def _load_taxonomy(dataset: str) -> Dict[int, str]:
    path, key = TAXONOMY_FILES[dataset]
    with open(resolve_path(path), encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["datasets"][key]["questions"] if key else payload["questions"]
    return {r["question_id"]: r["failure_category"] for r in records}


def tier3(payload: Dict, dataset: str) -> Dict:
    labels = _load_taxonomy(dataset)
    items = payload["items"]
    rows: Dict[str, Dict] = {}
    for category in CATEGORIES + ("unflagged_eligible",):
        if category == "unflagged_eligible":
            subset = [i for i in items if i["id"] not in labels]
        else:
            subset = [i for i in items if labels.get(i["id"]) == category]
        rows[category] = {
            "n": len(subset),
            "mean_D": _mean_or_none([_ens(r, "all", "D") for r in subset]),
            "mean_D_prime": _mean_or_none([_ens(r, "all", "D_prime") for r in subset]),
            "mean_one_minus_P": _mean_or_none(
                [1.0 - r["ensemble"]["all"]["P_mean"] for r in subset]
            ),
            "mean_F": _mean_or_none([r["ensemble"]["all"]["F_mean"] for r in subset]),
            "mean_KDA_cont": _mean_or_none([r["kda_cont"] for r in subset]),
        }
    control = rows["unflagged_eligible"]["mean_D"]
    control_prime = rows["unflagged_eligible"]["mean_D_prime"]
    for category in CATEGORIES:
        row = rows[category]
        row["below_control_D"] = (
            None if (row["mean_D"] is None or control is None or row["n"] == 0)
            else row["mean_D"] < control
        )
        row["below_control_D_prime"] = (
            None if (row["mean_D_prime"] is None or control_prime is None or row["n"] == 0)
            else row["mean_D_prime"] < control_prime
        )
    return {
        "caveat": TAXONOMY_CAVEAT,
        "n_flagged_labels_joined": sum(1 for i in items if i["id"] in labels),
        "rows": rows,
        "n_categories_below_control_D": sum(
            1 for c in CATEGORIES if rows[c]["below_control_D"] is True
        ),
        "n_categories_below_control_D_prime": sum(
            1 for c in CATEGORIES if rows[c]["below_control_D_prime"] is True
        ),
    }


def d_vs_d_prime(payload: Dict) -> Dict:
    """The Q2 comparison: what the (1 - P) gate does on this split."""
    dist = payload["distributions"]
    ens = dist["ensemble"].get("all", {})
    per_solver = {
        name: {
            "mean_D": cell["D"]["mean"],
            "mean_D_prime": cell["D_prime"]["mean"],
            "frac_D_zero": cell["D"]["frac_zero"],
            "frac_D_prime_zero": cell["D_prime"]["frac_zero"],
            "mean_one_minus_P": 1.0 - cell["P"]["mean"],
            "frac_one_minus_P_lt_0.01": (
                payload["pair_census"][name]["one_minus_P_lt_0.01"] / payload["n_eligible"]
            ),
        }
        for name, cell in dist["per_solver"].items()
    }
    out = {
        "ensemble_all": {
            "mean_D": ens.get("D", {}).get("mean"),
            "mean_D_prime": ens.get("D_prime", {}).get("mean"),
            "frac_D_zero": ens.get("D", {}).get("frac_zero"),
            "gate_attenuation": (
                ens["D"]["mean"] / ens["D_prime"]["mean"]
                if ens.get("D_prime", {}).get("mean") else None
            ),
        },
        "per_solver": per_solver,
    }
    if "ex_albert" in dist["ensemble"]:
        exa = dist["ensemble"]["ex_albert"]
        out["ensemble_ex_albert"] = {
            "mean_D": exa["D"]["mean"],
            "mean_D_prime": exa["D_prime"]["mean"],
            "gate_attenuation": (
                exa["D"]["mean"] / exa["D_prime"]["mean"] if exa["D_prime"]["mean"] else None
            ),
        }
    return out


def verdict(payload: Dict, t1: Dict, t2: Dict, t3: Optional[Dict]) -> Dict:
    """The five pre-registered criteria, adjudicated on D (the specified formulation)."""
    d_auc = t1["ensemble_all"]["D"]["auc"]
    kda_auc = t1["ensemble_all"]["KDA_cont"]["auc"]
    components = {
        s: t1["ensemble_all"][s]["auc"]
        for s in ("S", "F", "one_minus_P")
        if t1["ensemble_all"][s]["auc"] is not None
    }
    best_component = max(components, key=components.get) if components else None
    best_component_auc = components.get(best_component) if best_component else None

    within = t2["within_both_correct"]["D"]["auc"]
    rho = t2.get("rank_agreement_with_kda_cont", {}).get("spearman_D")
    n_below = t3["n_categories_below_control_D"] if t3 else None
    spread = t1["per_solver_D_auc_spread"]

    def _pass(value) -> Optional[bool]:
        return value

    criteria = {
        "1_tier1_auc": {
            "statement": f"Tier-1 AUC(context vs prior) >= {CRITERION_1_AUC} AND > KDA_cont's",
            "D_auc": d_auc,
            "kda_cont_auc": kda_auc,
            "passed": None if d_auc is None else (
                d_auc >= CRITERION_1_AUC and (kda_auc is None or d_auc > kda_auc)
            ),
        },
        "2_within_both_correct": {
            "statement": f"Within-both_correct AUC >= {CRITERION_2_AUC}",
            "auc": within,
            "n": t2["within_both_correct"]["n"],
            "passed": None if within is None else within >= CRITERION_2_AUC,
        },
        "3_not_a_repackaging": {
            "statement": f"Spearman(D, KDA_cont) < {CRITERION_3_RHO}",
            "spearman": rho,
            "passed": None if rho is None else abs(rho) < CRITERION_3_RHO,
        },
        "4_taxonomy_ordering": {
            "statement": f"Mean D below the unflagged control on >= "
                         f"{CRITERION_4_MIN_CATEGORIES} of 5 categories "
                         f"(corroboration only, not a gate)",
            "n_below_control": n_below,
            "passed": None if n_below is None else n_below >= CRITERION_4_MIN_CATEGORIES,
            "caveat": TAXONOMY_CAVEAT,
        },
        "5_product_form_earns_itself": {
            "statement": f"D's Tier-1 AUC exceeds its best single component by >= "
                         f"{CRITERION_5_MARGIN}",
            "D_auc": d_auc,
            "best_component": best_component,
            "best_component_auc": best_component_auc,
            "margin": (
                None if (d_auc is None or best_component_auc is None)
                else d_auc - best_component_auc
            ),
            "passed": None if (d_auc is None or best_component_auc is None)
            else (d_auc - best_component_auc) >= CRITERION_5_MARGIN,
        },
    }
    decided = [c["passed"] for c in criteria.values() if c["passed"] is not None]
    all_passed = bool(decided) and all(decided)

    if all_passed:
        overall = "VIABLE -- D is a viable core method for RQ2 on this split"
    elif criteria["5_product_form_earns_itself"]["passed"] is False:
        overall = ("NEEDS ANOTHER ITERATION -- the product form is not justified; a single "
                   "component matches or beats D")
    elif d_auc is not None and 0.6 <= d_auc < CRITERION_1_AUC:
        overall = ("NEEDS ANOTHER ITERATION -- better than KDA_cont but not decisive")
    else:
        overall = "NEEDS ANOTHER ITERATION"
    if spread is not None and spread > 0.2:
        overall += f" (per-solver AUC spread {spread:.3f} > 0.2: the kappa = 0.046 " \
                   "solver-dependence dominates)"

    return {
        "adjudicated_on": payload["tag"],
        "criteria": criteria,
        "n_criteria_passed": sum(1 for v in decided if v),
        "n_criteria_decided": len(decided),
        "overall": overall,
    }


def evaluate(path: str) -> Dict:
    with open(resolve_path(path), encoding="utf-8") as handle:
        payload = json.load(handle)
    tag = payload["tag"]
    dataset = "obqa" if "obqa" in tag else "sciq"
    single_solver = len(payload["solvers"]) == 1

    t1 = tier1(payload)
    t2 = tier2(payload)
    t3 = None
    try:
        t3 = tier3(payload, dataset)
    except (FileNotFoundError, KeyError) as exc:  # taxonomy not available for this split
        t3 = {"unavailable": str(exc), "caveat": TAXONOMY_CAVEAT}

    out = {
        "tag": tag,
        "source_scores_file": path,
        "n_eligible": payload["n_eligible"],
        "eligibility_rate": payload["eligibility_rate"],
        "solvers": payload["solvers"],
        "single_solver_run": single_solver,
        "tier1_setting_c_discrimination": t1,
        "tier2_ab_structure": t2,
        "tier3_manual_taxonomy": t3,
        "d_vs_d_prime": d_vs_d_prime(payload),
        "term1_only_divergence_by_cf_class": payload["term1_only_divergence_by_cf_class"],
        "pair_census": payload["pair_census"],
    }
    if single_solver:
        out["single_solver_note"] = (
            "|M| = 1. There is no ensemble to aggregate and no KDA_cont on this run "
            "(the (1 - P) weight cancels at |M| = 1). Read this split only for the "
            "D vs D' saturation comparison."
        )
    out["verdict"] = verdict(payload, t1, t2, t3 if t3 and "rows" in t3 else None)
    return out


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", nargs="+", default=list(DEFAULT_SCORE_FILES))
    parser.add_argument("--out", default="results/ex6_psfd_score/psfd_evaluation.json")
    args = parser.parse_args()

    evaluations = {}
    for path in args.scores:
        report = evaluate(path)
        evaluations[report["tag"]] = report

        print(f"\n=== {report['tag']} (n_eligible={report['n_eligible']}) ===")
        t1 = report["tier1_setting_c_discrimination"]["ensemble_all"]
        print("  Tier 1 AUC(context_dependent vs prior_dependent), ensemble:")
        for signal in SIGNALS:
            cell = t1[signal]
            print(f"    {signal:12s} {_fmt(cell['auc'])}"
                  f"   (n+={cell['n_positive']}, n-={cell['n_negative']})")
        comparison = report["d_vs_d_prime"]["ensemble_all"]
        print(f"  D vs D': mean D={_fmt(comparison['mean_D'])} "
              f"mean D'={_fmt(comparison['mean_D_prime'])} "
              f"gate attenuation={_fmt(comparison['gate_attenuation'], 3)} "
              f"zero-D={_fmt(comparison['frac_D_zero'], 3)}")
        print(f"  Verdict: {report['verdict']['overall']} "
              f"({report['verdict']['n_criteria_passed']}"
              f"/{report['verdict']['n_criteria_decided']} decided criteria passed)")

    out_path = ensure_parent(resolve_path(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "what": "Evaluation of the P/S/F/D formulation against existing ground "
                        "truth. Derived entirely from committed Setting-C runs.",
                "primary_split": PRIMARY_TAG,
                "pre_registered_criteria": {
                    "1_tier1_auc": CRITERION_1_AUC,
                    "2_within_both_correct_auc": CRITERION_2_AUC,
                    "3_max_spearman_vs_kda_cont": CRITERION_3_RHO,
                    "4_min_categories_below_control": CRITERION_4_MIN_CATEGORIES,
                    "5_component_margin_auc": CRITERION_5_MARGIN,
                },
                "evaluations": evaluations,
            },
            handle, ensure_ascii=False, indent=1,
        )
    print(f"\nWrote {os.path.relpath(out_path, resolve_path('.'))}")


if __name__ == "__main__":
    main()
