"""Re-measure the SciQ prior-dependence figures with the leaked items removed (ex7 #4).

`rescan_residual_leakage.py` finds the counterfactual passages where the gold answer is
still effectively stated, which ex2's whole-word `residual_answer_mentions` check missed.
Any Setting-C label on such an item is untrustworthy in one specific direction: a solver
that answers gold there may simply be reading the passage correctly, and gets scored
`prior_dependent` for it. The published prior-dependence rates are therefore an upper
bound.

This script recomputes ex2's per-model metrics with those items dropped, using the ex2
metric function itself so the corrected figures are produced by exactly the code that
produced the published ones. Three exclusion sets are reported:

    strict        the rescan's `flagged_strict` set -- gold still stated in a sentence
                  that answers the question. The primary correction.
    adjudicated   the subset of `strict` that a hand read confirms is a leak (see
                  ADJUDICATION below). A conservative lower bound on the correction.
    any           every `flagged_any` item, including mentions far from the answering
                  clause. An upper bound on the correction.

Read-only with respect to every committed run; writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ex2_counterfactual.run_counterfactual_experiment import (  # noqa: E402
    ENSEMBLE_KEY,
    compute_model_metrics,
)

# --- hand adjudication of the 30 strict flags -----------------------------------------
# This is *my* single reading of the 30 flagged passages, recorded so the conservative
# bound is reproducible. It is not an independent human annotation and must not be
# reported as one; the only rater-verified items in this list are the five the ex8 Q2
# annotators reached independently (0, 25, 222, 716, 805).
#
# CONFIRMED  the gold answer still stands in the sentence that answers the question, or
#            in a clause that states the same fact.
# BORDERLINE gold survives in a topical or adjacent clause; whether a reader would answer
#            gold from it is genuinely unclear.
# SPURIOUS   the residual is a question-stem echo, a generic head noun, or the gold string
#            buried inside an unrelated longer word.
ADJUDICATION: Dict[int, str] = {
    0: "CONFIRMED",     # "are calledoxidants" survives fused; only the title was rewritten
    25: "CONFIRMED",    # "deficiencies of the immune systems"
    40: "SPURIOUS",     # residual "voltage" is the question's own stem, not the answer slot
    57: "SPURIOUS",     # residual "melts" is the question's own stem
    93: "BORDERLINE",   # "ovary" survives in a section-header list of organs
    146: "SPURIOUS",    # "muscle fiber" without "skeletal" -- the discriminating word is gone
    178: "CONFIRMED",   # "Each lung is enclosed within a cavity ... surrounded by the pleura"
    207: "BORDERLINE",  # "neuron cell subtypes" survives in the same sentence as the target
    209: "SPURIOUS",    # "central nervous system" is not the gold "peripheral nervous system"
    222: "CONFIRMED",   # "dependent upon the chemical identity of the solute"
    258: "BORDERLINE",  # "specific metabolic pathways" survives in a low-overlap sentence
    272: "BORDERLINE",  # "actual charge" survives, but the question quotes that sentence
    348: "BORDERLINE",  # gold "cuticle" survives only inside "exocuticle"
    404: "CONFIRMED",   # "Amino acids are the fundamental building blocks of proteins"
    418: "CONFIRMED",   # "A vector is an organism that carries pathogens from one person"
    426: "CONFIRMED",   # "Colonies may include millions of individual insects"
    470: "CONFIRMED",   # heading "Production of Vaccines" plus "vaccination strategies"
    473: "CONFIRMED",   # "Although truly isolated systems are not really possible ..."
    522: "BORDERLINE",  # "12 hours" survives inside a counterfactual conditional
    557: "CONFIRMED",   # "bursae found at or near the shoulder, hip, knee, or elbow joints"
    628: "CONFIRMED",   # "Traits of arthropods include three body segments ..."
    659: "BORDERLINE",  # "ecosystem" survives in the tail of the rewritten sentence
    716: "CONFIRMED",   # "An antioxidant is a molecule that inhibits the oxidation ..."
    750: "BORDERLINE",  # "turns a driveshaft" -- the substitution also produced nonsense
    752: "SPURIOUS",    # gold "polar" survives only inside "nonpolar"
    774: "CONFIRMED",   # heading "Sebaceous Glands" plus "Most sebaceous glands are ..."
    805: "CONFIRMED",   # "Certain air pollutants form acids when dissolved in water"
    807: "CONFIRMED",   # "Atoms of different elements differ in size, mass ..."
    814: "CONFIRMED",   # "Earth's gravitational force causes the Moon to orbit Earth"
    824: "BORDERLINE",  # "new technologies" survives in the tail of the rewritten sentence
}


def _metrics_for(records: Sequence[Dict], drop: Sequence[int]) -> Dict[str, Dict]:
    dropped = set(drop)
    kept = [r for r in records if r["id"] not in dropped]
    eligible = [r for r in kept if r.get("counterfactual_valid")]
    targets = list(kept[0]["per_model"].keys()) + [ENSEMBLE_KEY]
    out = {}
    for target in targets:
        metrics = compute_model_metrics(kept, target, eligible)
        out[target] = {
            "n_eligible": metrics["n_counterfactual_eligible"],
            "classes_all_eligible": {
                name: block["count"]
                for name, block in metrics["counterfactual_classes_all_eligible"].items()
            },
            "prior_dependent_rate_all_eligible": (
                metrics["counterfactual_classes_all_eligible"]["prior_dependent"]["count"]
                / metrics["n_counterfactual_eligible"]
            ),
            "n_both_correct_eligible": metrics["both_correct_analysis"][
                "n_both_correct_eligible"
            ],
            "both_correct_prior_dependent": metrics["both_correct_analysis"][
                "prior_dependent"
            ]["count"],
            "both_correct_prior_dependent_pct": metrics["both_correct_analysis"][
                "prior_dependent"
            ]["percentage"],
            "both_correct_context_dependent": metrics["both_correct_analysis"][
                "context_dependent"
            ]["count"],
            "both_correct_context_dependent_pct": metrics["both_correct_analysis"][
                "context_dependent"
            ]["percentage"],
            "context_verified_accuracy_with_fact": metrics[
                "context_verified_accuracy_with_fact"
            ],
            "raw_accuracy_with_fact_on_eligible": metrics[
                "raw_accuracy_with_fact_on_eligible"
            ],
            "prior_inflation_of_accuracy_with_fact": metrics[
                "prior_inflation_of_accuracy_with_fact"
            ],
            "counterfactual_passage_vs_cf_target": metrics["accuracy"][
                "counterfactual_passage_vs_cf_target"
            ],
        }
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 6), round(centre + half, 6)]


def _enrichment(
    eligible: Sequence[Dict], subsets: Dict[str, Sequence[int]], unflagged: Sequence[int]
) -> Dict:
    """Is a flagged item more likely to be labelled prior_dependent than an unflagged one?

    This is the direct test of the mechanism. The aggregate correction above is small
    because the flagged set is small; the rate *within* the flagged set is the evidence
    that the construction defect manufactures prior_dependent labels.
    """
    targets = list(eligible[0]["per_model"].keys()) + [ENSEMBLE_KEY]
    keep = set(unflagged)
    out: Dict[str, Dict] = {}
    for target in targets:
        view = (
            (lambda r: r["ensemble"])
            if target == ENSEMBLE_KEY
            else (lambda r, t=target: r["per_model"][t])
        )
        block = {}
        for name, ids in list(subsets.items()) + [("unflagged", sorted(keep))]:
            chosen = set(ids)
            rows = [r for r in eligible if r["id"] in chosen]
            hits = sum(1 for r in rows if view(r)["counterfactual_class"] == "prior_dependent")
            block[name] = {
                "n": len(rows),
                "prior_dependent": hits,
                "rate": round(hits / len(rows), 6) if rows else None,
                "wilson95": _wilson(hits, len(rows)),
            }
        out[target] = block
    return out


def _delta(original: Dict, corrected: Dict) -> Dict:
    out = {}
    for key, value in original.items():
        other = corrected[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = {
                "original": value,
                "corrected": other,
                "delta": round(other - value, 6),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
    )
    parser.add_argument(
        "--rescan",
        default="results/ex7_counterfactual_construction/residual_leakage_rescan_sciq.json",
    )
    parser.add_argument(
        "--out",
        default="results/ex7_counterfactual_construction/prior_dependence_corrected_sciq.json",
    )
    args = parser.parse_args()

    run = json.load(open(args.run, encoding="utf-8"))
    rescan = json.load(open(args.rescan, encoding="utf-8"))
    records = run["results"]

    strict = rescan["report"]["flagged_strict_ids"]
    any_ids = rescan["report"]["flagged_any_ids"]
    assert set(ADJUDICATION) == set(strict), (
        "ADJUDICATION must cover exactly the strict flags",
        sorted(set(strict) - set(ADJUDICATION)),
        sorted(set(ADJUDICATION) - set(strict)),
    )
    confirmed = sorted(i for i, v in ADJUDICATION.items() if v == "CONFIRMED")

    exclusions = {
        "none_published": [],
        "adjudicated_confirmed": confirmed,
        "strict": sorted(strict),
        "any": sorted(any_ids),
    }

    # Faithfulness check: with nothing dropped the recomputation must reproduce the
    # committed per-model metrics exactly.
    baseline = _metrics_for(records, [])
    for target, block in baseline.items():
        published = run["summary"]["per_model_metrics"][target]
        assert block["n_eligible"] == published["n_counterfactual_eligible"], target
        for key in (
            "context_verified_accuracy_with_fact",
            "raw_accuracy_with_fact_on_eligible",
            "prior_inflation_of_accuracy_with_fact",
        ):
            assert abs(block[key] - published[key]) < 1e-12, (target, key)
        assert (
            block["both_correct_prior_dependent"]
            == published["both_correct_analysis"]["prior_dependent"]["count"]
        ), target

    computed = {name: _metrics_for(records, ids) for name, ids in exclusions.items()}

    eligible_records = [r for r in records if r.get("counterfactual_valid")]
    unflagged = [r["id"] for r in eligible_records if r["id"] not in set(any_ids)]

    report = {
        "inputs": {"run": args.run, "rescan": args.rescan},
        "exclusion_sets": {
            name: {"n": len(ids), "ids": ids} for name, ids in exclusions.items()
        },
        "adjudication_counts": {
            verdict: sum(1 for v in ADJUDICATION.values() if v == verdict)
            for verdict in ("CONFIRMED", "BORDERLINE", "SPURIOUS")
        },
        "faithfulness_check": "recomputation with no exclusions reproduces the committed "
        "per-model metrics exactly (asserted)",
        "metrics": computed,
        "prior_dependent_enrichment": _enrichment(
            eligible_records,
            {"flagged_strict": sorted(strict), "adjudicated_confirmed": confirmed},
            unflagged,
        ),
        "deltas_vs_published": {
            name: {
                target: _delta(baseline[target], computed[name][target])
                for target in baseline
            }
            for name in ("adjudicated_confirmed", "strict", "any")
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(report["exclusion_sets"], indent=1))
    print(json.dumps(report["adjudication_counts"], indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
