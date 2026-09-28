"""Human Q2 reading versus solver Setting-C behaviour, per the fixed annotator-2 analysis plan.

Implements docs/ex8_independent_labels/analysis_plan_annotator2.md section 5.2 exactly. The
statistics were first computed ad hoc for annotator 1 (rq2_annotation_results.md section B4); this
script fixes them as code so annotator 2's sheet goes through the identical procedure. Run with
`--verify-published` it asserts that it reproduces annotator 1's published figures before it may be
trusted on anything else.

Read-only. No model inference. Inputs: a filled Q2 sheet, the committed ex2 SciQ Setting-C run (for
the gold and counterfactual-target indices), the committed ex6 score file (for per-solver and
ensemble Setting-C classes and the metric values), and the build manifest (for the 55 stratum-A
items). The human reading is never combined with another rater's.

Definitions, verbatim from the plan:

  human reading   `reads CF target` if the chosen option index equals counterfactual_target_idx;
                  `reads gold` if it equals answer_idx; `none` if `none`; `reads third option`
                  otherwise. Blank or invalid answers are excluded and listed.
  prior-override  among (model, question) pairs on items read as the CF target, the share whose
                  per-solver Setting-C class is prior_dependent. Wilson 95% CI.
  artifact share  among pair-level prior_dependent labels on the 55 items, the share on items NOT
                  read as the CF target. Wilson 95% CI. The ensemble-level count is reported
                  alongside, without a verdict.
  consistency     (comparison mode) two raters are consistent on a finding when their 95% CIs
                  overlap. Counts are never pooled.

The Q2 AUC (F, D, KDA_cont separating CF-read from gold-read items) is computed and reported but is
NOT a replication test: the gold-read class was n = 5 for annotator 1, and one or two items move it
arbitrarily.

Usage:
    python code/ex8_independent_labels/q2_crosstab.py --q2 <filled_q2.csv> [--out report.json]
    python code/ex8_independent_labels/q2_crosstab.py --q2 <a.csv> --compare <b.csv> [--out ...]
    python code/ex8_independent_labels/q2_crosstab.py --q2 <annotator1.csv> --verify-published
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex8_independent_labels.compare_metrics import BOOTSTRAP_SEED, N_BOOTSTRAP, auc
from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

DEFAULT_MANIFEST = "results/ex8_independent_labels/annotation_manifest.json"
DEFAULT_RESULTS = "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json"
DEFAULT_SCORES = "results/ex6_psfd_score/psfd_scores_sciq_test_full.json"

READINGS = ("reads CF target", "reads gold", "reads third option", "none")
CLASSES = ("context_dependent", "prior_dependent", "unstable_other")
VALID = ("a", "b", "c", "d", "none")
LETTERS = "abcd"

# Annotator 1's published figures (rq2_annotation_results.md section B4). `--verify-published`
# must reproduce these exactly before this script is used on a second rater.
PUBLISHED_ANNOTATOR1 = {
    "prior_override": (40, 200),
    "artifact_share_pairs": (17, 57),
    "artifact_ensemble": (5, 16),
    "readings": {"reads CF target": 50, "reads gold": 5},
}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval -- the same formula used for the published annotator-1 figures."""
    if n == 0:
        return None, None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def _rate(k: int, n: int) -> Dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def read_q2(path: str) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
    """item_id -> answer for valid rows, plus (item_id, raw) for blank or invalid rows."""
    with open(resolve_path(path), encoding="utf-8") as handle:
        body = [line for line in handle if not line.startswith("#")]
    answers: Dict[int, str] = {}
    invalid: List[Tuple[int, str]] = []
    for row in csv.DictReader(body):
        raw = str(row.get("stated_answer", "")).strip().lower()
        if raw in VALID:
            answers[int(row["item_id"])] = raw
        else:
            invalid.append((int(row["item_id"]), raw))
    return answers, invalid


def reading_of(answer: str, record: Dict) -> str:
    if answer == "none":
        return "none"
    chosen = LETTERS.index(answer)
    if chosen == record["counterfactual_target_idx"]:
        return "reads CF target"
    if chosen == record["answer_idx"]:
        return "reads gold"
    return "reads third option"


def _auc_with_ci(pos: Sequence[float], neg: Sequence[float]) -> Dict:
    point = auc(pos, neg)
    if point is None:
        return {"auc": None, "ci95": [None, None], "n_pos": len(pos), "n_neg": len(neg)}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(N_BOOTSTRAP):
        p = [pos[rng.randrange(len(pos))] for _ in pos]
        n = [neg[rng.randrange(len(neg))] for _ in neg]
        value = auc(p, n)
        if value is not None:
            draws.append(value)
    draws.sort()
    return {"auc": point,
            "ci95": [draws[int(0.025 * len(draws))], draws[min(int(0.975 * len(draws)), len(draws) - 1)]],
            "n_pos": len(pos), "n_neg": len(neg)}


def load_context(manifest_path: str, results_path: str, scores_path: str):
    with open(resolve_path(manifest_path), encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = sorted(int(k) for k, v in manifest["stratum_membership"].items() if v == "A")
    if len(items) != manifest["q2"]["n"]:
        raise SystemExit("manifest stratum A size disagrees with the Q2 sheet size; refusing")
    with open(resolve_path(results_path), encoding="utf-8") as handle:
        records = {r["id"]: r for r in json.load(handle)["results"]}
    with open(resolve_path(scores_path), encoding="utf-8") as handle:
        payload = json.load(handle)
    return items, records, {i["id"]: i for i in payload["items"]}, payload["solvers"]


def analyse(q2_path: str, items: Sequence[int], records: Dict, scores: Dict,
            solvers: Sequence[str]) -> Dict:
    answers, invalid = read_q2(q2_path)
    unknown = sorted(set(answers) - set(items))
    if unknown:
        raise SystemExit(f"{q2_path}: item ids outside stratum A: {unknown}; refusing")
    readings = {i: reading_of(answers[i], records[i]) for i in items if i in answers}

    pair_table = {r: collections.Counter() for r in READINGS}
    ensemble_table = {r: collections.Counter() for r in READINGS}
    for item_id, reading in readings.items():
        for solver in solvers:
            pair_table[reading][scores[item_id]["per_solver"][solver]["counterfactual_class"]] += 1
        ensemble_table[reading][scores[item_id]["ensemble_counterfactual_class"]] += 1

    cf_items = [i for i, r in readings.items() if r == "reads CF target"]
    override_k = sum(1 for i in cf_items for s in solvers
                     if scores[i]["per_solver"][s]["counterfactual_class"] == "prior_dependent")
    prior_pairs = [(i, s) for i in readings for s in solvers
                   if scores[i]["per_solver"][s]["counterfactual_class"] == "prior_dependent"]
    artifact_pairs = [(i, s) for i, s in prior_pairs if readings[i] != "reads CF target"]
    ens_prior = [i for i in readings if scores[i]["ensemble_counterfactual_class"] == "prior_dependent"]
    ens_artifact = [i for i in ens_prior if readings[i] != "reads CF target"]

    per_solver = {}
    for solver in solvers:
        k = sum(1 for i in cf_items if scores[i]["per_solver"][solver]["counterfactual_class"] == "prior_dependent")
        per_solver[solver] = _rate(k, len(cf_items))

    coincide = sum(1 for i, r in readings.items()
                   if (r == "reads CF target" and scores[i]["ensemble_counterfactual_class"] == "context_dependent")
                   or (r == "reads gold" and scores[i]["ensemble_counterfactual_class"] == "prior_dependent"))

    gold_items = [i for i, r in readings.items() if r == "reads gold"]
    metric = {
        "F": lambda i: scores[i]["ensemble"]["all"]["F_mean"],
        "D": lambda i: scores[i]["ensemble"]["all"]["D"]["mean"],
        "KDA_cont": lambda i: scores[i]["kda_cont"],
    }
    q2_auc = {name: _auc_with_ci([fn(i) for i in cf_items], [fn(i) for i in gold_items])
              for name, fn in metric.items()}

    return {
        "sheet": q2_path,
        "n_items": len(items),
        "n_answered": len(readings),
        "invalid_or_blank": [{"item_id": i, "raw": raw} for i, raw in invalid],
        "reading_counts": {r: sum(1 for v in readings.values() if v == r) for r in READINGS},
        "pairs_by_reading": {r: {c: pair_table[r][c] for c in CLASSES} for r in READINGS},
        "ensemble_by_reading": {r: {c: ensemble_table[r][c] for c in CLASSES} for r in READINGS},
        "prior_override": _rate(override_k, len(cf_items) * len(solvers)),
        "prior_override_per_solver": per_solver,
        "artifact_share_pairs": _rate(len(artifact_pairs), len(prior_pairs)),
        "artifact_ensemble": {"k": len(ens_artifact), "n": len(ens_prior),
                              "items": sorted(ens_artifact),
                              "note": "reported alongside, no consistency verdict (plan 5.2)"},
        "views_coincide": {"k": coincide, "n": len(readings),
                           "note": "descriptive: CF-read & context_dependent, or gold-read & "
                                   "prior_dependent, at ensemble level"},
        "q2_auc_cf_vs_gold": q2_auc,
        "q2_auc_note": "NOT a replication test -- the gold-read class is tiny and moves arbitrarily.",
        "_readings": {str(i): r for i, r in readings.items()},
    }


def _overlap(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> Optional[bool]:
    if None in a or None in b:
        return None
    return a[0] <= b[1] and b[0] <= a[1]


def compare(rep_a: Dict, rep_b: Dict, label_a: str, label_b: str) -> Dict:
    ra = {int(k): v for k, v in rep_a["_readings"].items()}
    rb = {int(k): v for k, v in rep_b["_readings"].items()}
    shared = sorted(set(ra) & set(rb))
    differ = [{"item_id": i, label_a: ra[i], label_b: rb[i]} for i in shared if ra[i] != rb[i]]
    out = {
        "n_items_both_answered": len(shared),
        "item_agreement_four_way": ((len(shared) - len(differ)) / len(shared)) if shared else None,
        "items_that_differ": differ,
        "consistency": {},
        "rule": "consistent = the two raters' 95% CIs overlap; counts are never pooled",
    }
    for finding in ("prior_override", "artifact_share_pairs"):
        a, b = rep_a[finding], rep_b[finding]
        out["consistency"][finding] = {
            label_a: {"k": a["k"], "n": a["n"], "rate": a["rate"], "ci95": a["ci95"]},
            label_b: {"k": b["k"], "n": b["n"], "rate": b["rate"], "ci95": b["ci95"]},
            "consistent": _overlap(a["ci95"], b["ci95"]),
        }
    out["artifact_ensemble_side_by_side"] = {
        label_a: [rep_a["artifact_ensemble"]["k"], rep_a["artifact_ensemble"]["n"]],
        label_b: [rep_b["artifact_ensemble"]["k"], rep_b["artifact_ensemble"]["n"]],
        "note": "no verdict (plan 5.2)",
    }
    return out


def verify_published(rep: Dict) -> List[str]:
    failures = []
    for key in ("prior_override", "artifact_share_pairs", "artifact_ensemble"):
        got = (rep[key]["k"], rep[key]["n"])
        if got != PUBLISHED_ANNOTATOR1[key]:
            failures.append(f"{key}: got {got[0]}/{got[1]}, published {PUBLISHED_ANNOTATOR1[key][0]}/{PUBLISHED_ANNOTATOR1[key][1]}")
    for reading, count in PUBLISHED_ANNOTATOR1["readings"].items():
        if rep["reading_counts"][reading] != count:
            failures.append(f"readings[{reading}]: got {rep['reading_counts'][reading]}, published {count}")
    return failures


def _fmt_rate(cell: Dict) -> str:
    if cell["rate"] is None:
        return f"{cell['k']}/{cell['n']}  (undefined)"
    lo, hi = cell["ci95"]
    return f"{cell['k']}/{cell['n']} = {cell['rate']:.1%}  [{lo:.1%}, {hi:.1%}]"


def _print(rep: Dict, label: str) -> None:
    print(f"=== {label}: {rep['n_answered']}/{rep['n_items']} answered"
          + (f", {len(rep['invalid_or_blank'])} blank/invalid" if rep["invalid_or_blank"] else "") + " ===")
    print("  readings:", {k: v for k, v in rep["reading_counts"].items() if v})
    print(f"  {'reading':20s} {'pairs':>6s} {'ctx':>5s} {'prior':>6s} {'unst':>5s}")
    for r in READINGS:
        c = rep["pairs_by_reading"][r]
        n = sum(c.values())
        if n:
            print(f"  {r:20s} {n:6d} {c['context_dependent']:5d} {c['prior_dependent']:6d} {c['unstable_other']:5d}")
    print(f"  prior-override (CF-read pairs)   : {_fmt_rate(rep['prior_override'])}")
    print(f"  artifact share (prior pairs)     : {_fmt_rate(rep['artifact_share_pairs'])}")
    ae = rep["artifact_ensemble"]
    print(f"  artifact, ensemble level         : {ae['k']} of {ae['n']}")
    print(f"  views coincide                   : {rep['views_coincide']['k']}/{rep['views_coincide']['n']}")
    aucs = rep["q2_auc_cf_vs_gold"]
    parts = []
    for name, cell in aucs.items():
        parts.append(f"{name} " + ("n/a" if cell["auc"] is None else f"{cell['auc']:.3f}"))
    print(f"  Q2 AUC (not a replication test)  : {', '.join(parts)}  (n_neg={aucs['F']['n_neg']})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--q2", required=True, help="Filled Q2 sheet (the primary or only rater).")
    parser.add_argument("--compare", default="", help="A second rater's filled Q2 sheet.")
    parser.add_argument("--label-a", default="annotator1")
    parser.add_argument("--label-b", default="annotator2")
    parser.add_argument("--verify-published", action="store_true",
                        help="Assert that --q2 reproduces annotator 1's published figures.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    items, records, scores, solvers = load_context(args.manifest, args.results, args.scores)
    rep_a = analyse(args.q2, items, records, scores, solvers)
    report = {"what": "Human Q2 reading vs solver Setting-C behaviour, per analysis_plan_annotator2.md 5.2",
              args.label_a: rep_a}
    _print(rep_a, args.label_a)

    status = 0
    if args.verify_published:
        failures = verify_published(rep_a)
        report["verify_published"] = {"passed": not failures, "failures": failures}
        print("\nverify-published:", "PASSED -- reproduces 40/200, 17/57, 5 of 16, 50 CF / 5 gold"
              if not failures else "FAILED")
        for f in failures:
            print("  ", f)
        status = 0 if not failures else 1

    if args.compare:
        rep_b = analyse(args.compare, items, records, scores, solvers)
        report[args.label_b] = rep_b
        print()
        _print(rep_b, args.label_b)
        cmp = compare(rep_a, rep_b, args.label_a, args.label_b)
        report["comparison"] = cmp
        agree = cmp["item_agreement_four_way"]
        print(f"\n=== {args.label_a} vs {args.label_b} (counts never pooled) ===")
        print(f"  item agreement, four-way reading : "
              + ("n/a" if agree is None else f"{agree:.1%} of {cmp['n_items_both_answered']}"))
        print(f"  items that differ                : {[d['item_id'] for d in cmp['items_that_differ']]}")
        for finding, cell in cmp["consistency"].items():
            verdict = {True: "CONSISTENT (CIs overlap)", False: "NOT consistent (CIs do not overlap)",
                       None: "undefined (a rate has no denominator)"}[cell["consistent"]]
            print(f"  {finding:22s}: {verdict}")

    if args.out:
        for key in (args.label_a, args.label_b):
            if key in report:
                report[key].pop("_readings", None)
        path = ensure_parent(resolve_path(args.out))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        print(f"\nWrote {os.path.relpath(path, resolve_path('.'))}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
