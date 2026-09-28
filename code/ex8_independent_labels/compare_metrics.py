"""Compare KDA_cont, F, D and D' against the independent human label set. Pre-registered.

Every statistic, threshold and decision below was fixed in the RQ2 close-out plan BEFORE any
label existed, and this file was written and verified against synthetic data before the sheets
were annotated. Nothing here is chosen after seeing labels, and no metric is re-fit or
re-parameterised.

Reads: a filled Q1 sheet (optionally a second annotator's for the kappa gate), optionally the
filled Q2 sheets, the build manifest (for strata and sampling fractions) and the committed ex6
score file. Computes no metric of its own -- KDA_cont, F, D and D' are read as scored.

  PRIMARY      Sign test on strata A and B, per the nine steps in the plan (see `sign_test`).
               Sensitivity: the same test on stratum A alone.
  SECONDARY    AUC of each metric against the human label binarised at 1-2 vs 3-4, reported
               unweighted (high power, not population-representative) and inverse-probability
               weighted by stratum sampling fraction (population estimate, wider interval).
  TERTIARY     Spearman rho between each metric and the raw 1-4 rating, bootstrap CI.
  GATE         Cohen's kappa on the double-annotated block. Below KAPPA_GATE, every comparison
               is reported as provisional and nothing is adopted.
  Q2           What the counterfactual passage actually licenses, by human reading: a validity
               check on the construction and the only non-circular test of F's context-following
               claim available.

Usage:
    python code/ex8_independent_labels/compare_metrics.py --q1 <filled_q1.csv> \
        [--q1-b <second_annotator.csv>] [--q2 <filled_q2.csv>] [--q2-b <second.csv>] \
        [--out results/ex8_independent_labels/metric_comparison.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex8_independent_labels.kappa_independent_labels import KAPPA_GATE, score_q1, score_q2
from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

DEFAULT_MANIFEST = "results/ex8_independent_labels/annotation_manifest.json"
DEFAULT_SCORES = "results/ex6_psfd_score/psfd_scores_sciq_test_full.json"

METRICS = ("KDA_cont", "F", "D", "D_prime")
ALPHA = 0.05
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260916
# Pre-registered in the plan: D must beat F by this much on AUC before the more complex form
# is preferred. Same bar as evaluate_psfd.py's criterion 5, deliberately unchanged.
D_OVER_F_AUC_MARGIN = 0.02


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        mean_rank = (i + j + 1) / 2.0
        for p in range(i, j):
            ranks[order[p]] = mean_rank
        i = j
    return ranks


def _normalised_ranks(values: Sequence[float]) -> List[float]:
    ranks = average_ranks(values)
    n = len(values)
    return [0.0] * n if n < 2 else [(r - 1.0) / (n - 1.0) for r in ranks]


def binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def exact_binomial_two_sided(w: int, m: int) -> Optional[float]:
    if m == 0:
        return None
    return min(1.0, 2.0 * min(binomial_cdf(w, m), 1.0 - binomial_cdf(w - 1, m)))


def clopper_pearson(w: int, m: int, alpha: float = ALPHA) -> Tuple[Optional[float], Optional[float]]:
    """Exact interval by bisection on the binomial CDF; no scipy dependency."""
    if m == 0:
        return None, None
    def solve(target: float, upper: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            value = binomial_cdf(w, m, mid) if upper else 1.0 - binomial_cdf(w - 1, m, mid)
            if (value > target) == upper:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0
    low = 0.0 if w == 0 else solve(alpha / 2.0, False)
    high = 1.0 if w == m else solve(alpha / 2.0, True)
    return low, high


def auc(positive: Sequence[float], negative: Sequence[float],
        w_pos: Optional[Sequence[float]] = None,
        w_neg: Optional[Sequence[float]] = None) -> Optional[float]:
    """P(random positive > random negative), ties 0.5. Weighted when weights are given.

    The unweighted branch uses the same rank form as ex2/ex6 so figures stay comparable; the
    weighted branch is an explicit O(n^2) pair sum, which is fine at n <= 100 and avoids any
    ambiguity about how stratum weights enter.
    """
    if not positive or not negative:
        return None
    if w_pos is None or w_neg is None:
        merged = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
        rank_sum = 0.0
        i = 0
        while i < len(merged):
            j = i
            while j < len(merged) and merged[j][0] == merged[i][0]:
                j += 1
            mean_rank = (i + j + 1) / 2.0
            rank_sum += mean_rank * sum(1 for k in range(i, j) if merged[k][1] == 1)
            i = j
        n_pos, n_neg = len(positive), len(negative)
        return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    numerator = denominator = 0.0
    for p, wp in zip(positive, w_pos):
        for n, wn in zip(negative, w_neg):
            weight = wp * wn
            denominator += weight
            numerator += weight * (1.0 if p > n else (0.5 if p == n else 0.0))
    return numerator / denominator if denominator else None


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def bootstrap_ci(xs: Sequence[float], ys: Sequence[float],
                 n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> Dict:
    point = spearman(xs, ys)
    if point is None:
        return {"rho": None, "ci_lo": None, "ci_hi": None, "n_boot": 0}
    rng = random.Random(seed)
    n = len(xs)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        value = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        if value is not None:
            draws.append(value)
    draws.sort()
    if not draws:
        return {"rho": point, "ci_lo": None, "ci_hi": None, "n_boot": 0}
    return {
        "rho": point,
        "ci_lo": draws[int(0.025 * len(draws))],
        "ci_hi": draws[min(int(0.975 * len(draws)), len(draws) - 1)],
        "n_boot": len(draws),
    }


# ---------------------------------------------------------------------------------------
# The pre-registered sign test, steps 1-9
# ---------------------------------------------------------------------------------------


def sign_test(items: Sequence[Dict], metric_a: str, metric_b: str) -> Dict:
    """Is `metric_b` closer to the human ordering than `metric_a`, item by item?

    Steps, exactly as pre-registered:
      1 restrict to rated, usable items (done by the caller)
      2 orient the human score: d = 5 - rating
      3 average ranks within this set only
      4 normalise ranks to [0, 1]
      5 per-item absolute distance from each metric's normalised rank to the human's
      6 sign: +1 when metric_b is closer, -1 when metric_a is, 0 on an exact tie
      7 drop zeros
      8 two-sided exact binomial test at p = 0.5
      9 report w/m, p, and the 95% Clopper-Pearson interval
    """
    n = len(items)
    if n < 3:
        return {"comparison": f"{metric_b} closer to human than {metric_a}",
                "n": n, "m": 0, "w": None, "proportion_favouring_b": None,
                "ci95": [None, None], "p_value": None, "significant": False,
                "winner": None, "n_exact_ties_dropped": 0,
                "unavailable": "fewer than 3 rated items"}
    u_human = _normalised_ranks([5 - i["rating"] for i in items])
    u_a = _normalised_ranks([i[metric_a] for i in items])
    u_b = _normalised_ranks([i[metric_b] for i in items])

    signs = []
    for k in range(n):
        err_a = abs(u_a[k] - u_human[k])
        err_b = abs(u_b[k] - u_human[k])
        signs.append(1 if err_b < err_a else (-1 if err_a < err_b else 0))

    m = sum(1 for s in signs if s != 0)
    w = sum(1 for s in signs if s == 1)
    low, high = clopper_pearson(w, m)
    p = exact_binomial_two_sided(w, m)
    return {
        "comparison": f"{metric_b} closer to human than {metric_a}",
        "n": n, "m": m, "w": w,
        "proportion_favouring_b": (w / m) if m else None,
        "ci95": [low, high],
        "p_value": p,
        "significant": (p is not None and p < ALPHA),
        "winner": None if (p is None or p >= ALPHA)
                  else (metric_b if w / m > 0.5 else metric_a),
        "n_exact_ties_dropped": n - m,
    }


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------


def read_q1(path: str) -> Tuple[Dict[int, int], List[int]]:
    with open(resolve_path(path), encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    ratings: Dict[int, int] = {}
    unusable: List[int] = []
    for row in csv.DictReader(lines):
        item_id = int(row["item_id"])
        if str(row.get("unusable", "")).strip().upper() == "X":
            unusable.append(item_id)
            continue
        raw = str(row.get("rating", "")).strip()
        if raw in ("1", "2", "3", "4"):
            ratings[item_id] = int(raw)
    return ratings, unusable


def read_q2(path: str) -> Dict[int, str]:
    with open(resolve_path(path), encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    out: Dict[int, str] = {}
    for row in csv.DictReader(lines):
        value = str(row.get("stated_answer", "")).strip().lower()
        if value in ("a", "b", "c", "d", "none"):
            out[int(row["item_id"])] = value
    return out


def load_metrics(scores_path: str) -> Dict[int, Dict[str, float]]:
    with open(resolve_path(scores_path), encoding="utf-8") as handle:
        payload = json.load(handle)
    out = {}
    for item in payload["items"]:
        block = item["ensemble"]["all"]
        out[item["id"]] = {
            "KDA_cont": item["kda_cont"],
            "F": block["F_mean"],
            "D": block["D"]["mean"],
            "D_prime": block["D_prime"]["mean"],
        }
    return out


# ---------------------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------------------


def analyse_q2(answers: Dict[int, str], results_path: str) -> Dict:
    """Did a human reading the counterfactual passage land on the counterfactual target?"""
    with open(resolve_path(results_path), encoding="utf-8") as handle:
        records = {r["id"]: r for r in json.load(handle)["results"]}
    letters = "abcd"
    tally = {"target": 0, "gold": 0, "other_option": 0, "none": 0}
    per_item = {}
    for item_id, answer in answers.items():
        record = records[item_id]
        if answer == "none":
            verdict = "none"
        else:
            chosen = letters.index(answer)
            if chosen == record["counterfactual_target_idx"]:
                verdict = "target"
            elif chosen == record["answer_idx"]:
                verdict = "gold"
            else:
                verdict = "other_option"
        tally[verdict] += 1
        per_item[item_id] = verdict
    n = len(answers)
    return {
        "n": n,
        "counts": tally,
        "rate_reads_as_counterfactual_target": (tally["target"] / n) if n else None,
        "rate_still_reads_as_gold": (tally["gold"] / n) if n else None,
        "rate_no_clear_answer": (tally["none"] / n) if n else None,
        "per_item": per_item,
        "interpretation": (
            "`target` means the perturbed passage licenses the counterfactual answer to a human "
            "reader, so a solver that follows it is genuinely following the context. `gold` or "
            "`none` means the construction failed for that item, independently of any model."
        ),
    }


# ---------------------------------------------------------------------------------------


def decide(primary: Dict, aucs: Dict, gate_passed: Optional[bool]) -> Dict:
    f_auc = aucs.get("F", {}).get("unweighted")
    k_auc = aucs.get("KDA_cont", {}).get("unweighted")
    d_auc = aucs.get("D", {}).get("unweighted")

    if primary.get("p_value") is None:
        outcome = "NO DATA -- the sign test could not be computed"
    elif not primary["significant"]:
        outcome = ("INCONCLUSIVE at this sample size -- neither metric is closer to the human "
                   "ordering often enough to reject chance. Do not adopt either.")
    elif primary["winner"] == "F":
        if f_auc is not None and k_auc is not None and f_auc >= k_auc:
            outcome = "ADOPT F as the RQ2 metric."
        else:
            outcome = ("SPLIT -- F wins the sign test but does not match KDA_cont on AUC. "
                       "Report both; do not adopt.")
    else:
        outcome = ("KDA_cont WINS -- it survives its own critique. Report as a negative result "
                   "for the extension.")

    prefer_d = (d_auc is not None and f_auc is not None
                and (d_auc - f_auc) >= D_OVER_F_AUC_MARGIN)
    # The D-vs-F bar only decides anything when F is the metric in contention. When KDA_cont
    # wins outright, D beating F is a fact about two losing metrics and is reported separately
    # rather than appended to the verdict, where it would read as a recommendation.
    winner_is_f_or_open = primary.get("winner") == "F" or not primary.get("significant")
    if d_auc is not None and f_auc is not None:
        delta = d_auc - f_auc
        if winner_is_f_or_open:
            outcome += (f" D {'clears' if prefer_d else 'does not clear'} the "
                        f"{D_OVER_F_AUC_MARGIN} AUC bar over F (delta {delta:+.3f}), so the "
                        f"{'product' if prefer_d else 'simpler'} form "
                        f"{'is preferred' if prefer_d else 'stands'}.")
        else:
            outcome += (f" (Informational, not a recommendation: D vs F AUC delta "
                        f"{delta:+.3f}; neither is the winning metric here.)")

    # `None` means the gate was never evaluated (no second annotator); `False` means it was
    # evaluated and failed. These are different situations and must not print the same way --
    # the earlier `is False` test let the not-evaluated case through with no caveat at all.
    if gate_passed is None:
        outcome = ("KAPPA GATE NOT EVALUATED -- no second annotator. Results below are "
                   "exploratory, single-rater evidence and do not discharge the "
                   "pre-registered adoption condition. " + outcome)
    elif gate_passed is not True:
        outcome = ("PROVISIONAL -- the kappa gate did not pass, so the label set is not "
                   "reliable enough to license adoption. " + outcome)
    return {"outcome": outcome, "prefer_D_over_F": prefer_d,
            "kappa_gate_passed": gate_passed,
            "kappa_gate_status": ("not_evaluated" if gate_passed is None
                                  else ("passed" if gate_passed else "failed"))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--q1", required=True, help="Annotator 1's filled Q1 sheet.")
    parser.add_argument("--q1-b", default="", help="Annotator 2's filled Q1 sheet (kappa gate).")
    parser.add_argument("--q2", default="", help="Filled Q2 sheet.")
    parser.add_argument("--q2-b", default="", help="Annotator 2's filled Q2 sheet.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--results", default="results/ex2_counterfactual/results_counterfactual_sciq_test_full.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    with open(resolve_path(args.manifest), encoding="utf-8") as handle:
        manifest = json.load(handle)
    strata = {int(k): v for k, v in manifest["stratum_membership"].items()}
    fractions = manifest["sampling_fractions_for_reweighting"]
    metrics = load_metrics(args.scores)
    ratings, unusable = read_q1(args.q1)

    rows = [
        {"id": i, "rating": r, "stratum": strata[i],
         "weight": (1.0 / fractions[strata[i]]) if fractions.get(strata[i]) else 0.0,
         **{m: metrics[i][m] for m in METRICS}}
        for i, r in sorted(ratings.items()) if i in strata and i in metrics
    ]
    sign_eligible = set(manifest["sign_test_eligible_ids"])
    ab = [r for r in rows if r["id"] in sign_eligible]
    a_only = [r for r in rows if r["stratum"] == "A"]

    gate = None
    kappa_block = None
    if args.q1_b:
        kappa_block = score_q1(args.q1, args.q1_b, manifest["q1"]["double_block_ids"])
        gate = kappa_block.get("gate_passed")
        if args.q2 and args.q2_b:
            kappa_block["q2"] = score_q2(args.q2, args.q2_b, manifest["q2"]["double_block_ids"])

    primary = sign_test(ab, "KDA_cont", "F")
    report = {
        "what": "Pre-registered comparison of KDA_cont, F, D and D' against the independent "
                "human label set. No metric is computed or re-fit here.",
        "n_rated": len(rows), "n_unusable": len(unusable), "unusable_ids": sorted(unusable),
        "n_by_stratum": {s: sum(1 for r in rows if r["stratum"] == s) for s in ("A", "B", "C")},
        "kappa_gate": kappa_block,
        "primary_sign_test_A_and_B": primary,
        "sensitivity_sign_test_A_only": sign_test(a_only, "KDA_cont", "F"),
        "sign_test_D_vs_KDA": sign_test(ab, "KDA_cont", "D"),
        "sign_test_D_vs_F": sign_test(ab, "F", "D"),
    }

    aucs: Dict[str, Dict] = {}
    positive = [r for r in rows if r["rating"] <= 2]     # DEPENDENT
    negative = [r for r in rows if r["rating"] >= 3]     # NOT_DEPENDENT
    for metric in METRICS:
        pos_v = [r[metric] for r in positive]
        neg_v = [r[metric] for r in negative]
        # Pre-registered: bootstrap CIs on every AUC as well as every rho. Resampling is
        # stratified by class so both arms stay non-empty.
        rng = random.Random(BOOTSTRAP_SEED)
        draws = []
        for _ in range(N_BOOTSTRAP):
            if not pos_v or not neg_v:
                break
            p_s = [pos_v[rng.randrange(len(pos_v))] for _ in pos_v]
            n_s = [neg_v[rng.randrange(len(neg_v))] for _ in neg_v]
            value = auc(p_s, n_s)
            if value is not None:
                draws.append(value)
        draws.sort()
        aucs[metric] = {
            "unweighted": auc(pos_v, neg_v),
            "unweighted_ci_lo": draws[int(0.025 * len(draws))] if draws else None,
            "unweighted_ci_hi": draws[min(int(0.975 * len(draws)), len(draws) - 1)] if draws else None,
            "ipw_by_stratum": auc(
                pos_v, neg_v,
                [r["weight"] for r in positive], [r["weight"] for r in negative]),
            "n_positive": len(positive), "n_negative": len(negative),
        }
    report["auc_vs_binarised_human"] = aucs
    report["auc_note"] = ("positive class = rating 1-2 (DEPENDENT). Unweighted is the stratified "
                          "sample; ipw_by_stratum reweights to the 860-item population.")
    report["spearman_vs_rating"] = {
        m: bootstrap_ci([r[m] for r in rows], [5 - r["rating"] for r in rows]) for m in METRICS
    }

    if args.q2:
        report["q2_context_following"] = analyse_q2(read_q2(args.q2), args.results)
    report["decision"] = decide(primary, aucs, gate)

    fmt = lambda v, d=4: "n/a" if v is None else f"{v:.{d}f}"
    print(f"rated {report['n_rated']}  (A={report['n_by_stratum']['A']} "
          f"B={report['n_by_stratum']['B']} C={report['n_by_stratum']['C']}, "
          f"{report['n_unusable']} unusable)")
    if kappa_block:
        print(f"kappa gate: collapsed binary "
              f"{fmt(kappa_block['collapsed_binary']['kappa'], 3)} -> "
              f"{'PASSED' if gate else 'NOT PASSED'}")
    print(f"\nPRIMARY sign test (A+B), F vs KDA_cont:")
    print(f"  w/m = {primary['w']}/{primary['m']} = {fmt(primary['proportion_favouring_b'], 3)}"
          f"   p = {fmt(primary['p_value'], 4)}   winner: {primary['winner'] or 'none'}")
    print("\nAUC vs binarised human label (positive = DEPENDENT):")
    for metric in METRICS:
        cell = aucs[metric]
        print(f"  {metric:10s} unweighted {fmt(cell['unweighted'])}"
              f"  [{fmt(cell['unweighted_ci_lo'],3)}, {fmt(cell['unweighted_ci_hi'],3)}]"
              f"   IPW {fmt(cell['ipw_by_stratum'])}")
    print("\nSpearman vs rating (bootstrap 95% CI):")
    for metric in METRICS:
        cell = report["spearman_vs_rating"][metric]
        print(f"  {metric:10s} rho {fmt(cell['rho'], 3)}"
              f"  [{fmt(cell['ci_lo'], 3)}, {fmt(cell['ci_hi'], 3)}]")
    if "q2_context_following" in report:
        q2 = report["q2_context_following"]
        print(f"\nQ2: human reads the counterfactual passage as the CF target in "
              f"{fmt(q2['rate_reads_as_counterfactual_target'], 3)} of {q2['n']} items "
              f"(still gold {fmt(q2['rate_still_reads_as_gold'], 3)}, "
              f"no clear answer {fmt(q2['rate_no_clear_answer'], 3)})")
    print(f"\nDECISION: {report['decision']['outcome']}")

    if args.out:
        path = ensure_parent(resolve_path(args.out))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        print(f"\nWrote {os.path.relpath(path, resolve_path('.'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
