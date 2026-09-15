"""P / S / F / D dependency score — pure functions, no I/O and no CLI.

Everything here is a function of numbers that are already committed in a Setting-C
results file, so nothing in this module can trigger model inference.

The four signals, per solver `k` and question `q`, with `g` the gold option index and
`c` the counterfactual target index:

    P = p(gold | A)                      prior answerability   (Setting A: q + options)
    S = p(gold | B)                      material sufficiency  (Setting B: + gold passage)
    F = 0.5 * [ (p(gold|B) - p(gold|C))          <- term1: mass leaves the gold answer
              + (p(cf|C)  - p(cf|B))   ]         <- term2: mass arrives on the cf target
                                         counterfactual sensitivity

    D  = (1 - P) * S * max(F, 0)         the scalar score as specified
    D' =           S * max(F, 0)         the no-prior-gate variant

`D` and `D'` are two NAMED VARIANTS reported side by side, not a metric and a fallback.
The reason is measured, not hypothetical: on the saturated Qwen3-4B run 94.2% of
counterfactual-eligible items have `(1 - P) < 0.01`, so `D` is driven to ~0 across almost
the whole split. That is the same ignorance-mass collapse RQ1 identified in KDA_cont's
`(1 - P(R^q=1))` weight, reappearing inside the metric proposed to repair it. `D'` drops
the gate and isolates how much of D's behaviour came from it. Which one is right is a
formulation question that the comparison exists to inform; this module takes no side.

`term1` and `term2` are returned separately because `F > 0` can be produced by `term1`
alone — probability leaving the gold answer without arriving on the counterfactual
target. Those are `unstable_other`-shaped movements, which every committed ex2 estimator
scores as a failure under the strict convention. Counting that divergence is the point of
keeping the terms; no `min(term1, term2)` variant is implemented here.

Schema note. Two committed Setting-C layouts are supported by `solver_views`:

    per-model  `record["per_model"][name]["probabilities_{no,original,counterfactual}_passage"]`
               (the KDA_small 4-encoder runs, SciQ and OpenBookQA)
    flat       `record["probabilities_*"]` directly on the record
               (the |M| = 1 Qwen3-4B run)

`p(cf | B)` is the one quantity no run stores as a scalar; it is read out of the Setting-B
probability vector at the counterfactual target index, which every record carries.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Optional, Sequence

# The near-uniform member of the KDA_small ensemble. On SciQ its Setting-B probability
# vector has a spread below 0.05 on 607 of 860 eligible items, so it contributes a nearly
# constant ~0 to any mean over solvers. Every ensemble figure is therefore reported twice,
# with and without it, rather than picking one.
ALBERT = "Riiid/kda-albert-xlarge-v2-race"

VARIANTS = ("D", "D_prime")


# ---------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------


def solver_views(record: Dict, flat_solver_name: Optional[str] = None) -> Dict[str, Dict]:
    """Return `{solver_name: {probabilities_*}}` for either committed layout.

    `flat_solver_name` names the single solver of a flat-layout record; it is ignored when
    the record carries a `per_model` block.
    """
    if "per_model" in record:
        return dict(record["per_model"])
    if flat_solver_name is None:
        raise ValueError("flat-layout record needs an explicit solver name")
    return {flat_solver_name: record}


def compute_psfd(view: Dict, gold_idx: int, cf_idx: int) -> Dict[str, float]:
    """P/S/F/D for one (solver, question) pair, read straight off the three vectors."""
    if gold_idx == cf_idx:
        raise ValueError(
            f"counterfactual target index {cf_idx} equals the gold index; the "
            "perturbation would assert the answer it is supposed to displace"
        )
    p_a = view["probabilities_no_passage"]
    p_b = view["probabilities_original_passage"]
    p_c = view["probabilities_counterfactual_passage"]

    prior = p_a[gold_idx]
    sufficiency = p_b[gold_idx]
    term1 = p_b[gold_idx] - p_c[gold_idx]   # gold mass released by the perturbation
    term2 = p_c[cf_idx] - p_b[cf_idx]       # cf-target mass gained by the perturbation
    f_raw = 0.5 * (term1 + term2)
    f_clamped = max(f_raw, 0.0)

    return {
        "P": prior,
        "S": sufficiency,
        "F": f_clamped,
        "F_raw": f_raw,
        "term1": term1,
        "term2": term2,
        # `F > 0` carried by term1 alone: gold mass left, but not for the cf target.
        "term1_only_positive": bool(f_raw > 0.0 and term1 > 0.0 and term2 <= 0.0),
        "D": (1.0 - prior) * sufficiency * f_clamped,
        "D_prime": sufficiency * f_clamped,
        "p_gold_C": p_c[gold_idx],
        "p_cf_B": p_b[cf_idx],
        "p_cf_C": p_c[cf_idx],
    }


# ---------------------------------------------------------------------------------------
# Aggregation over solvers
# ---------------------------------------------------------------------------------------


def aggregate(
    values: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> Dict[str, Optional[float]]:
    """Mean over solvers, with spread and a percentile bootstrap interval.

    The interval is a bootstrap over an ensemble of FOUR members, which is far too small
    for the CI to be read as a population interval. It is reported because the Setting-C
    label agrees across solvers at only Cohen's kappa = 0.046, so the spread between
    solvers is a first-class result rather than noise around a point estimate. `sd`,
    `min` and `max` are the honest summaries; the CI is a convenience.
    """
    values = list(values)
    if not values:
        return {k: None for k in ("mean", "sd", "min", "max", "ci_lo", "ci_hi", "n")}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {
            "mean": mean, "sd": 0.0, "min": mean, "max": mean,
            "ci_lo": mean, "ci_hi": mean, "n": 1,
        }
    rng = random.Random(seed)
    boots = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot)
    )
    return {
        "mean": mean,
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "ci_lo": boots[int(0.025 * n_boot)],
        "ci_hi": boots[min(int(0.975 * n_boot), n_boot - 1)],
        "n": len(values),
    }


# ---------------------------------------------------------------------------------------
# Statistics (pure; the AUC convention is the one ex2 already uses)
# ---------------------------------------------------------------------------------------


def auc(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    """P(random positive > random negative), ties counted as 0.5.

    Rank-based (Mann-Whitney U). Byte-for-byte the convention of
    `run_counterfactual_experiment._auc`, so the numbers here are directly comparable to
    the committed `discrimination_auc_kept_vs_dropped` figures. 0.5 = no information.
    """
    if not positive or not negative:
        return None
    merged = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank_sum_positive = 0.0
    index = 0
    while index < len(merged):
        end = index
        while end < len(merged) and merged[end][0] == merged[index][0]:
            end += 1
        average_rank = (index + end + 1) / 2.0  # 1-based mean rank across the tie block
        for position in range(index, end):
            if merged[position][1] == 1:
                rank_sum_positive += average_rank
        index = end
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + end + 1) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, tie-corrected (Pearson on average ranks)."""
    if len(xs) != len(ys):
        raise ValueError("spearman needs equal-length inputs")
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return None if den == 0 else num / den
