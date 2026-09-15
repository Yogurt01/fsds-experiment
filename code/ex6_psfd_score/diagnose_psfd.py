"""Three follow-up diagnostics on the P/S/F/D evaluation. Analysis only, no new metric.

None of this proposes a formulation. `psfd.py`'s compute logic is untouched and no model
inference is run; every input is a committed Setting-C record or the score files
`compute_psfd_scores.py` already wrote.

  D1  Why is S anti-predictive? Distribution of S split by Setting-C class, to separate
      "near-ceiling in both classes, so the AUC is noise" from "a real inverse relation".
      Reported over all eligible items AND restricted to `both_correct`, because the
      ceiling argument only bites on the latter.

  D2  Ceiling check. Logistic regression on (P, S, F) predicting context_dependent vs
      prior_dependent, to ask whether ANY linear combination of the three signals beats F
      alone. THIS IS NOT A METRIC CANDIDATE -- it is fitted on the labels it is scored
      against, so its in-sample AUC is an upper bound, not a result. A 5-fold
      cross-validated AUC is reported alongside for that reason.

  D3  The `reasoning_shortcut` miss: the SciQ items in that category with the highest
      ensemble F, laid out for reading. No statistics.

  D4  Algebraic follow-on to D2. Regrouping F_raw's four terms by SETTING rather than by
      option gives F_raw = 0.5*(MarginB + MarginC), where MarginB = p(gold|B) - p(c|B)
      uses no Setting-C information and MarginC = p(c|C) - p(gold|C) is the quantity whose
      sign IS the context/prior label. Scoring the two halves separately says how much of
      F's own Tier-1 AUC is definitional.

Usage:
    python code/ex6_psfd_score/diagnose_psfd.py
"""

from __future__ import annotations

import argparse
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

from ex6_psfd_score.psfd import auc
from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

SCORE_FILES = {
    "sciq_test_full": "results/ex6_psfd_score/psfd_scores_sciq_test_full.json",
    "obqa_test_full": "results/ex6_psfd_score/psfd_scores_obqa_test_full.json",
    "obqa_test_exact_tier": "results/ex6_psfd_score/psfd_scores_obqa_test_exact_tier.json",
    "qwen3_4b_sciq": "results/ex6_psfd_score/psfd_scores_qwen3_4b_sciq.json",
}
SCIQ_SOURCE = "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json"
TAXONOMY = "results/ex5_failure_analysis/rq1_flagged_questions.json"
FEATURES = ("P", "S", "F")


def _load(path: str) -> Dict:
    with open(resolve_path(path), encoding="utf-8") as handle:
        return json.load(handle)


def _describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    values = list(values)
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
        "frac_above_0.9": sum(1 for v in values if v > 0.9) / len(values),
    }


def _ens_signal(item: Dict, name: str) -> Optional[float]:
    block = item["ensemble"].get("all")
    if block is None:
        return None
    return {"P": block["P_mean"], "S": block["S_mean"], "F": block["F_mean"],
            "D": block["D"]["mean"], "D_prime": block["D_prime"]["mean"]}[name]


# ---------------------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------------------


def diagnostic_1(payload: Dict) -> Dict:
    items = payload["items"]

    def split(subset: Sequence[Dict]) -> Dict:
        out = {}
        for cls in ("context_dependent", "prior_dependent", "unstable_other"):
            rows = [i for i in subset if i["ensemble_counterfactual_class"] == cls]
            out[cls] = {
                signal: _describe([_ens_signal(i, signal) for i in rows])
                for signal in ("S", "P", "F")
            }
        ctx = [_ens_signal(i, "S") for i in subset
               if i["ensemble_counterfactual_class"] == "context_dependent"]
        pri = [_ens_signal(i, "S") for i in subset
               if i["ensemble_counterfactual_class"] == "prior_dependent"]
        out["S_auc_context_vs_prior"] = auc(ctx, pri)
        # Standardised mean difference, so "small but real" can be told from "no signal".
        if len(ctx) > 1 and len(pri) > 1:
            pooled = math.sqrt(
                ((len(ctx) - 1) * statistics.variance(ctx)
                 + (len(pri) - 1) * statistics.variance(pri))
                / (len(ctx) + len(pri) - 2)
            )
            out["S_cohens_d_context_minus_prior"] = (
                (statistics.fmean(ctx) - statistics.fmean(pri)) / pooled if pooled else None
            )
        else:
            out["S_cohens_d_context_minus_prior"] = None
        return out

    both_correct = [i for i in items if i["ensemble_baseline_category"] == "both_correct"]
    return {
        "premise_check": (
            "The Setting-C class is assigned by `classify_counterfactual` from the Setting-C "
            "argmax alone (run_counterfactual_experiment.py:145-153). It does NOT require "
            "both_correct, so Tier-1 runs over every counterfactual-eligible item, not a "
            "both_correct subset. The ceiling argument is therefore tested twice: over all "
            "eligible items, and restricted to both_correct where it would actually apply."
        ),
        "all_eligible": split(items),
        "both_correct_only": split(both_correct),
    }


# ---------------------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------------------


def _standardise(rows: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    n_features = len(rows[0])
    means = [statistics.fmean([r[j] for r in rows]) for j in range(n_features)]
    sds = []
    for j in range(n_features):
        column = [r[j] for r in rows]
        sd = statistics.stdev(column) if len(column) > 1 else 0.0
        sds.append(sd if sd > 1e-12 else 1.0)
    return [[(r[j] - means[j]) / sds[j] for j in range(n_features)] for r in rows], means, sds


def _fit_logistic(
    x: Sequence[Sequence[float]], y: Sequence[int], ridge: float = 1e-4,
    iterations: int = 200,
) -> List[float]:
    """IRLS / Newton-Raphson with a small ridge. Returns [intercept, *coefficients]."""
    n_features = len(x[0])
    design = [[1.0, *row] for row in x]
    beta = [0.0] * (n_features + 1)
    for _ in range(iterations):
        gradient = [0.0] * (n_features + 1)
        hessian = [[0.0] * (n_features + 1) for _ in range(n_features + 1)]
        for row, label in zip(design, y):
            z = sum(b * v for b, v in zip(beta, row))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            w = max(p * (1.0 - p), 1e-9)
            residual = label - p
            for a in range(len(beta)):
                gradient[a] += residual * row[a]
                for b in range(len(beta)):
                    hessian[a][b] += w * row[a] * row[b]
        for a in range(len(beta)):
            gradient[a] -= ridge * beta[a]
            hessian[a][a] += ridge
        step = _solve(hessian, gradient)
        if step is None:
            break
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return beta


def _solve(matrix: List[List[float]], vector: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting."""
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]
    return [augmented[i][n] / augmented[i][i] for i in range(n)]


def _predict(beta: Sequence[float], rows: Sequence[Sequence[float]]) -> List[float]:
    return [
        1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, beta[0] + sum(
            b * v for b, v in zip(beta[1:], row))))))
        for row in rows
    ]


def diagnostic_2(payload: Dict, n_folds: int = 5, seed: int = 0) -> Dict:
    items = payload["items"]
    rows, labels, extra = [], [], []
    for item in items:
        cls = item["ensemble_counterfactual_class"]
        if cls not in ("context_dependent", "prior_dependent"):
            continue
        rows.append([_ens_signal(item, f) for f in FEATURES])
        labels.append(1 if cls == "context_dependent" else 0)
        extra.append((_ens_signal(item, "D"), _ens_signal(item, "D_prime")))
    if len(set(labels)) < 2 or len(rows) < 20:
        return {"unavailable": f"n={len(rows)} with {len(set(labels))} classes"}

    standardised, means, sds = _standardise(rows)
    beta = _fit_logistic(standardised, labels)
    fitted = _predict(beta, standardised)
    pos = [v for v, l in zip(fitted, labels) if l == 1]
    neg = [v for v, l in zip(fitted, labels) if l == 0]

    # Cross-validated AUC: the fit sees the labels, so in-sample AUC is an upper bound.
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    held_out_scores: List[Tuple[float, int]] = []
    for fold in range(n_folds):
        test_idx = {order[i] for i in range(fold, len(order), n_folds)}
        train = [i for i in range(len(rows)) if i not in test_idx]
        test = [i for i in range(len(rows)) if i in test_idx]
        if not test or len({labels[i] for i in train}) < 2:
            continue
        sub, m, s = _standardise([rows[i] for i in train])
        fold_beta = _fit_logistic(sub, [labels[i] for i in train])
        scaled = [[(rows[i][j] - m[j]) / s[j] for j in range(len(FEATURES))] for i in test]
        for score, i in zip(_predict(fold_beta, scaled), test):
            held_out_scores.append((score, labels[i]))
    cv_auc = auc(
        [s for s, l in held_out_scores if l == 1],
        [s for s, l in held_out_scores if l == 0],
    )

    def single(index: int) -> Optional[float]:
        return auc([r[index] for r, l in zip(rows, labels) if l == 1],
                   [r[index] for r, l in zip(rows, labels) if l == 0])

    return {
        "what": "Upper-bound check: does ANY linear combination of P, S, F beat F alone? "
                "Fitted on the labels it is scored against; NOT a metric candidate.",
        "n": len(rows),
        "n_context_dependent": sum(labels),
        "n_prior_dependent": len(labels) - sum(labels),
        "coefficients_standardised": {
            "intercept": beta[0],
            **{name: beta[i + 1] for i, name in enumerate(FEATURES)},
        },
        "feature_means": dict(zip(FEATURES, means)),
        "feature_sds": dict(zip(FEATURES, sds)),
        "auc_logistic_in_sample": auc(pos, neg),
        "auc_logistic_cross_validated": cv_auc,
        "auc_F_alone": single(FEATURES.index("F")),
        "auc_S_alone": single(FEATURES.index("S")),
        "auc_P_alone": single(FEATURES.index("P")),
        "auc_D": auc([d for (d, _), l in zip(extra, labels) if l == 1],
                     [d for (d, _), l in zip(extra, labels) if l == 0]),
        "auc_D_prime": auc([dp for (_, dp), l in zip(extra, labels) if l == 1],
                           [dp for (_, dp), l in zip(extra, labels) if l == 0]),
    }


# ---------------------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------------------


def _snippet(text: str, needle: Optional[str], width: int = 130) -> str:
    if not text:
        return ""
    position = text.lower().find(needle.lower()) if needle else -1
    if position < 0:
        return text[:width] + ("..." if len(text) > width else "")
    start = max(0, position - width // 3)
    end = min(len(text), position + len(needle) + 2 * width // 3)
    return ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")


def diagnostic_3(payload: Dict, top_n: int = 8) -> Dict:
    taxonomy = _load(TAXONOMY)["datasets"]["sciq"]["questions"]
    shortcut_ids = {
        r["question_id"] for r in taxonomy if r["failure_category"] == "reasoning_shortcut"
    }
    source = {r["id"]: r for r in _load(SCIQ_SOURCE)["results"]}

    candidates = [
        (i, _ens_signal(i, "F")) for i in payload["items"] if i["id"] in shortcut_ids
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)

    laid_out = []
    for item, f_value in candidates[:top_n]:
        record = source[item["id"]]
        laid_out.append({
            "id": item["id"],
            "ensemble_F": f_value,
            "ensemble_D": _ens_signal(item, "D"),
            "ensemble_S": _ens_signal(item, "S"),
            "ensemble_P": _ens_signal(item, "P"),
            "ensemble_counterfactual_class": item["ensemble_counterfactual_class"],
            "ensemble_baseline_category": item["ensemble_baseline_category"],
            "kda_cont": item["kda_cont"],
            "question": record["question"],
            "options": record["options"],
            "gold_answer": record["correct_answer"],
            "counterfactual_target": record["counterfactual_target"],
            "substitution_tier": record["substitution_tier"],
            "passage_snippet": _snippet(record["passage"], record["correct_answer"]),
            "counterfactual_snippet": _snippet(
                record["counterfactual_passage"], record["counterfactual_target"]
            ),
            "per_solver": {
                name: {
                    "pred_B": record["per_model"][name]["pred_original_passage"],
                    "pred_C": record["per_model"][name]["pred_counterfactual_passage"],
                    "class": record["per_model"][name]["counterfactual_class"],
                    "F": item["per_solver"][name]["F"],
                }
                for name in payload["solvers"]
            },
        })
    return {
        "category": "reasoning_shortcut",
        "n_in_category_and_eligible": len(candidates),
        "top_by_ensemble_F": laid_out,
    }


# ---------------------------------------------------------------------------------------
# D4
# ---------------------------------------------------------------------------------------


def _margins(item: Dict, solvers: Sequence[str]) -> Tuple[float, float, float]:
    """(MarginB, MarginC, F_raw) at ensemble level, averaged over solvers."""
    def mean_of(key: str) -> float:
        return statistics.fmean([item["per_solver"][n][key] for n in solvers])
    margin_b = mean_of("S") - mean_of("p_cf_B")          # Setting B only
    margin_c = mean_of("p_cf_C") - mean_of("p_gold_C")   # Setting C only
    return margin_b, margin_c, item["ensemble"]["all"]["F_raw_mean"]


def diagnostic_4(payload: Dict) -> Dict:
    """Split F_raw into its Setting-B and Setting-C halves and score each on Tier 1."""
    solvers = payload["solvers"]

    def collect(stratum: str) -> Dict:
        pos = {k: [] for k in ("MarginB", "MarginC", "F_raw", "F")}
        neg = {k: [] for k in pos}
        identity_ok = identity_n = 0
        for item in payload["items"]:
            cls = item["ensemble_counterfactual_class"]
            if cls not in ("context_dependent", "prior_dependent"):
                continue
            if stratum == "both_correct" and item["ensemble_baseline_category"] != "both_correct":
                continue
            margin_b, margin_c, f_raw = _margins(item, solvers)
            identity_n += 1
            if abs(0.5 * (margin_b + margin_c) - f_raw) < 1e-9:
                identity_ok += 1
            target = pos if cls == "context_dependent" else neg
            target["MarginB"].append(margin_b)
            target["MarginC"].append(margin_c)
            target["F_raw"].append(f_raw)
            target["F"].append(item["ensemble"]["all"]["F_mean"])
        return {
            "n_context_dependent": len(pos["MarginB"]),
            "n_prior_dependent": len(neg["MarginB"]),
            "identity_holds": f"{identity_ok}/{identity_n}",
            "auc": {k: auc(pos[k], neg[k]) for k in pos},
            "mean_MarginB_context": statistics.fmean(pos["MarginB"]) if pos["MarginB"] else None,
            "mean_MarginB_prior": statistics.fmean(neg["MarginB"]) if neg["MarginB"] else None,
        }

    return {
        "decomposition": "F_raw = 0.5*(MarginB + MarginC); "
                         "MarginB = p(gold|B) - p(c|B) uses no Setting-C information; "
                         "sign(MarginC) IS the context/prior label among these two classes.",
        "all_eligible": collect("all_eligible"),
        "both_correct_only": collect("both_correct"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/ex6_psfd_score/psfd_diagnostics.json")
    args = parser.parse_args()

    report = {
        "what": "Follow-up diagnostics on the P/S/F/D evaluation. Analysis only; no new "
                "formulation, no new inference.",
        "diagnostic_1_S_distribution_by_class": {},
        "diagnostic_2_logistic_ceiling": {},
        "diagnostic_3_reasoning_shortcut_examples": None,
        "diagnostic_4_margin_decomposition": {},
    }
    for tag, path in SCORE_FILES.items():
        payload = _load(path)
        report["diagnostic_1_S_distribution_by_class"][tag] = diagnostic_1(payload)
        report["diagnostic_2_logistic_ceiling"][tag] = diagnostic_2(payload)
        report["diagnostic_4_margin_decomposition"][tag] = diagnostic_4(payload)
        if tag == "sciq_test_full":
            report["diagnostic_3_reasoning_shortcut_examples"] = diagnostic_3(payload)

    out_path = ensure_parent(resolve_path(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    for tag in SCORE_FILES:
        d1 = report["diagnostic_1_S_distribution_by_class"][tag]["all_eligible"]
        d2 = report["diagnostic_2_logistic_ceiling"][tag]
        ctx = d1["context_dependent"]["S"]
        pri = d1["prior_dependent"]["S"]
        print(f"\n=== {tag} ===")
        print(f"  D1 S | context n={ctx['n']:4d} mean={ctx['mean']:.4f} sd={ctx['sd']:.4f}"
              f"  | prior n={pri['n']:4d} mean={pri['mean']:.4f} sd={pri['sd']:.4f}"
              f"  | AUC={d1['S_auc_context_vs_prior']:.4f}"
              f" d={d1['S_cohens_d_context_minus_prior']:+.3f}")
        if "unavailable" in d2:
            print(f"  D2 unavailable: {d2['unavailable']}")
            continue
        coefficients = d2["coefficients_standardised"]
        print(f"  D2 logistic in-sample AUC={d2['auc_logistic_in_sample']:.4f} "
              f"CV={d2['auc_logistic_cross_validated']:.4f} | F alone={d2['auc_F_alone']:.4f} "
              f"| D={d2['auc_D']:.4f} D'={d2['auc_D_prime']:.4f}")
        print(f"     coefficients (standardised): "
              + "  ".join(f"{k}={coefficients[k]:+.3f}" for k in ("P", "S", "F")))
        d4 = report["diagnostic_4_margin_decomposition"][tag]
        a4 = d4["all_eligible"]["auc"]
        print(f"  D4 identity {d4['all_eligible']['identity_holds']} | "
              f"MarginC={a4['MarginC']:.4f}  MarginB={a4['MarginB']:.4f}  "
              f"F_raw={a4['F_raw']:.4f} | MarginB within both_correct="
              f"{d4['both_correct_only']['auc']['MarginB']:.4f}")
    print(f"\nWrote {os.path.relpath(out_path, resolve_path('.'))}")


if __name__ == "__main__":
    main()
