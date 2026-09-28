"""Descriptive LLM-human agreement. NOT a kappa gate, NOT inter-annotator reliability.

Reports exact-match, within-1, and quadratic-weighted agreement between the supplementary
LLM-judge pass and annotator 1, for Q1, plus option-match for Q2. Nothing here feeds the
pre-registered gate, and the LLM labels are never combined with or substituted for the human's.

The number this produces is weak evidence in both directions. Task A in this same project found
an unvalidated Qwen3-4B judge over-crediting 13:1 against a stricter human check, so high
agreement does not validate the rubric or the labels, and low agreement does not by itself
indict them.

Usage:
    python code/ex8_independent_labels/score_llm_judge_agreement.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

D = "results/ex8_independent_labels/"
CAVEAT = ("SUPPLEMENTARY ONLY. This is LLM-human agreement, not inter-annotator kappa, and it "
          "does not discharge the pre-registered gate. Task A in this project measured an "
          "unvalidated Qwen3-4B judge over-crediting 13:1 against a stricter human check, so a "
          "high number here is weak evidence at best.")


def read(path: str, key: str) -> Dict[int, str]:
    body = [l for l in open(resolve_path(path), encoding="utf-8") if not l.startswith("#")]
    out = {}
    for row in csv.DictReader(body):
        value = str(row.get(key, "")).strip().lower()
        if value:
            out[int(row["item_id"])] = value
    return out


def weighted_agreement(pairs: Sequence[Tuple[int, int]]) -> Optional[float]:
    """Quadratic-weighted agreement, chance-corrected (same form as a weighted kappa but
    reported as `agreement` so it is never mistaken for the gate statistic)."""
    if not pairs:
        return None
    labels = (1, 2, 3, 4)
    n = len(pairs)
    weight = lambda a, b: ((a - b) ** 2) / 9.0
    marg_a = {l: sum(1 for a, _ in pairs if a == l) / n for l in labels}
    marg_b = {l: sum(1 for _, b in pairs if b == l) / n for l in labels}
    observed = sum(weight(a, b) for a, b in pairs) / n
    expected = sum(weight(a, b) * marg_a[a] * marg_b[b] for a in labels for b in labels)
    return (1 - observed / expected) if expected else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=D + "llm_judge_agreement.json")
    args = parser.parse_args()

    manifest = json.load(open(resolve_path(D + "annotation_manifest.json"), encoding="utf-8"))
    human1 = read(D + "q1_answerability_sheet.annotator1.csv", "rating")
    llm1 = read(D + "q1_llm_judge_check.csv", "llm_rating")
    human2 = read(D + "q2_context_following_sheet.annotator1.csv", "stated_answer")
    llm2 = read(D + "q2_llm_judge_check.csv", "llm_stated_answer")

    def q1_block(ids: Sequence[int], label: str) -> Dict:
        pairs = [(int(human1[i]), int(llm1[i])) for i in ids if i in human1 and i in llm1]
        if not pairs:
            return {"scope": label, "n": 0}
        exact = sum(1 for a, b in pairs if a == b) / len(pairs)
        within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
        collapse = lambda v: "DEP" if v <= 2 else "NOT"
        binary = sum(1 for a, b in pairs if collapse(a) == collapse(b)) / len(pairs)
        return {
            "scope": label, "n": len(pairs),
            "exact_match": exact, "within_1_point": within1,
            "collapsed_binary_match": binary,
            "quadratic_weighted_agreement": weighted_agreement(pairs),
            "mean_human": statistics.fmean(a for a, _ in pairs),
            "mean_llm": statistics.fmean(b for _, b in pairs),
            "llm_minus_human_mean": statistics.fmean(b - a for a, b in pairs),
            "confusion_human_x_llm": {f"h{a}_l{b}": sum(1 for x, y in pairs if x == a and y == b)
                                      for a in (1, 2, 3, 4) for b in (1, 2, 3, 4)},
        }

    def q2_block(ids: Sequence[int], label: str) -> Dict:
        pairs = [(human2[i], llm2[i]) for i in ids if i in human2 and i in llm2]
        if not pairs:
            return {"scope": label, "n": 0}
        return {
            "scope": label, "n": len(pairs),
            "option_match_rate": sum(1 for a, b in pairs if a == b) / len(pairs),
            "human_distribution": dict(collections.Counter(a for a, _ in pairs)),
            "llm_distribution": dict(collections.Counter(b for _, b in pairs)),
        }

    all_q1 = sorted(set(human1) & set(llm1))
    block_q1 = [i for i in manifest["q1"]["double_block_ids"] if i in human1 and i in llm1]
    all_q2 = sorted(set(human2) & set(llm2))
    block_q2 = [i for i in manifest["q2"]["double_block_ids"] if i in human2 and i in llm2]

    report = {
        "WHAT_THIS_IS_NOT": CAVEAT,
        "model": "Qwen3-4B-Instruct-2507, 4-bit NF4, greedy, one stateless call per item",
        "n_llm_unparsed_q1": sum(1 for i in human1 if i not in llm1),
        "n_llm_unparsed_q2": sum(1 for i in human2 if i not in llm2),
        "q1_full_sample": q1_block(all_q1, "all rated items"),
        "q1_double_block": q1_block(block_q1, "the 30-item block"),
        "q2_full_sample": q2_block(all_q2, "all 55 items"),
        "q2_double_block": q2_block(block_q2, "the 16-item block"),
    }

    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(CAVEAT + "\n")
    for key in ("q1_full_sample", "q1_double_block"):
        c = report[key]
        print(f"Q1 {c['scope']:20s} n={c['n']:3d}  exact {f(c['exact_match'])}"
              f"  within-1 {f(c['within_1_point'])}  binary {f(c['collapsed_binary_match'])}"
              f"  quad-weighted {f(c['quadratic_weighted_agreement'])}")
        print(f"      mean human {c['mean_human']:.2f} vs LLM {c['mean_llm']:.2f}"
              f"  (LLM - human = {c['llm_minus_human_mean']:+.2f})")
    for key in ("q2_full_sample", "q2_double_block"):
        c = report[key]
        print(f"Q2 {c['scope']:20s} n={c['n']:3d}  option match {f(c['option_match_rate'])}")
    print(f"\nunparsed: Q1 {report['n_llm_unparsed_q1']}, Q2 {report['n_llm_unparsed_q2']}")

    path = ensure_parent(resolve_path(args.out))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"Wrote {os.path.relpath(path, resolve_path('.'))}")


if __name__ == "__main__":
    main()
