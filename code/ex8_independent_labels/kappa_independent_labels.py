"""Inter-annotator agreement for the RQ2 independent label set. Read-only, no inference.

Adapted from code/ex5_failure_audit/kappa_corpus_defect.py, which already has the right shape:
two separate annotator files plus a manifest naming the double-annotated subset. Three things
change here.

  1. Q1 is an ordinal 1-4 scale, not a binary label, so both a quadratic-weighted kappa (which
     charges a 1-vs-4 disagreement more than a 1-vs-2) and an unweighted kappa on the collapsed
     binary are reported. The GATE is applied to the collapsed binary, because that is the form
     the metric comparison ultimately uses and because KAPPA_GATE was pre-registered against a
     binary judgement in compute_kappa.py.

  2. Q2 is a five-way categorical (a/b/c/d/none), scored with unweighted kappa. No gate: Q2 is
     a validity check on the counterfactual construction, not a measurement whose reliability
     licenses a downstream claim.

  3. `unusable` rows are excluded from every statistic and counted separately. An item either
     annotator marked unusable is dropped from both, so the pairs stay aligned.

The collapse for Q1 is fixed here and must not be changed after labels are seen:

    rating 1 or 2  ->  DEPENDENT      (an unaided student would probably fail; the passage matters)
    rating 3 or 4  ->  NOT_DEPENDENT  (an unaided student would probably succeed)

Scope. The GATE is always computed on the pre-registered double block -- 30 Q1 items, 16 Q2 items
-- and that never changes. `--full-sample` additionally scores every item both annotators rated
(100 Q1, 55 Q2) as a SUPPLEMENTARY reliability estimate, added 2026-09-22 when the second annotator
took the full sheets (docs/ex8_independent_labels/analysis_plan_annotator2.md section 4). The
full-sample figure never decides: if it and the block figure land on opposite sides of KAPPA_GATE,
the block governs and the report says so. Only the set of item ids passed in differs; the kappa
computation, the collapse rule, the gate and the unusable handling are shared.

Usage:
    python code/ex8_independent_labels/kappa_independent_labels.py \
        --q1-a <annotator1.csv> --q1-b <annotator2.csv> \
        [--q2-a <a.csv> --q2-b <b.csv>] [--full-sample] [--out report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

DEFAULT_MANIFEST = "results/ex8_independent_labels/annotation_manifest.json"

# Pre-registered, mirroring compute_kappa.py:38. Do not change after seeing results.
KAPPA_GATE = 0.70

RATINGS = ("1", "2", "3", "4")
BINARY = ("DEPENDENT", "NOT_DEPENDENT")
COLLAPSE = {"1": "DEPENDENT", "2": "DEPENDENT", "3": "NOT_DEPENDENT", "4": "NOT_DEPENDENT"}
Q2_LABELS = ("a", "b", "c", "d", "none")
Q2_ALIASES = {"a": "a", "b": "b", "c": "c", "d": "d",
              "none": "none", "n": "none", "x": "none", "-": "none"}


def _read(path: str, value_column: str) -> Tuple[Dict[int, str], List[int]]:
    """item_id -> raw value, plus the ids marked unusable. Comment lines are skipped."""
    with open(resolve_path(path), encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    values: Dict[int, str] = {}
    unusable: List[int] = []
    for row in csv.DictReader(lines):
        item_id = int(row["item_id"])
        if str(row.get("unusable", "")).strip().upper() == "X":
            unusable.append(item_id)
            continue
        raw = str(row.get(value_column, "")).strip().lower()
        if raw:
            values[item_id] = raw
    return values, unusable


def _kappa(pairs: Sequence[Tuple[str, str]], labels: Sequence[str],
           weights: Optional[Dict[Tuple[str, str], float]] = None) -> Optional[Dict]:
    """Cohen's kappa; quadratic-weighted when `weights` is supplied."""
    n = len(pairs)
    if n == 0:
        return None
    if weights is None:
        weights = {(a, b): (0.0 if a == b else 1.0) for a in labels for b in labels}
    marg_a = {l: sum(1 for a, _ in pairs if a == l) / n for l in labels}
    marg_b = {l: sum(1 for _, b in pairs if b == l) / n for l in labels}
    observed = sum(weights[(a, b)] for a, b in pairs) / n
    expected = sum(weights[(a, b)] * marg_a[a] * marg_b[b] for a in labels for b in labels)
    kappa = (1 - observed / expected) if expected else None
    return {
        "n": n,
        "observed_disagreement": observed,
        "expected_disagreement": expected,
        "kappa": kappa,
        "kappa_undefined_reason": None if kappa is not None
            else "degenerate margin: expected disagreement is zero",
        "raw_agreement": sum(1 for a, b in pairs if a == b) / n,
        "confusion": {f"A_{a}__B_{b}": sum(1 for x, y in pairs if x == a and y == b)
                      for a in labels for b in labels},
    }


def score_q1(path_a: str, path_b: str, subset: Sequence[int], role: str = "gate") -> Dict:
    """`role="gate"` is the pre-registered call. Anything else is supplementary and carries no
    gate verdict, so a full-sample figure cannot be mistaken for the gate by a later reader."""
    a, unusable_a = _read(path_a, "rating")
    b, unusable_b = _read(path_b, "rating")
    dropped = sorted(set(unusable_a) | set(unusable_b))
    shared = [i for i in subset
              if i in a and i in b and a[i] in RATINGS and b[i] in RATINGS]
    ordinal = [(a[i], b[i]) for i in shared]
    collapsed = [(COLLAPSE[a[i]], COLLAPSE[b[i]]) for i in shared]
    quad = {(x, y): ((int(x) - int(y)) ** 2) / 9.0 for x in RATINGS for y in RATINGS}

    binary = _kappa(collapsed, BINARY)
    gate = None if binary is None or binary["kappa"] is None else binary["kappa"] >= KAPPA_GATE
    subset_set = set(subset)
    out = {
        "role": role,
        "n_subset": len(subset),
        "n_scored": len(shared),
        "n_unusable_either_annotator": len(dropped),
        "n_unusable_in_subset": sum(1 for i in dropped if i in subset_set),
        "unusable_ids": dropped,
        "ordinal_quadratic_weighted": _kappa(ordinal, RATINGS, quad),
        "collapsed_binary": binary,
        "collapse_rule": "1,2 -> DEPENDENT; 3,4 -> NOT_DEPENDENT",
    }
    if role == "gate":
        out.update({"kappa_gate": KAPPA_GATE, "gate_passed": gate,
                    "gate_applies_to": "collapsed_binary"})
    else:
        out["not_the_gate"] = ("Supplementary reliability estimate. The pre-registered gate is "
                               "the double-block figure; this one does not decide.")
    return out


def score_q2(path_a: str, path_b: str, subset: Sequence[int], role: str = "block") -> Dict:
    a, _ = _read(path_a, "stated_answer")
    b, _ = _read(path_b, "stated_answer")
    norm = lambda v: Q2_ALIASES.get(v)
    shared = [i for i in subset if norm(a.get(i)) and norm(b.get(i))]
    pairs = [(norm(a[i]), norm(b[i])) for i in shared]
    return {
        "role": role,
        "n_subset": len(subset),
        "n_scored": len(shared),
        "unweighted": _kappa(pairs, Q2_LABELS),
        "no_gate_reason": "Q2 is a validity check on the counterfactual construction, not a "
                          "measurement whose reliability licenses a downstream claim.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--q1-a", required=True, help="Annotator 1's filled Q1 sheet.")
    parser.add_argument("--q1-b", required=True, help="Annotator 2's filled Q1 sheet.")
    parser.add_argument("--q2-a", default="", help="Annotator 1's filled Q2 sheet (optional).")
    parser.add_argument("--q2-b", default="", help="Annotator 2's filled Q2 sheet (optional).")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--full-sample", action="store_true",
                        help="Also score every item both annotators rated (supplementary; the "
                             "gate stays on the double block).")
    parser.add_argument("--out", default="", help="Write the report as JSON here.")
    args = parser.parse_args(argv)

    with open(resolve_path(args.manifest), encoding="utf-8") as handle:
        manifest = json.load(handle)

    report = {"manifest": args.manifest,
              "q1": score_q1(args.q1_a, args.q1_b, manifest["q1"]["double_block_ids"])}
    if args.q2_a and args.q2_b:
        report["q2"] = score_q2(args.q2_a, args.q2_b, manifest["q2"]["double_block_ids"])

    if args.full_sample:
        membership = {int(k): v for k, v in manifest["stratum_membership"].items()}
        q1_all = sorted(membership)                                   # every Q1 item
        q2_all = sorted(i for i, s in membership.items() if s == "A")  # Q2 covers stratum A
        if len(q1_all) != manifest["q1"]["n"] or len(q2_all) != manifest["q2"]["n"]:
            raise SystemExit("manifest item counts disagree with stratum membership; refusing to score")
        full = score_q1(args.q1_a, args.q1_b, q1_all, role="supplementary_full_sample")
        report["q1_full_sample"] = full
        if args.q2_a and args.q2_b:
            report["q2_full_sample"] = score_q2(args.q2_a, args.q2_b, q2_all,
                                                role="supplementary_full_sample")
        block_k = report["q1"]["collapsed_binary"]["kappa"] if report["q1"]["collapsed_binary"] else None
        full_k = full["collapsed_binary"]["kappa"] if full["collapsed_binary"] else None
        if block_k is not None and full_k is not None:
            split = (block_k >= KAPPA_GATE) != (full_k >= KAPPA_GATE)
            report["gate_disagreement"] = {
                "block_kappa": block_k, "full_sample_kappa": full_k, "disagree": split,
                "rule": "the double-block figure governs; the disagreement must be stated "
                        "alongside the gate result",
            }

    q1 = report["q1"]
    print(f"Q1  scored {q1['n_scored']} of {q1['n_subset']} double-block items"
          f"  ({q1['n_unusable_in_subset']} unusable in the block, "
          f"{q1['n_unusable_either_annotator']} in the whole sheet)")
    if q1["n_scored"] == 0:
        print("  Nothing to score yet -- fill the `rating` column in both sheets first.")
    else:
        for name in ("ordinal_quadratic_weighted", "collapsed_binary"):
            cell = q1[name]
            k = cell["kappa"]
            print(f"  {name:28s} kappa {'n/a' if k is None else f'{k:.3f}'}"
                  f"   raw agreement {cell['raw_agreement']:.3f}  (n={cell['n']})")
        print(f"  gate {KAPPA_GATE} on collapsed binary: "
              f"{'PASSED' if q1['gate_passed'] else 'NOT PASSED'}")
    if "q2" in report:
        q2 = report["q2"]
        k = q2["unweighted"]["kappa"] if q2["unweighted"] else None
        print(f"Q2  scored {q2['n_scored']} of {q2['n_subset']}"
              f"   kappa {'n/a' if k is None else f'{k:.3f}'}")

    if "q1_full_sample" in report:
        full = report["q1_full_sample"]
        print(f"\nSUPPLEMENTARY (not the gate) -- full sample, "
              f"{full['n_scored']} of {full['n_subset']} Q1 items")
        for name in ("ordinal_quadratic_weighted", "collapsed_binary"):
            cell = full[name]
            k = cell["kappa"] if cell else None
            print(f"  {name:28s} kappa {'n/a' if k is None else f'{k:.3f}'}"
                  + (f"   raw agreement {cell['raw_agreement']:.3f}  (n={cell['n']})" if cell else ""))
        if "q2_full_sample" in report:
            q2f = report["q2_full_sample"]
            k = q2f["unweighted"]["kappa"] if q2f["unweighted"] else None
            print(f"  Q2 full sample: {q2f['n_scored']} of {q2f['n_subset']}"
                  f"   kappa {'n/a' if k is None else f'{k:.3f}'}")
        gd = report.get("gate_disagreement")
        if gd and gd["disagree"]:
            print(f"  !! Block kappa {gd['block_kappa']:.3f} and full-sample kappa "
                  f"{gd['full_sample_kappa']:.3f} fall on opposite sides of {KAPPA_GATE}. "
                  f"The double block governs; state both.")

    if args.out:
        path = ensure_parent(resolve_path(args.out))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        print(f"\nWrote {os.path.relpath(path, resolve_path('.'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
