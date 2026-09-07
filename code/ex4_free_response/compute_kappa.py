"""Score a filled human-validation sheet against the LLM judge, with the kappa >= 0.70 gate.

The gate is **pre-registered**: it is declared here, in code, before the human labels exist, so
the decision rule cannot be adjusted after seeing the result.

    kappa >= 0.70   the judge is reliable enough; `acc_band_low` / `acc_band_high` may be
                    published as the headline, with kappa quoted alongside
    kappa <  0.70   the judge is NOT reliable enough; publish `acc_normalised` (the
                    deterministic floor) as the headline and the band only as a diagnostic

Cohen's kappa is computed against the judge's **forward** verdict (the one that would be quoted)
and again against its **both-directions-agree** verdict, since those are the two candidate
headline mechanisms. Agreement is also reported separately inside the `agree` and `flip` strata:
if the judge is reliable where it is self-consistent and unreliable where it flips, that is a
usable result even at a mediocre overall kappa, and it would justify reporting the band rather
than either endpoint.

Usage:
    uv run --active python code/ex4_free_response/compute_kappa.py \\
        --sheet results/ex4_free_response/validation_sheet_pilot_bidir.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent, resolve

KAPPA_GATE = 0.70          # pre-registered; do not change after seeing results
LABELS = ("CORRECT", "INCORRECT")


def cohens_kappa(pairs: Sequence[Sequence[str]]) -> Optional[Dict]:
    """Cohen's kappa for two raters over a binary label set."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n

    expected = 0.0
    for label in LABELS:
        p_a = sum(1 for a, _ in pairs if a == label) / n
        p_b = sum(1 for _, b in pairs if b == label) / n
        expected += p_a * p_b

    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    table = {
        f"human_{h}__judge_{j}": sum(1 for a, b in pairs if a == h and b == j)
        for h in LABELS for j in LABELS
    }
    return {
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
        "confusion": table,
    }


def read_sheet(path: str) -> List[Dict]:
    """Read the CSV, skipping the leading `#` instruction block."""
    with open(resolve(path), encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def normalise_verdict(value: str) -> Optional[str]:
    text = str(value or "").strip().upper()
    if text in ("CORRECT", "C", "1", "Y", "YES", "TRUE"):
        return "CORRECT"
    if text in ("INCORRECT", "I", "0", "N", "NO", "FALSE"):
        return "INCORRECT"
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score a validation sheet; apply the kappa gate.")
    parser.add_argument("--sheet", required=True, help="Filled validation CSV.")
    parser.add_argument("--out", default="", help="Write the report as JSON here (optional).")
    args = parser.parse_args(argv)

    rows = read_sheet(args.sheet)
    labelled, unlabelled = [], 0
    for row in rows:
        human = normalise_verdict(row.get("human_verdict"))
        if human is None:
            unlabelled += 1
            continue
        row["_human"] = human
        labelled.append(row)

    print(f"  rows in sheet     : {len(rows)}")
    print(f"  human-labelled    : {len(labelled)}")
    print(f"  still blank       : {unlabelled}")
    if not labelled:
        print("\n  Nothing to score yet -- fill the `human_verdict` column first.")
        return 1

    forward_pairs = [(r["_human"], normalise_verdict(r.get("judge_forward")) or "INCORRECT")
                     for r in labelled]
    # The "both directions agree" mechanism: CORRECT only when both passes say CORRECT.
    both_pairs = [
        (
            r["_human"],
            "CORRECT" if (normalise_verdict(r.get("judge_forward")) == "CORRECT"
                          and normalise_verdict(r.get("judge_reverse")) == "CORRECT")
            else "INCORRECT",
        )
        for r in labelled
    ]

    report = {
        "kappa_gate": KAPPA_GATE,
        "n_labelled": len(labelled),
        "forward_only": cohens_kappa(forward_pairs),
        "both_directions": cohens_kappa(both_pairs),
        "by_stratum": {},
    }

    strata: Dict[str, List] = {}
    for row, pair in zip(labelled, forward_pairs):
        strata.setdefault(row.get("stratum", "?"), []).append(pair)
    for name, pairs in sorted(strata.items()):
        report["by_stratum"][name] = cohens_kappa(pairs)

    def show(name: str, block: Optional[Dict]) -> None:
        if not block:
            return
        kappa = block["kappa"]
        print(f"  {name:34} n={block['n']:4}  agreement={block['observed_agreement']:.3f}  "
              f"kappa={'n/a' if kappa is None else f'{kappa:.3f}'}")

    print()
    print("  === Cohen's kappa, human vs judge ===")
    show("forward-only verdict", report["forward_only"])
    show("both-directions-agree verdict", report["both_directions"])
    print()
    print("  === by stratum (forward-only) ===")
    for name, block in report["by_stratum"].items():
        show(name, block)

    best = max(
        (b["kappa"] for b in (report["forward_only"], report["both_directions"])
         if b and b["kappa"] is not None),
        default=None,
    )
    report["best_kappa"] = best
    report["gate_passed"] = bool(best is not None and best >= KAPPA_GATE)

    print()
    print("  === PRE-REGISTERED GATE ===")
    if best is None:
        print("  kappa undefined (labels are degenerate) -- treat as NOT passed.")
    elif report["gate_passed"]:
        print(f"  kappa {best:.3f} >= {KAPPA_GATE}: GATE PASSED.")
        print("  -> publish acc_band_low / acc_band_high as the headline, quoting kappa.")
    else:
        print(f"  kappa {best:.3f} < {KAPPA_GATE}: GATE NOT PASSED.")
        print("  -> publish acc_normalised (deterministic floor) as the headline;")
        print("     report the band as a diagnostic only, and do not quote acc_judged.")

    if args.out:
        out_path = ensure_parent(resolve(args.out))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
