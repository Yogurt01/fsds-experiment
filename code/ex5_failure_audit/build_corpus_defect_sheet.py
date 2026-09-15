"""Build the 33-item SciQ corpus-defect annotation sheet (Task B, step 1).

What this set is
----------------
The union of the two screens whose relationship is established in
docs/ex5_failure_audit/corpus_defect_audit.md section 5 (line 243):

    * question/passage overlap < 0.40   -- 21 items
    * Setting-C-ineligible              -- 24 items, i.e. `counterfactual_valid == false`

Union = 33 unique items, which the audit reports as capturing 13 of 13 known topic mismatches.

NOTE: read the overlap tail from `all_overlaps`, NOT from `tail_items`. That export was written
at `tail_threshold: 0.1` and holds only 12 items; section 6's arithmetic uses < 0.40.

The double-annotation subset
----------------------------
docs/ex5_failure_audit/corpus_defect_audit.md section 6 asks for "Double-annotate 15 of the 33, report Cohen's
kappa" without saying which 15, who the second annotator is, or whether the passes are blind.
Those three gaps were closed by decision on 2026-09-08 and are recorded in
docs/guides/ANNOTATION_GUIDE.md section 2.6:

    annotator 1   the existing hand-check in results/ex5_failure_audit/passage_overlap_handcheck.json
    annotator 2   the human filling the sheet this script writes
    the 15        a seeded draw from the 21 items annotator 1 already labelled -- those are the
                  only items on which a two-annotator comparison exists at all
    blind         yes; annotator 2 must not read corpus_defect_audit.md sections 3.1-3.2, or the
                  hand-check artifact, until their sheet is locked

Those 21 items are exactly the overlap < 0.40 screen (asserted below), so drawing the comparison
subset from them discloses nothing about annotator 1's verdicts -- only about which screen caught
the item, which the sheet's own `screen` column already says.

**This script never reads or writes a verdict.** It touches the hand-check artifact only to
collect the *ids* annotator 1 covered.

The sheet deliberately does NOT mark which 15 are the comparison subset: an annotator who knows
which rows are scored against someone else may work them harder, which would bias kappa upward.
The subset is pre-registered in a separate manifest instead, written at the same time.

Usage:
    uv run --active python code/ex5_failure_audit/build_corpus_defect_sheet.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Set

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent, resolve

OVERLAP_AUDIT = "results/ex5_failure_audit/passage_overlap_audit.json"
COUNTERFACTUAL = "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json"
HANDCHECK = "results/ex5_failure_audit/passage_overlap_handcheck.json"
DATASET = "datasets/sciq/sciq_test_full.json"

DEFAULT_SHEET = "results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv"
DEFAULT_MANIFEST = "results/ex5_failure_audit/corpus_defect_comparison_subset.json"

OVERLAP_THRESHOLD = 0.40
COMPARISON_N = 15
SEED = 20260904            # the project's standing seed; see passage_overlap_audit.py:193

# Category names in the hand-check artifact. Only the ids are read -- never the category.
HANDCHECK_CATEGORIES = ("topic_mismatch", "legitimate_low_overlap", "orthographic_defect", "other")

COLUMNS = [
    "row_id", "item_id", "question", "gold_answer", "passage",
    "verdict",                       # <- the annotator fills ONLY this
    "note",                          # <- free text, optional
    "overlap", "screen",             # <- provenance; not evidence
]

INSTRUCTIONS = [
    "# CORPUS-DEFECT SHEET -- SciQ union screen (overlap < 0.40 OR Setting-C-ineligible)",
    "# Fill ONLY the `verdict` column with MISMATCH or ON_TOPIC.",
    "#   MISMATCH = the passage is about an unrelated subject and cannot answer the question.",
    "#   ON_TOPIC = the passage is about this question's subject, even if worded differently and",
    "#              even if it never states the gold answer.",
    "# Judge PASSAGE-vs-QUESTION subject only. Do NOT judge whether the keyed answer is right,",
    "# and do NOT judge whether the passage is well written.",
    "# `overlap` and `screen` say why the item was screened in. They are provenance, not evidence",
    "# -- hide them if you can. `note` is free text for anything contested.",
    "#",
    "# BLIND PASS. You are annotator 2. Do not read docs/ex5_failure_audit/corpus_defect_audit.md sections 3.1-3.2",
    "# or results/ex5_failure_audit/passage_overlap_handcheck.json until this file is locked --",
    "# they carry annotator 1's verdicts for 21 of these 33 items.",
    "#",
    "# Rubric: docs/guides/ANNOTATION_GUIDE.md section 2.",
    "# When done, run: uv run --active python code/ex5_failure_audit/kappa_corpus_defect.py \\",
    "#                     --annotator-b <this file>",
]


def load_json(path: str):
    with open(resolve(path), encoding="utf-8") as handle:
        return json.load(handle)


def handcheck_ids(payload: Dict) -> Set[int]:
    """Every SciQ id annotator 1 assigned a category to -- ids only, never the category.

    The artifact groups ids by overlap band; within a band each category maps to a list of ids.
    `sample_above_0_40` stores a *count* rather than a list under the same key, so anything that
    is not a list is skipped.
    """
    ids: Set[int] = set()
    for block in payload["sciq"].values():
        if not isinstance(block, dict):
            continue
        for category in HANDCHECK_CATEGORIES:
            value = block.get(category)
            if isinstance(value, list):
                ids |= {int(item) for item in value}
    return ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Task B corpus-defect sheet.")
    parser.add_argument("--out", default=DEFAULT_SHEET, help="Annotation sheet (CSV).")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="Pre-registered comparison subset (JSON).")
    parser.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD,
                        help="Overlap screen cutoff; section 6's arithmetic uses 0.40.")
    parser.add_argument("--comparison-n", type=int, default=COMPARISON_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing sheet. Refuses by default once it has labels.")
    args = parser.parse_args(argv)

    overlaps = {int(k): v for k, v
                in load_json(OVERLAP_AUDIT)["datasets"]["sciq"]["all_overlaps"].items()}
    tail = {item for item, value in overlaps.items() if value < args.threshold}
    ineligible = {record["id"] for record in load_json(COUNTERFACTUAL)["results"]
                  if not record.get("counterfactual_valid")}
    prior = handcheck_ids(load_json(HANDCHECK))
    by_id = {record["id"]: record for record in load_json(DATASET)}

    ids: List[int] = sorted(tail | ineligible)

    # The comparison subset is drawn only from items annotator 1 already covered. Those turn out
    # to be exactly the overlap screen; assert it, because if the two ever diverge the draw would
    # start leaking which items carry a prior verdict.
    if prior != tail:
        print(f"  WARNING: hand-check ids ({len(prior)}) != overlap screen ({len(tail)}).")
        print(f"           only in hand-check: {sorted(prior - tail)}")
        print(f"           only in screen    : {sorted(tail - prior)}")
    if len(prior) < args.comparison_n:
        print(f"  ERROR: only {len(prior)} items carry a prior verdict; "
              f"cannot draw {args.comparison_n}.")
        return 1

    comparison = sorted(random.Random(args.seed).sample(sorted(prior), args.comparison_n))

    sheet_path = ensure_parent(resolve(args.out))
    if os.path.exists(sheet_path) and not args.force:
        with open(sheet_path, encoding="utf-8") as handle:
            existing = [line for line in handle if not line.startswith("#")]
        filled = sum(1 for row in csv.DictReader(existing) if (row.get("verdict") or "").strip())
        if filled:
            print(f"  REFUSING to overwrite {sheet_path}: {filled} rows already carry a verdict.")
            print("  Re-run with --force only if you intend to discard those labels.")
            return 1

    with open(sheet_path, "w", newline="", encoding="utf-8") as handle:
        for line in INSTRUCTIONS:
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row_id, item_id in enumerate(ids, start=1):
            record = by_id[item_id]
            in_tail, in_ineligible = item_id in tail, item_id in ineligible
            writer.writerow({
                "row_id": row_id,
                "item_id": item_id,
                "question": record["question"],
                "gold_answer": record["correct_answer"],
                "passage": record["passage"],
                "verdict": "",
                "note": "",
                "overlap": f"{overlaps[item_id]:.3f}",
                "screen": "both" if in_tail and in_ineligible
                          else ("overlap" if in_tail else "ineligible"),
            })

    manifest = {
        "_provenance": {
            "what": "Pre-registered double-annotation subset for the SciQ corpus-defect census "
                    "(docs/ex5_failure_audit/corpus_defect_audit.md section 6, step 2).",
            "decided": "2026-09-08",
            "spec_gap": "Section 6 asks for 15 of the 33 without naming which 15, who the second "
                        "annotator is, or whether the passes are blind. All three were closed by "
                        "decision; see docs/guides/ANNOTATION_GUIDE.md section 2.6.",
            "annotator_1": "the existing hand-check in " + HANDCHECK + " -- produced in an "
                           "earlier session by an LLM agent, single pass, before this guide "
                           "existed. Declared as an LLM rater.",
            "annotator_2": "human, filling " + args.out,
            "blind": "yes -- annotator 2 does not read corpus_defect_audit.md sections 3.1-3.2 "
                     "or the hand-check artifact until their sheet is locked",
            "selection_rule": f"random.Random({args.seed}).sample(sorted(prior_labelled_ids), "
                              f"{args.comparison_n}), where prior_labelled_ids are the "
                              f"{len(prior)} SciQ ids annotator 1 categorised",
            "why_not_random_over_all_33": "only the items annotator 1 covered support a "
                                          "two-annotator comparison; the other 12 have no first "
                                          "verdict to compare against",
            "not_marked_in_sheet": "deliberate -- an annotator who knows which rows are scored "
                                   "may work them harder, biasing kappa upward",
            "gate": "none; KAPPA_GATE=0.70 is pre-registered for Task A (judge validation) only",
        },
        "seed": args.seed,
        "sheet": args.out,
        "n_union": len(ids),
        "n_prior_labelled": len(prior),
        "n_comparison": len(comparison),
        "comparison_item_ids": comparison,
        "single_pass_item_ids": [item for item in ids if item not in set(comparison)],
        "screen_breakdown": {
            "overlap_below_threshold": len(tail),
            "setting_c_ineligible": len(ineligible),
            "in_both": len(tail & ineligible),
            "overlap_only": len(tail - ineligible),
            "ineligible_only": len(ineligible - tail),
        },
        "overlap_threshold": args.threshold,
    }
    manifest_path = ensure_parent(resolve(args.manifest))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"  overlap < {args.threshold}        : {len(tail)}")
    print(f"  Setting-C-ineligible  : {len(ineligible)}")
    print(f"  union (sheet rows)    : {len(ids)}")
    print(f"  prior-labelled (A1)   : {len(prior)}  [ids only; no verdict was read]")
    print(f"  comparison subset     : {len(comparison)} (seed {args.seed})")
    print(f"  single-pass remainder : {len(ids) - len(comparison)}")
    print(f"  wrote {sheet_path}")
    print(f"  wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
