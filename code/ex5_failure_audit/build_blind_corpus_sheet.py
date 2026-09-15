"""Derive a blind annotation copy of the SciQ corpus-defect sheet (Task B, step 1).

Why this exists
---------------
`corpus_defect_sheet_sciq33.csv` carries the selection provenance that put each item on the sheet
(`overlap`, `screen`) alongside the fields the judgement actually needs. Those two columns do more
than weakly bias the topic call: the comparison subset was drawn from the items annotator 1
covered, and that set is exactly the overlap screen, so either column turns "which rows are
double-annotated" into a lookup. This script removes them, along with `gold_answer` and `row_id`,
so the file the annotator opens is uniform across all 33 rows.

It is a *derivation*, never a mutation: the source is opened read-only and its SHA-256 is recorded
in the manifest so the pair can be re-verified later.

What is dropped, and why
------------------------
    row_id        row position (1..33, ascending item_id); item_id is the stable identifier
    gold_answer   not part of the Task B judgement -- ANNOTATION_GUIDE.md section 2.1 lists answer
                  correctness as out of scope and section 2.2 Source 4 puts "on topic but does not
                  support the keyed answer" out of scope too. Showing it invites drift toward an
                  answer-correctness task, and invites reconstructing the ineligibility screen by
                  eye ("is the gold answer in this passage?"). Excluded by explicit decision.
    overlap       selection provenance; also partitions the sheet by annotator-1 coverage
    screen        selection provenance; same partition, categorically

What is kept
------------
    item_id       the stable identifier the Task B scorer already joins on
                  (kappa_corpus_defect.py:96). Not invented, not positional.
    question      one half of the judgement
    passage       the other half
    verdict       empty, for the annotator: ON_TOPIC / MISMATCH
    note          empty, optional -- section 2.3's tiebreak asks for a line on contested items

Because the scorer needs only `item_id` and `verdict`, the filled blind sheet can be passed
straight to `kappa_corpus_defect.py --annotator-b`; no merge step is required.

This script never reads, copies, or inspects any annotator-1 verdict or category.

Usage:
    uv run --active python code/ex5_failure_audit/build_blind_corpus_sheet.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent, resolve

SOURCE = "results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv"
DEFAULT_OUT = "results/ex5_failure_audit/corpus_defect_sheet_sciq33_BLIND.csv"
DEFAULT_MANIFEST = "results/ex5_failure_audit/corpus_defect_sheet_sciq33_BLIND_manifest.json"

COLUMNS = ["item_id", "question", "passage", "verdict", "note"]

DROPPED = {
    "row_id": "row position (1..33, assigned in ascending item_id order); redundant with item_id, "
              "which is the stable identifier the scorer joins on",
    "gold_answer": "not part of the Task B judgement (ANNOTATION_GUIDE.md sections 2.1 and 2.2 "
                   "Source 4); showing it invites drift toward answer correctness and toward "
                   "reconstructing the ineligibility screen by eye",
    "overlap": "selection provenance, not evidence (ANNOTATION_GUIDE.md section 2.5); it also "
               "partitions the sheet by annotator-1 coverage, which would identify the "
               "double-annotated subset",
    "screen": "selection provenance; same partition as overlap, categorically",
}

INSTRUCTIONS = [
    "# BLIND CORPUS-DEFECT SHEET (Task B) -- SciQ, 33 items",
    "# Derived from corpus_defect_sheet_sciq33.csv by",
    "#   code/ex5_failure_audit/build_blind_corpus_sheet.py",
    "# The source is unmodified. This copy carries no prior annotation and no selection metadata.",
    "#",
    "# Fill ONLY the `verdict` column with ON_TOPIC or MISMATCH.",
    "#   ON_TOPIC = the passage is about the same specific subject as the question.",
    "#   MISMATCH = the passage is about a different specific subject.",
    "# Same broad field is NOT enough for ON_TOPIC.",
    "# Low word overlap is NOT enough for MISMATCH.",
    "# A passage can be ON_TOPIC even if it never states the keyed answer.",
    "# Do NOT judge whether the keyed answer is right, or whether the passage is well written.",
    "# `note` is free text, for genuinely contested items only.",
    "# Full rubric, with worked examples: docs/guides/ANNOTATION_GUIDE.md sections 2.2-2.3.",
    "#",
    "# DROPPED COLUMNS AND WHY:",
    "#   row_id       row position; item_id is the stable identifier",
    "#   gold_answer  not part of this judgement (excluded by design; see manifest)",
    "#   overlap      selection provenance",
    "#   screen       selection provenance",
    "#",
    "# BLIND PASS. While labelling, do not open any of:",
    "#   docs/ex5_failure_audit/corpus_defect_audit.md sections 3.1-3.2",
    "#   results/ex5_failure_audit/passage_overlap_handcheck.json",
    "#   docs/guides/ANNOTATION_GUIDE.md section 2.6",
    "#   results/ex5_failure_audit/corpus_defect_comparison_subset.json",
    "# The first two carry annotator 1's verdicts; the last two identify the double-annotated",
    "# subset. Label all 33 rows uniformly, in file order.",
]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: str) -> List[Dict[str, str]]:
    with open(resolve(path), encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the blind Task B annotation sheet.")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing blind sheet. Refuses by default once it "
                             "carries verdicts.")
    args = parser.parse_args(argv)

    source_path = resolve(args.source)
    hash_before = sha256_file(source_path)
    rows = read_source(args.source)

    item_ids = [row["item_id"] for row in rows]
    if len(set(item_ids)) != len(item_ids):
        print(f"  ERROR: item_id is not unique in the source "
              f"({len(item_ids)} rows, {len(set(item_ids))} distinct).")
        return 1

    out_path = ensure_parent(resolve(args.out))
    if os.path.exists(out_path) and not args.force:
        with open(out_path, encoding="utf-8") as handle:
            existing = [line for line in handle if not line.startswith("#")]
        filled = sum(1 for row in csv.DictReader(existing) if (row.get("verdict") or "").strip())
        if filled:
            print(f"  REFUSING to overwrite {out_path}: {filled} rows already carry a verdict.")
            print("  Re-run with --force only if you intend to discard those labels.")
            return 1

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        for line in INSTRUCTIONS:
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "item_id": row["item_id"],
                "question": row["question"],
                "passage": row["passage"],
                "verdict": "",
                "note": "",
            })

    manifest = {
        "_provenance": {
            "what": "Blind annotation copy of the SciQ corpus-defect census sheet (Task B, "
                    "step 1 of docs/ex5_failure_audit/corpus_defect_audit.md section 6).",
            "created": "2026-09-09",
            "generator": "code/ex5_failure_audit/build_blind_corpus_sheet.py",
            "rubric": "docs/guides/ANNOTATION_GUIDE.md sections 2.2-2.3",
            "annotator": "annotator 2 (human)",
            "source_is_unmodified": True,
        },
        "source": {"path": args.source, "sha256": hash_before, "n_rows": len(rows)},
        "blind_copy": {
            "path": args.out,
            "columns": COLUMNS,
            "n_rows": len(rows),
            "row_order": "identical to source (ascending item_id); not sorted, shuffled or "
                         "reordered, by explicit instruction",
        },
        "retained_columns": {
            "item_id": "stable identifier; the Task B scorer joins on it "
                       "(kappa_corpus_defect.py:96)",
            "question": "one half of the judgement",
            "passage": "the other half",
            "verdict": "empty; the annotator fills ON_TOPIC / MISMATCH",
            "note": "empty; optional free text for contested items",
        },
        "dropped_columns": DROPPED,
        "stable_id": {
            "column": "item_id",
            "why": "it is the identifier the scoring protocol already uses, it is an index into "
                   "datasets/sciq/sciq_test_full.json rather than a row position, and it is "
                   "verified unique across all 33 rows",
            "verified_unique": len(set(item_ids)) == len(item_ids),
            "joins_to": [
                "results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv (the source)",
                "results/ex5_failure_audit/corpus_defect_comparison_subset.json "
                "(the pre-registered subset, by comparison_item_ids)",
            ],
            "no_merge_step_required": "kappa_corpus_defect.py reads only item_id and verdict, so "
                                      "the filled blind sheet can be passed directly as "
                                      "--annotator-b",
        },
        "blindness": {
            "annotator_1_information_exposed": False,
            "annotator_1_information_copied": False,
            "annotator_1_categories_inspected_to_build_this_sheet": False,
            "statement": "No annotator-1 verdict or category was read, copied, summarised or used "
                         "in deciding this sheet's contents. The generator reads only the source "
                         "CSV; it does not open passage_overlap_handcheck.json.",
            "comparison_subset_marked": False,
            "comparison_subset_marking_rationale": "an annotator who knows which rows are scored "
                                                   "against another rater may work them harder, "
                                                   "biasing kappa upward",
        },
        "residual_channels": {
            "note": "Recorded rather than claimed closed. None is removable without breaking the "
                    "task or an explicit instruction; all are mitigated behaviourally.",
            "item_id_cross_reference": "the 33 item_ids and the 15 comparison item_ids are both "
                                       "published in docs/guides/ANNOTATION_GUIDE.md (sections 2.5 and "
                                       "2.6.1) and in corpus_defect_comparison_subset.json, so "
                                       "subset membership is recoverable by consulting a document "
                                       "the annotator has already read. Mitigation: do not open "
                                       "those sections while labelling.",
            "row_order": "rows are in ascending item_id order, which the annotator was instructed "
                         "not to change; position therefore maps to item_id, but carries no "
                         "verdict or screen information",
            "metric_recomputation": "the overlap metric is a function of question and passage, "
                                    "which the task requires showing, so screen membership is in "
                                    "principle recomputable. Inherent to the annotation; cannot "
                                    "be removed.",
        },
        "design_caveats": {
            "gold_answer_excluded": "By explicit decision, the annotator is shown only question "
                                    "and passage and judges subject correspondence alone; the "
                                    "gold answer is not displayed. This is a deliberate "
                                    "narrowing of the task to what the rubric actually asks. It "
                                    "does mean the two annotators may not have had identical "
                                    "information in front of them, which should be stated beside "
                                    "the published kappa.",
        },
        "accepted_values": {
            "MISMATCH": ["MISMATCH", "M", "1", "Y", "YES"],
            "ON_TOPIC": ["ON_TOPIC", "ON-TOPIC", "ONTOPIC", "O", "0", "N", "NO"],
            "source": "code/ex5_failure_audit/kappa_corpus_defect.py ALIASES",
        },
    }
    manifest_path = ensure_parent(resolve(args.manifest))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        import json
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    hash_after = sha256_file(source_path)
    print(f"  source rows          : {len(rows)}")
    print(f"  blind rows           : {len(rows)}")
    print(f"  item_id distinct     : {len(set(item_ids))}")
    print(f"  columns kept         : {len(COLUMNS)}  {COLUMNS}")
    print(f"  columns dropped      : {len(DROPPED)}  {sorted(DROPPED)}")
    print(f"  source sha256 before : {hash_before}")
    print(f"  source sha256 after  : {hash_after}")
    print(f"  source untouched     : {hash_before == hash_after}")
    print(f"  wrote {out_path}")
    print(f"  wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
