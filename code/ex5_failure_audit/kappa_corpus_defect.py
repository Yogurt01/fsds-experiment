"""Cohen's kappa between two annotators on the SciQ corpus-defect census (Task B, step 2).

docs/ex5_failure_audit/corpus_defect_audit.md section 6 asks for "Double-annotate 15 of the 33, report Cohen's
kappa". This is the scorer for that step.

Estimator
---------
The same one as code/ex4_free_response/compute_kappa.py:42-66, with the label set swapped to
MISMATCH / ON_TOPIC:

    kappa = (Po - Pe) / (1 - Pe)
    Po    = share of double-annotated items where the two annotators agree
    Pe    = sum over labels L of P(A = L) * P(B = L)

**No gate is applied.** KAPPA_GATE = 0.70 is pre-registered in compute_kappa.py for Task A, the
free-response *judge validation*, and has no authority here. Section 6 says only that kappa
should be high and that "if it is not, the rubric rather than the sample is at fault"
(corpus_defect_audit.md:278) -- so a low kappa indicts the rubric in ANNOTATION_GUIDE.md
sections 2.2-2.3, and the response is to sharpen it and say so, never to re-label until the
number improves.

Annotator 1 is the existing hand-check
--------------------------------------
Per the 2026-09-08 decision recorded in ANNOTATION_GUIDE.md section 2.6, annotator 1 is
results/ex5_failure_audit/passage_overlap_handcheck.json rather than a second CSV. Reading it
directly means no second file carrying its verdicts is ever created, which is what keeps
annotator 2's pass blind. Its four categories collapse to the binary judgement as set out in
ANNOTATION_GUIDE.md section 2.2:

    topic_mismatch                                     -> MISMATCH
    legitimate_low_overlap / orthographic_defect / other -> ON_TOPIC

Pass `--annotator-a <sheet.csv>` instead to score two filled sheets against each other.

Usage:
    uv run --active python code/ex5_failure_audit/kappa_corpus_defect.py \\
        --annotator-b results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv \\
        --out results/ex5_failure_audit/corpus_defect_kappa.json
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

from utils.paths import ensure_parent, resolve

HANDCHECK = "results/ex5_failure_audit/passage_overlap_handcheck.json"
# Annotator 2's labels live in the BLIND copy, not in the sheet the builder emits. The blind
# workflow (build_blind_corpus_sheet.py) was added after this scorer was written, and the default
# below pointed at the unlabelled source until 2026-09-09, so the documented no-argument
# invocation scored nothing. Both files carry `item_id` and `verdict`, so either can be passed.
DEFAULT_SHEET = "results/ex5_failure_audit/corpus_defect_sheet_sciq33_BLIND.csv"
DEFAULT_MANIFEST = "results/ex5_failure_audit/corpus_defect_comparison_subset.json"

LABELS = ("MISMATCH", "ON_TOPIC")

# Accepted spellings in the `verdict` column. Mirrors the alias style of
# compute_kappa.py:76-82; values are upper-cased and stripped before lookup.
ALIASES = {
    "MISMATCH": "MISMATCH", "M": "MISMATCH", "1": "MISMATCH", "Y": "MISMATCH", "YES": "MISMATCH",
    "ON_TOPIC": "ON_TOPIC", "ON-TOPIC": "ON_TOPIC", "ONTOPIC": "ON_TOPIC", "O": "ON_TOPIC",
    "0": "ON_TOPIC", "N": "ON_TOPIC", "NO": "ON_TOPIC",
}

# The hand-check's four categories, collapsed to the binary judgement.
CATEGORY_TO_LABEL = {
    "topic_mismatch": "MISMATCH",
    "legitimate_low_overlap": "ON_TOPIC",
    "orthographic_defect": "ON_TOPIC",
    "other": "ON_TOPIC",
}


def load_json(path: str):
    with open(resolve(path), encoding="utf-8") as handle:
        return json.load(handle)


def normalise_verdict(value: str) -> Optional[str]:
    return ALIASES.get(str(value or "").strip().upper())


def read_sheet(path: str) -> Dict[int, str]:
    """item_id -> normalised verdict, for rows carrying a parsable value."""
    with open(resolve(path), encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    labels: Dict[int, str] = {}
    for row in csv.DictReader(lines):
        verdict = normalise_verdict(row.get("verdict"))
        if verdict is not None:
            labels[int(row["item_id"])] = verdict
    return labels


def read_handcheck(path: str = HANDCHECK) -> Dict[int, str]:
    """item_id -> binary label, from annotator 1's committed artifact.

    Ids are grouped by overlap band, then by category. `sample_above_0_40` stores a count rather
    than a list under the same key, so anything that is not a list is skipped.
    """
    labels: Dict[int, str] = {}
    for block in load_json(path)["sciq"].values():
        if not isinstance(block, dict):
            continue
        for category, label in CATEGORY_TO_LABEL.items():
            value = block.get(category)
            if isinstance(value, list):
                for item in value:
                    labels[int(item)] = label
    return labels


def cohens_kappa(pairs: Sequence[Tuple[str, str]]) -> Optional[Dict]:
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
    return {
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
        "kappa_undefined_reason": None if kappa is not None
            else "degenerate margin: both annotators used a single label, so Pe = 1",
        "confusion": {
            f"A_{a}__B_{b}": sum(1 for x, y in pairs if x == a and y == b)
            for a in LABELS for b in LABELS
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cohen's kappa, annotator vs annotator, on the corpus-defect census.")
    parser.add_argument("--annotator-a", default="handcheck",
                        help="'handcheck' (default) reads annotator 1 from --handcheck; "
                             "otherwise a filled sheet CSV.")
    parser.add_argument("--handcheck", default=HANDCHECK,
                        help="Annotator 1's committed artifact, used when --annotator-a is "
                             "'handcheck'.")
    parser.add_argument("--annotator-b", default=DEFAULT_SHEET, help="Annotator 2's filled sheet.")
    parser.add_argument("--subset", default=DEFAULT_MANIFEST,
                        help="Comparison-subset manifest, or 'all' to score every shared item.")
    parser.add_argument("--out", default="", help="Write the report as JSON here (optional).")
    args = parser.parse_args(argv)

    if args.annotator_a == "handcheck":
        a_labels, a_source = read_handcheck(args.handcheck), args.handcheck
        a_name = "annotator 1 -- prior hand-check (LLM agent, single pass)"
    else:
        a_labels, a_source = read_sheet(args.annotator_a), args.annotator_a
        a_name = "annotator 1 -- filled sheet"
    b_labels = read_sheet(args.annotator_b)

    if args.subset == "all":
        subset, subset_source = None, "all shared items"
    else:
        manifest = load_json(args.subset)
        subset = set(manifest["comparison_item_ids"])
        subset_source = args.subset

    shared = sorted(set(a_labels) & set(b_labels))
    scored = [item for item in shared if subset is None or item in subset]
    missing = sorted(subset - set(b_labels)) if subset else []

    print(f"  annotator A source    : {a_source}")
    print(f"  annotator B sheet     : {args.annotator_b}")
    print(f"  comparison subset     : {subset_source}"
          + (f" ({len(subset)} items)" if subset else ""))
    print(f"  labelled by A         : {len(a_labels)}")
    print(f"  labelled by B         : {len(b_labels)}")
    print(f"  scored (in both)      : {len(scored)}")
    if missing:
        print(f"  NOT YET LABELLED BY B : {missing}")
    if not scored:
        print("\n  Nothing to score yet -- fill the `verdict` column first.")
        return 1

    block = cohens_kappa([(a_labels[item], b_labels[item]) for item in scored])
    disagreements = [item for item in scored if a_labels[item] != b_labels[item]]

    report = {
        "_provenance": {
            "what": "Cohen's kappa for docs/ex5_failure_audit/corpus_defect_audit.md section 6, step 2.",
            "annotator_A": a_name,
            "annotator_A_source": a_source,
            "annotator_B": "annotator 2 -- human",
            "annotator_B_source": args.annotator_b,
            "blind": "annotator 2 labelled without reading corpus_defect_audit.md sections "
                     "3.1-3.2 or the hand-check artifact",
            "subset_source": subset_source,
            "gate": "none; KAPPA_GATE=0.70 is pre-registered for Task A only",
            "caveat": "one rater is an LLM agent; n is small, so kappa is unstable -- always "
                      "quote n alongside it",
        },
        "labels": list(LABELS),
        "n_labelled_A": len(a_labels),
        "n_labelled_B": len(b_labels),
        "n_scored": len(scored),
        "scored_item_ids": scored,
        "unlabelled_by_B": missing,
        "agreement": block,
        "disagreement_item_ids": disagreements,
    }

    kappa = block["kappa"]
    print()
    print("  === Cohen's kappa, annotator A vs annotator B ===")
    print(f"  observed agreement    : {block['observed_agreement']:.3f}")
    print(f"  expected agreement    : {block['expected_agreement']:.3f}")
    print(f"  kappa                 : "
          f"{'n/a (degenerate margin)' if kappa is None else f'{kappa:.3f}'}")
    print("  confusion:")
    for a in LABELS:
        for b in LABELS:
            print(f"    A={a:<9} B={b:<9} {block['confusion'][f'A_{a}__B_{b}']}")
    print(f"  disagreements         : {disagreements or 'none'}")
    print()
    print("  No gate applies here. A low kappa indicts the rubric, not the sample")
    print("  (corpus_defect_audit.md:278) -- sharpen ANNOTATION_GUIDE.md sections 2.2-2.3 and")
    print("  say so; do not re-label to move the number.")
    print("  Next: adjudicate the disagreements above, then publish the SciQ mismatch rate")
    print("  as <confirmed MISMATCH count>/884 quoting n, kappa and the annotator configuration.")

    if args.out:
        out_path = ensure_parent(resolve(args.out))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
