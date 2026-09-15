"""Derive a blind annotation copy of the judge-validation sheet (Task A).

Why this exists
---------------
`validation_sheet_pilot_bidir.csv` is self-contained for scoring: it carries the judge's own
verdicts so `compute_kappa.py` can read one file. That makes it hostile to the annotator, whose
independence from the judge is the entire measurement (see docs/NEXT_PHASE_HANDOFF.md section
4.2). The sheet's header block asks the annotator to hide those columns by hand; this script
removes them instead, so the file the human opens cannot leak.

It is a *derivation*, never a mutation: the original is opened read-only and its SHA-256 is
recorded in the manifest so the pair can be re-verified later.

What is dropped, and why
------------------------
    judge_forward   the judge's verdict -- the quantity being measured
    judge_reverse   the same verdict with reference and student swapped
    judge_agree     whether those two matched
    stratum         build_validation_sample.py:99 builds it as
                    f"{dataset}/{cell}/" + ("agree" if agree else "flip"), so it re-encodes
                    judge_agree verbatim; hiding judge_agree while keeping this hides nothing
    cell            A_prime / B_prime differ sharply in base rate of correctness, because B_prime
                    supplies the supporting fact in the prompt; knowing the cell tells the
                    annotator how likely the answer is to be right before they read it
    dataset         same argument, and a component of stratum
    item_id         not needed for the judgement, not unique across the sheet (the same source
                    item appears in both cells), and withholding it materially strengthens
                    pair_key: with item_id in hand, reversing the hash collapses to guessing
                    dataset x cell

What is kept
------------
    pair_key        NEW. sha256("dataset|cell|item_id")[:12]. Unique, opaque, and stable across
                    any regeneration of the sheet, since it is derived from the source pair
                    rather than from row position. Obfuscation, not secrecy: with 74 rows and
                    this docstring, it is brute-forceable by anyone who wants to. The threat
                    model is the annotator's own accidental exposure, not an adversary.
    row_id          the original's own key; unique, and it travels with the row, so the join
                    survives the annotator sorting or filtering in a spreadsheet
    question        rule A1 judges meaning *in the context of the question*
    gold_answer     the reference half of the comparison
    model_answer    the answer under evaluation
    human_verdict   empty, for the annotator

Usage:
    uv run --active python code/ex4_free_response/build_blind_sheet.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent, resolve

SOURCE = "results/ex4_free_response/validation_sheet_pilot_bidir.csv"
DEFAULT_OUT = "results/ex4_free_response/validation_sheet_pilot_bidir_BLIND.csv"
DEFAULT_MANIFEST = "results/ex4_free_response/validation_sheet_pilot_bidir_BLIND_manifest.json"

KEY_FIELDS = ("dataset", "cell", "item_id")
KEY_SEPARATOR = "|"
KEY_LENGTH = 12

COLUMNS = ["pair_key", "row_id", "question", "gold_answer", "model_answer", "human_verdict"]

DROPPED = {
    "judge_forward": "the judge's verdict -- the quantity this sheet exists to measure against",
    "judge_reverse": "the judge's verdict with reference and student swapped",
    "judge_agree": "whether the two judge passes matched; compute_kappa.py:111-119 scores every "
                   "non-matching row INCORRECT under the both-directions mechanism",
    "stratum": "build_validation_sample.py:99 concatenates dataset, cell and the judge-agreement "
               "flag, so it re-encodes judge_agree verbatim",
    "cell": "the two experimental cells differ sharply in base rate of correctness, because one "
            "supplies the supporting fact in the prompt and the other does not; knowing a row's "
            "cell signals how likely its answer is to be right before it is read. Value names "
            "and measured rates are withheld here on purpose -- see docs/guides/ANNOTATION_GUIDE.md "
            "section 1.1 and docs/ex4_free_response/plan_option_free_response_experiment.md, after labelling",
    "dataset": "the two datasets also differ sharply in base rate of correctness, for reasons "
               "unrelated to any individual item; also a component of stratum",
    "item_id": "not needed for the judgement, not unique across the sheet, and its absence "
               "strengthens pair_key against casual reversal",
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(row: Dict[str, str]) -> str:
    raw = KEY_SEPARATOR.join(str(row[field]) for field in KEY_FIELDS)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:KEY_LENGTH]


def read_source(path: str) -> List[Dict[str, str]]:
    with open(resolve(path), encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


# Terse reasons for the in-file header. The full reasoning lives in DROPPED and reaches the
# manifest; it is kept out of the CSV because some of it (e.g. *which* cell has the higher base
# rate of correctness) is itself mild priming for the annotator reading this file while labelling.
DROPPED_SHORT = {
    "judge_forward": "the judge's verdict -- the quantity being measured",
    "judge_reverse": "the judge's verdict, swapped direction",
    "judge_agree": "whether the two judge passes matched",
    "stratum": "re-encodes judge_agree verbatim",
    "cell": "correlates with the base rate of correctness",
    "dataset": "correlates with the base rate of correctness",
    "item_id": "not needed to judge; not unique; withholding it strengthens pair_key",
}


def instructions(manifest_path: str) -> List[str]:
    lines = [
        "# BLIND VALIDATION SHEET -- free-response judge (Task A)",
        "# Derived from validation_sheet_pilot_bidir.csv by code/ex4_free_response/build_blind_sheet.py.",
        "# The original is unmodified; this copy exists so the judge's verdicts cannot be seen.",
        "#",
        "# Fill ONLY the `human_verdict` column with CORRECT or INCORRECT.",
        "# CORRECT means: the model_answer means the same thing as the gold_answer, in the context",
        "# of the question. Ignore spelling, capitalisation and phrasing. A more specific or more",
        "# general answer counts as CORRECT only if it identifies the same thing.",
        "# Full rubric, with worked examples: docs/guides/ANNOTATION_GUIDE.md section 1.",
        "#",
        "# DROPPED COLUMNS AND WHY:",
    ]
    width = max(len(name) for name in DROPPED_SHORT)
    for name, reason in DROPPED_SHORT.items():
        lines.append(f"#   {name:<{width}}  {reason}")
    lines += [
        "#   (full reasoning is in the manifest, deliberately not repeated here -- some of it",
        "#    would itself prime the annotator reading this file)",
        "#",
        "# pair_key is an opaque per-item hash: unique, and stable across any regeneration of the",
        "#   sheet. row_id is the original's own key and survives sorting. Derivation: manifest.",
        f"# Manifest (drops, derivation, checksums): {os.path.basename(manifest_path)}",
        "#",
        "# When done, the verdicts are joined back on row_id and scored with compute_kappa.py.",
    ]
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the blind annotation copy of the sheet.")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing blind sheet. Refuses by default once it has "
                             "verdicts in it.")
    args = parser.parse_args(argv)

    source_path = resolve(args.source)
    source_hash_before = sha256_file(source_path)
    rows = read_source(args.source)

    keys = [pair_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        print(f"  ERROR: pair_key collides -- {len(keys)} rows, {len(set(keys))} distinct keys.")
        return 1

    out_path = ensure_parent(resolve(args.out))
    if os.path.exists(out_path) and not args.force:
        with open(out_path, encoding="utf-8") as handle:
            existing = [line for line in handle if not line.startswith("#")]
        filled = sum(1 for row in csv.DictReader(existing)
                     if (row.get("human_verdict") or "").strip())
        if filled:
            print(f"  REFUSING to overwrite {out_path}: {filled} rows already carry a verdict.")
            print("  Re-run with --force only if you intend to discard those labels.")
            return 1

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        for line in instructions(args.manifest):
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for key, row in zip(keys, rows):
            writer.writerow({
                "pair_key": key,
                "row_id": row["row_id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "model_answer": row["model_answer"],
                "human_verdict": "",
            })

    manifest = {
        "_provenance": {
            "what": "Blind annotation copy of the free-response judge-validation sheet (Task A).",
            "created": "2026-09-08",
            "why": "The source sheet carries the judge's own verdicts so compute_kappa.py can "
                   "read one file. Human independence from that judge is the whole measurement "
                   "(docs/NEXT_PHASE_HANDOFF.md section 4.2), so the annotator gets a copy the "
                   "verdicts cannot leak through.",
            "generator": "code/ex4_free_response/build_blind_sheet.py",
            "source_is_unmodified": True,
            "rubric": "docs/guides/ANNOTATION_GUIDE.md section 1",
        },
        "source": {
            "path": args.source,
            "sha256": source_hash_before,
            "n_rows": len(rows),
        },
        "blind_copy": {
            "path": args.out,
            "columns": COLUMNS,
            "n_rows": len(rows),
            "row_order": "identical to source; rows are copied in file order and not re-sorted",
        },
        "join_key": {
            "primary": "row_id",
            "primary_note": "the source's own key; unique across the sheet and carried on the "
                            "row, so the join survives the annotator sorting or filtering",
            "stable": "pair_key",
            "derivation": f"sha256('{KEY_SEPARATOR.join(KEY_FIELDS)}'"
                          f".format(**row).encode('utf-8')).hexdigest()[:{KEY_LENGTH}]",
            "derivation_plain": f"SHA-256 of the {KEY_SEPARATOR!r}-joined values of "
                                f"{list(KEY_FIELDS)}, truncated to {KEY_LENGTH} hex characters",
            "why_not_item_id_alone": "not unique -- the same source item appears in more than one "
                                     "cell, so item_id collides across rows",
            "why_not_row_id_alone": "row_id is assigned by position after the sampler's shuffle "
                                    "(build_validation_sample.py:136-139), so it would not "
                                    "survive a regeneration of the sheet",
            "reversibility": "pair_key is obfuscation, not secrecy: the derivation is documented "
                             "here and the input space is small, so it is brute-forceable by "
                             "anyone who wants to. The threat model is the annotator's own "
                             "accidental exposure, not an adversary.",
            "no_mapping_stored": "this manifest deliberately stores the derivation formula only, "
                                 "never a pair_key -> dataset/cell/item_id table, which would "
                                 "reintroduce the values it exists to withhold",
        },
        "dropped_columns": DROPPED,
        "kept_columns": {
            "pair_key": "opaque stable join key (new; not present in the source)",
            "row_id": "join key back to the source",
            "question": "rule A1 judges meaning in the context of the question",
            "gold_answer": "the reference half of the comparison",
            "model_answer": "the answer under evaluation",
            "human_verdict": "empty; the column the annotator fills",
        },
        "accepted_values": {
            "CORRECT": ["CORRECT", "C", "1", "Y", "YES", "TRUE"],
            "INCORRECT": ["INCORRECT", "I", "0", "N", "NO", "FALSE"],
            "source": "code/ex4_free_response/compute_kappa.py:76-82",
        },
    }
    manifest_path = ensure_parent(resolve(args.manifest))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    source_hash_after = sha256_file(source_path)
    print(f"  source rows           : {len(rows)}")
    print(f"  blind rows            : {len(rows)}")
    print(f"  pair_key distinct     : {len(set(keys))}")
    print(f"  columns kept          : {len(COLUMNS)}")
    print(f"  columns dropped       : {len(DROPPED)}")
    print(f"  source sha256 before  : {source_hash_before}")
    print(f"  source sha256 after   : {source_hash_after}")
    print(f"  source untouched      : {source_hash_before == source_hash_after}")
    print(f"  wrote {out_path}")
    print(f"  wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
