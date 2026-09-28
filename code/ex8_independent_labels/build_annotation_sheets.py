"""Build the two blind annotation sheets for the RQ2 independent label set. Read-only inputs.

No model inference. Reads the committed ex2 Setting-C run (for question, options, gold answer,
passage and counterfactual passage) and the committed ex6 score file (for the stratification
only -- no score reaches either sheet).

Two sheets, two separate passes:

  Q1  answerability without the passage, on stratum A + B + C.
      The annotator sees question, four options, gold answer and the ORIGINAL passage, and
      rates 1-4 how likely an unaided student is to answer correctly. This is a human-judged
      analogue of Setting A, and it is what KDA_cont's (1 - P) weight claims to capture.

  Q2  context following, on stratum A only.
      The annotator sees question, four options and the COUNTERFACTUAL passage -- not the
      original -- and says which option that passage states or implies. This is the part Q1
      cannot reach: it tests, by human reading, whether the perturbed passage actually
      licenses the counterfactual target, which is both a validity check on the construction
      and the only non-circular test of F's context-following claim available.

Stratification (computed from KDA_cont and F terciles over all 860 CF-eligible SciQ items):

  A  maximally discordant -- KDA-low & F-high, or KDA-high & F-low.  55 items, taken whole.
     Within these 55 the two metrics rank items at Spearman -0.736, so this is where they
     actually disagree and where a head-to-head can be decided.
  B  adjacent disagreement.  Sampled. Enters the primary sign test alongside A.
  C  concordant.  Sampled. Contributes only to the population-weighted AUC; the designated
     first cut if the time budget binds.

Blindness. Neither sheet carries a score, a Setting-C class, a bucket, a substitution tier, or
a stratum name. Stratum is withheld specifically because it is a deterministic function of
KDA_cont and F, so printing it would leak both rankings. Gold answer IS shown on Q1 by explicit
decision (the rating is unanswerable without knowing which option is correct) and withheld on
Q2 (where naming the gold would give away the substitution).

Ordering. Two properties, both deliberate:

  * The first `--double-n` rows of each sheet are the double-annotation block, built stratified
    rather than random, so stopping early still yields a representative kappa.
  * Every row after that is ordered by PRIORITY -- all remaining stratum A, then B, then C,
    shuffled within each -- so an annotator who simply works top to bottom and runs out of time
    loses stratum C first, which is the designated cut. The sheet still carries no stratum
    column: the priority is encoded in row order alone, so the annotator needs no knowledge of
    which stratum a row belongs to and nothing about either metric leaks.

The two sheets use different shuffle seeds so their row orders do not correspond.

Never overwrites: refuses if an output exists unless --force. Records SHA-256 of both sheets in
the manifest and in a LOCK file, following the Task A/B discipline.

Usage:
    python code/ex8_independent_labels/build_annotation_sheets.py
    python code/ex8_independent_labels/build_annotation_sheets.py --stratum-c 0 --double-n 20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

EX2_RESULTS = "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json"
EX6_SCORES = "results/ex6_psfd_score/psfd_scores_sciq_test_full.json"
OUT_DIR = "results/ex8_independent_labels"

Q1_SHEET = "q1_answerability_sheet.csv"
Q2_SHEET = "q2_context_following_sheet.csv"
MANIFEST = "annotation_manifest.json"
LOCK = "annotation_sheets.LOCK.json"

SEED_Q1 = 20260916
SEED_Q2 = 20260917   # deliberately different: the two sheets must not share a row order
SEED_SAMPLE = 7

Q1_COLUMNS = ["item_id", "question", "option_a", "option_b", "option_c", "option_d",
              "gold_answer", "passage", "rating", "unusable", "notes"]
Q2_COLUMNS = ["item_id", "question", "option_a", "option_b", "option_c", "option_d",
              "passage", "stated_answer", "notes"]

Q1_HEADER = [
    "# Q1 -- ANSWERABILITY WITHOUT THE PASSAGE.  Fill `rating` (1-4) for every row.",
    "# Imagine a typical secondary-school student who has studied general science but has NOT",
    "# studied this particular topic, and who has NOT read the passage. They see only the",
    "# question and the four options. How likely are they to pick the correct answer?",
    "#   1 = Very unlikely  (they would have to guess; the passage is doing all the work)",
    "#   2 = Unlikely       (might narrow to two options, could not reliably choose)",
    "#   3 = Likely         (general knowledge or a clue in the wording usually gets them there)",
    "#   4 = Very likely    (obvious from question and options alone)",
    "# No middle option, on purpose. Commit to a side.",
    "# If a row is unusable (passage about a different topic, incoherent question, duplicate",
    "# options), put X in `unusable` and leave `rating` blank. Do not rate it.",
    "# Do NOT judge whether the keyed answer is factually right, whether the passage is well",
    "# written, or whether YOU know the answer. Judge the imagined student.",
    "# Work top to bottom. Do not reorder rows.",
]

Q2_HEADER = [
    "# Q2 -- WHAT DOES THIS PASSAGE SAY?  Fill `stated_answer` for every row.",
    "# Read the passage. Which of the four options does THIS PASSAGE state or clearly imply",
    "# is the answer to the question?",
    "#   a / b / c / d  = that option",
    "#   none           = the passage does not clearly point to any of them",
    "# Answer ONLY from the passage in front of you. Do not use outside knowledge, do not",
    "# judge whether the passage is factually true, and do not try to recall this item from",
    "# any earlier pass -- several passages here have been edited on purpose.",
    "# Work top to bottom. Do not reorder rows.",
]


def _load(path: str) -> Dict:
    with open(resolve_path(path), encoding="utf-8") as handle:
        return json.load(handle)


def _terciles(values: Sequence[float]) -> List[int]:
    ordered = sorted(values)
    low, high = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]
    return [0 if v <= low else (1 if v <= high else 2) for v in values]


def stratify(scores: Dict) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    items = scores["items"]
    kda = _terciles([i["kda_cont"] for i in items])
    f = _terciles([i["ensemble"]["all"]["F_mean"] for i in items])
    membership: Dict[int, str] = {}
    strata: Dict[str, List[int]] = {"A": [], "B": [], "C": []}
    for item, a, b in zip(items, kda, f):
        if (a == 0 and b == 2) or (a == 2 and b == 0):
            name = "A"
        elif abs(a - b) == 1:
            name = "B"
        else:
            name = "C"
        membership[item["id"]] = name
        strata[name].append(item["id"])
    return membership, strata


def _stratified_front(ids_by_stratum: Dict[str, List[int]], front_n: int,
                      rng: random.Random) -> Tuple[List[int], List[int]]:
    """Split ids into a stratum-proportional front block and the remainder, each shuffled."""
    total = sum(len(v) for v in ids_by_stratum.values())
    front: List[int] = []
    rest: List[int] = []
    for name in sorted(ids_by_stratum):
        pool = list(ids_by_stratum[name])
        rng.shuffle(pool)
        take = round(front_n * len(pool) / total) if total else 0
        front.extend(pool[:take])
        rest.extend(pool[take:])
    # Rounding can miss the target by one or two; move items between blocks to land exactly.
    while len(front) < front_n and rest:
        front.append(rest.pop())
    while len(front) > front_n:
        rest.append(front.pop())
    rng.shuffle(front)
    rng.shuffle(rest)
    return front, rest


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _write_sheet(path: str, header: Sequence[str], columns: Sequence[str],
                 rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for line in header:
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stratum-b", type=int, default=25,
                        help="Items sampled from the adjacent-disagreement stratum.")
    parser.add_argument("--stratum-c", type=int, default=20,
                        help="Items sampled from the concordant stratum. First cut if time binds.")
    parser.add_argument("--double-n", type=int, default=30,
                        help="Size of the Q1 double-annotation front block.")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing sheets.")
    args = parser.parse_args()

    records = {r["id"]: r for r in _load(EX2_RESULTS)["results"]}
    scores = _load(EX6_SCORES)
    membership, strata = stratify(scores)

    rng = random.Random(SEED_SAMPLE)
    chosen: Dict[str, List[int]] = {"A": sorted(strata["A"])}
    for name, size in (("B", args.stratum_b), ("C", args.stratum_c)):
        pool = sorted(strata[name])
        rng.shuffle(pool)
        chosen[name] = sorted(pool[:size])

    q1_ids = {n: chosen[n] for n in ("A", "B", "C") if chosen[n]}
    front, rest = _stratified_front(q1_ids, args.double_n, random.Random(SEED_Q1))
    # Priority order for everything after the double block: A, then B, then C. An annotator who
    # runs short therefore drops stratum C first without needing to know what a stratum is.
    rest_by_stratum: Dict[str, List[int]] = {"A": [], "B": [], "C": []}
    for item_id in rest:
        rest_by_stratum[membership[item_id]].append(item_id)
    rng_rest = random.Random(SEED_Q1 + 1)
    ordered_rest: List[int] = []
    for name in ("A", "B", "C"):
        block = rest_by_stratum[name]
        rng_rest.shuffle(block)
        ordered_rest.extend(block)
    q1_order = front + ordered_rest

    # Q2 covers stratum A only. Its front block is the A-items already in Q1's front block, so
    # the double-annotated sets overlap and one early stop covers both sheets.
    a_set = set(chosen["A"])
    q2_front_ids = [i for i in front if i in a_set]
    q2_rest_ids = [i for i in chosen["A"] if i not in set(q2_front_ids)]
    rng2 = random.Random(SEED_Q2)
    rng2.shuffle(q2_front_ids)
    rng2.shuffle(q2_rest_ids)
    q2_order = q2_front_ids + q2_rest_ids

    def q1_row(item_id: int) -> Dict:
        r = records[item_id]
        options = r["options"]
        return {
            "item_id": item_id, "question": r["question"],
            "option_a": options[0], "option_b": options[1],
            "option_c": options[2], "option_d": options[3],
            "gold_answer": r["correct_answer"], "passage": r["passage"],
            "rating": "", "unusable": "", "notes": "",
        }

    def q2_row(item_id: int) -> Dict:
        r = records[item_id]
        options = r["options"]
        return {
            "item_id": item_id, "question": r["question"],
            "option_a": options[0], "option_b": options[1],
            "option_c": options[2], "option_d": options[3],
            "passage": r["counterfactual_passage"],
            "stated_answer": "", "notes": "",
        }

    out_dir = resolve_path(args.out_dir)
    q1_path = ensure_parent(os.path.join(out_dir, Q1_SHEET))
    q2_path = ensure_parent(os.path.join(out_dir, Q2_SHEET))
    for path in (q1_path, q2_path):
        if os.path.exists(path) and not args.force:
            raise SystemExit(f"{path} exists; refusing to overwrite without --force.")

    _write_sheet(q1_path, Q1_HEADER, Q1_COLUMNS, [q1_row(i) for i in q1_order])
    _write_sheet(q2_path, Q2_HEADER, Q2_COLUMNS, [q2_row(i) for i in q2_order])

    manifest = {
        "what": "Blind annotation sheets for the RQ2 independent label set. No score, class, "
                "bucket, tier or stratum appears in either sheet.",
        "built": "2026-09-16",
        "sources": {"ex2_results": EX2_RESULTS, "ex6_scores": EX6_SCORES},
        "seeds": {"sample": SEED_SAMPLE, "q1_order": SEED_Q1, "q2_order": SEED_Q2},
        "stratification": {
            "basis": "terciles of KDA_cont and of ensemble F over all 860 CF-eligible items",
            "A_max_discordant": {"population": len(strata["A"]), "sampled": len(chosen["A"])},
            "B_adjacent": {"population": len(strata["B"]), "sampled": len(chosen["B"])},
            "C_concordant": {"population": len(strata["C"]), "sampled": len(chosen["C"])},
        },
        "sampling_fractions_for_reweighting": {
            name: (len(chosen[name]) / len(strata[name]) if chosen[name] else 0.0)
            for name in ("A", "B", "C")
        },
        "q1": {"file": Q1_SHEET, "n": len(q1_order), "double_block_n": len(front),
               "double_block_ids": sorted(front),
               "row_order": "rows 1-%d double block (stratified); then remaining A, then B, "
                            "then C" % len(front),
               "priority_boundaries_1_indexed": {
                   "double_block": [1, len(front)],
                   **{name: ([len(front) + 1 + sum(len(rest_by_stratum[n]) for n in ("A","B","C")[:k]),
                              len(front) + sum(len(rest_by_stratum[n]) for n in ("A","B","C")[:k+1])]
                             if rest_by_stratum[name] else None)
                      for k, name in enumerate(("A", "B", "C"))},
               }},
        "q2": {"file": Q2_SHEET, "n": len(q2_order), "double_block_n": len(q2_front_ids),
               "double_block_ids": sorted(q2_front_ids)},
        "stratum_membership": {str(i): membership[i] for i in q1_order},
        "sign_test_eligible_ids": sorted(chosen["A"] + chosen["B"]),
        "note": "stratum_membership is recorded HERE, never in the sheets: it is a "
                "deterministic function of KDA_cont and F and would leak both rankings.",
    }
    manifest_path = ensure_parent(os.path.join(out_dir, MANIFEST))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)

    lock = {
        "what": "SHA-256 of the sheets as issued. Re-verify before scoring so a filled sheet "
                "can be proved to descend from this issue.",
        "built": "2026-09-16",
        Q1_SHEET: _sha256(q1_path),
        Q2_SHEET: _sha256(q2_path),
        MANIFEST: _sha256(manifest_path),
    }
    with open(ensure_parent(os.path.join(out_dir, LOCK)), "w", encoding="utf-8") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=1)

    rel = lambda p: os.path.relpath(p, resolve_path("."))
    print(f"Q1 {len(q1_order):3d} rows (A={len(chosen['A'])} B={len(chosen['B'])} "
          f"C={len(chosen['C'])}), double block = first {len(front)} -> {rel(q1_path)}")
    print(f"Q2 {len(q2_order):3d} rows (stratum A only), double block = first "
          f"{len(q2_front_ids)} -> {rel(q2_path)}")
    print(f"sign-test eligible (A+B) = {len(manifest['sign_test_eligible_ids'])} items")
    print(f"manifest -> {rel(manifest_path)}")


if __name__ == "__main__":
    main()
