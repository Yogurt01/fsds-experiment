"""Build the human-validation sample and annotation sheet for the free-response judge.

Why this exists
---------------
The pilot established that the LLM judge decides 76-100% of free-response items and flips 24.3%
of its verdicts when reference and student are swapped. The judge is therefore not an adjudicator
of last resort but the primary grading mechanism, and its agreement with a human is the validity
of the whole experiment. This script produces the sample a human grades; `compute_kappa.py`
scores the result against the pre-registered kappa >= 0.70 gate.

Sampling
--------
Stratified over the cells that matter for judge reliability:

    dataset      sciq / obqa
    cell         A_prime / B_prime
    agreement    judge agreed in both directions / flipped under swap

The flipped stratum is deliberately **over-sampled to its full size** where possible: those are
the items where the judge is demonstrably unstable, so they carry the most information about
whether its verdicts can be trusted. Sampling within a stratum is seeded and the seed recorded.

Output
------
A CSV with one row per item and an empty `human_verdict` column for the annotator to fill with
CORRECT or INCORRECT. The judge's own verdicts are written to a separate column that the
annotator is instructed not to read first -- see the header block written into the file.

Usage:
    uv run --active python code/ex4_free_response/build_validation_sample.py --tag pilot_bidir
    uv run --active python code/ex4_free_response/build_validation_sample.py --tag full --n 150
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent, resolve

DATASETS = ("sciq", "obqa")
CELLS = ("A_prime", "B_prime")

COLUMNS = [
    "row_id", "dataset", "cell", "item_id",
    "question", "gold_answer", "model_answer",
    "human_verdict",                 # <- the annotator fills ONLY this
    "judge_forward", "judge_reverse", "judge_agree", "stratum",
]

INSTRUCTIONS = [
    "# HUMAN VALIDATION SHEET -- free-response judge",
    "# Fill ONLY the `human_verdict` column with CORRECT or INCORRECT.",
    "# CORRECT means: the model_answer means the same thing as the gold_answer, in the context of",
    "# the question. Ignore spelling, capitalisation and phrasing. A more specific or more general",
    "# answer counts as CORRECT only if it identifies the same thing.",
    "# Please do NOT read the judge_forward / judge_reverse columns before deciding -- they are",
    "# included so the sheet is self-contained for scoring, not as a prompt. Hide them if you can.",
    "# When done, run: uv run --active python code/ex4_free_response/compute_kappa.py --sheet <this file>",
]


def load_records(tag: str, out_dir: str) -> List[Dict]:
    """Every judged item across both datasets and both cells, flattened."""
    rows: List[Dict] = []
    for dataset in DATASETS:
        name = f"results_free_response_{dataset}" + (f"_{tag}" if tag else "") + ".json"
        path = resolve(os.path.join(out_dir, name))
        if not os.path.isfile(path):
            print(f"  (missing, skipped: {path})")
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for cell in CELLS:
            for record in payload["results"].get(cell, []):
                if record["match_stage"] != "judge":
                    continue          # deterministic matches need no human check
                agree = record.get("judge_agree")
                rows.append(
                    {
                        "dataset": dataset,
                        "cell": cell,
                        "item_id": record["id"],
                        "question": record["question"],
                        "gold_answer": record["gold_answer"],
                        "model_answer": record["prediction"],
                        "judge_forward": _verdict(record, "forward"),
                        "judge_reverse": _verdict(record, "reverse"),
                        "judge_agree": "" if agree is None else str(bool(agree)),
                        "stratum": f"{dataset}/{cell}/" + ("agree" if agree else "flip"),
                    }
                )
    return rows


def _verdict(record: Dict, direction: str) -> str:
    blob = record.get("judge") or {}
    if direction in blob:
        return blob[direction]["verdict"]
    return blob.get("verdict", "")          # forward-only pilot format


def stratified_sample(rows: List[Dict], n_target: int, seed: int) -> List[Dict]:
    """Take every `flip` item, then fill the remainder proportionally from the `agree` strata."""
    rng = random.Random(seed)
    flips = [r for r in rows if r["stratum"].endswith("/flip")]
    agrees = [r for r in rows if not r["stratum"].endswith("/flip")]

    chosen = list(flips)
    if len(chosen) > n_target:
        chosen = rng.sample(chosen, n_target)

    remaining = n_target - len(chosen)
    if remaining > 0:
        by_stratum: Dict[str, List[Dict]] = {}
        for row in agrees:
            by_stratum.setdefault(row["stratum"], []).append(row)
        # Proportional allocation, then top up in a stable order if rounding leaves a shortfall.
        total = len(agrees)
        for stratum, items in sorted(by_stratum.items()):
            take = min(len(items), round(remaining * len(items) / total)) if total else 0
            chosen.extend(rng.sample(items, take))
        pool = [r for r in agrees if r not in chosen]
        rng.shuffle(pool)
        while len(chosen) < n_target and pool:
            chosen.append(pool.pop())

    rng.shuffle(chosen)                      # present in random order to avoid ordering effects
    for index, row in enumerate(chosen, start=1):
        row["row_id"] = index
        row["human_verdict"] = ""
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the judge validation sheet.")
    parser.add_argument("--tag", default="pilot_bidir", help="Tag of the run to sample from.")
    parser.add_argument("--results-dir", default="results/ex4_free_response")
    parser.add_argument("--n", type=int, default=150, help="Target sample size.")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out", default="", help="CSV path (default derives from --tag).")
    args = parser.parse_args(argv)

    rows = load_records(args.tag, args.results_dir)
    if not rows:
        print("No judged items found; run the experiment first.", file=sys.stderr)
        return 1

    sample = stratified_sample(rows, min(args.n, len(rows)), args.seed)

    out = args.out or f"results/ex4_free_response/validation_sheet_{args.tag}.csv"
    out_path = ensure_parent(resolve(out))
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        for line in INSTRUCTIONS:
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample)

    print(f"  judged items available : {len(rows)}")
    print(f"  sampled                : {len(sample)} (seed {args.seed})")
    counts: Dict[str, int] = {}
    for row in sample:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    for stratum, count in sorted(counts.items()):
        print(f"    {stratum:28} {count}")
    print(f"  wrote {out_path}")
    print("  Fill the `human_verdict` column, then run compute_kappa.py --sheet <file>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
