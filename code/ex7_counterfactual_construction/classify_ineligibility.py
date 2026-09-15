"""Split OpenBookQA's Setting-C-ineligible items by ROOT CAUSE. Read-only, no inference.

355 of 500 OBQA items (71.0%) have no lexical span to substitute. The existing docs treat
that as one coverage problem. Inspection says it is at least three, needing different fixes:

  (A) rule-to-instance -- `fact1` states a general rule and the gold answer is an INSTANCE
      the rule generalises over ("birds lay eggs" -> "emus"). There is no answer span in
      the fact, and there cannot be one. Unreachable by any matcher, lexical or semantic.

  (B) paraphrase / synonym -- the fact DOES state the answer's content, in other words
      ("usually coral lives in warm water" -> "tepid seas"). A semantic matcher would find
      this; a whole-word regex cannot. Reachable.

  (C) fact-insufficient -- `fact1` does not determine the gold answer at all; bridging to
      the option needs outside knowledge ("storms can cause a landslide" -> "mud";
      "wax is an electrical insulator" -> "rubber"). This is by OBQA's own design, where
      fact1 is one of several facts a solver must combine. NO counterfactual construction
      can work here: perturbing a fact that does not determine the answer cannot reliably
      flip it, which is a direct mechanism for the elevated `unstable_other` that survives
      even a clean `exact` swap on OBQA.

Only (B) is addressable by better matching; (A) needs the rule perturbed rather than the
answer span; (C) is out of reach of any construction. The three shares therefore set the
ceiling on every candidate strategy and are the main quantitative input to choosing one.

This script does NOT assign the labels. It computes discriminating signals, applies a
transparent heuristic, and writes a stratified hand-check sheet so the heuristic can later
be scored against human labels. The heuristic's own accuracy is unmeasured until that
happens; treat its split as indicative only.

Usage:
    python code/ex7_counterfactual_construction/classify_ineligibility.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

SOURCE = "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json"
OUT_DIR = "results/ex7_counterfactual_construction"
SHEET_N = 60
SEED = 7

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "to", "is", "are", "be", "and", "or", "it", "its",
    "that", "this", "for", "with", "as", "by", "at", "from", "will", "can", "when", "if",
    "some", "then", "into", "out", "up", "down", "more", "less", "their", "them", "they",
}
# Plural common nouns in a one-clause rule usually mark the CATEGORY the gold instantiates
# ("birds lay eggs", "a mammal is warm-blooded"), which is the (A) signature.
CATEGORY_HEADS = {
    "animals", "birds", "mammals", "reptiles", "insects", "plants", "trees", "fish",
    "humans", "people", "metals", "rocks", "minerals", "gases", "liquids", "solids",
    "objects", "materials", "organisms", "predators", "producers", "consumers",
}


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def _content(text: str) -> set:
    return {w for w in _words(text) if w not in STOPWORDS}


def signals(record: Dict) -> Dict:
    fact = record["passage"]
    gold = record["correct_answer"]
    question = record["question"]
    gold_words = _content(gold)
    fact_words = _content(fact)
    return {
        "gold_starts_capital": bool(re.match(r"^[A-Z]", gold.strip())),
        "gold_word_count": len(gold.split()),
        "gold_content_overlap_with_fact": (
            len(gold_words & fact_words) / len(gold_words) if gold_words else 0.0),
        "question_content_overlap_with_fact": (
            lambda q: len(q & fact_words) / len(q) if q else 0.0)(_content(question)),
        "fact_has_category_head": bool(fact_words & CATEGORY_HEADS),
        "fact_word_count": len(fact.split()),
    }


def heuristic_label(sig: Dict) -> str:
    """Transparent, deliberately simple. Accuracy UNMEASURED until the sheet is labelled.

    (A) is signalled by a proper-noun gold or an explicit category head in the fact; (B) by
    the gold sharing no surface with the fact while the QUESTION does -- i.e. the fact is
    on topic but words the answer differently.
    """
    if sig["gold_starts_capital"] or sig["fact_has_category_head"]:
        return "A_rule_to_instance"
    if sig["question_content_overlap_with_fact"] >= 0.5 and sig["gold_content_overlap_with_fact"] == 0.0:
        return "B_paraphrase"
    return "unclassified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    with open(resolve_path(SOURCE), encoding="utf-8") as handle:
        records = json.load(handle)["results"]
    ineligible = [r for r in records if r["substitution_tier"] == "none"]

    rows = []
    for record in ineligible:
        sig = signals(record)
        rows.append({
            "id": record["id"],
            "question": record["question"],
            "fact1": record["passage"],
            "gold_answer": record["correct_answer"],
            "options": record["options"],
            **sig,
            "heuristic_label": heuristic_label(sig),
        })

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["heuristic_label"]] = counts.get(row["heuristic_label"], 0) + 1

    # Stratified hand-check sheet, proportional to the heuristic's own strata.
    import random
    rng = random.Random(SEED)
    by_label: Dict[str, List[Dict]] = {}
    for row in rows:
        by_label.setdefault(row["heuristic_label"], []).append(row)
    sample: List[Dict] = []
    for label, group in sorted(by_label.items()):
        take = max(1, round(SHEET_N * len(group) / len(rows)))
        picked = sorted(group, key=lambda r: r["id"])
        rng.shuffle(picked)
        sample.extend(picked[:take])
    sample.sort(key=lambda r: r["id"])

    sheet_path = ensure_parent(resolve_path(
        os.path.join(args.out_dir, "ineligibility_handcheck_sheet_BLIND.csv")))
    with open(sheet_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # The heuristic label is withheld so an annotator is not anchored by it.
        writer.writerow(["item_id", "question", "fact1", "gold_answer", "options",
                         "cause_A_rule_to_instance__B_paraphrase__C_fact_insufficient__D_other",
                         "rationale"])
        for row in sample:
            writer.writerow([row["id"], row["question"], row["fact1"], row["gold_answer"],
                             " | ".join(row["options"]), "", ""])

    payload = {
        "what": "Root-cause split of OpenBookQA's Setting-C-ineligible items.",
        "source_results_file": SOURCE,
        "n_ineligible": len(ineligible),
        "n_split": len(records),
        "heuristic_counts": counts,
        "heuristic_shares": {k: v / len(rows) for k, v in counts.items()},
        "caveat": (
            "The heuristic is transparent but UNVALIDATED -- no human labels exist yet, so "
            "its accuracy is unknown and these shares are indicative, not measured. The "
            "blind sheet is the instrument for validating it; labelling it is a separate, "
            "unapproved step."
        ),
        "ceiling_note": (
            "Only cause (B) is reachable by a better matcher. Cause (A) requires perturbing "
            "the rule rather than the answer span, which is what target-anchored rewriting "
            "does. Cause (C), where fact1 does not determine the answer at all, is out of "
            "reach of ANY construction method and should be excluded rather than rewritten."
        ),
        "known_gap": (
            "The heuristic has no (C) detector, so (C) items fall into `unclassified`. A "
            "single-pass hand read of 18 `unclassified` items by one reader (no second "
            "rater, no kappa, indicative only) found roughly a third of that stratum to be "
            "(C). Sizing (C) properly is exactly what the hand-check sheet is for."
        ),
        "handcheck_sheet": os.path.relpath(sheet_path, resolve_path(".")),
        "handcheck_sheet_n": len(sample),
        "items": rows,
    }
    out = ensure_parent(resolve_path(os.path.join(args.out_dir, "obqa_ineligibility_causes.json")))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    print(f"OBQA Setting-C-ineligible items: {len(ineligible)} of {len(records)}")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:22s} {count:4d}  ({count/len(rows):5.1%})   [heuristic, UNVALIDATED]")
    print(f"\nWrote {os.path.relpath(out, resolve_path('.'))}")
    print(f"Wrote {payload['handcheck_sheet']} ({len(sample)} items, labels withheld)")


if __name__ == "__main__":
    main()
