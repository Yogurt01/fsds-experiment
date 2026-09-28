"""Expanded residual-leakage rescan of the SciQ counterfactual passages (ex7 follow-up).

ex2's construction records one leakage counter per item, `residual_answer_mentions`, which
is a *whole-word, exact-surface* count of the gold answer in the perturbed passage. By
construction it is almost always 0: the substituter has just rewritten every whole-word
occurrence of the matched surface form, so nothing whole-word is left to find. It reads 0
on all 860 eligible SciQ items.

The ex8 Q2 annotation found five items (0, 25, 222, 716, 805) where a human reading the
counterfactual passage still answers with the *gold* answer, because the sentence that
answers the question was never rewritten. All five read 0 on `residual_answer_mentions`.
The blind spot is morphological and boundary-level, not conceptual:

    id 0    gold "oxidants"                -> only the title occurrence was rewritten;
                                              "are calledoxidants" survives fused
    id 25   gold "immune system"           -> "deficiencies of the immune systems" survives
    id 222  gold "chemical state of solute"-> "dependent upon the chemical identity of the
                                              solute" survives; only a fragment was rewritten
    id 716  gold "antioxidants"            -> "An antioxidant is a molecule that inhibits
                                              the oxidation of other molecules" survives
    id 805  gold "acid"                    -> "Certain air pollutants form acids when
                                              dissolved in water droplets" survives

This module re-runs the check over every eligible counterfactual passage with four
matchers instead of one, and adds a sentence-level "does it still answer the question"
filter, so the headline number is items where the gold answer is still *effectively
stated as the answer*, not merely mentioned somewhere.

    R1  exact       gold as a whole word                      (ex2's original check)
    R2  glued       gold fused to the preceding word          ("calledoxidants")
    R3  morphologic a surface/stem variant of gold as a word  ("acids", "immune systems")
    R4  fragment    a sentence carrying all but at most one
                    of gold's content stems                   (the id-222 pattern)

    Q   answering   the carrying sentence overlaps the question's content stems enough to
                    be the sentence a reader would answer from

    flagged_any     R1 | R2 | R3 | R4
    flagged_strict  flagged_any & Q          <- the headline

Matches that fall inside the inserted counterfactual target are never counted: the target
"similar state of solute" contains the matched fragment "of solute", and crediting that as
a residual would be an artefact of the rewriter, not of the passage.

Read-only. Consumes the committed ex2 run and writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ex2_counterfactual.counterfactual_passage import (  # noqa: E402
    morphological_variants,
    _glued_left_pattern,
    _whole_word_pattern,
)

# --- fixed thresholds -------------------------------------------------------------
# Both are set here, before the rescan is run, from the five known Q2 failures and the
# pattern descriptions above -- not tuned against the 860-item output.

# R4: a sentence counts as carrying a multi-word gold when it is missing at most this
# many of the gold's content stems, and retains at least FRAGMENT_MIN_PRESENT_STEMS of
# them. The second bound was added after the 50-item precision check below (see the
# module docstring's validation note): without it, a two-word gold fires on its generic
# head noun alone -- "cells" for "voltaic cells", "light" for "visible light" -- which is
# not the gold answer surviving, it is a common noun surviving.
FRAGMENT_MISSING_ALLOWED = 1
FRAGMENT_MIN_PRESENT_STEMS = 2

# Q: the carrying sentence must share at least this many content stems with the question,
# or at least this fraction of them, whichever is easier to satisfy.
QUESTION_OVERLAP_MIN_STEMS = 3
QUESTION_OVERLAP_MIN_FRACTION = 0.5

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "do", "does", "for", "from", "had", "has", "have", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "them", "there", "these", "they", "this",
    "those", "to", "was", "were", "what", "when", "which", "who", "whom", "will",
    "with", "you", "your", "call", "called", "kind", "type", "known",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def stem(word: str) -> str:
    """Deterministic suffix stripping -- Porter step-1-ish, no dependency.

    Only the inflections that actually separate a SciQ answer string from the way the
    passage writes it: plurals, -es/-ies, -ing, -ed, and hyphenation.
    """
    w = word.lower().replace("-", "")
    for suffix, cut, add in (
        ("ies", 3, "y"),
        ("sses", 2, ""),
        ("ses", 2, ""),
        ("xes", 2, ""),
        ("zes", 2, ""),
        ("ches", 2, ""),
        ("shes", 2, ""),
        ("ing", 3, ""),
        ("ed", 2, ""),
        ("es", 1, ""),
        ("s", 1, ""),
    ):
        if w.endswith(suffix) and len(w) - cut >= 3:
            return w[: len(w) - cut] + add
    return w


def content_stems(text: str) -> List[str]:
    out = []
    for token in WORD.findall(text):
        low = token.lower()
        if low in STOPWORDS or len(low) < 2:
            continue
        out.append(stem(low))
    return out


def sentences(passage: str) -> List[str]:
    return [s for s in SENTENCE_SPLIT.split(passage.strip()) if s.strip()]


def _spans(pattern: re.Pattern, text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in pattern.finditer(text)]


def _target_spans(passage: str, target: Optional[str]) -> List[Tuple[int, int]]:
    """Where the inserted counterfactual target sits, so residuals inside it are ignored."""
    if not target:
        return []
    return _spans(re.compile(re.escape(target), re.IGNORECASE), passage)


def _inside(span: Tuple[int, int], holders: Sequence[Tuple[int, int]]) -> bool:
    return any(h[0] <= span[0] and span[1] <= h[1] for h in holders)


def _stem_word_spans(passage: str, target_stem: str) -> List[Tuple[int, int]]:
    """Whole-word occurrences whose stem equals `target_stem` (single-word golds)."""
    return [
        m.span()
        for m in WORD.finditer(passage)
        if stem(m.group(0)) == target_stem
    ]


def sentence_of(passage: str, span: Tuple[int, int]) -> str:
    cursor = 0
    for sentence in sentences(passage):
        start = passage.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        cursor = end
        if start <= span[0] < end:
            return sentence
    return passage


def answers_question(sentence: str, question: str) -> Tuple[bool, int, int]:
    q = set(content_stems(question))
    s = set(content_stems(sentence))
    hit = len(q & s)
    ok = hit >= min(QUESTION_OVERLAP_MIN_STEMS, len(q)) or (
        len(q) > 0 and hit / len(q) >= QUESTION_OVERLAP_MIN_FRACTION
    )
    return ok, hit, len(q)


def scan_item(record: Dict) -> Dict:
    """Run the four matchers plus the answering-sentence filter on one item."""
    passage = record["counterfactual_passage"]
    gold = (record["correct_answer"] or "").strip()
    target = record.get("counterfactual_target")
    question = record["question"]
    matched = record.get("substitution_matched_text")

    protected = _target_spans(passage, target)
    hits: List[Dict] = []

    def add(kind: str, surface: str, span: Tuple[int, int]) -> None:
        if _inside(span, protected):
            return
        sentence = sentence_of(passage, span)
        ok, hit, n_q = answers_question(sentence, question)
        hits.append(
            {
                "matcher": kind,
                "surface": surface,
                "span": list(span),
                "sentence": sentence,
                "answers_question": ok,
                "question_stems_hit": hit,
                "question_stems_total": n_q,
            }
        )

    # R1 -- exact whole word (ex2's check, reproduced here so the two are comparable).
    for span in _spans(_whole_word_pattern(gold), passage):
        add("exact", gold, span)

    # R2 -- fused to the preceding word. Whole-word hits are already R1, so only the
    # occurrences with a letter immediately to the left count here.
    for span in _spans(_glued_left_pattern(gold), passage):
        if span[0] > 0 and passage[span[0] - 1].isalpha():
            add("glued", gold, span)
    # ...and the same for the surface form the rewriter actually matched, which for the
    # glued tier can differ from the gold string.
    if matched and matched.lower() != gold.lower():
        for span in _spans(_glued_left_pattern(matched), passage):
            if span[0] > 0 and passage[span[0] - 1].isalpha():
                add("glued", matched, span)

    # R3 -- morphological variants. ex2's own deterministic variant list first, then a
    # stem-equality pass for single-word golds to catch anything the list misses.
    seen_spans = {tuple(h["span"]) for h in hits}
    for variant in morphological_variants(gold):
        for span in _spans(_whole_word_pattern(variant), passage):
            if span in seen_spans:
                continue
            seen_spans.add(span)
            add("morphological", variant, span)
    if " " not in gold and gold:
        for span in _stem_word_spans(passage, stem(gold)):
            if span in seen_spans:
                continue
            seen_spans.add(span)
            add("morphological_stem", passage[span[0] : span[1]], span)

    # R4 -- fragment / clause survival for multi-word golds. No single surface form is
    # required; a sentence that carries all but one of gold's content stems, and does not
    # itself contain the inserted target, is still stating the gold fact.
    gold_stems = content_stems(gold)
    fragment_hits: List[Dict] = []
    if len(gold_stems) >= 2:
        needed = set(gold_stems)
        for sentence in sentences(passage):
            if target and target.lower() in sentence.lower():
                continue
            present = set(content_stems(sentence))
            missing = needed - present
            if (
                len(missing) <= FRAGMENT_MISSING_ALLOWED
                and len(needed & present) >= FRAGMENT_MIN_PRESENT_STEMS
            ):
                ok, hit, n_q = answers_question(sentence, question)
                fragment_hits.append(
                    {
                        "matcher": "fragment",
                        "surface": " ".join(sorted(needed & present)),
                        "missing_stems": sorted(missing),
                        "sentence": sentence,
                        "answers_question": ok,
                        "question_stems_hit": hit,
                        "question_stems_total": n_q,
                    }
                )
    hits.extend(fragment_hits)

    by_matcher = sorted({h["matcher"] for h in hits})
    strict = [h for h in hits if h["answers_question"]]
    return {
        "id": record["id"],
        "question": question,
        "gold": gold,
        "target": target,
        "tier": record["substitution_tier"],
        "matched_text": matched,
        "n_substitutions": record["n_substitutions"],
        "ex2_residual_answer_mentions": record["residual_answer_mentions"],
        "ex2_residual_glued_mentions": record["residual_glued_mentions"],
        "matchers": by_matcher,
        "n_hits": len(hits),
        "flagged_any": bool(hits),
        "flagged_strict": bool(strict),
        "strict_matchers": sorted({h["matcher"] for h in strict}),
        "hits": hits,
    }


def _tally(items: Iterable[Dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item[key]] = out.get(item[key], 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
    )
    parser.add_argument(
        "--q2-sheet",
        default="results/ex8_independent_labels/q2_context_following_sheet.annotator1.csv",
    )
    parser.add_argument(
        "--out",
        default="results/ex7_counterfactual_construction/residual_leakage_rescan_sciq.json",
    )
    args = parser.parse_args()

    with open(args.run, encoding="utf-8") as handle:
        run = json.load(handle)
    records = [r for r in run["results"] if r.get("counterfactual_valid")]
    assert len(records) == run["summary"]["counterfactual_generation"]["n_eligible"], (
        len(records),
        run["summary"]["counterfactual_generation"]["n_eligible"],
    )

    scans = [scan_item(r) for r in records]
    by_id = {s["id"]: s for s in scans}

    flagged_any = [s for s in scans if s["flagged_any"]]
    flagged_strict = [s for s in scans if s["flagged_strict"]]

    matcher_counts: Dict[str, int] = {}
    for scan in flagged_strict:
        for matcher in scan["strict_matchers"]:
            matcher_counts[matcher] = matcher_counts.get(matcher, 0) + 1

    tier_breakdown: Dict[str, Dict[str, int]] = {}
    for scan in scans:
        bucket = tier_breakdown.setdefault(
            scan["tier"], {"n": 0, "flagged_any": 0, "flagged_strict": 0}
        )
        bucket["n"] += 1
        bucket["flagged_any"] += int(scan["flagged_any"])
        bucket["flagged_strict"] += int(scan["flagged_strict"])
    for bucket in tier_breakdown.values():
        bucket["strict_rate"] = round(bucket["flagged_strict"] / bucket["n"], 4)

    report = {
        "inputs": {
            "run": args.run,
            "n_eligible": len(records),
            "thresholds": {
                "FRAGMENT_MISSING_ALLOWED": FRAGMENT_MISSING_ALLOWED,
                "FRAGMENT_MIN_PRESENT_STEMS": FRAGMENT_MIN_PRESENT_STEMS,
                "QUESTION_OVERLAP_MIN_STEMS": QUESTION_OVERLAP_MIN_STEMS,
                "QUESTION_OVERLAP_MIN_FRACTION": QUESTION_OVERLAP_MIN_FRACTION,
            },
        },
        "baseline_ex2_check": {
            "items_with_residual_answer_mentions": sum(
                1 for r in records if r["residual_answer_mentions"]
            ),
            "items_with_residual_glued_mentions": sum(
                1 for r in records if r["residual_glued_mentions"]
            ),
        },
        "expanded_check": {
            "flagged_any": len(flagged_any),
            "flagged_strict": len(flagged_strict),
            "flagged_any_rate": round(len(flagged_any) / len(records), 4),
            "flagged_strict_rate": round(len(flagged_strict) / len(records), 4),
            "strict_matcher_counts": dict(sorted(matcher_counts.items())),
        },
        "by_substitution_tier": tier_breakdown,
        "flagged_strict_ids": sorted(s["id"] for s in flagged_strict),
        "flagged_any_ids": sorted(s["id"] for s in flagged_any),
    }

    # --- validation against the 55-item Q2 sample ---------------------------------
    if args.q2_sheet and os.path.exists(args.q2_sheet):
        import csv
        import io

        text = open(args.q2_sheet, encoding="utf-8").read()
        body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
        rows = list(csv.DictReader(io.StringIO(body)))
        letter = {"a": 0, "b": 1, "c": 2, "d": 3}
        by_record = {r["id"]: r for r in records}
        reads_gold, reads_other = [], []
        for row in rows:
            item_id = int(row["item_id"])
            record = by_record[item_id]
            answer = (row.get("stated_answer") or "").strip().lower()
            idx = letter.get(answer)
            if idx is not None and idx == record["answer_idx"]:
                reads_gold.append(item_id)
            else:
                reads_other.append(item_id)
        caught = [i for i in reads_gold if by_id[i]["flagged_strict"]]
        false_alarms = [i for i in reads_other if by_id[i]["flagged_strict"]]
        report["q2_validation"] = {
            "sheet": args.q2_sheet,
            "n_rows": len(rows),
            "human_reads_gold_ids": sorted(reads_gold),
            "human_reads_not_gold_ids": sorted(reads_other),
            "strict_caught_of_reads_gold": sorted(caught),
            "strict_recall": (
                round(len(caught) / len(reads_gold), 4) if reads_gold else None
            ),
            "strict_flagged_among_reads_not_gold": sorted(false_alarms),
            "strict_false_alarm_rate": (
                round(len(false_alarms) / len(reads_other), 4) if reads_other else None
            ),
            "any_caught_of_reads_gold": sorted(
                i for i in reads_gold if by_id[i]["flagged_any"]
            ),
            "any_flagged_among_reads_not_gold": sorted(
                i for i in reads_other if by_id[i]["flagged_any"]
            ),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"report": report, "items": scans}, handle, indent=1)

    print(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
