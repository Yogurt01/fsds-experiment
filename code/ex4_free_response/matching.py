"""Three-stage answer-matching cascade for the option-free free-response experiment.

Stage 1  normalised exact   case/punctuation/article-insensitive string equality
Stage 2  morphological      singular/plural/hyphenation variants, reusing the counterfactual
                            generator's `morphological_variants` so one matching semantics is
                            shared across the codebase
Stage 3  LLM judge          only the residual, adjudicated in context by the same local model

Stages 1 and 2 are deterministic, dependency-free and auditable; stage 3 is where a second model's
judgement enters, so the stage that decided each item is always recorded and accuracy is reported
at all three tiers (`strict` / `normalised` / `judged`).

Rationale for the cascade over the alternatives (exact-only, embedding threshold, judge-everything)
is in docs/plan_option_free_response_experiment.md section 4.2.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex2_counterfactual.counterfactual_passage import morphological_variants

LEADING_ARTICLES = ("the ", "a ", "an ")

# Prefixes an instruction-tuned model prepends despite being told not to.
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:answer|a|ans)\s*(?:is|:)\s*", re.IGNORECASE
)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def postprocess_generation(text: str) -> str:
    """Turn a raw generation into the candidate answer string.

    Greedy decoding with a short budget still produces the occasional `Answer: X` or a trailing
    sentence. Only the first line is kept, a leading answer-prefix is stripped, and surrounding
    quotes/periods are removed. The raw text is retained by the caller so every transformation
    stays inspectable.
    """
    first_line = str(text).strip().split("\n")[0].strip()
    first_line = _ANSWER_PREFIX.sub("", first_line)
    return first_line.strip().strip('"“”\'').rstrip(".").strip()


def normalise(text: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace, drop one leading article."""
    lowered = _PUNCT.sub(" ", str(text).lower())
    collapsed = _WS.sub(" ", lowered).strip()
    for article in LEADING_ARTICLES:
        if collapsed.startswith(article):
            return collapsed[len(article):].strip()
    return collapsed


def _variant_set(text: str) -> set:
    """`text` plus its morphological variants, all normalised."""
    forms = {text} | set(morphological_variants(text))
    # Also vary the final word alone, so "tree ring" reaches "tree rings".
    words = str(text).split()
    if len(words) > 1:
        for variant in morphological_variants(words[-1]):
            forms.add(" ".join(words[:-1] + [variant]))
    return {normalise(f) for f in forms if str(f).strip()}


def match_stage(prediction: str, gold: str) -> Optional[str]:
    """Return "strict" / "normalised", or None when the deterministic stages cannot decide."""
    if not str(prediction).strip():
        return None
    if normalise(prediction) == normalise(gold):
        return "strict"
    if _variant_set(prediction) & _variant_set(gold):
        return "normalised"
    return None


JUDGE_SYSTEM = (
    "You are grading a short-answer science exam. Decide whether the student's answer means the "
    "same thing as the reference answer, in the context of the question. Ignore spelling, "
    "capitalisation, and phrasing. A more specific or more general answer counts as correct only "
    "if it identifies the same thing. Reply with exactly one word: CORRECT or INCORRECT."
)


def judge_user_turn(question: str, gold: str, prediction: str) -> str:
    return (
        f"Question: {str(question).strip()}\n"
        f"Reference answer: {str(gold).strip()}\n"
        f"Student answer: {str(prediction).strip()}"
    )


def accuracy_tiers(records: Sequence[Dict]) -> Dict[str, float]:
    """Accuracy at every tier, reported as a **band** rather than a point estimate.

    The pilot established that the judge decides 76-100% of items and flips 24.3% of its verdicts
    when reference and student are swapped (docs/plan_option_free_response_experiment.md section
    8.4). A single `acc_judged` number therefore hides a large, systematic uncertainty. Instead:

        acc_strict       stage 1 only                                    (hardest floor)
        acc_normalised   stages 1-2, deterministic only                  (defensible floor)
        acc_band_low     deterministic + judge CORRECT in BOTH directions (conservative)
        acc_band_high    deterministic + judge CORRECT in EITHER direction (permissive)

    The width of [acc_band_low, acc_band_high] is the judge's order-instability made explicit.
    `acc_judged` is retained as the forward-only number for continuity with the first pilot, but
    it should not be quoted as the headline.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    deterministic = [r for r in records if r["match_stage"] in ("strict", "normalised")]
    judged_items = [r for r in records if r["match_stage"] == "judge"]

    n_det = len(deterministic)
    both = sum(1 for r in judged_items if r.get("judge_agree") and r.get("judge_forward_correct"))
    either = sum(
        1 for r in judged_items
        if r.get("judge_forward_correct") or r.get("judge_reverse_correct")
    )
    flips = sum(1 for r in judged_items if r.get("judge_agree") is False)

    return {
        "n": n,
        "acc_strict": sum(1 for r in records if r["match_stage"] == "strict") / n,
        "acc_normalised": n_det / n,
        "acc_judged": sum(1 for r in records if r["correct"]) / n,
        "acc_band_low": (n_det + both) / n,
        "acc_band_high": (n_det + either) / n,
        "band_width": (either - both) / n,
        "n_sent_to_judge": len(judged_items),
        "n_judge_forward_correct": sum(1 for r in judged_items if r.get("judge_forward_correct")),
        "n_judge_both_directions_correct": both,
        "n_judge_either_direction_correct": either,
        "n_judge_order_flips": flips,
        "judge_order_flip_rate": (flips / len(judged_items)) if judged_items else None,
    }
