"""Counterfactual passage generation for Pillar 3.

The perturbation is deliberately *minimal and lexical*: the ground-truth answer string
is located inside the SciQ support passage and rewritten as one of the question's own
distractors. Everything else in the passage is left untouched, so Setting C differs from
Setting B only in which option the context supports.

    original passage        "Dissociation is the separation of ions ..."
    correct_answer          "dissociation"
    chosen distractor       "combustion"
    counterfactual passage  "Combustion is the separation of ions ..."

A model that genuinely reads the context must now answer "combustion"; a model that is
riding a parametric prior or an option-surface shortcut will keep answering
"dissociation". That flip (or its absence) is the Pillar 3 signal.

Four matching tiers are tried in order, and the tier used is recorded per sample so the
analysis can be restricted to the cleanest ones (see `--min-substitution-tier`):

    exact        the answer appears verbatim as a whole word
    glued        the answer appears with its left boundary fused to the previous word
                 (a real SciQ artefact, e.g. "are calledoxidants")
    morphological a plural/singular/hyphenation variant of the answer appears
    partial      only a contiguous sub-phrase of a multi-word answer appears

If no tier matches, the passage simply does not state the fact lexically; the sample is
flagged `counterfactual_valid = False` and is excluded from every Setting C metric.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# Tier names ordered from the most to the least faithful substitution.
TIERS = ("exact", "glued", "morphological", "partial")
TIER_RANK = {tier: rank for rank, tier in enumerate(TIERS)}

# A partial match must be at least this many characters, so that a multi-word answer is
# never rewritten on the strength of a short function word such as "of" or "the".
MIN_PARTIAL_CHARS = 4


def _whole_word_pattern(text: str) -> re.Pattern:
    """Case-insensitive whole-word match ('oxidant' must not match 'oxidants')."""
    return re.compile(
        r"(?<![A-Za-z])" + re.escape(text) + r"(?![A-Za-z])", re.IGNORECASE
    )


def _glued_left_pattern(text: str) -> re.Pattern:
    """Whole-word on the right only, so 'calledoxidants' is still a hit."""
    return re.compile(re.escape(text) + r"(?![A-Za-z])", re.IGNORECASE)


def morphological_variants(phrase: str) -> List[str]:
    """Cheap surface variants of a phrase (no NLP dependency, deterministic order).

    Covers the inflection and hyphenation differences that make up almost all of the
    near-misses between a SciQ answer string and its support passage.
    """
    variants: List[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate != phrase and candidate not in variants:
            variants.append(candidate)

    lowered = phrase.lower()
    if lowered.endswith("ies"):
        add(phrase[:-3] + "y")
    if lowered.endswith("es"):
        add(phrase[:-2])
    if lowered.endswith("s"):
        add(phrase[:-1])
    if lowered.endswith("y"):
        add(phrase[:-1] + "ies")
    add(phrase + "s")
    add(phrase + "es")
    if "-" in phrase:
        add(phrase.replace("-", " "))
        add(phrase.replace("-", ""))
    if " " in phrase:
        add(phrase.replace(" ", "-"))
    return variants


def _match_case(source: str, replacement: str) -> str:
    """Give `replacement` the capitalisation pattern of the text it replaces."""
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def find_substitution_span(passage: str, answer: str) -> Optional[Tuple[str, str]]:
    """Locate the answer inside the passage.

    Returns (tier, matched_surface_form) for the first tier that hits, or None.
    """
    answer = answer.strip()
    if not answer:
        return None

    if _whole_word_pattern(answer).search(passage):
        return "exact", answer

    if _glued_left_pattern(answer).search(passage):
        return "glued", answer

    for variant in morphological_variants(answer):
        if _whole_word_pattern(variant).search(passage):
            return "morphological", variant

    # Longest contiguous word n-gram of a multi-word answer, longest first.
    words = answer.split()
    for size in range(len(words) - 1, 0, -1):
        for start in range(len(words) - size + 1):
            gram = " ".join(words[start : start + size])
            if len(gram) < MIN_PARTIAL_CHARS:
                continue
            for candidate in [gram] + morphological_variants(gram):
                if _whole_word_pattern(candidate).search(passage):
                    return "partial", candidate
    return None


def choose_distractor(
    passage: str, options: Sequence[str], answer_idx: int
) -> Tuple[int, Dict]:
    """Pick the distractor that will become the counterfactual target.

    Preference order (all deterministic, no randomness):
      1. the distractor must not already occur in the passage -- otherwise the perturbed
         passage would mention two competing targets;
      2. it should not overlap the answer as a substring (e.g. 'oxidants' inside
         'antioxidants'), which would blur the option surfaces;
      3. its word count should be as close as possible to the answer's, so the rewritten
         sentence stays grammatically plausible;
      4. ties are broken by the original option index.
    """
    answer = options[answer_idx]
    answer_words = len(answer.split())
    answer_lower = answer.lower()

    ranked = []
    for index, option in enumerate(options):
        if index == answer_idx:
            continue
        option_lower = option.lower()
        already_present = bool(_whole_word_pattern(option).search(passage))
        overlaps_answer = option_lower in answer_lower or answer_lower in option_lower
        ranked.append(
            (
                (
                    int(already_present),          # 0 is better
                    int(overlaps_answer),          # 0 is better
                    abs(len(option.split()) - answer_words),
                    index,
                ),
                index,
                {"already_present": already_present, "overlaps_answer": overlaps_answer},
            )
        )

    ranked.sort(key=lambda item: item[0])
    _, chosen_index, flags = ranked[0]
    return chosen_index, flags


def _substitute(passage: str, matched: str, tier: str, replacement: str) -> Tuple[str, int]:
    """Replace every occurrence of `matched` with `replacement`, preserving case."""
    pattern = _glued_left_pattern(matched) if tier == "glued" else _whole_word_pattern(matched)
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        count += 1
        return _match_case(match.group(0), replacement)

    return pattern.sub(repl, passage), count


def build_counterfactual(sample: Dict) -> Dict:
    """Generate the counterfactual passage for one SciQ sample.

    The returned dict is merged into the sample's result record; the original sample is
    never mutated and no raw data file is touched.
    """
    passage = sample["passage"]
    options = list(sample["options"])
    answer_idx = sample["answer_idx"]
    answer = options[answer_idx]

    target_idx, distractor_flags = choose_distractor(passage, options, answer_idx)
    target = options[target_idx]

    located = find_substitution_span(passage, answer)
    if located is None:
        # The passage never states the answer lexically, so there is nothing to flip.
        return {
            "counterfactual_passage": passage,
            "counterfactual_target": None,
            "counterfactual_target_idx": None,
            "counterfactual_valid": False,
            "substitution_tier": "none",
            "substitution_matched_text": None,
            "n_substitutions": 0,
            "residual_answer_mentions": 0,
            "residual_glued_mentions": 0,
            "distractor_already_in_passage": distractor_flags["already_present"],
            "distractor_overlaps_answer": distractor_flags["overlaps_answer"],
        }

    tier, matched = located
    perturbed, n_substitutions = _substitute(passage, matched, tier, target)
    if n_substitutions == 0:  # defensive: the locator and the rewriter must agree
        return {
            "counterfactual_passage": passage,
            "counterfactual_target": None,
            "counterfactual_target_idx": None,
            "counterfactual_valid": False,
            "substitution_tier": "none",
            "substitution_matched_text": matched,
            "n_substitutions": 0,
            "residual_answer_mentions": 0,
            "residual_glued_mentions": 0,
            "distractor_already_in_passage": distractor_flags["already_present"],
            "distractor_overlaps_answer": distractor_flags["overlaps_answer"],
        }

    # Leakage audit. A whole-word residual is impossible after a successful pass, but the
    # answer can survive fused inside a longer word ("exocuticle" keeps "cuticle"), which
    # is left alone on purpose: rewriting it would corrupt an unrelated term.
    residual_whole_word = len(_whole_word_pattern(answer).findall(perturbed))
    residual_glued = len(_glued_left_pattern(matched).findall(perturbed))

    return {
        "counterfactual_passage": perturbed,
        "counterfactual_target": target,
        "counterfactual_target_idx": target_idx,
        "counterfactual_valid": True,
        "substitution_tier": tier,
        "substitution_matched_text": matched,
        "n_substitutions": n_substitutions,
        "residual_answer_mentions": residual_whole_word,
        "residual_glued_mentions": residual_glued,
        "distractor_already_in_passage": distractor_flags["already_present"],
        "distractor_overlaps_answer": distractor_flags["overlaps_answer"],
    }


def tier_at_least(tier: str, minimum: str) -> bool:
    """True when `tier` is at least as faithful as `minimum` ('exact' is the strictest)."""
    if tier == "none":
        return False
    return TIER_RANK[tier] <= TIER_RANK[minimum]
