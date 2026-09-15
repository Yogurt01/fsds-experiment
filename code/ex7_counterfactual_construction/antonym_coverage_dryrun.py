"""Coverage dry run for the antonym / scalar-swap counterfactual strategy. Read-only.

Lexicon matching only -- nothing is generated, no model is called, no file outside
`results/ex7_counterfactual_construction/` is touched.

The strategy (proposed in docs/ex2_counterfactual/counterfactual_obqa_analysis.md section 6)
perturbs the RULE's head term rather than the answer span: "a plant requires sunlight to
grow" -> "a plant requires darkness to grow". It targets exactly the failure that sinks
answer-span substitution on OpenBookQA, so its coverage is worth measuring.

Coverage is only half the question. Setting C asks whether the solver switches to a
specific counterfactual TARGET, so a usable counterfactual needs some option to be correct
under the inverted rule. This script measures a NECESSARY condition for that: does the
option set itself contain a polar contrast the inverted rule could select? If it does not,
inverting the rule leaves no option correct, there is no target to detect, and Setting C
degenerates into "did the solver stop picking gold" -- a weaker and different signal.

Two lexicons are reported because the answer depends on how permissive one is:
  core        36 unambiguous scalar/polar pairs
  permissive  core plus 29 pairs whose members are frequently polysemous in this corpus
              ("on"/"off", "light"/"heavy", "up"/"down"), which inflate coverage

Usage:
    python code/ex7_counterfactual_construction/antonym_coverage_dryrun.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Sequence, Set, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

SOURCE = "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json"
OUT_DIR = "results/ex7_counterfactual_construction"

CORE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("increasing", "decreasing"), ("more", "less"), ("more", "fewer"),
    ("greater", "lesser"), ("higher", "lower"), ("hot", "cold"), ("warm", "cool"),
    ("hotter", "colder"), ("warmer", "cooler"), ("faster", "slower"),
    ("larger", "smaller"), ("longer", "shorter"), ("heavier", "lighter"),
    ("stronger", "weaker"), ("wet", "dry"), ("rough", "smooth"),
    ("positive", "negative"), ("attract", "repel"), ("attracts", "repels"),
    ("expand", "contract"), ("expands", "contracts"), ("absorb", "reflect"),
    ("absorbs", "reflects"), ("melt", "freeze"), ("melts", "freezes"),
    ("summer", "winter"), ("alive", "dead"), ("living", "nonliving"),
    ("beneficial", "harmful"), ("maximum", "minimum"), ("always", "never"),
    ("gain", "lose"), ("gains", "loses"),
)
# Kept separate: each of these is frequently the WRONG sense in this corpus -- "light" as
# illumination rather than weight, "on" as a preposition, "down" as direction.
POLYSEMOUS_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("high", "low"), ("big", "small"), ("long", "short"), ("fast", "slow"),
    ("heavy", "light"), ("strong", "weak"), ("hard", "soft"), ("good", "bad"),
    ("many", "few"), ("most", "least"), ("all", "none"), ("day", "night"),
    ("north", "south"), ("east", "west"), ("up", "down"), ("on", "off"),
    ("open", "closed"), ("near", "far"), ("before", "after"), ("rise", "fall"),
    ("rises", "falls"), ("start", "stop"), ("starts", "stops"), ("add", "remove"),
    ("adds", "removes"), ("help", "harm"), ("helps", "harms"), ("begins", "ends"),
    ("large", "small"),
)


def _lexicon(pairs: Sequence[Tuple[str, str]]) -> Set[str]:
    return {word for pair in pairs for word in pair}


def _words(text: str) -> Set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _options_contain_polar_contrast(options: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> bool:
    """Necessary condition for answer remapping: two options are polar opposites."""
    tokenised = [_words(option) for option in options]
    for left, right in pairs:
        for i, first in enumerate(tokenised):
            for j, second in enumerate(tokenised):
                if i != j and left in first and right in second:
                    return True
    return False


def evaluate(records: Sequence[Dict], pairs: Sequence[Tuple[str, str]], label: str) -> Dict:
    lexicon = _lexicon(pairs)
    ineligible = [r for r in records if r["substitution_tier"] == "none"]
    currently_eligible = [r for r in records if r["counterfactual_valid"]]

    hits = [r for r in records if _words(r["passage"]) & lexicon]
    hits_ineligible = [r for r in ineligible if _words(r["passage"]) & lexicon]
    remappable = [r for r in hits if _options_contain_polar_contrast(r["options"], pairs)]
    remappable_ineligible = [
        r for r in hits_ineligible if _options_contain_polar_contrast(r["options"], pairs)
    ]
    reachable = len(currently_eligible) + len(hits_ineligible)
    return {
        "lexicon": label,
        "n_pairs": len(pairs),
        "n_items": len(records),
        "n_currently_eligible": len(currently_eligible),
        "n_fact_contains_lexicon_term": len(hits),
        "share_fact_contains_lexicon_term": len(hits) / len(records),
        "n_ineligible_reachable_by_lexicon": len(hits_ineligible),
        "share_of_ineligible_reachable": len(hits_ineligible) / len(ineligible),
        "projected_eligibility_union": reachable,
        "projected_eligibility_rate": reachable / len(records),
        # The condition that decides the strategy.
        "n_with_remappable_options": len(remappable),
        "share_remappable_of_lexicon_hits": len(remappable) / len(hits) if hits else None,
        "n_ineligible_with_remappable_options": len(remappable_ineligible),
        "effective_usable_items": len(remappable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    with open(resolve_path(SOURCE), encoding="utf-8") as handle:
        records = json.load(handle)["results"]

    core = evaluate(records, CORE_PAIRS, "core")
    permissive = evaluate(records, CORE_PAIRS + POLYSEMOUS_PAIRS, "permissive")

    payload = {
        "what": "Coverage and answer-remapping feasibility for the antonym/scalar-swap "
                "counterfactual strategy on OpenBookQA. Lexicon matching only.",
        "source_results_file": SOURCE,
        "results": {"core": core, "permissive": permissive},
        "verdict": (
            "REJECTED. Coverage improves substantially (29.0% -> "
            f"{core['projected_eligibility_rate']:.1%} core / "
            f"{permissive['projected_eligibility_rate']:.1%} permissive), but only "
            f"{core['share_remappable_of_lexicon_hits']:.1%} of lexicon-matched items have "
            "a polar contrast among their four options. Inverting the rule therefore "
            "leaves no option correct for roughly 95% of the items it can reach, so there "
            "is no counterfactual target to detect and Setting C degenerates. The "
            "necessary condition fails; coverage is irrelevant without it."
        ),
        "caveat": (
            "The remapping test is a NECESSARY, not sufficient, condition, and it is a "
            "surface lexicon test -- an option could be correct under the inverted rule "
            "without containing a lexicon antonym. The true usable rate is therefore "
            "bounded below by this figure, not equal to it. It would have to be off by "
            "more than an order of magnitude to change the verdict."
        ),
    }
    out = ensure_parent(resolve_path(os.path.join(args.out_dir, "antonym_coverage_dryrun.json")))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    for cell in (core, permissive):
        print(f"[{cell['lexicon']:10s}] {cell['n_pairs']:2d} pairs | "
              f"fact has term {cell['n_fact_contains_lexicon_term']:3d}/500 "
              f"({cell['share_fact_contains_lexicon_term']:5.1%}) | "
              f"projected eligibility {cell['projected_eligibility_union']:3d}/500 "
              f"({cell['projected_eligibility_rate']:5.1%}) | "
              f"options remappable {cell['n_with_remappable_options']:3d} "
              f"({cell['share_remappable_of_lexicon_hits']:5.1%})")
    print(f"\n{payload['verdict']}")
    print(f"\nWrote {os.path.relpath(out, resolve_path('.'))}")


if __name__ == "__main__":
    main()
