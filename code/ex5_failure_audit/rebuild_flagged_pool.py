"""Rebuild the RQ1 flagged-question pool from the committed pipelines.

Why this exists
---------------
`results/ex5_failure_analysis/rq1_flagged_questions.json` was produced by a script that is not
in version control (see `docs/ex5_failure_audit/provenance_rq1_flagged_questions.md`). Everything in that artefact
except the hand-assigned `failure_category` turned out to be exactly recomputable from files that
*are* committed, so this module restores the missing generator by reimplementation and proves it
by byte-for-byte comparison against the surviving artefact (`--verify`).

It also does what the original could not: apply criterion **C3_prior_dependent** to OpenBookQA,
which had no Setting-C run when the pool was first built but does now.

Inputs (all read-only)
----------------------
    datasets/<split>.json                                   question text, options, gold
    results/ex1_reproduce_KDA_pipeline/...results_kda_small...   Settings A and B, 4 students
    results/ex2_counterfactual/results_counterfactual_*     Setting C ensemble class (optional)

Detection criteria, as specified in docs/ex5_failure_audit/rq1_test_split_failure_analysis.md section 1
-------------------------------------------------------------------------------------
    C1_kda_cont            KDA_cont >= 0.70            AND primary model correct in Setting A
    C1_kda_disc            KDA_disc_proxy == 1.0       AND primary model correct in Setting A
    C2_both_correct_conf   both_correct bucket         AND P(R^q=1) >= 0.70 on the primary model
    C3_prior_dependent     ensemble Setting-C class == "prior_dependent"

A question joins the pool if it meets ANY criterion. `KDA_disc_proxy` is the model-ensemble
analogue of discrete KDA (no human student annotations exist in this repository):

    KDA_disc_proxy(q) = |{m : wrong without fact AND correct with fact}| / |{m : wrong without fact}|

Usage:
    uv run --active python code/ex5_failure_audit/rebuild_flagged_pool.py --dataset sciq --verify
    uv run --active python code/ex5_failure_audit/rebuild_flagged_pool.py --dataset obqa --verify
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Set

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent, resolve

PRIMARY_MODEL = "Riiid/kda-mpnet-base-race"
KDA_CONT_THRESHOLD = 0.70
P_RQ_THRESHOLD = 0.70

DATASET_SPECS = {
    "sciq": {
        "data": "datasets/sciq/sciq_test_full.json",
        "kda": "results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json",
        "counterfactual": "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
        "index_field": "sciq_index",
    },
    "obqa": {
        "data": "datasets/openbookqa/obqa_test_full.json",
        "kda": "results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json",
        "counterfactual": "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json",
        "index_field": "obqa_index",
    },
}

STOPWORDS = frozenset(
    """a an the of to in on at for from by with as is are was were be been being do does did
    what which who when where why how that this these those it its they them their and or but
    if then than so such can could will would shall should may might must not no all some any
    each most more less other another same different called known used using you your we our
    i me my he she his her""".split()
)

# Previously `abs_in_distractor` and `numeric_odd` were excluded from the pass/fail decision as
# unrecoverable. Their rules were subsequently recovered (both are gold-centric); the set is kept
# empty so the exemption machinery stays visible but exempts nothing.
RARE_CUES: frozenset = frozenset()

_WORD_RE = re.compile(r"[a-z0-9]+")
# Absolute/extremity vocabulary for `abs_in_distractor`. Recovered empirically, including the
# exclusion of "only" -- see docs/ex5_failure_audit/provenance_rq1_flagged_questions.md section 4.3.
_ABSOLUTE_RE = re.compile(
    r"\b(always|never|all|none|every|exactly|nothing|must|entirely|completely)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def content_tokens(text: str) -> Set[str]:
    return {w for w in _WORD_RE.findall(str(text).lower()) if w not in STOPWORDS and len(w) > 1}


def gold_content_words_present(gold: str, passage: str) -> bool:
    """`gold_in_passage`: every content word of the gold answer occurs in the passage.

    Not a substring test and not the counterfactual locator's contiguous n-gram test. The
    matching is *directional*: a singular gold token matches a plural in the passage, but not the
    reverse. That asymmetry is not a design choice of ours -- it is what the surviving artefact
    does, recovered from two items that pin it down in opposite directions:

        SciQ #669  gold "safety precaution"      passage "...precautions..."  -> cue FIRES
        SciQ #146  gold "skeletal muscle fibers" passage "...muscle fiber..."  -> cue does NOT fire

    See docs/ex5_failure_audit/provenance_rq1_flagged_questions.md section 4.2.
    """
    gold_tokens = content_tokens(gold)
    if not gold_tokens:
        return False
    passage_tokens = content_tokens(passage)
    return all(
        word in passage_tokens
        or f"{word}s" in passage_tokens
        or f"{word}es" in passage_tokens
        for word in gold_tokens
    )


def unique_argmax(values: Sequence, pick=max) -> Optional[int]:
    """Index of the strict extremum, or None when it is tied."""
    best = pick(values)
    winners = [i for i, v in enumerate(values) if v == best]
    return winners[0] if len(winners) == 1 else None


# --------------------------------------------------------------------------------------
# Metrics recomputed from the raw Settings A/B probabilities
# --------------------------------------------------------------------------------------
def kda_cont(record: Dict, models: Sequence[str]) -> Optional[float]:
    """KDA_cont(q) = sum_m (1 - P_m^A) * P_m^B / sum_m (1 - P_m^A)."""
    idx = record["answer_idx"]
    pa = [record["per_model"][m]["probabilities_without_fact"][idx] for m in models]
    pb = [record["per_model"][m]["probabilities_with_fact"][idx] for m in models]
    denominator = sum(1.0 - p for p in pa)
    if denominator <= 0:
        return None
    return sum((1.0 - a) * b for a, b in zip(pa, pb)) / denominator


def kda_disc_proxy(record: Dict, models: Sequence[str]) -> Optional[float]:
    """Model-ensemble stand-in for discrete KDA; None when no model failed Setting A."""
    idx = record["answer_idx"]
    failed = [m for m in models if record["per_model"][m]["predicted_idx_without_fact"] != idx]
    if not failed:
        return None
    recovered = sum(
        1 for m in failed if record["per_model"][m]["predicted_idx_with_fact"] == idx
    )
    return recovered / len(failed)


def ensemble_mean(record: Dict, models: Sequence[str], key: str) -> float:
    idx = record["answer_idx"]
    return sum(record["per_model"][m][key][idx] for m in models) / len(models)


# --------------------------------------------------------------------------------------
# Detection criteria and structural cues
# --------------------------------------------------------------------------------------
def detection_flags(
    record: Dict, models: Sequence[str], cf_class: Optional[str]
) -> List[str]:
    """The Step-1 criteria. A question joins the pool if any of these fire."""
    idx = record["answer_idx"]
    primary = record["per_model"][PRIMARY_MODEL]
    primary_a_correct = primary["predicted_idx_without_fact"] == idx
    primary_b_correct = primary["predicted_idx_with_fact"] == idx
    primary_p_a = primary["probabilities_without_fact"][idx]

    flags: List[str] = []
    score = kda_cont(record, models)
    if score is not None and score >= KDA_CONT_THRESHOLD and primary_a_correct:
        flags.append("C1_kda_cont")

    proxy = kda_disc_proxy(record, models)
    if proxy is not None and proxy == 1.0 and primary_a_correct:
        flags.append("C1_kda_disc")

    if primary_a_correct and primary_b_correct and primary_p_a >= P_RQ_THRESHOLD:
        flags.append("C2_both_correct_conf")

    if cf_class == "prior_dependent":
        flags.append("C3_prior_dependent")

    return flags


def structural_cues(question: str, options: Sequence[str], gold_idx: int, passage: str) -> List[str]:
    """Automatic surface-artifact detectors. Independent of the manual failure_category."""
    cues: List[str] = []
    lengths = [len(str(o).strip()) for o in options]

    if unique_argmax(lengths, max) == gold_idx:
        cues.append("longest_gold")
    if unique_argmax(lengths, min) == gold_idx:
        cues.append("shortest_gold")

    if gold_content_words_present(str(options[gold_idx]), passage or ""):
        cues.append("gold_in_passage")

    stem = content_tokens(question)
    overlaps = [len(stem & content_tokens(o)) for o in options]
    if max(overlaps) > 0 and unique_argmax(overlaps, max) == gold_idx:
        cues.append("stem_overlap_gold")

    normalised = [str(o).strip().lower() for o in options]
    if len(set(normalised)) < len(normalised):
        cues.append("dup_distractors")

    # Gold-centric, like every other cue in this set: it fires only when the *gold* is the
    # unique option carrying a digit, not merely when some option is.
    numeric = [bool(re.search(r"\d", str(o))) for o in options]
    if sum(numeric) == 1 and numeric[gold_idx]:
        cues.append("numeric_odd")

    # Discriminative, not merely present: an absolute word in a distractor is only a cue if the
    # gold does *not* also carry one. Where all four options share it (SciQ #522, "Every N
    # hours") it distinguishes nothing and the cue does not fire.
    if not _ABSOLUTE_RE.search(str(options[gold_idx])) and any(
        _ABSOLUTE_RE.search(str(o)) for i, o in enumerate(options) if i != gold_idx
    ):
        cues.append("abs_in_distractor")

    return sorted(cues)


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def build_pool(dataset_key: str, use_c3: bool = True) -> Dict:
    spec = DATASET_SPECS[dataset_key]

    with open(resolve(spec["kda"]), encoding="utf-8") as handle:
        kda_payload = json.load(handle)
    kda_records = kda_payload["results"]
    models = list(kda_payload.get("summary", {}).get("models") or kda_records[0]["per_model"].keys())

    cf_classes: Dict[int, Optional[str]] = {}
    cf_path = resolve(spec["counterfactual"])
    cf_available = use_c3 and os.path.isfile(cf_path)
    if cf_available:
        with open(cf_path, encoding="utf-8") as handle:
            for record in json.load(handle)["results"]:
                cf_classes[record["id"]] = record["ensemble"].get("counterfactual_class")

    questions: List[Dict] = []
    for record in kda_records:
        cf_class = cf_classes.get(record["id"])
        flags = detection_flags(record, models, cf_class)
        if not flags:
            continue
        idx = record["answer_idx"]
        primary = record["per_model"][PRIMARY_MODEL]
        questions.append(
            {
                "question_id": record["id"],
                "question": record["question"],
                "options": record["options"],
                "gold_answer": record["correct_answer"],
                "gold_index": idx,
                "reference_passage": record["passage"],
                "kda_cont": kda_cont(record, models),
                "kda_disc_proxy": kda_disc_proxy(record, models),
                "p_rq_without_fact_ensemble": ensemble_mean(record, models, "probabilities_without_fact"),
                "p_rqf_with_fact_ensemble": ensemble_mean(record, models, "probabilities_with_fact"),
                "p_rq_without_fact_primary": primary["probabilities_without_fact"][idx],
                "p_rqf_with_fact_primary": primary["probabilities_with_fact"][idx],
                "detection_flags": flags,
                "structural_cues": structural_cues(
                    record["question"], record["options"], idx, record["passage"]
                ),
                "counterfactual_class": cf_class,
            }
        )

    return {
        "dataset": dataset_key,
        "split_size": len(kda_records),
        "n_flagged": len(questions),
        "models": models,
        "primary_model": PRIMARY_MODEL,
        "counterfactual_available": cf_available,
        "criteria": {
            "C1_kda_cont": f"KDA_cont >= {KDA_CONT_THRESHOLD} and primary correct in Setting A",
            "C1_kda_disc": "KDA_disc_proxy == 1.0 and primary correct in Setting A",
            "C2_both_correct_conf": f"both_correct and P(R^q=1) >= {P_RQ_THRESHOLD} on the primary model",
            "C3_prior_dependent": (
                "ensemble Setting-C class == prior_dependent"
                if cf_available else "NOT APPLIED (no counterfactual results found)"
            ),
        },
        "questions": questions,
    }


# --------------------------------------------------------------------------------------
# Verification against the surviving artefact
# --------------------------------------------------------------------------------------
def verify(pool: Dict, artefact_path: str, dataset_key: str) -> int:
    """Compare a rebuilt pool field-by-field against the original artefact."""
    with open(resolve(artefact_path), encoding="utf-8") as handle:
        artefact = json.load(handle)["datasets"][dataset_key]

    original = {q["question_id"]: q for q in artefact["questions"]}
    rebuilt = {q["question_id"]: q for q in pool["questions"]}

    print(f"  artefact flagged : {len(original)}")
    print(f"  rebuilt  flagged : {len(rebuilt)}")
    only_a = sorted(set(original) - set(rebuilt))
    only_b = sorted(set(rebuilt) - set(original))
    print(f"  only in artefact : {len(only_a)} {only_a[:10]}")
    print(f"  only in rebuild  : {len(only_b)} {only_b[:10]}")

    shared = sorted(set(original) & set(rebuilt))
    numeric_fields = (
        "kda_cont", "p_rq_without_fact_ensemble", "p_rqf_with_fact_ensemble",
        "p_rq_without_fact_primary", "p_rqf_with_fact_primary",
    )
    mismatches = 0
    for field in numeric_fields:
        bad = [
            q for q in shared
            if original[q].get(field) is not None
            and abs(original[q][field] - (rebuilt[q][field] or 0.0)) > 5e-3
        ]
        mismatches += len(bad)
        print(f"  {field:32} {len(shared) - len(bad):4}/{len(shared)} match")

    bad = [q for q in shared if sorted(original[q]["detection_flags"]) != sorted(rebuilt[q]["detection_flags"])]
    mismatches += len(bad)
    print(f"  {'detection_flags':32} {len(shared) - len(bad):4}/{len(shared)} match" +
          (f"   differing ids: {bad[:6]}" if bad else ""))

    # `abs_in_distractor` and `numeric_odd` are excluded from the pass/fail decision: the
    # artefact applies them inconsistently (it fires `abs` on "every day" in OBQA #210 but not on
    # "Every 24 hours" in SciQ #522), so their exact definitions are unrecoverable. Neither cue
    # appears in any published table. See docs/ex5_failure_audit/provenance_rq1_flagged_questions.md section 4.2.
    core_bad, rare_bad = [], []
    for q in shared:
        a = set(original[q].get("structural_cues") or [])
        b = set(rebuilt[q].get("structural_cues") or [])
        if a == b:
            continue
        (rare_bad if (a ^ b) <= RARE_CUES else core_bad).append(q)
    mismatches += len(core_bad)
    print(f"  {'structural_cues (published cues)':32} {len(shared) - len(core_bad):4}/{len(shared)} match" +
          (f"   differing ids: {core_bad[:6]}" if core_bad else ""))
    print(f"  {'structural_cues (rare cues only)':32} {len(rare_bad):4} known residual(s)" +
          (f": {rare_bad[:6]}" if rare_bad else ""))

    return 0 if (not only_a and not only_b and mismatches == 0) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild and verify the RQ1 flagged-question pool.")
    parser.add_argument("--dataset", default="sciq", choices=sorted(DATASET_SPECS))
    parser.add_argument("--out", default="", help="Write the rebuilt pool here (optional).")
    parser.add_argument(
        "--verify", action="store_true",
        help="Compare against results/ex5_failure_analysis/rq1_flagged_questions.json.",
    )
    parser.add_argument(
        "--artefact", default="results/ex5_failure_analysis/rq1_flagged_questions.json"
    )
    parser.add_argument(
        "--extra-labels", default="results/ex5_failure_analysis/obqa_new_admission_labels.json",
        help="Second-annotator labels for items the original artefact never saw.",
    )
    parser.add_argument(
        "--carry-labels", action="store_true",
        help="Copy the manual failure_category across from the artefact by question_id.",
    )
    parser.add_argument(
        "--no-c3", action="store_true",
        help="Skip criterion C3 (reproduces the original OBQA pool, built before Setting C existed).",
    )
    args = parser.parse_args(argv)

    pool = build_pool(args.dataset, use_c3=not args.no_c3)

    if args.carry_labels:
        # The manual `failure_category` cannot be recomputed (single-annotator judgement), so it
        # is carried across by question_id. Items newly admitted by C3 have no label and are
        # marked so they can be found and annotated.
        with open(resolve(args.artefact), encoding="utf-8") as handle:
            labels = {
                q["question_id"]: q.get("failure_category")
                for q in json.load(handle)["datasets"][args.dataset]["questions"]
            }
        extra: Dict[int, Dict] = {}
        if args.extra_labels and os.path.isfile(resolve(args.extra_labels)):
            with open(resolve(args.extra_labels), encoding="utf-8") as handle:
                blob = json.load(handle)
            extra = {int(k): v for k, v in blob.get("labels", {}).items()}
            pool["extra_labels_source"] = args.extra_labels
            pool["extra_labels_provenance"] = blob.get("_provenance")

        n_new = 0
        for question in pool["questions"]:
            qid = question["question_id"]
            label = labels.get(qid)
            if label:
                question["failure_category"] = label
                question["failure_category_status"] = "carried_from_original_artefact"
            elif qid in extra:
                question["failure_category"] = extra[qid]["failure_category"]
                question["failure_category_rationale"] = extra[qid].get("rationale")
                question["failure_category_status"] = "second_annotator_new_admission"
            else:
                question["failure_category"] = None
                question["failure_category_status"] = "UNLABELLED_new_admission"
                n_new += 1
        pool["n_unlabelled"] = n_new
        pool["n_second_annotator"] = sum(
            1 for q in pool["questions"]
            if q["failure_category_status"] == "second_annotator_new_admission"
        )
        print(f"  labels carried   : {sum(1 for q in pool['questions'] if q['failure_category_status'] == 'carried_from_original_artefact')}"
              f"; second-annotator: {pool['n_second_annotator']}; still unlabelled: {n_new}")
    print(f"=== {args.dataset.upper()} ===")
    print(f"  split size       : {pool['split_size']}")
    print(f"  flagged          : {pool['n_flagged']} ({100.0 * pool['n_flagged'] / pool['split_size']:.1f}%)")
    print(f"  C3 applied       : {pool['counterfactual_available']}")

    status = 0
    if args.verify:
        print("  --- verification against the surviving artefact ---")
        status = verify(pool, args.artefact, args.dataset)
        print("  RESULT: " + ("EXACT MATCH" if status == 0 else "DIFFERENCES FOUND (see above)"))

    if args.out:
        out_path = ensure_parent(resolve(args.out))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(pool, handle, ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
