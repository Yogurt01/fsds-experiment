"""Audit the Setting-C counterfactual rewrite as a measurement instrument. Read-only.

No model inference and no LLM calls. Every input is a committed ex2 Setting-C run, its
dry-run preview (which retains the two construction flags the results files drop), or an
ex6 score file. Nothing under `code/ex2_counterfactual/` or `results/ex2_counterfactual/`
is modified; outputs go to `results/ex7_counterfactual_construction/`.

Four blocks:

  A  Per-tier fidelity -- coverage of the gold answer by the matched span, token mismatch
     between matched span and inserted target, residual mentions, and whether the
     unmatched remainder of a multi-word gold survives in the perturbed passage.

  B  Surface-cue leakage. A model-free probe picks the option with the highest
     content-word overlap with the passage. The quantity of interest is the ASYMMETRY:
     does Setting C make the counterfactual target more surface-guessable than Setting B
     made the gold? A clean rewrite moves this by ~0; splicing a whole distractor option
     into a one-clause fact moves it a long way.

  C  Does fidelity explain the ex6 anomalies? Setting-C outcome and Tier-1 AUC, split by
     substitution tier, plus a passage-length control inside SciQ's clean `exact` tier.

  D  Leakage cross-check and discount artifact. Block B's probe is model-free, so before
     it can be used as a validation criterion it has to be shown to track real solver
     behaviour. This block tests the probe against the 4-encoder ensemble's actual
     Setting-C labels, then writes per-item leakage flags and re-runs the ex6 Tier-1 AUCs
     with high-leakage items excluded -- so the share of the published OBQA numbers that
     rests on a surface artifact can be read off directly.

Usage:
    python code/ex7_counterfactual_construction/audit_construction_fidelity.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex6_psfd_score.psfd import auc
from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

# tag -> (ex2 results, ex2 dry-run preview, ex6 score file)
RUNS = {
    "sciq_test_full": (
        "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
        "results/ex2_counterfactual/counterfactual_passages_preview_sciq.json",
        "results/ex6_psfd_score/psfd_scores_sciq_test_full.json",
    ),
    "obqa_test_full": (
        "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json",
        "results/ex2_counterfactual/counterfactual_passages_preview_obqa.json",
        "results/ex6_psfd_score/psfd_scores_obqa_test_full.json",
    ),
}
OUT_DIR = "results/ex7_counterfactual_construction"

# Deliberately small: these are the words whose presence in an option carries no topical
# information, so including them would let a long option win the overlap probe on filler.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "to", "is", "are", "be", "and", "or", "it", "its",
    "that", "this", "for", "with", "as", "by", "at", "from", "will", "can",
}
TIERS = ("exact", "glued", "morphological", "partial", "none")
LENGTH_BUCKETS = (("<=15", 0, 15), ("16-30", 16, 30), ("31-60", 31, 60),
                  ("61-120", 61, 120), (">120", 121, 10 ** 9))
# Criterion 5's bar: SciQ `exact`, the split that works, sits at -3.3 pp.
LEAKAGE_DELTA_BAR = 0.05


def _load(path: str) -> Dict:
    with open(resolve_path(path), encoding="utf-8") as handle:
        return json.load(handle)


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def _content(text: str) -> Set[str]:
    return {w for w in _words(text) if w not in STOPWORDS}


# ---------------------------------------------------------------------------------------
# Block B primitive
# ---------------------------------------------------------------------------------------


def overlap_pick(passage: str, options: Sequence[str]) -> Tuple[List[int], List[float]]:
    """Options with maximal content-word overlap with the passage, and every score.

    Deliberately model-free: this is the cue a solver could exploit without reading the
    passage for meaning. Score is the share of the OPTION's content words present in the
    passage, so a long option is not rewarded for length.
    """
    passage_words = _content(passage)
    scores = []
    for option in options:
        option_words = _content(option)
        scores.append(len(option_words & passage_words) / len(option_words) if option_words else 0.0)
    best = max(scores)
    return [i for i, s in enumerate(scores) if s == best], scores


def _tie_credit(picked: Sequence[int], target: int) -> float:
    """1/k credit on a k-way tie, so ties are neither free wins nor free losses."""
    return (1.0 / len(picked)) if target in picked else 0.0


# ---------------------------------------------------------------------------------------
# Block A
# ---------------------------------------------------------------------------------------


def block_a_fidelity(records: Sequence[Dict], preview: Dict[int, Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for tier in TIERS:
        rows = [r for r in records if r["substitution_tier"] == tier]
        if not rows:
            continue
        matched_rows = [r for r in rows if r.get("substitution_matched_text")]
        coverage = []
        for r in matched_rows:
            gold = _content(r["correct_answer"])
            matched = _content(r["substitution_matched_text"])
            if gold:
                coverage.append(len(gold & matched) / len(gold))
        remainder_survives = 0
        for r in matched_rows:
            gold_words = _words(r["correct_answer"])
            matched_words = set(_words(r["substitution_matched_text"]))
            remainder = [w for w in gold_words if w not in matched_words]
            perturbed = set(_words(r["counterfactual_passage"]))
            if remainder and all(w in perturbed for w in remainder):
                remainder_survives += 1
        out[tier] = {
            "n": len(rows),
            "n_eligible": sum(1 for r in rows if r["counterfactual_valid"]),
            "coverage_eq_1": sum(1 for c in coverage if c >= 0.999),
            "coverage_lt_0.5": sum(1 for c in coverage if c < 0.5),
            "n_with_coverage": len(coverage),
            "token_mismatch_gt_1": sum(
                1 for r in matched_rows
                if abs(len((r["counterfactual_target"] or "").split())
                       - len(r["substitution_matched_text"].split())) > 1
            ),
            "multiword_gold": sum(1 for r in rows if len(r["correct_answer"].split()) > 1),
            "residual_gold_phrase": sum(1 for r in rows if r["residual_answer_mentions"]),
            "residual_glued": sum(1 for r in rows if r["residual_glued_mentions"]),
            "over_substitution_gt_2": sum(1 for r in rows if r["n_substitutions"] > 2),
            "unmatched_gold_remainder_survives": remainder_survives,
            "distractor_already_in_passage": sum(
                1 for r in rows if preview.get(r["id"], {}).get("distractor_already_in_passage")
            ),
            "distractor_overlaps_answer": sum(
                1 for r in rows if preview.get(r["id"], {}).get("distractor_overlaps_answer")
            ),
            "note": (
                "coverage is word-set overlap and is INVALID for the morphological tier, "
                "where the match is an inflection of the gold by construction"
                if tier == "morphological" else None
            ),
        }
    return out


# ---------------------------------------------------------------------------------------
# Block B
# ---------------------------------------------------------------------------------------


def block_b_leakage(records: Sequence[Dict]) -> Tuple[Dict, Dict[int, Dict]]:
    per_item: Dict[int, Dict] = {}
    for r in records:
        if not r["counterfactual_valid"]:
            continue
        picked_b, scores_b = overlap_pick(r["passage"], r["options"])
        picked_c, scores_c = overlap_pick(r["counterfactual_passage"], r["options"])
        gold, target = r["answer_idx"], r["counterfactual_target_idx"]
        b_gold = _tie_credit(picked_b, gold)
        c_target = _tie_credit(picked_c, target)
        per_item[r["id"]] = {
            "id": r["id"],
            "substitution_tier": r["substitution_tier"],
            "probe_B_picks_gold": b_gold,
            "probe_C_picks_cf_target": c_target,
            "probe_C_picks_gold": _tie_credit(picked_c, gold),
            "probe_score_gold_B": scores_b[gold],
            "probe_score_target_C": scores_c[target],
            # The rewrite CREATED a surface cue: the gold was not surface-findable in B,
            # the target is surface-findable in C.
            "leakage_created": bool(b_gold < 1.0 and c_target == 1.0),
            "leakage_delta_item": c_target - b_gold,
        }
    by_tier: Dict[str, Dict] = {}
    for tier in ("ALL",) + TIERS:
        rows = [v for v in per_item.values()
                if tier == "ALL" or v["substitution_tier"] == tier]
        if len(rows) < 3:
            continue
        b = statistics.fmean([v["probe_B_picks_gold"] for v in rows])
        c = statistics.fmean([v["probe_C_picks_cf_target"] for v in rows])
        by_tier[tier] = {
            "n": len(rows),
            "B_picks_gold": b,
            "C_picks_cf_target": c,
            "C_picks_gold": statistics.fmean([v["probe_C_picks_gold"] for v in rows]),
            "leakage_delta": c - b,
            "passes_delta_bar": abs(c - b) <= LEAKAGE_DELTA_BAR,
            "n_leakage_created": sum(1 for v in rows if v["leakage_created"]),
        }
    return by_tier, per_item


# ---------------------------------------------------------------------------------------
# Block C
# ---------------------------------------------------------------------------------------


def _tier_outcomes(items: Sequence[Dict], solvers: Sequence[str]) -> Dict:
    pos_f = [i["ensemble"]["all"]["F_mean"] for i in items
             if i["ensemble_counterfactual_class"] == "context_dependent"]
    neg_f = [i["ensemble"]["all"]["F_mean"] for i in items
             if i["ensemble_counterfactual_class"] == "prior_dependent"]
    pos_d = [i["ensemble"]["all"]["D"]["mean"] for i in items
             if i["ensemble_counterfactual_class"] == "context_dependent"]
    neg_d = [i["ensemble"]["all"]["D"]["mean"] for i in items
             if i["ensemble_counterfactual_class"] == "prior_dependent"]
    solver_aucs = []
    for name in solvers:
        p = [i["per_solver"][name]["D"] for i in items
             if i["per_solver"][name]["counterfactual_class"] == "context_dependent"]
        n = [i["per_solver"][name]["D"] for i in items
             if i["per_solver"][name]["counterfactual_class"] == "prior_dependent"]
        value = auc(p, n)
        if value is not None:
            solver_aucs.append(value)
    n_pairs = len(items) * len(solvers)
    return {
        "n_items": len(items),
        "ensemble_unstable_other": sum(
            1 for i in items if i["ensemble_counterfactual_class"] == "unstable_other"),
        "ensemble_unstable_rate": (
            sum(1 for i in items if i["ensemble_counterfactual_class"] == "unstable_other")
            / len(items)) if items else None,
        "pair_unstable_rate": (
            sum(1 for i in items for n in solvers
                if i["per_solver"][n]["counterfactual_class"] == "unstable_other") / n_pairs
        ) if n_pairs else None,
        "F_auc": auc(pos_f, neg_f),
        "D_auc": auc(pos_d, neg_d),
        "n_context_dependent": len(pos_f),
        "n_prior_dependent": len(neg_f),
        "per_solver_D_auc_spread": (
            max(solver_aucs) - min(solver_aucs) if len(solver_aucs) > 1 else None),
    }


def block_c_outcomes(records: Dict[int, Dict], scores: Dict) -> Dict:
    solvers = scores["solvers"]
    by_tier = {
        tier: _tier_outcomes(
            [i for i in scores["items"] if records[i["id"]]["substitution_tier"] == tier],
            solvers,
        )
        for tier in TIERS
        if sum(1 for i in scores["items"]
               if records[i["id"]]["substitution_tier"] == tier) >= 3
    }
    by_length = {}
    exact = [i for i in scores["items"] if records[i["id"]]["substitution_tier"] == "exact"]
    for label, lo, hi in LENGTH_BUCKETS:
        bucket = [i for i in exact
                  if lo <= len(records[i["id"]]["passage"].split()) <= hi]
        if len(bucket) >= 5:
            by_length[label] = _tier_outcomes(bucket, solvers)
    return {
        "by_substitution_tier": by_tier,
        "by_passage_length_within_exact_tier": by_length,
        "why": "The length control isolates passage brevity from substitution quality: "
               "every item here had a clean whole-word `exact` swap.",
    }


# ---------------------------------------------------------------------------------------
# Block D
# ---------------------------------------------------------------------------------------


def block_d_crosscheck(
    records: Dict[int, Dict], scores: Dict, leakage: Dict[int, Dict], tier: str = "partial",
) -> Dict:
    """Does the model-free probe predict the real ensemble's Setting-C label?"""
    items = [i for i in scores["items"]
             if records[i["id"]]["substitution_tier"] == tier
             and i["ensemble_counterfactual_class"] in ("context_dependent", "prior_dependent")]
    if len(items) < 10:
        return {"unavailable": f"n={len(items)} at tier={tier}"}

    table = {"leak_ctx": 0, "leak_prior": 0, "noleak_ctx": 0, "noleak_prior": 0}
    deltas, labels = [], []
    for item in items:
        flag = leakage[item["id"]]["leakage_created"]
        ctx = item["ensemble_counterfactual_class"] == "context_dependent"
        key = ("leak_" if flag else "noleak_") + ("ctx" if ctx else "prior")
        table[key] += 1
        deltas.append(leakage[item["id"]]["leakage_delta_item"])
        labels.append(ctx)

    a, b, c, d = table["leak_ctx"], table["leak_prior"], table["noleak_ctx"], table["noleak_prior"]
    denominator = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
    phi = ((a * d - b * c) / denominator) if denominator else None
    return {
        "tier": tier,
        "n": len(items),
        "contingency": {
            "leakage_created & context_dependent": a,
            "leakage_created & prior_dependent": b,
            "no_leakage & context_dependent": c,
            "no_leakage & prior_dependent": d,
        },
        "p_context_dependent_given_leakage": a / (a + b) if (a + b) else None,
        "p_context_dependent_given_no_leakage": c / (c + d) if (c + d) else None,
        "phi_coefficient": phi,
        "auc_leakage_delta_predicts_context_dependent": auc(
            [x for x, l in zip(deltas, labels) if l],
            [x for x, l in zip(deltas, labels) if not l],
        ),
        "interpretation_rule": (
            "phi >= 0.3 or AUC >= 0.65 => the model-free probe tracks real solver "
            "susceptibility, and Phase-3 criterion 5 is a VALIDATED predictor. Below "
            "that it stays a plausible proxy only."
        ),
    }


def block_d_discount(scores: Dict, leakage: Dict[int, Dict]) -> Dict:
    """Re-run the ex6 Tier-1 AUCs with surface-leaking items excluded."""
    def tier1(items: Sequence[Dict]) -> Dict:
        out = {}
        for signal in ("F", "D", "D_prime"):
            if signal == "F":
                get = lambda i: i["ensemble"]["all"]["F_mean"]
            else:
                get = lambda i, s=signal: i["ensemble"]["all"][s]["mean"]
            pos = [get(i) for i in items if i["ensemble_counterfactual_class"] == "context_dependent"]
            neg = [get(i) for i in items if i["ensemble_counterfactual_class"] == "prior_dependent"]
            out[signal] = {"auc": auc(pos, neg), "n_positive": len(pos), "n_negative": len(neg)}
        kda_pos = [i["kda_cont"] for i in items
                   if i["ensemble_counterfactual_class"] == "context_dependent" and i["kda_cont"] is not None]
        kda_neg = [i["kda_cont"] for i in items
                   if i["ensemble_counterfactual_class"] == "prior_dependent" and i["kda_cont"] is not None]
        out["KDA_cont"] = {"auc": auc(kda_pos, kda_neg)}
        return out

    everything = scores["items"]
    clean = [i for i in everything if not leakage[i["id"]]["leakage_created"]]
    dropped = [i["id"] for i in everything if leakage[i["id"]]["leakage_created"]]
    return {
        "n_all": len(everything),
        "n_after_dropping_leaking_items": len(clean),
        "n_dropped": len(dropped),
        "dropped_ids": sorted(dropped),
        "tier1_as_published": tier1(everything),
        "tier1_excluding_leaking_items": tier1(clean),
        "how_to_read": (
            "The published ex6 OBQA Tier-1 numbers are the first block. The second is the "
            "same computation on items where the rewrite did NOT create a surface cue. "
            "The gap is the share of the published figure attributable to the artifact."
        ),
    }


# ---------------------------------------------------------------------------------------


def audit(tag: str) -> Tuple[Dict, Dict]:
    results_path, preview_path, scores_path = RUNS[tag]
    payload = _load(results_path)
    records = payload["results"]
    by_id = {r["id"]: r for r in records}
    preview = {s["id"]: s for s in _load(preview_path)["samples"]}
    scores = _load(scores_path)

    if len(preview) != len(records):
        raise SystemExit(f"[{tag}] preview/results length mismatch; refusing to write.")

    leak_summary, leak_items = block_b_leakage(records)
    report = {
        "tag": tag,
        "sources": {"ex2_results": results_path, "ex2_preview": preview_path,
                    "ex6_scores": scores_path},
        "n_samples": len(records),
        "n_eligible": sum(1 for r in records if r["counterfactual_valid"]),
        "block_a_fidelity_by_tier": block_a_fidelity(records, preview),
        "block_b_surface_leakage": leak_summary,
        "block_c_outcomes": block_c_outcomes(by_id, scores),
        "block_d_probe_crosscheck": block_d_crosscheck(by_id, scores, leak_items),
        "block_d_ex6_discount": block_d_discount(scores, leak_items),
    }
    artifact = {
        "what": "Per-item surface-cue leakage flags for a committed ex2 Setting-C run. "
                "Derived read-only; the ex2 run is unmodified.",
        "probe": "Model-free: pick the option with the highest share of its content words "
                 "present in the passage; 1/k credit on k-way ties.",
        "leakage_created": "gold was NOT surface-findable in Setting B, but the "
                           "counterfactual target IS in Setting C -- i.e. the rewrite "
                           "manufactured the cue.",
        "source_results_file": results_path,
        "summary": leak_summary,
        "items": [leak_items[k] for k in sorted(leak_items)],
    }
    return report, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    combined = {}
    for tag in RUNS:
        report, artifact = audit(tag)
        combined[tag] = report
        path = ensure_parent(resolve_path(
            os.path.join(args.out_dir, f"leakage_flags_{tag}.json")))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=1)

        print(f"\n=== {tag} (n={report['n_samples']}, eligible={report['n_eligible']}) ===")
        print("  Block B -- surface-cue leakage (probe picks, 1/k on ties):")
        for tier, cell in report["block_b_surface_leakage"].items():
            mark = "OK " if cell["passes_delta_bar"] else "FAIL"
            print(f"    {tier:14s} n={cell['n']:4d}  B->gold {cell['B_picks_gold']:6.1%}"
                  f"  C->target {cell['C_picks_cf_target']:6.1%}"
                  f"  delta {cell['leakage_delta']:+7.1%}  [{mark}]")
        cross = report["block_d_probe_crosscheck"]
        if "unavailable" not in cross:
            print(f"  Block D -- probe vs real ensemble label (tier={cross['tier']}, n={cross['n']}):")
            print(f"    P(context_dep | leakage)={cross['p_context_dependent_given_leakage']}"
                  f"  P(context_dep | no leakage)={cross['p_context_dependent_given_no_leakage']}")
            print(f"    phi={cross['phi_coefficient']}  "
                  f"AUC(delta -> context_dep)={cross['auc_leakage_delta_predicts_context_dependent']}")
        disc = report["block_d_ex6_discount"]
        pub = disc["tier1_as_published"]; cln = disc["tier1_excluding_leaking_items"]
        print(f"  Block D -- ex6 discount (dropping {disc['n_dropped']} leaking of {disc['n_all']}):")
        for signal in ("F", "D", "D_prime", "KDA_cont"):
            a, b = pub[signal]["auc"], cln[signal]["auc"]
            fa = "n/a" if a is None else f"{a:.4f}"
            fb = "n/a" if b is None else f"{b:.4f}"
            print(f"    {signal:9s} published {fa}  ->  leak-free {fb}")
        print(f"    -> wrote {os.path.relpath(path, resolve_path('.'))}")

    out = ensure_parent(resolve_path(os.path.join(args.out_dir, "construction_fidelity_audit.json")))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"what": "Audit of the ex2 counterfactual construction as a measurement "
                           "instrument. Read-only; no ex2/ex6 artifact modified.",
                   "leakage_delta_bar": LEAKAGE_DELTA_BAR,
                   "runs": combined}, handle, ensure_ascii=False, indent=1)
    print(f"\nWrote {os.path.relpath(out, resolve_path('.'))}")


if __name__ == "__main__":
    main()
