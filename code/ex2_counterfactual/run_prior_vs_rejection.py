"""E2: is `prior_dependent` at LLM scale "ignored the context" or "rejected a false one"?

The problem
-----------
E1 found that Qwen3-4B keeps the gold answer on 62.5% of its Setting-C target set
(`docs/ex2_counterfactual/e1_counterfactual_llm_scale.md`). The pipeline labels that `prior_dependent` -- the
model overrode the passage with its parametric prior. But the counterfactual passages are
deliberately FALSE ("Combustion is the separation of ions"), and ClashEval's central finding
is that models arbitrate prior against context by the context's PLAUSIBILITY. A 110M encoder
cannot assess that; a 4B instruct model can. So `prior_dependent` conflates two behaviours:

    (1) ignored the context          -- the failure the metric exists to catch
    (2) read it, judged it false,    -- competent reading, which the metric PENALISES
        and overrode it

E1's 62.5% is therefore an upper bound on genuine prior-dependence. This experiment
produces the lower bound, bracketing the true value.

Two independent probes
----------------------
A. INSTRUCTION ABLATION -- capability vs. choice.
   Setting C' re-runs the counterfactual passage with a system prompt that removes the
   model's discretion: treat the Fact as authoritative even where it contradicts belief.
   An item that was `prior_dependent` under C and follows the counterfactual under C' COULD
   read the context and chose not to -- behaviour (2). An item that stays on gold even when
   told to follow the passage is much closer to behaviour (1).

   Paired control: Setting B' runs the GOLD passage under the same system prompt. If the
   instruction were simply disruptive, B' accuracy would fall away from B's 0.9977. This
   control is what makes C' interpretable, and it is reported whether or not it is flattering.

B. PLAUSIBILITY GRADIENT -- intrinsic, no prompting at all.
   The model's own mean per-token NLL of the counterfactual passage minus that of the gold
   passage measures how much the perturbation damaged the text's plausibility UNDER THIS
   MODEL. If behaviour (2) dominates, `prior_dependent` items should carry a systematically
   larger plausibility penalty than `context_dependent` ones.

The probes are independent: A depends on instruction-following, B does not. A depends on
prompt wording, B does not. Agreement between them is therefore real evidence; disagreement
is informative about which confound is biting.

Pre-registered outcomes (share of E1's `prior_dependent` items that flip under C')
---------------------------------------------------------------------------------
    P1  >= 60% flip     mostly rejection of a false context. E1's 62.5% is a large
                        overestimate and the metric needs the context-priority instruction
                        built into its definition.
    P2  <= 20% flip     mostly genuine prior-dependence. E1's headline stands.
    P3  20-60% flip     mixed; the bracket is the result.

  and, for probe B, AUC of the plausibility penalty predicting `prior_dependent`:
    >= 0.60 supports the rejection hypothesis; ~0.50 rejects it.

The two probes can disagree, and that outcome is reported rather than resolved by fiat.

Correctness gates
-----------------
1. Setting C re-scored under the ORIGINAL system prompt must reproduce E1's predictions
   exactly on the checked subset. Everything here is a delta against E1, so if the harness
   does not reproduce E1 the deltas are meaningless.
2. The counterfactual passages must equal the committed dry-run preview (as in E1), so the
   perturbation is the same one E1 and the encoder runs saw.

Usage:
    python code/ex2_counterfactual/run_prior_vs_rejection.py --verify-c 50 --verify-only
    python code/ex2_counterfactual/run_prior_vs_rejection.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex1_reproduce_KDA.kda_qwen_eval import (
    DATASET_SPECS,
    SYSTEM_PROMPT,
    letter_token_ids,
    load_model,
    score_prompt,
)
from ex2_counterfactual.counterfactual_passage import build_counterfactual, tier_at_least
from ex2_counterfactual.run_counterfactual_qwen import (
    CF_PREVIEWS,
    build_prompt,
    cohen_kappa,
    verify_counterfactual_generation,
    wilson_interval,
)
from ex2_counterfactual.run_counterfactual_experiment import CF_CLASSES, TIERS, argmax
from utils.paths import MODELS_DIR, PROJECT_ROOT, ensure_parent
from utils.paths import resolve as resolve_path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "filelock",
    "huggingface_hub", "transformers", "datasets", "fsspec",
)

# E1's run, joined by id. Every result here is a delta against it.
E1_RESULTS = "results/ex2_counterfactual/results_counterfactual_qwen3_4b.json"

# The Setting B'/C' system prompt. It differs from kda_qwen_eval.SYSTEM_PROMPT by the two
# sentences that remove the model's discretion over whether to believe the Fact. The user
# turn is byte-identical to Settings B and C, so the system prompt is the ONLY difference.
CONTEXT_PRIORITY_SYSTEM_PROMPT = (
    "You are a careful student taking a multiple-choice exam. "
    "Answer using only the information in the Fact provided. "
    "Treat the Fact as true and authoritative, even where it contradicts what you believe. "
    "Answer with a single letter and nothing else."
)

logger = logging.getLogger("prior_vs_rejection")


def setup_logging(log_path: str, verbose: bool, append: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def render_chat_with_system(tokenizer, system_text: str, user_text: str) -> str:
    """`kda_qwen_eval.render_chat` with the system prompt exposed as a parameter."""
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def sequence_nll(model, tokenizer, text: str, device: str) -> Optional[Tuple[float, float, int]]:
    """Mean and total next-token NLL of `text` under the model, plus its token count.

    Probe B's plausibility measure. Scored on the bare passage with no prompt scaffolding,
    so it reflects the text itself rather than any instruction. Returns None for text too
    short to have a single prediction target.
    """
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].to(device)
    if ids.shape[-1] < 2:
        return None
    logits = model(input_ids=ids).logits[0, :-1, :].float()
    targets = ids[0, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_prob = log_probs[torch.arange(targets.shape[0], device=ids.device), targets]
    total = float(-token_log_prob.sum())
    n_tokens = int(targets.shape[0])
    return total / n_tokens, total, n_tokens


def verify_setting_c(
    model, tokenizer, letter_ids, device: str, samples: Sequence[Dict],
    e1_records: Dict[int, Dict], n_check: int,
) -> Dict:
    """Gate 1: Setting C under the original system prompt must reproduce E1 exactly."""
    checked, mismatches, max_delta = 0, [], 0.0
    for sample in samples:
        if checked >= n_check:
            break
        reference = e1_records.get(sample["id"])
        if reference is None or not sample["counterfactual_valid"]:
            continue
        probs_c, _ = score_prompt(
            model, tokenizer,
            render_chat_with_system(
                tokenizer, SYSTEM_PROMPT,
                build_prompt(sample, sample["counterfactual_passage"]),
            ),
            letter_ids, device,
        )
        if argmax(probs_c) != reference["predicted_idx_counterfactual_passage"]:
            mismatches.append(
                f"id={sample['id']}: {argmax(probs_c)} != "
                f"{reference['predicted_idx_counterfactual_passage']}"
            )
        max_delta = max(
            max_delta,
            abs(probs_c[sample["answer_idx"]] - reference["p_correct_counterfactual_passage"]),
        )
        checked += 1
    return {
        "reference": E1_RESULTS,
        "n_checked": checked,
        "n_prediction_mismatches": len(mismatches),
        "max_abs_probability_delta": max_delta,
        "mismatches": mismatches[:20],
        "passed": not mismatches,
    }


def score_dataset(
    model, tokenizer, letter_ids, device: str, samples: Sequence[Dict],
    e1_records: Dict[int, Dict], progress_every: int,
) -> List[Dict]:
    """Settings B' and C' plus both passage NLLs, joined to E1's B and C."""
    records: List[Dict] = []
    started = time.time()
    progress_every = progress_every or max(1, len(samples) // 20)

    for index, sample in enumerate(samples):
        answer_idx = sample["answer_idx"]
        cf_idx = sample["counterfactual_target_idx"]
        cf_valid = sample["counterfactual_valid"]
        e1 = e1_records.get(sample["id"])

        # Probe A: the same user turn as Settings B and C, context-priority system prompt.
        probs_b_prime, _ = score_prompt(
            model, tokenizer,
            render_chat_with_system(
                tokenizer, CONTEXT_PRIORITY_SYSTEM_PROMPT,
                build_prompt(sample, sample["passage"]),
            ),
            letter_ids, device,
        )
        probs_c_prime, _ = score_prompt(
            model, tokenizer,
            render_chat_with_system(
                tokenizer, CONTEXT_PRIORITY_SYSTEM_PROMPT,
                build_prompt(sample, sample["counterfactual_passage"]),
            ),
            letter_ids, device,
        )

        # Probe B: plausibility of each passage as bare text.
        nll_gold = sequence_nll(model, tokenizer, sample["passage"], device)
        nll_cf = sequence_nll(model, tokenizer, sample["counterfactual_passage"], device)

        pred_b_prime, pred_c_prime = argmax(probs_b_prime), argmax(probs_c_prime)
        class_c_prime = None
        if cf_valid:
            if pred_c_prime == cf_idx:
                class_c_prime = "context_dependent"
            elif pred_c_prime == answer_idx:
                class_c_prime = "prior_dependent"
            else:
                class_c_prime = "unstable_other"

        records.append({
            "id": sample["id"],
            "question": sample["question"],
            "options": sample["options"],
            "answer_idx": answer_idx,
            "counterfactual_target_idx": cf_idx,
            "counterfactual_valid": cf_valid,
            "substitution_tier": sample["substitution_tier"],
            # --- E1 join (Settings A/B/C, original system prompt) --------------------
            "e1_baseline_category": e1["baseline_category"] if e1 else None,
            "e1_counterfactual_class": e1["counterfactual_class"] if e1 else None,
            "e1_predicted_idx_counterfactual_passage": (
                e1["predicted_idx_counterfactual_passage"] if e1 else None
            ),
            "e1_is_correct_original_passage": (
                e1["is_correct_original_passage"] if e1 else None
            ),
            # --- probe A --------------------------------------------------------------
            "probabilities_original_passage_context_priority": probs_b_prime,
            "probabilities_counterfactual_passage_context_priority": probs_c_prime,
            "predicted_idx_original_passage_context_priority": pred_b_prime,
            "predicted_idx_counterfactual_passage_context_priority": pred_c_prime,
            "is_correct_original_passage_context_priority": pred_b_prime == answer_idx,
            "follows_counterfactual_context_priority": bool(cf_valid and pred_c_prime == cf_idx),
            "counterfactual_class_context_priority": class_c_prime,
            # --- probe B --------------------------------------------------------------
            "nll_mean_gold_passage": nll_gold[0] if nll_gold else None,
            "nll_total_gold_passage": nll_gold[1] if nll_gold else None,
            "n_tokens_gold_passage": nll_gold[2] if nll_gold else None,
            "nll_mean_counterfactual_passage": nll_cf[0] if nll_cf else None,
            "nll_total_counterfactual_passage": nll_cf[1] if nll_cf else None,
            "n_tokens_counterfactual_passage": nll_cf[2] if nll_cf else None,
            "plausibility_penalty": (
                nll_cf[0] - nll_gold[0] if (nll_cf and nll_gold) else None
            ),
        })

        if (index + 1) % progress_every == 0 or index + 1 == len(samples):
            elapsed = time.time() - started
            logger.info(
                "  %d/%d (%.1f%%) | %.2f s/question | elapsed %.1f s",
                index + 1, len(samples), 100.0 * (index + 1) / len(samples),
                elapsed / (index + 1), elapsed,
            )
    return records


def auc(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    """Rank-based AUC; 0.5 means the score carries no information about the two groups."""
    if not positive or not negative:
        return None
    merged = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank_sum, index = 0.0, 0
    while index < len(merged):
        end = index
        while end < len(merged) and merged[end][0] == merged[index][0]:
            end += 1
        average_rank = (index + end + 1) / 2.0
        for position in range(index, end):
            if merged[position][1] == 1:
                rank_sum += average_rank
        index = end
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def analyse(records: Sequence[Dict]) -> Dict:
    """Both probes, and the bracket they produce around genuine prior-dependence."""
    eligible = [r for r in records if r["counterfactual_valid"]]
    # E1's Setting-C target set: both_correct under the ORIGINAL prompt, and CF-eligible.
    target = [r for r in eligible if r["e1_baseline_category"] == "both_correct"]
    prior = [r for r in target if r["e1_counterfactual_class"] == "prior_dependent"]
    context = [r for r in target if r["e1_counterfactual_class"] == "context_dependent"]

    # --- probe A ---------------------------------------------------------------------
    flipped = [r for r in prior if r["counterfactual_class_context_priority"] == "context_dependent"]
    still_prior = [r for r in prior if r["counterfactual_class_context_priority"] == "prior_dependent"]
    became_unstable = [
        r for r in prior if r["counterfactual_class_context_priority"] == "unstable_other"
    ]
    context_held = [
        r for r in context if r["counterfactual_class_context_priority"] == "context_dependent"
    ]

    n_prior = len(prior)
    flip_share = len(flipped) / n_prior if n_prior else 0.0
    flip_ci = wilson_interval(len(flipped), n_prior)

    if flip_share >= 0.60:
        outcome = (
            f"P1 (>= 60% flip): {flip_share:.1%} of `prior_dependent` items follow the "
            "counterfactual once told to. Mostly rejection of a false context, not "
            "prior-dependence -- E1's 62.5% is a large overestimate."
        )
    elif flip_share <= 0.20:
        outcome = (
            f"P2 (<= 20% flip): only {flip_share:.1%} flip. Mostly genuine "
            "prior-dependence; E1's headline stands."
        )
    else:
        outcome = (
            f"P3 (20-60% flip): {flip_share:.1%} flip. Mixed -- the bracket is the result."
        )

    # --- the bracket -------------------------------------------------------------------
    n_target = len(target)
    upper = n_prior / n_target if n_target else 0.0            # E1: every prior_dependent
    lower = len(still_prior) / n_target if n_target else 0.0   # survives an explicit order

    # --- control -----------------------------------------------------------------------
    n_all = len(records)
    acc_b = sum(1 for r in records if r["e1_is_correct_original_passage"]) / n_all if n_all else 0.0
    acc_b_prime = (
        sum(1 for r in records if r["is_correct_original_passage_context_priority"]) / n_all
        if n_all else 0.0
    )

    # --- probe B -----------------------------------------------------------------------
    prior_penalty = [r["plausibility_penalty"] for r in prior if r["plausibility_penalty"] is not None]
    context_penalty = [
        r["plausibility_penalty"] for r in context if r["plausibility_penalty"] is not None
    ]
    penalty_auc = auc(prior_penalty, context_penalty)

    # The sharper form of probe B: WITHIN the prior_dependent set, does the plausibility
    # penalty predict which items refuse to flip? If rejection-of-an-implausible-context
    # drives the effect, the items that hold out against an explicit order should be the
    # ones whose perturbation is least believable. This test does not depend on the
    # prior/context split being meaningful, so it survives if probe B's across-class
    # comparison is confounded by the classes differing in some other way.
    stayed_penalty = [
        r["plausibility_penalty"] for r in still_prior if r["plausibility_penalty"] is not None
    ]
    flipped_penalty = [
        r["plausibility_penalty"] for r in flipped if r["plausibility_penalty"] is not None
    ]
    within_auc = auc(stayed_penalty, flipped_penalty)

    # Is the measure working at all? If the perturbation did not damage plausibility, a
    # null result above would be uninformative rather than evidence.
    all_penalty = [
        r["plausibility_penalty"] for r in target if r["plausibility_penalty"] is not None
    ]
    share_positive = (
        sum(1 for x in all_penalty if x > 0) / len(all_penalty) if all_penalty else 0.0
    )
    if penalty_auc is None:
        probe_b = "not computable"
    elif penalty_auc >= 0.60:
        probe_b = (
            f"SUPPORTS rejection (AUC {penalty_auc:.3f}): `prior_dependent` items carry a "
            "larger plausibility penalty."
        )
    elif penalty_auc <= 0.40:
        probe_b = (
            f"INVERTED (AUC {penalty_auc:.3f}): `prior_dependent` items carry a SMALLER "
            "plausibility penalty than `context_dependent` ones."
        )
    else:
        probe_b = (
            f"REJECTS rejection (AUC {penalty_auc:.3f} ~ chance): the plausibility penalty "
            "does not distinguish the two classes."
        )

    return {
        "n_samples": len(records),
        "n_counterfactual_eligible": len(eligible),
        "e1_target_set": n_target,
        "probe_a_instruction_ablation": {
            "system_prompt": CONTEXT_PRIORITY_SYSTEM_PROMPT,
            "n_prior_dependent_in_e1": n_prior,
            "flipped_to_context_dependent": {
                "count": len(flipped), "share": flip_share, "ci95": flip_ci,
            },
            "still_prior_dependent": {
                "count": len(still_prior),
                "share": len(still_prior) / n_prior if n_prior else 0.0,
            },
            "became_unstable": {
                "count": len(became_unstable),
                "share": len(became_unstable) / n_prior if n_prior else 0.0,
            },
            "context_dependent_held": {
                "count": len(context_held),
                "of": len(context),
                "share": len(context_held) / len(context) if context else 0.0,
            },
            "pre_registered_outcome": outcome,
        },
        "control_gold_passage_under_same_instruction": {
            "accuracy_setting_B_original_prompt": acc_b,
            "accuracy_setting_B_context_priority": acc_b_prime,
            "delta": acc_b_prime - acc_b,
            "note": (
                "If the instruction were merely disruptive rather than discretion-removing, "
                "gold-passage accuracy would fall. A near-zero delta is what makes the "
                "Setting C' result interpretable."
            ),
        },
        "probe_b_plausibility": {
            "measure": (
                "mean per-token NLL of the counterfactual passage minus that of the gold "
                "passage, scored as bare text with no prompt scaffolding"
            ),
            "measure_is_live": {
                "mean_penalty_all_target_items": mean(all_penalty),
                "share_with_positive_penalty": share_positive,
                "note": (
                    "A penalty above 0 means the perturbation made the passage less likely, "
                    "which is what it should do. If this were near 0 the null results below "
                    "would be uninformative rather than evidence."
                ),
            },
            "across_classes": {
                "mean_penalty_prior_dependent": mean(prior_penalty),
                "mean_penalty_context_dependent": mean(context_penalty),
                "auc_penalty_predicts_prior_dependent": penalty_auc,
            },
            "within_prior_dependent": {
                "mean_penalty_stayed_prior": mean(stayed_penalty),
                "mean_penalty_flipped": mean(flipped_penalty),
                "auc_penalty_predicts_staying_prior": within_auc,
                "note": (
                    "The decisive form. Under the rejection hypothesis the items that hold "
                    "out against an explicit order to follow the passage should be the ones "
                    "whose perturbation is least believable."
                ),
            },
            "verdict": probe_b,
        },
        "bracket_on_genuine_prior_dependence": {
            "upper_bound_e1": upper,
            "lower_bound_survives_explicit_instruction": lower,
            "n_target_set": n_target,
            "note": (
                "Upper: every item that keeps the gold answer under the ordinary prompt. "
                "Lower: those that keep it even when told the Fact is authoritative. The "
                "truth lies between; neither endpoint is the answer on its own."
            ),
        },
    }


def log_analysis(analysis: Dict) -> None:
    logger.info("")
    logger.info("=" * 100)
    logger.info("E2 -- PRIOR-DEPENDENCE vs REJECTION OF A FALSE CONTEXT")
    logger.info("=" * 100)
    probe_a = analysis["probe_a_instruction_ablation"]
    control = analysis["control_gold_passage_under_same_instruction"]
    probe_b = analysis["probe_b_plausibility"]
    bracket = analysis["bracket_on_genuine_prior_dependence"]

    logger.info("  E1 Setting-C target set: %d", analysis["e1_target_set"])
    logger.info("")
    logger.info("  CONTROL (gold passage, context-priority instruction):")
    logger.info(
        "    Acc_B %.4f -> Acc_B' %.4f (delta %+.4f)",
        control["accuracy_setting_B_original_prompt"],
        control["accuracy_setting_B_context_priority"], control["delta"],
    )
    logger.info("")
    logger.info("  PROBE A -- of %d E1 `prior_dependent` items, under C':",
                probe_a["n_prior_dependent_in_e1"])
    flipped = probe_a["flipped_to_context_dependent"]
    logger.info(
        "    flipped to context_dependent : %5d  %6.2f%%  95%% CI [%.2f%%, %.2f%%]",
        flipped["count"], flipped["share"] * 100,
        flipped["ci95"][0] * 100, flipped["ci95"][1] * 100,
    )
    logger.info(
        "    still prior_dependent        : %5d  %6.2f%%",
        probe_a["still_prior_dependent"]["count"],
        probe_a["still_prior_dependent"]["share"] * 100,
    )
    logger.info(
        "    became unstable_other        : %5d  %6.2f%%",
        probe_a["became_unstable"]["count"], probe_a["became_unstable"]["share"] * 100,
    )
    held = probe_a["context_dependent_held"]
    logger.info(
        "    (sanity) context_dependent items that stayed context_dependent: %d/%d = %.1f%%",
        held["count"], held["of"], held["share"] * 100,
    )
    logger.info("")
    logger.info("  PROBE A OUTCOME: %s", probe_a["pre_registered_outcome"])
    logger.info("")
    logger.info("  PROBE B -- plausibility penalty (mean per-token NLL, cf minus gold):")
    across, within = probe_b["across_classes"], probe_b["within_prior_dependent"]
    live = probe_b["measure_is_live"]
    logger.info(
        "    measure is live: mean penalty %+.4f over the target set, %.1f%% positive",
        live["mean_penalty_all_target_items"], live["share_with_positive_penalty"] * 100,
    )
    logger.info(
        "    across classes : prior %+.4f | context %+.4f | AUC %s",
        across["mean_penalty_prior_dependent"], across["mean_penalty_context_dependent"],
        "n/a" if across["auc_penalty_predicts_prior_dependent"] is None
        else f"{across['auc_penalty_predicts_prior_dependent']:.3f}",
    )
    logger.info(
        "    within prior   : stayed %+.4f | flipped %+.4f | AUC %s  <-- the decisive test",
        within["mean_penalty_stayed_prior"], within["mean_penalty_flipped"],
        "n/a" if within["auc_penalty_predicts_staying_prior"] is None
        else f"{within['auc_penalty_predicts_staying_prior']:.3f}",
    )
    logger.info("  PROBE B VERDICT: %s", probe_b["verdict"])
    logger.info("")
    logger.info(
        "  BRACKET on genuine prior-dependence: %.1f%% (lower, survives an explicit order) "
        "to %.1f%% (upper, E1)",
        bracket["lower_bound_survives_explicit_instruction"] * 100,
        bracket["upper_bound_e1"] * 100,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2: separate prior-dependence from rejection of a false context."
    )
    parser.add_argument(
        "--model-path", default=os.path.join(MODELS_DIR, "Qwen3-4B-Instruct-2507")
    )
    parser.add_argument("--dataset", default="sciq", choices=sorted(DATASET_SPECS))
    parser.add_argument(
        "--out", default="results/ex2_counterfactual/results_prior_vs_rejection_qwen3_4b.json"
    )
    parser.add_argument(
        "--log-file", default="results/ex2_counterfactual/prior_vs_rejection_qwen3_4b.log"
    )
    parser.add_argument("--min-substitution-tier", choices=TIERS, default="partial")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--verify-c", type=int, default=50,
        help="Gate 1: re-score N Setting-C items under the original prompt vs E1.",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--double-quant", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    args = parser.parse_args()

    log_path = ensure_parent(resolve_path(args.log_file))
    setup_logging(log_path, args.verbose, args.append_log)
    run_started = time.time()

    logger.info("=" * 100)
    logger.info("E2 -- PRIOR-DEPENDENCE vs REJECTION OF A FALSE CONTEXT")
    logger.info("=" * 100)
    logger.info("Workspace : %s", PROJECT_ROOT)
    logger.info("Python    : %s", platform.python_version())
    logger.info("torch     : %s (CUDA %s)", torch.__version__, torch.cuda.is_available())
    logger.info("C' system prompt: %s", CONTEXT_PRIORITY_SYSTEM_PROMPT)

    model_path = resolve_path(args.model_path)
    if not os.path.isdir(model_path):
        logger.error("Model directory not found: %s", model_path)
        return 1

    e1_path = resolve_path(E1_RESULTS)
    if not os.path.exists(e1_path):
        logger.error("E1 results not found at %s -- run run_counterfactual_qwen.py first.", e1_path)
        return 1
    with open(e1_path, encoding="utf-8") as handle:
        e1_records = {r["id"]: r for r in json.load(handle)["results"][args.dataset]}
    logger.info("Joined %d E1 records from %s", len(e1_records), E1_RESULTS)

    # --- counterfactual generation, identical to E1 -----------------------------------
    spec = DATASET_SPECS[args.dataset]
    with open(resolve_path(spec["path"]), encoding="utf-8") as handle:
        raw = json.load(handle)
    if args.limit:
        raw = raw[: args.limit]

    samples: List[Dict] = []
    for entry in raw:
        counterfactual = build_counterfactual(entry)
        if counterfactual["counterfactual_valid"] and not tier_at_least(
            counterfactual["substitution_tier"], args.min_substitution_tier
        ):
            counterfactual["counterfactual_valid"] = False
        merged = {**entry, **counterfactual}
        if merged["counterfactual_target_idx"] is None:
            merged["counterfactual_target_idx"] = (
                (merged["answer_idx"] + 1) % len(merged["options"])
            )
        samples.append(merged)
    n_eligible = sum(1 for s in samples if s["counterfactual_valid"])
    logger.info("Generated %d usable counterfactual passages of %d", n_eligible, len(samples))

    generation_gate = None
    if not args.limit and args.min_substitution_tier == "partial":
        generation_gate = verify_counterfactual_generation(args.dataset, samples)
        logger.info(
            "GATE 2 counterfactual generation vs committed preview: %s (%d compared)",
            "PASS" if generation_gate["passed"] else "FAIL", generation_gate["n_compared"],
        )
        if not generation_gate["passed"]:
            for mismatch in generation_gate["mismatches"]:
                logger.error("    %s", mismatch)
            return 1

    model, tokenizer, memory = load_model(model_path, args, logger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    letter_ids = letter_token_ids(tokenizer, logger)

    # --- gate 1 ------------------------------------------------------------------------
    c_gate = None
    if args.verify_c:
        c_gate = verify_setting_c(
            model, tokenizer, letter_ids, device, samples, e1_records, args.verify_c
        )
        logger.info(
            "GATE 1 Setting C under the ORIGINAL prompt vs E1: %s "
            "(%d checked, %d mismatches, max |dP| = %.2e)",
            "PASS" if c_gate["passed"] else "FAIL", c_gate["n_checked"],
            c_gate["n_prediction_mismatches"], c_gate["max_abs_probability_delta"],
        )
        if not c_gate["passed"]:
            for mismatch in c_gate["mismatches"]:
                logger.error("    %s", mismatch)
            logger.error(
                "This harness does not reproduce E1, so no delta against E1 is meaningful."
            )
            return 1

    if args.verify_only:
        logger.info("--verify-only: gates passed, stopping before the full run.")
        return 0

    logger.info("")
    logger.info("Scoring %d samples: Settings B' and C' + two passage NLLs each ...", len(samples))
    records = score_dataset(
        model, tokenizer, letter_ids, device, samples, e1_records, args.progress_every
    )
    analysis = analyse(records)
    log_analysis(analysis)

    out_path = ensure_parent(resolve_path(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": {
                    "experiment": (
                        "E2 -- separating prior-dependence from rejection of a false "
                        "context, by instruction ablation and an intrinsic plausibility "
                        "gradient"
                    ),
                    "model": os.path.basename(model_path),
                    "quantization": "4-bit NF4",
                    "dataset": spec["dataset_name"],
                    "builds_on": E1_RESULTS,
                    "original_system_prompt": SYSTEM_PROMPT,
                    "context_priority_system_prompt": CONTEXT_PRIORITY_SYSTEM_PROMPT,
                    "correctness_gates": {
                        "setting_c_vs_e1": c_gate,
                        "counterfactual_generation": generation_gate,
                    },
                    "memory": memory,
                    "analysis": analysis,
                    "total_runtime_seconds": round(time.time() - run_started, 2),
                },
                "results": records,
            },
            handle, ensure_ascii=False, indent=2,
        )
    logger.info("")
    logger.info("Wrote %s", out_path)
    logger.info("Total runtime: %.1f s", time.time() - run_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
