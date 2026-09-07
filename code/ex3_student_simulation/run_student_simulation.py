"""Persona-conditioned student simulation under Zero-Context (Setting A / No Fact).

Motivation
----------
`code/ex1_reproduce_KDA/kda_qwen_eval.py` showed that Qwen3-4B-Instruct-2507 is
parametrically saturated on SciQ: Acc_wof = 95.4% with no passage at all (OBQA: 82.6%).
A simulated student that already knows almost every answer cannot stratify questions by
difficulty, and KDA collapses. This script asks whether *persona conditioning* can
manufacture the missing ability spread without any external context -- i.e. whether we can
push one saturated model into three separable ability tiers using the prompt alone.

Two paradigms
-------------
Paradigm 1 -- JOINT (contrastive). All three profiles appear in a single prompt together
    with the question, and the model emits one letter per tier in one continuation. The
    tiers are visible to each other, so the model must allocate *relative* competence.

Paradigm 2 -- ISOLATED (roleplay). Three independent prompts, each asking the model to be
    exactly one student. No tier sees any other. This measures whether self-roleplay alone
    suppresses the parametric prior.

Scoring protocol
----------------
Both paradigms use the same letter-logit protocol as the ex1 baseline, so the numbers are
directly comparable and no answer is ever lost to free-text parsing:

    - the chat template is applied with add_generation_prompt=True;
    - logits at the final position are restricted to the option letters A/B/C/D;
    - all single-token spellings of a letter are combined with logsumexp;
    - a softmax over just those four values gives a proper distribution over the options.

For the joint paradigm the assistant turn is *teacher-forced* through a fixed scaffold:

    "Beginner:"      -> read letter distribution, commit the argmax letter
    "\nIntermediate:" -> read letter distribution, commit the argmax letter
    "\nAdvanced:"     -> read letter distribution, commit the argmax letter

Committing each tier's letter before scoring the next is what makes the paradigm genuinely
joint: the intermediate and advanced predictions are conditioned on the answers already
attributed to the weaker tiers. A KV cache is carried across the three reads, so a joint
question costs one long forward pass plus two short ones.

Usage:
    uv run --active python code/ex3_student_simulation/run_student_simulation.py --limit 50
    uv run --active python code/ex3_student_simulation/run_student_simulation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import re
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------------------
# Cross-stage imports: `code/` goes on sys.path, never the project root (a `datasets/`
# folder lives there and would shadow the HuggingFace `datasets` package).
# --------------------------------------------------------------------------------------
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex3_student_simulation.personas import (
    PERSONAS, TIERS, TIER_LABELS, all_persona_blocks, persona_block,
)
from utils.paths import MODELS_DIR, PROJECT_ROOT, ensure_parent, resolve

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
    "transformers", "datasets", "fsspec", "accelerate", "bitsandbytes",
)

LETTERS = ("A", "B", "C", "D")
PARADIGMS = ("joint", "isolated")

DATASET_SPECS = {
    "sciq": {
        "path": "datasets/sciq/sciq_test_full.json",
        "dataset_name": "allenai/sciq",
        "split": "test",
        "out": "results/ex3_student_simulation/results_persona_simulation_sciq.json",
        "subject": "science",
    },
    "obqa": {
        "path": "datasets/openbookqa/obqa_test_full.json",
        "dataset_name": "allenai/openbookqa",
        "split": "test",
        "out": "results/ex3_student_simulation/results_persona_simulation_obqa.json",
        "subject": "elementary science",
    },
}


def setup_logging(log_path: str, verbose: bool, append: bool) -> logging.Logger:
    """DEBUG records to `log_path`, INFO records to stdout."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(
        ensure_parent(log_path), mode="a" if append else "w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("student_sim")


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------
JOINT_SYSTEM = (
    "You are a psychometrics expert who models student behaviour. Given a student profile "
    "and an exam question, you predict the option that student would actually select -- "
    "which is often not the correct option. You never reveal the correct answer; you report "
    "predicted selections only, in the exact format requested."
)

ISOLATED_SYSTEM = (
    "You are role-playing a specific student taking a multiple-choice exam. Stay strictly "
    "in character: answer exactly as that student would, using only the knowledge and "
    "reasoning habits their profile describes, even when that leads to a wrong option. "
    "Answer with a single letter and nothing else."
)


def format_question(sample: Dict) -> str:
    """The question stem plus its four lettered options."""
    lines = [f"Question: {sample['question'].strip()}"]
    for letter, option in zip(LETTERS, sample["options"]):
        lines.append(f"{letter}. {str(option).strip()}")
    return "\n".join(lines)


def build_joint_user_turn(
    sample: Dict, dataset_key: str, spec: Dict, order: Sequence[str] = TIERS
) -> str:
    """All three profiles side by side, one question, one prediction per tier.

    `order` fixes the sequence the tiers are reported in. It is a parameter because the
    emission order turns out to matter: see `--joint-order`.
    """
    reply_lines = "\n".join(f"{TIER_LABELS[tier]}: <letter>" for tier in order)
    return "\n\n".join(
        [
            f"Three students of different ability levels are taking the same "
            f"{spec['subject']} exam. Their profiles:",
            all_persona_blocks(dataset_key),
            format_question(sample),
            (
                "For each of the three students, predict the option that student would "
                "actually choose, reasoning from their profile. Compare the three students "
                "against each other: a weaker student should be predicted to fall for the "
                "traps their profile describes, so their predicted answer will often be "
                "wrong. Two students may well choose the same option -- predict each one "
                "independently on their own merits, and do not force their answers to "
                "differ. Do not report the correct answer -- report each student's "
                "predicted selection.\n"
                "Reply with exactly three lines and nothing else:\n"
                f"{reply_lines}"
            ),
        ]
    )


def build_isolated_user_turn(sample: Dict, dataset_key: str, tier: str, spec: Dict) -> str:
    """One profile, adopted in the first person, one question."""
    return "\n\n".join(
        [
            f"You are this student, taking a {spec['subject']} exam:",
            persona_block(dataset_key, tier),
            (
                "Answer as this student genuinely would. If their knowledge or reasoning "
                "habits would lead them to a wrong option, choose that wrong option -- do "
                "not answer better than this student can."
            ),
            format_question(sample),
            "Answer with a single letter (A, B, C, or D).",
        ]
    )


def render_chat(tokenizer, system_prompt: str, user_text: str) -> str:
    """Apply the model's chat template and open the assistant turn."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------------------
# Letter token bookkeeping (identical protocol to ex1, kept local so the stages stay
# independently runnable)
# --------------------------------------------------------------------------------------
def letter_token_ids(tokenizer, logger: logging.Logger) -> Dict[str, List[int]]:
    """Every single-token spelling of each option letter (bare, space-prefixed)."""
    variants: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids: List[int] = []
        for spelling in (letter, f" {letter}"):
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        if not ids:
            raise RuntimeError(
                f"Option letter {letter!r} has no single-token spelling in this tokenizer; "
                "the letter-logit protocol cannot be applied."
            )
        variants[letter] = ids
        logger.debug("  letter %s -> token ids %s", letter, ids)
    return variants


def letter_probs_from_logits(logits: torch.Tensor, letter_ids: Dict[str, List[int]]) -> List[float]:
    """Total mass per letter across tokenisation variants, renormalised over A/B/C/D."""
    per_letter = torch.stack(
        [
            torch.logsumexp(logits[torch.tensor(ids, device=logits.device)], dim=0)
            for ids in (letter_ids[l] for l in LETTERS)
        ]
    )
    return torch.softmax(per_letter, dim=0).tolist()


class PrefixCachedScorer:
    """Reads letter distributions at successive points of one growing assistant turn.

    `logits_at(text)` returns the next-token logits for the *cumulative* `text`. Because the
    joint scaffold only ever appends, the previously cached tokens are almost always a
    prefix of the new tokenisation, and only the delta needs a forward pass. When a
    tokeniser boundary re-merge breaks that assumption the cache is dropped and the full
    text is recomputed, so the result is always exact.
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.cache = None
        self.cached_ids: List[int] = []
        self.forward_tokens = 0

    @torch.no_grad()
    def logits_at(self, text: str) -> torch.Tensor:
        ids: List[int] = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        reuse = len(self.cached_ids)
        if reuse and (reuse >= len(ids) or ids[:reuse] != self.cached_ids):
            self.reset()          # boundary re-merge (or no growth): recompute exactly
            reuse = 0

        delta = ids[reuse:]
        input_ids = torch.tensor([delta], device=self.device)
        attention_mask = torch.ones((1, len(ids)), dtype=torch.long, device=self.device)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = out.past_key_values
        self.cached_ids = ids
        self.forward_tokens += len(delta)
        return out.logits[0, -1, :].float()


# --------------------------------------------------------------------------------------
# Per-question scoring
# --------------------------------------------------------------------------------------
@torch.no_grad()
def score_joint(
    scorer: PrefixCachedScorer,
    tokenizer,
    sample: Dict,
    dataset_key: str,
    spec: Dict,
    letter_ids: Dict[str, List[int]],
    order: Sequence[str] = TIERS,
) -> Tuple[Dict[str, List[float]], int]:
    """Teacher-force one label per tier in `order` and read a distribution at each."""
    scorer.reset()
    text = render_chat(
        tokenizer, JOINT_SYSTEM, build_joint_user_turn(sample, dataset_key, spec, order)
    )

    probs: Dict[str, List[float]] = {}
    for position, tier in enumerate(order):
        text += ("" if position == 0 else "\n") + f"{TIER_LABELS[tier]}:"
        tier_probs = letter_probs_from_logits(scorer.logits_at(text), letter_ids)
        probs[tier] = tier_probs
        # Commit this tier's answer into the context before scoring the next one --
        # this is what makes the three predictions jointly, not independently, drawn.
        text += f" {LETTERS[max(range(4), key=lambda i: tier_probs[i])]}"

    return probs, scorer.forward_tokens


@torch.no_grad()
def score_isolated(
    scorer: PrefixCachedScorer,
    tokenizer,
    sample: Dict,
    dataset_key: str,
    spec: Dict,
    letter_ids: Dict[str, List[int]],
) -> Tuple[Dict[str, List[float]], int]:
    """Three fully independent single-persona passes."""
    probs: Dict[str, List[float]] = {}
    total_tokens = 0
    for tier in TIERS:
        scorer.reset()
        text = render_chat(
            tokenizer, ISOLATED_SYSTEM, build_isolated_user_turn(sample, dataset_key, tier, spec)
        )
        probs[tier] = letter_probs_from_logits(scorer.logits_at(text), letter_ids)
        total_tokens += scorer.forward_tokens
    return probs, total_tokens


# --------------------------------------------------------------------------------------
# Surface-heuristic probes for the error analysis
# --------------------------------------------------------------------------------------
STOPWORDS = frozenset(
    """a an the of to in on at for from by with as is are was were be been being do does did
    what which who whom whose when where why how that this these those it its they them their
    and or but if then than so such can could will would shall should may might must not no
    all some any each most more less other another same different called known used using
    you your we our i me my he she his her""".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> frozenset:
    """Lower-cased alphanumeric tokens with stopwords and 1-character noise removed."""
    return frozenset(
        tok for tok in _WORD_RE.findall(str(text).lower())
        if tok not in STOPWORDS and len(tok) > 1
    )


def unique_argmax(scores: Sequence[float]) -> Optional[int]:
    """Index of the strict maximum, or None when the top score is tied."""
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    return winners[0] if len(winners) == 1 else None


def longest_option_index(options: Sequence[str]) -> Optional[int]:
    """The uniquely longest option by character count, else None."""
    return unique_argmax([len(str(o).strip()) for o in options])


def surface_match_index(question: str, options: Sequence[str]) -> Optional[int]:
    """The option sharing the most content words with the stem, else None when tied.

    This is the "it repeats a word from the question, so it must be the answer" heuristic.
    Options sharing no content word at all are never a match. On SciQ and OBQA the options
    are mostly short noun phrases that share nothing with the stem, so this probe is sparse
    (~4-9% of questions are eligible); `lookalike_index` is the dense counterpart.
    """
    stem = content_tokens(question)
    overlaps = [len(stem & content_tokens(o)) for o in options]
    if max(overlaps) == 0:
        return None
    return unique_argmax(overlaps)


def char_trigrams(text: str) -> frozenset:
    """Character trigrams of the squeezed, lower-cased string."""
    squeezed = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    padded = f"  {squeezed}  "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def lookalike_index(options: Sequence[str], gold_idx: int) -> Optional[int]:
    """The distractor whose surface form most resembles the gold answer, else None if tied.

    The dense form of "misled by surface-level association": on SciQ the gold answer
    'oxidants' sits beside the distractor 'Oxygen', and a reader who recognises the shape of
    a word without knowing its meaning is drawn to the near-twin. Similarity is character
    trigram Jaccard, which needs no embeddings and no extra model.
    """
    gold = char_trigrams(options[gold_idx])
    scores = []
    for i, option in enumerate(options):
        if i == gold_idx:
            scores.append(-1.0)
            continue
        other = char_trigrams(option)
        union = gold | other
        scores.append(len(gold & other) / len(union) if union else 0.0)
    return unique_argmax(scores)


def entropy_bits(probs: Sequence[float]) -> float:
    """Shannon entropy in bits; 2.0 is uniform over four options."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def tier_metrics(records: Sequence[Dict], paradigm: str, tier: str) -> Dict:
    """Accuracy, calibration, answer-position spread and trap rates for one tier."""
    n = len(records)
    picks = [rec[paradigm][tier]["pred_idx"] for rec in records]
    golds = [rec["answer_idx"] for rec in records]
    probs = [rec[paradigm][tier]["probs"] for rec in records]
    correct = [int(p == g) for p, g in zip(picks, golds)]

    letter_counts = {letter: 0 for letter in LETTERS}
    for pick in picks:
        letter_counts[LETTERS[pick]] += 1

    error_positions = [i for i, c in enumerate(correct) if c == 0]
    error_letter_counts = {letter: 0 for letter in LETTERS}
    for i in error_positions:
        error_letter_counts[LETTERS[picks[i]]] += 1

    # --- Trap probes -------------------------------------------------------------------
    # Restricted to questions where the trap option exists, is unique, and is a *distractor*
    # (if the trap coincides with the gold answer it cannot explain an error). Among that
    # eligible pool we report the share of the tier's errors that landed on the trap; a
    # student erring at random over the three distractors would score 1/3.
    trap_stats = {}
    for name, finder in (
        ("longest_option", lambda rec: longest_option_index(rec["options"])),
        ("surface_match", lambda rec: surface_match_index(rec["question"], rec["options"])),
        ("lookalike_to_gold", lambda rec: lookalike_index(rec["options"], rec["answer_idx"])),
    ):
        eligible = 0
        eligible_errors = 0
        hits = 0
        for rec, pick, gold, ok in zip(records, picks, golds, correct):
            trap = finder(rec)
            if trap is None or trap == gold:
                continue
            eligible += 1
            if ok:
                continue
            eligible_errors += 1
            if pick == trap:
                hits += 1
        trap_stats[name] = {
            "eligible_questions": eligible,
            "eligible_errors": eligible_errors,
            "trap_hits": hits,
            "trap_hit_rate_among_errors": (hits / eligible_errors) if eligible_errors else None,
            "random_distractor_baseline": 1 / 3,
            "lift_over_random": (
                (hits / eligible_errors) / (1 / 3) if eligible_errors else None
            ),
        }

    return {
        "n_samples": n,
        "accuracy": sum(correct) / n if n else None,
        "n_correct": sum(correct),
        "n_errors": len(error_positions),
        "mean_p_correct": statistics.fmean([p[g] for p, g in zip(probs, golds)]) if n else None,
        "mean_p_selected": statistics.fmean([p[i] for p, i in zip(probs, picks)]) if n else None,
        "mean_entropy_bits": statistics.fmean([entropy_bits(p) for p in probs]) if n else None,
        "mean_p_selected_on_errors": (
            statistics.fmean([probs[i][picks[i]] for i in error_positions])
            if error_positions else None
        ),
        "answer_position_distribution": {
            letter: round(count / n, 4) for letter, count in letter_counts.items()
        } if n else None,
        "error_position_distribution": {
            letter: round(count / len(error_positions), 4)
            for letter, count in error_letter_counts.items()
        } if error_positions else None,
        "trap_probes": trap_stats,
    }


def jaccard(a: set, b: set) -> Optional[float]:
    union = a | b
    return (len(a & b) / len(union)) if union else None


def paradigm_metrics(records: Sequence[Dict], paradigm: str) -> Dict:
    """Per-tier metrics plus the monotonicity and spread checks for one paradigm."""
    per_tier = {tier: tier_metrics(records, paradigm, tier) for tier in TIERS}
    acc = {tier: per_tier[tier]["accuracy"] for tier in TIERS}

    error_sets = {
        tier: {
            rec["id"] for rec in records
            if rec[paradigm][tier]["pred_idx"] != rec["answer_idx"]
        }
        for tier in TIERS
    }

    return {
        "per_tier": per_tier,
        "accuracy_by_tier": acc,
        "ability_gap_advanced_minus_beginner": (
            acc["advanced"] - acc["beginner"]
            if acc["advanced"] is not None and acc["beginner"] is not None else None
        ),
        "monotonicity": {
            "strict_beginner_lt_intermediate": acc["beginner"] < acc["intermediate"],
            "intermediate_le_advanced": acc["intermediate"] <= acc["advanced"],
            "holds": acc["beginner"] < acc["intermediate"] <= acc["advanced"],
        },
        "error_set_overlap_jaccard": {
            "beginner_vs_intermediate": jaccard(error_sets["beginner"], error_sets["intermediate"]),
            "beginner_vs_advanced": jaccard(error_sets["beginner"], error_sets["advanced"]),
            "intermediate_vs_advanced": jaccard(error_sets["intermediate"], error_sets["advanced"]),
        },
        "error_containment": {
            # Share of the stronger tier's errors that the weaker tier also made. A coherent
            # ability ladder nests: what an advanced student misses, a beginner misses too.
            "advanced_errors_also_missed_by_beginner": (
                len(error_sets["advanced"] & error_sets["beginner"]) / len(error_sets["advanced"])
                if error_sets["advanced"] else None
            ),
            "intermediate_errors_also_missed_by_beginner": (
                len(error_sets["intermediate"] & error_sets["beginner"]) / len(error_sets["intermediate"])
                if error_sets["intermediate"] else None
            ),
        },
    }


def joint_differentiation_metrics(records: Sequence[Dict], order: Sequence[str]) -> Dict:
    """Does the joint scaffold pressure later tiers into *differing* from earlier ones?

    The joint prompt commits each tier's letter into the context before the next tier is
    scored. That is the mechanism that makes the prediction contrastive, but it also lets
    the model treat "three students" as "three different answers". This measures that
    directly: when the tiers scored earlier already agree with each other, how often does
    the last-scored tier break away, and how often is breaking away simply wrong?
    """
    n = len(records)
    if n == 0:
        return {}

    last = order[-1]
    earlier = list(order[:-1])

    all_distinct = 0
    earlier_agree = 0
    last_deviates = 0
    deviation_wrong = 0
    deviation_from_correct_consensus = 0
    for rec in records:
        picks = {tier: rec["joint"][tier]["pred_idx"] for tier in order}
        if len({picks[t] for t in order}) == 3:
            all_distinct += 1
        if len({picks[t] for t in earlier}) == 1:
            earlier_agree += 1
            consensus = picks[earlier[0]]
            if picks[last] != consensus:
                last_deviates += 1
                if picks[last] != rec["answer_idx"]:
                    deviation_wrong += 1
                if consensus == rec["answer_idx"]:
                    deviation_from_correct_consensus += 1

    return {
        "emission_order": list(order),
        "last_scored_tier": last,
        "all_three_tiers_distinct_rate": all_distinct / n,
        "chance_all_distinct_if_independent_uniform": 0.375,  # 4*3*2 / 4^3
        "n_questions_earlier_tiers_agree": earlier_agree,
        "last_tier_deviates_from_agreeing_earlier_tiers": (
            last_deviates / earlier_agree if earlier_agree else None
        ),
        "deviations_that_are_wrong": (
            deviation_wrong / last_deviates if last_deviates else None
        ),
        "deviations_away_from_a_correct_consensus": (
            deviation_from_correct_consensus / last_deviates if last_deviates else None
        ),
    }


def cross_paradigm_metrics(records: Sequence[Dict]) -> Dict:
    """How far the two paradigms disagree, tier by tier."""
    out = {}
    for tier in TIERS:
        agree = sum(
            1 for rec in records
            if rec["joint"][tier]["pred_idx"] == rec["isolated"][tier]["pred_idx"]
        )
        out[tier] = {
            "prediction_agreement": agree / len(records) if records else None,
            "accuracy_joint": sum(
                1 for rec in records if rec["joint"][tier]["pred_idx"] == rec["answer_idx"]
            ) / len(records) if records else None,
            "accuracy_isolated": sum(
                1 for rec in records if rec["isolated"][tier]["pred_idx"] == rec["answer_idx"]
            ) / len(records) if records else None,
        }
    return out


# --------------------------------------------------------------------------------------
# Model loading (mirrors ex1 so the two stages share one VRAM profile)
# --------------------------------------------------------------------------------------
def load_model(model_path: str, args, logger: logging.Logger):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=args.double_quant,
    )
    logger.info(
        "Quantisation : 4-bit NF4, compute dtype float16, double_quant=%s", args.double_quant
    )

    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.time() - started

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded in %.1fs | %.2fB parameters (packed)", load_seconds, n_params / 1e9)

    memory = {"load_seconds": round(load_seconds, 2), "n_parameters_packed": n_params}
    if torch.cuda.is_available():
        memory.update(
            {
                "weights_vram_gb": round((torch.cuda.memory_allocated() - before) / 1024**3, 3),
                "allocated_after_load_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "reserved_after_load_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "device_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
                ),
            }
        )
        logger.info(
            "VRAM after load: %.2f GB allocated / %.2f GB reserved of %.2f GB total",
            memory["allocated_after_load_gb"], memory["reserved_after_load_gb"],
            memory["device_total_gb"],
        )
    return model, tokenizer, memory


# --------------------------------------------------------------------------------------
# Evaluation driver
# --------------------------------------------------------------------------------------
def evaluate_dataset(
    model, tokenizer, letter_ids: Dict[str, List[int]],
    key: str, spec: Dict, args, logger: logging.Logger,
) -> Dict:
    joint_order = TIERS if args.joint_order == "ascending" else list(reversed(TIERS))
    data_path = resolve(spec["path"])
    with open(data_path, encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.limit:
        samples = samples[: args.limit]

    logger.info("")
    logger.info("=" * 100)
    logger.info(
        "DATASET %s (%s, %s) -- %d samples, zero-context (no passage)",
        key.upper(), spec["dataset_name"], spec["split"], len(samples),
    )
    logger.info("  file      : %s", data_path)
    logger.info("  paradigms : %s", ", ".join(args.paradigms))
    logger.info("  joint order: %s", " -> ".join(joint_order))
    logger.info("=" * 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = PrefixCachedScorer(model, tokenizer, device)
    records: List[Dict] = []
    progress_every = args.progress_every or max(1, len(samples) // 20)
    started = time.time()

    for index, sample in enumerate(samples):
        answer_idx = int(sample["answer_idx"])
        sample_started = time.time()

        record = {
            "id": sample.get("id", index),
            "question": sample["question"],
            "options": sample["options"],
            "answer_idx": answer_idx,
            "correct_answer": sample.get("correct_answer", sample["options"][answer_idx]),
            "longest_option_idx": longest_option_index(sample["options"]),
            "surface_match_idx": surface_match_index(sample["question"], sample["options"]),
            "lookalike_idx": lookalike_index(sample["options"], answer_idx),
        }

        token_counts = {}
        for paradigm in args.paradigms:
            if paradigm == "joint":
                probs, n_tokens = score_joint(
                    scorer, tokenizer, sample, key, spec, letter_ids, joint_order
                )
            else:
                probs, n_tokens = score_isolated(
                    scorer, tokenizer, sample, key, spec, letter_ids
                )
            token_counts[paradigm] = n_tokens
            record[paradigm] = {
                tier: {
                    "probs": probs[tier],
                    "pred_idx": int(max(range(4), key=lambda i: probs[tier][i])),
                    "p_correct": probs[tier][answer_idx],
                }
                for tier in TIERS
            }
            for tier in TIERS:
                record[paradigm][tier]["correct"] = int(
                    record[paradigm][tier]["pred_idx"] == answer_idx
                )

        record["prompt_tokens"] = token_counts
        record["latency_seconds"] = time.time() - sample_started
        records.append(record)

        logger.debug(
            "id=%s gold=%s | joint=%s | isolated=%s",
            record["id"], LETTERS[answer_idx],
            {t: LETTERS[record["joint"][t]["pred_idx"]] for t in TIERS} if "joint" in record else None,
            {t: LETTERS[record["isolated"][t]["pred_idx"]] for t in TIERS} if "isolated" in record else None,
        )

        if (index + 1) % progress_every == 0 or (index + 1) == len(samples):
            done = index + 1
            elapsed = time.time() - started
            running = " | ".join(
                f"{p[:4]} "
                + "/".join(
                    f"{sum(r[p][t]['correct'] for r in records) / done:.2f}" for t in TIERS
                )
                for p in args.paradigms
            )
            logger.info(
                "  [%4d/%4d] %5.1f%% | acc B/I/A  %s | %.2fs/sample | ETA %.1f min",
                done, len(samples), 100.0 * done / len(samples), running,
                elapsed / done, (elapsed / done) * (len(samples) - done) / 60.0,
            )

    scoring_seconds = time.time() - started

    metrics = {
        "dataset_key": key,
        "dataset_name": spec["dataset_name"],
        "split": spec["split"],
        "data_file": data_path,
        "setting": "zero_context_no_fact",
        "n_samples": len(records),
        "scoring_seconds": round(scoring_seconds, 2),
        "mean_latency_seconds": statistics.fmean([r["latency_seconds"] for r in records]),
        "joint_emission_order": list(joint_order),
        "paradigms": {p: paradigm_metrics(records, p) for p in args.paradigms},
    }
    if "joint" in args.paradigms:
        metrics["joint_differentiation"] = joint_differentiation_metrics(records, joint_order)
    if len(args.paradigms) == 2:
        metrics["cross_paradigm"] = cross_paradigm_metrics(records)
        joint_gap = metrics["paradigms"]["joint"]["ability_gap_advanced_minus_beginner"]
        isolated_gap = metrics["paradigms"]["isolated"]["ability_gap_advanced_minus_beginner"]
        metrics["gap_joint_minus_isolated"] = joint_gap - isolated_gap

    log_dataset_summary(metrics, args, logger)
    return {"metrics": metrics, "records": records}


def log_dataset_summary(metrics: Dict, args, logger: logging.Logger) -> None:
    logger.info("")
    logger.info("  %s -- ACCURACY STRATIFICATION", metrics["dataset_key"].upper())
    logger.info("  " + "-" * 96)
    logger.info(
        "    %-10s %10s %14s %14s %10s %10s",
        "paradigm", "beginner", "intermediate", "advanced", "gap", "monotone",
    )
    for paradigm in args.paradigms:
        block = metrics["paradigms"][paradigm]
        acc = block["accuracy_by_tier"]
        logger.info(
            "    %-10s %10.4f %14.4f %14.4f %10.4f %10s",
            paradigm, acc["beginner"], acc["intermediate"], acc["advanced"],
            block["ability_gap_advanced_minus_beginner"],
            "yes" if block["monotonicity"]["holds"] else "NO",
        )
    if "gap_joint_minus_isolated" in metrics:
        logger.info("    gap(joint) - gap(isolated) = %+.4f", metrics["gap_joint_minus_isolated"])
    if "joint_differentiation" in metrics:
        jd = metrics["joint_differentiation"]
        logger.info("")
        logger.info("    joint scaffold differentiation pressure (order %s)", " -> ".join(jd["emission_order"]))
        logger.info("      all three tiers distinct        : %.3f (chance if independent: 0.375)",
                    jd["all_three_tiers_distinct_rate"])
        logger.info("      last tier (%s) breaks a tie : %s  over %d agreeing questions",
                    jd["last_scored_tier"],
                    _fmt(jd["last_tier_deviates_from_agreeing_earlier_tiers"]),
                    jd["n_questions_earlier_tiers_agree"])
        logger.info("      those breaks that are wrong     : %s", _fmt(jd["deviations_that_are_wrong"]))
        logger.info("      breaks away from correct answer : %s",
                    _fmt(jd["deviations_away_from_a_correct_consensus"]))
    logger.info("")
    logger.info("    trap-hit rate among errors (random distractor baseline = 0.333)")
    for paradigm in args.paradigms:
        for tier in TIERS:
            probes = metrics["paradigms"][paradigm]["per_tier"][tier]["trap_probes"]
            logger.info(
                "      %-9s %-13s longest %s (n=%4d) | stem-overlap %s (n=%3d) | lookalike %s (n=%4d)",
                paradigm, tier,
                _fmt(probes["longest_option"]["trap_hit_rate_among_errors"]),
                probes["longest_option"]["eligible_errors"],
                _fmt(probes["surface_match"]["trap_hit_rate_among_errors"]),
                probes["surface_match"]["eligible_errors"],
                _fmt(probes["lookalike_to_gold"]["trap_hit_rate_among_errors"]),
                probes["lookalike_to_gold"]["eligible_errors"],
            )
    logger.info("  " + "-" * 96)


def _fmt(value: Optional[float]) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persona-conditioned student simulation (zero context) with Qwen3-4B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run --active python code/ex3_student_simulation/run_student_simulation.py --limit 50\n"
            "  uv run --active python code/ex3_student_simulation/run_student_simulation.py\n"
        ),
    )
    parser.add_argument(
        "--model-path", default=os.path.join(MODELS_DIR, "Qwen3-4B-Instruct-2507"),
        help="Local model directory (default: models/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["sciq", "obqa"], choices=sorted(DATASET_SPECS),
        help="Which datasets to score.",
    )
    parser.add_argument(
        "--paradigms", nargs="+", default=list(PARADIGMS), choices=list(PARADIGMS),
        help="Which prompting paradigms to run.",
    )
    parser.add_argument(
        "--joint-order", default="ascending", choices=["ascending", "descending"],
        help=(
            "Order the tiers are emitted in inside the joint scaffold. 'ascending' is "
            "Beginner->Intermediate->Advanced (the primary design); 'descending' reverses "
            "it, which is the control for emission-order artifacts."
        ),
    )
    parser.add_argument(
        "--out-dir", default="results/ex3_student_simulation",
        help="Directory for the per-dataset result files.",
    )
    parser.add_argument(
        "--log-file", default="results/ex3_student_simulation/student_simulation.log",
        help="Execution log path.",
    )
    parser.add_argument(
        "--tag", default="", help="Suffix for output filenames (e.g. 'smoke50')."
    )
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N samples.")
    parser.add_argument(
        "--double-quant", action="store_true",
        help="Enable bnb nested (double) quantisation; saves ~0.1GB VRAM.",
    )
    parser.add_argument("--progress-every", type=int, default=0, help="Progress cadence (0 = auto, ~5%%).")
    parser.add_argument("--verbose", action="store_true", help="Stream DEBUG logs to console.")
    parser.add_argument("--append-log", action="store_true", help="Append to the log file.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.paradigms = [p for p in PARADIGMS if p in args.paradigms]  # canonical order
    logger = setup_logging(resolve(args.log_file), args.verbose, args.append_log)

    model_path = resolve(args.model_path)
    if not os.path.isdir(model_path):
        logger.error(
            "Model directory not found: %s\nRun: uv run --active python code/pre_data/download_models.py",
            model_path,
        )
        return 1

    total_started = time.time()
    logger.info("=" * 100)
    logger.info("PERSONA-CONDITIONED STUDENT SIMULATION -- zero context (Setting A / no fact)")
    logger.info("=" * 100)
    logger.info("Workspace   : %s", PROJECT_ROOT)
    logger.info("Model path  : %s", model_path)
    logger.info("Python      : %s", platform.python_version())
    logger.info("Platform    : %s", platform.platform())
    logger.info("torch       : %s (cuda %s)", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info(
            "GPU         : %s (%.2f GB)", torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )
    logger.info("Datasets    : %s", ", ".join(args.datasets))
    logger.info("Paradigms   : %s", ", ".join(args.paradigms))
    logger.info("Tiers       : %s", ", ".join(TIERS))
    logger.info("Limit       : %s", args.limit or "none (full split)")
    logger.info("")

    model, tokenizer, memory = load_model(model_path, args, logger)
    logger.info("Resolving option-letter token ids ...")
    letter_ids = letter_token_ids(tokenizer, logger)
    logger.info("  %s", {l: letter_ids[l] for l in LETTERS})

    written: Dict[str, str] = {}
    all_metrics: Dict[str, Dict] = {}
    for key in args.datasets:
        spec = DATASET_SPECS[key]
        outcome = evaluate_dataset(model, tokenizer, letter_ids, key, spec, args, logger)
        all_metrics[key] = outcome["metrics"]

        if torch.cuda.is_available():
            memory["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            memory["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

        payload = {
            "summary": {
                "experiment": "persona_conditioned_student_simulation_zero_context",
                "model": os.path.basename(model_path),
                "model_path": model_path,
                "quantization": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": args.double_quant,
                },
                "setting": "zero_context (Setting A) -- no reference passage is ever shown",
                "tiers": TIERS,
                "paradigms": {
                    "joint": (
                        "all three profiles in one prompt; the assistant turn is teacher-forced "
                        "through 'Beginner:/Intermediate:/Advanced:' and each tier's committed "
                        "letter conditions the next"
                    ),
                    "isolated": "three independent single-persona roleplay prompts",
                },
                "scoring_protocol": (
                    "letter-logit: logits at the scored position restricted to A/B/C/D, "
                    "logsumexp over tokenisation variants, softmax over the four options"
                ),
                "personas": PERSONAS[key],
                "prompts": {
                    "joint_system": JOINT_SYSTEM,
                    "isolated_system": ISOLATED_SYSTEM,
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
                "memory": memory,
                "limit": args.limit or None,
                "metrics": all_metrics[key],
                "total_runtime_seconds": round(time.time() - total_started, 2),
            },
            "results": outcome["records"],
        }

        base = os.path.basename(spec["out"])
        if args.tag:
            stem, ext = os.path.splitext(base)
            base = f"{stem}_{args.tag}{ext}"
        out_path = ensure_parent(resolve(os.path.join(args.out_dir, base)))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        written[key] = out_path
        logger.info("  wrote %s", out_path)

    logger.info("")
    logger.info("=" * 100)
    logger.info("SUMMARY -- accuracy by ability tier (zero context)")
    logger.info("=" * 100)
    logger.info(
        "  %-8s %-10s %10s %14s %14s %10s", "dataset", "paradigm",
        "beginner", "intermediate", "advanced", "gap",
    )
    for key, metrics in all_metrics.items():
        for paradigm in args.paradigms:
            acc = metrics["paradigms"][paradigm]["accuracy_by_tier"]
            logger.info(
                "  %-8s %-10s %10.4f %14.4f %14.4f %10.4f", key, paradigm,
                acc["beginner"], acc["intermediate"], acc["advanced"],
                metrics["paradigms"][paradigm]["ability_gap_advanced_minus_beginner"],
            )
    logger.info("")
    for key, path in written.items():
        logger.info("Results %-6s: %s", key, path)
    logger.info("Log           : %s", resolve(args.log_file))
    logger.info("Runtime       : %.1f min", (time.time() - total_started) / 60.0)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
