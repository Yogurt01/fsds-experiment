"""Option-free free-response evaluation (cells A' and B').

Every other experiment in this repository scores a **closed set**: a softmax over four
multiple-choice logits, an option-string NLL, or a softmax over the letters A/B/C/D. None of them
can emit a token the option list did not supply, so every "zero-context accuracy" reported so far
is 4-way multiple choice with a 0.25 chance floor. This script is the first that asks the model to
*recall* an answer rather than *recognise* one.

    Cell A'   options removed, no fact       open-ended parametric recall   (the primary cell)
    Cell B'   options removed, fact present  open-ended reading comprehension

B' exists to make a low A' interpretable: it separates "the model does not know it" from "the
expected answer string is unguessable without seeing the options", which matters on OpenBookQA
where gold answers are annotator phrasings averaging 3.26 words.

Answers are graded by the three-stage cascade in `matching.py` (normalised exact -> morphological
-> LLM judge on the residual only), and accuracy is reported at all three tiers so a reader can
choose their own strictness.

Usage:
    uv run --active python code/ex4_free_response/run_free_response.py --limit 25 --tag pilot
    uv run --active python code/ex4_free_response/run_free_response.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex4_free_response.matching import (
    JUDGE_SYSTEM, accuracy_tiers, judge_user_turn, match_stage, postprocess_generation,
)
from utils.paths import MODELS_DIR, PROJECT_ROOT, ensure_parent, resolve

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
         "transformers", "datasets", "fsspec", "accelerate", "bitsandbytes")

CELLS = ("A_prime", "B_prime")

DATASET_SPECS = {
    "sciq": {"path": "datasets/sciq/sciq_test_full.json", "subject": "science"},
    "obqa": {"path": "datasets/openbookqa/obqa_test_full.json", "subject": "elementary science"},
}

SYSTEM_NO_FACT = (
    "You are a student taking a short-answer science exam. Answer with the answer only - a word "
    "or short phrase. Do not explain, do not write a sentence."
)
SYSTEM_WITH_FACT = (
    "You are a student taking a short-answer science exam. Answer using the reference text. "
    "Answer with the answer only - a word or short phrase. Do not explain, do not write a "
    "sentence."
)


def setup_logging(log_path: str, verbose: bool, append: bool) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler = logging.FileHandler(ensure_parent(log_path), mode="a" if append else "w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("free_response")


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
def build_user_turn(sample: Dict, cell: str) -> str:
    question = str(sample["question"]).strip()
    tail = "Answer with the answer only (a word or short phrase)."
    if cell == "B_prime":
        return f"Fact: {str(sample.get('passage', '')).strip()}\n\nQuestion: {question}\n\n{tail}"
    return f"Question: {question}\n\n{tail}"


def render_chat(tokenizer, system_prompt: str, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )


# --------------------------------------------------------------------------------------
# Generation and judging
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt_text: str, max_new_tokens: int, device: str) -> str:
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,                       # greedy: deterministic, comparable to the argmax settings
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)


def verdict_token_ids(tokenizer, logger: logging.Logger) -> Dict[str, int]:
    """First-token ids for CORRECT / INCORRECT; they must differ for the gate to work."""
    ids = {}
    for word in ("CORRECT", "INCORRECT"):
        encoded = tokenizer.encode(word, add_special_tokens=False)
        ids[word] = encoded[0]
    if ids["CORRECT"] == ids["INCORRECT"]:
        raise RuntimeError(
            "CORRECT and INCORRECT share a first token in this tokenizer; the judge gate "
            "cannot be applied as a two-way logit restriction."
        )
    logger.info("  judge verdict tokens: %s", ids)
    return ids


@torch.no_grad()
def judge(model, tokenizer, question: str, gold: str, prediction: str,
          verdict_ids: Dict[str, int], device: str) -> Dict:
    """Adjudicate one residual item; returns the verdict and its probability.

    Scored with the same letter-logit technique used elsewhere in the repo: the final-position
    logits are restricted to the two verdict tokens and softmaxed, so the judge is deterministic,
    calibrated, and free of any free-text parsing of its own.
    """
    text = render_chat(tokenizer, JUDGE_SYSTEM, judge_user_turn(question, gold, prediction))
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits = model(**encoded).logits[0, -1, :].float()
    pair = torch.stack([logits[verdict_ids["CORRECT"]], logits[verdict_ids["INCORRECT"]]])
    probs = torch.softmax(pair, dim=0).tolist()
    return {"verdict": "CORRECT" if probs[0] > probs[1] else "INCORRECT", "p_correct": probs[0]}


# --------------------------------------------------------------------------------------
# Model loading (same 4-bit configuration as every other Qwen run here)
# --------------------------------------------------------------------------------------
def load_model(model_path: str, args, logger: logging.Logger):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=args.double_quant,
    )
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quant, dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu", local_files_only=True,
    )
    model.eval()
    memory = {"load_seconds": round(time.time() - started, 2)}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory["allocated_after_load_gb"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 3)
        memory["device_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 3)
        logger.info("VRAM after load: %.2f GB of %.2f GB",
                    memory["allocated_after_load_gb"], memory["device_total_gb"])
    return model, tokenizer, memory


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def evaluate_dataset(model, tokenizer, verdict_ids, key, spec, args, logger) -> Dict:
    with open(resolve(spec["path"]), encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.limit:
        samples = samples[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("")
    logger.info("=" * 100)
    logger.info("DATASET %s -- %d samples, cells %s", key.upper(), len(samples), ", ".join(args.cells))
    logger.info("=" * 100)

    per_cell: Dict[str, List[Dict]] = {c: [] for c in args.cells}
    started = time.time()

    for index, sample in enumerate(samples):
        gold = sample.get("correct_answer", sample["options"][sample["answer_idx"]])
        for cell in args.cells:
            system = SYSTEM_WITH_FACT if cell == "B_prime" else SYSTEM_NO_FACT
            prompt = render_chat(tokenizer, system, build_user_turn(sample, cell))
            raw = generate(model, tokenizer, prompt, args.max_new_tokens, device)
            prediction = postprocess_generation(raw)

            stage = match_stage(prediction, gold)
            record = {
                "id": sample.get("id", index),
                "question": sample["question"],
                "gold_answer": gold,
                "options": sample["options"],
                "raw_generation": raw,
                "prediction": prediction,
                "match_stage": stage or "judge",
                "correct": stage is not None,
                "judge": None,
            }
            if stage is None:
                # Bidirectional grading. Semantic equivalence is symmetric, so swapping reference
                # and student should not change the verdict; the pilot found it changes 24.3% of
                # them. Both directions are recorded so the instability becomes a reported band
                # instead of hidden error.
                forward = judge(model, tokenizer, sample["question"], gold, prediction,
                                verdict_ids, device)
                reverse = judge(model, tokenizer, sample["question"], prediction, gold,
                                verdict_ids, device)
                record["judge"] = {"forward": forward, "reverse": reverse}
                record["judge_forward_correct"] = forward["verdict"] == "CORRECT"
                record["judge_reverse_correct"] = reverse["verdict"] == "CORRECT"
                record["judge_agree"] = forward["verdict"] == reverse["verdict"]
                # `correct` stays the forward-only verdict for continuity; the band is the
                # headline (see accuracy_tiers).
                record["correct"] = record["judge_forward_correct"]
            per_cell[cell].append(record)
            logger.debug("id=%s cell=%s gold=%r pred=%r stage=%s correct=%s",
                         record["id"], cell, gold, prediction, record["match_stage"], record["correct"])

        if (index + 1) % max(1, len(samples) // 10) == 0 or (index + 1) == len(samples):
            done = index + 1
            elapsed = time.time() - started
            summary = "  ".join(
                f"{c} {sum(r['correct'] for r in per_cell[c]) / done:.2f}" for c in args.cells
            )
            logger.info("  [%3d/%3d] %s | %.2fs/item | ETA %.1f min", done, len(samples), summary,
                        elapsed / done, (elapsed / done) * (len(samples) - done) / 60.0)

    metrics = {c: accuracy_tiers(per_cell[c]) for c in args.cells}
    metrics["seconds_per_item"] = (time.time() - started) / max(1, len(samples))
    return {"metrics": metrics, "records": per_cell}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Option-free free-response evaluation.")
    parser.add_argument("--model-path", default=os.path.join(MODELS_DIR, "Qwen3-4B-Instruct-2507"))
    parser.add_argument("--datasets", nargs="+", default=["sciq", "obqa"], choices=sorted(DATASET_SPECS))
    parser.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    parser.add_argument("--out-dir", default="results/ex4_free_response")
    parser.add_argument("--log-file", default="results/ex4_free_response/free_response.log")
    parser.add_argument("--tag", default="", help="Suffix for output filenames (e.g. 'pilot').")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--double-quant", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.cells = [c for c in CELLS if c in args.cells]
    logger = setup_logging(resolve(args.log_file), args.verbose, args.append_log)

    model_path = resolve(args.model_path)
    if not os.path.isdir(model_path):
        logger.error("Model directory not found: %s", model_path)
        return 1

    total_started = time.time()
    logger.info("=" * 100)
    logger.info("OPTION-FREE FREE-RESPONSE EVALUATION")
    logger.info("=" * 100)
    logger.info("Model    : %s", model_path)
    logger.info("Cells    : %s", ", ".join(args.cells))
    logger.info("Datasets : %s", ", ".join(args.datasets))
    logger.info("Limit    : %s", args.limit or "none (full split)")
    logger.info("Decoding : greedy, max_new_tokens=%d", args.max_new_tokens)

    model, tokenizer, memory = load_model(model_path, args, logger)
    verdict_ids = verdict_token_ids(tokenizer, logger)

    all_metrics: Dict[str, Dict] = {}
    for key in args.datasets:
        outcome = evaluate_dataset(model, tokenizer, verdict_ids, key, DATASET_SPECS[key], args, logger)
        all_metrics[key] = outcome["metrics"]

        if torch.cuda.is_available():
            memory["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)

        payload = {
            "summary": {
                "experiment": "option_free_free_response",
                "model": os.path.basename(model_path),
                "cells": {
                    "A_prime": "options removed, no fact (open-ended parametric recall)",
                    "B_prime": "options removed, fact present (open-ended reading comprehension)",
                },
                "prompts": {"system_no_fact": SYSTEM_NO_FACT, "system_with_fact": SYSTEM_WITH_FACT},
                "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens},
                "matching": "3-stage cascade: normalised exact -> morphological -> LLM judge (residual only)",
                "judge_system_prompt": JUDGE_SYSTEM,
                "memory": memory,
                "limit": args.limit or None,
                "metrics": all_metrics[key],
                "total_runtime_seconds": round(time.time() - total_started, 2),
            },
            "results": outcome["records"],
        }
        name = f"results_free_response_{key}" + (f"_{args.tag}" if args.tag else "") + ".json"
        out_path = ensure_parent(resolve(os.path.join(args.out_dir, name)))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        logger.info("  wrote %s", out_path)

    logger.info("")
    logger.info("=" * 100)
    logger.info("SUMMARY -- accuracy by matching tier")
    logger.info("=" * 100)
    logger.info("  %-6s %-9s %5s %9s %12s %22s %9s %8s",
                "data", "cell", "n", "strict", "normalised", "BAND [low, high]", "->judge", "flips")
    for key, metrics in all_metrics.items():
        for cell in args.cells:
            m = metrics[cell]
            band = f"[{m['acc_band_low']:.3f}, {m['acc_band_high']:.3f}]"
            flip = f"{m['judge_order_flip_rate']:.3f}" if m["judge_order_flip_rate"] is not None else "n/a"
            logger.info("  %-6s %-9s %5d %9.3f %12.3f %22s %9d %8s", key, cell, m["n"],
                        m["acc_strict"], m["acc_normalised"], band, m["n_sent_to_judge"], flip)

    # OBQA A' must always be read against its own B' ceiling, never against 1.0 or against MCQ A:
    # with the fact supplied, open-ended OBQA still only reaches ~0.44, which bounds A'.
    for key, metrics in all_metrics.items():
        if not {"A_prime", "B_prime"} <= set(args.cells):
            continue
        ceiling = metrics["B_prime"]["acc_band_low"]
        a_low = metrics["A_prime"]["acc_band_low"]
        logger.info("  %-6s A' relative to its own B' ceiling: %.3f / %.3f = %.3f",
                    key, a_low, ceiling, (a_low / ceiling) if ceiling else float("nan"))
    logger.info("Runtime : %.1f min", (time.time() - total_started) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
