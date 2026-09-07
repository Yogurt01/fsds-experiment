"""Reproduce the KDA evaluation with an instruction-tuned Qwen LLM as the simulated student.

This is the modern-LLM counterpart to `run_experiment.py`. Where that script scores an
ensemble of 355M-parameter RACE-tuned encoders, this one runs a single multi-billion
parameter instruct model under 4-bit NF4 quantisation so it fits a 4GB laptop GPU.

Scoring protocol
----------------
Each sample is scored twice, exactly mirroring the KDA construction:

    Phase 1 ("without fact")  prompt = question + lettered options
                              -> P(R^q = 1)      and  r^q      in {0, 1}
    Phase 2 ("with fact")     prompt = gold fact + question + lettered options
                              -> P(R^{q+f} = 1)  and  r^{q+f}  in {0, 1}

Both phases use a **single forward pass**. The chat template is applied with
`add_generation_prompt=True`, and the logits at the final position are restricted to the
option letters (A/B/C/D). A softmax over just those letters yields a proper distribution
over the options, from which:

    P(R = 1) = p[correct_letter]
    r        = 1 if argmax(p) == correct_letter else 0

This is the standard MMLU-harness style of multiple-choice scoring. It is not the same as
per-option sequence likelihood (which needs one pass per option and is length-biased); the
letter-logit form keeps every option on an equal footing and costs 4x less compute.

Because a letter may tokenise differently with and without a leading space, all
single-token spellings of a letter are collected and combined with `logsumexp` before the
softmax, so the probability reflects the model's total mass on "answer X".

Metrics
-------
Dataset-level aggregates over samples i (note: this is the *sample-level* aggregation
requested for the LLM study; `run_experiment.py` aggregates over models k per question):

    KDA_disc = sum_i (1 - r_i^q) * r_i^{q+f}  /  sum_i (1 - r_i^q)
    KDA_cont = sum_i P(R_i^q = 0) * P(R_i^{q+f} = 1)  /  sum_i P(R_i^q = 0)

with P(R^q = 0) = 1 - P(R^q = 1). Accuracy before/after the fact prompt and the
memorisation rate (the share of samples with r^q = 1) are logged alongside.

Usage:
    uv run --active python code/ex1_reproduce_KDA/kda_qwen_eval.py
    uv run --active python code/ex1_reproduce_KDA/kda_qwen_eval.py --limit 20 --datasets sciq
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
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------------------
# Cross-stage imports. This script lives in `code/<stage>/`, so `code/` itself is put on
# `sys.path`; `utils.paths` and `ex1_reproduce_KDA.kda_tiny` then resolve no matter which
# directory the script is launched from. The *project root* is deliberately NOT added --
# it contains a `datasets/` folder that would shadow the HuggingFace `datasets` package.
# --------------------------------------------------------------------------------------
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import MODELS_DIR, PROJECT_ROOT, ensure_parent, resolve

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

NOISY_LOGGERS = (
    "httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
    "transformers", "datasets", "fsspec", "accelerate", "bitsandbytes",
)

LETTERS = ("A", "B", "C", "D")

# Datasets are declared here so `--datasets sciq obqa` stays a short flag.
DATASET_SPECS = {
    "sciq": {
        "path": "datasets/sciq/sciq_test_full.json",
        "dataset_name": "allenai/sciq",
        "split": "test",
        "fact_field": "support paragraph",
    },
    "obqa": {
        "path": "datasets/openbookqa/obqa_test_full.json",
        "dataset_name": "allenai/openbookqa",
        "split": "test",
        "fact_field": "fact1",
    },
}

SYSTEM_PROMPT = (
    "You are a careful student taking a multiple-choice exam. "
    "Answer with a single letter and nothing else."
)


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

    return logging.getLogger("kda_qwen")


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------
def build_prompt(sample: Dict, with_fact: bool) -> str:
    """Render the user turn for one sample, with or without the gold fact prepended."""
    lines: List[str] = []
    if with_fact:
        fact = (sample.get("passage") or "").strip()
        lines.append(f"Fact: {fact}")
        lines.append("")
    lines.append(f"Question: {sample['question'].strip()}")
    for letter, option in zip(LETTERS, sample["options"]):
        lines.append(f"{letter}. {str(option).strip()}")
    lines.append("")
    lines.append("Answer with a single letter (A, B, C, or D).")
    return "\n".join(lines)


def render_chat(tokenizer, user_text: str) -> str:
    """Apply the model's chat template and open the assistant turn."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------------------
# Letter token bookkeeping
# --------------------------------------------------------------------------------------
def letter_token_ids(tokenizer, logger: logging.Logger) -> Dict[str, List[int]]:
    """Every single-token spelling of each option letter (bare, space-prefixed, ...).

    A letter can tokenise differently depending on what precedes it. Collecting all
    single-token variants and combining them with logsumexp gives the model's total
    probability mass on "the answer is X" rather than on one arbitrary spelling.
    """
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


@torch.no_grad()
def score_prompt(
    model,
    tokenizer,
    prompt_text: str,
    letter_ids: Dict[str, List[int]],
    device: str,
) -> Tuple[List[float], int]:
    """One forward pass -> normalised probability over the four option letters.

    Returns (probabilities in A/B/C/D order, prompt token count).
    """
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    n_tokens = int(encoded["input_ids"].shape[-1])

    logits = model(**encoded).logits[0, -1, :].float()

    # Total mass per letter across its tokenisation variants, then renormalise over
    # the four options only.
    per_letter = torch.stack(
        [torch.logsumexp(logits[torch.tensor(ids, device=logits.device)], dim=0)
         for ids in (letter_ids[l] for l in LETTERS)]
    )
    probs = torch.softmax(per_letter, dim=0)
    return probs.tolist(), n_tokens


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def compute_metrics(records: Sequence[Dict]) -> Dict:
    """Dataset-level KDA_disc / KDA_cont plus the accuracy and memorisation breakdown."""
    n = len(records)
    if n == 0:
        return {"n_samples": 0}

    r_q = [rec["r_q"] for rec in records]
    r_qf = [rec["r_q_plus_f"] for rec in records]
    p_q = [rec["p_correct_without_fact"] for rec in records]
    p_qf = [rec["p_correct_with_fact"] for rec in records]

    # KDA_disc: among samples answered WRONG without the fact, how many flip to correct?
    disc_denominator = sum(1 - r for r in r_q)
    disc_numerator = sum((1 - rq) * rqf for rq, rqf in zip(r_q, r_qf))
    kda_disc = (disc_numerator / disc_denominator) if disc_denominator else None

    # KDA_cont: the same quantity in probability space, weighted by ignorance mass.
    cont_denominator = sum(1.0 - p for p in p_q)
    cont_numerator = sum((1.0 - pq) * pqf for pq, pqf in zip(p_q, p_qf))
    kda_cont = (cont_numerator / cont_denominator) if cont_denominator > 0 else None

    # The four-bucket contingency table, same vocabulary as categorize_kda_results.py.
    buckets = {"both_correct": 0, "wrong_to_correct": 0, "both_wrong": 0, "correct_to_wrong": 0}
    for rq, rqf in zip(r_q, r_qf):
        if rq == 1 and rqf == 1:
            buckets["both_correct"] += 1
        elif rq == 0 and rqf == 1:
            buckets["wrong_to_correct"] += 1
        elif rq == 0 and rqf == 0:
            buckets["both_wrong"] += 1
        else:
            buckets["correct_to_wrong"] += 1

    acc_wof = sum(r_q) / n
    acc_wf = sum(r_qf) / n
    latencies = [rec["latency_seconds"] for rec in records]

    return {
        "n_samples": n,
        "kda_disc": kda_disc,
        "kda_disc_numerator": disc_numerator,
        "kda_disc_denominator": disc_denominator,
        "kda_cont": kda_cont,
        "kda_cont_numerator": cont_numerator,
        "kda_cont_denominator": cont_denominator,
        "accuracy_without_fact": acc_wof,
        "accuracy_with_fact": acc_wf,
        "accuracy_gain": acc_wf - acc_wof,
        "memorization_rate": acc_wof,  # share of samples with r^q = 1, by definition
        "mean_p_correct_without_fact": statistics.fmean(p_q),
        "mean_p_correct_with_fact": statistics.fmean(p_qf),
        "mean_probability_gain": statistics.fmean(p_qf) - statistics.fmean(p_q),
        "buckets": {
            name: {"count": count, "percentage": round(100.0 * count / n, 4)}
            for name, count in buckets.items()
        },
        "latency_seconds_per_sample": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
    }


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------
def load_model(model_path: str, args, logger: logging.Logger):
    """Load the tokenizer and the 4-bit quantised model, logging the VRAM footprint."""
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
        "Quantisation : 4-bit NF4, compute dtype float16, double_quant=%s",
        args.double_quant,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    else:
        before = 0

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
        weights_bytes = torch.cuda.memory_allocated() - before
        memory.update(
            {
                "weights_vram_gb": round(weights_bytes / 1024**3, 3),
                "allocated_after_load_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "reserved_after_load_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "device_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
                ),
            }
        )
        logger.info(
            "VRAM after load: %.2f GB allocated / %.2f GB reserved of %.2f GB total",
            memory["allocated_after_load_gb"],
            memory["reserved_after_load_gb"],
            memory["device_total_gb"],
        )

    return model, tokenizer, memory


# --------------------------------------------------------------------------------------
# Evaluation driver
# --------------------------------------------------------------------------------------
def evaluate_dataset(
    model,
    tokenizer,
    letter_ids: Dict[str, List[int]],
    key: str,
    spec: Dict,
    args,
    logger: logging.Logger,
) -> Dict:
    """Score one dataset end to end and return its records + metrics."""
    data_path = resolve(spec["path"])
    with open(data_path, encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.limit:
        samples = samples[: args.limit]

    logger.info("")
    logger.info("=" * 100)
    logger.info("DATASET %s (%s, %s) -- %d samples", key.upper(), spec["dataset_name"], spec["split"], len(samples))
    logger.info("  file      : %s", data_path)
    logger.info("  target fact: %s", spec["fact_field"])
    logger.info("=" * 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    records: List[Dict] = []
    progress_every = args.progress_every or max(1, len(samples) // 20)
    started = time.time()

    for index, sample in enumerate(samples):
        answer_idx = int(sample["answer_idx"])
        sample_started = time.time()

        probs_wof, ntok_wof = score_prompt(
            model, tokenizer, render_chat(tokenizer, build_prompt(sample, with_fact=False)),
            letter_ids, device,
        )
        probs_wf, ntok_wf = score_prompt(
            model, tokenizer, render_chat(tokenizer, build_prompt(sample, with_fact=True)),
            letter_ids, device,
        )
        latency = time.time() - sample_started

        pred_wof = int(max(range(4), key=lambda i: probs_wof[i]))
        pred_wf = int(max(range(4), key=lambda i: probs_wf[i]))

        record = {
            "id": sample.get("id", index),
            "dataset_row": sample.get("dataset_row"),
            "question": sample["question"],
            "options": sample["options"],
            "answer_idx": answer_idx,
            "correct_answer": sample.get("correct_answer", sample["options"][answer_idx]),
            "passage": sample.get("passage", ""),
            "probs_without_fact": probs_wof,
            "probs_with_fact": probs_wf,
            "predicted_idx_without_fact": pred_wof,
            "predicted_idx_with_fact": pred_wf,
            "p_correct_without_fact": probs_wof[answer_idx],
            "p_correct_with_fact": probs_wf[answer_idx],
            "r_q": int(pred_wof == answer_idx),
            "r_q_plus_f": int(pred_wf == answer_idx),
            "prompt_tokens_without_fact": ntok_wof,
            "prompt_tokens_with_fact": ntok_wf,
            "latency_seconds": latency,
        }
        records.append(record)

        logger.debug(
            "id=%s r^q=%d r^{q+f}=%d P^q=%.4f P^{q+f}=%.4f probs_wof=%s probs_wf=%s",
            record["id"], record["r_q"], record["r_q_plus_f"],
            record["p_correct_without_fact"], record["p_correct_with_fact"],
            [round(p, 4) for p in probs_wof], [round(p, 4) for p in probs_wf],
        )

        if (index + 1) % progress_every == 0 or (index + 1) == len(samples):
            done = index + 1
            elapsed = time.time() - started
            running_acc_wof = sum(r["r_q"] for r in records) / done
            running_acc_wf = sum(r["r_q_plus_f"] for r in records) / done
            logger.info(
                "  [%4d/%4d] %5.1f%% | Acc_wof %.3f -> Acc_wf %.3f | %.2fs/sample | ETA %.1f min",
                done, len(samples), 100.0 * done / len(samples),
                running_acc_wof, running_acc_wf,
                elapsed / done, (elapsed / done) * (len(samples) - done) / 60.0,
            )

    scoring_seconds = time.time() - started
    metrics = compute_metrics(records)
    metrics.update(
        {
            "dataset_key": key,
            "dataset_name": spec["dataset_name"],
            "split": spec["split"],
            "data_file": data_path,
            "fact_field": spec["fact_field"],
            "scoring_seconds": round(scoring_seconds, 2),
        }
    )

    logger.info("")
    logger.info("  %s RESULTS", key.upper())
    logger.info("  " + "-" * 96)
    logger.info("    KDA_disc            : %s", _fmt(metrics["kda_disc"]))
    logger.info("    KDA_cont            : %s", _fmt(metrics["kda_cont"]))
    logger.info("    Acc without fact    : %.4f", metrics["accuracy_without_fact"])
    logger.info("    Acc with fact       : %.4f  (delta %+.4f)", metrics["accuracy_with_fact"], metrics["accuracy_gain"])
    logger.info("    Memorization (r^q=1): %.2f%%", 100.0 * metrics["memorization_rate"])
    logger.info("    mean P(R^q=1)       : %.4f -> P(R^{q+f}=1) %.4f", metrics["mean_p_correct_without_fact"], metrics["mean_p_correct_with_fact"])
    logger.info("    buckets             : %s", {k: v["count"] for k, v in metrics["buckets"].items()})
    logger.info("    latency             : %.3fs/sample (median %.3fs)", metrics["latency_seconds_per_sample"]["mean"], metrics["latency_seconds_per_sample"]["median"])
    logger.info("  " + "-" * 96)

    return {"metrics": metrics, "records": records}


def _fmt(value: Optional[float]) -> str:
    return "n/a (zero denominator)" if value is None else f"{value:.4f}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KDA evaluation with a 4-bit quantised Qwen instruct model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run --active python code/ex1_reproduce_KDA/kda_qwen_eval.py\n"
            "  uv run --active python code/ex1_reproduce_KDA/kda_qwen_eval.py --limit 20 --datasets sciq\n"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(MODELS_DIR, "Qwen3-4B-Instruct-2507"),
        help="Local model directory (default: models/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["obqa", "sciq"], choices=sorted(DATASET_SPECS),
        help="Which datasets to score.",
    )
    parser.add_argument(
        "--out", default="results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json", help="Structured output path."
    )
    parser.add_argument(
        "--log-file", default="results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_eval.log", help="Execution log path."
    )
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N samples.")
    parser.add_argument(
        "--double-quant", action="store_true",
        help="Enable bnb nested (double) quantisation; saves ~0.1GB VRAM at a small accuracy cost.",
    )
    parser.add_argument("--progress-every", type=int, default=0, help="Progress cadence (0 = auto, ~5%%).")
    parser.add_argument("--verbose", action="store_true", help="Stream DEBUG logs to console.")
    parser.add_argument("--append-log", action="store_true", help="Append to the log file.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
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
    logger.info("KDA EVALUATION -- Qwen instruct model as simulated student")
    logger.info("=" * 100)
    logger.info("Workspace   : %s", PROJECT_ROOT)
    logger.info("Model path  : %s", model_path)
    logger.info("Python      : %s", platform.python_version())
    logger.info("Platform    : %s", platform.platform())
    logger.info("torch       : %s (cuda %s)", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU         : %s (%.2f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1024**3)
    try:
        import bitsandbytes
        logger.info("bitsandbytes: %s", bitsandbytes.__version__)
    except Exception:  # pragma: no cover - only reached if bnb is broken
        logger.warning("bitsandbytes not importable")
    logger.info("Datasets    : %s", ", ".join(args.datasets))
    logger.info("")

    model, tokenizer, memory = load_model(model_path, args, logger)
    logger.info("Resolving option-letter token ids ...")
    letter_ids = letter_token_ids(tokenizer, logger)
    logger.info("  %s", {l: letter_ids[l] for l in LETTERS})

    per_dataset: Dict[str, Dict] = {}
    all_records: Dict[str, List[Dict]] = {}
    for key in args.datasets:
        outcome = evaluate_dataset(model, tokenizer, letter_ids, key, DATASET_SPECS[key], args, logger)
        per_dataset[key] = outcome["metrics"]
        all_records[key] = outcome["records"]

    if torch.cuda.is_available():
        memory["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        memory["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        logger.info("")
        logger.info("Peak VRAM   : %.2f GB allocated / %.2f GB reserved",
                    memory["peak_allocated_gb"], memory["peak_reserved_gb"])

    payload = {
        "summary": {
            "experiment": "kda_evaluation_qwen3_4b_instruct_4bit",
            "model": os.path.basename(model_path),
            "model_path": model_path,
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": args.double_quant,
            },
            "scoring_protocol": (
                "single forward pass per phase; logits at the final position restricted to "
                "the option letters A/B/C/D, logsumexp over tokenisation variants, softmax "
                "over the four options"
            ),
            "formulas": {
                "kda_disc": "sum_i (1 - r_i^q) * r_i^{q+f} / sum_i (1 - r_i^q)",
                "kda_cont": "sum_i (1 - P_i(R^q=1)) * P_i(R^{q+f}=1) / sum_i (1 - P_i(R^q=1))",
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
            "per_dataset": per_dataset,
            "total_runtime_seconds": round(time.time() - total_started, 2),
        },
        "results": all_records,
    }

    out_path = ensure_parent(resolve(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 100)
    logger.info("SUMMARY")
    logger.info("=" * 100)
    logger.info("  %-8s %10s %10s %10s %10s %12s", "dataset", "KDA_disc", "KDA_cont", "Acc_wof", "Acc_wf", "memorized")
    for key, metrics in per_dataset.items():
        logger.info(
            "  %-8s %10s %10s %10.4f %10.4f %11.2f%%",
            key, _fmt(metrics["kda_disc"]), _fmt(metrics["kda_cont"]),
            metrics["accuracy_without_fact"], metrics["accuracy_with_fact"],
            100.0 * metrics["memorization_rate"],
        )
    logger.info("")
    logger.info("Results : %s", out_path)
    logger.info("Log     : %s", resolve(args.log_file))
    logger.info("Runtime : %.1f min", payload["summary"]["total_runtime_seconds"] / 60.0)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
