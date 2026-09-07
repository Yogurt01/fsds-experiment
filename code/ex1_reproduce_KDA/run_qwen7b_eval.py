#!/usr/bin/env python
"""Standalone KDA evaluation for Qwen2.5-7B-Instruct on Google Colab / Kaggle.

This file is **self-contained**: it imports nothing from this repository, builds its own
datasets, and can be dropped into a Colab or Kaggle session on its own. See
`docs/RUN_QWEN2.5_7B_CLOUD_GUIDE.md` for the accompanying step-by-step guide.

    pip install -q -U transformers accelerate bitsandbytes datasets pandas
    python run_qwen7b_eval.py --datasets obqa sciq --out-dir ./kda_out

Scoring protocol
----------------
Each sample is scored twice with a single batched forward pass per phase:

    Phase 1 ("without fact")  question + lettered options      -> P(R^q=1),      r^q
    Phase 2 ("with fact")     gold fact + question + options   -> P(R^{q+f}=1),  r^{q+f}

The chat template is applied with `add_generation_prompt=True` and the logits at the final
position are restricted to the option letters A/B/C/D. All single-token spellings of a
letter are combined with `logsumexp`, then a softmax over the four letters yields a proper
distribution over the options. Batching uses **left padding** so that position -1 is the
true final token of every sequence in the batch.

Metrics (dataset-level, aggregated over samples i)
--------------------------------------------------
    KDA_disc = sum_i (1 - r_i^q) * r_i^{q+f}                  / sum_i (1 - r_i^q)
    KDA_cont = sum_i (1 - P_i(R^q=1)) * P_i(R^{q+f}=1)        / sum_i (1 - P_i(R^q=1))

Outputs
-------
    <out-dir>/kda_qwen2.5_7b_results.json      full per-sample records + summary
    <out-dir>/kda_qwen2.5_7b_records.csv       one row per sample (flat, for pandas/Excel)
    <out-dir>/kda_qwen2.5_7b_summary.csv       one row per dataset
    <out-dir>/kda_qwen2.5_7b_eval.log         execution log
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NOISY = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
         "transformers", "datasets", "fsspec", "accelerate", "bitsandbytes")

LETTERS = ("A", "B", "C", "D")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a careful student taking a multiple-choice exam. "
    "Answer with a single letter and nothing else."
)


# ======================================================================================
# Logging
# ======================================================================================
def setup_logging(log_path: str, verbose: bool = False) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("kda_qwen7b")


# ======================================================================================
# Dataset construction -- KDA input format: passage / question / options / answer_idx
# ======================================================================================
def build_sciq(split: str, seed: int, logger: logging.Logger) -> List[Dict]:
    """allenai/sciq -> KDA format. The `support` paragraph is the target fact."""
    from datasets import load_dataset

    raw = load_dataset("allenai/sciq", split=split)
    rng = random.Random(seed)
    samples, dropped = [], 0
    for row_index, row in enumerate(raw):
        support = (row.get("support") or "").strip()
        if not support:
            dropped += 1
            continue
        correct = (row["correct_answer"] or "").strip()
        options = [correct,
                   (row["distractor1"] or "").strip(),
                   (row["distractor2"] or "").strip(),
                   (row["distractor3"] or "").strip()]
        rng.shuffle(options)
        samples.append({
            "passage": support,
            "question": (row["question"] or "").strip(),
            "options": options,
            "answer_idx": options.index(correct),
            "correct_answer": correct,
            "id": len(samples),
            "dataset_row": row_index,
        })
    logger.info("  SciQ %s: %d samples (%d dropped for empty support)", split, len(samples), dropped)
    return samples


def build_obqa(split: str, seed: int, logger: logging.Logger) -> List[Dict]:
    """allenai/openbookqa -> KDA format. `fact1` (from the `additional` config) is the fact."""
    from datasets import load_dataset

    raw = load_dataset("allenai/openbookqa", "additional", split=split)
    rng = random.Random(seed)
    samples, dropped = [], 0
    for row_index, row in enumerate(raw):
        fact = (row.get("fact1") or "").strip()
        choices = row["choices"]
        texts = [t.strip() for t in choices["text"]]
        labels = list(choices["label"])
        key = row["answerKey"]
        if not fact or key not in labels or len(texts) != 4:
            dropped += 1
            continue
        correct = texts[labels.index(key)]
        options = list(texts)
        rng.shuffle(options)
        samples.append({
            "passage": fact,
            "question": (row["question_stem"] or "").strip(),
            "options": options,
            "answer_idx": options.index(correct),
            "correct_answer": correct,
            "id": len(samples),
            "dataset_row": row_index,
        })
    logger.info("  OpenBookQA %s: %d samples (%d dropped)", split, len(samples), dropped)
    return samples


def load_local_json(path: str, logger: logging.Logger) -> List[Dict]:
    """Load a pre-prepared KDA-format JSON file (e.g. uploaded from this repo)."""
    with open(path, encoding="utf-8") as handle:
        samples = json.load(handle)
    logger.info("  loaded %d samples from %s", len(samples), path)
    return samples


DATASET_BUILDERS = {"sciq": build_sciq, "obqa": build_obqa}
DATASET_NAMES = {"sciq": "allenai/sciq", "obqa": "allenai/openbookqa"}
FACT_FIELDS = {"sciq": "support paragraph", "obqa": "fact1"}


# ======================================================================================
# Prompting
# ======================================================================================
def build_prompt(sample: Dict, with_fact: bool) -> str:
    lines: List[str] = []
    if with_fact:
        lines.append(f"Fact: {(sample.get('passage') or '').strip()}")
        lines.append("")
    lines.append(f"Question: {sample['question'].strip()}")
    for letter, option in zip(LETTERS, sample["options"]):
        lines.append(f"{letter}. {str(option).strip()}")
    lines.append("")
    lines.append("Answer with a single letter (A, B, C, or D).")
    return "\n".join(lines)


def render_chat(tokenizer, user_text: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def letter_token_ids(tokenizer, logger: logging.Logger) -> Dict[str, List[int]]:
    """All single-token spellings of each option letter (bare and space-prefixed)."""
    variants: Dict[str, List[int]] = {}
    for letter in LETTERS:
        ids: List[int] = []
        for spelling in (letter, f" {letter}"):
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        if not ids:
            raise RuntimeError(f"Letter {letter!r} has no single-token spelling in this tokenizer.")
        variants[letter] = ids
    logger.info("  option-letter token ids: %s", variants)
    return variants


# ======================================================================================
# Batched scoring
# ======================================================================================
@torch.no_grad()
def score_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    letter_ids: Dict[str, List[int]],
) -> List[List[float]]:
    """Batched forward pass -> per-prompt probability over the four option letters.

    Uses LEFT padding so that index -1 is the real final token for every row.
    """
    encoded = tokenizer(
        list(prompts), return_tensors="pt", padding=True,
        truncation=True, max_length=4096, add_special_tokens=False,
    ).to(model.device)

    logits = model(**encoded).logits[:, -1, :].float()

    per_letter = torch.stack(
        [torch.logsumexp(logits[:, torch.tensor(letter_ids[l], device=logits.device)], dim=-1)
         for l in LETTERS],
        dim=-1,
    )
    return torch.softmax(per_letter, dim=-1).tolist()


def chunked(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


# ======================================================================================
# Metrics
# ======================================================================================
def compute_metrics(records: Sequence[Dict]) -> Dict:
    n = len(records)
    if n == 0:
        return {"n_samples": 0}

    r_q = [r["r_q"] for r in records]
    r_qf = [r["r_q_plus_f"] for r in records]
    p_q = [r["p_correct_without_fact"] for r in records]
    p_qf = [r["p_correct_with_fact"] for r in records]

    disc_den = sum(1 - r for r in r_q)
    disc_num = sum((1 - a) * b for a, b in zip(r_q, r_qf))
    cont_den = sum(1.0 - p for p in p_q)
    cont_num = sum((1.0 - a) * b for a, b in zip(p_q, p_qf))

    buckets = {"both_correct": 0, "wrong_to_correct": 0, "both_wrong": 0, "correct_to_wrong": 0}
    for a, b in zip(r_q, r_qf):
        if a == 1 and b == 1:
            buckets["both_correct"] += 1
        elif a == 0 and b == 1:
            buckets["wrong_to_correct"] += 1
        elif a == 0 and b == 0:
            buckets["both_wrong"] += 1
        else:
            buckets["correct_to_wrong"] += 1

    return {
        "n_samples": n,
        "kda_disc": (disc_num / disc_den) if disc_den else None,
        "kda_disc_numerator": disc_num,
        "kda_disc_denominator": disc_den,
        "kda_cont": (cont_num / cont_den) if cont_den > 0 else None,
        "kda_cont_numerator": cont_num,
        "kda_cont_denominator": cont_den,
        "accuracy_without_fact": sum(r_q) / n,
        "accuracy_with_fact": sum(r_qf) / n,
        "accuracy_gain": (sum(r_qf) - sum(r_q)) / n,
        "memorization_rate": sum(r_q) / n,
        "mean_p_correct_without_fact": statistics.fmean(p_q),
        "mean_p_correct_with_fact": statistics.fmean(p_qf),
        "buckets": {k: {"count": v, "percentage": round(100.0 * v / n, 4)} for k, v in buckets.items()},
    }


# ======================================================================================
# Model loading
# ======================================================================================
def resolve_precision(requested: str, logger: logging.Logger) -> str:
    """Pick a precision that actually fits the attached accelerator."""
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        logger.warning("No CUDA device found -- falling back to fp32 on CPU (very slow).")
        return "fp32"

    total_gb = sum(
        torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())
    ) / 1024**3
    bf16_ok = torch.cuda.is_bf16_supported()
    logger.info("  detected %.1f GB across %d GPU(s); bf16 supported: %s",
                total_gb, torch.cuda.device_count(), bf16_ok)

    if total_gb >= 30 and bf16_ok:
        return "bf16"            # A100 40GB / H100
    if total_gb >= 28:
        return "fp16"            # 2x T4 (Kaggle), sharded
    if total_gb >= 15:
        return "8bit"            # single T4 16GB (Colab free)
    return "4bit"                # anything smaller


def load_model(model_id: str, precision: str, token: Optional[str], logger: logging.Logger):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading %s at precision '%s'", model_id, precision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    # Left padding is REQUIRED: we read the logits at index -1 of each row.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict = {"device_map": "auto", "token": token}
    if precision == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif precision == "fp16":
        kwargs["dtype"] = torch.float16
    elif precision == "fp32":
        kwargs["dtype"] = torch.float32
        kwargs["device_map"] = None
    elif precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["dtype"] = torch.float16
    elif precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["dtype"] = torch.float16
    else:
        raise ValueError(f"unknown precision {precision!r}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    load_seconds = time.time() - started

    memory = {"precision": precision, "load_seconds": round(load_seconds, 2)}
    if torch.cuda.is_available():
        memory["allocated_after_load_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        memory["reserved_after_load_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
        logger.info("  loaded in %.1fs | %.2f GB allocated", load_seconds,
                    memory["allocated_after_load_gb"])
    else:
        logger.info("  loaded in %.1fs (CPU)", load_seconds)
    return model, tokenizer, memory


# ======================================================================================
# Evaluation driver
# ======================================================================================
def evaluate_dataset(
    model, tokenizer, letter_ids, key: str, samples: List[Dict],
    batch_size: int, logger: logging.Logger,
) -> Dict:
    logger.info("")
    logger.info("=" * 96)
    logger.info("DATASET %s -- %d samples, batch size %d", key.upper(), len(samples), batch_size)
    logger.info("=" * 96)

    prompts_wof = [render_chat(tokenizer, build_prompt(s, False)) for s in samples]
    prompts_wf = [render_chat(tokenizer, build_prompt(s, True)) for s in samples]

    def run_phase(prompts: List[str], label: str) -> Tuple[List[List[float]], float]:
        out: List[List[float]] = []
        started = time.time()
        current = batch_size
        index = 0
        while index < len(prompts):
            batch = prompts[index:index + current]
            try:
                out.extend(score_batch(model, tokenizer, batch, letter_ids))
            except torch.cuda.OutOfMemoryError:
                # Halve the batch and retry; a single sample that still OOMs is fatal.
                torch.cuda.empty_cache()
                if current == 1:
                    raise
                current = max(1, current // 2)
                logger.warning("  OOM -- reducing batch size to %d and retrying", current)
                continue
            index += len(batch)
            if out and (len(out) % max(batch_size * 10, 100) < current or index >= len(prompts)):
                elapsed = time.time() - started
                logger.info("  %s %5d/%5d (%.1f%%) | %.3fs/sample | ETA %.1f min",
                            label, len(out), len(prompts), 100.0 * len(out) / len(prompts),
                            elapsed / max(len(out), 1),
                            (elapsed / max(len(out), 1)) * (len(prompts) - len(out)) / 60.0)
        return out, time.time() - started

    probs_wof, seconds_wof = run_phase(prompts_wof, "phase1 (no fact)  ")
    probs_wf, seconds_wf = run_phase(prompts_wf, "phase2 (with fact)")

    records: List[Dict] = []
    for sample, p_wof, p_wf in zip(samples, probs_wof, probs_wf):
        answer_idx = int(sample["answer_idx"])
        pred_wof = int(max(range(4), key=lambda i: p_wof[i]))
        pred_wf = int(max(range(4), key=lambda i: p_wf[i]))
        records.append({
            "id": sample.get("id"),
            "dataset_row": sample.get("dataset_row"),
            "question": sample["question"],
            "options": sample["options"],
            "answer_idx": answer_idx,
            "correct_answer": sample.get("correct_answer", sample["options"][answer_idx]),
            "passage": sample.get("passage", ""),
            "probs_without_fact": p_wof,
            "probs_with_fact": p_wf,
            "predicted_idx_without_fact": pred_wof,
            "predicted_idx_with_fact": pred_wf,
            "p_correct_without_fact": p_wof[answer_idx],
            "p_correct_with_fact": p_wf[answer_idx],
            "r_q": int(pred_wof == answer_idx),
            "r_q_plus_f": int(pred_wf == answer_idx),
        })

    metrics = compute_metrics(records)
    total_seconds = seconds_wof + seconds_wf
    metrics.update({
        "dataset_key": key,
        "dataset_name": DATASET_NAMES.get(key, key),
        "fact_field": FACT_FIELDS.get(key, "passage"),
        "scoring_seconds": round(total_seconds, 2),
        "seconds_per_sample": round(total_seconds / max(len(samples), 1), 4),
    })

    logger.info("")
    logger.info("  KDA_disc        : %s", _fmt(metrics["kda_disc"]))
    logger.info("  KDA_cont        : %s", _fmt(metrics["kda_cont"]))
    logger.info("  Acc without fact: %.4f", metrics["accuracy_without_fact"])
    logger.info("  Acc with fact   : %.4f (delta %+.4f)", metrics["accuracy_with_fact"], metrics["accuracy_gain"])
    logger.info("  Memorized (r^q=1): %.2f%%", 100.0 * metrics["memorization_rate"])
    logger.info("  buckets         : %s", {k: v["count"] for k, v in metrics["buckets"].items()})
    logger.info("  latency         : %.3fs/sample", metrics["seconds_per_sample"])

    return {"metrics": metrics, "records": records}


def _fmt(value: Optional[float]) -> str:
    return "n/a (zero denominator)" if value is None else f"{value:.4f}"


# ======================================================================================
# Export
# ======================================================================================
def export_csv(out_dir: str, per_dataset: Dict[str, Dict], all_records: Dict[str, List[Dict]],
               logger: logging.Logger) -> Tuple[str, str]:
    """Flat per-sample CSV plus a one-row-per-dataset summary CSV."""
    records_path = os.path.join(out_dir, "kda_qwen2.5_7b_records.csv")
    with open(records_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset", "id", "dataset_row", "question", "correct_answer", "answer_idx",
            "option_A", "option_B", "option_C", "option_D",
            "p_correct_without_fact", "p_correct_with_fact",
            "predicted_idx_without_fact", "predicted_idx_with_fact",
            "r_q", "r_q_plus_f", "bucket", "passage",
        ])
        for key, records in all_records.items():
            for record in records:
                if record["r_q"] == 1 and record["r_q_plus_f"] == 1:
                    bucket = "both_correct"
                elif record["r_q"] == 0 and record["r_q_plus_f"] == 1:
                    bucket = "wrong_to_correct"
                elif record["r_q"] == 0 and record["r_q_plus_f"] == 0:
                    bucket = "both_wrong"
                else:
                    bucket = "correct_to_wrong"
                options = list(record["options"]) + [""] * (4 - len(record["options"]))
                writer.writerow([
                    key, record["id"], record["dataset_row"], record["question"],
                    record["correct_answer"], record["answer_idx"],
                    options[0], options[1], options[2], options[3],
                    f'{record["p_correct_without_fact"]:.6f}',
                    f'{record["p_correct_with_fact"]:.6f}',
                    record["predicted_idx_without_fact"], record["predicted_idx_with_fact"],
                    record["r_q"], record["r_q_plus_f"], bucket, record["passage"],
                ])

    summary_path = os.path.join(out_dir, "kda_qwen2.5_7b_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset", "n_samples", "kda_disc", "kda_cont",
            "accuracy_without_fact", "accuracy_with_fact", "accuracy_gain",
            "memorization_rate", "mean_p_correct_without_fact", "mean_p_correct_with_fact",
            "both_correct", "wrong_to_correct", "both_wrong", "correct_to_wrong",
            "seconds_per_sample",
        ])
        for key, m in per_dataset.items():
            writer.writerow([
                key, m["n_samples"],
                "" if m["kda_disc"] is None else f'{m["kda_disc"]:.6f}',
                "" if m["kda_cont"] is None else f'{m["kda_cont"]:.6f}',
                f'{m["accuracy_without_fact"]:.6f}', f'{m["accuracy_with_fact"]:.6f}',
                f'{m["accuracy_gain"]:.6f}', f'{m["memorization_rate"]:.6f}',
                f'{m["mean_p_correct_without_fact"]:.6f}', f'{m["mean_p_correct_with_fact"]:.6f}',
                m["buckets"]["both_correct"]["count"], m["buckets"]["wrong_to_correct"]["count"],
                m["buckets"]["both_wrong"]["count"], m["buckets"]["correct_to_wrong"]["count"],
                m.get("seconds_per_sample", ""),
            ])

    logger.info("CSV records : %s", records_path)
    logger.info("CSV summary : %s", summary_path)
    return records_path, summary_path


# ======================================================================================
# CLI
# ======================================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone KDA evaluation for Qwen2.5-7B-Instruct on Colab / Kaggle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Hub id or local path.")
    parser.add_argument("--datasets", nargs="+", default=["obqa", "sciq"], choices=["obqa", "sciq"])
    parser.add_argument("--split", default="test", help="Split to score (default: test).")
    parser.add_argument(
        "--local-data", nargs="+", default=None, metavar="KEY=PATH",
        help="Use pre-prepared JSON instead of downloading, e.g. sciq=./sciq_test_full.json",
    )
    parser.add_argument(
        "--precision", default="auto", choices=["auto", "bf16", "fp16", "8bit", "4bit", "fp32"],
        help="auto picks bf16 on A100, fp16 on 2xT4, 8-bit on a single T4, 4-bit below that.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per forward pass.")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N samples.")
    parser.add_argument("--out-dir", default="./kda_out", help="Directory for JSON/CSV/log output.")
    parser.add_argument("--seed", type=int, default=42, help="Option-shuffle seed.")
    parser.add_argument("--token", default=None, help="HF token (else $HF_TOKEN).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(os.path.join(out_dir, "kda_qwen2.5_7b_eval.log"), args.verbose)

    total_started = time.time()
    token = args.token or os.environ.get("HF_TOKEN") or None

    logger.info("=" * 96)
    logger.info("KDA EVALUATION -- %s", args.model_id)
    logger.info("=" * 96)
    logger.info("Python   : %s", platform.python_version())
    logger.info("Platform : %s", platform.platform())
    logger.info("torch    : %s (cuda %s)", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info("GPU %d    : %s (%.1f GB)", i, props.name, props.total_memory / 1024**3)
    logger.info("HF token : %s", "provided" if token else "none (public model)")
    logger.info("Output   : %s", out_dir)
    logger.info("")

    local_map: Dict[str, str] = {}
    for item in args.local_data or []:
        if "=" not in item:
            logger.error("--local-data expects KEY=PATH, got %r", item)
            return 1
        key, path = item.split("=", 1)
        local_map[key.strip()] = path.strip()

    logger.info("Preparing datasets ...")
    datasets: Dict[str, List[Dict]] = {}
    for key in args.datasets:
        if key in local_map:
            samples = load_local_json(local_map[key], logger)
        else:
            samples = DATASET_BUILDERS[key](args.split, args.seed, logger)
        if args.limit:
            samples = samples[: args.limit]
        datasets[key] = samples

    precision = resolve_precision(args.precision, logger)
    model, tokenizer, memory = load_model(args.model_id, precision, token, logger)
    letter_ids = letter_token_ids(tokenizer, logger)

    per_dataset: Dict[str, Dict] = {}
    all_records: Dict[str, List[Dict]] = {}
    for key, samples in datasets.items():
        outcome = evaluate_dataset(model, tokenizer, letter_ids, key, samples, args.batch_size, logger)
        per_dataset[key] = outcome["metrics"]
        all_records[key] = outcome["records"]

    if torch.cuda.is_available():
        memory["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        memory["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

    payload = {
        "summary": {
            "experiment": "kda_evaluation_qwen2.5_7b_instruct",
            "model": args.model_id,
            "precision": precision,
            "batch_size": args.batch_size,
            "split": args.split,
            "seed": args.seed,
            "limit": args.limit or None,
            "scoring_protocol": (
                "single batched forward pass per phase; logits at the final position "
                "restricted to option letters A/B/C/D (left padding), logsumexp over "
                "tokenisation variants, softmax over the four options"
            ),
            "formulas": {
                "kda_disc": "sum_i (1 - r_i^q) * r_i^{q+f} / sum_i (1 - r_i^q)",
                "kda_cont": "sum_i (1 - P_i(R^q=1)) * P_i(R^{q+f}=1) / sum_i (1 - P_i(R^q=1))",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else [],
            },
            "memory": memory,
            "per_dataset": per_dataset,
            "total_runtime_seconds": round(time.time() - total_started, 2),
        },
        "results": all_records,
    }

    json_path = os.path.join(out_dir, "kda_qwen2.5_7b_results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    export_csv(out_dir, per_dataset, all_records, logger)

    logger.info("")
    logger.info("=" * 96)
    logger.info("SUMMARY")
    logger.info("=" * 96)
    logger.info("  %-8s %10s %10s %10s %10s %12s", "dataset", "KDA_disc", "KDA_cont", "Acc_wof", "Acc_wf", "memorized")
    for key, m in per_dataset.items():
        logger.info("  %-8s %10s %10s %10.4f %10.4f %11.2f%%", key,
                    _fmt(m["kda_disc"]), _fmt(m["kda_cont"]),
                    m["accuracy_without_fact"], m["accuracy_with_fact"],
                    100.0 * m["memorization_rate"])
    logger.info("")
    logger.info("JSON     : %s", json_path)
    logger.info("Runtime  : %.1f min", payload["summary"]["total_runtime_seconds"] / 60.0)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
