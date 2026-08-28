"""Run the KDA_tiny ensemble over a prepared multiple-choice dataset.

Defaults reproduce the original run on the 50 sampled SciQ questions
(datasets/sciq/sciq_50.json), but any file in the same KDA input format works --
for example datasets/openbookqa/obqa_test_full.json.

Writes structured results to results/results.json and a full execution log to
results/experiment.log while streaming progress to the console.
"""

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from typing import Dict, List

import torch

from kda_tiny import DEFAULT_MODELS, KDA_SMALL, MODEL_PRESETS, KDATiny
from paths import ensure_parent, resolve

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries log every HTTP request at INFO, which would bury the
# experiment's own records in both the console stream and experiment.log.
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "transformers",
    "datasets",
    "fsspec",
)


def setup_logging(log_path: str, verbose: bool, append: bool) -> logging.Logger:
    """Send DEBUG-level records to `log_path` and INFO-level records to stdout."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):  # avoid duplicate handlers on re-entry
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("kda_experiment")


INDEX_KEYS = ("sciq_index", "obqa_index", "source_index")


def source_index(sample: Dict):
    """Original row number in the upstream dataset, whatever the dataset calls it."""
    for key in INDEX_KEYS:
        if key in sample:
            return sample[key]
    return sample.get("id")


def qualitative_reason(record: Dict) -> str:
    """Explain a question's score from its ensemble probability components."""
    prior = record["mean_p_correct_without_fact"]
    posterior = record["mean_p_correct_with_fact"]
    gain = record["mean_probability_gain"]
    parts: List[str] = []

    if prior < 0.35:
        parts.append(
            f"students cannot guess it without the fact (mean P(R^q=1)={prior:.3f}, "
            "close to the 0.25 chance level)"
        )
    elif prior < 0.60:
        parts.append(f"background knowledge only helps partially (mean P(R^q=1)={prior:.3f})")
    else:
        parts.append(
            f"largely answerable without the passage (mean P(R^q=1)={prior:.3f})"
        )

    if posterior > 0.80:
        parts.append(
            f"the passage answers it directly and near-decisively "
            f"(mean P(R^q+f=1)={posterior:.3f})"
        )
    elif posterior > 0.50:
        parts.append(
            f"the passage points to the right option but students stay unsure "
            f"(mean P(R^q+f=1)={posterior:.3f})"
        )
    else:
        parts.append(
            f"the passage is NOT enough to select the correct option "
            f"(mean P(R^q+f=1)={posterior:.3f}); the target fact does not match the "
            "question or a distractor competes with it"
        )

    parts.append(f"probability gain from the fact = {gain:+.3f}")
    return "; ".join(parts)


def format_progress(done: int, total: int, elapsed: float, mean_kda: float, width: int = 24) -> str:
    """Render a single-line progress bar with a running mean and an ETA."""
    fraction = done / total if total else 1.0
    filled = int(round(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    return (
        f"[{bar}] {fraction * 100:5.1f}%  {done}/{total}  "
        f"running mean KDA={mean_kda:.4f}  elapsed={elapsed:.1f}s  ETA={eta:.1f}s"
    )


def shorten(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compute_model_metrics(records: List[Dict], model_names: List[str]) -> Dict:
    """Per-model and ensemble-level accuracy metrics."""
    n = len(records)
    per_model = {}
    mean_kda_tiny = statistics.fmean(r["kda_score"] for r in records)

    for name in model_names:
        stats = [r["per_model"][name] for r in records]
        accuracy_without_fact = sum(s["is_correct_without_fact"] for s in stats) / n
        accuracy_with_fact = sum(s["is_correct_with_fact"] for s in stats) / n
        per_model[name] = {
            "accuracy_without_fact": accuracy_without_fact,
            "accuracy_with_fact": accuracy_with_fact,
            "accuracy_gain": accuracy_with_fact - accuracy_without_fact,
            "mean_p_correct_without_fact": statistics.fmean(
                s["p_correct_without_fact"] for s in stats
            ),
            "mean_p_correct_with_fact": statistics.fmean(
                s["p_correct_with_fact"] for s in stats
            ),
            "mean_weight": statistics.fmean(s["weight"] for s in stats),
            # With |M| = 1 the weight (1 - P(R^q=1)) cancels between numerator and
            # denominator, so a single model's KDA_cont collapses to P(R^{q+f}=1).
            # Reported per model for comparison; only the ensemble value is a true
            # weighted KDA_cont.
            "mean_kda_tiny": statistics.fmean(s["p_correct_with_fact"] for s in stats),
        }

    ensemble_accuracy_without_fact = statistics.fmean(
        m["accuracy_without_fact"] for m in per_model.values()
    )
    ensemble_accuracy_with_fact = statistics.fmean(
        m["accuracy_with_fact"] for m in per_model.values()
    )

    # Secondary view: pool the probability distributions first, then take the argmax.
    pooled_without_fact = 0
    pooled_with_fact = 0
    for record in records:
        answer_idx = record["answer_idx"]
        n_models = len(model_names)
        avg_without = [
            sum(
                record["per_model"][name]["probabilities_without_fact"][i]
                for name in model_names
            )
            / n_models
            for i in range(4)
        ]
        avg_with = [
            sum(
                record["per_model"][name]["probabilities_with_fact"][i]
                for name in model_names
            )
            / n_models
            for i in range(4)
        ]
        pooled_without_fact += max(range(4), key=avg_without.__getitem__) == answer_idx
        pooled_with_fact += max(range(4), key=avg_with.__getitem__) == answer_idx

    return {
        "per_model": per_model,
        "ensemble_average": {
            "accuracy_without_fact": ensemble_accuracy_without_fact,
            "accuracy_with_fact": ensemble_accuracy_with_fact,
            "accuracy_gain": ensemble_accuracy_with_fact - ensemble_accuracy_without_fact,
            "mean_p_correct_without_fact": statistics.fmean(
                m["mean_p_correct_without_fact"] for m in per_model.values()
            ),
            "mean_p_correct_with_fact": statistics.fmean(
                m["mean_p_correct_with_fact"] for m in per_model.values()
            ),
            # The real KDA_cont: weighted across the whole ensemble, not an average
            # of the per-model degenerate values above.
            "mean_kda_tiny": mean_kda_tiny,
        },
        "pooled_probability_vote": {
            "accuracy_without_fact": pooled_without_fact / n,
            "accuracy_with_fact": pooled_with_fact / n,
            "accuracy_gain": (pooled_with_fact - pooled_without_fact) / n,
            "mean_kda_tiny": mean_kda_tiny,
        },
    }


def log_block(logger: logging.Logger, title: str, records: List[Dict]) -> None:
    logger.info("")
    logger.info(title)
    logger.info("=" * 78)
    for rank, record in enumerate(records, 1):
        logger.info("")
        logger.info(
            "[%d] id=%s (dataset row #%s)  KDA_tiny = %.4f",
            rank,
            record["id"],
            source_index(record),
            record["kda_score"],
        )
        logger.info("    Question : %s", shorten(record["question"]))
        logger.info(
            "    Options  : %s -> correct: '%s' (idx %d)",
            record["options"],
            record["correct_answer"],
            record["answer_idx"],
        )
        logger.info("    Passage  : %s", shorten(record["passage"]))
        for name, stats in record["per_model"].items():
            logger.info(
                "    Model    : %s | P(R^q=1)=%.3f -> P(R^q+f=1)=%.3f | weight=%.3f",
                name,
                stats["p_correct_without_fact"],
                stats["p_correct_with_fact"],
                stats["weight"],
            )
        logger.info("    Reason   : %s", record["reason"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the multi-model KDA_tiny metric on a prepared MCQ dataset."
    )
    parser.add_argument(
        "--data",
        default="datasets/sciq/sciq_50.json",
        help="Formatted samples in the KDA input format (relative to the project root).",
    )
    parser.add_argument(
        "--out", default="results/results.json", help="Structured output path."
    )
    parser.add_argument(
        "--log-file", default="results/experiment.log", help="Execution log path."
    )
    parser.add_argument(
        "--dataset-name",
        default="allenai/sciq",
        help="Dataset identifier recorded in the results summary.",
    )
    parser.add_argument(
        "--split", default="test", help="Split name recorded in the results summary."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Ensemble members; at least two are required for a non-degenerate KDA_cont. "
            "Accepts the preset name KDA_SMALL to expand to the paper's official "
            "four-model KDA_small suite."
        ),
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected if unset).")
    parser.add_argument(
        "--strategy",
        choices=("auto", "resident", "sequential"),
        default="auto",
        help=(
            "'resident' keeps every model in VRAM at once; 'sequential' sweeps the whole "
            "dataset one model at a time, unloading between passes so peak VRAM is set by "
            "the largest single checkpoint. 'auto' (default) picks sequential for three or "
            "more models."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Stream DEBUG logs to console.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help=(
            "Emit an INFO progress bar every N samples. 0 (default) selects automatically: "
            "one INFO line per sample for datasets of 100 samples or fewer, otherwise a "
            "progress bar every 5%% of the run. Per-sample records always reach the log "
            "file at DEBUG level regardless of this setting."
        ),
    )
    parser.add_argument("--append-log", action="store_true", help="Append to the log file.")
    parser.add_argument(
        "--allow-single-model",
        action="store_true",
        help="Permit |M| = 1 to reproduce the degenerate baseline on purpose.",
    )
    args = parser.parse_args()

    # Expand preset names (e.g. --models KDA_SMALL) into explicit checkpoint lists.
    resolved_models: List[str] = []
    for entry in args.models:
        preset = MODEL_PRESETS.get(entry.upper())
        if preset:
            resolved_models.extend(preset)
        else:
            resolved_models.append(entry)
    args.models = resolved_models

    # Anchor every path to the project root so the script is CWD-independent.
    args.data = resolve(args.data)
    args.out = ensure_parent(resolve(args.out))
    args.log_file = ensure_parent(resolve(args.log_file))

    logger = setup_logging(args.log_file, args.verbose, args.append_log)
    run_started = time.time()

    logger.info("=" * 78)
    logger.info("KDA_tiny ensemble experiment on %s (%s)", args.dataset_name, args.split)
    logger.info("=" * 78)
    logger.info("Python        : %s", platform.python_version())
    logger.info("Platform      : %s", platform.platform())
    logger.info("torch         : %s (CUDA available: %s)", torch.__version__, torch.cuda.is_available())
    logger.info("Ensemble |M|  : %d", len(args.models))
    for name in args.models:
        logger.info("  - %s", name)
    logger.info("Data file     : %s", args.data)
    logger.info("Results file  : %s", args.out)
    logger.info("Log file      : %s", args.log_file)

    try:
        with open(args.data, encoding="utf-8") as handle:
            samples = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read the dataset file '%s'", args.data)
        raise
    n_samples = len(samples)
    logger.info("Loaded %d questions from %s", n_samples, args.data)

    try:
        scorer = KDATiny(
            model_names=args.models,
            device=args.device,
            logger=logging.getLogger("kda_tiny"),
            allow_single_model=args.allow_single_model,
            strategy=args.strategy,
        )
    except Exception:
        logger.exception("Ensemble initialisation failed")
        raise

    # Console cadence: per-sample lines are readable for small sets, but a full split
    # would emit hundreds of them, so fall back to a periodic progress bar.
    per_sample_console = len(samples) <= 100 if args.progress_every == 0 else False
    if args.progress_every > 0:
        progress_every = args.progress_every
    else:
        progress_every = max(1, len(samples) // 20)
    if not per_sample_console:
        logger.info(
            "Console cadence: progress bar every %d samples "
            "(per-sample records go to %s at DEBUG level)",
            progress_every,
            args.log_file,
        )

    records: List[Dict] = []
    failures: List[Dict] = []
    running_kda_total = 0.0
    scoring_started = time.time()

    if scorer.strategy == "sequential":
        # One model resident at a time: each sweeps the dataset, then its weights are
        # released before the next is loaded. Progress is reported per model pass.
        last_reported = {"key": None}

        def report(model_name, model_position, n_models, position, n_samples):
            if position % progress_every and position != n_samples:
                return
            key = (model_position, position)
            if last_reported["key"] == key:
                return
            last_reported["key"] = key
            fraction = ((model_position - 1) * n_samples + position) / (n_models * n_samples)
            elapsed = time.time() - scoring_started
            eta = elapsed / fraction - elapsed if fraction > 0 else 0.0
            width = 24
            filled = int(round(fraction * width))
            logger.info(
                "[%s] %5.1f%%  model %d/%d (%s)  %d/%d samples  elapsed=%.1fs  ETA=%.1fs",
                "#" * filled + "-" * (width - filled),
                fraction * 100,
                model_position,
                n_models,
                model_name.split("/")[-1],
                position,
                n_samples,
                elapsed,
                eta,
            )

        try:
            results = scorer.score_dataset(samples, progress_callback=report)
        except Exception:
            logger.exception("Sequential scoring failed")
            raise

        for sample, result in zip(samples, results):
            record = {**sample, **result}
            record["reason"] = qualitative_reason(record)
            records.append(record)
            logger.debug(
                "id=%s KDA=%.4f | mean P(R^q=1)=%.3f -> P(R^q+f=1)=%.3f",
                sample["id"],
                result["kda_score"],
                result["mean_p_correct_without_fact"],
                result["mean_p_correct_with_fact"],
            )
            for name, stats in result["per_model"].items():
                logger.debug(
                    "        %s | without_fact=%s | with_fact=%s | "
                    "P(R^q=1)=%.4f P(R^q+f=1)=%.4f weight=%.4f | probs_wof=%s probs_wf=%s",
                    name,
                    stats["predicted_idx_without_fact"],
                    stats["predicted_idx_with_fact"],
                    stats["p_correct_without_fact"],
                    stats["p_correct_with_fact"],
                    stats["weight"],
                    [round(p, 4) for p in stats["probabilities_without_fact"]],
                    [round(p, 4) for p in stats["probabilities_with_fact"]],
                )
        samples = []  # skip the resident loop below

    for position, sample in enumerate(samples, start=1):
        sample_started = time.time()
        try:
            result = scorer.score(
                sample["passage"],
                sample["question"],
                sample["options"],
                sample["answer_idx"],
            )
        except Exception as error:  # keep going so one bad sample cannot kill the run
            logger.exception(
                "Scoring failed for sample id=%s (dataset row #%s)",
                sample.get("id"),
                source_index(sample),
            )
            failures.append(
                {"id": sample.get("id"), "source_index": source_index(sample), "error": repr(error)}
            )
            continue

        record = {**sample, **result}
        record["reason"] = qualitative_reason(record)
        record["scoring_seconds"] = round(time.time() - sample_started, 4)
        records.append(record)

        running_kda_total += result["kda_score"]

        sample_line = (
            "[%d/%d] id=%s KDA=%.4f | mean P(R^q=1)=%.3f -> P(R^q+f=1)=%.3f | %.3fs"
        )
        sample_args = (
            position,
            len(samples),
            sample["id"],
            result["kda_score"],
            result["mean_p_correct_without_fact"],
            result["mean_p_correct_with_fact"],
            record["scoring_seconds"],
        )
        if per_sample_console:
            logger.info(sample_line, *sample_args)
        else:
            logger.debug(sample_line, *sample_args)
            if position % progress_every == 0 or position == len(samples):
                logger.info(
                    "%s",
                    format_progress(
                        position,
                        len(samples),
                        time.time() - scoring_started,
                        running_kda_total / len(records),
                    ),
                )
        for name, stats in result["per_model"].items():
            logger.debug(
                "        %s | without_fact=%s | with_fact=%s | "
                "P(R^q=1)=%.4f P(R^q+f=1)=%.4f weight=%.4f | probs_wof=%s probs_wf=%s",
                name,
                stats["predicted_idx_without_fact"],
                stats["predicted_idx_with_fact"],
                stats["p_correct_without_fact"],
                stats["p_correct_with_fact"],
                stats["weight"],
                [round(p, 4) for p in stats["probabilities_without_fact"]],
                [round(p, 4) for p in stats["probabilities_with_fact"]],
            )

    scoring_seconds = time.time() - scoring_started
    if not records:
        logger.error("No sample was scored successfully; aborting.")
        sys.exit(1)
    if failures:
        logger.warning("%d sample(s) failed and were skipped", len(failures))

    scores = [r["kda_score"] for r in records]
    mean_kda = statistics.fmean(scores)
    metrics = compute_model_metrics(records, list(args.models))
    ensemble = metrics["ensemble_average"]

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "n_samples": len(records),
        "n_failed_samples": len(failures),
        "models": list(args.models),
        "ensemble_size": len(args.models),
        "total_parameters": scorer.total_parameters,
        "device": scorer.device,
        "mean_kda_tiny": mean_kda,
        "median_kda_tiny": statistics.median(scores),
        "std_kda_tiny": statistics.pstdev(scores),
        "min_kda_tiny": min(scores),
        "max_kda_tiny": max(scores),
        "accuracy_without_fact": ensemble["accuracy_without_fact"],
        "accuracy_with_fact": ensemble["accuracy_with_fact"],
        "accuracy_gain": ensemble["accuracy_gain"],
        "metrics": metrics,
        "n_zero_denominator": sum(r["zero_denominator"] for r in records),
        "zero_denominator_sample_ids": [
            {"id": r["id"], "source_index": source_index(r)}
            for r in records
            if r["zero_denominator"]
        ],
        "scoring_seconds": round(scoring_seconds, 2),
        "total_runtime_seconds": round(time.time() - run_started, 2),
        "formula": (
            "KDA_cont(q) = sum_m (1 - P_m(R^q=1)) * P_m(R^{q+f}=1) "
            "/ sum_m (1 - P_m(R^q=1)); 0 if the denominator is 0"
        ),
        "failures": failures,
    }

    try:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "results": records}, handle, ensure_ascii=False, indent=2)
    except OSError:
        logger.exception("Failed to write results to '%s'", args.out)
        raise

    ranked = sorted(records, key=lambda r: r["kda_score"], reverse=True)

    logger.info("")
    logger.info("=" * 78)
    logger.info(
        "KDA_tiny RESULTS OVER %d %s %s QUESTIONS",
        len(records),
        args.dataset_name,
        args.split.upper(),
    )
    logger.info("=" * 78)
    logger.info(
        "Ensemble            : %d models, %.1fM parameters (%s)",
        len(args.models),
        scorer.total_parameters / 1e6,
        scorer.device,
    )
    logger.info("Samples scored      : %d (%d failed)", len(records), len(failures))
    logger.info("")
    logger.info("KDA_cont score distribution:")
    logger.info("  %-10s %-10s %-10s %-10s %-10s", "mean", "median", "std", "min", "max")
    logger.info(
        "  %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f",
        mean_kda,
        summary["median_kda_tiny"],
        summary["std_kda_tiny"],
        summary["min_kda_tiny"],
        summary["max_kda_tiny"],
    )
    logger.info("")
    logger.info(
        "Zero-denominator safety check : %d question(s) where sum(1 - P_m(R^q=1)) = 0",
        summary["n_zero_denominator"],
    )
    if summary["zero_denominator_sample_ids"]:
        for entry in summary["zero_denominator_sample_ids"]:
            logger.warning(
                "  KDA forced to 0 for sample id=%s (dataset row #%s)",
                entry["id"],
                entry["source_index"],
            )
    else:
        logger.info("  No sample triggered the guard; every KDA score is a true ratio.")
    logger.info("")
    logger.info("Accuracy and KDA per model:")
    logger.info(
        "  %-45s %10s %10s %10s %12s",
        "model", "Acc_wof", "Acc_wf", "delta_Acc", "mean_KDA",
    )
    logger.info("  %s", "-" * 90)
    for name, model_metrics in metrics["per_model"].items():
        logger.info(
            "  %-45s %9.2f%% %9.2f%% %+9.2f%% %12.4f",
            name,
            model_metrics["accuracy_without_fact"] * 100,
            model_metrics["accuracy_with_fact"] * 100,
            model_metrics["accuracy_gain"] * 100,
            model_metrics["mean_kda_tiny"],
        )
    logger.info("  %s", "-" * 90)
    logger.info(
        "  %-45s %9.2f%% %9.2f%% %+9.2f%% %12.4f",
        "ENSEMBLE AVERAGE",
        ensemble["accuracy_without_fact"] * 100,
        ensemble["accuracy_with_fact"] * 100,
        ensemble["accuracy_gain"] * 100,
        ensemble["mean_kda_tiny"],
    )
    pooled = metrics["pooled_probability_vote"]
    logger.info(
        "  %-45s %9.2f%% %9.2f%% %+9.2f%% %12.4f",
        "ENSEMBLE pooled prob. vote",
        pooled["accuracy_without_fact"] * 100,
        pooled["accuracy_with_fact"] * 100,
        pooled["accuracy_gain"] * 100,
        pooled["mean_kda_tiny"],
    )
    logger.info("")
    logger.info(
        "  mean_KDA per model is the degenerate |M|=1 case (KDA_cont collapses to "
        "P(R^{q+f}=1));"
    )
    logger.info(
        "  the ensemble rows carry the true weighted KDA_cont over all %d models.",
        len(args.models),
    )
    logger.info("")
    logger.info(
        "Mean P(R^q=1) / P(R^q+f=1) : %.4f / %.4f",
        ensemble["mean_p_correct_without_fact"],
        ensemble["mean_p_correct_with_fact"],
    )
    logger.info("Scoring time        : %.2fs (%.3fs per question)", scoring_seconds, scoring_seconds / len(records))
    logger.info("Total runtime       : %.2fs", summary["total_runtime_seconds"])

    log_block(logger, "TOP 3 QUESTIONS BY HIGHEST KDA", ranked[:3])
    log_block(logger, "TOP 3 QUESTIONS BY LOWEST KDA", list(reversed(ranked[-3:])))

    logger.info("")
    logger.info("Results written to : %s", args.out)
    logger.info("Execution log      : %s", args.log_file)


if __name__ == "__main__":
    main()
