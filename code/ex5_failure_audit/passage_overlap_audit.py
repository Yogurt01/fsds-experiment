"""Benchmark-wide corpus-defect screen: does each question's passage actually match it?

Origin
------
The Setting-C ineligibility screen (`bucket_diagnostics.py`) surfaced 7 SciQ items that the
primary model fails in both settings. Reading them showed 5 were not hard questions at all -- the
support passage was about an unrelated topic (a reptile-skin question paired with a passage on the
cell cycle). Those 5 shared *literally no content word* with their own passage, against a split
mean of 0.838, which suggested a cheap detector for a corpus defect that no metric can correct.

This module runs that detector over **every item in both test splits**, independent of any model's
performance, so the defect rate is a property of the benchmark rather than of the `both_wrong`
bucket.

Metric
------
    overlap(q) = |content_words(question) & content_words(passage)| / |content_words(question)|

Content words are lower-cased alphanumeric tokens of length > 1 with stopwords removed, using the
same `content_tokens` the pool rebuild uses, so the number is comparable across the audit.

A caution that turns out to dominate the analysis
-------------------------------------------------
The metric does **not** mean the same thing on the two datasets, for the same structural reason
the counterfactual mechanism failed to transfer (see docs/counterfactual_experiment_methodology.md
section 5.1):

    SciQ  `support` is an EXTRACTIVE textbook paragraph containing the answer sentence, so a
          question normally shares most of its vocabulary with it. Zero overlap is anomalous.
    OBQA  `fact1` is a DEDUCTIVE one-clause rule; the question asks for an instance of it. Sharing
          no vocabulary is the design, not a defect. Zero overlap is the MODAL case.

So a single threshold cannot serve both splits. The script reports each distribution separately
and exports a per-dataset tail for hand-checking rather than asserting a defect rate directly.

Usage:
    uv run --active python code/ex5_failure_audit/passage_overlap_audit.py
    uv run --active python code/ex5_failure_audit/passage_overlap_audit.py --sciq-threshold 0.10 --obqa-sample 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex5_failure_audit.rebuild_flagged_pool import content_tokens
from utils.paths import ensure_parent, resolve

DATASET_SPECS = {
    "sciq": {"path": "datasets/sciq/sciq_test_full.json", "passage_field": "support paragraph"},
    "obqa": {"path": "datasets/openbookqa/obqa_test_full.json", "passage_field": "fact1"},
}

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 99)


def _stem(word: str) -> str:
    """Crude number-stripping, matching the `gold_in_passage` recovery in the pool rebuild."""
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def overlap(question: str, passage: str, stem: bool = True) -> float:
    """Share of the question's content words that also occur in the passage.

    Stemming is on by default and matters: without it the detector produces false positives on
    pure singular/plural differences. Three of the fifteen SciQ items below 0.10 unstemmed --
    #201 (`nebula`/`nebulas`), #251 (`snail`/`snails`, `invertebrate`/`invertebrates`) and #546
    (`adaptation`/`adaptations`) -- have passages that are exactly on topic and are rescued to
    0.250, 0.250 and 1.000 respectively once number is stripped. Every genuine topic mismatch
    stays at 0.000 under either variant, so stemming raises precision without costing recall.
    """
    q = content_tokens(question)
    if not q:
        return 0.0
    p = content_tokens(passage)
    if stem:
        q = {_stem(w) for w in q}
        p = {_stem(w) for w in p}
    return len(q & p) / len(q)


def describe(values: Sequence[float]) -> Dict:
    ordered = sorted(values)
    n = len(ordered)
    histogram = []
    for decile in range(10):
        low, high = decile / 10, (decile + 1) / 10
        count = sum(1 for v in ordered if (v >= low and (v < high or (decile == 9 and v <= 1.0))))
        histogram.append({"bin": f"[{low:.1f},{high:.1f})", "count": count,
                          "percentage": round(100.0 * count / n, 2)})
    return {
        "n": n,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "percentiles": {f"p{p}": ordered[min(n - 1, int(p / 100 * n))] for p in PERCENTILES},
        "n_exactly_zero": sum(1 for v in ordered if v == 0.0),
        "pct_exactly_zero": round(100.0 * sum(1 for v in ordered if v == 0.0) / n, 2),
        "n_below_0_10": sum(1 for v in ordered if v < 0.10),
        "n_below_0_25": sum(1 for v in ordered if v < 0.25),
        "histogram": histogram,
    }


def audit(dataset_key: str, threshold: float, sample_n: int, seed: int) -> Dict:
    spec = DATASET_SPECS[dataset_key]
    with open(resolve(spec["path"]), encoding="utf-8") as handle:
        samples = json.load(handle)

    scored = []
    for index, sample in enumerate(samples):
        scored.append(
            {
                "id": sample.get("id", index),
                "overlap": overlap(sample["question"], sample.get("passage", "")),
                "overlap_unstemmed": overlap(
                    sample["question"], sample.get("passage", ""), stem=False
                ),
                "question": sample["question"],
                "passage": sample.get("passage", ""),
                "gold_answer": sample.get("correct_answer", ""),
                "options": sample["options"],
            }
        )

    tail = [r for r in scored if r["overlap"] < threshold]
    # A census where the tail is small; a seeded random sample where it is not.
    if sample_n and len(tail) > sample_n:
        rng = random.Random(seed)
        exported = sorted(rng.sample(tail, sample_n), key=lambda r: r["id"])
        mode = f"seeded random sample of {sample_n} from {len(tail)} (seed {seed})"
    else:
        exported = sorted(tail, key=lambda r: r["id"])
        mode = f"census of all {len(tail)}"

    return {
        "dataset": dataset_key,
        "passage_field": spec["passage_field"],
        "distribution": describe([r["overlap"] for r in scored]),
        "tail_threshold": threshold,
        "n_in_tail": len(tail),
        "pct_in_tail": round(100.0 * len(tail) / len(scored), 2),
        "export_mode": mode,
        "tail_items": exported,
        "all_overlaps": {str(r["id"]): round(r["overlap"], 4) for r in scored},
    }


def print_report(payload: Dict) -> None:
    for key, block in payload["datasets"].items():
        d = block["distribution"]
        print("=" * 100)
        print(f"{key.upper()} (n={d['n']}) -- passage field: {block['passage_field']}")
        print("=" * 100)
        print(f"  mean={d['mean']:.3f}  median={d['median']:.3f}")
        print("  percentiles: " + "  ".join(f"{k}={v:.3f}" for k, v in d["percentiles"].items()))
        print(f"  exactly 0.000: {d['n_exactly_zero']} ({d['pct_exactly_zero']}%)   "
              f"<0.10: {d['n_below_0_10']}   <0.25: {d['n_below_0_25']}")
        print("  histogram:")
        for row in d["histogram"]:
            bar = "#" * max(0, round(60 * row["count"] / d["n"]))
            print(f"    {row['bin']} {row['count']:5} ({row['percentage']:5.1f}%) {bar}")
        print(f"  tail (< {block['tail_threshold']}): {block['n_in_tail']} items "
              f"({block['pct_in_tail']}%) -- exported as {block['export_mode']}")
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark-wide passage/question overlap audit.")
    parser.add_argument(
        "--sciq-threshold", type=float, default=0.10,
        help="SciQ tail cutoff. The split is right-skewed with an isolated spike at 0.",
    )
    parser.add_argument(
        "--obqa-threshold", type=float, default=0.001,
        help="OBQA tail cutoff. Zero overlap is the modal case here, so only exact zeros qualify.",
    )
    parser.add_argument(
        "--obqa-sample", type=int, default=25,
        help="OBQA zero-overlap items are too numerous to census; sample this many.",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out", default="results/ex5_failure_audit/passage_overlap_audit.json")
    args = parser.parse_args(argv)

    payload = {
        "metric": "share of question content words present in the passage",
        "seed": args.seed,
        "datasets": {
            "sciq": audit("sciq", args.sciq_threshold, 0, args.seed),
            "obqa": audit("obqa", args.obqa_threshold, args.obqa_sample, args.seed),
        },
    }
    print_report(payload)

    out_path = ensure_parent(resolve(args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
