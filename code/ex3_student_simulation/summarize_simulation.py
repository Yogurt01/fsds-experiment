"""Turn the persona-simulation result files into the tables used in the report.

Reads the per-dataset JSON written by `run_student_simulation.py` and prints Markdown:
accuracy stratification with Wilson intervals, paradigm comparison with exact McNemar
tests on the paired predictions, the joint-scaffold differentiation diagnostics, and the
error-pattern probes.

Usage:
    uv run --active python code/ex3_student_simulation/summarize_simulation.py
    uv run --active python code/ex3_student_simulation/summarize_simulation.py --tag smoke50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex3_student_simulation.personas import TIERS, TIER_LABELS
from utils.paths import resolve

LETTERS = ("A", "B", "C", "D")
PARADIGMS = ("joint", "isolated")
DATASETS = ("sciq", "obqa")

# Zero-context accuracy of the same model with no persona at all, from
# results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json. This is the saturation the
# personas are trying to break.
BASELINE_ACC_WOF = {"sciq": 0.9536, "obqa": 0.8260}


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Optional[tuple]:
    """95% Wilson score interval -- well behaved near 0 and 1, unlike the normal approx."""
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    `b` and `c` are the discordant counts. Under H0 each discordant pair is a fair coin, so
    the p-value is the two-sided binomial tail at n = b + c, p = 0.5.
    """
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def chi2_sf_df3(x: float) -> float:
    """Upper tail of a chi-square with 3 degrees of freedom (closed form, no scipy).

    P(X > x) = erfc(sqrt(x/2)) + sqrt(2x/pi) * exp(-x/2)
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def chi2_against(observed: Sequence[int], expected_p: Sequence[float]) -> tuple:
    """Pearson chi-square of an answer-letter histogram against a reference distribution."""
    n = sum(observed)
    stat = 0.0
    for obs, p in zip(observed, expected_p):
        exp = n * p
        if exp > 0:
            stat += (obs - exp) ** 2 / exp
    return stat, chi2_sf_df3(stat)


def paired_counts(records: Sequence[Dict], key_a, key_b) -> tuple:
    """(a right & b wrong, a wrong & b right) over the same questions."""
    b = sum(1 for r in records if key_a(r) and not key_b(r))
    c = sum(1 for r in records if not key_a(r) and key_b(r))
    return b, c


def fmt(value: Optional[float], places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def load(tag: str, out_dir: str) -> Dict[str, Dict]:
    payloads = {}
    for key in DATASETS:
        base = f"results_persona_simulation_{key}.json"
        if tag:
            stem, ext = os.path.splitext(base)
            base = f"{stem}_{tag}{ext}"
        path = resolve(os.path.join(out_dir, base))
        if not os.path.exists(path):
            print(f"<!-- missing: {path} -->")
            continue
        with open(path, encoding="utf-8") as handle:
            payloads[key] = json.load(handle)
    return payloads


def section_stratification(payloads: Dict[str, Dict]) -> List[str]:
    out = ["### Accuracy stratification (zero context, 95% Wilson CI)", ""]
    out.append("| Dataset | Paradigm | Beginner | Intermediate | Advanced | Gap (A-B) | Monotone |")
    out.append("|---|---|---|---|---|---|---|")
    for key, payload in payloads.items():
        metrics = payload["summary"]["metrics"]
        n = metrics["n_samples"]
        for paradigm in PARADIGMS:
            if paradigm not in metrics["paradigms"]:
                continue
            block = metrics["paradigms"][paradigm]
            cells = []
            for tier in TIERS:
                acc = block["accuracy_by_tier"][tier]
                lo, hi = wilson_ci(block["per_tier"][tier]["n_correct"], n)
                cells.append(f"{acc:.3f} [{lo:.3f}, {hi:.3f}]")
            out.append(
                f"| {key} | {paradigm} | " + " | ".join(cells)
                + f" | {block['ability_gap_advanced_minus_beginner']:+.3f} |"
                + (" yes |" if block["monotonicity"]["holds"] else " **no** |")
            )
    out.append("")
    out.append("Persona-free reference (same model, same zero-context prompt, ex1): "
               + ", ".join(f"{k} {v:.3f}" for k, v in BASELINE_ACC_WOF.items()) + ".")
    out.append("")
    return out


def section_paradigm_comparison(payloads: Dict[str, Dict]) -> List[str]:
    out = ["### Paradigm comparison (exact McNemar on paired predictions)", ""]
    out.append("| Dataset | Comparison | Discordant b/c | p | Note |")
    out.append("|---|---|---|---|---|")
    for key, payload in payloads.items():
        records = payload["results"]
        metrics = payload["summary"]["metrics"]
        for paradigm in PARADIGMS:
            if paradigm not in metrics["paradigms"]:
                continue
            b, c = paired_counts(
                records,
                lambda r, p=paradigm: r[p]["advanced"]["correct"] == 1,
                lambda r, p=paradigm: r[p]["beginner"]["correct"] == 1,
            )
            out.append(
                f"| {key} | {paradigm}: advanced vs beginner | {b}/{c} | "
                f"{fmt_p(mcnemar_exact(b, c))} | advanced-only-right / beginner-only-right |"
            )
        for tier in TIERS:
            if not all(p in metrics["paradigms"] for p in PARADIGMS):
                continue
            b, c = paired_counts(
                records,
                lambda r, t=tier: r["joint"][t]["correct"] == 1,
                lambda r, t=tier: r["isolated"][t]["correct"] == 1,
            )
            out.append(
                f"| {key} | {tier}: joint vs isolated | {b}/{c} | "
                f"{fmt_p(mcnemar_exact(b, c))} | joint-only-right / isolated-only-right |"
            )
    out.append("")

    out.append("| Dataset | Gap (joint) | Gap (isolated) | Joint - isolated |")
    out.append("|---|---|---|---|")
    for key, payload in payloads.items():
        metrics = payload["summary"]["metrics"]
        if "gap_joint_minus_isolated" not in metrics:
            continue
        out.append(
            f"| {key} | {metrics['paradigms']['joint']['ability_gap_advanced_minus_beginner']:+.3f} "
            f"| {metrics['paradigms']['isolated']['ability_gap_advanced_minus_beginner']:+.3f} "
            f"| {metrics['gap_joint_minus_isolated']:+.3f} |"
        )
    out.append("")

    out.append("| Dataset | Tier | Joint/isolated prediction agreement |")
    out.append("|---|---|---|")
    for key, payload in payloads.items():
        cross = payload["summary"]["metrics"].get("cross_paradigm")
        if not cross:
            continue
        for tier in TIERS:
            out.append(f"| {key} | {tier} | {fmt_pct(cross[tier]['prediction_agreement'])} |")
    out.append("")
    return out


def section_differentiation(payloads: Dict[str, Dict]) -> List[str]:
    out = ["### Joint-scaffold differentiation pressure", ""]
    out.append(
        "| Dataset | Order | All 3 distinct | Last tier breaks a consensus | Breaks that are wrong "
        "| Breaks away from a correct consensus |"
    )
    out.append("|---|---|---|---|---|---|")
    for key, payload in payloads.items():
        jd = payload["summary"]["metrics"].get("joint_differentiation")
        if not jd:
            continue
        out.append(
            f"| {key} | {'->'.join(t[:3] for t in jd['emission_order'])} "
            f"| {fmt_pct(jd['all_three_tiers_distinct_rate'])} "
            f"| {fmt_pct(jd['last_tier_deviates_from_agreeing_earlier_tiers'])} "
            f"(n={jd['n_questions_earlier_tiers_agree']}) "
            f"| {fmt_pct(jd['deviations_that_are_wrong'])} "
            f"| {fmt_pct(jd['deviations_away_from_a_correct_consensus'])} |"
        )
    out.append("")
    out.append("Chance rate for three independent uniform picks all differing: 37.5%.")
    out.append("")
    return out


def section_errors(payloads: Dict[str, Dict]) -> List[str]:
    out = ["### Error-pattern analysis", "", "#### Trap-hit rate among errors (random distractor = 0.333)", ""]
    out.append(
        "| Dataset | Paradigm | Tier | Errors | Longest option | Stem overlap | Lookalike to gold |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for key, payload in payloads.items():
        metrics = payload["summary"]["metrics"]
        for paradigm in PARADIGMS:
            if paradigm not in metrics["paradigms"]:
                continue
            for tier in TIERS:
                tm = metrics["paradigms"][paradigm]["per_tier"][tier]
                probes = tm["trap_probes"]
                cells = []
                for name in ("longest_option", "surface_match", "lookalike_to_gold"):
                    rate = probes[name]["trap_hit_rate_among_errors"]
                    cells.append(
                        f"{fmt(rate)} (n={probes[name]['eligible_errors']})"
                        if rate is not None else f"n/a (n=0)"
                    )
                out.append(
                    f"| {key} | {paradigm} | {tier} | {tm['n_errors']} | " + " | ".join(cells) + " |"
                )
    out.append("")

    out.append("#### Confidence and answer spread")
    out.append("")
    out.append(
        "| Dataset | Paradigm | Tier | Mean P(correct) | Mean P(selected) | Mean P(selected) on errors "
        "| Mean entropy (bits) | Answer distribution A/B/C/D | vs gold spread (chi2, p) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for key, payload in payloads.items():
        metrics = payload["summary"]["metrics"]
        records = payload["results"]
        n = len(records)
        gold_p = [
            sum(1 for r in records if r["answer_idx"] == i) / n for i in range(4)
        ]
        for paradigm in PARADIGMS:
            if paradigm not in metrics["paradigms"]:
                continue
            for tier in TIERS:
                tm = metrics["paradigms"][paradigm]["per_tier"][tier]
                dist = tm["answer_position_distribution"]
                observed = [
                    sum(1 for r in records if r[paradigm][tier]["pred_idx"] == i)
                    for i in range(4)
                ]
                stat, p_value = chi2_against(observed, gold_p)
                out.append(
                    f"| {key} | {paradigm} | {tier} | {fmt(tm['mean_p_correct'])} "
                    f"| {fmt(tm['mean_p_selected'])} | {fmt(tm['mean_p_selected_on_errors'])} "
                    f"| {fmt(tm['mean_entropy_bits'], 2)} "
                    f"| {'/'.join(f'{dist[l]:.2f}' for l in LETTERS)} "
                    f"| {stat:.1f}, {fmt_p(p_value)} |"
                )
    for key, payload in payloads.items():
        records = payload["results"]
        n = len(records)
        gold_p = [sum(1 for r in records if r["answer_idx"] == i) / n for i in range(4)]
        out.append(
            f"| {key} | *gold* | -- | -- | -- | -- | -- "
            f"| {'/'.join(f'{p:.2f}' for p in gold_p)} | -- |"
        )
    out.append("")

    out.append("#### Error-set structure (does the ability ladder nest?)")
    out.append("")
    out.append(
        "| Dataset | Paradigm | Jaccard B-I | Jaccard B-A | Jaccard I-A "
        "| Advanced errors also missed by beginner |"
    )
    out.append("|---|---|---|---|---|---|")
    for key, payload in payloads.items():
        metrics = payload["summary"]["metrics"]
        for paradigm in PARADIGMS:
            if paradigm not in metrics["paradigms"]:
                continue
            block = metrics["paradigms"][paradigm]
            ov = block["error_set_overlap_jaccard"]
            out.append(
                f"| {key} | {paradigm} | {fmt(ov['beginner_vs_intermediate'])} "
                f"| {fmt(ov['beginner_vs_advanced'])} | {fmt(ov['intermediate_vs_advanced'])} "
                f"| {fmt_pct(block['error_containment']['advanced_errors_also_missed_by_beginner'])} |"
            )
    out.append("")
    return out


def section_runtime(payloads: Dict[str, Dict]) -> List[str]:
    out = ["### Run metadata", ""]
    out.append("| Dataset | n | Setting | s/question | Total (min) | Peak VRAM (GB) |")
    out.append("|---|---|---|---|---|---|")
    for key, payload in payloads.items():
        summary = payload["summary"]
        metrics = summary["metrics"]
        out.append(
            f"| {key} | {metrics['n_samples']} | {metrics['setting']} "
            f"| {metrics['mean_latency_seconds']:.2f} "
            f"| {metrics['scoring_seconds'] / 60:.1f} "
            f"| {summary['memory'].get('peak_reserved_gb', 'n/a')} |"
        )
    out.append("")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Markdown tables for the persona simulation.")
    parser.add_argument("--tag", default="", help="Filename suffix of the run to summarise.")
    parser.add_argument("--out-dir", default="results/ex3_student_simulation")
    parser.add_argument("--out", default="", help="Write to this file instead of stdout.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payloads = load(args.tag, args.out_dir)
    if not payloads:
        print("no result files found", file=sys.stderr)
        return 1

    lines: List[str] = []
    lines += section_stratification(payloads)
    lines += section_paradigm_comparison(payloads)
    lines += section_differentiation(payloads)
    lines += section_errors(payloads)
    lines += section_runtime(payloads)

    text = "\n".join(lines)
    if args.out:
        with open(resolve(args.out), "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {resolve(args.out)}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
