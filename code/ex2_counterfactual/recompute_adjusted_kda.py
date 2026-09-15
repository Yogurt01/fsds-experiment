"""Re-derive the adjusted-KDA block of a committed Setting-C run, with both readings of
the sample-exclusion estimator.

Why this exists
---------------
`compute_adjusted_kda` used to report the sample-exclusion estimator in ONE form -- a mean
over the counterfactual-eligible set with dropped items zero-filled -- while README section
2.4 defines it as E_{q in Q_ctx}[KDA_cont(q)], a mean over the RETAINED set. The two are
related exactly:

    mean_over_eligible_zero_filled = mean_over_retained * (n_kept / n_eligible)

so the published figure was the retained mean multiplied by the keep rate, and its
"retention vs baseline" was therefore dominated by the keep rate rather than by any change
in score. Both forms are now computed and reported (see `compute_adjusted_kda`).

Every quantity in the adjusted-KDA block is DERIVED from the per-sample records, not from
model inference, so it can be recomputed exactly from a committed results file with no GPU
and no re-run. This script does that.

It never overwrites its input. The committed result files are immutable execution records
(PROJECT_READING_GUIDE section 5.5); the corrected numbers go to a separate artifact.

Faithfulness check
------------------
The recomputed hard/soft/baseline means must equal the committed ones. The committed
exclusion figure is checked against the `ensemble_both_correct_prior_only` sensitivity
entry rather than the headline, because the headline drop condition also changed on
2026-09-09: all four estimators now use the strict `unstable_other` convention, so Q_ctx
drops `both_correct & unstable_other` as well. If any check fails the recomputation has
drifted from the run that produced the file, and the script exits non-zero rather than
writing a corrected number nobody can trust.

Usage:
    python code/ex2_counterfactual/recompute_adjusted_kda.py            # all three runs
    python code/ex2_counterfactual/recompute_adjusted_kda.py --results <file> [<file> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex2_counterfactual.run_counterfactual_experiment import compute_adjusted_kda
from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

# The three Setting-C runs committed in this repository.
DEFAULT_RESULTS = (
    "results/ex2_counterfactual/results_counterfactual_sciq_test_full.json",
    "results/ex2_counterfactual/results_counterfactual_obqa_test_full.json",
    "results/ex2_counterfactual/results_counterfactual_obqa_test_exact_tier.json",
)

DEFAULT_OUT = "results/ex2_counterfactual/adjusted_kda_corrected.json"

# Committed values were written by `statistics.fmean` over the same records, so agreement
# should be at float-noise level; anything looser means the inputs are not what we think.
TOLERANCE = 1e-9


def recompute(path: str) -> Dict:
    """Recompute one run's adjusted-KDA block and verify it against the committed one."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload["summary"]
    records = payload["results"]
    eligible = [r for r in records if r["counterfactual_valid"]]
    model_names = summary["models"]

    corrected = compute_adjusted_kda(records, eligible, model_names)
    committed = summary["adjusted_kda"]
    prior_only = corrected["q_ctx_sensitivity"]["ensemble_both_correct_prior_only"]

    # --- faithfulness: the parts that existed before must come back unchanged ----------
    checks = {
        "kda_adjusted_hard.mean": (
            committed["kda_adjusted_hard"]["mean"], corrected["kda_adjusted_hard"]["mean"]
        ),
        "kda_adjusted_soft.mean": (
            committed["kda_adjusted_soft"]["mean"], corrected["kda_adjusted_soft"]["mean"]
        ),
        "kda_original_eligible.mean": (
            committed["kda_original_eligible"]["mean"],
            corrected["kda_original_eligible"]["mean"],
        ),
        # The committed `mean` is the zero-filled reading under the OLD prior-only drop
        # condition, so it is checked against the sensitivity entry that reproduces that
        # condition -- not against the headline, which now also drops `unstable_other`.
        "sample_exclusion.mean -> prior_only.mean_over_eligible_zero_filled": (
            committed["kda_adjusted_sample_exclusion"]["mean"],
            prior_only["mean_over_eligible_zero_filled"],
        ),
        "sample_exclusion.n_dropped -> prior_only.n_dropped": (
            float(committed["kda_adjusted_sample_exclusion"]["n_dropped_prior_dependent"]),
            float(prior_only["n_dropped"]),
        ),
    }
    failures = [
        f"{name}: committed {was!r} != recomputed {now!r}"
        for name, (was, now) in checks.items()
        if abs(was - now) > TOLERANCE
    ]
    # The headline drop set is a superset of the committed one (it adds `unstable_other`),
    # so the containment -- not equality -- is what must hold.
    dropped_was = set(committed["kda_adjusted_sample_exclusion"]["dropped_ids"])
    dropped_now = set(corrected["kda_adjusted_sample_exclusion"]["dropped_ids"])
    if not dropped_was <= dropped_now:
        failures.append(
            f"committed dropped_ids are not a subset of the strict-convention drop set: "
            f"{len(dropped_was - dropped_now)} committed id(s) no longer dropped"
        )

    exclusion = corrected["kda_adjusted_sample_exclusion"]
    return {
        "results_file": os.path.relpath(path, resolve_path(".")),
        "dataset": summary["dataset"],
        "min_substitution_tier": summary["counterfactual_generation"]["min_substitution_tier"],
        "models": model_names,
        "verification": {
            "passed": not failures,
            "failures": failures,
            "identity_check_max_abs_error": corrected["identity_check_max_abs_error"],
        },
        "kda_cont_eligible_mean": corrected["kda_original_eligible"]["mean"],
        "kda_adjusted_hard_mean": corrected["kda_adjusted_hard"]["mean"],
        "kda_adjusted_soft_mean": corrected["kda_adjusted_soft"]["mean"],
        "kda_adjusted_sample_exclusion": exclusion,
        "unstable_other_exposure": corrected["unstable_other_exposure"],
        "retention": {
            "hard": corrected["retention_ratio_hard"],
            "soft": corrected["retention_ratio_soft"],
            "sample_exclusion_over_retained": corrected[
                "retention_ratio_sample_exclusion_over_retained"
            ],
            "sample_exclusion_zero_filled": corrected["retention_ratio_sample_exclusion"],
        },
        "q_ctx_sensitivity": corrected["q_ctx_sensitivity"],
    }


def print_report(blocks: Sequence[Dict]) -> None:
    """Console summary: the correction, then the sensitivity that generalises it."""
    print("\n" + "=" * 100)
    print("SAMPLE-EXCLUSION ESTIMATOR -- BOTH READINGS")
    print("=" * 100)
    print(
        f"{'run':<34} {'n_elig':>6} {'n_kept':>6} {'baseline':>9} "
        f"{'zero-fill':>10} {'ret':>6} {'retained':>9} {'ret':>7}"
    )
    print("-" * 100)
    for block in blocks:
        ex = block["kda_adjusted_sample_exclusion"]
        label = f"{block['dataset']} ({block['min_substitution_tier']})"
        print(
            f"{label:<34} {ex['n_eligible']:>6} {ex['n_kept']:>6} "
            f"{block['kda_cont_eligible_mean']:>9.4f} "
            f"{ex['mean_over_eligible_zero_filled']:>10.4f} "
            f"{block['retention']['sample_exclusion_zero_filled'] * 100:>5.1f}% "
            f"{ex['mean_over_retained']:>9.4f} "
            f"{block['retention']['sample_exclusion_over_retained'] * 100:>6.1f}%"
        )

    print("\n" + "=" * 100)
    print("WHY THEY DIVERGE -- does KDA_cont discriminate prior-dependent items?")
    print("=" * 100)
    for block in blocks:
        ex = block["kda_adjusted_sample_exclusion"]
        auc = ex["discrimination_auc_kept_vs_dropped"]
        dropped = ex["mean_kda_of_dropped"]
        print(
            f"  {block['dataset']:<10} ({block['min_substitution_tier']:<9}) "
            f"kept {ex['mean_kda_of_kept']:.4f} | "
            f"dropped {'n/a' if dropped is None else f'{dropped:.4f}'} | "
            f"AUC {'n/a' if auc is None else f'{auc:.4f}'}"
        )
    print("  (AUC 0.5 = KDA_cont carries no information about prior-dependence)")

    print("\n" + "=" * 100)
    print("Q_ctx SENSITIVITY -- retained mean vs baseline, by definition")
    print("=" * 100)
    for block in blocks:
        base = block["kda_cont_eligible_mean"]
        print(f"\n  {block['dataset']} ({block['min_substitution_tier']}), baseline {base:.4f}")
        print(f"    {'definition':<44} {'dropped':>7} {'retained':>9} {'ret':>7} {'zero-fill ret':>14}")
        for name, sens in block["q_ctx_sensitivity"].items():
            retained_ret = sens["mean_over_retained"] / base if base else 0.0
            zero_ret = sens["mean_over_eligible_zero_filled"] / base if base else 0.0
            print(
                f"    {name[:44]:<44} {sens['n_dropped_prior_dependent']:>7} "
                f"{sens['mean_over_retained']:>9.4f} {retained_ret * 100:>6.1f}% "
                f"{zero_ret * 100:>13.1f}%"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the adjusted-KDA block of committed Setting-C runs with both "
            "readings of the sample-exclusion estimator. Never writes to its input."
        )
    )
    parser.add_argument(
        "--results", nargs="+", default=list(DEFAULT_RESULTS),
        help="Committed Setting-C result files (relative to the project root).",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, help="Where to write the corrected block."
    )
    args = parser.parse_args()

    blocks: List[Dict] = []
    for entry in args.results:
        path = resolve_path(entry)
        if os.path.abspath(path) == os.path.abspath(resolve_path(args.out)):
            print(f"Refusing to overwrite an input file: {path}", file=sys.stderr)
            return 1
        print(f"Reading {path}")
        blocks.append(recompute(path))

    failed = [b for b in blocks if not b["verification"]["passed"]]
    print_report(blocks)

    out_path = ensure_parent(resolve_path(args.out))
    payload = {
        "_provenance": {
            "generator": "code/ex2_counterfactual/recompute_adjusted_kda.py",
            "what": (
                "Adjusted-KDA blocks re-derived from the committed Setting-C result "
                "files, reporting BOTH readings of the sample-exclusion estimator plus a "
                "Q_ctx definition sweep. No model was run; every quantity here is a "
                "deterministic function of the per-sample records in the source files."
            ),
            "why": (
                "README section 2.4 defines KDA_adj^excl as a mean over the retained set; "
                "the original implementation reported a mean over the eligible set with "
                "dropped items zero-filled. The two differ by the keep rate."
            ),
            "source_files": [b["results_file"] for b in blocks],
            "inputs_unmodified": True,
        },
        "runs": blocks,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")

    if failed:
        print("\nVERIFICATION FAILED -- recomputation does not match the committed run:",
              file=sys.stderr)
        for block in failed:
            for failure in block["verification"]["failures"]:
                print(f"  {block['results_file']}: {failure}", file=sys.stderr)
        return 1
    print("Verification passed: every pre-existing figure re-derived exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
