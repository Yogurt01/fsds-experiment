# KDA Evaluation with `Qwen2.5-7B-Instruct` (4-bit NF4, Kaggle 2×T4)

Evaluation report for the **KDA** metric computed with `Qwen2.5-7B-Instruct` as the simulated
student, on the same full `test` splits as every other model in this project: **OpenBookQA** (500
questions) and **SciQ** (884 questions).

This is the Qwen2.5-7B counterpart to
[`kda_qwen3_4b_evaluation_report.md`](kda_qwen3_4b_evaluation_report.md), run per
[`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) §2 to test whether that report's central finding —
**parametric saturation collapses KDA's usable support to a few percent of the dataset** — is
specific to `Qwen3-4B-Instruct-2507` or general to modern instruct models. **It generalises, and at
a larger parameter count the collapse stays essentially flat on SciQ but deepens further on OBQA —
a mixed, not uniformly worsening, scale effect.**

---

## Headline finding

> On SciQ, **only 44 of 884 questions (5.0%)** contribute anything to $KDA_{disc}$ — marginally
> *more* than Qwen3-4B's 41 (4.6%), essentially flat. On OpenBookQA the usable support is **74 of
> 500 (14.8%)** — *fewer* than Qwen3-4B's 87 (17.4%), i.e. OBQA's support is **more** collapsed at
> 7B than at 4B. Moving from a 4B to a 7B instruct model does not uniformly deepen the collapse on
> both datasets — it leaves SciQ's roughly where it was and deepens OBQA's.

As with Qwen3-4B, the metric returns high, confident-looking numbers ($KDA_{disc} = 0.9773$ on
SciQ, $0.6081$ on OBQA) computed from a small minority of each dataset, with no signal in the
number itself that this happened.

---

## Contents

1. [Model configuration and environment](#1-model-configuration-and-environment)
2. [Memory footprint and latency](#2-memory-footprint-and-latency)
3. [Scoring protocol](#3-scoring-protocol)
4. [Results](#4-results)
5. [Finding 1 — the metric's support collapses, and the direction is dataset-dependent](#5-finding-1--the-metrics-support-collapses-and-the-direction-is-dataset-dependent)
6. [Finding 2 — probability saturation, still bimodal](#6-finding-2--probability-saturation-still-bimodal)
7. [Finding 3 — three-way comparison with `KDA_small` and Qwen3-4B](#7-finding-3--three-way-comparison-with-kda_small-and-qwen3-4b)
8. [Bucket transitions and item-level agreement with Qwen3-4B](#8-bucket-transitions-and-item-level-agreement-with-qwen3-4b)
9. [Limitations and caveats](#9-limitations-and-caveats)
10. [Reproduction](#10-reproduction)

---

## 1. Model configuration and environment

| | |
|---|---|
| **Model** | `Qwen/Qwen2.5-7B-Instruct` (downloaded fresh from the Hub on the cloud notebook) |
| **Quantisation** | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)` — the project's declared precision policy ([`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) §1.4): match Qwen3-4B's 4-bit NF4 so quantisation is not a confound in any Qwen2.5-vs-Qwen3 diff |
| **GPU** | 2× Tesla T4 (Kaggle), `device_map="auto"` |
| **Driver / libraries** | torch 2.10.0+cu128 |
| **Python / OS** | 3.12.13 · Linux-6.12.90+-x86_64-with-glibc2.35 |
| **Batch size** | 16 (batched forward pass per phase, vs Qwen3-4B's single-sequence batch of 1) |
| **Datasets** | rebuilt from `allenai/sciq` / `allenai/openbookqa` (`additional` config) with the same seed/logic as `datasets/`, not `--local-data`-pinned to the committed JSON files |
| **Target fact** | OpenBookQA → `fact1` · SciQ → `support` paragraph |
| **Script** | [`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](../../code/ex1_reproduce_KDA/run_qwen7b_eval.py) — the same standalone evaluator `RUN_QWEN2.5.md` Appendix A reproduces inline |

> **Dataset-construction caveat.** Unlike the Qwen3-4B run, this run did not pass `--local-data`, so
> it rebuilt SciQ/OBQA from the Hub rather than scoring the exact committed
> `datasets/{sciq,openbookqa}/*_test_full.json` files. `run_qwen7b_eval.py`'s dataset builders use
> the same seed (42) and the same construction logic as `code/pre_data/`, so sample composition and
> option order should match, but this was not bit-verified item-by-item the way the Qwen3-4B
> correctness gates were in [`e1_counterfactual_llm_scale.md`](../ex2_counterfactual/e1_counterfactual_llm_scale.md).
> The item-level agreement analysis in §8 below joins on integer `id`, which assumes the two runs'
> row orderings coincide; both files report `n=884` (SciQ) / `n=500` (OBQA) with no drops, consistent
> with that assumption but not an independent proof of it.

## 2. Memory footprint and latency

| Metric | Value |
|---|---:|
| Model load time | 110.39 s |
| Allocated after load | 1.354 GB |
| Reserved after load | 1.516 GB |
| **Peak allocated (whole run)** | **4.707 GB** |
| **Peak reserved (whole run)** | **11.633 GB** |
| Total runtime (load + both datasets) | 606.51 s (**10.1 min**) |

Peak reserved memory (11.6 GB) is well inside a single T4's 16 GB even before Kaggle's second GPU
is counted — 4-bit NF4 leaves ample headroom for a 7.62B model, unlike the 3.68 GiB local RTX 3050
run against `Qwen3-4B-Instruct-2507`, which finished at 95% utilisation.

### Latency

| Dataset | Samples | Scoring time | **Per sample** |
|---|---:|---:|---:|
| OpenBookQA | 500 | 98.64 s | **0.197 s** |
| SciQ | 884 | 375.48 s | **0.425 s** |

> **Not directly comparable to Qwen3-4B's per-sample timings.** This run used batch size 16 (16
> sequences scored per forward pass); the Qwen3-4B local run used batch size 1. The per-sample
> figures above reflect batched-T4 throughput, not single-sequence latency, so a batch-size-1
> apples-to-apples comparison would need a re-run at `--batch-size 1`, which was not done.

## 3. Scoring protocol

Identical to [§3 of the Qwen3-4B report](kda_qwen3_4b_evaluation_report.md#3-scoring-protocol):
each sample scored twice (without fact / with fact) via a single batched forward pass, logits at
the final position restricted to the four option letters (all single-token spellings combined with
`logsumexp`), softmax over the four options, no free-text generation or parsing. Same $KDA_{disc}$
/ $KDA_{cont}$ formulas, same dataset-level (not per-question-over-models) aggregation, so the same
[aggregation caveat](kda_qwen3_4b_evaluation_report.md#3-scoring-protocol) applies: KDA values here
are not numerically comparable to `kda_reproduction_summary.md`'s question-level ensemble KDA;
accuracies are.

## 4. Results

### 4.1. Headline metrics

| Dataset | $n$ | $KDA_{disc}$ | $KDA_{cont}$ | $Acc_{\text{wof}}$ | $Acc_{\text{wf}}$ | $\Delta$ | Memorisation ($r^q{=}1$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OpenBookQA** | 500 | 0.6081 | 0.6121 | 85.20% | 92.40% | **+7.20 pp** | **85.20%** |
| **SciQ** | 884 | 0.9773 | 0.9655 | 95.02% | 99.77% | **+4.75 pp** | **95.02%** |

### 4.2. Probability-space view

| Dataset | mean $P(R^q{=}1)$ | mean $P(R^{q+f}{=}1)$ | Gain |
|---|---:|---:|---:|
| OpenBookQA | 0.8475 | 0.9202 | +0.0728 |
| SciQ | 0.9491 | 0.9969 | +0.0477 |

### 4.3. Four-bucket contingency table

| Dataset | `both_correct` | `wrong_to_correct` | `both_wrong` | `correct_to_wrong` |
|---|---:|---:|---:|---:|
| **OpenBookQA** | 417 (83.40%) | 45 (9.00%) | 29 (5.80%) | 9 (1.80%) |
| **SciQ** | 839 (94.91%) | 43 (4.86%) | 1 (0.11%) | 1 (0.11%) |

## 5. Finding 1 — the metric's support collapses, and the direction is dataset-dependent

| Dataset | Denominator | Usable share | Discarded | Qwen3-4B denominator (usable share) |
|---|---:|---:|---:|---|
| **SciQ** | **44 / 884** | **4.98%** | 840 questions | 41/884 (4.64%) |
| **OpenBookQA** | **74 / 500** | **14.80%** | 426 questions | 87/500 (17.40%) |

$KDA_{cont}$'s denominator, out of a theoretical maximum of $n$:

| Dataset | $KDA_{cont}$ denominator | Maximum ($n$) | Effective support |
|---|---:|---:|---:|
| SciQ | 44.97 | 884 | **5.09%** |
| OpenBookQA | 76.26 | 500 | **15.25%** |

At a larger scale (7B vs 4B), SciQ's already-tiny usable support is essentially flat — a marginal
*increase* in denominator share (4.64% → 4.98%), both single-digit percentages of the dataset —
while OpenBookQA's usable support **shrinks further** (17.40% → 14.80% denominator share, i.e.
*fewer* of OBQA's questions remain usable at 7B), consistent with a "collapse deepens with scale"
pattern on OBQA specifically, not on SciQ. See §7 for the full three-way comparison.

## 6. Finding 2 — probability saturation, still bimodal

| $P(R^q{=}1)$ band | OpenBookQA | SciQ |
|---|---:|---:|
| $[0.90, 1.00]$ | 410 (82.00%) | 833 (94.23%) |
| $[0.75, 0.90)$ | 6 (1.20%) | 4 (0.45%) |
| $[0.50, 0.75)$ | 9 (1.80%) | 3 (0.34%) |
| $[0.25, 0.50)$ | 6 (1.20%) | 2 (0.23%) |
| $[0.00, 0.25)$ | 69 (13.80%) | 42 (4.75%) |
| **Middle bands $[0.25, 0.90)$** | **21 (4.20%)** | **9 (1.02%)** |

Still bimodal, and comparably so to Qwen3-4B (whose middle bands were 3.40% OBQA / 0.23% SciQ):
**182/500 (36.4%) OBQA** and **615/884 (69.6%) SciQ** samples have $P(R^q{=}1)$ exactly `1.0` in
float32; the largest value strictly below 1.0 in either dataset is the same float32 ceiling
Qwen3-4B hit, `0.9999998807907104`. $KDA_{cont} \approx KDA_{disc}$ holds again (0.9655 vs 0.9773
SciQ; 0.6121 vs 0.6081 OBQA — differences of ~0.01–0.004), so the continuous relaxation is redundant
at this scale too, for the same reason given in
[§6 of the Qwen3-4B report](kda_qwen3_4b_evaluation_report.md#6-finding-2--probability-saturation-makes-kda_cont-redundant):
the discrete metric's argmax collapse is the underlying cause, not letter-logit scoring artefacts.

## 7. Finding 3 — three-way comparison with `KDA_small` and Qwen3-4B

Against the four-model `KDA_small` encoder ensemble
([`kda_reproduction_summary.md`](kda_reproduction_summary.md)) and Qwen3-4B
([§7 of that report](kda_qwen3_4b_evaluation_report.md#7-finding-3--comparison-with-the-kda_small-encoder-baseline)),
extending that report's own comparison table with a third model row:

| | SciQ $Acc_{wof}$ | SciQ $Acc_{wf}$ | OBQA $Acc_{wof}$ | OBQA $Acc_{wf}$ |
|---|---:|---:|---:|---:|
| `KDA_small` (ensemble average) | 38.43% | 70.73% | 32.35% | 39.55% |
| `KDA_small` (pooled probability vote) | 45.48% | 88.57% | 34.80% | 40.60% |
| Qwen3-4B-Instruct-2507 (4-bit) | 95.36% | 99.77% | 82.60% | 92.00% |
| **Qwen2.5-7B-Instruct (4-bit)** | **95.02%** | **99.77%** | **85.20%** | **92.40%** |
| **Qwen2.5-7B vs. Qwen3-4B** | **−0.34 pp** | **0.00 pp** | **+2.60 pp** | **+0.40 pp** |
| **Qwen2.5-7B vs. `KDA_small` pooled vote** | **+49.54 pp** | **+11.20 pp** | **+50.40 pp** | **+51.80 pp** |

**Four `RUN_QWEN2.5.md` §2 comparison items, measured:**

1. **Saturation diff.** Mixed, not uniform: SciQ zero-context accuracy is essentially flat
   (95.36% → 95.02%, **−0.34 pp**), while OBQA rises (82.60% → 85.20%, **+2.60 pp**). The larger
   model does *not* saturate more on both datasets simultaneously — it saturates about the same on
   SciQ and somewhat more on OBQA.
2. **$KDA_{disc}$ denominator diff.** SciQ: 41 → 44 (denominator grows slightly, tracking the
   near-flat accuracy). OBQA: 87 → 74 (denominator *shrinks* — fewer usable questions at 7B than at
   4B), consistent with item 1's OBQA accuracy rise. Both remain small minorities of their datasets;
   neither model leaves KDA_disc with a healthy sample size.
3. **Bucket diff (`both_wrong` / `correct_to_wrong`).** SciQ: Qwen3-4B left 2 / 0; Qwen2.5-7B leaves
   **1 / 1** — the same near-zero regime, redistributed by one item. OBQA: Qwen3-4B left 36 / 4;
   Qwen2.5-7B leaves **29 / 9** — fewer pure reasoning failures, more cases where the fact actively
   hurt (context bias, §8.3 of the Qwen3-4B report), more than doubling that bucket's share (0.80% →
   1.80% of OBQA).
4. **Item-level agreement** — see §8 below.

Qwen2.5-7B's accuracy gain from the fact ($\Delta$) is +4.75 pp SciQ / +7.20 pp OBQA, close to
Qwen3-4B's +4.41 pp / +9.40 pp — both models leave the fact doing very little marginal work once
their strong prior is accounted for, and both dwarf the `KDA_small` pooled vote's +43.10 pp / +5.80
pp.

## 8. Bucket transitions and item-level agreement with Qwen3-4B

Joining Qwen3-4B's and Qwen2.5-7B's per-sample records on `id` (see the dataset-construction caveat
in §1 about this join's assumption):

| Dataset | Both correct w/o fact | Both wrong w/o fact | Qwen3-only correct | Qwen2.5-only correct | Overall $r^q$ agreement |
|---|---:|---:|---:|---:|---:|
| SciQ (n=884) | 824 (93.21%) | 25 (2.83%) | 19 (2.15%) | 16 (1.81%) | **849 (96.04%)** |
| OBQA (n=500) | 383 (76.60%) | 44 (8.80%) | 30 (6.00%) | 43 (8.60%) | **427 (85.40%)** |

**SciQ shows a strong shared memorisation profile** — 96.0% item-level agreement on whether the
model already knows the answer without the fact, with only 35 of 884 items (4.0%) where the two
models disagree. **OBQA shows more disagreement** (85.4% agreement, 73/500 items split), consistent
with OBQA's larger accuracy shift between the two models (§7 item 1): Qwen2.5-7B knows 43 OBQA items
cold that Qwen3-4B does not, against only 30 the other way, which is the same net +13-item (+2.6 pp)
gap reported as a percentage in §7.

## 9. Limitations and caveats

All limitations from
[§9 of the Qwen3-4B report](kda_qwen3_4b_evaluation_report.md#9-limitations-and-caveats) apply
equally here (single solver/seed, letter-logit sharpening, no option-order debiasing, no
calibration check, cross-metric-family KDA values not comparable). In addition:

* **Dataset construction not bit-pinned.** As noted in §1, this run rebuilt the datasets from the
  Hub rather than scoring `--local-data`-pinned copies of the exact committed JSON files. The §8
  item-level join assumes row-order correspondence by integer `id`, which is plausible (identical
  seed and construction logic, identical sample counts) but not independently verified the way the
  Qwen3-4B-vs-encoder correctness gates were elsewhere in this project.
* **Batch size differs from Qwen3-4B (16 vs 1).** The §2 latency figures are not apples-to-apples
  with Qwen3-4B's single-sequence timings; only the accuracy/KDA numbers are compared in §7.
* **No qualitative case-study pass.** Unlike §8 of the Qwen3-4B report, this report does not include
  a hand-inspected qualitative error taxonomy (memorisation examples, context-bias case studies,
  OpenBookQA label-noise spot-checks). The quantitative buckets in §4.3/§7 are directly comparable;
  the qualitative characterisation of *why* items land in each bucket is not re-derived here and is
  assumed to carry over from the Qwen3-4B report's own inspection of largely overlapping items (§8
  shows 93–96% item-level agreement on SciQ, somewhat less on OBQA).

## 10. Reproduction

Runbook: [`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) §2 (Entry 1), including a Colab/Kaggle
walkthrough and the script reproduced inline in its Appendix A.

```bash
python code/ex1_reproduce_KDA/run_qwen7b_eval.py --datasets obqa sciq --precision 4bit --batch-size 16 --out-dir ./kda_out
```

### Artefacts

| Path | Contents |
|---|---|
| [`results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_results.json`](../../results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_results.json) | Per-sample records + summary block (config, memory, per-dataset metrics) |
| [`results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_records.csv`](../../results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_records.csv) | One row per sample: probabilities, predictions, `r_q`, `r_q_plus_f`, bucket |
| [`results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_summary.csv`](../../results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_summary.csv) | One row per dataset: KDA_disc, KDA_cont, accuracies, bucket counts |
| [`results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_eval.log`](../../results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_eval.log) | Full execution log |
| [`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](../../code/ex1_reproduce_KDA/run_qwen7b_eval.py) | The standalone evaluation script |

---

## Related documents

* [`kda_qwen3_4b_evaluation_report.md`](kda_qwen3_4b_evaluation_report.md) — the Qwen3-4B run this
  report compares against throughout; the source of §7's Qwen3-4B row and every "vs. Qwen3-4B" diff.
* [`kda_reproduction_summary.md`](kda_reproduction_summary.md) — the `KDA_small` encoder baseline in
  §7.
* [`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) — the cloud runbook this run followed (Entry 1).
* [`../../README.md`](../../README.md) §6.4 — project overview; this run is the scale-generalisation
  check on the RQ1 evidence.
