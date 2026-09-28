# RQ1 — Final Report

> **Do modern LLM-based evaluators reliably measure whether a multiple-choice question genuinely
> requires the target learning material?**

**Written:** 2026-09-26. This report consolidates [`rq1_conclusion.md`](rq1_conclusion.md) and
[`rq1_research_synthesis.md`](../synthesis/rq1_research_synthesis.md), which are left unchanged.
Every quantitative claim cites its primary source, and bracketed keys such as [FA] link to it.
Figures a source later corrected are shown as *published → corrected*. The 21 departures from the
earlier reports are documented in [`rq1_final_report_corrections.md`](rq1_final_report_corrections.md).

**Scope.** This report covers the **KDA family** ($KDA_{disc}$, $KDA_{cont}$, and the adjusted
$KDA_{adj}^{hard/soft/excl}$) on SciQ test (884 items) and OpenBookQA (OBQA) test (500 items).
Every KDA value is a **model-ensemble or single-solver proxy**. No human-subject KDA measurement
exists in this evidence base ([FA] §5.5).

**Terms.** $R^q$ / $R^{q+f}$: the solver is correct without / with the target material.
$Acc_{wof}$: accuracy without the material. Buckets (without → with): `both_correct`,
`wrong_to_correct`, `correct_to_wrong`, `both_wrong`. **Setting C** gives the solver a counterfactual
passage asserting a different answer: the item is `context_dependent` if the solver follows it, and
`prior_dependent` if the solver keeps the gold.

---

## 1. Verdict

**No.** On SciQ and OBQA the KDA family does not reliably measure whether a question requires its
target material. The failure is **insensitivity, not inversion**: the metric does not score
material-independent questions lower; it is blind to them.

| A reliable instrument needs… | Result | § |
|---|---|---|
| Non-degenerate support | **Fails.** Usable support is 4.6–5.0% of SciQ at modern-LLM scale, on two solvers | 2.1 |
| Discrimination between dependent and independent items | **Fails.** $r(KDA_{cont}, P(R^q{=}1)) = +0.075$ on SciQ; zero-context solve rate is flat across KDA deciles | 2.2 |
| A solver-independent verdict | **Contrary evidence.** Per-solver KDA on the same items spans 0.2737–0.8146; the ignorance weight is near-constant for 2 of 4 encoder students and near-zero at LLM scale | 2.3 |
| Agreement with human judgement | **Fails.** Every unweighted CI contains chance (both annotators); population-weighted AUCs are below 0.5 | 2.4 |

The adjusted estimators do not repair the metric (§2.5–2.6). The source paper's own validation was
already weakest on these corpora (§2.7). Competing explanations are measured and cannot carry the
effect (§2.8).

---

## 2. Empirical findings

### 2.1 Support collapses, on two different solvers

$KDA_{disc}$'s denominator $\sum_i (1 - r_i^q)$ counts only questions the solver answers **wrong
without the material**. Every question it already knows is silently dropped.

| Solver | SciQ $Acc_{wof}$ | SciQ denominator | SciQ $KDA_{disc}$ / $KDA_{cont}$ | OBQA $Acc_{wof}$ | OBQA denominator | OBQA $KDA_{disc}$ / $KDA_{cont}$ |
|---|---:|---:|---|---:|---:|---|
| `KDA_small` encoder ensemble, pooled vote | 45.48% | — | — | 34.80% | — | — |
| Qwen3-4B-Instruct-2507 | **95.36%** | **41 / 884 = 4.6%** | 0.9512 / 0.9407 | **82.60%** | **87 / 500 = 17.4%** | 0.5862 / 0.5956 |
| Qwen2.5-7B-Instruct | **95.02%** | **44 / 884 = 5.0%** | 0.9773 / 0.9655 | **85.20%** | **74 / 500 = 14.8%** | 0.6081 / 0.6121 |

Sources: [KR] §1.1–1.3; [Q3] §4.1, §5; [Q25] §4.1, §5. Nothing in these scores signals that their
support has collapsed. Saturation diagnostics ([Q3] §4.2–4.3, §6–7; [Q25] §4.2–4.3, §6–7):

| Diagnostic | Qwen3-4B SciQ | Qwen3-4B OBQA | Qwen2.5-7B SciQ | Qwen2.5-7B OBQA |
|---|---:|---:|---:|---:|
| $KDA_{cont}$ denominator | 40.46 / 884 (4.58%) | 86.22 / 500 (17.24%) | 44.97 / 884 (5.09%) | 76.26 / 500 (15.25%) |
| Items with $P(R^q{=}1)$ exactly 1.0 | **777 / 884 (87.9%)** | **257 / 500 (51.4%)** | **615 / 884 (69.6%)** | — |
| Items in middle band $[0.25, 0.90)$ | 0.23% | 3.40% | 1.02% | 4.20% |
| `prior_share` (with-material-correct already correct without) | **0.9558** | **0.8891** | — | — |
| Accuracy gain from the material (`KDA_small`: +43.10 pp) | **+4.41 pp** | — | **+4.75 pp** | — |
| `both_correct` bucket | **843 / 884 (95.4%)** | — | — | — |
| `wrong_to_correct` bucket (≠ denominator, which adds `both_wrong`) | **39 / 884 (4.41%)** | **51 / 500 (10.2%)** | 43 / 884 | 45 / 500 |

**The collapse generalises across solvers** ([Q25] §7–8). SciQ support is flat (4.64% → 4.98%),
while OBQA support shrinks (17.40% → 14.80%), tracking a +2.60 pp rise in $Acc_{wof}$. The two
solvers agree item-by-item on "known without the material" for **96.04%** (SciQ) and **85.40%**
(OBQA) of items.

**Single-solver degeneracy.** With $|M| = 1$, $KDA_{cont}$ degenerates exactly to
$P(R^{q+f}{=}1)$, so every single-solver KDA figure is that quantity ([KR] §1.3; [E1] §5).

### 2.2 $KDA_{cont}$ does not discriminate dependent from independent items

These results cover the **full, unannotated** 884 / 500 populations under the `KDA_small` ensemble
([FA] §4.1–4.2, Tables 3, 5, 7, 9, 10).

| | $r(KDA_{cont}, P(R^q{=}1))$ | $r(KDA_{cont}, P(R^{q+f}{=}1))$ | $r(KDA_{cont}, \text{gain})$ |
|---|---:|---:|---:|
| SciQ | **+0.075** (ρ +0.084) | **+0.863** (ρ +0.847) | +0.802 |
| OBQA | **+0.301** (ρ +0.370) | **+0.878** (ρ +0.878) | +0.684 |

A necessity metric should correlate *negatively* with zero-context answerability. Both coefficients
are positive, and the metric tracks **with-material** answerability instead.

| Evidence of insensitivity | SciQ | OBQA |
|---|---|---|
| Zero-context solve rate, highest vs lowest $KDA_{cont}$ decile (flat, non-monotone) | **41.6%** vs 51.1% | **42.0%** vs 16.0% |
| Mean $KDA_{cont}$, flagged vs unflagged items | **0.491** (n = 328) vs 0.460 (n = 556) | **0.287** (n = 80) vs 0.268 (n = 420) |
| Mean $KDA_{cont}$, `both_correct` vs split | **0.495** vs 0.471 | **0.348** vs 0.271 (+28% relative) |
| Flagged share of high-KDA tail (base rate 37.1%) | 45.5% of top decile; 51.4% at $KDA_{cont} \ge 0.70$ | — |

**Flagged pool** ([FA] §1; [FTM] §3). An item is flagged by any of four flags (`C1_kda_cont`:
$KDA_{cont} \ge 0.70$; `C1_kda_disc`: discrete proxy $= 1.0$; `C2_both_correct_conf`;
`C3_prior_dependent`), which the source groups as "three operational criteria". The pool is
**328 / 884 = 37.1%** (SciQ) and **80 / 500 = 16.0%** (OBQA). *Caveat:* the C1 flags select on KDA
itself, and 91 SciQ items entered through C1 alone ([FTM] §3.1), so the flagged rows are not fully
independent of the metric. The correlations and deciles use no flags.

**Counterfactual corroboration, SciQ** ([FA] Table 6; [RL] §5.1). The `prior_dependent` share is
**115 / 860 = 13.4%** → **101 / 830 = 12.17%** after the ex7 strict leakage exclusion ("No published
conclusion changes"). Their mean unadjusted $KDA_{cont}$ (**0.474**) is indistinguishable from the
split mean (**0.477**), while their adjusted score collapses to **0.208**. These means were not
recomputed after the exclusion.

**What the 408 flagged items are** ([FA] §2; manual taxonomy, **single annotator, no κ**):

| Failure category | SciQ n | % of split [95% CI] | OBQA n | % of split [95% CI] |
|---|---:|---|---:|---|
| `common_knowledge` | 182 | 20.6% [18.1, 23.4] | 41 | 8.2% [6.1, 10.9] |
| `parametric_knowledge` | 48 | 5.4% [4.1, 7.1] | 0 | 0.0% [0.0, 0.8] |
| `option_leakage` | 45 | 5.1% [3.8, 6.7] | 13 | 2.6% [1.5, 4.4] |
| `reasoning_shortcut` | 44 | 5.0% [3.7, 6.6] | 16 | 3.2% [2.0, 5.1] |
| `material_not_necessary` | 9 | 1.0% [0.5, 1.9] | 10 | 2.0% [1.1, 3.6] |
| **Total flagged** | **328** | **37.1% [34.0, 40.3]** | **80** | **16.0% [13.0, 19.5]** |

**The datasets leak through different channels** ([FA] §4.3, Tables 12–13). OBQA has a
significant longest-option bias (**160 / 427 = 37.5%** vs 25% chance, **z = 5.95**); SciQ does not
(184 / 695 = 26.5%, z = 0.90). On SciQ, the gold appears verbatim in the passage for 838 / 884
items, and 494 / 884 (55.9%) are near-verbatim cloze deletions.

### 2.3 The estimator's internals are not properties of the question

**Numerator contamination** ([FA] §4.2, Table 8). **29.8%** (SciQ) and **30.8%** (OBQA) of
aggregate $KDA_{cont}$ numerator mass comes from pairs **already correct without the material**. On
**256 / 884 (29.0%)** and **133 / 500 (26.6%)** items, such pairs contribute over half the numerator.

**Weight degeneracy.** The weight $1 - P(R^q{=}1)$ barely varies across items for some students:

| Simulated student | SciQ weight mean | SciQ weight sd ([FA] Table 11) | OBQA weight sd (recomputed†) |
|---|---:|---:|---:|
| `kda-albert-xlarge-v2-race` | 0.749 | **0.044** | **0.041** |
| `kda-scibert-uncased-race` | 0.718 | **0.091** | **0.099** |
| `kda-mpnet-base-race` | 0.591 | 0.256 | 0.279 |
| `t5-small-ssm-nq` | 0.616 | 0.397 | 0.327 |

† [FA] says "three of four" OBQA students are near-constant. Recomputing from
`results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json` gives **two of
four**, as on SciQ (corrections, row 2). At LLM scale the weight is near zero: Qwen3-4B's SciQ mean
is **0.0458**, and **827 / 884 = 93.6%** of items have $P(R^q{=}1) \ge 0.999$ ([E1] §4).

**Solver-level spread** ([KR] §2.2). Per-solver KDA on the same SciQ items spans **0.2737–0.8146**.
`kda-mpnet-base-race` alone gives **0.8146** / **0.4355** (SciQ / OBQA) against the ensemble's
**0.4715** / **0.2712**. The source calls it "the model a single-model ($|M| = 1$) shortcut would most
distort"; its mean weight is the lowest of the four (0.5906 / 0.6206).

### 2.4 Against independent human labels, $KDA_{cont}$ performs at chance

This is the only independently labelled evidence ([AR]). Two annotators rated SciQ items (100 on
Q1, 55 on Q2) without seeing any model output or score. The Q1 wording ([AG]):

> **Imagine a typical secondary-school student who has studied general science but has NOT studied
> this particular topic, and who has NOT read the passage.** They see only the question and the four
> options. **How likely are they to pick the correct answer?**

Ratings 1–2 = DEPENDENT and 3–4 = NOT_DEPENDENT ([AR] §B2, §D3; bootstrap, 2000 resamples):

| $KDA_{cont}$ vs human label | Annotator 1 (27 pos / 73) | Annotator 2 (16 / 84) | CI excludes chance? |
|---|---|---|---|
| AUC, unweighted [95% CI] | 0.580 [0.459, 0.701] | 0.569 [0.420, 0.718] | **Neither** |
| AUC, IPW (population-weighted) | **0.455** | **0.387** | — |
| Spearman ρ vs raw rating | 0.124 [−0.087, 0.305] | 0.157 [−0.038, 0.336] | **Neither** |

**Pre-registered primary test** ([AR] §B1, §D3). The sign test gives **41/80 = 0.512, p = 0.911**
and **40/80 = 0.500, p = 1.000**, which triggers the **INCONCLUSIVE** branch. The test had 0.94
power against a true rate of 70%, so this is weak evidence that no large difference exists, not
merely low power (§B1a).

**Q2 inversion** ([AR] §B4c). On Q2 (which option the passage points to), $KDA_{cont}$'s AUC is
**inverted** at **0.072 [0.000, 0.196]** / **0.020 [0.000, 0.065]**. With n = 5 negatives, the
source calls this "directionally informative, not a result to quote alone".

**Prior override** ([AR] §B4, §D4). Where a human confirms the perturbed passage states the
counterfactual answer, solvers still give the original gold on **40/200 = 20.0% [15.0, 26.1]** of
pairs, under both annotators.

**Caveats** ([AR] §D2, §B5, §B6.1, §B6.5). κ is **0.870** on the pre-registered 30-item block and
**0.680** on the full sample (below 0.70); the gate **PASSED** on the block. One more disagreement
would move the block κ to 0.714. Q1 is a proxy for prior answerability and "does not test
context-dependence at all". Context-dependence is covered by §2.2.

### 2.5 The adjusted estimators do not repair the metric

**$KDA_{adj}^{excl}$ is inert** ([CM] §6; [CO] §2.3; `results/ex2_counterfactual/adjusted_kda_corrected.json`).
As a retained-set mean it returns **0.4793 vs a 0.4767 baseline**, because dropped and kept items
score alike: AUC **0.553** (SciQ), 0.566 (OBQA `exact`), **0.346** (OBQA `partial`, inverted). The
zero-filled "retention" of **90.4%** (published) → **89.6%** (strict `unstable_other` convention;
[UO]; [SYN] §4.1 Addendum) reflects *how many* items are dropped, not *which*.

**Hard and soft forms** ([PS] §1.2; [UO]; [NH] §3.3 item 5). Means over the 860 eligible SciQ
items are $KDA_{cont}$ **0.4767**, $KDA_{adj}^{hard}$ **0.3357** and $KDA_{adj}^{soft}$ **0.2641**.
These shift the level but are validated against no human label. The `unstable_other` convention
moves hard retention by up to **29 pp** on OBQA, and no adjusted $KDA_{disc}$ exists.

**The Setting-C label they depend on is solver-dependent** ([E1] §3). On byte-identical passages
for 860 SciQ items, the encoder ensemble and Qwen3-4B agree at **0.412** raw agreement, **κ = 0.046**
(0.064 on the 776 items neither called `unstable_other`). Of the 669 items the ensemble certified
`context_dependent`, Qwen3-4B calls **392 (58.6%)** `prior_dependent`. The Qwen side was not
covered by the ex7 leakage rescan ([RL]).

**$KDA_{cont}$ against counterfactual labels** ([PS] §3; [CCA] §3.4, §11). Separating
`context_dependent` from `prior_dependent` items, the AUC is **0.5494** on SciQ (chance) → **0.5507**
leak-free (22 dropped), and **0.3550** on OBQA `partial` (inverted) → **0.4163** leak-free (56
dropped). The source says the OBQA figures are "depressed, not inflated" by construction defects,
and OBQA Setting C is now closed as a stress test: "every OBQA Setting-C figure in this repository
carries a caveat rather than a conclusion". These labels cannot rank metrics derived from them
([PS] §14), but $KDA_{cont}$ is not so derived.

### 2.6 With-material success is substantially prior-driven

The table gives the share of `both_correct` items classed `prior_dependent` ([CO] §3.2). SciQ
values are corrected via `results/ex7_counterfactual_construction/prior_dependence_corrected_sciq.json`;
OBQA was not rescanned.

| | SciQ, published → ex7 strict | OBQA `partial` | OBQA `exact` |
|---|---|---:|---:|
| Primary model (`kda-mpnet-base-race`) | **8.2%** (n = 415) → **6.7%** (27 / 401) | 46.2% (n = 52) | 16.7% (n = 12) |
| Encoder ensemble | **22.3%** (n = 381) → **21.5%** (79 / 368) | 41.5% (n = 41) | **50.0%** (n = 16) |

**Context-verified accuracy, primary model** ([CO] §3.3; [RL] §5.1). OBQA falls from 0.5793 to
0.3517 (**−16.6 points**). SciQ falls from 0.9058 to **0.8581** (published), or **0.8651** under ex7
strict. The published OBQA inflation is **4.2×** SciQ's 4.0 points, and the ensemble's OBQA
prior-dependence is roughly double SciQ's ([CO] §0, §3.2).

**OBQA coverage is limited** ([CO] §1, §3.1). Only **145 / 500 (29.0%)** OBQA items are eligible,
vs **860 / 884 (97.3%)** SciQ items. 67% of the eligible pool is low-fidelity `partial`, and
`unstable_other` is 37.2% (vs 6.4% on SciQ).

### 2.7 The source metric's own validation was already weakest here

This section uses the published validation (Moon et al., EMNLP 2022) as recorded in [KP]. **Table 4**
gives the Pearson r between human-measured KDA and the automatic metrics:

| | OBQA | TabMCQ | **SciQ** | All (pooled) |
|---|---:|---:|---:|---:|
| $KDA_{cont}$ | 0.73\*\* | 0.16 | **0.17** | 0.74\*\* |
| $KDA_{disc}$ | 0.71\*\* | 0.3\*\* | **0.05** | 0.8\*\* |

(\*\* p < 0.01; \* p < 0.05.) On SciQ **both correlations are non-significant**. The headline
0.74 / 0.80 figures are pooled.

- **Table 10** ($KDA_{cont}$ vs expert Likert). For **human-authored** questions: OBQA **−0.57**,
  TabMCQ −0.18, SciQ **0.06**, All **−0.01**. Generated-question rows ("All" column): KDDG 0.42\*,
  DG 0.61\*\*, QDG 0.64\*\*, DGen models 0.51\*\*. Both benchmarks here are human-authored.
- **Table 11.** SciQ's average with-material correctness was already **0.96** (OBQA 0.71, TabMCQ
  0.99).
- **Table 9.** Expert κ averaged **0.20** across 21 pairs among 7 teachers (SciQ 0.31, OBQA 0.18,
  TabMCQ 0.07).
- **The paper's own limitations name both mechanisms** ([KP] §4). §6.2: the core assumption "breaks
  down for questions answerable via the question stem alone or by ruling out topically irrelevant
  distractors". §6.3: a PLM may already "know" a fact from pretraining.

### 2.8 Competing explanations are measured and bounded

**Corpus defects** ([CD] §4, §6, §8). Confirmed topic mismatches are **12 / 884 = 1.36%** of SciQ
(a lower bound). OBQA's **0 / 500** rests on a 25-item sample of 123 zero-overlap items. Rater κ is
**0.857 at n = 15** (human-vs-LLM), and one disagreement moves it by ≈0.06. Answer-key noise is not
estimable: 0 mis-keys and 1 contested key in 55 items "cannot distinguish 'rare' from 'absent'".

**The MCQ format overstates unaided recall** ([FR] §9.2, §10.1–10.3). Without options, SciQ recall
bands are **[0.722, 0.804]** (Qwen2.5-7B, n = 884; MCQ $Acc_{wof}$ **0.9502**) and **[0.680, 0.800]**
(Qwen3-4B, 25-item pilot; **0.9536**). OBQA's with-material ceiling is only **[0.448, 0.660]**, so its
bands are uninterpretable. *Judge caveats:* the cross-judge κ is **0.545**, below the 0.70 gate; the
Qwen2.5-7B bands are self-judged; only `acc_normalised` is headline-licensed (SciQ **0.535**); and
the human-vs-Qwen3-4B-judge gate passed at κ **0.754**.

**Counterfactual construction artifacts.** **17/57 = 29.8%** of `prior_dependent` pairs in the
55-item human sample are failed perturbations ([AR] §B4b, §D6). At corpus level, **30 / 860 = 3.5%**
of SciQ passages still state the gold ([RL] §3). This limits how precisely the counterfactual audit
is calibrated, but not §2.2, which rests on unannotated full populations.

**Prompting does not restore dynamic range** ([PSR] §3.1, §3.3, §7). Under Qwen3-4B, isolated
persona roleplay spreads SciQ accuracy by only **0.036**, narrower than its own CIs. Its joint-prompt
gaps are an emission-order artifact: reversing the order moves SciQ **+0.342 → +0.639** and OBQA
**−0.228 → +0.408**. Under Qwen2.5-7B the isolated spread is **0.033**. The last-emitted tier breaks
consensus on **44.0%** of SciQ items as beginner vs **4.4%** as advanced, but reversal *shrinks*
the SciQ gap (+0.543 → +0.369).

---

## 3. Interpretation

**The estimator conditions on an event that has almost vanished.** KDA is conditional on the
solver *not* knowing the answer, which at modern-LLM scale covers 4.6–5.0% of SciQ (§2.1). Nothing
in the output signals this. With a flat or near-zero weight (§2.3), the score collapses toward
$P(R^{q+f}{=}1)$, as the correlations show (§2.2). At $|M| = 1$ the score *is* $P(R^{q+f}{=}1)$.

**Insensitivity is harder to fix than inversion.** An inverted metric could be recalibrated; an
insensitive one carries no recoverable signal. Material-independent items score *above* average
(§2.2). Dropping independently identified prior-dependent items moves the score by 0.0026 (§2.5).

**The blind spots are one-sided and grow with capability.** Both invisible buckets sit on the
$r^q = 1$ side. `both_correct` holds 843 / 884 SciQ items under Qwen3-4B. `correct_to_wrong` is
0 (SciQ) / 4 (OBQA) under Qwen3-4B and 1 / 9 under Qwen2.5-7B ([Q3] §4.3; [Q25] §4.3). A metric of
whether the material *helps* cannot see material that *harms*.

**Human agreement does not rescue the construct** (§2.4). The result matches the paper's
non-significant SciQ correlations (0.05 / 0.17) and its near-zero correlation on human-authored
items (§2.7). It **confirms at modern scale a validity gap the metric already had**. The one clean
signal, `wrong_to_correct`, is rewarded as it should be. But those items are nearly all that
remains, and they are identified by the bucket assignment, not by the score.

---

## 4. Limitations

- **No human-subject KDA.** A comparison with human-measured KDA exists only in the source paper's
  data (§2.7).
- **Failure taxonomy.** It is single-annotator, single-pass, with no κ ([FA] §5.1); §2.2–2.3 do not
  depend on it. With C3 later applied, the OBQA pool is **100 / 500 = 20.0%**, with 20 new items
  labelled by a second annotator ([FTM] §6.1). §2.2 uses the original pool.
- **Human study.** SciQ only, 100 + 55 items, translation-assisted. The full-sample κ is 0.680
  (the gate passed on the 0.870 block), and Q1 is a prior-answerability proxy.
- **Corpus-defect κ** is human-vs-LLM at n = 15, and the SciQ mismatch rate is a lower bound.
- **Solver configuration.** Each modern solver was run in one configuration, with no option-order
  debiasing or calibration. Support collapse follows from $Acc_{wof}$ alone, but exact KDA values
  would shift.
- **Counterfactual labels.** The ex7 correction covers the SciQ encoder solvers only; OBQA and
  Qwen3-4B Setting-C figures are uncorrected. The 30 strict-leak items are a lexical lower bound
  ([RL] §6).
- **Cross-dataset KDA** is not on a common scale. The SciQ–OBQA gap (0.4715 vs 0.2712) is largely
  an extractive-vs-deductive mixture effect ([KR] §2.1).
- **Verbatim containment** has two measurements, not merged: 820 / 884 (92.8%) SciQ and 47 / 500
  (9.4%) OBQA ([KR] §2.1), versus 838 / 884 (94.8%) and 57 / 500 (11.4%) under a different
  criterion ([FA] §4.3).
- **Scope.** All relationships are correlational, and the evidence covers only two English
  science-MCQ benchmarks.

---

## 5. Conclusion

Modern LLM-based evaluators of the KDA family **do not reliably measure** whether an MCQ genuinely
requires its target material, on either SciQ or OpenBookQA. Four independent lines of evidence
converge:

1. **Support collapses.** Modern solvers answer 95.36% / 95.02% of SciQ and 82.60% / 85.20% of OBQA
   without the material. $KDA_{disc}$ then rests on 4.6–5.0% of SciQ and 14.8–17.4% of OBQA, yet
   still reports confident scores (0.9512 / 0.9773 on SciQ).
2. **The score does not discriminate.** $KDA_{cont}$ correlates positively with zero-context
   answerability (+0.075 / +0.301) and strongly with with-material answerability (+0.863 / +0.878).
   Its top decile is solved without the material 41.6% / 42.0% of the time.
3. **Its internals are not properties of the question.** About 30% of numerator mass comes from
   already-solved pairs. The ignorance weight is flat for half the encoder students and near zero
   at LLM scale. Per-solver scores on identical items span 0.2737–0.8146.
4. **It does not agree with humans.** Unweighted AUCs are 0.580 / 0.569, and every CI contains
   chance. Population-weighted AUCs are 0.455 / 0.387, and the pre-registered test is inconclusive.

The adjusted estimators do not repair this. The exclusion form is inert (0.4793 vs 0.4767), and the
others inherit a Setting-C label on which two solvers agree at κ = 0.046. Corpus defects (≥ 1.36%),
answer-key noise and construction artifacts are too small or too localised to explain the effect.
The failure is **insensitivity rather than inversion**. It worsens as solvers become more capable,
and it confirms at modern-LLM scale a validity gap already visible in the source paper's own
non-significant SciQ correlations.

---

## Sources

- **ex1:** [KR] reproduction summary · [Q3] Qwen3-4B eval · [Q25] Qwen2.5-7B eval.
- **ex2:** [CM] counterfactual methodology · [CO] OBQA counterfactual analysis · [E1] LLM-scale
  Setting C · [UO] `unstable_other` convention.
- **ex3–5:** [PSR] persona simulation · [FR] free response · [FA] failure analysis · [FTM] taxonomy
  methodology · [CD] corpus-defect audit.
- **ex6–8:** [PS] P/S/F/D formulation · [CCA] construction audit · [RL] leakage rescan ·
  [AR] annotation results · [AG] annotation guide.
- **Other:** [KP] KDA paper documentation · [NH] next-phase handoff · [SYN] earlier RQ1 synthesis.

[KR]: ../ex1_reproduce_KDA/kda_reproduction_summary.md
[Q3]: ../ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md
[Q25]: ../ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md
[CM]: ../ex2_counterfactual/counterfactual_experiment_methodology.md
[CO]: ../ex2_counterfactual/counterfactual_obqa_analysis.md
[E1]: ../ex2_counterfactual/e1_counterfactual_llm_scale.md
[UO]: ../ex2_counterfactual/unstable_other_convention.md
[PSR]: ../ex3_student_simulation/student_persona_simulation_report.md
[FR]: ../ex4_free_response/plan_option_free_response_experiment.md
[FA]: ../ex5_failure_audit/rq1_test_split_failure_analysis.md
[FTM]: ../ex5_failure_audit/failure_taxonomy_methodology.md
[CD]: ../ex5_failure_audit/corpus_defect_audit.md
[PS]: ../ex6_psfd_score/psfd_formulation.md
[CCA]: ../ex7_counterfactual_construction/counterfactual_construction_audit.md
[RL]: ../ex7_counterfactual_construction/residual_leakage_rescan.md
[AR]: ../ex8_independent_labels/rq2_annotation_results.md
[AG]: ../ex8_independent_labels/ANNOTATION_GUIDE_RQ2.md
[KP]: ../papers/KDA_Paper_Documentation.md
[NH]: ../NEXT_PHASE_HANDOFF.md
[SYN]: ../synthesis/rq1_research_synthesis.md
