# RQ1 — Conclusion

> **Do modern LLM-based evaluators reliably measure whether a multiple-choice question genuinely
> requires the target learning material?**

**Written:** 2026-09-25. **Evidence base:** all eight experiments in `docs/`, plus the source
metric's own validation data as recorded in [`docs/papers/KDA_Paper_Documentation.md`](../papers/KDA_Paper_Documentation.md).

**Scope.** This report assesses the **KDA family** of automated evaluators: $KDA_{disc}$,
$KDA_{cont}$, and the three adjusted estimators $KDA_{adj}^{hard}$, $KDA_{adj}^{soft}$,
$KDA_{adj}^{excl}$. Other evaluator prototypes examined elsewhere in this repository (the P/S/F/D
score, the Setting-C class label, LLM-as-judge) are **not** assessed here as evaluators; they appear
only where they supply a *measurement of* a KDA-family metric, or a caveat on such a measurement.

**Relation to prior work in this repository.** [`docs/synthesis/rq1_research_synthesis.md`](../synthesis/rq1_research_synthesis.md)
reached the same verdict from five reports. This report is standalone and adds four evidence sources
that postdate it: independent human labels (ex8), the circularity of Setting-C-derived evaluation
(ex6), a second saturated solver (ex1, Qwen2.5-7B), and open-ended recall measurement (ex4).

**Terminological note, load-bearing throughout.** Every $KDA_{disc}$ value in this repository is a
**model-ensemble proxy**, not the paper's human-student quantity. No human-subject KDA measurement
exists anywhere in this evidence base
([`rq1_test_split_failure_analysis.md`](../ex5_failure_audit/rq1_test_split_failure_analysis.md) §5.5;
[synthesis Appendix](../synthesis/rq1_research_synthesis.md)).

---

## 1. Verdict

**No.** On the two benchmarks studied (SciQ test, 884 items; OpenBookQA test, 500 items), the KDA
family does not reliably measure whether a question requires its target material.

The failure mode is **insensitivity, not inversion** — the metric does not systematically score
material-independent questions *lower*; it becomes numerically **blind** to them. Four independent
lines of evidence converge:

| Condition a reliable instrument must satisfy | Result | § |
|---|---|---|
| Non-degenerate support | **Fails.** Usable support collapses to 4.6–5.0% of SciQ at modern-LLM scale, on two different solvers | §3 |
| Discrimination between dependent and independent items | **Fails.** $r(KDA_{cont}, P(R^q{=}1)) = +0.075$ SciQ; zero-context solve rate is flat across KDA deciles | §4 |
| Verdict independent of the solver | **Not established; contrary evidence.** Per-solver KDA on the same SciQ items spans 0.2737–0.8146; the ignorance weight is near-constant for most solvers | §5 |
| Agreement with independent human judgement | **Fails.** Every $KDA_{cont}$ interval against human labels contains chance, under both annotators | §6 |

The adjusted estimators built to repair the metric do not repair it (§7). The source paper's own
validation was already weakest on precisely these corpora, and named both mechanisms as limitations
in 2022 (§8). Corpus defects and answer-key noise are measured and cannot account for the effect
(§9).

---

# Part I — Empirical findings

Sections 3–9 report what was measured. All causal and evaluative reading is deferred to §10.

## 3. Finding 1 — The metric's support collapses, and it generalises across solvers

$KDA_{disc}$'s denominator is $\sum_i (1 - r_i^q)$, the count of questions the solver answers
**wrong without the material**. Questions the solver already knows contribute zero to both numerator
and denominator and are silently dropped.

Zero-context accuracy ($Acc_{wof}$) and the resulting usable support:

| Solver | SciQ $Acc_{wof}$ | SciQ $KDA_{disc}$ denominator | OBQA $Acc_{wof}$ | OBQA $KDA_{disc}$ denominator |
|---|---:|---:|---:|---:|
| `KDA_small` ensemble, pooled vote (4 encoders) | 45.48% | — | 34.80% | — |
| Qwen3-4B-Instruct-2507 | **95.36%** | **41 / 884 = 4.6%** | **82.60%** | **87 / 500 = 17.4%** |
| Qwen2.5-7B-Instruct | **95.02%** | **44 / 884 = 5.0%** | **85.20%** | **74 / 500 = 14.8%** |

Sources: [`kda_reproduction_summary.md`](../ex1_reproduce_KDA/kda_reproduction_summary.md) §1.1–1.3;
[`kda_qwen3_4b_evaluation_report.md`](../ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md) §4.1, §5;
[`kda_qwen2.5_7b_evaluation_report.md`](../ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md) §4.1, §5.

The scores reported from that support are high and carry no signal that the support collapsed:
$KDA_{disc} = 0.9512$ (SciQ) / $0.5862$ (OBQA) for Qwen3-4B, and $0.9773$ / $0.6081$ for
Qwen2.5-7B. The continuous form does not help — its denominator is $\sum_i (1 - P(R_i^q{=}1))$,
measured at **40.46 / 884 (4.58%)** and **86.22 / 500 (17.24%)** for Qwen3-4B, and **44.97 / 884
(5.09%)** and **76.26 / 500 (15.25%)** for Qwen2.5-7B.

**The collapse is not solver-specific.** This is the part the earlier five-report synthesis could not
establish, since it rested on one modern solver. With a second, larger model the direction is mixed
rather than uniform: SciQ's support is essentially flat (4.64% → 4.98%) while OBQA's shrinks further
(17.40% → 14.80%), tracking an OBQA $Acc_{wof}$ rise of +2.60 pp
([`kda_qwen2.5_7b_evaluation_report.md`](../ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md) §7).
Item-level agreement between the two solvers on whether the answer is known without the material is
**96.04%** on SciQ and **85.40%** on OBQA (same document, §8).

**Supporting observations.**

- $KDA_{cont} \approx KDA_{disc}$ once saturated: 0.9407 vs 0.9512 (SciQ) and 0.5956 vs 0.5862
  (OBQA) for Qwen3-4B; 0.9655 vs 0.9773 and 0.6121 vs 0.6081 for Qwen2.5-7B.
- The probability distribution is bimodal, not graded. **777 / 884 SciQ samples (87.9%)** have
  $P(R^q{=}1)$ exactly `1.0` in float32 under Qwen3-4B; **615 / 884 (69.6%)** under Qwen2.5-7B,
  whose middle bands $[0.25, 0.90)$ hold **1.02%** of SciQ and **4.20%** of OBQA.
- `prior_share` — the fraction of with-material-correct answers already correct without it — is
  **0.9558** (SciQ) and **0.8891** (OBQA) under Qwen3-4B.
- The accuracy gain from supplying the material falls from **+43.10 pp** (`KDA_small` pooled vote,
  SciQ) to **+4.41 pp** (Qwen3-4B) and **+4.75 pp** (Qwen2.5-7B).
- Under $|M| = 1$ the ignorance weight cancels between numerator and denominator and $KDA_{cont}$
  degenerates exactly to $P(R^{q+f}{=}1)$
  ([`kda_reproduction_summary.md`](../ex1_reproduce_KDA/kda_reproduction_summary.md) §1.3;
  [`e1_counterfactual_llm_scale.md`](../ex2_counterfactual/e1_counterfactual_llm_scale.md) §5). Every
  single-solver KDA figure in this repository is therefore that degenerate quantity.

## 4. Finding 2 — $KDA_{cont}$ does not discriminate material-dependent from material-independent items

Measured on the **full, unannotated** 884 / 500 populations with the `KDA_small` ensemble
([`rq1_test_split_failure_analysis.md`](../ex5_failure_audit/rq1_test_split_failure_analysis.md)
§4.1–4.2, Tables 3, 5, 7, 9, 10):

| | $r(KDA_{cont}, P(R^q{=}1))$ | $r(KDA_{cont}, P(R^{q+f}{=}1))$ | $r(KDA_{cont}, \text{gain})$ |
|---|---:|---:|---:|
| SciQ | **+0.075** (ρ +0.084) | **+0.863** (ρ +0.847) | +0.802 |
| OBQA | **+0.301** (ρ +0.370) | **+0.878** (ρ +0.878) | +0.684 |

A necessity metric should correlate *negatively* with zero-context answerability. Both coefficients
are positive, and the metric's strong association is with **with-material** answerability.

Zero-context solve rate by $KDA_{cont}$ decile is "flat and non-monotone" (Table 10). The **highest**
decile is still solved without the material **41.6%** of the time on SciQ and **42.0%** on OBQA;
the lowest decile sits at 51.1% and 16.0%.

Items independently flagged as answerable without the material score **higher**, not lower:

| | flagged mean $KDA_{cont}$ | unflagged mean |
|---|---:|---:|
| SciQ | **0.491** (n = 328) | 0.460 (n = 556) |
| OBQA | **0.287** (n = 80) | 0.268 (n = 420) |

The `both_correct` bucket — questions that demonstrably did not need the material — carries a mean
$KDA_{cont}$ of **0.495** against a split mean of 0.471 on SciQ, and **0.348** against 0.271 on OBQA
(a +28% relative premium). The high-KDA tail is **enriched** for flagged items: 45.5% in the SciQ top
decile and 51.4% at $KDA_{cont} \ge 0.70$, against a 37.1% base rate.

Flagged pool construction: the union of four detection criteria yields **328 / 884 = 37.1%** (SciQ)
and **80 / 500 = 16.0%** (OBQA) items independently confirmed answerable without the material by at
least one criterion (same document, §1). Independent corroboration on SciQ from the counterfactual
probe: **115 / 860 = 13.4%** classified `prior_dependent`, whose mean unadjusted $KDA_{cont}$
(**0.474**) is statistically indistinguishable from the split mean (**0.477**), while their adjusted
score collapses to **0.208** (same document, Table 6).

## 5. Finding 3 — The estimator's internals are dominated by quantities that are not properties of the question

Three measurements, all on the full populations
([`rq1_test_split_failure_analysis.md`](../ex5_failure_audit/rq1_test_split_failure_analysis.md)
§4.2, Tables 8 and 11):

**Numerator contamination.** **29.8%** (SciQ) and **30.8%** (OBQA) of the aggregate $KDA_{cont}$
numerator mass is contributed by student–question pairs that were **already correct without the
material**. On **256 / 884 (29.0%)** SciQ and **133 / 500 (26.6%)** OBQA items, more than half the
numerator contribution comes from already-solved pairs.

**Weight degeneracy.** The ignorance weight $1 - P(R^q{=}1)$ is nearly constant for most simulated
students, so it cannot differentiate questions:

| Simulated student (SciQ) | weight mean | weight **sd** |
|---|---:|---:|
| `kda-albert-xlarge-v2-race` | 0.749 | **0.044** |
| `kda-scibert-uncased-race` | 0.718 | **0.091** |
| `kda-mpnet-base-race` | 0.591 | 0.256 |
| `t5-small-ssm-nq` | 0.616 | 0.397 |

Two of four students on SciQ, and three of four on OBQA, contribute a weight that is nearly constant
across all items.

**Solver-level spread.** Reporting a single strong member in place of the ensemble moves the score
substantially: `kda-mpnet-base-race` alone gives SciQ KDA **0.8146** and OBQA **0.4355**, against the
ensemble's **0.4715** and **0.2712**. The source identifies it as "the model a single-model
($|M| = 1$) shortcut would most distort", and notes that its own mean weight is the lowest of the
four (0.5906 SciQ / 0.6206 OBQA) because the ignorance weight shrinks as prior accuracy rises
([`kda_reproduction_summary.md`](../ex1_reproduce_KDA/kda_reproduction_summary.md) §2.2).

## 6. Finding 4 — Against independent human labels, $KDA_{cont}$ performs at chance

[`rq2_annotation_results.md`](../ex8_independent_labels/rq2_annotation_results.md) is the only
independently-labelled evidence in this project: human judgements made from the question, options and
passage with no model output, no score and no prior label in view. SciQ only, 100 items rated (Q1)
plus 55 (Q2), two annotators.

The literal human question (Q1), which defines the reference
([`ANNOTATION_GUIDE_RQ2.md`](../ex8_independent_labels/ANNOTATION_GUIDE_RQ2.md)):

> **Imagine a typical secondary-school student who has studied general science but has NOT studied
> this particular topic, and who has NOT read the passage.** They see only the question and the four
> options. **How likely are they to pick the correct answer?**

Rated 1–4 and binarised as 1–2 = DEPENDENT / 3–4 = NOT_DEPENDENT.

**$KDA_{cont}$ against the binarised human label** (§B2, §D3; bootstrap CIs, 2000 resamples):

| | Annotator 1 (27 positives / 73) | Annotator 2 (16 / 84) | CI excludes chance? |
|---|---|---|---|
| AUC | 0.580 [0.459, 0.701] | 0.569 [0.420, 0.718] | **Neither** |
| Spearman ρ vs raw rating | 0.124 [−0.087, 0.305] | 0.157 [−0.038, 0.336] | **Neither** |

**Pre-registered primary test** (§B1, §D3). The sign test comparing rank-normalised distance to the
human ordering returns **41/80 = 0.512, p = 0.911** (annotator 1) and **40/80 = 0.500, p = 1.000**
(annotator 2): "No metric is closer to the human ordering than any other at this sample size." The
pre-registered **INCONCLUSIVE** branch was triggered, under which nothing is adopted. The document
notes this is not primarily a power failure — at *m* = 80 the test had 0.94 power against a true rate
of 70% — but "weak evidence that no large difference exists" (§B1a).

**A second human-grounded reading, pointing the same way** (§B4c). On Q2, where annotators state
which option *the passage* points to, $KDA_{cont}$'s AUC is **inverted at 0.072 [0.000, 0.196]**
(annotator 1) and **0.020 [0.000, 0.065]** (annotator 2). The source attaches its own caveat: the
negative class is **n = 5**, so this is "directionally informative, not a result to quote alone."

**Two caveats travelling with these figures, both stated in the source.**

1. The inter-annotator gate — Cohen's κ on the collapsed binary over the pre-registered 30-item block
   — stands at **0.870**, against a **full-sample κ of 0.680** (100 items), below the 0.70 threshold.
   The gate **PASSED** (§D2, §B6.1). The block's positive class is 5 items; one further
   disagreement would move κ to 0.714.
2. Q1 asks a human to judge unaided answerability, which is a human proxy for Setting A — i.e. for
   the prior term — and the document states plainly that "Q1 does not test context-dependence at
   all" (§B5, §B6.5). The Q1 result therefore bears on whether $KDA_{cont}$ tracks *human-judged
   prior answerability*, which it does not; §4's population evidence is what speaks to
   context-dependence.

**One further independent comparison.** Where a human confirms the perturbed passage states the
counterfactual answer, solvers still answer the original gold on **40/200 = 20.0% [15.0, 26.1]** of
(model, question) pairs — identical under both annotators (§B4, §D4). The source calls this "the
cleanest evidence for the RQ1 thesis anywhere in this project."

## 7. Finding 5 — The adjusted estimators do not repair the metric

$KDA_{adj}^{excl}$ was built to drop prior-dependent questions from the estimate. Read as specified —
a mean over the retained set — it is **inert**: **0.4793 against a 0.4767 baseline**, because the
items it drops score the same as the items it keeps (**AUC 0.553** on SciQ; 0.566 on OBQA `exact`;
**0.346, inverted,** on OBQA `partial`). The previously published "90.4% retention" is the
zero-filled form, which is dominated by *how many* items are dropped rather than *which*
([synthesis §4.1 Addendum](../synthesis/rq1_research_synthesis.md);
[`counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) §6).

Means over the 860 counterfactual-eligible SciQ items: $KDA_{cont}$ **0.4767**,
$KDA_{adj}^{hard}$ **0.3357**, $KDA_{adj}^{soft}$ **0.2641**
([`psfd_formulation.md`](../ex6_psfd_score/psfd_formulation.md) §1.2). The hard and soft forms shift
the level but are not independently validated against any human label.

The treatment of the `unstable_other` class is a convention, not a measurement, and the choice moves
hard retention by up to **29 pp** on OBQA; the strict convention was adopted by decision
([`unstable_other_convention.md`](../ex2_counterfactual/unstable_other_convention.md)).

**All three adjusted forms depend on the Setting-C class label, and that label is a property of the
solver rather than of the question.** Scoring the same 860 eligible SciQ items on byte-identical
passages, the encoder ensemble and Qwen3-4B agree on the three-way class at raw agreement **0.412**,
**Cohen's κ = 0.046** (0.064 restricted to the 776 items neither called `unstable_other`); of the 669
items the encoder ensemble certified `context_dependent`, Qwen3-4B calls **392 (58.6%)**
`prior_dependent` ([`e1_counterfactual_llm_scale.md`](../ex2_counterfactual/e1_counterfactual_llm_scale.md) §3).
This is a measurement of the Setting-C label, not of $KDA_{cont}$; it bears on the adjusted
estimators because they are defined by that label.

**A measurement of $KDA_{cont}$ from the metric-comparison work.** Separating counterfactually
confirmed `context_dependent` from `prior_dependent` items, $KDA_{cont}$'s AUC is **0.5494** on SciQ
(at chance) and **0.3550** on OBQA `partial` (inverted)
([`psfd_formulation.md`](../ex6_psfd_score/psfd_formulation.md) §3). That document's own §14
establishes that this comparison cannot be used to *rank* candidate metrics derived from the same
counterfactual labels, because such evaluation "measures proximity to its own definition" — but
$KDA_{cont}$ is not derived from those labels, so its at-chance result on a reference it did not
define stands as a measurement of $KDA_{cont}$.

## 8. Finding 6 — The source metric's own validation was already weakest on these corpora

From the original metric's published validation, as recorded in
[`KDA_Paper_Documentation.md`](../papers/KDA_Paper_Documentation.md) (Moon et al., EMNLP 2022):

**Table 4 — Pearson correlation between human-measured KDA and the automatic metrics:**

| | OBQA | TabMCQ | **SciQ** | All (pooled) |
|---|---:|---:|---:|---:|
| $KDA_{cont}$ | 0.73\*\* | 0.16 | **0.17** | 0.74\*\* |
| $KDA_{disc}$ | 0.71\*\* | 0.3\*\* | **0.05** | 0.8\*\* |

(\*\* p < 0.01; \* p < 0.05.) On SciQ — the corpus this project stresses hardest — **both
correlations are non-significant**. The headline 0.74 / 0.80 is the pooled column.

**Table 10 — $KDA_{cont}$ vs expert Likert, split by question provenance:** for **human-authored**
questions the correlation is weak or negative (OBQA **−0.57**, TabMCQ −0.18, SciQ **0.06**, All
**−0.01**), while generated-question rows reach 0.51–0.64. Both benchmarks in this project are
human-authored.

**Table 11 — average correctness at the time of publication:** SciQ average with-material
correctness was already **0.96** (OBQA 0.71, TabMCQ 0.99) with 2022-era solvers. The ignorance
baseline the metric is defined against was already thin before modern LLMs.

**Table 9 — expert inter-rater agreement:** Cohen's κ averaged **0.20** across 21 pairwise
coefficients among 7 teachers (SciQ 0.31, OBQA 0.18, TabMCQ 0.07), with the source noting agreement
was higher (κ > 0.3–0.4) only for identifying clearly *bad* questions.

**The paper names both failure mechanisms in its own Limitations** (§4 of the documentation):

- §6.2, *Too Easy Questions*: "The core assumption — that a well-designed MCQ is answerable *iff*
  the student knows the fact — breaks down for questions answerable via the question stem alone or by
  ruling out topically irrelevant distractors, regardless of actual fact knowledge."
- §6.3, *Difficulty of Measuring PLM's Ignorance*: the authors "cannot guarantee a given PLM doesn't
  already 'know' a fact from pretraining (contamination), which would confound the 'without fact'
  baseline condition."

## 9. Finding 7 — Competing explanations, measured and bounded

**Corpus defects cannot carry the effect.** A census of all 1,384 test items with a lexical
question–passage overlap screen, followed by hand adjudication, puts confirmed topic mismatches at
**12 / 884 = 1.36%** on SciQ and **0 / 500** on OBQA
([`corpus_defect_audit.md`](../ex5_failure_audit/corpus_defect_audit.md) §8.1, §4). Caveats from the
source: the double-annotation κ is **0.857 at n = 15** with observed agreement 0.933, one of the two
raters is an LLM agent (so this is human-vs-LLM, not human-vs-human), one disagreement moves κ by
≈0.06, and the rate is a **lower bound** because items above the 0.40 overlap threshold were sampled
(20 items, 0 mismatches) rather than censused.

**Answer-key noise was not estimable and was dropped.** Across 55 items read in a prior round there
were **0 unambiguous mis-keys and 1 contested key**; the source records that estimating a rate near
2% to ±5 pp needs n ≈ 30 but that such an interval is compatible with 0%, so the estimate "cannot
distinguish 'rare' from 'absent'" and was deliberately abandoned
([`corpus_defect_audit.md`](../ex5_failure_audit/corpus_defect_audit.md) §6).

**The zero-context quantity is measured in a format that overstates unaided recall.** Stripping the
options and grading open-ended answers, the recall band on SciQ is **[0.722, 0.804]** at n = 884
(Qwen2.5-7B) against an MCQ $Acc_{wof}$ of **0.9502**, and **[0.680, 0.800]** in the 25-item Qwen3-4B
pilot against **0.9536**
([`plan_option_free_response_experiment.md`](../ex4_free_response/plan_option_free_response_experiment.md)
§9.2, §10.2). Caveat from the source: these bands rest on an LLM judge whose cross-model agreement is
**κ = 0.545**, below the project's pre-registered 0.70 gate, so they are diagnostic rather than
headline figures (same document, §10.1, §10.3). On OpenBookQA the corresponding bands are not
interpretable as knowledge at all, because the with-material ceiling is itself only **[0.448, 0.660]**
(§10.2).

**The counterfactual audit reference is itself partly defective.** Roughly **29.8% (17/57)** of
sampled `prior_dependent` labels on SciQ are the perturbation having failed rather than a solver
overriding its context, confirmed rater-free on all five checked items
([`rq2_annotation_results.md`](../ex8_independent_labels/rq2_annotation_results.md) §B4b, §D6). This
bounds how precisely the *audit* in §4 and §7 can be calibrated; it is not a defence of
$KDA_{cont}$, whose insensitivity in §4 rests on the full unannotated populations.

**Mitigation by prompting does not restore dynamic range.** Persona-conditioned zero-context
simulation produces a total accuracy spread of **0.036** on SciQ under isolated roleplay — narrower
than its own confidence intervals — while the large spreads produced by joint prompting are an
emission-order artifact: reversing the tier order moves the SciQ gap from **+0.342 to +0.639** and
OBQA's from **−0.228 to +0.408** under Qwen3-4B
([`student_persona_simulation_report.md`](../ex3_student_simulation/student_persona_simulation_report.md)
§3.1, §3.3). Under Qwen2.5-7B the same mechanism replicates (the last-emitted tier breaks an
established consensus on 44.0% of SciQ items against 4.4% when emitted first) while its effect on the
reported gap does not (§7.3 of the same document).

---

# Part II — Interpretation

## 10. Why the failure takes this form

*This section is interpretation. The measurements it draws on are in §3–§9.*

**The estimator conditions on a quantity that no longer exists.** $KDA$ is defined as
$P(R^{q+f}{=}1 \mid R^q{=}0)$ — it is a conditional on the solver *not* knowing the answer. That
conditioning event is the instrument's entire measurement aperture. As solver capability rises the
event becomes rare, so the aperture closes: 4.6–5.0% of SciQ at modern-LLM scale (§3). Nothing in the
metric's output signals that this has happened; a denominator of 41 still divides and still returns a
plausible-looking score.

**What remains is not a property of the question.** Once the weight is near-constant (§5) the ratio
collapses toward a weighted mean of $P(R^{q+f}{=}1)$, which is what the correlations show: +0.863 and
+0.878 against with-material answerability, versus +0.075 and +0.301 against the quantity the metric
is supposed to be sensitive to (§4). Two components therefore dominate the score — how answerable the
question is *with* the material, and how capable the simulated student is — and neither is the
property RQ1 asks about. The $|M| = 1$ degeneracy is the limiting case: with a single solver the
metric is exactly $P(R^{q+f}{=}1)$ under a different name.

**Insensitivity, not inversion — and this is the harder failure.** An inverted metric could be
repaired by a sign change or a recalibration. An insensitive one carries no recoverable signal:
material-independent items score *higher* than the rest (0.491 vs 0.460 on SciQ, §4), the top decile
is enriched for them (45.5% against a 37.1% base rate), and an estimator that drops them on
independent evidence changes the score by 0.0026 (§7). The distribution of scores and the
distribution of necessity are close to orthogonal.

**The blind spots are one-sided, so the problem compounds with capability.** Both invisible buckets
sit on the $r^q = 1$ side: `both_correct` (843 of 884 SciQ items under Qwen3-4B — pure parametric
prior) and `correct_to_wrong` (4 items under Qwen3-4B, 9 under Qwen2.5-7B — cases where the material
actively *hurt*). A metric that measures whether material *helps* cannot see material that *harms*,
and as $Acc_{wof}$ rises that side of the table absorbs the corpus.

**Human agreement does not rescue the construct.** The one independently-labelled test returns
intervals containing chance under both annotators (AUC 0.580 / 0.569; ρ 0.124 / 0.157), and the
pre-registered primary comparison is inconclusive at a sample size that had 0.94 power against a
large effect (§6). This is weak evidence of absence rather than absence of evidence — and it is
consistent with the source paper's own non-significant SciQ correlations (0.05 and 0.17) and its
near-zero correlation on human-authored items (§8). The finding is best read as a *confirmation at
modern scale of a validity gap the metric already had*, not as a new defect introduced by LLMs.

**What does still carry signal.** The `wrong_to_correct` bucket is a clean positive: items the solver
fails without the material and answers correctly with it — **39 / 884 (SciQ)** and **51 / 500
(OBQA)** under Qwen3-4B; **43 / 884** and **45 / 500** under Qwen2.5-7B (§3 sources). These are
exactly the items a knowledge-dependency metric should reward, and the metric does reward them. The
problem is not that it misjudges them; it is that they are all that remains, and they are identified
by the bucket assignment itself rather than by the score.

**What a reliable instrument would have to demonstrate.** On this evidence: a support set that does
not vanish as solvers improve; a negative association with zero-context answerability rather than a
positive one; stability of the per-question verdict under a change of solver; and agreement with
human judgement on labels not derived from the metric's own machinery. No KDA-family estimator
assessed here is shown to satisfy all four on these two benchmarks. $KDA_{disc}$ and $KDA_{cont}$
fail the first two directly (§3, §4) and the fourth where it has been tested (§6). The adjusted forms
inherit the same support collapse, and what discrimination they have comes from the Setting-C label
rather than from KDA — a label that is itself solver-dependent and, in the one sample checked against
humans, roughly 30% construction artifacts (§7, §9).

---

## 11. What this evidence does not establish

- **No human-subject KDA exists in this project.** Every $KDA_{disc}$ figure is a model-ensemble
  proxy. The comparison against *human-measured* KDA exists only in the source paper's own data (§8).
- **The 408-item failure taxonomy is single-annotator, single-pass, with no κ.** Its four-way
  category split is the most annotator-sensitive part of the evidence base. The population-level
  results in §4–§5 do not depend on those labels and rest on the full 884 / 500 populations.
- **The only inter-annotator κ on corpus defects is human-vs-LLM at n = 15** (κ = 0.857), and the
  overturned label was the LLM rater's. The SciQ mismatch rate of 1.36% is a lower bound.
- **The independent human study is one dataset (SciQ), 100 + 55 items, translation-assisted with the
  English read alongside**, with a full-sample κ of 0.680 below the pre-registered 0.70 threshold and a
  gate that passed on the pre-registered block. Its Q1 instrument is a proxy for prior answerability, not for context-dependence.
- **Each modern solver was run at a single seed and a single quantisation, with no option-order
  debiasing and no calibration check.** The support-collapse finding follows from $Acc_{wof}$ alone
  and does not depend on these choices; the exact KDA values would shift.
- **Cross-dataset KDA values are not on a common scale.** The SciQ–OBQA gap (0.4715 vs 0.2712) is
  substantially a mixture effect between extractive and deductive target facts, and the source
  states that "any comparison of absolute KDA values across datasets must therefore control for
  target-fact style" ([`kda_reproduction_summary.md`](../ex1_reproduce_KDA/kda_reproduction_summary.md) §2.1).
- **The flagged-rate comparison 37.1% (SciQ) vs 16.0% (OBQA) is not directly comparable** — SciQ had
  three detection criteria available and OBQA two.
- **Two different verbatim-containment measurements exist and should not be merged.**
  [`kda_reproduction_summary.md`](../ex1_reproduce_KDA/kda_reproduction_summary.md) §2.1 reports the
  gold answer present in its own passage for **820 / 884 (92.8%)** SciQ and **47 / 500 (9.4%)** OBQA
  items; [`rq1_test_split_failure_analysis.md`](../ex5_failure_audit/rq1_test_split_failure_analysis.md)
  §4.3 reports **838 / 884 (94.8%)** and **57 / 500 (11.4%)** under a different string-matching
  criterion.
- **All relationships reported here are correlational.** No attention- or attribution-level analysis
  isolates why any individual solver answered as it did.
- **Only two benchmarks, both English, both science MCQ.** Whether the same failure profile holds for
  other corpora, subjects or languages is untested here.

---

## 12. Source index

| Document | Sections used | Contributes |
|---|---|---|
| [`ex1_reproduce_KDA/kda_reproduction_summary.md`](../ex1_reproduce_KDA/kda_reproduction_summary.md) | §1.1–1.3, §2.1–2.2 | Encoder-ensemble baseline; single-solver degeneracy; per-solver spread; containment split |
| [`ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md`](../ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md) | §4.1–4.3, §5, §6, §8.5 | Support collapse; bimodality; buckets; `prior_share` |
| [`ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md`](../ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md) | §4.1–4.3, §5, §7, §8 | Second-solver replication; cross-solver item agreement |
| [`ex2_counterfactual/counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) | §6 | $KDA_{adj}^{excl}$ inertness; exclusion AUCs |
| [`ex2_counterfactual/e1_counterfactual_llm_scale.md`](../ex2_counterfactual/e1_counterfactual_llm_scale.md) | §3, §5 | Cross-solver agreement on the Setting-C label; why single-solver KDA is not reported at LLM scale |
| [`ex2_counterfactual/unstable_other_convention.md`](../ex2_counterfactual/unstable_other_convention.md) | — | `unstable_other` convention and its 29 pp movement |
| [`ex3_student_simulation/student_persona_simulation_report.md`](../ex3_student_simulation/student_persona_simulation_report.md) | §3.1, §3.3, §7.3 | Prompt-based mitigation fails; order artifact under both solvers |
| [`ex4_free_response/plan_option_free_response_experiment.md`](../ex4_free_response/plan_option_free_response_experiment.md) | §9.2, §10.1–10.3 | Open-ended recall bands; judge-reliability caveat |
| [`ex5_failure_audit/rq1_test_split_failure_analysis.md`](../ex5_failure_audit/rq1_test_split_failure_analysis.md) | §1, §4.1–4.4, §5 | Insensitivity; numerator contamination; weight degeneracy; flagged pool; limitations |
| [`ex5_failure_audit/corpus_defect_audit.md`](../ex5_failure_audit/corpus_defect_audit.md) | §4, §6, §8 | Corpus-defect census; mis-key non-estimability |
| [`ex6_psfd_score/psfd_formulation.md`](../ex6_psfd_score/psfd_formulation.md) | §1.2, §3, §14 | $KDA_{cont}$ Tier-1 AUCs; adjusted-estimator means; circularity caveat |
| [`ex8_independent_labels/rq2_annotation_results.md`](../ex8_independent_labels/rq2_annotation_results.md) | §B1–B5, §B6, §D2–D6 | Independent human labels; AUC/ρ/sign tests; κ gate; prior-override rate; construction artifacts |
| [`ex8_independent_labels/ANNOTATION_GUIDE_RQ2.md`](../ex8_independent_labels/ANNOTATION_GUIDE_RQ2.md) | Q1/Q2 instruments | Literal wording of the human reference question |
| [`papers/KDA_Paper_Documentation.md`](../papers/KDA_Paper_Documentation.md) | Tables 4, 9, 10, 11; §4 Limitations | Source metric's own validation and stated limitations |
| [`synthesis/rq1_research_synthesis.md`](../synthesis/rq1_research_synthesis.md) | §1, §2.1–2.3, §4.1 Addendum, Appendix | Prior five-report synthesis; carried-forward limitations |
| [`README.md`](../../README.md) | §2.1–2.4 | Metric definitions and the adjusted-estimator specifications |
