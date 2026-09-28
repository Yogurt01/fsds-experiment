# RQ2 Research Synthesis — How can we formulate a metric that separates context reliance from prior knowledge?

**Research question.** *How can we formulate a robust metric that measures whether an MCQ answer
strictly depends on the provided target knowledge material, versus prior knowledge or surface
shortcuts?* (README §2.3.)

**Written 2026-09-25. Planning document.** Nothing here is executed. Scope fixed with the lead:
**~1 week**, **SciQ only** (OpenBookQA stays closed per ex7 §11), **a new AI-free annotator is
available**.

**Evidence base.** Four internal reports and their result files. Every figure below was re-derived
from the underlying JSON before being quoted; none is taken from prose.

| Short name | File | Scope |
|---|---|---|
| **ex6** | `ex6_psfd_score/psfd_formulation.md` (+ `psfd_evaluation.json`, `psfd_scores_*.json`) | The P/S/F/D score, evaluated against Setting-C labels |
| **ex7** | `ex7_counterfactual_construction/counterfactual_construction_audit.md` (+ `construction_fidelity_audit.json`) | Construction fidelity, surface leakage, OBQA ineligibility |
| **ex7-rescan** | `ex7_counterfactual_construction/residual_leakage_rescan.md` (+ `residual_leakage_rescan_sciq.json`, `prior_dependence_corrected_sciq.json`) | 2026-09-24 expanded leakage check and corrected prior-dependence figures |
| **ex8** | `ex8_independent_labels/rq2_annotation_results.md` (+ `metric_comparison*.json`, `kappa_annotator1_vs_annotator2.json`, `q2_crosstab_*.json`) | Independent human label set, two annotators |

**Four claims in the source documents are superseded and appear below only in corrected form:**
ex7's *"zero whole-phrase leakage, 0/860"* (§4); ex6 §3's within-family AUC ordering (§2, per its own
§14); ex6 §8.1's `reasoning_shortcut` ❌ (a spec error, withdrawn by §13); and ex7's first report of
target type-incoherence as near-deterministic (extreme-group sampling inflation; φ = 0.253 at full n).

---

## 1. Executive verdict

**The tested formulations did not fail as arithmetic. They failed as measurement.** The decomposition
into P (prior answerability), S (material sufficiency) and F (counterfactual sensitivity) carries real
information the baseline estimator lacks. What has never been established is *which combination of
them is right*, and the reason is that **no experiment run so far was capable of answering that
question.** Three separable causes, which the rest of this document keeps apart because they have
different fixes:

1. **Evaluation circularity.** Every internal comparison scored candidates against Setting-C-derived
   labels that the candidates algebraically contain. `F` decomposes exactly into `½(MarginB +
   MarginC)`, and `MarginC` reproduces the Setting-C label at **AUC 1.0000 by construction**. Within
   the C-derived family, a higher score means "closer to re-deriving the label", not "better metric".
2. **Construct mismatch.** The independent labels show that "knowledge-dependency" resolves into two
   axes — prior-answerability and context-following — that rank the candidates in **opposite orders**.
   `D` leads on one, `F` leads on the other, and the head-to-head test lands at 0.512. No scalar
   combination rule repairs a construct that is not one-dimensional.
3. **Label contamination and instability.** The target the family is fitted to is itself noisy and
   biased: cross-solver agreement on the Setting-C class is **κ = 0.046**, and 3.5% of SciQ
   counterfactual passages still state the gold answer where the question is answered — items that
   carry `prior_dependent` at **4–5× the base rate**.

**The corrected reading of the whole RQ2 branch to date: the product form `(1−P)·S·max(F,0)` is
rejected, `F` alone is not a survivor either, and the next round should buy an uncontaminated
adjudication instrument rather than another functional form.**

---

## 2. What ex6 established about P/S/F/D — and what its own §14 withdrew

**Supported (ESTABLISHED).** The decomposition sees inside the `both_correct` bucket that `KDA_cont`
is structurally blind to. On SciQ, separating `context_dependent` from `prior_dependent`:

| | all eligible (n = 860; 669+/115−) | within `both_correct` (n = 381; 287+/85−) |
|---|---:|---:|
| `D` | **0.824** | **0.772** |
| `KDA_cont` | 0.549 | 0.617 |

This is a **family-versus-family** comparison — using Setting-C information at all, versus not — and
it survives every caveat below.

**Not supported (ESTABLISHED).** The product form. Pre-registered criterion 5 — *`D` must beat its
best single component by ≥ 0.02 AUC* — fails on **every** split, by **−0.075 to −0.399**. The overall
verdict was 4/5 on SciQ, but criterion 5 is the one that adjudicates the combination rule.

**The load-bearing finding is ex6 §14, and it removes the ability to read anything more into that
ordering.** Regrouping `F_raw`'s four terms by setting rather than by option:

```
F_raw = ½[ MarginB + MarginC ]        MarginB = p(gold|B) − p(c|B)     MarginC = p(c|C) − p(gold|C)
```

Identity verified to 1e-9 on 784/784 SciQ items.

| | `MarginC` | `MarginB` | `F` |
|---|---:|---:|---:|
| Tier-1 AUC, SciQ | **1.0000** | **0.454** | 0.899 |
| within `both_correct` | — | 0.538 | 0.912 |

`MarginC`'s sign **is** the label: `context_dependent` and `prior_dependent` are exactly
`argmax(C) = c` and `argmax(C) = gold`. So `F`'s 0.899 is substantially definitional, and the only
quantity in the experiment whose Tier-1 evaluation is *not* circular — `MarginB` — carries **no
Tier-1 signal at all** (chance, not anti-predictive; §11 shows the below-chance pooled value is a
composition effect). **"`F` alone is the metric" was never a supportable claim**, and §3 shows it did
not survive first contact with independent labels.

**The `(1−P)` gate inherits RQ1's saturation failure (ESTABLISHED).** On the encoder ensemble it
attenuates the score by ~41% (`gate_attenuation` 0.590) while producing almost no zeros. On the
saturated Qwen3-4B run it annihilates it:

| | encoders (SciQ ensemble) | Qwen3-4B |
|---|---:|---:|
| mean `D` | 0.1495 | **0.0356** |
| mean `D′` | 0.2534 | 0.3906 |
| items with `D` exactly 0 | 0.1% | **88.1%** |
| items with `D′` exactly 0 | — | 0.5% |
| items with `(1−P) < 0.01` | 0.0–11.2% per solver | **94.2%** |
| Tier-1 AUC, `D` vs `D′` | — | **0.600 vs 0.997** |

Note the decomposition: on three of four encoders `frac(D = 0)` equals `frac(D′ = 0)` **exactly**, so
those zeros come from `max(F,0)`, not from the gate. On Qwen 99.5% of the zeros come from the gate.
The gate is not merely shrinking `D` at LLM scale — inside the one bucket RQ2 most needs to see, it
removes the signal.

---

## 3. What ex8 established — the first non-circular read

ex8 built an independent label set precisely because §14 showed Setting-C labels cannot rank the
C-derived family. 100 Q1 items (answerability, 1–4) and 55 Q2 items (what the counterfactual passage
states), labelled blind to every score and class. Strata A/B/C = 55/25/20.

**The head-to-head is inconclusive, and it is not power-limited (ESTABLISHED).** Primary sign test
`F` vs `KDA_cont`, strata A+B: **w/m = 41/80 = 0.512** [0.398, 0.626], **p = 0.911**. Power at m = 80
was **0.94** against a true rate of 70%, so a difference large enough to matter would very likely have
shown. This is weak positive evidence that **no large difference exists** on the Q1 axis.

**`F` collapses to chance against independent labels (ESTABLISHED) — exactly as §14 predicted.**

| Metric | Q1 AUC (pos = DEPENDENT, n = 27/73) | 95% CI | Spearman ρ vs raw 1–4 | 95% CI |
|---|---:|---|---:|---|
| `KDA_cont` | 0.580 | [0.459, 0.701] | 0.124 | [−0.087, 0.305] |
| **`F`** | **0.490** | [0.356, 0.613] | **−0.103** | [−0.293, 0.094] |
| `D` | **0.697** | **[0.581, 0.810]** | **0.233** | **[0.038, 0.417]** |
| `D′` | 0.458 | [0.330, 0.577] | −0.167 | [−0.352, 0.026] |

`F` dominated every Setting-C comparison in ex6 at AUC 0.899 and lands at chance here. That is the
single clearest demonstration that the ex6 ordering was measuring circularity.

**`D`'s Q1 lead is rater-dependent and structurally advantaged (OPEN, do not rely on it).** Under
annotator 2, `D`'s AUC CI is [0.490, 0.759] and **no** metric's CI excludes chance. Independently of
that: Q1 asks a human to judge whether an unaided student could answer — a human proxy for Setting A,
i.e. for `P`. `D = (1−P)·S·max(F,0)` contains that term and `F` does not. A metric sharing a factor
with the thing the human is estimating has a built-in edge on Q1.

**Q2 inverts the ranking, and this is the pivot of the whole synthesis.** Separating items the human
reads as stating the counterfactual target from those read as still stating gold:

| Metric | Q2 AUC | 95% CI |
|---|---:|---|
| **`F`** | **0.964** | [0.908, 1.000] |
| `D` | 0.868 | [0.736, 0.964] |
| **`KDA_cont`** | **0.072** | **[0.000, 0.196]** |

Each metric wins the axis it was built for and loses the other. Two limits travel with this table: the
negative class is **n = 5**, and `F` shares a common cause with the Q2 label (a failed substitution
leaves the passage nearly unchanged, so the human reads gold *and* `F` is mechanically near zero).
Q2 as built is therefore a **construction-validity instrument**, not yet a metric discriminator.

**Two results that need no metric at all (ESTABLISHED, and the most durable output of ex8):**

- **Human-grounded prior-override: 40/200 pairs = 20.0%** [15.0, 26.1]. Where a human confirms the
  passage states the counterfactual answer, solvers still answer gold on one pair in five. Nothing in
  that chain derives from a metric or a Setting-C label. The per-solver ordering reproduces ex2 §6.3
  exactly, with closed-book NQ-finetuned T5 worst at 40%.
- **~30% of `prior_dependent` labels in the sample are construction artifacts**: 17/57 pair-level
  [19.5, 42.7], 5 of 16 ensemble-level. Confirmed rater-free in §D6.

**Reliability (ESTABLISHED).** Block κ **0.870** (30 items) passes the pre-registered gate;
full-sample κ **0.680** (100 items), quoted alongside as the plan requires. Both annotators read the
English source alongside a translation, each from a different tool, and no tool suggested any label
or rating. The gate was decided **PASSED** by the project lead. The estimate is imprecise — 5
positives in the block and a one-directional calibration offset — but those are limits on precision,
not on how the labels were produced.

---

## 4. Counterfactual construction — failure patterns, cause, corrected measurements

### 4.1 The original leakage check was near-tautological

ex2 records one leakage counter, `residual_answer_mentions`: a **whole-word, exact-surface** count of
the gold answer in the perturbed passage. It reads **0 on all 860 eligible SciQ items**, and ex7
reported that as a solved channel. It was guaranteed, not earned — the substituter has just rewritten
every whole-word occurrence of *that same surface form*, so nothing of that form can remain.

All five Q2 construction failures read 0 on it. The secondary glued counter fires on 16 items and
catches only two of the five.

### 4.2 Four failure patterns, one cause

| Pattern | Confirmed item | What survived |
|---|---|---|
| **Morphological residual** | 25, 716, 805 | `immune system` → title rewritten, *"deficiencies of the immune **systems**"* survives; `antioxidants` → *"An **antioxidant** is a molecule that inhibits the oxidation…"*; `acid` → *"pollutants form **acids** when dissolved…"* |
| **Fused-boundary residual** | 0 | Only the heading changed; *"are called**oxidants**"* survives fused to the previous word |
| **Fragment splice** | 222 | A `partial`-tier splice of `of solute` produced a non-sentence, while *"dependent upon the chemical identity of the solute"* survives intact |
| **Over-substitution / self-contradiction** | ex7 §1.1, `partial` tier | `percent` replaced at 6 sites; `polar` → *"secular and nonpolar"* |

**One cause explains all four.** The pipeline is **source-anchored** and matches **a single surface
form of the gold**, while a natural extractive passage restates the same fact in **other forms** — an
inflection, a fused token, a paraphrase, a heading. The rewriter changes the form it matched and
leaves the others standing. In four of the five confirmed failures *the original answering clause
survives verbatim*, so a solver answering gold there is **reading correctly, not overriding**.

### 4.3 Corrected measurements (ex7-rescan, 2026-09-24)

An expanded check adds glued, morphological/stem and fragment matchers plus a filter for whether the
surviving mention sits in a sentence that answers the question.

| Check | Items flagged (of 860) | Rate |
|---|---:|---:|
| ex2 `residual_answer_mentions` (published) | **0** | 0.0% |
| ex2 `residual_glued_mentions` | 16 | 1.9% |
| expanded, any | 109 | 12.7% |
| **expanded, strict** (gold still stated where the question is answered) | **30** | **3.5%** |

- **25 of the 30 are invisible to both existing counters.**
- **26 of the 30 sit on the `exact` tier** (3.2% of 813), against 3/36 `partial` and 1/11
  `morphological`. The lower tiers leak ~2.5× more often, but **restricting to `exact` would not have
  avoided this.** That contradicts the working assumption that `exact` was clean.
- Validation on the 55-item Q2 sample: strict recall **5/5** on the known failures; 2/50 flags among
  the rest, both explained (774 is a genuine leak the annotator did not act on; 146 is a real
  limitation of any lexical rule).

### 4.4 Implications for RQ2 — small at the aggregate, large at the mechanism

Recomputed through ex2's own metric function (which reproduces every committed figure exactly with no
exclusions):

| | published | strict exclusion (n = 830) | Δ |
|---|---:|---:|---:|
| ensemble `prior_dependent` share | 13.37% | 12.17% | −1.20 pp |
| ensemble context-verified accuracy | 78.49% | 78.67% | +0.19 pp |
| ensemble context-following rate | 77.79% | 78.92% | +1.12 pp |

**No published conclusion changes and every solver keeps its rank.** But the aggregate moves little
only because the flagged set is 3.5% of the corpus. The rate *within* it is the finding:

| `prior_dependent` rate | flagged_strict (n = 30) | adjudicated leaks (n = 16) | unflagged (n = 751) |
|---|---|---|---|
| ensemble | **46.7%** [30.2, 63.9] | **62.5%** [38.6, 81.5] | **11.6%** [9.5, 14.1] |
| mpnet-base | 43.3% [27.4, 60.8] | 75.0% [50.5, 89.8] | 7.1% [5.4, 9.1] |

**4.0× at the ensemble, 5.4× on adjudicated leaks, 10.6× on mpnet; intervals disjoint for four of the
five solvers.** The defect manufactures the very label the entire metric family is fitted to.

Two caveats travel with this: the 30 is a **lower bound on a lexical check** — item 222's failure is
semantic and was caught only because enough stems happened to survive — and the
CONFIRMED / BORDERLINE / SPURIOUS split (16 / 9 / 5) is a single unverified reading.

---

## 5. Open problems

| # | Problem | Why it blocks RQ2 |
|---|---|---|
| **O1** | **No uncontaminated adjudication instrument exists.** Setting-C labels are circular (§2); Q1 is a `P`-proxy that structurally favours `D` (§3); Q2 has n = 5 negatives and shares a cause with `F` | Without one, no comparison among C-derived candidates is falsifiable |
| **O2** | **The target is solver-dependent** — cross-solver κ on the Setting-C class is **0.046** | A metric of a *question* is being fitted to a label that is a property of the *solver* |
| **O3** | **Enrichment is not causation.** Flagged items may simply be more extractive — the passage repeats the gold — and hence differ in prior profile for reasons unrelated to the leak | The 4–5× figure could be confounded; the ~30% artifact claim rests on it |
| **O4** | **Whether RQ2 has a scalar answer at all is open.** Q1 and Q2 rank the candidates in opposite orders | A one-dimensional metric may be the wrong deliverable |
| **O5** | **Any term monotone in zero-context accuracy inherits RQ1's saturation failure** | `(1−P)` is unusable at LLM scale as specified; `D′` drops the prior signal entirely |

---

## 6. Hypotheses

Stated as hypotheses, not findings. Each is falsifiable by one experiment in §7.

- **H1 — the leak is causal, not confounded.** Repairing the residual gold in a flagged passage moves
  its `prior_dependent` labels toward `context_dependent`; matched unflagged controls do not move.
- **H2 — dependency is two-dimensional.** Prior-answerability and context-following are near
  orthogonal, so a two-component report `(P̂, F̂)` with an explicit decision rule dominates every
  scalar on a joint criterion.
- **H3 — the gate's failure is structural, not a tuning problem.** `(1−P)`'s attenuation of `D` is a
  deterministic function of zero-context confidence, so no threshold or reparameterisation of that
  term survives saturation; only replacing the multiplicative gate does.
- **H4 — the 3.5% is a lexical floor.** A meaning-level residual check finds leaks no surface matcher
  can (the item-222 class), raising the population rate.

---

## 7. Proposed one-week plan

Four experiments, SciQ only, ordered by value. Criteria are fixed here so they cannot move later. Each
can come out negative, and what a negative result means is stated.

### E1 — Repair-and-re-run *(rater-free; the highest-value experiment)*

**Tests H1, resolves O3.**

Extend the substituter to sweep **every** residual gold form after the primary substitution —
inflections, fused tokens, headings — case-preserved. Items whose only residual is a fragment splice
are **excluded, not repaired**: item 222's class is not mechanically fixable, and pretending otherwise
would manufacture a different defect. Re-score Settings B and C on the repaired flagged items **plus
matched unflagged controls**, matched on passage length, substitution-site count and gold-repetition
count — the three confounds that would otherwise explain the enrichment on their own. The repair
pipeline must be a no-op on the controls, which is itself a check.

**Criterion:** ≥ 50% of pair-level `prior_dependent` labels on the flagged set change class, **and**
the control set moves by ≤ 5 pp. If the flagged set moves and the controls do not, H1 holds and the
~30% artifact figure is causal. **If both move, or neither does, H1 is rejected** and §4.4's
enrichment must be restated as an association — which would also weaken ex8 §B4b.

Reuses the committed encoder ensemble; roughly 60 items × 4 solvers, no new inference infrastructure.

### E2 — Enlarge the Q2 negative class with the AI-free annotator *(~75 min, one sitting)*

**Attacks O1.**

Q2's five-item negative class is the binding constraint on the only non-circular discriminator in the
project. Deliberately **over-sample the 30 rescan-flagged items** — enriched for failed substitutions —
plus matched controls, plus an overlap block with annotators 1 and 2 so the new annotator's Q2
reliability can be measured.
Because the sample is deliberately enriched, every rate must be **IPW-weighted back to the 860-item
population**, reusing the stratum machinery already in `compare_metrics.py`. Pre-register that before
any label exists.

No Q1 pass is needed: the ex8 κ gate has already passed.

**Criterion:** Q2 negative class ≥ 20 after weighting, and Q2 κ ≥ 0.70 on the overlap block against
annotators 1 and 2. Below either, E3 is not run and the report says the instrument could not be built.

### E3 — Two-axis versus scalar, scored on E2's labels

**Tests H2, resolves O4. Depends on E2.**

Pre-registered comparison of the two-component rule `(P̂, F̂)` against `D`, `D′`, `F` and `KDA_cont` on
a joint label — an item counts as genuinely material-dependent only if it is both low prior-answerable
(Q1) **and** context-following (Q2). This is the first comparison in the project scored on labels that
are neither Setting-C-derived nor a proxy for a term inside one of the candidates.

**Criterion:** the two-component rule beats the best scalar by ≥ 0.05 AUC on the joint label.
**If it does not, H2 is rejected**, RQ2 does have a scalar answer, and the combination-rule search
resumes — but now with a non-circular instrument to score it against, which is the thing that was
missing all along.

### E4 — Saturation decomposition *(rater-free, cheap, committed data only)*

**Tests H3, resolves O5.**

Largely computable from the committed runs already (§2's table). The new content is the decomposition:
for each solver, partition `D`'s zeros into those caused by `max(F,0) = 0` and those caused by
`(1−P) ≈ 0`, and report it against Setting-A accuracy. The committed data already shows the split is
0% gate-attributable on three of four encoders and 99.5% on Qwen3-4B.

**Criterion:** if the gate-attributable share of zeros tracks Setting-A accuracy, `(1−P)` as a
multiplicative gate is rejected for LLM-scale solvers on structural grounds — closing the `D` vs `D′`
question without further formula search. **Note the honest limit:** there are five solver points and
four are clustered at Setting-A accuracy 0.27–0.51, so this is effectively a two-cluster comparison,
not a trend. It should be reported as a mechanism, not a dose-response curve.

---

## 8. Evidence-status ledger

| Claim | Status |
|---|---|
| P/S/F carry information `KDA_cont` lacks (0.824 vs 0.549; 0.772 vs 0.617 within `both_correct`) | **ESTABLISHED** |
| The product form `(1−P)·S·max(F,0)` is not justified — criterion 5 fails on every split | **ESTABLISHED** |
| `F = ½(MarginB + MarginC)`; `MarginC` AUC = 1.0000 by construction; `MarginB` at chance | **ESTABLISHED** (identity verified to 1e-9) |
| Tier-1 cannot rank within the C-derived family | **ESTABLISHED** |
| `F` is at chance against independent Q1 labels (0.490, ρ −0.103) | **ESTABLISHED** (holds under both annotators) |
| The `F` vs `KDA_cont` head-to-head is inconclusive and not power-limited (41/80, p 0.911) | **ESTABLISHED** |
| 20.0% human-grounded prior-override [15.0, 26.1] | **ESTABLISHED** (both annotators) |
| ~30% of `prior_dependent` labels in the Q2 sample are construction artifacts | **ESTABLISHED** as an association; rater-free on all 5 items; **causality is O3** |
| `(1−P)` annihilates `D` at LLM scale (88.1% exact zeros, AUC 0.600 vs 0.997) | **ESTABLISHED** |
| Expanded check flags 30/860 (3.5%); 25 invisible to both existing counters; 26 on `exact` | **ESTABLISHED** |
| Flagged items carry `prior_dependent` at 4–5× the base rate | **ESTABLISHED** as measured; **confounding not excluded (O3)** |
| Corrected aggregate figures move ~1 pp; no conclusion changes | **ESTABLISHED** |
| `D` leads on Q1 (AUC 0.697) | **OPEN** — rater-dependent, and structurally advantaged by construct overlap |
| ex8 κ gate | **ESTABLISHED** — PASSED (block 0.870; full sample 0.680) |
| The 3.5% is a lower bound; CONFIRMED/BORDERLINE/SPURIOUS is one unverified reading | **OPEN** |
| Setting-C labels are solver properties, not question properties (κ = 0.046) | **OPEN** — largest standing RQ2/RQ3 risk |
| Leak → `prior_dependent` is causal | **HYPOTHESIS (H1)** → E1 |
| Dependency is two-dimensional | **HYPOTHESIS (H2)** → E3 |
| The gate's failure is structural | **HYPOTHESIS (H3)** → E4 |
| A semantic residual check raises the 3.5% | **HYPOTHESIS (H4)** → not funded this week |
| E1–E4 | **PROPOSED** — none run |

---

## 9. Explicitly out of scope, with reasons

- **More Q1 annotation for the sign test.** ex8 §B1a: the study was already at 0.94 power against a
  70% true rate. Reaching *m* = 199 buys a difference too small to change which metric anyone should
  use. The annotator's hours go to E2 instead.
- **Further combination-rule search scored on Setting-C labels.** ex6 §14 makes it unfalsifiable: the
  score rises with proximity to the label's own definition.
- **Reopening OpenBookQA.** Closed at ex7 §11; reconfirmed SciQ-only. The target-anchored rewriting
  proposal (ex7 §4.2) and its V0–V6 validity protocol (§7) remain on file, unbuilt.
- **Re-running ex1/ex2 wholesale, or touching the committed hard / soft / sample-exclusion
  estimators.** They remain the project's method of record; E1 re-scores ~60 items, not the corpus.
- **H4's semantic residual check.** Real, but it competes with E1 for the same week and E1 answers the
  sharper question.

## Related

- [`rq1_research_synthesis.md`](rq1_research_synthesis.md) — the RQ1 verdict this branch responds to
- [`../ex6_psfd_score/psfd_formulation.md`](../ex6_psfd_score/psfd_formulation.md) §14 — the circularity result
- [`../ex7_counterfactual_construction/residual_leakage_rescan.md`](../ex7_counterfactual_construction/residual_leakage_rescan.md) — the corrected measurements in §4
- [`../ex8_independent_labels/rq2_annotation_results.md`](../ex8_independent_labels/rq2_annotation_results.md) §D6 — the rater-free confirmation
- [`../NEXT_PHASE_HANDOFF.md`](../NEXT_PHASE_HANDOFF.md) §3 — estimator status and the κ = 0.046 risk
