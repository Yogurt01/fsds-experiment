# RQ2 — independent label set: results

Three structurally separate parts; Part D was added when a second annotator's full sheets
arrived. Read them in this order and do not mix their numbers.

> # Gate: **PASSED** — both annotators used AI for translation only, reading the English source alongside
>
> **Decided by the project lead.** Both annotators confirmed, as relayed by the lead: they read the
> original English side by side with the translation throughout; each used a **different** LLM for
> translation; neither tool suggested labels, ratings, rankings or reasoning. Every judgement on both
> sheets was the annotator's own.
>
> | | |
> |---|---|
> | κ as computed | block **0.870** (30 items — the pre-registered gate) · full sample **0.680** (100 items, supplementary). The block governs by the rule fixed in advance; the two are always quoted together |
> | What it measures | agreement between two humans who each judged the English source for themselves |
> | **Gate status** | **PASSED** — the registered condition, independent human judgement applying the rubric, is met (§D0) |
> | Limits on the estimate | a 5-item positive class in the block, and a one-directional calibration offset between raters (§D2). These are limits on precision, not conditions of the gate |
> | Rater-free evidence | the 5 construction failures behind the Q2 findings were re-read in English with no rater involved: **all 5 confirmed** (§D6) |
>
> **Two questions are kept separate throughout this document**, and neither is evidence about the
> other: **(1)** does F/D outperform KDA_cont — empirical, answered by the sign test and AUC on
> their own terms; **(2)** is the label set trustworthy — methodological, about how the annotation
> was produced. The gate status below is argued only on (2).
>
> *History: "gate not evaluated — one annotator" → "PASSED on the block" → "NOT EVALUATED AS
> DESIGNED", on the belief that AI had suggested ratings → "PASSED WITH CAVEAT", after the
> translation-only correction, pending the reading protocol → **PASSED**, upon confirmation that both
> annotators read the English alongside, using different tools. This banner supersedes all four.*

| Part | What | Status |
|---|---|---|
| **B** | Annotator 1 (**translation assistance**, English read alongside), 100 Q1 + 55 Q2 | **Primary, pre-registered. Gated** — κ gate passed (see banner) |
| **D** | Annotator 2 (**translation assistance**, English read alongside), same 100 + 55 items | Replication check per the fixed plan. Reported separately, never merged with B. **Read §D0 first** |
| **C** | Qwen3-4B judge over the same items | **Supplementary only.** Not a rater, not a gate, never merged with B or D |

Parts appear in the order **B → D → C**: the two human parts first, the LLM part last.

Artifacts: `metric_comparison.json` (annotator 1) · `metric_comparison.annotator2.json` ·
`kappa_annotator1_vs_annotator2.json` · `q2_crosstab_annotator1_vs_annotator2.json` ·
`llm_judge_agreement.json` · `q1_llm_judge_check.csv` · `q2_llm_judge_check.csv` ·
`llm_judge_prompts.json`

---
---

# PART B — Human annotator 1: the pre-registered primary result

> **Status: gated.** Written as exploratory single-rater evidence. The gate status changed as the
> annotators' disclosure became more specific, and was decided **PASSED** by the project lead upon
> confirmation of the parallel reading protocol (§D0). The numbers below are unchanged throughout.
> Where a finding was checked by annotator 2, the section says so and points to Part D.

Single rater. Labels produced blind to every score, Setting-A/B/C output, Setting-C class and
stratum. Sheets verified against `annotation_sheets.LOCK.json`: all three SHA-256 match, columns
and row order unchanged, 100/100 and 55/55 valid values, zero `unusable`.

## B1. Primary test — inconclusive

Sign test, exactly as pre-registered (nine steps, rank-normalised distance to the human ordering).

| Test | w/m | proportion | 95% CI | p | winner |
|---|---|---:|---|---:|---|
| **Primary — F vs KDA_cont, strata A+B** | 41/80 | **0.512** | [0.398, 0.626] | **0.911** | **none** |
| Sensitivity — F vs KDA_cont, stratum A | 22/55 | 0.400 | [0.270, 0.541] | 0.177 | none |
| D vs KDA_cont | 44/78 | 0.564 | [0.447, 0.676] | 0.308 | none |
| D vs F | 46/79 | 0.582 | [0.466, 0.692] | 0.177 | none |

**No metric is closer to the human ordering than any other at this sample size.** The primary test
lands almost exactly on chance. Under the pre-registered decision rule this is the
**INCONCLUSIVE** branch — *"do not adopt either"* — and that is the branch it would have triggered
even if the gate had passed.

### B1a. Sample size required — what the inconclusive branch obliges us to state

The rule's inconclusive branch requires stating the *n* needed. Two-sided exact binomial,
α = 0.05, 80% power:

| If the true rate is… | *m* needed | power at the current *m* = 80 |
|---|---:|---:|
| 60% | **199** | 0.37 |
| 62.5% | **125** | 0.55 |
| **65%** | **90** | **0.72** |
| 70% | 49 | 0.94 |
| 75% | 30 | 1.00 |

**Read this before deciding whether more annotation is worth it.** At *m* = 80 the study was
already well powered for a decisive difference (0.94 at 70%) and it found **w/m = 0.512**. A true
effect large enough to matter would very likely have shown. So the inconclusive result is not
mainly a power failure — **it is weak evidence that no large difference exists** between F and
KDA_cont on this dimension.

Scaling up is cheap only in the 65% band: strata A+B hold 389 of the 860 eligible items, so *m* =
90 is drawable without redesign — **10 more items, ~25 minutes**. Chasing 60% needs *m* = 199, a
further **~5 hours** of Q1 annotation, to resolve a difference small enough that it would not
change which metric anyone should use. **Recommendation: do not buy more Q1 annotation for the
sign test.** If effort is available, §B7's second rater and the Q2 line of evidence (§B4) are both
worth more per hour.

## B2. AUC against the binarised human label

Positive class = DEPENDENT (rating 1–2), n = 27; negative = NOT_DEPENDENT (3–4), n = 73.
Bootstrap CIs, 2000 resamples, seed fixed.

| Metric | unweighted | 95% CI | IPW (population) |
|---|---:|---|---:|
| KDA_cont | 0.580 | [0.459, 0.701] | 0.455 |
| **F** | **0.490** | [0.356, 0.613] | 0.498 |
| **D** | **0.697** | **[0.581, 0.810]** | **0.668** |
| D′ | 0.458 | [0.330, 0.577] | 0.458 |

**D is the only metric whose CI excludes 0.5.** Every other CI contains chance.

## B3. Spearman against the raw 1–4 rating

| Metric | ρ | 95% CI |
|---|---:|---|
| KDA_cont | 0.124 | [−0.087, 0.305] |
| **F** | **−0.103** | [−0.293, 0.094] |
| **D** | **0.233** | **[0.038, 0.417]** |
| D′ | −0.167 | [−0.352, 0.026] |

Same pattern: D is the only metric whose interval excludes zero.

### B3a. Orientation check on F's negative ρ — verified, no bug

F's ρ = −0.103 is mildly wrong-signed rather than merely flat, and this project has caught two
orientation bugs before (reversed leakage direction in ex7, the `F_raw` term decomposition in
ex6), so it was checked by hand before being reported.

`compare_metrics.py:463` computes `spearman(metric, 5 − rating)`. Higher `d = 5 − rating` means
more passage-dependent; every metric also claims higher = more passage-dependent. **A positive ρ
therefore means agreement, and the sign as computed is the right way round.**

| | item | F | rating *h* | *d* = 5−*h* |
|---|---:|---:|---:|---:|
| 3 highest F | 876 · 15 · 147 | 0.561 · 0.536 · 0.515 | 4 · 2 · 4 | 1 · 3 · 1 |
| 3 lowest F | 327 · 712 · 805 | 0.104 · 0.091 · 0.040 | 4 · 3 · 3 | 1 · 2 · 2 |

The n=3 extremes are uninformative (mean *d* = 1.67 on both ends). The decile view is the
readable one: **bottom-decile F has mean *d* = 2.10, top-decile F has mean *d* = 1.80** — lower F
items are judged *slightly more* passage-dependent, the direction the negative ρ implies.

An independently written Spearman implementation reproduces **ρ(F, d) = −0.1026** against the
script's −0.103, and **ρ(F, rating) = +0.1026** — the exact negation, as it must be. **No
orientation fault. The negative correlation stands as reported.**

One wording correction that follows: with CI [−0.293, 0.094] spanning zero, the defensible claim
is **"F is at chance"**, not "F is inverted". The point estimate's sign should not be leaned on.

## B4. Q2 — the human-grounded test of context-following

**50 of 55 items (90.9%)** were read as stating the counterfactual target; 5 (9.1%) as still
stating the gold; **`none` was never used**. On SciQ the substitution does put a different,
readable answer into the passage.

That much was already reported. The test Q2 was actually built for — cross-tabulating the human
reading against what the solvers did — is below, and it is the only non-circular evidence in this
document. Nothing in it derives from a metric or from a Setting-C label.

### B4a. Genuine prior-override, measured against a human reading

(model, question) pairs over the 55 stratum-A items, 4 solvers:

| Human reads the CF passage as stating… | pairs | context_dependent | **prior_dependent** | unstable_other |
|---|---:|---:|---:|---:|
| **the counterfactual target** | 200 | 130 (65%) | **40 (20%)** | 30 (15%) |
| the gold answer | 20 | 3 (15%) | **17 (85%)** | 0 (0%) |

**Where a human confirms the passage states the counterfactual answer, solvers still answer gold
on 20.0% of pairs [95% CI 15.0–26.1%].** That is prior-override established against an
independent human reading, with no metric and no Setting-C-derived label in the chain. It is the
cleanest evidence for the RQ1 thesis anywhere in this project.

Per solver, on those same 50 human-confirmed items:

| Solver | prior_dependent | rate |
|---|---:|---:|
| `google/t5-small-ssm-nq` | 20/50 | **40.0%** |
| `Riiid/kda-albert-xlarge-v2-race` | 10/50 | 20.0% |
| `Riiid/kda-scibert-uncased-race` | 8/50 | 16.0% |
| `Riiid/kda-mpnet-base-race` | 2/50 | 4.0% |

The ordering reproduces ex2 §6.3 exactly — the closed-book NQ-finetuned T5, the solver with the
most parametric knowledge, is the worst offender — now on human-grounded rather than
self-referential labels.

### B4b. About 30% of `prior_dependent` labels are construction artifacts

The second row of the table above is the finding that complicates things. On the 5 items where the
human says the passage **still reads as gold**, the substitution failed: answering gold is correct
reading, not prior-override. Those items are labelled `prior_dependent` anyway.

| | count | share of all `prior_dependent` |
|---|---:|---:|
| Ensemble-level, these 55 items | 5 of 16 | **31%** |
| Pair-level, these 55 items | 17 of 57 | **29.8%** [95% CI 19.5–42.7%] |

**Roughly three in ten `prior_dependent` labels in this sample are the construction failing, not a
solver overriding its context.** *Confirmed by a direct reading of the 5 items with no
rater involved (§D6): all five counterfactual passages still state the gold answer or state nothing
clearly.* This is a measurement-validity problem for ex2's and ex6's
prior-dependence counts on SciQ that no amount of metric reformulation would have surfaced — only
a human reading the perturbed passage could. It is a smaller effect than OBQA's (ex7 §11) but it
is the same class of defect, and SciQ was the split believed clean.

### B4c. On Q2, F does what it claims — and KDA_cont is inverted

Separating items the human reads as the CF target from those read as gold:

| Metric | AUC | 95% CI (bootstrap, 2000) |
|---|---:|---|
| **F** | **0.964** | [0.908, 1.000] |
| D | 0.868 | [0.736, 0.964] |
| **KDA_cont** | **0.072** | **[0.000, 0.196]** |

*Corrected: these intervals are now the ones `q2_crosstab.py` produces. They were first
computed with one-off code that ordered items differently and read the upper bound one position
off, so the same seed drew different resamples — bootstrap noise in the third decimal. Point
estimates are unchanged; previously F [0.912, 1.000], D [0.728, 0.968], KDA_cont [0.000, 0.192].*

Mean F is 0.335 where the human reads the CF target against 0.110 where they read gold.

> ⚠️ **Two limits before this is quoted.** The negative class is **n = 5**; the intervals are tight
> only because the separation is near-total, and five items is five items. And **F and the human
> Q2 label share a common cause**: when a substitution fails, the perturbed passage barely differs
> from the original, so the human reads gold *and* F is mechanically near zero. This is not the
> ex6 §14 circularity — the human label touches no model output — but it does mean AUC 0.964
> substantially measures *"F detects failed substitutions"*, which is a construction-validity
> property rather than proof that F measures knowledge-dependency.

### B4d. Human view vs the Setting-C-derived view, same 55 items

| | items |
|---|---:|
| Human: passage states CF target · solver followed it | 39 |
| Human: passage states CF target · solver stayed on gold — **genuine prior-override** | 11 |
| Human: passage states gold · solver stayed on gold — **construction failed** | 5 |
| Human: passage states gold · solver followed | 0 |

**The two views coincide on 44/55 = 80% of items.** The 20% where they part is the whole payload:
11 items where the Setting-C label and the human agree something real happened, and 5 where the
Setting-C label reports prior-dependence that a human reading says is not there.

## B5. Reading the result

Two things are worth saying plainly, and they point in opposite directions.

**F performs at chance against independent human labels** (AUC 0.490, ρ −0.103, both intervals
spanning chance) — despite dominating every Setting-C-derived comparison in ex6, where it reached
AUC 0.899. That is exactly what ex6 §14 predicted: F's Tier-1 performance came from re-encoding
`MarginC`, which *is* the Setting-C label, not from tracking knowledge-dependency. This is the
first evidence on labels that cannot be accused of that circularity, and F does not survive it.

**But D's advantage is at least partly structural, not a discovery.** Q1 asks a human to judge
whether an unaided student could answer — which is a human proxy for Setting A, i.e. for `P`.
`D = (1−P)·S·max(F,0)` contains that term; `F` does not. A metric sharing a factor with the thing
the human is estimating has a built-in edge on Q1, and D's AUC 0.697 should be read with that in
mind rather than as evidence that the product form is right. **Q1 does not test context-dependence
at all** — that was the gap Q2 was added to cover, and Q2 as built is a per-item construction check
rather than a metric discriminator, so it cannot break the tie either.

**Q2 points the other way from Q1, and that is the most important thing in this report.** On the
Q1 dimension F is at chance and D leads; on the Q2 dimension — which is F's *actual* claim,
context-following, and the only non-circular test here — **F separates the human's reading at AUC
0.964 while KDA_cont is inverted at 0.072** (§B4c, with the n=5 and shared-cause caveats attached).
The two dimensions are measuring different things, and each metric wins the one it was built for:
KDA_cont/D on prior-answerability, F on context-following.

> **Replication check (Part D).** Annotator 2 reproduces the inconclusive sign test
> (40/80), F at chance (AUC 0.432, ρ −0.091), and every Q2 figure exactly. It does **not** reproduce
> §B2's specific claim that D is the only metric whose AUC CI excludes 0.5: under annotator 2, D's
> CI is [0.490, 0.759] and no metric's AUC CI excludes chance. D's Spearman association does
> replicate (0.232, CI excludes zero). So "D leads on Q1" should be treated as **rater-dependent**,
> while "F is at chance on Q1" and all of §B4 hold under both raters.

**Net: this round does not adjudicate F vs KDA_cont vs D on a single scale**, and the sign test
says no large difference exists on the Q1 dimension. It does supply three genuinely new results:
F's Setting-C dominance does not transfer to independent prior-answerability labels (§B3);
**20% human-grounded prior-override** (§B4a); and **~30% of `prior_dependent` labels on SciQ are
construction artifacts** (§B4b).

## B6. Limitations

1. **The κ gate passed.** Block κ 0.870, full sample 0.680 (§D2). Both annotators read the English
   source alongside a translation, using different tools, so the judgements are independent (§D0).
   The estimate is imprecise: the block's positive class is 5 items, and the raters differ
   systematically in calibration.
2. **Q1 rating skew: 27 DEPENDENT / 73 NOT_DEPENDENT**, with rating 4 alone on 55/100 items. The
   AUC positive class is n = 27, which is why those CIs are ±0.12 wide. Underpowered for anything
   short of a large effect.
3. **Q2 items were recognisable as edited.** The annotator left a note on all 55 rows; **45 of 55
   explicitly identify a substitution** ("*'organs' substitutes for the expected term 'DNA'*").
   **Zero notes say "recalled"** — this is not carryover from Sheet 1, it is inference from subject
   knowledge, and the sampled notes state the answer was given from the passage text regardless.
   The guide's instruction was followed. But Q2 did not measure what an *uninformed* reader takes
   from the passage, which is what the construction-validity reading assumes.
   *Update:* annotator 2's Q2 notes never mention a substitution (0 of 55), yet their
   readings match annotator 1's on 54 of 55 items (§D4). Whether or not annotator 2 noticed the
   edits, detecting them does not appear to have changed what was read. That weakens this
   limitation; it does not remove it, since not writing about an edit is not evidence of not
   seeing it.
4. **`none` was never used on Q2.** The escape hatch existed and went unused across 55 items.
5. **Construct alignment (§B5).** Q1 is a P-proxy, so metrics containing (1−P) are advantaged on it.
6. **Roughly 30% of `prior_dependent` labels in this sample are construction failures** (§B4b),
   which is a validity problem for ex2's and ex6's SciQ prior-dependence counts, not for this
   annotation. It is measured on 55 items and should be re-measured before any headline
   prior-dependence figure is quoted again.
7. **The Q2 metric comparison (§B4c) has n = 5 in its negative class** and shares a common cause
   with F. Directionally informative, not a result to quote alone.
8. **Single dataset, single annotator, one week.** SciQ only; OBQA is closed separately (ex7 §11).

## B7. What would resolve the gate

**A second annotator on the 30-item double block: ~75 minutes.** The block is already built,
issued and locked; `kappa_independent_labels.py` scores it against annotator 1's existing file with
no rework:

```bash
python code/ex8_independent_labels/kappa_independent_labels.py --q1-a <annotator1.csv> --q1-b <annotator2.csv>
```

It can be done after the fact and retroactively licenses — or withdraws — everything in Part B.

> **Update.** A second annotator is doing the **full** sheets (100 Q1, 55 Q2), not just
> the block. How those labels are used was fixed before any came back:
> [`analysis_plan_annotator2.md`](analysis_plan_annotator2.md), SHA-256 in
> `results/ex8_independent_labels/analysis_plan_annotator2.LOCK.json`. In short: annotator 1 stays
> primary and these Part B results stand; the gate is κ on the pre-registered 30-item block, with
> full-sample κ reported beside it; annotator 2's full sheets are run through the unchanged
> comparison as a separately reported replication; labels are never averaged.
>
> **Note on the plan's wording.** §5.2 of the locked plan says "three findings from the
> Q2 cross-tab", but it defines a consistent/not-consistent verdict for only **two** of them —
> prior-override rate and artifact share — and says the ensemble-level artifact count is "reported
> alongside", without a verdict. The analysis follows the definitions as written, so exactly two
> verdicts are issued (§D4). The plan file itself is not edited, since that would break its
> recorded hash.
>
> **Result:** κ computed a pass on the block (0.870; full sample 0.680), and annotator 2's run
> replicates the primary result. Both annotators used AI for translation only, reading the English
> source alongside, and the gate was decided **PASSED**. See Part D, §D0.

---
---

# PART D — Annotator 2 (translation assistance): gate, replication, and disclosure

> **Written under a plan fixed before these labels were scored:**
> [`analysis_plan_annotator2.md`](analysis_plan_annotator2.md). **Annotator 1 (Part B) remains the
> primary result.** Annotator 2 is used for two things only: evaluating the κ gate, and a
> replication run of the identical, unchanged procedure. The two raters' labels are never averaged,
> majority-voted, or merged anywhere, and no LLM-judge output enters this part.

Sections follow the order plan §7 sets: structural validation → κ gate → Q1 replication → Q2
replication → deviations.

## D0. Disclosure: translation assistance, both annotators — read before §D1–§D5

### What the annotators reported

As relayed by the project lead (a summary of their responses, not their own wording; written
statements should replace this summary if they exist). **Both annotators:**

> 1. **Translation only.** AI was used solely to render the guide and the item text — question,
>    options, passage — into the annotator's native language, to aid comprehension.
> 2. **English read alongside.** Both read the original English source side by side with the
>    translation throughout the annotation.
> 3. **Different tools.** Each annotator independently used a different LLM for translation.
> 4. **No AI judgement of any kind.** Neither tool suggested labels, ratings, rankings or reasoning.
>    Every rating and every `stated_answer` is the annotator's own.

Annotator 2 also reported **no prior exposure** to the results document or any mentor update — only
the annotation guide — and completing the work in **two separate sittings** (see Timing below).

**How the account developed.** Annotator 2's first answer was relayed as *"used AI suggestions during
annotation, but the final decision on each rating was their own."* That wording was read as
AI-suggested *ratings*, and the gate was marked *not evaluated as designed* on that basis. Both
annotators then clarified that the assistance was translation only, and a follow-up established the
reading protocol and tooling above. The explicit statement in item 4 supersedes the earlier wording.
The banner's history line keeps the sequence visible rather than silent.

For fairness: the guide in use at the time never named AI tools. Its opening says the labels are
*"judgements made by a person reading only the question, options and passage, with no model output …
in view"*, and the Q2 instructions say *"do not look anything up"*. That rules out AI assistance in
spirit, but it never said so explicitly. The gap was partly the guide's. The guide now states the
rule directly: translation is the one permitted use, it must be declared, and the English must be
read alongside — the protocol both annotators report having followed.

### D0a. Residual risk from translation — closed

**Why translation is a much weaker channel than AI-suggested ratings.** The concern that two humans
might be converging on a model's *judgement* does not arise:

- the model never saw the 1–4 scale, the `stated_answer` options, or any instruction to rate;
- it made no judgement about knowledge-dependency, answerability, or what a passage states;
- it returned no rating, no ranking and no recommendation — only the same text in another language.

Every rating in this label set is therefore a human decision.

**The channels translation could have opened.** Translation is not a neutral pipe: a translator must
disambiguate. Had both annotators read the same machine translation *instead of* the English, a
mistranslation would have been a **correlated** distortion — one that inflates agreement where
independent human error would not. Three concrete channels, and how the confirmed protocol closes
each:

| Channel | Where it would bite | Under the confirmed protocol |
|---|---|---|
| **Sense disambiguation** | a term with several readings is forced into one ("base" as alkali vs foundation, "residues", "oxidants") | **Reduced to a floor.** The English is in view, and different tools make a shared mis-disambiguation much less likely |
| **Normalisation of corrupted text** | machine translators repair what looks like an error — a fused *"calledoxidants"* becomes *"called oxidants"*, an ungrammatical splice is smoothed into a sentence. **Q2-specific** | **Closed.** Each annotator saw the corrupted English itself. §D6 separately verifies the decisive items with no rater involved |
| **Loss or creation of surface cues** | Q1's rubric names *"a giveaway word"*; a stem–option echo (*"Digestive…" → "digestive system"*) may vanish or appear in translation. **Q1-specific** | **Closed.** The giveaway word and any echo are in the English each annotator read |

**The floor, stated precisely.** Different LLMs learn from overlapping data and share tendencies, and
an annotator who did not know an English word could lean on the translation for it. That residual is
the one any non-native annotator using a dictionary carries — ordinary annotation practice, not a
departure from protocol.

**Why no remaining channel can make the pass misleading.** A passing κ misleads only if the raters
shared an input that pulled their labels together. Independent error — carelessness, speed,
idiosyncratic misreadings, two *different* translations — pushes κ **down**. With different tools and
the English read alongside, no shared input remains. The κ figures are therefore not an artifact of
how the labels were produced; any residual independent noise makes them, if anything, conservative.

**Consistent with the observed disagreement.** The raters disagree on 11 of 100 Q1 items, all in one
direction — the signature of a difference in personal calibration (§D2), not of a shared external
input.

### D0b. What this does to the Q2 54/55 match

§D4 reported that the two raters read 54 of 55 Q2 items the same way. The one route by which
translation could have manufactured that agreement — both reading a machine-repaired version of a
deliberately broken passage — is closed: both read the corrupted English itself. Three further things
bound it:

1. **Q2 is near-objective.** The edit places the counterfactual answer in the passage, so the split
   is driven mostly by the text.
2. **The findings depend only on which 5 items fall outside the group**, not on finer judgement.
3. **Those 5 items have been re-read directly in the original English, with no rater involved —
   §D6 — and all 5 are confirmed.**

The ordering of evidence for the Q2 findings is unchanged: **§D6's direct reading first, the 54/55
rater agreement second.**

### Timing — both statements are true

Full-precision file times (local, +0700; "born" is when the file was created):

| | born | last saved |
|---|---|---|
| Q1 sheet, annotator 2 | 2026-09-22 **00:21:11** (issued blank) | 2026-09-22 **01:34:18** |
| Q2 sheet, annotator 2 | 2026-09-22 **00:21:11** (issued blank) | 2026-09-22 **11:10:03** |

- **Same calendar day, and plausibly two sittings.** The last Q1 save is 01:34 at night and the last
  Q2 save is 11:10 the next morning — 9 h 36 min apart, very likely with sleep in between. That fits
  "two separate sittings". It does not meet the guide's instruction to do Q1 and Q2 on **different
  days**, which existed to reduce carryover from Q1's original passages to Q2's edited ones.
- **What the files cannot show.** Only the *last* save is recorded, so when Q2 work *started* is
  unknown — the gap between finishing Q1 and starting Q2 could be anything from zero to 9.5 hours.
  Both files keep their original creation time, so they were edited in place rather than replaced.
  Q1's final save came 73 minutes after the sheet was issued.
- **Carryover does not seem to have mattered.** Annotator 2's Q2 readings match annotator 1's on
  54 of 55 items (§D4). The deviation is real, but it has no visible effect on the Q2 results.

### What translation assistance does to the gate

**What κ was registered to measure.** Whether a second, independent human applying the same rubric
reproduces annotator 1's labels — which is what licenses treating the labels as a product of the
rubric rather than of one person.

**What it measures here.** Two humans, each deciding for themselves, over the identical English
source, with a translation alongside from different tools. The judgement channel is intact — no model
rated anything — and so is the text channel, since both read the same English. The condition the
gate tests is met, and the way it was tested does not compromise it.

### The observed patterns — partly explained, partly still open

Three patterns were flagged before any disclosure: a pace of at most 73 minutes for 100 Q1 ratings
with a written note each; 71% of Q2 notes opening with the same phrase; and annotator 2 rating
higher than annotator 1 on 31 items and lower on none.

The translation-only account explains these less completely than AI-suggested ratings would have,
and that is recorded rather than smoothed over:

- **Templated, fluent English notes** have a natural explanation: if the annotators wrote their notes
  in their own language and machine-translated them into English for the sheet, uniform phrasing and
  typographic curly quotes follow immediately.
- **Pace** remains partly unexplained, and reading two languages side by side would add time rather
  than save it. It is not impossible — the rubric is short, Q1 items are brief, many are easy, and
  translated notes are quick to paste — but 45 seconds per item including a note is fast. **Pace
  cannot inflate κ**: speed produces independent error, which lowers agreement with another rater.
- **One-directional leniency** is best explained as an ordinary difference in personal calibration at
  the 2/3 boundary. Translation flattening, an alternative considered earlier, is unlikely given that
  both annotators read the English.

None of this is an accusation, and none of it bears on the gate status. It is recorded because the
patterns were raised in this document.

### What it means for the replication claims

**Q1 — an independent human replication.** The inconclusive sign test and "F at chance" reproduce
under a second rater who judged the English source for themselves. The non-replication of D's AUC
claim also stands, and the calibration offset explains it: annotator 2's higher ratings shrank the
DEPENDENT class from 27 to 16.

**Q2 — reassuring, for a reason that does not depend on rater independence.** Q2 is near-objective:
the substitution literally puts the counterfactual answer in the text, so careful readers should land
on the same item split, and the convergence is driven mainly by the passages. The prior-override and
artifact figures depend only on *which items* are read as the counterfactual target. Both raters drew
that line at the same 5 items (0, 25, 222, 716, 805), and §D6 confirmed all 5 by reading the passages
directly, with no rater in the loop.

### Gate status: **PASSED** — decided by the project lead

> **This decision is argued only from how the labels were produced.** Whether F or D actually beats
> KDA_cont is a separate, empirical question, settled by the sign test and AUC on their own terms, and
> it is given no weight here. How much a methodological question matters cannot be set by how the
> substantive result happened to come out; the gate would deserve the same scrutiny had the result
> been decisive.

**Why PASSED.** The registered condition is independent human judgement applying the rubric, and it
is met: no model judged anything, both annotators read the identical English source, and their
translations came from different tools (§D0a). The only caveat that concerned how the labels were
produced — annotation over translated text — was to be removed by confirmation that the annotators
read the English alongside; that confirmation was given. The plan (§4) registers two outcomes,
*passes* and *fails*. On the pre-registered block, κ = 0.870 ≥ 0.70: **the gate passes.**

**Why not "passed with caveat".** That status was used while the reading protocol was unconfirmed. It
also carried three points that describe the precision of the estimate rather than the production of
the labels. Those points stay in this document — below and in §D2 — but a qualified pass is not an
outcome the plan registered, and attaching one to carry them would be a post hoc qualification of a
pre-registered result.

**Limits on the κ estimate**, all independent of how the labels were produced:

1. The block κ rests on only **5 DEPENDENT items** (annotator 1). One more disagreement gives 0.714,
   two more give 0.526.
2. The **full-sample κ is 0.680**, below the threshold. The block governs by the rule fixed in
   advance, and both figures are always quoted together.
3. The two raters differ **systematically**, annotator 2 rating higher on 31 items and lower on none.

A larger double-annotated block would tighten the estimate; nothing in the production record bears on
the status. Both annotators' data is retained in full and at full weight.

**What passing does to Part B.** Per plan §4, Part B moves from exploratory to gated. The decision
rule's branches become available, and the primary result is its INCONCLUSIVE branch as already
computed. No number changes.

## D1. Structural validation — passes; three protocol points to confirm with annotator 2

Checked against the **original blank issue** (`annotation_sheets.LOCK.json`), not annotator 1's
files. Before scoring, the plan, all three scoring scripts and the blank issue were re-verified
against their recorded hashes; all matched.

| Check | Q1 | Q2 |
|---|---|---|
| Rows | 100/100 ✅ | 55/55 ✅ |
| Columns, order, comment header, item order, immutable columns | unchanged ✅ | unchanged ✅ |
| Metric, class or stratum columns | none ✅ | none ✅ |
| Valid values; `unusable` conflicts | 100 valid, 0 unusable, 0 violations ✅ | 55 valid, 0 violations ✅ |
| Double block filled | rows 1–30 = manifest block, all answered ✅ | 16/16 ✅ |

Rating distribution — Q1: `1`×2, `2`×14, `3`×10, `4`×74, so **16 DEPENDENT / 84 NOT_DEPENDENT**
(annotator 1: 27 / 73). Q2: a 35, b 11, c 4, d 4, **none 1**.

**Exposure (plan §1) — resolved.** None of the 155 notes mentions the results document,
a mentor update, annotator 1, any metric, Setting A/B/C, an LLM, or recall, and **annotator 2 has
confirmed no prior exposure** to the results document or any mentor update — only the guide.

**Protocol observations, from file metadata — to be asked, not concluded.** The blank sheets were
issued at 00:21 on 2026-09-22. The Q1 sheet was last saved at 01:34, so if it was filled in that
file, 100 ratings plus a written note on every row took at most 73 minutes — under 45 seconds per
item, against the guide's 2.5-minute budget. The Q2 sheet was last saved at 11:10 the same day; the
guide asked for Q1 and Q2 on **different days**. And 71% of the Q2 notes open with the same phrase
("*The passage explicitly states…*"). Each has an innocent explanation — a copy filled elsewhere and
pasted in, a long overnight gap, a note-writing habit — and annotator 1's notes were also on every
row. But together they are also what an AI-assisted pass would look like, and that would matter: κ
would then partly measure human-versus-model agreement. **Three questions for annotator 2, before
these figures go into a mentor update:** was any AI tool used; were Q1 and Q2 done in separate
sittings; and had they seen the results document?

*Resolved — see §D0.* No exposure; two separate sittings on the same calendar day; AI used for
**translation only**, with the English source read alongside and a different tool per annotator. The
notes pattern has a natural explanation under that account; the pace is only partly explained, and
cannot inflate κ (§D0).

## D2. The κ gate — passes on the block (0.870); full sample 0.680

| | scope | n | collapsed-binary κ | raw agreement | quadratic-weighted κ | role |
|---|---|---:|---:|---:|---:|---|
| **Q1** | **pre-registered 30-item block** | 30 | **0.870** | 0.967 | 0.868 | **THE GATE — passes (≥ 0.70)** |
| Q1 | full sample | 100 | 0.680 | 0.890 | 0.713 | supplementary, does not decide |
| Q2 | 16-item block | 16 | 1.000 | — | — | no gate, as pre-registered |
| Q2 | full sample | 55 | 0.967 | — | — | supplementary |

**As computed, the gate passes: block κ = 0.870 — while the full-sample κ is 0.680, just under
0.70.** The two fall on opposite sides of the threshold, which is exactly the case the plan's rule
covers: the block governs, and the two are stated together. (κ is a statistic of the *pair* of
raters, so there is one block figure and one full-sample figure, not one per annotator.)

**The disagreement runs one way only.** On the 1–4 scale annotator 2 rated **higher than annotator 1
on 31 items, equal on 69, and lower on none** — the whole lower triangle of the confusion matrix is
empty:

| annotator 1 ↓ / annotator 2 → | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| **1** | 2 | 6 | 1 | 1 |
| **2** | 0 | 8 | 5 | 4 |
| **3** | 0 | 0 | 4 | 14 |
| **4** | 0 | 0 | 0 | 55 |

Mean rating 3.18 (annotator 1) against 3.56 (annotator 2). All 11 binary disagreements are
annotator 1 DEPENDENT / annotator 2 NOT_DEPENDENT. This is a **calibration offset** — annotator 2
judges items as more answerable without the passage — not random disagreement, and the rank
ordering agrees well: Spearman between the two raters **0.768 [0.657, 0.856]**. It also explains
most of the block/full gap: with 84% of annotator 2's ratings in NOT_DEPENDENT, chance agreement is
high, so 89% raw agreement becomes κ 0.680.

**How much weight the pass can bear.** The block κ rests on only **5 DEPENDENT items** (annotator 1)
and one disagreement. With one more disagreement it would be 0.714 — still a pass; with two more,
0.526 — a fail. As computed, the gate passed as specified, but on a narrow base, and
the full-sample figure is the better estimate of how reliably the rubric reproduces.

**What passing does to Part B.** Per plan §4, a pass moves Part B from exploratory to gated. The gate
was decided **PASSED** by the project lead upon confirmation of the parallel reading protocol (§D0),
so Part B is gated.

The figures above stand as computed. Read them as **agreement between two humans who each judged the
English source for themselves**, with a translation alongside from different tools (§D0).

*On the result files.* `metric_comparison.json` and `metric_comparison.annotator2.json` record
"KAPPA GATE NOT EVALUATED" because the plan ran them without a second rater's input, by design; the
gate result is in `kappa_annotator1_vs_annotator2.json`. Their decision strings also carry a
"product form is preferred" clause. Gating Part B does not promote it: D's Q1 AUC lead is
rater-dependent under the replication rule (§D3).

## D3. Q1 replication — **replicated**

Annotator 2's full sheets through the unchanged `compare_metrics.py`, exactly as the plan's command
specifies, output `metric_comparison.annotator2.json`.

| Sign test | Primary: annotator 1 | Replication: annotator 2 |
|---|---|---|
| **F vs KDA_cont, strata A+B** | 41/80 = 0.512, p 0.911, no winner | **40/80 = 0.500, p 1.000, no winner** |
| F vs KDA_cont, stratum A | 22/55 = 0.400, p 0.177 | 23/55 = 0.418, p 0.281 |
| D vs KDA_cont | 44/78 = 0.564, p 0.308 | 45/78 = 0.577, p 0.213 |
| D vs F | 46/79 = 0.582, p 0.177 | 47/79 = 0.595, p 0.115 |

Under the plan's pre-declared table, primary INCONCLUSIVE and replication INCONCLUSIVE is
**replicated**: no large difference between F and KDA_cont on Q1. *Both raters decided for
themselves, reading the English source with a translation alongside from different tools (§D0).*

Two features of this null make it robust on its own terms, and neither is an inference from what the
metrics are otherwise thought to be worth. First, the two raters **disagree on 11 of 100 Q1 items,
all in one direction**, so they were plainly not producing the same labels by a common route; a
shared external influence strong enough to drive the result would have shown as closer agreement.
Second, a shared influence inflates *agreement*, and the quantity here is a **null** — 41/80 and
40/80, both at chance. That failure mode manufactures false positives, not false nulls.
**Nothing is adopted.** Had annotator 2 produced a significant winner, the rule would have read it as
*not replicated* and still adopted nothing.

AUC against the binarised label (positive = DEPENDENT; annotator 1 n = 27, annotator 2 n = 16):

| Metric | annotator 1 | annotator 2 | CI excludes 0.5? |
|---|---|---|---|
| KDA_cont | 0.580 [0.459, 0.701] | 0.569 [0.420, 0.718] | neither |
| F | 0.490 [0.356, 0.613] | 0.432 [0.288, 0.579] | neither |
| **D** | **0.697 [0.581, 0.810]** | **0.632 [0.490, 0.759]** | **annotator 1 only — not replicated** |
| D′ | 0.458 [0.330, 0.577] | 0.388 [0.251, 0.527] | neither |

IPW (population) AUC, annotator 1 → annotator 2: KDA_cont 0.455 → 0.387 · F 0.498 → 0.433 ·
D 0.668 → 0.583 · D′ 0.458 → 0.374.

Spearman against the 1–4 rating:

| Metric | annotator 1 | annotator 2 | CI excludes 0? |
|---|---|---|---|
| KDA_cont | 0.124 [−0.087, 0.305] | 0.157 [−0.038, 0.336] | neither |
| F | −0.103 [−0.293, 0.094] | −0.091 [−0.279, 0.097] | neither |
| **D** | **0.233 [0.038, 0.417]** | **0.232 [0.048, 0.411]** | **both — replicated** |
| D′ | −0.167 [−0.352, 0.026] | −0.141 [−0.320, 0.038] | neither |

**The specific claim the plan names — "D is the only metric whose Q1 AUC CI excludes 0.5" — does
not replicate.** Under annotator 2 no metric's AUC CI excludes chance; D's lower bound is 0.490.
D's rank association with the rating does replicate. **F at chance replicates on every statistic.**
Annotator 2's smaller positive class (16 against 27, from the leniency offset in §D2) widens their
AUC intervals, which is part of why D's slips across 0.5.

Two lines in the replication run's printed output need explaining. It prints *"KAPPA GATE NOT
EVALUATED"* because the plan runs annotator 2 **without** `--q1-b` — annotator 2 is a replication
there, and the gate is evaluated separately in §D2. And it prints *"D clears the 0.02 AUC bar over
F, so the product form is preferred"*, as annotator 1's run did; under the INCONCLUSIVE branch no
metric is adopted, so that comparison selects nothing.

## D4. Q2 replication — **consistent on both findings, near-identical readings**

`q2_crosstab.py`, annotator 1 against annotator 2, with `--verify-published` confirming the script
reproduces annotator 1's published figures first. Output `q2_crosstab_annotator1_vs_annotator2.json`.

| Finding | annotator 1 | annotator 2 | Verdict (CIs overlap) |
|---|---|---|---|
| **Prior-override** (CF-read pairs answering gold) | 40/200 = **20.0%** [15.0, 26.1] | 40/200 = **20.0%** [15.0, 26.1] | **consistent** |
| **Artifact share** (prior_dependent pairs on items not read as CF) | 17/57 = **29.8%** [19.5, 42.7] | 17/57 = **29.8%** [19.5, 42.7] | **consistent** |
| Artifact share, ensemble level | 5 of 16 | 5 of 16 | reported alongside, no verdict |
| **Item-level agreement**, four-way reading | — | — | **54/55 = 98.2%** |

The figures are identical because the two raters split the 55 items into **the same 50 read as the
counterfactual target and the same 5 not** (items 0, 25, 222, 716, 805). The single difference is
item 222, which annotator 1 read as still stating the gold answer and annotator 2 marked `none` —
both "not the CF target", so neither rate moves.

**So the two strongest findings in this document — 20% human-grounded prior-override and ~30% of
`prior_dependent` labels being construction artifacts — hold under both raters.** *Both
annotators read the corrupted English itself, so a translator's repair of broken text could not drive
the split (§D0b), and **the decisive rater-free check on the original English confirmed all 5 items
(§D6)** — these findings do not rest on rater agreement at all.* Q2 has a
near-objective answer (the substitution literally places the target in the text), so this level of
agreement is what a sound instrument should produce, and it is also why Q2 agreement says little
about the harder Q1 judgement.

Q2 AUC, recomputed as the plan requires but **not a replication test** (gold-read class n = 5 and
n = 4): annotator 1 F 0.964 [0.908, 1.000], D 0.868 [0.736, 0.964], KDA_cont 0.072 [0.000, 0.196];
annotator 2 F 0.960 [0.895, 1.000], D 0.885 [0.745, 1.000], KDA_cont 0.020 [0.000, 0.065].

## D5. Deviations from the plan

**None in the analysis.** Every step ran as the locked plan specifies, with the scripts at their
recorded hashes. For the record:

- **Timing evidence is weaker than intended.** The plan and scripts were committed after annotator 2
  had finished, so the commit shows they were fixed before analysis, not before labelling. The
  locks' own recorded times (00:29 and 00:40) precede the last saves of annotator 2's sheets (01:34
  and 11:10), but those timestamps are editable.
- **Plan §1's exposure check — resolved:** annotator 2 confirms no prior exposure (§D0).
- **Protocol deviations by the annotators (§D0):** both used AI **translation** of the guide and
  item text, read alongside the English source — a use the guide in force at the time neither
  anticipated nor named, and now permits when declared; and annotator 2 did Q1 and Q2 in two sittings on the same calendar
  day rather than on different days. These are deviations from the *annotation
  protocol*, not from the *analysis plan*, which ran as written. Their effect is on how the results
  are interpreted, not on the numbers.
- **The plan's "three findings" wording** is addressed by the note in §B7: two verdicts are
  defined, and two were issued.
- `--verify-published` was run in the same call as the comparison. The plan requires that check
  before the script is used; running it alongside changes nothing.

---
---

## D6. Rater-free check of the 5 items — all 5 confirmed

The 20% prior-override and 29.8% artifact figures both hinge on 5 of the 55 items falling outside
the "reads the counterfactual target" group. Both annotators put the same 5 items there. The check
below does not depend on how anyone read them: **it goes to the passages themselves.**
Each was read directly against the question, options, gold answer and the original passage. No
annotator's label and no Part C output was consulted.

The question for each: **does the counterfactual passage clearly state the counterfactual target?**

| Item | gold → CF target | Does the CF passage state the CF target? | What actually happened |
|---|---|---|---|
| **0** | `oxidants` → `Oxygen` | **No — still states gold** | Only the heading changed ("Oxidants and Reductants" → "Oxygen and Reductants"). The answering clause still reads *"are calledoxidants (or oxidizing agents)"* — the word is fused to "called", so whole-word matching skipped it |
| **25** | `immune system` → `respiratory system` | **No — still states gold** | Only the title changed. The answering clause still reads *"chronic disease associated with deficiencies of the immune system**s**"* — plural, so the singular pattern did not match |
| **222** | `chemical state of solute` → `similar state of solute` | **No — states nothing clearly** | A `partial`-tier splice of the fragment "of solute" produced *"depend only upon the total concentration similar state of solute species"*, which is not a sentence. Meanwhile *"dependent upon the chemical identity of the solute"* survives and still points at gold |
| **716** | `antioxidants` → `neurotransmitters` | **No — still states gold** | Two plural occurrences were replaced, but the defining sentence — the one that answers the question — still reads *"An **antioxidant** is a molecule that inhibits the oxidation of other molecules"* |
| **805** | `acid` → `base` | **No — still states gold** | Downstream mentions became "base fog"/"base rain", but the answering clause still reads *"Certain air pollutants form **acids** when dissolved in water droplets"* — plural, unmatched. The result also contradicts itself: "base fog … pH of 4" |

**All five confirmed.** In four of the five the *original answering clause survives verbatim*, so a
solver answering gold on these items is reading the passage correctly, not overriding it. Labelling
those solver responses `prior_dependent` is a construction artifact.

**This is the strongest evidence in the document for the two Q2 findings**, because it involves no
rater — human, translation-assisted, or model. It confirms what both annotators concluded, which also means
their Q2 readings were correct on the cases that matter, whatever process produced them.

### A defect this exposes in the ex7 fidelity audit

The five failures share one mechanism: **the substitution replaced one surface form of the gold
answer while the answering clause carried a different form.** Singular against plural in 25, 716 and
805; a fused word boundary in 0; a fragment splice in 222.

ex7's leakage audit checked `residual_answer_mentions`, whole-phrase exact residuals, and reported
**0 across all 860 SciQ items** — treating whole-phrase leakage as a solved channel
([`counterfactual_construction_audit.md`](../ex7_counterfactual_construction/counterfactual_construction_audit.md) §1).
For these five items that metric reads 0, and the glued counter catches only 2 of the 5. **It has a
morphological blind spot: an inflected residual in the answering clause is invisible to it.** That
is a concrete, fixable gap in the audit, and it is what produced roughly 30% of the
`prior_dependent` labels in this sample. Flagged, not acted on.

---
---

# PART C — LLM-judge check: **SUPPLEMENTARY ONLY, NOT A SECOND ANNOTATOR**

> ## ⚠ Read this before any number in Part C
>
> **Note:** both human annotators used AI for **translation only**, reading the English source
> alongside (§D0). Their judgements are their own, so the comparison below remains model against
> human judgement.
>
> **This is not a human rater, not an inter-annotator κ, and not an input to the gate.** Its output
> was never merged with, averaged with, or substituted for either human annotator's labels anywhere,
> it plays no part in Part D, and
> `compare_metrics.py` was never run with it.
>
> **This project has already measured what an unvalidated LLM judge does here.** Task A found
> Qwen3-4B — the same model used below — **over-crediting 13:1** against a stricter human check,
> with κ = 0.069 in the regime where it was self-inconsistent. **High agreement below would be weak
> supplementary evidence at best; it would not validate the rubric or the labels.** As it happens,
> agreement is not high.

**Setup.** Local `Qwen3-4B-Instruct-2507`, 4-bit NF4, greedy decoding, loaded exactly as
`kda_qwen_eval.py` loads it. `api.anthropic.com` returned **HTTP 401** — no key in the environment
— so the `claude-sonnet-4-6` path was unavailable; this is the weaker of the two judges the plan
allowed. One **fresh stateless generation per item**, no state carried between items and no
exposure to this session's context. Each prompt is the rubric lifted verbatim from
`ANNOTATION_GUIDE_RQ2.md` §2 / §3 plus one item's fields — nothing else. Every prompt is saved to
`llm_judge_prompts.json`; a leakage audit for 20 banned terms (KDA, Setting A/B/C, stratum, metric,
AUC, ex6/ex7, RQ2, solver, …) returned **CLEAN** on both templates. 0 of 155 responses unparseable.

## C1. Q1 agreement with annotator 1

| Scope | n | exact | within-1 | collapsed binary | quad-weighted | mean human → LLM |
|---|---:|---:|---:|---:|---:|---|
| All items | 100 | 0.470 | 0.900 | 0.750 | 0.466 | 3.18 → 3.20 (+0.02) |
| 30-item block | 30 | 0.433 | 0.900 | 0.767 | 0.311 | 3.37 → 3.00 (−0.37) |

Exact agreement is under half; within-1 is high, so the two mostly differ by one scale point. The
confusion is concentrated in one cell — **19 items the human rated 4, the LLM rated 3** — i.e. the
judge is more conservative about calling an item obviously answerable, the opposite direction to
Task A's over-crediting.

## C2. Q2 — the judge and the human diverge sharply

| Reads the counterfactual passage as stating… | human | LLM |
|---|---:|---:|
| the counterfactual target | **50 (90.9%)** | **21 (38.2%)** |
| the gold answer | 5 (9.1%) | **33 (60.0%)** |
| none | 0 | 1 (1.8%) |

Option-match with the human: **0.473** overall, 0.438 on the block.

**Q2 is the one place here with a near-objective answer** — the substitution literally places the
counterfactual target in the text — and the judge answers with the *factually correct* option
instead 60% of the time. That is the prior-over-context behaviour this whole project is about,
appearing in the judge rather than the solver, and it is consistent with Task A.

Two reasons not to over-read it: the human had been told the passages were edited and detected the
edits on 45/55 items (§B6.3), which is not a level playing field; and a 4B model under 4-bit
quantisation is a weak reader.

## C3. What Part C is worth

**Very little on its own, in both directions.** Agreement is moderate on Q1 (exact 0.47,
quad-weighted 0.47) and poor on Q2 (0.47 option match). Per the Task A precedent, a *high* number
would not have validated anything; a *low* number likewise does not indict the human labels,
because the judge is the less reliable instrument on the one sub-task with a checkable answer.

The single defensible use: it is **not** evidence that annotator 1's ratings are anomalous. It does
nothing to discharge the κ gate, and **§B7 remains the only thing that will.**

---

## Related

- [`ANNOTATION_GUIDE_RQ2.md`](ANNOTATION_GUIDE_RQ2.md) — the rubric, verbatim, as both raters saw it
- [`docs/ex6_psfd_score/psfd_formulation.md`](../ex6_psfd_score/psfd_formulation.md) §14 — why Setting-C-derived labels could not settle this
- [`docs/ex5_failure_audit/annotation_final_conclusion.md`](../ex5_failure_audit/annotation_final_conclusion.md) — Task A, the 13:1 precedent behind Part C's caveat
- [`docs/ex7_counterfactual_construction/counterfactual_construction_audit.md`](../ex7_counterfactual_construction/counterfactual_construction_audit.md) §11 — OBQA close-out
